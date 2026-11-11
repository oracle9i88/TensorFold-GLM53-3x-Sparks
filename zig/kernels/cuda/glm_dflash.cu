// SPDX-License-Identifier: Apache-2.0
// Modified 2026 by Kristian Aaron (Krishando). Original: ashhart/TensorFold@d1018b318 (Apache-2.0).
// Modified 2026 in the glm53-tp3-3x-sparks branch (GLM-5.3 on three DGX Sparks); see docs/tp3-changes.txt.
// DFlash2 drafter kernels (ours, after z-lab/dflash, MIT): taps, norms+rope, convs, SwiGLU, ring and block attention.

#include <cuda_bf16.h>
#include <stdint.h>

namespace {
constexpr int D = 4096;
constexpr int HD = 128;
constexpr int KVH = 8;
constexpr int QH = 32;
constexpr int KVW = KVH * HD;  // 1024
constexpr int WIN = 2048;      // the sliding window (and the ring's slots)
constexpr int BLOCK = 8;
constexpr int GROUP = 16;  // channels sharing one dynamic tap weight
constexpr int GROUPS = D / GROUP;
constexpr int CH = 16;  // key chunks of the chunked attention (df_attn_part: a block a kv head and chunk)

__device__ __forceinline__ float bf(__nv_bfloat16 x) { return __bfloat162float(x); }
__device__ __forceinline__ __nv_bfloat16 tobf(float x) { return __float2bfloat16_rn(x); }

// The ring of one layer: K then V, [WIN][KVW] each, layers one after another.
__device__ __forceinline__ __nv_bfloat16* ringK(uint64_t base, int layer) {
    return reinterpret_cast<__nv_bfloat16*>(base) + (long long)layer * 2 * WIN * KVW;
}
__device__ __forceinline__ __nv_bfloat16* ringV(uint64_t base, int layer) { return ringK(base, layer) + (long long)WIN * KVW; }

// Qwen3's per-head RMSNorm then rotate-half RoPE of one head (64 threads: dims d and d + 64), in place or to out.
__device__ __forceinline__ void normRope(const __nv_bfloat16* x, const __nv_bfloat16* w, float eps, long long pos, float theta,
                                         __nv_bfloat16* out, float* red) {
    const int d = threadIdx.x;  // 0..63
    const float a = bf(x[d]), b = bf(x[d + 64]);
    float s = a * a + b * b;
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    if ((d & 31) == 0) red[d >> 5] = s;
    __syncthreads();
    const float rinv = rsqrtf((red[0] + red[1]) / HD + eps);
    __syncthreads();
    const float na = bf(tobf(bf(w[d]) * bf(tobf(a * rinv))));
    const float nb = bf(tobf(bf(w[d + 64]) * bf(tobf(b * rinv))));
    const float inv = powf(theta, -(float)(2 * d) / HD);
    const float ang = (float)pos * inv;
    const float c = cosf(ang), sn = sinf(ang);
    out[d] = tobf(na * c - nb * sn);
    out[d + 64] = tobf(nb * c + na * sn);
}

// DSpark's 64-wide heads use one warp per head, with each lane handling a pair
// of dimensions. The established 128-wide path remains separate and unchanged.
__device__ __forceinline__ void normRope64(const __nv_bfloat16* x, const __nv_bfloat16* w, float eps, long long pos, float theta,
                                           __nv_bfloat16* out, float* red) {
    const int d = threadIdx.x & 31;
    const float a = bf(x[d]), b = bf(x[d + 32]);
    float s = a * a + b * b;
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    if (threadIdx.x == 0) red[0] = s;
    __syncthreads();
    const float rinv = rsqrtf(red[0] / 64.0f + eps);
    __syncthreads();
    const float na = bf(tobf(bf(w[d]) * bf(tobf(a * rinv))));
    const float nb = bf(tobf(bf(w[d + 32]) * bf(tobf(b * rinv))));
    const float inv = powf(theta, -(float)(2 * d) / 64.0f);
    const float ang = (float)pos * inv;
    const float c = cosf(ang), sn = sinf(ang);
    out[d] = tobf(na * c - nb * sn);
    out[d + 32] = tobf(nb * c + na * sn);
}
}  // namespace

// A target layer's residual, its 4 streams' mean (TF_GLM_DFLASH2_TAP=s0: stream 0), into column block ``slot``.
extern "C" __global__ void df_tap(const __nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ out, int slot, int mode) {
    const int r = blockIdx.x;
    const __nv_bfloat16* xr = x + (long long)r * 4 * D;
    __nv_bfloat16* o = out + (long long)r * 5 * D + (long long)slot * D;
    for (int d = blockIdx.y * 1024 + threadIdx.x; d < (blockIdx.y + 1) * 1024; d += blockDim.x) {
        if (mode == 1) {
            o[d] = xr[d];
            continue;
        }
        const float a = ((bf(xr[d]) + bf(xr[D + d])) + bf(xr[2 * D + d])) + bf(xr[3 * D + d]);
        o[d] = tobf(a / 4.0f);
    }
}

// The block's queries (fused qkv, stride ldq) normed and rotated at pos[r] into q, keys into kb, values into vb.
extern "C" __global__ void df_block_qkv(const __nv_bfloat16* __restrict__ qkv, int ldq, const __nv_bfloat16* __restrict__ qn,
                                        const __nv_bfloat16* __restrict__ kn, const int* __restrict__ pos, float eps, float theta,
                                        __nv_bfloat16* __restrict__ q, __nv_bfloat16* __restrict__ kb, __nv_bfloat16* __restrict__ vb) {
    __shared__ float red[2];
    const int r = blockIdx.x, h = blockIdx.y;
    const __nv_bfloat16* row = qkv + (long long)r * ldq;
    if (h < QH) {
        normRope(row + h * HD, qn, eps, pos[r], theta, q + (long long)r * QH * HD + h * HD, red);
    } else {
        const int kh = h - QH;
        normRope(row + QH * HD + kh * HD, kn, eps, pos[r], theta, kb + (long long)r * KVW + kh * HD, red);
        const __nv_bfloat16* v = row + QH * HD + KVW + kh * HD;
        __nv_bfloat16* vo = vb + (long long)r * KVW + kh * HD;
        vo[threadIdx.x] = v[threadIdx.x];
        vo[threadIdx.x + 64] = v[threadIdx.x + 64];
    }
}

// Context rows' keys and values (k | v) normed, rotated at pos[r], into rings[r] at slot pos % WIN of ``layer``.
extern "C" __global__ void df_ctx_kv(const __nv_bfloat16* __restrict__ kv, const __nv_bfloat16* __restrict__ kn,
                                     const long long* __restrict__ pos, const uint64_t* __restrict__ rings, int layer, float eps,
                                     float theta) {
    __shared__ float red[2];
    const int r = blockIdx.x, kh = blockIdx.y;
    const long long p = pos[r];
    const int slot = (int)(p % WIN);
    const __nv_bfloat16* row = kv + (long long)r * 2 * KVW;
    normRope(row + kh * HD, kn, eps, p, theta, ringK(rings[r], layer) + (long long)slot * KVW + kh * HD, red);
    __nv_bfloat16* vo = ringV(rings[r], layer) + (long long)slot * KVW + kh * HD;
    vo[threadIdx.x] = row[KVW + kh * HD + threadIdx.x];
    vo[threadIdx.x + 64] = row[KVW + kh * HD + threadIdx.x + 64];
}

// Two-tap dynamic causal conv over a block: y[t] = sum_o (base[o] + dyn[t][o][c / 16]) * z[t - o]; add or store.
extern "C" __global__ void df_conv(const __nv_bfloat16* __restrict__ z, const __nv_bfloat16* __restrict__ dyn, int which,
                                   const __nv_bfloat16* __restrict__ base, __nv_bfloat16* __restrict__ out, int add) {
    const int r = blockIdx.x, t = r % BLOCK;
    const __nv_bfloat16* dr = dyn + (long long)r * 4 * GROUPS + which * 2 * GROUPS;
    for (int c = blockIdx.y * 1024 + threadIdx.x; c < (blockIdx.y + 1) * 1024; c += blockDim.x) {
        const int g = c / GROUP;
        float y = (bf(base[c]) + bf(dr[g])) * bf(z[(long long)r * D + c]);
        if (t > 0) y += (bf(base[D + c]) + bf(dr[GROUPS + g])) * bf(z[(long long)(r - 1) * D + c]);
        __nv_bfloat16* o = out + (long long)r * D + c;
        *o = add ? tobf(bf(*o) + bf(tobf(y))) : tobf(y);
    }
}

// SwiGLU over [gate | up] rows (width F each): bf16(bf16(silu(g)) * u). grid (R, F / 1024), 256 threads.
extern "C" __global__ void df_swiglu(const __nv_bfloat16* __restrict__ gu, __nv_bfloat16* __restrict__ out, int F) {
    const int r = blockIdx.x;
    const __nv_bfloat16* g = gu + (long long)r * 2 * F;
    for (int c = blockIdx.y * 1024 + threadIdx.x; c < (blockIdx.y + 1) * 1024 && c < F; c += blockDim.x) {
        const float x = bf(g[c]);
        const float s = bf(tobf(x / (1.0f + expf(-x))));
        out[(long long)r * F + c] = tobf(s * bf(g[F + c]));
    }
}

// Each block's 8 rows attend to their ring's window (max(from, p - 2047) .. start - 1) and all 8 block keys.
extern "C" __global__ void __launch_bounds__(256) df_attn(const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ kb,
                                                          const __nv_bfloat16* __restrict__ vb, const uint64_t* __restrict__ rings,
                                                          const long long* __restrict__ starts, const long long* __restrict__ froms,
                                                          int layer, float scale, __nv_bfloat16* __restrict__ o) {
    const int s = blockIdx.x, kh = blockIdx.y;
    const int j = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int r = s * BLOCK + j;
    const long long start = starts[s], p = start + j;
    long long lo = p - (WIN - 1);
    if (lo < froms[s]) lo = froms[s];
    if (lo < 0) lo = 0;
    float qv[4][4], m[4], l[4], acc[4][4];
    for (int h = 0; h < 4; ++h) {
        const __nv_bfloat16* qh = q + (long long)r * QH * HD + (kh * 4 + h) * HD + lane * 4;
        for (int i = 0; i < 4; ++i) qv[h][i] = bf(qh[i]) * scale;
        m[h] = -INFINITY;
        l[h] = 0.f;
        for (int i = 0; i < 4; ++i) acc[h][i] = 0.f;
    }
    const __nv_bfloat16* K = ringK(rings[s], layer);
    const __nv_bfloat16* V = ringV(rings[s], layer);
    const long long n_ctx = start > lo ? start - lo : 0;
    for (long long n = 0; n < n_ctx + BLOCK; ++n) {
        const __nv_bfloat16 *kr, *vr;
        if (n < n_ctx) {
            const int slot = (int)((lo + n) % WIN);
            kr = K + (long long)slot * KVW + kh * HD + lane * 4;
            vr = V + (long long)slot * KVW + kh * HD + lane * 4;
        } else {
            const long long rb = (long long)s * BLOCK + (n - n_ctx);
            kr = kb + rb * KVW + kh * HD + lane * 4;
            vr = vb + rb * KVW + kh * HD + lane * 4;
        }
        float kf[4], vf[4];
        for (int i = 0; i < 4; ++i) {
            kf[i] = bf(kr[i]);
            vf[i] = bf(vr[i]);
        }
        for (int h = 0; h < 4; ++h) {
            float d = qv[h][0] * kf[0] + qv[h][1] * kf[1] + qv[h][2] * kf[2] + qv[h][3] * kf[3];
            for (int off = 16; off > 0; off >>= 1) d += __shfl_xor_sync(0xffffffffu, d, off);
            const float mn = fmaxf(m[h], d);
            const float corr = __expf(m[h] - mn), pe = __expf(d - mn);
            l[h] = l[h] * corr + pe;
            for (int i = 0; i < 4; ++i) acc[h][i] = acc[h][i] * corr + pe * vf[i];
            m[h] = mn;
        }
    }
    for (int h = 0; h < 4; ++h) {
        __nv_bfloat16* oh = o + (long long)r * QH * HD + (kh * 4 + h) * HD + lane * 4;
        const float inv = 1.0f / l[h];
        for (int i = 0; i < 4; ++i) oh[i] = tobf(acc[h][i] * inv);
    }
}

// df_attn split over CH key chunks (running max, sum, fp32 output), merged by df_attn_merge: not latency-bound.
extern "C" __global__ void __launch_bounds__(256) df_attn_part(const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ kb,
        const __nv_bfloat16* __restrict__ vb, const uint64_t* __restrict__ rings, const long long* __restrict__ starts,
        const long long* __restrict__ froms, int layer, float scale, float* __restrict__ pm, float* __restrict__ pl,
        float* __restrict__ pacc) {
    const int s = blockIdx.x, kh = blockIdx.y, c = blockIdx.z;
    const int j = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int r = s * BLOCK + j;
    const long long start = starts[s], p = start + j;
    long long lo = p - (WIN - 1);
    if (lo < froms[s]) lo = froms[s];
    if (lo < 0) lo = 0;
    const long long n_ctx = start > lo ? start - lo : 0;
    const long long T = n_ctx + BLOCK, L = (T + CH - 1) / CH;
    const long long n0 = (long long)c * L, n1 = n0 + L < T ? n0 + L : T;
    float qv[4][4], m[4], l[4], acc[4][4];
    for (int h = 0; h < 4; ++h) {
        const __nv_bfloat16* qh = q + (long long)r * QH * HD + (kh * 4 + h) * HD + lane * 4;
        for (int i = 0; i < 4; ++i) qv[h][i] = bf(qh[i]) * scale;
        m[h] = -INFINITY;
        l[h] = 0.f;
        for (int i = 0; i < 4; ++i) acc[h][i] = 0.f;
    }
    const __nv_bfloat16* K = ringK(rings[s], layer);
    const __nv_bfloat16* V = ringV(rings[s], layer);
    for (long long n = n0; n < n1; ++n) {
        const __nv_bfloat16 *kr, *vr;
        if (n < n_ctx) {
            const int slot = (int)((lo + n) % WIN);
            kr = K + (long long)slot * KVW + kh * HD + lane * 4;
            vr = V + (long long)slot * KVW + kh * HD + lane * 4;
        } else {
            const long long rb = (long long)s * BLOCK + (n - n_ctx);
            kr = kb + rb * KVW + kh * HD + lane * 4;
            vr = vb + rb * KVW + kh * HD + lane * 4;
        }
        float kf[4], vf[4];
        for (int i = 0; i < 4; ++i) {
            kf[i] = bf(kr[i]);
            vf[i] = bf(vr[i]);
        }
        for (int h = 0; h < 4; ++h) {
            float d = qv[h][0] * kf[0] + qv[h][1] * kf[1] + qv[h][2] * kf[2] + qv[h][3] * kf[3];
            for (int off = 16; off > 0; off >>= 1) d += __shfl_xor_sync(0xffffffffu, d, off);
            const float mn = fmaxf(m[h], d);
            const float corr = __expf(m[h] - mn), pe = __expf(d - mn);
            l[h] = l[h] * corr + pe;
            for (int i = 0; i < 4; ++i) acc[h][i] = acc[h][i] * corr + pe * vf[i];
            m[h] = mn;
        }
    }
    for (int h = 0; h < 4; ++h) {
        const long long at = ((long long)r * QH + kh * 4 + h) * CH + c;
        float* pa = pacc + at * HD + lane * 4;
        for (int i = 0; i < 4; ++i) pa[i] = acc[h][i];
        if (lane == 0) {
            pm[at] = m[h];   // -inf with l = 0 for a chunk past the row's keys: the merge weighs it 0
            pl[at] = l[h];
        }
    }
}

// The chunks' partials of row r, head h into o[r][h] (bf16). grid (R, QH), 128 threads (a thread a dim).
extern "C" __global__ void df_attn_merge(const float* __restrict__ pm, const float* __restrict__ pl, const float* __restrict__ pacc,
                                         __nv_bfloat16* __restrict__ o) {
    const int r = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
    const long long base = ((long long)r * QH + h) * CH;
    float m = -INFINITY;
    for (int c = 0; c < CH; ++c) m = fmaxf(m, pm[base + c]);
    float l = 0.f, a = 0.f;
    for (int c = 0; c < CH; ++c) {
        const float w = __expf(pm[base + c] - m);
        l += pl[base + c] * w;
        a += pacc[(base + c) * HD + d] * w;
    }
    o[(long long)r * QH * HD + h * HD + d] = tobf(a / l);
}

// x += y (bf16 rows of D). grid (R, D / 1024), 256 threads.
extern "C" __global__ void df_add(__nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ y) {
    const long long r = blockIdx.x;
    for (int c = blockIdx.y * 1024 + threadIdx.x; c < (blockIdx.y + 1) * 1024; c += blockDim.x)
        x[r * D + c] = tobf(bf(x[r * D + c]) + bf(y[r * D + c]));
}

// out = bf16(g rank 0 + g rank 1): both ranks' gathered fp32 partials [2][R][D] of a split linear.
extern "C" __global__ void df_sum2(const float* __restrict__ g, __nv_bfloat16* __restrict__ out, int R) {
    const long long r = blockIdx.x;
    for (int c = blockIdx.y * 1024 + threadIdx.x; c < (blockIdx.y + 1) * 1024; c += blockDim.x) {
        const long long i = r * D + c;
        out[i] = tobf(g[i] + g[(long long)R * D + i]);
    }
}

// out = bf16(g rank 0 + g rank 1 + g rank 2): three ranks' gathered fp32 partials [3][R][D] (TF_GLM53_DRAFT_SPLIT3).
extern "C" __global__ void df_sum3(const float* __restrict__ g, __nv_bfloat16* __restrict__ out, int R) {
    const long long r = blockIdx.x;
    for (int c = blockIdx.y * 1024 + threadIdx.x; c < (blockIdx.y + 1) * 1024; c += blockDim.x) {
        const long long i = r * D + c;
        out[i] = tobf((g[i] + g[(long long)R * D + i]) + g[2LL * R * D + i]);
    }
}

// A rank's 16 largest logits a head row as global ids and fp32 values: per-chunk part, then merge; ties keep low id.
namespace {
constexpr int CHUNK_COLS = 4096;
constexpr int TOP = 16;

__device__ __forceinline__ bool better(float a, int ia, float b, int ib) { return a > b || (a == b && ia < ib); }

// The block's best (value, index) of sv/si [n] (256 threads), the winner's slot cleared to -inf.
__device__ void takeBest(float* sv, int* si, int n, float* rv, int* ri, float* outv, int* outi) {
    float bv = -INFINITY;
    int bi = 0x7fffffff, bs = -1;
    for (int j = threadIdx.x; j < n; j += blockDim.x)
        if (better(sv[j], si[j], bv, bi)) { bv = sv[j]; bi = si[j]; bs = j; }
    for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
        const int oi = __shfl_xor_sync(0xffffffffu, bi, o), os = __shfl_xor_sync(0xffffffffu, bs, o);
        if (better(ov, oi, bv, bi)) { bv = ov; bi = oi; bs = os; }
    }
    const int w = threadIdx.x >> 5;
    if ((threadIdx.x & 31) == 0) { rv[w] = bv; ri[w] = bi; ri[8 + w] = bs; }
    __syncthreads();
    if (threadIdx.x == 0) {
        int best = 0;
        for (int k = 1; k < (int)(blockDim.x >> 5); ++k)
            if (better(rv[k], ri[k], rv[best], ri[best])) best = k;
        *outv = rv[best];
        *outi = ri[best];
        if (ri[8 + best] >= 0) sv[ri[8 + best]] = -INFINITY;
    }
    __syncthreads();
}
}  // namespace

extern "C" __global__ void __launch_bounds__(256) df_top16_part(const __nv_bfloat16* __restrict__ logits, int cols, int offset,
                                                                float* __restrict__ pv, int* __restrict__ pi) {
    __shared__ float sv[CHUNK_COLS];
    __shared__ int si[CHUNK_COLS];
    __shared__ float rv[8];
    __shared__ int ri[16];
    const int r = blockIdx.x, c0 = blockIdx.y * CHUNK_COLS;
    const int n = min(CHUNK_COLS, cols - c0);
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        sv[j] = bf(logits[(long long)r * cols + c0 + j]);
        si[j] = offset + c0 + j;
    }
    __syncthreads();
    const long long base = ((long long)r * gridDim.y + blockIdx.y) * TOP;
    for (int t = 0; t < TOP; ++t) takeBest(sv, si, n, rv, ri, pv + base + t, pi + base + t);
}

extern "C" __global__ void __launch_bounds__(256) df_top16_merge(const float* __restrict__ pv, const int* __restrict__ pi, int chunks,
                                                                 int* __restrict__ out) {
    __shared__ float sv[64 * TOP];
    __shared__ int si[64 * TOP];
    __shared__ float rv[8];
    __shared__ int ri[16];
    __shared__ float ov[TOP];
    __shared__ int oi[TOP];
    const int r = blockIdx.x, n = chunks * TOP;
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        sv[j] = pv[(long long)r * n + j];
        si[j] = pi[(long long)r * n + j];
    }
    __syncthreads();
    for (int t = 0; t < TOP; ++t) takeBest(sv, si, n, rv, ri, ov + t, oi + t);
    if (threadIdx.x < TOP) {
        out[r * 2 * TOP + threadIdx.x] = oi[threadIdx.x];
        out[r * 2 * TOP + TOP + threadIdx.x] = __float_as_int(ov[threadIdx.x]);
    }
}

// A row's sums over each 64-column group (qmm's xs for a 4-bit linear's input), fp32 [rows][k / 64].
extern "C" __global__ void df_gsum(const __nv_bfloat16* __restrict__ x, int ldx, int k, float* __restrict__ xs) {
    const int r = blockIdx.x, groups = k / 64;
    const __nv_bfloat16* xr = x + (long long)r * ldx;
    for (int g = threadIdx.x; g < groups; g += blockDim.x) {
        float s = 0.f;
        for (int i = 0; i < 64; ++i) s += bf(xr[g * 64 + i]);
        xs[(long long)r * groups + g] = s;
    }
}

// GLM-5.3 DFlash kernels use separate shape-aware entry points; Flash above stays source-identical to the base.

namespace {
constexpr int MAX_HEADS_PER_KV = 8;
constexpr int TOP_MAX = 64;
__device__ __forceinline__ __nv_bfloat16* ringK(uint64_t base, int layer, int win, int kvw) {
    return reinterpret_cast<__nv_bfloat16*>(base) + (long long)layer * 2 * win * kvw;
}
__device__ __forceinline__ __nv_bfloat16* ringV(uint64_t base, int layer, int win, int kvw) { return ringK(base, layer, win, kvw) + (long long)win * kvw; }
}

// A target layer's residual: Flash averages its four streams, while glm_moe_dsa copies one plain hidden row.
extern "C" __global__ void df_tap_glm(const __nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ out, int slot, int mode, int hidden, int ntaps) {
    const int r = blockIdx.x;
    const __nv_bfloat16* xr = x + (long long)r * (mode == 1 ? hidden : 4 * hidden);
    __nv_bfloat16* o = out + (long long)r * ntaps * hidden + (long long)slot * hidden;
    for (int d = blockIdx.y * 1024 + threadIdx.x; d < min((blockIdx.y + 1) * 1024, hidden); d += blockDim.x) {
        if (mode == 1) {
            o[d] = xr[d];
            continue;
        }
        const float a = ((bf(xr[d]) + bf(xr[hidden + d])) + bf(xr[2 * hidden + d])) + bf(xr[3 * hidden + d]);
        o[d] = tobf(a / 4.0f);
    }
}


// The block's queries (fused qkv, stride ldq) normed and rotated at pos[r] into q, keys into kb, values into vb.
extern "C" __global__ void df_block_qkv_glm(const __nv_bfloat16* __restrict__ qkv, int ldq, const __nv_bfloat16* __restrict__ qn,
                                        const __nv_bfloat16* __restrict__ kn, const int* __restrict__ pos, float eps, float theta, int QH, int KVH, int head_dim, int KVW,
                                        __nv_bfloat16* __restrict__ q, __nv_bfloat16* __restrict__ kb, __nv_bfloat16* __restrict__ vb) {
    __shared__ float red[2];
    const int r = blockIdx.x, h = blockIdx.y;
    const __nv_bfloat16* row = qkv + (long long)r * ldq;
    if (h < QH) {
        if (head_dim == 64) normRope64(row + h * head_dim, qn, eps, pos[r], theta, q + (long long)r * QH * head_dim + h * head_dim, red);
        else normRope(row + h * head_dim, qn, eps, pos[r], theta, q + (long long)r * QH * head_dim + h * head_dim, red);
    } else {
        const int kh = h - QH;
        if (head_dim == 64) normRope64(row + QH * head_dim + kh * head_dim, kn, eps, pos[r], theta, kb + (long long)r * KVW + kh * head_dim, red);
        else normRope(row + QH * head_dim + kh * head_dim, kn, eps, pos[r], theta, kb + (long long)r * KVW + kh * head_dim, red);
        const __nv_bfloat16* v = row + QH * head_dim + KVW + kh * head_dim;
        __nv_bfloat16* vo = vb + (long long)r * KVW + kh * head_dim;
        if (head_dim == 64) {
            if (threadIdx.x < 32) { vo[threadIdx.x] = v[threadIdx.x]; vo[threadIdx.x + 32] = v[threadIdx.x + 32]; }
        } else {
            vo[threadIdx.x] = v[threadIdx.x];
            vo[threadIdx.x + 64] = v[threadIdx.x + 64];
        }
    }
}


// Context rows' keys and values (k | v) normed, rotated at pos[r], into rings[r] at slot pos % WIN of ``layer``.
template <int QH_, int KVH_, int WIN_, int HD_>
__device__ __forceinline__ void ctxKv(const __nv_bfloat16* __restrict__ kv, const __nv_bfloat16* __restrict__ kn,
                                     const long long* __restrict__ pos, const uint64_t* __restrict__ rings, int layer, float eps,
                                     float theta, int KVH, int head_dim, int KVW, int WIN) {
    __shared__ float red[2];
    const int r = blockIdx.x, kh = blockIdx.y;
    const long long p = pos[r];
    const int slot = (int)(p & (WIN_ - 1));
    constexpr int KVW_ = KVH_ * HD_;
    const __nv_bfloat16* row = kv + (long long)r * 2 * KVW_;
    if constexpr (HD_ == 64)
        normRope64(row + kh * HD_, kn, eps, p, theta, ringK(rings[r], layer, WIN_, KVW_) + (long long)slot * KVW_ + kh * HD_, red);
    else
        normRope(row + kh * HD_, kn, eps, p, theta, ringK(rings[r], layer, WIN_, KVW_) + (long long)slot * KVW_ + kh * HD_, red);
    __nv_bfloat16* vo = ringV(rings[r], layer, WIN_, KVW_) + (long long)slot * KVW_ + kh * HD_;
    if constexpr (HD_ == 64) {
        if (threadIdx.x < 32) { vo[threadIdx.x] = row[KVW + kh * HD_ + threadIdx.x]; vo[threadIdx.x + 32] = row[KVW + kh * HD_ + threadIdx.x + 32]; }
    } else {
        vo[threadIdx.x] = row[KVW + kh * HD_ + threadIdx.x];
        vo[threadIdx.x + 64] = row[KVW + kh * HD_ + threadIdx.x + 64];
    }
}

extern "C" __global__ void df_ctx_kv_glm(const __nv_bfloat16* __restrict__ kv, const __nv_bfloat16* __restrict__ kn, const long long* __restrict__ pos,
        const uint64_t* __restrict__ rings, int layer, float eps, float theta, int KVH, int head_dim, int KVW, int WIN) {
    if (head_dim == 64 && KVH == 64) ctxKv<64, 64, 2048, 64>(kv, kn, pos, rings, layer, eps, theta, KVH, head_dim, KVW, WIN);
    else ctxKv<32, 8, 2048, 128>(kv, kn, pos, rings, layer, eps, theta, KVH, head_dim, KVW, WIN);
}

extern "C" __global__ void df_ctx_kv_h8_glm(const __nv_bfloat16* __restrict__ kv, const __nv_bfloat16* __restrict__ kn, const long long* __restrict__ pos,
        const uint64_t* __restrict__ rings, int layer, float eps, float theta, int KVH, int head_dim, int KVW, int WIN) {
    ctxKv<64, 8, 2048, 128>(kv, kn, pos, rings, layer, eps, theta, KVH, head_dim, KVW, WIN);
}


// Two-tap dynamic causal conv over a block: y[t] = sum_o (base[o] + dyn[t][o][c / 16]) * z[t - o]; add or store.
extern "C" __global__ void df_conv_glm(const __nv_bfloat16* __restrict__ z, const __nv_bfloat16* __restrict__ dyn, int which,
                                   const __nv_bfloat16* __restrict__ base, __nv_bfloat16* __restrict__ out, int add,
                                   int D, int GROUP, int KERNEL, int BLOCK) {
    const int r = blockIdx.x, t = r % BLOCK;
    const int GROUPS = D / GROUP;
    const __nv_bfloat16* dr = dyn + (long long)r * 2 * KERNEL * GROUPS + which * KERNEL * GROUPS;
    for (int c = blockIdx.y * 1024 + threadIdx.x; c < min((blockIdx.y + 1) * 1024, D); c += blockDim.x) {
        const int g = c / GROUP;
        float y = (bf(base[c]) + bf(dr[g])) * bf(z[(long long)r * D + c]);
        for (int o = 1; o < KERNEL && o <= t; ++o)
            y += (bf(base[(long long)o * D + c]) + bf(dr[o * GROUPS + g])) * bf(z[(long long)(r - o) * D + c]);
        __nv_bfloat16* o = out + (long long)r * D + c;
        *o = add ? tobf(bf(*o) + bf(tobf(y))) : tobf(y);
    }
}


// df_attn split over CH key chunks (running max, sum, fp32 output), merged by df_attn_merge: not latency-bound.
template <int QH_, int KVH_, int WIN_, int HD_>
__device__ __forceinline__ void attnPart(const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ kb,
        const __nv_bfloat16* __restrict__ vb, const uint64_t* __restrict__ rings, const long long* __restrict__ starts,
        const long long* __restrict__ froms, int layer, float scale, int WIN, int BLOCK, int QH, int KVH, int HD, int KVW,
        float* __restrict__ pm, float* __restrict__ pl, float* __restrict__ pacc) {
    const int s = blockIdx.x, kh = blockIdx.y, c = blockIdx.z;
    const int j = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (j >= BLOCK) return;
    const int r = s * BLOCK + j;
    constexpr int HPKV = QH_ / KVH_;
    constexpr int VEC = HD_ / 32;
    constexpr int KVW_ = KVH_ * HD_;
    const long long start = starts[s], p = start + j;
    long long lo = p - (HD_ == 64 ? WIN_ : WIN_ - 1);
    if (lo < froms[s]) lo = froms[s];
    if (lo < 0) lo = 0;
    const long long n_ctx = start > lo ? start - lo : 0;
    const long long T = n_ctx + BLOCK, L = (T + CH - 1) / CH;
    const long long n0 = (long long)c * L, n1 = n0 + L < T ? n0 + L : T;
    float qv[MAX_HEADS_PER_KV][4], m[MAX_HEADS_PER_KV], l[MAX_HEADS_PER_KV], acc[MAX_HEADS_PER_KV][4];
#pragma unroll
    for (int h = 0; h < HPKV; ++h) {
        const __nv_bfloat16* qh = q + (long long)r * QH_ * HD_ + (kh * HPKV + h) * HD_ + lane * VEC;
        for (int i = 0; i < VEC; ++i) qv[h][i] = bf(qh[i]) * scale;
        m[h] = -INFINITY;
        l[h] = 0.f;
        for (int i = 0; i < 4; ++i) acc[h][i] = 0.f;
    }
    const __nv_bfloat16* K = ringK(rings[s], layer, WIN_, KVW_);
    const __nv_bfloat16* V = ringV(rings[s], layer, WIN_, KVW_);
    for (long long n = n0; n < n1; ++n) {
        // DSpark's Qwen block path is causal within the draft block. Preserve
        // the established DFlash2 block attention range for 128-wide heads.
        if constexpr (HD_ == 64) if (n >= n_ctx && n - n_ctx > j) continue;
        const __nv_bfloat16 *kr, *vr;
        if (n < n_ctx) {
            const int slot = (int)((lo + n) & (WIN_ - 1));
            kr = K + (long long)slot * KVW_ + kh * HD_ + lane * VEC;
            vr = V + (long long)slot * KVW_ + kh * HD_ + lane * VEC;
        } else {
            const long long rb = (long long)s * BLOCK + (n - n_ctx);
            kr = kb + rb * KVW_ + kh * HD_ + lane * VEC;
            vr = vb + rb * KVW_ + kh * HD_ + lane * VEC;
        }
        float kf[4], vf[4];
        for (int i = 0; i < VEC; ++i) {
            kf[i] = bf(kr[i]);
            vf[i] = bf(vr[i]);
        }
#pragma unroll
        for (int h = 0; h < HPKV; ++h) {
            float d;
            if constexpr (VEC == 4) d = qv[h][0] * kf[0] + qv[h][1] * kf[1] + qv[h][2] * kf[2] + qv[h][3] * kf[3];
            else d = qv[h][0] * kf[0] + qv[h][1] * kf[1];
            for (int off = 16; off > 0; off >>= 1) d += __shfl_xor_sync(0xffffffffu, d, off);
            const float mn = fmaxf(m[h], d);
            const float corr = __expf(m[h] - mn), pe = __expf(d - mn);
            l[h] = l[h] * corr + pe;
            for (int i = 0; i < VEC; ++i) acc[h][i] = acc[h][i] * corr + pe * vf[i];
            m[h] = mn;
        }
    }
#pragma unroll
    for (int h = 0; h < HPKV; ++h) {
        const long long at = ((long long)r * QH_ + kh * HPKV + h) * CH + c;
        float* pa = pacc + at * HD_ + lane * VEC;
        for (int i = 0; i < VEC; ++i) pa[i] = acc[h][i];
        if (lane == 0) {
            pm[at] = m[h];   // -inf with l = 0 for a chunk past the row's keys: the merge weighs it 0
            pl[at] = l[h];
        }
    }
}

extern "C" __global__ void __launch_bounds__(256) df_attn_part_glm(const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ kb,
        const __nv_bfloat16* __restrict__ vb, const uint64_t* __restrict__ rings, const long long* __restrict__ starts, const long long* __restrict__ froms,
        int layer, float scale, int WIN, int BLOCK, int QH, int KVH, int HD, int KVW, float* __restrict__ pm, float* __restrict__ pl, float* __restrict__ pacc) {
    if (HD == 64 && QH == 64 && KVH == 64) attnPart<64, 64, 2048, 64>(q, kb, vb, rings, starts, froms, layer, scale, WIN, BLOCK, QH, KVH, HD, KVW, pm, pl, pacc);
    else attnPart<32, 8, 2048, 128>(q, kb, vb, rings, starts, froms, layer, scale, WIN, BLOCK, QH, KVH, HD, KVW, pm, pl, pacc);
}

extern "C" __global__ void __launch_bounds__(256) df_attn_part_h8_glm(const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ kb,
        const __nv_bfloat16* __restrict__ vb, const uint64_t* __restrict__ rings, const long long* __restrict__ starts, const long long* __restrict__ froms,
        int layer, float scale, int WIN, int BLOCK, int QH, int KVH, int HD, int KVW, float* __restrict__ pm, float* __restrict__ pl, float* __restrict__ pacc) {
    if (HD == 64 && QH == 64 && KVH == 64) attnPart<64, 64, 2048, 64>(q, kb, vb, rings, starts, froms, layer, scale, WIN, BLOCK, QH, KVH, HD, KVW, pm, pl, pacc);
    else attnPart<64, 8, 2048, 128>(q, kb, vb, rings, starts, froms, layer, scale, WIN, BLOCK, QH, KVH, HD, KVW, pm, pl, pacc);
}


// The chunks' partials of row r, head h into o[r][h] (bf16). grid (R, QH), 128 threads (a thread a dim).
extern "C" __global__ void df_attn_merge_glm(const float* __restrict__ pm, const float* __restrict__ pl, const float* __restrict__ pacc,
                                         __nv_bfloat16* __restrict__ o, int QH, int HD) {
    const int r = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
    const long long base = ((long long)r * QH + h) * CH;
    float m = -INFINITY;
    for (int c = 0; c < CH; ++c) m = fmaxf(m, pm[base + c]);
    float l = 0.f, a = 0.f;
    for (int c = 0; c < CH; ++c) {
        const float w = __expf(pm[base + c] - m);
        l += pl[base + c] * w;
        a += pacc[(base + c) * HD + d] * w;
    }
    o[(long long)r * QH * HD + h * HD + d] = tobf(a / l);
}


// out = bf16(g rank 0 + g rank 1): both ranks' gathered fp32 partials [2][R][D] of a split linear.
extern "C" __global__ void df_sum2_glm(const float* __restrict__ g, __nv_bfloat16* __restrict__ out, int R, int D) {
    const long long r = blockIdx.x;
    for (int c = blockIdx.y * 1024 + threadIdx.x; c < D; c += blockDim.x) {
        const long long i = r * D + c;
        out[i] = tobf(g[i] + g[(long long)R * D + i]);
    }
}

// out = bf16(g rank 0 + g rank 1 + g rank 2): three ranks' gathered fp32 partials [3][R][D] (TF_GLM53_DRAFT_SPLIT3).
extern "C" __global__ void df_sum3_glm(const float* __restrict__ g, __nv_bfloat16* __restrict__ out, int R, int D) {
    const long long r = blockIdx.x;
    const int end = min(D, (int)(blockIdx.y + 1) * 1024);
    for (int c = blockIdx.y * 1024 + threadIdx.x; c < end; c += blockDim.x) {
        const long long i = r * D + c;
        out[i] = tobf((g[i] + g[(long long)R * D + i]) + g[2LL * R * D + i]);
    }
}


extern "C" __global__ void __launch_bounds__(256) df_top16_part_glm(const __nv_bfloat16* __restrict__ logits, int cols, int offset, int TOP,
                                                                float* __restrict__ pv, int* __restrict__ pi) {
    __shared__ float sv[CHUNK_COLS];
    __shared__ int si[CHUNK_COLS];
    __shared__ float rv[8];
    __shared__ int ri[16];
    const int r = blockIdx.x, c0 = blockIdx.y * CHUNK_COLS;
    const int n = min(CHUNK_COLS, cols - c0);
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        sv[j] = bf(logits[(long long)r * cols + c0 + j]);
        si[j] = offset + c0 + j;
    }
    __syncthreads();
    const long long base = ((long long)r * gridDim.y + blockIdx.y) * TOP;
    for (int t = 0; t < TOP; ++t) takeBest(sv, si, n, rv, ri, pv + base + t, pi + base + t);
}

extern "C" __global__ void __launch_bounds__(256) df_top16_merge_glm(const float* __restrict__ pv, const int* __restrict__ pi, int chunks,
                                                                 int TOP, int* __restrict__ out) {
    __shared__ float sv[64 * TOP_MAX];
    __shared__ int si[64 * TOP_MAX];
    __shared__ float rv[8];
    __shared__ int ri[16];
    __shared__ float ov[TOP_MAX];
    __shared__ int oi[TOP_MAX];
    const int r = blockIdx.x, n = chunks * TOP;
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        sv[j] = pv[(long long)r * n + j];
        si[j] = pi[(long long)r * n + j];
    }
    __syncthreads();
    for (int t = 0; t < TOP; ++t) takeBest(sv, si, n, rv, ri, ov + t, oi + t);
    if (threadIdx.x < TOP) {
        out[r * 2 * TOP + threadIdx.x] = oi[threadIdx.x];
        out[r * 2 * TOP + TOP + threadIdx.x] = __float_as_int(ov[threadIdx.x]);
    }
}

// The DSpark block path needs a simple bf16 residual add over an arbitrary row span.
extern "C" __global__ void df_add_glm(__nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ y, int rows, int width) {
    const int row = blockIdx.x;
    const int col = blockIdx.y * blockDim.x + threadIdx.x;
    if (row < rows && col < width) {
        const long long at = (long long)row * width + col;
        x[at] = tobf(bf(x[at]) + bf(y[at]));
    }
}
