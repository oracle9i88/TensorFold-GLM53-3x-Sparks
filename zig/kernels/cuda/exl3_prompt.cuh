// SPDX-License-Identifier: Apache-2.0
// Modified 2026 by Kristian Aaron (Krishando). Original: ashhart/TensorFold@d1018b318 (Apache-2.0).
// Common EXL3 prompt helpers, after @BobClawblaw (#976) and MiaAI-Lab (Apache-2.0, patches 0004 and 0009).
#pragma once
#include "tf_vblock.cuh"

namespace tf_exl3_prompt {
using namespace tf_exl3;
template <int K2, int CB>
__device__ __forceinline__ void decode_tile_p(const uint32_t* tile, int lane, uint32_t (&b0)[2], uint32_t (&b1)[2]) {
    uint32_t w[lane_words<K2>()];
    int word, offset;
    lane_start<K2>(lane, word, offset);
    for (int i = 0; i < lane_words<K2>(); ++i) w[i] = tile[(word + i) % tile_words<K2>()];
    decode_lane<K2, CB>(w, lane, b0, b1);
}
__device__ __forceinline__ void cp_async16(void* dst, const void* src, bool valid) {
    const unsigned d = (unsigned)__cvta_generic_to_shared(dst);
    const int n = valid ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(d), "l"(src), "r"(n));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }
__device__ __forceinline__ void ldmatrix_x4(uint32_t (&a)[4], const half* p) {
    const unsigned s = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]) : "r"(s));
}
}
namespace tf_exl3_prompt {

constexpr int TPM = 64, TNW = 8, TMT = 16 / TNW, TNB = TPM / 8, TT = TNW * 32, TPH = TPM / 2;
constexpr int TBK = 32, TKT = TBK / 16, TSTAGES = 3, TXS = TBK + 8;
constexpr int TWORDS = 16 * TKT * 32;
constexpr int GU_ES = 128 + 4, DN_ES = 256 + 4;                        // the epilogue sums' row strides (floats)
template <int XR>
constexpr int tstage_halves() { return XR * TXS + 2 * TWORDS; }
constexpr int tsmem_bytes() {
    const int gu = TSTAGES * tstage_halves<2 * TPM>() * 2;   // gate and up rows apart (no shared rotation)
    const int ep = 2 * TPH * GU_ES * 4, dn = TPH * DN_ES * 4;
    return gu > ep ? (gu > dn ? gu : dn) : (ep > dn ? ep : dn);
}

// Which (matrix, weight tile) warp-tile q (0..15) of column block cb is.
template <int MATS>
__device__ __forceinline__ void ttile(int q, int cb, int& mat, int& nt) {
    mat = MATS == 2 ? q >> 3 : 0;
    nt = MATS == 2 ? cb * 8 + (q & 7) : cb * 16 + q;
}

// Main loop: acc[i][nb] = warp tile i x members nb * 8 .. + 7 over K; XR rows a stage (TPM if X0 == X1, else 2 TPM).
// K is the inputs this expert has, KX the row stride of X (K unless a span table makes experts' K differ).
template <int MATS, int XR, int CB>
__device__ __forceinline__ void tmain(const half* __restrict__ X0, const half* __restrict__ X1, const uint64_t* __restrict__ tp0,
        const uint64_t* __restrict__ tp1, const int* rows_sh, int cnt, int e, int K, int KX, int NTILES, int cb, int xdiv,
        int k2a, int k2b, half* stage, float (&acc)[TMT][TNB][4]) {
    constexpr int SH = tstage_halves<XR>();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int mat, nt_unused;
    ttile<MATS>(warp * TMT, cb, mat, nt_unused);
    const int chunks = K / TBK;
    const uint32_t* T0 = reinterpret_cast<const uint32_t*>(tp0[e]);
    const uint32_t* T1 = reinterpret_cast<const uint32_t*>(tp1[e]);
    const int nblocks = (cnt + 15) >> 4;   // 16-member blocks in use
#pragma unroll
    for (int i = 0; i < TMT; ++i)
#pragma unroll
        for (int j = 0; j < TNB; ++j)
#pragma unroll
            for (int c = 0; c < 4; ++c) acc[i][j][c] = 0.f;
    auto load = [&](int chunk) {
        half* dst = stage + (size_t)(chunk % TSTAGES) * SH;
        const int k0 = chunk * TBK;
        for (int idx = threadIdx.x; idx < XR * (TBK / 8); idx += TT) {
            const int m = idx / (TPM * (TBK / 8)), rem = idx % (TPM * (TBK / 8)), i = rem / (TBK / 8), j = rem % (TBK / 8);
            if (i >= nblocks * 16) continue;   // a 16-member block without a member: never read
            const int r = rows_sh[i];
            const half* src = (m ? X1 : X0) + (size_t)(r < 0 ? 0 : (r / xdiv)) * KX + k0 + j * 8;
            cp_async16(dst + ((size_t)m * TPM + i) * TXS + j * 8, src, r >= 0);
        }
        uint32_t* wdst = reinterpret_cast<uint32_t*>(dst + (size_t)XR * TXS);
        for (int idx = threadIdx.x; idx < 16 * TKT * 8; idx += TT) {   // stage is sized for K2=8
            const int q = idx / (TKT * 8), kk = (idx / 8) % TKT, part = idx % 8;
            int wm, nt;
            ttile<MATS>(q, cb, wm, nt);
            const int k2 = wm ? k2b : k2a;
            if (part >= k2) continue;   // a 4*k2-word tile: the stage's remaining words are never decoded
            const uint32_t* src = (wm ? T1 : T0) + (((size_t)chunk * TKT + kk) * NTILES + nt) * (4 * k2) + part * 4;
            cp_async16(wdst + ((size_t)q * TKT + kk) * 32 + part * 4, src, true);
        }
    };
#pragma unroll
    for (int c = 0; c < TSTAGES - 1; ++c) {
        if (c < chunks) load(c);
        cp_async_commit();
    }
    for (int c = 0; c < chunks; ++c) {
        cp_async_wait<TSTAGES - 2>();
        TF_TASK_SYNC();
        if (c + TSTAGES - 1 < chunks) load(c + TSTAGES - 1);
        cp_async_commit();
        const half* base = stage + (size_t)(c % TSTAGES) * SH;
        const half* xs = base + (size_t)(XR == TPM ? 0 : mat) * TPM * TXS;
        const uint32_t* ws = reinterpret_cast<const uint32_t*>(base + (size_t)XR * TXS);
#pragma unroll
        for (int kk = 0; kk < TKT; ++kk) {
            uint32_t a[TMT][4];
#pragma unroll
            for (int i = 0; i < TMT; ++i) {
                uint32_t b0[2], b1[2];
                const int k2 = mat ? k2b : k2a;
                switch (k2) {
                    case 2: decode_tile_p<2, CB>(ws + ((size_t)(warp * TMT + i) * TKT + kk) * 32, lane, b0, b1); break;
                    case 3: decode_tile_p<3, CB>(ws + ((size_t)(warp * TMT + i) * TKT + kk) * 32, lane, b0, b1); break;
                    case 4: decode_tile_p<4, CB>(ws + ((size_t)(warp * TMT + i) * TKT + kk) * 32, lane, b0, b1); break;
                    case 5: decode_tile_p<5, CB>(ws + ((size_t)(warp * TMT + i) * TKT + kk) * 32, lane, b0, b1); break;
                    case 6: decode_tile_p<6, CB>(ws + ((size_t)(warp * TMT + i) * TKT + kk) * 32, lane, b0, b1); break;
                    case 7: decode_tile_p<7, CB>(ws + ((size_t)(warp * TMT + i) * TKT + kk) * 32, lane, b0, b1); break;
                    case 8: decode_tile_p<8, CB>(ws + ((size_t)(warp * TMT + i) * TKT + kk) * 32, lane, b0, b1); break;
                    default: b0[0] = b0[1] = b1[0] = b1[1] = 0; break;
                }
                a[i][0] = b0[0]; a[i][1] = b1[0]; a[i][2] = b0[1]; a[i][3] = b1[1];
            }
#pragma unroll
            for (int jb = 0; jb < TPM / 16; ++jb) {
                if (jb < nblocks) {
                    uint32_t r[4];
                    const int row = jb * 16 + (lane & 7) + ((lane >> 4) & 1) * 8;
                    ldmatrix_x4(r, xs + (size_t)row * TXS + kk * 16 + ((lane >> 3) & 1) * 8);
                    const uint32_t bl[2] = {r[0], r[1]}, bh[2] = {r[2], r[3]};
#pragma unroll
                    for (int i = 0; i < TMT; ++i) {
                        mma16816(acc[i][2 * jb], a[i], bl);
                        mma16816(acc[i][2 * jb + 1], a[i], bh);
                    }
                }
            }
        }
    }
    cp_async_wait<0>();
    TF_TASK_SYNC();   // the stages become the epilogue's sums
}

// acc (members TPH * hf .. + TPH - 1) -> E[member - TPH * hf][col] fp32, row stride ES; col0: first tile's column.
__device__ __forceinline__ void tspill(float* E, int ES, int col0, const float (&acc)[TMT][TNB][4], int hf) {
    const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
#pragma unroll
    for (int i = 0; i < TMT; ++i)
#pragma unroll
        for (int q = 0; q < TPM / 16; ++q) {
            const int nb = hf * (TPM / 16) + q;
            const int tok = q * 8 + 2 * t, col = col0 + i * 16 + g;
            E[(size_t)tok * ES + col] = acc[i][nb][0];
            E[(size_t)(tok + 1) * ES + col] = acc[i][nb][1];
            E[(size_t)tok * ES + col + 8] = acc[i][nb][2];
            E[(size_t)(tok + 1) * ES + col + 8] = acc[i][nb][3];
        }
}

template <int XR, int CB>
__device__ __forceinline__ void tgateup(const half* __restrict__ Xg, const half* __restrict__ Xu, const uint64_t* __restrict__ Tg,
        const uint64_t* __restrict__ Tu, const int* __restrict__ items, const int* __restrict__ counts, const int* __restrict__ members,
        const half* __restrict__ svh_g, const half* __restrict__ svh_u, const half* __restrict__ suh_d, half* __restrict__ xd,
        int K, int N, int xdiv, float limit, int act_mode, const int* k2g, const int* k2u, const int* widths,
        unsigned char* smem_raw, int* rows_sh) {
    const int cb = TF_VBID.x, item = TF_VBID.y;
    if (item >= counts[0]) return;
    const int e = items[3 * item], first = items[3 * item + 1], cnt = items[3 * item + 2];
    const int ne = widths ? widths[e] : N;   // this expert's columns; N stays the row stride of svh, suh_d and xd
    if (cb * 128 >= ne) return;
    if (threadIdx.x < TPM) rows_sh[threadIdx.x] = (int)threadIdx.x < cnt ? members[first + threadIdx.x] : -1;
    TF_TASK_SYNC();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    float acc[TMT][TNB][4];
    const int k2g_e = k2g[e], k2u_e = k2u[e];
    tmain<2, XR, CB>(Xg, Xu, Tg, Tu, rows_sh, cnt, e, K, K, ne >> 4, cb, xdiv, k2g_e, k2u_e, reinterpret_cast<half*>(smem_raw), acc);
    float* Eg = reinterpret_cast<float*>(smem_raw);
    int mat, nt;
    ttile<2>(warp * TMT, cb, mat, nt);
    const int col0 = (nt - cb * 8) * 16;
    const int n = cb * 128 + 4 * lane;
    for (int hf = 0; hf < 2 && hf * TPH < cnt; ++hf) {
        if (hf) TF_TASK_SYNC();   // the first half's rows are read
        tspill(Eg + (size_t)mat * TPH * GU_ES, GU_ES, col0, acc, hf);
        TF_TASK_SYNC();
        for (int i = warp; i < TPH && hf * TPH + i < cnt; i += TNW) {
            const int p = rows_sh[hf * TPH + i];
            float gv[4], uv[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                gv[j] = Eg[(size_t)i * GU_ES + 4 * lane + j];
                uv[j] = Eg[(size_t)(TPH + i) * GU_ES + 4 * lane + j];
            }
            fwht128(gv, lane);
            fwht128(uv, lane);
            float v[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                float gg = gv[j] * HAD_SCALE * __half2float(svh_g[(size_t)e * N + n + j]);
                float uu = uv[j] * HAD_SCALE * __half2float(svh_u[(size_t)e * N + n + j]);
                if (act_mode == 0) { gg = bf16r(gg); uu = bf16r(uu); }
                if (limit > 0) { gg = upTo(gg, limit); uu = within(uu, limit); }
                float act = gg / (1.f + expf(-gg));
                if (act_mode == 0) act = bf16r(act);
                act *= uu;
                if (act_mode == 0) act = bf16r(act);
                v[j] = act * __half2float(suh_d[(size_t)e * N + n + j]);
            }
            fwht128(v, lane);
            half* o = xd + (size_t)p * N + n;
#pragma unroll
            for (int j = 0; j < 4; ++j) o[j] = __float2half_rn(v[j] * HAD_SCALE);
        }
    }
}

}  // namespace tf_exl3_prompt

// The wide prompt path (TF_GLM53_PROMPT_EXPERTS=2), ours on the 64-member path above (same credits).
// - The plan's items hold up to 128 members (router tile 128). Items of 64 members or fewer run the 64-member kernels
//   above unchanged (tf_exl3_prompt_*_s entries); larger items run the kernels below, so an expert with 65-128 rows is
//   one decode pass, not two.
// - Below, every kernel is 8 warps (256 threads, 64 fp32 accumulators a thread, 2 CTAs an SM). Each warp owns ONE 16x16
//   weight tile of one matrix and multiplies it into all 16 member n8 tiles: a decoded tile feeds up to 16 MMAs.
// - A CTA covers one 128-column Hadamard block of one matrix: up (wup), gate (wgate) or down (wdown). wup's epilogue
//   stores uu (after its fwht, svh_u and, in act_mode 0, its bf16 rounding: the value tgateup holds) as fp32 [pair][N]
//   in the y region, which down writes only later; wgate's epilogue reads it and finishes SwiGLU and the down rotation.
// - Numerics: the decoded A fragments, the X rows, the stage size (TBK 32), the k16 order (ascending, one m16n8k16 chain
//   from +0 per output) and the epilogue arithmetic are the 64-member path's; uu crosses kernels as an exact fp32. A
//   member's column of B touches no other column and pad members load zeros, so every output has the bits of
//   TF_GLM53_PROMPT_EXPERTS=1 whatever the item's member count or the member's place in it (row invariance).
namespace tf_exl3_prompt {

constexpr int WPM = 128, WNB = WPM / 8, WR = 32;   // members an item, n8 tiles a warp, epilogue rows a round
constexpr int WTILES = 8, WTHREADS = WTILES * 32;  // weight tiles (= warps) a CTA: one 128-column block
template <int XR, int TILES>
constexpr int wstage_halves() { return XR * TXS + 2 * TILES * TKT * 32; }   // X rows, then trellis words sized for K2=8
constexpr int wmax(int a, int b) { return a > b ? a : b; }
constexpr int wsmem_bytes() { return wmax(TSTAGES * wstage_halves<WPM, WTILES>() * 2, WR * GU_ES * 4); }
static_assert(wsmem_bytes() == 36864, "exl3_prompt.zig wide_smem");

// Which (matrix, weight tile) warp tile q of column block cb is: MATS 2 splits the tiles gate | up; MATS 1 one matrix.
template <int MATS, int TILES>
__device__ __forceinline__ void wtile(int q, int cb, int& mat, int& nt) {
    constexpr int PER = MATS == 2 ? TILES / 2 : TILES;
    mat = MATS == 2 ? q / PER : 0;
    nt = cb * PER + q % PER;
}

// Main loop: acc[nb] = this warp's tile x members nb * 8 .. + 7 over K; XR rows a stage (WPM if X0 == X1, else 2 WPM).
// K is the inputs this expert has, KX the row stride of X (as tmain).
template <int MATS, int TILES, int XR, int CB>
__device__ __forceinline__ void wmain(const half* __restrict__ X0, const half* __restrict__ X1, const uint64_t* __restrict__ tp0,
        const uint64_t* __restrict__ tp1, const int* rows_sh, int cnt, int e, int K, int KX, int NTILES, int cb, int xdiv,
        int k2a, int k2b, half* stage, float (&acc)[WNB][4]) {
    constexpr int THREADS = TILES * 32;
    constexpr int SH = wstage_halves<XR, TILES>();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int mat, nt_own;
    wtile<MATS, TILES>(warp, cb, mat, nt_own);
    (void)nt_own;
    const int k2 = mat ? k2b : k2a;
    const int chunks = K / TBK;
    const uint32_t* T0 = reinterpret_cast<const uint32_t*>(tp0[e]);
    const uint32_t* T1 = reinterpret_cast<const uint32_t*>(tp1[e]);
    const int nblocks = (cnt + 15) >> 4;   // 16-member blocks in use
#pragma unroll
    for (int j = 0; j < WNB; ++j)
#pragma unroll
        for (int c = 0; c < 4; ++c) acc[j][c] = 0.f;
    auto load = [&](int chunk) {
        half* dst = stage + (size_t)(chunk % TSTAGES) * SH;
        const int k0 = chunk * TBK;
        for (int idx = threadIdx.x; idx < XR * (TBK / 8); idx += THREADS) {
            const int m = idx / (WPM * (TBK / 8)), rem = idx % (WPM * (TBK / 8)), i = rem / (TBK / 8), j = rem % (TBK / 8);
            if (i >= nblocks * 16) continue;   // a 16-member block without a member: never read
            const int r = rows_sh[i];
            const half* src = (m ? X1 : X0) + (size_t)(r < 0 ? 0 : (r / xdiv)) * KX + k0 + j * 8;
            cp_async16(dst + ((size_t)m * WPM + i) * TXS + j * 8, src, r >= 0);   // pad members: zeros
        }
        uint32_t* wdst = reinterpret_cast<uint32_t*>(dst + (size_t)XR * TXS);
        for (int idx = threadIdx.x; idx < TILES * TKT * 8; idx += THREADS) {
            const int q = idx / (TKT * 8), kk = (idx / 8) % TKT, part = idx % 8;
            int wm, wnt;
            wtile<MATS, TILES>(q, cb, wm, wnt);
            const int qk2 = wm ? k2b : k2a;
            if (part >= qk2) continue;   // a 4*k2-word tile: the stage's remaining words are never decoded
            const uint32_t* src = (wm ? T1 : T0) + (((size_t)chunk * TKT + kk) * NTILES + wnt) * (4 * qk2) + part * 4;
            cp_async16(wdst + ((size_t)q * TKT + kk) * 32 + part * 4, src, true);
        }
    };
#pragma unroll
    for (int c = 0; c < TSTAGES - 1; ++c) {
        if (c < chunks) load(c);
        cp_async_commit();
    }
    for (int c = 0; c < chunks; ++c) {
        cp_async_wait<TSTAGES - 2>();
        TF_TASK_SYNC();
        if (c + TSTAGES - 1 < chunks) load(c + TSTAGES - 1);
        cp_async_commit();
        const half* base = stage + (size_t)(c % TSTAGES) * SH;
        const half* xs = base + (size_t)(XR == WPM ? 0 : mat) * WPM * TXS;
        const uint32_t* ws = reinterpret_cast<const uint32_t*>(base + (size_t)XR * TXS) + (size_t)warp * TKT * 32;
#pragma unroll
        for (int kk = 0; kk < TKT; ++kk) {
            uint32_t b0[2], b1[2];
            switch (k2) {
                case 2: decode_tile_p<2, CB>(ws + kk * 32, lane, b0, b1); break;
                case 3: decode_tile_p<3, CB>(ws + kk * 32, lane, b0, b1); break;
                case 4: decode_tile_p<4, CB>(ws + kk * 32, lane, b0, b1); break;
                case 5: decode_tile_p<5, CB>(ws + kk * 32, lane, b0, b1); break;
                case 6: decode_tile_p<6, CB>(ws + kk * 32, lane, b0, b1); break;
                case 7: decode_tile_p<7, CB>(ws + kk * 32, lane, b0, b1); break;
                case 8: decode_tile_p<8, CB>(ws + kk * 32, lane, b0, b1); break;
                default: b0[0] = b0[1] = b1[0] = b1[1] = 0; break;
            }
            const uint32_t a[4] = {b0[0], b1[0], b0[1], b1[1]};
#pragma unroll
            for (int jb = 0; jb < WPM / 16; ++jb) {
                if (jb < nblocks) {
                    uint32_t r[4];
                    const int row = jb * 16 + (lane & 7) + ((lane >> 4) & 1) * 8;
                    ldmatrix_x4(r, xs + (size_t)row * TXS + kk * 16 + ((lane >> 3) & 1) * 8);
                    const uint32_t bl[2] = {r[0], r[1]}, bh[2] = {r[2], r[3]};
                    mma16816(acc[2 * jb], a, bl);
                    mma16816(acc[2 * jb + 1], a, bh);
                }
            }
        }
    }
    cp_async_wait<0>();
    TF_TASK_SYNC();   // the stages become the epilogue's sums
}

// acc (members WR * rd .. + WR - 1) -> E[member - WR * rd][col] fp32, row stride ES; col0: the tile's first column.
// rd is a constant after the callers' unrolled round loops, so acc stays in registers.
__device__ __forceinline__ void wspill(float* E, int ES, int col0, const float (&acc)[WNB][4], int rd) {
    const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
#pragma unroll
    for (int q = 0; q < WR / 8; ++q) {
        const int nb = rd * (WR / 8) + q;
        const int tok = q * 8 + 2 * t, col = col0 + g;
        E[(size_t)tok * ES + col] = acc[nb][0];
        E[(size_t)(tok + 1) * ES + col] = acc[nb][1];
        E[(size_t)tok * ES + col + 8] = acc[nb][2];
        E[(size_t)(tok + 1) * ES + col + 8] = acc[nb][3];
    }
}

// Shared front of wup/wgate/wdown: the item (64-member items return: the _s kernels own them), its member rows, and
// the main loop over one matrix. Returns the item's member count (0: nothing to do).
// With a span table (TF_GLM53_MOE_BALANCE), wmode 1 (wup, wgate) bounds the expert's columns (blocks past them return
// 0) and wmode 2 (wdown) its inputs; K and N stay the row strides.
template <int CB>
__device__ __forceinline__ int wfront(const half* __restrict__ X, const uint64_t* __restrict__ T, const int* __restrict__ items,
        const int* __restrict__ counts, const int* __restrict__ members, const int* __restrict__ k2s, int K, int N, int xdiv,
        const int* __restrict__ widths, int wmode, unsigned char* smem_raw, int* rows_sh, int& e, float (&acc)[WNB][4]) {
    const int cb = TF_VBID.x, item = TF_VBID.y;
    if (item >= counts[0]) return 0;
    const int cnt = items[3 * item + 2];
    if (cnt <= TPM) return 0;
    e = items[3 * item];
    const int ne = (widths && wmode == 1) ? widths[e] : N;
    const int ke = (widths && wmode == 2) ? widths[e] : K;
    if (cb * 128 >= ne) return 0;
    const int first = items[3 * item + 1];
    if (threadIdx.x < WPM) rows_sh[threadIdx.x] = (int)threadIdx.x < cnt ? members[first + threadIdx.x] : -1;
    TF_TASK_SYNC();
    const int k2 = k2s[e];
    wmain<1, WTILES, WPM, CB>(X, X, T, T, rows_sh, cnt, e, ke, K, ne >> 4, cb, xdiv, k2, k2, reinterpret_cast<half*>(smem_raw), acc);
    return cnt;
}

// Up block (128 columns, item > 64 members): uu[pair][cb * 128 ..] fp32, tgateup's uu before the clamp.
template <int CB>
__device__ __forceinline__ void wup(const half* __restrict__ Xu, const uint64_t* __restrict__ Tu, const int* __restrict__ items,
        const int* __restrict__ counts, const int* __restrict__ members, const half* __restrict__ svh_u, float* __restrict__ uu_out,
        int K, int N, int xdiv, int act_mode, const int* __restrict__ k2u, const int* __restrict__ widths,
        unsigned char* smem_raw, int* rows_sh) {
    float acc[WNB][4];
    int e = 0;
    const int cnt = wfront<CB>(Xu, Tu, items, counts, members, k2u, K, N, xdiv, widths, 1, smem_raw, rows_sh, e, acc);
    if (cnt == 0) return;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    float* Eu = reinterpret_cast<float*>(smem_raw);
    const int n = TF_VBID.x * 128 + 4 * lane;
#pragma unroll
    for (int rd = 0; rd < WPM / WR; ++rd) {
        if (rd * WR < cnt) {   // cnt is the CTA's: every thread takes the same rounds and barriers
            if (rd) TF_TASK_SYNC();   // the previous round's rows are read
            wspill(Eu, GU_ES, warp * 16, acc, rd);
            TF_TASK_SYNC();
            for (int i = warp; i < WR && rd * WR + i < cnt; i += WTILES) {
                const int p = rows_sh[rd * WR + i];
                float uv[4];
#pragma unroll
                for (int j = 0; j < 4; ++j) uv[j] = Eu[(size_t)i * GU_ES + 4 * lane + j];
                fwht128(uv, lane);
                float* o = uu_out + (size_t)p * N + n;
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    float uu = uv[j] * HAD_SCALE * __half2float(svh_u[(size_t)e * N + n + j]);
                    if (act_mode == 0) uu = bf16r(uu);
                    o[j] = uu;
                }
            }
        }
    }
}

// Gate block (128 columns, item > 64 members): xd[pair][cb * 128 ..] = tgateup's epilogue, uu read from wup.
template <int CB>
__device__ __forceinline__ void wgate(const half* __restrict__ Xg, const uint64_t* __restrict__ Tg, const int* __restrict__ items,
        const int* __restrict__ counts, const int* __restrict__ members, const half* __restrict__ svh_g, const half* __restrict__ suh_d,
        const float* __restrict__ uu_in, half* __restrict__ xd, int K, int N, int xdiv, float limit, int act_mode,
        const int* __restrict__ k2g, const int* __restrict__ widths, unsigned char* smem_raw, int* rows_sh) {
    float acc[WNB][4];
    int e = 0;
    const int cnt = wfront<CB>(Xg, Tg, items, counts, members, k2g, K, N, xdiv, widths, 1, smem_raw, rows_sh, e, acc);
    if (cnt == 0) return;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    float* Eg = reinterpret_cast<float*>(smem_raw);
    const int n = TF_VBID.x * 128 + 4 * lane;
#pragma unroll
    for (int rd = 0; rd < WPM / WR; ++rd) {
        if (rd * WR < cnt) {
            if (rd) TF_TASK_SYNC();
            wspill(Eg, GU_ES, warp * 16, acc, rd);
            TF_TASK_SYNC();
            for (int i = warp; i < WR && rd * WR + i < cnt; i += WTILES) {
                const int p = rows_sh[rd * WR + i];
                float gv[4];
#pragma unroll
                for (int j = 0; j < 4; ++j) gv[j] = Eg[(size_t)i * GU_ES + 4 * lane + j];
                fwht128(gv, lane);
                const float* u = uu_in + (size_t)p * N + n;
                float v[4];
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    float gg = gv[j] * HAD_SCALE * __half2float(svh_g[(size_t)e * N + n + j]);
                    float uu = u[j];   // wup's value: already bf16-rounded in act_mode 0 (bf16r is idempotent)
                    if (act_mode == 0) gg = bf16r(gg);
                    if (limit > 0) { gg = upTo(gg, limit); uu = within(uu, limit); }
                    float act = gg / (1.f + expf(-gg));
                    if (act_mode == 0) act = bf16r(act);
                    act *= uu;
                    if (act_mode == 0) act = bf16r(act);
                    v[j] = act * __half2float(suh_d[(size_t)e * N + n + j]);
                }
                fwht128(v, lane);
                half* o = xd + (size_t)p * N + n;
#pragma unroll
                for (int j = 0; j < 4; ++j) o[j] = __float2half_rn(v[j] * HAD_SCALE);
            }
        }
    }
}

// Down block (128 model-width columns, item > 64 members): y[pair][cb * 128 ..], one Hadamard block of the 64-member
// down's two.
template <int CB>
__device__ __forceinline__ void wdown(const half* __restrict__ Xd, const uint64_t* __restrict__ Td, const int* __restrict__ items,
        const int* __restrict__ counts, const int* __restrict__ members, const half* __restrict__ svh_d, float* __restrict__ y,
        int K, int D, const int* __restrict__ k2d, const int* __restrict__ widths, unsigned char* smem_raw, int* rows_sh) {
    float acc[WNB][4];
    int e = 0;
    const int cnt = wfront<CB>(Xd, Td, items, counts, members, k2d, K, D, 1, widths, 2, smem_raw, rows_sh, e, acc);
    if (cnt == 0) return;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    float* Ed = reinterpret_cast<float*>(smem_raw);
    const int n = TF_VBID.x * 128 + 4 * lane;
#pragma unroll
    for (int rd = 0; rd < WPM / WR; ++rd) {
        if (rd * WR < cnt) {
            if (rd) TF_TASK_SYNC();
            wspill(Ed, GU_ES, warp * 16, acc, rd);
            TF_TASK_SYNC();
            for (int i = warp; i < WR && rd * WR + i < cnt; i += WTILES) {
                const int p = rows_sh[rd * WR + i];
                float v[4];
#pragma unroll
                for (int j = 0; j < 4; ++j) v[j] = Ed[(size_t)i * GU_ES + 4 * lane + j];
                fwht128(v, lane);
                float* o = y + (size_t)p * D + n;
#pragma unroll
                for (int j = 0; j < 4; ++j) o[j] = v[j] * HAD_SCALE * __half2float(svh_d[(size_t)e * D + n + j]);
            }
        }
    }
}

}  // namespace tf_exl3_prompt
