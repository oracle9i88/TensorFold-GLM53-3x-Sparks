// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
#include "../tf_vblock.cuh"
// Adapted from bertholomus/bertholomus/TensorFold@ddfad35:src/tensorfold/families/glm_moe_dsa/cuda/mla_pe.py and dcp.py (Apache-2.0).
// DCP merge follows ashhart/TensorFold PR #159 by drowzeys (Apache-2.0). Changes: CUDA K3/K4 kernels for this engine.
// GLM-5.3 MLA K3/K4 kernels. kv_b stays in its checkpoint E4M3 + 128x128 scale form.
// Tensor-core attend follows the absorbed-MLA head-batched layout used by FlashMLA (DeepSeek, MIT) as a design reference; no code copied.
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <math.h>
#include <stdint.h>

#include "../glm_glue_norm.cuh"
#include "mla_index.h"
#include "rope.cuh"
#include "../glm_triton.cuh"

namespace {
constexpr int LD = 512, PD = 64, ND = 192, VD = 256, KR = 448, TOPK = 2048;
constexpr int DECODE_ABSORB_COLUMNS_PER_THREAD = LD / 128;
constexpr int DECODE_EXPAND_TILE = 32;
constexpr int DECODE_V2_EXPAND_C = 32;
constexpr int DECODE_MLA_V2_THREADS = 128;
namespace tf_mla {
__device__ __forceinline__ float bf(__nv_bfloat16 x) { return __bfloat162float(x); }
}
__device__ __forceinline__ __nv_bfloat16 rb(float x) { return __float2bfloat16_rn(x); }
__device__ __forceinline__ float kvb(const uint8_t* w, const float* s, int row, int col) {
  const __nv_fp8_e4m3 v = *reinterpret_cast<const __nv_fp8_e4m3*>(w + (int64_t)row * LD + col);
  return tf_mla::bf(rb((float)v * s[(row / 128) * 4 + col / 128]));
}
}

// proj[row, 576] = kv_a. Every current-chunk row is written: owned rows to latent/rope,
// peer rows to the stage plane at absolute slots. The stage plane also holds fetched history.
// Rope tables are indexed by row within this launch.
__device__ __forceinline__ void tf_glm53_kv_write_task(const __nv_bfloat16* proj, const __nv_bfloat16* norm,
                                               const float* cos, const float* sin, const int64_t* positions,
                                               __nv_bfloat16* latent, __nv_bfloat16* rope,
                                               __nv_bfloat16* stage_latent, __nv_bfloat16* stage_rope,
                                               int rows, int stride, int rank, int world) {
  const int r = TF_VBID.x, t = threadIdx.x;
  const int64_t p = positions[r];
  const int owner = (int)(p % world), slot = glm53_stage_plane_slot(p, world);
  if (owner != rank && stage_latent == nullptr) return;
  __nv_bfloat16* dst_lat = owner == rank ? latent : stage_latent;
  __nv_bfloat16* dst_rope = owner == rank ? rope : stage_rope;
  glm_glue::rmsnorm<512, 512, 4, 4>(proj, stride, norm,
                                    dst_lat + (int64_t)slot * LD, 0, nullptr, 1.e-5f, 0);
  for (int i = t; i < PD / 2; i += TF_VBDIM.x) {
    const float x = tf_mla::bf(proj[(int64_t)r * stride + LD + 2 * i]);
    const float y = tf_mla::bf(proj[(int64_t)r * stride + LD + 2 * i + 1]);
    const float c = cos[r * (PD / 2) + i], s = sin[r * (PD / 2) + i];
    dst_rope[(int64_t)slot * PD + 2 * i] = rb(glm53_rope::even(x, c, y, s));
    dst_rope[(int64_t)slot * PD + 2 * i + 1] = rb(glm53_rope::odd(y, c, x, s));
  }
}

extern "C" __global__ void tf_glm53_kv_write(const __nv_bfloat16* proj, const __nv_bfloat16* norm,
                                               const float* cos, const float* sin, const int64_t* positions,
                                               __nv_bfloat16* latent, __nv_bfloat16* rope,
                                               __nv_bfloat16* stage_latent, __nv_bfloat16* stage_rope,
                                               int rows, int stride, int rank, int world) {
    TF_VBLOCK_ENTER();
  tf_glm53_kv_write_task(proj, norm, cos, sin, positions, latent, rope, stage_latent, stage_rope, rows, stride, rank, world);
}

// q is [rows, local_heads, 256] (q_nope | q_pe); packed q_pe is [rows, local_heads, 64].
__device__ __forceinline__ void tf_glm53_qpe_pack_task(const __nv_bfloat16* q, const float* cos, const float* sin,
                                              __nv_bfloat16* qpe, int rows, int local_heads) {
  const int h = TF_VBID.y, r = TF_VBID.x, i = threadIdx.x;
  if (r >= rows || h >= local_heads || i >= PD / 2) return;
  const int64_t base = ((int64_t)r * local_heads + h) * 256 + ND;
  const float x = tf_mla::bf(q[base + 2 * i]), y = tf_mla::bf(q[base + 2 * i + 1]);
  const float c = cos[r * (PD / 2) + i], s = sin[r * (PD / 2) + i];
  qpe[((int64_t)r * local_heads + h) * PD + 2 * i] = rb(glm53_rope::even(x, c, y, s));
  qpe[((int64_t)r * local_heads + h) * PD + 2 * i + 1] = rb(glm53_rope::odd(y, c, x, s));
}

extern "C" __global__ void tf_glm53_qpe_pack(const __nv_bfloat16* q, const float* cos, const float* sin,
                                              __nv_bfloat16* qpe, int rows, int local_heads) {
    TF_VBLOCK_ENTER();
  tf_glm53_qpe_pack_task(q, cos, sin, qpe, rows, local_heads);
}

__device__ __forceinline__ void tf_glm53_absorb_task(const __nv_bfloat16* q, const uint8_t* kvb_codes,
                                            const float* kvb_scales, __nv_bfloat16* qa,
                                            int rows, int local_heads, int head0) {
  const int t = threadIdx.x, h = TF_VBID.y, r = TF_VBID.z;
  if (r >= rows || h >= local_heads) return;
  const int64_t qbase = ((int64_t)r * local_heads + h) * 256;
  const int gh = head0 + h;
  for (int c = t; c < LD; c += TF_VBDIM.x) {
    float acc = 0.f;
    for (int d = 0; d < ND; ++d)
      acc = __fmaf_rn(tf_mla::bf(q[qbase + d]), kvb(kvb_codes, kvb_scales, gh * KR + d, c), acc);
    qa[((int64_t)r * local_heads + h) * LD + c] = rb(acc);
  }
}

extern "C" __global__ void tf_glm53_absorb(const __nv_bfloat16* q, const uint8_t* kvb_codes,
                                            const float* kvb_scales, __nv_bfloat16* qa,
                                            int rows, int local_heads, int head0) {
    TF_VBLOCK_ENTER();
  tf_glm53_absorb_task(q, kvb_codes, kvb_scales, qa, rows, local_heads, head0);
}

// Decode/verify absorb: one CTA owns one row/head, and lanes span contiguous kv_b columns.
// The k loop and its FMA order are fixed regardless of the number of rows in the launch.
__device__ __forceinline__ void tf_glm53_absorb_decode_task(const __nv_bfloat16* q, const uint8_t* kvb_codes,
                                            const float* kvb_scales, __nv_bfloat16* qa,
                                            int rows, int local_heads, int head0) {
  const int r = TF_VBID.z, h = TF_VBID.y, t = threadIdx.x;
  if (r >= rows || h >= local_heads) return;
  const int threads_per_block = (TF_VBDIM.x / 32) * 32;
  const int64_t qbase = ((int64_t)r * local_heads + h) * 256;
  const int gh = head0 + h;
  const int64_t out_base = ((int64_t)r * local_heads + h) * LD;
  float acc[DECODE_ABSORB_COLUMNS_PER_THREAD] = {};
  #pragma unroll 1
  for (int d = 0; d < ND; ++d) {
    const float qv = tf_mla::bf(q[qbase + d]);
    #pragma unroll
    for (int item = 0; item < DECODE_ABSORB_COLUMNS_PER_THREAD; ++item) {
      const int c = t + item * threads_per_block;
      acc[item] = __fmaf_rn(qv, kvb(kvb_codes, kvb_scales, gh * KR + d, c), acc[item]);
    }
  }
  #pragma unroll
  for (int item = 0; item < DECODE_ABSORB_COLUMNS_PER_THREAD; ++item) {
    const int c = t + item * threads_per_block;
    qa[out_base + c] = rb(acc[item]);
  }
}

extern "C" __global__ void tf_glm53_absorb_decode(const __nv_bfloat16* q, const uint8_t* kvb_codes,
                                            const float* kvb_scales, __nv_bfloat16* qa,
                                            int rows, int local_heads, int head0) {
    TF_VBLOCK_ENTER();
  tf_glm53_absorb_decode_task(q, kvb_codes, kvb_scales, qa, rows, local_heads, head0);
}

// One CTA per head owns the full decode row group. Each weight row is fetched as
// coalesced 16-byte vectors once, then applied to every row's independent FMA chain.
__device__ __forceinline__ void tf_glm53_absorb_decode_v2_task(const __nv_bfloat16* q, const uint8_t* kvb_codes,
                                            const float* kvb_scales, __nv_bfloat16* qa,
                                            int rows, int local_heads, int head0) {
  const int h = TF_VBID.y, t = threadIdx.x;
  if (TF_VBDIM.x != DECODE_MLA_V2_THREADS || rows > 8) __trap();
  if (h >= local_heads) return;
  const int gh = head0 + h;
  __shared__ __align__(16) uint8_t weight_row[LD];
  float acc[8][LD / 128] = {};
  for (int d = 0; d < ND; ++d) {
    const uint4* src = reinterpret_cast<const uint4*>(kvb_codes + (int64_t)(gh * KR + d) * LD);
    if (t < LD / 16) {
      const int vi = t;
      const uint4 x = src[vi];
      reinterpret_cast<uint4*>(weight_row)[vi] = x;
    }
    __syncthreads();
    for (int r = 0; r < rows; ++r) {
      const float qv = tf_mla::bf(q[((int64_t)r * local_heads + h) * 256 + d]);
      #pragma unroll
      for (int item = 0; item < LD / 128; ++item) {
        const int c = t + item * 128;
        const __nv_fp8_e4m3 code = *reinterpret_cast<const __nv_fp8_e4m3*>(weight_row + c);
        const float w = tf_mla::bf(rb((float)code * kvb_scales[((gh * KR + d) / 128) * 4 + c / 128]));
        acc[r][item] = __fmaf_rn(qv, w, acc[r][item]);
      }
    }
    __syncthreads();
  }
  for (int r = 0; r < rows; ++r) {
    const int64_t out_base = ((int64_t)r * local_heads + h) * LD;
    #pragma unroll
    for (int item = 0; item < LD / 128; ++item)
      qa[out_base + t + item * 128] = rb(acc[r][item]);
  }
}

extern "C" __global__ void tf_glm53_absorb_decode_v2(const __nv_bfloat16* q, const uint8_t* kvb_codes,
                                            const float* kvb_scales, __nv_bfloat16* qa,
                                            int rows, int local_heads, int head0) {
  TF_VBLOCK_ENTER();
  tf_glm53_absorb_decode_v2_task(q, kvb_codes, kvb_scales, qa, rows, local_heads, head0);
}

// Split the output columns across CTAs, and stage each CTA's complete weight slab
// before starting the ordered per-output FMA chain.  The absolute kv_b row for a
// head is not always 128-row aligned (448 rows/head), so three scale entries are
// needed for odd global heads to preserve kvb()'s exact block selection.
template <bool absorb_rows_together>
__device__ __forceinline__ void tf_glm53_absorb_decode_v3_task_t(const __nv_bfloat16* q, const uint8_t* kvb_codes,
                                            const float* kvb_scales, __nv_bfloat16* qa,
                                            int rows, int local_heads, int head0) {
  const int h = TF_VBID.y, t = threadIdx.x;
  if (TF_VBDIM.x != 128 || rows > 8) __trap();
  if (h >= local_heads) return;
  const int gh = head0 + h;
  const int column_block = TF_VBID.x;
  const int column0 = column_block * 64;
  const int first_row = gh * KR;
  __shared__ __align__(16) uint8_t weight_slab[ND * 64];
  __shared__ __align__(16) __nv_bfloat16 q_slab[8 * ND];
  __shared__ float scales[3];

  // Each lane copies six aligned 16-byte vectors. The complete slab is issued
  // before the barrier, exposing independent global loads instead of a serial d loop.
  uint4* dst = reinterpret_cast<uint4*>(weight_slab);
  uint4 staged[6];
  #pragma unroll
  for (int j = 0; j < 6; ++j) {
    const int i = t + j * 128;
    const int d = i / 4, vector = i % 4;
    const uint4* src = reinterpret_cast<const uint4*>(kvb_codes + (int64_t)(first_row + d) * LD + column0);
    staged[j] = src[vector];
  }
  #pragma unroll
  for (int j = 0; j < 6; ++j) dst[t + j * 128] = staged[j];
  for (int i = t; i < rows * ND; i += 128)
    q_slab[i] = q[((int64_t)(i / ND) * local_heads + h) * 256 + (i % ND)];
  if (t < 3) scales[t] = kvb_scales[(first_row / 128 + t) * 4 + column0 / 128];
  __syncthreads();

  // Threads 0-63 take rows 0, 2, 4, 6 and 64-127 rows 1, 3, 5, 7. Each weight is decoded once and fed to all of a
  // thread's rows (TF_GLM53_ABSORB_ROWS=1), instead of one full d loop per row; every output still accumulates
  // d = 0..ND-1 in order from 0, with the same operands, so the results are bit-identical either way.
  const int r0 = t / 64;
  const int c = column0 + (t % 64);
  if constexpr (absorb_rows_together) {
    const int mine = (rows - r0 + 1) / 2;
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    #pragma unroll 4
    for (int d = 0; d < ND; ++d) {
      const int scale_index = (first_row + d) / 128 - first_row / 128;
      const __nv_fp8_e4m3 code = *reinterpret_cast<const __nv_fp8_e4m3*>(weight_slab + d * 64 + (t % 64));
      const float w = tf_mla::bf(rb((float)code * scales[scale_index]));
      #pragma unroll
      for (int j = 0; j < 4; ++j)
        if (j < mine) acc[j] = __fmaf_rn(tf_mla::bf(q_slab[(r0 + 2 * j) * ND + d]), w, acc[j]);
    }
    #pragma unroll
    for (int j = 0; j < 4; ++j)
      if (j < mine) qa[((int64_t)(r0 + 2 * j) * local_heads + h) * LD + c] = rb(acc[j]);
    return;
  }
  for (int rr = r0; rr < rows; rr += 2) {
    float acc = 0.f;
    #pragma unroll 1
    for (int d = 0; d < ND; ++d) {
      const int scale_index = (first_row + d) / 128 - first_row / 128;
      const __nv_fp8_e4m3 code = *reinterpret_cast<const __nv_fp8_e4m3*>(weight_slab + d * 64 + (t % 64));
      const float w = tf_mla::bf(rb((float)code * scales[scale_index]));
      acc = __fmaf_rn(tf_mla::bf(q_slab[rr * ND + d]), w, acc);
    }
    qa[((int64_t)rr * local_heads + h) * LD + c] = rb(acc);
  }
}

__device__ __forceinline__ void tf_glm53_absorb_decode_v3_task(const __nv_bfloat16* q, const uint8_t* kvb_codes,
                                            const float* kvb_scales, __nv_bfloat16* qa,
                                            int rows, int local_heads, int head0) {
  tf_glm53_absorb_decode_v3_task_t<false>(q, kvb_codes, kvb_scales, qa, rows, local_heads, head0);
}

extern "C" __global__ void tf_glm53_absorb_decode_v3(const __nv_bfloat16* q, const uint8_t* kvb_codes,
                                            const float* kvb_scales, __nv_bfloat16* qa,
                                            int rows, int local_heads, int head0) {
  TF_VBLOCK_ENTER();
  tf_glm53_absorb_decode_v3_task(q, kvb_codes, kvb_scales, qa, rows, local_heads, head0);
}

// TF_GLM53_ABSORB_ROWS=1: the same kernel, each decoded weight fed to all of a thread's rows (bit-identical).
extern "C" __global__ void tf_glm53_absorb_decode_v3r(const __nv_bfloat16* q, const uint8_t* kvb_codes,
                                            const float* kvb_scales, __nv_bfloat16* qa,
                                            int rows, int local_heads, int head0) {
  TF_VBLOCK_ENTER();
  tf_glm53_absorb_decode_v3_task_t<true>(q, kvb_codes, kvb_scales, qa, rows, local_heads, head0);
}

// DCP decode reads rank-local slots directly. Gather-on-prefill reads global token positions and
// maps every peer key through a stage plane populated at absolute slots (history fetch + current chunk).
__device__ __forceinline__ void tf_glm53_attend_task(const __nv_bfloat16* qa, const __nv_bfloat16* qpe,
                                            const __nv_bfloat16* latent, const __nv_bfloat16* rope,
                                            const __nv_bfloat16* stage_latent, const __nv_bfloat16* stage_rope,
                                            const int64_t* positions, const int32_t* global_tokens,
                                            const int32_t* local_slots, const int32_t* global_counts,
                                            const int32_t* local_counts, const int32_t* dense_len,
                                            __nv_bfloat16* partial, int rows,
                                            int local_heads, int rank, int world, int prefill) {
  const int r = TF_VBID.x, h = TF_VBID.y, d = threadIdx.x;
  if (r >= rows || h >= local_heads) return;
  const int64_t qpos = positions[r];
  const bool dense = qpos < 2048;
  // qpos is authoritative for dense rows; dense_len is part of the shared contract for consumers.
  (void)dense_len;
  const int dense_global = (int)(qpos + 1);
  // Gather-on-prefill owns all heads locally: dense rows read every absolute key from
  // the owner plane, including current-chunk peer rows in stage_latent/stage_rope.
  // Decode keeps DCP rank-local partials and the caller merges them.
  const int n = dense ? (prefill ? dense_global : (dense_global <= rank ? 0 : (dense_global - 1 - rank) / world + 1)) : (prefill ? global_counts[r] : local_counts[r]);
  float maximum = -CUDART_INF_F, denom = 0.f, value0 = 0.f, value1 = 0.f;
  for (int tile = 0; tile < n; tile += 16) {
    const int end = min(n, tile + 16);
    for (int j = tile; j < end; ++j) {
      int64_t keypos;
      int slot;
      const __nv_bfloat16 *kl, *kr;
      if (dense) {
        if (prefill) {
          keypos = j;
          slot = glm53_stage_plane_slot(keypos, world);
          const bool owned = keypos % world == rank;
          kl = owned ? latent : stage_latent; kr = owned ? rope : stage_rope;
        } else {
          slot = j;
          keypos = (int64_t)j * world + rank;
          if (keypos > qpos) continue;
          kl = latent; kr = rope;
        }
      } else if (prefill) {
        keypos = global_tokens[(int64_t)r * TOPK + j];
        if (keypos < 0 || keypos > qpos) continue;
        slot = glm53_stage_plane_slot(keypos, world);
        const bool owned = keypos % world == rank;
        kl = owned ? latent : stage_latent; kr = owned ? rope : stage_rope;
      } else {
        slot = local_slots[(int64_t)r * TOPK + j];
        if (slot < 0) continue;
        keypos = (int64_t)slot * world + rank;
        if (keypos > qpos) continue;
        kl = latent; kr = rope;
      }
      float score = 0.f;
      const int group = h / 32, head = h % 32;
      const int64_t query = local_heads < 32 ? (int64_t)r * local_heads + h : (((int64_t)group * rows + r) * 32 + head);
      for (int k = 0; k < LD; ++k)
        score = __fmaf_rn(tf_mla::bf(qa[query * LD + k]), tf_mla::bf(kl[(int64_t)slot * LD + k]), score);
      for (int k = 0; k < PD; ++k)
        score = __fmaf_rn(tf_mla::bf(qpe[query * PD + k]), tf_mla::bf(kr[(int64_t)slot * PD + k]), score);
      score *= 0.0625f;
      const float next = fmaxf(maximum, score);
      const float alpha = isinf(maximum) ? 0.f : expf(maximum - next);
      const float p = expf(score - next), pb = tf_mla::bf(rb(p));
      denom = denom * alpha + p;
      value0 = value0 * alpha + pb * tf_mla::bf(kl[(int64_t)slot * LD + d]);
      value1 = value1 * alpha + pb * tf_mla::bf(kl[(int64_t)slot * LD + d + TF_VBDIM.x]);
      maximum = next;
    }
  }
  if (local_heads < 32) {
    const int64_t base = ((int64_t)r * local_heads + h) * LD;
    partial[base + d] = rb(denom == 0.f ? 0.f : value0 / denom);
    partial[base + d + TF_VBDIM.x] = rb(denom == 0.f ? 0.f : value1 / denom);
    if (d == 0) {
      const int64_t partial_bytes = (int64_t)rows * local_heads * LD * (int)sizeof(__nv_bfloat16);
      float* group_lse = reinterpret_cast<float*>(reinterpret_cast<unsigned char*>(partial) + partial_bytes);
      group_lse[(int64_t)r * local_heads + h] = denom == 0.f ? -CUDART_INF_F : maximum + logf(denom);
    }
  } else {
    const int group = h / 32, head = h % 32;
    constexpr int group_row_bytes = 32 * LD * sizeof(__nv_bfloat16) + 32 * sizeof(float);
    const int64_t group_byte = (int64_t)group * rows * group_row_bytes;
    const int64_t group_partial = group_byte / (int)sizeof(__nv_bfloat16);
    const int64_t base = group_partial + ((int64_t)r * 32 + head) * LD;
    partial[base + d] = rb(denom == 0.f ? 0.f : value0 / denom);
    partial[base + d + TF_VBDIM.x] = rb(denom == 0.f ? 0.f : value1 / denom);
    if (d == 0) {
      float* group_lse = reinterpret_cast<float*>(reinterpret_cast<unsigned char*>(partial) +
          group_byte + rows * 32 * LD * (int)sizeof(__nv_bfloat16));
      group_lse[(int64_t)r * 32 + head] = denom == 0.f ? -CUDART_INF_F : maximum + logf(denom);
    }
  }
}

extern "C" __global__ void tf_glm53_attend(const __nv_bfloat16* qa, const __nv_bfloat16* qpe,
                                            const __nv_bfloat16* latent, const __nv_bfloat16* rope,
                                            const __nv_bfloat16* stage_latent, const __nv_bfloat16* stage_rope,
                                            const int64_t* positions, const int32_t* global_tokens,
                                            const int32_t* local_slots, const int32_t* global_counts,
                                            const int32_t* local_counts, const int32_t* dense_len,
                                            __nv_bfloat16* partial, int rows,
                                            int local_heads, int rank, int world, int prefill) {
    TF_VBLOCK_ENTER();
  tf_glm53_attend_task(qa, qpe, latent, rope, stage_latent, stage_rope, positions, global_tokens,
      local_slots, global_counts, local_counts, dense_len, partial, rows, local_heads, rank, world, prefill);
}

namespace {
__device__ __forceinline__ void mla_mma16816(float* c, const uint32_t* a, const uint32_t* b) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ uint32_t mla_pack(__nv_bfloat16 lo, __nv_bfloat16 hi) {
  return (uint32_t)__bfloat16_as_ushort(lo) | ((uint32_t)__bfloat16_as_ushort(hi) << 16);
}
}

// Scalar contracts (kvb() addresses [checkpoint_row][checkpoint_col]):
//   absorb:  qa[r,h,j] = sum_k q[r,h,k]  * kvb(gh*KR + k,     j), k in [0,ND),     j in [0,LD)
//   expand: out[r,h,j] = sum_k ol[r,h,k] * kvb(gh*KR + ND+j, k), k in [0,LD),     j in [0,VD)
// Thus each product is X[M,K] * W[K,N]. In row.col m16n8k16, A is X (row-major);
// the shared B tile is [N][K], which is column-major B[K,N]. A/C lanes own row=lane/4
// (and row+8), col=2*(lane%4) (and col+1); B lanes own a column lane/4 and K pairs
// 2*(lane%4) and +8. We load those packed B pairs manually from shared memory (no
// ldmatrix transpose), and store C's four values as [row,col:col+1], [row+8,col:col+1].
// Fixed 16-row tiles make each row's MMA and accumulation order independent of call size.
// Every CTA dequantizes one kv_b 16x8 tile once, then reuses it across its 16 input rows.
template<bool EXPAND>
__device__ __forceinline__ void tf_glm53_prompt_mla_task(const __nv_bfloat16* x, const uint8_t* codes,
    const float* scales, __nv_bfloat16* out, int rows, int local_heads, int head0, unsigned char* smem_raw) {
  constexpr int MT = 16, NT = 8, KT = 16;
  constexpr int THREADS = 32;
  static_assert(THREADS == 32, "prompt MLA launch requires one warp");
  static_assert((MT * KT) % THREADS == 0 && (NT * KT) % THREADS == 0, "prompt MLA shared tiles must divide evenly across the warp");
  __nv_bfloat16* as = reinterpret_cast<__nv_bfloat16*>(smem_raw);
  __nv_bfloat16* bs = as + MT * KT;
  const int lane = threadIdx.x & 31;
  const int group = lane >> 2, pair = lane & 3;
  const int h = blockIdx.y, row0 = blockIdx.z * MT, gh = head0 + h, ntile = blockIdx.x;
  if (h >= local_heads) return;
  constexpr int kdim = EXPAND ? LD : ND;
  constexpr int odim = EXPAND ? VD : LD;
  constexpr int kvrow0 = EXPAND ? ND : 0;
  const int in_stride = EXPAND ? LD : 256;
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
  for (int kt = 0; kt < kdim; kt += KT) {
    for (int i = threadIdx.x; i < MT * KT; i += blockDim.x) {
      const int r = i / KT, k = i % KT, rr = row0 + r;
      as[i] = rr < rows ? x[((int64_t)rr * local_heads + h) * in_stride + kt + k] : __float2bfloat16_rn(0.f);
    }
    for (int i = threadIdx.x; i < NT * KT; i += blockDim.x) {
      // Absorb's source is [K][N], so consecutive lanes should walk adjacent N bytes.
      // Expand's [N][K] access already walks adjacent K bytes with the original mapping.
      const int n = EXPAND ? i / KT : i % NT;
      const int k = EXPAND ? i % KT : i / NT;
      const int col = ntile * NT + n;
      // absorb reads kv_b K rows (row = k, col = output j); expand reads V rows (row = ND + output j, col = k).
      const float w = col < odim ? (EXPAND ? kvb(codes, scales, gh * KR + kvrow0 + col, kt + k) : kvb(codes, scales, gh * KR + kvrow0 + kt + k, col)) : 0.f;
      bs[n * KT + k] = col < odim ? rb(w) : __float2bfloat16_rn(0.f);
    }
    __syncthreads();
    const int k0 = 2 * pair;
    const uint32_t a[4] = {*reinterpret_cast<const uint32_t*>(as + group * KT + k0),
                           *reinterpret_cast<const uint32_t*>(as + (group + 8) * KT + k0),
                           *reinterpret_cast<const uint32_t*>(as + group * KT + k0 + 8),
                           *reinterpret_cast<const uint32_t*>(as + (group + 8) * KT + k0 + 8)};
    const uint32_t b[2] = {*reinterpret_cast<const uint32_t*>(bs + group * KT + k0),
                           *reinterpret_cast<const uint32_t*>(bs + group * KT + k0 + 8)};
    mla_mma16816(acc, a, b);
    __syncthreads();
  }
  const int row = row0 + group, col = ntile * NT + 2 * pair;
  if (row < rows) {
    const int64_t dst = ((int64_t)row * local_heads + h) * odim + col;
    out[dst] = rb(acc[0]); out[dst + 1] = rb(acc[1]);
  }
  if (row + 8 < rows) {
    const int64_t dst = ((int64_t)(row + 8) * local_heads + h) * odim + col;
    out[dst] = rb(acc[2]); out[dst + 1] = rb(acc[3]);
  }
}

extern "C" __global__ void tf_glm53_absorb_prompt_mma(const __nv_bfloat16* q, const uint8_t* kvb_codes,
    const float* kvb_scales, __nv_bfloat16* qa, int rows, int local_heads, int head0) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  tf_glm53_prompt_mla_task<false>(q, kvb_codes, kvb_scales, qa, rows, local_heads, head0, smem_raw);
}

extern "C" __global__ void tf_glm53_expand_v_prompt_mma(const __nv_bfloat16* ol, const uint8_t* kvb_codes,
    const float* kvb_scales, __nv_bfloat16* out, int rows, int local_heads, int head0) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  tf_glm53_prompt_mla_task<true>(ol, kvb_codes, kvb_scales, out, rows, local_heads, head0, smem_raw);
}

constexpr int PROMPT_MLA_BIG_THREADS = 256;

// Grouped prompt MLA: one CTA owns a fixed 64x64 output tile per head. At each K step it stages
// A[64][16] and the dequantized/rounded B[64][16] once, then each of 8 warps computes four
// m16n8k16 tiles from those shared slices. The outer K loop and each tile's MMA order match
// the 16x8 kernel, so no row's arithmetic depends on the CTA's other rows.
template<bool EXPAND>
__device__ __forceinline__ void tf_glm53_prompt_mla_big_task(const __nv_bfloat16* x, const uint8_t* codes,
    const float* scales, __nv_bfloat16* out, int rows, int local_heads, int head0, unsigned char* smem_raw) {
  constexpr int MT = 16, NT = 8, KT = 16, BIG_M = 64, BIG_N = 64, B_STRIDE = KT + 2;
  if (blockDim.x != PROMPT_MLA_BIG_THREADS) { asm volatile("trap;"); return; }
  __nv_bfloat16* as = reinterpret_cast<__nv_bfloat16*>(smem_raw);
  __nv_bfloat16* bs = as + BIG_M * KT;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int group = lane >> 2, pair = lane & 3;
  const int h = blockIdx.y, row0 = blockIdx.z * BIG_M, gh = head0 + h, ntile0 = blockIdx.x;
  if (h >= local_heads) return;
  constexpr int kdim = EXPAND ? LD : ND;
  constexpr int odim = EXPAND ? VD : LD;
  constexpr int kvrow0 = EXPAND ? ND : 0;
  const int in_stride = EXPAND ? LD : 256;
  float acc[4][4] = {};
  for (int kt = 0; kt < kdim; kt += KT) {
    for (int i = tid; i < BIG_M * KT; i += blockDim.x) {
      const int r = i / KT, k = i % KT, rr = row0 + r;
      as[i] = rr < rows ? x[((int64_t)rr * local_heads + h) * in_stride + kt + k] : __float2bfloat16_rn(0.f);
    }
    for (int i = tid; i < BIG_N * KT; i += blockDim.x) {
      const int n = EXPAND ? i / KT : i % BIG_N;
      const int k = EXPAND ? i % KT : i / BIG_N;
      const int col = ntile0 * BIG_N + n;
      const float w = col < odim ? (EXPAND ? kvb(codes, scales, gh * KR + kvrow0 + col, kt + k) : kvb(codes, scales, gh * KR + kvrow0 + kt + k, col)) : 0.f;
      bs[n * B_STRIDE + k] = col < odim ? rb(w) : __float2bfloat16_rn(0.f);
    }
    __syncthreads();
    const int k0 = 2 * pair;
    const int nsub = warp;
    const uint32_t b[2] = {*reinterpret_cast<const uint32_t*>(bs + (nsub * NT + group) * B_STRIDE + k0),
                           *reinterpret_cast<const uint32_t*>(bs + (nsub * NT + group) * B_STRIDE + k0 + 8)};
    #pragma unroll
    for (int part = 0; part < 4; ++part) {
      const int tile = warp + part * 8;
      const int mtile = tile / 8;
      const uint32_t a[4] = {*reinterpret_cast<const uint32_t*>(as + (mtile * MT + group) * KT + k0),
                             *reinterpret_cast<const uint32_t*>(as + (mtile * MT + group + 8) * KT + k0),
                             *reinterpret_cast<const uint32_t*>(as + (mtile * MT + group) * KT + k0 + 8),
                             *reinterpret_cast<const uint32_t*>(as + (mtile * MT + group + 8) * KT + k0 + 8)};
      mla_mma16816(acc[part], a, b);
    }
    __syncthreads();
  }
  #pragma unroll
  for (int part = 0; part < 4; ++part) {
    const int tile = warp + part * 8;
    const int mtile = tile / 8, nsub = tile % 8;
    const int row = row0 + mtile * MT + group;
    const int col = ntile0 * BIG_N + nsub * NT + 2 * pair;
    if (row < rows) {
      const int64_t dst = ((int64_t)row * local_heads + h) * odim + col;
      out[dst] = rb(acc[part][0]); out[dst + 1] = rb(acc[part][1]);
    }
    if (row + 8 < rows) {
      const int64_t dst = ((int64_t)(row + 8) * local_heads + h) * odim + col;
      out[dst] = rb(acc[part][2]); out[dst + 1] = rb(acc[part][3]);
    }
  }
}

extern "C" __global__ void tf_glm53_absorb_prompt_mma_big(const __nv_bfloat16* q, const uint8_t* kvb_codes,
    const float* kvb_scales, __nv_bfloat16* qa, int rows, int local_heads, int head0) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  tf_glm53_prompt_mla_big_task<false>(q, kvb_codes, kvb_scales, qa, rows, local_heads, head0, smem_raw);
}

extern "C" __global__ void tf_glm53_expand_v_prompt_mma_big(const __nv_bfloat16* ol, const uint8_t* kvb_codes,
    const float* kvb_scales, __nv_bfloat16* out, int rows, int local_heads, int head0) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  tf_glm53_prompt_mla_big_task<true>(ol, kvb_codes, kvb_scales, out, rows, local_heads, head0, smem_raw);
}

// Tensor-core attend: one block = one row r and one 32-head group (absorbed MLA: every head of a row attends the
// same keys, so heads are the M dimension). Per 16-key tile: S = Q[32 x 576] . K^T (bf16 MMA, fp32; warp w owns
// output tile w & 3 over K half w >> 2, the halves summed in order), x 0.0625, invalid keys -inf; online softmax per
// head over the tile (m = max(m_old, tile max), alpha = exp(m_old - m), denominator += sum_j exp(s_j - m) in fp32,
// P = bf16(exp(s_j - m)) as the scalar kernel's pb); O[32 x 512] = O * alpha + P . V (V = the keys' latent rows;
// warp w owns columns 64w .. 64w + 63). Output layout, LSE and empty-row rules as tf_glm53_attend.
constexpr int MMA_QS = 576 + 8;              // smem row stride (bf16) of Q and K rows: 1168 B, 16-byte aligned
constexpr int MMA_KT = 16;                   // keys a tile
constexpr int MMA_SMEM = 32 * MMA_QS * 2 + 2 * MMA_KT * MMA_QS * 2 + 2 * 32 * MMA_KT * 4 + 32 * (MMA_KT + 8) * 2 + 3 * 32 * 4 + 4 * MMA_KT * 4;

__device__ __forceinline__ void tf_glm53_attend_mma_task(const __nv_bfloat16* qa, const __nv_bfloat16* qpe,
                                            const __nv_bfloat16* latent, const __nv_bfloat16* rope,
                                            const __nv_bfloat16* stage_latent, const __nv_bfloat16* stage_rope,
                                            const int64_t* positions, const int32_t* global_tokens,
                                            const int32_t* local_slots, const int32_t* global_counts,
                                            const int32_t* local_counts, __nv_bfloat16* partial, int rows,
                                            int local_heads, int rank, int world, int prefill,
                                            float* split_ws, int split_tiles) {
  const int r = TF_VBID.x, group = TF_VBID.y, split = TF_VBID.z, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  if (r >= rows || group >= (local_heads + 31) / 32) return;
  extern __shared__ __align__(16) unsigned char mma_smem[];
  __nv_bfloat16* qs = reinterpret_cast<__nv_bfloat16*>(mma_smem);                 // [32][MMA_QS]
  __nv_bfloat16* ks0 = qs + 32 * MMA_QS;                                            // [2][16][MMA_QS]: double buffer
  float* spart = reinterpret_cast<float*>(ks0 + 2 * MMA_KT * MMA_QS);              // [2][32][16]
  __nv_bfloat16* ps = reinterpret_cast<__nv_bfloat16*>(spart + 2 * 32 * MMA_KT);    // [32][MMA_KT + 8]
  float* hmax = reinterpret_cast<float*>(ps + 32 * (MMA_KT + 8));                   // [32]
  float* hden = hmax + 32;
  float* halpha = hden + 32;
  int* kslot0 = reinterpret_cast<int*>(halpha + 32);                               // [2][16]
  int* kplane0 = kslot0 + 2 * MMA_KT;                                               // [2][16]: 0 owner plane, 1 stage
  const int64_t qrow = local_heads < 32 ? (int64_t)r * local_heads : (((int64_t)group * rows + r) * 32);
  for (int x = tid; x < 32 * 72; x += 256) {   // 72 16-byte pieces a head: 64 latent + 8 rope
    const int h = x / 72, c = x % 72;
    uint4 val;
    if (local_heads < 32 && h >= local_heads) {
      val = make_uint4(0, 0, 0, 0);
    } else {
      const uint4* src = c < 64 ? reinterpret_cast<const uint4*>(qa + (qrow + h) * LD) + c : reinterpret_cast<const uint4*>(qpe + (qrow + h) * PD) + (c - 64);
      val = *src;
    }
    reinterpret_cast<uint4*>(qs + h * MMA_QS)[c] = val;
  }
  if (tid < 32) { hmax[tid] = -CUDART_INF_F; hden[tid] = 0.f; }
  const int64_t qpos = positions[r];
  const bool dense = qpos < 2048;
  const int dense_global = (int)(qpos + 1);
  const int n = dense ? (prefill ? dense_global : (dense_global <= rank ? 0 : (dense_global - 1 - rank) / world + 1)) : (prefill ? global_counts[r] : local_counts[r]);
  // split_tiles > 0 (decode): this block takes key tiles [split * split_tiles, (split + 1) * split_tiles) and leaves
  // its unnormalized O, max and denominator in split_ws for tf_glm53_attend_split_merge; 0: every tile, final output.
  const int first_key = split_tiles > 0 ? split * split_tiles * MMA_KT : 0;
  const int end_key = split_tiles > 0 ? min(n, first_key + split_tiles * MMA_KT) : n;
  const int g = lane >> 2, q = lane & 3;
  float o[2][8][4];
#pragma unroll
  for (int m = 0; m < 2; ++m)
#pragma unroll
    for (int t = 0; t < 8; ++t)
#pragma unroll
      for (int c = 0; c < 4; ++c) o[m][t][c] = 0.f;
  // The key tile at `tile` into buffer b: its 16 slots and planes (threads 0..15), then (after a sync) 16-byte
  // cp.async pieces, zero-filled for skipped keys.
  auto meta = [&](int tile, int b) {
    if (tid < MMA_KT) {
      const int j = tile + tid;
      int slot = -1, plane = 0;
      if (j < end_key) {
        int64_t keypos;
        if (dense) {
          if (prefill) { keypos = j; slot = glm53_stage_plane_slot(keypos, world); plane = keypos % world == rank ? 0 : 1; }
          else { slot = j; keypos = (int64_t)j * world + rank; if (keypos > qpos) slot = -1; }
        } else if (prefill) {
          keypos = global_tokens[(int64_t)r * TOPK + j];
          if (keypos < 0 || keypos > qpos) slot = -1;
          else { slot = glm53_stage_plane_slot(keypos, world); plane = keypos % world == rank ? 0 : 1; }
        } else {
          slot = local_slots[(int64_t)r * TOPK + j];
          if (slot >= 0) { keypos = (int64_t)slot * world + rank; if (keypos > qpos) slot = -1; }
        }
      }
      kslot0[b * MMA_KT + tid] = slot;
      kplane0[b * MMA_KT + tid] = plane;
    }
  };
  auto fetch = [&](int b) {
    __nv_bfloat16* ks = ks0 + b * MMA_KT * MMA_QS;
    for (int x = tid; x < MMA_KT * 72; x += 256) {
      const int j = x / 72, c = x % 72, slot = kslot0[b * MMA_KT + j];
      const __nv_bfloat16* kl = kplane0[b * MMA_KT + j] ? stage_latent : latent;
      const __nv_bfloat16* kr = kplane0[b * MMA_KT + j] ? stage_rope : rope;
      const void* src = slot < 0 ? (const void*)latent : (c < 64 ? (const void*)(reinterpret_cast<const uint4*>(kl + (int64_t)slot * LD) + c) : (const void*)(reinterpret_cast<const uint4*>(kr + (int64_t)slot * PD) + (c - 64)));
      const unsigned d = (unsigned)__cvta_generic_to_shared(reinterpret_cast<uint4*>(ks + j * MMA_QS) + c);
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(d), "l"(src), "r"(slot < 0 ? 0 : 16));
    }
    asm volatile("cp.async.commit_group;\n" ::);
  };
  int buf = 0;
  if (first_key < end_key) { meta(first_key, 0); TF_TASK_SYNC(); fetch(0); }
  for (int tile = first_key; tile < end_key; tile += MMA_KT) {
    const int next = tile + MMA_KT;
    if (next < end_key) meta(next, buf ^ 1);     // the other buffer's last readers finished before the previous PV sync
    TF_TASK_SYNC();
    if (next < end_key) { fetch(buf ^ 1); asm volatile("cp.async.wait_group 1;\n" ::); }
    else asm volatile("cp.async.wait_group 0;\n" ::);
    TF_TASK_SYNC();   // this tile's keys are in ks0[buf]
    const __nv_bfloat16* ks = ks0 + buf * MMA_KT * MMA_QS;
    const int* kslot = kslot0 + buf * MMA_KT;
    {   // S partials: tile (mt, nt) = warp & 3 over K steps [18 * half, 18 * half + 18)
      const int t = warp & 3, half = warp >> 2, mt = t >> 1, nt = t & 1;
      float acc[4] = {0.f, 0.f, 0.f, 0.f};
      const __nv_bfloat16* a0 = qs + (mt * 16 + g) * MMA_QS;
      const __nv_bfloat16* a1 = a0 + 8 * MMA_QS;
      const __nv_bfloat16* b0 = ks + (nt * 8 + g) * MMA_QS;
#pragma unroll 2
      for (int st = 0; st < 18; ++st) {
        const int k0 = (half * 18 + st) * 16 + 2 * q;
        const uint32_t a[4] = {*reinterpret_cast<const uint32_t*>(a0 + k0), *reinterpret_cast<const uint32_t*>(a1 + k0),
                               *reinterpret_cast<const uint32_t*>(a0 + k0 + 8), *reinterpret_cast<const uint32_t*>(a1 + k0 + 8)};
        const uint32_t b[2] = {*reinterpret_cast<const uint32_t*>(b0 + k0), *reinterpret_cast<const uint32_t*>(b0 + k0 + 8)};
        mla_mma16816(acc, a, b);
      }
      float* sp = spart + half * 32 * MMA_KT;
      const int row = mt * 16 + g, col = nt * 8 + 2 * q;
      sp[row * MMA_KT + col] = acc[0];
      sp[row * MMA_KT + col + 1] = acc[1];
      sp[(row + 8) * MMA_KT + col] = acc[2];
      sp[(row + 8) * MMA_KT + col + 1] = acc[3];
    }
    TF_TASK_SYNC();
    if (tid < 32) {   // head tid's online softmax over the tile
      const int h = tid;
      float s[MMA_KT];
      float m = hmax[h];
#pragma unroll
      for (int j = 0; j < MMA_KT; ++j) {
        s[j] = kslot[j] < 0 ? -CUDART_INF_F : (spart[h * MMA_KT + j] + spart[32 * MMA_KT + h * MMA_KT + j]) * 0.0625f;
        m = fmaxf(m, s[j]);
      }
      const float alpha = isinf(hmax[h]) ? 0.f : expf(hmax[h] - m);
      float sum = 0.f;
#pragma unroll
      for (int j = 0; j < MMA_KT; ++j) {
        const float p = isinf(s[j]) ? 0.f : expf(s[j] - m);
        sum += p;
        ps[h * (MMA_KT + 8) + j] = rb(p);
      }
      if (isinf(m)) { halpha[h] = 1.f; }   // nothing valid yet: O and the denominator stay zero
      else { halpha[h] = alpha; hden[h] = hden[h] * alpha + sum; hmax[h] = m; }
    }
    TF_TASK_SYNC();
    {   // O = O * alpha + P . V over this warp's 64 columns
      const float al0 = halpha[g], al1 = halpha[g + 8], al2 = halpha[16 + g], al3 = halpha[24 + g];
      uint32_t a[2][4];
#pragma unroll
      for (int mt = 0; mt < 2; ++mt) {
        const __nv_bfloat16* p0 = ps + (mt * 16 + g) * (MMA_KT + 8) + 2 * q;
        const __nv_bfloat16* p1 = p0 + 8 * (MMA_KT + 8);
        a[mt][0] = *reinterpret_cast<const uint32_t*>(p0);
        a[mt][1] = *reinterpret_cast<const uint32_t*>(p1);
        a[mt][2] = *reinterpret_cast<const uint32_t*>(p0 + 8);
        a[mt][3] = *reinterpret_cast<const uint32_t*>(p1 + 8);
      }
#pragma unroll
      for (int t = 0; t < 8; ++t) {
        const int col = warp * 64 + t * 8 + g;
        const uint32_t b[2] = {mla_pack(ks[(2 * q) * MMA_QS + col], ks[(2 * q + 1) * MMA_QS + col]),
                               mla_pack(ks[(2 * q + 8) * MMA_QS + col], ks[(2 * q + 9) * MMA_QS + col])};
        o[0][t][0] *= al0; o[0][t][1] *= al0; o[0][t][2] *= al1; o[0][t][3] *= al1;
        o[1][t][0] *= al2; o[1][t][1] *= al2; o[1][t][2] *= al3; o[1][t][3] *= al3;
        mla_mma16816(o[0][t], a[0], b);
        mla_mma16816(o[1][t], a[1], b);
      }
    }
    buf ^= 1;
  }
  TF_TASK_SYNC();
  if (split_tiles > 0) {   // [group][row][split][head][LD + 2]: O (relative to the head's max), max, denominator
    const int splits = TF_VNBLK.z;
    float* w = split_ws + (((int64_t)group * rows + r) * splits + split) * 32 * (LD + 2);
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        const int head = mt * 16 + g + 8 * hh;
#pragma unroll
        for (int t = 0; t < 8; ++t) {
          const int col = warp * 64 + t * 8 + 2 * q;
          w[head * (LD + 2) + col] = o[mt][t][2 * hh];
          w[head * (LD + 2) + col + 1] = o[mt][t][2 * hh + 1];
        }
      }
    if (tid < 32) { w[tid * (LD + 2) + LD] = hmax[tid]; w[tid * (LD + 2) + LD + 1] = hden[tid]; }
    return;
  }
  if (local_heads < 32) {
    const int64_t partial_bytes = (int64_t)rows * local_heads * LD * (int)sizeof(__nv_bfloat16);
    float* group_lse = reinterpret_cast<float*>(reinterpret_cast<unsigned char*>(partial) + partial_bytes);
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        const int head = mt * 16 + g + 8 * hh;
        if (head < local_heads) {
          const float den = hden[head];
          __nv_bfloat16* dst = partial + ((int64_t)r * local_heads + head) * LD;
#pragma unroll
          for (int t = 0; t < 8; ++t) {
            const int col = warp * 64 + t * 8 + 2 * q;
            dst[col] = rb(den == 0.f ? 0.f : o[mt][t][2 * hh] / den);
            dst[col + 1] = rb(den == 0.f ? 0.f : o[mt][t][2 * hh + 1] / den);
          }
        }
      }
    if (tid < local_heads) {
      group_lse[(int64_t)r * local_heads + tid] = hden[tid] == 0.f ? -CUDART_INF_F : hmax[tid] + logf(hden[tid]);
    }
  } else {
    constexpr int group_row_bytes = 32 * LD * sizeof(__nv_bfloat16) + 32 * sizeof(float);
    const int64_t group_byte = (int64_t)group * rows * group_row_bytes;
    const int64_t group_partial = group_byte / (int)sizeof(__nv_bfloat16);
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        const int head = mt * 16 + g + 8 * hh;
        const float den = hden[head];
        __nv_bfloat16* dst = partial + group_partial + ((int64_t)r * 32 + head) * LD;
#pragma unroll
        for (int t = 0; t < 8; ++t) {
          const int col = warp * 64 + t * 8 + 2 * q;
          dst[col] = rb(den == 0.f ? 0.f : o[mt][t][2 * hh] / den);
          dst[col + 1] = rb(den == 0.f ? 0.f : o[mt][t][2 * hh + 1] / den);
        }
      }
    if (tid < 32) {
      float* group_lse = reinterpret_cast<float*>(reinterpret_cast<unsigned char*>(partial) + group_byte + rows * 32 * LD * (int)sizeof(__nv_bfloat16));
      group_lse[(int64_t)r * 32 + tid] = hden[tid] == 0.f ? -CUDART_INF_F : hmax[tid] + logf(hden[tid]);
    }
  }
}

extern "C" __global__ void __launch_bounds__(256) tf_glm53_attend_mma(const __nv_bfloat16* qa, const __nv_bfloat16* qpe,
                                            const __nv_bfloat16* latent, const __nv_bfloat16* rope,
                                            const __nv_bfloat16* stage_latent, const __nv_bfloat16* stage_rope,
                                            const int64_t* positions, const int32_t* global_tokens,
                                            const int32_t* local_slots, const int32_t* global_counts,
                                            const int32_t* local_counts, const int32_t* dense_len,
                                            __nv_bfloat16* partial, int rows,
                                            int local_heads, int rank, int world, int prefill,
                                            float* split_ws, int split_tiles) {
  TF_VBLOCK_ENTER();
  (void)dense_len;
  tf_glm53_attend_mma_task(qa, qpe, latent, rope, stage_latent, stage_rope, positions, global_tokens, local_slots,
      global_counts, local_counts, partial, rows, local_heads, rank, world, prefill, split_ws, split_tiles);
}

// The decode splits of one (row, head) combined in split order: M = max m_s, L = sum l_s e^(m_s - M),
// O = (sum O_s e^(m_s - M)) / L -> BF16, LSE = M + log L (L == 0: zeros and -inf, as tf_glm53_attend).
extern "C" __global__ void __launch_bounds__(256) tf_glm53_attend_split_merge(const float* split_ws, __nv_bfloat16* partial,
                                            int rows, int local_heads, int splits) {
  TF_VBLOCK_ENTER();
  const int r = TF_VBID.x, h = TF_VBID.y, d = threadIdx.x;
  if (r >= rows || h >= local_heads) return;
  const int group = h / 32, head = h % 32;
  const float* w = split_ws + ((int64_t)group * rows + r) * splits * 32 * (LD + 2) + head * (LD + 2);
  const int64_t stride = 32 * (LD + 2);
  float m = -CUDART_INF_F;
  for (int s = 0; s < splits; ++s) m = fmaxf(m, w[s * stride + LD]);
  float l = 0.f, v0 = 0.f, v1 = 0.f;
  if (!isinf(m)) {
    for (int s = 0; s < splits; ++s) {
      const float ms = w[s * stride + LD];
      if (isinf(ms)) continue;
      const float e = expf(ms - m);
      l += w[s * stride + LD + 1] * e;
      v0 += w[s * stride + d] * e;
      v1 += w[s * stride + d + 256] * e;
    }
  }
  if (local_heads < 32) {
    __nv_bfloat16* dst = partial + ((int64_t)r * local_heads + h) * LD;
    dst[d] = rb(l == 0.f ? 0.f : v0 / l);
    dst[d + 256] = rb(l == 0.f ? 0.f : v1 / l);
    if (d == 0) {
      const int64_t partial_bytes = (int64_t)rows * local_heads * LD * (int)sizeof(__nv_bfloat16);
      float* group_lse = reinterpret_cast<float*>(reinterpret_cast<unsigned char*>(partial) + partial_bytes);
      group_lse[(int64_t)r * local_heads + h] = l == 0.f ? -CUDART_INF_F : m + logf(l);
    }
  } else {
    constexpr int group_row_bytes = 32 * LD * sizeof(__nv_bfloat16) + 32 * sizeof(float);
    const int64_t group_byte = (int64_t)group * rows * group_row_bytes;
    __nv_bfloat16* dst = partial + group_byte / (int)sizeof(__nv_bfloat16) + ((int64_t)r * 32 + head) * LD;
    dst[d] = rb(l == 0.f ? 0.f : v0 / l);
    dst[d + 256] = rb(l == 0.f ? 0.f : v1 / l);
    if (d == 0) {
      float* group_lse = reinterpret_cast<float*>(reinterpret_cast<unsigned char*>(partial) + group_byte + rows * 32 * LD * (int)sizeof(__nv_bfloat16));
      group_lse[(int64_t)r * 32 + head] = l == 0.f ? -CUDART_INF_F : m + logf(l);
    }
  }
}

// A3 batches two independent decode splits per CTA: (row, 32-head group, chunk), with 16 chunks.
// Empty splits write only max/denominator; split_merge skips O whenever max is infinite.
// Each nonempty split runs attend_mma's per-row tiles and reduction unchanged, so its split record and the merged row
// are byte-identical to attend_mma + split_merge. Per-row math: tf_glm53_attend_mma (this file); no code from SMLA-P.
constexpr int VERIFY_SPLITS = 32;
constexpr int VERIFY_SPLIT_KEYS = 4 * MMA_KT;
constexpr int VERIFY_CHUNKS = 16;
constexpr int VERIFY_SPC = VERIFY_SPLITS / VERIFY_CHUNKS;
constexpr int VERIFY_SMEM = MMA_SMEM + VERIFY_SPC * VERIFY_SPLIT_KEYS * 4;
static_assert(VERIFY_SPLITS * VERIFY_SPLIT_KEYS == TOPK && VERIFY_SPC * VERIFY_CHUNKS == VERIFY_SPLITS, "decode verify split layout");
static_assert(VERIFY_SMEM == 81536, "mla.zig attend_verify_smem mirrors this layout");
static_assert(VERIFY_SMEM <= 99 * 1024, "must fit the sm_120/sm_121 opt-in shared memory");

extern "C" __global__ void __launch_bounds__(256) tf_glm53_attend_verify(const __nv_bfloat16* qa, const __nv_bfloat16* qpe,
                                            const __nv_bfloat16* latent, const __nv_bfloat16* rope,
                                            const int64_t* positions, const int32_t* local_slots,
                                            const int32_t* local_counts, float* split_ws,
                                            int rows, int local_heads, int rank, int world) {
  const int r = blockIdx.x, group = blockIdx.y, chunk = blockIdx.z, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  if (r >= rows || group >= (local_heads + 31) / 32) return;
  extern __shared__ __align__(16) unsigned char mma_smem[];
  __nv_bfloat16* qs = reinterpret_cast<__nv_bfloat16*>(mma_smem);
  __nv_bfloat16* ks0 = qs + 32 * MMA_QS;
  float* spart = reinterpret_cast<float*>(ks0 + 2 * MMA_KT * MMA_QS);
  __nv_bfloat16* ps = reinterpret_cast<__nv_bfloat16*>(spart + 2 * 32 * MMA_KT);
  float* hmax = reinterpret_cast<float*>(ps + 32 * (MMA_KT + 8));
  float* hden = hmax + 32;
  float* halpha = hden + 32;
  int* kslot0 = reinterpret_cast<int*>(halpha + 32);
  int* kplane0 = kslot0 + 2 * MMA_KT;
  int* vslot = reinterpret_cast<int*>(mma_smem + MMA_SMEM);
  const int64_t qpos = positions[r];
  const bool dense = qpos < 2048;
  const int dense_global = (int)(qpos + 1);
  const int n = dense ? (dense_global <= rank ? 0 : (dense_global - 1 - rank) / world + 1) : local_counts[r];
  const int s0 = chunk, s1 = chunk + VERIFY_CHUNKS;
  const int first0 = s0 * VERIFY_SPLIT_KEYS, first1 = s1 * VERIFY_SPLIT_KEYS;
  const int end0 = min(n, first0 + VERIFY_SPLIT_KEYS), end1 = min(n, first1 + VERIFY_SPLIT_KEYS);
  const int nt0 = first0 < end0 ? (end0 - first0 + MMA_KT - 1) / MMA_KT : 0;
  const int nt1 = first1 < end1 ? (end1 - first1 + MMA_KT - 1) / MMA_KT : 0;
  float* w0 = split_ws + (((int64_t)group * rows + r) * VERIFY_SPLITS + s0) * 32 * (LD + 2);
  float* w1 = split_ws + (((int64_t)group * rows + r) * VERIFY_SPLITS + s1) * 32 * (LD + 2);
  if (nt0 == 0 && tid < 32) { w0[tid * (LD + 2) + LD] = -CUDART_INF_F; w0[tid * (LD + 2) + LD + 1] = 0.f; }
  if (nt1 == 0 && tid < 32) { w1[tid * (LD + 2) + LD] = -CUDART_INF_F; w1[tid * (LD + 2) + LD + 1] = 0.f; }
  const int total = nt0 + nt1;
  if (total == 0) return;
  for (int x = tid; x < 2 * VERIFY_SPLIT_KEYS; x += 256) {
    const int k = x / VERIFY_SPLIT_KEYS, e = x % VERIFY_SPLIT_KEYS, j = (k ? first1 : first0) + e;
    const int end = k ? end1 : end0;
    int slot = -1;
    if (j < end) {
      if (dense) { slot = j; const int64_t keypos = (int64_t)j * world + rank; if (keypos > qpos) slot = -1; }
      else { slot = local_slots[(int64_t)r * TOPK + j]; if (slot >= 0) { const int64_t keypos = (int64_t)slot * world + rank; if (keypos > qpos) slot = -1; } }
    }
    vslot[x] = slot;
  }
  __syncthreads();
  auto offset = [&](int i) { return i < nt0 ? i * MMA_KT : VERIFY_SPLIT_KEYS + (i - nt0) * MMA_KT; };
  auto meta = [&](int i, int b) {
    if (tid < MMA_KT) { kslot0[b * MMA_KT + tid] = vslot[offset(i) + tid]; kplane0[b * MMA_KT + tid] = 0; }
  };
  auto fetch = [&](int b) {
    __nv_bfloat16* ks = ks0 + b * MMA_KT * MMA_QS;
    for (int x = tid; x < MMA_KT * 72; x += 256) {
      const int j = x / 72, c = x % 72, slot = kslot0[b * MMA_KT + j];
      const __nv_bfloat16* kl = latent; const __nv_bfloat16* kr = rope;
      const void* src = slot < 0 ? (const void*)latent : (c < 64 ? (const void*)(reinterpret_cast<const uint4*>(kl + (int64_t)slot * LD) + c) : (const void*)(reinterpret_cast<const uint4*>(kr + (int64_t)slot * PD) + (c - 64)));
      const unsigned d = (unsigned)__cvta_generic_to_shared(reinterpret_cast<uint4*>(ks + j * MMA_QS) + c);
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(d), "l"(src), "r"(slot < 0 ? 0 : 16));
    }
    asm volatile("cp.async.commit_group;\n" ::);
  };
  meta(0, 0); __syncthreads(); fetch(0);
  const int64_t qrow = local_heads < 32 ? (int64_t)r * local_heads : (((int64_t)group * rows + r) * 32);
  for (int x = tid; x < 32 * 72; x += 256) {
    const int h = x / 72, c = x % 72;
    uint4 val;
    if (local_heads < 32 && h >= local_heads) {
      val = make_uint4(0, 0, 0, 0);
    } else {
      const uint4* src = c < 64 ? reinterpret_cast<const uint4*>(qa + (qrow + h) * LD) + c : reinterpret_cast<const uint4*>(qpe + (qrow + h) * PD) + (c - 64);
      val = *src;
    }
    reinterpret_cast<uint4*>(qs + h * MMA_QS)[c] = val;
  }
  if (tid < 32) { hmax[tid] = -CUDART_INF_F; hden[tid] = 0.f; }
  const int g = lane >> 2, q = lane & 3;
  float o[2][8][4];
#pragma unroll
  for (int m = 0; m < 2; ++m) for (int t = 0; t < 8; ++t) for (int c = 0; c < 4; ++c) o[m][t][c] = 0.f;
  int buf = 0;
  for (int i = 0; i < total; ++i) {
    const int next = i + 1;
    if (next < total) meta(next, buf ^ 1);
    __syncthreads();
    if (next < total) { fetch(buf ^ 1); asm volatile("cp.async.wait_group 1;\n" ::); } else asm volatile("cp.async.wait_group 0;\n" ::);
    __syncthreads();
    const __nv_bfloat16* ks = ks0 + buf * MMA_KT * MMA_QS;
    const int* kslot = kslot0 + buf * MMA_KT;
    { const int t = warp & 3, half = warp >> 2, mt = t >> 1, nt = t & 1; float acc[4] = {0.f,0.f,0.f,0.f};
      const __nv_bfloat16* a0 = qs + (mt * 16 + g) * MMA_QS; const __nv_bfloat16* a1 = a0 + 8 * MMA_QS; const __nv_bfloat16* b0 = ks + (nt * 8 + g) * MMA_QS;
#pragma unroll 2
      for (int st = 0; st < 18; ++st) { const int k0 = (half * 18 + st) * 16 + 2 * q;
        const uint32_t a[4] = {*reinterpret_cast<const uint32_t*>(a0+k0),*reinterpret_cast<const uint32_t*>(a1+k0),*reinterpret_cast<const uint32_t*>(a0+k0+8),*reinterpret_cast<const uint32_t*>(a1+k0+8)};
        const uint32_t b[2] = {*reinterpret_cast<const uint32_t*>(b0+k0),*reinterpret_cast<const uint32_t*>(b0+k0+8)}; mla_mma16816(acc,a,b); }
      float* sp=spart+half*32*MMA_KT; const int row=mt*16+g,col=nt*8+2*q; sp[row*MMA_KT+col]=acc[0]; sp[row*MMA_KT+col+1]=acc[1]; sp[(row+8)*MMA_KT+col]=acc[2]; sp[(row+8)*MMA_KT+col+1]=acc[3]; }
    __syncthreads();
    if (tid < 32) { const int h=tid; float s[MMA_KT]; float m=hmax[h];
#pragma unroll
      for(int j=0;j<MMA_KT;++j){s[j]=kslot[j]<0?-CUDART_INF_F:(spart[h*MMA_KT+j]+spart[32*MMA_KT+h*MMA_KT+j])*0.0625f;m=fmaxf(m,s[j]);}
      const float alpha=isinf(hmax[h])?0.f:expf(hmax[h]-m);float sum=0.f;
#pragma unroll
      for(int j=0;j<MMA_KT;++j){const float p=isinf(s[j])?0.f:expf(s[j]-m);sum+=p;ps[h*(MMA_KT+8)+j]=rb(p);}
      if(isinf(m)){halpha[h]=1.f;}else{halpha[h]=alpha;hden[h]=hden[h]*alpha+sum;hmax[h]=m;} }
    __syncthreads();
    { const float al0=halpha[g],al1=halpha[g+8],al2=halpha[16+g],al3=halpha[24+g];uint32_t a[2][4];
#pragma unroll
      for(int mt=0;mt<2;++mt){const __nv_bfloat16* p0=ps+(mt*16+g)*(MMA_KT+8)+2*q;const __nv_bfloat16* p1=p0+8*(MMA_KT+8);a[mt][0]=*reinterpret_cast<const uint32_t*>(p0);a[mt][1]=*reinterpret_cast<const uint32_t*>(p1);a[mt][2]=*reinterpret_cast<const uint32_t*>(p0+8);a[mt][3]=*reinterpret_cast<const uint32_t*>(p1+8);}
#pragma unroll
      for(int t=0;t<8;++t){const int col=warp*64+t*8+g;const uint32_t b[2]={mla_pack(ks[(2*q)*MMA_QS+col],ks[(2*q+1)*MMA_QS+col]),mla_pack(ks[(2*q+8)*MMA_QS+col],ks[(2*q+9)*MMA_QS+col])};o[0][t][0]*=al0;o[0][t][1]*=al0;o[0][t][2]*=al1;o[0][t][3]*=al1;o[1][t][0]*=al2;o[1][t][1]*=al2;o[1][t][2]*=al3;o[1][t][3]*=al3;mla_mma16816(o[0][t],a[0],b);mla_mma16816(o[1][t],a[1],b);} }
    buf ^= 1;
    if (next == nt0 || next == total) {
      __syncthreads(); float* w = i < nt0 ? w0 : w1;
#pragma unroll
      for(int mt=0;mt<2;++mt) for(int hh=0;hh<2;++hh){const int head=mt*16+g+8*hh;
#pragma unroll
        for(int t=0;t<8;++t){const int col=warp*64+t*8+2*q;w[head*(LD+2)+col]=o[mt][t][2*hh];w[head*(LD+2)+col+1]=o[mt][t][2*hh+1];}}
      if(tid<32){w[tid*(LD+2)+LD]=hmax[tid];w[tid*(LD+2)+LD+1]=hden[tid];}
#pragma unroll
      for(int mt=0;mt<2;++mt) for(int t=0;t<8;++t) for(int c=0;c<4;++c)o[mt][t][c]=0.f;
      if(tid<32){hmax[tid]=-CUDART_INF_F;hden[tid]=0.f;}
    }
  }
}

// SMLA-P (TF_GLM53_SMLA_P=1): the prefill attend with query blocking .
// One block = SMLA_QB = 2 adjacent query rows x one 32-head group: M = 64 query-heads. QB = 4 does not fit: O for
// 128 query-heads x 512 columns in fp32 is 256 KB, the whole register file. With QB = 2, O is 64 fp32 a thread at
// 512 threads, Q is held in registers as MMA A fragments (36 a thread) and shared memory holds the key tiles.
//
// Tiles and row invariance. A row's key tile t is its OWN list entries [16t, 16t + 16) (dense: positions 16t..; sparse:
// global_tokens), exactly attend_mma's tiles; key column e of the tile is entry 16t + e. Every row of the block
// steps through its tiles in lockstep, and step t loads the UNION of the rows' tile-t keys once (at most 32 rows,
// dedupe in warp 0: lane b*16 + e is row b's entry e; a key seen in a lower lane reuses that lane's union row).
// MMA B fragments are gathered per row from the union rows with ldmatrix (one lane address a key row), so a row's
// scores, P and P.V see the same key data in the same columns, tile boundaries and order whether its block-mate is
// another row, an empty row or none. A row whose list has no tile t is skipped for step t (its warps branch around
// the MMAs, its softmax state is untouched), so extra steps of a longer block-mate are exact no-ops. Hence a row's
// bits do not depend on the block (tested bit-exact: alone, in a block, in a shifted pairing).
//
// Math (as attend_mma, with two reassociations): S = Q[64 x 576] . K^T as bf16 mma.sync with fp32 accumulation; warp
// w owns m-tile w >> 2 (16 query-heads of row (w >> 3)) over K quarter w & 3 (9 k16 steps); the four partial sums
// are added in quarter order (attend_mma: two halves), x 0.0625, invalid keys -inf. Online softmax per query-head
// over the 16-key tile: 8 threads a query-head, max and sum by a fixed xor tree (attend_mma: serial over j), m =
// max(m_old, tile max), alpha = exp(m_old - m), P = bf16(exp(s - m)), denominator in fp32. O = O * alpha + P . V
// (V = the keys' latent rows); warp w owns value columns 32w .. 32w + 31 for all 64 query-heads.
// Output layout, LSE, empty-row rule (zeros and -inf), invalid keys (< 0 or > position) and the owner/stage plane
// mapping (p % world == rank ? latent : stage, at slot p / world) are tf_glm53_attend's prefill contract.
constexpr int SMLA_QB = 2;                                // query rows a block
constexpr int SMLA_KT = 16;                               // own list entries a row's tile (attend_mma's MMA_KT)
constexpr int SMLA_THREADS = 512;                         // 16 warps
constexpr int SMLA_MH = SMLA_QB * 32;                     // query-heads a block (the MMA M dimension)
constexpr int SMLA_UK = SMLA_QB * SMLA_KT;                // union capacity a step: the step's distinct key rows
constexpr int SMLA_KS = 576 + 8;                          // key row stride (bf16): 1168 B, 16-byte aligned
constexpr int SMLA_KQ = 4;                                // K quarters of the score MMA
constexpr int SMLA_KSTEPS = 576 / 16 / SMLA_KQ;           // 9 k16 steps a warp
constexpr int SMLA_PS = SMLA_KT + 8;                      // P row stride (bf16)
constexpr int SMLA_META = 3;                              // step ring depth of the union tables
constexpr int SMLA_OFF_K = 0;                                                   // [2][UK][KS] bf16 union key rows
constexpr int SMLA_OFF_Z = SMLA_OFF_K + 2 * SMLA_UK * SMLA_KS * 2;              // [KS] bf16 zero row (invalid keys)
constexpr int SMLA_OFF_S = SMLA_OFF_Z + SMLA_KS * 2;                            // [KQ][MH][KT] fp32 score partials
constexpr int SMLA_OFF_P = SMLA_OFF_S + SMLA_KQ * SMLA_MH * SMLA_KT * 4;        // [MH][PS] bf16 P
constexpr int SMLA_OFF_H = SMLA_OFF_P + SMLA_MH * SMLA_PS * 2;                  // [3][MH] fp32 max, denominator, alpha
constexpr int SMLA_OFF_POS = SMLA_OFF_H + 3 * SMLA_MH * 4;                      // [QB] int64 positions
constexpr int SMLA_OFF_N = SMLA_OFF_POS + SMLA_QB * 8;                          // [QB] int list lengths (16 B)
constexpr int SMLA_OFF_MAP = SMLA_OFF_N + 16;                                   // [META][UK] entry -> union row, -1 invalid
constexpr int SMLA_OFF_USLOT = SMLA_OFF_MAP + SMLA_META * SMLA_UK * 4;          // [META][UK] union row -> plane slot
constexpr int SMLA_OFF_UPLANE = SMLA_OFF_USLOT + SMLA_META * SMLA_UK * 4;       // [META][UK] 0 owner plane, 1 stage
constexpr int SMLA_OFF_UCOUNT = SMLA_OFF_UPLANE + SMLA_META * SMLA_UK * 4;      // [META] union rows (16 B)
constexpr int SMLA_SMEM = SMLA_OFF_UCOUNT + 16;
static_assert(SMLA_QB == 2, "warp roles are laid out for M = 64: 4 m-tiles x 4 K quarters (scores), 16 x 32 value columns (P.V)");
static_assert(SMLA_UK == 32, "warp 0 builds a step's union with one lane an entry");
static_assert((SMLA_THREADS / 32) == 2 * SMLA_QB * SMLA_KQ && 512 / (SMLA_THREADS / 32) == 32, "warp roles");
static_assert(SMLA_SMEM == 97344, "mla.zig smla_p_smem mirrors this layout");
static_assert(SMLA_SMEM <= 99 * 1024, "SMLA-P must fit the sm_120/sm_121 opt-in shared memory");

namespace {
__device__ __forceinline__ void smla_ldsm_x4(uint32_t* r, unsigned addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr) : "memory");
}
__device__ __forceinline__ void smla_ldsm_x4_trans(uint32_t* r, unsigned addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr) : "memory");
}
__device__ __forceinline__ uint32_t smla_ld32(const __nv_bfloat16* p) { return *reinterpret_cast<const uint32_t*>(p); }
}

// Same arguments as tf_glm53_attend; only the prefill fields are read (local_slots, local_counts, dense_len are the
// decode contract). The host launches it for prefill only (prefill = 1); grid (ceil(rows / 2), local_heads / 32).
extern "C" __global__ void __launch_bounds__(SMLA_THREADS, 1) tf_glm53_smla_p(const __nv_bfloat16* qa, const __nv_bfloat16* qpe,
                                            const __nv_bfloat16* latent, const __nv_bfloat16* rope,
                                            const __nv_bfloat16* stage_latent, const __nv_bfloat16* stage_rope,
                                            const int64_t* positions, const int32_t* global_tokens,
                                            const int32_t* local_slots, const int32_t* global_counts,
                                            const int32_t* local_counts, const int32_t* dense_len,
                                            __nv_bfloat16* partial, int rows,
                                            int local_heads, int rank, int world, int prefill) {
  (void)local_slots; (void)local_counts; (void)dense_len;
  const int row0 = blockIdx.x * SMLA_QB, group = blockIdx.y, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  if (prefill == 0) __trap();
  if (row0 >= rows || group >= (local_heads + 31) / 32) return;
  extern __shared__ __align__(16) unsigned char smla_smem[];
  __nv_bfloat16* kbuf = reinterpret_cast<__nv_bfloat16*>(smla_smem + SMLA_OFF_K);
  __nv_bfloat16* kzero = reinterpret_cast<__nv_bfloat16*>(smla_smem + SMLA_OFF_Z);
  float* spart = reinterpret_cast<float*>(smla_smem + SMLA_OFF_S);
  __nv_bfloat16* ps = reinterpret_cast<__nv_bfloat16*>(smla_smem + SMLA_OFF_P);
  float* hmax = reinterpret_cast<float*>(smla_smem + SMLA_OFF_H);
  float* hden = hmax + SMLA_MH;
  float* halpha = hden + SMLA_MH;
  int64_t* rpos = reinterpret_cast<int64_t*>(smla_smem + SMLA_OFF_POS);
  int* rn = reinterpret_cast<int*>(smla_smem + SMLA_OFF_N);
  int* kmap0 = reinterpret_cast<int*>(smla_smem + SMLA_OFF_MAP);
  int* uslot0 = reinterpret_cast<int*>(smla_smem + SMLA_OFF_USLOT);
  int* uplane0 = reinterpret_cast<int*>(smla_smem + SMLA_OFF_UPLANE);
  int* ucount = reinterpret_cast<int*>(smla_smem + SMLA_OFF_UCOUNT);
  const int g = lane >> 2, q = lane & 3;

  if (tid < SMLA_QB) {   // row b: position and list length (dense rows: every key 0..position)
    const int r = row0 + tid;
    int64_t qpos = -1;
    int n = 0;
    if (r < rows) {
      qpos = positions[r];
      n = qpos < 2048 ? (int)(qpos + 1) : global_counts[r];
    }
    rpos[tid] = qpos;
    rn[tid] = n;
  }
  if (tid < SMLA_MH) { hmax[tid] = -CUDART_INF_F; hden[tid] = 0.f; halpha[tid] = 1.f; }
  for (int x = tid; x < SMLA_KS / 8; x += SMLA_THREADS) reinterpret_cast<uint4*>(kzero)[x] = make_uint4(0u, 0u, 0u, 0u);

  // Score role: m-tile smt (query-heads 16 smt .. 16 smt + 15 of the block; row sb), K quarter kq. Its Q A fragments
  // for 9 k16 steps stay in registers: a[0] (head g, k0), a[1] (g + 8, k0), a[2] (g, k0 + 8), a[3] (g + 8, k0 + 8).
  const int smt = warp >> 2, sb = smt >> 1, kq = warp & 3;
  uint32_t qf[SMLA_KSTEPS][4];
  {
    const int r = row0 + sb;
    const int head_base = (smt & 1) * 16 + g;
#pragma unroll
    for (int st = 0; st < SMLA_KSTEPS; ++st) {
      const int k0 = (kq * SMLA_KSTEPS + st) * 16 + 2 * q;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int h = head_base + 8 * (i & 1);
        const int kk = k0 + 8 * (i >> 1);
        uint32_t v = 0u;
        if (r < rows) {
          if (local_heads < 32) {
            if (h < local_heads) {
              const int64_t qr = (int64_t)r * local_heads + h;
              v = kk < LD ? smla_ld32(qa + qr * LD + kk) : smla_ld32(qpe + qr * PD + (kk - LD));
            }
          } else {
            const int64_t qr = ((int64_t)group * rows + r) * 32 + h;
            v = kk < LD ? smla_ld32(qa + qr * LD + kk) : smla_ld32(qpe + qr * PD + (kk - LD));
          }
        }
        qf[st][i] = v;
      }
    }
  }
  __syncthreads();
  int steps = 0;
#pragma unroll
  for (int b = 0; b < SMLA_QB; ++b) steps = max(steps, (rn[b] + SMLA_KT - 1) / SMLA_KT);

  // Warp 0, lane b * 16 + e: row b's raw list entry step * 16 + e (dense rows: the position itself); -1 past the list.
  // Loaded two steps ahead (the register `ahead`), so the global read is off the step's critical path.
  auto entry = [&](int step) -> int {
    const int b = lane / SMLA_KT, j = step * SMLA_KT + lane % SMLA_KT;
    if (b >= SMLA_QB || j >= rn[b]) return -1;
    return rpos[b] < 2048 ? j : global_tokens[(int64_t)(row0 + b) * TOPK + j];
  };
  // Warp 0: step `step`'s union from the lanes' raw entries. Invalid keys (< 0 or > position) map to -1; a key
  // equal to a lower lane's key takes that lane's union row; the other keys get rows in lane order.
  auto meta = [&](int step, int raw) {
    const int mb = step % SMLA_META;
    const int b = lane / SMLA_KT;
    const int key = b < SMLA_QB && raw >= 0 && (int64_t)raw <= rpos[b] ? raw : -1;
    int first = -1;
#pragma unroll
    for (int i = 0; i < 32; ++i) {
      const int other = __shfl_sync(0xffffffffu, key, i);
      if (first < 0 && i < lane && key >= 0 && other == key) first = i;
    }
    const bool fresh = key >= 0 && first < 0;
    const unsigned owners = __ballot_sync(0xffffffffu, fresh);
    const int u = __popc(owners & ((1u << lane) - 1u));
    const int urow = __shfl_sync(0xffffffffu, u, first >= 0 ? first : lane);
    kmap0[mb * SMLA_UK + lane] = key >= 0 ? urow : -1;
    if (fresh) {
      uslot0[mb * SMLA_UK + u] = glm53_stage_plane_slot(key, world);
      uplane0[mb * SMLA_UK + u] = key % world == rank ? 0 : 1;
    }
    if (lane == 0) ucount[mb] = __popc(owners);
  };
  // All threads: 16-byte cp.async pieces of the step's union rows (72 a row: 64 latent + 8 rope) into buffer step & 1.
  auto fetch = [&](int step) {
    const int mb = step % SMLA_META;
    __nv_bfloat16* kb = kbuf + (step & 1) * SMLA_UK * SMLA_KS;
    const int count = ucount[mb];
    for (int x = tid; x < count * 72; x += SMLA_THREADS) {
      const int u = x / 72, c = x % 72, slot = uslot0[mb * SMLA_UK + u];
      const bool staged = uplane0[mb * SMLA_UK + u] != 0;
      const __nv_bfloat16* src = c < 64 ? (staged ? stage_latent : latent) + (int64_t)slot * LD + c * 8
                                        : (staged ? stage_rope : rope) + (int64_t)slot * PD + (c - 64) * 8;
      const unsigned d = (unsigned)__cvta_generic_to_shared(kb + u * SMLA_KS + c * 8);
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(d), "l"(src) : "memory");
    }
    asm volatile("cp.async.commit_group;\n" ::: "memory");
  };

  float o[2 * SMLA_QB][4][4];
#pragma unroll
  for (int m = 0; m < 2 * SMLA_QB; ++m)
#pragma unroll
    for (int t = 0; t < 4; ++t)
#pragma unroll
      for (int c = 0; c < 4; ++c) o[m][t][c] = 0.f;

  int ahead = -1;   // warp 0: its lane's raw entry for step t + 1 during step t
  if (steps > 0) {
    if (warp == 0) { meta(0, entry(0)); ahead = entry(1); }
    __syncthreads();
    fetch(0);
  }
  for (int t = 0; t < steps; ++t) {
    const int mb = t % SMLA_META;
    const bool more = t + 1 < steps;
    // Ring slot (t + 1) % 3 was last read in step t - 2; key buffer (t + 1) & 1 in step t - 1 (before this sync).
    if (more && warp == 0) { meta(t + 1, ahead); ahead = entry(t + 2); }
    __syncthreads();
    if (more) { fetch(t + 1); asm volatile("cp.async.wait_group 1;\n" ::: "memory"); }
    else asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    __syncthreads();   // step t's union rows are in buffer t & 1
    const __nv_bfloat16* kb = kbuf + (t & 1) * SMLA_UK * SMLA_KS;
    const int* kmap = kmap0 + mb * SMLA_UK;
    if (t * SMLA_KT < rn[sb]) {   // scores: m-tile smt x the row's 16 keys (2 n-tiles) over K quarter kq
      float acc[2][4];
#pragma unroll
      for (int nt = 0; nt < 2; ++nt)
#pragma unroll
        for (int c = 0; c < 4; ++c) acc[nt][c] = 0.f;
      // ldmatrix x4: lanes 0-7 key rows 0-7 at k0, 8-15 rows 0-7 at k0 + 8, 16-23 rows 8-15 at k0, 24-31 rows 8-15 at
      // k0 + 8 -> {b0, b1} of n-tile 0, {b0, b1} of n-tile 1.
      const int u = kmap[sb * SMLA_KT + (lane >> 4) * 8 + (lane & 7)];
      const __nv_bfloat16* kr = u < 0 ? kzero : kb + u * SMLA_KS;
      const unsigned base = (unsigned)__cvta_generic_to_shared(kr + ((lane >> 3) & 1) * 8) + kq * SMLA_KSTEPS * 32;
#pragma unroll
      for (int st = 0; st < SMLA_KSTEPS; ++st) {
        uint32_t bk[4];
        smla_ldsm_x4(bk, base + st * 32);
        mla_mma16816(acc[0], qf[st], bk);
        mla_mma16816(acc[1], qf[st], bk + 2);
      }
      float* sp = spart + kq * SMLA_MH * SMLA_KT;
      const int row = smt * 16 + g;
#pragma unroll
      for (int nt = 0; nt < 2; ++nt) {
        const int col = nt * 8 + 2 * q;
        sp[row * SMLA_KT + col] = acc[nt][0];
        sp[row * SMLA_KT + col + 1] = acc[nt][1];
        sp[(row + 8) * SMLA_KT + col] = acc[nt][2];
        sp[(row + 8) * SMLA_KT + col + 1] = acc[nt][3];
      }
    }
    __syncthreads();
    {   // online softmax: query-head qh by 8 threads, keys 2 sub and 2 sub + 1; a warp's 4 query-heads share a row
      const int qh = tid >> 3, sub = tid & 7, b = qh >> 5;
      if (t * SMLA_KT < rn[b]) {
        const float old = hmax[qh], den = hden[qh];
        float s[2];
#pragma unroll
        for (int i = 0; i < 2; ++i) {
          const int j = 2 * sub + i;
          const float* sq = spart + qh * SMLA_KT + j;
          s[i] = kmap[b * SMLA_KT + j] < 0 ? -CUDART_INF_F
              : (((sq[0] + sq[SMLA_MH * SMLA_KT]) + sq[2 * SMLA_MH * SMLA_KT]) + sq[3 * SMLA_MH * SMLA_KT]) * 0.0625f;
        }
        float mx = fmaxf(s[0], s[1]);
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 4));
        const float m = fmaxf(old, mx);
        const float alpha = isinf(old) ? 0.f : expf(old - m);
        const float p0 = isinf(s[0]) ? 0.f : expf(s[0] - m);
        const float p1 = isinf(s[1]) ? 0.f : expf(s[1] - m);
        float sum = p0 + p1;
        sum += __shfl_xor_sync(0xffffffffu, sum, 1);
        sum += __shfl_xor_sync(0xffffffffu, sum, 2);
        sum += __shfl_xor_sync(0xffffffffu, sum, 4);
        ps[qh * SMLA_PS + 2 * sub] = rb(p0);
        ps[qh * SMLA_PS + 2 * sub + 1] = rb(p1);
        __syncwarp();
        if (sub == 0) {
          if (isinf(m)) { halpha[qh] = 1.f; }   // nothing valid yet: O and the denominator stay zero
          else { halpha[qh] = alpha; hden[qh] = den * alpha + sum; hmax[qh] = m; }
        }
      }
    }
    __syncthreads();
    {   // O = O * alpha + P . V over this warp's 32 columns, row by row (each row's own key rows)
      const int c0 = warp * 32;
#pragma unroll
      for (int b = 0; b < SMLA_QB; ++b) {
        if (t * SMLA_KT >= rn[b]) continue;
        // ldmatrix x4 trans: lanes 0-7 key rows 0-7, 8-15 rows 8-15 (n-tile 2 pair), 16-31 the same at +8 columns
        // (n-tile 2 pair + 1) -> {b0, b1} of each n-tile; the row addresses are the row's own entries.
        const int u = kmap[b * SMLA_KT + (lane & 15)];
        const __nv_bfloat16* vr = u < 0 ? kzero : kb + u * SMLA_KS;
        const unsigned vbase = (unsigned)__cvta_generic_to_shared(vr + c0 + (lane >> 4) * 8);
#pragma unroll
        for (int pair = 0; pair < 2; ++pair) {
          uint32_t bv[4];
          smla_ldsm_x4_trans(bv, vbase + pair * 32);
#pragma unroll
          for (int mt2 = 0; mt2 < 2; ++mt2) {
            const int mt = 2 * b + mt2;
            const __nv_bfloat16* p0 = ps + (mt * 16 + g) * SMLA_PS + 2 * q;
            const __nv_bfloat16* p1 = p0 + 8 * SMLA_PS;
            const uint32_t a[4] = {smla_ld32(p0), smla_ld32(p1), smla_ld32(p0 + 8), smla_ld32(p1 + 8)};
            const float al0 = halpha[mt * 16 + g], al1 = halpha[mt * 16 + g + 8];
#pragma unroll
            for (int h = 0; h < 2; ++h) {
              float* acc = o[mt][2 * pair + h];
              acc[0] *= al0; acc[1] *= al0; acc[2] *= al1; acc[3] *= al1;
              mla_mma16816(acc, a, bv + 2 * h);
            }
          }
        }
      }
    }
  }
  __syncthreads();
  if (local_heads < 32) {
    const int64_t partial_bytes = (int64_t)rows * local_heads * LD * (int)sizeof(__nv_bfloat16);
    float* group_lse = reinterpret_cast<float*>(reinterpret_cast<unsigned char*>(partial) + partial_bytes);
    const int c0 = warp * 32;
#pragma unroll
    for (int mt = 0; mt < 2 * SMLA_QB; ++mt) {
      const int r = row0 + (mt >> 1);
      if (r >= rows) continue;
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        const int head = (mt & 1) * 16 + g + 8 * hh;
        if (head < local_heads) {
          const float den = hden[mt * 16 + g + 8 * hh];
          __nv_bfloat16* dst = partial + ((int64_t)r * local_heads + head) * LD;
#pragma unroll
          for (int nt = 0; nt < 4; ++nt) {
            const int col = c0 + nt * 8 + 2 * q;
            dst[col] = rb(den == 0.f ? 0.f : o[mt][nt][2 * hh] / den);
            dst[col + 1] = rb(den == 0.f ? 0.f : o[mt][nt][2 * hh + 1] / den);
          }
        }
      }
    }
    if (tid < SMLA_MH) {
      const int r = row0 + (tid >> 5);
      const int head = tid & 31;
      if (r < rows && head < local_heads) {
        group_lse[(int64_t)r * local_heads + head] = hden[tid] == 0.f ? -CUDART_INF_F : hmax[tid] + logf(hden[tid]);
      }
    }
  } else {
    constexpr int group_row_bytes = 32 * LD * sizeof(__nv_bfloat16) + 32 * sizeof(float);
    const int64_t group_byte = (int64_t)group * rows * group_row_bytes;
    const int64_t group_partial = group_byte / (int)sizeof(__nv_bfloat16);
    const int c0 = warp * 32;
#pragma unroll
    for (int mt = 0; mt < 2 * SMLA_QB; ++mt) {
      const int r = row0 + (mt >> 1);
      if (r >= rows) continue;
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        const int head = (mt & 1) * 16 + g + 8 * hh;
        const float den = hden[mt * 16 + g + 8 * hh];
        __nv_bfloat16* dst = partial + group_partial + ((int64_t)r * 32 + head) * LD;
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
          const int col = c0 + nt * 8 + 2 * q;
          dst[col] = rb(den == 0.f ? 0.f : o[mt][nt][2 * hh] / den);
          dst[col + 1] = rb(den == 0.f ? 0.f : o[mt][nt][2 * hh + 1] / den);
        }
      }
    }
    if (tid < SMLA_MH && row0 + (tid >> 5) < rows) {
      float* group_lse = reinterpret_cast<float*>(reinterpret_cast<unsigned char*>(partial) + group_byte + rows * 32 * LD * (int)sizeof(__nv_bfloat16));
      group_lse[(int64_t)(row0 + (tid >> 5)) * 32 + (tid & 31)] = hden[tid] == 0.f ? -CUDART_INF_F : hmax[tid] + logf(hden[tid]);
    }
  }
}

// SMLA-P3 (TF_GLM53_SMLA_P3=1, valid only with TF_GLM53_SMLA_P=1): tf_glm53_smla_p with the same bits and fewer
// barriers. Two kernels on the same stream:
//  1. tf_glm53_smla_union: one warp a row pair runs tf_glm53_smla_p's per-step union build (the `entry` + `meta` code,
//     unchanged) for every step of the pair and writes one SMLA3_REC record a step to a workspace table.
//  2. tf_glm53_smla_p3: tf_glm53_smla_p's attend with three changes, none of which touches a value:
//     - the step's union record comes from the table by cp.async (2 steps ahead, the same 3-deep ring), not from a
//       serial warp-0 build before the step's first barrier;
//     - warps 4m .. 4m + 3 form m-tile group m (16 query-heads of row m >> 1). Scores keep tf_glm53_smla_p's roles
//       (warp = m-tile x K quarter); the quarter sum + softmax already ran on exactly these 128 threads (qh = tid >> 3);
//       P.V moves to warp (m, c) = m-tile m x value columns 128c .. 128c + 127. Every shared value a group writes in a
//       step (S partial rows, P rows, max / denominator / alpha) is read only by the same group, so the two
//       __syncthreads around the softmax become named barriers of 128 threads (ids SMLA3_BAR0 + m);
//     - the key double buffer and the record ring keep one __syncthreads a step (after the step's cp.async lands,
//       before the next fetch overwrites the buffer read in the step before): 4 full barriers -> 1 full + 2 named.
//     - blocks run longest-first: row0 = (gridDim.x - 1 - blockIdx.x) * 2 (a block's bits do not depend on its order).
// Bit contract (same as tf_glm53_smla_p): each (row-head, key) score is the same 4 quarter chains (9 k16 MMAs each, the
// same A and B fragments) summed ((q0 + q1) + q2) + q3, x 0.0625, -inf for invalid keys; the softmax is the same code
// on the same threads; every O element gets the same per-step sequence O *= alpha; O = mma(P, V, O) with the same
// A (P rows of its m-tile) and B (its 8 value columns of the step's 16 keys) fragments; only the issuing warp changes.
// mma.sync output element (i, j) depends only on A row i, B column j and C (i, j), and the element keeps its position
// in the m16 x n8 tile (n-tiles stay 8-column aligned). The output, LSE, empty-row and invalid-key rules are unchanged.
//
// Table record (SMLA3_REC int32 = 176 B = 11 x 16 B, 16-byte aligned for cp.async), [pairs][SMLA3_STEPS][SMLA3_REC]:
//   bytes 0 .. 31  int8 kmap[32]: entry e = b * 16 + j of the step (row b, tile column j) -> union row, -1 invalid
//   words 8 .. 39  union row u -> plane slot | plane << 31 (plane 0 = owner, 1 = stage); 0 for u >= count
//   word  40       count (distinct valid keys of the step)
//   words 41 .. 43 0
constexpr int SMLA3_REC = 44;                             // int32 a step record
constexpr int SMLA3_REC_BYTES = SMLA3_REC * 4;            // 176
constexpr int SMLA3_REC_SLOT = 8;                         // first slot|plane word
constexpr int SMLA3_REC_COUNT = SMLA3_REC_SLOT + SMLA_UK; // word 40
constexpr int SMLA3_STEPS = TOPK / SMLA_KT;               // 128: a list holds <= 2048 entries (dense: position + 1 <= 2048)
constexpr int SMLA3_UNION_WARPS = 4;                      // row pairs a union block (128 threads)
constexpr int SMLA3_BAR0 = 8;                             // named barrier ids 8 .. 11 (0 is __syncthreads; TF_TASK_SYNC uses 1)
constexpr int SMLA3_PV_COLS = LD / SMLA_KQ;               // 128 value columns a P.V warp
constexpr int SMLA3_PV_NT = SMLA3_PV_COLS / 8;            // 16 n-tiles a P.V warp
// Shared memory (SMLA-P's layout with the three union tables replaced by the record ring):
constexpr int SMLA3_OFF_K = 0;                                            // [2][UK][KS] bf16 union key rows: row u = key u's
                                                                          // 512 latent | 64 rope dims, dims contiguous
                                                                          // (scores: B n-major -> ldmatrix; P.V: B k-major
                                                                          // -> ldmatrix.trans)
constexpr int SMLA3_OFF_Z = SMLA3_OFF_K + 2 * SMLA_UK * SMLA_KS * 2;      // [KS] bf16 zero row (invalid keys)
constexpr int SMLA3_OFF_S = SMLA3_OFF_Z + SMLA_KS * 2;                    // [KQ][MH][KT] fp32 quarter partials, [qh][key]
constexpr int SMLA3_OFF_P = SMLA3_OFF_S + SMLA_KQ * SMLA_MH * SMLA_KT * 4; // [MH][PS] bf16 P, [qh][key] keys contiguous (A row-major)
constexpr int SMLA3_OFF_H = SMLA3_OFF_P + SMLA_MH * SMLA_PS * 2;          // [3][MH] fp32 max, denominator, alpha
constexpr int SMLA3_OFF_POS = SMLA3_OFF_H + 3 * SMLA_MH * 4;              // [QB] int64 positions
constexpr int SMLA3_OFF_N = SMLA3_OFF_POS + SMLA_QB * 8;                  // [QB] int list lengths (16 B)
constexpr int SMLA3_OFF_REC = SMLA3_OFF_N + 16;                           // [META][REC] int32 step record ring
constexpr int SMLA3_SMEM = SMLA3_OFF_REC + SMLA_META * SMLA3_REC_BYTES;
static_assert(SMLA3_REC_BYTES % 16 == 0 && SMLA3_OFF_REC % 16 == 0, "record cp.async pieces are 16 B");
static_assert(SMLA3_REC_BYTES / 16 <= SMLA_THREADS, "one thread a 16-byte record piece");
static_assert(SMLA3_REC_COUNT + 4 == SMLA3_REC, "record layout");
static_assert(SMLA3_OFF_P == SMLA_OFF_P && SMLA3_OFF_H == SMLA_OFF_H, "P3 keeps SMLA-P's tiles at the same offsets");
static_assert(SMLA3_PV_COLS * SMLA_KQ == LD && SMLA3_PV_NT % 2 == 0, "P.V warp (m, c): 4 x 128 columns, n-tile pairs");
static_assert(SMLA3_SMEM == 96704, "mla.zig smla_p3_smem mirrors this layout");
static_assert(SMLA3_SMEM <= 99 * 1024, "SMLA-P3 must fit the sm_120/sm_121 opt-in shared memory");

namespace {
__device__ __forceinline__ void smla3_group_sync(int m) {
  asm volatile("bar.sync %0, 128;\n" ::"r"(SMLA3_BAR0 + m) : "memory");
}
}

// Grid ceil(ceil(rows / 2) / SMLA3_UNION_WARPS), block SMLA3_UNION_WARPS x 32. Warp = row pair p (rows 2p, 2p + 1);
// lane b * 16 + j is row b's entry j of a step, as warp 0 of tf_glm53_smla_p. Steps past the pair's own steps are not
// written (the attend never reads them).
extern "C" __global__ void __launch_bounds__(SMLA3_UNION_WARPS * 32) tf_glm53_smla_union(const int64_t* positions, const int32_t* global_tokens,
                                            const int32_t* global_counts, int32_t* utab, int rows, int rank, int world) {
  const int lane = threadIdx.x & 31;
  const int pair = blockIdx.x * SMLA3_UNION_WARPS + (threadIdx.x >> 5);
  const int row0 = pair * SMLA_QB;
  if (row0 >= rows) return;
  const int b = lane / SMLA_KT;   // SMLA_UK == 32: every lane is a row b < SMLA_QB
  const int r = row0 + b;
  int64_t qpos = -1;   // tf_glm53_smla_p's rpos[b] / rn[b]
  int n = 0;
  if (r < rows) {
    qpos = positions[r];
    n = qpos < 2048 ? (int)(qpos + 1) : global_counts[r];
  }
  const int other = __shfl_xor_sync(0xffffffffu, n, SMLA_KT);
  const int steps = max((n + SMLA_KT - 1) / SMLA_KT, (other + SMLA_KT - 1) / SMLA_KT);
  if (steps > SMLA3_STEPS) __trap();   // a list longer than TOPK entries breaks the table (and the [rows][TOPK] token list)
  int32_t* rec_pair = utab + (int64_t)pair * SMLA3_STEPS * SMLA3_REC;
  for (int step = 0; step < steps; ++step) {
    // tf_glm53_smla_p `entry(step)`: row b's raw list entry step * 16 + j (dense rows: the position itself); -1 past it.
    const int j = step * SMLA_KT + lane % SMLA_KT;
    const int raw = j >= n ? -1 : (qpos < 2048 ? j : global_tokens[(int64_t)r * TOPK + j]);
    // tf_glm53_smla_p `meta(step, raw)`, unchanged: invalid keys (< 0 or > position) -> -1; a key equal to a lower
    // lane's key takes that lane's union row; the other keys get rows in lane order.
    const int key = raw >= 0 && (int64_t)raw <= qpos ? raw : -1;
    int first = -1;
#pragma unroll
    for (int i = 0; i < 32; ++i) {
      const int o = __shfl_sync(0xffffffffu, key, i);
      if (first < 0 && i < lane && key >= 0 && o == key) first = i;
    }
    const bool fresh = key >= 0 && first < 0;
    const unsigned owners = __ballot_sync(0xffffffffu, fresh);
    const int u = __popc(owners & ((1u << lane) - 1u));
    const int urow = __shfl_sync(0xffffffffu, u, first >= 0 ? first : lane);
    const int count = __popc(owners);
    int32_t* rec = rec_pair + (int64_t)step * SMLA3_REC;
    reinterpret_cast<int8_t*>(rec)[lane] = (int8_t)(key >= 0 ? urow : -1);
    // Fresh lanes own words SMLA3_REC_SLOT + u (u < count); lanes >= count zero their own word: disjoint stores.
    if (fresh) rec[SMLA3_REC_SLOT + u] = (int32_t)((uint32_t)glm53_stage_plane_slot(key, world) | ((uint32_t)(key % world == rank ? 0 : 1) << 31));
    if (lane >= count) rec[SMLA3_REC_SLOT + lane] = 0;
    if (lane == 0) rec[SMLA3_REC_COUNT] = count;
    if (lane < 3) rec[SMLA3_REC_COUNT + 1 + lane] = 0;
  }
}

// Same arguments as tf_glm53_smla_p plus the union table (tf_glm53_smla_union's output for the same rows and lists);
// grid (ceil(rows / 2), local_heads / 32), SMLA_THREADS threads, SMLA3_SMEM dynamic shared memory.
extern "C" __global__ void __launch_bounds__(SMLA_THREADS, 1) tf_glm53_smla_p3(const __nv_bfloat16* qa, const __nv_bfloat16* qpe,
                                            const __nv_bfloat16* latent, const __nv_bfloat16* rope,
                                            const __nv_bfloat16* stage_latent, const __nv_bfloat16* stage_rope,
                                            const int64_t* positions, const int32_t* global_tokens,
                                            const int32_t* local_slots, const int32_t* global_counts,
                                            const int32_t* local_counts, const int32_t* dense_len,
                                            __nv_bfloat16* partial, int rows,
                                            int local_heads, int rank, int world, int prefill, const int32_t* utab) {
  (void)global_tokens; (void)local_slots; (void)local_counts; (void)dense_len; (void)rank; (void)world;   // the table holds slots and planes
  // Longest-first: dense pieces have rising list lengths along the rows, so the last pairs start first.
  const int row0 = (int)(gridDim.x - 1 - blockIdx.x) * SMLA_QB, group = blockIdx.y, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  if (prefill == 0) __trap();
  if (row0 >= rows || group >= (local_heads + 31) / 32) return;
  extern __shared__ __align__(16) unsigned char smla_smem[];
  __nv_bfloat16* kbuf = reinterpret_cast<__nv_bfloat16*>(smla_smem + SMLA3_OFF_K);
  __nv_bfloat16* kzero = reinterpret_cast<__nv_bfloat16*>(smla_smem + SMLA3_OFF_Z);
  float* spart = reinterpret_cast<float*>(smla_smem + SMLA3_OFF_S);
  __nv_bfloat16* ps = reinterpret_cast<__nv_bfloat16*>(smla_smem + SMLA3_OFF_P);
  float* hmax = reinterpret_cast<float*>(smla_smem + SMLA3_OFF_H);
  float* hden = hmax + SMLA_MH;
  float* halpha = hden + SMLA_MH;
  int64_t* rpos = reinterpret_cast<int64_t*>(smla_smem + SMLA3_OFF_POS);
  int* rn = reinterpret_cast<int*>(smla_smem + SMLA3_OFF_N);
  int* rec0 = reinterpret_cast<int*>(smla_smem + SMLA3_OFF_REC);
  const int g = lane >> 2, q = lane & 3;

  if (tid < SMLA_QB) {   // row b: position and list length (dense rows: every key 0..position)
    const int r = row0 + tid;
    int64_t qpos = -1;
    int n = 0;
    if (r < rows) {
      qpos = positions[r];
      n = qpos < 2048 ? (int)(qpos + 1) : global_counts[r];
    }
    rpos[tid] = qpos;
    rn[tid] = n;
  }
  if (tid < SMLA_MH) { hmax[tid] = -CUDART_INF_F; hden[tid] = 0.f; halpha[tid] = 1.f; }
  for (int x = tid; x < SMLA_KS / 8; x += SMLA_THREADS) reinterpret_cast<uint4*>(kzero)[x] = make_uint4(0u, 0u, 0u, 0u);

  // Score role (as tf_glm53_smla_p): m-tile smt (query-heads 16 smt .. 16 smt + 15 of the block; row sb), K quarter kq.
  // Q A fragments for 9 k16 steps in registers: a[0] (head g, k0), a[1] (g + 8, k0), a[2] (g, k0 + 8), a[3] (g + 8, k0 + 8).
  // P.V role: m-tile smt x value columns 128 pc .. 128 pc + 127 (pc = warp & 3). Group smt = warps 4 smt .. 4 smt + 3.
  const int smt = warp >> 2, sb = smt >> 1, kq = warp & 3, pc = warp & 3;
  uint32_t qf[SMLA_KSTEPS][4];
  {
    const int r = row0 + sb;
    const int head_base = (smt & 1) * 16 + g;
#pragma unroll
    for (int st = 0; st < SMLA_KSTEPS; ++st) {
      const int k0 = (kq * SMLA_KSTEPS + st) * 16 + 2 * q;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int h = head_base + 8 * (i & 1);
        const int kk = k0 + 8 * (i >> 1);
        uint32_t v = 0u;
        if (r < rows) {
          if (local_heads < 32) {
            if (h < local_heads) {
              const int64_t qr = (int64_t)r * local_heads + h;
              v = kk < LD ? smla_ld32(qa + qr * LD + kk) : smla_ld32(qpe + qr * PD + (kk - LD));
            }
          } else {
            const int64_t qr = ((int64_t)group * rows + r) * 32 + h;
            v = kk < LD ? smla_ld32(qa + qr * LD + kk) : smla_ld32(qpe + qr * PD + (kk - LD));
          }
        }
        qf[st][i] = v;
      }
    }
  }
  __syncthreads();
  int steps = 0;
#pragma unroll
  for (int b = 0; b < SMLA_QB; ++b) steps = max(steps, (rn[b] + SMLA_KT - 1) / SMLA_KT);
  if (steps > SMLA3_STEPS) __trap();   // tf_glm53_smla_union traps on the same lists

  const unsigned char* rec_pair = reinterpret_cast<const unsigned char*>(utab) + (int64_t)(row0 / SMLA_QB) * SMLA3_STEPS * SMLA3_REC_BYTES;
  // Threads 0 .. 10: step `step`'s 176-byte record into ring slot step % 3 (joins the caller's commit group).
  auto record = [&](int step) {
    if (tid < SMLA3_REC_BYTES / 16) {
      const unsigned d = (unsigned)__cvta_generic_to_shared(rec0 + (step % SMLA_META) * SMLA3_REC + tid * 4);
      const unsigned char* src = rec_pair + (int64_t)step * SMLA3_REC_BYTES + tid * 16;
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(d), "l"(src) : "memory");
    }
  };
  // All threads: tf_glm53_smla_p's `fetch` (16-byte pieces of the step's union rows, 72 a row: 64 latent + 8 rope, into
  // buffer step & 1), with the slots, planes and count read from ring slot step % 3 (joins the caller's commit group).
  auto fetch = [&](int step) {
    const int* rec = rec0 + (step % SMLA_META) * SMLA3_REC;
    __nv_bfloat16* kb = kbuf + (step & 1) * SMLA_UK * SMLA_KS;
    const int count = rec[SMLA3_REC_COUNT];
    for (int x = tid; x < count * 72; x += SMLA_THREADS) {
      const int u = x / 72, c = x % 72;
      const uint32_t sp = (uint32_t)rec[SMLA3_REC_SLOT + u];
      const int slot = (int)(sp & 0x7fffffffu);
      const bool staged = (sp >> 31) != 0u;
      const __nv_bfloat16* src = c < 64 ? (staged ? stage_latent : latent) + (int64_t)slot * LD + c * 8
                                        : (staged ? stage_rope : rope) + (int64_t)slot * PD + (c - 64) * 8;
      const unsigned d = (unsigned)__cvta_generic_to_shared(kb + u * SMLA_KS + c * 8);
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(d), "l"(src) : "memory");
    }
  };

  // P.V accumulators of warp (smt, pc): o[nt] = query-heads 16 smt + {g, g + 8} x columns 128 pc + 8 nt + {2q, 2q + 1}.
  float o[SMLA3_PV_NT][4];
#pragma unroll
  for (int t = 0; t < SMLA3_PV_NT; ++t)
#pragma unroll
    for (int c = 0; c < 4; ++c) o[t][c] = 0.f;

  if (steps > 0) {
    record(0);
    asm volatile("cp.async.commit_group;\n" ::: "memory");
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    __syncthreads();   // record 0 is in ring slot 0
    fetch(0);
    if (steps > 1) record(1);
    asm volatile("cp.async.commit_group;\n" ::: "memory");
  }
  for (int t = 0; t < steps; ++t) {
    const int mb = t % SMLA_META;
    // The one full barrier of the step: this thread's step-t key pieces and record t + 1 (committed in step t - 1 or
    // the prologue) have landed; after the barrier everyone's have, and every warp is done with step t - 1 (key buffer
    // (t + 1) & 1 and ring slot (t + 2) % 3 = (t - 1) % 3 are free; S, P and the softmax state are group-private).
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    __syncthreads();
    if (t + 1 < steps) {
      fetch(t + 1);
      if (t + 2 < steps) record(t + 2);
      asm volatile("cp.async.commit_group;\n" ::: "memory");
    }
    const __nv_bfloat16* kb = kbuf + (t & 1) * SMLA_UK * SMLA_KS;
    const signed char* kmap = reinterpret_cast<const signed char*>(rec0 + mb * SMLA3_REC);
    if (t * SMLA_KT < rn[sb]) {   // scores (tf_glm53_smla_p, unchanged): m-tile smt x the row's 16 keys over quarter kq
      float acc[2][4];
#pragma unroll
      for (int nt = 0; nt < 2; ++nt)
#pragma unroll
        for (int c = 0; c < 4; ++c) acc[nt][c] = 0.f;
      // ldmatrix x4 (no .trans: key rows are the n index, dims the contiguous k index): lanes 0-7 key rows 0-7 at k0,
      // 8-15 rows 0-7 at k0 + 8, 16-23 rows 8-15 at k0, 24-31 rows 8-15 at k0 + 8 -> {b0, b1} of n-tile 0, of n-tile 1.
      const int u = kmap[sb * SMLA_KT + (lane >> 4) * 8 + (lane & 7)];
      const __nv_bfloat16* kr = u < 0 ? kzero : kb + u * SMLA_KS;
      const unsigned base = (unsigned)__cvta_generic_to_shared(kr + ((lane >> 3) & 1) * 8) + kq * SMLA_KSTEPS * 32;
#pragma unroll
      for (int st = 0; st < SMLA_KSTEPS; ++st) {
        uint32_t bk[4];
        smla_ldsm_x4(bk, base + st * 32);
        mla_mma16816(acc[0], qf[st], bk);
        mla_mma16816(acc[1], qf[st], bk + 2);
      }
      float* sp = spart + kq * SMLA_MH * SMLA_KT;
      const int row = smt * 16 + g;
#pragma unroll
      for (int nt = 0; nt < 2; ++nt) {
        const int col = nt * 8 + 2 * q;
        sp[row * SMLA_KT + col] = acc[nt][0];
        sp[row * SMLA_KT + col + 1] = acc[nt][1];
        sp[(row + 8) * SMLA_KT + col] = acc[nt][2];
        sp[(row + 8) * SMLA_KT + col + 1] = acc[nt][3];
      }
    }
    smla3_group_sync(smt);   // the group's 4 quarter partials of its 16 query-heads are in S
    {   // online softmax (tf_glm53_smla_p, unchanged): query-head qh by 8 threads; tid >> 3 lies in m-tile smt
      const int qh = tid >> 3, sub = tid & 7, b = qh >> 5;
      if (t * SMLA_KT < rn[b]) {
        const float old = hmax[qh], den = hden[qh];
        float s[2];
#pragma unroll
        for (int i = 0; i < 2; ++i) {
          const int j = 2 * sub + i;
          const float* sq = spart + qh * SMLA_KT + j;
          s[i] = kmap[b * SMLA_KT + j] < 0 ? -CUDART_INF_F
              : (((sq[0] + sq[SMLA_MH * SMLA_KT]) + sq[2 * SMLA_MH * SMLA_KT]) + sq[3 * SMLA_MH * SMLA_KT]) * 0.0625f;
        }
        float mx = fmaxf(s[0], s[1]);
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 4));
        const float m = fmaxf(old, mx);
        const float alpha = isinf(old) ? 0.f : expf(old - m);
        const float p0 = isinf(s[0]) ? 0.f : expf(s[0] - m);
        const float p1 = isinf(s[1]) ? 0.f : expf(s[1] - m);
        float sum = p0 + p1;
        sum += __shfl_xor_sync(0xffffffffu, sum, 1);
        sum += __shfl_xor_sync(0xffffffffu, sum, 2);
        sum += __shfl_xor_sync(0xffffffffu, sum, 4);
        ps[qh * SMLA_PS + 2 * sub] = rb(p0);
        ps[qh * SMLA_PS + 2 * sub + 1] = rb(p1);
        __syncwarp();
        if (sub == 0) {
          if (isinf(m)) { halpha[qh] = 1.f; }   // nothing valid yet: O and the denominator stay zero
          else { halpha[qh] = alpha; hden[qh] = den * alpha + sum; hmax[qh] = m; }
        }
      }
    }
    smla3_group_sync(smt);   // the group's P rows and alphas are in shared memory
    if (t * SMLA_KT < rn[sb]) {   // O = O * alpha + P . V: m-tile smt x columns 128 pc .. 128 pc + 127 (row sb's own keys)
      // A = P of m-tile smt, stored [qh][key] (keys contiguous): a0 rows g k 2q, a1 rows g + 8 k 2q, a2 rows g k 2q + 8,
      // a3 rows g + 8 k 2q + 8 (rows 0-7 / 8-15, k 0-7 / 8-15), loaded once a step.
      const __nv_bfloat16* p0 = ps + (smt * 16 + g) * SMLA_PS + 2 * q;
      const __nv_bfloat16* p1 = p0 + 8 * SMLA_PS;
      const uint32_t a[4] = {smla_ld32(p0), smla_ld32(p1), smla_ld32(p0 + 8), smla_ld32(p1 + 8)};
      const float al0 = halpha[smt * 16 + g], al1 = halpha[smt * 16 + g + 8];
      // B = V, stored [key][column] (key = the MMA k index, columns contiguous = n): ldmatrix x4 .trans. Lanes 0-7 key
      // rows 0-7, 8-15 rows 8-15 (n-tile 2 pair), 16-31 the same at +8 columns (n-tile 2 pair + 1) -> {b0, b1} of each.
      const int u = kmap[sb * SMLA_KT + (lane & 15)];
      const __nv_bfloat16* vr = u < 0 ? kzero : kb + u * SMLA_KS;
      const unsigned vbase = (unsigned)__cvta_generic_to_shared(vr + pc * SMLA3_PV_COLS + (lane >> 4) * 8);
#pragma unroll
      for (int pair = 0; pair < SMLA3_PV_NT / 2; ++pair) {
        uint32_t bv[4];
        smla_ldsm_x4_trans(bv, vbase + pair * 32);
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          float* acc = o[2 * pair + h];
          acc[0] *= al0; acc[1] *= al0; acc[2] *= al1; acc[3] *= al1;
          mla_mma16816(acc, a, bv + 2 * h);
        }
      }
    }
  }
  __syncthreads();
  if (local_heads < 32) {
    const int64_t partial_bytes = (int64_t)rows * local_heads * LD * (int)sizeof(__nv_bfloat16);
    float* group_lse = reinterpret_cast<float*>(reinterpret_cast<unsigned char*>(partial) + partial_bytes);
    const int r = row0 + sb;
    if (r < rows) {
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        const int head = (smt & 1) * 16 + g + 8 * hh;
        if (head < local_heads) {
          const float den = hden[smt * 16 + g + 8 * hh];
          __nv_bfloat16* dst = partial + ((int64_t)r * local_heads + head) * LD;
#pragma unroll
          for (int nt = 0; nt < SMLA3_PV_NT; ++nt) {
            const int col = pc * SMLA3_PV_COLS + nt * 8 + 2 * q;
            dst[col] = rb(den == 0.f ? 0.f : o[nt][2 * hh] / den);
            dst[col + 1] = rb(den == 0.f ? 0.f : o[nt][2 * hh + 1] / den);
          }
        }
      }
    }
    if (tid < SMLA_MH) {
      const int r_lse = row0 + (tid >> 5);
      const int head = tid & 31;
      if (r_lse < rows && head < local_heads) {
        group_lse[(int64_t)r_lse * local_heads + head] = hden[tid] == 0.f ? -CUDART_INF_F : hmax[tid] + logf(hden[tid]);
      }
    }
  } else {
    constexpr int group_row_bytes = 32 * LD * sizeof(__nv_bfloat16) + 32 * sizeof(float);
    const int64_t group_byte = (int64_t)group * rows * group_row_bytes;
    const int64_t group_partial = group_byte / (int)sizeof(__nv_bfloat16);
    {
      const int r = row0 + sb;
      if (r < rows) {
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
          const int head = (smt & 1) * 16 + g + 8 * hh;
          const float den = hden[smt * 16 + g + 8 * hh];
          __nv_bfloat16* dst = partial + group_partial + ((int64_t)r * 32 + head) * LD;
#pragma unroll
          for (int nt = 0; nt < SMLA3_PV_NT; ++nt) {
            const int col = pc * SMLA3_PV_COLS + nt * 8 + 2 * q;
            dst[col] = rb(den == 0.f ? 0.f : o[nt][2 * hh] / den);
            dst[col + 1] = rb(den == 0.f ? 0.f : o[nt][2 * hh + 1] / den);
          }
        }
      }
    }
    if (tid < SMLA_MH && row0 + (tid >> 5) < rows) {
      float* group_lse = reinterpret_cast<float*>(reinterpret_cast<unsigned char*>(partial) + group_byte + rows * 32 * LD * (int)sizeof(__nv_bfloat16));
      group_lse[(int64_t)(row0 + (tid >> 5)) * 32 + (tid & 31)] = hden[tid] == 0.f ? -CUDART_INF_F : hmax[tid] + logf(hden[tid]);
    }
  }
}

// SMLA-FP8 (TF_GLM53_SMLA_FP8=1, valid only with TF_GLM53_SMLA_P=1): tf_glm53_smla_p with the score GEMM S = Q.K^T in
// e4m3 (score part only). Lossy: behind a KLD gate. The KV
// cache and the stage plane stay bf16; the step's bf16 union key rows are converted to e4m3 in shared memory.
// Same grid, block, union steps, per-row 16-entry tiles, online softmax (fp32), P.V (bf16), output, LSE, empty-row and
// invalid-key rules as tf_glm53_smla_p. The differences:
//  - Q: one power-of-two fp32 scale a (row, head), sq = the smallest 2^k with amax(|q|, 576 dims) / 2^k <= 448 (1 for
//    an all-zero row-head); codes e4m3(q / sq) (RN, satfinite) held as m16n8k32 A fragments (36 registers a thread).
//  - K: after a step's bf16 union rows land, warp w converts union rows w and w + 16: sk by the same rule over the
//    row's 576 dims, codes e4m3(k / sk) into a [32][592] byte tile; sk into a [32] fp32 table.
//  - Scores: warp w owns m-tile w >> 2 (row w >> 3), n-tile (w >> 1) & 1 (8 of the row's 16 keys) and K half w & 1
//    (dims 288 (w & 1) .. + 287: 9 k32 steps), e4m3 mma.sync m16n8k32 with fp32 accumulation. Pair (w, w ^ 1): the
//    half-0 warp stores its sums and arrives at named barrier 1 + (w >> 1); the half-1 warp waits there and adds its
//    own (h0 + h1, one fixed order). s = ((h0 + h1) * (sq * sk)) * 0.0625: sq * sk and 0.0625 are powers of two, so
//    only h0 + h1 rounds. Invalid keys read e4m3 row 0 (the tile is zeroed once, so stale rows are finite); their
//    columns are replaced by -inf exactly as in tf_glm53_smla_p (MMA columns are independent).
// Row invariance holds as in tf_glm53_smla_p: a key's e4m3 row and scale depend on the key alone, a row-head's codes and
// scale on the row-head alone, the tiles and the summation order on nothing else (tested bit-exact).
// Shared memory (101,184 B): bf16 latent [2][32][520] (V and the latent source of K), bf16 rope [2][32][72] (K source
// only: dead after the step's conversion, so the step's [64][16] fp32 scores live in it), e4m3 K [32][592], zero V row,
// P, max/denominator/alpha, Q and K scales, and the same union ring as tf_glm53_smla_p. The e4m3 tile is single-buffered
// (converted each step from the double-buffered bf16 rows): no space is left under 99 KB for a deeper cp.async ring.
constexpr int SMF_VS = 512 + 8;                            // bf16 latent row stride: 1040 B (ldmatrix rows on distinct banks)
constexpr int SMF_RS = 64 + 8;                             // bf16 rope row stride: 144 B
constexpr int SMF_ES = 576 + 16;                           // e4m3 key row stride: 592 B
constexpr int SMF_KSTEPS = 576 / 32 / 2;                   // 9 k32 steps a warp (K halves)
constexpr int SMF_OFF_V = 0;                                                    // [2][UK][VS] bf16 latent rows
constexpr int SMF_OFF_R = SMF_OFF_V + 2 * SMLA_UK * SMF_VS * 2;                 // [2][UK][RS] bf16 rope rows; scores alias [t & 1]
constexpr int SMF_OFF_E = SMF_OFF_R + 2 * SMLA_UK * SMF_RS * 2;                 // [UK][ES] e4m3 key rows
constexpr int SMF_OFF_Z = SMF_OFF_E + SMLA_UK * SMF_ES;                         // [VS] bf16 zero row (invalid keys in P.V)
constexpr int SMF_OFF_P = SMF_OFF_Z + SMF_VS * 2;                               // [MH][PS] bf16 P
constexpr int SMF_OFF_H = SMF_OFF_P + SMLA_MH * SMLA_PS * 2;                    // [3][MH] fp32 max, denominator, alpha
constexpr int SMF_OFF_QS = SMF_OFF_H + 3 * SMLA_MH * 4;                         // [MH] fp32 Q scales
constexpr int SMF_OFF_KSC = SMF_OFF_QS + SMLA_MH * 4;                           // [UK] fp32 key scales (this step)
constexpr int SMF_OFF_POS = SMF_OFF_KSC + SMLA_UK * 4;                          // [QB] int64 positions
constexpr int SMF_OFF_N = SMF_OFF_POS + SMLA_QB * 8;                            // [QB] int list lengths (16 B)
constexpr int SMF_OFF_MAP = SMF_OFF_N + 16;                                     // [META][UK] entry -> union row, -1 invalid
constexpr int SMF_OFF_USLOT = SMF_OFF_MAP + SMLA_META * SMLA_UK * 4;            // [META][UK] union row -> plane slot
constexpr int SMF_OFF_UPLANE = SMF_OFF_USLOT + SMLA_META * SMLA_UK * 4;         // [META][UK] 0 owner plane, 1 stage
constexpr int SMF_OFF_UCOUNT = SMF_OFF_UPLANE + SMLA_META * SMLA_UK * 4;        // [META] union rows (16 B)
constexpr int SMF_SMEM = SMF_OFF_UCOUNT + 16;
static_assert(SMF_SMEM == 101184, "mla.zig smla_fp8_smem mirrors this layout");
static_assert(SMF_SMEM <= 99 * 1024, "SMLA-FP8 must fit the sm_120/sm_121 opt-in shared memory");
static_assert(SMLA_MH * SMLA_KT * 4 <= SMLA_UK * SMF_RS * 2, "a step's fp32 scores fit its rope buffer");
static_assert(SMF_OFF_R % 16 == 0 && SMF_OFF_E % 16 == 0 && SMF_OFF_Z % 16 == 0 && SMF_OFF_P % 16 == 0 && SMF_OFF_POS % 8 == 0, "alignment");
static_assert((SMF_VS * 2) % 16 == 0 && (SMF_RS * 2) % 16 == 0 && SMF_ES % 16 == 0, "16-byte rows for cp.async and ldmatrix");
static_assert(SMF_KSTEPS * 32 * 2 == 576 && SMF_KSTEPS % 2 == 1, "the score loop pairs k32 steps and ends with one x2 step");
static_assert(SMLA_THREADS / 32 == 4 * 2 * 2, "warp roles: 4 m-tiles x 2 n-tiles x 2 K halves");

namespace {
__device__ __forceinline__ void smf_ldsm_x2(uint32_t* r, unsigned addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n" : "=r"(r[0]), "=r"(r[1]) : "r"(addr) : "memory");
}
__device__ __forceinline__ void smf_mma_e4m3(float* c, const uint32_t* a, const uint32_t* b) {
#if __CUDA_ARCH__ >= 890
  asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#else
  (void)c; (void)a; (void)b;
  __trap();
#endif
}
__device__ __forceinline__ float smf_lo(uint32_t w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float smf_hi(uint32_t w) { return __uint_as_float(w & 0xffff0000u); }
__device__ __forceinline__ float smf_amax2(uint32_t w) { return fmaxf(fabsf(smf_lo(w)), fabsf(smf_hi(w))); }
// |max| of a lane's 18 bf16 values: 16 latent dims (two uint4) and 2 rope dims (one word).
__device__ __forceinline__ float smf_amax18(uint4 a, uint4 b, uint32_t c) {
  float m = fmaxf(smf_amax2(a.x), smf_amax2(a.y));
  m = fmaxf(m, fmaxf(smf_amax2(a.z), smf_amax2(a.w)));
  m = fmaxf(m, fmaxf(smf_amax2(b.x), smf_amax2(b.y)));
  m = fmaxf(m, fmaxf(smf_amax2(b.z), smf_amax2(b.w)));
  return fmaxf(m, smf_amax2(c));
}
__device__ __forceinline__ float smf_warp_max(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}
// The smallest power of two 2^k with amax / 2^k <= 448 (the e4m3 maximum), k clamped to [-126, 127]; 1 for amax 0.
__device__ __forceinline__ float smf_scale(float amax) {
  if (!(amax > 0.f)) return 1.f;
  int e = 0;
  const float m = frexpf(amax, &e);   // amax = m 2^e, m in [0.5, 1): m 512 in [256, 512) is <= 448 iff m <= 0.875
  const int k = max(-126, min(127, e - 9 + (m > 0.875f ? 1 : 0)));
  return __int_as_float((k + 127) << 23);
}
// Two bf16 (one word, low half first) times an exact power-of-two inverse scale as two e4m3 codes (low byte first).
__device__ __forceinline__ uint32_t smf_e4m3x2(uint32_t w, float inv) {
  return (uint32_t)__nv_cvt_float2_to_fp8x2(make_float2(smf_lo(w) * inv, smf_hi(w) * inv), __NV_SATFINITE, __NV_E4M3);
}
__device__ __forceinline__ uint32_t smf_e4m3x4(uint32_t w0, uint32_t w1, float inv) {
  return smf_e4m3x2(w0, inv) | (smf_e4m3x2(w1, inv) << 16);
}
}

// Same arguments, launch shape and output contract as tf_glm53_smla_p; dynamic shared memory SMF_SMEM.
extern "C" __global__ void __launch_bounds__(SMLA_THREADS, 1) tf_glm53_smla_p_fp8(const __nv_bfloat16* qa, const __nv_bfloat16* qpe,
                                            const __nv_bfloat16* latent, const __nv_bfloat16* rope,
                                            const __nv_bfloat16* stage_latent, const __nv_bfloat16* stage_rope,
                                            const int64_t* positions, const int32_t* global_tokens,
                                            const int32_t* local_slots, const int32_t* global_counts,
                                            const int32_t* local_counts, const int32_t* dense_len,
                                            __nv_bfloat16* partial, int rows,
                                            int local_heads, int rank, int world, int prefill) {
  (void)local_slots; (void)local_counts; (void)dense_len;
  const int row0 = blockIdx.x * SMLA_QB, group = blockIdx.y, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  if (prefill == 0) __trap();
  if (row0 >= rows || group >= (local_heads + 31) / 32) return;
  extern __shared__ __align__(16) unsigned char smf_smem[];
  __nv_bfloat16* vbuf = reinterpret_cast<__nv_bfloat16*>(smf_smem + SMF_OFF_V);
  __nv_bfloat16* rbuf = reinterpret_cast<__nv_bfloat16*>(smf_smem + SMF_OFF_R);
  unsigned char* ebuf = smf_smem + SMF_OFF_E;
  __nv_bfloat16* vzero = reinterpret_cast<__nv_bfloat16*>(smf_smem + SMF_OFF_Z);
  __nv_bfloat16* ps = reinterpret_cast<__nv_bfloat16*>(smf_smem + SMF_OFF_P);
  float* hmax = reinterpret_cast<float*>(smf_smem + SMF_OFF_H);
  float* hden = hmax + SMLA_MH;
  float* halpha = hden + SMLA_MH;
  float* qsc = reinterpret_cast<float*>(smf_smem + SMF_OFF_QS);
  float* ksc = reinterpret_cast<float*>(smf_smem + SMF_OFF_KSC);
  int64_t* rpos = reinterpret_cast<int64_t*>(smf_smem + SMF_OFF_POS);
  int* rn = reinterpret_cast<int*>(smf_smem + SMF_OFF_N);
  int* kmap0 = reinterpret_cast<int*>(smf_smem + SMF_OFF_MAP);
  int* uslot0 = reinterpret_cast<int*>(smf_smem + SMF_OFF_USLOT);
  int* uplane0 = reinterpret_cast<int*>(smf_smem + SMF_OFF_UPLANE);
  int* ucount = reinterpret_cast<int*>(smf_smem + SMF_OFF_UCOUNT);
  const int g = lane >> 2, q = lane & 3;

  if (tid < SMLA_QB) {   // row b: position and list length (dense rows: every key 0..position)
    const int r = row0 + tid;
    int64_t qpos = -1;
    int n = 0;
    if (r < rows) {
      qpos = positions[r];
      n = qpos < 2048 ? (int)(qpos + 1) : global_counts[r];
    }
    rpos[tid] = qpos;
    rn[tid] = n;
  }
  if (tid < SMLA_MH) { hmax[tid] = -CUDART_INF_F; hden[tid] = 0.f; halpha[tid] = 1.f; }
  for (int x = tid; x < SMF_VS / 8; x += SMLA_THREADS) reinterpret_cast<uint4*>(vzero)[x] = make_uint4(0u, 0u, 0u, 0u);
  for (int x = tid; x < SMLA_UK * SMF_ES / 16; x += SMLA_THREADS) reinterpret_cast<uint4*>(ebuf)[x] = make_uint4(0u, 0u, 0u, 0u);
  // Q scales: warp w, query-heads 4w .. 4w + 3 (row qh >> 5, head qh & 31); lane: latent dims 16 lane .. 16 lane + 15
  // and rope dims 2 lane, 2 lane + 1.
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int qh = warp * 4 + i, r = row0 + (qh >> 5), h = qh & 31;
    float amax = 0.f;
    if (r < rows && (local_heads >= 32 || h < local_heads)) {
      const int64_t qr = local_heads < 32 ? (int64_t)r * local_heads + h : (((int64_t)group * rows + r) * 32 + h);
      const uint4* lat = reinterpret_cast<const uint4*>(qa + qr * LD + 16 * lane);
      amax = smf_amax18(lat[0], lat[1], smla_ld32(qpe + qr * PD + 2 * lane));
    }
    amax = smf_warp_max(amax);
    if (lane == 0) qsc[qh] = smf_scale(amax);
  }
  __syncthreads();

  // Score role: m-tile smt (query-heads 16 smt .. 16 smt + 15; row sb), n-tile nt (row keys 8 nt .. 8 nt + 7), K half kh.
  // Its Q A fragments for 9 k32 steps stay in registers: a[0] (head g, k0 .. k0 + 3), a[1] (g + 8, k0), a[2] (g,
  // k0 + 16), a[3] (g + 8, k0 + 16), k0 = step 32 + 4 q (the m16n8k32 e4m3 layout).
  const int smt = warp >> 2, sb = smt >> 1, nt = (warp >> 1) & 1, kh = warp & 1;
  uint32_t qf[SMF_KSTEPS][4];
  {
    const int r = row0 + sb;
    const int head_base = (smt & 1) * 16 + g;
    const float inv0 = __frcp_rn(qsc[smt * 16 + g]), inv1 = __frcp_rn(qsc[smt * 16 + g + 8]);
#pragma unroll
    for (int st = 0; st < SMF_KSTEPS; ++st) {
      const int k0 = (kh * SMF_KSTEPS + st) * 32 + 4 * q;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int h = head_base + 8 * (i & 1);
        const int kk = k0 + 16 * (i >> 1);
        uint32_t v = 0u;
        if (r < rows) {
          if (local_heads < 32) {
            if (h < local_heads) {
              const int64_t qr = (int64_t)r * local_heads + h;
              const uint2 x = kk < LD ? *reinterpret_cast<const uint2*>(qa + qr * LD + kk) : *reinterpret_cast<const uint2*>(qpe + qr * PD + (kk - LD));
              v = smf_e4m3x4(x.x, x.y, (i & 1) ? inv1 : inv0);
            }
          } else {
            const int64_t qr = ((int64_t)group * rows + r) * 32 + h;
            const uint2 x = kk < LD ? *reinterpret_cast<const uint2*>(qa + qr * LD + kk) : *reinterpret_cast<const uint2*>(qpe + qr * PD + (kk - LD));
            v = smf_e4m3x4(x.x, x.y, (i & 1) ? inv1 : inv0);
          }
        }
        qf[st][i] = v;
      }
    }
  }
  int steps = 0;
#pragma unroll
  for (int b = 0; b < SMLA_QB; ++b) steps = max(steps, (rn[b] + SMLA_KT - 1) / SMLA_KT);

  // Union tables: exactly tf_glm53_smla_p's entry() and meta().
  auto entry = [&](int step) -> int {
    const int b = lane / SMLA_KT, j = step * SMLA_KT + lane % SMLA_KT;
    if (b >= SMLA_QB || j >= rn[b]) return -1;
    return rpos[b] < 2048 ? j : global_tokens[(int64_t)(row0 + b) * TOPK + j];
  };
  auto meta = [&](int step, int raw) {
    const int mb = step % SMLA_META;
    const int b = lane / SMLA_KT;
    const int key = b < SMLA_QB && raw >= 0 && (int64_t)raw <= rpos[b] ? raw : -1;
    int first = -1;
#pragma unroll
    for (int i = 0; i < 32; ++i) {
      const int other = __shfl_sync(0xffffffffu, key, i);
      if (first < 0 && i < lane && key >= 0 && other == key) first = i;
    }
    const bool fresh = key >= 0 && first < 0;
    const unsigned owners = __ballot_sync(0xffffffffu, fresh);
    const int u = __popc(owners & ((1u << lane) - 1u));
    const int urow = __shfl_sync(0xffffffffu, u, first >= 0 ? first : lane);
    kmap0[mb * SMLA_UK + lane] = key >= 0 ? urow : -1;
    if (fresh) {
      uslot0[mb * SMLA_UK + u] = glm53_stage_plane_slot(key, world);
      uplane0[mb * SMLA_UK + u] = key % world == rank ? 0 : 1;
    }
    if (lane == 0) ucount[mb] = __popc(owners);
  };
  // All threads: 16-byte cp.async pieces of the step's bf16 union rows (64 latent pieces into the latent buffer, 8
  // rope pieces into the rope buffer), buffer step & 1.
  auto fetch = [&](int step) {
    const int mb = step % SMLA_META;
    __nv_bfloat16* vb = vbuf + (step & 1) * SMLA_UK * SMF_VS;
    __nv_bfloat16* rbs = rbuf + (step & 1) * SMLA_UK * SMF_RS;
    const int count = ucount[mb];
    for (int x = tid; x < count * 72; x += SMLA_THREADS) {
      const int u = x / 72, c = x % 72, slot = uslot0[mb * SMLA_UK + u];
      const bool staged = uplane0[mb * SMLA_UK + u] != 0;
      const __nv_bfloat16* src = c < 64 ? (staged ? stage_latent : latent) + (int64_t)slot * LD + c * 8
                                        : (staged ? stage_rope : rope) + (int64_t)slot * PD + (c - 64) * 8;
      __nv_bfloat16* dst = c < 64 ? vb + u * SMF_VS + c * 8 : rbs + u * SMF_RS + (c - 64) * 8;
      const unsigned d = (unsigned)__cvta_generic_to_shared(dst);
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(d), "l"(src) : "memory");
    }
    asm volatile("cp.async.commit_group;\n" ::: "memory");
  };

  float o[2 * SMLA_QB][4][4];
#pragma unroll
  for (int m = 0; m < 2 * SMLA_QB; ++m)
#pragma unroll
    for (int t = 0; t < 4; ++t)
#pragma unroll
      for (int c = 0; c < 4; ++c) o[m][t][c] = 0.f;

  int ahead = -1;   // warp 0: its lane's raw entry for step t + 1 during step t
  if (steps > 0) {
    if (warp == 0) { meta(0, entry(0)); ahead = entry(1); }
    __syncthreads();
    fetch(0);
  }
  for (int t = 0; t < steps; ++t) {
    const int mb = t % SMLA_META;
    const bool more = t + 1 < steps;
    // Ring slot (t + 1) % 3 was last read in step t - 2; bf16 buffers (t + 1) & 1 (and the scores in rope buffer
    // (t + 1) & 1) in step t - 1, before this sync.
    if (more && warp == 0) { meta(t + 1, ahead); ahead = entry(t + 2); }
    __syncthreads();
    if (more) { fetch(t + 1); asm volatile("cp.async.wait_group 1;\n" ::: "memory"); }
    else asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    __syncthreads();   // step t's bf16 union rows are in buffers t & 1
    const __nv_bfloat16* vb = vbuf + (t & 1) * SMLA_UK * SMF_VS;
    const __nv_bfloat16* rbt = rbuf + (t & 1) * SMLA_UK * SMF_RS;
    float* sc = reinterpret_cast<float*>(rbuf + (t & 1) * SMLA_UK * SMF_RS);   // [MH][KT] scores, after the conversion
    const int* kmap = kmap0 + mb * SMLA_UK;
    {   // e4m3 keys: warp w converts union rows w and w + 16 (lane: latent dims 16 lane .., rope dims 2 lane ..)
      const int count = ucount[mb];
      for (int u = warp; u < count; u += SMLA_THREADS / 32) {
        const uint4* lat = reinterpret_cast<const uint4*>(vb + u * SMF_VS + 16 * lane);
        const uint4 a0 = lat[0], a1 = lat[1];
        const uint32_t pe = smla_ld32(rbt + u * SMF_RS + 2 * lane);
        const float sk = smf_scale(smf_warp_max(smf_amax18(a0, a1, pe)));
        const float inv = __frcp_rn(sk);
        unsigned char* er = ebuf + u * SMF_ES;
        *reinterpret_cast<uint4*>(er + 16 * lane) = make_uint4(smf_e4m3x4(a0.x, a0.y, inv), smf_e4m3x4(a0.z, a0.w, inv),
                                                               smf_e4m3x4(a1.x, a1.y, inv), smf_e4m3x4(a1.z, a1.w, inv));
        *reinterpret_cast<uint16_t*>(er + LD + 2 * lane) = (uint16_t)smf_e4m3x2(pe, inv);
        if (lane == 0) ksc[u] = sk;
      }
    }
    __syncthreads();   // e4m3 keys and scales ready; rope buffer t & 1 is free for the scores
    if (t * SMLA_KT < rn[sb]) {   // scores: m-tile smt x the row's keys 8 nt .. 8 nt + 7 over K half kh
      float acc[4] = {0.f, 0.f, 0.f, 0.f};
      // ldmatrix x4: lanes 0-7 key rows 0-7 at byte k0, 8-15 at k0 + 16, 16-23 at k0 + 32, 24-31 at k0 + 48 ->
      // {b0, b1} of k32 step st, {b0, b1} of step st + 1. Invalid keys read row 0 (their column is discarded).
      const int u = kmap[sb * SMLA_KT + nt * 8 + (lane & 7)];
      const unsigned base = (unsigned)__cvta_generic_to_shared(ebuf + (u < 0 ? 0 : u) * SMF_ES) + kh * SMF_KSTEPS * 32 + (lane >> 3) * 16;
#pragma unroll
      for (int st = 0; st + 1 < SMF_KSTEPS; st += 2) {
        uint32_t bk[4];
        smla_ldsm_x4(bk, base + st * 32);
        smf_mma_e4m3(acc, qf[st], bk);
        smf_mma_e4m3(acc, qf[st + 1], bk + 2);
      }
      {
        uint32_t bk[2];
        smf_ldsm_x2(bk, base + (SMF_KSTEPS - 1) * 32);
        smf_mma_e4m3(acc, qf[SMF_KSTEPS - 1], bk);
      }
      float* s0 = sc + (smt * 16 + g) * SMLA_KT + nt * 8 + 2 * q;
      float* s1 = s0 + 8 * SMLA_KT;
      const int pair_bar = 1 + (warp >> 1);
      if (kh == 0) {
        s0[0] = acc[0]; s0[1] = acc[1]; s1[0] = acc[2]; s1[1] = acc[3];
        asm volatile("bar.arrive %0, 64;\n" ::"r"(pair_bar) : "memory");
      } else {
        asm volatile("bar.sync %0, 64;\n" ::"r"(pair_bar) : "memory");
        s0[0] = s0[0] + acc[0]; s0[1] = s0[1] + acc[1]; s1[0] = s1[0] + acc[2]; s1[1] = s1[1] + acc[3];
      }
    }
    __syncthreads();
    {   // online softmax, as tf_glm53_smla_p: query-head qh by 8 threads, keys 2 sub and 2 sub + 1
      const int qh = tid >> 3, sub = tid & 7, b = qh >> 5;
      if (t * SMLA_KT < rn[b]) {
        const float old = hmax[qh], den = hden[qh], qs = qsc[qh];
        float s[2];
#pragma unroll
        for (int i = 0; i < 2; ++i) {
          const int j = 2 * sub + i;
          const int u = kmap[b * SMLA_KT + j];
          s[i] = u < 0 ? -CUDART_INF_F : (sc[qh * SMLA_KT + j] * (qs * ksc[u])) * 0.0625f;
        }
        float mx = fmaxf(s[0], s[1]);
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 4));
        const float m = fmaxf(old, mx);
        const float alpha = isinf(old) ? 0.f : expf(old - m);
        const float p0 = isinf(s[0]) ? 0.f : expf(s[0] - m);
        const float p1 = isinf(s[1]) ? 0.f : expf(s[1] - m);
        float sum = p0 + p1;
        sum += __shfl_xor_sync(0xffffffffu, sum, 1);
        sum += __shfl_xor_sync(0xffffffffu, sum, 2);
        sum += __shfl_xor_sync(0xffffffffu, sum, 4);
        ps[qh * SMLA_PS + 2 * sub] = rb(p0);
        ps[qh * SMLA_PS + 2 * sub + 1] = rb(p1);
        __syncwarp();
        if (sub == 0) {
          if (isinf(m)) { halpha[qh] = 1.f; }   // nothing valid yet: O and the denominator stay zero
          else { halpha[qh] = alpha; hden[qh] = den * alpha + sum; hmax[qh] = m; }
        }
      }
    }
    __syncthreads();
    {   // O = O * alpha + P . V (bf16), as tf_glm53_smla_p, from the bf16 latent rows
      const int c0 = warp * 32;
#pragma unroll
      for (int b = 0; b < SMLA_QB; ++b) {
        if (t * SMLA_KT >= rn[b]) continue;
        const int u = kmap[b * SMLA_KT + (lane & 15)];
        const __nv_bfloat16* vr = u < 0 ? vzero : vb + u * SMF_VS;
        const unsigned vbase = (unsigned)__cvta_generic_to_shared(vr + c0 + (lane >> 4) * 8);
#pragma unroll
        for (int pair = 0; pair < 2; ++pair) {
          uint32_t bv[4];
          smla_ldsm_x4_trans(bv, vbase + pair * 32);
#pragma unroll
          for (int mt2 = 0; mt2 < 2; ++mt2) {
            const int mt = 2 * b + mt2;
            const __nv_bfloat16* p0 = ps + (mt * 16 + g) * SMLA_PS + 2 * q;
            const __nv_bfloat16* p1 = p0 + 8 * SMLA_PS;
            const uint32_t a[4] = {smla_ld32(p0), smla_ld32(p1), smla_ld32(p0 + 8), smla_ld32(p1 + 8)};
            const float al0 = halpha[mt * 16 + g], al1 = halpha[mt * 16 + g + 8];
#pragma unroll
            for (int h = 0; h < 2; ++h) {
              float* acc = o[mt][2 * pair + h];
              acc[0] *= al0; acc[1] *= al0; acc[2] *= al1; acc[3] *= al1;
              mla_mma16816(acc, a, bv + 2 * h);
            }
          }
        }
      }
    }
  }
  __syncthreads();
  if (local_heads < 32) {
    const int64_t partial_bytes = (int64_t)rows * local_heads * LD * (int)sizeof(__nv_bfloat16);
    float* group_lse = reinterpret_cast<float*>(reinterpret_cast<unsigned char*>(partial) + partial_bytes);
    const int c0 = warp * 32;
#pragma unroll
    for (int mt = 0; mt < 2 * SMLA_QB; ++mt) {
      const int r = row0 + (mt >> 1);
      if (r >= rows) continue;
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        const int head = (mt & 1) * 16 + g + 8 * hh;
        if (head < local_heads) {
          const float den = hden[mt * 16 + g + 8 * hh];
          __nv_bfloat16* dst = partial + ((int64_t)r * local_heads + head) * LD;
#pragma unroll
          for (int nt4 = 0; nt4 < 4; ++nt4) {
            const int col = c0 + nt4 * 8 + 2 * q;
            dst[col] = rb(den == 0.f ? 0.f : o[mt][nt4][2 * hh] / den);
            dst[col + 1] = rb(den == 0.f ? 0.f : o[mt][nt4][2 * hh + 1] / den);
          }
        }
      }
    }
    if (tid < SMLA_MH) {
      const int r_lse = row0 + (tid >> 5);
      const int head = tid & 31;
      if (r_lse < rows && head < local_heads) {
        group_lse[(int64_t)r_lse * local_heads + head] = hden[tid] == 0.f ? -CUDART_INF_F : hmax[tid] + logf(hden[tid]);
      }
    }
  } else {
    constexpr int group_row_bytes = 32 * LD * sizeof(__nv_bfloat16) + 32 * sizeof(float);
    const int64_t group_byte = (int64_t)group * rows * group_row_bytes;
    const int64_t group_partial = group_byte / (int)sizeof(__nv_bfloat16);
    const int c0 = warp * 32;
#pragma unroll
    for (int mt = 0; mt < 2 * SMLA_QB; ++mt) {
      const int r = row0 + (mt >> 1);
      if (r >= rows) continue;
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        const int head = (mt & 1) * 16 + g + 8 * hh;
        const float den = hden[mt * 16 + g + 8 * hh];
        __nv_bfloat16* dst = partial + group_partial + ((int64_t)r * 32 + head) * LD;
#pragma unroll
        for (int nt4 = 0; nt4 < 4; ++nt4) {
          const int col = c0 + nt4 * 8 + 2 * q;
          dst[col] = rb(den == 0.f ? 0.f : o[mt][nt4][2 * hh] / den);
          dst[col + 1] = rb(den == 0.f ? 0.f : o[mt][nt4][2 * hh + 1] / den);
        }
      }
    }
    if (tid < SMLA_MH && row0 + (tid >> 5) < rows) {
      float* group_lse = reinterpret_cast<float*>(reinterpret_cast<unsigned char*>(partial) + group_byte + rows * 32 * LD * (int)sizeof(__nv_bfloat16));
      group_lse[(int64_t)(row0 + (tid >> 5)) * 32 + (tid & 31)] = hden[tid] == 0.f ? -CUDART_INF_F : hmax[tid] + logf(hden[tid]);
    }
  }
}

// Decode exchanges only the peer's 32-head BF16 partial plus FP32 LSE record (1,028 bytes/head).
// The owner combines local and peer contributions in global rank order.
__device__ __forceinline__ void tf_glm53_mla_merge_task(const __nv_bfloat16* partial_local, const __nv_bfloat16* partial_peer,
                                                __nv_bfloat16* out,
                                                float* lse_out, int rows, int local_heads, int rank) {
  const int r = TF_VBID.x, h = TF_VBID.y, d = threadIdx.x;
  if (r >= rows || h >= local_heads) return;
  const int64_t row = ((int64_t)r * local_heads + h) * LD;
  const int64_t head = (int64_t)r * local_heads + h;
  const __nv_bfloat16* partial0 = rank == 0 ? partial_local : partial_peer;
  const __nv_bfloat16* partial1 = rank == 0 ? partial_peer : partial_local;
  const int64_t lse_bytes = (int64_t)rows * local_heads * LD * (int)sizeof(__nv_bfloat16);
  const float* lse_local = reinterpret_cast<const float*>(reinterpret_cast<const unsigned char*>(partial_local) + lse_bytes);
  const float* lse_peer = reinterpret_cast<const float*>(reinterpret_cast<const unsigned char*>(partial_peer) + lse_bytes);
  const float* lse0 = rank == 0 ? lse_local : lse_peer;
  const float* lse1 = rank == 0 ? lse_peer : lse_local;
  const float l0 = lse0[head], l1 = lse1[head];
  const float m = fmaxf(l0, l1);
  const float a = l0 == -CUDART_INF_F ? 0.f : glm_triton::exp_approx(l0 - m);
  const float b = l1 == -CUDART_INF_F ? 0.f : glm_triton::exp_approx(l1 - m);
  const float den = a + b;
  for (int i = d; i < LD; i += TF_VBDIM.x) {
    float acc = __fmaf_rn(tf_mla::bf(partial0[row + i]), a, 0.f);
    acc = __fmaf_rn(tf_mla::bf(partial1[row + i]), b, acc);
    out[row + i] = rb(den == 0.f ? 0.f : glm_triton::div_full(acc, den));
  }
  if (d == 0) lse_out[head] = den == 0.f ? -CUDART_INF_F : m + logf(den);
}

extern "C" __global__ void tf_glm53_mla_merge(const __nv_bfloat16* partial_local, const __nv_bfloat16* partial_peer,
                                                __nv_bfloat16* out, float* lse_out, int rows, int local_heads, int rank) {
    TF_VBLOCK_ENTER();
  tf_glm53_mla_merge_task(partial_local, partial_peer, out, lse_out, rows, local_heads, rank);
}

// Expand all 256 value columns from native FP8 kv_b rows.
__device__ __forceinline__ void tf_glm53_expand_v_task(const __nv_bfloat16* ol, const uint8_t* kvb_codes,
                                               const float* kvb_scales, __nv_bfloat16* out,
                                               int rows, int local_heads, int head0) {
  const int r = TF_VBID.x, h = TF_VBID.y;
  if (r >= rows || h >= local_heads) return;
  const int gh = head0 + h;
  for (int j = threadIdx.x; j < VD; j += TF_VBDIM.x) {
    float acc = 0.f;
    for (int c = 0; c < LD; ++c)
      acc = __fmaf_rn(tf_mla::bf(ol[((int64_t)r * local_heads + h) * LD + c]), kvb(kvb_codes, kvb_scales, gh * KR + ND + j, c), acc);
    out[((int64_t)r * local_heads + h) * VD + j] = rb(acc);
  }
}

extern "C" __global__ void tf_glm53_expand_v(const __nv_bfloat16* ol, const uint8_t* kvb_codes,
                                               const float* kvb_scales, __nv_bfloat16* out,
                                               int rows, int local_heads, int head0) {
    TF_VBLOCK_ENTER();
  tf_glm53_expand_v_task(ol, kvb_codes, kvb_scales, out, rows, local_heads, head0);
}

// Stage 32 value rows by 32 latent columns with contiguous lanes reading contiguous FP8
// bytes. One thread then owns each output and performs the scalar kernel's exact c-order FMA chain.
__device__ __forceinline__ void tf_glm53_expand_v_decode_task(const __nv_bfloat16* ol, const uint8_t* kvb_codes,
                                               const float* kvb_scales, __nv_bfloat16* out,
                                               int rows, int local_heads, int head0) {
  const int r = TF_VBID.x, h = TF_VBID.y, t = threadIdx.x;
  if (r >= rows || h >= local_heads) return;
  const int warps_per_block = TF_VBDIM.x / 32;
  const int threads_per_block = warps_per_block * 32;
  const int gh = head0 + h;
  const int j0 = TF_VBID.z * DECODE_EXPAND_TILE;
  const int64_t ol_base = ((int64_t)r * local_heads + h) * LD;
  __shared__ float tile[DECODE_EXPAND_TILE][DECODE_EXPAND_TILE];
  float sum = 0.f;
  #pragma unroll 1
  for (int c0 = 0; c0 < LD; c0 += DECODE_EXPAND_TILE) {
    for (int item = t; item < DECODE_EXPAND_TILE * DECODE_EXPAND_TILE; item += threads_per_block) {
      const int j = j0 + item / DECODE_EXPAND_TILE;
      const int c = c0 + item % DECODE_EXPAND_TILE;
      tile[item / DECODE_EXPAND_TILE][item % DECODE_EXPAND_TILE] =
          kvb(kvb_codes, kvb_scales, gh * KR + ND + j, c);
    }
    __syncthreads();
    if (t < DECODE_EXPAND_TILE) {
      #pragma unroll 1
      for (int c = 0; c < DECODE_EXPAND_TILE; ++c)
        sum = __fmaf_rn(tf_mla::bf(ol[ol_base + c0 + c]), tile[t][c], sum);
    }
    __syncthreads();
  }
  if (t < DECODE_EXPAND_TILE)
    out[((int64_t)r * local_heads + h) * VD + j0 + t] = rb(sum);
}

extern "C" __global__ void tf_glm53_expand_v_decode(const __nv_bfloat16* ol, const uint8_t* kvb_codes,
                                               const float* kvb_scales, __nv_bfloat16* out,
                                               int rows, int local_heads, int head0) {
    TF_VBLOCK_ENTER();
  tf_glm53_expand_v_decode_task(ol, kvb_codes, kvb_scales, out, rows, local_heads, head0);
}

// Each CTA owns 128 value outputs and stages a 128x32 byte weight tile. Every
// thread owns one output column and keeps one ordered accumulator
// per row; all R chains visit c=0..511 exactly as in the scalar kernel.
__device__ __forceinline__ void tf_glm53_expand_v_decode_v2_task(const __nv_bfloat16* ol, const uint8_t* kvb_codes,
                                               const float* kvb_scales, __nv_bfloat16* out,
                                               int rows, int local_heads, int head0) {
  const int h = TF_VBID.y, t = threadIdx.x;
  if (TF_VBDIM.x != DECODE_MLA_V2_THREADS || rows > 8) __trap();
  if (h >= local_heads) return;
  const int gh = head0 + h;
  const int j0 = TF_VBID.z * 128;
  __shared__ __align__(16) uint8_t tile[128][DECODE_V2_EXPAND_C];
  float acc[8] = {};
  const int kv_row0 = gh * KR + ND + j0;
  for (int c0 = 0; c0 < LD; c0 += DECODE_V2_EXPAND_C) {
    // The source has LD-byte rows, while the shared tile has 32-byte rows.
    // Load this thread's output row explicitly; flattening the tile into a
    // contiguous copy incorrectly mapped source columns 32..63 to output row 1.
    const uint4* src = reinterpret_cast<const uint4*>(kvb_codes + (int64_t)(kv_row0 + t) * LD + c0);
    uint4* dst = reinterpret_cast<uint4*>(&tile[t][0]);
    dst[0] = src[0];
    dst[1] = src[1];
    __syncthreads();
    const int row = kv_row0 + t;
    for (int r = 0; r < rows; ++r) {
      const int64_t ol_base = ((int64_t)r * local_heads + h) * LD + c0;
      #pragma unroll
      for (int c = 0; c < DECODE_V2_EXPAND_C; ++c) {
        const __nv_fp8_e4m3 code = *reinterpret_cast<const __nv_fp8_e4m3*>(&tile[t][c]);
        const float w = tf_mla::bf(rb((float)code * kvb_scales[(row / 128) * 4 + (c0 + c) / 128]));
        acc[r] = __fmaf_rn(tf_mla::bf(ol[ol_base + c]), w, acc[r]);
      }
    }
    __syncthreads();
  }
  for (int r = 0; r < rows; ++r)
    out[((int64_t)r * local_heads + h) * VD + j0 + t] = rb(acc[r]);
}

extern "C" __global__ void tf_glm53_expand_v_decode_v2(const __nv_bfloat16* ol, const uint8_t* kvb_codes,
                                               const float* kvb_scales, __nv_bfloat16* out,
                                               int rows, int local_heads, int head0) {
  TF_VBLOCK_ENTER();
  tf_glm53_expand_v_decode_v2_task(ol, kvb_codes, kvb_scales, out, rows, local_heads, head0);
}
