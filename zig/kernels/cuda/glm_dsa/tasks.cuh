// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
// Modified 2026 in the glm53-tp3-3x-sparks branch (GLM-5.3 on three DGX Sparks); see docs/tp3-changes.txt.
// GLM-5.3 decode task entry points. Keep arithmetic in task functions so a
// single-CTA megakernel can call them with explicit block coordinates.
#pragma once

#include "../tf_vblock.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include "../glm_glue_norm.cuh"
#include "../glm_glue_rows.cuh"

struct tf_glm53_swiglu_plain_params {
  const __nv_bfloat16* gu; __nv_bfloat16* out; int rows, width;
  static constexpr int block_size = 256, smem_bytes = 0;
};
struct tf_glm53_lowrank_partial_params {
  const __nv_bfloat16* x; const float* a; float* partials;
  int rows, hidden, rank, x_stride, a_stride, splits;
  static constexpr int block_size = 256, smem_bytes = 0;
};
struct tf_glm53_lowrank_decode_params {
  const __nv_bfloat16* x; const float* a; float* partials;
  int rows, hidden, rank, x_stride, a_stride, splits;
  static constexpr int block_size = 256, smem_bytes = 8 * 32 * sizeof(float);
};
struct tf_glm53_lowrank_reduce_params {
  const float* partials; float* mid; int total, splits;
  static constexpr int block_size = 256, smem_bytes = 0;
};
struct tf_glm53_add_partial_params {
  const float* add; float* acc; int count;
  static constexpr int block_size = 256, smem_bytes = 0;
};
struct tf_glm53_lowrank_combine_bf16_params {
  const float* routed; const float* shared; const float* b; const float* mid;
  __nv_bfloat16* out; int rows, hidden, rank, b_stride;
  static constexpr int block_size = 256, smem_bytes = 0;
};
struct tf_glm53_moe_post_params {
  const float* y; const float* weights; const float* shared; const float* b; const float* mid;
  __nv_bfloat16* out; int rows, hidden, rank, b_stride;
  static constexpr int block_size = 256, smem_bytes = 0;
};
struct tf_glm53_residual_rmsnorm_6144_params {
  __nv_bfloat16* x; const float* parts; const __nv_bfloat16* w;
  __nv_bfloat16* normed; int rows; float eps;
  static constexpr int block_size = 256, smem_bytes = 32;
};
struct tf_glm53_ablate_6144_params {
  __nv_bfloat16* x; const void* parts; int kind; const float* r; const float* a;
  const __nv_bfloat16* w; __nv_bfloat16* normed; int rows; float eps;
  static constexpr int block_size = 256, smem_bytes = 32;
};
struct tf_glm53_moe_residual_add_params {
  const __nv_bfloat16* x; __nv_bfloat16* xout; const __nv_bfloat16* parts; int count;
  static constexpr int block_size = 256, smem_bytes = 0;
};
struct tf_glm53_lowrank_finish_params {
  const float* b; const float* mid; float* acc;
  int rows, hidden, rank, b_stride;
  static constexpr int block_size = 256, smem_bytes = 0;
};
struct tf_glm53_embed_pick_params {
  const __nv_bfloat16* gathered; const int* owners; __nv_bfloat16* out;
  int rows, hidden, world;
  static constexpr int block_size = 256, smem_bytes = 0;
};

__device__ __forceinline__ float tf_glm53_triton_exp(float x) {
  float y;
  asm("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(x * 1.4426950408889634074f));
  return y;
}
__device__ __forceinline__ float tf_glm53_triton_div(float x, float y) {
  float z;
  asm("div.full.f32 %0, %1, %2;" : "=f"(z) : "f"(x), "f"(y));
  return z;
}

__device__ __forceinline__ void tf_glm53_swiglu_plain_task(const tf_glm53_swiglu_plain_params& p, int block_idx, int, char*) {
  const int i = block_idx * tf_glm53_swiglu_plain_params::block_size + threadIdx.x;
  const int total = p.rows * p.width;
  if (i >= total) return;
  const int row = i / p.width, col = i % p.width;
  const float g = __bfloat162float(p.gu[row * 2 * p.width + col]);
  const float u = __bfloat162float(p.gu[row * 2 * p.width + p.width + col]);
  const __nv_bfloat16 sg = __float2bfloat16_rn(tf_glm53_triton_div(g, 1.0f + tf_glm53_triton_exp(-g)));
  p.out[i] = __float2bfloat16_rn(__bfloat162float(sg) * u);
}

__device__ __forceinline__ void tf_glm53_lowrank_partial_task(const tf_glm53_lowrank_partial_params& p, int block_idx, int, char*) {
  const int i = block_idx * tf_glm53_lowrank_partial_params::block_size + threadIdx.x;
  const int total = p.rows * p.rank * p.splits;
  if (i >= total) return;
  const int split = i / (p.rows * p.rank), row = (i / p.rank) % p.rows, rr = i % p.rank;
  const int per = p.hidden / p.splits, begin = split * per, end = begin + per;
  float sum = 0.0f;
  for (int tile = begin; tile < end; tile += 64)
    for (int k = tile; k < tile + 64; ++k)
      sum = fmaf(__bfloat162float(p.x[row * p.x_stride + k]), p.a[k * p.a_stride + rr], sum);
  p.partials[i] = sum;
}

// A block covers one row, K split, and 32 adjacent rank columns. Each warp
// streams contiguous A values; the eight warp sums are folded in a fixed order.
__device__ __forceinline__ void tf_glm53_lowrank_decode_task(const tf_glm53_lowrank_decode_params& p, int block_idx, int, char* raw_smem) {
  const int rank_tiles = (p.rank + 31) / 32;
  const int rank_tile = block_idx % rank_tiles;
  const int split = (block_idx / rank_tiles) % p.splits;
  const int row = block_idx / (rank_tiles * p.splits);
  if (row >= p.rows) return;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int rr = rank_tile * 32 + lane;
  const int per = p.hidden / p.splits;
  const int begin = split * per;
  float sum = 0.0f;
  if (rr < p.rank) {
    for (int k = begin + warp; k < begin + per; k += 8)
      sum = fmaf(__bfloat162float(p.x[row * p.x_stride + k]), p.a[k * p.a_stride + rr], sum);
  }
  float* warp_sums = reinterpret_cast<float*>(raw_smem);
  warp_sums[warp * 32 + lane] = sum;
  __syncthreads();
  if (warp == 0 && rr < p.rank) {
    float total = warp_sums[lane];
#pragma unroll
    for (int w = 1; w < 8; ++w) total += warp_sums[w * 32 + lane];
    const int i = (split * p.rows + row) * p.rank + rr;
    p.partials[i] = total;
  }
}

__device__ __forceinline__ void tf_glm53_lowrank_reduce_task(const tf_glm53_lowrank_reduce_params& p, int block_idx, int, char*) {
  const int i = block_idx * tf_glm53_lowrank_reduce_params::block_size + threadIdx.x;
  if (i >= p.total) return;
  float sum = p.partials[i];
  for (int split = 1; split < p.splits; ++split) sum += p.partials[split * p.total + i];
  p.mid[i] = sum;
}

__device__ __forceinline__ void tf_glm53_add_partial_task(const tf_glm53_add_partial_params& p, int block_idx, int, char*) {
  const int i = block_idx * tf_glm53_add_partial_params::block_size + threadIdx.x;
  if (i < p.count) p.acc[i] = __fadd_rn(p.acc[i], p.add[i]);
}

__device__ __forceinline__ void tf_glm53_lowrank_combine_bf16_task(const tf_glm53_lowrank_combine_bf16_params& p, int block_idx, int, char*) {
  const int i = block_idx * tf_glm53_lowrank_combine_bf16_params::block_size + threadIdx.x;
  if (i >= p.rows * p.hidden) return;
  const int row = i / p.hidden, col = i % p.hidden;
  float shared_lowrank = p.shared[i];
  for (int r = 0; r < p.rank; ++r) shared_lowrank = fmaf(p.mid[row * p.rank + r], p.b[r * p.b_stride + col], shared_lowrank);
  p.out[i] = __float2bfloat16_rn(__fadd_rn(p.routed[i], shared_lowrank));
}

__device__ __forceinline__ void tf_glm53_lowrank_combine_bf16_decode_task(const tf_glm53_lowrank_combine_bf16_params& p, int block_idx) {
  const int i = block_idx * 32 + threadIdx.x;
  if (i >= p.rows * p.hidden) return;
  const int row = i / p.hidden, col = i % p.hidden;
  float shared_lowrank = p.shared[i];
  for (int r = 0; r < p.rank; ++r) shared_lowrank = fmaf(p.mid[row * p.rank + r], p.b[r * p.b_stride + col], shared_lowrank);
  p.out[i] = __float2bfloat16_rn(__fadd_rn(p.routed[i], shared_lowrank));
}

__device__ __forceinline__ void tf_glm53_moe_post_task(const tf_glm53_moe_post_params& p, int block_idx, int, char*) {
  const int i = block_idx * tf_glm53_moe_post_params::block_size + threadIdx.x;
  if (i >= p.rows * p.hidden) return;
  const int row = i / p.hidden, col = i % p.hidden;
  float routed = 0.f;
  for (int q = 0; q < 8; ++q) routed = fmaf(p.weights[row * 8 + q], p.y[(row * 8 + q) * p.hidden + col], routed);
  float sl = p.shared[i];
  for (int j = 0; j < p.rank; ++j) sl = fmaf(p.mid[row * p.rank + j], p.b[j * p.b_stride + col], sl);
  p.out[i] = __float2bfloat16_rn(__fadd_rn(routed, sl));
}

// PreparedAllReduceRMSNorm epilogue idea from NVIDIA flare-local-inference
// @47581cc (Apache-2.0); no code copied. Keep the residual BF16 boundary.
__device__ __forceinline__ void tf_glm53_residual_rmsnorm_6144_task(const tf_glm53_residual_rmsnorm_6144_params& p, int block_idx, int, char*) {
  constexpr int S = 8, NW = 8, T = NW * 32, TILE = S * T, REPS = (6144 + TILE - 1) / TILE;
  const int row = block_idx, t = threadIdx.x;
  const int world = ((unsigned int)p.rows >> 24) != 0 ? (int)((unsigned int)p.rows >> 24) : 2;
  const int rows = (int)((unsigned int)p.rows & 0x00FFFFFF);
  const int64_t rs = (int64_t)rows * 6144;
  float residual[REPS * S];
#pragma unroll
  for (int rep = 0; rep < REPS; ++rep) {
    const int d = rep * TILE + t * S;
    if (d >= 6144) continue;
    float sum_parts[S];
    glm_glue::ld8(p.parts + (int64_t)row * 6144 + d, sum_parts);
    for (int r = 1; r < world; ++r) {
      float part[S];
      glm_glue::ld8(p.parts + (int64_t)r * rs + (int64_t)row * 6144 + d, part);
#pragma unroll
      for (int j = 0; j < S; ++j) {
        sum_parts[j] = __fadd_rn(sum_parts[j], part[j]);
      }
    }
    glm_glue::ld8(p.x + (int64_t)row * 6144 + d, residual + rep * S);
#pragma unroll
    for (int j = 0; j < S; ++j) {
      residual[rep * S + j] = __fadd_rn(residual[rep * S + j], glm_glue::rbf(sum_parts[j]));
    }
    glm_glue::st8(p.x + (int64_t)row * 6144 + d, residual + rep * S);
  }
  TF_TASK_SYNC();
  glm_glue::rmsnorm<6144, 8192, 8, 8>(p.x, 6144, p.w, p.normed, 6144, nullptr, p.eps, 0, row);
}

__device__ __forceinline__ void tf_glm53_ablate_6144_task(const tf_glm53_ablate_6144_params& p, int row, int, char*) {
  __shared__ float sums[8];
  const int t = threadIdx.x;
  const int world = ((unsigned int)p.rows >> 24) != 0 ? (int)((unsigned int)p.rows >> 24) : 2;
  const int rows = (int)((unsigned int)p.rows & 0x00FFFFFF);
  const float* p0 = static_cast<const float*>(p.parts);
  const __nv_bfloat16* b0 = static_cast<const __nv_bfloat16*>(p.parts);
  float acc = 0.0f;
  for (int u = 0; u < 24; ++u) {
    const int j = t + 256 * u;
    float d = 0.0f;
    for (int r = 0; r < world; ++r) {
      const float val = p.kind == 0 ? p0[(int64_t)(r * rows + row) * 6144 + j]
        : __bfloat162float(b0[(int64_t)(r * rows + row) * 6144 + j]);
      d = __fadd_rn(d, val);
    }
    acc = fmaf(p.r[j], glm_glue::rbf(d), acc);
  }
  for (int o = 16; o >= 1; o >>= 1) acc += __shfl_xor_sync(0xffffffff, acc, o);
  if ((t & 31) == 0) sums[t >> 5] = acc;
  __syncthreads();
  if (t == 0) {
    float total = sums[0];
    for (int w = 1; w < 8; ++w) total = __fadd_rn(total, sums[w]);
    sums[0] = total;
  }
  __syncthreads();
  const float dot = sums[0];
  const float scale = __fmul_rn(*p.a, dot);
  for (int j = t; j < 6144; j += 256) {
    const float old = __bfloat162float(p.x[(int64_t)row * 6144 + j]);
    const float value = __fadd_rn(old, -__fmul_rn(scale, p.r[j]));
    p.x[(int64_t)row * 6144 + j] = __float2bfloat16_rn(value);
  }
  if (p.normed) {
    TF_TASK_SYNC();
    glm_glue::rmsnorm<6144, 8192, 8, 8>(p.x, 6144, p.w, p.normed, 6144, nullptr, p.eps, 0, row);
  }
}

__device__ __forceinline__ void tf_glm53_moe_residual_add_task(const tf_glm53_moe_residual_add_params& p, int block_idx, int, char*) {
  const int world = ((unsigned int)p.count >> 28) != 0 ? (int)((unsigned int)p.count >> 28) : 2;
  const int count = (int)((unsigned int)p.count & 0x0FFFFFFF);
  const int i = block_idx * tf_glm53_moe_residual_add_params::block_size + threadIdx.x;
  if (i >= count) return;
  float rank_sum = __bfloat162float(p.parts[i]);
  for (int r = 1; r < world; ++r) {
    rank_sum = __fadd_rn(rank_sum, __bfloat162float(p.parts[(int64_t)r * count + i]));
  }
  const float residual = __fadd_rn(__bfloat162float(p.x[i]), __bfloat162float(__float2bfloat16_rn(rank_sum)));
  p.xout[i] = __float2bfloat16_rn(residual);
}

__device__ __forceinline__ void tf_glm53_lowrank_finish_task(const tf_glm53_lowrank_finish_params& p, int block_idx, int, char*) {
  const int i = block_idx * tf_glm53_lowrank_finish_params::block_size + threadIdx.x;
  if (i >= p.rows * p.hidden) return;
  const int row = i / p.hidden, col = i % p.hidden;
  float sum = p.acc[i];
  for (int r = 0; r < p.rank; ++r) sum = fmaf(p.mid[row * p.rank + r], p.b[r * p.b_stride + col], sum);
  p.acc[i] = sum;
}

__device__ __forceinline__ void tf_glm53_embed_pick_task(const tf_glm53_embed_pick_params& p, int block_idx, int, char*) {
  const int i = block_idx * tf_glm53_embed_pick_params::block_size + threadIdx.x;
  if (i >= p.rows * p.hidden) return;
  const int row = i / p.hidden, col = i % p.hidden, owner = p.owners[row];
  p.out[i] = owner >= 0 && owner < p.world ? p.gathered[owner * p.rows * p.hidden + i] : __float2bfloat16_rn(0.0f);
}
