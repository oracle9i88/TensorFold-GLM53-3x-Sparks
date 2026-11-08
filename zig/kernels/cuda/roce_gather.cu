// SPDX-License-Identifier: Apache-2.0
// Modified 2026 by Kristian Aaron (Krishando). Original: ashhart/TensorFold@d1018b318 (Apache-2.0).
// Two-rank one-shot RoCE all-gather and rank-order all-reduce after
// flashinfer/experimental/b12x/comm/roce/_oneshot_cute.py (local-inference-lab/flashinfer@4092b28, Apache-2.0).
// Also see docs/roce-gather.md for this implementation's constraints.

#include <stdint.h>

namespace {

constexpr int C_SEQ = 0, C_NBYTES = 1, C_ERROR = 3, C_HCA = 4, C_ERROR_SEQ = 5, C_ABORT = 6;  // ctrl words
constexpr int C_ERROR_KIND = 7;
constexpr int FLAG_STRIDE = 128;
// counters (device memory, u32): epoch, poison, then stage and tail arrivals for each grid size class
constexpr int K_EPOCH = 0, K_POISON = 1, K_CLASSES = 4;

__device__ __forceinline__ uint32_t ld_relaxed_gpu(const uint32_t* p) {
    uint32_t v;
    asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}
__device__ __forceinline__ uint32_t ld_relaxed_sys(const uint32_t* p) {
    uint32_t v;
    asm volatile("ld.relaxed.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}
__device__ __forceinline__ uint32_t ld_acquire_sys(const uint32_t* p) {
    uint32_t v;
    asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}
__device__ __forceinline__ uint64_t globaltimer() {
    uint64_t v;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(v));
    return v;
}
__device__ __forceinline__ uint4 ld_relaxed_sys4(const uint4* p) {
    uint4 v;
    asm volatile("ld.relaxed.sys.global.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p) : "memory");
    return v;
}
__device__ __forceinline__ void st_relaxed_sys(uint32_t* p, uint32_t v) {
    asm volatile("st.relaxed.sys.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ void st_release_gpu(uint32_t* p, uint32_t v) {
    asm volatile("st.release.gpu.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ uint32_t arrive(uint32_t* p) {
    uint32_t old;
    asm volatile("atom.relaxed.gpu.global.add.u32 %0, [%1], 1;" : "=r"(old) : "l"(p) : "memory");
    return old;
}
__device__ __forceinline__ void fence_sys() { asm volatile("fence.sc.sys;" ::: "memory"); }
__device__ __forceinline__ void fence_gpu() { asm volatile("fence.sc.gpu;" ::: "memory"); }

}  // namespace

// in: this rank's shard (packs of 16 bytes); out: [rank 0's shard | rank 1's shard]. klass: log2(gridDim.x).
extern "C" __global__ void __launch_bounds__(512) roce_gather(
    const uint4* __restrict__ in, uint4* __restrict__ out, int packs, int rank, char* recv, char* flags, char* send,
    uint32_t* ctrl, long long slot_bytes, uint32_t* counters, int klass, uint64_t timeout_ns, int n_hca) {
    const int tid = threadIdx.x;
    uint32_t* stage = counters + 2 + klass;
    uint32_t* tail = counters + 2 + K_CLASSES + klass;
    __shared__ uint32_t poisoned;
    if (tid == 0) poisoned = ld_relaxed_gpu(counters + K_POISON) | ld_relaxed_sys(ctrl + C_ABORT);
    __syncthreads();
    if (poisoned != 0u) {
        const int at = blockIdx.x * blockDim.x + tid, stride = gridDim.x * blockDim.x;
        const uint4 nan4 = make_uint4(0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu);
        for (int i = at; i < 2 * packs; i += stride) out[i] = nan4;
        return;
    }
    const uint32_t seq = ld_relaxed_gpu(counters + K_EPOCH) + 1u;
    const uint32_t slot = seq & 1u;
    const int at = blockIdx.x * blockDim.x + tid, stride = gridDim.x * blockDim.x;

    uint4* mine = reinterpret_cast<uint4*>(send + (long long)slot * slot_bytes);  // 1. stage
    for (int i = at; i < packs; i += stride) mine[i] = in[i];
    __syncthreads();
    if (tid == 0) {  // 2. the doorbell, by the last block to stage
        fence_sys();
        if ((arrive(stage) + 1u) % gridDim.x == 0u) {
            st_relaxed_sys(ctrl + C_NBYTES + slot, (uint32_t)packs * 16u);
            fence_sys();
            st_relaxed_sys(ctrl + C_SEQ, seq);
        }
    }
    if (tid < n_hca) {  // 3. the peer's stripes
        const uint32_t* flag = reinterpret_cast<const uint32_t*>(flags + ((long long)slot * n_hca + tid) * FLAG_STRIDE);
        const uint64_t deadline = globaltimer() + timeout_ns;
        while (true) {
            const uint32_t peer_seq = ld_acquire_sys(flag);
            if (peer_seq == seq) {
                if (ld_relaxed_sys(flag + 1) != (uint32_t)packs * 16u) {
                    st_relaxed_sys(ctrl + C_HCA, (uint32_t)tid);
                    st_relaxed_sys(ctrl + C_ERROR_SEQ, seq);
                    st_relaxed_sys(ctrl + C_ERROR_KIND, 2u);
                    st_relaxed_sys(ctrl + C_ERROR, 1u);
                    st_release_gpu(counters + K_POISON, 1u);
                }
                break;
            }
            if (ld_relaxed_sys(ctrl + C_ABORT) != 0u || globaltimer() >= deadline) {
                st_relaxed_sys(ctrl + C_HCA, (uint32_t)tid);
                st_relaxed_sys(ctrl + C_ERROR_SEQ, seq);
                st_relaxed_sys(ctrl + C_ERROR_KIND, 1u);
                st_relaxed_sys(ctrl + C_ERROR, 1u);
                st_release_gpu(counters + K_POISON, 1u);
                break;
            }
        }
    }
    __syncthreads();
    if (tid == 0) poisoned = ld_relaxed_gpu(counters + K_POISON) | ld_relaxed_sys(ctrl + C_ABORT);
    __syncthreads();
    if (poisoned == 0u) {  // 4. both shards in rank order
        const uint4* theirs = reinterpret_cast<const uint4*>(recv + (long long)slot * slot_bytes);
        uint4* own_out = out + (long long)rank * packs;
        uint4* peer_out = out + (long long)(1 - rank) * packs;
        for (int i = at; i < packs; i += stride) {
            own_out[i] = in[i];
            peer_out[i] = ld_relaxed_sys4(theirs + i);
        }
    }
    fence_gpu();  // 5. the epoch, by the last block to finish
    __syncthreads();
    if (tid == 0 && (arrive(tail) + 1u) % gridDim.x == 0u) {
        fence_gpu();
        if (ld_relaxed_sys(ctrl + C_ERROR) == 0u && ld_relaxed_sys(ctrl + C_ABORT) == 0u) st_release_gpu(counters + K_EPOCH, seq);
    }
}

// A two-rank sum over roce_gather's rank-ordered device staging buffer.
#include <cuda_bf16.h>

extern "C" __global__ void roce_reduce_f32(const float* __restrict__ all, float* __restrict__ out, const uint32_t* ctrl, int count) {
    __shared__ uint32_t poisoned;
    if (threadIdx.x == 0) poisoned = ld_acquire_sys(ctrl + C_ERROR) | ld_acquire_sys(ctrl + C_ABORT);
    __syncthreads();
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) out[i] = poisoned == 0u ? all[i] + all[count + i] : __uint_as_float(0xffffffffu);
}

extern "C" __global__ void roce_reduce_bf16(const __nv_bfloat16* __restrict__ all,
                                              __nv_bfloat16* __restrict__ out, const uint32_t* ctrl, int count) {
    __shared__ uint32_t poisoned;
    if (threadIdx.x == 0) poisoned = ld_acquire_sys(ctrl + C_ERROR) | ld_acquire_sys(ctrl + C_ABORT);
    __syncthreads();
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        if (poisoned != 0u) {
            out[i] = __ushort_as_bfloat16(0xffffu);
            return;
        }
        const float a = __bfloat162float(all[i]);
        const float b = __bfloat162float(all[count + i]);
        out[i] = __float2bfloat16_rn(a + b);
    }
}

// The final token writer/sampler calls this at the end of the captured step.
extern "C" __global__ void roce_status_gate(const uint32_t* ctrl, uint32_t* token, uint32_t* status) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        const bool poisoned = ld_acquire_sys(ctrl + C_ERROR) != 0u || ld_acquire_sys(ctrl + C_ABORT) != 0u;
        *status = poisoned ? 1u : 0u;
        if (poisoned) *token = 0xffffffffu;
    }
}

// -- TP=3 over a cabled triangle -------------------------------------------------------------------------------------
// Each rank keeps two of roce_gather's two-rank exchanges, one with each neighbour, each with its own region,
// counters and host proxy (roce.zig initTriangle). One launch runs both: it stages the shard into both send slots,
// rings both doorbells, waits for both neighbours' flags, then writes all three shards in rank order. Two roce_gather
// launches in a row would chain the pairs ({0,1}, then {0,2}, then {1,2}) and cost three exchanges, not one.
struct RocePair {
    char* recv;
    char* flags;
    char* send;
    uint32_t* ctrl;
    uint32_t* counters;
    long long slot_bytes;
    int peer_slot;  // the neighbour's rank: where its shard lands in `out`
    int n_hca;
};

namespace {

__device__ __forceinline__ uint32_t pair_poisoned(const RocePair& p) {
    return ld_relaxed_gpu(p.counters + K_POISON) | ld_relaxed_sys(p.ctrl + C_ABORT);
}

__device__ __forceinline__ void pair_fail(const RocePair& p, int h, uint32_t seq, uint32_t kind) {
    st_relaxed_sys(p.ctrl + C_HCA, (uint32_t)h);
    st_relaxed_sys(p.ctrl + C_ERROR_SEQ, seq);
    st_relaxed_sys(p.ctrl + C_ERROR_KIND, kind);
    st_relaxed_sys(p.ctrl + C_ERROR, 1u);
    st_release_gpu(p.counters + K_POISON, 1u);
}

// roce_gather's step 3 for one HCA stripe of pair `p`.
__device__ __forceinline__ void pair_wait(const RocePair& p, uint32_t slot, uint32_t seq, int h, int packs, uint64_t timeout_ns) {
    const uint32_t* flag = reinterpret_cast<const uint32_t*>(p.flags + ((long long)slot * p.n_hca + h) * FLAG_STRIDE);
    const uint64_t deadline = globaltimer() + timeout_ns;
    while (true) {
        if (ld_acquire_sys(flag) == seq) {
            if (ld_relaxed_sys(flag + 1) != (uint32_t)packs * 16u) pair_fail(p, h, seq, 2u);
            return;
        }
        if (ld_relaxed_sys(p.ctrl + C_ABORT) != 0u || globaltimer() >= deadline) {
            pair_fail(p, h, seq, 1u);
            return;
        }
    }
}

}  // namespace

// in: this rank's shard (packs of 16 bytes); out: three shards in rank order. Pair a's arrival counters order both
// pairs' doorbells and epochs (each launch adds a whole grid to one class, as roce_gather does).
extern "C" __global__ void __launch_bounds__(512) roce_gather_tri(
    const uint4* __restrict__ in, uint4* __restrict__ out, int packs, int rank, RocePair a, RocePair b, int klass,
    uint64_t timeout_ns) {
    const int tid = threadIdx.x;
    uint32_t* stage = a.counters + 2 + klass;
    uint32_t* tail = a.counters + 2 + K_CLASSES + klass;
    const int at = blockIdx.x * blockDim.x + tid, stride = gridDim.x * blockDim.x;
    __shared__ uint32_t poisoned;
    if (tid == 0) poisoned = pair_poisoned(a) | pair_poisoned(b);
    __syncthreads();
    if (poisoned != 0u) {
        const uint4 nan4 = make_uint4(0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu);
        for (int i = at; i < packs; i += stride) {
            out[(long long)rank * packs + i] = nan4;
            out[(long long)a.peer_slot * packs + i] = nan4;
            out[(long long)b.peer_slot * packs + i] = nan4;
        }
        return;
    }
    const uint32_t seq_a = ld_relaxed_gpu(a.counters + K_EPOCH) + 1u, slot_a = seq_a & 1u;
    const uint32_t seq_b = ld_relaxed_gpu(b.counters + K_EPOCH) + 1u, slot_b = seq_b & 1u;

    uint4* mine_a = reinterpret_cast<uint4*>(a.send + (long long)slot_a * a.slot_bytes);  // 1. stage, both pairs
    uint4* mine_b = reinterpret_cast<uint4*>(b.send + (long long)slot_b * b.slot_bytes);
    for (int i = at; i < packs; i += stride) {
        const uint4 v = in[i];
        mine_a[i] = v;
        mine_b[i] = v;
    }
    __syncthreads();
    if (tid == 0) {  // 2. both doorbells, by the last block to stage
        fence_sys();
        if ((arrive(stage) + 1u) % gridDim.x == 0u) {
            st_relaxed_sys(a.ctrl + C_NBYTES + slot_a, (uint32_t)packs * 16u);
            st_relaxed_sys(b.ctrl + C_NBYTES + slot_b, (uint32_t)packs * 16u);
            fence_sys();
            st_relaxed_sys(a.ctrl + C_SEQ, seq_a);
            st_relaxed_sys(b.ctrl + C_SEQ, seq_b);
        }
    }
    // 3. both neighbours' stripes, each pair's waiters in a warp of their own
    if (tid < a.n_hca) pair_wait(a, slot_a, seq_a, tid, packs, timeout_ns);
    else if (tid >= 32 && tid < 32 + b.n_hca) pair_wait(b, slot_b, seq_b, tid - 32, packs, timeout_ns);
    __syncthreads();
    if (tid == 0) poisoned = pair_poisoned(a) | pair_poisoned(b);
    __syncthreads();
    if (poisoned == 0u) {  // 4. three shards in rank order
        const uint4* theirs_a = reinterpret_cast<const uint4*>(a.recv + (long long)slot_a * a.slot_bytes);
        const uint4* theirs_b = reinterpret_cast<const uint4*>(b.recv + (long long)slot_b * b.slot_bytes);
        uint4* own_out = out + (long long)rank * packs;
        uint4* a_out = out + (long long)a.peer_slot * packs;
        uint4* b_out = out + (long long)b.peer_slot * packs;
        for (int i = at; i < packs; i += stride) {
            own_out[i] = in[i];
            a_out[i] = ld_relaxed_sys4(theirs_a + i);
            b_out[i] = ld_relaxed_sys4(theirs_b + i);
        }
    }
    fence_gpu();  // 5. both epochs, by the last block to finish
    __syncthreads();
    if (tid == 0 && (arrive(tail) + 1u) % gridDim.x == 0u) {
        fence_gpu();
        const uint32_t bad = ld_relaxed_sys(a.ctrl + C_ERROR) | ld_relaxed_sys(a.ctrl + C_ABORT) |
                             ld_relaxed_sys(b.ctrl + C_ERROR) | ld_relaxed_sys(b.ctrl + C_ABORT);
        if (bad == 0u) {
            st_release_gpu(a.counters + K_EPOCH, seq_a);
            st_release_gpu(b.counters + K_EPOCH, seq_b);
        }
    }
}

// roce_status_gate over both pairs of a triangle (one launch, so a clean pair cannot overwrite the other's status).
extern "C" __global__ void roce_status_gate_tri(const uint32_t* ctrl_a, const uint32_t* ctrl_b, uint32_t* token,
                                                uint32_t* status) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        const uint32_t bad = ld_acquire_sys(ctrl_a + C_ERROR) | ld_acquire_sys(ctrl_a + C_ABORT) |
                             ld_acquire_sys(ctrl_b + C_ERROR) | ld_acquire_sys(ctrl_b + C_ABORT);
        *status = bad != 0u ? 1u : 0u;
        if (bad != 0u) *token = 0xffffffffu;
    }
}
