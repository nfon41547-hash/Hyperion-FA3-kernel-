/*
 * unified_kernel.cu – Hyperion HALO Unified Hybrid Kernel.
 *
 * Combines the Hyperion Apex V3 persistent kernel (INT4-K / FP8-V,
 * 6-stage cp.async pipeline, WFSA) with a TensorCore WMMA compute path.
 *
 * Runtime path selection
 * ----------------------
 * The kernel inspects the `use_tensorcore` flag passed in at launch:
 *   - true  → use WMMA fragments for QK dot-product (higher throughput on
 *             compute-bound workloads with large head_dim).
 *   - false → use the scalar INT4/FP8 path from Apex V3 (better latency for
 *             small batch sizes where memory bandwidth is the bottleneck).
 *
 * This file is intended to be compiled by
 * torch.utils.cpp_extension.load_inline() at Python startup.
 *
 * Compilation flags (recommended)
 * --------------------------------
 *   -O3 -arch=sm_86 --use_fast_math -Xptxas=-v,-dlcm=ca --maxrregcount=96
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <float.h>

using namespace nvcuda;

// ============================================================
// Shared constants
// ============================================================
#define WARP_SIZE         32
#define THREADS_PER_BLOCK 256
#define WARPS_PER_BLOCK   8
#define EXP_CLAMP         -80.0f
#define SMEM_PAD          8
#define WORK_CHUNK_SIZE   128
#define CP_ASYNC_STAGES   6
#define VEC_SIZE          4
#define ALIGN_BOUNDARY    128
#define WMMA_M            16
#define WMMA_N            16
#define WMMA_K            16

// ============================================================
// Persistent work counter
// ============================================================
__device__ int g_unified_work_counter = 0;

__global__ void unified_reset_counter() {
    if (threadIdx.x == 0 && blockIdx.x == 0)
        g_unified_work_counter = 0;
}

// ============================================================
// Utility: 128-byte alignment
// ============================================================
__device__ __host__ __forceinline__
size_t align128(size_t x) {
    return (x + ALIGN_BOUNDARY - 1) & ~(size_t)(ALIGN_BOUNDARY - 1);
}

// ============================================================
// INT4 dequant helpers
// ============================================================
__device__ __forceinline__ float int4_lo(uint8_t b) {
    int v = b & 0xF; if (v >= 8) v -= 16; return (float)v;
}
__device__ __forceinline__ float int4_hi(uint8_t b) {
    int v = (b >> 4) & 0xF; if (v >= 8) v -= 16; return (float)v;
}

// ============================================================
// FP8 E4M3 → FP32
// ============================================================
__device__ __forceinline__
float fp8_to_fp32(uint8_t x) {
    if (x == 0) return 0.f;
    float sign = (x >> 7) ? -1.f : 1.f;
    int exp = (x >> 3) & 0xF;
    float mant = (float)(x & 0x7);
    if (exp == 0) return sign * ldexpf(mant, -9);
    return sign * ldexpf(1.f + mant / 8.f, exp - 7);
}

// ============================================================
// XOR bank swizzle
// ============================================================
__device__ __forceinline__
int xor_swizzle(int row, int col, int stride) {
    int idx  = row * stride + col;
    int bank = (idx >> 2) & 31;
    int lane = threadIdx.x & 31;
    bank ^= (lane >> 3);
    return ((idx >> 5) << 5) | bank;
}

// ============================================================
// Warp reductions
// ============================================================
__device__ __forceinline__
float warp_max(float v) {
    for (int off = 16; off > 0; off >>= 1)
        v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, off));
    return v;
}
__device__ __forceinline__
float warp_sum(float v) {
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_xor_sync(0xffffffff, v, off);
    return v;
}

// ============================================================
// cp.async helpers (Ampere)
// ============================================================
__device__ __forceinline__
void cp_async_16B(void* dst, const void* src) {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                 :: "r"(__cvta_generic_to_shared(dst)), "l"(src));
#endif
}
__device__ __forceinline__ void cp_async_commit() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;\n" ::);
#endif
}
__device__ __forceinline__ void cp_async_wait_all() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_all;\n" ::);
#endif
}

// ============================================================
// Persistent fetch
// ============================================================
__device__ __forceinline__
int persistent_fetch(int chunk) {
    __shared__ int base;
    if (threadIdx.x == 0)
        base = atomicAdd(&g_unified_work_counter, chunk);
    __syncthreads();
    return base;
}

// ============================================================
// WorkItem
// ============================================================
struct WorkItem {
    int q_idx;
    int bh;
    int kv_sig;
    int prefetch_hint;
};

// ============================================================
// Unified hybrid kernel
// ============================================================
__launch_bounds__(THREADS_PER_BLOCK, 2)
__global__ void hyperion_unified_kernel(
    const WorkItem*  __restrict__ worklist,
    int total_work,
    const half*      __restrict__ Q,
    const float*     __restrict__ Q_scale,
    const uint8_t*   __restrict__ K_cache,
    const float*     __restrict__ K_scale,
    const uint8_t*   __restrict__ V_cache,
    const float*     __restrict__ V_scale,
    const int32_t*   __restrict__ context_lens,
    const int32_t*   __restrict__ block_tables,
    half*            __restrict__ O,
    int seq_q,
    int head_dim,
    int k_packed_bytes,
    int block_size,
    int num_blocks,
    float inv_sqrt_d,
    bool causal,
    bool use_tensorcore   /* runtime path selector */
) {
    const int lane    = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;

    // ---- Shared memory ----
    extern __shared__ uint8_t smem_unified[];
    const int STRIDE_K = k_packed_bytes + SMEM_PAD;
    const int STRIDE_V = head_dim       + SMEM_PAD;

    size_t k_stage_bytes = align128((size_t)STRIDE_K * block_size);
    size_t v_stage_bytes = align128((size_t)STRIDE_V * block_size);
    size_t one_stage     = k_stage_bytes + v_stage_bytes;
    size_t all_stages    = CP_ASYNC_STAGES * one_stage;

    uint8_t* k_smem[CP_ASYNC_STAGES];
    uint8_t* v_smem[CP_ASYNC_STAGES];
    for (int s = 0; s < CP_ASYNC_STAGES; ++s) {
        k_smem[s] = smem_unified + s * one_stage;
        v_smem[s] = smem_unified + s * one_stage + k_stage_bytes;
    }

    // TensorCore shared scratch (overlaid after pipeline stages)
    half* tc_smem = (half*)(smem_unified + all_stages);

    int* cta_ctr = (int*)(smem_unified + all_stages + align128((size_t)WMMA_M * head_dim * sizeof(half)));
    float* score_buf = (float*)(smem_unified + all_stages
                                + align128((size_t)WMMA_M * head_dim * sizeof(half))
                                + align128(4));

    // ---- Persistent work-stealing loop ----
    while (true) {
        int work_base = persistent_fetch(WORK_CHUNK_SIZE);
        if (work_base >= total_work) break;
        int work_end = min(work_base + WORK_CHUNK_SIZE, total_work);

        for (int widx = work_base + warp_id;
             widx < work_end;
             widx += WARPS_PER_BLOCK) {

            if (widx >= total_work) break;

            const int4* wl_v = reinterpret_cast<const int4*>(worklist);
            int4 raw = wl_v[widx];
            int q_idx = raw.x;
            int bh    = raw.y;

            int ctx_len   = context_lens[bh];
            int num_tiles = (ctx_len + block_size - 1) / block_size;
            const int32_t* btable = block_tables + bh * num_blocks;

            float acc_frag[VEC_SIZE] = {};
            float m_i = -FLT_MAX;
            float l_i = 0.f;

            if (threadIdx.x == 0) *cta_ctr = 0;
            __syncthreads();

            int stage = 0;
            int total_pipeline = num_tiles + CP_ASYNC_STAGES;

            for (int tile = 0; tile < total_pipeline; ++tile) {

                // ---- LOAD (warp 0-3): cp.async K/V tiles ----
                if (warp_id < 4) {
                    int load_tile    = tile;
                    int target_stage = stage;
                    if (load_tile < num_tiles) {
                        int ll = warp_id * WARP_SIZE + lane;
                        for (int vec = ll; vec < block_size; vec += 4 * WARP_SIZE) {
                            int g_idx = load_tile * block_size + vec;
                            if (g_idx >= ctx_len) continue;
                            int blk_id = btable[g_idx / block_size];
                            if (blk_id < 0 || blk_id >= num_blocks) continue;
                            int64_t slot = (int64_t)blk_id * block_size + g_idx % block_size;

                            // K cp.async
                            const uint8_t* k_src = K_cache + slot * k_packed_bytes;
                            int k_off = (g_idx % block_size) * STRIDE_K;
                            for (int cp = lane * 16; cp + 16 <= k_packed_bytes; cp += WARP_SIZE * 16)
                                cp_async_16B(k_smem[target_stage] + k_off + cp, k_src + cp);

                            // V cp.async
                            const uint8_t* v_src = V_cache + slot * head_dim;
                            int v_off = (g_idx % block_size) * STRIDE_V;
                            for (int cp = lane * 16; cp + 16 <= head_dim; cp += WARP_SIZE * 16)
                                cp_async_16B(v_smem[target_stage] + v_off + cp, v_src + cp);
                        }
                        if (lane == 0 && warp_id == 0) cp_async_commit();
                    }
                }
                __syncthreads();

                // ---- COMPUTE ----
                if (tile >= CP_ASYNC_STAGES) {
                    int compute_tile  = tile - CP_ASYNC_STAGES;
                    int smem_stage    = compute_tile % CP_ASYNC_STAGES;

                    if (threadIdx.x == 0) cp_async_wait_all();
                    __syncthreads();

                    if (!use_tensorcore) {
                        // ---- Scalar INT4/FP8 path (Apex V3 style) ----
                        if (warp_id == 4 || warp_id == 5) {
                            int rows_per_warp = (block_size + 1) / 2;
                            int row_start = (warp_id - 4) * rows_per_warp;
                            int row_end   = min(row_start + rows_per_warp, block_size);
                            for (int row = row_start; row < row_end; ++row) {
                                int kv_row = compute_tile * block_size + row;
                                if (kv_row >= ctx_len || (causal && kv_row > q_idx)) {
                                    if (lane == 0) score_buf[row] = -FLT_MAX;
                                    continue;
                                }
                                float dot = 0.f;
                                int k_base = row * STRIDE_K;
                                for (int b = lane; b < k_packed_bytes; b += WARP_SIZE) {
                                    uint8_t kp = k_smem[smem_stage][k_base + b];
                                    int d0 = b * 2, d1 = d0 + 1;
                                    float q0 = (d0 < head_dim) ? __half2float(Q[(bh * seq_q + q_idx) * head_dim + d0]) : 0.f;
                                    float q1 = (d1 < head_dim) ? __half2float(Q[(bh * seq_q + q_idx) * head_dim + d1]) : 0.f;
                                    dot += q0 * int4_lo(kp) + q1 * int4_hi(kp);
                                }
                                dot = warp_sum(dot);
                                int64_t s = (int64_t)btable[kv_row / block_size] * block_size + kv_row % block_size;
                                dot = dot * K_scale[s] * inv_sqrt_d;
                                if (lane == 0) score_buf[row] = dot;
                            }
                        }
                        __syncthreads();

                        if (warp_id == 6 || warp_id == 7) {
                            int dim_per_warp = (head_dim + 1) / 2;
                            int dim_start = (warp_id - 6) * dim_per_warp;
                            int dim_end   = min(dim_start + dim_per_warp, head_dim);
                            for (int row = 0; row < block_size; ++row) {
                                float s = score_buf[row];
                                if (s <= -FLT_MAX * 0.5f) continue;
                                int kv_row = compute_tile * block_size + row;
                                if (kv_row >= ctx_len) continue;
                                float m_new = fmaxf(m_i, s);
                                float alpha = __expf(fmaxf(m_i - m_new, EXP_CLAMP));
                                float w     = __expf(fmaxf(s - m_new, EXP_CLAMP));
                                int64_t slot = (int64_t)btable[kv_row / block_size] * block_size + kv_row % block_size;
                                float vs = V_scale[slot];
                                int v_base = row * STRIDE_V;
                                int d_base = lane * VEC_SIZE;
                                for (int d = dim_start + lane; d < dim_end; d += WARP_SIZE) {
                                    float v_val = fp8_to_fp32(v_smem[smem_stage][v_base + d]) * vs;
                                    int fi = d - d_base;
                                    if (fi >= 0 && fi < VEC_SIZE)
                                        acc_frag[fi] = acc_frag[fi] * alpha + w * v_val;
                                }
                                l_i = l_i * alpha + w;
                                m_i = m_new;
                            }
                        }
                    } else {
                        // ---- TensorCore WMMA path ----
                        // Warp 0: compute QK scores via WMMA and store in score_buf
                        if (warp_id == 0) {
                            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_q;
                            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> frag_k;
                            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_acc;
                            wmma::fill_fragment(frag_acc, 0.f);

                            // Load Q fragment from global mem into tc_smem first
                            const half* q_ptr = Q + (bh * seq_q + q_idx) * head_dim;
                            for (int d = lane; d < head_dim; d += WARP_SIZE)
                                tc_smem[d] = (d < head_dim) ? q_ptr[d] : __float2half(0.f);

                            // Convert K tile from INT4 to FP16 in tc_smem
                            // (rows: WMMA_K, cols: head_dim)
                            for (int row = lane; row < min(block_size, WMMA_K); row += WARP_SIZE) {
                                int kv_row = compute_tile * block_size + row;
                                int k_base = row * STRIDE_K;
                                for (int b = 0; b < k_packed_bytes; ++b) {
                                    uint8_t kp = k_smem[smem_stage][k_base + b];
                                    tc_smem[WMMA_M * head_dim + row * head_dim + b * 2]     = __float2half(int4_lo(kp));
                                    tc_smem[WMMA_M * head_dim + row * head_dim + b * 2 + 1] = __float2half(int4_hi(kp));
                                }
                                (void)kv_row; // suppress unused warning
                            }
                            __syncwarp();

                            for (int k_block = 0; k_block < head_dim; k_block += WMMA_K) {
                                wmma::load_matrix_sync(frag_q, tc_smem + k_block, head_dim);
                                wmma::load_matrix_sync(frag_k, tc_smem + WMMA_M * head_dim + k_block, head_dim);
                                wmma::mma_sync(frag_acc, frag_q, frag_k, frag_acc);
                            }

                            // Store scores
                            for (int j = 0; j < WMMA_N && j < block_size; ++j) {
                                int kv_row = compute_tile * block_size + j;
                                bool masked = (kv_row >= ctx_len) || (causal && kv_row > q_idx);
                                int64_t s = masked ? 0LL
                                    : (int64_t)btable[kv_row / block_size] * block_size + kv_row % block_size;
                                float ks = masked ? 0.f : K_scale[s];
                                score_buf[j] = masked ? -FLT_MAX : frag_acc.x[j] * ks * inv_sqrt_d;
                            }
                        }
                        __syncthreads();

                        // All warps participate in V accumulation
                        for (int row = 0; row < block_size; ++row) {
                            float s = score_buf[row];
                            if (s <= -FLT_MAX * 0.5f) continue;
                            int kv_row = compute_tile * block_size + row;
                            if (kv_row >= ctx_len) continue;
                            float m_new = fmaxf(m_i, s);
                            float alpha = __expf(fmaxf(m_i - m_new, EXP_CLAMP));
                            float w     = __expf(fmaxf(s - m_new, EXP_CLAMP));
                            int64_t slot = (int64_t)btable[kv_row / block_size] * block_size + kv_row % block_size;
                            float vs = V_scale[slot];
                            int v_base = row * STRIDE_V;
                            int d_base = lane * VEC_SIZE;
                            for (int d = lane; d < head_dim; d += WARP_SIZE) {
                                float v_val = fp8_to_fp32(v_smem[smem_stage][v_base + d]) * vs;
                                int fi = d - d_base;
                                if (fi >= 0 && fi < VEC_SIZE)
                                    acc_frag[fi] = acc_frag[fi] * alpha + w * v_val;
                            }
                            l_i = l_i * alpha + w;
                            m_i = m_new;
                        }
                    }
                    __syncthreads();
                }
                stage = (stage + 1) % CP_ASYNC_STAGES;
            }

            // ---- Write output ----
            if (warp_id >= 6) {
                float inv_l = (l_i > 1e-6f) ? 1.f / l_i : 1.f;
                half* out_row = O + (bh * seq_q + q_idx) * head_dim;
                for (int i = 0; i < VEC_SIZE; ++i) {
                    int d = lane * VEC_SIZE + i;
                    if (d < head_dim)
                        out_row[d] = __float2half(acc_frag[i] * inv_l);
                }
            }
        }
    }
}

// ============================================================
// Python bindings
// ============================================================
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("hyperion_unified_kernel", &hyperion_unified_kernel,
          "Unified hybrid attention kernel (INT4-K / FP8-V + optional TensorCore)");
    m.def("unified_reset_counter", &unified_reset_counter,
          "Reset persistent work counter");
}
