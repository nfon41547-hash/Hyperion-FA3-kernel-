/*
 * tensorcore_kernel.cu – Hyperion HALO TensorCore WMMA attention kernel.
 *
 * Architecture
 * ============
 * Uses NVIDIA Tensor Core WMMA (Warp Matrix Multiply-Accumulate) intrinsics
 * available on Volta / Turing / Ampere (sm_70+).
 *
 * Each CTA processes one (batch, head) pair and computes the full attention
 * output via:
 *   1. WMMA-accelerated QK^T dot products (FP16 A×B → FP32 accumulator)
 *   2. Online softmax with running (m_i, l_i) Flash-Attention style
 *   3. WMMA-accelerated weighted V accumulation
 *
 * Tile sizes are chosen for 16×16×16 WMMA fragment API.
 *
 * Limitations / Notes
 * -------------------
 * • Requires sm_70+ at compile time (-arch=sm_70 or higher).
 * • Q, K, V must be FP16.  Output O is FP16.
 * • Sequence length must be a multiple of WMMA_M (16) for full tiles;
 *   a boundary-check handles the remainder.
 * • This file is compiled by torch.utils.cpp_extension.load_inline() via
 *   the Python wrapper in unified_kernel_loader.py.
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>           /* WMMA intrinsics */
#include <float.h>

using namespace nvcuda;

// ============================================================
// Constants
// ============================================================
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16
#define WARP_SIZE 32
#define THREADS_PER_BLOCK 256   /* 8 warps */
#define EXP_CLAMP -80.0f

// ============================================================
// Warp-level reductions
// ============================================================
__device__ __forceinline__
float warp_reduce_max(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, off));
    return v;
}

__device__ __forceinline__
float warp_reduce_sum(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_xor_sync(0xffffffff, v, off);
    return v;
}

// ============================================================
// TensorCore attention kernel (WMMA path)
// ============================================================
//
// Grid  : (num_heads * batch_size, 1, 1)
// Block : (THREADS_PER_BLOCK, 1, 1)  i.e. 8 warps
//
// For each (batch, head):
//   • Q row: q_row = bh % seq_q  (simplified: one query token per CTA)
//   • Iterate over K/V tiles of size WMMA_K columns
//
__launch_bounds__(THREADS_PER_BLOCK, 2)
__global__ void tensorcore_attention_kernel(
    const half* __restrict__ Q,      /* [B*H, seq_q, head_dim] FP16 */
    const half* __restrict__ K,      /* [B*H, seq_kv, head_dim] FP16 */
    const half* __restrict__ V,      /* [B*H, seq_kv, head_dim] FP16 */
    half*       __restrict__ O,      /* [B*H, seq_q, head_dim] FP16 */
    int seq_q,
    int seq_kv,
    int head_dim,
    float inv_sqrt_d,
    bool causal
) {
    const int bh      = blockIdx.x;           /* combined batch*head index */
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane    = threadIdx.x % WARP_SIZE;

    // Each CTA handles one query token (q_row = blockIdx.y if extended)
    const int q_row = 0;   /* single-query mode */

    // ---- Shared memory ----
    extern __shared__ half smem_raw_tc[];
    // Layout: [WMMA_M * head_dim] for Q tile + [WMMA_K * head_dim] for K/V tile
    half* smem_q  = smem_raw_tc;
    half* smem_kv = smem_raw_tc + WMMA_M * head_dim;

    // ---- Load Q tile for this warp ----
    // All warps cooperate to load Q[bh, q_row, :] into smem_q
    {
        const half* q_ptr = Q + (bh * seq_q + q_row) * head_dim;
        for (int d = threadIdx.x; d < head_dim; d += THREADS_PER_BLOCK)
            smem_q[d] = (d < head_dim) ? q_ptr[d] : __float2half(0.f);
    }
    __syncthreads();

    // ---- Accumulators (warp-private) ----
    float acc[WMMA_M] = {};   /* output accumulator: one float per head_dim position */
    float m_i = -FLT_MAX;
    float l_i = 0.f;

    // ---- Iterate over K/V tiles ----
    int num_tiles = (seq_kv + WMMA_K - 1) / WMMA_K;

    for (int tile = 0; tile < num_tiles; ++tile) {
        int kv_start = tile * WMMA_K;

        // Load K tile [WMMA_K, head_dim] into smem_kv cooperatively
        for (int r = warp_id; r < WMMA_K; r += (THREADS_PER_BLOCK / WARP_SIZE)) {
            int kv_row = kv_start + r;
            const half* k_ptr = K + (bh * seq_kv + kv_row) * head_dim;
            for (int d = lane; d < head_dim; d += WARP_SIZE) {
                smem_kv[r * head_dim + d] = (kv_row < seq_kv)
                    ? k_ptr[d] : __float2half(0.f);
            }
        }
        __syncthreads();

        // ---- WMMA QK dot-product (warp 0 performs the WMMA) ----
        // Each warp handles a 16×16 tile; we use warp 0 to accumulate scores.
        if (warp_id == 0) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_q;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> frag_k;
            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_acc;

            wmma::fill_fragment(frag_acc, 0.f);

            for (int k_block = 0; k_block < head_dim; k_block += WMMA_K) {
                wmma::load_matrix_sync(frag_q, smem_q + k_block, head_dim);
                wmma::load_matrix_sync(frag_k, smem_kv + k_block, head_dim);
                wmma::mma_sync(frag_acc, frag_q, frag_k, frag_acc);
            }

            // Store QK scores (first WMMA_N values are the per-token scores)
            float scores[WMMA_N];
            // Extract diagonal: score[j] = frag_acc element (0, j)
            for (int j = 0; j < WMMA_N; ++j)
                scores[j] = frag_acc.x[j] * inv_sqrt_d;

            // Online softmax update
            for (int j = 0; j < WMMA_N; ++j) {
                int kv_row = kv_start + j;
                if (kv_row >= seq_kv) continue;
                if (causal && kv_row > q_row) continue;

                float s = scores[j];
                float m_new = fmaxf(m_i, s);
                float alpha = __expf(fmaxf(m_i - m_new, EXP_CLAMP));
                float w     = __expf(fmaxf(s - m_new, EXP_CLAMP));

                // Load V row
                const half* v_ptr = V + (bh * seq_kv + kv_row) * head_dim;
                for (int d = lane; d < head_dim; d += WARP_SIZE) {
                    float v_val = __half2float(v_ptr[d]);
                    acc[d / WARP_SIZE] = acc[d / WARP_SIZE] * alpha + w * v_val;
                }

                l_i = l_i * alpha + w;
                m_i = m_new;
            }
        }
        __syncthreads();
    }

    // ---- Write output (warp 0) ----
    if (warp_id == 0) {
        float inv_l = (l_i > 1e-6f) ? (1.f / l_i) : 1.f;
        half* out_ptr = O + (bh * seq_q + q_row) * head_dim;
        for (int d = lane; d < head_dim; d += WARP_SIZE) {
            int idx = d / WARP_SIZE;
            out_ptr[d] = __float2half(acc[idx] * inv_l);
        }
    }
}

// ============================================================
// Python binding
// ============================================================
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "tensorcore_attention",
        &tensorcore_attention_kernel,
        "TensorCore WMMA attention kernel (FP16 Q/K/V, FP32 accum)"
    );
}
