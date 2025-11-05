#pragma once

#include <vector>
#include <algorithm>
#include <execution>
#include <immintrin.h>
#include <cassert>
#include <cstring>

namespace matmul {

// Micro-kernel dimensions (register blocking)
constexpr size_t MR = 6;  // 6 rows - fits in registers with 16 cols
constexpr size_t NR = 16; // 16 cols = 2 AVX2 registers

// Cache blocking parameters
constexpr size_t MC = 256;  // M dimension cache block (L2)
constexpr size_t KC = 256;  // K dimension cache block (L2)
constexpr size_t NC = 4096; // N dimension cache block (L3)

// Parallelism threshold
constexpr size_t PARALLEL_THRESHOLD = 512;

/**
 * @brief Professional-grade GEMM implementation
 *
 * Based on BLIS/Goto's algorithm:
 * 1. Pack B panels into contiguous memory (NC × KC)
 * 2. For each MC × KC block of A:
 *    - Pack A panel
 *    - Compute MC × NC with packed data
 * 3. Micro-kernel: 6×16 register-blocked GEMM
 */
class TiledMatMul {
public:
    using Matrix = std::vector<float>;

    static void multiply(const Matrix& A, const Matrix& B, Matrix& C,
                        size_t M, size_t N, size_t K) {
        assert(A.size() == M * K);
        assert(B.size() == K * N);
        assert(C.size() == M * N);

        // Zero output
        std::fill(std::execution::par_unseq, C.begin(), C.end(), 0.0f);

        // Adaptive parallelism: use sequential for small matrices
        if (M < PARALLEL_THRESHOLD || N < PARALLEL_THRESHOLD) {
            gemm_sequential(A, B, C, M, N, K);
        } else {
            gemm_parallel(A, B, C, M, N, K);
        }
    }

private:
    /**
     * @brief Sequential GEMM for small matrices
     */
    static void gemm_sequential(const Matrix& A, const Matrix& B, Matrix& C,
                                size_t M, size_t N, size_t K) {
        // Allocate packing buffers
        std::vector<float> B_packed(KC * NC);
        std::vector<float> A_packed(MC * KC);

        // Loop over N dimension (panels of B)
        for (size_t j = 0; j < N; j += NC) {
            size_t jb = std::min(NC, N - j);

            // Loop over K dimension
            for (size_t p = 0; p < K; p += KC) {
                size_t pb = std::min(KC, K - p);

                // Pack B panel: B[p:p+pb, j:j+jb] -> B_packed
                pack_B_panel(B, B_packed.data(), p, pb, j, jb, N);

                // Loop over M dimension
                for (size_t i = 0; i < M; i += MC) {
                    size_t ib = std::min(MC, M - i);

                    // Pack A panel: A[i:i+ib, p:p+pb] -> A_packed
                    pack_A_panel(A, A_packed.data(), i, ib, p, pb, K);

                    // Macro-kernel: C[i:i+ib, j:j+jb] += A_packed @ B_packed
                    macro_kernel(A_packed.data(), B_packed.data(), C,
                                ib, jb, pb, i, j, N, p == 0);
                }
            }
        }
    }

    /**
     * @brief Parallel GEMM for large matrices
     */
    static void gemm_parallel(const Matrix& A, const Matrix& B, Matrix& C,
                              size_t M, size_t N, size_t K) {
        // Pre-allocate packing buffer for B (shared across threads)
        std::vector<float> B_packed(KC * NC);

        // Generate work items for parallelization
        struct WorkItem {
            size_t i, j, p;
            size_t ib, jb, pb;
        };
        std::vector<WorkItem> work_items;

        for (size_t j = 0; j < N; j += NC) {
            size_t jb = std::min(NC, N - j);
            for (size_t p = 0; p < K; p += KC) {
                size_t pb = std::min(KC, K - p);
                for (size_t i = 0; i < M; i += MC) {
                    size_t ib = std::min(MC, M - i);
                    work_items.push_back({i, j, p, ib, jb, pb});
                }
            }
        }

        // Process panels of B sequentially (each uses all threads for i-dimension)
        for (size_t j = 0; j < N; j += NC) {
            size_t jb = std::min(NC, N - j);

            for (size_t p = 0; p < K; p += KC) {
                size_t pb = std::min(KC, K - p);

                // Pack B panel once
                pack_B_panel(B, B_packed.data(), p, pb, j, jb, N);

                // Parallel over M dimension
                std::vector<size_t> i_blocks;
                for (size_t i = 0; i < M; i += MC) {
                    i_blocks.push_back(i);
                }

                std::for_each(std::execution::par, i_blocks.begin(), i_blocks.end(),
                    [&](size_t i) {
                        size_t ib = std::min(MC, M - i);

                        // Thread-local A packing buffer
                        std::vector<float> A_packed(MC * KC);
                        pack_A_panel(A, A_packed.data(), i, ib, p, pb, K);

                        // Compute this block
                        macro_kernel(A_packed.data(), B_packed.data(), C,
                                    ib, jb, pb, i, j, N, p == 0);
                    });
            }
        }
    }

    /**
     * @brief Pack panel of B into row-major format for SIMD efficiency
     *
     * B_packed[k, j] = B[p+k, j_start+j] for k=0..pb-1, j=0..jb-1
     * Packed in row-major: consecutive j's are contiguous (for SIMD loads)
     */
    static void pack_B_panel(const Matrix& B, float* B_packed,
                            size_t p, size_t pb, size_t j_start, size_t jb, size_t N) {
        for (size_t k = 0; k < pb; ++k) {
            for (size_t j = 0; j < jb; ++j) {
                B_packed[k * jb + j] = B[(p + k) * N + (j_start + j)];
            }
        }
    }

    /**
     * @brief Pack panel of A into row-major format
     *
     * A_packed[i, k] = A[i_start+i, p+k] for i=0..ib-1, k=0..pb-1
     */
    static void pack_A_panel(const Matrix& A, float* A_packed,
                            size_t i_start, size_t ib, size_t p, size_t pb, size_t K) {
        for (size_t i = 0; i < ib; ++i) {
            for (size_t k = 0; k < pb; ++k) {
                A_packed[i * pb + k] = A[(i_start + i) * K + (p + k)];
            }
        }
    }

    /**
     * @brief Macro-kernel: compute C[i:i+ib, j:j+jb] += A_packed @ B_packed
     */
    static void macro_kernel(const float* A_packed, const float* B_packed, Matrix& C,
                            size_t ib, size_t jb, size_t pb,
                            size_t i_start, size_t j_start, size_t N,
                            bool first_k_iter) {
        // Loop over MR×NR tiles
        for (size_t i = 0; i < ib; i += MR) {
            size_t mr = std::min(MR, ib - i);

            for (size_t j = 0; j < jb; j += NR) {
                size_t nr = std::min(NR, jb - j);

                // Call micro-kernel
                // B_packed layout: B_packed[k * jb + j]
                micro_kernel(A_packed + i * pb, B_packed + j,
                            C, mr, nr, pb, jb,
                            i_start + i, j_start + j, N,
                            first_k_iter);
            }
        }
    }

    /**
     * @brief Micro-kernel: 6×16 register-blocked GEMM
     *
     * Computes C[i:i+6, j:j+16] += A[i:i+6, 0:pb] @ B[0:pb, j:j+16]
     * where A and B are in packed format
     *
     * Register allocation:
     * - 12 YMM registers for C (6 rows × 2 cols)
     * - 2 YMM for B
     * - 1 YMM for A broadcast
     */
    static void micro_kernel(const float* A, const float* B, Matrix& C,
                            size_t mr, size_t nr, size_t kc, size_t jb_packed,
                            size_t i_start, size_t j_start, size_t N,
                            bool first_k_iter) {
        // Handle full 6×16 case with maximum optimization
        if (mr == MR && nr == NR) {
            micro_kernel_6x16(A, B, C, kc, jb_packed, i_start, j_start, N, first_k_iter);
        } else {
            // Generic fallback for edge cases
            micro_kernel_generic(A, B, C, mr, nr, kc, jb_packed, i_start, j_start, N, first_k_iter);
        }
    }

    /**
     * @brief Optimized 6×16 micro-kernel
     */
    static void micro_kernel_6x16(const float* __restrict__ A,
                                  const float* __restrict__ B,
                                  Matrix& C, size_t kc, size_t jb_packed,
                                  size_t i_start, size_t j_start, size_t N,
                                  bool first_k_iter) {
        // C accumulator registers: 6 rows × 2 columns (16 floats) = 12 registers
        __m256 c_00_07, c_01_08;  // Row 0
        __m256 c_10_17, c_11_18;  // Row 1
        __m256 c_20_27, c_21_28;  // Row 2
        __m256 c_30_37, c_31_38;  // Row 3
        __m256 c_40_47, c_41_48;  // Row 4
        __m256 c_50_57, c_51_58;  // Row 5

        // Load or zero C
        if (first_k_iter) {
            c_00_07 = _mm256_setzero_ps(); c_01_08 = _mm256_setzero_ps();
            c_10_17 = _mm256_setzero_ps(); c_11_18 = _mm256_setzero_ps();
            c_20_27 = _mm256_setzero_ps(); c_21_28 = _mm256_setzero_ps();
            c_30_37 = _mm256_setzero_ps(); c_31_38 = _mm256_setzero_ps();
            c_40_47 = _mm256_setzero_ps(); c_41_48 = _mm256_setzero_ps();
            c_50_57 = _mm256_setzero_ps(); c_51_58 = _mm256_setzero_ps();
        } else {
            const float* c_ptr = &C[i_start * N + j_start];
            c_00_07 = _mm256_loadu_ps(c_ptr); c_01_08 = _mm256_loadu_ps(c_ptr + 8);
            c_ptr += N;
            c_10_17 = _mm256_loadu_ps(c_ptr); c_11_18 = _mm256_loadu_ps(c_ptr + 8);
            c_ptr += N;
            c_20_27 = _mm256_loadu_ps(c_ptr); c_21_28 = _mm256_loadu_ps(c_ptr + 8);
            c_ptr += N;
            c_30_37 = _mm256_loadu_ps(c_ptr); c_31_38 = _mm256_loadu_ps(c_ptr + 8);
            c_ptr += N;
            c_40_47 = _mm256_loadu_ps(c_ptr); c_41_48 = _mm256_loadu_ps(c_ptr + 8);
            c_ptr += N;
            c_50_57 = _mm256_loadu_ps(c_ptr); c_51_58 = _mm256_loadu_ps(c_ptr + 8);
        }

        // Main computational loop
        for (size_t k = 0; k < kc; ++k) {
            // Prefetch next iteration
            _mm_prefetch((const char*)(A + kc), _MM_HINT_T0);
            _mm_prefetch((const char*)(B + 16), _MM_HINT_T0);

            // Load B[k, 0:16] from row-major packed format
            // B_packed[k * jb_packed + j] where j is column offset within panel
            const float* b_row = B + k * jb_packed;
            __m256 b_0_7 = _mm256_loadu_ps(b_row);
            __m256 b_8_15 = _mm256_loadu_ps(b_row + 8);

            // Row 0
            __m256 a0 = _mm256_broadcast_ss(A + k);
            c_00_07 = _mm256_fmadd_ps(a0, b_0_7, c_00_07);
            c_01_08 = _mm256_fmadd_ps(a0, b_8_15, c_01_08);

            // Row 1
            __m256 a1 = _mm256_broadcast_ss(A + kc + k);
            c_10_17 = _mm256_fmadd_ps(a1, b_0_7, c_10_17);
            c_11_18 = _mm256_fmadd_ps(a1, b_8_15, c_11_18);

            // Row 2
            __m256 a2 = _mm256_broadcast_ss(A + 2 * kc + k);
            c_20_27 = _mm256_fmadd_ps(a2, b_0_7, c_20_27);
            c_21_28 = _mm256_fmadd_ps(a2, b_8_15, c_21_28);

            // Row 3
            __m256 a3 = _mm256_broadcast_ss(A + 3 * kc + k);
            c_30_37 = _mm256_fmadd_ps(a3, b_0_7, c_30_37);
            c_31_38 = _mm256_fmadd_ps(a3, b_8_15, c_31_38);

            // Row 4
            __m256 a4 = _mm256_broadcast_ss(A + 4 * kc + k);
            c_40_47 = _mm256_fmadd_ps(a4, b_0_7, c_40_47);
            c_41_48 = _mm256_fmadd_ps(a4, b_8_15, c_41_48);

            // Row 5
            __m256 a5 = _mm256_broadcast_ss(A + 5 * kc + k);
            c_50_57 = _mm256_fmadd_ps(a5, b_0_7, c_50_57);
            c_51_58 = _mm256_fmadd_ps(a5, b_8_15, c_51_58);
        }

        // Store results
        float* c_ptr = &C[i_start * N + j_start];
        _mm256_storeu_ps(c_ptr, c_00_07); _mm256_storeu_ps(c_ptr + 8, c_01_08);
        c_ptr += N;
        _mm256_storeu_ps(c_ptr, c_10_17); _mm256_storeu_ps(c_ptr + 8, c_11_18);
        c_ptr += N;
        _mm256_storeu_ps(c_ptr, c_20_27); _mm256_storeu_ps(c_ptr + 8, c_21_28);
        c_ptr += N;
        _mm256_storeu_ps(c_ptr, c_30_37); _mm256_storeu_ps(c_ptr + 8, c_31_38);
        c_ptr += N;
        _mm256_storeu_ps(c_ptr, c_40_47); _mm256_storeu_ps(c_ptr + 8, c_41_48);
        c_ptr += N;
        _mm256_storeu_ps(c_ptr, c_50_57); _mm256_storeu_ps(c_ptr + 8, c_51_58);
    }

    /**
     * @brief Generic micro-kernel for edge cases
     */
    static void micro_kernel_generic(const float* A, const float* B, Matrix& C,
                                    size_t mr, size_t nr, size_t kc, size_t jb_packed,
                                    size_t i_start, size_t j_start, size_t N,
                                    bool first_k_iter) {
        for (size_t i = 0; i < mr; ++i) {
            for (size_t j = 0; j < nr; ++j) {
                float sum = first_k_iter ? 0.0f : C[(i_start + i) * N + (j_start + j)];
                for (size_t k = 0; k < kc; ++k) {
                    // A[i,k] * B[k,j] where B is packed as B[k * jb_packed + j]
                    sum += A[i * kc + k] * B[k * jb_packed + j];
                }
                C[(i_start + i) * N + (j_start + j)] = sum;
            }
        }
    }

public:
    static Matrix create_matrix(size_t rows, size_t cols, float value = 0.0f) {
        return Matrix(rows * cols, value);
    }

    static Matrix create_random_matrix(size_t rows, size_t cols) {
        Matrix mat(rows * cols);
        std::generate(std::execution::par_unseq, mat.begin(), mat.end(),
            []() { return static_cast<float>(rand()) / RAND_MAX; });
        return mat;
    }

    static float compute_error(const Matrix& A, const Matrix& B, const Matrix& C,
                               size_t M, size_t N, size_t K) {
        Matrix C_ref = create_matrix(M, N);

        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < N; ++j) {
                float sum = 0.0f;
                for (size_t k = 0; k < K; ++k) {
                    sum += A[i * K + k] * B[k * N + j];
                }
                C_ref[i * N + j] = sum;
            }
        }

        float max_error = 0.0f;
        for (size_t i = 0; i < M * N; ++i) {
            float error = std::abs(C[i] - C_ref[i]);
            float relative_error = error / (std::abs(C_ref[i]) + 1e-7f);
            max_error = std::max(max_error, relative_error);
        }

        return max_error;
    }
};

} // namespace matmul
