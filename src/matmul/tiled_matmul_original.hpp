#pragma once

#include <vector>
#include <algorithm>
#include <execution>
#include <immintrin.h>  // AVX2/AVX-512
#include <cassert>
#include <numeric>

namespace matmul {

// Tile sizes optimized for i7-12700K cache hierarchy
// L1: 48KB -> 6K floats per core
// L2: 1.25MB -> 320K floats per core
// L3: 25MB -> 6.5M floats shared

constexpr size_t L1_TILE = 64;   // 64x64 = 4K floats = 16KB (fits in L1)
constexpr size_t L2_TILE = 256;  // 256x256 = 64K floats = 256KB (fits in L2)
constexpr size_t L3_TILE = 1024; // 1024x1024 = 1M floats = 4MB (fits in L3)
constexpr size_t SIMD_WIDTH = 8; // AVX2 = 8 floats, AVX-512 = 16 floats

/**
 * @brief High-performance matrix multiplication: C = A * B
 *
 * Optimizations:
 * - 3-level cache blocking (L3 -> L2 -> L1)
 * - AVX2 vectorization (8-wide SIMD)
 * - C++17 parallel execution policies
 * - Cache-friendly memory layout
 *
 * @param A Input matrix (M x K)
 * @param B Input matrix (K x N)
 * @param C Output matrix (M x N)
 * @param M Number of rows in A and C
 * @param N Number of columns in B and C
 * @param K Number of columns in A and rows in B
 */
class TiledMatMul {
public:
    // Matrix stored in row-major format
    using Matrix = std::vector<float>;

    /**
     * @brief Compute C = A * B with optimal cache blocking and parallelism
     */
    static void multiply(const Matrix& A, const Matrix& B, Matrix& C,
                        size_t M, size_t N, size_t K) {
        assert(A.size() == M * K);
        assert(B.size() == K * N);
        assert(C.size() == M * N);

        // Zero output
        std::fill(std::execution::par_unseq, C.begin(), C.end(), 0.0f);

        // Generate L3 tile indices
        std::vector<std::tuple<size_t, size_t, size_t>> l3_tiles;
        for (size_t i = 0; i < M; i += L3_TILE) {
            for (size_t j = 0; j < N; j += L3_TILE) {
                for (size_t k = 0; k < K; k += L3_TILE) {
                    l3_tiles.emplace_back(i, j, k);
                }
            }
        }

        // Parallel L3 blocking - distribute across cores
        std::for_each(std::execution::par, l3_tiles.begin(), l3_tiles.end(),
            [&](const auto& tile) {
                auto [i0, j0, k0] = tile;
                size_t i_end = std::min(i0 + L3_TILE, M);
                size_t j_end = std::min(j0 + L3_TILE, N);
                size_t k_end = std::min(k0 + L3_TILE, K);

                multiply_l3_tile(A, B, C, M, N, K,
                                i0, i_end, j0, j_end, k0, k_end);
            });
    }

private:
    /**
     * @brief L3 cache blocking
     */
    static void multiply_l3_tile(const Matrix& A, const Matrix& B, Matrix& C,
                                 size_t M, size_t N, size_t K,
                                 size_t i0, size_t i_end,
                                 size_t j0, size_t j_end,
                                 size_t k0, size_t k_end) {
        // L2 blocking
        for (size_t i = i0; i < i_end; i += L2_TILE) {
            size_t i2_end = std::min(i + L2_TILE, i_end);

            for (size_t j = j0; j < j_end; j += L2_TILE) {
                size_t j2_end = std::min(j + L2_TILE, j_end);

                for (size_t k = k0; k < k_end; k += L2_TILE) {
                    size_t k2_end = std::min(k + L2_TILE, k_end);

                    multiply_l2_tile(A, B, C, M, N, K,
                                    i, i2_end, j, j2_end, k, k2_end);
                }
            }
        }
    }

    /**
     * @brief L2 cache blocking
     */
    static void multiply_l2_tile(const Matrix& A, const Matrix& B, Matrix& C,
                                 size_t M, size_t N, size_t K,
                                 size_t i0, size_t i_end,
                                 size_t j0, size_t j_end,
                                 size_t k0, size_t k_end) {
        // L1 blocking
        for (size_t i = i0; i < i_end; i += L1_TILE) {
            size_t i1_end = std::min(i + L1_TILE, i_end);

            for (size_t j = j0; j < j_end; j += L1_TILE) {
                size_t j1_end = std::min(j + L1_TILE, j_end);

                for (size_t k = k0; k < k_end; k += L1_TILE) {
                    size_t k1_end = std::min(k + L1_TILE, k_end);

                    multiply_l1_tile(A, B, C, M, N, K,
                                    i, i1_end, j, j1_end, k, k1_end);
                }
            }
        }
    }

    /**
     * @brief L1 cache blocking with AVX2 vectorization
     */
    static void multiply_l1_tile(const Matrix& A, const Matrix& B, Matrix& C,
                                 size_t M, size_t N, size_t K,
                                 size_t i0, size_t i_end,
                                 size_t j0, size_t j_end,
                                 size_t k0, size_t k_end) {
        // Compute micro-kernel
        for (size_t i = i0; i < i_end; ++i) {
            for (size_t k = k0; k < k_end; ++k) {
                // Broadcast A[i,k]
                __m256 a_broadcast = _mm256_set1_ps(A[i * K + k]);

                // Vectorized inner loop over j
                size_t j = j0;

                // Process 8 elements at a time with AVX2
                for (; j + SIMD_WIDTH <= j_end; j += SIMD_WIDTH) {
                    // Load B[k, j:j+8]
                    __m256 b_vec = _mm256_loadu_ps(&B[k * N + j]);

                    // Load C[i, j:j+8]
                    __m256 c_vec = _mm256_loadu_ps(&C[i * N + j]);

                    // C[i,j:j+8] += A[i,k] * B[k,j:j+8]
                    c_vec = _mm256_fmadd_ps(a_broadcast, b_vec, c_vec);

                    // Store back
                    _mm256_storeu_ps(&C[i * N + j], c_vec);
                }

                // Handle remaining elements
                for (; j < j_end; ++j) {
                    C[i * N + j] += A[i * K + k] * B[k * N + j];
                }
            }
        }
    }

public:
    /**
     * @brief Create a matrix filled with a value
     */
    static Matrix create_matrix(size_t rows, size_t cols, float value = 0.0f) {
        return Matrix(rows * cols, value);
    }

    /**
     * @brief Create a matrix with random values
     */
    static Matrix create_random_matrix(size_t rows, size_t cols) {
        Matrix mat(rows * cols);
        std::generate(std::execution::par_unseq, mat.begin(), mat.end(),
            []() { return static_cast<float>(rand()) / RAND_MAX; });
        return mat;
    }

    /**
     * @brief Validate correctness against naive implementation
     */
    static float compute_error(const Matrix& A, const Matrix& B, const Matrix& C,
                               size_t M, size_t N, size_t K) {
        Matrix C_ref = create_matrix(M, N);

        // Naive matmul
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < N; ++j) {
                float sum = 0.0f;
                for (size_t k = 0; k < K; ++k) {
                    sum += A[i * K + k] * B[k * N + j];
                }
                C_ref[i * N + j] = sum;
            }
        }

        // Compute max relative error
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
