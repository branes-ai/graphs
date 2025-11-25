/**
 * Matrix-Matrix Multiplication CUDA Kernel
 * For validating Jetson Orin Nano SM occupancy modeling
 *
 * Jetson Orin Nano Specs (SM 8.7 Ampere):
 * - SMs: 8
 * - CUDA cores per SM: 128 (FP32)
 * - Tensor cores per SM: 4 (3rd gen)
 * - Shared memory per SM: 48 KB default (up to 100 KB)
 * - L1 cache per SM: 128 KB unified (shared + L1)
 * - L2 cache: 512 KB
 * - Warp size: 32 threads
 * - Max threads per SM: 1536
 * - Max threads per block: 1024
 * - Max warps per SM: 48
 * - Registers per SM: 65536
 * - Memory bandwidth: 68 GB/s (LPDDR5)
 * - Peak FP32: ~1.6 TFLOPS
 */

#include <cuda_runtime.h>
#include <stdio.h>

// Tile dimensions for shared memory blocking
#define TILE_M 32       // Tile height (rows of C computed per block)
#define TILE_N 32       // Tile width (cols of C computed per block)
#define TILE_K 32       // Tile depth (reduction dimension)

// Block configuration
#define BLOCK_DIM_X 32  // Threads in X (maps to N dimension)
#define BLOCK_DIM_Y 8   // Threads in Y (maps to M dimension)
// Total threads per block: 32 * 8 = 256 (8 warps)

/**
 * Naive Matrix Multiplication: C = A * B
 * - No shared memory optimization
 * - For baseline comparison
 *
 * Memory access pattern (per thread):
 * - Loads K elements from row of A (strided)
 * - Loads K elements from column of B (strided, poor coalescing)
 * - Stores 1 element to C
 */
__global__ void matmul_naive(
    const float* __restrict__ A,    // M x K
    const float* __restrict__ B,    // K x N
    float* __restrict__ C,          // M x N
    int M, int K, int N
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

/**
 * Tiled Matrix Multiplication with Shared Memory
 * C = A * B where A is MxK, B is KxN, C is MxN
 *
 * ============================================
 * SHARED MEMORY LAYOUT
 * ============================================
 *
 * Each block computes a TILE_M x TILE_N tile of C
 *
 *     Shared Memory (per block):
 *     +------------------+------------------+
 *     |    A_shared      |    B_shared      |
 *     |   TILE_M x TILE_K|   TILE_K x TILE_N|
 *     |   32 x 32 floats |   32 x 32 floats |
 *     |   = 4 KB         |   = 4 KB         |
 *     +------------------+------------------+
 *     Total: 8 KB per block
 *
 * ============================================
 * WARP ASSIGNMENT
 * ============================================
 *
 * Block dimensions: 32 x 8 = 256 threads (8 warps)
 *
 * Thread layout in block (threadIdx.x, threadIdx.y):
 *
 *     threadIdx.x: 0  1  2  3  4  ... 31
 *     threadIdx.y:
 *         0       [W0--------------------------]  <- Warp 0
 *         1       [W1--------------------------]  <- Warp 1
 *         2       [W2--------------------------]  <- Warp 2
 *         3       [W3--------------------------]  <- Warp 3
 *         4       [W4--------------------------]  <- Warp 4
 *         5       [W5--------------------------]  <- Warp 5
 *         6       [W6--------------------------]  <- Warp 6
 *         7       [W7--------------------------]  <- Warp 7
 *
 * Each warp processes one row of 32 elements in the output tile
 * But we need 32 rows, so each thread computes 4 rows (TILE_M / BLOCK_DIM_Y)
 *
 * ============================================
 * MEMORY ACCESS PATTERN
 * ============================================
 *
 * Loading A_shared (coalesced):
 *   - Each warp loads one row of 32 floats
 *   - Thread 0-31 in warp load consecutive elements
 *   - Coalesced: 128-byte transaction
 *
 * Loading B_shared (coalesced):
 *   - Each warp loads partial rows
 *   - Thread 0-31 load consecutive elements
 *   - Coalesced: 128-byte transaction
 *
 * ============================================
 * L1 CACHE BEHAVIOR
 * ============================================
 *
 * Ampere L1 cache:
 * - 128 KB unified (L1 + shared memory)
 * - Default: 48 KB shared, 80 KB L1
 * - Can configure: up to 100 KB shared, 28 KB L1
 *
 * With shared memory tiling:
 * - Global memory reads cached in L1
 * - Reuse within tile iterations
 * - L1 hit rate depends on tile size vs L1 size
 */
__global__ void matmul_tiled(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int K, int N
) {
    // Shared memory for tiles of A and B
    __shared__ float A_shared[TILE_M][TILE_K];
    __shared__ float B_shared[TILE_K][TILE_N];

    // Thread indices
    int tx = threadIdx.x;  // 0-31, column within tile
    int ty = threadIdx.y;  // 0-7, row group within tile

    // Block indices (which tile of C we're computing)
    int bx = blockIdx.x;   // tile column in C
    int by = blockIdx.y;   // tile row in C

    // Global row/col this thread contributes to
    // Each thread computes multiple rows (TILE_M / BLOCK_DIM_Y = 4)
    int row_start = by * TILE_M + ty;
    int col = bx * TILE_N + tx;

    // Accumulators for 4 output elements (one per row this thread handles)
    float sum[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    // Number of tiles along K dimension
    int num_tiles = (K + TILE_K - 1) / TILE_K;

    // Iterate over tiles of A and B
    for (int t = 0; t < num_tiles; t++) {
        // ============================================
        // PHASE 1: Load tiles into shared memory
        // ============================================

        // Load A tile: each thread loads 4 elements (one per row it handles)
        // Access pattern: A[row_start + i*8][t*TILE_K + tx]
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int row = row_start + i * BLOCK_DIM_Y;
            int a_col = t * TILE_K + tx;
            if (row < M && a_col < K) {
                A_shared[ty + i * BLOCK_DIM_Y][tx] = A[row * K + a_col];
            } else {
                A_shared[ty + i * BLOCK_DIM_Y][tx] = 0.0f;
            }
        }

        // Load B tile: each thread loads 4 elements
        // Access pattern: B[t*TILE_K + ty + i*8][col]
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int b_row = t * TILE_K + ty + i * BLOCK_DIM_Y;
            if (b_row < K && col < N) {
                B_shared[ty + i * BLOCK_DIM_Y][tx] = B[b_row * N + col];
            } else {
                B_shared[ty + i * BLOCK_DIM_Y][tx] = 0.0f;
            }
        }

        // Synchronize to ensure tiles are loaded
        __syncthreads();

        // ============================================
        // PHASE 2: Compute partial products
        // ============================================

        // Each thread computes 4 elements of C (one per row)
        // Uses the cached tiles in shared memory
        #pragma unroll
        for (int k = 0; k < TILE_K; k++) {
            float b_val = B_shared[k][tx];  // Same B value for all rows

            #pragma unroll
            for (int i = 0; i < 4; i++) {
                sum[i] += A_shared[ty + i * BLOCK_DIM_Y][k] * b_val;
            }
        }

        // Synchronize before loading next tiles
        __syncthreads();
    }

    // ============================================
    // PHASE 3: Write results to global memory
    // ============================================
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int row = row_start + i * BLOCK_DIM_Y;
        if (row < M && col < N) {
            C[row * N + col] = sum[i];
        }
    }
}

/**
 * Double-buffered tiled GEMM
 * Overlaps computation with memory loads using two shared memory buffers
 */
__global__ void matmul_double_buffered(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int K, int N
) {
    // Double buffers for A and B tiles
    __shared__ float A_shared[2][TILE_M][TILE_K];
    __shared__ float B_shared[2][TILE_K][TILE_N];
    // Total shared memory: 2 * 8 KB = 16 KB per block

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int row_start = by * TILE_M + ty;
    int col = bx * TILE_N + tx;

    float sum[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    int num_tiles = (K + TILE_K - 1) / TILE_K;
    int curr_buf = 0;

    // Prefetch first tile
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int row = row_start + i * BLOCK_DIM_Y;
        int a_col = tx;
        if (row < M && a_col < K) {
            A_shared[0][ty + i * BLOCK_DIM_Y][tx] = A[row * K + a_col];
        } else {
            A_shared[0][ty + i * BLOCK_DIM_Y][tx] = 0.0f;
        }

        int b_row = ty + i * BLOCK_DIM_Y;
        if (b_row < K && col < N) {
            B_shared[0][ty + i * BLOCK_DIM_Y][tx] = B[b_row * N + col];
        } else {
            B_shared[0][ty + i * BLOCK_DIM_Y][tx] = 0.0f;
        }
    }
    __syncthreads();

    for (int t = 0; t < num_tiles; t++) {
        int next_buf = 1 - curr_buf;

        // Prefetch next tile while computing current
        if (t + 1 < num_tiles) {
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                int row = row_start + i * BLOCK_DIM_Y;
                int a_col = (t + 1) * TILE_K + tx;
                if (row < M && a_col < K) {
                    A_shared[next_buf][ty + i * BLOCK_DIM_Y][tx] = A[row * K + a_col];
                } else {
                    A_shared[next_buf][ty + i * BLOCK_DIM_Y][tx] = 0.0f;
                }

                int b_row = (t + 1) * TILE_K + ty + i * BLOCK_DIM_Y;
                if (b_row < K && col < N) {
                    B_shared[next_buf][ty + i * BLOCK_DIM_Y][tx] = B[b_row * N + col];
                } else {
                    B_shared[next_buf][ty + i * BLOCK_DIM_Y][tx] = 0.0f;
                }
            }
        }

        // Compute using current buffer
        #pragma unroll
        for (int k = 0; k < TILE_K; k++) {
            float b_val = B_shared[curr_buf][k][tx];
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                sum[i] += A_shared[curr_buf][ty + i * BLOCK_DIM_Y][k] * b_val;
            }
        }

        __syncthreads();
        curr_buf = next_buf;
    }

    // Write results
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int row = row_start + i * BLOCK_DIM_Y;
        if (row < M && col < N) {
            C[row * N + col] = sum[i];
        }
    }
}

/**
 * Print detailed occupancy analysis
 */
void print_occupancy_analysis() {
    printf("=======================================================\n");
    printf("  Jetson Orin Nano SM Occupancy Analysis - MatMul\n");
    printf("=======================================================\n\n");

    printf("Hardware Configuration:\n");
    printf("  SMs: 8\n");
    printf("  CUDA cores per SM: 128\n");
    printf("  Tensor cores per SM: 4\n");
    printf("  Warp size: 32\n");
    printf("  Max threads per SM: 1536\n");
    printf("  Max warps per SM: 48\n");
    printf("  Max blocks per SM: 16\n");
    printf("  Registers per SM: 65536\n");
    printf("  L1/Shared per SM: 128 KB (unified)\n");
    printf("  Default shared: 48 KB\n");
    printf("  L2 cache: 512 KB\n\n");

    printf("-------------------------------------------------------\n");
    printf("Kernel: matmul_tiled (single buffer)\n");
    printf("-------------------------------------------------------\n");
    printf("  Block size: %d x %d = %d threads (%d warps)\n",
           BLOCK_DIM_X, BLOCK_DIM_Y, BLOCK_DIM_X * BLOCK_DIM_Y, BLOCK_DIM_X * BLOCK_DIM_Y / 32);
    printf("  Tile size: %d x %d x %d\n", TILE_M, TILE_N, TILE_K);
    printf("  Shared memory per block: %lu bytes (%.1f KB)\n",
           sizeof(float) * (TILE_M * TILE_K + TILE_K * TILE_N),
           sizeof(float) * (TILE_M * TILE_K + TILE_K * TILE_N) / 1024.0f);
    printf("  Registers per thread: ~32 (estimated)\n\n");

    int threads_per_block = BLOCK_DIM_X * BLOCK_DIM_Y;
    int warps_per_block = threads_per_block / 32;
    size_t shared_per_block = sizeof(float) * (TILE_M * TILE_K + TILE_K * TILE_N);
    int regs_per_thread = 32;

    // Occupancy limiters
    int blocks_by_threads = 1536 / threads_per_block;
    int blocks_by_warps = 48 / warps_per_block;
    int blocks_by_smem = (48 * 1024) / shared_per_block;
    int blocks_by_regs = 65536 / (threads_per_block * regs_per_thread);
    int blocks_by_limit = 16;

    int blocks_per_sm = blocks_by_threads;
    if (blocks_by_warps < blocks_per_sm) blocks_per_sm = blocks_by_warps;
    if ((int)blocks_by_smem < blocks_per_sm) blocks_per_sm = blocks_by_smem;
    if (blocks_by_regs < blocks_per_sm) blocks_per_sm = blocks_by_regs;
    if (blocks_by_limit < blocks_per_sm) blocks_per_sm = blocks_by_limit;

    int active_warps = blocks_per_sm * warps_per_block;
    float occupancy = (float)active_warps / 48.0f * 100.0f;

    printf("Occupancy Limiting Factors:\n");
    printf("  By thread limit (1536): %d blocks\n", blocks_by_threads);
    printf("  By warp limit (48):     %d blocks\n", blocks_by_warps);
    printf("  By shared mem (48KB):   %d blocks\n", (int)blocks_by_smem);
    printf("  By registers (65536):   %d blocks\n", blocks_by_regs);
    printf("  By HW block limit (16): %d blocks\n", blocks_by_limit);
    printf("  ---------------------------------\n");
    printf("  Active blocks per SM:   %d\n", blocks_per_sm);
    printf("  Active warps per SM:    %d / 48\n", active_warps);
    printf("  Theoretical occupancy:  %.1f%%\n\n", occupancy);

    printf("-------------------------------------------------------\n");
    printf("Kernel: matmul_double_buffered\n");
    printf("-------------------------------------------------------\n");
    size_t shared_double = 2 * sizeof(float) * (TILE_M * TILE_K + TILE_K * TILE_N);
    printf("  Shared memory per block: %lu bytes (%.1f KB)\n", shared_double, shared_double / 1024.0f);

    int blocks_by_smem_double = (48 * 1024) / shared_double;
    int blocks_per_sm_double = blocks_by_threads;
    if (blocks_by_warps < blocks_per_sm_double) blocks_per_sm_double = blocks_by_warps;
    if ((int)blocks_by_smem_double < blocks_per_sm_double) blocks_per_sm_double = blocks_by_smem_double;
    if (blocks_by_regs < blocks_per_sm_double) blocks_per_sm_double = blocks_by_regs;
    if (blocks_by_limit < blocks_per_sm_double) blocks_per_sm_double = blocks_by_limit;

    int active_warps_double = blocks_per_sm_double * warps_per_block;
    float occupancy_double = (float)active_warps_double / 48.0f * 100.0f;

    printf("  By shared mem (48KB):   %d blocks (limiting factor)\n", (int)blocks_by_smem_double);
    printf("  Active blocks per SM:   %d\n", blocks_per_sm_double);
    printf("  Active warps per SM:    %d / 48\n", active_warps_double);
    printf("  Theoretical occupancy:  %.1f%%\n\n", occupancy_double);

    printf("-------------------------------------------------------\n");
    printf("Warp Execution Timeline (matmul_tiled)\n");
    printf("-------------------------------------------------------\n");
    printf("  Phase 1: Load A and B tiles from global memory\n");
    printf("    - 8 warps load 32x32 + 32x32 = 2048 floats\n");
    printf("    - Each warp: 256 floats (8 transactions of 32 floats)\n");
    printf("    - Memory latency hidden by warp scheduling\n\n");
    printf("  Phase 2: __syncthreads() barrier\n");
    printf("    - All 8 warps wait for tile load completion\n\n");
    printf("  Phase 3: Compute using shared memory\n");
    printf("    - 32 iterations over K dimension\n");
    printf("    - Each iteration: 8 FMAs per thread\n");
    printf("    - Total: 256 FMAs per thread per tile\n");
    printf("    - No bank conflicts with 32-wide access\n\n");
    printf("  Phase 4: __syncthreads() before next tile\n\n");

    printf("-------------------------------------------------------\n");
    printf("Shared Memory Bank Analysis\n");
    printf("-------------------------------------------------------\n");
    printf("  32 banks, 4 bytes per bank\n");
    printf("  A_shared[row][k]: row-major, threads read same k -> broadcast\n");
    printf("  B_shared[k][col]: col-major access, threads in warp read\n");
    printf("                    consecutive columns -> no conflict\n\n");
}

// Host code
int main(int argc, char** argv) {
    print_occupancy_analysis();

    // Matrix dimensions
    int M = 1024;
    int K = 1024;
    int N = 1024;

    printf("=======================================================\n");
    printf("  Running Matrix-Matrix Multiplication Test\n");
    printf("=======================================================\n");
    printf("Matrix sizes: A(%dx%d) * B(%dx%d) = C(%dx%d)\n", M, K, K, N, M, N);
    printf("Total FLOPs: %.2f GFLOP\n", 2.0 * M * K * N / 1e9);

    // Allocate host memory
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    float *h_A = (float*)malloc(size_A);
    float *h_B = (float*)malloc(size_B);
    float *h_C = (float*)malloc(size_C);

    // Initialize
    for (int i = 0; i < M * K; i++) h_A[i] = 1.0f;
    for (int i = 0; i < K * N; i++) h_B[i] = 1.0f;

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    // Grid configuration
    dim3 block(BLOCK_DIM_X, BLOCK_DIM_Y);  // 32 x 8 = 256 threads
    dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);

    printf("\nLaunch Configuration:\n");
    printf("  Grid: %d x %d blocks = %d total blocks\n", grid.x, grid.y, grid.x * grid.y);
    printf("  Block: %d x %d threads = %d threads\n", block.x, block.y, block.x * block.y);
    printf("  Blocks per SM (if even): %.1f\n", (float)(grid.x * grid.y) / 8.0f);

    // CUDA events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warm up
    matmul_tiled<<<grid, block>>>(d_A, d_B, d_C, M, K, N);
    cudaDeviceSynchronize();

    // Benchmark
    int iterations = 100;

    printf("\n--- matmul_naive ---\n");
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        matmul_naive<<<grid, block>>>(d_A, d_B, d_C, M, K, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_naive;
    cudaEventElapsedTime(&ms_naive, start, stop);
    ms_naive /= iterations;
    float gflops_naive = (2.0 * M * K * N) / (ms_naive * 1e6);
    printf("  Time: %.3f ms\n", ms_naive);
    printf("  Performance: %.1f GFLOPS\n", gflops_naive);

    printf("\n--- matmul_tiled ---\n");
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        matmul_tiled<<<grid, block>>>(d_A, d_B, d_C, M, K, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_tiled;
    cudaEventElapsedTime(&ms_tiled, start, stop);
    ms_tiled /= iterations;
    float gflops_tiled = (2.0 * M * K * N) / (ms_tiled * 1e6);
    printf("  Time: %.3f ms\n", ms_tiled);
    printf("  Performance: %.1f GFLOPS\n", gflops_tiled);
    printf("  Speedup vs naive: %.2fx\n", ms_naive / ms_tiled);

    printf("\n--- matmul_double_buffered ---\n");
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        matmul_double_buffered<<<grid, block>>>(d_A, d_B, d_C, M, K, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_double;
    cudaEventElapsedTime(&ms_double, start, stop);
    ms_double /= iterations;
    float gflops_double = (2.0 * M * K * N) / (ms_double * 1e6);
    printf("  Time: %.3f ms\n", ms_double);
    printf("  Performance: %.1f GFLOPS\n", gflops_double);
    printf("  Speedup vs naive: %.2fx\n", ms_naive / ms_double);

    // Jetson Orin Nano peak performance
    float peak_gflops = 1600.0f;  // ~1.6 TFLOPS FP32
    printf("\n--- Efficiency ---\n");
    printf("  Jetson Orin Nano peak FP32: %.0f GFLOPS\n", peak_gflops);
    printf("  Naive efficiency: %.1f%%\n", gflops_naive / peak_gflops * 100);
    printf("  Tiled efficiency: %.1f%%\n", gflops_tiled / peak_gflops * 100);
    printf("  Double-buffered efficiency: %.1f%%\n", gflops_double / peak_gflops * 100);

    // Verify
    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);
    printf("\nVerification: C[0][0] = %.1f (expected: %.1f)\n", h_C[0], (float)K);

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
