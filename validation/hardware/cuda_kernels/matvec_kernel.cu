/**
 * Matrix-Vector Multiplication CUDA Kernel
 * For validating Jetson Orin Nano SM occupancy modeling
 *
 * Jetson Orin Nano Specs:
 * - GPU: Ampere architecture (SM 8.7)
 * - SMs: 8
 * - CUDA cores per SM: 128
 * - Shared memory per SM: 48 KB (configurable up to 100 KB with reduced L1)
 * - L1 cache per SM: 128 KB (combined with shared memory)
 * - Warp size: 32 threads
 * - Max threads per SM: 1536
 * - Max threads per block: 1024
 * - Max warps per SM: 48
 * - Registers per SM: 65536
 */

#include <cuda_runtime.h>
#include <stdio.h>

// Kernel configuration for occupancy analysis
#define BLOCK_SIZE 256      // Threads per block (8 warps)
#define TILE_SIZE 256       // Elements per tile in shared memory

/**
 * Matrix-Vector Multiplication: y = A * x
 *
 * Memory Layout:
 * - A: M x N matrix (row-major)
 * - x: N x 1 vector
 * - y: M x 1 vector
 *
 * Parallelization Strategy:
 * - Each thread computes one element of y
 * - Each block handles BLOCK_SIZE rows of A
 * - Shared memory caches tiles of x for reuse
 *
 * Warp Assignment:
 * - Block of 256 threads = 8 warps
 * - Warp 0: threads 0-31   -> rows 0-31
 * - Warp 1: threads 32-63  -> rows 32-63
 * - ...
 * - Warp 7: threads 224-255 -> rows 224-255
 */
__global__ void matvec_shared_memory(
    const float* __restrict__ A,    // M x N matrix
    const float* __restrict__ x,    // N x 1 vector
    float* __restrict__ y,          // M x 1 output vector
    int M,                          // Number of rows
    int N                           // Number of columns
) {
    // Shared memory for caching tiles of vector x
    // This reduces global memory bandwidth by factor of BLOCK_SIZE
    __shared__ float x_shared[TILE_SIZE];

    // Thread identification
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // Warp and lane identification (for occupancy analysis)
    int warp_id = tid / 32;         // Which warp within block (0-7)
    int lane_id = tid % 32;         // Position within warp (0-31)

    // Accumulator for dot product
    float sum = 0.0f;

    // Process vector x in tiles to maximize shared memory reuse
    // Each tile: TILE_SIZE elements of x cached in shared memory
    for (int tile = 0; tile < (N + TILE_SIZE - 1) / TILE_SIZE; tile++) {
        // Collaborative loading: all threads in block load x into shared memory
        // This is a coalesced memory access pattern
        int x_idx = tile * TILE_SIZE + tid;
        if (x_idx < N) {
            x_shared[tid] = x[x_idx];
        } else {
            x_shared[tid] = 0.0f;
        }

        // Synchronize to ensure all threads have loaded their portion
        __syncthreads();

        // Compute partial dot product using cached x values
        if (row < M) {
            int tile_start = tile * TILE_SIZE;
            #pragma unroll 8
            for (int j = 0; j < TILE_SIZE && (tile_start + j) < N; j++) {
                // Access pattern: A[row][tile_start + j]
                // Each thread in a warp accesses different rows (strided)
                sum += A[row * N + tile_start + j] * x_shared[j];
            }
        }

        // Synchronize before loading next tile
        __syncthreads();
    }

    // Write result
    if (row < M) {
        y[row] = sum;
    }
}

/**
 * Alternative: Warp-level reduction for better occupancy
 * Each warp computes one output element collaboratively
 *
 * Occupancy characteristics:
 * - Fewer threads per output element
 * - Better for small M, large N
 * - Uses warp shuffle for fast reduction
 */
__global__ void matvec_warp_reduce(
    const float* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M,
    int N
) {
    // Each warp handles one row
    int warp_id_global = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane_id = threadIdx.x % 32;

    if (warp_id_global >= M) return;

    int row = warp_id_global;
    float sum = 0.0f;

    // Each lane processes every 32nd element
    // Lane 0: elements 0, 32, 64, ...
    // Lane 1: elements 1, 33, 65, ...
    for (int j = lane_id; j < N; j += 32) {
        sum += A[row * N + j] * x[j];
    }

    // Warp-level reduction using shuffle
    // No shared memory needed - uses registers
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // Lane 0 writes the final result
    if (lane_id == 0) {
        y[row] = sum;
    }
}

/**
 * Occupancy Calculator
 * Reports theoretical occupancy for Jetson Orin Nano
 */
void print_occupancy_info() {
    printf("=== Jetson Orin Nano SM Occupancy Analysis ===\n\n");

    printf("Hardware Specs:\n");
    printf("  SMs: 8\n");
    printf("  CUDA cores per SM: 128\n");
    printf("  Warp size: 32\n");
    printf("  Max threads per SM: 1536\n");
    printf("  Max warps per SM: 48\n");
    printf("  Max blocks per SM: 16\n");
    printf("  Shared memory per SM: 48-100 KB (configurable)\n");
    printf("  Registers per SM: 65536\n\n");

    printf("Kernel: matvec_shared_memory\n");
    printf("  Block size: %d threads (%d warps)\n", BLOCK_SIZE, BLOCK_SIZE/32);
    printf("  Shared memory per block: %lu bytes\n", TILE_SIZE * sizeof(float));
    printf("  Registers per thread: ~16 (estimated)\n\n");

    // Calculate theoretical occupancy
    int threads_per_block = BLOCK_SIZE;
    int warps_per_block = threads_per_block / 32;
    int shared_mem_per_block = TILE_SIZE * sizeof(float);
    int registers_per_thread = 16;  // Estimate

    // Limiting factors for blocks per SM:
    // 1. Thread limit: 1536 / 256 = 6 blocks
    // 2. Warp limit: 48 / 8 = 6 blocks
    // 3. Shared memory: 48KB / 1KB = 48 blocks (not limiting)
    // 4. Registers: 65536 / (256 * 16) = 16 blocks
    // 5. Block limit: 16 blocks

    int blocks_by_threads = 1536 / threads_per_block;
    int blocks_by_warps = 48 / warps_per_block;
    int blocks_by_smem = (48 * 1024) / shared_mem_per_block;
    int blocks_by_regs = 65536 / (threads_per_block * registers_per_thread);
    int blocks_by_limit = 16;

    int blocks_per_sm = blocks_by_threads;
    if (blocks_by_warps < blocks_per_sm) blocks_per_sm = blocks_by_warps;
    if (blocks_by_smem < blocks_per_sm) blocks_per_sm = blocks_by_smem;
    if (blocks_by_regs < blocks_per_sm) blocks_per_sm = blocks_by_regs;
    if (blocks_by_limit < blocks_per_sm) blocks_per_sm = blocks_by_limit;

    int active_warps = blocks_per_sm * warps_per_block;
    float occupancy = (float)active_warps / 48.0f * 100.0f;

    printf("Occupancy Calculation:\n");
    printf("  Blocks limited by threads: %d\n", blocks_by_threads);
    printf("  Blocks limited by warps: %d\n", blocks_by_warps);
    printf("  Blocks limited by shared mem: %d\n", blocks_by_smem);
    printf("  Blocks limited by registers: %d\n", blocks_by_regs);
    printf("  Blocks limited by HW limit: %d\n", blocks_by_limit);
    printf("  => Active blocks per SM: %d\n", blocks_per_sm);
    printf("  => Active warps per SM: %d / 48\n", active_warps);
    printf("  => Theoretical occupancy: %.1f%%\n\n", occupancy);

    printf("Memory Bandwidth Analysis:\n");
    printf("  Without shared memory: each thread loads entire x vector\n");
    printf("  With shared memory: x loaded once per block, reused %d times\n", BLOCK_SIZE);
    printf("  Bandwidth reduction: %.1fx\n", (float)BLOCK_SIZE);
}

// Host code for testing
int main(int argc, char** argv) {
    print_occupancy_info();

    // Matrix dimensions
    int M = 4096;  // rows
    int N = 4096;  // columns

    printf("=== Running Matrix-Vector Test ===\n");
    printf("Matrix size: %d x %d\n", M, N);
    printf("Vector size: %d\n", N);

    // Allocate host memory
    float *h_A, *h_x, *h_y;
    h_A = (float*)malloc(M * N * sizeof(float));
    h_x = (float*)malloc(N * sizeof(float));
    h_y = (float*)malloc(M * sizeof(float));

    // Initialize with test data
    for (int i = 0; i < M * N; i++) h_A[i] = 1.0f;
    for (int i = 0; i < N; i++) h_x[i] = 1.0f;

    // Allocate device memory
    float *d_A, *d_x, *d_y;
    cudaMalloc(&d_A, M * N * sizeof(float));
    cudaMalloc(&d_x, N * sizeof(float));
    cudaMalloc(&d_y, M * sizeof(float));

    // Copy to device
    cudaMemcpy(d_A, h_A, M * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice);

    // Launch configuration
    int num_blocks = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    printf("\nLaunch configuration:\n");
    printf("  Grid: %d blocks\n", num_blocks);
    printf("  Block: %d threads\n", BLOCK_SIZE);
    printf("  Total threads: %d\n", num_blocks * BLOCK_SIZE);
    printf("  Blocks per SM (if evenly distributed): %.1f\n", (float)num_blocks / 8.0f);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warm-up run
    matvec_shared_memory<<<num_blocks, BLOCK_SIZE>>>(d_A, d_x, d_y, M, N);
    cudaDeviceSynchronize();

    // Timed runs
    int num_iterations = 100;
    cudaEventRecord(start);
    for (int i = 0; i < num_iterations; i++) {
        matvec_shared_memory<<<num_blocks, BLOCK_SIZE>>>(d_A, d_x, d_y, M, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    float avg_time_ms = milliseconds / num_iterations;

    // Calculate achieved bandwidth
    // Reads: M*N (matrix) + N (vector) floats
    // Writes: M floats
    size_t bytes_read = (size_t)M * N * sizeof(float) + N * sizeof(float);
    size_t bytes_written = M * sizeof(float);
    float bandwidth_gb_s = (bytes_read + bytes_written) / (avg_time_ms * 1e6);

    printf("\nPerformance:\n");
    printf("  Average time: %.3f ms\n", avg_time_ms);
    printf("  Achieved bandwidth: %.1f GB/s\n", bandwidth_gb_s);
    printf("  Jetson Orin Nano peak: ~68 GB/s (LPDDR5)\n");
    printf("  Bandwidth utilization: %.1f%%\n", bandwidth_gb_s / 68.0f * 100.0f);

    // Verify result
    cudaMemcpy(h_y, d_y, M * sizeof(float), cudaMemcpyDeviceToHost);
    printf("\nVerification: y[0] = %.1f (expected: %.1f)\n", h_y[0], (float)N);

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_x);
    cudaFree(d_y);
    free(h_A);
    free(h_x);
    free(h_y);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
