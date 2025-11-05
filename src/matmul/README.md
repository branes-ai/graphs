# High-Performance Tiled Matrix Multiplication

Optimized matrix multiplication implementation for Intel Core i7-12700K with multi-level cache blocking, AVX2 vectorization, and C++17 parallel algorithms.

## Features

- **3-Level Cache Blocking**: L1 (64×64), L2 (256×256), L3 (1024×1024) tiles optimized for i7-12700K
- **AVX2 SIMD**: 8-wide vectorization with FMA instructions
- **C++17 Parallelism**: `std::execution::par` for multi-core execution
- **Zero Overfetch**: Optimal tile sizes prevent redundant memory transfers
- **Cache-Friendly Layout**: Row-major ordering for sequential access patterns

## Hardware Target

**Intel Core i7-12700K Specifications:**
- 12 cores (8 P-cores + 4 E-cores), 20 threads
- Max Frequency: 5.0 GHz
- L1 Data: 48 KB per P-core (4-5 cycle latency)
- L2: 1.25 MB per P-core (12-15 cycle latency)
- L3: 25 MB shared (40-50 cycle latency)
- Cache Line: 64 bytes
- Theoretical Peak: ~640 GFLOPS (P-cores, AVX2)

## Build Instructions

### Prerequisites

```bash
# Ubuntu/Debian
sudo apt-get install build-essential cmake libtbb-dev

# macOS
brew install cmake tbb

# Fedora/RHEL
sudo dnf install cmake tbb-devel
```

### Compile

```bash
cd src/matmul
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build .
```

## Usage

### Run Benchmark Suite

```bash
./matmul_benchmark
```

This runs benchmarks for 128×128, 256×256, 512×512, 1024×1024, 2048×2048, and 4096×4096 matrices.

### Run Custom Size

```bash
./matmul_benchmark --size 4096
```

### Validate Correctness

```bash
./matmul_benchmark --validate
```

### Example Output

```
================================================================================
Multi-Level Tiled Matrix Multiplication Benchmark
Target: Intel Core i7-12700K (12 cores @ 5.0 GHz)
Optimizations: L1/L2/L3 blocking + AVX2 + C++17 parallel algorithms
================================================================================

Testing 4096x4096...
Matrix: 4096x4096 = 4096x4096 * 4096x4096
  Time:         523.45 ms
  Throughput:   262.89 GFLOPS

Cache Analysis:
  Total memory: 192.00 MB
  ⚠ Exceeds L3 cache - using DRAM
  Tile sizes: L1=64x64, L2=256x256, L3=1024x1024

Performance Analysis:
  Achieved:      262.9 GFLOPS
  Theoretical:   640.0 GFLOPS (P-cores only)
  Efficiency:    41.1%
  Status: ⚠ Fair - room for improvement
```

## Implementation Details

### Memory Requirements (4k×4k example)

For 4096×4096 matrices (FP32):
- **A matrix**: 4096 × 4096 × 4 bytes = 64 MB
- **B matrix**: 4096 × 4096 × 4 bytes = 64 MB
- **C matrix**: 4096 × 4096 × 4 bytes = 64 MB
- **Total**: 192 MB

### Tile Strategy

**L3 Tiles (1024×1024)**:
- Each tile: 1M floats = 4 MB
- Fits comfortably in L3 cache (25 MB)
- Distributed across cores for parallelism

**L2 Tiles (256×256)**:
- Each tile: 64K floats = 256 KB
- Fits in L2 cache (1.25 MB)
- Reuses data within single core

**L1 Tiles (64×64)**:
- Each tile: 4K floats = 16 KB
- Fits in L1 cache (48 KB)
- Maximizes register reuse

### SIMD Vectorization

```cpp
// Broadcast A[i,k] to 8-wide vector
__m256 a_broadcast = _mm256_set1_ps(A[i * K + k]);

// Load 8 elements of B[k, j:j+8]
__m256 b_vec = _mm256_loadu_ps(&B[k * N + j]);

// FMA: C[i,j:j+8] += A[i,k] * B[k,j:j+8]
c_vec = _mm256_fmadd_ps(a_broadcast, b_vec, c_vec);
```

### Parallelism Strategy

```cpp
// Generate all L3 tiles
std::vector<Tile> l3_tiles = generate_tiles();

// Parallel execution across tiles
std::for_each(std::execution::par, l3_tiles.begin(), l3_tiles.end(),
    [&](const Tile& tile) {
        process_tile(tile);  // Each core gets independent tiles
    });
```

## Performance Expectations

| Matrix Size | Memory | Expected GFLOPS | Cache Level |
|-------------|--------|-----------------|-------------|
| 128×128     | 192 KB | ~400 GFLOPS     | L2 cache    |
| 512×512     | 3 MB   | ~450 GFLOPS     | L3 cache    |
| 1024×1024   | 12 MB  | ~500 GFLOPS     | L3 cache    |
| 2048×2048   | 48 MB  | ~350 GFLOPS     | DRAM        |
| 4096×4096   | 192 MB | ~260 GFLOPS     | DRAM        |

Performance degrades for large matrices due to DRAM bandwidth limitations (~70-80 GB/s).

## Optimization Checklist

✅ Multi-level cache blocking (L1/L2/L3)
✅ AVX2 SIMD vectorization (8-wide)
✅ FMA (fused multiply-add) instructions
✅ C++17 parallel algorithms (multi-core)
✅ Cache-friendly row-major layout
✅ Loop unrolling (compiler flags)
✅ Fast math optimizations
✅ Native CPU tuning (-march=native)

## Future Optimizations

- [ ] AVX-512 support (16-wide, 2× throughput)
- [ ] Register blocking for micro-kernels
- [ ] B matrix transposition for better locality
- [ ] Prefetching hints (`_mm_prefetch`)
- [ ] NUMA-aware memory allocation
- [ ] Dynamic tile size selection
- [ ] Mixed precision (FP16/BF16 compute)

## API Reference

### TiledMatMul Class

```cpp
#include "tiled_matmul.hpp"
using namespace matmul;

// Create matrices
auto A = TiledMatMul::create_random_matrix(4096, 4096);
auto B = TiledMatMul::create_random_matrix(4096, 4096);
auto C = TiledMatMul::create_matrix(4096, 4096);

// Compute C = A * B
TiledMatMul::multiply(A, B, C, 4096, 4096, 4096);

// Validate (optional)
float error = TiledMatMul::compute_error(A, B, C, 4096, 4096, 4096);
std::cout << "Max relative error: " << error << "\n";
```

## References

- [Intel Optimization Manual](https://www.intel.com/content/www/us/en/developer/articles/technical/intel-sdm.html)
- [BLIS: BLAS-like Library Instantiation Software](https://github.com/flame/blis)
- [Goto's Algorithm for GEMM](https://www.cs.utexas.edu/~flame/pubs/GotoTOMS_revision.pdf)
- [C++17 Parallel Algorithms](https://en.cppreference.com/w/cpp/algorithm/execution_policy_tag_t)

## License

This implementation is provided as-is for educational and research purposes.
