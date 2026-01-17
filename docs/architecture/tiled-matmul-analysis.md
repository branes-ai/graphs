# Matmul tiling and reuse

● Cache and Memory System Specifications

  CPU: Intel Core i7-12700K (12th Gen Alder Lake)
  - Cores: 12 cores (8 P-cores + 4 E-cores), 20 threads total
  - Max Frequency: 5.0 GHz

  Cache Hierarchy

  Per-Core Caches:
  - L1 Data Cache: 48 KB (per P-core), 32 KB (per E-core)
    - 8-way set associative
    - Latency: 4-5 cycles (1 ns @ 5 GHz)
  - L1 Instruction Cache: 32 KB (per P-core), 64 KB (per E-core)
    - Latency: 4-5 cycles (1 ns @ 5 GHz)
  - L2 Cache: 1.25 MB per P-core, ~2 MB per E-core cluster
    - 16-way set associative
    - Total: 12 MB across all cores
    - Latency: 12-15 cycles (3 ns @ 5 GHz)

  Shared Cache:
  - L3 Cache: 25 MB (shared across all cores)
    - 10-way set associative
    - Latency: 40-50 cycles (10 ns @ 5 GHz)

  Cache Line Size: 64 bytes (all levels)

  Memory System

  - Total RAM: 32 GB (31.9 GB available)
  - Memory Type: DDR5 (typical for 12th gen)
  - Estimated Bandwidth: ~70-80 GB/s (DDR5-4800 dual channel)
  - Memory Latency: ~80-100 ns

  Cache Speed Summary

  | Level | Size        | Latency (cycles) | Latency (ns @ 5GHz) | Bandwidth   |
  |-------|-------------|------------------|---------------------|-------------|
  | L1D   | 48 KB/32 KB | 4-5              | ~1 ns               | ~1 TB/s     |
  | L2    | 1.25-2 MB   | 12-15            | ~3 ns               | ~500 GB/s   |
  | L3    | 25 MB       | 40-50            | ~10 ns              | ~300 GB/s   |
  | DRAM  | 32 GB       | 400-500          | ~80-100 ns          | ~70-80 GB/s |

  Implications for Your 4k×4k Matmul

  - Data Size: 3 × 4096² × 4 bytes = 192 MB (doesn't fit in L3)
  - Working Set (512 KB per tile): Fits in L2 cache! ✅
  - Expected Bandwidth: With good tiling, you should achieve near L2 bandwidth (500 GB/s) rather than DRAM bandwidth (70 GB/s)

  This means your output-stationary schedule with 16×16 tiles is well-sized for this CPU's cache hierarchy.
