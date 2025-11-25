# Session Log: CUDA Kernel Validation for Jetson Orin Nano SM Occupancy

**Date**: 2025-11-25
**Focus**: Creating CUDA kernels to validate Jetson Orin Nano hardware mapper SM occupancy modeling

## Objective

Create CUDA kernels that demonstrate and measure SM occupancy, shared memory usage, and warp assignment on Jetson Orin Nano. These serve as ground truth for validating the GPU hardware mapper's occupancy predictions.

## Background

The Jetson Orin Nano mapper (`src/graphs/hardware/mappers/gpu.py`) models:
- SM allocation based on workload size
- Warp occupancy limited by threads, warps, shared memory, registers
- Memory bandwidth utilization
- Compute vs memory-bound classification

Validation requires comparing these predictions against actual CUDA kernel behavior.

## Deliverables

### 1. Matrix-Vector Kernel (`matvec_kernel.cu`)

Two implementations demonstrating different resource trade-offs:

| Kernel | Threads | Warps | Shared Mem | Occupancy |
|--------|---------|-------|------------|-----------|
| `matvec_shared_memory` | 256 | 8 | 1 KB | 100% |
| `matvec_warp_reduce` | varies | 1/output | 0 | ~100% |

**Key features:**
- Shared memory version caches vector tiles for bandwidth reduction
- Warp reduction version uses `__shfl_down_sync` for register-only dot product
- Detailed comments on memory coalescing patterns

### 2. Matrix-Matrix Kernel (`matmul_kernel.cu`)

Three implementations showing optimization progression:

| Kernel | Shared Mem | Blocks/SM | Occupancy | Expected Speedup |
|--------|------------|-----------|-----------|------------------|
| `matmul_naive` | 0 | 6 | 100% | 1x |
| `matmul_tiled` | 8 KB | 5 | 83% | ~10-20x |
| `matmul_double_buffered` | 16 KB | 2 | 33% | ~15-25x |

**Key features:**
- 32×32 tiles for optimal shared memory bank utilization
- 32×8 thread blocks (8 warps) with 4 output rows per thread
- Double buffering overlaps global memory loads with compute

### 3. Build Infrastructure (`Makefile`)

```bash
make                    # Build for SM 8.7 (Orin Nano)
make profile-matmul     # Profile with ncu/nvprof
make ptx                # Generate PTX for register analysis
make sass               # Generate SASS for instruction analysis
```

## Jetson Orin Nano Specifications (SM 8.7)

| Resource | Limit |
|----------|-------|
| SMs | 8 |
| CUDA cores/SM | 128 |
| Tensor cores/SM | 4 |
| Warp size | 32 |
| Max threads/SM | 1536 |
| Max warps/SM | 48 |
| Max blocks/SM | 16 |
| Registers/SM | 65536 |
| L1+Shared/SM | 128 KB (unified) |
| Default shared | 48 KB |
| L2 cache | 512 KB |
| Memory BW | 68 GB/s |
| Peak FP32 | ~1.6 TFLOPS |

## Occupancy Calculation Formula

```
blocks_per_sm = min(
    1536 / threads_per_block,      # Thread limit
    48 / warps_per_block,          # Warp limit
    48KB / shared_per_block,       # Shared memory limit
    65536 / total_regs_per_block,  # Register limit
    16                              # HW block limit
)

occupancy = (blocks_per_sm * warps_per_block) / 48 * 100%
```

## Warp Assignment Visualization

For 32×8 thread block (matmul_tiled):

```
threadIdx.x: 0  1  2  ... 31
threadIdx.y:
    0       [Warp 0----------]
    1       [Warp 1----------]
    2       [Warp 2----------]
    3       [Warp 3----------]
    4       [Warp 4----------]
    5       [Warp 5----------]
    6       [Warp 6----------]
    7       [Warp 7----------]
```

Each warp executes in lockstep (SIMT). Divergent branches serialize execution.

## Shared Memory Bank Conflicts

32 banks × 4 bytes per bank:
- **No conflict**: Warp accesses consecutive 4-byte elements
- **Broadcast**: All threads access same address (free)
- **Conflict**: Multiple threads access different addresses in same bank (serialized)

In `matmul_tiled`:
- `A_shared[ty + i*8][k]`: Row-major, all threads read same `k` → broadcast
- `B_shared[k][tx]`: Column access, threads read consecutive `tx` → no conflict

## Validation Process

1. **Build and run on Jetson Orin Nano**:
   ```bash
   make && ./matmul_kernel
   ```

2. **Compare printed occupancy** with mapper predictions:
   ```python
   from graphs.hardware.mappers.gpu import create_jetson_orin_nano_8gb_mapper
   mapper = create_jetson_orin_nano_8gb_mapper()
   ```

3. **Profile for actual metrics**:
   ```bash
   ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active ./matmul_kernel
   ```

4. **Compare achieved vs theoretical**:
   - Bandwidth utilization (actual vs 68 GB/s)
   - GFLOPS (actual vs 1.6 TFLOPS peak)
   - Occupancy (achieved vs theoretical)

## Files Created

| Path | Description |
|------|-------------|
| `validation/hardware/cuda_kernels/matvec_kernel.cu` | Matrix-vector kernels |
| `validation/hardware/cuda_kernels/matmul_kernel.cu` | Matrix-matrix kernels |
| `validation/hardware/cuda_kernels/Makefile` | Build infrastructure |
| `validation/hardware/cuda_kernels/README.md` | Documentation |

## Next Steps

1. Run kernels on actual Jetson Orin Nano hardware
2. Collect `ncu` metrics for achieved occupancy, bandwidth, GFLOPS
3. Compare with GPU mapper predictions
4. Adjust mapper parameters if discrepancies found
5. Add tensor core kernels for FP16/INT8 validation

## References

- [CUDA Occupancy Calculator](https://developer.nvidia.com/cuda-occupancy-calculator)
- [Jetson Orin Nano Developer Kit](https://developer.nvidia.com/embedded/jetson-orin-nano)
- [NVIDIA Ampere Architecture Whitepaper](https://www.nvidia.com/en-us/data-center/nvidia-ampere-gpu-architecture/)
