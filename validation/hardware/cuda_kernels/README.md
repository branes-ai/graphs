# CUDA Kernel Validation for Jetson Orin Nano

This directory contains CUDA kernels for validating the SM occupancy model in the Jetson Orin Nano hardware mapper.

## Kernels

### 1. Matrix-Vector (`matvec_kernel.cu`)

Two implementations:
- **`matvec_shared_memory`**: Uses shared memory to cache vector tiles
- **`matvec_warp_reduce`**: Uses warp shuffle for reduction (no shared memory)

Key occupancy metrics:
- Block size: 256 threads (8 warps)
- Shared memory: 1 KB per block
- Theoretical occupancy: 100% (6 blocks × 8 warps = 48 warps/SM)

### 2. Matrix-Matrix (`matmul_kernel.cu`)

Three implementations:
- **`matmul_naive`**: Baseline without optimization
- **`matmul_tiled`**: Shared memory tiling (8 KB per block)
- **`matmul_double_buffered`**: Double buffering (16 KB per block)

Key occupancy metrics:
- Block size: 32×8 = 256 threads (8 warps)
- Tile size: 32×32
- Tiled shared memory: 8 KB → 5 blocks/SM → 83% occupancy
- Double-buffered: 16 KB → 2 blocks/SM → 33% occupancy

## Jetson Orin Nano Specifications

| Parameter | Value |
|-----------|-------|
| Architecture | Ampere (SM 8.7) |
| SMs | 8 |
| CUDA cores/SM | 128 |
| Tensor cores/SM | 4 |
| Warp size | 32 |
| Max threads/SM | 1536 |
| Max warps/SM | 48 |
| Max blocks/SM | 16 |
| Registers/SM | 65536 |
| L1/Shared (unified) | 128 KB |
| Default shared | 48 KB |
| L2 cache | 512 KB |
| Memory BW | 68 GB/s (LPDDR5) |
| Peak FP32 | ~1.6 TFLOPS |

## Building

```bash
# For Jetson Orin Nano (default)
make

# For other GPUs, edit NVCC_FLAGS in Makefile:
# -arch=sm_89 for RTX 4090
# -arch=sm_86 for RTX 3090
# -arch=sm_80 for A100
```

## Running

```bash
# Run with occupancy analysis printed
./matvec_kernel
./matmul_kernel
```

## Profiling

```bash
# Using Nsight Compute (ncu)
make profile-matvec
make profile-matmul

# Detailed occupancy
make occupancy
```

## Occupancy Calculation

The theoretical occupancy is limited by:

1. **Thread limit**: `1536 / threads_per_block`
2. **Warp limit**: `48 / warps_per_block`
3. **Shared memory**: `48KB / shared_per_block`
4. **Registers**: `65536 / (threads × regs_per_thread)`
5. **Block limit**: 16

Occupancy = `min(all limits) × warps_per_block / 48`

## Warp Scheduling

On Ampere, warp schedulers issue instructions to warps:
- 4 warp schedulers per SM
- Each scheduler can issue 1 instruction per cycle
- Warps are scheduled in round-robin among eligible warps
- Memory latency (~400 cycles) hidden by switching warps

## Shared Memory Bank Conflicts

32 banks, 4 bytes per bank:
- Consecutive 32-bit accesses across warp → no conflict
- Same bank accessed by multiple threads → serialization
- Broadcast: all threads access same address → no conflict

## Validating Against Hardware Mapper

Compare these kernels' occupancy with:
```python
from graphs.hardware.mappers.gpu import create_jetson_orin_nano_8gb_mapper

mapper = create_jetson_orin_nano_8gb_mapper()
# Check mapper's SM allocation logic
```

The mapper should predict:
- Blocks per SM based on resource limits
- Warp occupancy percentage
- Memory-bound vs compute-bound classification
- Latency estimates using roofline model
