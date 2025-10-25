"""
Ampere Ampereone 128 Resource Model hardware resource model.

Extracted from resource_model.py during refactoring.
"""

from ...resource_model import (
    HardwareResourceModel,
    HardwareType,
    Precision,
    PrecisionProfile,
    ClockDomain,
    ComputeResource,
    TileSpecialization,
    KPUComputeResource,
    PerformanceCharacteristics,
    ThermalOperatingPoint,
)


def ampere_ampereone_128_resource_model() -> HardwareResourceModel:
    """
    Ampere AmpereOne 128-core ARM Server Processor (A128-30X mid-tier).

    ARCHITECTURE:
    - 128 Ampere 64-bit ARM v8.6+ cores
    - 2×128-bit SIMD units per core (NEON + SVE)
    - Coherent mesh interconnect with distributed snoop filtering
    - TSMC 5nm process
    - 67% of 192-core flagship

    PERFORMANCE:
    - Clock: Up to 3.6 GHz (consistent across all cores)
    - Peak FP32: ~3.69 TFLOPS (128 cores × 8 ops/cycle × 3.6 GHz)
    - Peak FP16/BF16: ~7.37 TFLOPS (128 cores × 16 ops/cycle × 3.6 GHz)
    - Peak INT8: ~14.75 TOPS (128 cores × 32 ops/cycle × 3.6 GHz)

    CACHE HIERARCHY:
    - L1 Data: 64 KB per core (8 MB total)
    - L1 Instruction: 16 KB per core (2 MB total)
    - L2: 2 MB per core (256 MB total)
    - System Cache (L3-like): 48 MB shared

    MEMORY:
    - 8-channel DDR5-5200 (up to 4TB)
    - Peak bandwidth: 332.8 GB/s (8 × 41.6 GB/s)

    POWER:
    - TDP: 210W (A128-30X)
    - Idle: ~40W (estimate)
    - Dynamic: ~170W

    AI ACCELERATION:
    - Native FP16/BF16 support (2×128-bit SIMD)
    - Native INT8/INT16 support
    - Ampere AIO (AI Optimizer) for ML frameworks
    - Better than x86 for AI inference (wider SIMD for low precision)

    CONNECTIVITY:
    - 128 lanes PCIe 5.0 with 32 controllers

    USE CASES:
    - Cloud-native workloads (microservices, containers)
    - AI inference at scale (cloud servers)
    - High-performance computing (HPC)
    - Cost-effective datacenter deployments

    CALIBRATION STATUS:
    ⚠ ESTIMATED - Based on Ampere published specs
    - SIMD calculations based on ARM v8.6+ NEON/SVE
    - Memory bandwidth from DDR5-5200 8-channel configuration

    REFERENCES:
    - Ampere AmpereOne Family Product Brief (2024)
    - ARM v8.6+ Architecture Reference Manual
    - DDR5-5200 specifications
    """
    # Performance calculations
    num_cores = 128
    clock_hz = 3.6e9  # 3.6 GHz
    simd_units_per_core = 2  # 2×128-bit SIMD units

    # SIMD operations per cycle per core
    # 128-bit SIMD: FP32=4, FP16=8, INT8=16 ops per unit
    fp32_ops_per_core = simd_units_per_core * 4  # 8 FP32 ops/cycle
    fp16_ops_per_core = simd_units_per_core * 8  # 16 FP16/BF16 ops/cycle
    int8_ops_per_core = simd_units_per_core * 16  # 32 INT8 ops/cycle

    # Peak performance
    peak_fp32 = num_cores * fp32_ops_per_core * clock_hz  # 3.69 TFLOPS
    peak_fp16 = num_cores * fp16_ops_per_core * clock_hz  # 7.37 TFLOPS
    peak_int8 = num_cores * int8_ops_per_core * clock_hz  # 14.75 TOPS

    # Memory bandwidth (8-channel DDR5-5200, same as 192-core)
    channels = 8
    ddr5_rate = 5200e6  # 5200 MT/s
    bytes_per_transfer = 8  # 64-bit per channel
    peak_bandwidth = channels * ddr5_rate * bytes_per_transfer  # 332.8 GB/s

    # Power and energy
    tdp = 210.0  # Watts (A128-30X, lower than 192-core)
    idle_power = 40.0  # Estimated
    dynamic_power = tdp - idle_power  # 170W

    # Energy per operation (at peak)
    energy_per_flop_fp32 = dynamic_power / peak_fp32  # ~46 pJ/FLOP
    energy_per_byte = 25e-12  # 25 pJ/byte (server-class)

    return HardwareResourceModel(
        name="Ampere-AmpereOne-128core",
        hardware_type=HardwareType.CPU,
        compute_units=num_cores,  # 128 cores
        threads_per_unit=1,  # No SMT/HyperThreading (single thread per core)
        warps_per_unit=1,  # No warp concept in CPUs
        warp_size=16,  # Effective SIMD width for INT8 (per unit)

        precision_profiles={
            Precision.FP64: PrecisionProfile(
                precision=Precision.FP64,
                peak_ops_per_sec=peak_fp32 / 2,  # 1.84 TFLOPS (half of FP32)
                tensor_core_supported=False,
                relative_speedup=0.5,
                bytes_per_element=8,
            ),
            Precision.FP32: PrecisionProfile(
                precision=Precision.FP32,
                peak_ops_per_sec=peak_fp32,  # 3.69 TFLOPS
                tensor_core_supported=False,
                relative_speedup=1.0,
                bytes_per_element=4,
            ),
            Precision.BF16: PrecisionProfile(
                precision=Precision.BF16,
                peak_ops_per_sec=peak_fp16,  # 7.37 TFLOPS (native support)
                tensor_core_supported=True,  # Native ARM SIMD support
                relative_speedup=2.0,
                bytes_per_element=2,
            ),
            Precision.FP16: PrecisionProfile(
                precision=Precision.FP16,
                peak_ops_per_sec=peak_fp16,  # 7.37 TFLOPS (native support)
                tensor_core_supported=True,  # Native ARM SIMD support
                relative_speedup=2.0,
                bytes_per_element=2,
            ),
            Precision.INT16: PrecisionProfile(
                precision=Precision.INT16,
                peak_ops_per_sec=peak_fp16,  # 7.37 TOPS (same as FP16)
                tensor_core_supported=True,  # Native ARM SIMD support
                relative_speedup=2.0,
                bytes_per_element=2,
                accumulator_precision=Precision.INT32,
            ),
            Precision.INT8: PrecisionProfile(
                precision=Precision.INT8,
                peak_ops_per_sec=peak_int8,  # 14.75 TOPS (native support)
                tensor_core_supported=True,  # Native ARM SIMD support
                relative_speedup=4.0,
                bytes_per_element=1,
                accumulator_precision=Precision.INT32,
            ),
        },
        default_precision=Precision.FP32,

        peak_bandwidth=peak_bandwidth,  # 332.8 GB/s (8-channel DDR5-5200)
        l1_cache_per_unit=64 * 1024,  # 64 KB L1D per core
        l2_cache_total=256 * 1024 * 1024,  # 256 MB total L2 (2 MB × 128 cores)
        main_memory=512 * 1024**3,  # 512 GB (typical server config, up to 4TB)
        energy_per_flop_fp32=energy_per_flop_fp32,  # ~46 pJ/FLOP
        energy_per_byte=energy_per_byte,  # 25 pJ/byte
        min_occupancy=0.5,
        max_concurrent_kernels=num_cores,  # One kernel per core
        wave_quantization=1,
    )


