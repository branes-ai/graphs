"""
Intel Granite Rapids Resource Model hardware resource model.

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


def intel_granite_rapids_resource_model() -> HardwareResourceModel:
    """
    Intel Xeon Granite Rapids (Next-Gen 2024-2025) - 128-core Datacenter Processor.

    ARCHITECTURE:
    - 128 Redwood Cove P-cores (256 threads with HyperThreading)
    - Intel 3 process (Enhanced FinFET)
    - Tile-based chiplet design (new for Intel)
    - 2× core count vs Sapphire Rapids flagship

    PERFORMANCE:
    - Clock: 2.0 GHz base, 3.8 GHz single-core boost, 3.2 GHz all-core boost (estimated)
    - Peak FP32: ~6.55 TFLOPS @ 3.2 GHz all-core (128 cores × 16 ops/cycle)
    - Peak INT8: ~209 TOPS with AMX (128 cores × 2 tiles × 256 ops/cycle @ 3.2 GHz)
    - Enhanced AMX with sparsity acceleration

    CACHE HIERARCHY:
    - L1 Data: 48 KB per core (6.1 MB total)
    - L1 Instruction: 32 KB per core (4.1 MB total)
    - L2: 2 MB per core (256 MB total)
    - L3 (LLC): 320 MB shared (distributed across tiles)

    MEMORY:
    - 8-channel DDR5-5600 (improved from DDR5-4800)
    - 12-channel DDR5-5600 on HBM SKUs
    - Peak bandwidth: 358.4 GB/s (8 × 44.8 GB/s) or 537.6 GB/s (12-channel)

    POWER:
    - TDP: 500W (high core count)
    - Idle: ~100W (estimate)
    - Dynamic: ~400W

    AI ACCELERATION:
    - Enhanced AMX with INT4, FP8 support
    - Sparsity acceleration (structured sparsity)
    - VNNI improvements for INT8
    - Better AMX efficiency than Sapphire Rapids

    CONNECTIVITY:
    - PCIe 6.0: 96 lanes
    - CXL 2.0 support
    - UPI (Ultra Path Interconnect) for multi-socket

    USE CASES:
    - Large-scale AI training and inference
    - HPC workloads
    - Cloud computing at scale
    - Next-generation datacenter deployments

    CALIBRATION STATUS:
    ⚠ PROJECTED - Based on Intel roadmap and industry estimates
    - Not yet shipping (2024-2025 timeline)
    - Specs are projections based on Intel disclosures

    REFERENCES:
    - Intel Xeon Roadmap 2024
    - Intel 3 Process Technology Brief
    - Industry analyst projections
    """
    # Performance calculations
    num_cores = 128
    base_clock_hz = 2.0e9  # 2.0 GHz
    boost_clock_hz = 3.8e9  # 3.8 GHz (single core, estimated)
    all_core_boost_hz = 3.2e9  # 3.2 GHz (estimated, higher than Sapphire Rapids)

    # AVX-512: 16 FP32 ops/cycle per core
    fp32_ops_per_core = 16  # AVX-512
    fp16_ops_per_core = 32  # AVX-512 FP16

    # Enhanced AMX: 2 tiles per core, each tile 16×16 INT8
    amx_tiles_per_core = 2
    amx_ops_per_tile = 256  # 16×16 matrix

    # Peak performance (all-core boost)
    peak_fp32 = num_cores * fp32_ops_per_core * all_core_boost_hz  # 6.55 TFLOPS
    peak_fp16 = num_cores * fp16_ops_per_core * all_core_boost_hz  # 13.11 TFLOPS
    peak_int8_amx = num_cores * amx_tiles_per_core * amx_ops_per_tile * all_core_boost_hz  # 209.7 TOPS

    # Memory bandwidth (8-channel DDR5-5600, improved)
    channels = 8
    ddr5_rate = 5600e6  # 5600 MT/s (up from 4800)
    bytes_per_transfer = 8  # 64-bit per channel
    peak_bandwidth = channels * ddr5_rate * bytes_per_transfer  # 358.4 GB/s

    # Power and energy
    tdp = 500.0  # Watts (higher due to 128 cores)
    idle_power = 100.0  # Estimated
    dynamic_power = tdp - idle_power  # 400W

    # Energy per operation (at peak)
    energy_per_flop_fp32 = dynamic_power / peak_fp32  # ~61 pJ/FLOP
    energy_per_byte = 28e-12  # 28 pJ/byte (improved process)

    return HardwareResourceModel(
        name="Intel-Xeon-Granite-Rapids",
        hardware_type=HardwareType.CPU,
        compute_units=num_cores,  # 128 cores
        threads_per_unit=2,  # HyperThreading (SMT)
        warps_per_unit=1,  # No warp concept in CPUs
        warp_size=16,  # AVX-512 width for FP32

        precision_profiles={
            Precision.FP64: PrecisionProfile(
                precision=Precision.FP64,
                peak_ops_per_sec=peak_fp32 / 2,  # 3.28 TFLOPS (half of FP32)
                tensor_core_supported=False,
                relative_speedup=0.5,
                bytes_per_element=8,
            ),
            Precision.FP32: PrecisionProfile(
                precision=Precision.FP32,
                peak_ops_per_sec=peak_fp32,  # 6.55 TFLOPS
                tensor_core_supported=False,
                relative_speedup=1.0,
                bytes_per_element=4,
            ),
            Precision.BF16: PrecisionProfile(
                precision=Precision.BF16,
                peak_ops_per_sec=peak_int8_amx / 2,  # 104.9 TFLOPS (Enhanced AMX BF16)
                tensor_core_supported=True,  # Enhanced AMX
                relative_speedup=16.0,  # AMX is much faster than SIMD
                bytes_per_element=2,
            ),
            Precision.FP16: PrecisionProfile(
                precision=Precision.FP16,
                peak_ops_per_sec=peak_fp16,  # 13.11 TFLOPS (AVX-512 FP16)
                tensor_core_supported=False,
                relative_speedup=2.0,
                bytes_per_element=2,
            ),
            Precision.INT8: PrecisionProfile(
                precision=Precision.INT8,
                peak_ops_per_sec=peak_int8_amx,  # 209.7 TOPS (Enhanced AMX INT8)
                tensor_core_supported=True,  # Enhanced AMX
                relative_speedup=32.0,  # AMX is very fast for INT8
                bytes_per_element=1,
                accumulator_precision=Precision.INT32,
            ),
            Precision.INT4: PrecisionProfile(
                precision=Precision.INT4,
                peak_ops_per_sec=peak_int8_amx * 2,  # 419.4 TOPS (Enhanced AMX INT4)
                tensor_core_supported=True,  # Enhanced AMX with INT4 support
                relative_speedup=64.0,
                bytes_per_element=0.5,
                accumulator_precision=Precision.INT32,
            ),
        },
        default_precision=Precision.FP32,

        peak_bandwidth=peak_bandwidth,  # 358.4 GB/s (8-channel DDR5-5600)
        l1_cache_per_unit=48 * 1024,  # 48 KB L1D per core
        l2_cache_total=256 * 1024 * 1024,  # 256 MB total L2 (2 MB × 128 cores)
        main_memory=512 * 1024**3,  # 512 GB (typical server config)
        energy_per_flop_fp32=energy_per_flop_fp32,  # ~61 pJ/FLOP
        energy_per_byte=energy_per_byte,  # 28 pJ/byte
        min_occupancy=0.5,
        max_concurrent_kernels=num_cores,  # One kernel per core
        wave_quantization=1,
    )


