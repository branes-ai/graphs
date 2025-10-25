"""
Intel Xeon Platinum 8592Plus Resource Model hardware resource model.

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


def intel_xeon_platinum_8592plus_resource_model() -> HardwareResourceModel:
    """
    Intel Xeon Platinum 8592+ (Sapphire Rapids) - Flagship 64-core Datacenter Processor.

    ARCHITECTURE:
    - 64 Golden Cove cores (128 threads with HyperThreading)
    - Intel 7 process (10nm Enhanced SuperFin)
    - Monolithic die design
    - 7% more cores than 8490H

    PERFORMANCE:
    - Clock: 1.9 GHz base, 3.9 GHz single-core boost, 3.0 GHz all-core boost
    - Peak FP32: ~3.07 TFLOPS @ 3.0 GHz all-core (64 cores × 16 ops/cycle)
    - Peak INT8: ~98.3 TOPS with AMX (64 cores × 2 tiles × 256 ops/cycle @ 3.0 GHz)
    - AMX: Advanced Matrix Extensions for AI acceleration

    CACHE HIERARCHY:
    - L1 Data: 48 KB per core (3.1 MB total)
    - L1 Instruction: 32 KB per core (2.0 MB total)
    - L2: 2 MB per core (128 MB total)
    - L3 (LLC): 120 MB shared

    MEMORY:
    - 8-channel DDR5-4800
    - Up to 4TB capacity
    - Peak bandwidth: 307.2 GB/s (8 × 38.4 GB/s)

    POWER:
    - TDP: 350W (same as 8490H)
    - Idle: ~85W (estimate)
    - Dynamic: ~265W

    AI ACCELERATION:
    - AMX (Advanced Matrix Extensions): INT8, BF16 matrix operations
    - VNNI (Vector Neural Network Instructions): INT8 dot products
    - Deep Learning Boost
    - Highest AMX performance in Sapphire Rapids lineup

    CONNECTIVITY:
    - PCIe 5.0: 80 lanes
    - CXL 1.1 support

    USE CASES:
    - AI training and inference (flagship AI SKU)
    - HPC workloads
    - Database servers
    - Virtualization hosts

    CALIBRATION STATUS:
    ⚠ ESTIMATED - Based on Intel published specs
    - AMX performance is theoretical peak
    - Sustained performance depends on thermal limits

    REFERENCES:
    - Intel Xeon Scalable Processors (4th Gen) Product Brief
    - Intel AMX Architecture Guide
    """
    # Performance calculations
    num_cores = 64
    base_clock_hz = 1.9e9  # 1.9 GHz
    boost_clock_hz = 3.9e9  # 3.9 GHz (single core)
    all_core_boost_hz = 3.0e9  # 3.0 GHz (realistic sustained, higher than 8490H)

    # AVX-512: 16 FP32 ops/cycle per core
    fp32_ops_per_core = 16  # AVX-512
    fp16_ops_per_core = 32  # AVX-512 FP16

    # AMX: 2 tiles per core, each tile 16×16 INT8
    amx_tiles_per_core = 2
    amx_ops_per_tile = 256  # 16×16 matrix

    # Peak performance (all-core boost)
    peak_fp32 = num_cores * fp32_ops_per_core * all_core_boost_hz  # 3.07 TFLOPS
    peak_fp16 = num_cores * fp16_ops_per_core * all_core_boost_hz  # 6.14 TFLOPS
    peak_int8_amx = num_cores * amx_tiles_per_core * amx_ops_per_tile * all_core_boost_hz  # 98.3 TOPS

    # Memory bandwidth (8-channel DDR5-4800, same as 8490H)
    channels = 8
    ddr5_rate = 4800e6  # 4800 MT/s
    bytes_per_transfer = 8  # 64-bit per channel
    peak_bandwidth = channels * ddr5_rate * bytes_per_transfer  # 307.2 GB/s

    # Power and energy
    tdp = 350.0  # Watts (same as 8490H)
    idle_power = 85.0  # Estimated
    dynamic_power = tdp - idle_power  # 265W

    # Energy per operation (at peak)
    energy_per_flop_fp32 = dynamic_power / peak_fp32  # ~86 pJ/FLOP
    energy_per_byte = 30e-12  # 30 pJ/byte (datacenter-class)

    return HardwareResourceModel(
        name="Intel-Xeon-Platinum-8592+",
        hardware_type=HardwareType.CPU,
        compute_units=num_cores,  # 64 cores
        threads_per_unit=2,  # HyperThreading (SMT)
        warps_per_unit=1,  # No warp concept in CPUs
        warp_size=16,  # AVX-512 width for FP32

        precision_profiles={
            Precision.FP64: PrecisionProfile(
                precision=Precision.FP64,
                peak_ops_per_sec=peak_fp32 / 2,  # 1.54 TFLOPS (half of FP32)
                tensor_core_supported=False,
                relative_speedup=0.5,
                bytes_per_element=8,
            ),
            Precision.FP32: PrecisionProfile(
                precision=Precision.FP32,
                peak_ops_per_sec=peak_fp32,  # 3.07 TFLOPS
                tensor_core_supported=False,
                relative_speedup=1.0,
                bytes_per_element=4,
            ),
            Precision.BF16: PrecisionProfile(
                precision=Precision.BF16,
                peak_ops_per_sec=peak_int8_amx / 2,  # 49.2 TFLOPS (AMX BF16)
                tensor_core_supported=True,  # AMX
                relative_speedup=16.0,  # AMX is much faster than SIMD
                bytes_per_element=2,
            ),
            Precision.FP16: PrecisionProfile(
                precision=Precision.FP16,
                peak_ops_per_sec=peak_fp16,  # 6.14 TFLOPS (AVX-512 FP16)
                tensor_core_supported=False,
                relative_speedup=2.0,
                bytes_per_element=2,
            ),
            Precision.INT8: PrecisionProfile(
                precision=Precision.INT8,
                peak_ops_per_sec=peak_int8_amx,  # 98.3 TOPS (AMX INT8)
                tensor_core_supported=True,  # AMX
                relative_speedup=32.0,  # AMX is very fast for INT8
                bytes_per_element=1,
                accumulator_precision=Precision.INT32,
            ),
        },
        default_precision=Precision.FP32,

        peak_bandwidth=peak_bandwidth,  # 307.2 GB/s (8-channel DDR5-4800)
        l1_cache_per_unit=48 * 1024,  # 48 KB L1D per core
        l2_cache_total=128 * 1024 * 1024,  # 128 MB total L2 (2 MB × 64 cores)
        main_memory=512 * 1024**3,  # 512 GB (typical server config, up to 4TB)
        energy_per_flop_fp32=energy_per_flop_fp32,  # ~86 pJ/FLOP
        energy_per_byte=energy_per_byte,  # 30 pJ/byte
        min_occupancy=0.5,
        max_concurrent_kernels=num_cores,  # One kernel per core
        wave_quantization=1,
    )


