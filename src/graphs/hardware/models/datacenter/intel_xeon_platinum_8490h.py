"""
Intel Xeon Platinum 8490H Resource Model hardware resource model.

Extracted from resource_model.py during refactoring.
"""

from ...resource_model import (
    HardwareResourceModel,
    HardwareType,
    Precision,
    PrecisionProfile,
    ComputeFabric,
    get_base_alu_energy,
    ClockDomain,
    ComputeResource,
    TileSpecialization,
    KPUComputeResource,
    PerformanceCharacteristics,
    ThermalOperatingPoint,
)


def intel_xeon_platinum_8490h_resource_model() -> HardwareResourceModel:
    """
    Intel Xeon Platinum 8490H (Sapphire Rapids) - Datacenter Server Processor.

    ARCHITECTURE:
    - 60 Golden Cove cores (120 threads with HyperThreading)
    - Intel 7 process (10nm Enhanced SuperFin)
    - Monolithic die design

    PERFORMANCE:
    - Clock: 2.0 GHz base, 3.5 GHz single-core boost, 2.9 GHz all-core boost
    - Peak FP32: ~2.78 TFLOPS @ 2.9 GHz all-core (60 cores × 16 ops/cycle)
    - Peak INT8: ~88.7 TOPS with AMX (60 cores × 2 tiles × 256 ops/cycle @ 2.9 GHz)
    - AMX: Advanced Matrix Extensions for AI acceleration

    CACHE HIERARCHY:
    - L1 Data: 48 KB per core (2.9 MB total)
    - L1 Instruction: 32 KB per core (1.9 MB total)
    - L2: 2 MB per core (120 MB total)
    - L3 (LLC): 112.5 MB shared

    MEMORY:
    - 8-channel DDR5-4800
    - Up to 4TB capacity
    - Peak bandwidth: 307 GB/s (8 × 38.4 GB/s)

    POWER:
    - TDP: 350W
    - Idle: ~80W (estimate)
    - Dynamic: ~270W

    AI ACCELERATION:
    - AMX (Advanced Matrix Extensions): INT8, BF16 matrix operations
    - VNNI (Vector Neural Network Instructions): INT8 dot products
    - Deep Learning Boost
    - Better AI performance than previous Xeon generations

    CONNECTIVITY:
    - PCIe 5.0: 80 lanes
    - CXL 1.1 support

    USE CASES:
    - AI training and inference
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
    num_cores = 60
    base_clock_hz = 2.0e9  # 2.0 GHz
    boost_clock_hz = 3.5e9  # 3.5 GHz (single core)
    all_core_boost_hz = 2.9e9  # 2.9 GHz (realistic sustained)

    # AVX-512: 16 FP32 ops/cycle per core
    fp32_ops_per_core = 16  # AVX-512
    fp16_ops_per_core = 32  # AVX-512 FP16

    # AMX: 2 tiles per core, each tile 16×16 INT8
    amx_tiles_per_core = 2
    amx_ops_per_tile = 256  # 16×16 matrix

    # Peak performance (all-core boost)
    peak_fp32 = num_cores * fp32_ops_per_core * all_core_boost_hz  # 2.78 TFLOPS
    peak_fp16 = num_cores * fp16_ops_per_core * all_core_boost_hz  # 5.57 TFLOPS
    peak_int8_amx = num_cores * amx_tiles_per_core * amx_ops_per_tile * all_core_boost_hz  # 88.7 TOPS

    # Memory bandwidth (8-channel DDR5-4800)
    channels = 8
    ddr5_rate = 4800e6  # 4800 MT/s
    bytes_per_transfer = 8  # 64-bit per channel
    peak_bandwidth = channels * ddr5_rate * bytes_per_transfer  # 307.2 GB/s

    # Power and energy
    tdp = 350.0  # Watts
    idle_power = 80.0  # Estimated
    dynamic_power = tdp - idle_power  # 270W
    energy_per_byte = 30e-12  # 30 pJ/byte (datacenter-class DDR5)

    # ========================================================================
    # Scalar Fabric (Custom Datacenter for 5+ GHz capability)
    # ========================================================================
    # x86 cores are custom high-frequency designs (2.75× multiplier)
    scalar_fabric = ComputeFabric(
        fabric_type="scalar_alu",
        circuit_type="custom_datacenter",  # 2.75× for high-frequency custom circuits
        num_units=num_cores * 2,           # 60 cores × 2 ALUs/core = 120 ALUs
        ops_per_unit_per_clock={
            Precision.FP64: 2,               # FMA: 2 ops/clock
            Precision.FP32: 2,               # FMA: 2 ops/clock
        },
        core_frequency_hz=base_clock_hz,   # 2.0 GHz base
        process_node_nm=10,                 # Intel 7 (10nm-class)
        energy_per_flop_fp32=get_base_alu_energy(10, 'custom_datacenter'),  # 2.1 pJ × 2.75 = 5.775 pJ
        energy_scaling={
            Precision.FP64: 2.0,
            Precision.FP32: 1.0,
        }
    )

    # ========================================================================
    # AVX-512 SIMD Fabric (SIMD Packed for vector operations)
    # ========================================================================
    avx512_fabric = ComputeFabric(
        fabric_type="avx512",
        circuit_type="simd_packed",        # 0.90× (10% more efficient for packed ops)
        num_units=num_cores,               # 60 cores × 1 AVX-512 unit/core
        ops_per_unit_per_clock={
            Precision.FP64: 8,               # 512-bit / 64-bit = 8 FP64/cycle
            Precision.FP32: 16,              # 512-bit / 32-bit = 16 FP32/cycle
            Precision.FP16: 32,              # 512-bit / 16-bit = 32 FP16/cycle
        },
        core_frequency_hz=all_core_boost_hz,  # 2.9 GHz all-core
        process_node_nm=10,
        energy_per_flop_fp32=get_base_alu_energy(10, 'simd_packed'),  # 2.1 pJ × 0.90 = 1.89 pJ
        energy_scaling={
            Precision.FP64: 2.0,
            Precision.FP32: 1.0,
            Precision.FP16: 0.5,
            Precision.INT8: 0.125,
        }
    )

    # ========================================================================
    # AMX Matrix Fabric (Tensor Core-like for matrix operations)
    # ========================================================================
    amx_fabric = ComputeFabric(
        fabric_type="amx_tile",
        circuit_type="tensor_core",        # 0.85× (15% more efficient, like Tensor Cores)
        num_units=num_cores * amx_tiles_per_core,  # 60 cores × 2 tiles = 120 tiles
        ops_per_unit_per_clock={
            Precision.BF16: 128,             # 16×16 matrix / 2 = 128 ops/tile/cycle
            Precision.INT8: 256,             # 16×16 matrix = 256 ops/tile/cycle
        },
        core_frequency_hz=all_core_boost_hz,  # 2.9 GHz all-core
        process_node_nm=10,
        energy_per_flop_fp32=get_base_alu_energy(10, 'tensor_core'),  # 2.1 pJ × 0.85 = 1.785 pJ
        energy_scaling={
            Precision.BF16: 0.5,
            Precision.INT8: 0.125,
        }
    )

    # Calculate peak ops from fabrics
    scalar_fp64_peak = scalar_fabric.get_peak_ops_per_sec(Precision.FP64)
    scalar_fp32_peak = scalar_fabric.get_peak_ops_per_sec(Precision.FP32)
    avx512_fp32_peak = avx512_fabric.get_peak_ops_per_sec(Precision.FP32)
    avx512_fp16_peak = avx512_fabric.get_peak_ops_per_sec(Precision.FP16)
    amx_bf16_peak = amx_fabric.get_peak_ops_per_sec(Precision.BF16)
    amx_int8_peak = amx_fabric.get_peak_ops_per_sec(Precision.INT8)

    return HardwareResourceModel(
        name="Intel-Xeon-Platinum-8490H",
        hardware_type=HardwareType.CPU,

        # NEW: Compute fabrics (Scalar + AVX-512 + AMX)
        compute_fabrics=[scalar_fabric, avx512_fabric, amx_fabric],

        # Legacy fields
        compute_units=num_cores,  # 60 cores
        threads_per_unit=2,  # HyperThreading (SMT)
        warps_per_unit=1,  # No warp concept in CPUs
        warp_size=16,  # AVX-512 width for FP32

        # Legacy precision profiles (calculated from fabrics)
        precision_profiles={
            Precision.FP64: PrecisionProfile(
                precision=Precision.FP64,
                peak_ops_per_sec=scalar_fp64_peak,  # Scalar ALUs for FP64
                tensor_core_supported=False,
                relative_speedup=0.5,
                bytes_per_element=8,
            ),
            Precision.FP32: PrecisionProfile(
                precision=Precision.FP32,
                peak_ops_per_sec=avx512_fp32_peak,  # AVX-512 for FP32
                tensor_core_supported=False,
                relative_speedup=1.0,
                bytes_per_element=4,
            ),
            Precision.BF16: PrecisionProfile(
                precision=Precision.BF16,
                peak_ops_per_sec=amx_bf16_peak,  # AMX BF16
                tensor_core_supported=True,  # AMX
                relative_speedup=16.0,  # AMX is much faster than SIMD
                bytes_per_element=2,
            ),
            Precision.FP16: PrecisionProfile(
                precision=Precision.FP16,
                peak_ops_per_sec=avx512_fp16_peak,  # AVX-512 FP16
                tensor_core_supported=False,
                relative_speedup=2.0,
                bytes_per_element=2,
            ),
            Precision.INT8: PrecisionProfile(
                precision=Precision.INT8,
                peak_ops_per_sec=amx_int8_peak,  # AMX INT8
                tensor_core_supported=True,  # AMX
                relative_speedup=32.0,  # AMX is very fast for INT8
                bytes_per_element=1,
                accumulator_precision=Precision.INT32,
            ),
        },
        default_precision=Precision.FP32,

        peak_bandwidth=peak_bandwidth,  # 307.2 GB/s (8-channel DDR5-4800)
        l1_cache_per_unit=48 * 1024,  # 48 KB L1D per core
        l2_cache_total=120 * 1024 * 1024,  # 120 MB total L2 (2 MB × 60 cores)
        main_memory=512 * 1024**3,  # 512 GB (typical server config, up to 4TB)

        # Legacy energy (use AVX-512 fabric as baseline for FP32)
        energy_per_flop_fp32=avx512_fabric.energy_per_flop_fp32,  # 1.89 pJ (SIMD packed)
        energy_per_byte=energy_per_byte,  # 30 pJ/byte
        energy_scaling={
            Precision.FP64: 2.0,
            Precision.FP32: 1.0,
            Precision.BF16: 0.5,
            Precision.FP16: 0.5,
            Precision.INT8: 0.125,
        },
        min_occupancy=0.5,
        max_concurrent_kernels=num_cores,  # One kernel per core
        wave_quantization=1,

        # Thermal operating point
        thermal_operating_points={
            "default": ThermalOperatingPoint(
                name="default",
                tdp_watts=350.0,
                cooling_solution="datacenter-liquid",
                performance_specs={}
            ),
        },
    )


