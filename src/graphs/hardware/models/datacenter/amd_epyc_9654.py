"""
Amd Epyc 9654 Resource Model hardware resource model.

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


def amd_epyc_9654_resource_model() -> HardwareResourceModel:
    """
    AMD EPYC 9654 (Genoa) - Datacenter Server Processor.

    ARCHITECTURE:
    - 96 Zen 4 cores (192 threads with SMT)
    - TSMC 5nm process
    - Chiplet design (12× 8-core CCDs + I/O die)

    PERFORMANCE:
    - Clock: 2.4 GHz base, 3.7 GHz boost
    - Peak FP32: ~1.84 TFLOPS @ 2.4 GHz base (96 cores × 8 effective ops/cycle)
    - AVX-512: Double-pumped 256-bit (not native 512-bit)
    - Peak INT8: ~7.37 TOPS (96 cores × 32 ops/cycle @ 2.4 GHz)

    CACHE HIERARCHY:
    - L1 Data: 32 KB per core (3.1 MB total)
    - L1 Instruction: 32 KB per core (3.1 MB total)
    - L2: 1 MB per core (96 MB total)
    - L3: 384 MB shared (32 MB per CCD × 12 CCDs)

    MEMORY:
    - 12-channel DDR5-4800
    - Up to 6TB capacity
    - Peak bandwidth: 460.8 GB/s (12 × 38.4 GB/s)

    POWER:
    - TDP: 360W (can be tuned up to 400W)
    - Idle: ~90W (estimate)
    - Dynamic: ~270W

    AI ACCELERATION:
    - AVX-512 support (double-pumped, slower than native)
    - AVX2 for wider compatibility
    - No dedicated AI accelerator (unlike Intel AMX)

    CONNECTIVITY:
    - PCIe 5.0: 128 lanes
    - CXL 1.1+ support

    USE CASES:
    - Cloud computing (highest core density)
    - Virtualization (many VMs)
    - Database servers
    - Scientific computing

    CALIBRATION STATUS:
    ⚠ ESTIMATED - Based on AMD published specs
    - AVX-512 is double-pumped, so effective throughput is lower

    REFERENCES:
    - AMD EPYC 9004 Series Processors Product Brief
    - AMD Zen 4 Architecture
    """
    # Performance calculations
    num_cores = 96
    base_clock_hz = 2.4e9  # 2.4 GHz
    boost_clock_hz = 3.7e9  # 3.7 GHz (single core)

    # AVX-512 (double-pumped): Effective 8 FP32 ops/cycle
    # Native 256-bit × 2 ops = 8 FP32 per cycle (but takes 2 cycles for 512-bit)
    fp32_ops_per_core = 8  # Effective (double-pumped AVX-512)
    fp16_ops_per_core = 16  # Effective
    int8_ops_per_core = 32  # Effective

    # Peak performance (base clock, conservative)
    peak_fp32 = num_cores * fp32_ops_per_core * base_clock_hz  # 1.84 TFLOPS
    peak_fp16 = num_cores * fp16_ops_per_core * base_clock_hz  # 3.69 TFLOPS
    peak_int8 = num_cores * int8_ops_per_core * base_clock_hz  # 7.37 TOPS

    # Memory bandwidth (12-channel DDR5-4800)
    channels = 12
    ddr5_rate = 4800e6  # 4800 MT/s
    bytes_per_transfer = 8  # 64-bit per channel
    peak_bandwidth = channels * ddr5_rate * bytes_per_transfer  # 460.8 GB/s

    # Power and energy
    tdp = 360.0  # Watts
    idle_power = 90.0  # Estimated
    dynamic_power = tdp - idle_power  # 270W
    energy_per_byte = 28e-12  # 28 pJ/byte (datacenter-class DDR5)

    # ========================================================================
    # Scalar Fabric (Custom Datacenter for 5+ GHz capability)
    # ========================================================================
    scalar_fabric = ComputeFabric(
        fabric_type="scalar_alu",
        circuit_type="custom_datacenter",  # 2.75× for high-frequency custom circuits
        num_units=num_cores * 2,           # 96 cores × 2 ALUs/core = 192 ALUs
        ops_per_unit_per_clock={
            Precision.FP64: 2,               # FMA: 2 ops/clock
            Precision.FP32: 2,               # FMA: 2 ops/clock
        },
        core_frequency_hz=base_clock_hz,   # 2.4 GHz base
        process_node_nm=5,                  # TSMC 5nm
        energy_per_flop_fp32=get_base_alu_energy(5, 'custom_datacenter'),  # 1.5 pJ × 2.75 = 4.125 pJ
        energy_scaling={
            Precision.FP64: 2.0,
            Precision.FP32: 1.0,
        }
    )

    # ========================================================================
    # AVX-512 SIMD Fabric (Double-pumped on 256-bit units)
    # ========================================================================
    # AMD's AVX-512 is double-pumped: takes 2 cycles for one 512-bit op
    # Effective throughput: 8 FP32 ops/cycle
    avx512_fabric = ComputeFabric(
        fabric_type="avx512_double_pump",
        circuit_type="simd_packed",        # 0.90× (10% more efficient for packed ops)
        num_units=num_cores,               # 96 cores × 1 AVX-512 unit/core
        ops_per_unit_per_clock={
            Precision.FP64: 4,               # Double-pumped: 256-bit native
            Precision.FP32: 8,               # Double-pumped: 256-bit native
            Precision.FP16: 16,              # Double-pumped
            Precision.INT8: 32,              # Double-pumped
        },
        core_frequency_hz=base_clock_hz,   # 2.4 GHz base
        process_node_nm=5,
        energy_per_flop_fp32=get_base_alu_energy(5, 'simd_packed'),  # 1.5 pJ × 0.90 = 1.35 pJ
        energy_scaling={
            Precision.FP64: 2.0,
            Precision.FP32: 1.0,
            Precision.FP16: 0.5,
            Precision.INT8: 0.125,
        }
    )

    # Calculate peak ops from fabrics
    scalar_fp64_peak = scalar_fabric.get_peak_ops_per_sec(Precision.FP64)
    scalar_fp32_peak = scalar_fabric.get_peak_ops_per_sec(Precision.FP32)
    avx512_fp32_peak = avx512_fabric.get_peak_ops_per_sec(Precision.FP32)
    avx512_fp16_peak = avx512_fabric.get_peak_ops_per_sec(Precision.FP16)
    avx512_int8_peak = avx512_fabric.get_peak_ops_per_sec(Precision.INT8)

    return HardwareResourceModel(
        name="AMD-EPYC-9654-Genoa",
        hardware_type=HardwareType.CPU,

        # NEW: Compute fabrics (Scalar + AVX-512, no AMX on AMD)
        compute_fabrics=[scalar_fabric, avx512_fabric],

        # Legacy fields
        compute_units=num_cores,  # 96 cores
        threads_per_unit=2,  # SMT (2 threads per core)
        warps_per_unit=1,  # No warp concept in CPUs
        warp_size=8,  # Effective SIMD width for FP32 (double-pumped)

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
            Precision.FP16: PrecisionProfile(
                precision=Precision.FP16,
                peak_ops_per_sec=avx512_fp16_peak,  # AVX-512 for FP16
                tensor_core_supported=False,
                relative_speedup=2.0,
                bytes_per_element=2,
            ),
            Precision.INT8: PrecisionProfile(
                precision=Precision.INT8,
                peak_ops_per_sec=avx512_int8_peak,  # AVX-512 for INT8
                tensor_core_supported=False,  # No AMX equivalent
                relative_speedup=4.0,
                bytes_per_element=1,
                accumulator_precision=Precision.INT32,
            ),
        },
        default_precision=Precision.FP32,

        peak_bandwidth=peak_bandwidth,  # 460.8 GB/s (12-channel DDR5-4800)
        l1_cache_per_unit=32 * 1024,  # 32 KB L1D per core
        l2_cache_total=96 * 1024 * 1024,  # 96 MB total L2 (1 MB × 96 cores)
        main_memory=512 * 1024**3,  # 512 GB (typical server config, up to 6TB)

        # Legacy energy (use AVX-512 fabric as baseline for FP32)
        energy_per_flop_fp32=avx512_fabric.energy_per_flop_fp32,  # 1.35 pJ (SIMD packed)
        energy_per_byte=energy_per_byte,  # 28 pJ/byte
        energy_scaling={
            Precision.FP64: 2.0,
            Precision.FP32: 1.0,
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
                tdp_watts=360.0,
                cooling_solution="datacenter-liquid",
                performance_specs={}
            ),
        },
    )


