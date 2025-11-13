"""
Amd Epyc Turin Resource Model hardware resource model.

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


def amd_epyc_turin_resource_model() -> HardwareResourceModel:
    """
    AMD EPYC Turin (Zen 5, Next-Gen 2024-2025) - 192-core Datacenter Processor.

    ARCHITECTURE:
    - 192 Zen 5 cores (384 threads with SMT)
    - TSMC 3nm process (N3)
    - Chiplet design (24× 8-core CCDs + I/O die)
    - 50% more cores than EPYC 9754

    PERFORMANCE:
    - Clock: 2.5 GHz base, 3.8 GHz boost (estimated)
    - Peak FP32: ~3.84 TFLOPS @ 2.5 GHz base (192 cores × 8 effective ops/cycle)
    - AVX-512: Native 512-bit (improved from double-pumped Zen 4)
    - Peak INT8: ~15.36 TOPS (192 cores × 32 ops/cycle @ 2.5 GHz)

    CACHE HIERARCHY:
    - L1 Data: 48 KB per core (9.2 MB total, increased from 32 KB)
    - L1 Instruction: 32 KB per core (6.1 MB total)
    - L2: 1 MB per core (192 MB total)
    - L3: 768 MB shared (32 MB per CCD × 24 CCDs)

    MEMORY:
    - 12-channel DDR5-6000 (up from DDR5-4800)
    - Up to 6TB capacity
    - Peak bandwidth: 576 GB/s (12 × 48 GB/s)
    - 25% more bandwidth than EPYC 9000 series

    POWER:
    - TDP: 500W (higher core count)
    - Idle: ~120W (estimate)
    - Dynamic: ~380W

    AI ACCELERATION:
    - Native AVX-512 support (improved from double-pumped)
    - AVX2 for compatibility
    - Possible AI matrix accelerator (rumored, not confirmed)
    - Better INT8 performance than Zen 4

    CONNECTIVITY:
    - PCIe 6.0: 160 lanes
    - CXL 2.0 support

    USE CASES:
    - Cloud computing (extreme core density)
    - Virtualization (384 threads!)
    - Database servers (massive concurrent connections)
    - Large-scale AI inference

    CALIBRATION STATUS:
    ⚠ PROJECTED - Based on AMD roadmap and industry estimates
    - Not yet shipping (2024-2025 timeline)
    - Specs are projections based on AMD disclosures

    REFERENCES:
    - AMD EPYC Roadmap 2024
    - AMD Zen 5 Architecture Disclosures
    - Industry analyst projections
    """
    # Performance calculations
    num_cores = 192
    base_clock_hz = 2.5e9  # 2.5 GHz (estimated)
    boost_clock_hz = 3.8e9  # 3.8 GHz (single core, estimated)

    # Native AVX-512: Effective 8 FP32 ops/cycle (improved throughput)
    fp32_ops_per_core = 8  # Native AVX-512 (not double-pumped)
    fp16_ops_per_core = 16  # Native AVX-512 FP16
    int8_ops_per_core = 32  # Native AVX-512 INT8

    # Peak performance (base clock, conservative)
    peak_fp32 = num_cores * fp32_ops_per_core * base_clock_hz  # 3.84 TFLOPS
    peak_fp16 = num_cores * fp16_ops_per_core * base_clock_hz  # 7.68 TFLOPS
    peak_int8 = num_cores * int8_ops_per_core * base_clock_hz  # 15.36 TOPS

    # Memory bandwidth (12-channel DDR5-6000, improved)
    channels = 12
    ddr5_rate = 6000e6  # 6000 MT/s (up from 4800)
    bytes_per_transfer = 8  # 64-bit per channel
    peak_bandwidth = channels * ddr5_rate * bytes_per_transfer  # 576 GB/s

    # Power and energy
    tdp = 500.0  # Watts (higher for 192 cores)
    idle_power = 120.0  # Estimated
    dynamic_power = tdp - idle_power  # 380W
    energy_per_byte = 25e-12  # 25 pJ/byte (DDR5-6000)

    # ========================================================================
    # Scalar Fabric (Custom Datacenter)
    # ========================================================================
    scalar_fabric = ComputeFabric(
        fabric_type="scalar_alu",
        circuit_type="custom_datacenter",
        num_units=num_cores * 2,           # 192 cores × 2 ALUs = 384 ALUs
        ops_per_unit_per_clock={
            Precision.FP64: 2,
            Precision.FP32: 2,
        },
        core_frequency_hz=base_clock_hz,   # 2.5 GHz
        process_node_nm=3,                  # TSMC 3nm
        energy_per_flop_fp32=get_base_alu_energy(3, 'custom_datacenter'),  # 1.2 × 2.75 = 3.3 pJ
        energy_scaling={
            Precision.FP64: 2.0,
            Precision.FP32: 1.0,
        }
    )

    # ========================================================================
    # AVX-512 SIMD Fabric (Native, not double-pumped)
    # ========================================================================
    avx512_fabric = ComputeFabric(
        fabric_type="avx512_native",
        circuit_type="simd_packed",
        num_units=num_cores,
        ops_per_unit_per_clock={
            Precision.FP64: 8,   # Native 512-bit
            Precision.FP32: 16,  # Native 512-bit
            Precision.FP16: 32,
            Precision.INT8: 64,
        },
        core_frequency_hz=base_clock_hz,
        process_node_nm=3,
        energy_per_flop_fp32=get_base_alu_energy(3, 'simd_packed'),  # 1.2 × 0.90 = 1.08 pJ
        energy_scaling={
            Precision.FP64: 2.0,
            Precision.FP32: 1.0,
            Precision.FP16: 0.5,
            Precision.INT8: 0.125,
        }
    )

    # Calculate peaks
    scalar_fp64_peak = scalar_fabric.get_peak_ops_per_sec(Precision.FP64)
    avx512_fp32_peak = avx512_fabric.get_peak_ops_per_sec(Precision.FP32)
    avx512_fp16_peak = avx512_fabric.get_peak_ops_per_sec(Precision.FP16)
    avx512_int8_peak = avx512_fabric.get_peak_ops_per_sec(Precision.INT8)

    return HardwareResourceModel(
        name="AMD-EPYC-Turin-Zen5",
        hardware_type=HardwareType.CPU,

        # NEW: Compute fabrics
        compute_fabrics=[scalar_fabric, avx512_fabric],

        # Legacy fields
        compute_units=num_cores,  # 192 cores
        threads_per_unit=2,  # SMT (2 threads per core)
        warps_per_unit=1,  # No warp concept in CPUs
        warp_size=8,  # SIMD width for FP32

        precision_profiles={
            Precision.FP64: PrecisionProfile(
                precision=Precision.FP64,
                peak_ops_per_sec=scalar_fp64_peak,  # Scalar (half of FP32)
                tensor_core_supported=False,
                relative_speedup=0.5,
                bytes_per_element=8,
            ),
            Precision.FP32: PrecisionProfile(
                precision=Precision.FP32,
                peak_ops_per_sec=avx512_fp32_peak,  # AVX-512
                tensor_core_supported=False,
                relative_speedup=1.0,
                bytes_per_element=4,
            ),
            Precision.FP16: PrecisionProfile(
                precision=Precision.FP16,
                peak_ops_per_sec=avx512_fp16_peak,  # AVX-512
                tensor_core_supported=False,
                relative_speedup=2.0,
                bytes_per_element=2,
            ),
            Precision.INT8: PrecisionProfile(
                precision=Precision.INT8,
                peak_ops_per_sec=avx512_int8_peak,  # AVX-512
                tensor_core_supported=False,  # No AMX-like accelerator yet
                relative_speedup=4.0,
                bytes_per_element=1,
                accumulator_precision=Precision.INT32,
            ),
        },
        default_precision=Precision.FP32,

        peak_bandwidth=peak_bandwidth,  # 576 GB/s (12-channel DDR5-6000)
        l1_cache_per_unit=48 * 1024,  # 48 KB L1D per core (increased from 32 KB)
        l2_cache_total=192 * 1024 * 1024,  # 192 MB total L2 (1 MB × 192 cores)
        main_memory=512 * 1024**3,  # 512 GB (typical server config, up to 6TB)
        energy_per_flop_fp32=avx512_fabric.energy_per_flop_fp32,  # 1.08 pJ
        energy_per_byte=energy_per_byte,
        energy_scaling={
            Precision.FP64: 2.0,
            Precision.FP32: 1.0,
            Precision.FP16: 0.5,
            Precision.INT8: 0.125,
        },  # 26 pJ/byte
        min_occupancy=0.5,
        max_concurrent_kernels=num_cores,  # One kernel per core
        wave_quantization=1,
    )


