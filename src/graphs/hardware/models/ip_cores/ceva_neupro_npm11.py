"""
Ceva Neupro Npm11 Resource Model hardware resource model.

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


def ceva_neupro_npm11_resource_model() -> HardwareResourceModel:
    """
    CEVA NeuPro-M NPM11 Neural Processing IP Core.

    ARCHITECTURE:
    - Licensable NPU IP core for edge AI acceleration
    - Single NeuPro-M engine configuration
    - Heterogeneous: Tensor + Vector + Scalar units
    - Designed for SoC integration (mobile, automotive, IoT)

    PERFORMANCE:
    - Peak: 20 TOPS INT8 @ 1.25 GHz
    - Scalable architecture (2-256 TOPS range)
    - Optimized for CNNs, RNNs, and transformer models

    PRECISION SUPPORT:
    - INT8: Native, primary mode
    - INT16: Native support
    - FP16: Supported
    - INT4: Supported (2× INT8 throughput)

    MEMORY:
    - Configurable local memory (SRAM)
    - External DRAM access via SoC interconnect
    - Typical: 2-4 MB local SRAM
    - Bandwidth depends on SoC integration

    POWER:
    - 2W typical for NPM11 @ 1.0 GHz
    - ~10 TOPS/W efficiency
    - Highly power-efficient for edge AI

    USE CASES:
    - Mobile devices (always-on AI)
    - Automotive ADAS (sensor fusion)
    - IoT devices (edge AI)
    - Smart cameras and drones

    CALIBRATION STATUS:
    ⚠ ESTIMATED - Based on CEVA published specs
    - Need empirical benchmarking on actual silicon implementations
    - Performance varies based on SoC integration and memory configuration

    REFERENCES:
    - CEVA NeuPro-M Product Brief
    - NPM11 configuration specifications
    - CEVA press releases (2021-2024)
    """
    # Model as 64 equivalent processing elements
    # 20 TOPS @ 1.25 GHz → 16e12 ops/sec → 250 ops/cycle → 64 units × 312.5 ops/unit/cycle
    num_npu_units = 64
    sustained_clock_hz = 1.0e9  # 1.0 GHz sustained

    # ========================================================================
    # Multi-Fabric Architecture (CEVA NeuPro-M NPM11 - Neural Processing IP)
    # ========================================================================
    # Tensor Fabric (INT8/INT16/INT4 tensor operations)
    # ========================================================================
    tensor_fabric = ComputeFabric(
        fabric_type="neupro_tensor",
        circuit_type="tensor_core",      # Tensor operations
        num_units=64,                    # 64 tensor units
        ops_per_unit_per_clock={
            Precision.INT8: 312,         # 312 INT8 MACs/cycle/unit
            Precision.INT16: 156,        # 156 INT16 MACs/cycle/unit
            Precision.INT4: 624,         # 624 INT4 MACs/cycle/unit (2× INT8)
        },
        core_frequency_hz=sustained_clock_hz,  # 1.0 GHz
        process_node_nm=16,              # 16nm (typical for edge AI IP)
        energy_per_flop_fp32=get_base_alu_energy(16, 'tensor_core'),  # 2.295 pJ
        energy_scaling={
            Precision.INT8: 0.12,        # INT8 is very efficient
            Precision.INT16: 0.20,       # INT16
            Precision.INT4: 0.06,        # INT4 (2× efficiency)
        }
    )

    # ========================================================================
    # Vector Fabric (FP16 vector operations)
    # ========================================================================
    vector_fabric = ComputeFabric(
        fabric_type="neupro_vector",
        circuit_type="simd_packed",      # Vector operations
        num_units=64,                    # 64 vector units
        ops_per_unit_per_clock={
            Precision.FP16: 156,         # 156 FP16 MACs/cycle/unit
        },
        core_frequency_hz=sustained_clock_hz,  # 1.0 GHz
        process_node_nm=16,              # 16nm
        energy_per_flop_fp32=get_base_alu_energy(16, 'simd_packed'),  # 2.43 pJ
        energy_scaling={
            Precision.FP16: 0.40,        # FP16
        }
    )

    # Tensor INT8: 64 units × 312 ops/cycle × 1.0 GHz = 19.97 TOPS ≈ 20 TOPS ✓
    # Vector FP16: 64 units × 156 ops/cycle × 1.0 GHz = 9.98 TFLOPS ≈ 10 TFLOPS ✓

    clock = ClockDomain(
        base_clock_hz=800e6,       # 800 MHz minimum
        max_boost_clock_hz=1.25e9, # 1.25 GHz peak
        sustained_clock_hz=1.0e9,  # 1.0 GHz sustained
        dvfs_enabled=True,
    )

    compute_resource = ComputeResource(
        resource_type="CEVA-NeuPro-M-NPM11",
        num_units=num_npu_units,
        ops_per_unit_per_clock={
            Precision.INT8: 312,   # 312 INT8 MACs/cycle/unit
            Precision.INT16: 156,  # 156 INT16 MACs/cycle/unit
            Precision.FP16: 156,   # 156 FP16 MACs/cycle/unit
            Precision.INT4: 624,   # 624 INT4 MACs/cycle/unit (2× INT8)
        },
        clock_domain=clock,
    )

    # Peak INT8: 64 units × 312 ops/cycle × 1.25 GHz = 25 TOPS (conservative vs 20 TOPS spec)
    # This accounts for realistic efficiency

    thermal_2w = ThermalOperatingPoint(
        name="2W-edge-ai",
        tdp_watts=2.0,
        cooling_solution="passive-mobile",
        performance_specs={
            Precision.INT8: PerformanceCharacteristics(
                precision=Precision.INT8,
                compute_resource=compute_resource,
                instruction_efficiency=0.90,  # High NPU efficiency
                memory_bottleneck_factor=0.75,  # Depends on SoC integration
                efficiency_factor=0.70,  # 70% effective utilization
                tile_utilization=0.85,  # Good tensor utilization
                native_acceleration=True,
            ),
            Precision.INT16: PerformanceCharacteristics(
                precision=Precision.INT16,
                compute_resource=compute_resource,
                instruction_efficiency=0.88,
                memory_bottleneck_factor=0.70,
                efficiency_factor=0.65,
                tile_utilization=0.80,
                native_acceleration=True,
            ),
            Precision.FP16: PerformanceCharacteristics(
                precision=Precision.FP16,
                compute_resource=compute_resource,
                instruction_efficiency=0.85,
                memory_bottleneck_factor=0.70,
                efficiency_factor=0.60,
                tile_utilization=0.80,
                native_acceleration=True,
            ),
            Precision.INT4: PerformanceCharacteristics(
                precision=Precision.INT4,
                compute_resource=compute_resource,
                instruction_efficiency=0.92,
                memory_bottleneck_factor=0.80,
                efficiency_factor=0.75,
                tile_utilization=0.85,
                native_acceleration=True,
            ),
        }
    )

    return HardwareResourceModel(
        name="CEVA-NeuPro-M-NPM11",
        hardware_type=HardwareType.DSP,

        # NEW: Multi-fabric architecture (Tensor + Vector units)
        compute_fabrics=[tensor_fabric, vector_fabric],

        compute_units=num_npu_units,
        threads_per_unit=4,
        warps_per_unit=1,
        warp_size=32,

        thermal_operating_points={
            "2W": thermal_2w,
        },
        default_thermal_profile="2W",

        precision_profiles={
            Precision.INT8: PrecisionProfile(
                precision=Precision.INT8,
                peak_ops_per_sec=20e12,  # 20 TOPS INT8
                tensor_core_supported=True,
                relative_speedup=1.0,
                bytes_per_element=1,
            ),
            Precision.INT16: PrecisionProfile(
                precision=Precision.INT16,
                peak_ops_per_sec=10e12,  # 10 TOPS INT16
                tensor_core_supported=True,
                relative_speedup=0.5,
                bytes_per_element=2,
            ),
            Precision.FP16: PrecisionProfile(
                precision=Precision.FP16,
                peak_ops_per_sec=10e12,  # 10 TFLOPS FP16
                tensor_core_supported=True,
                relative_speedup=0.5,
                bytes_per_element=2,
            ),
        },
        default_precision=Precision.INT8,

        # Memory (depends on SoC integration, typical values)
        peak_bandwidth=50e9,  # 50 GB/s (typical for mobile SoC integration)
        l1_cache_per_unit=64 * 1024,  # 64 KB per unit
        l2_cache_total=2 * 1024 * 1024,  # 2 MB shared cache
        main_memory=8 * 1024**3,  # Up to 8 GB

        # Energy (edge-optimized)
        energy_per_flop_fp32=tensor_fabric.energy_per_flop_fp32,  # 2.295 pJ (16nm, tensor_core)
        energy_per_byte=12e-12,  # 12 pJ/byte
        energy_scaling={
            Precision.INT8: 0.12,
            Precision.INT16: 0.20,
            Precision.FP16: 0.40,
            Precision.INT4: 0.06,
        },

        min_occupancy=0.70,
        max_concurrent_kernels=8,
        wave_quantization=4,
    )


# ============================================================================
# Cadence Tensilica Vision DSP IP
# ============================================================================

