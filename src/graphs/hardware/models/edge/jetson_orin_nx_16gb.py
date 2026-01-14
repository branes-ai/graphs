"""
Jetson Orin NX 16GB Resource Model hardware resource model.

MEMORY: 16 GB LPDDR5

The Jetson Orin NX sits between the Orin Nano and Orin AGX in NVIDIA's
edge AI lineup. It provides double the compute resources of the Nano
while maintaining a compact form factor.

References:
- NVIDIA Jetson Orin NX Series Data Sheet (DS-10712-001_v1.6)
- https://developer.nvidia.com/embedded/jetson-orin-nx
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
    BOMCostProfile,
)


def jetson_orin_nx_16gb_resource_model() -> HardwareResourceModel:
    """
    NVIDIA Jetson Orin NX 16GB with realistic DVFS-aware power modeling.

    MEMORY: 16 GB LPDDR5 @ 3200 MHz

    Configuration: NX variant (2048 CUDA cores, 16 Ampere SMs, 64 Tensor Cores)

    CRITICAL REALITY CHECK - Performance Specifications:
    - Marketing claim (Super): 157 TOPS INT8 (sparse, all engines)
    - Marketing claim (original): 100 TOPS INT8 (sparse, all engines)
    - Dense networks GPU only: ~5.3 TOPS INT8 (16 SMs x 512 ops/SM/clock x 650 MHz)
    - Customer empirical data: 3-6% of peak at typical power budgets
    - Root cause: Same as other Jetsons - DVFS thermal throttling + memory bottlenecks

    Power Profiles with Realistic DVFS Behavior:
    ============================================

    10W Mode (Low Power - Battery-Powered Applications):
    - Base clock: 204 MHz (minimum)
    - Boost clock: 918 MHz (datasheet)
    - Sustained clock: 350 MHz (empirical under thermal load)
    - Thermal throttle factor: 38% (severe throttling)
    - Effective INT8: ~2.0 TOPS (5% of dense peak)
    - Use case: Battery-powered robots, drones

    15W Mode (Moderate Power - Compact Edge Devices):
    - Sustained clock: 500 MHz (54% of boost)
    - Effective INT8: ~4 TOPS (9% of dense peak)
    - Use case: Compact edge AI devices with passive cooling

    25W Mode (Balanced - Typical Edge AI Deployment):
    - Sustained clock: 650 MHz (71% of boost)
    - Effective INT8: ~8 TOPS (18% of dense peak)
    - Use case: Standard edge AI deployment with active cooling

    40W Mode (MAXN_SUPER - Maximum Performance):
    - Sustained clock: 850 MHz (93% of boost)
    - Effective INT8: ~15 TOPS (35% of dense peak)
    - Use case: Peak performance with robust cooling solution

    References:
    - Jetson Orin NX Super Specs: 157 TOPS, 102 GB/s, 10-40W
    - Jetson Orin NX 16GB Specs: 100 TOPS, 102 GB/s, 10-25W
    - NVIDIA Official: 1024 CUDA cores, 32 Tensor Cores
    - https://connecttech.com/orin-module-comparison/
    """
    # Physical hardware specs (NX has same SMs as Nano, half of AGX)
    # Per NVIDIA specs: Orin NX 16GB has 1024 CUDA cores, 32 Tensor Cores
    # Reference: https://connecttech.com/orin-module-comparison/
    num_sms = 8  # 1024 CUDA cores / 128 cores per SM
    cuda_cores_per_sm = 128  # Ampere SM: 128 CUDA cores
    tensor_cores_per_sm = 4  # NX has 4 TCs/SM like other Orins
    total_tensor_cores = num_sms * tensor_cores_per_sm  # 32 TCs
    # ops are flops, so MACs count as 2 ops
    fp32_ops_per_sm_per_clock = 256  # 128 CUDA cores x 1 FMA = 128 MACs/clock/SM
    # Tensor Core throughput
    int8_ops_per_sm_per_clock = 1024  # 4 TCs/SM x 128 MACs/TC = 512 MACs/clock/SM
    fp16_ops_per_sm_per_clock = 512  # 4 TCs/SM x 64 MACs/TC = 256 MACs/clock/SM

    # Baseline frequency (sustained at 25W with active cooling)
    baseline_freq_hz = 650e6  # 650 MHz sustained (typical)

    # | Precision          | Ops/clock/SM |         Notes                       |
    # +--------------------+--------------+-------------------------------------+
    # | FP32               |      128     | 1 op/clock per CUDA core            |
    # | FP16               |      256     | 2x FP32 rate (packed)               |
    # | FP64               |        2     | Severely limited on Jetson/consumer |
    # | INT32              |       64     | Shares with half the FP32 units     |
    # | INT8               |      256     | via dp4a instruction                |
    # | Tensor Core (INT8) |      512+    | 4x4x4 matrix ops per clock (64 MACs)|
    # | Tensor Core (FP16) |      512+    | Matrix ops only, layout constraints |

    # ========================================================================
    # Multi-Fabric Architecture (NVIDIA Ampere GPU)
    # ========================================================================
    # CUDA Core Fabric (standard ALU operations)
    # ========================================================================
    cuda_fabric = ComputeFabric(
        fabric_type="cuda_core",
        circuit_type="standard_cell",
        num_units=num_sms * cuda_cores_per_sm,  # 16 SMs x 128 cores/SM = 2,048 cores
        ops_per_unit_per_clock={
            Precision.FP64: 0.03125,  # 1/32 rate (emulated)
            Precision.FP32: 2,  # FMA = 2 ops/clock (mul + add)
            Precision.FP16: 2,  # FP16 emulated on CUDA cores
            Precision.INT8: 2,  # INT8 emulated on CUDA cores
        },
        core_frequency_hz=baseline_freq_hz,  # 650 MHz sustained (25W)
        process_node_nm=8,  # Samsung 8nm
        energy_per_flop_fp32=get_base_alu_energy(8, 'standard_cell'),  # 1.9 pJ
        energy_scaling={
            Precision.FP64: 2.0,  # Double precision = 2x energy
            Precision.FP32: 1.0,  # Baseline
            Precision.FP16: 0.5,  # Half precision (emulated on CUDA cores)
            Precision.INT8: 0.125,  # INT8 (emulated on CUDA cores)
        }
    )

    # ========================================================================
    # Tensor Core Fabric (15% more efficient for fused MAC+accumulate)
    # ========================================================================
    tensor_fabric = ComputeFabric(
        fabric_type="tensor_core",
        circuit_type="tensor_core",
        num_units=total_tensor_cores,  # 16 SMs x 4 TCs/SM = 64 TCs
        ops_per_unit_per_clock={
            Precision.FP16: 64,  # 64 FP16 ops/clock/TC (Ampere 2nd gen)
            Precision.INT8: 64,  # 64 INT8 ops/clock/TC
        },
        core_frequency_hz=baseline_freq_hz,  # 650 MHz sustained (25W)
        process_node_nm=8,
        energy_per_flop_fp32=get_base_alu_energy(8, 'tensor_core'),  # 1.62 pJ (15% better)
        energy_scaling={
            Precision.FP16: 0.5,  # Half precision
            Precision.INT8: 0.25,  # INT8
        }
    )

    # CUDA cores:
    # FP32: 128 cores x 1 ops/clock   = 128 ops/clock per SM
    # Total: 16 SMs x 128 ops/clock/SM = 2,048 ops/clock
    # At 918 MHz boost: 2,048 x 918e6 = 1.88 TFLOPS FP32 (peak)

    # INT8: 128 cores x 2 ops/clock   = 256 ops/clock per SM
    # Total: 16 SMs x 256 ops/clock/SM = 4,096 ops/clock

    # Tensor cores:
    # Tensor Core INT8: 4x4x4 matmul      = 64 MACs/clock per Tensor Core
    # Total: 64 TCs x 64 MACs/TC/clock    = 4,096 MACs/clock
    # At 918 MHz boost:     4,096 x 918e6 = 3.76 TOPS INT8 (peak)
    # At 650 MHz sustained: 4,096 x 650e6 = 2.66 TOPS INT8 (sustained)

    # ========================================================================
    # 10W MODE: Low power deployment (battery-powered devices)
    # ========================================================================
    clock_10w = ClockDomain(
        base_clock_hz=204e6,
        max_boost_clock_hz=918e6,
        sustained_clock_hz=350e6,  # 38% throttle
        dvfs_enabled=True,
    )

    compute_resource_10w = ComputeResource(
        resource_type="Ampere-SM-TensorCore",
        num_units=num_sms,
        ops_per_unit_per_clock={
            Precision.INT8: int8_ops_per_sm_per_clock,
            Precision.FP16: fp16_ops_per_sm_per_clock,
            Precision.FP32: fp32_ops_per_sm_per_clock,
        },
        clock_domain=clock_10w,
    )

    thermal_10w = ThermalOperatingPoint(
        name="10W-battery",
        tdp_watts=10.0,
        cooling_solution="passive-heatsink-small",
        performance_specs={
            Precision.INT8: PerformanceCharacteristics(
                precision=Precision.INT8,
                compute_resource=compute_resource_10w,
                instruction_efficiency=0.78,
                memory_bottleneck_factor=0.50,
                efficiency_factor=0.35,  # 35% of sustained (5% of peak)
                native_acceleration=True,
            ),
            Precision.FP16: PerformanceCharacteristics(
                precision=Precision.FP16,
                compute_resource=compute_resource_10w,
                efficiency_factor=0.60,
                native_acceleration=True,
            ),
            Precision.FP32: PerformanceCharacteristics(
                precision=Precision.FP32,
                compute_resource=compute_resource_10w,
                efficiency_factor=0.18,
                native_acceleration=True,
            ),
        }
    )

    # ========================================================================
    # 15W MODE: Moderate power (compact edge devices)
    # ========================================================================
    clock_15w = ClockDomain(
        base_clock_hz=306e6,
        max_boost_clock_hz=918e6,
        sustained_clock_hz=500e6,  # 54% of boost
        dvfs_enabled=True,
    )

    compute_resource_15w = ComputeResource(
        resource_type="Ampere-SM-TensorCore",
        num_units=num_sms,
        ops_per_unit_per_clock={
            Precision.INT8: int8_ops_per_sm_per_clock,
            Precision.FP16: fp16_ops_per_sm_per_clock,
            Precision.FP32: fp32_ops_per_sm_per_clock,
        },
        clock_domain=clock_15w,
    )

    thermal_15w = ThermalOperatingPoint(
        name="15W-compact",
        tdp_watts=15.0,
        cooling_solution="passive-heatsink",
        performance_specs={
            Precision.INT8: PerformanceCharacteristics(
                precision=Precision.INT8,
                compute_resource=compute_resource_15w,
                instruction_efficiency=0.82,
                memory_bottleneck_factor=0.55,
                efficiency_factor=0.42,  # 42% of sustained (9% of peak)
                native_acceleration=True,
            ),
            Precision.FP16: PerformanceCharacteristics(
                precision=Precision.FP16,
                compute_resource=compute_resource_15w,
                efficiency_factor=0.50,
                native_acceleration=True,
            ),
            Precision.FP32: PerformanceCharacteristics(
                precision=Precision.FP32,
                compute_resource=compute_resource_15w,
                efficiency_factor=0.25,
                native_acceleration=True,
            ),
        }
    )

    # ========================================================================
    # 25W MODE: Balanced configuration (typical edge deployment)
    # ========================================================================
    clock_25w = ClockDomain(
        base_clock_hz=306e6,
        max_boost_clock_hz=918e6,
        sustained_clock_hz=650e6,  # 71% of boost
        dvfs_enabled=True,
    )

    compute_resource_25w = ComputeResource(
        resource_type="Ampere-SM-TensorCore",
        num_units=num_sms,
        ops_per_unit_per_clock={
            Precision.INT8: int8_ops_per_sm_per_clock,
            Precision.FP16: fp16_ops_per_sm_per_clock,
            Precision.FP32: fp32_ops_per_sm_per_clock,
        },
        clock_domain=clock_25w,
    )

    thermal_25w = ThermalOperatingPoint(
        name="25W-standard",
        tdp_watts=25.0,
        cooling_solution="active-fan",
        performance_specs={
            Precision.INT8: PerformanceCharacteristics(
                precision=Precision.INT8,
                compute_resource=compute_resource_25w,
                instruction_efficiency=0.88,
                memory_bottleneck_factor=0.62,
                efficiency_factor=0.55,  # 55% of sustained (18% of peak)
                native_acceleration=True,
            ),
            Precision.FP16: PerformanceCharacteristics(
                precision=Precision.FP16,
                compute_resource=compute_resource_25w,
                efficiency_factor=0.55,
                native_acceleration=True,
            ),
            Precision.FP32: PerformanceCharacteristics(
                precision=Precision.FP32,
                compute_resource=compute_resource_25w,
                efficiency_factor=0.35,
                native_acceleration=True,
            ),
        }
    )

    # ========================================================================
    # 40W MODE: MAXN_SUPER (maximum performance)
    # ========================================================================
    clock_40w = ClockDomain(
        base_clock_hz=306e6,
        max_boost_clock_hz=918e6,
        sustained_clock_hz=850e6,  # 93% of boost
        dvfs_enabled=True,
    )

    compute_resource_40w = ComputeResource(
        resource_type="Ampere-SM-TensorCore",
        num_units=num_sms,
        ops_per_unit_per_clock={
            Precision.INT8: int8_ops_per_sm_per_clock,
            Precision.FP16: fp16_ops_per_sm_per_clock,
            Precision.FP32: fp32_ops_per_sm_per_clock,
        },
        clock_domain=clock_40w,
    )

    thermal_40w = ThermalOperatingPoint(
        name="40W-maxn-super",
        tdp_watts=40.0,
        cooling_solution="active-fan-high",
        performance_specs={
            Precision.INT8: PerformanceCharacteristics(
                precision=Precision.INT8,
                compute_resource=compute_resource_40w,
                instruction_efficiency=0.92,
                memory_bottleneck_factor=0.70,
                efficiency_factor=0.70,  # 70% of sustained (35% of peak)
                native_acceleration=True,
            ),
            Precision.FP16: PerformanceCharacteristics(
                precision=Precision.FP16,
                compute_resource=compute_resource_40w,
                efficiency_factor=0.65,
                native_acceleration=True,
            ),
            Precision.FP32: PerformanceCharacteristics(
                precision=Precision.FP32,
                compute_resource=compute_resource_40w,
                efficiency_factor=0.45,
                native_acceleration=True,
            ),
        }
    )

    # ========================================================================
    # BOM Cost Profile
    # ========================================================================
    bom_cost = BOMCostProfile(
        silicon_die_cost=180.0,
        package_cost=35.0,
        memory_cost=80.0,  # 16GB LPDDR5
        pcb_assembly_cost=15.0,
        thermal_solution_cost=5.0,
        other_costs=10.0,
        total_bom_cost=325.0,
        margin_multiplier=1.54,
        retail_price=499.0,  # Approximate street price
        volume_tier="10K+",
        process_node="8nm",
        year=2025,
        notes="Mid-range edge AI module. 16GB LPDDR5. Positioned between Nano and AGX. "
              "Popular for robotics and industrial applications requiring more memory than Nano.",
    )

    # ========================================================================
    # Hardware Resource Model
    # ========================================================================
    return HardwareResourceModel(
        name="Jetson-Orin-NX-16GB",
        hardware_type=HardwareType.GPU,

        # NEW: Multi-fabric architecture
        compute_fabrics=[cuda_fabric, tensor_fabric],

        compute_units=num_sms,
        threads_per_unit=64,
        warps_per_unit=2,
        warp_size=32,

        # Thermal operating points with DVFS modeling
        thermal_operating_points={
            "10W": thermal_10w,  # Battery-powered devices
            "15W": thermal_15w,  # Compact edge devices
            "25W": thermal_25w,  # Standard edge AI deployment
            "40W": thermal_40w,  # MAXN_SUPER maximum performance
        },
        default_thermal_profile="25W",  # Balanced default for edge AI

        # Legacy precision profiles (backward compatibility)
        # NX has 8 SMs (half of AGX's 16 SMs)
        precision_profiles={
            Precision.INT8: PrecisionProfile(
                precision=Precision.INT8,
                peak_ops_per_sec=2.662e12,  # 8 SMs x 512 ops/clock x 650 MHz (realistic peak)
                tensor_core_supported=True,
                bytes_per_element=1,
            ),
            Precision.FP16: PrecisionProfile(
                precision=Precision.FP16,
                peak_ops_per_sec=1.331e12,  # Half of INT8
                tensor_core_supported=True,
                bytes_per_element=2,
            ),
            Precision.FP32: PrecisionProfile(
                precision=Precision.FP32,
                peak_ops_per_sec=0.666e12,  # Standard CUDA cores
                tensor_core_supported=False,
                bytes_per_element=4,
            ),
        },
        default_precision=Precision.INT8,

        peak_bandwidth=102e9,  # 102 GB/s (LPDDR5 @ 3200 MHz)
        l1_cache_per_unit=128 * 1024,  # 128 KB per SM
        l2_cache_total=2 * 1024 * 1024,  # 2 MB (same as Nano, half of AGX)
        main_memory=16 * 1024**3,  # 16 GB
        energy_per_flop_fp32=cuda_fabric.energy_per_flop_fp32,  # 1.9 pJ (8nm Samsung, CUDA cores)
        energy_per_byte=18e-12,  # 18 pJ/byte
        min_occupancy=0.3,
        max_concurrent_kernels=8,  # More than Nano
        wave_quantization=2,
        bom_cost_profile=bom_cost,
    )
