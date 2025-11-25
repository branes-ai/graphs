"""
Jetson Orin Nano 8GB Resource Model hardware resource model.

MEMORY: 8 GB LPDDR5

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
    BOMCostProfile,
)


def jetson_orin_nano_8gb_resource_model() -> HardwareResourceModel:
    """
    NVIDIA Jetson Orin Nano 8GB with realistic DVFS-aware power modeling.

    MEMORY: 8 GB LPDDR5

    Configuration: Nano variant (1024 CUDA cores, 8 Ampere SMs, 32 Tensor Cores)

    CRITICAL REALITY CHECK - Performance Specifications:
    - Marketing claim (Super): 67 TOPS INT8 (sparse, all engines)
    - Marketing claim (original): 40 TOPS INT8 (sparse, all engines)
    - Dense networks GPU only: ~2.66 TOPS INT8 (8 SMs x 512 ops/SM/clock x 650 MHz)
    - Customer empirical data: 2-4% of peak at typical power budgets
    - Root cause: Same as AGX - severe DVFS thermal throttling + memory bottlenecks

    Power Profiles with Realistic DVFS Behavior:
    ============================================

    7W Mode (Low Power - Battery-Powered Drones/Robots):
    - Base clock: 204 MHz (minimum)
    - Boost clock: 918 MHz (datasheet)
    - Sustained clock: 300 MHz (empirical under thermal load)
    - Thermal throttle factor: 33% (severe throttling!)
    - Effective INT8: ~1.5 TOPS (7% of 21 TOPS GPU dense peak)
    - Use case: Battery-powered drones, small robots (avoid thermal shutdown)

    15W Mode (Balanced - Typical Edge AI Deployment):
    - Sustained clock: 500 MHz (54% of boost)
    - Effective INT8: ~4 TOPS (19% of 21 TOPS GPU dense peak)
    - Use case: Edge AI devices with passive cooling

    References:
    - Jetson Orin Nano Super Specs: 67 TOPS, 102 GB/s, 15W max
    - Jetson Orin Nano 8GB Specs: 40 TOPS, 68 GB/s, 7-15W
    - TechPowerUp GPU Database: 1024 CUDA cores, 32 Tensor cores
    """
    # Physical hardware specs (Nano has half the SMs of AGX)
    num_sms = 8  # 1024 CUDA cores / 128 cores per SM
    cuda_cores_per_sm = 128  # Ampere SM: 128 CUDA cores
    tensor_cores_per_sm = 4  # Nano has 4 TCs/SM like AGX
    total_tensor_cores = num_sms * tensor_cores_per_sm  # 32 TCs
    # ops are flops, so MACs count as 2 ops
    fp32_ops_per_sm_per_clock =  256  # 128 CUDA cores × 1 FMA = 128 MACs/clock/SM
    # Tensor Core throughput
    int8_ops_per_sm_per_clock = 1024  # 4 TCs/SM × 128 MACs/TC = 512 MACs/clock/SM
    fp16_ops_per_sm_per_clock =  512  # 4 TCs/SM ×  64 MACs/TC = 256 MACs/clock/SM

    # Baseline frequency (sustained at 15W with active cooling)
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
        num_units=num_sms * cuda_cores_per_sm,  # 8 SMs × 128 cores/SM = 1,024 cores
        ops_per_unit_per_clock={
            Precision.FP64: 0.03125,  # 1/32 rate (emulated)
            Precision.FP32: 2,         # FMA = 2 ops/clock (mul + add)
            Precision.FP16: 2,         # FP16 emulated on CUDA cores
            Precision.INT8: 2,         # INT8 emulated on CUDA cores
        },
        core_frequency_hz=baseline_freq_hz,  # 650 MHz sustained (15W)
        process_node_nm=8,             # Samsung 8nm
        energy_per_flop_fp32=get_base_alu_energy(8, 'standard_cell'),  # 1.9 pJ
        energy_scaling={
            Precision.FP64: 2.0,       # Double precision = 2× energy
            Precision.FP32: 1.0,       # Baseline
            Precision.FP16: 0.5,       # Half precision (emulated on CUDA cores)
            Precision.INT8: 0.125,     # INT8 (emulated on CUDA cores)
        }
    )

    # ========================================================================
    # Tensor Core Fabric (15% more efficient for fused MAC+accumulate)
    # ========================================================================
    tensor_fabric = ComputeFabric(
        fabric_type="tensor_core",
        circuit_type="tensor_core",
        num_units=total_tensor_cores,  #  8 SMs × 4 TCs/SM = 32 TCs
        ops_per_unit_per_clock={
            Precision.FP16: 64,        # 64 FP16 ops/clock/TC (Ampere 2nd gen)
            Precision.INT8: 64,        # 64 INT8 ops/clock/TC
        },
        core_frequency_hz=baseline_freq_hz,  # 650 MHz sustained (15W)
        process_node_nm=8,
        energy_per_flop_fp32=get_base_alu_energy(8, 'tensor_core'),  # 1.62 pJ (15% better)
        energy_scaling={
            Precision.FP16: 0.5,       # Half precision
            Precision.INT8: 0.25,      # INT8
        }
    )

    # CUDA cores:
    # FP32: 128 cores × 1 ops/clock   = 128 ops/clock per SM
    # Total: 8 SMs × 128 ops/clock/SM = 1,024 ops/clock
    # At 918 MHz boost: 1,024 × 918e6 = 1.88 TFLOPS FP32 (peak)
    
    # INT8: 128 cores × 2 ops/clock   = 256 ops/clock per SM
    # Total: 8 SMs × 256 ops/clock/SM = 2,048 ops/clock

    # Peak INT8:     2,048 × 1300 MHz = 2.662 TOPS     <--- this is AGX
    # Boost 918 MHz: 2,048 ×  918 MHz = 3.76 TOPS INT8 (peak)
    # Nominal INT8:  2,048 ×  650 MHz = 1.331 TOPS
    # Sustained INT8:2,048 ×  300 MHz = 0.614 TOPS
    # Effective: 1.228 × 0.40 = 0.98 TOPS (4.7% of 21 TOPS dense peak)

    # Tensor cores: 
    # Tensor Core INT8: 4×4×4 matmul      = 64 MACs/clock per Tensor Core
    # Total: 32 TCs × 64 MACs/TC/clock    = 2,048 MACs/clock
    # At 918 MHz boost:     2,048 × 918e6 = 1.88 TOPS INT8 (peak)
    # At 650 MHz sustained: 2,048 × 650e6 = 1.33 TOPS INT8 (sustained)

    # ========================================================================
    # 7W MODE: Low power deployment (battery-powered devices)
    # ========================================================================
    clock_7w = ClockDomain(
        base_clock_hz=204e6,
        max_boost_clock_hz=918e6,
        sustained_clock_hz=300e6,  # 33% throttle
        dvfs_enabled=True,
    )

    compute_resource_7w = ComputeResource(
        resource_type="Ampere-SM-TensorCore",
        num_units=num_sms,
        ops_per_unit_per_clock={
            Precision.INT8: int8_ops_per_sm_per_clock,
            Precision.FP16: fp16_ops_per_sm_per_clock,
            Precision.FP32: fp32_ops_per_sm_per_clock,
        },
        clock_domain=clock_7w,
    )

    thermal_7w = ThermalOperatingPoint(
        name="7W-battery",
        tdp_watts=7.0,
        cooling_solution="passive-heatsink-small",
        performance_specs={
            Precision.INT8: PerformanceCharacteristics(
                precision=Precision.INT8,
                compute_resource=compute_resource_7w,
                instruction_efficiency=0.80,
                memory_bottleneck_factor=0.55,
                efficiency_factor=0.40,  # 40% of sustained (4% of peak!)
                native_acceleration=True,
            ),
            Precision.FP16: PerformanceCharacteristics(
                precision=Precision.FP16,
                compute_resource=compute_resource_7w,
                efficiency_factor=0.65,
                native_acceleration=True,
            ),
            Precision.FP32: PerformanceCharacteristics(
                precision=Precision.FP32,
                compute_resource=compute_resource_7w,
                efficiency_factor=0.20,
                native_acceleration=True,
            ),
        }
    )

    # ========================================================================
    # 15W MODE: Balanced configuration (typical edge deployment)
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

    # Sustained INT8: 8 × 512 × 500 MHz = 2.0 TOPS
    # Effective: 2.0 × 0.50 = 1.0 TOPS (9.7% of 21 TOPS dense peak)

    thermal_15w = ThermalOperatingPoint(
        name="15W-standard",
        tdp_watts=15.0,
        cooling_solution="passive-heatsink",
        performance_specs={
            Precision.INT8: PerformanceCharacteristics(
                precision=Precision.INT8,
                compute_resource=compute_resource_15w,
                instruction_efficiency=0.85,
                memory_bottleneck_factor=0.60,
                efficiency_factor=0.50,  # 50% of sustained (10% of peak)
                native_acceleration=True,
            ),
            Precision.FP16: PerformanceCharacteristics(
                precision=Precision.FP16,
                compute_resource=compute_resource_15w,
                efficiency_factor=0.45,
                native_acceleration=True,
            ),
            Precision.FP32: PerformanceCharacteristics(
                precision=Precision.FP32,
                compute_resource=compute_resource_15w,
                efficiency_factor=0.30,
                native_acceleration=True,
            ),
        }
    )

    # ========================================================================
    # BOM Cost Profile
    # ========================================================================
    bom_cost = BOMCostProfile(
        silicon_die_cost=120.0,
        package_cost=25.0,
        memory_cost=40.0,
        pcb_assembly_cost=10.0,
        thermal_solution_cost=3.0,
        other_costs=7.0,
        total_bom_cost=205.0,
        margin_multiplier=1.46,
        retail_price=299.0,
        volume_tier="10K+",
        process_node="12nm",
        year=2025,
        notes="Entry-level edge AI module. 8GB LPDDR5. NVIDIA pricing strategy: lower margins for market penetration. Competes with embedded GPUs and TPUs.",
    )

    # ========================================================================
    # Hardware Resource Model
    # ========================================================================
    return HardwareResourceModel(
        name="Jetson-Orin-Nano-8GB",
        hardware_type=HardwareType.GPU,

        # NEW: Multi-fabric architecture
        compute_fabrics=[cuda_fabric, tensor_fabric],

        compute_units=num_sms,
        threads_per_unit=64,
        warps_per_unit=2,
        warp_size=32,

        # Thermal operating points with DVFS modeling
        thermal_operating_points={
            "7W": thermal_7w,   # Battery-powered devices
            "15W": thermal_15w,  # Standard edge AI deployment
        },
        default_thermal_profile="7W",  # Most realistic for embodied AI

        # Legacy precision profiles (backward compatibility)
        precision_profiles={
            Precision.INT8: PrecisionProfile(
                precision=Precision.INT8,
                peak_ops_per_sec=2.662e12,  # 8 SMs × 512 ops/clock × 650 MHz (realistic peak)
                tensor_core_supported=True,
                bytes_per_element=1,
            ),
        },
        default_precision=Precision.INT8,

        peak_bandwidth=68e9,  # 68 GB/s (original) or 102 GB/s (Super)
        l1_cache_per_unit=128 * 1024,
        l2_cache_total=2 * 1024 * 1024,  # 2 MB (half of AGX)
        main_memory=8 * 1024**3,  # 8 GB
        energy_per_flop_fp32=cuda_fabric.energy_per_flop_fp32,  # 1.9 pJ (8nm Samsung, CUDA cores)
        energy_per_byte=18e-12,
        min_occupancy=0.3,
        max_concurrent_kernels=4,  # Fewer than AGX
        wave_quantization=2,  # Smaller wave size
        bom_cost_profile=bom_cost,
    )


