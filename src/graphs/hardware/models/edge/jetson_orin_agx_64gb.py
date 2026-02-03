"""
Jetson Orin AGX 64GB Resource Model hardware resource model.

MEMORY: 64 GB LPDDR5

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


def jetson_orin_agx_64gb_resource_model() -> HardwareResourceModel:
    """
    NVIDIA Jetson Orin AGX 64GB with realistic DVFS-aware multi-power-profile modeling.

    MEMORY: 64 GB LPDDR5

    Configuration: AGX variant (2048 CUDA cores, 16 Ampere SMs, 64 Tensor Cores)

    CRITICAL REALITY CHECK - Performance Specifications:
    - Marketing claim: 275 TOPS INT8 (sparse networks, all engines: GPU+DLA+PVA)
    - Dense networks total: 138 TOPS INT8 (GPU + 2×DLA)
      - GPU only (dense): 5.3 TOPS INT8 @ 1.3 GHz ← Relevant for PyTorch workloads
        (64 Tensor Cores × 64 MACs/TC/clock × 1.3 GHz = 5.3 TOPS)
      - 2×DLA (dense): 52.5 TOPS INT8
    - GPU sparse: 10.6 TOPS INT8 (2:4 structured sparsity gives 2× speedup)
    - Customer empirical data: 2-4% of peak at typical power budgets
    - Root cause: Severe DVFS thermal throttling + memory bottlenecks

    Power Profiles with Realistic DVFS Behavior:
    ============================================

    15W Mode (Passive Cooling - What Customers Actually Deploy):
    - Base clock: 306 MHz (guaranteed minimum)
    - Boost clock: 1.02 GHz (datasheet spec, rarely sustained)
    - Sustained clock: 400 MHz (empirical under thermal load)
    - Thermal throttle factor: 39% (severe throttling!)
    - Peak INT8: 1.6 TOPS (4,096 MACs/clock × 400 MHz)
    - Effective INT8: ~0.3 TOPS (20% of peak due to memory bottlenecks)
    - Use case: Battery-powered robots, drones (must avoid thermal shutdown)

    30W Mode (Active Cooling - Better but Still Throttles):
    - Sustained clock: 650 MHz (64% of boost)
    - Peak INT8: 2.7 TOPS (4,096 MACs/clock × 650 MHz)
    - Effective INT8: ~1.1 TOPS (40% of peak)
    - Use case: Tethered robots with active cooling

    50W Mode (High Performance - Active Cooling Required):
    - Sustained clock: 900 MHz (87% of boost)
    - Peak INT8: 3.7 TOPS (4,096 MACs/clock × 900 MHz)
    - Effective INT8: ~2.2 TOPS (60% of peak)
    - Use case: Tethered systems with robust cooling

    MAXN Mode (Unconstrained - Experimental):
    - Maximum clocks, no power cap (can exceed 60W)
    - WARNING: Hardware throttling engaged when power exceeds TDP
    - Not recommended for prolonged heavy workloads
    - Use case: Benchmarking and short bursts only

    References:
    - Jetson Orin AGX Datasheet: NVIDIA Technical Brief
    - Empirical measurements: Customer lab data (2-4% of peak @ 15W)
    - DVFS behavior: Observed clock throttling under sustained load
    """
    # ========================================================================
    # CUDA Core Fabric (Standard Cell FP32 ALUs)
    # ========================================================================
    # Physical hardware specs (constant across power modes)
    # Official specs: 2048 CUDA cores total, Ampere architecture
    num_sms = 16                     # 2048 CUDA cores ÷ 128 cores/SM = 16 SMs
    cuda_cores_per_sm = 128          # Ampere architecture standard
    tensor_cores_per_sm = 4          # 64 Tensor cores total ÷ 16 SMs = 4 per SM
    total_tensor_cores = 64          # Total tensor cores across all SMs

    # Use 30W mode as baseline for fabric specs (most realistic deployment)
    baseline_freq_hz = 650e6  # 650 MHz sustained (30W mode)

    cuda_fabric = ComputeFabric(
        fabric_type="cuda_core",
        circuit_type="standard_cell",
        num_units=num_sms * cuda_cores_per_sm,  # 16 SMs × 128 CUDA cores/SM = 2,048 cores
        ops_per_unit_per_clock={
            Precision.FP64: 2,         # FMA: 2 ops/clock
            Precision.FP32: 2,         # FMA: 2 ops/clock
        },
        core_frequency_hz=baseline_freq_hz,  # 650 MHz sustained (30W)
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
        num_units=num_sms * tensor_cores_per_sm,  # 16 SMs × 4 Tensor Cores/SM = 64 TCs
        ops_per_unit_per_clock={
            Precision.FP16: 64,        # 64 FP16 ops/clock/TC (Ampere 2nd gen)
            Precision.INT8: 64,        # 64 INT8 ops/clock/TC
        },
        core_frequency_hz=baseline_freq_hz,  # 650 MHz sustained (30W)
        process_node_nm=8,
        energy_per_flop_fp32=get_base_alu_energy(8, 'tensor_core'),  # 1.62 pJ (15% better)
        energy_scaling={
            Precision.FP16: 0.5,       # Half precision
            Precision.INT8: 0.125,     # INT8
        }
    )

    # Tensor Core INT8: 4×4×4 matmul = 64 MACs/clock per Tensor Core
    # Total: 64 TCs × 64 MACs/TC/clock = 4,096 MACs/clock
    # At 1.3 GHz: 4,096 × 1.3e9 = 5.3 TOPS INT8
    int8_ops_per_sm_per_clock = 256  # 4 TCs/SM × 64 MACs/TC = 256 MACs/SM/clock
    fp32_ops_per_sm_per_clock = 256  # CUDA core capability: 128 cores × 2 ops (FMA)
    fp16_ops_per_sm_per_clock = 256  # Tensor Core FP16: 4 TCs/SM × 64 MACs/TC

    # ========================================================================
    # 15W MODE: Realistic deployment configuration (passive cooling)
    # ========================================================================
    clock_15w = ClockDomain(
        base_clock_hz=306e6,         # 306 MHz guaranteed minimum
        max_boost_clock_hz=1.02e9,   # 1.02 GHz datasheet boost
        sustained_clock_hz=400e6,    # 400 MHz empirical (39% throttle!)
        dvfs_enabled=True,
    )

    compute_resource_15w_int8 = ComputeResource(
        resource_type="Ampere-SM-TensorCore",
        num_units=num_sms,
        ops_per_unit_per_clock={
            Precision.INT8: int8_ops_per_sm_per_clock,
            Precision.FP16: fp16_ops_per_sm_per_clock,
            Precision.FP32: fp32_ops_per_sm_per_clock,
        },
        clock_domain=clock_15w,
    )

    # Peak INT8: 64 TCs × 64 MACs/TC/clock × 1.02 GHz = 4.2 TOPS (boost clock)
    # Sustained INT8: 64 TCs × 64 MACs/TC × 400 MHz = 1.6 TOPS (throttled)
    # Effective INT8: 1.6 TOPS × 0.20 empirical derate = 0.3 TOPS (memory-bound)

    thermal_15w = ThermalOperatingPoint(
        name="15W-passive",
        tdp_watts=15.0,
        cooling_solution="passive-heatsink",
        performance_specs={
            Precision.INT8: PerformanceCharacteristics(
                precision=Precision.INT8,
                compute_resource=compute_resource_15w_int8,
                instruction_efficiency=0.85,
                memory_bottleneck_factor=0.60,
                efficiency_factor=0.47,  # 47% of sustained (3% of peak!)
                native_acceleration=True,
            ),
            Precision.FP16: PerformanceCharacteristics(
                precision=Precision.FP16,
                compute_resource=compute_resource_15w_int8,
                efficiency_factor=0.40,  # Worse (more memory bound)
                native_acceleration=True,
            ),
            Precision.FP32: PerformanceCharacteristics(
                precision=Precision.FP32,
                compute_resource=compute_resource_15w_int8,
                efficiency_factor=0.25,  # Much worse
                native_acceleration=True,
            ),
        }
    )

    # ========================================================================
    # 30W MODE: Balanced configuration (active fan cooling)
    # ========================================================================
    clock_30w = ClockDomain(
        base_clock_hz=612e6,
        max_boost_clock_hz=1.15e9,
        sustained_clock_hz=650e6,    # 650 MHz sustained (57% throttle)
        dvfs_enabled=True,
    )

    compute_resource_30w = ComputeResource(
        resource_type="Ampere-SM-TensorCore",
        num_units=num_sms,
        ops_per_unit_per_clock={
            Precision.INT8: int8_ops_per_sm_per_clock,
            Precision.FP16: fp16_ops_per_sm_per_clock,
            Precision.FP32: fp32_ops_per_sm_per_clock,
        },
        clock_domain=clock_30w,
    )

    # Sustained INT8: 64 TCs × 64 MACs/TC × 650 MHz = 2.7 TOPS
    # Effective: 2.7 × 0.40 = 1.1 TOPS (40% efficiency, memory-bound)

    thermal_30w = ThermalOperatingPoint(
        name="30W-active",
        tdp_watts=30.0,
        cooling_solution="active-fan",
        performance_specs={
            Precision.INT8: PerformanceCharacteristics(
                precision=Precision.INT8,
                compute_resource=compute_resource_30w,
                efficiency_factor=0.60,  # Better (10% of peak)
                native_acceleration=True,
            ),
            Precision.FP16: PerformanceCharacteristics(
                precision=Precision.FP16,
                compute_resource=compute_resource_30w,
                efficiency_factor=0.50,
                native_acceleration=True,
            ),
            Precision.FP32: PerformanceCharacteristics(
                precision=Precision.FP32,
                compute_resource=compute_resource_30w,
                efficiency_factor=0.35,
                native_acceleration=True,
            ),
        }
    )

    # ========================================================================
    # 50W MODE: High performance (active cooling required)
    # ========================================================================
    # CALIBRATED 2026-02-03: Measured via Conv2D microbenchmarks on Orin AGX
    # GPU clock observed at 306-816 MHz range during benchmarks
    clock_50w = ClockDomain(
        base_clock_hz=828e6,
        max_boost_clock_hz=1.2e9,
        sustained_clock_hz=900e6,    # 900 MHz sustained (75% of boost)
        dvfs_enabled=True,
    )

    compute_resource_50w = ComputeResource(
        resource_type="Ampere-SM-TensorCore",
        num_units=num_sms,
        ops_per_unit_per_clock={
            Precision.INT8: int8_ops_per_sm_per_clock,
            Precision.FP16: fp16_ops_per_sm_per_clock,
            Precision.FP32: fp32_ops_per_sm_per_clock,
        },
        clock_domain=clock_50w,
    )

    # CALIBRATION DATA (2026-02-03, Conv2D microbenchmarks):
    # - FP32 Conv2D average: 968 GFLOPS (26% of theoretical peak)
    # - FP16 Conv2D average: 1012 GFLOPS (27% of theoretical peak)
    # - BF16 Conv2D average: 1057 GFLOPS (29% of theoretical peak)
    # - Depthwise convs: 3-80 GFLOPS (severely memory-bound)
    # - Standard 3x3 convs: 400-4000 GFLOPS (size-dependent)
    # - GEMM: 1592 GFLOPS FP32, 3406 GFLOPS FP16 (higher than Conv2D)
    # Note: INT8 requires TensorRT for native support, not available in PyTorch

    thermal_50w = ThermalOperatingPoint(
        name="50W-performance",
        tdp_watts=50.0,
        cooling_solution="active-fan-high",
        performance_specs={
            Precision.INT8: PerformanceCharacteristics(
                precision=Precision.INT8,
                compute_resource=compute_resource_50w,
                # INT8 estimated from FP16 scaling (TensorRT typically 1.5-2x FP16)
                efficiency_factor=0.40,  # Conservative estimate without TensorRT calibration
                native_acceleration=True,
            ),
            Precision.FP16: PerformanceCharacteristics(
                precision=Precision.FP16,
                compute_resource=compute_resource_50w,
                # CALIBRATED: 1012 GFLOPS / 3686 GFLOPS theoretical = 27%
                efficiency_factor=0.27,
                native_acceleration=True,
            ),
            Precision.FP32: PerformanceCharacteristics(
                precision=Precision.FP32,
                compute_resource=compute_resource_50w,
                # CALIBRATED: 968 GFLOPS / 3686 GFLOPS theoretical = 26%
                efficiency_factor=0.26,
                native_acceleration=True,
            ),
            Precision.BF16: PerformanceCharacteristics(
                precision=Precision.BF16,
                compute_resource=compute_resource_50w,
                # CALIBRATED: 1057 GFLOPS / 3686 GFLOPS theoretical = 29%
                efficiency_factor=0.29,
                native_acceleration=True,
            ),
        }
    )

    # ========================================================================
    # MAXN MODE: Unconstrained (experimental - hardware throttling engaged)
    # ========================================================================
    clock_maxn = ClockDomain(
        base_clock_hz=918e6,
        max_boost_clock_hz=1.3e9,
        sustained_clock_hz=1.0e9,    # 1.0 GHz sustained (77% of boost)
        dvfs_enabled=True,
    )

    compute_resource_maxn = ComputeResource(
        resource_type="Ampere-SM-TensorCore",
        num_units=num_sms,
        ops_per_unit_per_clock={
            Precision.INT8: int8_ops_per_sm_per_clock,
            Precision.FP16: fp16_ops_per_sm_per_clock,
            Precision.FP32: fp32_ops_per_sm_per_clock,
        },
        clock_domain=clock_maxn,
    )

    # MAXN is unconstrained but throttles under sustained load
    # Peak INT8: 64 TCs × 64 MACs/TC × 1.0 GHz = 4.1 TOPS
    # Effective: varies due to thermal throttling

    thermal_maxn = ThermalOperatingPoint(
        name="MAXN-unconstrained",
        tdp_watts=60.0,  # Can exceed this, triggering throttling
        cooling_solution="active-fan-max",
        performance_specs={
            Precision.INT8: PerformanceCharacteristics(
                precision=Precision.INT8,
                compute_resource=compute_resource_maxn,
                efficiency_factor=0.75,  # Best case before throttling
                native_acceleration=True,
            ),
            Precision.FP16: PerformanceCharacteristics(
                precision=Precision.FP16,
                compute_resource=compute_resource_maxn,
                efficiency_factor=0.65,
                native_acceleration=True,
            ),
            Precision.FP32: PerformanceCharacteristics(
                precision=Precision.FP32,
                compute_resource=compute_resource_maxn,
                efficiency_factor=0.50,
                native_acceleration=True,
            ),
        }
    )

    # ========================================================================
    # BOM Cost Profile
    # ========================================================================
    bom_cost = BOMCostProfile(
        silicon_die_cost=380.0,
        package_cost=60.0,
        memory_cost=160.0,
        pcb_assembly_cost=35.0,
        thermal_solution_cost=15.0,
        other_costs=20.0,
        total_bom_cost=670.0,
        margin_multiplier=1.34,
        retail_price=899.0,
        volume_tier="10K+",
        process_node="8nm",
        year=2025,
        notes="High-end edge AI module. 64GB LPDDR5. NVIDIA pricing: competitive margin for robotics/automotive. Competes with datacenter inference cards.",
    )

    # ========================================================================
    # Legacy Precision Profiles (calculated from fabrics for backward compatibility)
    # ========================================================================
    # Calculate peak ops using fabrics (use 30W sustained as baseline)
    cuda_fp64_peak = cuda_fabric.get_peak_ops_per_sec(Precision.FP64)
    cuda_fp32_peak = cuda_fabric.get_peak_ops_per_sec(Precision.FP32)
    tensor_fp16_peak = tensor_fabric.get_peak_ops_per_sec(Precision.FP16)
    tensor_int8_peak = tensor_fabric.get_peak_ops_per_sec(Precision.INT8)

    # ========================================================================
    # Hardware Resource Model (uses NEW thermal operating points + compute fabrics)
    # ========================================================================
    return HardwareResourceModel(
        name="Jetson-Orin-AGX-64GB",
        hardware_type=HardwareType.GPU,

        # NEW: Compute fabrics
        compute_fabrics=[cuda_fabric, tensor_fabric],

        # Legacy fields (for backward compatibility)
        compute_units=num_sms,
        threads_per_unit=64,
        warps_per_unit=2,
        warp_size=32,

        # GPU Microarchitecture (Ampere)
        cuda_cores_per_sm=cuda_cores_per_sm,
        tensor_cores_per_sm=tensor_cores_per_sm,
        ops_per_clock_per_core=2.0,  # FMA: 2 ops/clock for FP32
        sm_boost_clock_hz=1.3e9,     # 1.3 GHz boost (MAXN mode)
        sm_sustained_clock_hz=650e6,  # 650 MHz sustained (30W mode typical)

        # NEW: Thermal operating points with DVFS modeling
        # Power modes per NVIDIA nvpmodel: 15W, 30W, 50W, MAXN
        thermal_operating_points={
            "15W": thermal_15w,   # Battery-powered deployment (passive cooling)
            "30W": thermal_30w,   # Balanced (active cooling, typical deployment)
            "50W": thermal_50w,   # High performance (robust active cooling)
            "MAXN": thermal_maxn, # Unconstrained (experimental, throttles under load)
        },
        default_thermal_profile="30W",  # Balanced default for edge AI with active cooling

        # Legacy precision profiles (calculated from fabrics)
        precision_profiles={
            Precision.FP64: PrecisionProfile(
                precision=Precision.FP64,
                peak_ops_per_sec=cuda_fp64_peak,  # ~2.7 TFLOPS @ 650 MHz
                tensor_core_supported=False,
                relative_speedup=1.0,
                bytes_per_element=8,
            ),
            Precision.FP32: PrecisionProfile(
                precision=Precision.FP32,
                peak_ops_per_sec=cuda_fp32_peak,  # ~2.7 TFLOPS @ 650 MHz (CUDA cores)
                tensor_core_supported=False,
                relative_speedup=1.0,
                bytes_per_element=4,
            ),
            Precision.FP16: PrecisionProfile(
                precision=Precision.FP16,
                peak_ops_per_sec=tensor_fp16_peak,  # ~2.7 TOPS @ 650 MHz (Tensor Cores)
                tensor_core_supported=True,
                relative_speedup=1.0,
                bytes_per_element=2,
                accumulator_precision=Precision.FP32,
            ),
            Precision.INT8: PrecisionProfile(
                precision=Precision.INT8,
                peak_ops_per_sec=tensor_int8_peak,  # ~2.7 TOPS @ 650 MHz (Tensor Cores)
                tensor_core_supported=True,
                relative_speedup=1.0,
                bytes_per_element=1,
                accumulator_precision=Precision.INT32,
            ),
        },
        default_precision=Precision.INT8,

        peak_bandwidth=204.8e9,       # 204.8 GB/s LPDDR5
        l1_cache_per_unit=128 * 1024,  # 128 KB per SM
        l2_cache_total=4 * 1024 * 1024,  # 4 MB
        main_memory=64 * 1024**3,     # 64 GB LPDDR5

        # Legacy energy (use CUDA fabric as baseline)
        energy_per_flop_fp32=cuda_fabric.energy_per_flop_fp32,  # 1.9 pJ
        energy_per_byte=15e-12,       # 15 pJ/byte (LPDDR5)
        energy_scaling={
            Precision.FP64: 2.0,
            Precision.FP32: 1.0,
            Precision.FP16: 0.5,
            Precision.INT8: 0.125,
        },

        min_occupancy=0.3,
        max_concurrent_kernels=8,
        wave_quantization=4,
        bom_cost_profile=bom_cost,
    )


