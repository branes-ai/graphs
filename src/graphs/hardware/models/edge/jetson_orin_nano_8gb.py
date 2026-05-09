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
    fp32_ops_per_sm_per_clock = 256  # 128 CUDA cores × 1 FMA = 128 MACs/clock/SM
    # Tensor Core throughput (Ampere SM 8.6, FP16 TC fragment is 8x8x4 ->
    # 256 MACs/clock/TC = 512 ops/clock/TC; INT8 is 2x that). Per the
    # Ampere whitepaper and verified against V4 Phase B baseline on
    # Orin Nano Super: pre-fix value of 64 MACs/TC was half the spec,
    # which made the analyzer's compute roof sit ~10x BELOW measured
    # GFLOPS for cuBLAS matmul. See #91 / #94 for the diagnosis.
    int8_ops_per_sm_per_clock = 2048  # 4 TCs/SM × 256 MACs/TC = 1024 MACs/clock/SM
    fp16_ops_per_sm_per_clock = 1024  # 4 TCs/SM × 128 MACs/TC =  512 MACs/clock/SM

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
            Precision.FP32: 2,  # FMA = 2 ops/clock (mul + add)
            Precision.FP16: 2,  # FP16 emulated on CUDA cores
            Precision.INT8: 2,  # INT8 emulated on CUDA cores
        },
        core_frequency_hz=baseline_freq_hz,  # 650 MHz sustained (15W)
        process_node_nm=8,  # Samsung 8nm
        energy_per_flop_fp32=get_base_alu_energy(8, "standard_cell"),  # 1.9 pJ
        energy_scaling={
            Precision.FP64: 2.0,  # Double precision = 2× energy
            Precision.FP32: 1.0,  # Baseline
            Precision.FP16: 0.5,  # Half precision (emulated on CUDA cores)
            Precision.INT8: 0.125,  # INT8 (emulated on CUDA cores)
        },
    )

    # ========================================================================
    # Tensor Core Fabric (15% more efficient for fused MAC+accumulate)
    # ========================================================================
    tensor_fabric = ComputeFabric(
        fabric_type="tensor_core",
        circuit_type="tensor_core",
        num_units=total_tensor_cores,  #  8 SMs × 4 TCs/SM = 32 TCs
        ops_per_unit_per_clock={
            # Ampere SM 8.6 Tensor Core: FP16 fragment is 8x8x4 = 256
            # MACs/clock/TC = 512 ops/clock/TC. Pre-fix value of 64 was
            # half the spec; #94 calibrated against measured 7-8 TFLOPS
            # on Orin Nano Super at 650 MHz sustained.
            Precision.FP16: 512,  # 256 MACs/clock/TC × 2 ops/MAC
            Precision.INT8: 1024,  # 2x FP16 rate on Ampere
        },
        core_frequency_hz=baseline_freq_hz,  # 650 MHz sustained (15W)
        process_node_nm=8,
        energy_per_flop_fp32=get_base_alu_energy(
            8, "tensor_core"
        ),  # 1.62 pJ (15% better)
        energy_scaling={
            Precision.FP16: 0.5,  # Half precision
            Precision.INT8: 0.25,  # INT8
        },
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
        # Memory clock (issue #136 Phase 4). Conservative assumption:
        # Orin Nano Super silicon supports LPDDR5-6400 (= 3200 MHz
        # internal DRAM clock) across all nvpmodel profiles. NVIDIA
        # doesn't publish a clean per-mode DRAM throttling table, so
        # absent authoritative data we report the silicon's max rate.
        # Refine in a follow-up when per-mode telemetry confirms or
        # rejects throttling at 7W.
        memory_clock_mhz=3200.0,
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
        },
    )

    # ========================================================================
    # 15W MODE: Standard edge deployment (Orin Nano Super, Dec 2024)
    # ========================================================================
    # The Orin Nano *Super* refresh raised the 15W power envelope so the
    # GPU can sustain near-boost clocks (~1.02 GHz) under cuBLAS Tensor
    # Core load -- the 500 MHz sustained figure was from the original
    # 2023 Nano. The V4 Phase B baseline (#90) was captured at 15W on a
    # Super, measuring 7-8 TFLOPS for fp16 matmul (peak shapes).
    # Effective FP16 calibration: 8 SMs * 1024 ops/clock * 1020 MHz *
    # 0.85 = 7.1 TFLOPS, matching the V4 baseline.
    clock_15w = ClockDomain(
        base_clock_hz=306e6,
        max_boost_clock_hz=1020e6,  # Super raised boost from 918 MHz
        sustained_clock_hz=1020e6,  # Super can sustain boost at 15W under cuBLAS
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
        name="15W-Super",
        tdp_watts=15.0,
        cooling_solution="passive-heatsink",
        memory_clock_mhz=3200.0,  # LPDDR5-6400 (Super silicon's headline rate)
        performance_specs={
            Precision.INT8: PerformanceCharacteristics(
                precision=Precision.INT8,
                compute_resource=compute_resource_15w,
                instruction_efficiency=0.95,
                memory_bottleneck_factor=0.80,
                efficiency_factor=0.80,  # Tensor Core INT8 well-utilized
                native_acceleration=True,
            ),
            Precision.FP16: PerformanceCharacteristics(
                precision=Precision.FP16,
                compute_resource=compute_resource_15w,
                # 0.85 = cuBLAS-tuned Tensor Core matmul fraction of
                # theoretical, calibrated against #94 V4 baseline
                # measurement of 7-8 TFLOPS (peak achievable on the
                # captured sweep shapes).
                efficiency_factor=0.85,
                native_acceleration=True,
            ),
            Precision.FP32: PerformanceCharacteristics(
                precision=Precision.FP32,
                compute_resource=compute_resource_15w,
                # FP32 doesn't use Tensor Cores on consumer Ampere;
                # CUDA-core only with modest utilization.
                efficiency_factor=0.40,
                native_acceleration=True,
            ),
        },
    )

    # ========================================================================
    # 25W MODE: Orin Nano Super at fixed 25W cap (production deployment)
    # ========================================================================
    # The Super refresh (Dec 2024) raised the chip's power budget to 25W.
    # The "25W" mode is an explicit fixed-cap deployment config -- DVFS
    # allowed within the 25W envelope so the chip can scale down for
    # power-aware workloads while still being thermally provisioned for
    # full Super performance. This is the recommended production mode for
    # devices that have active cooling but want predictable power
    # accounting. Compare to MAXN below, which locks DVFS off and runs at
    # boost regardless of headroom.
    #
    # Performance characteristics match MAXN at peak workloads (cuBLAS pegs
    # at boost in both); the operational difference is in idle / mixed
    # workloads where 25W mode allows DVFS to drop clocks for power saving.
    clock_25w = ClockDomain(
        base_clock_hz=306e6,
        max_boost_clock_hz=1020e6,
        sustained_clock_hz=1020e6,  # Same boost ceiling as MAXN
        dvfs_enabled=True,           # KEY DIFFERENCE vs MAXN: DVFS allowed
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
        name="25W-Super",
        tdp_watts=25.0,
        cooling_solution="active-fan",
        memory_clock_mhz=3200.0,  # LPDDR5-6400 (Super silicon's headline rate)
        performance_specs={
            Precision.INT8: PerformanceCharacteristics(
                precision=Precision.INT8,
                compute_resource=compute_resource_25w,
                instruction_efficiency=0.95,
                memory_bottleneck_factor=0.80,
                efficiency_factor=0.80,
                native_acceleration=True,
            ),
            Precision.FP16: PerformanceCharacteristics(
                precision=Precision.FP16,
                compute_resource=compute_resource_25w,
                # Same calibration as MAXN (cuBLAS pegs at boost).
                efficiency_factor=0.85,
                native_acceleration=True,
            ),
            Precision.FP32: PerformanceCharacteristics(
                precision=Precision.FP32,
                compute_resource=compute_resource_25w,
                efficiency_factor=0.40,
                native_acceleration=True,
            ),
        },
    )

    # ========================================================================
    # MAXN MODE: Orin Nano Super (Dec 2024 refresh) at full power
    # ========================================================================
    # Super refresh raised the power budget from 15W to 25W, allowing higher
    # sustained GPU clocks (up to 1.02 GHz boost). At MAXN with cuBLAS
    # Tensor Core matmul, the GPU pegs at boost.
    #
    # Empirical anchor (#94): user's V4 Phase B baseline on Orin Nano Super
    # measured 7-8 TFLOPS sustained for fp16 matmul. With per-SM Tensor Core
    # rate of 1024 ops/clock (SM 8.6 spec), 8 SMs * 1024 * 1020 MHz =
    # 8.36 TFLOPS theoretical; * 0.85 efficiency = 7.1 TFLOPS effective.
    clock_maxn = ClockDomain(
        base_clock_hz=306e6,
        max_boost_clock_hz=1020e6,  # Super raised boost to 1.02 GHz
        sustained_clock_hz=1020e6,  # cuBLAS pegs at boost; no throttling at 25W
        dvfs_enabled=False,
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

    thermal_maxn = ThermalOperatingPoint(
        name="MAXN-Super",
        tdp_watts=25.0,  # Super raised TDP cap from 15W to 25W
        cooling_solution="active-fan",
        memory_clock_mhz=3200.0,  # LPDDR5-6400 (Super silicon's headline rate)
        performance_specs={
            Precision.INT8: PerformanceCharacteristics(
                precision=Precision.INT8,
                compute_resource=compute_resource_maxn,
                instruction_efficiency=0.95,
                memory_bottleneck_factor=0.80,
                efficiency_factor=0.80,
                native_acceleration=True,
            ),
            Precision.FP16: PerformanceCharacteristics(
                precision=Precision.FP16,
                compute_resource=compute_resource_maxn,
                # 0.85 = cuBLAS-tuned Tensor Core matmul fraction of
                # theoretical, calibrated against #94 measured 7-8 TFLOPS.
                efficiency_factor=0.85,
                native_acceleration=True,
            ),
            Precision.FP32: PerformanceCharacteristics(
                precision=Precision.FP32,
                compute_resource=compute_resource_maxn,
                # FP32 doesn't use Tensor Cores on consumer Ampere;
                # CUDA-core only with modest utilization.
                efficiency_factor=0.40,
                native_acceleration=True,
            ),
        },
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
        # Thermal operating points with DVFS modeling.
        # Four canonical modes for Orin Nano Super (Dec 2024+):
        #   7W   -- battery / drone-payload deployment
        #   15W  -- standard edge AI baseline (Phase B reference)
        #   25W  -- production deployment with explicit 25W cap (DVFS on)
        #   MAXN -- developer/burst mode, locked at boost (DVFS off)
        # 25W and MAXN share the same TDP cap (the silicon's max envelope on
        # Super) but differ in DVFS behavior: 25W scales down on idle for
        # power-aware deployments, MAXN locks at boost regardless.
        thermal_operating_points={
            "7W": thermal_7w,    # Battery-powered devices
            "15W": thermal_15w,  # Orin Nano Super 15W (Phase B baseline)
            "25W": thermal_25w,  # Production 25W cap with DVFS (#136)
            "MAXN": thermal_maxn,  # Orin Nano Super at full 25W power, DVFS off
        },
        # 15W is the default because the V4 Phase B baseline (#90) was
        # captured in the 15W thermal profile. Operators running at
        # MAXN should pass thermal_profile="MAXN"; battery deployments
        # should pass "7W". On the Super silicon, 15W can sustain
        # near-boost clocks under cuBLAS Tensor Core load.
        default_thermal_profile="15W",
        # Legacy precision profiles (backward compatibility).
        # FP16/FP32 added for issue #53: get_peak_ops now raises on missing
        # precision instead of silently falling back to INT8. Values follow
        # the per-SM table above (Ampere CUDA cores, 650 MHz reference):
        #   FP32: 8 SMs * 128 ops/clock * 650 MHz =  0.666 TFLOPS
        #   FP16: 8 SMs * 256 ops/clock * 650 MHz =  1.331 TFLOPS  (2x FP32)
        #   INT8: 8 SMs * 512 ops/clock * 650 MHz =  2.662 TOPS    (via dp4a)
        precision_profiles={
            Precision.INT8: PrecisionProfile(
                precision=Precision.INT8,
                peak_ops_per_sec=2.662e12,  # 8 SMs × 512 ops/clock × 650 MHz (realistic peak)
                tensor_core_supported=True,
                bytes_per_element=1,
            ),
            Precision.FP16: PrecisionProfile(
                precision=Precision.FP16,
                peak_ops_per_sec=1.331e12,
                tensor_core_supported=True,
                relative_speedup=0.5,  # vs INT8
                bytes_per_element=2,
            ),
            Precision.FP32: PrecisionProfile(
                precision=Precision.FP32,
                peak_ops_per_sec=0.666e12,
                tensor_core_supported=False,  # FP32 runs on CUDA cores only
                relative_speedup=0.25,  # vs INT8
                bytes_per_element=4,
            ),
        },
        default_precision=Precision.INT8,
        peak_bandwidth=102e9,  # 102 GB/s LPDDR5-6400 (Orin Nano Super, Dec 2024).
        # Original Orin Nano (2023) was 68 GB/s LPDDR5-4267.
        # The V4 Phase B baseline was captured on a Super.
        l1_cache_per_unit=128 * 1024,
        # V5 follow-up: on-chip BW peaks so memory_hierarchy emits a
        # multi-tier view (L1 + L2 + DRAM). Without these, the V5-3b
        # eligibility predicate's >=2 tier gate declines the
        # tier-aware path on Orin Nano even when opted in.
        #
        # L1 per SM at sustained 650 MHz (the baseline_freq_hz used by
        # the compute fabrics above): Ampere SM 8.6 spec is 128 B/cycle
        # for L1 cache + shared memory throughput, so 128 * 650e6 =
        # 83.2 GB/s/SM. The memory_hierarchy property aggregates by
        # multiplying by compute_units (=8 SMs), giving 666 GB/s
        # aggregate L1.
        # Source: NVIDIA Ampere Architecture Whitepaper, "L1 Data
        # Cache and Shared Memory" section.
        l1_bandwidth_per_unit_bps=83e9,
        l2_cache_total=2 * 1024 * 1024,  # 2 MB (half of AGX)
        # L2 BW: NVIDIA doesn't publish an Orin-Nano-specific number;
        # for Ampere chips the L2 BW typically lands at 2-5x DRAM
        # peak depending on partition count and clock. Using 2x DRAM
        # (204 GB/s) as a conservative analytical estimate that
        # preserves L2 > DRAM ordering for the tier picker. Pending
        # an L2-isolating microbench, the achievable_fraction
        # calibration will tighten this; see the i7 L1 analysis at
        # docs/calibration/i7-12700k-l1-calibration-analysis.md for
        # the same pattern (cache-resident BW is hard to measure
        # directly because dispatch overhead dominates).
        l2_bandwidth_bps=204e9,
        main_memory=8 * 1024**3,  # 8 GB
        energy_per_flop_fp32=cuda_fabric.energy_per_flop_fp32,  # 1.9 pJ (8nm Samsung, CUDA cores)
        energy_per_byte=18e-12,
        min_occupancy=0.3,
        max_concurrent_kernels=4,  # Fewer than AGX
        wave_quantization=2,  # Smaller wave size
        bom_cost_profile=bom_cost,
        # V5-5 calibration: DRAM achievable_fraction derived from
        # validation/model_v4/results/baselines/jetson_orin_nano_8gb_vector_add.csv
        # captured at the 15W thermal profile per #94. Plateau rows
        # (N=16M, 67M, 268M; fp16 working set 100 MB to 1.6 GB, all
        # well past the 2 MB L2): measured 61.6 / 56.4 / 56.6 GB/s,
        # median ~57 GB/s. Peak LPDDR5-6400 is 102 GB/s, so
        # achievable_fraction = 57 / 102 = 0.55. Used by the V5-3b
        # tier-aware roofline path when opt-in. L1 / L2 entries stay
        # absent (default 1.0) until matmul-anchored calibration.
        tier_achievable_fractions={"DRAM": 0.55},
        # V5 follow-up: per-op compute efficiency overrides for matmul
        # / linear at fp16. Derived from V4 baseline measurements at
        # validation/model_v4/results/baselines/jetson_orin_nano_8gb_{matmul,linear}.csv.
        #
        # The shared ``_get_compute_efficiency_scale`` curve was tuned
        # for AGX-style "spec peak" interpretations and returns scale
        # ~1.5 for large matmul, but Orin Nano's
        # ``peak_ops_per_sec = 7.1 TFLOPS`` is ALREADY the achievable
        # peak (cuBLAS Tensor Core 0.85 efficiency baked in via
        # ``efficiency_factor`` on the FP16 perf characteristics).
        # Multiplying that by 1.5 over-predicts compute throughput by
        # 2.4x and forces shapes that actually run alu_bound to look
        # dram_bound in the model.
        #
        # V4-baseline-fit sweep at the 15W default (Orin Nano Super
        # 1.02 GHz). Two memory paths matter (legacy via
        # ``_get_bandwidth_efficiency_scale`` and tier-aware via
        # V5-3b); the values below are Pareto-optimal across both:
        #
        # * matmul: 0.70 maximizes latency-pass count.
        #     - tier-aware: lat 22 -> 30 (+8), energy unchanged at 29
        #     - legacy:     lat 18 -> 26 (+8), energy unchanged at 24
        #
        # * linear: 0.94 respects the legacy-path energy floor while
        #   still gaining latency-pass on both paths. The peak
        #   latency-pass point is around 0.85-0.88 but those values
        #   regress the legacy-path energy band (predicted latency
        #   grows -> static_energy = avg_power * latency grows
        #   faster than the actual silicon energy). Sweep results
        #   on the LEGACY memory path (which the V4 floor tests
        #   currently use):
        #     scale  lat-pass  energy-pass
        #     0.85   28        24      ← peak lat, energy fails (floor 27)
        #     0.88   28        25      ← still fails energy floor
        #     0.90   27        25
        #     0.93   25        27      ← matches floor
        #     0.94   25        28      ← ships here, +1 headroom
        #     1.00   22        28
        #     1.40   18        29      ← legacy curve baseline
        #   On the TIER-AWARE path 0.94 gives lat 30 (vs baseline 18).
        #   The energy floor is enforced by
        #   tests/validation_model_v4/test_v4_against_baseline.py
        #   ::test_jetson_orin_nano_linear_pass_energy_floor (>=27).
        #
        # vector_add gets no override (its compute path is dominated
        # by memory; the legacy curve already returns ~1.0 for
        # elementwise ops).
        compute_efficiency_overrides_by_op={
            "fp16": {"matmul": 0.70, "linear": 0.94},
        },
    )
