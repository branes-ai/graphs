"""
Qrb5165 Resource Model hardware resource model.

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


def qrb5165_resource_model() -> HardwareResourceModel:
    """
    Qualcomm QRB5165 (Dragonwing) Robotics Platform with Hexagon 698 DSP.

    Based on Snapdragon 865 SoC, optimized for robotics and edge AI applications.

    ARCHITECTURE:
    - Heterogeneous compute: Kryo 585 CPU + Adreno 650 GPU + Hexagon 698 DSP
    - Primary AI accelerator: Hexagon 698 DSP with Tensor Accelerator (HTA)
    - Dataflow-style execution with vector extensions (HVX)
    - Process: 7nm TSMC FinFET

    CRITICAL REALITY CHECK - Performance Specifications:
    - Marketing claim: 15 TOPS INT8 (5th gen AI Engine, all engines combined)
    - Hexagon DSP breakdown:
      * Hexagon Vector eXtensions (HVX): Vector/tensor operations
      * Hexagon Tensor Accelerator (HTA): Dedicated tensor core-like units
      * Hexagon Scalar Accelerator: Control/scalar ops
    - Expected effective: ~5-6 TOPS INT8 under sustained 7W operation
    - Root cause: Thermal throttling + memory bandwidth bottleneck

    CPU CONFIGURATION:
    - 1× Kryo 585 Prime (Cortex-A77) @ 2.84 GHz
    - 3× Kryo 585 Gold (Cortex-A77) @ 2.42 GHz
    - 4× Kryo 585 Silver (Cortex-A55) @ 1.81 GHz

    GPU (Adreno 650):
    - ~1.0 TFLOPS FP32
    - OpenCL, Vulkan compute support
    - Useful for graphics + light AI workloads

    DSP (Hexagon 698):
    - 15 TOPS INT8 (combined with HVX + HTA)
    - INT8/INT16 native support
    - FP16 supported but slower
    - 4× HVX 1024-bit vector units

    Power Profile:
    ============

    7W Mode (Typical Robotics Deployment):
    - Sustained DSP clock: ~900 MHz (throttled from peak)
    - Effective INT8: ~6 TOPS (40% of 15 TOPS peak)
    - Use case: Battery-powered robots, drones, edge devices
    - Thermal throttle factor: ~60% (moderate throttling)

    Memory:
    - LPDDR5 @ 2750 MHz (quad-channel)
    - Bandwidth: 44 GB/s
    - Capacity: Up to 16GB

    References:
    - Qualcomm QRB5165 Product Brief (87-28730-1 REV D)
    - Snapdragon 865 specifications
    - Hexagon 698 DSP architecture
    """
    # ========================================================================
    # HEXAGON DSP ARCHITECTURE MODELING
    # ========================================================================
    # Hexagon 698 has:
    # - 4× HVX (Hexagon Vector eXtensions) units: 1024-bit SIMD
    # - Tensor Accelerator (HTA): Dedicated for matrix operations
    # - We model as equivalent "DSP cores" for compatibility

    # 15 TOPS INT8 @ ~1.5 GHz peak
    # → 15e12 ops/sec / 1.5e9 Hz = 10,000 ops/cycle
    # If we model as 32 "DSP processing elements":
    # → 10,000 / 32 = 312.5 ops/cycle/unit

    num_dsp_units = 32  # Equivalent processing elements (HVX + HTA combined)
    num_hvx_units = 4   # Actual HVX SIMD units
    peak_clock_hz = 1.5e9  # 1.5 GHz peak
    sustained_clock_hz = 900e6  # 900 MHz sustained @ 7W

    # ========================================================================
    # Multi-Fabric Architecture (Qualcomm Hexagon DSP)
    # ========================================================================
    # HVX Vector Fabric (SIMD operations: conv, pool, activations)
    # ========================================================================
    hvx_fabric = ComputeFabric(
        fabric_type="hvx_vector",
        circuit_type="simd_packed",    # SIMD vector operations
        num_units=num_hvx_units,       # 4× HVX units
        ops_per_unit_per_clock={
            Precision.INT8: 256,        # 1024-bit / 4-bit = 256 INT8 ops/cycle
            Precision.INT16: 128,       # 1024-bit / 8-bit = 128 INT16 ops/cycle
            Precision.FP16: 64,         # Emulated, slower
            Precision.INT4: 512,        # 1024-bit / 2-bit = 512 INT4 ops/cycle
        },
        core_frequency_hz=sustained_clock_hz,  # 900 MHz sustained
        process_node_nm=7,              # 7nm TSMC
        energy_per_flop_fp32=get_base_alu_energy(7, 'simd_packed'),  # 1.62 pJ
        energy_scaling={
            Precision.INT8: 0.15,       # INT8 is very efficient
            Precision.INT16: 0.25,
            Precision.FP16: 0.50,
            Precision.INT4: 0.08,
        }
    )

    # ========================================================================
    # HTA Tensor Fabric (Matrix operations: dense layers, attention)
    # ========================================================================
    hta_fabric = ComputeFabric(
        fabric_type="hta_tensor",
        circuit_type="tensor_core",     # Tensor accelerator (like GPU Tensor Cores)
        num_units=1,                    # Single HTA unit
        ops_per_unit_per_clock={
            Precision.INT8: 9024,       # HTA dominates INT8: ~60% of 15 TOPS / 1.5 GHz
            Precision.INT16: 4512,      # Half of INT8
            Precision.FP16: 2256,       # Slower
        },
        core_frequency_hz=sustained_clock_hz,  # 900 MHz sustained
        process_node_nm=7,
        energy_per_flop_fp32=get_base_alu_energy(7, 'tensor_core'),  # 1.53 pJ (15% better)
        energy_scaling={
            Precision.INT8: 0.15,
            Precision.INT16: 0.25,
            Precision.FP16: 0.50,
        }
    )

    # HVX INT8: 4 units × 256 ops/cycle × 900 MHz = 0.92 TOPS
    # HTA INT8: 1 unit × 9024 ops/cycle × 900 MHz = 8.12 TOPS
    # Total sustained: 0.92 + 8.12 = 9.04 TOPS (60% of 15 TOPS peak) ✓

    # ========================================================================
    # CLOCK DOMAIN - 7W Thermal Envelope
    # ========================================================================
    clock_7w = ClockDomain(
        base_clock_hz=800e6,        # 800 MHz minimum
        max_boost_clock_hz=1.5e9,   # 1.5 GHz peak
        sustained_clock_hz=900e6,   # 900 MHz sustained @ 7W (60% throttle)
        dvfs_enabled=True,
    )

    # ========================================================================
    # COMPUTE RESOURCE
    # ========================================================================
    compute_resource_7w = ComputeResource(
        resource_type="Hexagon-698-DSP-HVX-HTA",
        num_units=num_dsp_units,
        ops_per_unit_per_clock={
            Precision.INT8: 312,    # 312 INT8 ops/cycle/unit (optimized)
            Precision.INT16: 156,   # 156 INT16 ops/cycle/unit (0.5× INT8)
            Precision.FP16: 78,     # 78 FP16 ops/cycle/unit (slower, not native)
            Precision.INT4: 624,    # 624 INT4 ops/cycle/unit (2× INT8, experimental)
        },
        clock_domain=clock_7w,
    )

    # Peak INT8: 32 units × 312 ops/cycle × 1.5 GHz = 14.98 TOPS ≈ 15 TOPS ✓
    # Sustained @ 7W: 32 × 312 × 900 MHz = 8.99 TOPS
    # Effective: 8.99 × 0.60 = 5.4 TOPS (36% of 15 TOPS peak)

    # ========================================================================
    # THERMAL PROFILE (7W Robotics/Edge Deployment)
    # ========================================================================
    thermal_7w = ThermalOperatingPoint(
        name="7W-robotics",
        tdp_watts=7.0,
        cooling_solution="passive-heatsink",
        performance_specs={
            Precision.INT8: PerformanceCharacteristics(
                precision=Precision.INT8,
                compute_resource=compute_resource_7w,
                instruction_efficiency=0.85,  # Good DSP efficiency
                memory_bottleneck_factor=0.70,  # 44 GB/s is limiting for 15 TOPS
                efficiency_factor=0.60,  # 60% effective (realistic for sustained load)
                tile_utilization=0.80,  # Good HVX utilization
                native_acceleration=True,
            ),
            Precision.INT16: PerformanceCharacteristics(
                precision=Precision.INT16,
                compute_resource=compute_resource_7w,
                instruction_efficiency=0.82,
                memory_bottleneck_factor=0.65,  # More bandwidth per op
                efficiency_factor=0.55,
                tile_utilization=0.75,
                native_acceleration=True,
            ),
            Precision.FP16: PerformanceCharacteristics(
                precision=Precision.FP16,
                compute_resource=compute_resource_7w,
                instruction_efficiency=0.70,  # Not as optimized as INT8
                memory_bottleneck_factor=0.60,
                efficiency_factor=0.45,  # Lower efficiency for FP
                tile_utilization=0.70,
                native_acceleration=False,  # Emulated on INT/fixed hardware
            ),
            Precision.INT4: PerformanceCharacteristics(
                precision=Precision.INT4,
                compute_resource=compute_resource_7w,
                instruction_efficiency=0.80,
                memory_bottleneck_factor=0.75,  # Less bandwidth needed
                efficiency_factor=0.65,  # Slightly better than INT8
                tile_utilization=0.75,
                native_acceleration=True,
            ),
        }
    )

    # ========================================================================
    # BOM COST PROFILE (Estimated @ 10K units)
    # ========================================================================
    # QRB5165 is based on Snapdragon 865 (7nm, 2020 platform)
    # More expensive than QCS6490 (6nm, 2022) due to older process node
    bom_cost = BOMCostProfile(
        silicon_die_cost=55.0,        # 7nm die (larger than 6nm)
        package_cost=14.0,             # Advanced flip-chip package
        memory_cost=18.0,              # 4GB LPDDR5 on-package (higher than LPDDR4X)
        pcb_assembly_cost=7.0,         # SMT assembly
        thermal_solution_cost=3.0,     # Heatsink for 7W
        other_costs=6.0,               # Testing, connectors, robotics certification
        total_bom_cost=0,              # Auto-calculated
        margin_multiplier=2.7,         # Qualcomm robotics platform margin
        retail_price=0,                # Auto-calculated
        volume_tier="10K+",
        process_node="7nm",
        year=2021,  # Platform introduction year
        notes="Robotics platform based on Snapdragon 865. Higher BOM than QCS6490 due to 7nm vs 6nm process."
    )

    # BOM: $55 + $14 + $18 + $7 + $3 + $6 = $103
    # Retail: $103 × 2.7 = $278 (Qualcomm Robotics RB5 development kit ~$449)

    # ========================================================================
    # HARDWARE RESOURCE MODEL
    # ========================================================================
    return HardwareResourceModel(
        name="Qualcomm-QRB5165-Hexagon698",
        hardware_type=HardwareType.DSP,

        # NEW: Multi-fabric architecture (HVX vector + HTA tensor)
        compute_fabrics=[hvx_fabric, hta_fabric],

        compute_units=num_dsp_units,
        threads_per_unit=4,  # HVX units per "processing element"
        warps_per_unit=1,
        warp_size=32,  # Vector lane width (approximation)

        # BOM cost profile
        bom_cost_profile=bom_cost,

        # Thermal operating points
        thermal_operating_points={
            "7W": thermal_7w,
        },
        default_thermal_profile="7W",

        # Legacy precision profiles (backward compatibility)
        precision_profiles={
            Precision.INT8: PrecisionProfile(
                precision=Precision.INT8,
                peak_ops_per_sec=15e12,  # 15 TOPS INT8
                tensor_core_supported=True,  # HTA acts like tensor cores
                relative_speedup=1.0,
                bytes_per_element=1,
            ),
            Precision.INT16: PrecisionProfile(
                precision=Precision.INT16,
                peak_ops_per_sec=7.5e12,  # 7.5 TOPS INT16 (0.5× INT8)
                tensor_core_supported=True,
                relative_speedup=0.5,
                bytes_per_element=2,
            ),
            Precision.FP16: PrecisionProfile(
                precision=Precision.FP16,
                peak_ops_per_sec=3.75e12,  # 3.75 TFLOPS FP16 (estimated)
                tensor_core_supported=False,
                relative_speedup=0.25,
                bytes_per_element=2,
            ),
        },
        default_precision=Precision.INT8,

        # ====================================================================
        # MEMORY HIERARCHY - LPDDR5 External Memory
        # ====================================================================
        # QRB5165 uses LPDDR5 (not on-chip like Hailo)
        # - 4 channels × 16-bit × 2750 MHz × 2 (DDR) = 44 GB/s
        # - Hexagon has local caches/scratchpad
        # ====================================================================
        peak_bandwidth=44e9,  # 44 GB/s LPDDR5 @ 2750 MHz
        l1_cache_per_unit=128 * 1024,  # 128 KB L1 per DSP unit (estimated)
        l2_cache_total=4 * 1024 * 1024,  # 4 MB shared L2 (estimated)
        main_memory=16 * 1024**3,  # Up to 16 GB LPDDR5

        # Energy (use HVX fabric as baseline for general-purpose DSP operations)
        energy_per_flop_fp32=hvx_fabric.energy_per_flop_fp32,  # 1.62 pJ (7nm, SIMD packed)
        energy_per_byte=15e-12,  # 15 pJ/byte (LPDDR5)
        energy_scaling={
            Precision.INT8: 0.15,   # 15% of FP32 energy
            Precision.INT16: 0.25,  # 25% of FP32 energy
            Precision.FP16: 0.50,   # 50% of FP32 energy
            Precision.INT4: 0.08,   # 8% of FP32 energy
        },

        # Scheduling (DSP has sophisticated scheduling)
        min_occupancy=0.60,
        max_concurrent_kernels=8,  # Can run multiple layers concurrently
        wave_quantization=4,  # Process in groups of 4 units
    )


