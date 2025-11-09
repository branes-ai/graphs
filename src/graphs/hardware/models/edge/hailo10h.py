"""
Hailo-10H Resource Model - Generative AI Edge Accelerator

Dataflow architecture optimized for transformers, LLMs, and vision-language models.
Target: Edge Gen AI, on-device LLMs, vision-language models, multimodal AI.

Configuration:
- 40 TOPS INT4, 20 TOPS INT8
- Enhanced dataflow with KV cache support for transformers
- 4-8GB LPDDR4X external memory (for model weights + KV cache)
- 16nm process (same as Hailo-8)
- 2.5W typical power consumption

Competitor to: Qualcomm QCS6490, Edge TPUs, mobile NPUs
"""

from ...resource_model import (
    HardwareResourceModel,
    HardwareType,
    Precision,
    PrecisionProfile,
    ClockDomain,
    ComputeResource,
    PerformanceCharacteristics,
    ThermalOperatingPoint,
    BOMCostProfile,
)


def hailo10h_resource_model() -> HardwareResourceModel:
    """
    Hailo-10H Generative AI Edge Accelerator.

    ARCHITECTURE:
    - Enhanced dataflow architecture for transformers
    - KV cache management (critical for LLM inference)
    - 4-8GB LPDDR4X for model weights and activations
    - Optimized for INT4 quantization (2× performance vs INT8)
    - Supports attention mechanisms, LayerNorm, SwiGLU

    PERFORMANCE:
    - 40 TOPS INT4 (primary use case for LLMs)
    - 20 TOPS INT8 (for vision tasks)
    - Realistic: ~32 TOPS INT4 sustained (80% efficiency)
    - Realistic: ~17 TOPS INT8 sustained (85% efficiency)

    POWER PROFILE:
    - 2.5W typical (same excellent thermal as Hailo-8)
    - 16 TOPS/W INT4 (best in class for edge LLMs)

    USE CASES:
    - On-device LLMs (2B parameter models: Phi-2, TinyLLaMA)
    - Vision-Language Models (LLaVA, CLIP)
    - Stable Diffusion (image generation <5 seconds)
    - Multimodal AI assistants

    REAL-WORLD PERFORMANCE:
    - First token: <1 second (2B LLMs)
    - Token generation: 10 tokens/sec sustained
    - Stable Diffusion 2.1: <5 seconds per image

    CALIBRATION STATUS: ✅ DOCUMENTED
    - Hailo published benchmarks (April 2024 launch)
    - Real-world LLM performance validated
    - Production scheduled for 2026 (automotive AEC-Q100 Grade 2)
    """
    # Physical hardware
    num_dataflow_units = 40  # More units than Hailo-8 for transformer workloads
    int4_ops_per_unit_per_clock = 1000  # Optimized for INT4
    int8_ops_per_unit_per_clock = 500   # Same as Hailo-8 for compatibility

    # ========================================================================
    # 2.5W MODE: Single operating point (no DVFS, optimized thermal)
    # ========================================================================
    clock_2_5w = ClockDomain(
        base_clock_hz=1.0e9,        # 1.0 GHz (lower than Hailo-8, more units)
        max_boost_clock_hz=1.0e9,   # No boost, constant frequency
        sustained_clock_hz=1.0e9,   # No throttling
        dvfs_enabled=False,          # Fixed frequency
    )

    compute_resource_2_5w = ComputeResource(
        resource_type="Hailo-10H-Transformer-Dataflow",
        num_units=num_dataflow_units,
        ops_per_unit_per_clock={
            Precision.INT4: 1000,  # 40 units × 1000 ops/clock × 1.0 GHz = 40 TOPS INT4 ✓
            Precision.INT8: 500,   # 40 units × 500 ops/clock × 1.0 GHz = 20 TOPS INT8 ✓
        },
        clock_domain=clock_2_5w,
    )

    thermal_2_5w = ThermalOperatingPoint(
        name="2.5W-passive",
        tdp_watts=2.5,
        cooling_solution="passive-heatsink-small",
        performance_specs={
            Precision.INT4: PerformanceCharacteristics(
                precision=Precision.INT4,
                compute_resource=compute_resource_2_5w,
                instruction_efficiency=0.92,  # Transformer-optimized dataflow
                memory_bottleneck_factor=0.85,  # External DRAM (vs all-on-chip)
                efficiency_factor=0.80,  # 80% → ~32 TOPS INT4 effective
                native_acceleration=True,
            ),
            Precision.INT8: PerformanceCharacteristics(
                precision=Precision.INT8,
                compute_resource=compute_resource_2_5w,
                instruction_efficiency=0.95,
                memory_bottleneck_factor=0.88,
                efficiency_factor=0.85,  # 85% → ~17 TOPS INT8 effective
                native_acceleration=True,
            ),
        }
    )

    # ========================================================================
    # BOM COST PROFILE (Estimated @ 10K units, 2025)
    # ========================================================================
    bom_cost = BOMCostProfile(
        silicon_die_cost=30.0,       # 16nm die (slightly larger than Hailo-8)
        package_cost=10.0,            # Package with LPDDR4X interface
        memory_cost=20.0,             # 4GB LPDDR4X on-module (for KV cache + weights)
        pcb_assembly_cost=5.0,        # More complex than Hailo-8
        thermal_solution_cost=1.0,    # Tiny heatsink (2.5W)
        other_costs=4.0,              # Testing, connectors, certification
        total_bom_cost=0,             # Auto-calculated: $70
        margin_multiplier=3.5,        # High margin for cutting-edge edge Gen AI
        retail_price=240.0,           # Estimated retail (Hailo-8 is $160, this adds GenAI)
        volume_tier="10K+",
        process_node="16nm",
        year=2025,
        notes="First edge Gen AI accelerator. KV cache support for LLMs. Higher BOM than Hailo-8 "
              "due to external LPDDR4X. Production 2026 for automotive (AEC-Q100 Grade 2)."
    )

    # BOM: $30 + $10 + $20 + $5 + $1 + $4 = $70
    # Retail: $240 (estimated, not yet available)
    # Cost per INT4 TOPS: $70/40 = $1.75 BOM, $6.00 retail

    # ========================================================================
    # Hardware Resource Model
    # ========================================================================
    return HardwareResourceModel(
        name="Hailo-10H",
        hardware_type=HardwareType.KPU,  # Enhanced dataflow for transformers
        compute_units=num_dataflow_units,
        threads_per_unit=128,
        warps_per_unit=1,
        warp_size=1,

        # Thermal operating points
        thermal_operating_points={
            "2.5W": thermal_2_5w,
        },
        default_thermal_profile="2.5W",

        # Legacy precision profiles
        precision_profiles={
            Precision.INT4: PrecisionProfile(
                precision=Precision.INT4,
                peak_ops_per_sec=40e12,  # 40 TOPS INT4 (primary use case)
                tensor_core_supported=True,
                relative_speedup=2.0,  # 2× INT8
                bytes_per_element=0.5,
            ),
            Precision.INT8: PrecisionProfile(
                precision=Precision.INT8,
                peak_ops_per_sec=20e12,  # 20 TOPS INT8 (secondary)
                tensor_core_supported=True,
                relative_speedup=1.0,
                bytes_per_element=1,
            ),
        },
        default_precision=Precision.INT4,  # Primary use case is INT4 LLMs

        # Memory hierarchy (hybrid: on-chip + external DRAM)
        peak_bandwidth=40e9,  # ~40 GB/s LPDDR4X bandwidth
        l1_cache_per_unit=512 * 1024,  # 512 KB per unit (on-chip SRAM)
        l2_cache_total=12 * 1024 * 1024,  # 12 MB on-chip (larger for KV cache)
        main_memory=8 * 1024**3,  # 8 GB LPDDR4X (for model weights + KV cache)

        # Energy (16nm, transformer-optimized)
        energy_per_flop_fp32=0.55e-12,  # 0.55 pJ/FLOP (slightly higher than Hailo-8)
        energy_per_byte=15e-12,          # 15 pJ/byte (LPDDR4X, not on-chip)

        # Scheduling
        min_occupancy=0.75,  # Transformers have varying occupancy
        max_concurrent_kernels=1,  # Single model execution
        wave_quantization=1,

        # BOM cost
        bom_cost_profile=bom_cost,
    )
