"""
TPU Edge Pro Resource Model hardware resource model.

Hypothetical Google TPU Edge Pro @ 30W with FP32 support.

This is a realistic "what if Google made a 30W edge TPU with FP32 support"
model for fair comparison with KPU T256, Jetson Orin AGX, and ARM Cortex CPUs.

Design Rationale:
- Scale up Coral Edge TPU (2W) to 30W thermal envelope (15× power budget)
- Add FP32/BF16 support (following TPU v4 ISA evolution)
- Maintain systolic array dataflow (static, minimal control overhead)
- Add L2 SRAM stage for better memory hierarchy
- Target edge/datacenter hybrid use case (like KPU T256)

Scaling Strategy:
- 2W → 30W = 15× power budget
- 64×64 → 128×128 systolic array (4× compute)
- 500 MHz → 850 MHz clock (1.7× frequency)
- 0.5 MB → 4 MB L2 SRAM (8× on-chip memory)
- 4 GB/s → 128 GB/s bandwidth (32× memory bandwidth, 16× LPDDR5 channels)

Architecture Comparison:
                Coral Edge TPU    TPU Edge Pro      KPU T256
Power:          2W                30W               30W
Precision:      INT8 only         INT8/BF16/FP32    INT8/BF16/FP32
Array:          64×64 (est)       128×128           256×16×16
Clock:          500 MHz           850 MHz           1.2 GHz
Memory:         3-stage           4-stage           4-stage
Control:        Static dataflow   Static dataflow   Dynamic dataflow
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
from ...architectural_energy import TPUTileEnergyModel, SystolicArrayEnergyModel


def tpu_edge_pro_resource_model() -> HardwareResourceModel:
    """
    Google TPU Edge Pro @ 30W (hypothetical, realistic scaling from Coral).

    ============================================================================
    TPU EDGE PRO @ 30W - DESIGN PHILOSOPHY
    ============================================================================

    Target: Fair comparison with 30W competitors (KPU T256, Jetson Orin AGX)
    - 30W thermal envelope (15W/30W/45W profiles)
    - 128×128 systolic array (16,384 PEs)
    - FP32/BF16/INT8 support (following TPU v4 ISA)
    - Static dataflow (minimal control overhead)
    - 4-stage memory hierarchy (DRAM → L2 → Scratchpad → Accumulator)

    Compute Capabilities:
    - INT8:  55.5 TOPS peak (2× BF16, 85% efficiency → 47.2 TOPS effective)
    - BF16:  27.7 TFLOPS peak (1× base, 85% efficiency → 23.5 TFLOPS effective)
    - FP32:  13.9 TFLOPS peak (0.5× BF16, 85% efficiency → 11.8 TFLOPS effective)

    Why 85% efficiency?
    - Systolic arrays are highly efficient for dense matrix operations
    - Higher than GPU (5-12%) due to static dataflow (no warp divergence)
    - Lower than KPU (70-80%) due to poor memory reuse (DRAM ↔ SRAM)

    Memory Hierarchy (4-stage):
    - L3 (DRAM):      32 GB LPDDR5 (128 GB/s bandwidth, 16× channels)
    - L2 (SRAM):      4 MB on-chip SRAM (new for 30W!)
    - L1 (Scratchpad): 2 MB unified buffer (systolic array scratchpad)
    - L0 (Accumulators): 256 KB accumulator registers

    Energy Model:
    - INT8 MAC:  0.4 pJ (same as Coral Edge TPU)
    - BF16 MAC:  0.6 pJ (1.5× INT8, add FP16 accumulator)
    - FP32 MAC:  1.2 pJ (2× BF16, add FP32 accumulator + rounding)
    - Base ALU:  0.6 pJ (unencumbered FP32 MAC without accumulator)

    Control Overhead:
    - Static dataflow (minimal control)
    - No instruction fetch/decode (systolic array is hardwired)
    - No cache coherence (explicit weight loading)
    - Schedule setup energy: ~50 nJ one-time cost per layer

    Key Advantages vs competitors:
    ✓ Minimal control overhead (static dataflow)
    ✓ High efficiency (85% for systolic arrays)
    ✓ No SIMT scheduling overhead (unlike GPU)
    ✓ No instruction fetch overhead (unlike CPU)

    Key Disadvantages vs KPU:
    ✗ Poor memory reuse (DRAM → SRAM → Scratchpad)
    ✗ No distributed L3 memory
    ✗ Static dataflow less flexible than token-based dataflow
    ✗ Large systolic array underutilized for small workloads
    """

    # ========================================================================
    # COMPUTE CONFIGURATION
    # ========================================================================

    # Clock domain (850 MHz, modest boost from Coral's 500 MHz)
    clock_domain = ClockDomain(
        base_clock_hz=850e6,       # 850 MHz
        max_boost_clock_hz=900e6,  # 900 MHz max
        sustained_clock_hz=850e6,  # Sustained at 30W
        dvfs_enabled=True,         # Can scale down to 15W
    )

    # Systolic array configuration
    array_width = 128
    array_height = 128
    num_pes = array_width * array_height  # 16,384 PEs

    # Performance calculations
    clock_hz = clock_domain.base_clock_hz

    # INT8: 2 MACs per PE per cycle (weight stationary + double buffering)
    int8_ops_per_clock = num_pes * 2
    int8_tops = int8_ops_per_clock * clock_hz  # 55.5 TOPS peak

    # BF16: 1 MAC per PE per cycle (base precision)
    bf16_ops_per_clock = num_pes * 1
    bf16_tflops = bf16_ops_per_clock * clock_hz  # 27.7 TFLOPS peak

    # FP32: 0.5 MACs per PE per cycle (requires 2 cycles per MAC)
    fp32_ops_per_clock = num_pes * 0.5
    fp32_tflops = fp32_ops_per_clock * clock_hz  # 13.9 TFLOPS peak

    # Systolic array efficiency (85% for dense matmuls)
    efficiency = 0.85

    # ========================================================================
    # ENERGY MODEL (Systolic Array with FP32 Support)
    # ========================================================================

    # Base ALU energy (FP32, unencumbered MAC without accumulator)
    # This is for fair comparison with CPU/GPU/KPU base ALU energy
    energy_per_flop_fp32 = 0.6e-12  # 0.6 pJ (static dataflow, minimal control)

    # Full MAC energies (MAC + accumulator + rounding)
    mac_energy_int8 = 0.4e-12   # 0.4 pJ (same as Coral Edge TPU)
    mac_energy_bf16 = 0.6e-12   # 0.6 pJ (1.5× INT8, add BF16 accumulator)
    mac_energy_fp32 = 1.2e-12   # 1.2 pJ (2× BF16, add FP32 accumulator + rounding)

    # Memory hierarchy energy
    energy_per_byte = 12e-12  # 12 pJ/byte (LPDDR5, similar to KPU T256)

    # ========================================================================
    # MEMORY HIERARCHY (4-Stage)
    # ========================================================================

    # Stage 1: DRAM (off-chip LPDDR5)
    main_memory = 32 * 1024**3  # 32 GB LPDDR5
    peak_bandwidth = 128e9      # 128 GB/s (16× channels @ 8 GB/s each)
    dram_energy_per_byte = 12e-12  # 12 pJ/byte

    # Stage 2: L2 SRAM (on-chip, new for 30W)
    l2_cache_total = 4 * 1024 * 1024  # 4 MB
    l2_energy_per_byte = 1.0e-12      # 1.0 pJ/byte

    # Stage 3: L1 Unified Buffer (systolic array scratchpad)
    l1_cache_per_unit = 2 * 1024 * 1024  # 2 MB
    l1_energy_per_byte = 0.5e-12         # 0.5 pJ/byte

    # Stage 4: Accumulator registers (per MXU)
    accumulator_size = 256 * 1024  # 256 KB
    accumulator_energy_per_element = 0.4e-12  # 0.4 pJ per 32-bit write

    # ========================================================================
    # THERMAL OPERATING POINTS
    # ========================================================================

    # Compute resource (single systolic array)
    compute_resource = ComputeResource(
        resource_type="Systolic-Array",
        num_units=1,
        ops_per_unit_per_clock={
            Precision.INT8: int8_ops_per_clock,
            Precision.BF16: bf16_ops_per_clock,
            Precision.FP32: fp32_ops_per_clock,
        },
        clock_domain=clock_domain,
    )

    # Performance characteristics (same efficiency across precisions for systolic arrays)
    perf_int8 = PerformanceCharacteristics(
        precision=Precision.INT8,
        compute_resource=compute_resource,
        efficiency_factor=efficiency,
        native_acceleration=True,
        tile_utilization=1.0,
    )
    perf_bf16 = PerformanceCharacteristics(
        precision=Precision.BF16,
        compute_resource=compute_resource,
        efficiency_factor=efficiency,
        native_acceleration=True,
        tile_utilization=1.0,
    )
    perf_fp32 = PerformanceCharacteristics(
        precision=Precision.FP32,
        compute_resource=compute_resource,
        efficiency_factor=efficiency,
        native_acceleration=True,
        tile_utilization=1.0,
    )

    # Three thermal profiles (15W, 30W, 45W)
    thermal_operating_points = {
        "15W": ThermalOperatingPoint(
            name="15W",
            tdp_watts=15.0,
            cooling_solution="Passive (heatsink)",
            performance_specs={
                Precision.INT8: perf_int8,
                Precision.BF16: perf_bf16,
                Precision.FP32: perf_fp32,
            },
        ),
        "30W": ThermalOperatingPoint(
            name="30W",
            tdp_watts=30.0,
            cooling_solution="Active (fan)",
            performance_specs={
                Precision.INT8: perf_int8,
                Precision.BF16: perf_bf16,
                Precision.FP32: perf_fp32,
            },
        ),
        "45W": ThermalOperatingPoint(
            name="45W",
            tdp_watts=45.0,
            cooling_solution="Active (high-performance fan)",
            performance_specs={
                Precision.INT8: perf_int8,
                Precision.BF16: perf_bf16,
                Precision.FP32: perf_fp32,
            },
        ),
    }

    # ========================================================================
    # TPU TILE ENERGY MODEL
    # ========================================================================

    tile_energy_model = TPUTileEnergyModel(
        # Array configuration (128×128 systolic array)
        array_width=128,
        array_height=128,
        num_arrays=1,  # Single large systolic array

        # Tile configuration (moderate tiles for 30W)
        weight_tile_size=16 * 1024,  # 16 KiB per tile
        weight_fifo_depth=2,         # 2 tiles buffered

        # Pipeline (moderate pipeline for 850 MHz)
        pipeline_fill_cycles=128,    # 128 cycles to fill pipeline
        clock_frequency_hz=850e6,    # 850 MHz

        # Accumulator (256 KB for roofline knee)
        accumulator_size=256 * 1024,  # 256 KB
        accumulator_width=128,        # 128 elements wide

        # Unified Buffer (L1 scratchpad)
        unified_buffer_size=2 * 1024 * 1024,  # 2 MB

        # Energy coefficients (LPDDR5 + on-chip SRAM)
        weight_memory_energy_per_byte=12.0e-12,     # 12 pJ/byte (LPDDR5)
        weight_fifo_energy_per_byte=1.0e-12,        # 1.0 pJ/byte (L2 SRAM)
        unified_buffer_read_energy_per_byte=0.5e-12,  # 0.5 pJ/byte (L1)
        unified_buffer_write_energy_per_byte=0.5e-12, # 0.5 pJ/byte (L1)
        accumulator_write_energy_per_element=0.4e-12, # 0.4 pJ (32-bit write)
        accumulator_read_energy_per_element=0.3e-12,  # 0.3 pJ (32-bit read)
        weight_shift_in_energy_per_element=0.3e-12,   # 0.3 pJ (shift register)
        activation_stream_energy_per_element=0.2e-12, # 0.2 pJ (stream)
        mac_energy=mac_energy_bf16,  # 0.6 pJ per BF16 MAC (base precision)
    )

    # ========================================================================
    # BOM COST PROFILE
    # ========================================================================

    bom_cost = BOMCostProfile(
        silicon_die_cost=80.0,    # Larger die than Coral (128×128 array + 4 MB L2)
        package_cost=15.0,        # Advanced packaging
        memory_cost=30.0,         # 32 GB LPDDR5
        pcb_assembly_cost=10.0,   # More complex PCB
        thermal_solution_cost=5.0, # Active cooling (fan)
        other_costs=10.0,
        total_bom_cost=150.0,
        margin_multiplier=3.0,
        retail_price=450.0,
        volume_tier="10K+",
        process_node="7nm",       # Advanced node for 30W efficiency
        year=2025,
        notes="Hypothetical 30W Edge TPU with FP32 support. Scaled up from Coral Edge TPU. "
              "128×128 systolic array, 4 MB L2 SRAM, 32 GB LPDDR5. Target: Fair comparison "
              "with KPU T256 and Jetson Orin AGX at 30W thermal envelope.",
    )

    # ========================================================================
    # HARDWARE RESOURCE MODEL
    # ========================================================================

    model = HardwareResourceModel(
        name="TPU-Edge-Pro",
        hardware_type=HardwareType.TPU,
        compute_units=1,  # Single systolic array (128×128)
        threads_per_unit=num_pes,  # 16,384 PEs
        warps_per_unit=1,
        warp_size=1,

        precision_profiles={
            Precision.FP32: PrecisionProfile(
                precision=Precision.FP32,
                peak_ops_per_sec=fp32_tflops,
                tensor_core_supported=True,  # Systolic array acts like tensor cores
                relative_speedup=0.5,        # 0.5× BF16
                bytes_per_element=4,
                accumulator_precision=Precision.FP32,
            ),
            Precision.BF16: PrecisionProfile(
                precision=Precision.BF16,
                peak_ops_per_sec=bf16_tflops,
                tensor_core_supported=True,
                relative_speedup=1.0,  # Base precision
                bytes_per_element=2,
                accumulator_precision=Precision.FP32,
            ),
            Precision.INT8: PrecisionProfile(
                precision=Precision.INT8,
                peak_ops_per_sec=int8_tops,
                tensor_core_supported=True,
                relative_speedup=2.0,  # 2× BF16
                bytes_per_element=1,
                accumulator_precision=Precision.INT32,
            ),
        },
        default_precision=Precision.BF16,

        peak_bandwidth=peak_bandwidth,
        l1_cache_per_unit=l1_cache_per_unit,
        l2_cache_total=l2_cache_total,
        main_memory=main_memory,
        energy_per_flop_fp32=energy_per_flop_fp32,
        energy_per_byte=energy_per_byte,
        energy_scaling={
            Precision.INT8: mac_energy_int8 / energy_per_flop_fp32,  # 0.67× base
            Precision.BF16: mac_energy_bf16 / energy_per_flop_fp32,  # 1.0× base
            Precision.FP32: mac_energy_fp32 / energy_per_flop_fp32,  # 2.0× base
        },
        min_occupancy=0.7,  # Systolic arrays need high utilization
        max_concurrent_kernels=1,  # Single model at a time
        wave_quantization=1,
        thermal_operating_points=thermal_operating_points,
        default_thermal_profile="30W",
        bom_cost_profile=bom_cost,
    )

    # Attach tile energy model
    model.tile_energy_model = tile_energy_model

    # Attach architectural energy model (first-principles control overhead)
    model.architecture_energy_model = SystolicArrayEnergyModel()

    return model
