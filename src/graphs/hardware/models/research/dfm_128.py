"""
Data Flow Machine Resource Model - DFM-128 Reference Architecture

This models a classic Data Flow Machine (Jack Dennis, MIT) with modern
microarchitectural parameters inspired by superscalar out-of-order cores.

ARCHITECTURE OVERVIEW
=====================

Token-Based Execution:
- Instructions represented as tokens in CAM (Content Addressable Memory)
- Tokens contain opcode + operand slots (data or pointers)
- When all operands available, token becomes "ready"
- Ready tokens dispatched to processing elements
- Results routed back as data tokens to waiting instructions

Control Flow:
1. CAM holds instruction tokens (128 slots)
2. CAM controller performs associative search for ready tokens
3. Ready tokens collected and queued
4. Routing network distributes tokens to PEs
5. PEs execute and produce data tokens
6. Data tokens routed back to CAM
7. Token matching updates operand slots
8. Instruction becomes ready when operands complete

HARDWARE CONFIGURATION
======================

DFM-128 Specification:
- CAM Size: 128 instruction slots
- Processing Elements: 8 PEs (VLIW-like datapath)
  - 4x Integer ALUs (INT)
  - 2x Floating-Point Units (FP)
  - 1x Special Function Unit (SFU - transcendentals, etc.)
  - 1x Load/Store Unit (LSU)
- Routing Network: Crossbar (128 CAM → 8 PEs → 128 CAM)
- Token Matching: Fully associative (128-way CAM lookup)
- Clock: 2.0 GHz

MICROARCHITECTURAL SIMILARITY
==============================

Modern superscalar processors (e.g., Intel/AMD x86) use DFM concepts:
- Register renaming → Token operand slots
- Reservation stations → CAM instruction slots
- Issue queue → CAM ready queue
- Execution ports → Processing elements
- Bypass network → Token routing

The DFM-128 models what a pure dataflow machine would look like
if implemented with similar technology to a high-performance x86 core.

ENERGY CHARACTERISTICS
=======================

DFM has unique energy profile:
- CAM lookups are expensive (associative search across 128 entries)
- Token matching overhead dominates for small workloads
- No instruction fetch/decode (saves energy)
- Routing network energy (crossbar is power-hungry)
- Better efficiency at high ILP (instruction-level parallelism)

WHY NO COMMERCIAL DFM?
======================

1. CAM Energy: Associative search across 128 entries is expensive
2. Token Traffic: Routing overhead can exceed instruction fetch savings
3. Debugging: No program counter makes debugging difficult
4. Compilation: Extracting fine-grained dataflow is hard

However, modern x86/ARM cores use DFM ideas internally!
"""

from graphs.hardware.resource_model import (
    HardwareResourceModel,
    HardwareType,
    Precision,
    PrecisionProfile,
    ComputeFabric,
    get_base_alu_energy,
    ThermalOperatingPoint
)


def dfm_128_resource_model() -> HardwareResourceModel:
    """
    Create resource model for DFM-128 Data Flow Machine.

    SPECIFICATION
    =============

    Architecture:        Data Flow Machine (Token-Based Execution)
    CAM Slots:           128 instruction tokens
    Processing Elements: 8 PEs (4 INT, 2 FP, 1 SFU, 1 LSU)
    Clock:               2.0 GHz
    Process:             7nm (comparable to modern superscalar)

    Functional Units:
    - 4× Integer ALUs:   16 INT ops/cycle (4 units × 4 ops/unit)
    - 2× FP Units:       16 FP32 ops/cycle (2 units × 8 ops/unit)
    - 1× SFU:            2 SFU ops/cycle (transcendentals, etc.)
    - 1× LSU:            2 loads or stores per cycle

    Peak Performance (FP32):
    - 2.0 GHz × 16 FP32/cycle = 32 GFLOPS
    - With FMA: 64 GFLOPS

    Memory:
    - L1 Cache: 64 KB (32 KB I$ + 32 KB D$)
    - L2 Cache: 512 KB (unified)
    - Memory Bandwidth: 25.6 GB/s (DDR4-3200, single channel)

    Power:
    - TDP: 35W (comparable to high-performance embedded core)
    - Idle: 5W
    - Peak: 40W

    COMPARISON TO X86
    =================

    Similar to:
    - Intel Skylake/Coffee Lake single core
    - AMD Zen 2 single core
    - But with explicit token-based dataflow

    Key Difference:
    - X86 hides dataflow graph extraction in hardware (renaming, scheduling)
    - DFM exposes it explicitly (compiler generates dataflow graph)

    Returns:
        HardwareResourceModel configured for DFM-128
    """

    # Compute units = 8 PEs (each can execute one token per cycle)
    compute_units = 8
    clock_hz = 2.0e9  # 2.0 GHz

    # ========================================================================
    # Multi-Fabric Architecture (DFM-128 - Data Flow Machine)
    # ========================================================================
    # Processing Element Fabric (Heterogeneous: INT ALUs + FP Units)
    # ========================================================================
    pe_fabric = ComputeFabric(
        fabric_type="dfm_processing_element",
        circuit_type="simd_packed",      # VLIW-like datapath with SIMD operations
        num_units=8,                     # 8 Processing Elements
        ops_per_unit_per_clock={
            # FP32: 2 FP units × 8 ops/unit × 2 (FMA) / 8 PEs = 4 FP32 ops/PE/cycle
            Precision.FP32: 4,           # 4 FP32 ops/cycle/PE (with FMA)
            # FP16: 2× FP32
            Precision.FP16: 8,           # 8 FP16 ops/cycle/PE
            # BF16: Same as FP16
            Precision.BF16: 8,           # 8 BF16 ops/cycle/PE
            # INT8: 4 INT ALUs × 4 ops/unit / 8 PEs = 2 INT8 ops/PE/cycle
            Precision.INT8: 2,           # 2 INT8 ops/cycle/PE
            # INT4: 2× INT8
            Precision.INT4: 4,           # 4 INT4 ops/cycle/PE
        },
        core_frequency_hz=clock_hz,      # 2.0 GHz
        process_node_nm=7,               # 7nm (comparable to modern superscalar)
        energy_per_flop_fp32=get_base_alu_energy(7, 'simd_packed'),  # 1.62 pJ
        energy_scaling={
            Precision.FP32: 1.0,         # Baseline
            Precision.FP16: 0.50,        # Half precision
            Precision.BF16: 0.50,        # Brain float
            Precision.INT8: 0.20,        # INT8
            Precision.INT4: 0.10,        # INT4
        }
    )

    # Peak FP32: 8 PEs × 4 ops/cycle × 2.0 GHz = 64 GFLOPS ✓
    # Peak INT8: 8 PEs × 2 ops/cycle × 2.0 GHz = 32 GOPS ✓

    # Peak performance (with FMA)
    # 2 FP units × 8 FP32/cycle × 2 (FMA) × 2.0 GHz = 64 GFLOPS
    peak_ops_per_sec_fp32 = 64e9

    # Integer performance
    # 4 INT units × 4 INT ops/cycle × 2.0 GHz = 32 GOPS
    peak_ops_per_sec_int8 = 32e9

    # Memory bandwidth (single DDR4-3200 channel)
    peak_memory_bandwidth = 25.6e9  # bytes/sec

    # Cache sizes
    l1_cache_size = 64 * 1024  # 64 KB
    l2_cache_size = 512 * 1024  # 512 KB
    total_memory = 16 * 1024**3  # Assume 16 GB system memory

    # Energy parameters (based on 7nm process)
    energy_per_flop_fp32 = 2.5e-12  # 2.5 pJ per FLOP (similar to modern CPU)
    energy_per_byte = 15.0e-12  # 15 pJ per byte (DDR4 memory access)

    # Create thermal profiles
    thermal_profiles = {
        # Low power mode (embedded usage)
        '15W': ThermalOperatingPoint(
            name='15W-low-power',
            tdp_watts=15.0,
            cooling_solution='passive-heatsink',
            performance_specs={}  # Uses main precision profiles
        ),

        # Nominal mode (default)
        '25W': ThermalOperatingPoint(
            name='25W-nominal',
            tdp_watts=25.0,
            cooling_solution='active-fan',
            performance_specs={}  # Uses main precision profiles
        ),

        # Performance mode
        '35W': ThermalOperatingPoint(
            name='35W-performance',
            tdp_watts=35.0,
            cooling_solution='active-fan-enhanced',
            performance_specs={}  # Uses main precision profiles
        ),
    }

    return HardwareResourceModel(
        name='DFM-128',
        hardware_type=HardwareType.CPU,  # Closest match (general-purpose compute)

        # NEW: Multi-fabric architecture (Dataflow Processing Elements)
        compute_fabrics=[pe_fabric],

        # Compute resources
        compute_units=compute_units,
        threads_per_unit=1,  # Each PE executes one token at a time
        warps_per_unit=1,  # No warp concept in DFM
        warp_size=1,  # Single token execution

        # Precision profiles
        precision_profiles={
            Precision.FP32: PrecisionProfile(
                precision=Precision.FP32,
                peak_ops_per_sec=peak_ops_per_sec_fp32,  # 64 GFLOPS
                tensor_core_supported=False,
                relative_speedup=1.0,
                bytes_per_element=4,
            ),
            Precision.FP16: PrecisionProfile(
                precision=Precision.FP16,
                peak_ops_per_sec=peak_ops_per_sec_fp32 * 2,  # 128 GFLOPS
                tensor_core_supported=False,
                relative_speedup=2.0,
                bytes_per_element=2,
            ),
            Precision.BF16: PrecisionProfile(
                precision=Precision.BF16,
                peak_ops_per_sec=peak_ops_per_sec_fp32 * 2,  # 128 GFLOPS
                tensor_core_supported=False,
                relative_speedup=2.0,
                bytes_per_element=2,
            ),
            Precision.INT8: PrecisionProfile(
                precision=Precision.INT8,
                peak_ops_per_sec=peak_ops_per_sec_int8,  # 32 GOPS
                tensor_core_supported=False,
                relative_speedup=1.0,  # Relative to INT baseline
                bytes_per_element=1,
                accumulator_precision=Precision.INT32,
            ),
            Precision.INT4: PrecisionProfile(
                precision=Precision.INT4,
                peak_ops_per_sec=peak_ops_per_sec_int8 * 2,  # 64 GOPS
                tensor_core_supported=False,
                relative_speedup=2.0,
                bytes_per_element=0.5,
                accumulator_precision=Precision.INT32,
            ),
        },
        default_precision=Precision.FP32,

        # Memory hierarchy
        peak_bandwidth=peak_memory_bandwidth,
        l1_cache_per_unit=l1_cache_size // compute_units,  # 8 KB per PE
        l2_cache_total=l2_cache_size,
        main_memory=total_memory,

        # Energy model
        energy_per_flop_fp32=pe_fabric.energy_per_flop_fp32,  # 1.62 pJ (7nm, simd_packed)
        energy_per_byte=energy_per_byte,

        # Scheduling
        min_occupancy=0.25,
        max_concurrent_kernels=compute_units,  # 8 PEs
        wave_quantization=1,

        # Thermal profiles
        thermal_operating_points=thermal_profiles,
    )
