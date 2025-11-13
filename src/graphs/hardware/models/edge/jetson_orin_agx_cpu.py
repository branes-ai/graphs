"""
Jetson Orin AGX CPU Resource Model - ARM Cortex-A78AE 12-core configuration.

This models the ARM CPU subsystem inside the Jetson Orin AGX SoC, separate from
the GPU. Useful for comparing stored-program CPU execution against data-parallel
GPU and domain-flow KPU architectures.

MEMORY: Shared 64 GB LPDDR5 with GPU (partitioned for comparison)

Extracted to enable fair architecture comparisons at same power budget (30W).
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


def jetson_orin_agx_cpu_resource_model() -> HardwareResourceModel:
    """
    ARM Cortex-A78AE CPU inside Jetson Orin AGX (12-core configuration).

    ARCHITECTURE:
    - 12× ARM Cortex-A78AE cores @ 2.2 GHz sustained (all-core turbo)
    - ARMv8.2-A ISA with crypto + NEON extensions
    - ARM NEON SIMD: 128-bit vectors (Advanced SIMD)
      * 4× FP32 operations per cycle (FMA: 2 ops × 2 pipes)
      * 8× FP16 operations per cycle (FMA: 2 ops × 4 lanes)
      * 32× INT8 operations per cycle with dotprod (4 lanes × 8 elements)
    - Instruction Set: AArch64 with dotprod extension (INT8 dot products)
    - Out-of-order execution (128-entry ROB)
    - Branch prediction (TAGE predictor)

    CACHE HIERARCHY:
    - L1: 64 KB per core (32 KB I$ + 32 KB D$)
    - L2: 512 KB per core (private, inclusive of L1)
    - L3: 2 MB shared across all cores
    - Total cache: 6.75 MB (768 KB L1 + 6 MB L2 + 2 MB L3)

    MEMORY:
    - Shared LPDDR5 @ 204.8 GB/s total bandwidth
    - CPU allocation: ~51.2 GB/s (25% of total, rest for GPU/DLA)
    - ECC protected (automotive-grade reliability)

    POWER: 30W TDP (CPU-only, estimated allocation)

    REALITY CHECK - Power Budget:
    - Jetson Orin AGX total package: 60W max
    - GPU @ 30W mode: ~30W (GPU + memory controllers)
    - CPU @ 30W mode: ~20W (12 cores + L3 + fabric)
    - Remaining: ~10W (memory, I/O, misc)
    - For fair comparison @ 30W: CPU can use full 30W budget

    KEY CHARACTERISTICS:
    - Stored Program Architecture (explicit instruction fetch)
    - MIMD execution model (12-way thread-level parallelism)
    - Sequential control flow with branch prediction
    - Explicit memory loads/stores (no automatic dataflow)
    - Cache coherency protocol (ARM ACE - MESI variant)
    - Out-of-order execution hides instruction latency

    PERFORMANCE ESTIMATE (30W all-core sustained):
    - FP32 peak: 12 cores × 8 ops/cycle × 2.2 GHz = 211 GFLOPS
    - FP32 effective: 211 × 0.7 efficiency = 148 GFLOPS (~70% of peak)
    - INT8 peak (NEON dotprod): 12 cores × 32 ops/cycle × 2.2 GHz = 844 GOPS
    - INT8 effective: 844 × 0.6 efficiency = 506 GOPS (~60% due to memory bound)

    ENERGY CHARACTERISTICS:
    - Instruction fetch overhead: ~2 pJ per instruction
    - Memory request overhead: ~10 pJ per cache line (beyond data transfer)
    - Branch misprediction penalty: ~0.3 pJ per branch
    - Pipeline control: ~0.5 pJ per cycle
    - Less efficient than spatial architectures (GPU/KPU) due to
      sequential control flow and explicit instruction fetch

    USE CASE:
    - Host CPU for robotics control loops
    - Lightweight inference (batch=1)
    - General-purpose computation
    - Comparison baseline for architecture studies

    CALIBRATION STATUS:
    ⚠ ESTIMATED - Based on ARM Cortex-A78 public specs and Jetson docs
    - Clock speeds: Documented (Jetson Orin datasheet)
    - SIMD capabilities: ARM Cortex-A78 TRM (Technical Reference Manual)
    - Power budget: Estimated from 60W total package allocation
    - Performance: Scaled from ARM published benchmarks

    REFERENCES:
    - ARM Cortex-A78 Technical Reference Manual (ARM DDI 0593)
    - Jetson Orin Series SoC Technical Reference Manual
    - NVIDIA Jetson Orin Modules Datasheet (DA-10676-001)
    - ARM Neoverse V1 Performance (similar microarchitecture)
    """

    # ========================================================================
    # PHYSICAL CONFIGURATION
    # ========================================================================
    num_cores = 12  # Cortex-A78AE (automotive enhanced)
    simd_width_int8 = 16  # NEON 128-bit = 16 × INT8 elements
    simd_width_fp32 = 4   # NEON 128-bit = 4 × FP32 elements

    # ========================================================================
    # CLOCK DOMAIN - 30W Mode (All-Core Sustained)
    # ========================================================================
    # Cortex-A78AE specifications:
    # - Base: 1.2 GHz (minimum guaranteed)
    # - Single-core turbo: 2.6 GHz (brief bursts)
    # - All-core sustained: 2.2 GHz @ 30W (thermal equilibrium)
    clock_30w = ClockDomain(
        base_clock_hz=1.2e9,         # 1.2 GHz minimum
        max_boost_clock_hz=2.6e9,    # 2.6 GHz single-core boost
        sustained_clock_hz=2.2e9,    # 2.2 GHz all-core sustained
        dvfs_enabled=True,
    )

    # ========================================================================
    # COMPUTE RESOURCE - NEON SIMD
    # ========================================================================
    # ARM Cortex-A78 has 2 Advanced SIMD (NEON) pipelines:
    # - Pipeline 0: 128-bit ASIMD (FP, INT)
    # - Pipeline 1: 128-bit ASIMD (FP, INT)
    #
    # Operations per cycle per core:
    # - FP32: 4 elements × 2 ops (FMA) × 2 pipes = 8 ops/cycle (with NEON)
    # - FP16: 8 elements × 2 ops (FMA) × 2 pipes = 16 ops/cycle (actually limited to 8)
    # - INT8: Dotprod instruction: 4 lanes × (4 × 8-bit MACs) = 4 × 4 = 16 MACs/lane
    #         Total: 4 × 4 × 2 pipes = 32 INT8 ops/cycle
    #
    # NOTE: ARM dotprod is a 4-element dot product that processes 4 bytes at a time
    #       Result: (a0*b0 + a1*b1 + a2*b2 + a3*b3)
    #       Each dotprod counts as 4 MACs (8 INT8 ops)

    compute_resource = ComputeResource(
        resource_type="ARM-Cortex-A78AE-NEON",
        num_units=num_cores,
        ops_per_unit_per_clock={
            Precision.INT8: 32,   # NEON dotprod: 4 lanes × 4 MACs × 2 pipes
            Precision.FP16: 8,    # NEON FP16: 8-wide SIMD (conservative)
            Precision.FP32: 8,    # NEON FP32: 4-wide × 2 ops (FMA) × 2 pipes
        },
        clock_domain=clock_30w,
    )

    # ========================================================================
    # THERMAL OPERATING POINT - 30W CPU-Only
    # ========================================================================
    thermal_30w = ThermalOperatingPoint(
        name="30W-cpu",
        tdp_watts=30.0,
        cooling_solution="active-fan",
        performance_specs={
            Precision.INT8: PerformanceCharacteristics(
                precision=Precision.INT8,
                compute_resource=compute_resource,
                instruction_efficiency=0.70,  # 70% efficiency (instruction fetch overhead)
                memory_bottleneck_factor=0.80,  # 80% limited by memory bandwidth
                efficiency_factor=0.60,  # Overall: 60% of peak (memory-bound AI workloads)
                native_acceleration=True,  # NEON dotprod native
            ),
            Precision.FP16: PerformanceCharacteristics(
                precision=Precision.FP16,
                compute_resource=compute_resource,
                instruction_efficiency=0.70,
                memory_bottleneck_factor=0.75,  # More memory-bound than INT8
                efficiency_factor=0.55,  # 55% of peak
                native_acceleration=True,  # NEON FP16 native
            ),
            Precision.FP32: PerformanceCharacteristics(
                precision=Precision.FP32,
                compute_resource=compute_resource,
                instruction_efficiency=0.70,
                memory_bottleneck_factor=0.70,  # Most memory-bound
                efficiency_factor=0.50,  # 50% of peak (typical for AI on CPU)
                native_acceleration=True,  # NEON FP32 native
            ),
        }
    )

    # ========================================================================
    # PRECISION PROFILES (Legacy - for backward compatibility)
    # ========================================================================
    # Peak performance calculations (theoretical maximum):
    # INT8: 12 cores × 32 ops/cycle × 2.2 GHz = 844.8 GOPS
    # FP32: 12 cores × 8 ops/cycle × 2.2 GHz = 211.2 GFLOPS
    # FP16: 12 cores × 8 ops/cycle × 2.2 GHz = 211.2 GFLOPS

    precision_profiles = {
        Precision.INT8: PrecisionProfile(
            precision=Precision.INT8,
            peak_ops_per_sec=844.8e9,  # 844.8 GOPS INT8
            tensor_core_supported=False,  # CPU uses NEON SIMD, not tensor cores
            bytes_per_element=1,
        ),
        Precision.FP16: PrecisionProfile(
            precision=Precision.FP16,
            peak_ops_per_sec=211.2e9,  # 211.2 GFLOPS FP16
            tensor_core_supported=False,
            bytes_per_element=2,
        ),
        Precision.FP32: PrecisionProfile(
            precision=Precision.FP32,
            peak_ops_per_sec=211.2e9,  # 211.2 GFLOPS FP32
            tensor_core_supported=False,
            bytes_per_element=4,
        ),
    }

    # ========================================================================
    # BOM COST (CPU portion of Jetson Orin AGX)
    # ========================================================================
    # Note: This is the incremental cost for the CPU subsystem within the
    # larger Jetson Orin AGX SoC ($899 retail). Not a standalone product.
    bom_cost = BOMCostProfile(
        silicon_die_cost=50.0,  # CPU die area fraction (~13% of total die)
        package_cost=0.0,  # Shared with GPU (counted in GPU model)
        memory_cost=0.0,  # Shared LPDDR5 (counted in GPU model)
        pcb_assembly_cost=0.0,  # Shared (counted in GPU model)
        thermal_solution_cost=0.0,  # Shared heatsink (counted in GPU model)
        other_costs=10.0,  # L3 cache, fabric interconnect
        total_bom_cost=60.0,  # Incremental CPU portion
        margin_multiplier=1.0,  # Part of larger SoC, no separate margin
        retail_price=0.0,  # Not sold separately
        volume_tier="Integrated",
        process_node="8nm",
        year=2025,
        notes="ARM Cortex-A78AE CPU subsystem within Jetson Orin AGX. 12 cores for host processing and lightweight inference. Shared memory/package with GPU.",
    )

    # ========================================================================
    # HARDWARE RESOURCE MODEL
    # ========================================================================
    return HardwareResourceModel(
        name="Jetson-Orin-AGX-CPU",
        hardware_type=HardwareType.CPU,
        compute_units=num_cores,
        threads_per_unit=1,  # No SMT (Simultaneous Multi-Threading) in Cortex-A78AE
        warps_per_unit=1,  # No warp concept in CPUs (one thread per core)
        warp_size=simd_width_int8,  # SIMD width for CPUMapper (16-wide INT8)

        # Thermal operating points
        thermal_operating_points={"30W": thermal_30w},
        default_thermal_profile="30W",

        # Precision profiles (legacy)
        precision_profiles=precision_profiles,
        default_precision=Precision.INT8,

        # Memory hierarchy (shared with GPU, but modeled separately for CPU)
        peak_bandwidth=51.2e9,  # 51.2 GB/s (25% of 204.8 GB/s LPDDR5 bandwidth)
        l1_cache_per_unit=64 * 1024,  # 64 KB per core (32 KB I$ + 32 KB D$)
        l2_cache_total=8 * 1024 * 1024,  # 8 MB total (512 KB × 12 cores L2 + 2 MB shared L3)
        main_memory=64 * 1024**3,  # Shared 64 GB LPDDR5

        # Energy coefficients
        energy_per_flop_fp32=2.09e-11,  # ~20.9 pJ/FLOP (ARM Cortex-A78AE, embedded RISC)
        energy_per_byte=20e-12,  # ~20 pJ per byte (DRAM access)

        # Energy scaling by precision
        energy_scaling={
            Precision.FP64: 2.0,  # 2× FP32 (double precision)
            Precision.FP32: 1.0,  # Baseline
            Precision.FP16: 0.5,  # 0.5× FP32 (half precision)
            Precision.INT8: 0.25,  # 0.25× FP32 (integer)
        },

        # Occupancy and concurrency
        min_occupancy=0.5,  # Minimum 50% core utilization for efficiency
        max_concurrent_kernels=12,  # One thread per core (no SMT)
        wave_quantization=1,  # CPUs allocate cores individually (no wave grouping)

        # BOM cost
        bom_cost_profile=bom_cost,
    )
