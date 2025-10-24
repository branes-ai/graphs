"""
Hailo Hardware Mapper - Hailo-8 and Hailo-10H Neural Processors

This module implements mappers for Hailo's dataflow-based neural processors:
- Hailo-8: Computer vision optimized (26 TOPS INT8)
- Hailo-10H: Transformer/LLM optimized (40 TOPS INT4, 20 TOPS INT8)

Hailo Architecture:
- Structure-driven dataflow architecture
- Distributed on-chip memory fabric
- No Von Neumann bottleneck
- Custom compilation per network topology
- Native INT8/INT4 quantization (no FP16/BF16)

Key Features:
- Hailo-8: All on-chip (no DRAM), CNN-optimized, 2.5W typical
- Hailo-10H: 4-8GB LPDDR4X, transformer-optimized, KV cache, 2.5W typical

Use Case: Embodied AI for drones and robots with ~10W power budget
"""

from dataclasses import dataclass
from typing import List, Tuple

from ...resource_model import (
    HardwareMapper,
    HardwareResourceModel,
    HardwareType,
    HardwareAllocation,
    GraphHardwareAllocation,
    Precision,
    PrecisionProfile,
    ClockDomain,
    ComputeResource,
    ThermalOperatingPoint,
    PerformanceCharacteristics,
    BottleneckType,
)
from graphs.transform.partitioning import FusedSubgraph, FusionReport
from graphs.ir.structures import SubgraphDescriptor, ParallelismDescriptor


class HailoMapper(HardwareMapper):
    """
    Hailo hardware mapper using dataflow architecture.

    Implements realistic Hailo mapping considering:
    - Dataflow compilation (resource graph mapping)
    - Distributed memory fabric
    - INT8/INT4 quantization efficiency
    - Network-specific resource allocation
    - Power efficiency constraints
    """

    def __init__(self, resource_model: HardwareResourceModel):
        super().__init__(resource_model)

        # Validate this is a KPU (Hailo) model
        if resource_model.hardware_type != HardwareType.KPU:
            raise ValueError(f"HailoMapper requires KPU resource model, got {resource_model.hardware_type}")

        # Hailo-specific parameters
        self.dataflow_cores = resource_model.compute_units  # Number of dataflow processing elements

    def map_subgraph(
        self,
        subgraph: FusedSubgraph,
        execution_stage: int,
        concurrent_subgraphs: int,
        precision: Precision = Precision.INT8
    ) -> HardwareAllocation:
        """
        Map a single fused subgraph to Hailo dataflow architecture.

        Algorithm:
        1. Determine resource requirements (dataflow mapping)
        2. Calculate memory bandwidth needs
        3. Estimate latency using dataflow efficiency
        4. Account for quantization overhead
        """
        # Get parallelism
        if subgraph.parallelism is None:
            # Fallback: assume single-core execution
            units_required = 1
        else:
            # Hailo's dataflow compiler allocates resources per layer
            # Parallelism comes from layer-level distribution
            batch = subgraph.parallelism.batch
            channels = subgraph.parallelism.channels

            # Estimate dataflow resource allocation
            # Larger batch/channel counts benefit from more distributed resources
            units_required = min(batch * max(1, channels // 64), self.dataflow_cores)

        units_allocated = max(1, min(units_required, self.dataflow_cores))

        # Calculate occupancy (what fraction of dataflow resources are busy)
        occupancy = units_allocated / self.dataflow_cores

        # Calculate utilization
        utilization = occupancy

        # Calculate latency using dataflow roofline model
        ops = subgraph.total_flops if subgraph.total_flops > 0 else subgraph.total_macs * 2
        bytes_transferred = (
            subgraph.total_input_bytes +
            subgraph.total_output_bytes +
            subgraph.total_weight_bytes
        )

        # Hailo's dataflow architecture minimizes external memory access
        # Most operations happen on distributed on-chip memory
        # Only input/output and weights need external transfers
        external_bytes = bytes_transferred

        compute_time, memory_time, bottleneck = self._calculate_latency(
            ops=ops,
            bytes_transferred=external_bytes,
            allocated_units=units_allocated,
            occupancy=occupancy,
            precision=precision
        )

        # Dataflow compilation overhead (negligible at runtime, done offline)
        # No overhead during inference

        estimated_latency = max(compute_time, memory_time)

        # Calculate energy
        compute_energy, memory_energy = self._calculate_energy(
            ops=ops,
            bytes_transferred=external_bytes,
            precision=precision
        )
        total_energy = compute_energy + memory_energy

        # Check if this can run in parallel with others
        is_parallel = concurrent_subgraphs > 1

        return HardwareAllocation(
            subgraph_id=str(subgraph.subgraph_id),
            subgraph_name=", ".join(subgraph.node_names[:2]),
            precision=precision,
            threads_required=units_allocated,  # Dataflow "threads"
            warps_required=0,  # N/A for dataflow
            compute_units_allocated=units_allocated,
            compute_units_ideal=units_required,
            occupancy=occupancy,
            utilization=utilization,
            bottleneck=bottleneck,
            compute_time=compute_time,
            memory_time=memory_time,
            estimated_latency=estimated_latency,
            compute_energy=compute_energy,
            memory_energy=memory_energy,
            total_energy=total_energy,
            execution_stage=execution_stage,
            is_parallel=is_parallel,
        )

    def map_graph(
        self,
        fusion_report: FusionReport,
        execution_stages: List[List[int]],
        batch_size: int = 1,
        precision: Precision = Precision.INT8
    ) -> GraphHardwareAllocation:
        """
        Map entire computation graph to Hailo dataflow architecture.

        Args:
            fusion_report: Output from Phase 1 fusion partitioner
            execution_stages: Execution stages with subgraph indices
            batch_size: Batch size (scales parallelism)
            precision: Numerical precision

        Returns:
            Complete hardware allocation
        """
        subgraph_allocations: List[HardwareAllocation] = []
        latency_breakdown: dict[int, float] = {}

        peak_units_used = 0
        total_units_used = 0
        total_units_samples = 0

        # Process each execution stage
        for stage_id, subgraph_indices in enumerate(execution_stages):
            stage_allocations = []
            concurrent_subgraphs = len(subgraph_indices)

            # Map each subgraph in this stage
            for subgraph_idx in subgraph_indices:
                if subgraph_idx >= len(fusion_report.fused_subgraphs):
                    continue

                subgraph = fusion_report.fused_subgraphs[subgraph_idx]

                # Scale parallelism by batch size
                if subgraph.parallelism and batch_size > 1:
                    import copy
                    subgraph = copy.copy(subgraph)
                    subgraph.parallelism = copy.copy(subgraph.parallelism)
                    subgraph.parallelism.batch *= batch_size
                    subgraph.parallelism.total_threads *= batch_size

                allocation = self.map_subgraph(
                    subgraph=subgraph,
                    execution_stage=stage_id,
                    concurrent_subgraphs=concurrent_subgraphs,
                    precision=precision
                )
                stage_allocations.append(allocation)
                subgraph_allocations.append(allocation)

            # For parallel subgraphs: max units used, max latency
            if stage_allocations:
                stage_units_used = max(a.compute_units_allocated for a in stage_allocations)
                stage_latency = max(a.estimated_latency for a in stage_allocations)

                peak_units_used = max(peak_units_used, stage_units_used)
                total_units_used += stage_units_used
                total_units_samples += 1

                latency_breakdown[stage_id] = stage_latency

        # Calculate aggregate metrics
        total_subgraphs = len(subgraph_allocations)
        total_execution_stages = len(execution_stages)
        average_units_used = total_units_used / total_units_samples if total_units_samples > 0 else 0

        total_units = self.resource_model.compute_units
        peak_utilization = peak_units_used / total_units
        average_utilization = average_units_used / total_units

        # Total latency = sum of stage latencies (sequential)
        total_latency = sum(latency_breakdown.values())

        # Total energy
        total_energy = sum(a.total_energy for a in subgraph_allocations)

        # Naive latency (assuming 100% utilization)
        total_ops = fusion_report.total_flops
        peak_ops_per_sec = self.resource_model.get_peak_ops(precision)
        naive_latency = total_ops / peak_ops_per_sec if peak_ops_per_sec > 0 else 0

        # Correction factor
        latency_correction_factor = total_latency / naive_latency if naive_latency > 0 else 1.0

        # Bottleneck analysis
        from graphs.ir.structures import BottleneckType
        compute_bound_count = sum(1 for a in subgraph_allocations if a.bottleneck == BottleneckType.COMPUTE_BOUND)
        memory_bound_count = sum(1 for a in subgraph_allocations if a.bottleneck == BottleneckType.MEMORY_BOUND)
        bandwidth_bound_count = sum(1 for a in subgraph_allocations if a.bottleneck == BottleneckType.BANDWIDTH_BOUND)
        balanced_count = sum(1 for a in subgraph_allocations if a.bottleneck == BottleneckType.BALANCED)

        return GraphHardwareAllocation(
            model_name="Unknown",
            hardware_name=self.resource_model.name,
            batch_size=batch_size,
            model_precision=precision,
            subgraph_allocations=subgraph_allocations,
            total_subgraphs=total_subgraphs,
            total_execution_stages=total_execution_stages,
            peak_compute_units_used=peak_units_used,
            average_compute_units_used=average_units_used,
            peak_utilization=peak_utilization,
            average_utilization=average_utilization,
            total_latency=total_latency,
            latency_breakdown=latency_breakdown,
            total_energy=total_energy,
            naive_latency=naive_latency,
            latency_correction_factor=latency_correction_factor,
            compute_bound_count=compute_bound_count,
            memory_bound_count=memory_bound_count,
            bandwidth_bound_count=bandwidth_bound_count,
            balanced_count=balanced_count,
        )


def create_hailo8_mapper() -> HailoMapper:
    """
    Create hardware mapper for Hailo-8 AI processor.

    ARCHITECTURE:
    - Dataflow-based neural processor (no Von Neumann bottleneck)
    - Distributed on-chip memory fabric
    - ALL computation happens on-chip (no external DRAM!)
    - Custom compilation per network topology
    - 16nm TSMC process

    PERFORMANCE:
    - 26 TOPS INT8
    - 2.5W typical power (8.65W TDP max)
    - 672 fps on ResNet-50
    - 2.8 TOPS/W efficiency

    PRECISION SUPPORT:
    - INT8: Native, primary mode (a8_w8)
    - INT4: Supported for weights (a8_w4)
    - FP16/BF16: NOT supported (INT-only architecture)

    MEMORY:
    - All on-chip distributed SRAM
    - No external DRAM interface
    - Exact SRAM capacity: proprietary (not disclosed)
    - Estimated: ~10-20 MB total on-chip

    INTERFACE:
    - PCIe Gen 3.0 x2 or x4 (depending on module)
    - M.2 2242/2280 form factors available

    USE CASE:
    - Computer vision: object detection, segmentation, tracking
    - CNN-optimized workloads
    - Drone/robot vision at 2.5W
    - Real-time inference with minimal latency

    CALIBRATION STATUS:
    ⚠ ESTIMATED - Based on published specs and architecture analysis
    - efficiency_factor values are conservative estimates
    - Need empirical benchmarking to refine coefficients
    """
    from ...resource_model import (
        HardwareResourceModel,
        HardwareType,
        Precision,
        PrecisionProfile,
        ClockDomain,
        ComputeResource,
        ThermalOperatingPoint,
        PerformanceCharacteristics,
    )

    # ========================================================================
    # CLOCK DOMAIN (Dataflow architecture - not traditional clocked design)
    # ========================================================================
    # Hailo uses a dataflow architecture, but we model with effective clock
    clock = ClockDomain(
        base_clock_hz=800e6,       # Estimated: ~800 MHz
        max_boost_clock_hz=1.0e9,  # Estimated: ~1 GHz
        sustained_clock_hz=900e6,  # Estimated: ~900 MHz sustained
        dvfs_enabled=False,         # Dataflow arch doesn't use traditional DVFS
    )

    # ========================================================================
    # COMPUTE RESOURCE (Dataflow processing elements)
    # ========================================================================
    # Hailo distributes compute across many small processing elements
    # We model as equivalent "cores" for compatibility
    # Actual architecture is much more fine-grained

    # 26 TOPS INT8 @ 900 MHz sustained
    # → 26e12 ops/sec / 900e6 Hz = 28,888 ops/cycle
    # If we model as 64 "dataflow units":
    # → 28,888 / 64 = 451 ops/cycle/unit

    num_dataflow_units = 64  # Estimated equivalent processing elements

    dataflow_compute = ComputeResource(
        resource_type="Hailo-Dataflow-Element",
        num_units=num_dataflow_units,
        ops_per_unit_per_clock={
            Precision.INT8: 450,    # 450 INT8 ops/cycle/unit
            Precision.INT4: 900,    # 900 INT4 ops/cycle/unit (2× INT8)
            Precision.INT16: 225,   # 225 INT16 ops/cycle/unit (0.5× INT8)
        },
        clock_domain=clock,
    )

    # Peak INT8: 64 units × 450 ops/cycle × 900 MHz = 25.92 TOPS ≈ 26 TOPS ✓

    # ========================================================================
    # THERMAL PROFILE (Low-power edge inference)
    # ========================================================================
    thermal_profile = ThermalOperatingPoint(
        name="edge-inference-2.5W",
        tdp_watts=2.5,
        cooling_solution="passive",
        performance_specs={
            Precision.INT8: PerformanceCharacteristics(
                precision=Precision.INT8,
                compute_resource=dataflow_compute,
                instruction_efficiency=0.95,  # Dataflow is very efficient
                memory_bottleneck_factor=0.90,  # All on-chip, minimal bottleneck
                efficiency_factor=0.85,  # 85% of sustained (excellent for edge)
                tile_utilization=0.90,  # Good resource allocation by compiler
                native_acceleration=True,
            ),
            Precision.INT4: PerformanceCharacteristics(
                precision=Precision.INT4,
                compute_resource=dataflow_compute,
                instruction_efficiency=0.92,  # Slightly lower than INT8
                memory_bottleneck_factor=0.88,
                efficiency_factor=0.80,  # 80% (4-bit has more overhead)
                tile_utilization=0.88,
                native_acceleration=True,
            ),
            Precision.INT16: PerformanceCharacteristics(
                precision=Precision.INT16,
                compute_resource=dataflow_compute,
                instruction_efficiency=0.90,
                memory_bottleneck_factor=0.85,
                efficiency_factor=0.75,  # 75% (less optimized than INT8)
                tile_utilization=0.85,
                native_acceleration=True,
            ),
        }
    )

    # ========================================================================
    # HARDWARE RESOURCE MODEL
    # ========================================================================
    model = HardwareResourceModel(
        name="Hailo-8-Vision-Processor",
        hardware_type=HardwareType.KPU,
        compute_units=num_dataflow_units,
        threads_per_unit=1,  # Each dataflow unit is independent
        warps_per_unit=1,
        warp_size=1,  # N/A for dataflow

        # Precision profiles (for legacy compatibility)
        precision_profiles={
            Precision.INT8: PrecisionProfile(
                precision=Precision.INT8,
                peak_ops_per_sec=26e12,  # 26 TOPS
                tensor_core_supported=True,  # Dataflow accelerator
                relative_speedup=1.0,
                bytes_per_element=1,
            ),
            Precision.INT4: PrecisionProfile(
                precision=Precision.INT4,
                peak_ops_per_sec=52e12,  # 52 TOPS INT4 (2× INT8)
                tensor_core_supported=True,
                relative_speedup=2.0,
                bytes_per_element=0.5,
            ),
        },
        default_precision=Precision.INT8,

        # ====================================================================
        # MEMORY HIERARCHY - Hailo-8 All-On-Chip Design
        # ====================================================================
        # Hailo-8 has NO external DRAM!
        # All memory is distributed on-chip SRAM
        #
        # Architecture:
        # - Distributed memory fabric integrated with compute
        # - Each layer gets custom memory allocation during compilation
        # - No Von Neumann bottleneck (data stays local)
        #
        # Estimated memory (not disclosed by Hailo):
        # - ~10-20 MB total on-chip SRAM
        # - Distributed across processing elements
        # - Very high bandwidth (on-chip, not limited by DRAM)
        #
        # For modeling purposes:
        # - l1_cache_per_unit: local SRAM per processing element
        # - l2_cache_total: shared memory pool
        # - main_memory: 0 (no DRAM!)
        # - peak_bandwidth: on-chip bandwidth (very high)
        # ====================================================================
        peak_bandwidth=200e9,  # 200 GB/s on-chip bandwidth (estimated)
        l1_cache_per_unit=256 * 1024,  # 256 KB per dataflow unit (estimated)
        l2_cache_total=16 * 1024 * 1024,  # 16 MB shared pool (estimated)
        main_memory=0,  # NO DRAM! (all on-chip)

        # Energy (ultra-low power for edge)
        energy_per_flop_fp32=0,  # N/A (no FP32)
        energy_per_byte=5e-12,  # 5 pJ/byte (very low, on-chip)

        # Scheduling (dataflow compilation, not runtime scheduling)
        min_occupancy=0.70,  # Compiler achieves good utilization
        max_concurrent_kernels=32,  # Estimated concurrent layer executions
        wave_quantization=1,

        # Thermal profiles
        thermal_operating_points={
            "edge-inference-2.5W": thermal_profile,
        },
        default_thermal_profile="edge-inference-2.5W",
    )

    return HailoMapper(model)


def create_hailo10h_mapper() -> HailoMapper:
    """
    Create hardware mapper for Hailo-10H AI accelerator.

    ARCHITECTURE:
    - 2nd generation dataflow neural processor
    - Enhanced for transformers and LLMs
    - External DRAM interface (4-8GB LPDDR4X)
    - KV cache support for token generation
    - Process node: likely 12nm or better (not disclosed)

    PERFORMANCE:
    - 40 TOPS INT4 (generative AI optimized)
    - 20 TOPS INT8
    - 2.5W typical power
    - 16 TOPS/W efficiency @ INT4

    PRECISION SUPPORT:
    - INT4: Primary for LLM weights (QuaROT, GPTQ quantization)
    - INT8: For activations and traditional CNNs
    - FP16/BF16: NOT supported (INT-only architecture)

    MEMORY:
    - 4GB or 8GB LPDDR4X on-module
    - ~17 GB/s bandwidth (LPDDR4X spec)
    - Direct DDR interface for large model support
    - On-chip SRAM for intermediate results

    INTERFACE:
    - PCIe Gen 3.0 x4
    - M.2 2242/2280 form factors

    USE CASE:
    - Transformer inference (BERT, GPT, LLaMA)
    - Vision-Language Models (VLMs)
    - Stable Diffusion
    - Multi-modal AI at edge
    - Drone/robot intelligence with language understanding

    KEY FEATURES:
    - KV cache for autoregressive generation
    - Optimized attention mechanisms
    - Low-latency token generation
    - Power-efficient generative AI

    CALIBRATION STATUS:
    ⚠ ESTIMATED - Based on published specs and architecture analysis
    - efficiency_factor values are conservative estimates
    - LLM-specific optimizations not yet benchmarked
    """
    from ...resource_model import (
        HardwareResourceModel,
        HardwareType,
        Precision,
        PrecisionProfile,
        ClockDomain,
        ComputeResource,
        ThermalOperatingPoint,
        PerformanceCharacteristics,
    )

    # ========================================================================
    # CLOCK DOMAIN (2nd gen dataflow core)
    # ========================================================================
    clock = ClockDomain(
        base_clock_hz=900e6,       # Estimated: ~900 MHz
        max_boost_clock_hz=1.2e9,  # Estimated: ~1.2 GHz
        sustained_clock_hz=1.1e9,  # Estimated: ~1.1 GHz sustained
        dvfs_enabled=False,
    )

    # ========================================================================
    # COMPUTE RESOURCE (Enhanced dataflow for transformers)
    # ========================================================================
    # 40 TOPS INT4 @ 1.1 GHz sustained
    # → 40e12 ops/sec / 1.1e9 Hz = 36,363 ops/cycle
    # If we model as 128 "dataflow units" (larger than Hailo-8):
    # → 36,363 / 128 = 284 ops/cycle/unit INT4

    num_dataflow_units = 128  # More processing elements than Hailo-8

    dataflow_compute = ComputeResource(
        resource_type="Hailo-10H-Dataflow-Element",
        num_units=num_dataflow_units,
        ops_per_unit_per_clock={
            Precision.INT4: 280,    # 280 INT4 ops/cycle/unit
            Precision.INT8: 140,    # 140 INT8 ops/cycle/unit (0.5× INT4)
            Precision.INT16: 70,    # 70 INT16 ops/cycle/unit (0.25× INT4)
        },
        clock_domain=clock,
    )

    # Peak INT4: 128 units × 280 ops/cycle × 1.1 GHz = 39.42 TOPS ≈ 40 TOPS ✓
    # Peak INT8: 128 units × 140 ops/cycle × 1.1 GHz = 19.71 TOPS ≈ 20 TOPS ✓

    # ========================================================================
    # THERMAL PROFILE (Low-power generative AI)
    # ========================================================================
    thermal_profile = ThermalOperatingPoint(
        name="edge-genai-2.5W",
        tdp_watts=2.5,
        cooling_solution="passive",
        performance_specs={
            Precision.INT4: PerformanceCharacteristics(
                precision=Precision.INT4,
                compute_resource=dataflow_compute,
                instruction_efficiency=0.92,  # Transformer-optimized
                memory_bottleneck_factor=0.75,  # DRAM interface limits
                efficiency_factor=0.80,  # 80% (LLMs are memory-bound)
                tile_utilization=0.88,
                native_acceleration=True,
            ),
            Precision.INT8: PerformanceCharacteristics(
                precision=Precision.INT8,
                compute_resource=dataflow_compute,
                instruction_efficiency=0.94,
                memory_bottleneck_factor=0.78,
                efficiency_factor=0.82,  # 82% (better than INT4 for CNN)
                tile_utilization=0.90,
                native_acceleration=True,
            ),
            Precision.INT16: PerformanceCharacteristics(
                precision=Precision.INT16,
                compute_resource=dataflow_compute,
                instruction_efficiency=0.88,
                memory_bottleneck_factor=0.70,
                efficiency_factor=0.70,  # 70% (less optimized)
                tile_utilization=0.85,
                native_acceleration=True,
            ),
        }
    )

    # ========================================================================
    # HARDWARE RESOURCE MODEL
    # ========================================================================
    model = HardwareResourceModel(
        name="Hailo-10H-GenAI-Processor",
        hardware_type=HardwareType.KPU,
        compute_units=num_dataflow_units,
        threads_per_unit=1,
        warps_per_unit=1,
        warp_size=1,

        # Precision profiles
        precision_profiles={
            Precision.INT4: PrecisionProfile(
                precision=Precision.INT4,
                peak_ops_per_sec=40e12,  # 40 TOPS INT4
                tensor_core_supported=True,
                relative_speedup=2.0,  # vs INT8
                bytes_per_element=0.5,
            ),
            Precision.INT8: PrecisionProfile(
                precision=Precision.INT8,
                peak_ops_per_sec=20e12,  # 20 TOPS INT8
                tensor_core_supported=True,
                relative_speedup=1.0,
                bytes_per_element=1,
            ),
        },
        default_precision=Precision.INT4,  # INT4 primary for LLMs

        # ====================================================================
        # MEMORY HIERARCHY - Hailo-10H with External DRAM
        # ====================================================================
        # Hailo-10H has 4-8GB LPDDR4X on-module for large models
        #
        # Architecture:
        # - On-chip SRAM for intermediate results
        # - LPDDR4X for model weights and KV cache
        # - Direct DDR interface for large model support
        #
        # LPDDR4X bandwidth: ~17 GB/s per module
        # (Actual may be higher with multiple channels)
        # ====================================================================
        peak_bandwidth=17e9,  # 17 GB/s LPDDR4X
        l1_cache_per_unit=128 * 1024,  # 128 KB per dataflow unit (estimated)
        l2_cache_total=24 * 1024 * 1024,  # 24 MB shared on-chip (estimated)
        main_memory=8 * 1024**3,  # 8 GB LPDDR4X (max config)

        # Energy (ultra-low power for edge generative AI)
        energy_per_flop_fp32=0,  # N/A (no FP32)
        energy_per_byte=8e-12,  # 8 pJ/byte (LPDDR4X slightly higher than on-chip)

        # Scheduling
        min_occupancy=0.65,  # Transformers have variable layer utilization
        max_concurrent_kernels=64,  # More concurrent than Hailo-8
        wave_quantization=1,

        # Thermal profiles
        thermal_operating_points={
            "edge-genai-2.5W": thermal_profile,
        },
        default_thermal_profile="edge-genai-2.5W",
    )

    return HailoMapper(model)
