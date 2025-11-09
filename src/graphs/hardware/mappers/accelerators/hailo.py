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
        precision: Precision = Precision.INT8,
        model_size_bytes: int = 0
    ) -> GraphHardwareAllocation:
        """
        Map entire computation graph to Hailo dataflow architecture.

        NOTE: For Hailo-8, if model weights don't fit in on-chip SRAM (8 MB),
        we must stream weights layer-by-layer from host DRAM via PCIe.
        This adds significant latency and energy overhead.

        Args:
            fusion_report: Output from Phase 1 fusion partitioner
            execution_stages: Execution stages with subgraph indices
            batch_size: Batch size (scales parallelism)
            precision: Numerical precision
            model_size_bytes: Total model weight size in bytes (for PCIe overhead calculation)

        Returns:
            Complete hardware allocation
        """
        subgraph_allocations: List[HardwareAllocation] = []
        latency_breakdown: dict[int, float] = {}

        peak_units_used = 0
        total_units_used = 0
        total_units_samples = 0

        # Calculate total weight size for PCIe streaming overhead check
        # Use provided model_size_bytes if available, otherwise try to get from subgraphs
        # fusion_report can have either .fused_subgraphs or .subgraphs
        subgraphs = getattr(fusion_report, 'fused_subgraphs', None) or getattr(fusion_report, 'subgraphs', [])
        if model_size_bytes > 0:
            total_weight_bytes = model_size_bytes
        else:
            total_weight_bytes = sum(
                sg.total_weight_bytes
                for sg in subgraphs
            )

        # Check if model fits on-chip (Hailo-8 specific constraint)
        on_chip_memory_bytes = self.resource_model.l2_cache_total if self.resource_model.main_memory == 0 else float('inf')
        requires_pcie_streaming = total_weight_bytes > on_chip_memory_bytes


        # PCIe Gen3 x4 parameters (typical for Hailo-8 M.2 module)
        pcie_bandwidth_bytes_per_sec = 4e9  # 4 GB/s (PCIe Gen3 x4)
        pcie_energy_per_byte = 25e-12  # 25 pJ/byte (PCIe transfer energy)

        # Process each execution stage
        for stage_id, subgraph_indices in enumerate(execution_stages):
            stage_allocations = []
            concurrent_subgraphs = len(subgraph_indices)

            # Map each subgraph in this stage
            for subgraph_idx in subgraph_indices:
                if subgraph_idx >= len(subgraphs):
                    continue

                subgraph = subgraphs[subgraph_idx]

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

                # Add PCIe streaming overhead if model doesn't fit on-chip
                if requires_pcie_streaming:
                    # Transfer time for this layer's weights from host DRAM via PCIe
                    # For now, distribute total weight transfer evenly across all subgraphs
                    # (In reality, weight transfer happens layer-by-layer)
                    weight_bytes_per_subgraph = total_weight_bytes / len(subgraphs)
                    pcie_transfer_time = weight_bytes_per_subgraph / pcie_bandwidth_bytes_per_sec
                    pcie_transfer_energy = weight_bytes_per_subgraph * pcie_energy_per_byte

                    # Add overhead to allocation (modifying the object)
                    allocation.estimated_latency += pcie_transfer_time
                    allocation.memory_time += pcie_transfer_time
                    allocation.memory_energy += pcie_transfer_energy
                    allocation.total_energy += pcie_transfer_energy

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
    - Spatial graph mapping architecture (not dataflow)
    - Distributed on-chip memory fabric
    - ALL computation happens on-chip (no external DRAM!)
    - Network compiled to hardware configuration
    - 16nm TSMC process

    PERFORMANCE:
    - 26 TOPS INT8
    - 2.5W typical power
    - Excellent power efficiency

    USE CASE:
    - Computer vision: object detection, segmentation, tracking
    - CNN-optimized workloads
    - Drone/robot vision at 2.5W
    """
    from ...models.edge.hailo8 import hailo8_resource_model
    return HailoMapper(hailo8_resource_model())


def create_hailo10h_mapper() -> HailoMapper:
    """
    Create hardware mapper for Hailo-10H AI accelerator.

    ARCHITECTURE:
    - 2nd generation spatial graph mapper
    - Enhanced for transformers and LLMs
    - External DRAM interface (4-8GB LPDDR4X)

    PERFORMANCE:
    - 40 TOPS INT4, 20 TOPS INT8
    - 2.5W typical power

    USE CASE:
    - Transformer inference, vision-language models
    - Edge generative AI
    """
    from ...models.edge.hailo10h import hailo10h_resource_model
    return HailoMapper(hailo10h_resource_model())
