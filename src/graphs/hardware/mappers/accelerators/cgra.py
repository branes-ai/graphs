"""
CGRA Hardware Mapper - Maps fused subgraphs to Stanford Plasticine spatial fabric

This module implements realistic CGRA mapping using spatial dataflow execution,
fundamentally different from temporal execution (GPU/CPU/TPU/KPU/DPU).

Key characteristics:
- Spatial dataflow: Entire computation graph mapped to fabric
- Greedy place-and-route: Conservative NP-hard problem approximation
- Reconfiguration overhead: 1000 cycles (CGRA's Achilles heel)
- Medium-grained PCUs: Balanced coverage vs fabric overhead
- Power budget: 15W (embodied AI range: 10-25W)

CGRA vs Tile-Based (KPU/DPU):
- CGRA: Spatial execution (entire subgraph mapped to fabric simultaneously)
- Tile-based: Temporal execution (operations execute sequentially)
- CGRA: Reconfiguration overhead for each new subgraph pattern
- Tile-based: Minimal overhead (just load/store data)

Example:
  ResNet-18 has 27 fused subgraphs
  - Each subgraph requires fabric reconfiguration (1000 cycles)
  - Total reconfig overhead: 27 × 1000 = 27,000 cycles
  - At 1 GHz: 27 microseconds overhead
  - Compare to compute time: ~5-10ms
  - Overhead: ~0.3% (acceptable)
"""

from typing import List, Dict, Tuple
from dataclasses import dataclass
import math

from ...resource_model import (
    HardwareMapper,
    HardwareResourceModel,
    HardwareAllocation,
    GraphHardwareAllocation,
    Precision,
)
from graphs.transform.partitioning import FusedSubgraph, FusionReport
from graphs.core.structures import BottleneckType


@dataclass
class PlaceAndRouteResult:
    """
    Result of greedy place-and-route for a subgraph.

    The place-and-route problem is NP-hard, so we use a greedy heuristic:
    1. Count operations and types in the subgraph
    2. Find the critical path (longest dependency chain)
    3. Map critical path to PCUs first (priority)
    4. Map parallel operations to remaining PCUs
    5. Check if graph fits; partition if needed
    """
    num_operations: int  # Total operations in subgraph
    critical_path_length: int  # Length of longest dependency chain
    parallel_width: int  # Maximum parallelism available

    pcus_required: int  # PCUs needed for this subgraph
    pcus_allocated: int  # PCUs actually allocated (≤ 32)
    fits_in_fabric: bool  # True if subgraph fits without partitioning

    num_partitions: int  # How many reconfigurations needed
    reconfiguration_cycles: int  # Cycles spent on reconfiguration
    spatial_efficiency: float  # Fraction of PCUs actually used


class CGRAMapper(HardwareMapper):
    """
    CGRA hardware mapper using spatial dataflow execution.

    Implements realistic CGRA mapping considering:
    - Greedy place-and-route algorithm
    - Conservative reconfiguration overhead (1000 cycles)
    - Medium-grained PCU allocation
    - Spatial execution (entire graph mapped)
    - Fabric constraints (32 PCUs max)
    """

    def __init__(self, resource_model: HardwareResourceModel):
        super().__init__(resource_model)

        # Validate this is a CGRA model
        if resource_model.hardware_type.value != "cgra":
            raise ValueError(f"CGRAMapper requires CGRA resource model, got {resource_model.hardware_type}")

        # CGRA-specific parameters
        self.num_pcus = resource_model.compute_units  # 32 PCUs typical
        self.ops_per_pcu = resource_model.threads_per_unit  # 8 ops per PCU
        self.clock_freq = 1.0e9  # 1 GHz

        # Reconfiguration overhead (CGRA's Achilles heel - be conservative)
        self.reconfiguration_cycles = 1000  # Cycles to load new configuration

    def _greedy_place_and_route(
        self,
        subgraph: FusedSubgraph,
        precision: Precision
    ) -> PlaceAndRouteResult:
        """
        Greedy heuristic for place-and-route (NP-hard problem).

        Algorithm:
        1. Count operations in subgraph
        2. Estimate critical path length (dependency depth)
        3. Estimate parallel width (max concurrent ops)
        4. Allocate PCUs greedily (critical path first)
        5. Check if fits in fabric (32 PCUs)

        Args:
            subgraph: Fused subgraph to map
            precision: Numerical precision

        Returns:
            PlaceAndRouteResult with allocation details
        """
        # Estimate critical path length for spatial dataflow
        # The critical path depends on the actual computation structure, not total FLOPs
        # For Conv2D: critical path ~ output_height (sequential dependency in spatial dimension)
        # For MatMul: critical path ~ sqrt(inner_dimension) (reduction tree depth)
        #
        # Heuristic: Use cube root of FLOPs as proxy for critical path depth
        # - Too low: Just counting graph nodes (ignores computation depth)
        # - Too high: Using total FLOPs / 1000 (ignores parallelism)
        # - Balanced: FLOPs^(1/3) accounts for 3D nature of Conv (H×W×C)
        #
        # Example: 236 MFLOP → 236M^(1/3) ≈ 618 operations critical path
        flops = subgraph.total_flops if subgraph.total_flops > 0 else subgraph.total_macs * 2
        num_nodes = max(1, len(subgraph.node_names))

        if flops > 0:
            # Cube root provides good balance for typical CNN ops
            critical_path_length = max(10, int(flops ** (1.0/3.0)))
            num_operations = int(flops / 1000)  # Rough op count for area estimation
        else:
            # Fallback for non-compute ops (activation, pooling)
            critical_path_length = num_nodes * 10
            num_operations = num_nodes

        # Estimate parallel width
        # Use parallelism info from subgraph if available
        if subgraph.parallelism is not None:
            parallel_width = max(1, subgraph.parallelism.total_threads // self.ops_per_pcu)
        else:
            # Fallback: assume sqrt(num_operations) parallelism
            parallel_width = max(1, int(math.sqrt(num_operations)))

        # PCUs required: max of critical path length and parallel width
        # (Critical path sets minimum, parallelism sets ideal)
        pcus_required = max(critical_path_length, parallel_width)
        pcus_required = min(pcus_required, num_operations)  # Can't exceed total ops

        # Check if fits in fabric
        fits_in_fabric = pcus_required <= self.num_pcus

        if fits_in_fabric:
            # Entire subgraph fits - single configuration
            pcus_allocated = pcus_required
            num_partitions = 1
            reconfiguration_cycles = self.reconfiguration_cycles
        else:
            # Need multiple reconfigurations (partition subgraph)
            pcus_allocated = self.num_pcus
            num_partitions = math.ceil(pcus_required / self.num_pcus)
            # Conservative: each partition requires full reconfiguration
            reconfiguration_cycles = num_partitions * self.reconfiguration_cycles

        # Spatial efficiency: what fraction of allocated PCUs are actually used
        spatial_efficiency = min(1.0, pcus_required / pcus_allocated) if pcus_allocated > 0 else 0.0

        return PlaceAndRouteResult(
            num_operations=num_operations,
            critical_path_length=critical_path_length,
            parallel_width=parallel_width,
            pcus_required=pcus_required,
            pcus_allocated=pcus_allocated,
            fits_in_fabric=fits_in_fabric,
            num_partitions=num_partitions,
            reconfiguration_cycles=reconfiguration_cycles,
            spatial_efficiency=spatial_efficiency,
        )

    def _get_bytes_per_element(self, precision: Precision) -> int:
        """Get bytes per element for a precision"""
        bytes_map = {
            Precision.FP64: 8,
            Precision.FP32: 4,
            Precision.FP16: 2,
            Precision.BF16: 2,
            Precision.FP8_E4M3: 1,
            Precision.FP8_E5M2: 1,
            Precision.INT32: 4,
            Precision.INT16: 2,
            Precision.INT8: 1,
            Precision.INT4: 0.5,  # Packed
        }
        return bytes_map.get(precision, 4)

    def map_subgraph(
        self,
        subgraph: FusedSubgraph,
        execution_stage: int,
        concurrent_subgraphs: int,
        precision: Precision = Precision.INT8  # CGRA default is INT8
    ) -> HardwareAllocation:
        """
        Map a single fused subgraph to CGRA spatial fabric.

        Algorithm:
        1. Run greedy place-and-route to determine PCU allocation
        2. Calculate reconfiguration overhead (conservative: 1000 cycles)
        3. Calculate spatial execution time (all ops execute in parallel)
        4. Calculate latency = reconfiguration + execution
        5. Calculate energy

        Key difference from temporal execution:
        - Temporal (GPU/KPU/DPU): Operations execute sequentially
        - Spatial (CGRA): Operations execute simultaneously on fabric
        """
        # Run greedy place-and-route
        pnr_result = self._greedy_place_and_route(subgraph, precision)

        # Calculate occupancy (what fraction of PCUs are busy)
        occupancy = pnr_result.pcus_allocated / self.num_pcus

        # Calculate utilization (considering spatial efficiency)
        utilization = occupancy * pnr_result.spatial_efficiency

        # Calculate latency
        ops = subgraph.total_flops if subgraph.total_flops > 0 else subgraph.total_macs * 2
        bytes_transferred = (
            subgraph.total_input_bytes +
            subgraph.total_output_bytes +
            subgraph.total_weight_bytes
        )

        # SPATIAL EXECUTION: Operations execute in parallel on fabric
        # Execution time = critical_path_length / clock_freq
        # (Not total_ops / peak_ops like temporal execution)
        critical_path_cycles = pnr_result.critical_path_length * 10  # ~10 cycles per operation
        execution_time = critical_path_cycles / self.clock_freq

        # Add reconfiguration overhead (CGRA's Achilles heel)
        reconfiguration_time = pnr_result.reconfiguration_cycles / self.clock_freq

        # For bottleneck analysis, calculate roofline compute/memory time
        # But don't use it for latency - spatial execution is fundamentally different
        compute_time, memory_time, bottleneck = self._calculate_latency(
            ops=ops,
            bytes_transferred=bytes_transferred,
            allocated_units=pnr_result.pcus_allocated,
            occupancy=occupancy,
            precision=precision
        )

        # Total latency = reconfiguration + spatial execution
        # For CGRA spatial dataflow, use critical path model, NOT roofline model
        # Roofline is for temporal execution (KPU/DPU/GPU)
        spatial_execution_latency = execution_time + reconfiguration_time
        estimated_latency = spatial_execution_latency

        # Calculate energy
        compute_energy, memory_energy = self._calculate_energy(
            ops=ops,
            bytes_transferred=bytes_transferred,
            precision=precision
        )
        # Add reconfiguration energy (small but non-zero)
        reconfiguration_energy = reconfiguration_time * 15.0  # 15W during reconfig
        total_energy = compute_energy + memory_energy + reconfiguration_energy

        # Check if this can run in parallel with others
        is_parallel = concurrent_subgraphs > 1

        return HardwareAllocation(
            subgraph_id=str(subgraph.subgraph_id),
            subgraph_name=", ".join(subgraph.node_names[:2]),
            precision=precision,
            threads_required=pnr_result.pcus_allocated * self.ops_per_pcu,
            warps_required=0,  # No warp concept in spatial execution
            compute_units_allocated=pnr_result.pcus_allocated,
            compute_units_ideal=pnr_result.pcus_required,
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
        Map entire computation graph to CGRA.

        Args:
            fusion_report: Output from Phase 1 fusion partitioner
            execution_stages: Execution stages with subgraph indices
            batch_size: Batch size (scales parallelism)
            precision: Numerical precision (INT8 preferred)

        Returns:
            Complete hardware allocation
        """
        subgraph_allocations: List[HardwareAllocation] = []
        latency_breakdown: Dict[int, float] = {}

        peak_pcus_used = 0
        total_pcus_used = 0
        total_pcus_samples = 0

        # Process each execution stage
        for stage_id, subgraph_indices in enumerate(execution_stages):
            stage_allocations = []
            concurrent_subgraphs = len(subgraph_indices)

            # Map each subgraph in this stage
            for subgraph_idx in subgraph_indices:
                if subgraph_idx >= len(fusion_report.fused_subgraphs):
                    continue

                subgraph = fusion_report.fused_subgraphs[subgraph_idx]
                allocation = self.map_subgraph(
                    subgraph=subgraph,
                    execution_stage=stage_id,
                    concurrent_subgraphs=concurrent_subgraphs,
                    precision=precision
                )
                stage_allocations.append(allocation)
                subgraph_allocations.append(allocation)

                # Track utilization
                pcus_used = allocation.compute_units_allocated
                total_pcus_used += pcus_used
                total_pcus_samples += 1
                peak_pcus_used = max(peak_pcus_used, pcus_used)

            # Stage latency: max latency of parallel ops
            if stage_allocations:
                stage_latency = max(alloc.estimated_latency for alloc in stage_allocations)
                latency_breakdown[stage_id] = stage_latency

        # Total latency: sum of all stage latencies
        total_latency = sum(latency_breakdown.values())

        # Total energy: sum of all operation energies
        total_energy = sum(alloc.total_energy for alloc in subgraph_allocations)

        # Utilization stats
        avg_pcus_used = total_pcus_used / total_pcus_samples if total_pcus_samples > 0 else 0
        avg_utilization = avg_pcus_used / self.num_pcus
        peak_utilization = peak_pcus_used / self.num_pcus

        # Naive latency (assuming 100% utilization)
        total_ops = fusion_report.total_flops
        peak_ops_per_sec = self.resource_model.get_peak_ops(precision)
        naive_latency = total_ops / peak_ops_per_sec if peak_ops_per_sec > 0 else 0

        # Correction factor
        latency_correction_factor = total_latency / naive_latency if naive_latency > 0 else 1.0

        # Bottleneck analysis
        compute_bound_count = sum(1 for alloc in subgraph_allocations if alloc.bottleneck == BottleneckType.COMPUTE_BOUND)
        memory_bound_count = sum(1 for alloc in subgraph_allocations if alloc.bottleneck == BottleneckType.MEMORY_BOUND)
        bandwidth_bound_count = sum(1 for alloc in subgraph_allocations if alloc.bottleneck == BottleneckType.BANDWIDTH_BOUND)
        balanced_count = sum(1 for alloc in subgraph_allocations if alloc.bottleneck == BottleneckType.BALANCED)

        return GraphHardwareAllocation(
            model_name="Unknown",  # Will be set by caller
            hardware_name=self.resource_model.name,
            batch_size=batch_size,
            model_precision=precision,
            subgraph_allocations=subgraph_allocations,
            total_subgraphs=len(fusion_report.fused_subgraphs),
            total_execution_stages=len(execution_stages),
            peak_compute_units_used=peak_pcus_used,
            average_compute_units_used=avg_pcus_used,
            peak_utilization=peak_utilization,
            average_utilization=avg_utilization,
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


def create_plasticine_v2_mapper() -> CGRAMapper:
    """
    Create a CGRA mapper for Stanford Plasticine-v2.

    Configuration:
    - 32 PCUs (medium granularity)
    - 6.14 TOPS INT8 @ 60% efficiency
    - 15W power (embodied AI range)
    - Conservative reconfiguration: 1000 cycles

    Returns:
        CGRAMapper instance
    """
    from ...models.accelerators.stanford_plasticine_cgra import stanford_plasticine_cgra_resource_model
    return CGRAMapper(stanford_plasticine_cgra_resource_model())
