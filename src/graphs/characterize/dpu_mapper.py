"""
DPU Hardware Mapper - Maps fused subgraphs to Xilinx Vitis AI DPU tiles

This module implements realistic DPU mapping for Xilinx Versal AI Edge devices,
considering:
- Tile-based processing with AIE (Adaptive Intelligence Engine)
- Scratchpad memory constraints (64KB per tile typical)
- FPGA reconfigurability and flexibility
- Native INT8 acceleration (best performance)
- Power efficiency (15-20W for embodied AI)

Key characteristics:
- B4096 configuration: 4096 MACs @ 1.25 GHz
- Realistic peak: 7.68 TOPS INT8 @ 75% efficiency
- 64 tiles (estimate for B4096)
- Scratchpad-based memory hierarchy
- Optimized for edge deployment

DPU vs KPU:
- Slower absolute performance (7.68 vs 100 TOPS)
- More power efficient (17.5W vs 25W)
- FPGA-based (reconfigurable vs fixed ASIC)
- Similar tiling approach (scratchpad constraints)
- Better suited for custom operations

Example:
  ResNet-18 Conv layer with 3×3 kernel, 64 channels:
  - Input: 1×64×56×56 (802KB) - doesn't fit in 64KB!
  - Need tiling: Process 8×56 strips at a time
  - 64/8 = 8 iterations required
  - Each iteration: load input (25KB), weights (36KB), compute, store (25KB)
"""

from typing import List, Dict, Tuple
from dataclasses import dataclass
import math

from .hardware_mapper import (
    HardwareMapper,
    HardwareResourceModel,
    HardwareAllocation,
    GraphHardwareAllocation,
    Precision,
)
from .fusion_partitioner import FusedSubgraph, FusionReport
from .graph_structures import BottleneckType


@dataclass
class DPUTileConfiguration:
    """
    DPU tile configuration for an operation.

    DPU uses AIE tile-based processing where each operation is broken
    into tiles that fit in the scratchpad memory (64KB typical).
    """
    scratchpad_size: int  # 64KB typical
    input_bytes_per_tile: int  # Bytes of input per tile
    weight_bytes_per_tile: int  # Bytes of weights per tile
    output_bytes_per_tile: int  # Bytes of output per tile
    total_bytes_per_tile: int  # Total bytes needed per tile

    num_tiles_required: int  # How many tiles to process all data
    tiles_per_iteration: int  # How many tiles can run in parallel
    num_iterations: int  # How many iterations to process all tiles

    fits_in_scratchpad: bool  # True if operation fits in scratchpad
    tiling_overhead: float  # Overhead factor due to tiling (1.0 = no overhead)


class DPUMapper(HardwareMapper):
    """
    Xilinx Vitis AI DPU hardware mapper using AIE tile allocation.

    Implements realistic DPU mapping considering:
    - AIE tile allocation based on parallelism
    - Scratchpad memory constraints (64KB per tile)
    - Tiling strategy when data doesn't fit
    - Quantization benefits (INT8 native)
    - FPGA reconfiguration overhead (negligible after initial load)
    - Power efficiency for embodied AI
    """

    def __init__(self, resource_model: HardwareResourceModel):
        super().__init__(resource_model)

        # Validate this is a DPU model
        if resource_model.hardware_type.value != "dpu":
            raise ValueError(f"DPUMapper requires DPU resource model, got {resource_model.hardware_type}")

        # DPU-specific parameters
        self.num_tiles = resource_model.compute_units  # 64 tiles typical
        self.scratchpad_per_tile = resource_model.l1_cache_per_unit  # 64KB
        self.threads_per_tile = resource_model.threads_per_unit  # 64 threads

    def _analyze_tiling(
        self,
        subgraph: FusedSubgraph,
        precision: Precision
    ) -> DPUTileConfiguration:
        """
        Analyze tiling requirements for a subgraph.

        Args:
            subgraph: Fused subgraph to analyze
            precision: Numerical precision (affects memory footprint)

        Returns:
            DPUTileConfiguration with tiling analysis
        """
        scratchpad_size = self.scratchpad_per_tile

        # Get bytes per element for this precision
        bytes_per_element = self._get_bytes_per_element(precision)

        # Calculate memory footprint per operation
        # For a Conv2D: need input tile + weights + output tile in scratchpad
        input_bytes = subgraph.total_input_bytes
        weight_bytes = subgraph.total_weight_bytes
        output_bytes = subgraph.total_output_bytes

        # Adjust for precision (if different from FP32)
        # Subgraph bytes are in FP32, scale to actual precision
        scale_factor = bytes_per_element / 4.0  # 4 bytes = FP32
        input_bytes = int(input_bytes * scale_factor)
        weight_bytes = int(weight_bytes * scale_factor)
        output_bytes = int(output_bytes * scale_factor)

        total_bytes = input_bytes + weight_bytes + output_bytes

        # Check if fits in scratchpad
        fits_in_scratchpad = total_bytes <= scratchpad_size

        # Initialize variables (will be overwritten below)
        input_per_tile = input_bytes
        weight_per_tile = weight_bytes
        output_per_tile = output_bytes
        bytes_per_tile = total_bytes

        if fits_in_scratchpad:
            # Entire operation fits - no tiling needed
            num_tiles_required = 1
            tiles_per_iteration = 1
            num_iterations = 1
            tiling_overhead = 1.0  # No overhead
        else:
            # Need tiling - split operation into chunks
            # Strategy: Keep weights in scratchpad, tile input/output

            # Weights typically smaller and reused
            if weight_bytes > scratchpad_size * 0.8:
                # Weights too large - need to tile weights too (rare)
                # Pessimistic: assume we can fit 80% of scratchpad per tile
                bytes_per_tile = int(scratchpad_size * 0.8)
                num_tiles_required = math.ceil(total_bytes / bytes_per_tile)
            else:
                # Weights fit, tile input/output
                # Reserve space for weights, split remaining between input/output
                remaining_space = scratchpad_size - weight_bytes

                # Input and output ratio
                io_bytes = input_bytes + output_bytes
                if io_bytes > 0:
                    input_fraction = input_bytes / io_bytes
                    output_fraction = output_bytes / io_bytes

                    input_per_tile = int(remaining_space * input_fraction * 0.8)  # 80% efficiency
                    output_per_tile = int(remaining_space * output_fraction * 0.8)
                else:
                    input_per_tile = 0
                    output_per_tile = 0

                weight_per_tile = weight_bytes
                bytes_per_tile = input_per_tile + weight_per_tile + output_per_tile

                # Calculate number of tiles needed
                if input_per_tile > 0:
                    num_tiles_required = math.ceil(input_bytes / input_per_tile)
                else:
                    num_tiles_required = 1

            # How many tiles can run in parallel? (up to total DPU tiles)
            tiles_per_iteration = min(num_tiles_required, self.num_tiles)

            # How many iterations to process all tiles?
            num_iterations = math.ceil(num_tiles_required / tiles_per_iteration)

            # Tiling overhead: each iteration has load/store overhead
            # DPU has slightly higher overhead than KPU due to FPGA fabric routing
            # Estimate 12% overhead per iteration for data movement
            tiling_overhead = 1.0 + (num_iterations - 1) * 0.12

        return DPUTileConfiguration(
            scratchpad_size=scratchpad_size,
            input_bytes_per_tile=input_per_tile,
            weight_bytes_per_tile=weight_per_tile,
            output_bytes_per_tile=output_per_tile,
            total_bytes_per_tile=bytes_per_tile,
            num_tiles_required=num_tiles_required,
            tiles_per_iteration=tiles_per_iteration,
            num_iterations=num_iterations,
            fits_in_scratchpad=fits_in_scratchpad,
            tiling_overhead=tiling_overhead,
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
        precision: Precision = Precision.INT8  # DPU default is INT8
    ) -> HardwareAllocation:
        """
        Map a single fused subgraph to DPU tiles.

        Algorithm:
        1. Analyze tiling requirements (does it fit in 64KB scratchpad?)
        2. Determine tile allocation based on parallelism
        3. Calculate number of iterations needed
        4. Calculate occupancy (limited by tile count)
        5. Calculate latency using roofline model + tiling overhead
        6. Calculate energy (DPU is power-efficient at 17.5W)
        """
        # Analyze tiling
        tile_config = self._analyze_tiling(subgraph, precision)

        # Get parallelism
        if subgraph.parallelism is None:
            # Fallback: assume minimal parallelism
            threads_required = self.num_tiles * self.threads_per_tile
            tiles_allocated = self.num_tiles
        else:
            # DPU parallelism is limited by tile count
            parallelism = subgraph.parallelism.total_threads

            # Calculate tiles needed based on threads
            threads_per_tile = self.threads_per_tile
            tiles_needed = math.ceil(parallelism / threads_per_tile)

            # But also limited by tiling requirements
            tiles_needed = max(tiles_needed, tile_config.tiles_per_iteration)

            # Allocate up to all tiles
            tiles_allocated = min(tiles_needed, self.num_tiles)

        tiles_allocated = max(1, tiles_allocated)  # At least 1 tile
        threads_required = tiles_allocated * self.threads_per_tile

        # Calculate occupancy (what fraction of tiles are busy)
        occupancy = tiles_allocated / self.num_tiles

        # Calculate utilization
        utilization = tiles_allocated / self.num_tiles

        # Calculate latency using roofline model
        ops = subgraph.total_flops if subgraph.total_flops > 0 else subgraph.total_macs * 2
        bytes_transferred = (
            subgraph.total_input_bytes +
            subgraph.total_output_bytes +
            subgraph.total_weight_bytes
        )

        # Multiply by tiling overhead (more iterations = more overhead)
        # This accounts for loading/storing tiles multiple times
        ops_with_tiling = int(ops * tile_config.tiling_overhead)
        bytes_with_tiling = int(bytes_transferred * tile_config.num_iterations)

        compute_time, memory_time, bottleneck = self._calculate_latency(
            ops=ops_with_tiling,
            bytes_transferred=bytes_with_tiling,
            allocated_units=tiles_allocated,
            occupancy=occupancy,
            precision=precision
        )

        estimated_latency = max(compute_time, memory_time)

        # Calculate energy
        compute_energy, memory_energy = self._calculate_energy(
            ops=ops,
            bytes_transferred=bytes_transferred,
            precision=precision
        )
        total_energy = compute_energy + memory_energy

        # Check if this can run in parallel with others
        is_parallel = concurrent_subgraphs > 1

        return HardwareAllocation(
            subgraph_id=str(subgraph.subgraph_id),
            subgraph_name=", ".join(subgraph.node_names[:2]),
            precision=precision,
            threads_required=threads_required,
            warps_required=0,  # DPU uses tiles, not warps
            compute_units_allocated=tiles_allocated,
            compute_units_ideal=tiles_allocated,
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
        Map entire computation graph to DPU.

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

        peak_tiles_used = 0
        total_tiles_used = 0
        total_tiles_samples = 0

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
                tiles_used = allocation.compute_units_allocated
                total_tiles_used += tiles_used
                total_tiles_samples += 1
                peak_tiles_used = max(peak_tiles_used, tiles_used)

            # Stage latency: max latency of parallel ops
            if stage_allocations:
                stage_latency = max(alloc.estimated_latency for alloc in stage_allocations)
                latency_breakdown[stage_id] = stage_latency

        # Total latency: sum of all stage latencies
        total_latency = sum(latency_breakdown.values())

        # Total energy: sum of all operation energies
        total_energy = sum(alloc.total_energy for alloc in subgraph_allocations)

        # Utilization stats
        avg_tiles_used = total_tiles_used / total_tiles_samples if total_tiles_samples > 0 else 0
        avg_utilization = avg_tiles_used / self.num_tiles
        peak_utilization = peak_tiles_used / self.num_tiles

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
            peak_compute_units_used=peak_tiles_used,
            average_compute_units_used=avg_tiles_used,
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


def create_dpu_vitis_ai_mapper() -> DPUMapper:
    """
    Create a DPU mapper for Xilinx Vitis AI (B4096 on Versal VE2302).

    Configuration:
    - B4096: 4096 MACs @ 1.25 GHz
    - 7.68 TOPS INT8 @ 75% efficiency
    - 64 tiles with 64KB scratchpad each
    - 17.5W power (embodied AI target)
    - Optimized for edge deployment

    Returns:
        DPUMapper instance
    """
    from .hardware_mapper import xilinx_vitis_ai_dpu_resource_model
    return DPUMapper(xilinx_vitis_ai_dpu_resource_model())
