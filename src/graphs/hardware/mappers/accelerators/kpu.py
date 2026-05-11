"""
KPU Hardware Mapper - Maps fused subgraphs to KPU tiles with scratchpad constraints

This module implements realistic KPU mapping considering:
- Tile-based processing (64 tiles typical)
- Scratchpad memory constraints (256KB per tile)
- Tiling overhead (load/process/store cycles)
- Quantization optimization (INT8/INT4 preferred)
- Energy efficiency (10× better than CPU)

Key differences from GPU:
- Fewer tiles (64 vs 132 SMs)
- Strict scratchpad memory limit (256KB vs flexible SM memory)
- Explicit tiling required (must fit in scratchpad)
- Higher tiling overhead (smaller tiles = more iterations)
- Optimized for quantized inference (INT8/INT4)

Example:
  ResNet-18 Conv layer with 3x3 kernel, 64 channels:
  - Input: 1x64x56x56 (802KB) - doesn't fit in 256KB!
  - Need tiling: Process 16x56 strips at a time
  - 64/16 = 4 iterations required
  - Each iteration: load input (50KB), weights (36KB), compute, store (50KB)
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
class TileConfiguration:
    """
    KPU tile configuration for an operation.

    KPU uses tile-based processing where each operation is broken
    into tiles that fit in the 256KB scratchpad memory.
    """
    scratchpad_size: int  # 256KB typical
    input_bytes_per_tile: int  # Bytes of input per tile
    weight_bytes_per_tile: int  # Bytes of weights per tile
    output_bytes_per_tile: int  # Bytes of output per tile
    total_bytes_per_tile: int  # Total bytes needed per tile

    num_tiles_required: int  # How many tiles to process all data
    tiles_per_iteration: int  # How many tiles can run in parallel
    num_iterations: int  # How many iterations to process all tiles

    fits_in_scratchpad: bool  # True if operation fits in scratchpad
    tiling_overhead: float  # Overhead factor due to tiling (1.0 = no overhead)


class KPUMapper(HardwareMapper):
    """
    KPU hardware mapper using tile + scratchpad allocation.

    Implements realistic KPU mapping considering:
    - Tile allocation based on parallelism
    - Scratchpad memory constraints (256KB per tile)
    - Tiling strategy when data doesn't fit
    - Quantization benefits (INT8/INT4 preferred)
    - Tiling overhead
    """

    # Idle power modeling (nanoscale leakage)
    IDLE_POWER_FRACTION = 0.5  # 50% of TDP consumed at idle due to leakage

    def __init__(
        self,
        resource_model: HardwareResourceModel,
        thermal_profile: str = None
    ):
        """
        Initialize KPU mapper.

        Args:
            resource_model: KPU resource model
            thermal_profile: Thermal profile name (e.g., "6W", "10W")
                           If None, uses default from resource model
        """
        super().__init__(resource_model, thermal_profile=thermal_profile)

        # Validate this is a KPU model
        if resource_model.hardware_type.value != "kpu":
            raise ValueError(f"KPUMapper requires KPU resource model, got {resource_model.hardware_type}")

        # KPU-specific parameters
        self.num_tiles = resource_model.compute_units  # 64, 256, or 768 tiles
        self.scratchpad_per_tile = resource_model.l1_cache_per_unit  # 256KB
        self.threads_per_tile = resource_model.threads_per_unit  # 256 threads

        # Aggregate on-chip capacity (issue #51): the KPU's reason-for-being is
        # weight stationarity. Weight-residency decisions need to see the full
        # tile-fabric L1 + shared L2, not just one tile's scratchpad.
        # KPU-T64: 64 * 256KB + 4MB = 20 MB; KPU-T256: 256 * 256KB + 16MB = 80 MB.
        self.l2_cache_total = resource_model.l2_cache_total
        self.total_on_chip_bytes = (
            self.num_tiles * self.scratchpad_per_tile + self.l2_cache_total
        )

    def compute_energy_with_idle_power(
        self,
        latency: float,
        dynamic_energy: float
    ) -> Tuple[float, float]:
        """
        Compute total energy including idle power consumption.

        Modern nanoscale SoCs consume ~50% of TDP at idle due to leakage currents.
        This method adds idle energy to the dynamic energy.

        Args:
            latency: Total execution time (seconds)
            dynamic_energy: Energy from compute + memory transfers (Joules)

        Returns:
            (total_energy, average_power) in Joules and Watts
        """
        # Get TDP from thermal operating point
        tdp_watts = None
        if self.thermal_profile and self.resource_model.thermal_operating_points:
            thermal_point = self.resource_model.thermal_operating_points.get(self.thermal_profile)
            if thermal_point:
                tdp_watts = thermal_point.tdp_watts

        # Fallback logic: try "default" profile, then first available
        if tdp_watts is None and self.resource_model.thermal_operating_points:
            default_thermal = self.resource_model.thermal_operating_points.get("default")
            if default_thermal:
                tdp_watts = default_thermal.tdp_watts
            else:
                # Use first available thermal profile
                first_profile = next(iter(self.resource_model.thermal_operating_points.values()), None)
                if first_profile:
                    tdp_watts = first_profile.tdp_watts

        # Final fallback: estimate TDP from dynamic power
        if tdp_watts is None:
            dynamic_power = dynamic_energy / latency if latency > 0 else 0
            tdp_watts = dynamic_power * 2.0  # Assume 50% headroom

        # Calculate idle energy (constant power × time)
        idle_power = tdp_watts * self.IDLE_POWER_FRACTION
        idle_energy = idle_power * latency

        # Total energy = idle + dynamic
        total_energy = idle_energy + dynamic_energy

        # Average power during execution
        average_power = total_energy / latency if latency > 0 else 0

        return total_energy, average_power

    def _analyze_tiling(
        self,
        subgraph: FusedSubgraph,
        precision: Precision
    ) -> TileConfiguration:
        """
        Analyze tiling and on-chip residency for a subgraph.

        Implements the weight-stationary execution model that is the KPU's
        architectural reason for existing (issue #51). For each subgraph:

        1. Compare weight footprint against the aggregate on-chip capacity
           (all tiles' L1 + shared L2), not a single tile's 256KB scratchpad.
        2. If weights fit on-chip, model weight loads as a one-shot prologue
           cost and stream activations through the resident weights.
        3. If weights don't fit, outer-tile execution: weights load once per
           outer pass over the activation stream, not once per inner tile.

        Note: ``subgraph.total_*_bytes`` are already at the analysis precision
        (issue #52 / PR #54). The mapper used to multiply by
        ``bytes_per_element/4`` here, which double-scaled at fp16/int8/int4.
        That obsolete adjustment has been removed.
        """
        scratchpad_size = self.scratchpad_per_tile
        on_chip_total = self.total_on_chip_bytes

        # Memory footprint at analysis precision (no double-scaling).
        input_bytes = subgraph.total_input_bytes
        weight_bytes = subgraph.total_weight_bytes
        output_bytes = subgraph.total_output_bytes
        total_bytes = input_bytes + weight_bytes + output_bytes

        # Per-tile fit (legacy diagnostic): does the *entire* working set fit
        # in a single tile's scratchpad? Useful for very small ops.
        fits_in_scratchpad = total_bytes <= scratchpad_size

        # On-chip residency for weights. Reserve a fraction of on-chip for the
        # activation working set so weights and activations co-exist.
        # 80% for weights / 20% for activations is a common dataflow heuristic.
        weight_on_chip_budget = int(on_chip_total * 0.8)
        weights_fit_on_chip = weight_bytes <= weight_on_chip_budget

        if weights_fit_on_chip:
            # Weight-stationary: weights load *once* into the tile fabric and
            # stay resident while activations stream through them.
            outer_weight_loads = 1
            # Activation working set per outer pass = on-chip minus weights.
            activation_budget = max(scratchpad_size, on_chip_total - weight_bytes)
        else:
            # Weights exceed the on-chip budget: outer-tile execution. Each
            # outer pass loads a *different slab* of weights, streams the
            # full activation set through it, then loads the next slab. The
            # slabs together cover the whole weight set exactly once.
            outer_weight_loads = max(1, math.ceil(weight_bytes / weight_on_chip_budget))
            # Activation working set co-resides with the active weight slab,
            # so the budget is whatever's left after reserving for weights
            # (consistent with the 80/20 split above).
            activation_budget = max(scratchpad_size, on_chip_total - weight_on_chip_budget)

        # How many times the activation stream cycles through DRAM. For
        # well-fitted activations this is 1; for very large activations (rare
        # for inference) it grows. Multiplied by outer_weight_loads when
        # weights are tiled, since each weight slab needs the full activation
        # pass.
        io_bytes = input_bytes + output_bytes
        if io_bytes <= 0:
            activation_iterations = 1
        else:
            activation_iterations = max(1, math.ceil(io_bytes / activation_budget)) * outer_weight_loads

        # tiles_per_iteration drives parallelism, not memory traffic. Use the
        # number of tiles needed to hold one outer pass.
        tiles_required = max(1, math.ceil(total_bytes / scratchpad_size))
        tiles_per_iteration = min(tiles_required, self.num_tiles)
        num_iterations = max(1, math.ceil(tiles_required / tiles_per_iteration))

        # Small prologue-load overhead when there is more than one outer pass.
        # Replaces the previous (num_iterations - 1) * 10% blanket inflation,
        # which compounded with the weight re-fetch bug.
        tiling_overhead = 1.0 + (max(0, outer_weight_loads - 1)) * 0.05

        # Per-tile decomposition (kept for diagnostics in TileConfiguration).
        input_per_tile = input_bytes // max(1, tiles_per_iteration)
        weight_per_tile = weight_bytes if weights_fit_on_chip else weight_on_chip_budget
        output_per_tile = output_bytes // max(1, tiles_per_iteration)
        bytes_per_tile = input_per_tile + weight_per_tile + output_per_tile

        config = TileConfiguration(
            scratchpad_size=scratchpad_size,
            input_bytes_per_tile=input_per_tile,
            weight_bytes_per_tile=weight_per_tile,
            output_bytes_per_tile=output_per_tile,
            total_bytes_per_tile=bytes_per_tile,
            num_tiles_required=tiles_required,
            tiles_per_iteration=tiles_per_iteration,
            num_iterations=num_iterations,
            fits_in_scratchpad=fits_in_scratchpad,
            tiling_overhead=tiling_overhead,
        )
        # Stash the residency outputs on the config so map_subgraph can use
        # them without re-deriving (extra attributes don't break the dataclass).
        config.weights_fit_on_chip = weights_fit_on_chip
        config.outer_weight_loads = outer_weight_loads
        config.activation_iterations = activation_iterations
        return config

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
        precision: Precision = Precision.INT8  # KPU default is INT8
    ) -> HardwareAllocation:
        """
        Map a single fused subgraph to KPU tiles.

        Algorithm:
        1. Analyze tiling requirements (does it fit in 256KB scratchpad?)
        2. Determine tile allocation based on parallelism
        3. Calculate number of iterations needed
        4. Calculate occupancy (limited by tile count)
        5. Calculate latency using roofline model + tiling overhead
        """
        # Analyze tiling
        tile_config = self._analyze_tiling(subgraph, precision)

        # Get parallelism
        if subgraph.parallelism is None:
            # Fallback: assume minimal parallelism
            threads_required = self.num_tiles * self.threads_per_tile
            tiles_allocated = self.num_tiles
        else:
            # KPU parallelism is limited by tile count
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

        # Calculate latency using roofline model.
        ops = subgraph.total_flops if subgraph.total_flops > 0 else subgraph.total_macs * 2

        # Weight-stationary memory accounting (issue #51). Each weight byte
        # is loaded from DRAM exactly once for this subgraph (split across
        # ``outer_weight_loads`` slabs when weights don't fit on-chip; each
        # slab covers a different weight tile, never the same one reloaded).
        # What gets repeated is the activation stream cycling through each
        # weight slab. Previously the mapper computed
        # ``(input + output + weight) * num_iterations`` which treats both
        # weights *and* activations as re-fetched on every inner tile --
        # exactly the access pattern the KPU dataflow is designed to avoid.
        activation_traffic_bytes = (
            (subgraph.total_input_bytes + subgraph.total_output_bytes)
            * tile_config.activation_iterations
        )
        weight_traffic_bytes = subgraph.total_weight_bytes
        bytes_transferred = activation_traffic_bytes + weight_traffic_bytes

        ops_with_tiling = int(ops * tile_config.tiling_overhead)

        compute_time, memory_time, bottleneck = self._calculate_latency(
            ops=ops_with_tiling,
            bytes_transferred=bytes_transferred,
            allocated_units=tiles_allocated,
            occupancy=occupancy,
            precision=precision
        )

        estimated_latency = max(compute_time, memory_time)

        # Calculate energy. Use the same precision-correct, stationary-aware
        # byte count so DRAM energy reflects what is actually re-fetched.
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
            warps_required=0,  # KPU uses tiles, not warps
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
        Map entire computation graph to KPU.

        Args:
            fusion_report: Output from Phase 1 fusion partitioner
            execution_stages: Execution stages with subgraph indices
            batch_size: Batch size (scales parallelism)
            precision: Numerical precision (INT8/INT4 preferred)

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

            # For parallel subgraphs: max tiles used, max latency
            if stage_allocations:
                stage_tiles_used = max(a.compute_units_allocated for a in stage_allocations)
                stage_latency = max(a.estimated_latency for a in stage_allocations)

                peak_tiles_used = max(peak_tiles_used, stage_tiles_used)
                total_tiles_used += stage_tiles_used
                total_tiles_samples += 1

                latency_breakdown[stage_id] = stage_latency

        # Calculate aggregate metrics
        total_subgraphs = len(subgraph_allocations)
        total_execution_stages = len(execution_stages)
        average_tiles_used = total_tiles_used / total_tiles_samples if total_tiles_samples > 0 else 0

        total_tiles = self.resource_model.compute_units
        peak_utilization = peak_tiles_used / total_tiles
        average_utilization = average_tiles_used / total_tiles

        # Total latency = sum of stage latencies (sequential)
        total_latency = sum(latency_breakdown.values())

        # Total energy with idle power
        dynamic_energy = sum(a.total_energy for a in subgraph_allocations)
        total_energy, average_power = self.compute_energy_with_idle_power(
            total_latency, dynamic_energy
        )

        # Naive latency (assuming 100% utilization)
        total_ops = fusion_report.total_flops
        peak_ops_per_sec = self.resource_model.get_peak_ops(precision)
        naive_latency = total_ops / peak_ops_per_sec if peak_ops_per_sec > 0 else 0

        # Correction factor
        latency_correction_factor = total_latency / naive_latency if naive_latency > 0 else 1.0

        # Bottleneck analysis
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
            peak_compute_units_used=peak_tiles_used,
            average_compute_units_used=average_tiles_used,
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


def create_kpu_t64_mapper(thermal_profile: str = None) -> KPUMapper:
    """
    Create KPU mapper for Stillwater KPU-T64 (edge AI / embodied AI accelerator).

    Args:
        thermal_profile: Thermal profile name (e.g., "3W", "6W", "10W")
                        If None, uses default ("6W")

    Returns:
        KPUMapper configured for KPU-T64 with heterogeneous tiles (44/13/7)
    """
    from ...models.accelerators.kpu_t64 import kpu_t64_resource_model
    from ...architectural_energy import KPUTileEnergyAdapter
    from ...physical_spec_loader import load_physical_spec_or_none

    model = kpu_t64_resource_model()

    # Wrap tile energy model with adapter to conform to ArchitecturalEnergyModel interface
    model.architecture_energy_model = KPUTileEnergyAdapter(model.tile_energy_model)

    mapper = KPUMapper(model, thermal_profile=thermal_profile)
    mapper.physical_spec = load_physical_spec_or_none(
        vendor="stillwater", base_id="kpu_t64_32x32_lp5x4_16nm_tsmc_ffp"
    )
    return mapper


def create_kpu_t128_mapper(thermal_profile: str = None) -> KPUMapper:
    """
    Create KPU mapper for Stillwater KPU-T128 (mid-range embodied AI).

    Introduced in M0.5. 128 tiles with 24x24 PE arrays per tile,
    scheduled OUTPUT_STATIONARY on a distributed domain-flow fabric.

    Args:
        thermal_profile: Thermal profile name (e.g., "6W", "12W", "18W")
                        If None, uses default ("12W")

    Returns:
        KPUMapper configured for KPU-T128 with heterogeneous tile roles (89/26/13)
    """
    from ...models.accelerators.kpu_t128 import kpu_t128_resource_model
    from ...architectural_energy import KPUTileEnergyAdapter
    from ...physical_spec_loader import load_physical_spec_or_none

    model = kpu_t128_resource_model()
    model.architecture_energy_model = KPUTileEnergyAdapter(model.tile_energy_model)

    mapper = KPUMapper(model, thermal_profile=thermal_profile)
    mapper.physical_spec = load_physical_spec_or_none(
        vendor="stillwater", base_id="kpu_t128_32x32_lp5x8_16nm_tsmc_ffp"
    )
    return mapper


def create_kpu_t256_mapper(thermal_profile: str = None) -> KPUMapper:
    """
    Create KPU mapper for Stillwater KPU-T256 (high-performance edge/datacenter AI).

    Args:
        thermal_profile: Thermal profile name (e.g., "15W", "30W", "50W")
                        If None, uses default ("30W")

    Returns:
        KPUMapper configured for KPU-T256 with heterogeneous tiles (179/51/26)
    """
    from ...models.accelerators.kpu_t256 import kpu_t256_resource_model
    from ...architectural_energy import KPUTileEnergyAdapter
    from ...physical_spec_loader import load_physical_spec_or_none

    model = kpu_t256_resource_model()

    # Wrap tile energy model with adapter to conform to ArchitecturalEnergyModel interface
    model.architecture_energy_model = KPUTileEnergyAdapter(model.tile_energy_model)

    mapper = KPUMapper(model, thermal_profile=thermal_profile)
    mapper.physical_spec = load_physical_spec_or_none(
        vendor="stillwater", base_id="kpu_t256_32x32_lp5x16_16nm_tsmc_ffp"
    )
    return mapper


def create_kpu_t768_mapper(thermal_profile: str = None) -> KPUMapper:
    """
    Create KPU mapper for Stillwater KPU-T768 (datacenter AI inference).

    Args:
        thermal_profile: Thermal profile name (e.g., "30W", "60W", "100W")
                        If None, uses default ("60W")

    Returns:
        KPUMapper configured for KPU-T768 with heterogeneous tiles (537/154/77)
    """
    from ...models.accelerators.kpu_t768 import kpu_t768_resource_model
    from ...architectural_energy import KPUTileEnergyAdapter
    from ...physical_spec_loader import load_physical_spec_or_none

    model = kpu_t768_resource_model()

    # Wrap tile energy model with adapter to conform to ArchitecturalEnergyModel interface
    model.architecture_energy_model = KPUTileEnergyAdapter(model.tile_energy_model)

    mapper = KPUMapper(model, thermal_profile=thermal_profile)
    mapper.physical_spec = load_physical_spec_or_none(
        vendor="stillwater", base_id="kpu_t768_16x8_hbm3x16_7nm_tsmc_hpc"
    )
    return mapper
