"""
GPU Hardware Mapper - Maps fused subgraphs to GPU Streaming Multiprocessors (SMs)

This module implements realistic SM allocation for GPUs like NVIDIA H100.

Key Algorithm:
1. Calculate thread requirements from parallelism descriptor
2. Map threads → warps → SMs
3. Account for wave quantization (SMs allocated in groups)
4. Calculate occupancy and utilization
5. Use precision-aware roofline model for latency

Example:
  ResNet-18 Conv layer with 200K threads:
  - 200K threads / 32 = 6,250 warps
  - 6,250 warps / 64 warps/SM = 98 SMs needed
  - With wave quantization (4): round up to 100 SMs
  - Utilization: 100/132 = 75.8%
"""

from typing import List, Dict, Tuple
from dataclasses import dataclass
import math

from ..resource_model import (
    HardwareMapper,
    HardwareResourceModel,
    HardwareAllocation,
    GraphHardwareAllocation,
    Precision,
)
from graphs.transform.partitioning import FusedSubgraph, FusionReport
from graphs.ir.structures import BottleneckType


class GPUMapper(HardwareMapper):
    """
    GPU hardware mapper using SM (Streaming Multiprocessor) allocation.

    Implements realistic GPU mapping considering:
    - Thread → warp → SM hierarchy
    - Wave quantization (SMs allocated in groups)
    - Occupancy limits
    - Concurrent kernel execution
    """

    def __init__(
        self,
        resource_model: HardwareResourceModel,
        thermal_profile: str = None
    ):
        """
        Initialize GPU mapper.

        Args:
            resource_model: GPU resource model
            thermal_profile: Thermal profile name (e.g., "15W", "30W", "60W")
                           If None, uses default from resource model
        """
        super().__init__(resource_model, thermal_profile=thermal_profile)

        # Validate this is a GPU model
        if resource_model.hardware_type.value != "gpu":
            raise ValueError(f"GPUMapper requires GPU resource model, got {resource_model.hardware_type}")

    def map_subgraph(
        self,
        subgraph: FusedSubgraph,
        execution_stage: int,
        concurrent_subgraphs: int,
        precision: Precision = Precision.FP32
    ) -> HardwareAllocation:
        """
        Map a single fused subgraph to GPU SMs.

        Algorithm:
        1. Get thread count from parallelism descriptor
        2. threads → warps (/ 32)
        3. warps → SMs (/ warps_per_SM)
        4. Apply wave quantization
        5. Calculate occupancy
        6. Calculate latency using roofline model
        """
        # Get thread requirements
        if subgraph.parallelism is None:
            # Fallback: assume minimal parallelism
            threads_required = 256
        else:
            threads_required = subgraph.parallelism.total_threads

        # Calculate warp requirements
        warp_size = self.resource_model.warp_size
        warps_required = math.ceil(threads_required / warp_size)

        # Calculate ideal SM allocation (before quantization)
        warps_per_sm = self.resource_model.warps_per_unit
        sms_ideal = math.ceil(warps_required / warps_per_sm)

        # Apply wave quantization (SMs allocated in groups)
        wave_size = self.resource_model.wave_quantization
        sms_allocated = math.ceil(sms_ideal / wave_size) * wave_size

        # Cap at total available SMs
        sms_allocated = min(sms_allocated, self.resource_model.compute_units)

        # Calculate occupancy (warps actually used / max warps possible)
        max_warps_possible = sms_allocated * warps_per_sm
        occupancy = min(1.0, warps_required / max_warps_possible) if max_warps_possible > 0 else 0.0

        # Calculate utilization (SMs used / total SMs available)
        utilization = sms_allocated / self.resource_model.compute_units

        # Calculate latency using roofline model
        ops = subgraph.total_flops if subgraph.total_flops > 0 else subgraph.total_macs * 2
        bytes_transferred = (
            subgraph.total_input_bytes +
            subgraph.total_output_bytes +
            subgraph.total_weight_bytes
        )

        compute_time, memory_time, bottleneck = self._calculate_latency(
            ops=ops,
            bytes_transferred=bytes_transferred,
            allocated_units=sms_allocated,
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
            subgraph_name=", ".join(subgraph.node_names[:2]),  # First 2 names
            precision=precision,
            threads_required=threads_required,
            warps_required=warps_required,
            compute_units_allocated=sms_allocated,
            compute_units_ideal=sms_ideal,
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
        precision: Precision = Precision.FP32
    ) -> GraphHardwareAllocation:
        """
        Map entire computation graph to GPU.

        Algorithm:
        1. For each execution stage:
           - Map all subgraphs in that stage
           - Calculate max SMs used (subgraphs run in parallel)
        2. Aggregate metrics across all stages
        3. Calculate total latency (sum across stages)
        4. Compare to naive estimate

        Args:
            fusion_report: Output from Phase 1 fusion partitioner
            execution_stages: e.g., [[0,1,2], [3], [4,5]] means:
                              Stage 0: subgraphs 0,1,2 parallel
                              Stage 1: subgraph 3 (sequential)
                              Stage 2: subgraphs 4,5 parallel
            batch_size: Batch size (scales parallelism)
            precision: Numerical precision

        Returns:
            Complete hardware allocation
        """
        subgraph_allocations: List[HardwareAllocation] = []
        latency_breakdown: Dict[int, float] = {}

        peak_sms_used = 0
        total_sms_used = 0
        total_sms_samples = 0

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
                    # Create scaled copy
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

            # For parallel subgraphs in same stage:
            # - Total SMs = max across subgraphs (they share SMs)
            # - Latency = max latency (they run concurrently)
            if stage_allocations:
                stage_sms_used = max(a.compute_units_allocated for a in stage_allocations)
                stage_latency = max(a.estimated_latency for a in stage_allocations)

                peak_sms_used = max(peak_sms_used, stage_sms_used)
                total_sms_used += stage_sms_used
                total_sms_samples += 1

                latency_breakdown[stage_id] = stage_latency

        # Calculate aggregate metrics
        total_subgraphs = len(subgraph_allocations)
        total_execution_stages = len(execution_stages)
        average_sms_used = total_sms_used / total_sms_samples if total_sms_samples > 0 else 0

        total_compute_units = self.resource_model.compute_units
        peak_utilization = peak_sms_used / total_compute_units
        average_utilization = average_sms_used / total_compute_units

        # Total latency = sum of stage latencies (stages are sequential)
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
        compute_bound_count = sum(1 for a in subgraph_allocations if a.bottleneck == BottleneckType.COMPUTE_BOUND)
        memory_bound_count = sum(1 for a in subgraph_allocations if a.bottleneck == BottleneckType.MEMORY_BOUND)
        bandwidth_bound_count = sum(1 for a in subgraph_allocations if a.bottleneck == BottleneckType.BANDWIDTH_BOUND)
        balanced_count = sum(1 for a in subgraph_allocations if a.bottleneck == BottleneckType.BALANCED)

        return GraphHardwareAllocation(
            model_name="Unknown",  # Will be set by caller
            hardware_name=self.resource_model.name,
            batch_size=batch_size,
            model_precision=precision,
            subgraph_allocations=subgraph_allocations,
            total_subgraphs=total_subgraphs,
            total_execution_stages=total_execution_stages,
            peak_compute_units_used=peak_sms_used,
            average_compute_units_used=average_sms_used,
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


def create_h100_mapper(thermal_profile: str = None) -> GPUMapper:
    """
    Create GPU mapper for NVIDIA H100 PCIe.

    Args:
        thermal_profile: Thermal profile name (if applicable)

    Returns:
        GPUMapper configured for H100
    """
    from ..resource_model import h100_pcie_resource_model
    return GPUMapper(h100_pcie_resource_model(), thermal_profile=thermal_profile)


def create_jetson_orin_agx_mapper(thermal_profile: str = None) -> GPUMapper:
    """
    Create GPU mapper for NVIDIA Jetson Orin AGX (edge AI platform).

    Args:
        thermal_profile: Thermal profile name (e.g., "15W", "30W", "60W")
                        If None, uses default ("15W")

    Returns:
        GPUMapper configured for Jetson Orin AGX
    """
    from ..resource_model import jetson_orin_agx_resource_model
    return GPUMapper(jetson_orin_agx_resource_model(), thermal_profile=thermal_profile)


def create_jetson_orin_nano_mapper(thermal_profile: str = None) -> GPUMapper:
    """
    Create GPU mapper for NVIDIA Jetson Orin Nano (compact edge AI platform).

    Args:
        thermal_profile: Thermal profile name (e.g., "7W", "15W")
                        If None, uses default ("7W")

    Returns:
        GPUMapper configured for Jetson Orin Nano
    """
    from ..resource_model import jetson_orin_nano_resource_model
    return GPUMapper(jetson_orin_nano_resource_model(), thermal_profile=thermal_profile)


def create_jetson_thor_mapper(thermal_profile: str = None) -> GPUMapper:
    """
    Create GPU mapper for NVIDIA Jetson Thor (next-gen edge AI).

    Args:
        thermal_profile: Thermal profile name (e.g., "30W", "60W", "100W")
                        If None, uses default ("30W")

    Returns:
        GPUMapper configured for Jetson Thor
    """
    from ..resource_model import jetson_thor_resource_model
    return GPUMapper(jetson_thor_resource_model(), thermal_profile=thermal_profile)


# ============================================================================
# ARM Mali GPU IP Mappers
# ============================================================================

def create_arm_mali_g78_mp20_mapper(thermal_profile: str = None) -> GPUMapper:
    """
    Create GPU mapper for ARM Mali-G78 MP20 GPU IP Core.

    ARCHITECTURE:
    - Licensable mobile GPU IP core for SoC integration
    - 2nd generation Valhall architecture
    - 20 shader cores (MP20 configuration)
    - Unified shader architecture (graphics + compute)

    PERFORMANCE:
    - Graphics: 1.94 TFLOPS FP32 @ 848 MHz
    - Compute: ~2 TOPS INT8 (not AI-optimized)
    - FP16: 3.88 TFLOPS (2× FP32)

    PRECISION SUPPORT:
    - FP32: Native (graphics primary)
    - FP16: Native (2× FP32 throughput)
    - INT8: Supported but not optimized

    MEMORY:
    - 2 MB L2 cache (typical)
    - 40 GB/s bandwidth (typical SoC integration)
    - Up to 8 GB external memory

    POWER:
    - 5W typical TDP @ 848 MHz
    - Passive mobile cooling
    - DVFS for power management

    USE CASE:
    - Mobile gaming (flagship smartphones)
    - Computational photography
    - Light AI inference (not primary use)
    - AR/VR rendering
    - Used in Google Tensor (Pixel 6/6 Pro)

    CALIBRATION STATUS:
    ⚠ ESTIMATED - Based on ARM specs and Google Tensor
    - Graphics-optimized, not AI-optimized
    - For AI workloads, pair with dedicated NPU

    REFERENCES:
    - ARM Mali-G78 Product Brief (2020)
    - Google Tensor SoC documentation
    - AnandTech Mali-G78 analysis

    Args:
        thermal_profile: Thermal profile name (e.g., "5W")
                        If None, uses default ("5W")

    Returns:
        GPUMapper configured for ARM Mali-G78 MP20
    """
    from ..resource_model import arm_mali_g78_mp20_resource_model
    return GPUMapper(arm_mali_g78_mp20_resource_model(), thermal_profile=thermal_profile)
