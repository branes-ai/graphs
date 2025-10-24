"""
DSP Hardware Mapper - Digital Signal Processors for AI Acceleration

This module implements mappers for DSP-based AI accelerators:
- Qualcomm Hexagon 698 (QRB5165): 15 TOPS INT8, robotics platform
- (Future) Texas Instruments C7x DSP
- (Future) Cadence Tensilica Vision DSPs
- (Future) CEVA NeuPro DSPs

DSP Architecture Characteristics:
- Heterogeneous compute: Vector + Tensor + Scalar units
- Dataflow-style execution with sophisticated scheduling
- Optimized for signal processing + AI workloads
- Typically integrated in SoCs (CPU + GPU + DSP)

Key Features:
- Native INT8/INT16 support
- High efficiency on edge devices (2-10 TOPS/W)
- Integrated sensor processing capabilities
- Lower power than GPUs, higher flexibility than fixed accelerators

Use Cases:
- Robotics platforms (sensor fusion + vision)
- Mobile devices (always-on AI)
- Edge IoT (audio + vision processing)
- Automotive (ADAS sensor fusion)
"""

from dataclasses import dataclass
from typing import List, Tuple

from .hardware_mapper import (
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
    qrb5165_resource_model,
)
from .fusion_partitioner import FusedSubgraph, FusionReport
from .graph_structures import SubgraphDescriptor, ParallelismDescriptor


class DSPMapper(HardwareMapper):
    """
    Generic DSP hardware mapper for AI acceleration.

    Implements realistic DSP mapping considering:
    - Vector vs tensor resource allocation
    - Memory bandwidth constraints
    - Thermal throttling (DVFS)
    - Quantization efficiency (INT8/INT16)
    - Power efficiency optimization
    """

    def __init__(self, resource_model: HardwareResourceModel):
        super().__init__(resource_model)

        # Validate this is a DSP model
        if resource_model.hardware_type != HardwareType.DSP:
            raise ValueError(f"DSPMapper requires DSP resource model, got {resource_model.hardware_type}")

        # DSP-specific parameters
        self.dsp_units = resource_model.compute_units  # Number of DSP processing elements
        self.vector_width = resource_model.threads_per_unit * 32  # Estimated vector width in bits

    def map_subgraph(
        self,
        subgraph: FusedSubgraph,
        execution_stage: int,
        concurrent_subgraphs: int,
        precision: Precision = Precision.INT8
    ) -> HardwareAllocation:
        """
        Map a single fused subgraph to DSP architecture.

        Algorithm:
        1. Determine vector vs tensor resource requirements
        2. Calculate memory bandwidth needs
        3. Estimate latency using roofline model
        4. Account for DVFS thermal throttling
        """
        # Get parallelism
        if subgraph.parallelism is None:
            # Fallback: assume single DSP unit execution
            units_required = 1
        else:
            # DSP distributes work across vector and tensor units
            # Larger workloads benefit from more parallel DSP units
            batch = subgraph.parallelism.batch
            channels = subgraph.parallelism.channels

            # Estimate DSP resource allocation based on parallelism
            # Conv/Linear layers can use multiple vector + tensor units
            units_required = min(batch * max(1, channels // 32), self.dsp_units)

        units_allocated = max(1, min(units_required, self.dsp_units))

        # Calculate occupancy (what fraction of DSP resources are busy)
        occupancy = units_allocated / self.dsp_units

        # Calculate utilization (effective usage considering memory bottlenecks)
        utilization = occupancy

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
            allocated_units=units_allocated,
            occupancy=occupancy,
            precision=precision
        )

        # DSP has sophisticated scheduling - minimal overhead
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
            threads_required=units_allocated * self.resource_model.threads_per_unit,
            warps_required=0,  # N/A for DSP
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
        Map entire computation graph to DSP architecture.

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
        compute_bound = sum(1 for a in subgraph_allocations if a.bottleneck == BottleneckType.COMPUTE_BOUND)
        memory_bound = sum(1 for a in subgraph_allocations if a.bottleneck == BottleneckType.MEMORY_BOUND)
        bandwidth_bound = sum(1 for a in subgraph_allocations if a.bottleneck == BottleneckType.BANDWIDTH_BOUND)
        balanced = sum(1 for a in subgraph_allocations if a.bottleneck == BottleneckType.BALANCED)

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
            compute_bound_count=compute_bound,
            memory_bound_count=memory_bound,
            bandwidth_bound_count=bandwidth_bound,
            balanced_count=balanced,
        )


# ============================================================================
# Qualcomm Hexagon DSP Mappers
# ============================================================================

def create_qrb5165_mapper() -> DSPMapper:
    """
    Create hardware mapper for Qualcomm QRB5165 Robotics Platform.

    ARCHITECTURE:
    - Hexagon 698 DSP with HVX (vector) and HTA (tensor) accelerators
    - 4× HVX 1024-bit SIMD vector units
    - Dedicated tensor accelerator for matrix operations
    - Heterogeneous: CPU (Kryo 585) + GPU (Adreno 650) + DSP (Hexagon 698)

    PERFORMANCE:
    - 15 TOPS INT8 (peak)
    - ~6 TOPS INT8 (effective @ 7W sustained)
    - ~2 TOPS/W efficiency

    PRECISION SUPPORT:
    - INT8: Native, optimized (primary mode)
    - INT16: Native, good performance
    - FP16: Supported but slower
    - INT4: Experimental support

    MEMORY:
    - LPDDR5 up to 2750 MHz
    - 44 GB/s bandwidth
    - Up to 16GB capacity

    POWER:
    - 7W typical TDP
    - DVFS: 60% sustained clock (900 MHz from 1.5 GHz peak)
    - Moderate thermal throttling

    USE CASE:
    - Battery-powered robots and drones
    - Edge AI with sensor fusion (camera + IMU + GNSS)
    - Robotics platforms requiring multi-modal processing
    - Qualcomm ecosystem (ROS, Snapdragon SDK)

    CALIBRATION STATUS:
    ⚠ ESTIMATED - Based on published specs and Snapdragon 865 architecture
    - efficiency_factor values are conservative estimates
    - Need empirical benchmarking on actual QRB5165 hardware

    REFERENCES:
    - Qualcomm QRB5165 Product Brief (87-28730-1 REV D)
    - Snapdragon 865 specifications
    - Hexagon 698 DSP architecture documentation
    """
    model = qrb5165_resource_model()
    return DSPMapper(model)


# ============================================================================
# Future DSP Mappers (Placeholders)
# ============================================================================

# TODO: Add TI C7x DSP mapper
# def create_ti_c7x_mapper() -> DSPMapper:
#     """Texas Instruments C7x DSP for automotive/ADAS"""
#     pass

# TODO: Add Cadence Tensilica Vision DSP mapper
# def create_cadence_vision_dsp_mapper() -> DSPMapper:
#     """Cadence Tensilica Vision DSPs"""
#     pass

# TODO: Add CEVA NeuPro DSP mapper
# def create_ceva_neupro_mapper() -> DSPMapper:
#     """CEVA NeuPro AI DSPs"""
#     pass
