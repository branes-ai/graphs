"""
DSP Hardware Mapper - Digital Signal Processors for AI Acceleration

This module implements mappers for DSP-based AI accelerators:

Qualcomm DSPs:
- Qualcomm Hexagon 698 (QRB5165): 15 TOPS INT8, robotics platform

Texas Instruments C7x DSPs:
- TI TDA4VM (Jacinto 7): 8 TOPS INT8, automotive ADAS
- TI TDA4VL: 4 TOPS INT8, entry-level ADAS
- TI TDA4AL: 8 TOPS INT8, mid-range ADAS (MMAv2)
- TI TDA4VH: 32 TOPS INT8, high-performance L3-4 autonomy

Licensable IP Cores:
- CEVA NeuPro-M NPM11: 20 TOPS INT8, edge AI NPU IP
- Cadence Tensilica Vision Q8: 3.8 TOPS INT8, vision DSP IP
- Synopsys ARC EV7x: 35 TOPS INT8, embedded vision processor IP

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
- SoC integration (IP licensing)
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional

from ..resource_model import (
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
from ..models.edge.qrb5165 import qrb5165_resource_model
from ..models.automotive.ti_tda4vm import ti_tda4vm_resource_model
from ..models.automotive.ti_tda4vl import ti_tda4vl_resource_model
from ..models.automotive.ti_tda4al import ti_tda4al_resource_model
from ..models.automotive.ti_tda4vh import ti_tda4vh_resource_model
from ..models.ip_cores.ceva_neupro_npm11 import ceva_neupro_npm11_resource_model
from ..models.ip_cores.cadence_vision_q8 import cadence_vision_q8_resource_model
from ..models.ip_cores.synopsys_arc_ev7x import synopsys_arc_ev7x_resource_model
from graphs.transform.partitioning import FusedSubgraph, FusionReport
from graphs.core.structures import SubgraphDescriptor, ParallelismDescriptor


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

    # Idle power modeling (nanoscale leakage)
    IDLE_POWER_FRACTION = 0.5  # 50% of TDP consumed at idle due to leakage

    def __init__(self, resource_model: HardwareResourceModel, thermal_profile: Optional[str] = None):
        super().__init__(resource_model, thermal_profile=thermal_profile)

        # Validate this is a DSP model
        if resource_model.hardware_type != HardwareType.DSP:
            raise ValueError(f"DSPMapper requires DSP resource model, got {resource_model.hardware_type}")

        # DSP-specific parameters
        self.dsp_units = resource_model.compute_units  # Number of DSP processing elements
        self.vector_width = resource_model.threads_per_unit * 32  # Estimated vector width in bits

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
# Texas Instruments C7x DSP Mappers
# ============================================================================

def create_ti_tda4vm_mapper(thermal_profile: str = "10W") -> DSPMapper:
    """
    Create hardware mapper for Texas Instruments TDA4VM (Jacinto 7) ADAS Processor.

    ARCHITECTURE:
    - C7x floating-point vector DSP @ 1.0 GHz
    - Matrix Multiply Accelerator (MMA) integrated with C7x
    - Heterogeneous: Cortex-A72 CPU + C7x DSP + MMA
    - Automotive-grade: ASIL-D/SIL-3 safety certification

    PERFORMANCE:
    - C7x DSP: 80 GFLOPS FP32, 256 GOPS INT16
    - MMA: 8 TOPS INT8 (peak)
    - ~5 TOPS INT8 (effective @ 10W)
    - ~6.5 TOPS INT8 (effective @ 20W)

    PRECISION SUPPORT:
    - INT8: Native via MMA (primary mode for CNNs)
    - INT16: Native via C7x vector units
    - FP32: Native via C7x DSP (80 GFLOPS)
    - FP16: Emulated via C7x

    MEMORY:
    - LPDDR4x @ 3733 MT/s (dual-channel)
    - 60 GB/s bandwidth
    - 8 MB MSMC on-chip SRAM for DSP
    - Up to 8GB capacity

    POWER PROFILES:
    - 10W: Front camera ADAS (lane detection, object detection)
      * Sustained: 850 MHz, ~5 TOPS effective
      * Use case: Single front-facing camera system

    - 20W: Full ADAS system (multi-camera, radar fusion, parking)
      * Sustained: 950 MHz, ~6.5 TOPS effective
      * Use case: 4-6 cameras + radar/lidar fusion

    AUTOMOTIVE FEATURES:
    - Temperature: -40°C to 125°C (AEC-Q100 Grade 2)
    - Safety: ASIL-D/SIL-3 with R5F safety cores
    - Deterministic scheduling for real-time
    - Secure boot and runtime protection

    USE CASE:
    - ADAS Level 2-3 (lane keep, adaptive cruise, auto park)
    - Multi-camera surround view
    - Sensor fusion (camera + radar + lidar)
    - Object detection and tracking
    - Lane detection and segmentation

    CALIBRATION STATUS:
    ⚠ ESTIMATED - Based on TI published specs and automotive benchmarks
    - efficiency_factor values are conservative for automotive
    - Need empirical benchmarking on actual TDA4VM hardware

    REFERENCES:
    - TI TDA4VM Datasheet (SPRS927)
    - Jacinto 7 Technical Reference Manual
    - TI Edge AI documentation
    - Automotive ADAS performance data

    Args:
        thermal_profile: Power mode - "10W" (front camera) or "20W" (full system)

    Returns:
        DSPMapper configured for TI TDA4VM
    """
    model = ti_tda4vm_resource_model()
    mapper = DSPMapper(model)

    # Set the thermal profile if specified
    if thermal_profile in model.thermal_operating_points:
        model._active_thermal_profile = thermal_profile

    return mapper


def create_ti_tda4vl_mapper(thermal_profile: str = "7W") -> DSPMapper:
    """
    Create hardware mapper for Texas Instruments TDA4VL (Entry-Level ADAS).

    ARCHITECTURE:
    - Entry-level Jacinto 7 for cost-sensitive ADAS applications
    - C7x DSP + MMAv2 (newer generation, more efficient than TDA4VM's MMAv1)
    - 4 TOPS INT8 (half of TDA4VM)
    - Automotive-grade: ASIL-B/C

    PERFORMANCE:
    - 4 TOPS INT8 @ 1.0 GHz (peak)
    - ~2-3 TOPS INT8 (effective @ 7W sustained)
    - ~3 TOPS INT8 (effective @ 12W sustained)

    PRECISION SUPPORT:
    - INT8: Native via MMAv2 (primary mode)
    - INT16: Native via C7x
    - FP32: Native via C7x DSP (40 GFLOPS)

    MEMORY:
    - LPDDR4x @ 3733 MT/s
    - 60 GB/s bandwidth
    - Up to 4GB capacity

    POWER PROFILES:
    - 7W: Entry-level ADAS (single camera, lane keep, TSR)
    - 12W: Multi-function ADAS (front + side cameras)

    USE CASES:
    - Entry-level ADAS (Lane Keep Assist, Traffic Sign Recognition)
    - Single front-facing camera systems
    - Cost-sensitive automotive markets

    CALIBRATION STATUS:
    ⚠ ESTIMATED - Based on TI published specs and TDA4VM benchmarks

    Args:
        thermal_profile: Power mode - "7W" (entry-level) or "12W" (multi-function)

    Returns:
        DSPMapper configured for TI TDA4VL
    """
    model = ti_tda4vl_resource_model()
    mapper = DSPMapper(model)

    if thermal_profile in model.thermal_operating_points:
        model._active_thermal_profile = thermal_profile

    return mapper


def create_ti_tda4al_mapper(thermal_profile: str = "10W") -> DSPMapper:
    """
    Create hardware mapper for Texas Instruments TDA4AL (Mid-Range ADAS).

    ARCHITECTURE:
    - Mid-range Jacinto 7 with newer MMAv2 architecture
    - C7x DSP + MMAv2 (more efficient than TDA4VM's MMAv1)
    - 8 TOPS INT8 (same as TDA4VM but more efficient)
    - Automotive-grade: ASIL-D/SIL-3

    PERFORMANCE:
    - 8 TOPS INT8 @ 1.0 GHz (peak)
    - ~5 TOPS INT8 (effective @ 10W sustained)
    - ~6.5 TOPS INT8 (effective @ 18W sustained)
    - Better power efficiency than TDA4VM @ same power level

    PRECISION SUPPORT:
    - INT8: Native via MMAv2 (primary mode)
    - INT16: Native via C7x
    - FP32: Native via C7x DSP (80 GFLOPS)

    MEMORY:
    - LPDDR4x @ 3733 MT/s
    - 60 GB/s bandwidth
    - Up to 8GB capacity

    POWER PROFILES:
    - 10W: Front camera ADAS
    - 18W: Multi-camera ADAS (vs TDA4VM @ 20W for similar performance)

    USE CASES:
    - ADAS Level 2-3 (replaces TDA4VM in newer designs)
    - Better power efficiency for same performance
    - Multi-camera sensor fusion

    CALIBRATION STATUS:
    ⚠ ESTIMATED - Based on TI published specs and TDA4VM benchmarks

    Args:
        thermal_profile: Power mode - "10W" (front camera) or "18W" (multi-camera)

    Returns:
        DSPMapper configured for TI TDA4AL
    """
    model = ti_tda4al_resource_model()
    mapper = DSPMapper(model)

    if thermal_profile in model.thermal_operating_points:
        model._active_thermal_profile = thermal_profile

    return mapper


def create_ti_tda4vh_mapper(thermal_profile: str = "20W") -> DSPMapper:
    """
    Create hardware mapper for Texas Instruments TDA4VH (High-Performance ADAS).

    ARCHITECTURE:
    - High-performance Jacinto 7 for Level 3-4 autonomous driving
    - 4× C7x DSP + 4× MMAv2 accelerators (4× TDA4VM)
    - 8× Cortex-A72 @ 2.0 GHz (vs 2× in TDA4VM)
    - 32 TOPS INT8 (4× TDA4VM)
    - Automotive-grade: ASIL-D/SIL-3

    PERFORMANCE:
    - 32 TOPS INT8 @ 1.0 GHz (peak)
    - ~20 TOPS INT8 (effective @ 20W sustained)
    - ~25 TOPS INT8 (effective @ 35W sustained)

    PRECISION SUPPORT:
    - INT8: Native via 4× MMAv2 (primary mode)
    - INT16: Native via 4× C7x
    - FP32: Native via 4× C7x DSP (320 GFLOPS total)

    MEMORY:
    - LPDDR5 @ 6400 MT/s
    - 100 GB/s bandwidth (higher than TDA4VM)
    - Up to 16GB capacity

    POWER PROFILES:
    - 20W: Multi-camera Level 2+ ADAS (4-6 cameras)
    - 35W: Full Level 3-4 autonomy stack (8-12 cameras + lidar/radar)

    USE CASES:
    - Advanced ADAS Level 3-4
    - Highway pilot, urban pilot
    - 8-12 camera surround view + lidar/radar fusion
    - Multi-modal sensor fusion

    CALIBRATION STATUS:
    ⚠ ESTIMATED - Based on TI published specs and TDA4VM scaling

    Args:
        thermal_profile: Power mode - "20W" (L2+ ADAS) or "35W" (L3-4 autonomy)

    Returns:
        DSPMapper configured for TI TDA4VH
    """
    model = ti_tda4vh_resource_model()
    mapper = DSPMapper(model)

    if thermal_profile in model.thermal_operating_points:
        model._active_thermal_profile = thermal_profile

    return mapper


# ============================================================================
# CEVA NeuPro Neural Processing IP Mappers
# ============================================================================

def create_ceva_neupro_npm11_mapper() -> DSPMapper:
    """
    Create hardware mapper for CEVA NeuPro-M NPM11 Neural Processing IP.

    ARCHITECTURE:
    - Licensable NPU IP core for SoC integration
    - Single NeuPro-M engine configuration
    - Heterogeneous: Tensor + Vector + Scalar units
    - Designed for mobile, automotive, and IoT applications

    PERFORMANCE:
    - 20 TOPS INT8 @ 1.25 GHz (peak)
    - ~14 TOPS INT8 (effective @ 2W sustained)
    - ~10 TOPS/W efficiency

    PRECISION SUPPORT:
    - INT8: Native, primary mode
    - INT16: Native (10 TOPS @ 1.25 GHz)
    - FP16: Supported (10 TFLOPS @ 1.25 GHz)
    - INT4: Native (40 TOPS @ 1.25 GHz, 2× INT8)

    MEMORY:
    - 2-4 MB local SRAM (configurable)
    - 50 GB/s bandwidth (typical SoC integration)
    - Up to 8 GB external DRAM

    POWER:
    - 2W typical TDP
    - Passive mobile cooling
    - Highly power-efficient for edge AI

    USE CASE:
    - Mobile devices (always-on AI, camera processing)
    - Automotive ADAS (sensor fusion, camera AI)
    - IoT devices (edge inference)
    - Smart cameras and drones
    - Wearables and hearables

    CALIBRATION STATUS:
    ⚠ ESTIMATED - Based on CEVA published specifications
    - Need empirical benchmarking on actual silicon
    - Performance varies based on SoC integration

    REFERENCES:
    - CEVA NeuPro-M Product Brief
    - NPM11 configuration datasheet
    - CEVA press releases (2021-2024)
    """
    model = ceva_neupro_npm11_resource_model()
    return DSPMapper(model)


# ============================================================================
# Cadence Tensilica Vision DSP IP Mappers
# ============================================================================

def create_cadence_vision_q8_mapper() -> DSPMapper:
    """
    Create hardware mapper for Cadence Tensilica Vision Q8 DSP IP (7th Gen).

    ARCHITECTURE:
    - Licensable vision DSP IP core for SoC integration
    - 7th generation Tensilica Vision DSP (flagship)
    - 1024-bit SIMD engine
    - Vector + Scalar architecture

    PERFORMANCE:
    - 3.8 TOPS INT8/INT16 @ 1.0 GHz
    - 129 GFLOPS FP32
    - 2× performance of Vision Q7

    PRECISION SUPPORT:
    - INT8/INT16: 3.8 TOPS each (vision-optimized)
    - FP32: 129 GFLOPS
    - FP16: ~258 GFLOPS (estimated)

    MEMORY:
    - 512 KB - 2 MB local SRAM (configurable)
    - 40 GB/s bandwidth (typical SoC integration)
    - Up to 4 GB external memory

    POWER:
    - 1W typical TDP
    - Passive mobile cooling
    - ~3-7 TOPS/W efficiency

    USE CASE:
    - Automotive ADAS cameras (ISP + AI vision)
    - Mobile device cameras (computational photography)
    - Surveillance cameras (edge analytics)
    - AR/VR vision processing
    - Drone vision systems

    CALIBRATION STATUS:
    ⚠ ESTIMATED - Based on Cadence published specifications
    - Need empirical benchmarking on actual silicon
    - 2021 announcement, multiple tape-outs since then

    REFERENCES:
    - Cadence Tensilica Vision Q8 Product Brief (2021)
    - Tensilica Vision DSP Family documentation
    - Cadence DesignWare IP catalog
    """
    model = cadence_vision_q8_resource_model()
    return DSPMapper(model)


# ============================================================================
# Synopsys ARC EV Embedded Vision Processor IP Mappers
# ============================================================================

def create_synopsys_arc_ev7x_mapper() -> DSPMapper:
    """
    Create hardware mapper for Synopsys ARC EV7x Embedded Vision Processor IP.

    ARCHITECTURE:
    - Licensable embedded vision processor IP
    - 1-4 core heterogeneous configuration
    - Each core: 512-bit vector DSP (VPU) + DNN accelerator
    - ARCv2 RISC ISA base
    - DNN accelerator: 880-14,080 MACs (scalable)

    PERFORMANCE:
    - 35 TOPS INT8 @ 1.0 GHz (4-core, full config)
    - ~24 TOPS INT8 (effective @ 5W sustained)
    - ~7-10 TOPS/W efficiency
    - 4× performance of ARC EV6x

    PRECISION SUPPORT:
    - INT8: 35 TOPS (primary mode via DNN accelerator)
    - INT16: 17.5 TOPS (via VPUs)
    - INT32: Supported (8.75 TOPS)
    - FP32: ~8.8 GFLOPS (4 cores × 2.2 GFLOPS)

    MEMORY:
    - 2-8 MB local memory (configurable)
    - 60 GB/s bandwidth (automotive SoC integration)
    - Up to 8 GB external DRAM

    POWER:
    - 5W typical TDP (4-core configuration)
    - Automotive passive cooling
    - Automotive-grade power management

    USE CASE:
    - Automotive ADAS Level 2-3 (camera processing)
    - Multi-camera surround view systems
    - Surveillance edge analytics
    - Drone/robot vision systems
    - AR/VR embedded vision

    CALIBRATION STATUS:
    ⚠ ESTIMATED - Based on Synopsys published specifications
    - 2019 announcement, automotive tape-outs since then
    - Need empirical benchmarking on actual silicon

    REFERENCES:
    - Synopsys ARC EV7x Product Brief (2019)
    - EE Times: "Synopsys ARC Embedded Vision Processors Deliver 35 TOPS" (2019)
    - Synopsys DesignWare ARC EV Processor IP catalog
    - ARC EV7xFS (functional safety variant) documentation
    """
    model = synopsys_arc_ev7x_resource_model()
    return DSPMapper(model)


def create_qualcomm_sa8775p_mapper(thermal_profile: str = "30W") -> DSPMapper:
    """
    Create hardware mapper for Qualcomm SA8775P Snapdragon Ride.

    ARCHITECTURE:
    - Hexagon DSP with dual HMX (Hexagon Matrix eXtensions)
    - Quad HVX vector processors
    - 5nm TSMC automotive-grade (ASIL D certified)
    - Mixed precision: INT4/INT8/INT16/FP16

    PERFORMANCE:
    - 32 TOPS mixed precision (estimated)
    - 16-20 TOPS INT8 sustained (~50-60% efficiency)
    - Better efficiency than Jetson Orin AGX (2-6%)
    - 20-45W TDP range (automotive active cooling)

    PRECISION SUPPORT:
    - INT8: Primary for CNN inference
    - INT4: Supported for quantized models
    - INT16/FP16: Mixed precision workflows

    MEMORY:
    - LPDDR5 system memory (automotive-grade)
    - Large L2 cache for Hexagon DSP
    - TCM (Tightly-Coupled Memory) for low-latency access

    USE CASE:
    - Automotive ADAS Level 2+/3 (highway pilot, parking)
    - Multi-camera vision processing
    - Cockpit compute (HMI, voice, driver monitoring)
    - Robotics (AMR, inspection robots)

    CALIBRATION STATUS:
    ⚠ ESTIMATED - Limited public benchmarks
    - TOPS estimate based on automotive SoC market positioning
    - Hexagon efficiency: 50-60% (better than GPU, worse than dedicated accelerators)
    """
    from ..models.automotive.qualcomm_sa8775p import qualcomm_sa8775p_resource_model
    model = qualcomm_sa8775p_resource_model()
    return DSPMapper(model, thermal_profile=thermal_profile)


def create_qualcomm_snapdragon_ride_mapper(thermal_profile: str = "100W") -> DSPMapper:
    """
    Create hardware mapper for Qualcomm Snapdragon Ride Flex.

    ARCHITECTURE:
    - Multi-chip automotive SoC (ASIL D certified)
    - Hexagon AI accelerators with HMX/HVX
    - Dedicated computer vision accelerator
    - ISP and video codec engines
    - 4nm TSMC process

    PERFORMANCE:
    - 700 TOPS mixed precision (peak advertised)
    - ~350-420 TOPS INT8 sustained (~50-60% efficiency)
    - Scalable architecture (Ride-1 to Ride-6+ configurations)
    - 65-130W TDP range (automotive liquid cooling)

    PRECISION SUPPORT:
    - INT4: Optimal for quantized models
    - INT8: High performance CNN inference
    - INT16/FP16: Mixed precision support
    - BF16: Supported for training/fine-tuning

    MEMORY:
    - 32-64GB LPDDR5 (automotive-grade)
    - Large cache hierarchy
    - High bandwidth for multi-sensor fusion

    USE CASE:
    - Level 4/5 autonomous driving
    - Multi-camera sensor fusion (up to 12 cameras)
    - Radar/Lidar processing
    - End-to-end neural planning

    CALIBRATION STATUS:
    ⚠ ESTIMATED - Pre-production platform
    - TOPS based on Qualcomm marketing materials
    - Efficiency estimates based on Hexagon architecture analysis
    - First vehicles with Snapdragon Ride expected 2025-2026
    """
    from ..models.automotive.qualcomm_snapdragon_ride import qualcomm_snapdragon_ride_resource_model
    model = qualcomm_snapdragon_ride_resource_model()
    return DSPMapper(model, thermal_profile=thermal_profile)
