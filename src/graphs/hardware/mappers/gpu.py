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
    - Sequential execution mode for small DNNs (batch=1, limited parallelism)
    """

    # Kernel launch overhead (empirical: ~5-10 µs per kernel)
    KERNEL_LAUNCH_OVERHEAD = 10e-6  # 10 microseconds

    # Idle power fraction (nanoscale transistor leakage)
    # Modern GPUs consume ~50% TDP at idle
    IDLE_POWER_FRACTION = 0.5

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

    def should_use_sequential_execution(
        self,
        fusion_report: FusionReport,
        batch_size: int
    ) -> bool:
        """
        Determine if we should model sequential kernel execution.

        Sequential mode is appropriate when:
        - Small batch size (< 8)
        - Small average workload per subgraph (< 100M FLOPs)

        This represents the common case of single-sample inference on small DNNs
        where there isn't enough parallelism to saturate many SMs.

        Args:
            fusion_report: Fusion partitioning report
            batch_size: Batch size

        Returns:
            True if sequential execution should be modeled
        """
        num_subgraphs = len(fusion_report.fused_subgraphs)
        if num_subgraphs == 0:
            return False

        avg_flops_per_subgraph = fusion_report.total_flops / num_subgraphs

        # Threshold: If average subgraph has < 200M FLOPs, use sequential mode
        # This catches ResNet-18/50, MobileNet, and other small inference workloads
        # Each kernel can't efficiently saturate many SMs at this scale
        return (batch_size < 8 and avg_flops_per_subgraph < 200e6)

    def determine_sm_allocation(
        self,
        subgraph: FusedSubgraph
    ) -> int:
        """
        Determine how many SMs to allocate for a subgraph in sequential mode.

        SM allocation based on PARALLELISM (output elements), not FLOPs.
        Utilization is about hardware occupancy, not efficiency.

        Key insight: GPU kernels launch enough threads to saturate hardware.
        - 1 thread per output element (for matmul, conv, elementwise ops)
        - Threads distributed across SMs via warp scheduler
        - Utilization should be ~100% for any workload with >10K outputs

        Thread allocation:
        - Output elements → threads (1:1 mapping)
        - Threads → warps (/ 32)
        - Warps → SMs (/ warps_per_SM, typically 48 for Ampere)
        - SM count capped at total available SMs

        Args:
            subgraph: Fused subgraph to analyze

        Returns:
            Number of SMs to allocate (1 to total_sms)
        """
        # Calculate output elements (determines thread count)
        # Each output element gets 1 thread (for matmul, conv, elementwise)
        # For now, estimate from total_output_bytes (assumes 4 bytes/element for FP32)
        # TODO: Use actual element count from output_tensors when available
        bytes_per_element = 4  # FP32 default (conservative estimate)
        output_elements = max(1, subgraph.total_output_bytes // bytes_per_element)

        # Convert to warps (32 threads/warp)
        warp_size = 32
        warps_required = math.ceil(output_elements / warp_size)

        # Convert to SMs (assume 48 warps/SM for Ampere)
        warps_per_sm = getattr(self.resource_model, 'warps_per_unit', 48)
        sms_needed = math.ceil(warps_required / warps_per_sm)

        # Cap at total available SMs
        total_sms = self.resource_model.compute_units
        sms_allocated = min(sms_needed, total_sms)

        # Ensure at least 1 SM (even for tiny kernels)
        return max(1, sms_allocated)

    def compute_sequential_latency(
        self,
        fusion_report: FusionReport,
        precision: Precision
    ) -> Tuple[float, List[HardwareAllocation]]:
        """
        Compute latency assuming sequential kernel execution.

        For small DNNs with limited parallelism:
        - Each subgraph launches 1 kernel
        - Kernels execute sequentially (no overlap)
        - Each kernel uses 1-8 SMs (not all 132)
        - Add kernel launch overhead (~10 µs per kernel)

        This is 100× more accurate than assuming full SM parallelization.

        Args:
            fusion_report: Fusion partitioning report
            precision: Numerical precision

        Returns:
            (total_latency, list of allocations)
        """
        allocations = []
        total_latency = 0.0

        for idx, sg in enumerate(fusion_report.fused_subgraphs):
            # Determine SM allocation for this subgraph
            sms_allocated = self.determine_sm_allocation(sg)

            # Compute per-SM throughput using microarchitecture parameters
            if (self.resource_model.cuda_cores_per_sm is not None and
                self.resource_model.sm_sustained_clock_hz is not None and
                self.resource_model.ops_per_clock_per_core is not None):
                # Use microarchitecture model: CUDA cores × ops/clock × frequency
                # This is more accurate than dividing peak performance by SM count
                sm_flops = (self.resource_model.cuda_cores_per_sm *
                            self.resource_model.ops_per_clock_per_core *
                            self.resource_model.sm_sustained_clock_hz)
            else:
                # Fallback to precision profile (for older models without microarch data)
                peak_flops = self.resource_model.get_peak_ops(precision)
                sm_flops = peak_flops / self.resource_model.compute_units

            # Memory bandwidth is SHARED across all SMs (not per-SM!)
            # Unlike compute which scales linearly, bandwidth is a global resource
            peak_bandwidth = self.resource_model.peak_bandwidth

            # Roofline model on allocated SMs
            ops = sg.total_flops if sg.total_flops > 0 else sg.total_macs * 2
            bytes_transferred = (
                sg.total_input_bytes +
                sg.total_output_bytes +
                sg.total_weight_bytes
            )

            compute_time = ops / (sm_flops * sms_allocated)
            memory_time = bytes_transferred / peak_bandwidth  # Bandwidth is shared, not per-SM!

            # Bottleneck is the limiting factor
            kernel_time = max(compute_time, memory_time)
            bottleneck = (BottleneckType.COMPUTE_BOUND if compute_time > memory_time
                         else BottleneckType.BANDWIDTH_BOUND)

            # Add kernel launch overhead (unless disabled for micro-benchmarks)
            # Check execution_context for disable_launch_overhead flag
            disable_overhead = getattr(self, 'disable_launch_overhead', False)
            if disable_overhead:
                kernel_latency = kernel_time  # Pure compute time only
            else:
                kernel_latency = kernel_time + self.KERNEL_LAUNCH_OVERHEAD

            # Calculate energy for this kernel
            compute_energy, memory_energy = self._calculate_energy(
                ops=ops,
                bytes_transferred=bytes_transferred,
                precision=precision
            )

            # Utilization: fraction of allocated SMs actually used
            utilization = sms_allocated / self.resource_model.compute_units

            allocation = HardwareAllocation(
                subgraph_id=str(sg.subgraph_id),
                subgraph_name=", ".join(sg.node_names[:2]),
                precision=precision,
                threads_required=0,  # Not relevant in sequential mode
                warps_required=0,
                compute_units_allocated=sms_allocated,
                compute_units_ideal=sms_allocated,
                occupancy=1.0,  # Assume full occupancy on allocated SMs
                utilization=utilization,
                bottleneck=bottleneck,
                compute_time=compute_time,
                memory_time=memory_time,
                estimated_latency=kernel_latency,
                compute_energy=compute_energy,
                memory_energy=memory_energy,
                total_energy=compute_energy + memory_energy,
                execution_stage=idx,  # Each kernel is its own stage
                is_parallel=False,
            )

            allocations.append(allocation)
            total_latency += kernel_latency

        return total_latency, allocations

    def compute_energy_with_idle_power(
        self,
        latency: float,
        dynamic_energy: float
    ) -> Tuple[float, float]:
        """
        Compute total energy including idle power consumption.

        Modern GPUs consume significant power even at idle due to:
        - Transistor leakage in nanoscale processes (7nm, 5nm, 4nm)
        - Always-on circuitry (memory controllers, PCIe, etc.)
        - Typical idle consumption: ~50% of TDP

        Power model:
        P_total = P_idle + P_dynamic
        P_idle = TDP × 0.5
        P_dynamic = dynamic_energy / latency

        Args:
            latency: Total execution time (seconds)
            dynamic_energy: Energy from computation and memory transfers (Joules)

        Returns:
            (total_energy, average_power)
        """
        if latency <= 0:
            return dynamic_energy, 0.0

        # Get TDP from thermal operating point if available
        tdp_watts = None
        if self.thermal_profile and self.resource_model.thermal_operating_points:
            thermal_point = self.resource_model.thermal_operating_points.get(self.thermal_profile)
            if thermal_point:
                tdp_watts = thermal_point.tdp_watts

        # If no thermal profile, estimate TDP from dynamic power
        # For datacenter GPUs: dynamic power is typically ~50-80% of TDP
        if tdp_watts is None:
            # Estimate TDP as 2× peak dynamic power (assuming 50% utilization at peak)
            dynamic_power = dynamic_energy / latency if latency > 0 else 0
            tdp_watts = dynamic_power * 2.0

        # Idle power: ~50% TDP constantly consumed
        idle_power = tdp_watts * self.IDLE_POWER_FRACTION
        idle_energy = idle_power * latency

        # Total energy = idle + dynamic
        total_energy = idle_energy + dynamic_energy

        # Average power during execution
        average_power = total_energy / latency

        return total_energy, average_power

    def map_graph(
        self,
        fusion_report: FusionReport,
        execution_stages: List[List[int]],
        batch_size: int = 1,
        precision: Precision = Precision.FP32
    ) -> GraphHardwareAllocation:
        """
        Map entire computation graph to GPU.

        Two execution modes:
        1. **Sequential mode** (small DNNs, batch=1):
           - Each subgraph launches 1 kernel sequentially
           - Kernels use 1-8 SMs (not all 132)
           - Add kernel launch overhead (~10 µs)
           - More accurate for small models like ResNet-18/50

        2. **Parallel mode** (large DNNs or large batch):
           - Subgraphs in same stage run in parallel
           - Allocate SMs based on thread requirements
           - Wave quantization and occupancy modeling

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
        # Detect execution mode
        use_sequential = self.should_use_sequential_execution(fusion_report, batch_size)

        if use_sequential:
            # Sequential SM execution mode (accurate for small DNNs)
            total_latency, subgraph_allocations = self.compute_sequential_latency(
                fusion_report, precision
            )

            # Aggregate metrics
            total_subgraphs = len(subgraph_allocations)
            average_sms_used = sum(a.compute_units_allocated for a in subgraph_allocations) / max(1, total_subgraphs)
            peak_sms_used = max((a.compute_units_allocated for a in subgraph_allocations), default=0)
            average_utilization = average_sms_used / self.resource_model.compute_units
            peak_utilization = peak_sms_used / self.resource_model.compute_units

            # Total energy (dynamic from subgraphs + idle power baseline)
            dynamic_energy = sum(a.total_energy for a in subgraph_allocations)
            total_energy, average_power = self.compute_energy_with_idle_power(
                total_latency, dynamic_energy
            )

            # Naive latency
            total_ops = fusion_report.total_flops
            peak_ops_per_sec = self.resource_model.get_peak_ops(precision)
            naive_latency = total_ops / peak_ops_per_sec if peak_ops_per_sec > 0 else 0

            # Latency breakdown (each subgraph is its own stage in sequential mode)
            latency_breakdown = {i: a.estimated_latency for i, a in enumerate(subgraph_allocations)}

            # Bottleneck counts
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
                total_execution_stages=total_subgraphs,  # Each subgraph is a stage
                peak_compute_units_used=peak_sms_used,
                average_compute_units_used=average_sms_used,
                peak_utilization=peak_utilization,
                average_utilization=average_utilization,
                total_latency=total_latency,
                latency_breakdown=latency_breakdown,
                total_energy=total_energy,
                naive_latency=naive_latency,
                latency_correction_factor=total_latency / naive_latency if naive_latency > 0 else 1.0,
                compute_bound_count=compute_bound_count,
                memory_bound_count=memory_bound_count,
                bandwidth_bound_count=bandwidth_bound_count,
                balanced_count=balanced_count,
            )

        # Otherwise, use parallel execution mode (original logic)
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

        # Total energy (dynamic from subgraphs + idle power baseline)
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


def create_h100_pcie_80gb_mapper(thermal_profile: str = None) -> GPUMapper:
    """
    Create GPU mapper for NVIDIA H100 PCIe 80GB.

    FORM FACTOR: PCIe (add-in card)
    MEMORY: 80 GB HBM2e

    Args:
        thermal_profile: Thermal profile name (if applicable)

    Returns:
        GPUMapper configured for H100 PCIe 80GB
    """
    from ..models.datacenter.h100_pcie_80gb import h100_pcie_80gb_resource_model
    from ..architectural_energy import DataParallelEnergyModel

    # Create resource model
    resource_model = h100_pcie_80gb_resource_model()

    # Configure architectural energy model for GPU (DATA_PARALLEL)
    resource_model.architecture_energy_model = DataParallelEnergyModel(
        instruction_fetch_energy=2.0e-12,
        operand_fetch_overhead=10.0e-12,
        coherence_energy_per_request=5.0e-12,  # GPU-specific coherence machinery
        thread_scheduling_overhead=1.0e-12,
        warp_divergence_penalty=3.0e-12,
        memory_coalescing_overhead=2.0e-12,
    )

    return GPUMapper(resource_model, thermal_profile=thermal_profile)


def create_h100_sxm5_80gb_mapper(thermal_profile: str = None) -> GPUMapper:
    """
    Create GPU mapper for NVIDIA H100 SXM5 80GB.

    FORM FACTOR: SXM (Server Module with high-bandwidth interconnect)
    MEMORY: 80 GB HBM2e

    Args:
        thermal_profile: Thermal profile name (if applicable)

    Returns:
        GPUMapper configured for H100 SXM5 80GB
    """
    from ..models.datacenter.h100_sxm5_80gb import h100_sxm5_80gb_resource_model
    from ..architectural_energy import DataParallelEnergyModel

    # Create resource model
    resource_model = h100_sxm5_80gb_resource_model()

    # Configure architectural energy model for GPU (DATA_PARALLEL)
    resource_model.architecture_energy_model = DataParallelEnergyModel(
        instruction_fetch_energy=2.0e-12,
        operand_fetch_overhead=10.0e-12,
        coherence_energy_per_request=5.0e-12,  # GPU-specific coherence machinery
        thread_scheduling_overhead=1.0e-12,
        warp_divergence_penalty=3.0e-12,
        memory_coalescing_overhead=2.0e-12,
    )

    return GPUMapper(resource_model, thermal_profile=thermal_profile)


def create_b100_sxm6_192gb_mapper(thermal_profile: str = None) -> GPUMapper:
    """
    Create GPU mapper for NVIDIA B100 SXM6 192GB (Blackwell - 2024).

    FORM FACTOR: SXM (Server Module with high-bandwidth interconnect)
    MEMORY: 192 GB HBM3e

    ARCHITECTURE:
    - Dual GB100 dies in single package (10 TB/s inter-die NV-HBI)
    - 5th generation Tensor Cores with FP4/FP6 support
    - 132 SMs (estimated) with 528 Tensor Cores
    - 208 billion transistors (TSMC 4NP process)
    - SXM form factor enables higher power and NVLink connectivity

    KEY ADVANCES:
    - FP4 precision: 7 PFLOPS (dense), 14 PFLOPS (sparse) - 9.3× vs H100 FP8
    - FP6 precision: 3.5 PFLOPS (dense), 7 PFLOPS (sparse)
    - 8 TB/s HBM3e bandwidth (4× H100's 2 TB/s)
    - 192 GB memory (2.4× H100 PCIe 80 GB)
    - Blackwell MoE architecture optimizations

    COMPUTE PERFORMANCE:
    - FP4: 7/14 PFLOPS (dense/sparse)
    - FP6: 3.5/7 PFLOPS (dense/sparse)
    - FP8: 3.5/7 PFLOPS (dense/sparse) - 2.3× vs H100
    - FP16/BF16: 1.8/3.5 PFLOPS (dense/sparse)
    - INT8: 3.5/7 POPS (dense/sparse)

    POWER: 700W TDP (SXM allows higher power than PCIe variants)

    USE CASE:
    - Datacenter AI training (GPT-4 scale models)
    - Large-scale inference with extreme efficiency (FP4/FP6)
    - Embodied AI model development (predecessor to Jetson Thor)
    - MoE models (Mixtral, GPT-4 MoE) with Blackwell optimizations

    CALIBRATION STATUS:
    ✅ VALIDATED - Official NVIDIA Blackwell specs (Nov 2024)

    Args:
        thermal_profile: Thermal profile name (if applicable)

    Returns:
        GPUMapper configured for B100 SXM6 192GB
    """
    from ..models.datacenter.b100_sxm6_192gb import b100_sxm6_192gb_resource_model
    from ..architectural_energy import DataParallelEnergyModel

    # Create resource model
    resource_model = b100_sxm6_192gb_resource_model()

    # Configure architectural energy model for GPU (DATA_PARALLEL)
    # Blackwell improvements: 20% better energy efficiency from 4nm process
    resource_model.architecture_energy_model = DataParallelEnergyModel(
        instruction_fetch_energy=1.6e-12,  # 20% improvement from H100
        operand_fetch_overhead=8.0e-12,  # 20% improvement from H100
        coherence_energy_per_request=4.0e-12,  # GPU-specific coherence machinery
        thread_scheduling_overhead=0.8e-12,  # 20% improvement from H100
        warp_divergence_penalty=2.4e-12,  # 20% improvement from H100
        memory_coalescing_overhead=1.6e-12,  # 20% improvement from H100
    )

    return GPUMapper(resource_model, thermal_profile=thermal_profile)


def create_a100_sxm4_80gb_mapper(thermal_profile: str = None) -> GPUMapper:
    """
    Create GPU mapper for NVIDIA A100 SXM4 80GB (Ampere - 2020).

    FORM FACTOR: SXM4 (Server Module)
    MEMORY: 80 GB HBM2e

    ARCHITECTURE:
    - 3rd generation Tensor Cores (TF32, BF16, FP64)
    - 108 SMs × 128 CUDA cores = 13,824 CUDA cores
    - SM sustained clock: 1300 MHz (boost: 1410 MHz)
    - 2 TB/s HBM2e bandwidth (same as H100)

    KEY ADVANCES:
    - First GPU with TF32 for AI training (1× precision of FP32, 8× speed)
    - First GPU with BF16 (wider dynamic range than FP16)
    - Multi-Instance GPU (MIG) for resource partitioning

    COMPUTE PERFORMANCE:
    - FP64: 9.7 TFLOPS (HPC applications)
    - TF32: 156 TFLOPS (AI training)
    - FP16/BF16: 312 TFLOPS with Tensor Cores
    - INT8: 624 TOPS

    POWER: 400W TDP (datacenter)

    USE CASE: AI training and HPC workloads (2020-2023 flagship)

    Args:
        thermal_profile: Thermal profile name (if applicable)

    Returns:
        GPUMapper configured for A100 SXM4 80GB
    """
    from ..models.datacenter.a100_sxm4_80gb import a100_sxm4_80gb_resource_model
    return GPUMapper(a100_sxm4_80gb_resource_model(), thermal_profile=thermal_profile)


def create_v100_sxm3_32gb_mapper(thermal_profile: str = None) -> GPUMapper:
    """
    Create GPU mapper for NVIDIA V100 SXM3 32GB (Volta - 2017).

    FORM FACTOR: SXM3 (Server Module)
    MEMORY: 32 GB HBM2

    ARCHITECTURE:
    - 1st generation Tensor Cores (FP16 matrix multiply)
    - 80 SMs x 64 CUDA cores = 5,120 CUDA cores
    - SM sustained clock: 1400 MHz (boost: 1530 MHz)
    - 900 GB/s HBM2 bandwidth

    KEY INNOVATION:
    - First GPU with Tensor Cores (8 per SM)
    - Revolutionized deep learning training speed
    - Independent thread scheduling (per-thread control flow)

    COMPUTE PERFORMANCE:
    - FP64: 7.8 TFLOPS (HPC workloads)
    - FP32: 15.7 TFLOPS (validated against published specs)
    - FP16: 31.4 TFLOPS
    - FP16 Tensor Core: 125 TFLOPS (mixed precision training)

    POWER: 300W TDP (SXM2), 250W TDP (PCIe)

    USE CASE: AI training breakthrough (2017-2020 flagship)

    Args:
        thermal_profile: Thermal profile name (if applicable)

    Returns:
        GPUMapper configured for V100 SXM2 32GB
    """
    from ..models.datacenter.v100_sxm3_32gb import v100_sxm3_32gb_resource_model
    return GPUMapper(v100_sxm3_32gb_resource_model(), thermal_profile=thermal_profile)


def create_t4_pcie_16gb_mapper(thermal_profile: str = None) -> GPUMapper:
    """
    Create GPU mapper for NVIDIA T4 PCIe 16GB (Turing - 2018).

    FORM FACTOR: PCIe (add-in card, inference-optimized)
    MEMORY: 16 GB GDDR6

    ARCHITECTURE:
    - 2nd generation Tensor Cores (INT8, INT4 support)
    - 40 SMs × 64 CUDA cores = 2,560 CUDA cores
    - SM sustained clock: 1470 MHz (boost: 1590 MHz)
    - 320 GB/s GDDR6 bandwidth

    KEY FEATURES:
    - Inference-optimized (small die, low power)
    - First GPU with INT8 Tensor Cores (2× INT8 vs FP16)
    - Turing architecture (RT cores for ray tracing, not used in AI)

    COMPUTE PERFORMANCE:
    - FP32: 8.1 TFLOPS
    - FP16: 65 TFLOPS with Tensor Cores
    - INT8: 130 TOPS (inference optimized)
    - INT4: 260 TOPS

    POWER: 70W TDP (extremely efficient for datacenter inference)

    USE CASE:
    - Edge inference and cloud inference (2018-present)
    - Ideal for video streaming inference
    - Cost-effective alternative to V100/A100

    Args:
        thermal_profile: Thermal profile name (if applicable)

    Returns:
        GPUMapper configured for T4 PCIe 16GB
    """
    from ..models.datacenter.t4_pcie_16gb import t4_pcie_16gb_resource_model
    return GPUMapper(t4_pcie_16gb_resource_model(), thermal_profile=thermal_profile)


def create_jetson_orin_agx_64gb_mapper(thermal_profile: str = None) -> GPUMapper:
    """
    Create GPU mapper for NVIDIA Jetson Orin AGX 64GB (edge AI platform).

    MEMORY: 64 GB LPDDR5

    Args:
        thermal_profile: Thermal profile name (e.g., "15W", "30W", "60W")
                        If None, uses default ("15W")

    Returns:
        GPUMapper configured for Jetson Orin AGX 64GB
    """
    from ..models.edge.jetson_orin_agx_64gb import jetson_orin_agx_64gb_resource_model
    from ..architectural_energy import DataParallelEnergyModel

    # Create resource model
    resource_model = jetson_orin_agx_64gb_resource_model()

    # Configure architectural energy model for GPU (DATA_PARALLEL)
    # Jetson Orin AGX: Edge GPU with lower power than datacenter
    resource_model.architecture_energy_model = DataParallelEnergyModel(
        # Compute unit breakdown
        cuda_core_mac_energy=0.8e-12,           # ~0.8 pJ per MAC (FP32)
        tensor_core_mac_energy=0.3e-12,         # ~0.3 pJ per MAC (INT8/FP16)
        tensor_core_utilization=0.80,           # 80% ops use Tensor Cores
        register_file_energy_per_access=0.6e-12,  # ~0.6 pJ (similar to ALU energy)

        # Memory hierarchy (NVIDIA Ampere nomenclature)
        # Register File → Shared Memory/L1 (unified) → L2 → DRAM
        shared_memory_l1_unified_energy_per_byte=0.25e-12,  # Unified Shared Mem/L1
        l2_cache_energy_per_byte=0.8e-12,
        dram_energy_per_byte=10.0e-12,

        # Memory access patterns
        shared_mem_l1_hit_rate=0.95,           # Unified Shared Mem/L1 hit rate
        l2_hit_rate=0.90,

        # Instruction pipeline stages
        instruction_fetch_energy=1.5e-12,      # Edge: lower than datacenter
        instruction_decode_energy=0.4e-12,
        instruction_execute_energy=0.25e-12,

        # SIMT control overheads (DATA_PARALLEL architecture)
        operand_fetch_overhead=8.0e-12,        # Edge: lower than datacenter
        coherence_energy_per_request=4.0e-12,  # GPU coherence machinery
        thread_scheduling_overhead=0.8e-12,
        warp_divergence_penalty=2.5e-12,
        memory_coalescing_overhead=1.8e-12,
        barrier_sync_energy=2.0e-12,
    )

    return GPUMapper(resource_model, thermal_profile=thermal_profile)


def create_jetson_orin_nano_8gb_mapper(thermal_profile: str = None) -> GPUMapper:
    """
    Create GPU mapper for NVIDIA Jetson Orin Nano 8GB (compact edge AI platform).

    MEMORY: 8 GB LPDDR5

    Args:
        thermal_profile: Thermal profile name (e.g., "7W", "15W")
                        If None, uses default ("7W")

    Returns:
        GPUMapper configured for Jetson Orin Nano 8GB
    """
    from ..models.edge.jetson_orin_nano_8gb import jetson_orin_nano_8gb_resource_model
    return GPUMapper(jetson_orin_nano_8gb_resource_model(), thermal_profile=thermal_profile)


def create_jetson_thor_128gb_mapper(thermal_profile: str = None) -> GPUMapper:
    """
    Create GPU mapper for NVIDIA Jetson Thor 128GB (next-gen edge AI).

    MEMORY: 128 GB LPDDR5X

    Args:
        thermal_profile: Thermal profile name (e.g., "30W", "60W", "100W")
                        If None, uses default ("30W")

    Returns:
        GPUMapper configured for Jetson Thor 128GB
    """
    from ..models.automotive.jetson_thor_128gb import jetson_thor_128gb_resource_model
    return GPUMapper(jetson_thor_128gb_resource_model(), thermal_profile=thermal_profile)


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
    from ..models.mobile.arm_mali_g78_mp20 import arm_mali_g78_mp20_resource_model
    return GPUMapper(arm_mali_g78_mp20_resource_model(), thermal_profile=thermal_profile)


# ============================================================================
# DEPRECATED: Old mapper names (backward compatibility)
# These will be removed in version X.Y+1.0 (6 months)
# ============================================================================

def create_h100_mapper(thermal_profile: str = None) -> GPUMapper:
    """
    DEPRECATED: Use create_h100_pcie_80gb_mapper() instead.

    This function is deprecated and will be removed in a future version.
    Please update your code to use the new naming convention that includes
    form factor and memory size for clarity:

        create_h100_pcie_80gb_mapper()

    Args:
        thermal_profile: Thermal profile name

    Returns:
        GPUMapper configured for H100 PCIe 80GB
    """
    import warnings
    warnings.warn(
        "create_h100_mapper() is deprecated and will be removed in a future version. "
        "Use create_h100_pcie_80gb_mapper() instead for explicit form factor and memory size.",
        DeprecationWarning,
        stacklevel=2
    )
    return create_h100_pcie_80gb_mapper(thermal_profile)


def create_a100_mapper(thermal_profile: str = None) -> GPUMapper:
    """
    DEPRECATED: Use create_a100_sxm4_80gb_mapper() instead.

    This function is deprecated and will be removed in a future version.
    Please update your code to use the new naming convention that includes
    form factor and memory size for clarity:

        create_a100_sxm4_80gb_mapper()

    Args:
        thermal_profile: Thermal profile name

    Returns:
        GPUMapper configured for A100 SXM4 80GB
    """
    import warnings
    warnings.warn(
        "create_a100_mapper() is deprecated and will be removed in a future version. "
        "Use create_a100_sxm4_80gb_mapper() instead for explicit form factor and memory size.",
        DeprecationWarning,
        stacklevel=2
    )
    return create_a100_sxm4_80gb_mapper(thermal_profile)


def create_v100_mapper(thermal_profile: str = None) -> GPUMapper:
    """
    DEPRECATED: Use create_v100_sxm2_32gb_mapper() instead.

    This function is deprecated and will be removed in a future version.
    Please update your code to use the new naming convention that includes
    form factor and memory size for clarity:

        create_v100_sxm2_32gb_mapper()

    Args:
        thermal_profile: Thermal profile name

    Returns:
        GPUMapper configured for V100 SXM2 32GB
    """
    import warnings
    warnings.warn(
        "create_v100_mapper() is deprecated and will be removed in a future version. "
        "Use create_v100_sxm2_32gb_mapper() instead for explicit form factor and memory size.",
        DeprecationWarning,
        stacklevel=2
    )
    return create_v100_sxm2_32gb_mapper(thermal_profile)


def create_t4_mapper(thermal_profile: str = None) -> GPUMapper:
    """
    DEPRECATED: Use create_t4_pcie_16gb_mapper() instead.

    This function is deprecated and will be removed in a future version.
    Please update your code to use the new naming convention that includes
    form factor and memory size for clarity:

        create_t4_pcie_16gb_mapper()

    Args:
        thermal_profile: Thermal profile name

    Returns:
        GPUMapper configured for T4 PCIe 16GB
    """
    import warnings
    warnings.warn(
        "create_t4_mapper() is deprecated and will be removed in a future version. "
        "Use create_t4_pcie_16gb_mapper() instead for explicit form factor and memory size.",
        DeprecationWarning,
        stacklevel=2
    )
    return create_t4_pcie_16gb_mapper(thermal_profile)


def create_jetson_orin_agx_mapper(thermal_profile: str = None) -> GPUMapper:
    """
    DEPRECATED: Use create_jetson_orin_agx_64gb_mapper() instead.

    This function is deprecated and will be removed in a future version.
    Please update your code to use the new naming convention that includes
    memory size for clarity:

        create_jetson_orin_agx_64gb_mapper()

    Args:
        thermal_profile: Thermal profile name

    Returns:
        GPUMapper configured for Jetson Orin AGX 64GB
    """
    import warnings
    warnings.warn(
        "create_jetson_orin_agx_mapper() is deprecated and will be removed in a future version. "
        "Use create_jetson_orin_agx_64gb_mapper() instead for explicit memory size.",
        DeprecationWarning,
        stacklevel=2
    )
    return create_jetson_orin_agx_64gb_mapper(thermal_profile)


def create_jetson_orin_nano_mapper(thermal_profile: str = None) -> GPUMapper:
    """
    DEPRECATED: Use create_jetson_orin_nano_8gb_mapper() instead.

    This function is deprecated and will be removed in a future version.
    Please update your code to use the new naming convention that includes
    memory size for clarity:

        create_jetson_orin_nano_8gb_mapper()

    Args:
        thermal_profile: Thermal profile name

    Returns:
        GPUMapper configured for Jetson Orin Nano 8GB
    """
    import warnings
    warnings.warn(
        "create_jetson_orin_nano_mapper() is deprecated and will be removed in a future version. "
        "Use create_jetson_orin_nano_8gb_mapper() instead for explicit memory size.",
        DeprecationWarning,
        stacklevel=2
    )
    return create_jetson_orin_nano_8gb_mapper(thermal_profile)


def create_jetson_thor_mapper(thermal_profile: str = None) -> GPUMapper:
    """
    DEPRECATED: Use create_jetson_thor_128gb_mapper() instead.

    This function is deprecated and will be removed in a future version.
    Please update your code to use the new naming convention that includes
    memory size for clarity:

        create_jetson_thor_128gb_mapper()

    Args:
        thermal_profile: Thermal profile name

    Returns:
        GPUMapper configured for Jetson Thor 128GB
    """
    import warnings
    warnings.warn(
        "create_jetson_thor_mapper() is deprecated and will be removed in a future version. "
        "Use create_jetson_thor_128gb_mapper() instead for explicit memory size.",
        DeprecationWarning,
        stacklevel=2
    )
    return create_jetson_thor_128gb_mapper(thermal_profile)
