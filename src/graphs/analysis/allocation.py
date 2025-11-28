"""
Hardware Allocation and Execution Planning Data Structures

This module defines data structures for tracking how computational subgraphs
are mapped to hardware resources and executed sequentially.

Key Concepts:
- SubgraphAllocation: Hardware resource allocation for a single subgraph
- ExecutionPlan: Complete execution sequence across all subgraphs
- Provides detailed tracking of resource utilization, power, and latency
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from graphs.ir.structures import SubgraphDescriptor, BottleneckType


@dataclass
class HardwareAllocation:
    """
    Hardware resources allocated to a subgraph

    Different hardware types have different allocation units:
    - GPU: SM count, threads per SM, warps, waves
    - TPU: Systolic array tiles, MXU size
    - KPU: Compute tiles, L3 cache tiles
    - CPU: Core count, SIMD lanes
    - DSP: Vector units, HVX threads
    """
    hardware_type: str  # 'GPU', 'TPU', 'KPU', 'CPU', 'DSP'

    # Generic allocation metrics
    allocated_units: int  # Number of compute units allocated (SMs, cores, tiles, etc.)
    total_available_units: int  # Total units in hardware
    utilization: float  # Fraction of allocated unit capacity used (0.0-1.0)

    # Hardware-specific details (stored as dict for flexibility)
    allocation_details: Dict[str, any] = field(default_factory=dict)

    # Examples of allocation_details:
    # GPU: {'SM_count': 10, 'threads_per_SM': 2048, 'warps': 32, 'waves': 5}
    # TPU: {'array_tiles': 4, 'tile_size': '128x128', 'mxu_utilization': 0.85}
    # KPU: {'compute_tiles': 8, 'l3_tiles': 2, 'tile_schedule': 'spatial'}
    # CPU: {'cores': 4, 'simd_lanes': 8, 'vector_width': 256}
    # DSP: {'vector_units': 2, 'hvx_threads': 4, 'lane_utilization': 0.6}

    def utilization_percentage(self) -> float:
        """Return utilization as percentage (0-100)"""
        return self.utilization * 100.0

    def allocated_percentage(self) -> float:
        """Return percentage of total hardware allocated (0-100)"""
        if self.total_available_units == 0:
            return 0.0
        return (self.allocated_units / self.total_available_units) * 100.0


@dataclass
class SubgraphAllocation:
    """
    Complete allocation information for a single subgraph

    Tracks:
    - Which subgraph this is
    - What hardware resources were allocated
    - Power and latency estimates on those resources
    - Bottleneck analysis
    """
    # Subgraph identification
    subgraph_id: int
    subgraph_descriptor: SubgraphDescriptor  # Full subgraph info

    # Operation summary
    operation_types: List[str]  # e.g., ['conv2d', 'batchnorm', 'relu']
    fusion_pattern: str  # e.g., 'conv_bn_relu'

    # Computational characteristics
    flops: int
    memory_bytes: int  # Total memory traffic (input + output + weights)
    arithmetic_intensity: float  # FLOPs / memory_bytes

    # Parallelism available
    total_threads: int  # Total parallelism in subgraph

    # Hardware allocation
    hardware_allocation: HardwareAllocation

    # Performance estimates (on allocated resources)
    compute_time_ms: float  # Time if compute-bound
    memory_time_ms: float  # Time if memory-bound
    actual_latency_ms: float  # max(compute_time, memory_time)

    # Power estimates
    idle_power_watts: float  # Idle power during this subgraph
    dynamic_power_watts: float  # Dynamic power based on utilization
    total_power_watts: float  # idle + dynamic

    # Bottleneck analysis
    bottleneck_type: BottleneckType
    bottleneck_explanation: str = ""

    # Data dependencies
    depends_on: List[int] = field(default_factory=list)  # Other subgraph IDs
    dependency_type: str = "sequential"  # 'sequential', 'independent', 'partial'

    def is_compute_bound(self) -> bool:
        """Check if this subgraph is compute-bound"""
        return self.bottleneck_type == BottleneckType.COMPUTE_BOUND

    def is_memory_bound(self) -> bool:
        """Check if this subgraph is memory-bound"""
        return self.bottleneck_type in [BottleneckType.MEMORY_BOUND, BottleneckType.BANDWIDTH_BOUND]

    def energy_joules(self) -> float:
        """Calculate energy consumed (Power × Time)"""
        return self.total_power_watts * (self.actual_latency_ms / 1000.0)


@dataclass
class ExecutionPlan:
    """
    Complete execution plan for a model on specific hardware

    Tracks sequential execution of all subgraphs with:
    - Resource allocation per subgraph
    - Power and latency per subgraph
    - Total power and latency
    - Utilization statistics
    """
    # Model information
    model_name: str
    total_operations: int
    total_flops: int
    total_memory_traffic: int

    # Hardware information
    hardware_name: str
    hardware_type: str  # 'GPU', 'TPU', 'KPU', 'CPU', 'DSP'
    peak_flops: float  # Peak FLOPS of hardware
    memory_bandwidth: float  # GB/s
    tdp_watts: float

    # Execution sequence
    subgraph_allocations: List[SubgraphAllocation] = field(default_factory=list)
    num_subgraphs: int = 0

    # Aggregated metrics (computed from subgraphs)
    total_latency_ms: float = 0.0  # Sum of sequential subgraph latencies
    average_power_watts: float = 0.0  # Time-weighted average power
    peak_power_watts: float = 0.0  # Maximum power across subgraphs
    total_energy_joules: float = 0.0  # Sum of energy across subgraphs

    # Utilization statistics
    average_utilization: float = 0.0  # Average utilization across subgraphs
    peak_utilization: float = 0.0  # Maximum utilization achieved
    min_utilization: float = 1.0  # Minimum utilization (identifies bottlenecks)

    # Bottleneck analysis
    compute_bound_subgraphs: int = 0
    memory_bound_subgraphs: int = 0

    # Efficiency metrics
    hardware_efficiency: float = 0.0  # Fraction of peak FLOPS achieved
    memory_efficiency: float = 0.0  # Fraction of peak bandwidth used

    # Optimization suggestions
    bottleneck_warnings: List[str] = field(default_factory=list)
    optimization_suggestions: List[str] = field(default_factory=list)

    def compute_aggregates(self):
        """Compute aggregate statistics from subgraph allocations"""
        if not self.subgraph_allocations:
            return

        self.num_subgraphs = len(self.subgraph_allocations)

        # Total latency (sequential execution)
        self.total_latency_ms = sum(alloc.actual_latency_ms for alloc in self.subgraph_allocations)

        # Power: time-weighted average
        if self.total_latency_ms > 0:
            weighted_power_sum = sum(
                alloc.total_power_watts * alloc.actual_latency_ms
                for alloc in self.subgraph_allocations
            )
            self.average_power_watts = weighted_power_sum / self.total_latency_ms

        # Peak power
        self.peak_power_watts = max(
            (alloc.total_power_watts for alloc in self.subgraph_allocations),
            default=0.0
        )

        # Total energy
        self.total_energy_joules = sum(alloc.energy_joules() for alloc in self.subgraph_allocations)

        # Utilization statistics
        utilizations = [alloc.hardware_allocation.utilization for alloc in self.subgraph_allocations]
        if utilizations:
            self.average_utilization = sum(utilizations) / len(utilizations)
            self.peak_utilization = max(utilizations)
            self.min_utilization = min(utilizations)

        # Bottleneck counts
        self.compute_bound_subgraphs = sum(
            1 for alloc in self.subgraph_allocations if alloc.is_compute_bound()
        )
        self.memory_bound_subgraphs = sum(
            1 for alloc in self.subgraph_allocations if alloc.is_memory_bound()
        )

        # Efficiency metrics
        if self.total_latency_ms > 0 and self.peak_flops > 0:
            achieved_flops = self.total_flops / (self.total_latency_ms / 1000.0)  # FLOPs/sec
            self.hardware_efficiency = achieved_flops / self.peak_flops

        if self.total_latency_ms > 0 and self.memory_bandwidth > 0:
            achieved_bandwidth = self.total_memory_traffic / (self.total_latency_ms / 1000.0)  # bytes/sec
            achieved_bandwidth_gb = achieved_bandwidth / 1e9  # GB/s
            self.memory_efficiency = achieved_bandwidth_gb / self.memory_bandwidth

    def generate_warnings_and_suggestions(self):
        """Generate bottleneck warnings and optimization suggestions"""
        self.bottleneck_warnings.clear()
        self.optimization_suggestions.clear()

        # Low utilization warning
        if self.average_utilization < 0.2:
            self.bottleneck_warnings.append(
                f"[!] Low hardware utilization (avg {self.average_utilization*100:.1f}%) - "
                f"insufficient parallelism to utilize {self.hardware_name}"
            )
            self.optimization_suggestions.append(
                "• Increase batch size to improve parallelism"
            )
            self.optimization_suggestions.append(
                "• Consider using smaller hardware target for this workload"
            )

        # Memory-bound workload
        if self.memory_bound_subgraphs > self.compute_bound_subgraphs * 2:
            self.bottleneck_warnings.append(
                f"[!] Memory-bound workload ({self.memory_bound_subgraphs}/{self.num_subgraphs} subgraphs)"
            )
            self.optimization_suggestions.append(
                "• Optimize data layout for better cache locality"
            )
            self.optimization_suggestions.append(
                "• Consider using higher precision for compute-bound ops (trade memory for compute)"
            )

        # Low efficiency warning
        if self.hardware_efficiency < 0.1:
            self.bottleneck_warnings.append(
                f"[!] Low hardware efficiency ({self.hardware_efficiency*100:.1f}% of peak FLOPS)"
            )
            self.optimization_suggestions.append(
                "• Profile individual kernels to identify bottlenecks"
            )
            self.optimization_suggestions.append(
                "• Consider operator fusion to reduce memory traffic"
            )

        # High variance in utilization
        if self.num_subgraphs > 1:
            util_variance = self.peak_utilization - self.min_utilization
            if util_variance > 0.5:
                self.bottleneck_warnings.append(
                    f"[!] High variance in utilization ({self.min_utilization*100:.1f}% - {self.peak_utilization*100:.1f}%)"
                )
                self.optimization_suggestions.append(
                    "• Balance workload across operations (some ops under-utilizing hardware)"
                )

    def throughput_fps(self) -> float:
        """Calculate throughput in frames per second"""
        if self.total_latency_ms == 0:
            return 0.0
        return 1000.0 / self.total_latency_ms

    def energy_per_inference_mj(self) -> float:
        """Energy per inference in millijoules"""
        return self.total_energy_joules * 1000.0
