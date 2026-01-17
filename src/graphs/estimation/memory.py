"""
Memory Estimation and Analysis

This module provides tools for analyzing memory footprint of neural network models
by simulating execution and tracking tensor allocations/deallocations.

Key Insight: Peak memory ≠ sum of all tensors!
It's the maximum of concurrent allocations over time.

Classes:
    MemoryTimelineEntry: Memory state at a single execution step
    MemoryDescriptor: Per-subgraph memory analysis
    MemoryReport: Complete memory analysis for entire model
    MemoryEstimator: Simulates execution to estimate memory usage
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from graphs.core.structures import SubgraphDescriptor, PartitionReport, OperationType
from graphs.hardware.resource_model import HardwareResourceModel


@dataclass
class MemoryTimelineEntry:
    """
    Memory state at a single point in execution.

    Captures what tensors are alive, what was allocated/freed,
    and total memory usage at this execution step.
    """

    step: int                        # Execution step number (0-indexed)
    subgraph_id: str                 # Subgraph executing at this step
    subgraph_name: str               # Human-readable name

    # Memory state
    total_memory_bytes: int          # Total allocated at this point
    activation_memory_bytes: int     # Live activations (excluding weights)
    workspace_memory_bytes: int      # Temporary buffers (im2col, etc.)

    # Tensors alive at this point
    live_tensors: List[str] = field(default_factory=list)  # IDs of live tensors
    num_live_tensors: int = 0

    # Events at this step
    allocated_tensors: List[str] = field(default_factory=list)  # Just allocated
    freed_tensors: List[str] = field(default_factory=list)      # Just freed

    def total_memory_mb(self) -> float:
        """Return total memory in megabytes"""
        return self.total_memory_bytes / (1024 ** 2)

    def total_memory_gb(self) -> float:
        """Return total memory in gigabytes"""
        return self.total_memory_bytes / (1024 ** 3)

    def format_summary(self) -> str:
        """One-line summary of this timeline entry"""
        alloc_str = f"+{len(self.allocated_tensors)}" if self.allocated_tensors else ""
        freed_str = f"-{len(self.freed_tensors)}" if self.freed_tensors else ""
        event_str = f" [{alloc_str}{freed_str}]" if (alloc_str or freed_str) else ""

        return (f"Step {self.step:3d}: {self.subgraph_name:30s} "
                f"{self.total_memory_mb():7.1f} MB  "
                f"({self.num_live_tensors} tensors alive){event_str}")


@dataclass
class MemoryDescriptor:
    """
    Memory footprint analysis for a single subgraph.

    Tracks all memory associated with executing one computational subgraph,
    including inputs, outputs, weights, and temporary workspace.
    """

    subgraph_id: str
    subgraph_name: str
    operation_type: OperationType

    # Memory breakdown (bytes)
    input_memory_bytes: int          # Input activations
    output_memory_bytes: int         # Output activations
    weight_memory_bytes: int         # Parameters (Conv/Linear)
    workspace_memory_bytes: int      # Temporary buffers

    # Lifetime tracking
    inputs_live_after: bool = True   # Do inputs remain alive after execution?
    can_reuse_input_buffer: bool = False  # Can output overwrite input?

    # Access patterns
    total_read_bytes: int = 0        # Total memory reads
    total_write_bytes: int = 0       # Total memory writes
    reuse_factor: float = 1.0        # Average times each byte is accessed

    # Optimization potential
    can_checkpoint: bool = False     # Can trade compute for memory?
    checkpoint_savings_bytes: int = 0  # Memory saved if checkpointed
    can_quantize: bool = False       # Can reduce precision?
    quantization_savings_bytes: int = 0  # FP32→INT8 savings (4× for weights)

    explanation: str = ""            # Human-readable description

    @property
    def total_memory_bytes(self) -> int:
        """Total memory used by this subgraph (excluding inputs)"""
        return (self.output_memory_bytes +
                self.weight_memory_bytes +
                self.workspace_memory_bytes)

    @property
    def total_memory_mb(self) -> float:
        """Total memory in megabytes"""
        return self.total_memory_bytes / (1024 ** 2)

    @property
    def peak_memory_bytes(self) -> int:
        """Peak memory during execution (inputs + outputs + weights + workspace)"""
        return (self.input_memory_bytes +
                self.output_memory_bytes +
                self.weight_memory_bytes +
                self.workspace_memory_bytes)

    @property
    def peak_memory_mb(self) -> float:
        """Peak memory in megabytes"""
        return self.peak_memory_bytes / (1024 ** 2)

    def format_summary(self) -> str:
        """One-line summary of memory for this subgraph"""
        return (f"{self.subgraph_name:30s}  "
                f"Peak: {self.peak_memory_mb:6.1f} MB  "
                f"(In: {self.input_memory_bytes/1024**2:5.1f} MB, "
                f"Out: {self.output_memory_bytes/1024**2:5.1f} MB, "
                f"W: {self.weight_memory_bytes/1024**2:5.1f} MB)")


@dataclass
class MemoryReport:
    """
    Complete memory analysis for entire model.

    Provides peak memory, breakdown by component, optimization suggestions,
    and detailed timeline of memory usage throughout execution.
    """

    # Peak memory (maximum at any point in time)
    peak_memory_bytes: int
    peak_memory_mb: float
    peak_memory_gb: float

    # Breakdown by component
    activation_memory_bytes: int     # Peak activations (excluding weights)
    weight_memory_bytes: int         # Total parameters (persistent)
    workspace_memory_bytes: int      # Maximum workspace needed

    # Efficiency metrics
    average_memory_bytes: float      # Average over execution
    memory_utilization: float        # avg / peak (0.0-1.0)
    fragmentation_waste_bytes: int = 0  # Wasted due to fragmentation

    # Hardware fit analysis
    fits_in_l2_cache: bool = False   # GPU L2 / CPU L3
    fits_in_shared_memory: bool = False  # GPU shared mem per SM
    fits_on_device: bool = True      # Total device memory
    l2_cache_size_bytes: int = 0     # For reference
    device_memory_bytes: int = 0     # For reference

    # Timeline (critical for understanding peaks)
    memory_timeline: List[MemoryTimelineEntry] = field(default_factory=list)
    peak_at_step: int = 0            # Which step has peak memory
    peak_at_subgraph: str = ""       # Which subgraph causes peak
    peak_at_subgraph_name: str = ""  # Human-readable name

    # Optimization opportunities
    total_checkpoint_savings_bytes: int = 0
    total_quantization_savings_bytes: int = 0
    optimization_suggestions: List[str] = field(default_factory=list)

    # Per-subgraph details
    subgraph_descriptors: List[MemoryDescriptor] = field(default_factory=list)

    def format_report(self, show_timeline: bool = False, timeline_steps: int = 10) -> str:
        """
        Generate human-readable memory report.

        Args:
            show_timeline: Whether to include memory timeline
            timeline_steps: How many timeline entries to show

        Returns:
            Formatted string report
        """
        lines = []

        # Header
        lines.append("=" * 80)
        lines.append("MEMORY ANALYSIS REPORT")
        lines.append("=" * 80)
        lines.append("")

        # Peak memory summary
        lines.append(f"Peak Memory: {self.peak_memory_mb:.1f} MB ({self.peak_memory_gb:.3f} GB)")
        lines.append(f"  Activations:  {self.activation_memory_bytes/1024**2:6.1f} MB "
                    f"({self.activation_memory_bytes/self.peak_memory_bytes*100:.1f}%)")
        lines.append(f"  Weights:      {self.weight_memory_bytes/1024**2:6.1f} MB "
                    f"({self.weight_memory_bytes/self.peak_memory_bytes*100:.1f}%)")
        lines.append(f"  Workspace:    {self.workspace_memory_bytes/1024**2:6.1f} MB "
                    f"({self.workspace_memory_bytes/self.peak_memory_bytes*100:.1f}%)")
        lines.append("")

        # Efficiency
        lines.append(f"Average Memory: {self.average_memory_bytes/1024**2:.1f} MB "
                    f"({self.memory_utilization*100:.0f}% utilization)")
        lines.append("")

        # Hardware fit
        lines.append("Hardware Fit Analysis:")
        lines.append(f"  {'[OK]' if self.fits_on_device else '[X]'} Fits on device "
                    f"({self.device_memory_bytes/1024**3:.0f} GB available)")
        if self.l2_cache_size_bytes > 0:
            lines.append(f"  {'[OK]' if self.fits_in_l2_cache else '[X]'} Fits in L2 cache "
                        f"({self.l2_cache_size_bytes/1024**2:.0f} MB available)")
        if hasattr(self, 'shared_memory_size_bytes') and self.shared_memory_size_bytes > 0:
            lines.append(f"  {'[OK]' if self.fits_in_shared_memory else '[X]'} Fits in shared memory "
                        f"({self.shared_memory_size_bytes/1024:.0f} KB available)")
        lines.append("")

        # Peak location
        lines.append(f"Peak occurs at: {self.peak_at_subgraph_name} (step {self.peak_at_step})")
        if self.memory_timeline:
            peak_entry = self.memory_timeline[self.peak_at_step]
            lines.append(f"  At this point:")
            lines.append(f"    - {peak_entry.num_live_tensors} tensors alive")
            lines.append(f"    - Activations: {peak_entry.activation_memory_bytes/1024**2:.1f} MB")
            if peak_entry.allocated_tensors:
                lines.append(f"    - Just allocated: {', '.join(peak_entry.allocated_tensors[:3])}")
        lines.append("")

        # Timeline
        if show_timeline and self.memory_timeline:
            lines.append(f"Memory Timeline (showing {timeline_steps} of {len(self.memory_timeline)} steps):")
            lines.append("  " + "-" * 76)
            lines.append(f"  {'Step':<6} {'Subgraph':<30} {'Memory':>10} {'Live':>6} {'Event':<20}")
            lines.append("  " + "-" * 76)

            # Show first timeline_steps entries
            for entry in self.memory_timeline[:timeline_steps]:
                event = ""
                if entry.allocated_tensors:
                    event = f"+{len(entry.allocated_tensors)}"
                if entry.freed_tensors:
                    event += f" -{len(entry.freed_tensors)}"

                lines.append(f"  {entry.step:<6} {entry.subgraph_name[:28]:<30} "
                           f"{entry.total_memory_mb():>9.1f}M {entry.num_live_tensors:>6} {event:<20}")

            if len(self.memory_timeline) > timeline_steps:
                lines.append(f"  ... ({len(self.memory_timeline) - timeline_steps} more steps)")
            lines.append("")

        # Optimization opportunities
        if self.optimization_suggestions:
            lines.append("Optimization Opportunities:")
            for suggestion in self.optimization_suggestions:
                lines.append(f"  {suggestion}")
            lines.append("")

        # Estimated savings
        if self.total_checkpoint_savings_bytes > 0 or self.total_quantization_savings_bytes > 0:
            lines.append("Estimated Savings:")
            if self.total_checkpoint_savings_bytes > 0:
                new_peak = self.peak_memory_bytes - self.total_checkpoint_savings_bytes
                reduction = (1 - new_peak/self.peak_memory_bytes) * 100
                lines.append(f"  With checkpointing: {self.peak_memory_mb:.1f} MB → "
                           f"{new_peak/1024**2:.1f} MB ({reduction:.0f}% reduction)")

            if self.total_quantization_savings_bytes > 0:
                new_peak = self.peak_memory_bytes - self.total_quantization_savings_bytes
                reduction = (1 - new_peak/self.peak_memory_bytes) * 100
                lines.append(f"  With quantization:  {self.peak_memory_mb:.1f} MB → "
                           f"{new_peak/1024**2:.1f} MB ({reduction:.0f}% reduction)")

            if self.total_checkpoint_savings_bytes > 0 and self.total_quantization_savings_bytes > 0:
                combined_savings = self.total_checkpoint_savings_bytes + self.total_quantization_savings_bytes
                new_peak = self.peak_memory_bytes - combined_savings
                reduction = (1 - new_peak/self.peak_memory_bytes) * 100
                lines.append(f"  Combined:           {self.peak_memory_mb:.1f} MB → "
                           f"{new_peak/1024**2:.1f} MB ({reduction:.0f}% reduction)")
            lines.append("")

        lines.append("=" * 80)

        return "\n".join(lines)

    def __str__(self) -> str:
        """String representation (short summary)"""
        return (f"MemoryReport(peak={self.peak_memory_mb:.1f}MB, "
                f"avg={self.average_memory_bytes/1024**2:.1f}MB, "
                f"util={self.memory_utilization*100:.0f}%)")


# ============================================================================
# Memory Estimator - Simulates Execution to Track Memory Usage
# ============================================================================


class MemoryEstimator:
    """
    Estimates memory footprint by simulating graph execution.

    Algorithm:
    1. Walk dependency graph in execution order (topological sort)
    2. For each subgraph:
       a. Allocate output tensors
       b. Allocate workspace if needed (im2col buffers, etc.)
       c. Record memory state in timeline
       d. Free inputs that are no longer needed
       e. Free workspace immediately (temporary)
    3. Track peak memory across all steps
    4. Analyze timeline for optimization opportunities

    Key Insight: Memory ≠ sum of all tensors!
    It's the maximum of concurrent allocations over time.
    """

    def __init__(self, resource_model: HardwareResourceModel):
        """
        Initialize memory estimator.

        Args:
            resource_model: Hardware specs for fit analysis
        """
        self.resource_model = resource_model

    def estimate_memory(
        self,
        subgraphs: List[SubgraphDescriptor],
        partition_report: PartitionReport
    ) -> MemoryReport:
        """
        Main entry point - estimate memory by simulating execution.

        Args:
            subgraphs: List of computational subgraphs
            partition_report: Graph partitioning results with dependency info

        Returns:
            MemoryReport with peak memory, timeline, and optimization suggestions
        """
        # Get execution order from dependency graph
        execution_order = self._get_execution_order(partition_report)

        # Simulate execution and track memory
        peak_memory, timeline = self._simulate_execution(subgraphs, execution_order)

        # Analyze component breakdown
        total_weight_memory = sum(sg.total_weight_bytes for sg in subgraphs)
        max_workspace = max((self._estimate_workspace(sg) for sg in subgraphs), default=0)

        # Calculate average memory over timeline
        if timeline:
            avg_memory = sum(entry.total_memory_bytes for entry in timeline) / len(timeline)
            memory_utilization = avg_memory / peak_memory if peak_memory > 0 else 0
        else:
            avg_memory = 0
            memory_utilization = 0

        # Find peak location
        peak_step = 0
        peak_at_subgraph = ""
        peak_at_subgraph_name = ""
        if timeline:
            peak_entry = max(timeline, key=lambda e: e.total_memory_bytes)
            peak_step = peak_entry.step
            peak_at_subgraph = peak_entry.subgraph_id
            peak_at_subgraph_name = peak_entry.subgraph_name

        # Calculate activation memory (peak - weights)
        activation_memory = max((entry.activation_memory_bytes for entry in timeline), default=0)

        # Analyze optimization opportunities
        optimizations = self._analyze_optimizations(subgraphs, peak_memory, timeline)

        # Check hardware fit
        hardware_fit = self._check_hardware_fit(peak_memory)

        # Create per-subgraph descriptors
        subgraph_descriptors = [
            self._create_subgraph_descriptor(sg) for sg in subgraphs
        ]

        return MemoryReport(
            peak_memory_bytes=peak_memory,
            peak_memory_mb=peak_memory / (1024 ** 2),
            peak_memory_gb=peak_memory / (1024 ** 3),
            activation_memory_bytes=activation_memory,
            weight_memory_bytes=total_weight_memory,
            workspace_memory_bytes=max_workspace,
            average_memory_bytes=avg_memory,
            memory_utilization=memory_utilization,
            fits_in_l2_cache=hardware_fit.get('fits_in_l2', False),
            fits_in_shared_memory=hardware_fit.get('fits_in_shared', False),
            fits_on_device=hardware_fit.get('fits_on_device', True),
            l2_cache_size_bytes=hardware_fit.get('l2_size', 0),
            device_memory_bytes=hardware_fit.get('device_size', 0),
            memory_timeline=timeline,
            peak_at_step=peak_step,
            peak_at_subgraph=peak_at_subgraph,
            peak_at_subgraph_name=peak_at_subgraph_name,
            total_checkpoint_savings_bytes=optimizations['checkpoint_savings'],
            total_quantization_savings_bytes=optimizations['quantization_savings'],
            optimization_suggestions=optimizations['suggestions'],
            subgraph_descriptors=subgraph_descriptors,
        )

    def _get_execution_order(self, partition_report: PartitionReport) -> List[str]:
        """
        Get execution order from dependency graph (topological sort).

        Args:
            partition_report: Contains subgraphs with dependency info

        Returns:
            List of subgraph IDs in execution order
        """
        # Build dependency graph from subgraph depends_on fields
        import networkx as nx
        dep_graph = nx.DiGraph()

        # Add all subgraphs as nodes
        for sg in partition_report.subgraphs:
            dep_graph.add_node(sg.node_id)

        # Add edges based on depends_on
        for sg in partition_report.subgraphs:
            for dependency_id in sg.depends_on:
                # Add edge from dependency to this subgraph
                dep_graph.add_edge(dependency_id, sg.node_id)

        # Topological sort using Kahn's algorithm
        # Calculate in-degree for each node
        in_degree = {node: 0 for node in dep_graph.nodes()}
        for node in dep_graph.nodes():
            for successor in dep_graph.successors(node):
                in_degree[successor] += 1

        # Queue of nodes with no incoming edges
        queue = [node for node, degree in in_degree.items() if degree == 0]
        execution_order = []

        while queue:
            # Take node with no dependencies
            node = queue.pop(0)
            execution_order.append(node)

            # Reduce in-degree of successors
            for successor in dep_graph.successors(node):
                in_degree[successor] -= 1
                if in_degree[successor] == 0:
                    queue.append(successor)

        # If we didn't process all nodes, there's a cycle (shouldn't happen)
        if len(execution_order) != len(dep_graph.nodes()):
            # Fallback: just use node IDs in order
            execution_order = [sg.node_id for sg in partition_report.subgraphs]

        return execution_order

    def _simulate_execution(
        self,
        subgraphs: List[SubgraphDescriptor],
        execution_order: List[str]
    ) -> Tuple[int, List[MemoryTimelineEntry]]:
        """
        Simulate execution and track memory allocations.

        Returns:
            (peak_memory_bytes, timeline)
        """
        # Create lookup for subgraphs
        subgraph_map = {sg.node_id: sg for sg in subgraphs}

        # Track live tensors: tensor_id -> size_bytes
        live_tensors = {}
        timeline = []
        peak_memory = 0

        # Allocate persistent weight memory upfront
        for sg in subgraphs:
            if sg.total_weight_bytes > 0:
                weight_id = f"{sg.node_id}_weights"
                live_tensors[weight_id] = sg.total_weight_bytes

        # Simulate execution step by step
        for step, sg_id in enumerate(execution_order):
            sg = subgraph_map.get(sg_id)
            if sg is None:
                continue

            allocated = []
            freed = []

            # 1. Allocate output tensor
            output_id = f"{sg_id}_output"
            if sg.total_output_bytes > 0:
                live_tensors[output_id] = sg.total_output_bytes
                allocated.append(output_id)

            # 2. Allocate workspace (temporary)
            workspace_id = f"{sg_id}_workspace"
            workspace_size = self._estimate_workspace(sg)
            if workspace_size > 0:
                live_tensors[workspace_id] = workspace_size
                allocated.append(workspace_id)

            # 3. Calculate current memory state
            current_memory = sum(live_tensors.values())
            # Activation memory = all non-weight tensors EXCLUDING workspace
            activation_memory = sum(
                size for tid, size in live_tensors.items()
                if not tid.endswith('_weights') and not tid.endswith('_workspace')
            )

            # 4. Record timeline entry
            timeline.append(MemoryTimelineEntry(
                step=step,
                subgraph_id=sg_id,
                subgraph_name=sg.node_name,
                total_memory_bytes=current_memory,
                activation_memory_bytes=activation_memory,
                workspace_memory_bytes=workspace_size,
                live_tensors=list(live_tensors.keys()),
                num_live_tensors=len(live_tensors),
                allocated_tensors=allocated.copy(),
                freed_tensors=[],  # Will be updated below
            ))

            # Track peak
            if current_memory > peak_memory:
                peak_memory = current_memory

            # 5. Free workspace immediately (it's temporary)
            if workspace_size > 0:
                del live_tensors[workspace_id]
                freed.append(workspace_id)

            # 6. Free dead input tensors
            dead_tensors = self._free_dead_tensors(
                sg, live_tensors, subgraphs, execution_order, step
            )
            freed.extend(dead_tensors)

            # Update timeline entry with freed tensors
            timeline[-1].freed_tensors = freed

        return peak_memory, timeline

    def _free_dead_tensors(
        self,
        current_sg: SubgraphDescriptor,
        live_tensors: Dict[str, int],
        all_subgraphs: List[SubgraphDescriptor],
        execution_order: List[str],
        current_step: int
    ) -> List[str]:
        """
        Free tensors that are no longer needed.

        A tensor can be freed if:
        1. It's not an input to any remaining subgraph
        2. It's not the final output of the graph

        Args:
            current_sg: Subgraph we just executed
            live_tensors: Currently allocated tensors (modified in place)
            all_subgraphs: All subgraphs in the graph
            execution_order: Order of execution
            current_step: Current execution step

        Returns:
            List of tensor IDs that were freed
        """
        freed = []

        # Get remaining subgraphs (haven't executed yet)
        remaining_ids = set(execution_order[current_step + 1:])

        # Build map of which tensors are needed by which future subgraphs
        subgraph_map = {sg.node_id: sg for sg in all_subgraphs}

        # Check each output tensor (skip weights, they're persistent)
        for tensor_id in list(live_tensors.keys()):
            if tensor_id.endswith('_weights'):
                continue  # Weights stay alive
            if tensor_id.endswith('_workspace'):
                continue  # Workspace already handled

            # Extract subgraph ID from tensor_id (format: "sgid_output")
            if '_output' in tensor_id:
                producing_sg_id = tensor_id.replace('_output', '')
            else:
                continue

            # Check if this tensor is needed by any remaining subgraph
            is_needed = False
            for remaining_id in remaining_ids:
                remaining_sg = subgraph_map.get(remaining_id)
                if remaining_sg and producing_sg_id in remaining_sg.depends_on:
                    is_needed = True
                    break

            # If not needed and not the current output, free it
            if not is_needed and producing_sg_id != current_sg.node_id:
                del live_tensors[tensor_id]
                freed.append(tensor_id)

        return freed

    def _estimate_workspace(self, subgraph: SubgraphDescriptor) -> int:
        """
        Estimate workspace memory needed for operation.

        Many operations need temporary buffers:
        - Conv2d: im2col buffer (~2× input size)
        - MatMul/Linear: transpose buffers
        - Attention: QKV projection buffers

        Args:
            subgraph: Subgraph to analyze

        Returns:
            Workspace size in bytes
        """
        op_type = subgraph.operation_type

        if op_type == OperationType.CONV2D:
            # Im2col expands input by kernel size
            # Conservative estimate: 2× input size
            # (exact formula depends on kernel size, padding, etc.)
            return subgraph.total_input_bytes * 2

        elif op_type in (OperationType.MATMUL, OperationType.LINEAR):
            # May need transpose buffers
            # Estimate: max(input, output) / 2
            return max(subgraph.total_input_bytes, subgraph.total_output_bytes) // 2

        elif op_type == OperationType.MULTIHEAD_ATTENTION:
            # Needs QKV projection buffers
            # Conservative: same as output
            return subgraph.total_output_bytes

        # Most other ops don't need significant workspace
        return 0

    def _analyze_optimizations(
        self,
        subgraphs: List[SubgraphDescriptor],
        peak_memory: int,
        timeline: List[MemoryTimelineEntry]
    ) -> Dict[str, any]:
        """
        Identify optimization opportunities.

        Detects:
        1. Activation checkpointing potential (large activation memory)
        2. Quantization potential (large weight memory)
        3. In-place operation opportunities (ReLU, Dropout)
        4. Buffer reuse potential

        Returns:
            Dict with checkpoint_savings, quantization_savings, suggestions
        """
        suggestions = []
        checkpoint_savings = 0
        quantization_savings = 0

        if not subgraphs:
            return {
                'checkpoint_savings': 0,
                'quantization_savings': 0,
                'suggestions': []
            }

        # 1. Activation checkpointing
        total_activation_memory = sum(sg.total_output_bytes for sg in subgraphs)

        if total_activation_memory > peak_memory * 0.5:
            # More than half of peak is activations - checkpointing helps
            # Recompute 60% of activations → save 60% of activation memory
            checkpoint_savings = int(total_activation_memory * 0.6)
            suggestions.append(
                f"[OK] Activation checkpointing: Save ~{checkpoint_savings/1024**2:.0f} MB "
                f"by recomputing 60% of activations (33% more compute)"
            )

        # 2. Quantization (FP32 → INT8 = 4× reduction)
        total_weight_memory = sum(sg.total_weight_bytes for sg in subgraphs)

        if total_weight_memory > peak_memory * 0.3:
            # Weights are >30% of peak - quantization helps
            # FP32 (4 bytes) → INT8 (1 byte) = 75% reduction
            quantization_savings = int(total_weight_memory * 0.75)
            suggestions.append(
                f"[OK] INT8 quantization: Save ~{quantization_savings/1024**2:.0f} MB "
                f"in weights (4x compression)"
            )

        # 3. In-place operations
        inplace_ops = [
            sg for sg in subgraphs
            if sg.operation_type in [
                OperationType.RELU,
                OperationType.RELU6,
                OperationType.DROPOUT,
            ]
        ]

        if inplace_ops:
            inplace_savings = sum(sg.total_output_bytes for sg in inplace_ops)
            suggestions.append(
                f"[OK] In-place ops: {len(inplace_ops)} ReLU/Dropout ops could save "
                f"~{inplace_savings/1024**2:.0f} MB"
            )

        # 4. Buffer reuse analysis
        if timeline:
            max_concurrent = max(entry.num_live_tensors for entry in timeline)
            total_tensors = len(subgraphs)

            if max_concurrent < total_tensors * 0.5:
                suggestions.append(
                    f"[OK] Buffer reuse: Only {max_concurrent} tensors alive at once "
                    f"(out of {total_tensors} total) - good reuse potential"
                )

        return {
            'checkpoint_savings': checkpoint_savings,
            'quantization_savings': quantization_savings,
            'suggestions': suggestions,
        }

    def _check_hardware_fit(self, peak_memory: int) -> Dict[str, any]:
        """
        Check if memory fits in various hardware constraints.

        Returns:
            Dict with fit flags and sizes
        """
        result = {}

        # L2/L3 cache
        if hasattr(self.resource_model, 'l2_cache_total'):
            l2_size = self.resource_model.l2_cache_total
            result['fits_in_l2'] = peak_memory <= l2_size
            result['l2_size'] = l2_size
        else:
            result['fits_in_l2'] = False
            result['l2_size'] = 0

        # GPU shared memory per SM
        if hasattr(self.resource_model, 'shared_memory_per_sm'):
            shared_size = self.resource_model.shared_memory_per_sm
            result['fits_in_shared'] = peak_memory <= shared_size
            result['shared_size'] = shared_size
        else:
            result['fits_in_shared'] = False
            result['shared_size'] = 0

        # Total device memory
        if hasattr(self.resource_model, 'main_memory'):
            device_size = self.resource_model.main_memory
            result['fits_on_device'] = peak_memory <= device_size
            result['device_size'] = device_size
        else:
            # Assume it fits if no limit specified
            result['fits_on_device'] = True
            result['device_size'] = 0

        return result

    def _create_subgraph_descriptor(
        self,
        sg: SubgraphDescriptor
    ) -> MemoryDescriptor:
        """
        Create MemoryDescriptor for a subgraph.

        Args:
            sg: Subgraph from partitioner

        Returns:
            MemoryDescriptor with optimization analysis
        """
        workspace = self._estimate_workspace(sg)

        # Checkpointing: beneficial for large activation-heavy ops
        can_checkpoint = sg.total_output_bytes > 10 * 1024 * 1024  # > 10 MB
        checkpoint_savings = sg.total_output_bytes if can_checkpoint else 0

        # Quantization: beneficial for ops with weights
        can_quantize = sg.total_weight_bytes > 5 * 1024 * 1024  # > 5 MB
        quantization_savings = int(sg.total_weight_bytes * 0.75) if can_quantize else 0

        return MemoryDescriptor(
            subgraph_id=sg.node_id,
            subgraph_name=sg.node_name,
            operation_type=sg.operation_type,
            input_memory_bytes=sg.total_input_bytes,
            output_memory_bytes=sg.total_output_bytes,
            weight_memory_bytes=sg.total_weight_bytes,
            workspace_memory_bytes=workspace,
            can_checkpoint=can_checkpoint,
            checkpoint_savings_bytes=checkpoint_savings,
            can_quantize=can_quantize,
            quantization_savings_bytes=quantization_savings,
            explanation=f"{sg.node_name} ({sg.operation_type.value})",
        )
