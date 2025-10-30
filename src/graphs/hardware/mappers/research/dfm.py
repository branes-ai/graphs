"""
Data Flow Machine Mapper - Token-Based Execution Model

Maps computational graphs to a Data Flow Machine architecture using
token-based dataflow execution.

EXECUTION MODEL
===============

Token-Based Dataflow:
1. Each operation becomes an instruction token
2. Token contains: opcode + operand slots (data or wait-for-data)
3. CAM stores tokens waiting for operands
4. When operands arrive, token becomes "ready"
5. CAM controller dispatches ready tokens to PEs
6. PEs execute and produce result tokens
7. Result tokens routed back to CAM, matched to waiting tokens
8. Cycle repeats until computation completes

Instruction Token Format (conceptual):
  [opcode | operand1 | operand2 | result_dest | dependencies]

CAM Operation:
  - Associative search: Find all tokens with satisfied dependencies
  - Content matching: Match result tokens to waiting operand slots
  - Ready queue: Collect tokens ready for execution
  - Issue logic: Dispatch ready tokens to available PEs

Processing Elements (PEs):
  - 4× Integer ALUs (INT ops)
  - 2× FP Units (FP ops)
  - 1× Special Function Unit (SFU - transcendentals)
  - 1× Load/Store Unit (LSU - memory ops)

MAPPING STRATEGY
================

1. Token Generation:
   - Convert each subgraph operation to instruction token(s)
   - Large operations decomposed into multiple tokens
   - Track token dependencies (data flow edges)

2. CAM Slot Management:
   - 128-slot CAM window
   - Estimate in-flight tokens (window size)
   - Calculate CAM pressure and stalls

3. PE Allocation:
   - Route INT ops to INT ALUs
   - Route FP ops to FP Units
   - Route special ops to SFU
   - Route memory ops to LSU

4. Token Lifetime Analysis:
   - Time from token creation to result production
   - Includes: wait time + dispatch time + execute time + routing time
   - CAM stalls when >128 in-flight tokens

5. Energy Accounting:
   - CAM lookups (associative search)
   - Token matching (operand slot writes)
   - Token routing (crossbar network)
   - PE execution
   - Result routing

COMPARISON TO SUPERSCALAR x86
==============================

DFM Explicit Token Flow:
  - Compiler generates dataflow graph
  - Tokens explicitly represent dependencies
  - CAM holds in-flight instructions
  - Routing network explicit

x86 Implicit Dataflow Extraction:
  - Hardware extracts dataflow from instruction stream
  - Register renaming creates "tokens" (physical registers)
  - Reservation stations = CAM slots
  - Issue logic = CAM controller
  - Execution ports = PEs

The DFM makes explicit what x86 does in hardware!
"""

from typing import List, Dict, Optional
from dataclasses import dataclass

from graphs.ir.structures import (
    SubgraphDescriptor,
    PartitionReport,
    OperationType
)
from graphs.transform.partitioning.fusion_partitioner import FusionReport, FusedSubgraph
from graphs.hardware.resource_model import (
    HardwareResourceModel,
    HardwareMapper,
    HardwareAllocation,
    GraphHardwareAllocation,
    Precision,
    BottleneckType
)


@dataclass
class TokenDescriptor:
    """Description of an instruction token in the DFM"""
    token_id: int
    operation_type: OperationType
    flops: int
    bytes: int
    dependencies: List[int]  # token_ids this token depends on
    ready_cycle: int  # cycle when token becomes ready
    issue_cycle: int  # cycle when token dispatched to PE
    complete_cycle: int  # cycle when result available
    pe_assigned: str  # 'INT', 'FP', 'SFU', 'LSU'


class DFMMapper(HardwareMapper):
    """
    Hardware mapper for Data Flow Machine architecture.

    Maps subgraphs to token-based execution model with:
    - 128-slot CAM for in-flight tokens
    - 8 Processing Elements (4 INT, 2 FP, 1 SFU, 1 LSU)
    - Crossbar routing network
    - Token matching and dispatch logic
    """

    def __init__(
        self,
        resource_model: HardwareResourceModel,
        thermal_profile: Optional[str] = None
    ):
        """
        Initialize DFM mapper.

        Args:
            resource_model: DFM-128 resource model
            thermal_profile: Thermal operating point (default: nominal)
        """
        super().__init__(resource_model, thermal_profile)

        # DFM-specific parameters
        self.cam_slots = 128  # Instruction tokens that can be in-flight
        self.num_int_units = 4  # Integer ALUs
        self.num_fp_units = 2  # Floating-point units
        self.num_sfu_units = 1  # Special function units
        self.num_lsu_units = 1  # Load/store units

        # Token routing latency (cycles)
        self.token_routing_latency = 1  # 1 cycle through crossbar

        # CAM lookup latency (cycles)
        self.cam_lookup_latency = 1  # 1 cycle for associative search

    def _estimate_pe_for_operation(self, op_type: OperationType) -> str:
        """
        Determine which PE type handles this operation.

        Args:
            op_type: Operation type

        Returns:
            PE type: 'INT', 'FP', 'SFU', 'LSU'
        """
        if op_type in [OperationType.CONV2D, OperationType.LINEAR, OperationType.MATMUL]:
            return 'FP'  # Matrix operations go to FP units
        elif op_type in [OperationType.ACTIVATION, OperationType.NORMALIZATION]:
            return 'SFU'  # Non-linear activations go to SFU
        elif op_type == OperationType.POOLING:
            return 'INT'  # Pooling is essentially integer reductions
        elif op_type == OperationType.ATTENTION:
            return 'FP'  # Attention uses matmul, softmax
        elif op_type == OperationType.ELEMENTWISE:
            return 'INT'  # Element-wise ops go to integer ALUs
        else:
            return 'INT'  # Default to integer

    def _estimate_token_count(self, subgraph: SubgraphDescriptor) -> int:
        """
        Estimate number of instruction tokens for a subgraph.

        Large operations decomposed into multiple tokens.

        Args:
            subgraph: Subgraph descriptor

        Returns:
            Number of instruction tokens
        """
        # Base: one token per operation
        base_tokens = 1

        # Large FLOPs operations decomposed into multiple tokens
        # (each token represents ~1K FLOPs to model instruction granularity)
        if subgraph.flops > 1000:
            flop_tokens = max(1, subgraph.flops // 1000)
            return min(flop_tokens, 1000)  # Cap at 1K tokens per subgraph

        return base_tokens

    def _calculate_cam_pressure(
        self,
        partition_report: PartitionReport,
        batch_size: int
    ) -> Dict:
        """
        Calculate CAM utilization and pressure.

        With 128 slots, analyze how many tokens are in-flight at any time.

        Args:
            partition_report: Graph partition information
            batch_size: Batch size

        Returns:
            Dict with CAM statistics
        """
        total_tokens = 0
        max_concurrent_tokens = 0

        # Estimate tokens per execution stage (dependency level)
        for subgraph in partition_report.subgraphs:
            tokens = self._estimate_token_count(subgraph)
            total_tokens += tokens

            # Tokens at same dependency level can be in-flight concurrently
            # Assume avg ~16 tokens per stage (empirical from x86 studies)
            stage_tokens = min(tokens, 16)
            max_concurrent_tokens = max(max_concurrent_tokens, stage_tokens)

        # Account for batch size (more batches = more tokens)
        max_concurrent_tokens = min(max_concurrent_tokens * batch_size, self.cam_slots)

        cam_utilization = max_concurrent_tokens / self.cam_slots
        cam_stalls = 1.0 if max_concurrent_tokens > self.cam_slots else 0.0

        return {
            'total_tokens': total_tokens,
            'max_concurrent_tokens': max_concurrent_tokens,
            'cam_utilization': cam_utilization,
            'cam_stalls': cam_stalls,
            'cam_slots_used': min(max_concurrent_tokens, self.cam_slots)
        }

    def map_single_subgraph(
        self,
        subgraph: SubgraphDescriptor,
        batch_size: int,
        precision: Precision,
        concurrent_subgraphs: int = 1
    ) -> HardwareAllocation:
        """
        Map a single subgraph to DFM token-based execution.

        Args:
            subgraph: Subgraph to map
            batch_size: Batch size
            precision: Numerical precision
            concurrent_subgraphs: Number of concurrent subgraphs (for dependencies)

        Returns:
            HardwareAllocation for this subgraph
        """
        # Determine PE type for this operation
        pe_type = self._estimate_pe_for_operation(subgraph.operation_type)

        # Estimate number of tokens
        num_tokens = self._estimate_token_count(subgraph)

        # Number of PEs available for this operation type
        if pe_type == 'INT':
            available_pes = self.num_int_units
        elif pe_type == 'FP':
            available_pes = self.num_fp_units
        elif pe_type == 'SFU':
            available_pes = self.num_sfu_units
        else:  # LSU
            available_pes = self.num_lsu_units

        # Tokens execute in waves across available PEs
        # Each PE can execute one token per cycle
        waves = (num_tokens + available_pes - 1) // available_pes

        # Operations and bytes
        ops = subgraph.flops * batch_size
        bytes_transferred = (
            subgraph.total_input_bytes +
            subgraph.total_output_bytes +
            subgraph.total_weight_bytes
        ) * batch_size

        # Calculate timing using roofline
        compute_time, memory_time, bottleneck = self._calculate_time_roofline(
            ops, bytes_transferred, precision
        )

        # Add DFM-specific overheads
        # Token routing: 1 cycle per token (through crossbar)
        routing_time = (num_tokens * self.token_routing_latency) / (self.resource_model.base_clock_ghz * 1e9)

        # CAM lookup overhead: 1 cycle per token match
        cam_time = (num_tokens * self.cam_lookup_latency) / (self.resource_model.base_clock_ghz * 1e9)

        # Total latency: max(compute, memory) + routing + CAM
        estimated_latency = max(compute_time, memory_time) + routing_time + cam_time

        # Energy calculation
        compute_energy, memory_energy = self._calculate_energy(
            ops, bytes_transferred, precision
        )
        total_energy = compute_energy + memory_energy

        # Utilization: how well we use the PEs
        # Perfect utilization = tokens exactly fill PE cycles
        ideal_cycles = num_tokens / available_pes
        actual_cycles = waves
        pe_utilization = ideal_cycles / actual_cycles if actual_cycles > 0 else 0.0

        # Overall utilization considers all 8 PEs
        utilization = (available_pes / 8.0) * pe_utilization

        return HardwareAllocation(
            subgraph_id=str(subgraph.node_id),
            subgraph_name=subgraph.node_name,
            precision=precision,
            threads_required=num_tokens,  # Each token is like a "thread"
            warps_required=0,  # Not applicable to DFM
            compute_units_allocated=available_pes,
            compute_units_ideal=available_pes,
            occupancy=pe_utilization,
            utilization=utilization,
            bottleneck=bottleneck,
            compute_time=compute_time,
            memory_time=memory_time,
            estimated_latency=estimated_latency,
            compute_energy=compute_energy,
            memory_energy=memory_energy,
            total_energy=total_energy,
            execution_stage=0,  # Will be filled by analyze_partition
            is_parallel=concurrent_subgraphs > 1
        )

    def analyze_partition(
        self,
        partition_report: PartitionReport,
        batch_size: int,
        precision: Precision
    ) -> GraphHardwareAllocation:
        """
        Analyze complete partition with token-based execution model.

        Args:
            partition_report: Complete graph partition
            batch_size: Batch size
            precision: Numerical precision

        Returns:
            Complete hardware allocation with CAM analysis
        """
        # Calculate CAM pressure
        cam_stats = self._calculate_cam_pressure(partition_report, batch_size)

        # Map each subgraph to tokens and PEs
        subgraph_allocations: List[HardwareAllocation] = []

        for subgraph in partition_report.subgraphs:
            allocation = self.map_single_subgraph(
                subgraph, batch_size, precision, concurrent_subgraphs=1
            )
            subgraph_allocations.append(allocation)

        # Calculate total metrics
        total_latency = sum(a.estimated_latency for a in subgraph_allocations)
        total_energy = sum(a.total_energy for a in subgraph_allocations)

        # Average utilization across all subgraphs
        avg_utilization = (
            sum(a.utilization for a in subgraph_allocations) / len(subgraph_allocations)
            if subgraph_allocations else 0.0
        )

        # Peak utilization (when CAM is fullest)
        peak_utilization = cam_stats['cam_utilization']

        # Bottleneck counts
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
            total_subgraphs=len(subgraph_allocations),
            total_execution_stages=1,  # DFM executes in dataflow order
            peak_compute_units_used=8,  # All 8 PEs can be active
            average_compute_units_used=8 * avg_utilization,
            peak_utilization=peak_utilization,
            average_utilization=avg_utilization,
            total_latency=total_latency,
            latency_breakdown={0: total_latency},
            total_energy=total_energy,
            naive_latency=total_latency * 0.5,  # Assume 2× speedup from dataflow
            latency_correction_factor=2.0,
            compute_bound_count=compute_bound,
            memory_bound_count=memory_bound,
            bandwidth_bound_count=bandwidth_bound,
            balanced_count=balanced,
        )

    def map_subgraph(
        self,
        subgraph: FusedSubgraph,
        execution_stage: int,
        concurrent_subgraphs: int,
        precision: Precision = Precision.FP32
    ) -> HardwareAllocation:
        """
        Map a single fused subgraph to hardware resources (required by ABC).

        This is an adapter that converts FusedSubgraph to SubgraphDescriptor
        and calls our internal mapping logic.

        Args:
            subgraph: Fused subgraph from Phase 1
            execution_stage: Which execution stage
            concurrent_subgraphs: How many subgraphs run in parallel
            precision: Numerical precision

        Returns:
            HardwareAllocation
        """
        # Convert FusedSubgraph to SubgraphDescriptor for our internal logic
        # FusedSubgraph has compatible fields
        subgraph_desc = SubgraphDescriptor(
            node_id=str(subgraph.subgraph_id),
            node_name=", ".join(subgraph.node_names[:2]),
            operation_type=subgraph.operation_type,
            fusion_pattern=subgraph.fusion_pattern,
            flops=subgraph.ops_total,
            macs=subgraph.macs_total,
            input_tensors=[],  # Not needed for DFM mapping
            output_tensors=[],
            weight_tensors=[],
            total_input_bytes=subgraph.input_memory_traffic,
            total_output_bytes=subgraph.output_memory_traffic,
            total_weight_bytes=subgraph.weight_memory_traffic,
            arithmetic_intensity=subgraph.arithmetic_intensity,
            parallelism=subgraph.parallelism,
            depends_on=[],
            dependency_type="sequential"
        )

        allocation = self.map_single_subgraph(
            subgraph_desc,
            batch_size=1,  # Will be scaled by map_graph
            precision=precision,
            concurrent_subgraphs=concurrent_subgraphs
        )

        # Set execution stage
        allocation.execution_stage = execution_stage
        return allocation

    def map_graph(
        self,
        fusion_report: FusionReport,
        execution_stages: List[List[int]],
        batch_size: int = 1,
        precision: Precision = Precision.FP32
    ) -> GraphHardwareAllocation:
        """
        Map entire computation graph to hardware (required by ABC).

        Args:
            fusion_report: Output from Phase 1 fusion partitioner
            execution_stages: List of execution stages
            batch_size: Batch size
            precision: Numerical precision

        Returns:
            Complete hardware allocation
        """
        # Convert FusionReport to PartitionReport for our internal logic
        partition_report = PartitionReport(
            subgraphs=[
                SubgraphDescriptor(
                    node_id=str(sg.subgraph_id),
                    node_name=", ".join(sg.node_names[:2]),
                    operation_type=sg.operation_type,
                    fusion_pattern=sg.fusion_pattern,
                    flops=sg.ops_total,
                    macs=sg.macs_total,
                    input_tensors=[],
                    output_tensors=[],
                    weight_tensors=[],
                    total_input_bytes=sg.input_memory_traffic,
                    total_output_bytes=sg.output_memory_traffic,
                    total_weight_bytes=sg.weight_memory_traffic,
                    arithmetic_intensity=sg.arithmetic_intensity,
                    parallelism=sg.parallelism,
                    depends_on=[],
                    dependency_type="sequential"
                )
                for sg in fusion_report.fused_subgraphs
            ],
            total_subgraphs=fusion_report.num_subgraphs,
            total_flops=fusion_report.total_flops,
            total_macs=fusion_report.total_macs,
            total_memory_traffic=fusion_report.total_bytes_transferred,
            average_arithmetic_intensity=fusion_report.avg_arithmetic_intensity,
            min_arithmetic_intensity=fusion_report.min_arithmetic_intensity,
            max_arithmetic_intensity=fusion_report.max_arithmetic_intensity,
            operation_type_counts={},
            fusion_pattern_counts={}
        )

        return self.analyze_partition(partition_report, batch_size, precision)


def create_dfm_128_mapper(thermal_profile: str = None) -> DFMMapper:
    """
    Create DFM mapper for DFM-128 Data Flow Machine.

    ARCHITECTURE:
    - 128-slot CAM for instruction tokens
    - 8 Processing Elements:
      - 4× Integer ALUs
      - 2× FP Units
      - 1× Special Function Unit
      - 1× Load/Store Unit
    - Crossbar routing network
    - Token-based dataflow execution

    MAPPING STRATEGY:
    - Convert operations to instruction tokens
    - Track token dependencies (dataflow edges)
    - Manage 128-slot CAM window
    - Route tokens to appropriate PEs
    - Account for routing and CAM overhead

    ENERGY MODEL:
    - CAM lookups (associative search)
    - Token matching (operand writes)
    - Token routing (crossbar network)
    - PE execution
    - Result routing

    Args:
        thermal_profile: Thermal operating point ('15W', '25W', '35W')
                        Default: '25W' (nominal)

    Returns:
        DFMMapper configured for DFM-128

    Example:
        >>> mapper = create_dfm_128_mapper(thermal_profile='35W')
        >>> # Use with UnifiedAnalyzer
        >>> analyzer = UnifiedAnalyzer()
        >>> result = analyzer.analyze_model_with_custom_hardware(
        ...     model=model,
        ...     input_tensor=input_tensor,
        ...     model_name='resnet18',
        ...     hardware_mapper=mapper,
        ...     precision=Precision.FP32
        ... )
    """
    from ...models.research.dfm_128 import dfm_128_resource_model
    from ...architectural_energy import DataFlowMachineEnergyModel

    model = dfm_128_resource_model()

    # Configure architectural energy model for DFM (DATA_FLOW_MACHINE)
    model.architecture_energy_model = DataFlowMachineEnergyModel(
        cam_lookup_per_cycle=5.0e-12,      # 5 pJ per 128-way associative search
        token_matching_energy=3.0e-12,      # 3 pJ per token match
        token_queue_management=1.5e-12,     # 1.5 pJ per queue operation
        graph_traversal_per_node=2.0e-12,   # 2 pJ per dataflow graph node
        compute_efficiency=0.40,             # 40% overhead (60% reduction vs CPU)
        memory_efficiency=0.45,              # 45% overhead (55% reduction vs CPU)
        cam_lookups_per_op=0.5,             # ~1 CAM lookup per 2 ops
    )

    return DFMMapper(model, thermal_profile=thermal_profile)
