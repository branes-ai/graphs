"""
Attention-Enhanced Fusion Partitioner

This module extends the fusion-based partitioner with attention-specific fusion patterns,
including sequential fusion for attention operations and parallel fusion for Q, K, V projections.

Phase 3 of Enhanced Attention Fusion Plan:
- Sequential patterns: matmul → mul → softmax, softmax → dropout → matmul
- Parallel patterns: Q_proj || K_proj || V_proj (3 parallel linear operations)
- Special handling for attention computation chains
"""

from typing import List, Set, Dict, Tuple, Optional
from dataclasses import dataclass, replace
import torch
import torch.nn.functional as F
from torch.fx import GraphModule, Node

from .fusion_partitioner import (
    FusionBasedPartitioner,
    FusedSubgraph,
    FusionReport,
)

from graphs.core.structures import OperationType, BottleneckType


class AttentionFusionPartitioner(FusionBasedPartitioner):
    """
    Enhanced fusion partitioner with attention-specific patterns.

    Extends FusionBasedPartitioner to handle:
    1. Attention-specific sequential fusion patterns
    2. Parallel fusion for Q, K, V projections
    3. Attention computation chains (matmul → mul → softmax)
    4. Attention application chains (softmax → dropout → matmul)

    Key Improvements:
    - Fuses attention operations that current partitioner misses
    - Identifies and merges parallel Q, K, V projections
    - Expected memory reduction: 40-60% for attention blocks (vs 5.7% currently)
    """

    def __init__(self):
        super().__init__()
        self.parallel_fusions_created = 0  # Track parallel fusion successes

    def partition(self, fx_graph: GraphModule) -> FusionReport:
        """
        Partition FX graph with attention-specific fusion enhancements.

        Steps:
        1. Perform standard sequential fusion (with attention patterns)
        2. Detect parallel Q, K, V projection patterns
        3. Merge parallel projections into single fused subgraphs
        4. Update statistics and return report

        Args:
            fx_graph: Traced PyTorch FX graph with shape propagation

        Returns:
            FusionReport with enhanced fusion for attention operations
        """
        # Step 1: Standard sequential fusion (with attention patterns added to _is_fusible)
        report = super().partition(fx_graph)

        # Step 2: Post-process to detect and merge parallel Q, K, V projections
        if self.fused_subgraphs:
            self._merge_parallel_qkv_projections(fx_graph)

            # Regenerate report with merged subgraphs
            report = self._generate_report(
                self.fused_subgraphs,
                len([n for n in fx_graph.graph.nodes if n.op in ['call_module', 'call_function']])
            )

        return report

    def _is_fusible(self, node1: Node, node2: Node) -> bool:
        """
        Check if two nodes can be fused, with attention-specific patterns.

        Extends parent class with attention operation patterns:
        - matmul → mul (Q @ K^T → scale)
        - mul → softmax (scale → softmax)
        - matmul → mul (combined: Q @ K^T can fuse with scale)
        - softmax → matmul (attention weights @ V)
        - dropout → matmul (for attention apply with dropout)
        - transpose → reshape (multi-head manipulation)
        - reshape → transpose (multi-head manipulation)
        - linear → reshape (projection → head split)
        - reshape → linear (head concat → output projection)

        Args:
            node1: First node
            node2: Second node

        Returns:
            True if nodes can be fused
        """
        # First check parent class patterns (Conv, BN, ReLU, Linear, etc.)
        if super()._is_fusible(node1, node2):
            return True

        # Add attention-specific patterns
        type1 = self._get_node_type(node1)
        type2 = self._get_node_type(node2)

        # Attention-specific fusible patterns
        attention_patterns = [
            # Attention score computation chain
            ('matmul', 'mul'),        # Q @ K^T → scale
            ('mul', 'softmax'),       # scale → softmax
            ('matmul', 'softmax'),    # Q @ K^T → softmax (if scale is fused inline)

            # Attention apply chain
            ('softmax', 'matmul'),    # attention_weights @ V
            ('dropout', 'matmul'),    # dropout(attention_weights) @ V

            # Multi-head manipulation
            ('linear', 'reshape'),    # projection → split heads
            ('reshape', 'transpose'), # reshape → transpose for multi-head
            ('transpose', 'reshape'), # transpose → reshape (head concat)
            ('transpose', 'matmul'),  # transposed tensors → matmul
            ('reshape', 'contiguous'),# reshape → contiguous
            ('contiguous', 'linear'), # contiguous → output projection

            # Size extraction (dynamic shapes from decomposed attention)
            ('size', 'reshape'),      # size extraction → reshape
            ('size', 'view'),         # size extraction → view

            # Special attention patterns
            ('matmul', 'transpose'),  # Q @ K^T → transpose result
            ('transpose', 'softmax'), # transpose → softmax
        ]

        for pattern in attention_patterns:
            if (type1, type2) == pattern:
                return True

        return False

    def _merge_parallel_qkv_projections(self, fx_graph: GraphModule) -> None:
        """
        Detect and merge parallel Q, K, V projection patterns.

        Pattern to detect:
        ```
        LayerNorm output
          ├─→ Linear (Q_proj) → ...
          ├─→ Linear (K_proj) → ...
          └─→ Linear (V_proj) → ...
        ```

        These three Linear operations:
        1. Share the same input (output of LayerNorm or similar)
        2. Execute in parallel (no dependencies between them)
        3. Have similar output shapes (embed_dim)
        4. Are typically followed by reshape/transpose operations

        When detected, merge into a single "parallel-fused" subgraph.

        Args:
            fx_graph: The FX graph to analyze
        """
        # Build a map from node → subgraph
        node_to_subgraph = {}
        for sg in self.fused_subgraphs:
            for node_id in sg.node_ids:
                node_to_subgraph[node_id] = sg

        # Find all call_function and call_module nodes
        all_nodes = {n.name: n for n in fx_graph.graph.nodes
                    if n.op in ['call_function', 'call_module']}

        # Track which subgraphs we've already merged
        merged_subgraphs = set()
        parallel_groups = []

        # Look for patterns: one input feeding multiple Linear operations
        for node_name, node in all_nodes.items():
            if node_name not in node_to_subgraph:
                continue

            # Check if this is a linear operation
            if not self._is_linear_op(node, fx_graph):
                continue

            # Get the consumers of this node's producer (siblings)
            if not node.all_input_nodes:
                continue

            producer = node.all_input_nodes[0]  # Typically LayerNorm or similar

            # Find all consumers of the producer that are Linear operations
            linear_siblings = []
            for consumer in producer.users:
                if consumer.name in node_to_subgraph and self._is_linear_op(consumer, fx_graph):
                    linear_siblings.append(consumer)

            # If we have 3 linear siblings (Q, K, V), this is a parallel pattern
            if len(linear_siblings) == 3:
                # Get their subgraphs
                sibling_subgraphs = [node_to_subgraph[sib.name] for sib in linear_siblings]

                # Check if any are already merged
                if any(sg.subgraph_id in merged_subgraphs for sg in sibling_subgraphs):
                    continue

                # Check that they're all separate subgraphs
                subgraph_ids = [sg.subgraph_id for sg in sibling_subgraphs]
                if len(set(subgraph_ids)) == 3:
                    # This is a valid parallel Q, K, V pattern!
                    parallel_groups.append({
                        'subgraphs': sibling_subgraphs,
                        'nodes': linear_siblings,
                        'producer': producer,
                    })

                    # Mark as merged
                    for sg in sibling_subgraphs:
                        merged_subgraphs.add(sg.subgraph_id)

        # Merge the detected parallel groups
        for group in parallel_groups:
            self._create_parallel_fused_subgraph(group)
            self.parallel_fusions_created += 1

    def _is_linear_op(self, node: Node, fx_graph: GraphModule) -> bool:
        """Check if a node is a Linear operation"""
        if node.op == 'call_module':
            try:
                module = fx_graph.get_submodule(node.target)
                return isinstance(module, torch.nn.Linear)
            except:
                return False
        elif node.op == 'call_function':
            # Check for F.linear
            return node.target == F.linear or (hasattr(node.target, '__name__') and node.target.__name__ == 'linear')
        return False

    def _create_parallel_fused_subgraph(self, parallel_group: Dict) -> None:
        """
        Create a single fused subgraph from parallel Q, K, V projections.

        Merges three separate subgraphs into one "parallel-fused" subgraph,
        combining their FLOPs, memory, and updating fusion patterns.

        Args:
            parallel_group: Dict with 'subgraphs', 'nodes', 'producer' keys
        """
        subgraphs_to_merge = parallel_group['subgraphs']

        # Create new merged subgraph
        merged_id = subgraphs_to_merge[0].subgraph_id

        # Combine node lists
        all_node_ids = []
        all_node_names = []
        all_operation_types = []

        for sg in subgraphs_to_merge:
            all_node_ids.extend(sg.node_ids)
            all_node_names.extend(sg.node_names)
            all_operation_types.extend(sg.operation_types)

        # Combine metrics
        total_flops = sum(sg.total_flops for sg in subgraphs_to_merge)
        total_macs = sum(sg.total_macs for sg in subgraphs_to_merge)
        total_weight_bytes = sum(sg.total_weight_bytes for sg in subgraphs_to_merge)

        # For parallel operations, inputs are shared, outputs are separate
        # Only count shared input once
        total_input_bytes = subgraphs_to_merge[0].total_input_bytes  # Shared input
        total_output_bytes = sum(sg.total_output_bytes for sg in subgraphs_to_merge)

        # Internal bytes: intermediate tensors within each subgraph
        internal_bytes = sum(sg.internal_bytes for sg in subgraphs_to_merge)

        # Additionally, by fusing in parallel, we save the redundant reads of the shared input
        # Each parallel op reads the input, but fused they read it once
        shared_input_savings = subgraphs_to_merge[0].total_input_bytes * 2  # Saved 2 extra reads
        internal_bytes += shared_input_savings

        # Update arithmetic intensity
        external_bytes = total_input_bytes + total_output_bytes + total_weight_bytes
        arithmetic_intensity = total_flops / max(1, external_bytes)

        # Classify bottleneck
        if arithmetic_intensity > 50:
            bottleneck = BottleneckType.COMPUTE_BOUND
        elif arithmetic_intensity > 10:
            bottleneck = BottleneckType.BALANCED
        elif arithmetic_intensity > 1:
            bottleneck = BottleneckType.MEMORY_BOUND
        else:
            bottleneck = BottleneckType.BANDWIDTH_BOUND

        # Create new fused subgraph with parallel fusion pattern
        merged_subgraph = FusedSubgraph(
            subgraph_id=merged_id,
            node_ids=all_node_ids,
            node_names=all_node_names,
            operation_types=all_operation_types,
            total_flops=total_flops,
            total_macs=total_macs,
            total_input_bytes=total_input_bytes,
            total_output_bytes=total_output_bytes,
            internal_bytes=internal_bytes,
            total_weight_bytes=total_weight_bytes,
            parallelism=subgraphs_to_merge[0].parallelism,  # Use first one's parallelism
            fusion_pattern=f"Parallel_QKV_Linear ({len(subgraphs_to_merge)} parallel)",
            num_operators=sum(sg.num_operators for sg in subgraphs_to_merge),
            depends_on=[],  # Will be updated if needed
            arithmetic_intensity=arithmetic_intensity,
            recommended_bottleneck=bottleneck
        )

        # Remove old subgraphs and add merged one
        for sg in subgraphs_to_merge:
            if sg in self.fused_subgraphs:
                self.fused_subgraphs.remove(sg)

        self.fused_subgraphs.append(merged_subgraph)

    def get_attention_fusion_stats(self) -> Dict:
        """
        Get statistics specific to attention fusion enhancements.

        Returns:
            Dict with attention fusion metrics
        """
        stats = {
            'parallel_fusions_created': self.parallel_fusions_created,
            'parallel_fusion_subgraphs': [
                sg for sg in self.fused_subgraphs
                if 'Parallel_QKV' in sg.fusion_pattern
            ],
            'attention_sequential_fusions': [
                sg for sg in self.fused_subgraphs
                if any(pattern in sg.fusion_pattern.lower()
                       for pattern in ['matmul', 'softmax', 'transpose', 'reshape'])
            ],
        }

        # Calculate memory savings from parallel fusion
        parallel_subgraphs = stats['parallel_fusion_subgraphs']
        if parallel_subgraphs:
            total_parallel_savings = sum(
                sg.data_movement_reduction() * (sg.total_input_bytes + sg.total_output_bytes + sg.total_weight_bytes)
                for sg in parallel_subgraphs
            )
            stats['parallel_fusion_memory_saved_bytes'] = total_parallel_savings
        else:
            stats['parallel_fusion_memory_saved_bytes'] = 0

        return stats

    def print_attention_fusion_summary(self) -> str:
        """
        Generate a summary report of attention-specific fusion enhancements.

        Returns:
            Formatted string with attention fusion statistics
        """
        stats = self.get_attention_fusion_stats()

        lines = []
        lines.append("=" * 80)
        lines.append("ATTENTION FUSION ENHANCEMENTS (Phase 3)")
        lines.append("=" * 80)
        lines.append("")

        # Parallel fusion stats
        lines.append(f"Parallel Q,K,V Fusions Created: {stats['parallel_fusions_created']}")
        if stats['parallel_fusions_created'] > 0:
            parallel_sgs = stats['parallel_fusion_subgraphs']
            lines.append(f"  Total parallel-fused subgraphs: {len(parallel_sgs)}")

            if parallel_sgs:
                avg_ops = sum(sg.num_operators for sg in parallel_sgs) / len(parallel_sgs)
                lines.append(f"  Average operators per parallel fusion: {avg_ops:.1f}")

                total_savings_mb = stats['parallel_fusion_memory_saved_bytes'] / (1024 * 1024)
                lines.append(f"  Memory saved by parallel fusion: {total_savings_mb:.2f} MB")
                lines.append("")

                lines.append("  Parallel fusion details:")
                for i, sg in enumerate(parallel_sgs[:5], 1):  # Show first 5
                    reduction_pct = sg.data_movement_reduction() * 100
                    lines.append(f"    {i}. Subgraph #{sg.subgraph_id}: {sg.num_operators} ops, "
                               f"{reduction_pct:.1f}% memory reduction")

                if len(parallel_sgs) > 5:
                    lines.append(f"    ... and {len(parallel_sgs) - 5} more")
        else:
            lines.append("  ⚠ No parallel Q,K,V patterns detected")
            lines.append("  This may indicate:")
            lines.append("    - Model doesn't use decomposed attention")
            lines.append("    - Pattern detection needs tuning")
            lines.append("    - Attention operations already fused by other patterns")

        lines.append("")

        # Sequential attention fusion stats
        attn_sequential = stats['attention_sequential_fusions']
        lines.append(f"Attention Sequential Fusions: {len(attn_sequential)}")
        if attn_sequential:
            # Count by pattern type
            from collections import Counter
            pattern_types = []
            for sg in attn_sequential:
                if 'matmul' in sg.fusion_pattern.lower():
                    pattern_types.append('matmul-based')
                if 'softmax' in sg.fusion_pattern.lower():
                    pattern_types.append('softmax-based')
                if 'transpose' in sg.fusion_pattern.lower() or 'reshape' in sg.fusion_pattern.lower():
                    pattern_types.append('shape-manipulation')

            pattern_counts = Counter(pattern_types)
            lines.append("  Pattern types:")
            for pattern_type, count in pattern_counts.most_common():
                lines.append(f"    - {pattern_type}: {count}")

        lines.append("")
        lines.append("=" * 80)

        return "\n".join(lines)
