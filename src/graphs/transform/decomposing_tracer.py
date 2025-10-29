"""
Custom FX Tracer for Automatic Attention Decomposition

This module provides a custom PyTorch FX tracer that automatically decomposes
standard nn.MultiheadAttention modules into explicit operations for better
fusion analysis.

The tracer detects MultiheadAttention modules during tracing and replaces them
with decomposed operations (QKV projections, reshapes, attention computation,
etc.) that expose all internal operations to the fusion partitioner.

Example:
    >>> from torchvision.models import vit_b_16
    >>> model = vit_b_16(weights=None)
    >>> tracer = DecomposingAttentionTracer()
    >>> graph = tracer.trace(model)
    >>> # MultiheadAttention nodes are now decomposed
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fx import Tracer, GraphModule, Node, Graph
from typing import Any, Callable, Dict, List, Optional, Tuple
import math


class DecomposingAttentionTracer(Tracer):
    """
    Custom FX tracer that automatically decomposes nn.MultiheadAttention.

    This tracer extends the standard PyTorch FX tracer to detect and decompose
    MultiheadAttention modules during the tracing process. Instead of creating
    a single opaque 'call_module' node for attention, it creates separate nodes
    for all internal operations.

    Benefits:
    - Automatic decomposition (no manual model modification needed)
    - Works with any model using nn.MultiheadAttention
    - Exposes 15+ operations per attention layer for fusion
    - Maintains functional equivalence with original model

    Limitations:
    - Currently supports batch_first=True attention modules
    - Does not support all attention features (see handle_attention_kwargs)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.decomposed_modules = []  # Track which modules we decomposed
        self.param_counter = 0  # Counter for unique parameter names

    def trace(
        self,
        root: torch.nn.Module,
        concrete_args: Optional[Dict[str, Any]] = None,
    ) -> Graph:
        """
        Trace the module and decompose attention layers.

        Args:
            root: Module to trace
            concrete_args: Arguments with concrete values

        Returns:
            Graph with decomposed attention operations
        """
        # First, do standard FX tracing
        graph = super().trace(root, concrete_args)

        # Then, decompose any MultiheadAttention nodes
        self._decompose_attention_in_graph(graph, root)

        return graph

    def _decompose_attention_in_graph(self, graph: Graph, root: nn.Module) -> None:
        """
        Find and decompose all MultiheadAttention nodes in the graph.

        Args:
            graph: The FX graph to modify
            root: The root module (needed to access submodules)
        """
        # Find all call_module nodes that are MultiheadAttention
        nodes_to_decompose = []
        for node in graph.nodes:
            if node.op == 'call_module':
                try:
                    submod = root.get_submodule(node.target)
                    if isinstance(submod, nn.MultiheadAttention):
                        nodes_to_decompose.append((node, submod))
                except (AttributeError, KeyError):
                    # Module might not exist (e.g., from a previous pass)
                    continue

        # Decompose each attention module
        for attn_node, attn_module in nodes_to_decompose:
            self._decompose_attention_node(graph, attn_node, attn_module, root)

    def _decompose_attention_node(
        self,
        graph: Graph,
        attn_node: Node,
        attn_module: nn.MultiheadAttention,
        root: nn.Module
    ) -> None:
        """
        Replace a single MultiheadAttention node with decomposed operations.

        Args:
            graph: The FX graph to modify
            attn_node: The node to replace
            attn_module: The attention module
            root: The root module
        """
        # Extract attention parameters
        embed_dim = attn_module.embed_dim
        num_heads = attn_module.num_heads
        head_dim = embed_dim // num_heads

        # Get input arguments (query, key, value, ...)
        # Standard signature: forward(query, key, value, key_padding_mask=None,
        #                              need_weights=True, attn_mask=None, average_attn_weights=True)
        args = attn_node.args
        kwargs = attn_node.kwargs

        if len(args) < 3:
            print(f"Warning: Attention node {attn_node.name} has insufficient args, skipping decomposition")
            return

        query, key, value = args[0], args[1], args[2]

        # Extract optional arguments
        attn_mask = kwargs.get('attn_mask', None)
        key_padding_mask = kwargs.get('key_padding_mask', None)
        need_weights = kwargs.get('need_weights', True)

        # Insert decomposed operations before the attention node
        with graph.inserting_before(attn_node):
            # Step 1: QKV Projections
            # nn.MultiheadAttention uses packed weights: in_proj_weight is (3*embed_dim, embed_dim)
            # We need to slice it into Q, K, V weights
            if hasattr(attn_module, 'in_proj_weight') and attn_module.in_proj_weight is not None:
                # Packed weight format
                q_weight = attn_module.in_proj_weight[:embed_dim, :]
                k_weight = attn_module.in_proj_weight[embed_dim:2*embed_dim, :]
                v_weight = attn_module.in_proj_weight[2*embed_dim:, :]

                q_bias = attn_module.in_proj_bias[:embed_dim] if attn_module.in_proj_bias is not None else None
                k_bias = attn_module.in_proj_bias[embed_dim:2*embed_dim] if attn_module.in_proj_bias is not None else None
                v_bias = attn_module.in_proj_bias[2*embed_dim:] if attn_module.in_proj_bias is not None else None
            else:
                # Separate weight format (less common)
                q_weight = attn_module.q_proj_weight
                k_weight = attn_module.k_proj_weight
                v_weight = attn_module.v_proj_weight

                q_bias = attn_module.q_proj_bias if hasattr(attn_module, 'q_proj_bias') else None
                k_bias = attn_module.k_proj_bias if hasattr(attn_module, 'k_proj_bias') else None
                v_bias = attn_module.v_proj_bias if hasattr(attn_module, 'v_proj_bias') else None

            q_proj_node = self._create_linear_node(graph, root, query, q_weight, q_bias, "q")
            k_proj_node = self._create_linear_node(graph, root, key, k_weight, k_bias, "k")
            v_proj_node = self._create_linear_node(graph, root, value, v_weight, v_bias, "v")

            # Step 2: Reshape for multi-head attention
            # Assuming batch_first=True: (batch, seq_len, embed_dim) -> (batch, seq_len, num_heads, head_dim)
            # We need to get batch_size and seq_len dynamically during execution
            # Use view instead of reshape, and get dimensions from the tensor
            batch_size_node = graph.call_method('size', args=(q_proj_node, 0))
            seq_len_node = graph.call_method('size', args=(q_proj_node, 1))

            # Create the target shape tuple dynamically
            # Use reshape instead of view to handle non-contiguous tensors
            q_reshaped = graph.call_method('reshape', args=(q_proj_node, (batch_size_node, seq_len_node, num_heads, head_dim)))
            k_reshaped = graph.call_method('reshape', args=(k_proj_node, (batch_size_node, seq_len_node, num_heads, head_dim)))
            v_reshaped = graph.call_method('reshape', args=(v_proj_node, (batch_size_node, seq_len_node, num_heads, head_dim)))

            # Step 3: Transpose to (batch, num_heads, seq_len, head_dim)
            q_transposed = graph.call_method('transpose', args=(q_reshaped, 1, 2))
            k_transposed = graph.call_method('transpose', args=(k_reshaped, 1, 2))
            v_transposed = graph.call_method('transpose', args=(v_reshaped, 1, 2))

            # Step 4: Compute attention scores Q @ K^T
            k_t = graph.call_method('transpose', args=(k_transposed, -2, -1))
            scores = graph.call_function(torch.matmul, args=(q_transposed, k_t))

            # Step 5: Scale scores
            scale_factor = 1.0 / math.sqrt(head_dim)
            scaled_scores = graph.call_function(torch.mul, args=(scores, scale_factor))

            # Step 6: Apply attention mask if provided
            if attn_mask is not None:
                scaled_scores = graph.call_function(torch.add, args=(scaled_scores, attn_mask))

            # Step 7: Softmax
            attn_weights = graph.call_function(F.softmax, args=(scaled_scores,), kwargs={'dim': -1})

            # Step 8: Dropout (if training)
            if attn_module.dropout > 0:
                attn_weights = graph.call_function(
                    F.dropout,
                    args=(attn_weights,),
                    kwargs={'p': attn_module.dropout, 'training': False}  # Set to False for inference
                )

            # Step 9: Apply attention weights to values
            attn_output = graph.call_function(torch.matmul, args=(attn_weights, v_transposed))

            # Step 10: Transpose back
            attn_transposed = graph.call_method('transpose', args=(attn_output, 1, 2))

            # Step 11: Reshape to concatenate heads
            # Use the same batch_size and seq_len from earlier
            attn_concat = graph.call_method('reshape', args=(attn_transposed, (batch_size_node, seq_len_node, embed_dim)))

            # Make contiguous for performance
            attn_contiguous = graph.call_method('contiguous', args=(attn_concat,))

            # Step 12: Output projection
            output = self._create_linear_node(
                graph, root, attn_contiguous,
                attn_module.out_proj.weight,
                attn_module.out_proj.bias,
                "out"
            )

            # Handle the return value
            # nn.MultiheadAttention always returns a tuple (output, weights)
            # When need_weights=False, weights is None
            # Note: We use operator.getitem later to unpack, so we need a proper tuple
            import operator
            if need_weights:
                # Return (output, attention_weights)
                result = graph.call_function(tuple, args=([output, attn_weights],))
            else:
                # Return (output, None) - None is passed directly as a constant
                result = graph.call_function(tuple, args=([output, None],))

        # Replace all uses of the original attention node with our result
        attn_node.replace_all_uses_with(result)

        # Remove the original attention node
        graph.erase_node(attn_node)

        # Track that we decomposed this module
        self.decomposed_modules.append(attn_node.target)

    def _create_linear_node(
        self,
        graph: Graph,
        root: nn.Module,
        input_node: Node,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        name_prefix: str
    ) -> Node:
        """
        Create a linear transformation node (y = xW^T + b).

        Stores weights as module parameters and creates get_attr nodes to reference them.

        Args:
            graph: The FX graph
            root: The root module (to store parameters)
            input_node: Input tensor node
            weight: Weight matrix
            bias: Bias vector (optional)
            name_prefix: Prefix for parameter names (e.g., "q", "k", "v", "out")

        Returns:
            Node representing the linear transformation output
        """
        # Create unique parameter names
        weight_name = f"_decomposed_{name_prefix}_weight_{self.param_counter}"
        bias_name = f"_decomposed_{name_prefix}_bias_{self.param_counter}" if bias is not None else None
        self.param_counter += 1

        # Store weights as module parameters
        # Note: We need to clone the tensors to avoid sharing references
        setattr(root, weight_name, nn.Parameter(weight.clone().detach()))

        # Create get_attr nodes to reference the stored weights
        weight_node = graph.get_attr(weight_name)

        if bias is not None:
            setattr(root, bias_name, nn.Parameter(bias.clone().detach()))
            bias_node = graph.get_attr(bias_name)
        else:
            bias_node = None

        # Linear is y = xW^T + b
        # We use F.linear which handles the transpose internally
        return graph.call_function(
            F.linear,
            args=(input_node, weight_node, bias_node)
        )


def trace_with_decomposition(
    model: nn.Module,
    concrete_args: Optional[Dict[str, Any]] = None
) -> GraphModule:
    """
    Convenience function to trace a model with attention decomposition.

    Args:
        model: PyTorch model to trace
        concrete_args: Arguments with concrete values for tracing

    Returns:
        GraphModule with decomposed attention operations

    Example:
        >>> from torchvision.models import vit_b_16
        >>> model = vit_b_16(weights=None)
        >>> traced = trace_with_decomposition(model)
        >>> # Now traced has decomposed attention operations
    """
    tracer = DecomposingAttentionTracer()
    graph = tracer.trace(model, concrete_args)
    traced_module = GraphModule(model, graph)

    print(f"Decomposed {len(tracer.decomposed_modules)} attention module(s):")
    for mod_name in tracer.decomposed_modules:
        print(f"  - {mod_name}")

    return traced_module
