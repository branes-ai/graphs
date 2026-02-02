"""
PyTorch Dynamo Frontend

Traces PyTorch models using torch.export (Dynamo) and partitions them into
subgraphs for performance analysis.

This is the primary frontend for PyTorch models, supporting:
- nn.Module tracing via torch.export.export()
- Shape propagation via torch.fx.passes.shape_prop.ShapeProp
- Fusion-based partitioning into SubgraphDescriptor objects

Usage:
    from graphs.frontends.dynamo import trace_and_partition

    model = torchvision.models.resnet18()
    input_tensor = torch.randn(1, 3, 224, 224)

    fx_graph, partition_report = trace_and_partition(model, input_tensor)
    print(f"Partitioned into {len(partition_report.subgraphs)} subgraphs")
    print(f"Total FLOPs: {partition_report.total_flops / 1e9:.2f} GFLOPs")
"""

from typing import Tuple, Optional

import torch
import torch.nn as nn
from torch.fx import GraphModule, symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp

from graphs.core.structures import PartitionReport
from graphs.transform.partitioning.fusion_partitioner import FusionBasedPartitioner

# torch.export.export() is stable from PyTorch 2.4+.
# On older versions (e.g. JetPack's 2.1.0a0) it segfaults.
_TORCH_VERSION = tuple(int(x) for x in torch.__version__.split('+')[0].split('a')[0].split('.')[:2])
_HAS_STABLE_EXPORT = _TORCH_VERSION >= (2, 4)


def _trace_model(
    model: nn.Module,
    input_tensor: torch.Tensor,
    verbose: bool = False,
) -> GraphModule:
    """Trace model using Dynamo export (>= 2.4) or symbolic_trace (fallback)."""
    if _HAS_STABLE_EXPORT:
        if verbose:
            print("  Tracing model with PyTorch Dynamo export...")
        try:
            exported_program = torch.export.export(model, (input_tensor,))
            fx_graph = exported_program.module()
            if verbose:
                print("    [OK] Dynamo export successful")
            return fx_graph
        except Exception as e:
            if verbose:
                print(f"    [X] Dynamo export failed: {e}")
                print("    Falling back to symbolic_trace...")

    # Fallback: torch.fx.symbolic_trace (works on all PyTorch >= 1.8)
    if verbose:
        print("  Tracing model with torch.fx.symbolic_trace...")
    try:
        fx_graph = symbolic_trace(model)
        if verbose:
            print("    [OK] symbolic_trace successful")
        return fx_graph
    except Exception as e:
        raise RuntimeError(f"Failed to trace model: {e}")


def trace_and_partition(
    model: nn.Module,
    input_tensor: torch.Tensor,
    verbose: bool = False,
) -> Tuple[GraphModule, PartitionReport]:
    """
    Trace a PyTorch model and partition into subgraphs.

    Uses PyTorch Dynamo export (state-of-the-art) for reliable tracing of
    complex models including YOLO, transformers, and segmentation models.

    Args:
        model: PyTorch model to trace
        input_tensor: Example input tensor for shape propagation
        verbose: Print progress messages

    Returns:
        Tuple of (fx_graph, partition_report):
        - fx_graph: The traced FX GraphModule
        - partition_report: PartitionReport with SubgraphDescriptor objects

    Raises:
        RuntimeError: If tracing fails

    Example:
        >>> import torch
        >>> from torchvision import models
        >>> from graphs.frontends.dynamo import trace_and_partition
        >>>
        >>> model = models.resnet18()
        >>> input_tensor = torch.randn(1, 3, 224, 224)
        >>> fx_graph, report = trace_and_partition(model, input_tensor)
        >>> print(f"Subgraphs: {len(report.subgraphs)}")
    """
    # Set model to evaluation mode (CRITICAL for BatchNorm with batch=1)
    # This prevents "Expected more than 1 value per channel" errors
    model.eval()

    # Warm-up model (important for lazy initialization)
    with torch.no_grad():
        try:
            _ = model(input_tensor)
        except Exception as e:
            if verbose:
                print(f"    Note: Warm-up failed ({e}), continuing anyway...")

    fx_graph = _trace_model(model, input_tensor, verbose)

    # Shape propagation
    shape_prop = ShapeProp(fx_graph)
    shape_prop.propagate(input_tensor)

    # Partition using FusionBasedPartitioner
    # FusionBasedPartitioner works better with Dynamo's flattened graph structure
    if verbose:
        print("  Running fusion-based partitioning...")

    partitioner = FusionBasedPartitioner()
    partition_report = partitioner.partition(fx_graph)

    if verbose:
        print(f"    Partitioned into {len(partition_report.subgraphs)} subgraphs")
        print(f"    Total FLOPs: {partition_report.total_flops / 1e9:.2f} GFLOPs")

    return fx_graph, partition_report


def trace_only(
    model: nn.Module,
    input_tensor: torch.Tensor,
    verbose: bool = False,
) -> GraphModule:
    """
    Trace a PyTorch model without partitioning.

    Useful when you want to inspect the raw FX graph before partitioning,
    or when using a custom partitioning strategy.

    Args:
        model: PyTorch model to trace
        input_tensor: Example input tensor for shape propagation
        verbose: Print progress messages

    Returns:
        The traced FX GraphModule with shape metadata

    Raises:
        RuntimeError: If tracing fails
    """
    model.eval()

    # Warm-up
    with torch.no_grad():
        try:
            _ = model(input_tensor)
        except Exception:
            pass

    fx_graph = _trace_model(model, input_tensor, verbose)

    # Shape propagation
    shape_prop = ShapeProp(fx_graph)
    shape_prop.propagate(input_tensor)

    return fx_graph


def get_model_stats(fx_graph: GraphModule) -> dict:
    """
    Extract basic statistics from a traced FX graph.

    Args:
        fx_graph: Traced FX GraphModule

    Returns:
        Dict with node counts by operation type
    """
    stats = {
        'total_nodes': 0,
        'call_function': 0,
        'call_module': 0,
        'call_method': 0,
        'get_attr': 0,
        'placeholder': 0,
        'output': 0,
    }

    for node in fx_graph.graph.nodes:
        stats['total_nodes'] += 1
        op = node.op
        if op in stats:
            stats[op] += 1

    return stats
