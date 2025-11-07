"""
Basic Dynamo Graph Extraction Patterns

This module demonstrates how to use torch._dynamo (the backend of torch.compile)
to extract computational graphs from complex PyTorch models.

Dynamo vs FX:
- FX (symbolic_trace): Limited to statically traceable code, fails on control flow
- Dynamo: Handles dynamic control flow, more robust for real-world models

Key Advantages of Dynamo:
1. Supports dynamic control flow (if/else, loops)
2. Works with models that have data-dependent shapes
3. Handles complex models like YOLO, DETR, Transformers
4. Provides graph breaks for unsupported operations (partial tracing)

Usage:
    python experiments/dynamo/basic_dynamo_tracing.py
"""

import torch
import torch._dynamo as dynamo
from typing import List, Dict, Any, Optional
import sys


class DynamoGraphExtractor:
    """
    Custom backend for torch.compile that extracts graph information
    without actually compiling the model.
    """

    def __init__(self):
        self.graphs: List[torch.fx.Graph] = []
        self.graph_modules: List[torch.fx.GraphModule] = []
        self.subgraph_count = 0

    def __call__(self, gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
        """
        Called by Dynamo for each graph partition.

        Args:
            gm: GraphModule containing the traced operations
            example_inputs: Example inputs used for shape propagation

        Returns:
            The original graph module (passthrough - no optimization)
        """
        self.subgraph_count += 1
        self.graphs.append(gm.graph)
        self.graph_modules.append(gm)

        print(f"\n{'='*80}")
        print(f"Graph Partition {self.subgraph_count}")
        print(f"{'='*80}")

        # Print graph in tabular format
        print("\nGraph Structure:")
        gm.graph.print_tabular()

        # Print detailed node information
        print("\nDetailed Node Information:")
        for node in gm.graph.nodes:
            self._print_node_info(node)

        # Count operations
        print("\nOperation Statistics:")
        self._print_op_stats(gm.graph)

        # Return the original graph module (no optimization)
        return gm.forward

    def _print_node_info(self, node: torch.fx.Node):
        """Print detailed information about a graph node."""
        print(f"\n  Node: {node.name}")
        print(f"    Op: {node.op}")
        print(f"    Target: {node.target}")
        print(f"    Args: {node.args}")
        print(f"    Kwargs: {node.kwargs}")

        # Print metadata if available
        if hasattr(node, 'meta'):
            meta = node.meta
            if 'tensor_meta' in meta:
                tm = meta['tensor_meta']
                print(f"    Shape: {tm.shape if hasattr(tm, 'shape') else 'unknown'}")
                print(f"    Dtype: {tm.dtype if hasattr(tm, 'dtype') else 'unknown'}")

    def _print_op_stats(self, graph: torch.fx.Graph):
        """Print operation statistics for the graph."""
        op_counts = {}
        for node in graph.nodes:
            if node.op == 'call_function':
                target_name = str(node.target).split('.')[-1].replace("'>", "")
                op_counts[target_name] = op_counts.get(target_name, 0) + 1
            elif node.op == 'call_module':
                op_counts['module_call'] = op_counts.get('module_call', 0) + 1

        for op, count in sorted(op_counts.items()):
            print(f"    {op}: {count}")

    def get_combined_graph(self) -> Optional[torch.fx.GraphModule]:
        """
        Get the combined graph module (if only one partition).
        For multiple partitions, returns the first one.
        """
        if len(self.graph_modules) > 0:
            return self.graph_modules[0]
        return None

    def reset(self):
        """Reset the extractor for a new model."""
        self.graphs = []
        self.graph_modules = []
        self.subgraph_count = 0


def trace_model_with_dynamo(
    model: torch.nn.Module,
    example_input: torch.Tensor,
    verbose: bool = True
) -> DynamoGraphExtractor:
    """
    Trace a PyTorch model using Dynamo and extract its computational graph.

    Args:
        model: PyTorch model to trace
        example_input: Example input tensor for shape propagation
        verbose: Print detailed information

    Returns:
        DynamoGraphExtractor containing the extracted graphs
    """
    # Create custom backend
    extractor = DynamoGraphExtractor()

    # Reset dynamo state (clears cached traces)
    dynamo.reset()

    # Compile with our custom backend
    compiled_model = torch.compile(
        model,
        backend=extractor,
        fullgraph=False,  # Allow graph breaks for unsupported ops
    )

    # Run the model to trigger tracing
    if verbose:
        print("\n" + "="*80)
        print("Starting Dynamo Tracing...")
        print("="*80)

    with torch.no_grad():
        _ = compiled_model(example_input)

    if verbose:
        print(f"\n{'='*80}")
        print(f"Tracing Complete: {extractor.subgraph_count} graph partition(s) found")
        print(f"{'='*80}\n")

    return extractor


# ============================================================================
# Example 1: Simple CNN
# ============================================================================

class SimpleCNN(torch.nn.Module):
    """Simple CNN for basic demonstration."""

    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool2d(2)
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc = torch.nn.Linear(32 * 56 * 56, 10)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# ============================================================================
# Example 2: Model with Control Flow (where FX fails)
# ============================================================================

class ModelWithControlFlow(torch.nn.Module):
    """Model with dynamic control flow that FX cannot trace."""

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 16, 3)
        self.bn = torch.nn.BatchNorm2d(16)

    def forward(self, x):
        x = self.conv(x)

        # Dynamic control flow - FX would fail here
        if x.sum() > 0:
            x = self.bn(x)

        return x


# ============================================================================
# Example 3: Model with Loops
# ============================================================================

class ModelWithLoops(torch.nn.Module):
    """Model with loops that demonstrates Dynamo's capabilities."""

    def __init__(self, num_layers: int = 3):
        super().__init__()
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList([
            torch.nn.Conv2d(16 if i > 0 else 3, 16, 3, padding=1)
            for i in range(num_layers)
        ])

    def forward(self, x):
        # Loop over layers - Dynamo handles this gracefully
        for conv in self.convs:
            x = torch.relu(conv(x))
        return x


# ============================================================================
# Main Examples
# ============================================================================

def example_simple_cnn():
    """Example: Trace simple CNN."""
    print("\n" + "="*80)
    print("EXAMPLE 1: Simple CNN")
    print("="*80)

    model = SimpleCNN()
    example_input = torch.randn(1, 3, 224, 224)

    extractor = trace_model_with_dynamo(model, example_input)

    return extractor


def example_control_flow():
    """Example: Trace model with control flow (where FX fails)."""
    print("\n" + "="*80)
    print("EXAMPLE 2: Model with Control Flow (FX would fail)")
    print("="*80)

    model = ModelWithControlFlow()
    example_input = torch.randn(1, 3, 224, 224)

    try:
        # Try FX first to show it fails
        print("\nAttempting FX symbolic_trace (will likely fail)...")
        from torch.fx import symbolic_trace
        _ = symbolic_trace(model)
        print("FX succeeded (unexpected!)")
    except Exception as e:
        print(f"FX failed as expected: {type(e).__name__}")

    print("\nNow trying Dynamo...")
    extractor = trace_model_with_dynamo(model, example_input)

    return extractor


def example_loops():
    """Example: Trace model with loops."""
    print("\n" + "="*80)
    print("EXAMPLE 3: Model with Loops")
    print("="*80)

    model = ModelWithLoops(num_layers=3)
    example_input = torch.randn(1, 3, 224, 224)

    extractor = trace_model_with_dynamo(model, example_input)

    return extractor


def compare_fx_vs_dynamo():
    """Compare FX and Dynamo on the same simple model."""
    print("\n" + "="*80)
    print("COMPARISON: FX vs Dynamo")
    print("="*80)

    model = SimpleCNN()
    example_input = torch.randn(1, 3, 224, 224)

    # FX tracing
    print("\n--- FX Symbolic Trace ---")
    from torch.fx import symbolic_trace
    try:
        fx_traced = symbolic_trace(model)
        print("\nFX Graph:")
        fx_traced.graph.print_tabular()
    except Exception as e:
        print(f"FX failed: {e}")

    # Dynamo tracing
    print("\n--- Dynamo Trace ---")
    dynamo_extractor = trace_model_with_dynamo(model, example_input)

    return dynamo_extractor


if __name__ == "__main__":
    # Run examples
    print("Dynamo Graph Extraction Examples")
    print("="*80)

    if len(sys.argv) > 1:
        example = sys.argv[1]
        if example == "simple":
            example_simple_cnn()
        elif example == "control":
            example_control_flow()
        elif example == "loops":
            example_loops()
        elif example == "compare":
            compare_fx_vs_dynamo()
        else:
            print(f"Unknown example: {example}")
            print("Available: simple, control, loops, compare")
    else:
        # Run all examples
        example_simple_cnn()
        example_control_flow()
        example_loops()
        compare_fx_vs_dynamo()
