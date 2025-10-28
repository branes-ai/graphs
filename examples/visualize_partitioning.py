#!/usr/bin/env python
"""
Visualize Graph Partitioning - API Example
===========================================

This example demonstrates the core API for visualizing FX graph partitioning.
Shows how to:
1. Load and trace a model with PyTorch FX
2. Apply graph partitioning
3. Visualize the results side-by-side

For production use with full features (range selection, multiple models),
see the CLI tool: cli/graph_explorer.py

Usage:
    python examples/visualize_partitioning.py
"""

import torch
import torchvision.models as models
from torch.fx import symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp

from graphs.transform.partitioning.graph_partitioner import GraphPartitioner


def main():
    print("=" * 80)
    print("Graph Partitioning Visualization - API Example")
    print("=" * 80)
    print()

    # Step 1: Load and trace a model
    print("Step 1: Load and trace ResNet-18 with PyTorch FX")
    print("-" * 80)
    model = models.resnet18(weights=None)
    model.eval()

    input_tensor = torch.randn(1, 3, 224, 224)
    fx_graph = symbolic_trace(model)
    print(f"✓ Model traced: {len(list(fx_graph.graph.nodes))} FX nodes")
    print()

    # Step 2: Propagate shapes (required for partitioning)
    print("Step 2: Propagate tensor shapes through the graph")
    print("-" * 80)
    shape_prop = ShapeProp(fx_graph)
    shape_prop.propagate(input_tensor)
    print("✓ Shapes propagated")
    print()

    # Step 3: Partition the graph
    print("Step 3: Partition graph into fused subgraphs")
    print("-" * 80)
    partitioner = GraphPartitioner()
    report = partitioner.partition(fx_graph)
    print(f"✓ Created {report.total_subgraphs} subgraphs")
    print()

    # Step 4: Visualize (basic - show first 15 nodes)
    print("Step 4: Visualize FX graph and partitioned subgraphs side-by-side")
    print("-" * 80)
    visualization = partitioner.visualize_partitioning(fx_graph, start=0, end=15)
    print(visualization)

    # Example variations
    print()
    print("=" * 80)
    print("API Variations")
    print("=" * 80)
    print()

    # Variation 1: Show all nodes
    print("# To visualize all nodes:")
    print("visualization = partitioner.visualize_partitioning(fx_graph)")
    print()

    # Variation 2: Show specific range
    print("# To show first 50 nodes:")
    print("visualization = partitioner.visualize_partitioning(fx_graph, start=0, end=50)")
    print()
    print("# To show nodes 20-40:")
    print("visualization = partitioner.visualize_partitioning(fx_graph, start=20, end=40)")
    print()

    # Variation 3: Access partition report data
    print("# To access partition report data:")
    print(f"print(f'Total subgraphs: {{report.total_subgraphs}}')")
    print(f"print(f'Total FLOPs: {{report.total_flops / 1e9:.2f}} GFLOPs')")
    print(f"print(f'Total memory traffic: {{report.total_memory_traffic / 1e6:.2f}} MB')")
    print()
    print(f"Total subgraphs: {report.total_subgraphs}")
    print(f"Total FLOPs: {report.total_flops / 1e9:.2f} GFLOPs")
    print(f"Total memory traffic: {report.total_memory_traffic / 1e6:.2f} MB")
    print()

    # Next steps
    print("=" * 80)
    print("Next Steps")
    print("=" * 80)
    print()
    print("For production use with advanced features:")
    print("  - Range selection (--start, --end, --around)")
    print("  - Multiple models (--model)")
    print("  - Save to file (--output)")
    print()
    print("See: cli/graph_explorer.py")
    print("     cli/docs/graph_explorer.md")
    print()


if __name__ == "__main__":
    main()
