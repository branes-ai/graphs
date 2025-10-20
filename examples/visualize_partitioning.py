#!/usr/bin/env python
"""
Visualize Graph Partitioning
=============================

This script demonstrates the side-by-side visualization of FX graph
and partitioned subgraphs.

Usage:
    python examples/visualize_partitioning.py [max_nodes]

Arguments:
    max_nodes: Optional, maximum number of nodes to display (default: 20)
"""

import torch
import torchvision.models as models
from torch.fx import symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp
import sys
sys.path.insert(0, 'src')

from graphs.characterize.graph_partitioner import GraphPartitioner


def main():
    # Get max_nodes from command line, default to 20
    max_nodes = int(sys.argv[1]) if len(sys.argv) > 1 else 20

    print("Graph Partitioning Visualization Demo")
    print("=" * 80)
    print()

    # Load and trace ResNet-18
    print("Loading ResNet-18...")
    model = models.resnet18(weights=None)
    model.eval()

    print("Tracing with PyTorch FX...")
    input_tensor = torch.randn(1, 3, 224, 224)
    fx_graph = symbolic_trace(model)

    print("Propagating shapes...")
    shape_prop = ShapeProp(fx_graph)
    shape_prop.propagate(input_tensor)

    print("Partitioning graph...")
    partitioner = GraphPartitioner()
    report = partitioner.partition(fx_graph)

    print(f"Created {report.total_subgraphs} subgraphs from {len(list(fx_graph.graph.nodes))} FX nodes")
    print()

    # Generate and display visualization
    print("Generating side-by-side visualization...")
    print()
    visualization = partitioner.visualize_partitioning(fx_graph, max_nodes=max_nodes)
    print(visualization)

    # Usage tips
    print()
    print("=" * 80)
    print("TIPS:")
    print("=" * 80)
    print()
    print("- Increase max_nodes to see more of the graph:")
    print(f"  python {sys.argv[0]} 50")
    print()
    print("- See the entire graph (may be very long):")
    print(f"  python {sys.argv[0]} 999")
    print()
    print("- Pipe to a file for easier viewing:")
    print(f"  python {sys.argv[0]} 999 > partitioning_viz.txt")
    print()


if __name__ == "__main__":
    main()
