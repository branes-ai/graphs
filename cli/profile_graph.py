#!/usr/bin/env python
"""
Graph Profiler - Computational Graph Characterization
======================================================

Profiles PyTorch models by characterizing each operator in the computational graph:
- Execution order and hierarchical structure
- Computational requirements (MACs, FLOPs)
- Memory bandwidth demands
- Parameter counts and tensor shapes
- Operation-level resource analysis
"""

import torch
import torchvision.models as models
from torch.fx import symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp
import sys
import argparse
sys.path.insert(0, 'src')

from graphs.characterize.graph_partitioner import GraphPartitioner
from graphs.characterize.table_formatter import format_partition_table


def show_table(model_name: str, show_shapes: bool = False, input_shape=(1, 3, 224, 224)):
    """Show hierarchical table for a model"""

    print("=" * 100)
    print(f"HIERARCHICAL MODULE TABLE: {model_name}")
    print("=" * 100)

    # Load model
    if model_name == 'resnet18':
        model = models.resnet18(weights=None)
    elif model_name == 'resnet34':
        model = models.resnet34(weights=None)
    elif model_name == 'resnet50':
        model = models.resnet50(weights=None)
    elif model_name == 'mobilenet_v2':
        model = models.mobilenet_v2(weights=None)
    else:
        print(f"Unknown model: {model_name}")
        return

    model.eval()
    input_tensor = torch.randn(*input_shape)

    # Trace
    print("\n[1/3] Tracing with PyTorch FX...")
    fx_graph = symbolic_trace(model)
    ShapeProp(fx_graph).propagate(input_tensor)

    # Partition
    print("[2/3] Running graph partitioner...")
    partitioner = GraphPartitioner()
    report = partitioner.partition(fx_graph)

    # Format table
    print("[3/3] Formatting hierarchical table...")
    table = format_partition_table(fx_graph, report, show_shapes=show_shapes)

    # Display
    print("\n" + "=" * 100)
    print("GRAPH PROFILE")
    print("=" * 100)
    print()
    print(table)

    # Summary
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params / 1e6:.2f}M ({total_params:,})")
    print(f"Total FLOPs: {report.total_flops / 1e9:.3f} GFLOPs ({report.total_flops:,})")
    print(f"Total MACs: {report.total_macs / 1e9:.3f} GMACs ({report.total_macs:,})")

    total_memory = sum(sg.total_input_bytes + sg.total_output_bytes + sg.total_weight_bytes
                      for sg in report.subgraphs)
    print(f"Total memory: {total_memory / 1e6:.2f} MB ({total_memory:,} bytes)")

    print(f"\nSubgraphs: {len(report.subgraphs)}")
    print(f"Average AI: {report.average_arithmetic_intensity:.2f} FLOPs/byte")


def main():
    parser = argparse.ArgumentParser(
        description='Profile computational graphs: characterize execution order, compute (MACs/FLOPs), and memory demands',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Profile ResNet-18
  python cli/profile_graph.py --model resnet18

  # Show tensor shapes and parameter details
  python cli/profile_graph.py --model resnet18 --showshape

  # Profile MobileNet V2
  python cli/profile_graph.py --model mobilenet_v2
        """
    )

    parser.add_argument('--model', type=str, default='resnet18',
                       choices=['resnet18', 'resnet34', 'resnet50', 'mobilenet_v2'],
                       help='Model to analyze')

    parser.add_argument('--showshape', action='store_true',
                       help='Show parameter shapes (weight/bias dimensions)')

    parser.add_argument('--input-shape', type=int, nargs=4, default=[1, 3, 224, 224],
                       metavar=('B', 'C', 'H', 'W'),
                       help='Input tensor shape')

    args = parser.parse_args()

    show_table(args.model, show_shapes=args.showshape, input_shape=tuple(args.input_shape))


if __name__ == "__main__":
    main()
