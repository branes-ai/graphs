#!/usr/bin/env python
"""
Graph Explorer CLI
==================

Command-line tool to explore FX computational graphs interactively.
Provides three modes: model discovery, graph summary, and detailed visualization.

Usage:
    # Discover available models
    python cli/graph_explorer.py

    # Get model summary statistics
    python cli/graph_explorer.py --model resnet18

    # Visualize first 50 nodes
    python cli/graph_explorer.py --model resnet18 --max-nodes 50

    # Visualize specific range (nodes 20-50)
    python cli/graph_explorer.py --model resnet18 --start 20 --end 50

    # Investigate around node 35 (±10 nodes context)
    python cli/graph_explorer.py --model resnet18 --around 35 --context 10

Command-line Options:
    --model:       Model name (resnet18, mobilenet_v2, etc.)
    --start:       Start node index (0-based)
    --end:         End node index (exclusive)
    --around:      Center node for context view
    --context:     Number of nodes before/after center (default: 10)
    --max-nodes:   Maximum nodes to display (from start)
    --input-shape: Input tensor shape (default: 1,3,224,224)
    --output:      Save visualization to file

"""

import torch
import torchvision.models as models
from torch.fx import symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp
import argparse
import sys
from typing import Optional, Tuple

from graphs.transform.partitioning.graph_partitioner import GraphPartitioner


class GraphExplorerCLI:
    """Command-line interface for exploring FX computational graphs"""

    SUPPORTED_MODELS = {
        'resnet18': models.resnet18,
        'resnet34': models.resnet34,
        'resnet50': models.resnet50,
        'resnet101': models.resnet101,
        'resnet152': models.resnet152,
        'mobilenet_v2': models.mobilenet_v2,
        'mobilenet_v3_small': models.mobilenet_v3_small,
        'mobilenet_v3_large': models.mobilenet_v3_large,
        'efficientnet_b0': models.efficientnet_b0,
        'efficientnet_b1': models.efficientnet_b1,
        'efficientnet_b2': models.efficientnet_b2,
        'efficientnet_b3': models.efficientnet_b3,
        'efficientnet_b4': models.efficientnet_b4,
        'vit_b_16': models.vit_b_16,
        'vit_b_32': models.vit_b_32,
        'vit_l_16': models.vit_l_16,
        'swin_t': models.swin_t,
        'swin_s': models.swin_s,
        'swin_b': models.swin_b,
        'convnext_tiny': models.convnext_tiny,
        'convnext_small': models.convnext_small,
        'convnext_base': models.convnext_base,
    }

    def __init__(self):
        self.fx_graph = None
        self.model_name = None
        self.partitioner = GraphPartitioner()
        self.report = None

    @classmethod
    def show_model_list(cls):
        """Display organized list of supported models"""
        print("=" * 80)
        print("ERROR: Please specify a model with --model")
        print("=" * 80)
        print()
        print("Usage: ./cli/graph_explorer.py --model MODEL_NAME [OPTIONS]")
        print()
        print("=" * 80)
        print("SUPPORTED MODELS")
        print("=" * 80)
        print()

        # Organize by family
        families = {
            "ResNet": [k for k in cls.SUPPORTED_MODELS.keys() if k.startswith('resnet')],
            "MobileNet": [k for k in cls.SUPPORTED_MODELS.keys() if k.startswith('mobilenet')],
            "EfficientNet": [k for k in cls.SUPPORTED_MODELS.keys() if k.startswith('efficientnet')],
            "Vision Transformer (ViT)": [k for k in cls.SUPPORTED_MODELS.keys() if k.startswith('vit')],
            "Swin Transformer": [k for k in cls.SUPPORTED_MODELS.keys() if k.startswith('swin')],
            "ConvNeXt": [k for k in cls.SUPPORTED_MODELS.keys() if k.startswith('convnext')],
        }

        for family, models in families.items():
            if models:
                print(f"{family}:")
                for model in sorted(models):
                    print(f"  - {model}")
                print()

        print("=" * 80)
        print("EXAMPLES")
        print("=" * 80)
        print()
        print("# Get model summary")
        print("./cli/graph_explorer.py --model resnet18")
        print()
        print("# Visualize specific range")
        print("./cli/graph_explorer.py --model resnet18 --max-nodes 20")
        print()
        print("# Investigate around specific node")
        print("./cli/graph_explorer.py --model resnet18 --around 35 --context 10")
        print()

    def show_summary(self):
        """Display model summary without full visualization"""
        print()
        print("=" * 80)
        print(f"MODEL SUMMARY: {self.model_name}")
        print("=" * 80)
        print()

        # Basic statistics
        total_nodes = len(list(self.fx_graph.graph.nodes))
        print(f"Total FX Nodes:        {total_nodes}")
        print(f"Partitioned Subgraphs: {self.report.total_subgraphs}")
        print(f"Nodes Not Partitioned: {total_nodes - self.report.total_subgraphs}")
        print()

        # Computation statistics
        print(f"Total FLOPs:           {self.report.total_flops / 1e9:.2f} GFLOPs")
        print(f"Total MACs:            {self.report.total_macs / 1e6:.2f} M")
        print(f"Total Memory Traffic:  {self.report.total_memory_traffic / 1e6:.2f} MB")
        print()

        # Arithmetic intensity
        print(f"Arithmetic Intensity:")
        print(f"  Average: {self.report.average_arithmetic_intensity:.1f} FLOPs/byte")
        print(f"  Range:   {self.report.min_arithmetic_intensity:.1f} - {self.report.max_arithmetic_intensity:.1f} FLOPs/byte")
        print()

        # Bottleneck distribution
        if self.report.bottleneck_distribution:
            print("Bottleneck Distribution:")
            total_bottlenecks = sum(self.report.bottleneck_distribution.values())
            for bottleneck, count in sorted(self.report.bottleneck_distribution.items()):
                percentage = (count / total_bottlenecks) * 100
                print(f"  {bottleneck:20s}: {count:3d} ({percentage:5.1f}%)")
            print()

        # Operation type distribution
        if self.report.operation_type_counts:
            print("Operation Type Distribution:")
            total_ops = sum(self.report.operation_type_counts.values())
            # Show top 10 most common
            sorted_ops = sorted(self.report.operation_type_counts.items(),
                              key=lambda x: x[1], reverse=True)[:10]
            for op_type, count in sorted_ops:
                percentage = (count / total_ops) * 100
                print(f"  {op_type:20s}: {count:3d} ({percentage:5.1f}%)")
            if len(self.report.operation_type_counts) > 10:
                print(f"  ... and {len(self.report.operation_type_counts) - 10} more")
            print()

        # Partition reason distribution
        if self.report.partition_reason_distribution:
            print("Partition Reason Distribution:")
            total_reasons = sum(self.report.partition_reason_distribution.values())
            for reason, count in sorted(self.report.partition_reason_distribution.items()):
                percentage = (count / total_reasons) * 100
                print(f"  {reason:30s}: {count:3d} ({percentage:5.1f}%)")
            print()

        # Guidance
        print("=" * 80)
        print("NEXT STEPS: Visualize Specific Sections")
        print("=" * 80)
        print()
        print("The graph has too many nodes to display all at once.")
        print("Use one of these options to visualize specific sections:")
        print()
        print("# Show first 20 nodes")
        print(f"./cli/graph_explorer.py --model {self.model_name} --max-nodes 20")
        print()
        print("# Show nodes 20-50")
        print(f"./cli/graph_explorer.py --model {self.model_name} --start 20 --end 50")
        print()
        print("# Investigate around node 35 (±10 context)")
        print(f"./cli/graph_explorer.py --model {self.model_name} --around 35 --context 10")
        print()
        print("# Save full visualization to file")
        print(f"./cli/graph_explorer.py --model {self.model_name} --output full_viz.txt")
        print()

    def load_and_trace_model(self, model_name: str, input_shape=(1, 3, 224, 224)):
        """Load model and trace with FX"""
        print("=" * 80)
        print(f"LOADING MODEL: {model_name}")
        print("=" * 80)

        if model_name not in self.SUPPORTED_MODELS:
            print(f"Error: Model '{model_name}' not supported")
            print(f"Supported models: {', '.join(sorted(self.SUPPORTED_MODELS.keys()))}")
            sys.exit(1)

        print(f"Loading {model_name}...")
        model = self.SUPPORTED_MODELS[model_name](weights=None)
        model.eval()
        self.model_name = model_name

        print(f"Tracing with PyTorch FX...")
        input_tensor = torch.randn(*input_shape)
        self.fx_graph = symbolic_trace(model)

        print("Propagating shapes...")
        shape_prop = ShapeProp(self.fx_graph)
        shape_prop.propagate(input_tensor)

        print("Partitioning graph...")
        self.report = self.partitioner.partition(self.fx_graph)

        total_nodes = len(list(self.fx_graph.graph.nodes))
        print(f"Created {self.report.total_subgraphs} subgraphs from {total_nodes} FX nodes")
        print()

        return self.fx_graph, self.report

    def determine_range(self, args) -> Tuple[Optional[int], Optional[int]]:
        """Determine start/end range based on arguments"""
        total_nodes = len(list(self.fx_graph.graph.nodes))

        # Priority 1: --around with --context
        if args.around is not None:
            context = args.context if args.context is not None else 10
            start = max(0, args.around - context)
            end = min(total_nodes, args.around + context + 1)
            print(f"Showing nodes around #{args.around} (context: ±{context} nodes)")
            print(f"Range: {start} to {end-1} (total: {end-start} nodes)")
            return start, end

        # Priority 2: --start and/or --end
        if args.start is not None or args.end is not None:
            start = args.start if args.start is not None else 0
            end = args.end if args.end is not None else total_nodes
            start = max(0, start)
            end = min(total_nodes, end)
            print(f"Showing nodes {start} to {end-1} (total: {end-start} nodes)")
            return start, end

        # Priority 3: --max-nodes
        if args.max_nodes is not None:
            start = 0
            end = min(args.max_nodes, total_nodes)
            print(f"Showing first {end} nodes")
            return start, end

        # Default: show all nodes
        print(f"Showing all {total_nodes} nodes")
        return None, None

    def visualize(self, args):
        """Generate and display visualization"""
        print()
        print("=" * 80)
        print("GENERATING VISUALIZATION")
        print("=" * 80)
        print()

        # Determine range
        start, end = self.determine_range(args)

        # Generate visualization
        if start is not None and end is not None:
            # Calculate max_nodes for backward compatibility with visualize_partitioning
            max_nodes = end - start
            # Note: GraphPartitioner.visualize_partitioning doesn't support start offset yet
            # For now, we'll use max_nodes approach
            visualization = self.partitioner.visualize_partitioning(
                self.fx_graph,
                max_nodes=end
            )
            # TODO: Update visualize_partitioning to support start/end range
            if start > 0:
                print(f"Note: Displaying from beginning to node {end-1}")
                print(f"      (Range selection starting at node {start} not yet fully supported)")
                print()
        else:
            visualization = self.partitioner.visualize_partitioning(self.fx_graph)

        # Display or save
        if args.output:
            with open(args.output, 'w') as f:
                f.write(visualization)
            print(f"Visualization saved to: {args.output}")
            print(f"View with: cat {args.output}")
        else:
            print(visualization)

    def show_tips(self, args):
        """Display usage tips"""
        print()
        print("=" * 80)
        print("TIPS")
        print("=" * 80)
        print()
        print("Range Selection:")
        print(f"  # Show nodes 20-50")
        print(f"  python cli/graph_explorer.py --model {self.model_name} --start 20 --end 50")
        print()
        print(f"  # Investigate around node 35 (±10 context)")
        print(f"  python cli/graph_explorer.py --model {self.model_name} --around 35 --context 10")
        print()
        print(f"  # Show first 100 nodes")
        print(f"  python cli/graph_explorer.py --model {self.model_name} --max-nodes 100")
        print()
        print("Output to File:")
        print(f"  python cli/graph_explorer.py --model {self.model_name} --output viz.txt")
        print()
        print("Different Models:")
        print(f"  python cli/graph_explorer.py --model mobilenet_v2")
        print(f"  python cli/graph_explorer.py --model vit_b_16")
        print()


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description='Visualize FX graph partitioning side-by-side',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize all nodes
  %(prog)s --model resnet18

  # Visualize first 50 nodes
  %(prog)s --model resnet18 --max-nodes 50

  # Visualize specific range
  %(prog)s --model resnet18 --start 20 --end 50

  # Investigate around node 35
  %(prog)s --model resnet18 --around 35 --context 10

  # Save to file
  %(prog)s --model mobilenet_v2 --output viz.txt
        """
    )

    # Model selection
    parser.add_argument('--model', type=str, default=None,
                        help='Model name (required)')
    parser.add_argument('--input-shape', type=str, default='1,3,224,224',
                        help='Input tensor shape as comma-separated values (default: 1,3,224,224)')

    # Range selection (mutually exclusive groups)
    range_group = parser.add_argument_group('range selection',
                                            'Choose one method to select node range')

    # Method 1: Explicit start/end
    range_group.add_argument('--start', type=int, default=None,
                            help='Start node index (0-based, inclusive)')
    range_group.add_argument('--end', type=int, default=None,
                            help='End node index (exclusive)')

    # Method 2: Context around a node
    range_group.add_argument('--around', type=int, default=None,
                            help='Center node for context view')
    range_group.add_argument('--context', type=int, default=10,
                            help='Number of nodes before/after center (default: 10)')

    # Method 3: Simple max-nodes (backward compatible)
    range_group.add_argument('--max-nodes', '-n', type=int, default=None,
                            help='Maximum nodes to display from start')

    # Output options
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Save visualization to file')

    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_args()

    # Level 1: No model specified → show model list
    if args.model is None:
        GraphExplorerCLI.show_model_list()
        sys.exit(1)

    # Parse input shape
    try:
        input_shape = tuple(map(int, args.input_shape.split(',')))
    except ValueError:
        print(f"Error: Invalid input shape '{args.input_shape}'")
        print("Expected format: 1,3,224,224")
        sys.exit(1)

    # Validate mutually exclusive range options
    range_methods = sum([
        args.around is not None,
        args.start is not None or args.end is not None,
        args.max_nodes is not None
    ])
    if range_methods > 1:
        print("Error: Cannot use multiple range selection methods simultaneously")
        print("Choose one: --start/--end, --around/--context, or --max-nodes")
        sys.exit(1)

    # Create CLI and load model
    cli = GraphExplorerCLI()
    cli.load_and_trace_model(args.model, input_shape)

    # Level 2: Model only (no range) → show summary
    # Level 3: Model + range → show visualization
    if range_methods == 0 and args.output is None:
        # Summary mode (no visualization range specified, not saving to file)
        cli.show_summary()
    else:
        # Visualization mode
        cli.visualize(args)
        cli.show_tips(args)


if __name__ == "__main__":
    main()
