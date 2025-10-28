#!/usr/bin/env python
"""
Partition Analyzer CLI
======================

Command-line tool to analyze and compare different partitioning strategies for FX graphs.
Compares unfused (baseline) vs fusion strategies, quantifying benefits of operator fusion.

Usage:
    # Compare all strategies on ResNet-18
    python cli/partition_analyzer.py --model resnet18 --strategy all

    # Test fusion strategy with visualization
    python cli/partition_analyzer.py --model mobilenet_v2 --strategy fusion --visualize

    # Compare unfused vs fusion
    python cli/partition_analyzer.py --model efficientnet_b0 --strategy all --compare

    # Visualize specific node range
    python cli/partition_analyzer.py --model resnet18 --strategy fusion --visualize --start 5 --end 20

    # Investigate around specific node
    python cli/partition_analyzer.py --model resnet18 --strategy fusion --visualize --around 10 --context 5

Command-line Options:
    --model:       Choose model (resnet18, mobilenet_v2, etc.)
    --strategy:    Select strategy (unfused, fusion, all)
    --compare:     Show side-by-side comparison
    --quantify:    Show detailed metrics
    --visualize:   Show graph visualization

    Range Selection (for visualization):
    --start:       Start node (1-based, inclusive)
    --end:         End node (1-based, inclusive)
    --around:      Center node for context view
    --context:     Nodes before/after center (default: 10)
    --max-nodes:   Maximum nodes from start (default: 20)

    --input-shape: Customize input tensor dimensions

"""

import torch
import torchvision.models as models
from torch.fx import symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp
import sys
import argparse
from typing import Dict, Any

from graphs.transform.partitioning.graph_partitioner import GraphPartitioner
from graphs.transform.partitioning.fusion_partitioner import FusionBasedPartitioner


class PartitionAnalyzerCLI:
    """Command-line interface for analyzing partitioning strategies"""

    SUPPORTED_MODELS = {
        'resnet18': models.resnet18,
        'resnet34': models.resnet34,
        'resnet50': models.resnet50,
        'mobilenet_v2': models.mobilenet_v2,
        'efficientnet_b0': models.efficientnet_b0,
        'efficientnet_b1': models.efficientnet_b1,
        'efficientnet_b2': models.efficientnet_b2,
        'efficientnet_b3': models.efficientnet_b3,
        'efficientnet_b4': models.efficientnet_b4,
        'efficientnet_b5': models.efficientnet_b5,
        'efficientnet_b6': models.efficientnet_b6,
        'efficientnet_b7': models.efficientnet_b7,
        'vit_b_16': models.vit_b_16,
        'vit_b_32': models.vit_b_32,
        'vit_l_16': models.vit_l_16,
        'swin_t': models.swin_t,
        'swin_s': models.swin_s,
        'swin_b': models.swin_b,
    }

    STRATEGIES = {
        'unfused': 'One subgraph per operator (baseline)',
        'fusion': 'Aggregate operators to minimize data movement',
    }

    def __init__(self):
        self.fx_graph = None
        self.model_name = None
        self.results = {}

    def load_and_trace_model(self, model_name: str, input_shape=(1, 3, 224, 224)):
        """Load model and trace with FX"""
        print("=" * 80)
        print(f"Loading and Tracing: {model_name}")
        print("=" * 80)

        if model_name not in self.SUPPORTED_MODELS:
            print(f"Error: Unknown model '{model_name}'")
            print(f"Supported models: {', '.join(self.SUPPORTED_MODELS.keys())}")
            return False

        print(f"\n[1/3] Loading {model_name}...")
        model = self.SUPPORTED_MODELS[model_name](weights=None)
        model.eval()

        print("[2/3] Tracing with PyTorch FX...")
        input_tensor = torch.randn(*input_shape)
        try:
            self.fx_graph = symbolic_trace(model)
        except Exception as e:
            print(f"Error tracing model: {e}")
            return False

        print("[3/3] Propagating shapes...")
        ShapeProp(self.fx_graph).propagate(input_tensor)

        self.model_name = model_name
        print(f"\nSuccess: {len(list(self.fx_graph.graph.nodes))} FX nodes")
        return True

    def determine_range(self, args) -> tuple:
        """Determine start/end range based on arguments

        Note: User-provided node numbers are 1-based (display numbering).
        This method converts them to 0-based array indices for slicing.
        """
        total_nodes = len(list(self.fx_graph.graph.nodes))

        # Priority 1: --around with --context
        if args.around is not None:
            context = args.context if args.context is not None else 10
            # Convert 1-based display node number to 0-based index
            center_idx = args.around - 1
            start = max(0, center_idx - context)
            end = min(total_nodes, center_idx + context + 1)
            print(f"Showing nodes around #{args.around} (context: ±{context} nodes)")
            print(f"Range: nodes {start+1} to {end} (total: {end-start} nodes)")
            return start, end

        # Priority 2: --start and/or --end
        if args.start is not None or args.end is not None:
            # Convert 1-based display numbers to 0-based indices
            # start: subtract 1 (node 5 -> index 4)
            # end: keep as-is (node 10 -> slice index 10, since slicing is exclusive on end)
            start = (args.start - 1) if args.start is not None else 0
            end = args.end if args.end is not None else total_nodes
            start = max(0, start)
            end = min(total_nodes, end)
            print(f"Showing nodes {start+1} to {end} (total: {end-start} nodes)")
            return start, end

        # Priority 3: --max-nodes (backward compatible)
        if args.max_nodes is not None:
            start = 0
            end = min(args.max_nodes, total_nodes)
            print(f"Showing first {end} nodes")
            return start, end

        # Default: first 20 nodes
        print(f"Showing first 20 nodes (default)")
        return 0, min(20, total_nodes)

    def apply_strategy(self, strategy: str) -> Dict[str, Any]:
        """Apply a partitioning strategy and return results"""
        print(f"\nApplying '{strategy}' partitioning strategy...")

        if strategy == 'unfused':
            partitioner = GraphPartitioner()
            report = partitioner.partition(self.fx_graph)

            return {
                'strategy': strategy,
                'partitioner': partitioner,
                'report': report,
                'num_subgraphs': report.total_subgraphs,
                'total_flops': report.total_flops,
                'total_memory': report.total_memory_traffic,
                'avg_ai': report.average_arithmetic_intensity,
            }

        elif strategy == 'fusion':
            partitioner = FusionBasedPartitioner()
            report = partitioner.partition(self.fx_graph)

            # Calculate average arithmetic intensity
            avg_ai = 0.0
            if report.fused_subgraphs:
                ai_sum = sum(sg.arithmetic_intensity for sg in report.fused_subgraphs)
                avg_ai = ai_sum / len(report.fused_subgraphs)

            return {
                'strategy': strategy,
                'partitioner': partitioner,
                'report': report,
                'num_subgraphs': report.total_subgraphs,
                'total_flops': report.total_flops,
                'total_memory': report.total_memory_traffic_fused,
                'avg_ai': avg_ai,
                'avg_fusion_size': report.avg_fusion_size,
                'data_movement_reduction': report.data_movement_reduction,
            }

        else:
            print(f"Error: Unknown strategy '{strategy}'")
            return None

    def quantify_results(self, strategy: str):
        """Print quantitative metrics for a strategy"""
        if strategy not in self.results:
            print(f"Error: No results for strategy '{strategy}'")
            return

        result = self.results[strategy]
        report = result['report']

        print("\n" + "=" * 80)
        print(f"METRICS: {strategy.upper()} STRATEGY")
        print("=" * 80)

        print(f"\nSubgraphs: {result['num_subgraphs']}")
        print(f"Total FLOPs: {result['total_flops'] / 1e9:.2f} G")
        print(f"Memory Traffic: {result['total_memory'] / 1e6:.2f} MB")
        print(f"Arithmetic Intensity: {result['avg_ai']:.2f} FLOPs/byte")

        if strategy == 'fusion':
            print(f"\nFusion Stats:")
            print(f"  Avg ops per subgraph: {result['avg_fusion_size']:.1f}")
            print(f"  Data movement reduction: {result['data_movement_reduction'] * 100:.1f}%")
            print(f"\nTop fusion patterns:")
            for pattern, count in sorted(report.fusion_patterns.items(),
                                        key=lambda x: x[1], reverse=True)[:5]:
                print(f"  {pattern}: {count}")

        # Bottleneck distribution
        if hasattr(report, 'bottleneck_distribution'):
            print(f"\nBottleneck Distribution:")
            for bottleneck, count in sorted(report.bottleneck_distribution.items(),
                                           key=lambda x: x[1], reverse=True):
                pct = count / result['num_subgraphs'] * 100
                print(f"  {bottleneck}: {count} ({pct:.0f}%)")

        # Partition reasons (if available)
        if hasattr(report, 'partition_reason_distribution'):
            print(f"\nPartition Reasons:")
            for reason, count in sorted(report.partition_reason_distribution.items(),
                                       key=lambda x: x[1], reverse=True):
                pct = count / result['num_subgraphs'] * 100
                print(f"  {reason}: {count} ({pct:.0f}%)")

    def compare_strategies(self):
        """Compare all applied strategies"""
        if len(self.results) < 2:
            print("\nNeed at least 2 strategies to compare")
            return

        print("\n" + "=" * 80)
        print("STRATEGY COMPARISON")
        print("=" * 80)

        # Create comparison table
        strategies = list(self.results.keys())

        print(f"\nModel: {self.model_name}")
        print(f"FX Nodes: {len(list(self.fx_graph.graph.nodes))}")
        print()

        # Header
        header = f"{'Metric':<30}"
        for strategy in strategies:
            header += f"{strategy.upper():<20}"
        print(header)
        print("-" * (30 + 20 * len(strategies)))

        # Subgraphs
        row = f"{'Subgraphs':<30}"
        for strategy in strategies:
            row += f"{self.results[strategy]['num_subgraphs']:<20}"
        print(row)

        # FLOPs
        row = f"{'FLOPs (G)':<30}"
        for strategy in strategies:
            flops = self.results[strategy]['total_flops'] / 1e9
            row += f"{flops:<20.2f}"
        print(row)

        # Memory
        row = f"{'Memory Traffic (MB)':<30}"
        for strategy in strategies:
            mem = self.results[strategy]['total_memory'] / 1e6
            row += f"{mem:<20.2f}"
        print(row)

        # Arithmetic Intensity
        row = f"{'Arithmetic Intensity':<30}"
        for strategy in strategies:
            ai = self.results[strategy]['avg_ai']
            row += f"{ai:<20.2f}"
        print(row)

        # Fusion-specific metrics
        if 'fusion' in self.results:
            print()
            row = f"{'Avg Fusion Size':<30}"
            for strategy in strategies:
                if strategy == 'fusion':
                    size = self.results[strategy]['avg_fusion_size']
                    row += f"{size:<20.1f}"
                else:
                    row += f"{'1.0':<20}"
            print(row)

            row = f"{'Data Movement Reduction':<30}"
            for strategy in strategies:
                if strategy == 'fusion':
                    reduction = self.results[strategy]['data_movement_reduction'] * 100
                    row += f"{reduction:<20.1f}%"
                else:
                    row += f"{'0.0%':<20}"
            print(row)

        # Efficiency gains
        if 'unfused' in self.results and 'fusion' in self.results:
            print("\n" + "=" * 80)
            print("FUSION EFFICIENCY GAINS")
            print("=" * 80)

            unfused = self.results['unfused']
            fused = self.results['fusion']

            kernel_reduction = unfused['num_subgraphs'] / max(1, fused['num_subgraphs'])
            memory_reduction = (unfused['total_memory'] - fused['total_memory']) / 1e6

            print(f"\nKernel Launches: {kernel_reduction:.1f}x reduction "
                  f"({unfused['num_subgraphs']} -> {fused['num_subgraphs']})")
            print(f"Memory Traffic: {memory_reduction:.2f} MB saved "
                  f"({fused['data_movement_reduction'] * 100:.1f}% reduction)")
            print(f"Arithmetic Intensity: {fused['avg_ai'] / max(0.01, unfused['avg_ai']):.2f}x improvement")

    def visualize_strategy(self, strategy: str, args, start: int = None, end: int = None, use_color: bool = False, no_color: bool = False):
        """Visualize partitioning for a strategy"""
        if strategy not in self.results:
            print(f"Error: No results for strategy '{strategy}'")
            return

        result = self.results[strategy]
        partitioner = result['partitioner']

        print("\n" + "=" * 80)
        print(f"VISUALIZATION: {strategy.upper()} STRATEGY")
        print("=" * 80)
        print()

        # Choose visualization method
        if use_color and strategy == 'fusion' and hasattr(partitioner, 'visualize_partitioning_colored'):
            # Use color-coded visualization for fusion
            color_enabled = not no_color
            viz = partitioner.visualize_partitioning_colored(self.fx_graph,
                                                            start=start,
                                                            end=end,
                                                            use_color=color_enabled if no_color else None)
            print(viz)
        elif hasattr(partitioner, 'visualize_partitioning'):
            # Use standard visualization
            viz = partitioner.visualize_partitioning(self.fx_graph, start=start, end=end)
            print(viz)
        else:
            print(f"Error: Visualization not implemented for '{strategy}' strategy")
            print("Note: Both 'unfused' and 'fusion' strategies support visualization")

    def run(self, args):
        """Main execution flow"""
        # Load and trace model
        if not self.load_and_trace_model(args.model, args.input_shape):
            return 1

        # Determine which strategies to run
        if args.strategy == 'all':
            strategies = list(self.STRATEGIES.keys())
        else:
            strategies = [args.strategy]

        # Apply each strategy
        print("\n" + "=" * 80)
        print("APPLYING PARTITIONING STRATEGIES")
        print("=" * 80)

        for strategy in strategies:
            result = self.apply_strategy(strategy)
            if result:
                self.results[strategy] = result
                print(f"  {strategy}: {result['num_subgraphs']} subgraphs created")

        # Quantify results
        if args.quantify or not args.compare:
            for strategy in strategies:
                self.quantify_results(strategy)

        # Compare strategies
        if args.compare and len(self.results) > 1:
            self.compare_strategies()

        # Visualize
        if args.visualize:
            # Determine range for visualization
            start, end = self.determine_range(args)
            for strategy in strategies:
                if strategy in self.results:
                    self.visualize_strategy(strategy,
                                          args=args,
                                          start=start,
                                          end=end,
                                          use_color=args.color,
                                          no_color=args.no_color)

        # Export to DOT/Graphviz
        if args.export_dot:
            if 'fusion' in self.results:
                partitioner = self.results['fusion']['partitioner']
                if hasattr(partitioner, 'export_to_graphviz'):
                    print(f"\nExporting fusion graph to {args.export_dot}...")
                    partitioner.export_to_graphviz(self.fx_graph, args.export_dot)
                else:
                    print("\nDOT export not available for this partitioner")
            else:
                print("\nDOT export requires --strategy fusion or --strategy all")

        # Balance analysis (fusion strategy only)
        if args.analyze_balance:
            if 'fusion' in self.results:
                partitioner = self.results['fusion']['partitioner']
                if hasattr(partitioner, 'analyze_balance'):
                    print()
                    analysis = partitioner.analyze_balance()
                    print(analysis)
                else:
                    print("\nBalance analysis not available for fusion partitioner")
            else:
                print("\nBalance analysis requires --strategy fusion or --strategy all")

        # Summary
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"\nModel: {self.model_name}")
        print(f"Strategies tested: {', '.join(self.results.keys())}")
        print(f"FX graph size: {len(list(self.fx_graph.graph.nodes))} nodes")

        if args.visualize:
            print(f"\nTo see more nodes in visualization:")
            print(f"  # Show first 50 nodes")
            print(f"  python cli/partition_analyzer.py --model {args.model} "
                  f"--strategy {strategies[0]} --visualize --max-nodes 50")
            print(f"  # Show specific range (nodes 20-50)")
            print(f"  python cli/partition_analyzer.py --model {args.model} "
                  f"--strategy {strategies[0]} --visualize --start 20 --end 50")
            print(f"  # Investigate around node 35")
            print(f"  python cli/partition_analyzer.py --model {args.model} "
                  f"--strategy {strategies[0]} --visualize --around 35 --context 10")

        return 0


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description='Graph Partitioning CLI - Apply and compare partitioning strategies',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare all strategies on ResNet-18
  python cli/partition_analyzer.py --model resnet18 --strategy all --compare

  # Test fusion with visualization (first 20 nodes)
  python cli/partition_analyzer.py --model mobilenet_v2 --strategy fusion --visualize

  # Visualize specific node range (nodes 5-20, inclusive)
  python cli/partition_analyzer.py --model resnet18 --strategy fusion --visualize --start 5 --end 20

  # Investigate around node 10 (±5 nodes context)
  python cli/partition_analyzer.py --model resnet18 --strategy fusion --visualize --around 10 --context 5

  # Analyze fusion balance
  python cli/partition_analyzer.py --model resnet50 --strategy fusion --analyze-balance

  # Full analysis with visualization and balance
  python cli/partition_analyzer.py --model efficientnet_b0 --strategy fusion --visualize --analyze-balance
        """
    )

    parser.add_argument('--model', type=str, default='resnet18',
                       choices=list(PartitionAnalyzerCLI.SUPPORTED_MODELS.keys()),
                       help='Model to partition')

    parser.add_argument('--strategy', type=str, default='all',
                       choices=list(PartitionAnalyzerCLI.STRATEGIES.keys()) + ['all'],
                       help='Partitioning strategy to apply')

    parser.add_argument('--compare', action='store_true',
                       help='Compare strategies (requires --strategy all or multiple runs)')

    parser.add_argument('--quantify', action='store_true',
                       help='Show detailed metrics for each strategy')

    parser.add_argument('--visualize', action='store_true',
                       help='Show side-by-side visualization of partitioning')

    parser.add_argument('--analyze-balance', action='store_true',
                       help='Analyze fusion balance and quality (requires --strategy fusion)')

    # Range selection (mutually exclusive groups)
    range_group = parser.add_argument_group('range selection',
                                            'Choose one method to select node range for visualization')

    # Method 1: Explicit start/end
    range_group.add_argument('--start', type=int, default=None,
                            help='Start node index (1-based, inclusive)')
    range_group.add_argument('--end', type=int, default=None,
                            help='End node index (1-based, inclusive)')

    # Method 2: Context around a node
    range_group.add_argument('--around', type=int, default=None,
                            help='Center node for context view (1-based)')
    range_group.add_argument('--context', type=int, default=10,
                            help='Number of nodes before/after center (default: 10)')

    # Method 3: Simple max-nodes (backward compatible)
    range_group.add_argument('--max-nodes', '-n', type=int, default=20,
                            help='Maximum nodes to display from start (default: 20)')

    parser.add_argument('--input-shape', type=int, nargs=4, default=[1, 3, 224, 224],
                       metavar=('B', 'C', 'H', 'W'),
                       help='Input tensor shape (default: 1 3 224 224)')

    parser.add_argument('--color', action='store_true',
                       help='Use color-coded visualization (fusion strategy only)')

    parser.add_argument('--no-color', action='store_true',
                       help='Disable colors (ASCII-only output)')

    parser.add_argument('--export-dot', type=str, metavar='FILE',
                       help='Export fusion graph to DOT/Graphviz format (e.g., --export-dot fusion.dot)')

    return parser.parse_args()


def main():
    args = parse_args()

    cli = PartitionAnalyzerCLI()
    return cli.run(args)


if __name__ == "__main__":
    sys.exit(main())
