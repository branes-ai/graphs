#!/usr/bin/env python
"""
Graph Partitioning CLI
======================

Command-line tool to apply different partitioning strategies to FX graphs,
quantify results, and visualize the partitioning.

Usage:
    # Compare all strategies on ResNet-18
    python cli/partitioner.py --model resnet18 --strategy all

    # Test fusion strategy with visualization
    python cli/partitioner.py --model mobilenet_v2 --strategy fusion --visualize

    # Compare unfused vs fusion
    python cli/partitioner.py --model efficientnet_b0 --strategy all --compare

Command-line Options:
    --model: Choose model (resnet18, mobilenet_v2, etc.)
    --strategy:    Select strategy (unfused, fusion, all)
    --compare:     Show side-by-side comparison
    --quantify:    Show detailed metrics
    --visualize:   Show graph visualization
    --max-nodes:   Control visualization length
    --input-shape: Customize input tensor dimensions

"""


import torch
import torchvision.models as models
from torch.fx import symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp
import sys
import argparse
from typing import Dict, Any
sys.path.insert(0, 'src')

from graphs.characterize.graph_partitioner import GraphPartitioner
from graphs.characterize.fusion_partitioner import FusionBasedPartitioner


class PartitionCLI:
    """Command-line interface for graph partitioning"""

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

    def visualize_strategy(self, strategy: str, max_nodes: int = 20):
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

        # Check if partitioner has visualization method
        if hasattr(partitioner, 'visualize_partitioning'):
            viz = partitioner.visualize_partitioning(self.fx_graph, max_nodes=max_nodes)
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
            for strategy in strategies:
                if strategy in self.results:
                    self.visualize_strategy(strategy, max_nodes=args.max_nodes)

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
            print(f"\nTo see full visualization, run:")
            print(f"  python cli/partitioner.py --model {args.model} "
                  f"--strategy {strategies[0]} --visualize --max-nodes 999")

        return 0


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description='Graph Partitioning CLI - Apply and compare partitioning strategies',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare all strategies on ResNet-18
  python cli/partitioner.py --model resnet18 --strategy all --compare

  # Test fusion with visualization
  python cli/partitioner.py --model mobilenet_v2 --strategy fusion --visualize

  # Analyze fusion balance
  python cli/partitioner.py --model resnet50 --strategy fusion --analyze-balance

  # Full analysis with visualization and balance
  python cli/partitioner.py --model efficientnet_b0 --strategy fusion --visualize --analyze-balance
        """
    )

    parser.add_argument('--model', type=str, default='resnet18',
                       choices=list(PartitionCLI.SUPPORTED_MODELS.keys()),
                       help='Model to partition')

    parser.add_argument('--strategy', type=str, default='all',
                       choices=list(PartitionCLI.STRATEGIES.keys()) + ['all'],
                       help='Partitioning strategy to apply')

    parser.add_argument('--compare', action='store_true',
                       help='Compare strategies (requires --strategy all or multiple runs)')

    parser.add_argument('--quantify', action='store_true',
                       help='Show detailed metrics for each strategy')

    parser.add_argument('--visualize', action='store_true',
                       help='Show side-by-side visualization of partitioning')

    parser.add_argument('--analyze-balance', action='store_true',
                       help='Analyze fusion balance and quality (requires --strategy fusion)')

    parser.add_argument('--max-nodes', type=int, default=20,
                       help='Maximum nodes to show in visualization')

    parser.add_argument('--input-shape', type=int, nargs=4, default=[1, 3, 224, 224],
                       metavar=('B', 'C', 'H', 'W'),
                       help='Input tensor shape (default: 1 3 224 224)')

    return parser.parse_args()


def main():
    args = parse_args()

    cli = PartitionCLI()
    return cli.run(args)


if __name__ == "__main__":
    sys.exit(main())
