#!/usr/bin/env python
"""
Model Comparison Script
=======================

Compare multiple models side-by-side to understand their characteristics.

Usage:
    python examples/compare_models.py
    python examples/compare_models.py --models resnet18 mobilenet_v2 efficientnet_b0
"""

import torch
import torchvision.models as models
from torch.fx import symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp
import argparse

from graphs.transform.partitioning import GraphPartitioner
from graphs.analysis.concurrency import ConcurrencyAnalyzer


# Model definitions
AVAILABLE_MODELS = {
    'resnet18': lambda: models.resnet18(weights=None),
    'resnet50': lambda: models.resnet50(weights=None),
    'resnet101': lambda: models.resnet101(weights=None),
    'mobilenet_v2': lambda: models.mobilenet_v2(weights=None),
    'mobilenet_v3_small': lambda: models.mobilenet_v3_small(weights=None),
    'mobilenet_v3_large': lambda: models.mobilenet_v3_large(weights=None),
    'efficientnet_b0': lambda: models.efficientnet_b0(weights=None),
    'efficientnet_b2': lambda: models.efficientnet_b2(weights=None),
    'vgg16': lambda: models.vgg16(weights=None),
    'densenet121': lambda: models.densenet121(weights=None),
    'shufflenet_v2_x1_0': lambda: models.shufflenet_v2_x1_0(weights=None),
    'deeplabv3_resnet50': lambda: models.segmentation.deeplabv3_resnet50(weights=None),
    'fcn_resnet50': lambda: models.segmentation.fcn_resnet50(weights=None),
    'squeezenet1_0': lambda: models.squeezenet1_0(weights=None),
    'vit_b_16': lambda: models.vit_b_16(weights=None),
    'vit_l_16': lambda: models.vit_l_16(weights=None),
    'vit_b_32': lambda: models.vit_b_32(weights=None),
    'vit_l_32': lambda: models.vit_l_32(weights=None),
}


def analyze_model(model_name, model_fn, input_shape=(1, 3, 224, 224), verbose=False):
    """Analyze a single model and return metrics"""

    if verbose:
        print(f"\nAnalyzing {model_name}...")

    try:
        # Load and trace
        model = model_fn()
        model.eval()

        fx_graph = symbolic_trace(model)
        input_tensor = torch.randn(*input_shape)
        ShapeProp(fx_graph).propagate(input_tensor)

        # Partition
        partitioner = GraphPartitioner()
        report = partitioner.partition(fx_graph)

        # Analyze concurrency
        analyzer = ConcurrencyAnalyzer()
        concurrency = analyzer.analyze(report)

        # Compute metrics
        total_weights = sum(sg.total_weight_bytes for sg in report.subgraphs) / 1e6
        max_activation = max(sg.total_input_bytes + sg.total_output_bytes
                            for sg in report.subgraphs) / 1e6

        # Count operation types
        conv_count = report.operation_type_counts.get('conv2d', 0)
        depthwise_count = report.operation_type_counts.get('conv2d_depthwise', 0)
        pointwise_count = report.operation_type_counts.get('conv2d_pointwise', 0)

        # Bottleneck distribution
        compute_bound = sum(1 for sg in report.subgraphs
                           if sg.recommended_bottleneck.value == 'compute_bound')
        memory_bound = sum(1 for sg in report.subgraphs
                          if sg.recommended_bottleneck.value == 'memory_bound')

        return {
            'name': model_name,
            'success': True,
            'flops_gflops': report.total_flops / 1e9,
            'macs_gmacs': report.total_macs / 1e9,
            'subgraphs': report.total_subgraphs,
            'arithmetic_intensity': report.average_arithmetic_intensity,
            'weights_mb': total_weights,
            'max_activation_mb': max_activation,
            'stages': concurrency.num_stages,
            'max_parallel': concurrency.max_parallel_ops_per_stage,
            'critical_path_length': concurrency.critical_path_length,
            'critical_path_gflops': concurrency.critical_path_flops / 1e9,
            'conv_count': conv_count,
            'depthwise_count': depthwise_count,
            'pointwise_count': pointwise_count,
            'compute_bound_pct': compute_bound / report.total_subgraphs * 100,
            'memory_bound_pct': memory_bound / report.total_subgraphs * 100,
            'avg_threads': sum(sg.parallelism.total_threads for sg in report.subgraphs
                              if sg.parallelism) / max(1, report.total_subgraphs),
        }

    except Exception as e:
        if verbose:
            print(f"  Failed: {e}")
        return {
            'name': model_name,
            'success': False,
            'error': str(e)
        }


def print_comparison_table(results):
    """Print a formatted comparison table"""

    successful = [r for r in results if r['success']]

    if not successful:
        print("No successful analyses to compare.")
        return

    print("\n" + "=" * 120)
    print("MODEL COMPARISON")
    print("=" * 120)

    # Table 1: Computation
    print("\nComputation:")
    print("-" * 120)
    print(f"{'Model':<20} {'FLOPs (G)':<12} {'MACs (G)':<12} {'Subgraphs':<12} {'Conv':<8} {'Depthwise':<12} {'Pointwise':<12}")
    print("-" * 120)

    for r in successful:
        print(f"{r['name']:<20} {r['flops_gflops']:>11.2f} {r['macs_gmacs']:>11.2f} "
              f"{r['subgraphs']:>11} {r['conv_count']:>7} {r['depthwise_count']:>11} {r['pointwise_count']:>11}")

    # Table 2: Memory
    print("\nMemory:")
    print("-" * 120)
    print(f"{'Model':<20} {'Weights (MB)':<15} {'Max Act (MB)':<15} {'Total Mem (MB)':<15} {'Arith Int':<15}")
    print("-" * 120)

    for r in successful:
        total_mem = r['weights_mb'] + r['max_activation_mb']
        print(f"{r['name']:<20} {r['weights_mb']:>14.1f} {r['max_activation_mb']:>14.1f} "
              f"{total_mem:>14.1f} {r['arithmetic_intensity']:>14.1f}")

    # Table 3: Concurrency
    print("\nConcurrency:")
    print("-" * 120)
    print(f"{'Model':<20} {'Stages':<10} {'Max Parallel':<15} {'Critical Path':<20} {'Crit FLOPs (G)':<15} {'Avg Threads':<15}")
    print("-" * 120)

    for r in successful:
        print(f"{r['name']:<20} {r['stages']:>9} {r['max_parallel']:>14} "
              f"{r['critical_path_length']:>19} {r['critical_path_gflops']:>14.2f} {r['avg_threads']:>14,.0f}")

    # Table 4: Bottleneck
    print("\nBottleneck Distribution:")
    print("-" * 120)
    print(f"{'Model':<20} {'Compute-Bound %':<18} {'Memory-Bound %':<18} {'Characterization':<30}")
    print("-" * 120)

    for r in successful:
        # Characterize the model
        if r['arithmetic_intensity'] > 40:
            char = "Highly compute-intensive"
        elif r['arithmetic_intensity'] > 20:
            char = "Compute-intensive"
        elif r['arithmetic_intensity'] > 10:
            char = "Balanced"
        else:
            char = "Memory-intensive"

        print(f"{r['name']:<20} {r['compute_bound_pct']:>17.1f} {r['memory_bound_pct']:>17.1f} {char:<30}")

    # Analysis summary
    print("\n" + "=" * 120)
    print("ANALYSIS SUMMARY")
    print("=" * 120)

    # Find extremes
    most_flops = max(successful, key=lambda r: r['flops_gflops'])
    least_flops = min(successful, key=lambda r: r['flops_gflops'])
    most_parallel = max(successful, key=lambda r: r['max_parallel'])
    most_compute = max(successful, key=lambda r: r['arithmetic_intensity'])
    most_memory = min(successful, key=lambda r: r['arithmetic_intensity'])

    print(f"\nMost Compute-Intensive:   {most_flops['name']} ({most_flops['flops_gflops']:.2f} GFLOPs)")
    print(f"Most Efficient:           {least_flops['name']} ({least_flops['flops_gflops']:.2f} GFLOPs)")
    print(f"Most Graph Parallelism:   {most_parallel['name']} ({most_parallel['max_parallel']} parallel ops)")
    print(f"Most Compute-Bound:       {most_compute['name']} (AI={most_compute['arithmetic_intensity']:.1f})")
    print(f"Most Memory-Bound:        {most_memory['name']} (AI={most_memory['arithmetic_intensity']:.1f})")

    # Hardware recommendations
    print("\nHardware Recommendations:")
    print("-" * 120)

    for r in successful:
        if r['arithmetic_intensity'] > 40:
            hw_rec = "High-FLOPS GPU (e.g., H100, A100)"
        elif r['arithmetic_intensity'] > 20:
            hw_rec = "Balanced GPU (e.g., V100, A10)"
        elif r['arithmetic_intensity'] > 10:
            hw_rec = "Memory-optimized or balanced"
        else:
            hw_rec = "High-bandwidth memory critical (e.g., edge devices, mobile)"

        # Batching recommendation
        if r['max_parallel'] < 16:
            batch_rec = f"needs batch≥{128 // r['max_parallel']} for GPU utilization"
        else:
            batch_rec = "good single-sample parallelism"

        print(f"{r['name']:<20} → {hw_rec:<50} ({batch_rec})")


def main():
    parser = argparse.ArgumentParser(description='Compare neural network models')
    parser.add_argument('--models', nargs='+',
                       help='Models to compare (example: resnet18, mobilenet_v2, efficientnet_b0)',
                       default=['squeezenet1_0', 'shufflenet_v2_x1_0', 'mobilenet_v3_small', 'mobilenet_v2', 'mobilenet_v3_large', 'efficientnet_b0', 'efficientnet_b2', 'densenet121', 'resnet18', 'fcn_resnet50', 'deeplabv3_resnet50', 'vit_b_16', 'vit_b_32', 'vgg16', 'vit_l_16', 'vit_l_32'])
    parser.add_argument('--verbose', action='store_true',
                       help='Show detailed progress')

    args = parser.parse_args()

    # Validate model names
    invalid_models = [m for m in args.models if m not in AVAILABLE_MODELS]
    if invalid_models:
        print(f"Error: Unknown models: {invalid_models}")
        print(f"Available models: {list(AVAILABLE_MODELS.keys())}")
        return

    print("=" * 120)
    print("Model Comparison Tool")
    print("=" * 120)
    print(f"\nComparing {len(args.models)} models: {', '.join(args.models)}")

    # Analyze each model
    results = []
    for model_name in args.models:
        model_fn = AVAILABLE_MODELS[model_name]

        # Special case: EfficientNet-B2 uses different input size
        input_shape = (1, 3, 260, 260) if model_name == 'efficientnet_b2' else (1, 3, 224, 224)

        result = analyze_model(model_name, model_fn, input_shape, verbose=args.verbose)
        results.append(result)

    # Print comparison
    print_comparison_table(results)

    # Print failed analyses
    failed = [r for r in results if not r['success']]
    if failed:
        print("\n" + "=" * 120)
        print("FAILED ANALYSES")
        print("=" * 120)
        for r in failed:
            print(f"{r['name']}: {r['error']}")


if __name__ == "__main__":
    main()
