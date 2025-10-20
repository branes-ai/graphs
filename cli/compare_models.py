#!/usr/bin/env python
"""
Model Comparison Tool
=====================

Compare multiple DNN models across key metrics:
- Model size (parameters, memory)
- Computational cost (FLOPs, MACs)
- Efficiency ratios (FLOPs/param, memory/FLOP)
- Architecture characteristics (depth, width)
- Arithmetic intensity distribution
"""

import torch
import torchvision.models as models
from torch.fx import symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp
import sys
import argparse
from typing import Dict, List
from dataclasses import dataclass
sys.path.insert(0, 'src')

from graphs.characterize.graph_partitioner import GraphPartitioner


@dataclass
class ModelMetrics:
    """Comprehensive metrics for a single model"""
    name: str

    # Size metrics
    total_params: int
    trainable_params: int
    model_size_mb: float

    # Compute metrics
    total_flops: int
    total_macs: int

    # Memory metrics
    total_memory: int
    input_memory: int
    output_memory: int
    weight_memory: int

    # Architecture metrics
    num_layers: int
    max_channels: int

    # Arithmetic intensity
    avg_ai: float
    compute_bound_ratio: float  # % of ops with AI > 50
    memory_bound_ratio: float   # % of ops with AI < 10

    # Efficiency ratios
    flops_per_param: float
    memory_per_flop: float
    params_per_mac: float


def profile_model(model_name: str, model_fn, input_shape=(1, 3, 224, 224)) -> ModelMetrics:
    """Profile a single model and extract all metrics"""

    # Load and trace model
    model = model_fn(weights=None)
    model.eval()
    input_tensor = torch.randn(*input_shape)

    fx_graph = symbolic_trace(model)
    ShapeProp(fx_graph).propagate(input_tensor)

    # Run partitioner
    partitioner = GraphPartitioner()
    report = partitioner.partition(fx_graph)

    # Size metrics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_mb = total_params * 4 / (1024 * 1024)  # Assuming float32

    # Compute metrics
    total_flops = report.total_flops
    total_macs = report.total_macs

    # Memory metrics
    total_input = sum(sg.total_input_bytes for sg in report.subgraphs)
    total_output = sum(sg.total_output_bytes for sg in report.subgraphs)
    total_weight = sum(sg.total_weight_bytes for sg in report.subgraphs)
    total_memory = total_input + total_output + total_weight

    # Architecture metrics
    num_layers = len(report.subgraphs)

    # Find max channels (approximate from parameter shapes)
    max_channels = 0
    for name, param in model.named_parameters():
        if 'weight' in name and len(param.shape) >= 2:
            max_channels = max(max_channels, max(param.shape[0], param.shape[1]))

    # Arithmetic intensity analysis
    avg_ai = report.average_arithmetic_intensity

    # Compute AI distribution
    compute_bound_count = sum(1 for sg in report.subgraphs if sg.arithmetic_intensity > 50)
    memory_bound_count = sum(1 for sg in report.subgraphs if sg.arithmetic_intensity < 10)
    total_count = len(report.subgraphs)

    compute_bound_ratio = (compute_bound_count / total_count * 100) if total_count > 0 else 0
    memory_bound_ratio = (memory_bound_count / total_count * 100) if total_count > 0 else 0

    # Efficiency ratios
    flops_per_param = total_flops / total_params if total_params > 0 else 0
    memory_per_flop = total_memory / total_flops if total_flops > 0 else 0
    params_per_mac = total_params / total_macs if total_macs > 0 else 0

    return ModelMetrics(
        name=model_name,
        total_params=total_params,
        trainable_params=trainable_params,
        model_size_mb=model_size_mb,
        total_flops=total_flops,
        total_macs=total_macs,
        total_memory=total_memory,
        input_memory=total_input,
        output_memory=total_output,
        weight_memory=total_weight,
        num_layers=num_layers,
        max_channels=max_channels,
        avg_ai=avg_ai,
        compute_bound_ratio=compute_bound_ratio,
        memory_bound_ratio=memory_bound_ratio,
        flops_per_param=flops_per_param,
        memory_per_flop=memory_per_flop,
        params_per_mac=params_per_mac,
    )


def format_number(num: float, unit: str = '') -> str:
    """Format large numbers with K/M/G/T suffixes"""
    if num >= 1e12:
        return f"{num / 1e12:.2f}T{unit}"
    elif num >= 1e9:
        return f"{num / 1e9:.2f}G{unit}"
    elif num >= 1e6:
        return f"{num / 1e6:.2f}M{unit}"
    elif num >= 1e3:
        return f"{num / 1e3:.2f}K{unit}"
    else:
        return f"{num:.2f}{unit}"


def print_comparison_table(metrics_list: List[ModelMetrics], sort_by: str = 'params'):
    """Print comparison table"""

    # Sort metrics
    sort_keys = {
        'params': lambda m: m.total_params,
        'flops': lambda m: m.total_flops,
        'macs': lambda m: m.total_macs,
        'memory': lambda m: m.total_memory,
        'efficiency': lambda m: m.flops_per_param,
        'ai': lambda m: m.avg_ai,
    }

    if sort_by in sort_keys:
        metrics_list = sorted(metrics_list, key=sort_keys[sort_by])

    print("=" * 140)
    print("MODEL COMPARISON")
    print("=" * 140)

    # Model Size Section
    print("\n" + "─" * 140)
    print("MODEL SIZE")
    print("─" * 140)
    print(f"{'Model':<20} {'Parameters':<15} {'Trainable':<15} {'Size (MB)':<12} {'Layers':<10} {'Max Width':<10}")
    print("─" * 140)

    for m in metrics_list:
        print(f"{m.name:<20} {format_number(m.total_params):<15} {format_number(m.trainable_params):<15} "
              f"{m.model_size_mb:<12.2f} {m.num_layers:<10} {m.max_channels:<10}")

    # Computational Cost Section
    print("\n" + "─" * 140)
    print("COMPUTATIONAL COST")
    print("─" * 140)
    print(f"{'Model':<20} {'MACs':<15} {'FLOPs':<15} {'FLOPs/Param':<15} {'Params/MAC':<15}")
    print("─" * 140)

    for m in metrics_list:
        print(f"{m.name:<20} {format_number(m.total_macs):<15} {format_number(m.total_flops):<15} "
              f"{format_number(m.flops_per_param):<15} {m.params_per_mac:<15.4f}")

    # Memory Traffic Section
    print("\n" + "─" * 140)
    print("MEMORY TRAFFIC")
    print("─" * 140)
    print(f"{'Model':<20} {'Total':<15} {'Input':<15} {'Output':<15} {'Weights':<15} {'Bytes/FLOP':<12}")
    print("─" * 140)

    for m in metrics_list:
        print(f"{m.name:<20} {format_number(m.total_memory, 'B'):<15} "
              f"{format_number(m.input_memory, 'B'):<15} {format_number(m.output_memory, 'B'):<15} "
              f"{format_number(m.weight_memory, 'B'):<15} {m.memory_per_flop:<12.2f}")

    # Arithmetic Intensity Section
    print("\n" + "─" * 140)
    print("ARITHMETIC INTENSITY")
    print("─" * 140)
    print(f"{'Model':<20} {'Avg AI':<12} {'Compute-Bound %':<18} {'Memory-Bound %':<18} {'Classification':<20}")
    print("─" * 140)

    for m in metrics_list:
        if m.avg_ai > 50:
            classification = "Compute-bound"
        elif m.avg_ai > 10:
            classification = "Balanced"
        else:
            classification = "Memory-bound"

        print(f"{m.name:<20} {m.avg_ai:<12.2f} {m.compute_bound_ratio:<18.1f} "
              f"{m.memory_bound_ratio:<18.1f} {classification:<20}")

    # Efficiency Summary
    print("\n" + "─" * 140)
    print("EFFICIENCY RANKINGS")
    print("─" * 140)

    # Most parameter-efficient
    by_params = sorted(metrics_list, key=lambda m: m.total_params)
    print(f"Smallest model:           {by_params[0].name:<20} ({format_number(by_params[0].total_params)} params)")
    print(f"Largest model:            {by_params[-1].name:<20} ({format_number(by_params[-1].total_params)} params)")

    # Most compute-efficient
    by_flops = sorted(metrics_list, key=lambda m: m.total_flops)
    print(f"Least FLOPs:              {by_flops[0].name:<20} ({format_number(by_flops[0].total_flops)} FLOPs)")
    print(f"Most FLOPs:               {by_flops[-1].name:<20} ({format_number(by_flops[-1].total_flops)} FLOPs)")

    # Most memory-efficient
    by_memory = sorted(metrics_list, key=lambda m: m.total_memory)
    print(f"Least memory:             {by_memory[0].name:<20} ({format_number(by_memory[0].total_memory, 'B')})")
    print(f"Most memory:              {by_memory[-1].name:<20} ({format_number(by_memory[-1].total_memory, 'B')})")

    # Best FLOPs/param ratio
    by_efficiency = sorted(metrics_list, key=lambda m: m.flops_per_param, reverse=True)
    print(f"Most compute per param:   {by_efficiency[0].name:<20} ({format_number(by_efficiency[0].flops_per_param)} FLOPs/param)")
    print(f"Least compute per param:  {by_efficiency[-1].name:<20} ({format_number(by_efficiency[-1].flops_per_param)} FLOPs/param)")

    # Best AI
    by_ai = sorted(metrics_list, key=lambda m: m.avg_ai, reverse=True)
    print(f"Highest AI (compute):     {by_ai[0].name:<20} ({by_ai[0].avg_ai:.2f} FLOPs/byte)")
    print(f"Lowest AI (memory):       {by_ai[-1].name:<20} ({by_ai[-1].avg_ai:.2f} FLOPs/byte)")

    print("=" * 140)


def main():
    parser = argparse.ArgumentParser(
        description='Compare multiple DNN models across key metrics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare ResNet family
  python cli/compare_models.py resnet18 resnet34 resnet50

  # Compare mobile architectures
  python cli/compare_models.py mobilenet_v2 efficientnet_b0 squeezenet1_0

  # Compare transformers
  python cli/compare_models.py vit_b_16 swin_t

  # Sort by different metrics
  python cli/compare_models.py resnet18 mobilenet_v2 --sort-by flops
        """
    )

    parser.add_argument('models', nargs='+', type=str,
                       help='Models to compare (space-separated)')

    parser.add_argument('--sort-by', type=str,
                       choices=['params', 'flops', 'macs', 'memory', 'efficiency', 'ai'],
                       default='params',
                       help='Sort results by metric')

    parser.add_argument('--input-shape', type=int, nargs=4, default=[1, 3, 224, 224],
                       metavar=('B', 'C', 'H', 'W'),
                       help='Input tensor shape')

    args = parser.parse_args()

    # Profile each model
    metrics_list = []

    for model_name in args.models:
        try:
            print(f"Profiling {model_name}...")
            model_fn = getattr(models, model_name, None)

            if model_fn is None:
                print(f"  Error: Model '{model_name}' not found in torchvision")
                continue

            metrics = profile_model(model_name, model_fn, input_shape=tuple(args.input_shape))
            metrics_list.append(metrics)
            print(f"  ✓ Complete")

        except Exception as e:
            print(f"  ✗ Failed: {str(e)[:60]}")

    if not metrics_list:
        print("\nNo models were successfully profiled.")
        return 1

    # Print comparison
    print()
    print_comparison_table(metrics_list, sort_by=args.sort_by)

    return 0


if __name__ == "__main__":
    sys.exit(main())
