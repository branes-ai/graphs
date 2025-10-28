#!/usr/bin/env python
"""
Model-to-Hardware Comparison Tool
==================================

Analyzes a single DNN model across multiple hardware architectures to determine
the best deployment target based on operation characteristics and resource constraints.

Usage:
    python cli/compare_models.py resnet50
    python cli/compare_models.py mobilenet_v2 --precision int8
    python cli/compare_models.py vit_b_16 --hardware cpu gpu tpu kpu
    python cli/compare_models.py resnet18 --deployment edge
"""

import torch
import torchvision.models as models
from torch.fx import symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp
import argparse
from typing import List, Dict, Tuple

from graphs.transform.partitioning import FusionBasedPartitioner
from graphs.analysis.concurrency import ConcurrencyAnalyzer
from graphs.hardware.resource_model import Precision
from graphs.ir.structures import OperationType

# Hardware mapper imports
from graphs.hardware.mappers.cpu import (
    create_intel_xeon_platinum_8490h_mapper,
    create_amd_epyc_9654_mapper,
)
from graphs.hardware.mappers.gpu import (
    create_h100_mapper,
    create_a100_mapper,
    create_v100_mapper,
    create_t4_mapper,
    create_jetson_orin_nano_mapper,
)
from graphs.hardware.mappers.dsp import (
    create_qrb5165_mapper,
    create_ti_tda4vm_mapper,
)
from graphs.hardware.mappers.accelerators.tpu import create_tpu_v4_mapper
from graphs.hardware.mappers.accelerators.kpu import (
    create_kpu_t64_mapper,
    create_kpu_t256_mapper,
)
from graphs.hardware.mappers.accelerators.dpu import create_dpu_vitis_ai_mapper
from graphs.hardware.mappers.accelerators.cgra import create_plasticine_v2_mapper


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
    'efficientnet_b4': lambda: models.efficientnet_b4(weights=None),
    'vgg16': lambda: models.vgg16(weights=None),
    'densenet121': lambda: models.densenet121(weights=None),
    'shufflenet_v2_x1_0': lambda: models.shufflenet_v2_x1_0(weights=None),
    'squeezenet1_0': lambda: models.squeezenet1_0(weights=None),
    'vit_b_16': lambda: models.vit_b_16(weights=None),
    'vit_l_16': lambda: models.vit_l_16(weights=None),
    'vit_b_32': lambda: models.vit_b_32(weights=None),
    'vit_l_32': lambda: models.vit_l_32(weights=None),
    'convnext_small': lambda: models.convnext_small(weights=None),
    'convnext_base': lambda: models.convnext_base(weights=None),
}

# Model input shapes (some models need non-standard sizes)
MODEL_INPUT_SHAPES = {
    'efficientnet_b2': (1, 3, 260, 260),
    'efficientnet_b4': (1, 3, 380, 380),
}


# Hardware configurations organized by deployment scenario
HARDWARE_CONFIGS = {
    'datacenter': [
        ('Intel Xeon 8490H', create_intel_xeon_platinum_8490h_mapper, 'CPU'),
        ('AMD EPYC 9654', create_amd_epyc_9654_mapper, 'CPU'),
        ('NVIDIA V100', create_v100_mapper, 'GPU'),
        ('NVIDIA T4', create_t4_mapper, 'GPU'),
        ('NVIDIA A100', create_a100_mapper, 'GPU'),
        ('NVIDIA H100', create_h100_mapper, 'GPU'),
        ('Google TPU v4', create_tpu_v4_mapper, 'TPU'),
    ],
    'edge': [
        ('Jetson Orin Nano', lambda: create_jetson_orin_nano_mapper('7W'), 'GPU'),
        ('Stillwater KPU-T64', lambda: create_kpu_t64_mapper('6W'), 'KPU'),
        ('Qualcomm QRB5165', create_qrb5165_mapper, 'DSP'),
        ('Xilinx DPU', create_dpu_vitis_ai_mapper, 'DPU'),
    ],
    'embedded': [
        ('Qualcomm QRB5165', create_qrb5165_mapper, 'DSP'),
        ('TI TDA4VM', create_ti_tda4vm_mapper, 'DSP'),
        ('Stillwater KPU-T64', lambda: create_kpu_t64_mapper('3W'), 'KPU'),
        ('CGRA Plasticine', create_plasticine_v2_mapper, 'CGRA'),
    ],
}


def analyze_model_characteristics(model_name: str, model_fn, input_shape: tuple, verbose: bool = False):
    """Analyze model characteristics independent of hardware"""

    if verbose:
        print(f"\n[1/3] Analyzing {model_name} characteristics...")

    # Load and trace model
    model = model_fn()
    model.eval()

    fx_graph = symbolic_trace(model)
    input_tensor = torch.randn(*input_shape)
    ShapeProp(fx_graph).propagate(input_tensor)

    # Partition with fusion
    partitioner = FusionBasedPartitioner()
    fusion_report = partitioner.partition(fx_graph)

    # Extract execution stages (simplified - group subgraphs into stages of 3)
    # TODO: Update ConcurrencyAnalyzer to work with FusionReport and compute proper dependencies
    # Execution stages are lists of subgraph indices
    n = len(fusion_report.fused_subgraphs)
    stages = []
    i = 0
    while i < n:
        stage_size = min(3, n - i)
        stages.append(list(range(i, i + stage_size)))
        i += stage_size

    # Compute operation type distribution from fused subgraphs
    # Each subgraph can have multiple operation types, so count all
    op_type_counts = {}
    for sg in fusion_report.fused_subgraphs:
        for op_type in sg.operation_types:
            op_key = op_type.value if hasattr(op_type, 'value') else str(op_type)
            op_type_counts[op_key] = op_type_counts.get(op_key, 0) + 1

    total_ops = sum(op_type_counts.values())

    conv_ops = (op_type_counts.get('conv2d', 0) +
                op_type_counts.get('conv2d_depthwise', 0) +
                op_type_counts.get('conv2d_pointwise', 0))

    # Attention operations (for transformers)
    attention_ops = (op_type_counts.get('matmul', 0) +
                    op_type_counts.get('linear', 0) +
                    op_type_counts.get('multihead_attention', 0))

    # Compute FLOP distribution by operation type
    total_flops = fusion_report.total_flops
    conv_flops = 0
    attention_flops = 0
    other_flops = 0

    for sg in fusion_report.fused_subgraphs:
        # Check if any operation in this subgraph is conv or attention
        has_conv = any(op in [OperationType.CONV2D, OperationType.CONV2D_DEPTHWISE, OperationType.CONV2D_POINTWISE]
                      for op in sg.operation_types)
        has_attention = any(op in [OperationType.MATMUL, OperationType.LINEAR, OperationType.MULTIHEAD_ATTENTION]
                           for op in sg.operation_types)

        if has_conv:
            conv_flops += sg.total_flops
        elif has_attention:
            attention_flops += sg.total_flops
        else:
            other_flops += sg.total_flops

    # Estimate arithmetic intensity and parallelism
    total_bytes = sum(sg.total_input_bytes + sg.total_output_bytes for sg in fusion_report.fused_subgraphs)
    arithmetic_intensity = total_flops / max(1, total_bytes)

    # Simple parallelism estimate (number of independent subgraphs)
    max_parallel_ops = min(16, total_ops)  # Conservative estimate
    num_stages = max(1, total_ops // max_parallel_ops)

    return {
        'model_name': model_name,
        'fusion_report': fusion_report,
        'execution_stages': stages,
        'total_flops': total_flops,
        'total_params': sum(sg.total_weight_bytes for sg in fusion_report.fused_subgraphs) / 4,  # Assuming FP32
        'arithmetic_intensity': arithmetic_intensity,
        'max_parallel_ops': max_parallel_ops,
        'num_stages': num_stages,
        'conv_op_pct': (conv_ops / max(1, total_ops)) * 100,
        'attention_op_pct': (attention_ops / max(1, total_ops)) * 100,
        'conv_flop_pct': (conv_flops / max(1, total_flops)) * 100,
        'attention_flop_pct': (attention_flops / max(1, total_flops)) * 100,
        'other_flop_pct': (other_flops / max(1, total_flops)) * 100,
    }


def benchmark_on_hardware(model_chars: dict, hardware_configs: list, precision: str, batch_size: int, verbose: bool = False):
    """Benchmark model on multiple hardware targets"""

    if verbose:
        print(f"\n[2/3] Benchmarking on {len(hardware_configs)} hardware targets...")

    precision_enum = getattr(Precision, precision.upper())
    results = []

    for hw_name, mapper_fn, hw_class in hardware_configs:
        try:
            mapper = mapper_fn()

            # Map graph to hardware
            allocation = mapper.map_graph(
                fusion_report=model_chars['fusion_report'],
                execution_stages=model_chars['execution_stages'],
                batch_size=batch_size,
                precision=precision_enum
            )

            # Calculate FPS
            fps = 1.0 / allocation.total_latency if allocation.total_latency > 0 else 0

            results.append({
                'hardware_name': hw_name,
                'hardware_class': hw_class,
                'latency_ms': allocation.total_latency * 1000,
                'fps': fps,
                'energy_j': allocation.total_energy,
                'utilization_pct': allocation.average_utilization * 100,
                'fps_per_watt': fps / (allocation.total_energy / allocation.total_latency) if allocation.total_latency > 0 and allocation.total_energy > 0 else 0,
                'power_w': allocation.total_energy / allocation.total_latency if allocation.total_latency > 0 else 0,
            })

            if verbose:
                print(f"  ✓ {hw_name}: {allocation.total_latency*1000:.2f} ms")

        except Exception as e:
            if verbose:
                print(f"  ✗ {hw_name}: {e}")
            results.append({
                'hardware_name': hw_name,
                'hardware_class': hw_class,
                'error': str(e),
            })

    return results


def print_model_summary(model_chars: dict):
    """Print model characteristics summary"""

    print("\n" + "=" * 100)
    print(f"MODEL: {model_chars['model_name'].upper()}")
    print("=" * 100)

    print("\n1. MODEL CHARACTERISTICS")
    print("-" * 100)
    print(f"   Parameters:              {model_chars['total_params']/1e6:>8.1f} M")
    print(f"   FLOPs:                   {model_chars['total_flops']/1e9:>8.2f} GFLOPs")
    print(f"   Arithmetic Intensity:    {model_chars['arithmetic_intensity']:>8.1f} (FLOPs/Byte)")
    print(f"   Graph Parallelism:       {model_chars['max_parallel_ops']:>8} ops/stage")
    print(f"   Execution Stages:        {model_chars['num_stages']:>8}")

    # Determine model type
    if model_chars['conv_flop_pct'] > 80:
        model_type = "Convolutional Network (CNN)"
    elif model_chars['attention_flop_pct'] > 50:
        model_type = "Transformer (Attention-based)"
    elif model_chars['conv_flop_pct'] > 50:
        model_type = "Hybrid (Conv-dominant)"
    else:
        model_type = "Hybrid (Attention-dominant)"

    print(f"\n   Model Type:              {model_type}")

    print("\n2. OPERATION TYPE BREAKDOWN")
    print("-" * 100)
    print(f"   Convolution:             {model_chars['conv_flop_pct']:>6.1f}% of FLOPs")
    print(f"   Attention/MatMul:        {model_chars['attention_flop_pct']:>6.1f}% of FLOPs")
    print(f"   Other:                   {model_chars['other_flop_pct']:>6.1f}% of FLOPs")

    # Compute/Memory characterization
    if model_chars['arithmetic_intensity'] > 40:
        char = "Highly Compute-Intensive"
    elif model_chars['arithmetic_intensity'] > 20:
        char = "Compute-Intensive"
    elif model_chars['arithmetic_intensity'] > 10:
        char = "Balanced"
    else:
        char = "Memory-Intensive"

    print(f"\n   Characterization:        {char}")


def print_hardware_comparison(results: list):
    """Print hardware comparison table"""

    # Filter out failed results
    successful = [r for r in results if 'error' not in r]
    failed = [r for r in results if 'error' in r]

    if not successful:
        print("\n⚠ No successful hardware benchmarks")
        return

    print("\n3. HARDWARE COMPARISON")
    print("-" * 100)
    print(f"{'Rank':<6} {'Hardware':<25} {'Class':<8} {'Latency':<12} {'FPS':<10} {'Energy':<12} {'Util %':<8} {'FPS/W':<10}")
    print("-" * 100)

    # Sort by latency (fastest first)
    ranked = sorted(successful, key=lambda r: r['latency_ms'])

    for i, r in enumerate(ranked, 1):
        print(f"{i:<6} {r['hardware_name']:<25} {r['hardware_class']:<8} "
              f"{r['latency_ms']:>10.2f} ms {r['fps']:>9.1f} {r['energy_j']:>10.6f} J "
              f"{r['utilization_pct']:>7.1f} {r['fps_per_watt']:>9.2f}")

    if failed:
        print(f"\n⚠ Failed benchmarks: {', '.join(r['hardware_name'] for r in failed)}")


def print_deployment_recommendations(model_chars: dict, results: list):
    """Print deployment scenario recommendations"""

    successful = [r for r in results if 'error' not in r]
    if not successful:
        return

    print("\n4. DEPLOYMENT RECOMMENDATIONS")
    print("-" * 100)

    # Categorize hardware by deployment
    edge_hw = [r for r in successful if r['hardware_class'] in ['KPU', 'DSP', 'DPU', 'CGRA'] or 'Jetson' in r['hardware_name']]
    datacenter_hw = [r for r in successful if r['hardware_class'] in ['CPU', 'GPU', 'TPU'] and 'Jetson' not in r['hardware_name']]

    # Edge deployment (battery-powered)
    if edge_hw:
        # Best for battery life (lowest power)
        best_battery = min(edge_hw, key=lambda r: r['power_w'])
        # Best for performance (highest FPS)
        best_perf = max(edge_hw, key=lambda r: r['fps'])
        # Best efficiency (FPS/W)
        best_eff = max(edge_hw, key=lambda r: r['fps_per_watt']) if all(r['fps_per_watt'] > 0 for r in edge_hw) else best_battery

        print("\n   Edge (Battery-Powered):")
        print(f"     → {best_battery['hardware_name']:<25} ({best_battery['power_w']:.1f}W, battery life optimized)")

        print("\n   Edge (Tethered/Performance):")
        print(f"     → {best_perf['hardware_name']:<25} ({best_perf['fps']:.1f} FPS, {best_perf['latency_ms']:.2f} ms latency)")

    # Datacenter deployment
    if datacenter_hw:
        # Best for CNNs (Intel AMX wins)
        # Best for Transformers (AMD bandwidth wins)
        if model_chars['conv_flop_pct'] > 70:
            # CNN-heavy: prefer Intel or GPU
            intel_hw = [r for r in datacenter_hw if 'Intel' in r['hardware_name'] or r['hardware_class'] == 'GPU']
            if intel_hw:
                best_dc = max(intel_hw, key=lambda r: r['fps'])
                print("\n   Datacenter (CNN Workload):")
                print(f"     → {best_dc['hardware_name']:<25} ({best_dc['fps']:.1f} FPS)")
                if 'Intel' in best_dc['hardware_name']:
                    print(f"       Reason: AMX acceleration for Conv2D operations ({model_chars['conv_flop_pct']:.0f}% of FLOPs)")

        elif model_chars['attention_flop_pct'] > 50:
            # Transformer: prefer AMD or TPU
            amd_hw = [r for r in datacenter_hw if 'AMD' in r['hardware_name'] or r['hardware_class'] == 'TPU']
            if amd_hw:
                best_dc = max(amd_hw, key=lambda r: r['fps'])
                print("\n   Datacenter (Transformer Workload):")
                print(f"     → {best_dc['hardware_name']:<25} ({best_dc['fps']:.1f} FPS)")
                if 'AMD' in best_dc['hardware_name']:
                    print(f"       Reason: High memory bandwidth for attention operations ({model_chars['attention_flop_pct']:.0f}% of FLOPs)")

    # Embedded/Automotive
    dsp_hw = [r for r in successful if r['hardware_class'] == 'DSP']
    if dsp_hw:
        best_dsp = max(dsp_hw, key=lambda r: r['fps'])
        print("\n   Embedded/Automotive:")
        print(f"     → {best_dsp['hardware_name']:<25} ({best_dsp['fps']:.1f} FPS, {best_dsp['power_w']:.1f}W)")
        if 'TDA4VM' in best_dsp['hardware_name']:
            print(f"       Features: ASIL-D certified, -40°C to 125°C operating range")


def print_architectural_insights(model_chars: dict, results: list):
    """Print architectural insights and WHY certain hardware wins"""

    successful = [r for r in results if 'error' not in r]
    if not successful:
        return

    print("\n5. ARCHITECTURAL INSIGHTS")
    print("-" * 100)

    # Why certain hardware wins
    if model_chars['conv_flop_pct'] > 70:
        print("   ✓ Conv-heavy ({:.0f}% of FLOPs) → Intel AMX/GPU Tensor Cores provide 8-10× speedup".format(model_chars['conv_flop_pct']))
        print("     - Conv2D operations map perfectly to matrix multiply units")
        print("     - Systolic arrays (TPU/KPU) also excel at convolution")

    if model_chars['attention_flop_pct'] > 50:
        print("   ✓ Attention-heavy ({:.0f}% of FLOPs) → High memory bandwidth critical".format(model_chars['attention_flop_pct']))
        print("     - Self-attention reads large matrices (Q, K, V)")
        print("     - AMD EPYC (460-576 GB/s) outperforms Intel (307-358 GB/s) for Transformers")

    if model_chars['arithmetic_intensity'] < 10:
        print("   ⚠ Memory-intensive (AI={:.1f}) → Bandwidth-bound on most hardware".format(model_chars['arithmetic_intensity']))
        print("     - Quantization (INT8) may not improve latency on CPUs")
        print("     - Edge accelerators (KPU/DSP) provide better efficiency")

    if model_chars['max_parallel_ops'] < 16:
        print("   ⚠ Low graph parallelism ({} ops/stage) → GPU utilization limited at batch=1".format(model_chars['max_parallel_ops']))
        print("     - Consider batch≥{} for better GPU utilization".format(128 // model_chars['max_parallel_ops']))
        print("     - Tile-based accelerators (KPU) achieve higher utilization")


def main():
    parser = argparse.ArgumentParser(
        description='Compare a single DNN model across multiple hardware architectures',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze ResNet-50 on all hardware
  python cli/compare_models.py resnet50

  # Analyze MobileNet-V2 with INT8 precision
  python cli/compare_models.py mobilenet_v2 --precision int8

  # Focus on edge deployment scenario
  python cli/compare_models.py efficientnet_b0 --deployment edge

  # Test on specific hardware subset
  python cli/compare_models.py vit_b_16 --hardware cpu gpu tpu
        """
    )

    parser.add_argument('model',
                       help='Model to analyze (e.g., resnet50, mobilenet_v2, vit_b_16)')

    parser.add_argument('--precision', choices=['fp32', 'bf16', 'int8', 'int4'],
                       default='int8',
                       help='Precision for inference (default: int8)')

    parser.add_argument('--batch-size', type=int, default=1,
                       help='Batch size for inference (default: 1)')

    parser.add_argument('--deployment', choices=['datacenter', 'edge', 'embedded', 'all'],
                       default='all',
                       help='Deployment scenario filter (default: all)')

    parser.add_argument('--verbose', action='store_true',
                       help='Show detailed progress')

    args = parser.parse_args()

    # Validate model
    if args.model not in AVAILABLE_MODELS:
        print(f"Error: Unknown model '{args.model}'")
        print(f"Available models: {', '.join(sorted(AVAILABLE_MODELS.keys()))}")
        return 1

    # Get model function and input shape
    model_fn = AVAILABLE_MODELS[args.model]
    input_shape = MODEL_INPUT_SHAPES.get(args.model, (1, 3, 224, 224))

    # Select hardware configs based on deployment
    if args.deployment == 'all':
        hw_configs = []
        for configs in HARDWARE_CONFIGS.values():
            hw_configs.extend(configs)
        # Remove duplicates (some hardware appears in multiple scenarios)
        seen = set()
        unique_configs = []
        for name, fn, cls in hw_configs:
            if name not in seen:
                seen.add(name)
                unique_configs.append((name, fn, cls))
        hw_configs = unique_configs
    else:
        hw_configs = HARDWARE_CONFIGS[args.deployment]

    # Step 1: Analyze model characteristics
    model_chars = analyze_model_characteristics(args.model, model_fn, input_shape, args.verbose)

    # Step 2: Benchmark on hardware
    results = benchmark_on_hardware(model_chars, hw_configs, args.precision, args.batch_size, args.verbose)

    # Step 3: Print analysis
    if args.verbose:
        print("\n[3/3] Generating recommendations...")

    print_model_summary(model_chars)
    print_hardware_comparison(results)
    print_deployment_recommendations(model_chars, results)
    print_architectural_insights(model_chars, results)

    print("\n" + "=" * 100)
    print("ANALYSIS COMPLETE")
    print("=" * 100)

    return 0


if __name__ == "__main__":
    exit(main())
