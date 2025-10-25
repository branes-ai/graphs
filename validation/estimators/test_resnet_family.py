#!/usr/bin/env python
"""Compare different ResNet architectures (18, 34, 50)"""

import torch
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import torchvision.models as models
import pandas as pd
from torch.fx import symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp

from src.graphs.transform.partitioning import FusionBasedPartitioner
from src.graphs.hardware.mappers.cpu import create_intel_cpu_mapper, create_amd_cpu_mapper
from src.graphs.hardware.mappers.gpu import create_h100_mapper
from src.graphs.hardware.mappers.accelerators.tpu import create_tpu_v4_mapper
from src.graphs.hardware.mappers.accelerators.kpu import create_kpu_t64_mapper
from src.graphs.hardware.resource_model import Precision

def format_number(n):
    """Format large numbers with SI prefixes"""
    if n >= 1e12:
        return f"{n/1e12:.2f}T"
    elif n >= 1e9:
        return f"{n/1e9:.2f}G"
    elif n >= 1e6:
        return f"{n/1e6:.2f}M"
    elif n >= 1e3:
        return f"{n/1e3:.2f}K"
    else:
        return f"{n:.2f}"

def characterize_model(model, model_name, batch_size=1):
    """Characterize a single model"""
    model.eval()

    # Create input
    input_tensor = torch.randn(batch_size, 3, 224, 224)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())

    try:
        # FX trace and shape propagation
        fx_graph = symbolic_trace(model)
        shape_prop = ShapeProp(fx_graph)
        shape_prop.propagate(input_tensor)

        # Partition the graph
        partitioner = FusionBasedPartitioner()
        fusion_report = partitioner.partition(fx_graph)

        # Extract execution stages
        def extract_execution_stages(fusion_report):
            subgraphs = fusion_report.fused_subgraphs
            n = len(subgraphs)
            if n == 0:
                return []
            stages = []
            i = 0
            while i < n:
                stage_size = min(3, n - i)
                stages.append(list(range(i, i + stage_size)))
                i += stage_size
            return stages

        execution_stages = extract_execution_stages(fusion_report)

        # Create hardware mappers
        mappers = {
            "CPU": create_intel_cpu_mapper("avx512"),
            "GPU": create_h100_mapper(),
            "TPU": create_tpu_v4_mapper(),
            "Stillwater KPU-T64": create_kpu_t64_mapper(),
        }

        results = []
        for arch_name, mapper in mappers.items():
            try:
                allocation = mapper.map_graph(
                    fusion_report=fusion_report,
                    execution_stages=execution_stages,
                    batch_size=batch_size,
                    precision=Precision.FP32
                )

                results.append({
                    "Model": model_name,
                    "Architecture": arch_name,
                    "Batch": batch_size,
                    "Parameters": total_params,
                    "FLOPs": fusion_report.total_flops,
                    "Latency_ms": allocation.total_latency * 1000,
                    "Energy_J": allocation.total_energy,
                    "Utilization": allocation.average_utilization
                })
            except Exception as e:
                print(f"   ✗ {arch_name} failed: {e}")

        return results, total_params

    except Exception as e:
        print(f"   ✗ Failed to characterize {model_name}: {e}")
        return [], 0

def main():
    print("=" * 80)
    print("ResNet Family Characterization")
    print("=" * 80)

    # Models to compare
    models_to_test = [
        ("ResNet-18", models.resnet18),
        ("ResNet-34", models.resnet34),
        ("ResNet-50", models.resnet50),
    ]

    all_results = []

    for model_name, model_fn in models_to_test:
        print(f"\n{model_name}:")
        print("-" * 80)

        # Load model
        model = model_fn(weights=None)

        # Characterize
        results, params = characterize_model(model, model_name)

        if results:
            print(f"   Parameters: {format_number(params)} ({params:,})")
            print(f"   FLOPs (GPU): {format_number(results[1]['FLOPs'])}")
            print(f"   Latency (GPU): {results[1]['Latency_ms']:.3f} ms")
            all_results.extend(results)
        else:
            print(f"   Failed to characterize")

    if not all_results:
        print("\n✗ No results to display")
        return 1

    # Create DataFrame
    df = pd.DataFrame(all_results)

    # Summary table by model
    print("\n" + "=" * 80)
    print("Summary: Computational Complexity by Model")
    print("=" * 80)

    summary = df[df['Architecture'] == 'GPU'].copy()
    summary['FLOPs_G'] = summary['FLOPs'] / 1e9
    summary['Params_M'] = summary['Parameters'] / 1e6

    print(summary[['Model', 'Params_M', 'FLOPs_G', 'Latency_ms', 'Utilization']].to_string(index=False))

    # GPU Performance comparison
    print("\n" + "=" * 80)
    print("GPU Performance Comparison")
    print("=" * 80)

    gpu_df = df[df['Architecture'] == 'GPU'].copy()
    gpu_df = gpu_df.sort_values('FLOPs')

    # Normalize to ResNet-18
    base_flops = gpu_df.iloc[0]['FLOPs']
    base_latency = gpu_df.iloc[0]['Latency_ms']

    gpu_df['FLOPs_Ratio'] = gpu_df['FLOPs'] / base_flops
    gpu_df['Latency_Ratio'] = gpu_df['Latency_ms'] / base_latency

    print(gpu_df[['Model', 'FLOPs_Ratio', 'Latency_Ratio']].to_string(index=False))

    # Architecture comparison for ResNet-50
    print("\n" + "=" * 80)
    print("ResNet-50 Across Architectures")
    print("=" * 80)

    r50_df = df[df['Model'] == 'ResNet-50'].copy()
    r50_df = r50_df.sort_values('Latency_ms')

    # Speedup vs CPU
    cpu_latency = r50_df[r50_df['Architecture'] == 'CPU']['Latency_ms'].values[0]
    r50_df['Speedup_vs_CPU'] = cpu_latency / r50_df['Latency_ms']

    print(r50_df[['Architecture', 'Latency_ms', 'Energy_J', 'Speedup_vs_CPU']].to_string(index=False))

    # Save to CSV
    output_path = 'results/validation/resnet_family_results.csv'
    df.to_csv(output_path, index=False)
    print(f"\n✓ Results saved to {output_path}")

    return 0

if __name__ == "__main__":
    exit(main())
