#!/usr/bin/env python
"""Characterization test for EfficientNet family from torchvision"""

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
            "Intel Core i7": create_intel_cpu_mapper("avx512"),
            "AMD Ryzen 7": create_amd_cpu_mapper(),
            "H100-PCIe": create_h100_mapper(),
            "TPU v4": create_tpu_v4_mapper(),
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
        import traceback
        traceback.print_exc()
        return [], 0

def main():
    print("=" * 80)
    print("EfficientNet Family Characterization")
    print("=" * 80)

    # Models to compare
    models_to_test = [
        ("EfficientNet-B0", models.efficientnet_b0),
        ("EfficientNet-B1", models.efficientnet_b1),
        ("EfficientNet-B2", models.efficientnet_b2),
        ("EfficientNet-V2-S", models.efficientnet_v2_s),
        ("EfficientNet-V2-M", models.efficientnet_v2_m),
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
            print(f"   FLOPs (H100): {format_number(results[2]['FLOPs'])}")
            print(f"   Latency (H100): {results[2]['Latency_ms']:.3f} ms")
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

    summary = df[df['Architecture'] == 'H100-PCIe'].copy()
    summary['FLOPs_G'] = summary['FLOPs'] / 1e9
    summary['Params_M'] = summary['Parameters'] / 1e6

    print(summary[['Model', 'Params_M', 'FLOPs_G', 'Latency_ms', 'Utilization']].to_string(index=False))

    # Comparison with ResNet-18 (reference)
    print("\n" + "=" * 80)
    print("Efficiency Comparison (vs ResNet-18: 11.7M params, 3.79G FLOPs)")
    print("=" * 80)

    resnet18_params = 11.69e6
    resnet18_flops = 3.79e9

    summary_cmp = df[df['Architecture'] == 'H100-PCIe'].copy()
    summary_cmp['Param_Ratio'] = summary_cmp['Parameters'] / resnet18_params
    summary_cmp['FLOP_Ratio'] = summary_cmp['FLOPs'] / resnet18_flops

    print(summary_cmp[['Model', 'Param_Ratio', 'FLOP_Ratio']].to_string(index=False))

    # H100 Performance comparison
    print("\n" + "=" * 80)
    print("H100-PCIe Performance")
    print("=" * 80)

    h100_df = df[df['Architecture'] == 'H100-PCIe'].copy()
    h100_df = h100_df.sort_values('Latency_ms')
    h100_df['Throughput_FPS'] = 1000 / h100_df['Latency_ms']

    print(h100_df[['Model', 'Latency_ms', 'Throughput_FPS']].to_string(index=False))

    # Edge deployment comparison (KPU-T100)
    print("\n" + "=" * 80)
    print("Edge Deployment (KPU-T100 - IoT/Battery-Powered)")
    print("=" * 80)

    kpu_df = df[df['Architecture'] == 'KPU-T100'].copy()
    kpu_df = kpu_df.sort_values('Energy_J')
    kpu_df['Throughput_FPS'] = 1000 / kpu_df['Latency_ms']

    print(kpu_df[['Model', 'Latency_ms', 'Throughput_FPS', 'Energy_J']].to_string(index=False))

    # Save to CSV
    output_path = 'results/validation/efficientnet_results.csv'
    df.to_csv(output_path, index=False)
    print(f"\n✓ Results saved to {output_path}")

    return 0

if __name__ == "__main__":
    exit(main())
