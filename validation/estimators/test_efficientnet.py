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

# DEPRECATED: from src.graphs.characterize.arch_profiles import (
    intel_i7_profile, amd_ryzen7_profile, h100_pcie_profile,
    tpu_v4_profile, kpu_t2_profile, kpu_t100_profile
)
# DEPRECATED: from src.graphs.characterize.fused_ops import default_registry
# DEPRECATED: from src.graphs.characterize.walker import FXGraphWalker
#
# TODO: Update to use new partitioning system:
#   from src.graphs.transform.partitioning import FusionBasedPartitioner
#   from src.graphs.hardware.resource_model import Precision
# See validation/hardware/test_all_hardware.py for example usage

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

        # Characterize across architectures
        registry = default_registry()
        results = []

        architectures = [
            ("Intel Core i7", intel_i7_profile),
            ("AMD Ryzen 7", amd_ryzen7_profile),
            ("H100-PCIe", h100_pcie_profile),
            ("TPU v4", tpu_v4_profile),
            ("KPU-T2", kpu_t2_profile),
            ("KPU-T100", kpu_t100_profile)
        ]

        for arch_name, arch_profile in architectures:
            walker = FXGraphWalker(arch_profile, registry)
            metrics = walker.walk(fx_graph)

            results.append({
                "Model": model_name,
                "Architecture": arch_name,
                "Batch": batch_size,
                "Parameters": total_params,
                "FLOPs": metrics['FLOPs'],
                "Memory_MB": metrics['Memory'] / (1024**2),
                "Tiles": metrics['Tiles'],
                "Latency_ms": metrics['Latency'] * 1000,
                "Energy_J": metrics['Energy']
            })

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

    print(summary[['Model', 'Params_M', 'FLOPs_G', 'Memory_MB', 'Tiles', 'Latency_ms']].to_string(index=False))

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

    # Edge deployment comparison (KPU-T2)
    print("\n" + "=" * 80)
    print("Edge Deployment (KPU-T2 - IoT/Battery-Powered)")
    print("=" * 80)

    kpu_df = df[df['Architecture'] == 'KPU-T2'].copy()
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
