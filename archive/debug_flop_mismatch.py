#!/usr/bin/env python
"""
Debug FLOP Count Mismatch
==========================

Investigates the discrepancy between our partitioner and fvcore by:
1. Comparing MACs vs FLOPs
2. Breaking down by operation type
3. Identifying which operations contribute to the difference
"""

import torch
import torchvision.models as models
from torch.fx import symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp
import sys
sys.path.insert(0, 'src')

from graphs.characterize.graph_partitioner import GraphPartitioner

try:
    from fvcore.nn import FlopCountAnalysis
    FVCORE_AVAILABLE = True
except ImportError:
    FVCORE_AVAILABLE = False
    print("ERROR: fvcore not installed. Install with: pip install fvcore")
    sys.exit(1)


def analyze_mismatch():
    """Detailed analysis of FLOP counting mismatch"""

    print("=" * 80)
    print("FLOP COUNT MISMATCH INVESTIGATION")
    print("=" * 80)

    # Load ResNet-18
    print("\n[1/4] Loading ResNet-18...")
    model = models.resnet18(weights=None)
    model.eval()
    input_tensor = torch.randn(1, 3, 224, 224)

    # Trace
    print("[2/4] Tracing with PyTorch FX...")
    fx_graph = symbolic_trace(model)
    ShapeProp(fx_graph).propagate(input_tensor)

    # Count with our tool
    print("[3/4] Counting with our partitioner...")
    partitioner = GraphPartitioner()
    report = partitioner.partition(fx_graph)

    # Count with fvcore
    print("[4/4] Counting with fvcore...")
    flops_analysis = FlopCountAnalysis(model, input_tensor)
    fvcore_total = flops_analysis.total()

    # Analysis
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    our_flops = report.total_flops
    our_macs = report.total_macs

    print(f"\nOur Tool:")
    print(f"  Total FLOPs: {our_flops / 1e9:.3f} GFLOPs ({our_flops:,})")
    print(f"  Total MACs:  {our_macs / 1e9:.3f} GMACs  ({our_macs:,})")

    print(f"\nFVCore:")
    print(f"  Total:       {fvcore_total / 1e9:.3f} GFLOPs ({fvcore_total:,})")

    print(f"\nComparison:")
    print(f"  Our FLOPs / FVCore: {our_flops / fvcore_total:.2f}x")
    print(f"  Our MACs / FVCore:  {our_macs / fvcore_total:.2f}x")

    # Breakdown by operation type
    print("\n" + "=" * 80)
    print("BREAKDOWN BY OPERATION TYPE")
    print("=" * 80)

    flops_by_type = {}
    macs_by_type = {}
    count_by_type = {}

    for sg in report.subgraphs:
        op_type = sg.operation_type.value
        flops_by_type[op_type] = flops_by_type.get(op_type, 0) + sg.flops
        macs_by_type[op_type] = macs_by_type.get(op_type, 0) + sg.macs
        count_by_type[op_type] = count_by_type.get(op_type, 0) + 1

    print(f"\n{'Operation Type':<20} {'Count':<8} {'FLOPs':<15} {'MACs':<15} {'% of Total':<12}")
    print("-" * 80)

    for op_type in sorted(flops_by_type.keys(), key=lambda k: flops_by_type[k], reverse=True):
        flops = flops_by_type[op_type]
        macs = macs_by_type[op_type]
        count = count_by_type[op_type]
        pct = (flops / our_flops * 100) if our_flops > 0 else 0

        flops_str = f"{flops / 1e9:.3f}G" if flops > 1e9 else f"{flops / 1e6:.1f}M"
        macs_str = f"{macs / 1e9:.3f}G" if macs > 1e9 else f"{macs / 1e6:.1f}M"

        print(f"{op_type:<20} {count:<8} {flops_str:<15} {macs_str:<15} {pct:>6.1f}%")

    # Calculate hypothetical values
    print("\n" + "=" * 80)
    print("HYPOTHESIS TESTING")
    print("=" * 80)

    # Hypothesis 1: FVCore counts only Conv2d and Linear MACs
    conv_linear_macs = (macs_by_type.get('conv2d', 0) +
                        macs_by_type.get('conv2d_pointwise', 0) +
                        macs_by_type.get('linear', 0))

    print(f"\nHypothesis 1: FVCore counts only Conv2d + Linear MACs")
    print(f"  Our Conv2d + Linear MACs: {conv_linear_macs / 1e9:.3f} GMACs")
    print(f"  FVCore total:             {fvcore_total / 1e9:.3f} GFLOPs")
    print(f"  Ratio:                    {conv_linear_macs / fvcore_total:.2f}x")
    print(f"  Match: {'YES' if abs(conv_linear_macs / fvcore_total - 1.0) < 0.05 else 'NO'}")

    # Hypothesis 2: FVCore counts Conv2d + Linear FLOPs (2x MACs)
    conv_linear_flops = (flops_by_type.get('conv2d', 0) +
                         flops_by_type.get('conv2d_pointwise', 0) +
                         flops_by_type.get('linear', 0))

    print(f"\nHypothesis 2: FVCore counts Conv2d + Linear FLOPs (2x MACs)")
    print(f"  Our Conv2d + Linear FLOPs: {conv_linear_flops / 1e9:.3f} GFLOPs")
    print(f"  FVCore total:              {fvcore_total / 1e9:.3f} GFLOPs")
    print(f"  Ratio:                     {conv_linear_flops / fvcore_total:.2f}x")
    print(f"  Match: {'YES' if abs(conv_linear_flops / fvcore_total - 1.0) < 0.05 else 'NO'}")

    # What we're counting extra
    print(f"\n" + "=" * 80)
    print("OPERATIONS NOT COUNTED BY FVCORE")
    print("=" * 80)

    extra_flops = our_flops - conv_linear_flops
    extra_macs = our_macs - conv_linear_macs

    print(f"\nExtra FLOPs beyond Conv2d/Linear: {extra_flops / 1e9:.3f} GFLOPs ({extra_flops / our_flops * 100:.1f}% of total)")
    print(f"Extra MACs beyond Conv2d/Linear:  {extra_macs / 1e9:.3f} GMACs\n")

    for op_type in ['batchnorm', 'relu', 'relu6', 'maxpool2d', 'avgpool2d']:
        if op_type in flops_by_type:
            flops = flops_by_type[op_type]
            count = count_by_type[op_type]
            print(f"  {op_type:<15}: {flops / 1e6:8.1f}M FLOPs ({count} operations)")

    # Recommendations
    print(f"\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)

    if abs(conv_linear_macs / fvcore_total - 1.0) < 0.05:
        print("\n✓ FVCore counts only Conv2d/Linear MACs (multiply-accumulates)")
        print("✓ Our tool counts ALL operations including activations and normalization")
        print("\nTo align with FVCore and ML community conventions:")
        print("  1. Report MACs instead of FLOPs for Conv2d/Linear")
        print("  2. Make activations/batch-norm optional in reporting")
        print("  3. Clearly label what is included in the count")
    else:
        print("\nNeeds further investigation - mismatch not fully explained")


if __name__ == "__main__":
    analyze_mismatch()
