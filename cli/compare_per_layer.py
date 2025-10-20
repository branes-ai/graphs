#!/usr/bin/env python
"""
Per-Layer FLOP Comparison
==========================

Detailed layer-by-layer comparison to identify specific discrepancies
"""

import torch
import torch.nn as nn
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
    print("ERROR: fvcore not installed")
    sys.exit(1)


def manual_conv2d_macs(in_channels, out_channels, kernel_size, input_shape,
                       stride=1, padding=0, groups=1):
    """Manually calculate MACs for Conv2d using standard formula"""
    B, C_in, H_in, W_in = input_shape

    if isinstance(kernel_size, tuple):
        K_h, K_w = kernel_size
    else:
        K_h = K_w = kernel_size

    if isinstance(stride, tuple):
        S_h, S_w = stride
    else:
        S_h = S_w = stride

    if isinstance(padding, tuple):
        P_h, P_w = padding
    else:
        P_h = P_w = padding

    H_out = (H_in + 2 * P_h - K_h) // S_h + 1
    W_out = (W_in + 2 * P_w - K_w) // S_w + 1

    C_in_per_group = in_channels // groups

    # MACs (multiply-accumulates) - this is the standard formula
    macs = B * out_channels * H_out * W_out * C_in_per_group * K_h * K_w

    return macs


def compare_first_conv():
    """Compare the very first conv layer in detail"""

    print("=" * 80)
    print("DETAILED ANALYSIS: First Conv Layer in ResNet-18")
    print("=" * 80)

    # Create just the first conv
    model = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.eval()

    input_shape = (1, 3, 224, 224)
    input_tensor = torch.randn(*input_shape)

    # Manual calculation
    manual_macs = manual_conv2d_macs(
        in_channels=3,
        out_channels=64,
        kernel_size=7,
        input_shape=input_shape,
        stride=2,
        padding=3,
        groups=1
    )

    print(f"\nLayer: Conv2d(3 -> 64, kernel=7x7, stride=2, padding=3)")
    print(f"Input shape: {input_shape}")
    print(f"Output shape: (1, 64, 112, 112)")

    print(f"\nManual calculation:")
    print(f"  H_out = (224 + 2*3 - 7) // 2 + 1 = 112")
    print(f"  W_out = (224 + 2*3 - 7) // 2 + 1 = 112")
    print(f"  MACs = 1 * 64 * 112 * 112 * 3 * 7 * 7")
    print(f"  MACs = {manual_macs:,}")
    print(f"  FLOPs (2x) = {2 * manual_macs:,}")

    # FVCore
    flops_analysis = FlopCountAnalysis(model, input_tensor)
    fvcore_count = flops_analysis.total()

    print(f"\nFVCore count: {fvcore_count:,}")
    print(f"Ratio (Manual MACs / FVCore): {manual_macs / fvcore_count:.3f}")
    print(f"Ratio (Manual FLOPs / FVCore): {2 * manual_macs / fvcore_count:.3f}")

    # Our tool
    class SimpleConv(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = model
        def forward(self, x):
            return self.conv(x)

    wrapper = SimpleConv()
    fx_graph = symbolic_trace(wrapper)
    ShapeProp(fx_graph).propagate(input_tensor)

    partitioner = GraphPartitioner()
    report = partitioner.partition(fx_graph)

    our_macs = report.total_macs
    our_flops = report.total_flops

    print(f"\nOur tool:")
    print(f"  MACs:  {our_macs:,}")
    print(f"  FLOPs: {our_flops:,}")
    print(f"Ratio (Our MACs / FVCore): {our_macs / fvcore_count:.3f}")
    print(f"Ratio (Our FLOPs / FVCore): {our_flops / fvcore_count:.3f}")

    # Analysis
    print(f"\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)

    if abs(manual_macs - fvcore_count) / fvcore_count < 0.01:
        print("\n✓ FVCore counts MACs (not FLOPs)")
        print("  - Each MAC is a multiply-accumulate operation")
        print("  - Community convention: report MACs but sometimes label as 'FLOPs'")

    if abs(our_macs - manual_macs) / manual_macs < 0.01:
        print("\n✓ Our MAC calculation is correct")
    else:
        print(f"\n✗ Our MAC calculation differs by {abs(our_macs - manual_macs) / manual_macs * 100:.1f}%")

    if abs(our_flops - 2 * manual_macs) / (2 * manual_macs) < 0.01:
        print("✓ Our FLOP calculation (2x MACs) is correct")
    else:
        print(f"✗ Our FLOP calculation differs by {abs(our_flops - 2 * manual_macs) / (2 * manual_macs) * 100:.1f}%")


def compare_resnet18_layers():
    """Compare all conv layers in ResNet-18"""

    print("\n" + "=" * 80)
    print("LAYER-BY-LAYER COMPARISON: ResNet-18")
    print("=" * 80)

    model = models.resnet18(weights=None)
    model.eval()
    input_tensor = torch.randn(1, 3, 224, 224)

    # Get fvcore per-module counts
    flops_analysis = FlopCountAnalysis(model, input_tensor)
    fvcore_by_module = flops_analysis.by_module()

    # Get our counts
    fx_graph = symbolic_trace(model)
    ShapeProp(fx_graph).propagate(input_tensor)
    partitioner = GraphPartitioner()
    report = partitioner.partition(fx_graph)

    print(f"\n{'Layer':<30} {'Our MACs':<15} {'FVCore':<15} {'Ratio':<10}")
    print("-" * 70)

    # Try to match layers (simplified - just show conv layers)
    conv_count = 0
    for sg in report.subgraphs:
        if 'conv' in sg.operation_type.value:
            conv_count += 1
            layer_name = sg.node_name[:28]
            our_macs = sg.macs

            # Try to find matching fvcore module
            fv_match = None
            for fv_name, fv_count in fvcore_by_module.items():
                if sg.node_name in fv_name or fv_name.split('.')[-1] in sg.node_name:
                    fv_match = fv_count
                    break

            if fv_match:
                ratio = our_macs / fv_match
                print(f"{layer_name:<30} {our_macs / 1e6:>8.2f}M      {fv_match / 1e6:>8.2f}M      {ratio:>6.2f}x")
            else:
                print(f"{layer_name:<30} {our_macs / 1e6:>8.2f}M      {'N/A':<15}")

    print(f"\nTotal conv layers: {conv_count}")


if __name__ == "__main__":
    compare_first_conv()
    compare_resnet18_layers()
