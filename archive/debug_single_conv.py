#!/usr/bin/env python
"""
Debug Single Conv Layer
========================

Deep dive into why our partitioner miscalculates FLOPs for a simple Conv2d
"""

import torch
import torch.nn as nn
from torch.fx import symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp
import sys
sys.path.insert(0, 'src')

from graphs.characterize.graph_partitioner import GraphPartitioner


def debug_conv():
    """Debug the first conv layer calculation"""

    print("=" * 80)
    print("DEBUGGING SINGLE CONV2D LAYER")
    print("=" * 80)

    # Create simple conv
    conv = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
    conv.eval()

    input_tensor = torch.randn(1, 3, 224, 224)

    # Wrap in module for FX
    class SimpleConv(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = conv
        def forward(self, x):
            return self.conv(x)

    model = SimpleConv()

    # Trace
    fx_graph = symbolic_trace(model)
    ShapeProp(fx_graph).propagate(input_tensor)

    # Print FX graph
    print("\nFX Graph Nodes:")
    print("-" * 80)
    for i, node in enumerate(fx_graph.graph.nodes):
        print(f"{i}: op={node.op:<15} name={node.name:<20} target={node.target}")
        if hasattr(node, 'meta') and 'tensor_meta' in node.meta:
            meta = node.meta['tensor_meta']
            print(f"   tensor_meta.shape = {meta.shape}")
            print(f"   tensor_meta.dtype = {meta.dtype}")

    # Manually walk through the calculation logic
    print("\n" + "=" * 80)
    print("MANUAL WALK-THROUGH OF PARTITIONER LOGIC")
    print("=" * 80)

    for node in fx_graph.graph.nodes:
        if node.op == 'call_module':
            print(f"\nProcessing node: {node.name}")

            module = fx_graph.get_submodule(node.target)
            print(f"  Module type: {type(module)}")

            if hasattr(node, 'meta') and 'tensor_meta' in node.meta:
                meta = node.meta['tensor_meta']
                print(f"  Input shape from meta: {meta.shape}")

                if isinstance(module, nn.Conv2d):
                    print(f"\n  Conv2d parameters:")
                    print(f"    in_channels: {module.in_channels}")
                    print(f"    out_channels: {module.out_channels}")
                    print(f"    kernel_size: {module.kernel_size}")
                    print(f"    stride: {module.stride}")
                    print(f"    padding: {module.padding}")
                    print(f"    groups: {module.groups}")

                    # Extract from meta (THIS IS WHAT THE BUG LIKELY IS)
                    B, C_in, H, W = meta.shape
                    print(f"\n  Extracted from meta.shape:")
                    print(f"    B (batch): {B}")
                    print(f"    C_in (input channels): {C_in}")
                    print(f"    H (height): {H}")
                    print(f"    W (width): {W}")

                    C_out = module.out_channels
                    K_h, K_w = (module.kernel_size if isinstance(module.kernel_size, tuple)
                               else (module.kernel_size, module.kernel_size))
                    S_h, S_w = (module.stride if isinstance(module.stride, tuple)
                               else (module.stride, module.stride))
                    P = module.padding if isinstance(module.padding, int) else module.padding[0]
                    groups = module.groups

                    H_out = (H + 2 * P - K_h) // S_h + 1
                    W_out = (W + 2 * P - K_w) // S_w + 1

                    print(f"\n  Calculated output dimensions:")
                    print(f"    H_out: ({H} + 2*{P} - {K_h}) // {S_h} + 1 = {H_out}")
                    print(f"    W_out: ({W} + 2*{P} - {K_w}) // {S_w} + 1 = {W_out}")

                    C_in_per_group = C_in // groups

                    macs = B * C_out * H_out * W_out * C_in_per_group * K_h * K_w

                    print(f"\n  MAC calculation:")
                    print(f"    MACs = B * C_out * H_out * W_out * C_in_per_group * K_h * K_w")
                    print(f"    MACs = {B} * {C_out} * {H_out} * {W_out} * {C_in_per_group} * {K_h} * {K_w}")
                    print(f"    MACs = {macs:,}")
                    print(f"    FLOPs = {2 * macs:,}")

                    # What SHOULD it be?
                    print(f"\n  CORRECT calculation (using INPUT shape):")
                    B_correct = 1
                    C_in_correct = 3
                    H_correct = 224
                    W_correct = 224
                    H_out_correct = (H_correct + 2 * P - K_h) // S_h + 1
                    W_out_correct = (W_correct + 2 * P - K_w) // S_w + 1
                    macs_correct = B_correct * C_out * H_out_correct * W_out_correct * C_in_correct * K_h * K_w
                    print(f"    MACs = {B_correct} * {C_out} * {H_out_correct} * {W_out_correct} * {C_in_correct} * {K_h} * {K_w}")
                    print(f"    MACs = {macs_correct:,}")
                    print(f"    FLOPs = {2 * macs_correct:,}")

                    print(f"\n  ERROR:")
                    print(f"    Using meta.shape = {meta.shape}")
                    print(f"    But meta.shape is the OUTPUT shape, not INPUT shape!")
                    print(f"    We need to get INPUT shape from node.args[0]")

            # Check args
            print(f"\n  Node args:")
            for i, arg in enumerate(node.args):
                print(f"    args[{i}]: {arg}")
                if hasattr(arg, 'meta') and 'tensor_meta' in arg.meta:
                    arg_meta = arg.meta['tensor_meta']
                    print(f"      -> tensor_meta.shape = {arg_meta.shape}")

    # Run actual partitioner
    print("\n" + "=" * 80)
    print("ACTUAL PARTITIONER OUTPUT")
    print("=" * 80)

    partitioner = GraphPartitioner()
    report = partitioner.partition(fx_graph)

    print(f"\nTotal MACs: {report.total_macs:,}")
    print(f"Total FLOPs: {report.total_flops:,}")

    for sg in report.subgraphs:
        print(f"\nSubgraph: {sg.node_name}")
        print(f"  Operation: {sg.operation_type.value}")
        print(f"  MACs: {sg.macs:,}")
        print(f"  FLOPs: {sg.flops:,}")


if __name__ == "__main__":
    debug_conv()
