#!/usr/bin/env python
"""Quick test script to validate Conv2D characterization"""

import torch
from torch.fx import symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp

from graphs.models.conv2d_stack import make_conv2d
from graphs.characterize.arch_profiles import cpu_profile
from graphs.characterize.fused_ops import default_registry
from graphs.characterize.walker import FXGraphWalker

def main():
    print("=" * 60)
    print("Testing Conv2D Characterization")
    print("=" * 60)

    # Create model
    model = make_conv2d(in_channels=3, out_channels=16, num_layers=3, kernel_size=3)
    print(f"\nModel: {model}")

    # Create input
    input_tensor = torch.randn(32, 3, 64, 64)
    print(f"Input shape: {input_tensor.shape}")

    # Trace and propagate shapes
    print("\n1. Tracing model with FX...")
    fx_graph = symbolic_trace(model)

    print("2. Propagating shapes...")
    shape_prop = ShapeProp(fx_graph)
    shape_prop.propagate(input_tensor)

    # Print graph nodes
    print("\n3. Graph nodes:")
    for i, node in enumerate(fx_graph.graph.nodes):
        if node.op == 'call_module':
            mod = fx_graph.get_submodule(node.target)
            meta = node.meta.get('tensor_meta')
            shape = meta.shape if meta else "No shape"
            print(f"   [{i}] {node.name}: {type(mod).__name__} -> {shape}")

    # Test characterization
    print("\n4. Running characterization...")
    registry = default_registry()
    walker = FXGraphWalker(cpu_profile, registry)
    metrics = walker.walk(fx_graph)

    print("\n" + "=" * 60)
    print("Results:")
    print("=" * 60)
    for key, value in metrics.items():
        if key in ['FLOPs', 'Memory']:
            print(f"{key:12s}: {value:,}")
        else:
            print(f"{key:12s}: {value}")

    # Verify non-zero
    if metrics['FLOPs'] > 0:
        print("\n✓ SUCCESS: Conv2D characterization working!")
    else:
        print("\n✗ FAILED: Conv2D still producing zeros")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
