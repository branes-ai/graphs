#!/usr/bin/env python
"""
Debug FVCore by_module() Output
================================

Investigates what fvcore.by_module() returns to understand the matching issue
"""

import torch
import torchvision.models as models
from fvcore.nn import FlopCountAnalysis

# Load ResNet-18
model = models.resnet18(weights=None)
model.eval()
input_tensor = torch.randn(1, 3, 224, 224)

# Get fvcore analysis
print("=" * 80)
print("FVCORE by_module() INVESTIGATION")
print("=" * 80)

flops_analysis = FlopCountAnalysis(model, input_tensor)
total_flops = flops_analysis.total()

print(f"\nTotal FLOPs: {total_flops / 1e9:.3f} GFLOPs")

# Get by_module breakdown
by_module = flops_analysis.by_module()

print(f"\nType of by_module: {type(by_module)}")
print(f"Number of entries: {len(by_module)}")

print("\n" + "=" * 80)
print("FIRST 20 ENTRIES FROM by_module()")
print("=" * 80)
print(f"{'Module Name':<50} {'FLOPs':<15}")
print("-" * 65)

for i, (name, flops) in enumerate(by_module.items()):
    if i >= 20:
        break
    print(f"{name:<50} {flops / 1e6:>10.2f}M")

print("\n" + "=" * 80)
print("SEARCHING FOR SPECIFIC LAYERS")
print("=" * 80)

# Check if specific layers exist
test_names = ['conv1', 'layer1.0.conv1', 'layer1.0.bn1', 'fc', 'layer2.0.downsample.0']

for test_name in test_names:
    print(f"\nSearching for '{test_name}':")
    found = False
    for fv_name, fv_flops in by_module.items():
        if test_name in fv_name:
            print(f"  Found: {fv_name} -> {fv_flops / 1e6:.2f}M FLOPs")
            found = True
    if not found:
        print(f"  NOT FOUND")

print("\n" + "=" * 80)
print("CHECKING WHAT CONTAINS 'conv1'")
print("=" * 80)

for fv_name, fv_flops in by_module.items():
    if 'conv1' in fv_name.lower():
        print(f"{fv_name:<50} {fv_flops / 1e6:>10.2f}M")

print("\n" + "=" * 80)
print("ALL MODULE NAMES (first 50)")
print("=" * 80)

for i, name in enumerate(by_module.keys()):
    if i >= 50:
        print(f"... and {len(by_module) - 50} more")
        break
    print(f"{i+1:3}. {name}")
