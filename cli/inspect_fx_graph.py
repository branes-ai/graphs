#!/usr/bin/env python
"""
Inspect FX Graph for Unsupported Operations
============================================

Shows what types of nodes exist in the FX graph and identifies
the unsupported operators.
"""

import torch
import torchvision.models as models
from torch.fx import symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp

# Load ResNet-18
print("=" * 80)
print("FX GRAPH NODE ANALYSIS")
print("=" * 80)

model = models.resnet18(weights=None)
model.eval()
input_tensor = torch.randn(1, 3, 224, 224)

# Trace
fx_graph = symbolic_trace(model)
ShapeProp(fx_graph).propagate(input_tensor)

# Analyze nodes by type
nodes_by_op = {}
for node in fx_graph.graph.nodes:
    op = node.op
    if op not in nodes_by_op:
        nodes_by_op[op] = []
    nodes_by_op[op].append(node)

print(f"\nTotal nodes: {len(fx_graph.graph.nodes)}")
print("\nNodes by operation type:")
for op, nodes in nodes_by_op.items():
    print(f"  {op:<20}: {len(nodes)} nodes")

# Look for call_function nodes (like add)
print("\n" + "=" * 80)
print("CALL_FUNCTION NODES")
print("=" * 80)

if 'call_function' in nodes_by_op:
    for node in nodes_by_op['call_function']:
        target_name = node.target.__name__ if hasattr(node.target, '__name__') else str(node.target)
        print(f"\nNode: {node.name}")
        print(f"  Target: {target_name}")
        print(f"  Args: {[str(arg)[:50] for arg in node.args]}")
        if hasattr(node, 'meta') and 'tensor_meta' in node.meta:
            meta = node.meta['tensor_meta']
            print(f"  Output shape: {meta.shape}")

# Look for call_method nodes (like add_)
print("\n" + "=" * 80)
print("CALL_METHOD NODES")
print("=" * 80)

if 'call_method' in nodes_by_op:
    for node in nodes_by_op['call_method']:
        print(f"\nNode: {node.name}")
        print(f"  Method: {node.target}")
        print(f"  Args: {[str(arg)[:50] for arg in node.args]}")
        if hasattr(node, 'meta') and 'tensor_meta' in node.meta:
            meta = node.meta['tensor_meta']
            print(f"  Output shape: {meta.shape}")

# Look for MaxPool specifically
print("\n" + "=" * 80)
print("MAXPOOL NODES")
print("=" * 80)

for node in fx_graph.graph.nodes:
    if 'pool' in node.name.lower() or (hasattr(node, 'target') and 'pool' in str(node.target).lower()):
        print(f"\nNode: {node.name}")
        print(f"  Op: {node.op}")
        print(f"  Target: {node.target}")
        if node.op == 'call_module':
            module = fx_graph.get_submodule(node.target)
            print(f"  Module type: {type(module)}")
            if hasattr(module, 'kernel_size'):
                print(f"  Kernel size: {module.kernel_size}")
            if hasattr(module, 'stride'):
                print(f"  Stride: {module.stride}")
        if hasattr(node, 'meta') and 'tensor_meta' in node.meta:
            meta = node.meta['tensor_meta']
            print(f"  Output shape: {meta.shape}")

# Look for add operations
print("\n" + "=" * 80)
print("ADD OPERATIONS")
print("=" * 80)

add_count = 0
for node in fx_graph.graph.nodes:
    if 'add' in node.name.lower() or (node.op == 'call_function' and hasattr(node.target, '__name__') and 'add' in node.target.__name__):
        add_count += 1
        print(f"\nNode: {node.name}")
        print(f"  Op: {node.op}")
        if node.op == 'call_function':
            print(f"  Function: {node.target.__name__}")
        elif node.op == 'call_method':
            print(f"  Method: {node.target}")
        print(f"  Args: {len(node.args)} arguments")
        if hasattr(node, 'meta') and 'tensor_meta' in node.meta:
            meta = node.meta['tensor_meta']
            print(f"  Output shape: {meta.shape}")

print(f"\nTotal add operations found: {add_count}")
