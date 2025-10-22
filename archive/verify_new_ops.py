#!/usr/bin/env python
"""
Verify New Operations Support
==============================

Confirms that our partitioner now handles MaxPool and Add operations
"""

import torch
import torchvision.models as models
from torch.fx import symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp
import sys
sys.path.insert(0, 'src')

from graphs.characterize.graph_partitioner import GraphPartitioner

# Load ResNet-18
print("=" * 80)
print("VERIFYING NEW OPERATIONS SUPPORT")
print("=" * 80)

model = models.resnet18(weights=None)
model.eval()
input_tensor = torch.randn(1, 3, 224, 224)

# Trace
fx_graph = symbolic_trace(model)
ShapeProp(fx_graph).propagate(input_tensor)

# Partition
partitioner = GraphPartitioner()
report = partitioner.partition(fx_graph)

# Count operations by type
op_counts = {}
for sg in report.subgraphs:
    op_type = sg.operation_type.value
    op_counts[op_type] = op_counts.get(op_type, 0) + 1

print("\nOperations Partitioned:")
print(f"{'Operation Type':<25} {'Count':<10} {'Total FLOPs':<15} {'Total MACs':<15}")
print("-" * 70)

# Group by operation type
flops_by_type = {}
macs_by_type = {}
for sg in report.subgraphs:
    op_type = sg.operation_type.value
    flops_by_type[op_type] = flops_by_type.get(op_type, 0) + sg.flops
    macs_by_type[op_type] = macs_by_type.get(op_type, 0) + sg.macs

for op_type in sorted(op_counts.keys()):
    count = op_counts[op_type]
    flops = flops_by_type[op_type]
    macs = macs_by_type[op_type]

    flops_str = f"{flops / 1e9:.3f}G" if flops > 1e9 else f"{flops / 1e6:.1f}M"
    macs_str = f"{macs / 1e9:.3f}G" if macs > 1e9 else f"{macs / 1e6:.1f}M"

    print(f"{op_type:<25} {count:<10} {flops_str:<15} {macs_str:<15}")

print("\n" + "=" * 80)
print("SPECIFIC OPERATION DETAILS")
print("=" * 80)

# Check for MaxPool
maxpool_ops = [sg for sg in report.subgraphs if sg.operation_type.value == 'maxpool']
print(f"\nMaxPool operations: {len(maxpool_ops)}")
for sg in maxpool_ops:
    print(f"  {sg.node_name}: FLOPs={sg.flops:,}, shape={sg.parallelism.total_threads if sg.parallelism else 'N/A'}")

# Check for AdaptiveAvgPool
adaptiveavgpool_ops = [sg for sg in report.subgraphs if sg.operation_type.value == 'adaptiveavgpool']
print(f"\nAdaptiveAvgPool operations: {len(adaptiveavgpool_ops)}")
for sg in adaptiveavgpool_ops:
    print(f"  {sg.node_name}: FLOPs={sg.flops:,}, shape={sg.parallelism.total_threads if sg.parallelism else 'N/A'}")

# Check for elementwise (add) operations
elementwise_ops = [sg for sg in report.subgraphs if sg.operation_type.value == 'elementwise']
print(f"\nElementwise (Add) operations: {len(elementwise_ops)}")
for sg in elementwise_ops:
    print(f"  {sg.node_name}: FLOPs={sg.flops:,} ({sg.flops / 1e6:.2f}M)")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print(f"\nTotal subgraphs: {len(report.subgraphs)}")
print(f"Total FLOPs: {report.total_flops / 1e9:.3f} GFLOPs")
print(f"Total MACs: {report.total_macs / 1e9:.3f} GMACs")

# Check if we found the expected operations
expected_ops = {
    'maxpool': 1,
    'adaptiveavgpool': 1,
    'elementwise': 8
}

print("\nExpected vs Actual:")
for op, expected_count in expected_ops.items():
    actual_count = op_counts.get(op, 0)
    status = "✓" if actual_count == expected_count else "✗"
    print(f"  {status} {op}: expected {expected_count}, got {actual_count}")
