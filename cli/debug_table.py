#!/usr/bin/env python
"""Debug table formatter to see what paths are generated"""

import torch
import torchvision.models as models
from torch.fx import symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp
import sys
sys.path.insert(0, 'src')

from graphs.characterize.graph_partitioner import GraphPartitioner
from graphs.characterize.table_formatter import HierarchicalTableFormatter

# Load model
model = models.resnet18(weights=None)
model.eval()
input_tensor = torch.randn(1, 3, 224, 224)

# Trace and partition
fx_graph = symbolic_trace(model)
ShapeProp(fx_graph).propagate(input_tensor)
partitioner = GraphPartitioner()
report = partitioner.partition(fx_graph)

# Create formatter and build hierarchy
formatter = HierarchicalTableFormatter()

# Debug: Check what parent paths would be generated
print("Modules before aggregation:")
print(f"  Total: {len(formatter.module_stats)} modules\n")

formatter._build_module_hierarchy(fx_graph, report)

# Debug aggregation
print("\nModules after aggregation:")
print(f"  Total: {len(formatter.module_stats)} modules")

# Print layer-related paths
print("\nLayer-related paths:")
for path in sorted(formatter.module_stats.keys()):
    if 'layer' in path:
        level = path.count('.')
        stats = formatter.module_stats[path]
        print(f"  {'  ' * level}{path} (level={stats.level}, dots={level})")

# Check for parent paths
print("\nLooking for layer parents (no dots):")
for path in formatter.module_stats.keys():
    if path.startswith('layer') and '.' not in path:
        stats = formatter.module_stats[path]
        print(f"  Found: {path} (params={stats.parameters}, macs={stats.macs})")
