"""
Simple KPU test to debug issues
"""

import torch
import torch.fx
from torch.fx.passes.shape_prop import ShapeProp
from torchvision import models
from typing import List

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.graphs.transform.partitioning import FusionBasedPartitioner
from src.graphs.hardware.mappers.accelerators.kpu import create_kpu_t64_mapper
from src.graphs.hardware.resource_model import Precision


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


# Load model
print("Loading ResNet-18...")
model = models.resnet18(pretrained=False)
model.eval()

# Trace
print("Tracing...")
input_tensor = torch.randn(1, 3, 224, 224)
fx_graph = torch.fx.symbolic_trace(model)
ShapeProp(fx_graph).propagate(input_tensor)

# Partition
print("Partitioning...")
partitioner = FusionBasedPartitioner()
fusion_report = partitioner.partition(fx_graph)
execution_stages = extract_execution_stages(fusion_report)

print(f"Subgraphs: {len(fusion_report.fused_subgraphs)}")
print(f"Stages: {len(execution_stages)}")

# Create KPU mapper
print("\nCreating KPU mapper...")
kpu_mapper = create_kpu_t64_mapper()
print(f"KPU model: {kpu_mapper.resource_model.name}")
print(f"Precision profiles available: {list(kpu_mapper.resource_model.precision_profiles.keys())}")

# Try to map with INT8
print("\nMapping with INT8...")
try:
    allocation = kpu_mapper.map_graph(
        fusion_report=fusion_report,
        execution_stages=execution_stages,
        batch_size=1,
        precision=Precision.INT8
    )
    print(f"SUCCESS!")
    print(f"Latency: {allocation.total_latency*1000:.3f} ms")
    print(f"Energy: {allocation.total_energy:.3f} J")
    print(f"Utilization: {allocation.average_utilization:.1%}")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()

# Try with FP32
print("\nMapping with FP32...")
try:
    allocation = kpu_mapper.map_graph(
        fusion_report=fusion_report,
        execution_stages=execution_stages,
        batch_size=1,
        precision=Precision.FP32
    )
    print(f"SUCCESS!")
    print(f"Latency: {allocation.total_latency*1000:.3f} ms")
    print(f"Energy: {allocation.total_energy:.3f} J")
    print(f"Utilization: {allocation.average_utilization:.1%}")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
