#!/usr/bin/env python
"""Quick test script to validate Conv2D characterization"""

import torch
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from torch.fx import symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp

from src.graphs.subgraphs.conv2d_stack import make_conv2d
from src.graphs.transform.partitioning import FusionBasedPartitioner
from src.graphs.hardware.mappers.cpu import create_intel_cpu_mapper
from src.graphs.hardware.resource_model import Precision

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

    # Partition the graph
    partitioner = FusionBasedPartitioner()
    fusion_report = partitioner.partition(fx_graph)

    # Extract execution stages
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

    execution_stages = extract_execution_stages(fusion_report)

    # Create CPU mapper and run mapping
    cpu_mapper = create_intel_cpu_mapper("avx512")
    allocation = cpu_mapper.map_graph(
        fusion_report=fusion_report,
        execution_stages=execution_stages,
        batch_size=32,
        precision=Precision.FP32
    )

    print("\n" + "=" * 60)
    print("Results:")
    print("=" * 60)
    print(f"FLOPs       : {fusion_report.total_flops:,}")
    print(f"Subgraphs   : {fusion_report.total_subgraphs}")
    print(f"Stages      : {allocation.total_execution_stages}")
    print(f"Latency     : {allocation.total_latency:.6f} seconds")
    print(f"Energy      : {allocation.total_energy:.6f} Joules")
    print(f"Utilization : {allocation.average_utilization:.1%}")

    # Verify non-zero
    if fusion_report.total_flops > 0:
        print("\n✓ SUCCESS: Conv2D characterization working!")
    else:
        print("\n✗ FAILED: Conv2D still producing zeros")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
