#!/usr/bin/env python
"""Characterization test for ResNet-18 from torchvision"""

import torch
import torchvision.models as models
from torch.fx import symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.graphs.transform.partitioning import FusionBasedPartitioner
from src.graphs.hardware.mappers.cpu import create_intel_cpu_mapper, create_amd_cpu_mapper
from src.graphs.hardware.mappers.gpu import create_h100_pcie_80gb_mapper
from src.graphs.hardware.mappers.accelerators.tpu import create_tpu_v4_mapper
from src.graphs.hardware.mappers.accelerators.kpu import create_kpu_t64_mapper
from src.graphs.hardware.resource_model import Precision

def format_number(n):
    """Format large numbers with SI prefixes"""
    if n >= 1e12:
        return f"{n/1e12:.2f}T"
    elif n >= 1e9:
        return f"{n/1e9:.2f}G"
    elif n >= 1e6:
        return f"{n/1e6:.2f}M"
    elif n >= 1e3:
        return f"{n/1e3:.2f}K"
    else:
        return f"{n:.2f}"

def analyze_graph_structure(fx_graph):
    """Analyze the FX graph structure"""
    module_lookup = dict(fx_graph.named_modules())

    # Count different node types
    node_types = {}
    for node in fx_graph.graph.nodes:
        if node.op == 'call_module':
            mod = module_lookup.get(node.target)
            if mod:
                mod_type = type(mod).__name__
                node_types[mod_type] = node_types.get(mod_type, 0) + 1

    return node_types

def main():
    print("=" * 70)
    print("ResNet-18 Characterization Test")
    print("=" * 70)

    # Load ResNet-18
    print("\n1. Loading ResNet-18 from torchvision...")
    model = models.resnet18(pretrained=False)
    model.eval()

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params:,} ({format_number(total_params)})")

    # Create input (standard ImageNet size)
    batch_size = 1
    input_tensor = torch.randn(batch_size, 3, 224, 224)
    print(f"   Input shape: {list(input_tensor.shape)}")

    # FX Trace
    print("\n2. Tracing model with FX...")
    try:
        fx_graph = symbolic_trace(model)
        print("   ✓ FX tracing successful")
    except Exception as e:
        print(f"   ✗ FX tracing failed: {e}")
        return 1

    # Shape propagation
    print("\n3. Propagating shapes...")
    try:
        shape_prop = ShapeProp(fx_graph)
        shape_prop.propagate(input_tensor)
        print("   ✓ Shape propagation successful")
    except Exception as e:
        print(f"   ✗ Shape propagation failed: {e}")
        return 1

    # Analyze graph structure
    print("\n4. Analyzing graph structure...")
    node_types = analyze_graph_structure(fx_graph)
    print("   Layer types found:")
    for layer_type, count in sorted(node_types.items(), key=lambda x: -x[1]):
        print(f"      {layer_type:20s}: {count:3d}")

    # Show first few nodes
    print("\n5. First 10 graph nodes:")
    module_lookup = dict(fx_graph.named_modules())
    call_module_nodes = [n for n in fx_graph.graph.nodes if n.op == 'call_module']
    for i, node in enumerate(call_module_nodes[:10]):
        mod = module_lookup.get(node.target)
        meta = node.meta.get('tensor_meta')
        shape = list(meta.shape) if meta else "No shape"
        print(f"   [{i:2d}] {node.name:25s}: {type(mod).__name__:15s} → {shape}")

    # Characterization across architectures
    print("\n6. Running characterization across architectures...")
    print("-" * 70)

    # Partition the graph using FusionBasedPartitioner
    partitioner = FusionBasedPartitioner()
    fusion_report = partitioner.partition(fx_graph)

    # Extract execution stages (simple sequential grouping)
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

    print(f"   Partitioned into {fusion_report.total_subgraphs} fused subgraphs")
    print(f"   Total FLOPs: {format_number(fusion_report.total_flops)}")

    # Create hardware mappers
    mappers = {
        "CPU (Intel)": create_intel_cpu_mapper("avx512"),
        "CPU (AMD)": create_amd_cpu_mapper(),
        "GPU (H100)": create_h100_pcie_80gb_mapper(),
        "TPU (v4)": create_tpu_v4_mapper(),
        "Stillwater KPU-T64": create_kpu_t64_mapper(),
    }

    results = []
    for arch_name, mapper in mappers.items():
        try:
            allocation = mapper.map_graph(
                fusion_report=fusion_report,
                execution_stages=execution_stages,
                batch_size=batch_size,
                precision=Precision.FP32
            )

            results.append((arch_name, allocation))

            print(f"\n{arch_name}:")
            print(f"   FLOPs:    {format_number(fusion_report.total_flops):>10s} ({fusion_report.total_flops:,})")
            print(f"   Latency:  {allocation.total_latency:>10.6f} seconds")
            print(f"   Energy:   {allocation.total_energy:>10.6f} Joules")
            print(f"   Util:     {allocation.average_utilization:>10.1%}")
        except Exception as e:
            print(f"\n{arch_name}: FAILED - {e}")

    # Theoretical comparison for ResNet-18
    print("\n" + "=" * 70)
    print("Theoretical Comparison")
    print("=" * 70)
    print("\nResNet-18 theoretical values (ImageNet, batch=1):")
    print("   FLOPs:    ~1.8 GFLOPs")
    print("   Params:   ~11.7M")
    print(f"   Our FLOPs: {format_number(fusion_report.total_flops)}")
    print(f"   Our Params: {format_number(total_params)}")

    # Check if we got reasonable results
    if fusion_report.total_flops > 1e9:  # At least 1 GFLOP
        print("\n✓ SUCCESS: ResNet-18 characterization produced reasonable results!")

        # Speedup comparison
        print("\n" + "=" * 70)
        print("Architecture Speedup Comparison (vs CPU)")
        print("=" * 70)

        # Get CPU baseline latency (first CPU result)
        cpu_latency = None
        for arch_name, allocation in results:
            if "CPU" in arch_name:
                cpu_latency = allocation.total_latency
                break

        if cpu_latency:
            for arch_name, allocation in results:
                speedup = cpu_latency / allocation.total_latency
                print(f"   {arch_name:15s}: {speedup:>8.2f}×")

        return 0
    else:
        print("\n✗ WARNING: FLOPs seem too low, check fusion patterns")
        return 1

if __name__ == "__main__":
    exit(main())
