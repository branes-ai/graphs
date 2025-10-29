#!/usr/bin/env python3
"""
Demonstration: Decomposed Multi-Head Attention for Enhanced Fusion

This script demonstrates the benefits of using DecomposedMultiheadAttention
instead of standard nn.MultiheadAttention for graph fusion optimization.

Key Results Expected:
- Standard attention: 2 ops (LayerNorm → MultiheadAttention), ~5.7% memory reduction
- Decomposed attention: 16+ ops, ~45% memory reduction (8× improvement!)

Usage:
    python examples/demo_decomposed_attention.py
"""

import torch
import torch.fx
from torch.fx.passes.shape_prop import ShapeProp

from graphs.subgraphs.attention import make_attention_block
from graphs.transform.partitioning import GraphPartitioner
from graphs.analysis import MemoryEstimator
from graphs.hardware.resource_model import (
    HardwareResourceModel,
    HardwareType,
    Precision,
    PrecisionProfile,
)


def create_gpu_hardware():
    """Create a realistic GPU hardware model (H100-like)"""
    return HardwareResourceModel(
        name="H100",
        hardware_type=HardwareType.GPU,
        compute_units=132,  # 132 SMs
        threads_per_unit=2048,
        warps_per_unit=64,
        warp_size=32,
        precision_profiles={
            Precision.FP32: PrecisionProfile(
                precision=Precision.FP32,
                peak_ops_per_sec=51e12,  # 51 TFLOPS FP32
                tensor_core_supported=False,
                relative_speedup=1.0,
                bytes_per_element=4,
                accumulator_precision=Precision.FP32,
            ),
            Precision.FP16: PrecisionProfile(
                precision=Precision.FP16,
                peak_ops_per_sec=989e12,  # 989 TFLOPS FP16 with tensor cores
                tensor_core_supported=True,
                relative_speedup=19.4,
                bytes_per_element=2,
                accumulator_precision=Precision.FP32,
            ),
        },
        default_precision=Precision.FP32,
        peak_bandwidth=3352e9,  # 3352 GB/s (HBM3)
        l1_cache_per_unit=256 * 1024,  # 256 KB per SM
        l2_cache_total=60 * 1024 * 1024,  # 60 MB L2
        main_memory=80 * 1024**3,  # 80 GB HBM3
        energy_per_flop_fp32=20e-12,  # 20 pJ/FLOP
        energy_per_byte=10e-12,  # 10 pJ/byte
    )


def trace_and_analyze(model, input_tensor, model_name, hardware, verbose=True):
    """
    Trace model, partition graph, and analyze memory usage.

    Args:
        model: PyTorch model to analyze
        input_tensor: Input tensor for shape propagation
        model_name: Name for display
        verbose: If True, print detailed information

    Returns:
        dict: Analysis results
    """
    print(f"\n{'='*80}")
    print(f"Analyzing: {model_name}")
    print(f"{'='*80}")

    # Step 1: FX Tracing
    if verbose:
        print("\n[Step 1] FX Tracing...")

    try:
        traced = torch.fx.symbolic_trace(model)
    except Exception as e:
        print(f"  ERROR: FX tracing failed: {e}")
        return None

    # Step 2: Shape Propagation
    if verbose:
        print("[Step 2] Shape Propagation...")

    try:
        ShapeProp(traced).propagate(input_tensor)
    except Exception as e:
        print(f"  ERROR: Shape propagation failed: {e}")
        return None

    # Count nodes in traced graph
    num_nodes = len([n for n in traced.graph.nodes if n.op in ['call_function', 'call_method', 'call_module']])

    if verbose:
        print(f"  ✓ Graph traced successfully")
        print(f"  ✓ Total nodes: {num_nodes}")

    # Step 3: Graph Partitioning
    if verbose:
        print("\n[Step 3] Graph Partitioning...")

    partitioner = GraphPartitioner()

    try:
        partition_report = partitioner.partition(traced)
    except Exception as e:
        print(f"  ERROR: Partitioning failed: {e}")
        return None

    if verbose:
        print(f"  ✓ Created {len(partition_report.subgraphs)} subgraphs")
        print(f"  ✓ Total FLOPs: {sum(sg.flops for sg in partition_report.subgraphs):,}")

    # Step 4: Memory Analysis
    if verbose:
        print("\n[Step 4] Memory Analysis...")

    memory_estimator = MemoryEstimator(hardware)

    try:
        memory_report = memory_estimator.estimate_memory(
            partition_report.subgraphs,
            partition_report
        )
    except Exception as e:
        print(f"  ERROR: Memory estimation failed: {e}")
        return None

    peak_mb = memory_report.peak_memory_bytes / (1024 * 1024)
    activation_mb = memory_report.activation_memory_bytes / (1024 * 1024)
    weight_mb = memory_report.weight_memory_bytes / (1024 * 1024)

    if verbose:
        print(f"  ✓ Peak memory: {peak_mb:.2f} MB")
        print(f"    - Activations: {activation_mb:.2f} MB ({activation_mb/peak_mb*100:.1f}%)")
        print(f"    - Weights: {weight_mb:.2f} MB ({weight_mb/peak_mb*100:.1f}%)")

    # Calculate memory reduction from fusion
    # Estimate: without fusion, all intermediate tensors would be live
    total_intermediate_bytes = sum(sg.total_output_bytes for sg in partition_report.subgraphs)
    unfused_peak = memory_report.weight_memory_bytes + total_intermediate_bytes
    memory_reduction_pct = (1 - peak_mb * 1024 * 1024 / unfused_peak) * 100 if unfused_peak > 0 else 0

    if verbose:
        print(f"\n  Memory Reduction from Fusion: {memory_reduction_pct:.1f}%")

    return {
        'num_nodes': num_nodes,
        'num_subgraphs': len(partition_report.subgraphs),
        'total_flops': partition_report.total_flops,
        'peak_memory_mb': peak_mb,
        'activation_memory_mb': activation_mb,
        'weight_memory_mb': weight_mb,
        'memory_reduction_pct': memory_reduction_pct,
        'partition_report': partition_report,
        'memory_report': memory_report,
    }


def print_comparison_table(standard_results, decomposed_results):
    """Print comparison table between standard and decomposed attention."""

    print("\n" + "="*80)
    print("COMPARISON: Standard vs Decomposed Attention")
    print("="*80)

    if standard_results is None or decomposed_results is None:
        print("  ERROR: Cannot compare - one or both analyses failed")
        return

    print(f"\n{'Metric':<40} {'Standard':<20} {'Decomposed':<20} {'Improvement'}")
    print("-"*80)

    # Number of traced nodes
    std_nodes = standard_results['num_nodes']
    dec_nodes = decomposed_results['num_nodes']
    node_ratio = dec_nodes / std_nodes if std_nodes > 0 else 0
    print(f"{'FX Graph Nodes':<40} {std_nodes:<20} {dec_nodes:<20} {node_ratio:.1f}×")

    # Number of fused subgraphs
    std_subgraphs = standard_results['num_subgraphs']
    dec_subgraphs = decomposed_results['num_subgraphs']
    subgraph_ratio = dec_subgraphs / std_subgraphs if std_subgraphs > 0 else 0
    print(f"{'Fused Subgraphs':<40} {std_subgraphs:<20} {dec_subgraphs:<20} {subgraph_ratio:.1f}×")

    # Total FLOPs
    std_flops = standard_results['total_flops']
    dec_flops = decomposed_results['total_flops']
    flops_ratio = dec_flops / std_flops if std_flops > 0 else 0
    print(f"{'Total FLOPs':<40} {std_flops:,}{'':<13} {dec_flops:,}{'':<13} {flops_ratio:.1f}×")

    # Peak memory
    std_mem = standard_results['peak_memory_mb']
    dec_mem = decomposed_results['peak_memory_mb']
    mem_ratio = std_mem / dec_mem if dec_mem > 0 else 0
    print(f"{'Peak Memory (MB)':<40} {std_mem:.2f}{'':<16} {dec_mem:.2f}{'':<16} {mem_ratio:.2f}×")

    # Memory reduction from fusion
    std_reduction = standard_results['memory_reduction_pct']
    dec_reduction = decomposed_results['memory_reduction_pct']
    reduction_improvement = dec_reduction / std_reduction if std_reduction > 0 else 0
    print(f"{'Memory Reduction from Fusion (%)':<40} {std_reduction:.1f}%{'':<16} {dec_reduction:.1f}%{'':<16} {reduction_improvement:.1f}×")

    print("\n" + "="*80)
    print("KEY INSIGHTS:")
    print("="*80)

    if node_ratio > 1:
        print(f"✓ Decomposed attention exposes {node_ratio:.1f}× more operations to the partitioner")

    if subgraph_ratio > 1:
        print(f"✓ Fusion creates {subgraph_ratio:.1f}× more fused subgraphs (better optimization)")

    if dec_reduction > std_reduction:
        print(f"✓ Memory reduction improved from {std_reduction:.1f}% to {dec_reduction:.1f}% ({reduction_improvement:.1f}× better!)")

    if dec_reduction >= 30:
        print(f"✓ SUCCESS: Achieved target of >30% memory reduction ({dec_reduction:.1f}%)")
    else:
        print(f"⚠ Target not met: {dec_reduction:.1f}% < 30% (may need additional fusion patterns)")


def visualize_subgraph_breakdown(results, title):
    """Visualize subgraph breakdown with ASCII art."""

    if results is None:
        return

    partition_report = results['partition_report']

    print(f"\n{'='*80}")
    print(f"{title} - Subgraph Breakdown")
    print(f"{'='*80}")

    print(f"\n{'Subgraph':<12} {'Operations':<40} {'FLOPs':<15} {'Memory (KB)'}")
    print("-"*80)

    for i, sg in enumerate(partition_report.subgraphs[:10]):  # Show first 10
        ops_str = str(sg.operation_type)

        flops_str = f"{sg.flops:,}" if sg.flops > 0 else "0"
        mem_kb = sg.total_output_bytes / 1024

        print(f"SG-{i:<9} {ops_str:<40} {flops_str:<15} {mem_kb:.1f}")

    if len(partition_report.subgraphs) > 10:
        print(f"... ({len(partition_report.subgraphs) - 10} more subgraphs)")


def main():
    """Main demonstration function."""

    print("="*80)
    print("ENHANCED ATTENTION FUSION - PHASE 1 PROOF OF CONCEPT")
    print("="*80)
    print("\nThis demo compares standard vs decomposed multi-head attention to show")
    print("the benefits of explicit operation decomposition for graph fusion.")
    print("\nTarget: >30% memory reduction (vs current ~5.7%)")

    # Configuration
    batch_size = 4
    seq_len = 196  # ViT-Base patch count (14×14)
    embed_dim = 768  # ViT-Base embedding dimension
    num_heads = 12  # ViT-Base number of attention heads

    print(f"\nConfiguration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Embedding dimension: {embed_dim}")
    print(f"  Number of heads: {num_heads}")

    # Create input tensor
    input_tensor = torch.randn(batch_size, seq_len, embed_dim)

    # Create hardware model for memory analysis
    hardware = create_gpu_hardware()

    # Test 1: Standard MultiheadAttention (baseline)
    print("\n" + "="*80)
    print("TEST 1: Standard nn.MultiheadAttention (Baseline)")
    print("="*80)

    standard_model = make_attention_block(
        embed_dim=embed_dim,
        num_heads=num_heads,
        dropout=0.1,
        use_decomposed=False,  # Use standard attention
    )

    standard_results = trace_and_analyze(
        standard_model,
        input_tensor,
        "Standard Attention Block",
        hardware,
        verbose=True,
    )

    # Test 2: Decomposed MultiheadAttention
    print("\n" + "="*80)
    print("TEST 2: DecomposedMultiheadAttention (Enhanced)")
    print("="*80)

    decomposed_model = make_attention_block(
        embed_dim=embed_dim,
        num_heads=num_heads,
        dropout=0.1,
        use_decomposed=True,  # Use decomposed attention
    )

    decomposed_results = trace_and_analyze(
        decomposed_model,
        input_tensor,
        "Decomposed Attention Block",
        hardware,
        verbose=True,
    )

    # Compare results
    print_comparison_table(standard_results, decomposed_results)

    # Visualize subgraph breakdown
    visualize_subgraph_breakdown(standard_results, "Standard Attention")
    visualize_subgraph_breakdown(decomposed_results, "Decomposed Attention")

    print("\n" + "="*80)
    print("DEMONSTRATION COMPLETE")
    print("="*80)
    print("\nNext Steps:")
    print("  1. Extend fusion patterns to better handle attention operations")
    print("  2. Add support for parallel fusion (Q, K, V projections)")
    print("  3. Test on full transformer models (ViT, BERT, etc.)")
    print("  4. Validate accuracy (ensure outputs match standard attention)")


if __name__ == "__main__":
    main()
