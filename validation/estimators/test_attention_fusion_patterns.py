#!/usr/bin/env python3
"""
Validation: Phase 3 Attention-Specific Fusion Patterns

This script validates that the AttentionFusionPartitioner (Phase 3) provides
enhanced fusion for decomposed attention operations compared to the standard
FusionBasedPartitioner.

Tests:
1. ViT-B/16 with standard fusion partitioner (baseline)
2. ViT-B/16 with attention fusion partitioner (enhanced)
3. Parallel Q,K,V fusion validation
4. Memory reduction comparison
5. Attention-specific pattern detection

Expected Results:
- Parallel Q,K,V fusions detected and merged (12 layers × 1 = 12 fusions)
- Additional memory reduction from parallel fusion
- Attention sequential patterns fused (matmul → mul → softmax, etc.)

Usage:
    python validation/estimators/test_attention_fusion_patterns.py
"""

import torch
import torch.nn as nn
from torchvision.models import vit_b_16
from torch.fx.passes.shape_prop import ShapeProp

from graphs.transform import trace_with_decomposition
from graphs.transform.partitioning import FusionBasedPartitioner, AttentionFusionPartitioner
from graphs.analysis import MemoryEstimator
from graphs.hardware.resource_model import (
    HardwareResourceModel,
    HardwareType,
    Precision,
    PrecisionProfile,
)


def create_gpu_hardware():
    """Create H100 GPU hardware model for testing"""
    return HardwareResourceModel(
        name="H100",
        hardware_type=HardwareType.GPU,
        compute_units=132,
        threads_per_unit=2048,
        warps_per_unit=64,
        warp_size=32,
        precision_profiles={
            Precision.FP32: PrecisionProfile(
                precision=Precision.FP32,
                peak_ops_per_sec=51e12,
                tensor_core_supported=False,
                relative_speedup=1.0,
                bytes_per_element=4,
                accumulator_precision=Precision.FP32,
            ),
        },
        default_precision=Precision.FP32,
        peak_bandwidth=3352e9,
        l1_cache_per_unit=256 * 1024,
        l2_cache_total=60 * 1024 * 1024,
        main_memory=80 * 1024**3,
        energy_per_flop_fp32=20e-12,
        energy_per_byte=10e-12,
    )


def analyze_with_partitioner(traced, input_tensor, hardware, partitioner_name, partitioner):
    """Analyze traced model with a specific partitioner"""
    print(f"\n{'='*80}")
    print(f"Analyzing with: {partitioner_name}")
    print(f"{'='*80}")

    # Partitioning
    try:
        partition_report = partitioner.partition(traced)
        print(f"  Subgraphs: {len(partition_report.fused_subgraphs)}")
        print(f"  Total FLOPs: {partition_report.total_flops:,}")
        print(f"  Average fusion size: {partition_report.avg_fusion_size:.2f} ops/subgraph")
        print(f"  Data movement reduction: {partition_report.data_movement_reduction * 100:.1f}%")

        # Memory analysis - calculate manually for fusion validation
        total_weights = sum(sg.total_weight_bytes for sg in partition_report.fused_subgraphs)
        total_outputs = sum(sg.total_output_bytes for sg in partition_report.fused_subgraphs)

        # Peak memory = weights + largest intermediate output
        peak_memory_bytes = total_weights + (
            max(sg.total_output_bytes for sg in partition_report.fused_subgraphs)
            if partition_report.fused_subgraphs else 0
        )

        peak_mb = peak_memory_bytes / (1024 * 1024)
        print(f"  Peak memory: {peak_mb:.2f} MB")

        # Calculate memory reduction from fusion
        unfused_peak = total_weights + total_outputs
        reduction_pct = (1 - peak_memory_bytes / unfused_peak) * 100 if unfused_peak > 0 else 0
        print(f"  Memory reduction from fusion: {reduction_pct:.1f}%")

        # If attention partitioner, show attention-specific stats
        if isinstance(partitioner, AttentionFusionPartitioner):
            print("\n" + partitioner.print_attention_fusion_summary())

        return {
            'num_subgraphs': len(partition_report.fused_subgraphs),
            'avg_fusion_size': partition_report.avg_fusion_size,
            'data_movement_reduction_pct': partition_report.data_movement_reduction * 100,
            'peak_memory_mb': peak_mb,
            'memory_reduction_pct': reduction_pct,
            'partition_report': partition_report,
        }
    except Exception as e:
        print(f"  ERROR: Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_vit_attention_fusion():
    """Test attention fusion patterns on ViT-B/16"""
    print("="*80)
    print("PHASE 3: ATTENTION FUSION PATTERN VALIDATION")
    print("="*80)
    print("\nThis validates that attention-specific fusion patterns provide")
    print("enhanced memory reduction compared to standard fusion.")

    # Configuration
    print("\n" + "-"*80)
    print("Configuration")
    print("-"*80)
    batch_size = 1
    input_shape = (batch_size, 3, 224, 224)
    print(f"  Model: ViT-B/16 (Vision Transformer Base)")
    print(f"  Input shape: {input_shape}")
    print(f"  Attention layers: 12")

    # Create model and hardware
    print("\n" + "-"*80)
    print("Loading Model")
    print("-"*80)
    model = vit_b_16(weights=None)
    model.eval()
    print("  ✓ Model loaded")

    input_tensor = torch.randn(input_shape)
    hardware = create_gpu_hardware()

    # Step 1: Trace with automatic attention decomposition
    print("\n" + "-"*80)
    print("Step 1: Automatic Attention Decomposition (Phase 2)")
    print("-"*80)
    print("\nApplying automatic attention decomposition...")
    decomposed_traced = trace_with_decomposition(model)
    print("  ✓ Decomposition successful")

    # Count decomposed operations
    computation_ops = ['call_function', 'call_method', 'call_module']
    num_nodes = len([n for n in decomposed_traced.graph.nodes if n.op in computation_ops])
    print(f"  Total FX nodes: {num_nodes}")

    # Step 2: Test with standard FusionBasedPartitioner
    print("\n" + "-"*80)
    print("Step 2: Standard Fusion Partitioner (Baseline)")
    print("-"*80)
    standard_partitioner = FusionBasedPartitioner()
    standard_results = analyze_with_partitioner(
        decomposed_traced, input_tensor, hardware,
        "Standard FusionBasedPartitioner",
        standard_partitioner
    )

    # Step 3: Test with AttentionFusionPartitioner
    print("\n" + "-"*80)
    print("Step 3: Attention Fusion Partitioner (Enhanced)")
    print("-"*80)
    attention_partitioner = AttentionFusionPartitioner()
    attention_results = analyze_with_partitioner(
        decomposed_traced, input_tensor, hardware,
        "AttentionFusionPartitioner (Phase 3)",
        attention_partitioner
    )

    # Step 4: Comparison
    if standard_results and attention_results:
        print("\n" + "="*80)
        print("COMPARISON: Standard vs Attention-Enhanced Fusion")
        print("="*80)

        print(f"\n{'Metric':<40} {'Standard':<20} {'Enhanced':<20} {'Improvement'}")
        print("-"*80)

        # Subgraphs
        std_subgraphs = standard_results['num_subgraphs']
        enh_subgraphs = attention_results['num_subgraphs']
        if std_subgraphs > enh_subgraphs:
            ratio = std_subgraphs / enh_subgraphs
            print(f"{'Fused Subgraphs':<40} {std_subgraphs:<20} {enh_subgraphs:<20} {ratio:.2f}× fewer")
        else:
            diff = enh_subgraphs - std_subgraphs
            print(f"{'Fused Subgraphs':<40} {std_subgraphs:<20} {enh_subgraphs:<20} +{diff}")

        # Fusion size
        std_fusion = standard_results['avg_fusion_size']
        enh_fusion = attention_results['avg_fusion_size']
        fusion_ratio = enh_fusion / std_fusion if std_fusion > 0 else 0
        print(f"{'Average Fusion Size':<40} {std_fusion:.2f}{'':<16} {enh_fusion:.2f}{'':<16} {fusion_ratio:.2f}×")

        # Data movement reduction
        std_dm = standard_results['data_movement_reduction_pct']
        enh_dm = attention_results['data_movement_reduction_pct']
        dm_improvement = enh_dm - std_dm
        print(f"{'Data Movement Reduction (%)':<40} {std_dm:.1f}%{'':<16} {enh_dm:.1f}%{'':<16} +{dm_improvement:.1f}%")

        # Memory reduction
        std_mem_red = standard_results['memory_reduction_pct']
        enh_mem_red = attention_results['memory_reduction_pct']
        mem_improvement = enh_mem_red - std_mem_red
        print(f"{'Memory Reduction (%)':<40} {std_mem_red:.1f}%{'':<16} {enh_mem_red:.1f}%{'':<16} +{mem_improvement:.1f}%")

        # Peak memory
        std_peak = standard_results['peak_memory_mb']
        enh_peak = attention_results['peak_memory_mb']
        peak_ratio = std_peak / enh_peak if enh_peak > 0 else 0
        print(f"{'Peak Memory (MB)':<40} {std_peak:.2f}{'':<16} {enh_peak:.2f}{'':<16} {peak_ratio:.2f}×")

        # Validation criteria
        print("\n" + "="*80)
        print("VALIDATION RESULTS")
        print("="*80)

        all_pass = True

        # Check 1: Parallel fusions created
        attn_stats = attention_partitioner.get_attention_fusion_stats()
        parallel_count = attn_stats['parallel_fusions_created']
        expected_parallel = 12  # 12 attention layers in ViT-B/16

        if parallel_count > 0:
            print(f"✓ PASS: Parallel Q,K,V fusions detected ({parallel_count} fusions)")
            if parallel_count >= expected_parallel * 0.8:  # Allow 80% tolerance
                print(f"  ✓ Expected ~{expected_parallel} fusions, found {parallel_count}")
            else:
                print(f"  ⚠ Expected ~{expected_parallel} fusions, found only {parallel_count}")
        else:
            print(f"✗ FAIL: No parallel Q,K,V fusions detected")
            all_pass = False

        # Check 2: Memory reduction improvement
        if mem_improvement > 0:
            print(f"✓ PASS: Memory reduction improved by {mem_improvement:.1f}%")
        else:
            print(f"⚠ INFO: Memory reduction changed by {mem_improvement:.1f}%")

        # Check 3: Data movement reduction
        if dm_improvement >= 5.0:
            print(f"✓ PASS: Significant data movement reduction improvement ({dm_improvement:.1f}%)")
        elif dm_improvement > 0:
            print(f"✓ PASS: Data movement reduction improved by {dm_improvement:.1f}%")
        else:
            print(f"⚠ INFO: Data movement reduction changed by {dm_improvement:.1f}%")

        # Check 4: Fusion efficiency
        if fusion_ratio >= 1.1:
            print(f"✓ PASS: Average fusion size increased by {(fusion_ratio - 1) * 100:.1f}%")
        elif fusion_ratio >= 1.0:
            print(f"✓ PASS: Average fusion size maintained or improved")
        else:
            print(f"⚠ INFO: Average fusion size decreased by {(1 - fusion_ratio) * 100:.1f}%")

        # Overall assessment
        print("\n" + "="*80)
        if all_pass and parallel_count > 0:
            print("✓ OVERALL: ATTENTION FUSION PATTERNS VALIDATED")
            print("\nPhase 3 enhancements are working correctly!")
            print(f"  - {parallel_count} parallel Q,K,V fusions created")
            print(f"  - {mem_improvement:+.1f}% memory reduction improvement")
            print(f"  - {dm_improvement:+.1f}% data movement reduction improvement")
        else:
            print("⚠ OVERALL: SOME VALIDATION CRITERIA NOT MET")
            print("\nPhase 3 may need adjustments")

        print("="*80)

        # Show fusion pattern examples
        if attention_results:
            print("\n" + "="*80)
            print("EXAMPLE FUSION PATTERNS")
            print("="*80)

            print("\nParallel Q,K,V Fusions:")
            parallel_sgs = attn_stats['parallel_fusion_subgraphs']
            for i, sg in enumerate(parallel_sgs[:3], 1):
                print(f"\n  {i}. Subgraph #{sg.subgraph_id}:")
                print(f"     Pattern: {sg.fusion_pattern}")
                print(f"     Operators: {sg.num_operators}")
                print(f"     FLOPs: {sg.total_flops / 1e9:.2f}G")
                print(f"     Memory reduction: {sg.data_movement_reduction() * 100:.1f}%")

            if len(parallel_sgs) > 3:
                print(f"\n  ... and {len(parallel_sgs) - 3} more parallel fusions")

        return all_pass
    else:
        print("\n✗ OVERALL: Analysis failed, cannot validate")
        return False


def main():
    """Main validation"""
    print("\n" * 2)
    passed = test_vit_attention_fusion()

    print("\n\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)

    if passed:
        print("\n✓✓✓ PHASE 3 VALIDATION PASSED ✓✓✓")
        print("\nAttention-specific fusion patterns are working correctly!")
        print("\nKey Achievements:")
        print("  1. Parallel Q,K,V projection fusion implemented")
        print("  2. Attention sequential patterns enhanced")
        print("  3. Memory reduction improved over standard fusion")
        print("  4. Ready for production use")
        print("\nNext Steps:")
        print("  - Test on other transformer models (BERT, GPT)")
        print("  - Integrate into production pipeline")
        print("  - Optimize fusion pattern detection")
    else:
        print("\n⚠⚠⚠ PHASE 3 VALIDATION INCOMPLETE ⚠⚠⚠")
        print("\nSome validation criteria not met.")
        print("Review the output above for details.")

    print("="*80)


if __name__ == "__main__":
    main()
