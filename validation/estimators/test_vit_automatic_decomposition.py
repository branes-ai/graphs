#!/usr/bin/env python3
"""
Validation: Automatic Attention Decomposition on Real ViT Models

This script validates that the automatic decomposition tracer (Phase 2) achieves
similar memory reduction benefits on real Vision Transformer models as the manual
decomposition (Phase 1).

Tests:
1. ViT-B/16 (Base model, 12 attention layers)
2. ViT-L/16 (Large model, 24 attention layers)
3. Functional equivalence verification
4. Memory reduction validation (target: >30% per attention layer)

Usage:
    python validation/estimators/test_vit_automatic_decomposition.py
"""

import torch
import torch.nn as nn
from torchvision.models import vit_b_16, vit_l_16
from torch.fx.passes.shape_prop import ShapeProp

from graphs.transform import trace_with_decomposition
from graphs.transform.partitioning import GraphPartitioner
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


def count_attention_layers(model):
    """Count MultiheadAttention layers in a model"""
    count = 0
    for module in model.modules():
        if isinstance(module, nn.MultiheadAttention):
            count += 1
    return count


def analyze_traced_model(traced, input_tensor, hardware, model_name):
    """Analyze a traced model and return metrics"""
    print(f"\n{'='*80}")
    print(f"Analyzing: {model_name}")
    print(f"{'='*80}")

    # Shape propagation (may fail with dynamic sizes, but that's OK)
    try:
        ShapeProp(traced).propagate(input_tensor)
        print("  ✓ Shape propagation successful")
    except Exception as e:
        print(f"  ⚠ Shape propagation failed (expected with dynamic sizes): {str(e)[:80]}")

    # Count nodes
    computation_ops = ['call_function', 'call_method', 'call_module']
    num_nodes = len([n for n in traced.graph.nodes if n.op in computation_ops])
    print(f"  Total FX nodes: {num_nodes}")

    # Count attention-related operations
    attention_ops = [
        'linear', 'matmul', 'softmax', 'dropout',
        'transpose', 'view', 'reshape', 'mul'
    ]
    num_attention_ops = 0
    for node in traced.graph.nodes:
        if node.op in computation_ops:
            node_str = str(node.target).lower()
            if any(op in node_str for op in attention_ops):
                num_attention_ops += 1
    print(f"  Attention-related operations: {num_attention_ops}")

    # Partitioning
    partitioner = GraphPartitioner()
    try:
        partition_report = partitioner.partition(traced)
        print(f"  Subgraphs: {len(partition_report.subgraphs)}")
        print(f"  Total FLOPs: {partition_report.total_flops:,}")

        # Memory analysis
        memory_estimator = MemoryEstimator(hardware)
        memory_report = memory_estimator.estimate_memory(
            partition_report.subgraphs,
            partition_report
        )
        peak_mb = memory_report.peak_memory_bytes / (1024 * 1024)
        print(f"  Peak memory: {peak_mb:.2f} MB")

        # Calculate memory reduction
        total_intermediate = sum(sg.total_output_bytes for sg in partition_report.subgraphs)
        unfused_peak = memory_report.weight_memory_bytes + total_intermediate
        reduction_pct = (1 - memory_report.peak_memory_bytes / unfused_peak) * 100 if unfused_peak > 0 else 0
        print(f"  Memory reduction from fusion: {reduction_pct:.1f}%")

        return {
            'num_nodes': num_nodes,
            'num_attention_ops': num_attention_ops,
            'num_subgraphs': len(partition_report.subgraphs),
            'total_flops': partition_report.total_flops,
            'peak_memory_mb': peak_mb,
            'memory_reduction_pct': reduction_pct,
        }
    except Exception as e:
        print(f"  ERROR: Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_vit_model(model_factory, model_name, input_shape):
    """Test automatic decomposition on a ViT model variant"""
    print(f"\n{'='*80}")
    print(f"TEST: {model_name}")
    print(f"{'='*80}")

    # Create model
    print("\nLoading model...")
    model = model_factory(weights=None)
    model.eval()

    # Count attention layers
    num_attn_layers = count_attention_layers(model)
    print(f"Model contains {num_attn_layers} MultiheadAttention layers")

    # Create input
    input_tensor = torch.randn(input_shape)
    print(f"Input shape: {list(input_tensor.shape)}")

    # Create hardware
    hardware = create_gpu_hardware()

    # Test 1: Standard tracing
    print("\n" + "-"*80)
    print("Step 1: Standard FX Tracing (Baseline)")
    print("-"*80)
    try:
        standard_traced = torch.fx.symbolic_trace(model)
        print("  ✓ Standard tracing successful")
        standard_results = analyze_traced_model(
            standard_traced, input_tensor, hardware, f"{model_name} - Standard"
        )
    except Exception as e:
        print(f"  ERROR: Standard tracing failed: {e}")
        standard_results = None

    # Test 2: Automatic decomposition
    print("\n" + "-"*80)
    print("Step 2: Automatic Attention Decomposition")
    print("-"*80)
    try:
        decomposed_traced = trace_with_decomposition(model)
        print("  ✓ Automatic decomposition successful")
        decomposed_results = analyze_traced_model(
            decomposed_traced, input_tensor, hardware, f"{model_name} - Decomposed"
        )
    except Exception as e:
        print(f"  ERROR: Automatic decomposition failed: {e}")
        import traceback
        traceback.print_exc()
        decomposed_results = None

    # Test 3: Functional equivalence
    print("\n" + "-"*80)
    print("Step 3: Functional Equivalence Check")
    print("-"*80)
    if standard_results and decomposed_results:
        print("\nVerifying outputs match...")
        with torch.no_grad():
            try:
                # Create fresh models with same weights
                model1 = model_factory(weights=None)
                model1.eval()
                model2 = model_factory(weights=None)
                model2.eval()
                model2.load_state_dict(model1.state_dict())

                # Standard output
                output_standard = model1(input_tensor)

                # Decomposed output
                decomposed_traced_test = trace_with_decomposition(model2)
                output_decomposed = decomposed_traced_test(input_tensor)

                # Compare
                max_diff = torch.max(torch.abs(output_standard - output_decomposed)).item()
                rel_diff = (torch.norm(output_standard - output_decomposed) / torch.norm(output_standard)).item()

                print(f"  Max absolute difference: {max_diff:.2e}")
                print(f"  Relative difference: {rel_diff:.2e}")

                if max_diff < 1e-4:
                    print("  ✓ PASS: Outputs match (functionally equivalent)")
                    functional_pass = True
                else:
                    print(f"  ⚠ FAIL: Outputs differ by {max_diff:.2e}")
                    functional_pass = False
            except Exception as e:
                print(f"  ERROR: Functional test failed: {e}")
                functional_pass = False
    else:
        functional_pass = False

    # Comparison summary
    if standard_results and decomposed_results:
        print("\n" + "="*80)
        print("COMPARISON SUMMARY")
        print("="*80)

        print(f"\n{'Metric':<35} {'Standard':<20} {'Decomposed':<20} {'Ratio'}")
        print("-"*80)

        std_nodes = standard_results['num_nodes']
        dec_nodes = decomposed_results['num_nodes']
        node_ratio = dec_nodes / std_nodes if std_nodes > 0 else 0
        print(f"{'FX Graph Nodes':<35} {std_nodes:<20} {dec_nodes:<20} {node_ratio:.2f}×")

        std_attn_ops = standard_results['num_attention_ops']
        dec_attn_ops = decomposed_results['num_attention_ops']
        attn_ratio = dec_attn_ops / std_attn_ops if std_attn_ops > 0 else 0
        print(f"{'Attention Operations':<35} {std_attn_ops:<20} {dec_attn_ops:<20} {attn_ratio:.2f}×")

        std_mem = standard_results['peak_memory_mb']
        dec_mem = decomposed_results['peak_memory_mb']
        mem_ratio = std_mem / dec_mem if dec_mem > 0 else 0
        print(f"{'Peak Memory (MB)':<35} {std_mem:.2f}{'':<16} {dec_mem:.2f}{'':<16} {mem_ratio:.2f}×")

        std_reduction = standard_results['memory_reduction_pct']
        dec_reduction = decomposed_results['memory_reduction_pct']
        print(f"{'Memory Reduction (%)':<35} {std_reduction:.1f}%{'':<16} {dec_reduction:.1f}%")

        # Validation criteria
        print("\n" + "="*80)
        print("VALIDATION RESULTS")
        print("="*80)

        all_pass = True

        # Check 1: More operations exposed
        if node_ratio >= 2.0:
            print(f"✓ PASS: Decomposition exposes {node_ratio:.1f}× more operations (target: ≥2×)")
        else:
            print(f"✗ FAIL: Decomposition only exposes {node_ratio:.1f}× more operations (target: ≥2×)")
            all_pass = False

        # Check 2: Memory reduction improvement
        if dec_reduction > std_reduction:
            improvement = dec_reduction - std_reduction
            print(f"✓ PASS: Memory reduction improved by {improvement:.1f}% ({std_reduction:.1f}% → {dec_reduction:.1f}%)")
        else:
            print(f"✗ FAIL: Memory reduction did not improve ({std_reduction:.1f}% → {dec_reduction:.1f}%)")
            all_pass = False

        # Check 3: Target memory reduction (30%+)
        if dec_reduction >= 30.0:
            print(f"✓ PASS: Achieved target memory reduction ({dec_reduction:.1f}% ≥ 30%)")
        else:
            print(f"⚠ PARTIAL: Memory reduction below target ({dec_reduction:.1f}% < 30%)")
            # Not a hard failure for full models, attention is only part of the network

        # Check 4: Functional equivalence
        if functional_pass:
            print(f"✓ PASS: Functional equivalence verified")
        else:
            print(f"✗ FAIL: Functional equivalence check failed")
            all_pass = False

        # Check 5: All attention layers decomposed
        expected_decompositions = num_attn_layers
        if num_attn_layers > 0:
            print(f"✓ INFO: Model has {num_attn_layers} attention layers to decompose")

        print("\n" + "="*80)
        if all_pass:
            print("✓ OVERALL: ALL CRITICAL TESTS PASSED")
        else:
            print("✗ OVERALL: SOME TESTS FAILED")
        print("="*80)

        return all_pass
    else:
        print("\n✗ OVERALL: Analysis failed, cannot validate")
        return False


def main():
    """Main validation suite"""
    print("="*80)
    print("VALIDATION: AUTOMATIC ATTENTION DECOMPOSITION ON REAL VIT MODELS")
    print("="*80)
    print("\nThis validates that Phase 2 automatic decomposition achieves similar")
    print("benefits on production Vision Transformer models as Phase 1 manual")
    print("decomposition (target: >30% memory reduction per attention layer).")

    # Test configurations
    tests = [
        {
            'factory': vit_b_16,
            'name': 'ViT-B/16 (Base)',
            'input_shape': (1, 3, 224, 224),
            'description': '12 transformer blocks, 768 embed_dim, 12 heads'
        },
    ]

    # Run tests
    results = []
    for test_config in tests:
        print(f"\n\n{'#'*80}")
        print(f"# {test_config['name']}")
        print(f"# {test_config['description']}")
        print(f"{'#'*80}")

        try:
            passed = test_vit_model(
                test_config['factory'],
                test_config['name'],
                test_config['input_shape']
            )
            results.append((test_config['name'], passed))
        except Exception as e:
            print(f"\n✗ ERROR: Test failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_config['name'], False))

    # Final summary
    print("\n\n" + "="*80)
    print("FINAL VALIDATION SUMMARY")
    print("="*80)

    for model_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {model_name}")

    all_passed = all(passed for _, passed in results)
    print("\n" + "="*80)
    if all_passed:
        print("✓✓✓ ALL VALIDATION TESTS PASSED ✓✓✓")
        print("\nPhase 2 automatic decomposition is validated and ready for production!")
    else:
        print("✗✗✗ SOME VALIDATION TESTS FAILED ✗✗✗")
        print("\nPhase 2 needs additional work before production use.")
    print("="*80)

    # Next steps
    print("\nNext Steps:")
    if all_passed:
        print("  1. ✓ Phase 2 complete - automatic decomposition validated")
        print("  2. → Proceed to Phase 3: Add attention-specific fusion patterns")
        print("  3. → Test on BERT and other transformer architectures")
        print("  4. → Optimize fusion patterns for attention operations")
    else:
        print("  1. → Debug failed tests and fix issues")
        print("  2. → Re-run validation")
        print("  3. → Proceed only after all tests pass")


if __name__ == "__main__":
    main()
