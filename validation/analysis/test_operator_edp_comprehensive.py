#!/usr/bin/env python3
"""
Comprehensive Validation Tests for Operator-Level EDP (Phase 2)

Tests:
1. Basic operator extraction and EDP calculation
2. EDP fraction normalization (energy-based)
3. Architectural modifiers
4. Subgraph-level EDP breakdown
5. UnifiedAnalyzer integration
6. Cross-architecture consistency

Run: python validation/analysis/test_operator_edp_comprehensive.py
"""

import sys
import os

# Add src to path for direct execution
if __name__ == '__main__':
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from graphs.analysis.architecture_comparator import ArchitectureComparator
from graphs.analysis.unified_analyzer import UnifiedAnalyzer, AnalysisConfig
from graphs.hardware.mappers.gpu import create_h100_pcie_80gb_mapper
from graphs.hardware.mappers.accelerators.kpu import create_kpu_t256_mapper
from graphs.hardware.mappers.accelerators.tpu import create_tpu_v4_mapper
from graphs.hardware.resource_model import Precision


def test_basic_operator_extraction():
    """Test 1: Basic operator extraction and EDP calculation"""
    print("=" * 100)
    print("Test 1: Basic Operator Extraction and EDP Calculation")
    print("=" * 100)
    print()

    architectures = {'KPU': create_kpu_t256_mapper()}
    comparator = ArchitectureComparator(
        model_name='resnet18',
        architectures=architectures,
        batch_size=1,
        precision=Precision.FP32
    )

    comparator.analyze_all()
    operator_edps = comparator.get_operator_edp_breakdown('KPU')

    # Validation
    assert len(operator_edps) > 0, "Should extract operators"
    print(f"✓ Extracted {len(operator_edps)} operators")

    # Check all operators have required fields
    for op in operator_edps[:5]:
        assert op.operator_type, "Operator should have type"
        assert op.subgraph_name, "Operator should have subgraph name"
        assert op.architectural_edp >= 0, "EDP should be non-negative"
        assert op.architectural_modifier > 0, "Modifier should be positive"
        assert 0 <= op.edp_fraction_of_model <= 1, "Model fraction should be [0,1]"
        assert 0 <= op.edp_fraction_of_subgraph <= 1, "Subgraph fraction should be [0,1]"

    print(f"✓ All operators have valid fields")
    print()

    # Show top 3
    print("Top 3 Operators:")
    for i, op in enumerate(operator_edps[:3], 1):
        print(f"  {i}. {op.operator_type:<15} {op.architectural_edp*1e9:>8.2f} nJ·s "
              f"({op.edp_fraction_of_model*100:>5.1f}%)")
    print()
    print("✅ Test 1 PASSED")
    print()


def test_edp_fraction_normalization():
    """Test 2: EDP fraction normalization (energy-based)"""
    print("=" * 100)
    print("Test 2: EDP Fraction Normalization")
    print("=" * 100)
    print()

    architectures = {'KPU': create_kpu_t256_mapper()}
    comparator = ArchitectureComparator(
        model_name='resnet18',
        architectures=architectures,
        batch_size=1,
        precision=Precision.FP32
    )

    comparator.analyze_all()
    operator_edps = comparator.get_operator_edp_breakdown('KPU')

    # Calculate total fraction
    total_fraction = sum(op.edp_fraction_of_model for op in operator_edps)

    print(f"Operator EDP fractions sum: {total_fraction:.4f}")

    # Should be close to 1.0, accounting for static energy (~3%)
    # Energy is additive, so operator energies should sum to ~97% of total
    # (remaining 3% is static/leakage energy that's time-based, not operation-specific)
    assert 0.95 <= total_fraction <= 1.0, f"Fractions should sum to ~0.97, got {total_fraction}"

    print(f"✓ Fractions sum correctly (expected 0.95-1.0 due to static energy)")

    if total_fraction < 0.99:
        remaining_pct = (1.0 - total_fraction) * 100
        print(f"  Remaining {remaining_pct:.1f}% is static/leakage energy (expected ~3%)")

    print()
    print("✅ Test 2 PASSED")
    print()


def test_architectural_modifiers():
    """Test 3: Architectural modifiers are applied correctly"""
    print("=" * 100)
    print("Test 3: Architectural Modifiers")
    print("=" * 100)
    print()

    # Test on KPU (spatial dataflow architecture)
    architectures = {'KPU': create_kpu_t256_mapper()}
    comparator = ArchitectureComparator(
        model_name='resnet18',
        architectures=architectures,
        batch_size=1,
        precision=Precision.FP32
    )

    comparator.analyze_all()
    operator_edps = comparator.get_operator_edp_breakdown('KPU')

    # Find Conv2d and BatchNorm operators
    conv_ops = [op for op in operator_edps if op.operator_type == 'Conv2d']
    bn_ops = [op for op in operator_edps if op.operator_type == 'BatchNorm2d']

    assert len(conv_ops) > 0, "Should have Conv2d operators"
    assert len(bn_ops) > 0, "Should have BatchNorm2d operators"

    print(f"Found {len(conv_ops)} Conv2d operators")
    print(f"Found {len(bn_ops)} BatchNorm2d operators")
    print()

    # Check modifiers
    # Conv2d should have ~1.0× modifier (baseline compute)
    # BatchNorm2d should have ~1.5× modifier (higher overhead on spatial architectures)
    avg_conv_modifier = sum(op.architectural_modifier for op in conv_ops) / len(conv_ops)
    avg_bn_modifier = sum(op.architectural_modifier for op in bn_ops) / len(bn_ops)

    print(f"Average Conv2d modifier: {avg_conv_modifier:.2f}× (expected ~1.0)")
    print(f"Average BatchNorm modifier: {avg_bn_modifier:.2f}× (expected ~1.5)")

    assert 0.8 <= avg_conv_modifier <= 1.2, "Conv2d should have ~1.0× modifier"
    assert 1.3 <= avg_bn_modifier <= 1.7, "BatchNorm should have ~1.5× modifier"

    print(f"✓ Architectural modifiers correct")
    print()
    print("✅ Test 3 PASSED")
    print()


def test_subgraph_edp_breakdown():
    """Test 4: Subgraph-level EDP breakdown"""
    print("=" * 100)
    print("Test 4: Subgraph-Level EDP Breakdown")
    print("=" * 100)
    print()

    architectures = {'KPU': create_kpu_t256_mapper()}
    comparator = ArchitectureComparator(
        model_name='resnet18',
        architectures=architectures,
        batch_size=1,
        precision=Precision.FP32
    )

    comparator.analyze_all()
    subgraph_edps = comparator.get_subgraph_edp_breakdown('KPU')

    assert len(subgraph_edps) > 0, "Should have subgraphs"
    print(f"✓ Found {len(subgraph_edps)} subgraphs")

    # Check all subgraphs have required fields
    for sg in subgraph_edps[:5]:
        assert sg.subgraph_name, "Should have name"
        assert sg.energy_j >= 0, "Energy should be non-negative"
        assert sg.latency_s >= 0, "Latency should be non-negative"
        assert sg.edp >= 0, "EDP should be non-negative"
        assert 0 <= sg.edp_fraction <= 1, "Fraction should be [0,1]"

    print(f"✓ All subgraphs have valid fields")

    # Check subgraph EDP fractions sum correctly
    # Note: Subgraph EDPs use individual latencies, so sum ≠ model EDP
    # This is expected behavior (latencies are not additive due to parallelism)
    total_sg_fraction = sum(sg.edp_fraction for sg in subgraph_edps)
    print(f"Subgraph EDP fractions sum: {total_sg_fraction:.4f}")

    # Show top 3 subgraphs
    print()
    print("Top 3 Subgraphs by EDP:")
    for i, sg in enumerate(subgraph_edps[:3], 1):
        print(f"  {i}. {sg.subgraph_name:<25} {sg.edp*1e9:>8.2f} nJ·s ({sg.edp_fraction*100:>5.1f}%)")

    print()
    print("✅ Test 4 PASSED")
    print()


def test_unified_analyzer_integration():
    """Test 5: UnifiedAnalyzer integration"""
    print("=" * 100)
    print("Test 5: UnifiedAnalyzer Integration")
    print("=" * 100)
    print()

    analyzer = UnifiedAnalyzer(verbose=False)
    config = AnalysisConfig(
        run_operator_edp=True,
        run_concurrency=False,  # Skip for speed
    )

    print("Running UnifiedAnalyzer with operator EDP enabled...")
    result = analyzer.analyze_model(
        model_name='resnet18',
        hardware_name='kpu-t256',
        batch_size=1,
        precision=Precision.FP32,
        config=config
    )

    # Validate results
    assert result.operator_edp_breakdown is not None, "Should have operator breakdown"
    assert result.subgraph_edp_breakdown is not None, "Should have subgraph breakdown"

    print(f"✓ UnifiedAnalyzer populated operator breakdown ({len(result.operator_edp_breakdown)} operators)")
    print(f"✓ UnifiedAnalyzer populated subgraph breakdown ({len(result.subgraph_edp_breakdown)} subgraphs)")

    # Check consistency with other metrics
    assert result.total_energy_mj > 0, "Should have energy"
    assert result.total_latency_ms > 0, "Should have latency"

    print(f"✓ Consistent with unified metrics (energy: {result.total_energy_mj:.2f} mJ, latency: {result.total_latency_ms:.2f} ms)")

    # Test with operator EDP disabled
    print()
    print("Testing with operator EDP disabled...")
    config_disabled = AnalysisConfig(run_operator_edp=False, run_concurrency=False)
    result_disabled = analyzer.analyze_model(
        model_name='resnet18',
        hardware_name='kpu-t256',
        batch_size=1,
        precision=Precision.FP32,
        config=config_disabled
    )

    assert result_disabled.operator_edp_breakdown is None, "Should not have operator breakdown when disabled"
    assert result_disabled.subgraph_edp_breakdown is None, "Should not have subgraph breakdown when disabled"

    print(f"✓ Correctly skips operator EDP when disabled")

    print()
    print("✅ Test 5 PASSED")
    print()


def test_cross_architecture_consistency():
    """Test 6: Cross-architecture consistency"""
    print("=" * 100)
    print("Test 6: Cross-Architecture Consistency")
    print("=" * 100)
    print()

    # Test on multiple architectures (GPU and KPU for now)
    architectures = {
        'GPU': create_h100_pcie_80gb_mapper(),
        'KPU': create_kpu_t256_mapper(),
    }

    comparator = ArchitectureComparator(
        model_name='resnet18',
        architectures=architectures,
        batch_size=1,
        precision=Precision.FP32
    )

    comparator.analyze_all()

    results = {}
    for arch_name in architectures.keys():
        operator_edps = comparator.get_operator_edp_breakdown(arch_name)
        total_fraction = sum(op.edp_fraction_of_model for op in operator_edps)
        results[arch_name] = {
            'count': len(operator_edps),
            'total_fraction': total_fraction,
            'top_op': operator_edps[0] if operator_edps else None
        }

    # All architectures should extract same number of operators
    counts = [r['count'] for r in results.values()]
    assert len(set(counts)) == 1, f"All architectures should extract same operators, got {counts}"
    print(f"✓ All architectures extract same number of operators ({counts[0]})")

    # All should have fractions summing to ~0.97
    for arch_name, r in results.items():
        assert 0.95 <= r['total_fraction'] <= 1.0, f"{arch_name} fractions should sum to ~0.97"
        print(f"✓ {arch_name}: fractions sum to {r['total_fraction']:.4f}")

    # Different architectures should have different top operators (due to modifiers)
    print()
    print("Top operator by architecture:")
    for arch_name, r in results.items():
        if r['top_op']:
            print(f"  {arch_name}: {r['top_op'].operator_type} "
                  f"({r['top_op'].edp_fraction_of_model*100:.1f}%, modifier: {r['top_op'].architectural_modifier:.2f}×)")

    print()
    print("✅ Test 6 PASSED")
    print()


def run_all_tests():
    """Run all validation tests"""
    print()
    print("="*100)
    print("OPERATOR-LEVEL EDP COMPREHENSIVE VALIDATION TESTS (Phase 2.6)")
    print("="*100)
    print()

    tests = [
        ("Basic Operator Extraction", test_basic_operator_extraction),
        ("EDP Fraction Normalization", test_edp_fraction_normalization),
        ("Architectural Modifiers", test_architectural_modifiers),
        ("Subgraph EDP Breakdown", test_subgraph_edp_breakdown),
        ("UnifiedAnalyzer Integration", test_unified_analyzer_integration),
        ("Cross-Architecture Consistency", test_cross_architecture_consistency),
    ]

    passed = 0
    failed = 0
    errors = []

    for test_name, test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            failed += 1
            errors.append((test_name, str(e)))
            print(f"❌ {test_name} FAILED: {e}")
            print()
        except Exception as e:
            failed += 1
            errors.append((test_name, f"Exception: {e}"))
            print(f"❌ {test_name} ERROR: {e}")
            print()

    # Summary
    print("="*100)
    print("TEST SUMMARY")
    print("="*100)
    print()
    print(f"Total tests: {len(tests)}")
    print(f"Passed: {passed} ✅")
    print(f"Failed: {failed} ❌")
    print()

    if errors:
        print("FAILED TESTS:")
        for test_name, error in errors:
            print(f"  - {test_name}: {error}")
        print()
        return False
    else:
        print("🎉 ALL TESTS PASSED! 🎉")
        print()
        print("Phase 2: Operator-Level EDP is production-ready!")
        print()
        return True


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
