#!/usr/bin/env python
"""
Unit Tests for FusionBasedPartitioner

This test suite validates:
1. Fusion pattern detection accuracy
2. Metrics calculations (FLOPs, memory, arithmetic intensity)
3. Cross-validation with fvcore and torch.profiler
4. Fusion quality across different architectures

Run: python tests/test_fusion_partitioner.py
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torch.fx import symbolic_trace, GraphModule
from torch.fx.passes.shape_prop import ShapeProp
from collections import Counter
from typing import Dict, List, Optional
import pytest

from graphs.transform.partitioning import FusionBasedPartitioner
from graphs.ir.structures import OperationType


class ResultsTracker:
    """Track test results and generate summary (not a pytest test class)"""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.warnings = 0
        self.details = []

    def pass_test(self, name: str, message: str = ""):
        self.passed += 1
        self.details.append(f"✓ {name}: {message}")

    def fail_test(self, name: str, message: str):
        self.failed += 1
        self.details.append(f"✗ {name}: {message}")

    def warn(self, message: str):
        self.warnings += 1
        self.details.append(f"⚠️  {message}")

    def print_summary(self):
        print("\n" + "=" * 80)
        print("TEST SUMMARY")
        print("=" * 80)
        for detail in self.details:
            print(detail)
        print()
        print(f"Passed: {self.passed}")
        print(f"Failed: {self.failed}")
        print(f"Warnings: {self.warnings}")
        print(f"Success Rate: {self.passed / max(1, self.passed + self.failed) * 100:.1f}%")
        print("=" * 80)

    def all_passed(self) -> bool:
        return self.failed == 0


@pytest.fixture
def results():
    """Pytest fixture to provide ResultsTracker object"""
    return ResultsTracker()


def trace_model(model: nn.Module, input_shape=(1, 3, 224, 224)) -> Optional[GraphModule]:
    """Trace a model with FX and shape propagation"""
    try:
        input_tensor = torch.randn(*input_shape)
        fx_graph = symbolic_trace(model)
        ShapeProp(fx_graph).propagate(input_tensor)
        return fx_graph
    except Exception as e:
        print(f"Error tracing model: {e}")
        return None


def test_fusion_pattern_detection(results: ResultsTracker):
    """Test 1: Validate fusion pattern detection"""
    print("\n" + "=" * 80)
    print("TEST 1: Fusion Pattern Detection")
    print("=" * 80)

    # Create simple test models with known patterns
    class ConvBNReLU(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 64, 3, padding=1)
            self.bn = nn.BatchNorm2d(64)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.conv(x)
            x = self.bn(x)
            x = self.relu(x)
            return x

    print("\n[1.1] Testing Conv->BN->ReLU pattern...")
    model = ConvBNReLU()
    fx_graph = trace_model(model)

    if fx_graph is None:
        results.fail_test("Conv-BN-ReLU trace", "Failed to trace model")
        return

    partitioner = FusionBasedPartitioner()
    report = partitioner.partition(fx_graph)

    # Should find Conv2d_BatchNorm2d_ReLU pattern
    pattern_found = any('Conv2d_BatchNorm2d_ReLU' in sg.fusion_pattern
                       for sg in report.fused_subgraphs)

    if pattern_found:
        results.pass_test("Conv-BN-ReLU detection", "Pattern correctly identified")
    else:
        patterns = [sg.fusion_pattern for sg in report.fused_subgraphs]
        results.fail_test("Conv-BN-ReLU detection", f"Pattern not found. Got: {patterns}")

    # Verify fusion size
    fused_patterns = [sg for sg in report.fused_subgraphs
                     if 'Conv2d_BatchNorm2d_ReLU' in sg.fusion_pattern]
    if fused_patterns and fused_patterns[0].num_operators == 3:
        results.pass_test("Conv-BN-ReLU fusion size", "3 operators fused correctly")
    else:
        actual_size = fused_patterns[0].num_operators if fused_patterns else 0
        results.fail_test("Conv-BN-ReLU fusion size",
                         f"Expected 3 operators, got {actual_size}")

    print("\n[1.2] Testing ResNet patterns (Conv-BN, add-ReLU)...")
    model = models.resnet18(weights=None)
    model.eval()
    fx_graph = trace_model(model)

    if fx_graph:
        partitioner = FusionBasedPartitioner()
        report = partitioner.partition(fx_graph)

        # Should find Conv2d_BatchNorm2d pattern
        conv_bn_count = sum(1 for sg in report.fused_subgraphs
                           if sg.fusion_pattern == 'Conv2d_BatchNorm2d')

        if conv_bn_count > 0:
            results.pass_test("Conv-BN detection", f"Found {conv_bn_count} instances")
        else:
            results.fail_test("Conv-BN detection", "Pattern not found in ResNet")

        # Should find add_ReLU pattern
        add_relu_count = sum(1 for sg in report.fused_subgraphs
                            if sg.fusion_pattern == 'add_ReLU')

        if add_relu_count > 0:
            results.pass_test("add-ReLU detection", f"Found {add_relu_count} instances")
        else:
            results.fail_test("add-ReLU detection", "Pattern not found in ResNet")


def test_metrics_calculations(results: ResultsTracker):
    """Test 2: Validate FLOPs, memory, and AI calculations"""
    print("\n" + "=" * 80)
    print("TEST 2: Metrics Calculations")
    print("=" * 80)

    print("\n[2.1] Testing on ResNet-18...")
    model = models.resnet18(weights=None)
    model.eval()
    fx_graph = trace_model(model)

    if fx_graph is None:
        results.fail_test("ResNet-18 trace", "Failed to trace model")
        return

    partitioner = FusionBasedPartitioner()
    report = partitioner.partition(fx_graph)

    # Test FLOPs calculation
    print(f"\nTotal FLOPs: {report.total_flops / 1e9:.2f} GFLOPs")
    expected_flops = 3.64e9  # ResNet-18 expected FLOPs (unfused baseline)
    actual_flops = report.total_flops
    error = abs(actual_flops - expected_flops) / expected_flops

    if error < 0.3:  # Within 30% tolerance
        results.pass_test("FLOPs calculation",
                         f"{actual_flops / 1e9:.2f}G vs expected ~3.64G (error {error * 100:.1f}%)")
    else:
        results.fail_test("FLOPs calculation",
                         f"{actual_flops / 1e9:.2f}G vs expected ~3.64G (error {error * 100:.1f}%)")

    # Test memory calculations
    print(f"\nMemory Traffic (fused): {report.total_memory_traffic_fused / 1e6:.2f} MB")
    print(f"Memory Traffic (unfused): {report.total_memory_traffic_unfused / 1e6:.2f} MB")

    if report.total_memory_traffic_fused < report.total_memory_traffic_unfused:
        reduction = (report.total_memory_traffic_unfused - report.total_memory_traffic_fused) / report.total_memory_traffic_unfused * 100
        results.pass_test("Memory reduction",
                         f"{reduction:.1f}% reduction from fusion")
    else:
        results.fail_test("Memory reduction",
                         "Fusion should reduce memory traffic")

    # Test arithmetic intensity
    avg_ai = sum(sg.arithmetic_intensity for sg in report.fused_subgraphs) / len(report.fused_subgraphs)
    print(f"\nAverage Arithmetic Intensity: {avg_ai:.2f} FLOPs/byte")

    if avg_ai > 0:
        results.pass_test("Arithmetic intensity", f"{avg_ai:.2f} FLOPs/byte")
    else:
        results.fail_test("Arithmetic intensity", "AI should be > 0")

    # Test that all subgraphs have valid metrics
    invalid_count = sum(1 for sg in report.fused_subgraphs
                       if sg.total_flops == 0 and len(sg.operation_types) > 0)

    if invalid_count == 0:
        results.pass_test("Subgraph metrics", "All subgraphs have valid FLOPs")
    else:
        results.warn(f"{invalid_count} subgraphs have zero FLOPs")


def test_fvcore_comparison(results: ResultsTracker):
    """Test 3: Cross-validate with fvcore FlopCountAnalysis"""
    print("\n" + "=" * 80)
    print("TEST 3: Cross-validation with fvcore")
    print("=" * 80)

    try:
        from fvcore.nn import FlopCountAnalysis
    except ImportError:
        results.warn("fvcore not installed, skipping cross-validation")
        return

    print("\n[3.1] Comparing FLOPs with fvcore on ResNet-18...")
    model = models.resnet18(weights=None)
    model.eval()

    input_tensor = torch.randn(1, 3, 224, 224)

    # Get fvcore FLOPs (note: fvcore counts MACs, not FLOPs)
    fvcore_macs = FlopCountAnalysis(model, input_tensor).total()
    fvcore_flops = fvcore_macs * 2  # Convert MACs to FLOPs

    # Get our FLOPs
    fx_graph = trace_model(model)
    if fx_graph is None:
        results.fail_test("fvcore comparison", "Failed to trace model")
        return

    partitioner = FusionBasedPartitioner()
    report = partitioner.partition(fx_graph)
    our_flops = report.total_flops

    print(f"\nfvcore MACs:   {fvcore_macs / 1e9:.3f} G")
    print(f"fvcore FLOPs:  {fvcore_flops / 1e9:.3f} GFLOPs (MACs × 2)")
    print(f"Our FLOPs:     {our_flops / 1e9:.3f} GFLOPs")

    error = abs(our_flops - fvcore_flops) / fvcore_flops

    if error < 0.15:  # Within 15%
        results.pass_test("fvcore FLOPs match",
                         f"Error {error * 100:.1f}% (within tolerance)")
    else:
        results.warn(f"fvcore FLOPs differ by {error * 100:.1f}% (may be due to different counting methods)")

    # Test torch.profiler integration
    print("\n[3.2] Cross-validation with torch.profiler...")

    try:
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU],
            record_shapes=True,
            with_flops=True
        ) as prof:
            with torch.no_grad():
                _ = model(input_tensor)

        # Get profiler FLOPs (if available)
        total_flops_profiler = sum(event.flops for event in prof.events() if event.flops > 0)

        if total_flops_profiler > 0:
            print(f"\nProfiler FLOPs: {total_flops_profiler / 1e9:.3f} GFLOPs")
            print(f"Our FLOPs:      {our_flops / 1e9:.3f} GFLOPs")

            prof_error = abs(our_flops - total_flops_profiler) / total_flops_profiler
            if prof_error < 0.2:  # Within 20%
                results.pass_test("torch.profiler FLOPs match",
                               f"Error {prof_error * 100:.1f}% (within tolerance)")
            else:
                results.warn(f"torch.profiler FLOPs differ by {prof_error * 100:.1f}%")
        else:
            results.warn("torch.profiler did not report FLOPs (expected on some platforms)")

    except Exception as e:
        results.warn(f"torch.profiler test failed: {e}")


def test_fusion_quality(results: ResultsTracker):
    """Test 4: Validate fusion quality metrics"""
    print("\n" + "=" * 80)
    print("TEST 4: Fusion Quality Analysis")
    print("=" * 80)

    models_to_test = [
        ('resnet18', models.resnet18),
        ('mobilenet_v2', models.mobilenet_v2),
    ]

    for model_name, model_fn in models_to_test:
        print(f"\n[4.{models_to_test.index((model_name, model_fn)) + 1}] Testing {model_name}...")

        model = model_fn(weights=None)
        model.eval()
        fx_graph = trace_model(model)

        if fx_graph is None:
            results.warn(f"Could not trace {model_name}")
            continue

        partitioner = FusionBasedPartitioner()
        report = partitioner.partition(fx_graph)

        # Calculate fusion efficiency
        total_ops = sum(sg.num_operators for sg in report.fused_subgraphs)
        fusion_efficiency = total_ops / len(report.fused_subgraphs)

        print(f"  Subgraphs: {len(report.fused_subgraphs)}")
        print(f"  Total operators: {total_ops}")
        print(f"  Fusion efficiency: {fusion_efficiency:.2f}×")

        # Test fusion efficiency is > 1 (some fusion happened)
        if fusion_efficiency > 1.5:
            results.pass_test(f"{model_name} fusion efficiency",
                            f"{fusion_efficiency:.2f}× (good fusion)")
        elif fusion_efficiency > 1.0:
            results.warn(f"{model_name} fusion efficiency is low: {fusion_efficiency:.2f}×")
        else:
            results.fail_test(f"{model_name} fusion efficiency",
                            f"{fusion_efficiency:.2f}× (no fusion occurred)")

        # Test data movement reduction
        if report.data_movement_reduction > 0.1:  # >10% reduction
            results.pass_test(f"{model_name} data movement",
                            f"{report.data_movement_reduction * 100:.1f}% reduction")
        else:
            results.warn(f"{model_name} low data movement reduction: "
                        f"{report.data_movement_reduction * 100:.1f}%")


def test_balance_analysis(results: ResultsTracker):
    """Test 5: Validate balance analysis functionality"""
    print("\n" + "=" * 80)
    print("TEST 5: Balance Analysis")
    print("=" * 80)

    print("\n[5.1] Testing balance analysis on ResNet-50...")
    model = models.resnet50(weights=None)
    model.eval()
    fx_graph = trace_model(model)

    if fx_graph is None:
        results.fail_test("Balance analysis", "Failed to trace ResNet-50")
        return

    partitioner = FusionBasedPartitioner()
    report = partitioner.partition(fx_graph)

    # Run balance analysis
    analysis = partitioner.analyze_balance()

    # Test that analysis contains expected sections
    expected_sections = [
        "FUSION SIZE DISTRIBUTION",
        "FUSION QUALITY ANALYSIS",
        "TOP FUSION PATTERNS",
        "BOTTLENECK DISTRIBUTION",
        "MISSED FUSION OPPORTUNITIES",
        "FUSION STRATEGY COMPARISON",
        "RECOMMENDATIONS"
    ]

    for section in expected_sections:
        if section in analysis:
            results.pass_test(f"Balance section: {section}", "Present")
        else:
            results.fail_test(f"Balance section: {section}", "Missing from analysis")

    # Test categorization works
    if "Structural Operations:" in analysis and "Potentially Fusible Operations:" in analysis:
        results.pass_test("Single-op categorization", "Structural vs fusible separation")
    else:
        results.fail_test("Single-op categorization", "Not properly categorized")

    # Test baseline comparison works
    if "Baseline (Sequential Only):" in analysis and "Actual (Smart Fusion):" in analysis:
        results.pass_test("Baseline comparison", "Present in analysis")
    else:
        results.fail_test("Baseline comparison", "Missing from analysis")


def test_edge_cases(results: ResultsTracker):
    """Test 6: Edge cases and error handling"""
    print("\n" + "=" * 80)
    print("TEST 6: Edge Cases")
    print("=" * 80)

    # Test 1: Single layer model
    print("\n[6.1] Testing single layer model...")
    class SingleConv(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 64, 3)

        def forward(self, x):
            return self.conv(x)

    model = SingleConv()
    fx_graph = trace_model(model)

    if fx_graph:
        partitioner = FusionBasedPartitioner()
        report = partitioner.partition(fx_graph)

        if len(report.fused_subgraphs) > 0:
            results.pass_test("Single layer", f"{len(report.fused_subgraphs)} subgraph(s)")
        else:
            results.fail_test("Single layer", "No subgraphs created")
    else:
        results.fail_test("Single layer", "Failed to trace")

    # Test 2: Model with skip connections
    print("\n[6.2] Testing model with skip connections...")
    class SkipConnection(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(64, 64, 3, padding=1)
            self.conv2 = nn.Conv2d(64, 64, 3, padding=1)

        def forward(self, x):
            identity = x
            out = self.conv1(x)
            out = self.conv2(out)
            out = out + identity
            return out

    model = SkipConnection()
    fx_graph = trace_model(model, input_shape=(1, 64, 32, 32))

    if fx_graph:
        partitioner = FusionBasedPartitioner()
        report = partitioner.partition(fx_graph)

        # Should handle add operation
        has_add = any('add' in sg.fusion_pattern.lower()
                     for sg in report.fused_subgraphs)

        if has_add:
            results.pass_test("Skip connection", "add operation handled")
        else:
            results.warn("Skip connection: add not found in patterns")
    else:
        results.fail_test("Skip connection", "Failed to trace")


def test_diverse_architectures(results: ResultsTracker):
    """Test 7: Validate on diverse modern architectures"""
    print("\n" + "=" * 80)
    print("TEST 7: Diverse Architecture Validation")
    print("=" * 80)

    architectures = [
        ('ResNet-18', models.resnet18, (1, 3, 224, 224)),
        ('MobileNet-V2', models.mobilenet_v2, (1, 3, 224, 224)),
        ('EfficientNet-B0', models.efficientnet_b0, (1, 3, 224, 224)),
        ('ViT-B/16', models.vit_b_16, (1, 3, 224, 224)),
    ]

    results_table = []

    for name, model_fn, input_shape in architectures:
        print(f"\n[7.{architectures.index((name, model_fn, input_shape)) + 1}] Testing {name}...")

        try:
            model = model_fn(weights=None)
            model.eval()
            fx_graph = trace_model(model, input_shape)

            if fx_graph is None:
                results.warn(f"{name}: Failed to trace")
                continue

            partitioner = FusionBasedPartitioner()
            report = partitioner.partition(fx_graph)

            total_ops = sum(sg.num_operators for sg in report.fused_subgraphs)
            fusion_efficiency = total_ops / len(report.fused_subgraphs)

            results_table.append({
                'name': name,
                'subgraphs': len(report.fused_subgraphs),
                'operators': total_ops,
                'efficiency': fusion_efficiency,
                'flops_g': report.total_flops / 1e9,
                'memory_mb': report.total_memory_traffic_fused / 1e6,
                'reduction': report.data_movement_reduction * 100
            })

            print(f"  Subgraphs: {len(report.fused_subgraphs)}")
            print(f"  Fusion efficiency: {fusion_efficiency:.2f}×")
            print(f"  FLOPs: {report.total_flops / 1e9:.2f}G")
            print(f"  Data movement reduction: {report.data_movement_reduction * 100:.1f}%")

            # Validation checks
            if fusion_efficiency >= 1.5:
                results.pass_test(f"{name} fusion", f"{fusion_efficiency:.2f}× efficiency")
            else:
                results.warn(f"{name} low fusion efficiency: {fusion_efficiency:.2f}×")

        except Exception as e:
            results.fail_test(f"{name}", f"Exception: {e}")

    # Print comparison table
    if results_table:
        print("\n" + "=" * 80)
        print("ARCHITECTURE COMPARISON")
        print("=" * 80)
        print(f"{'Architecture':<20} {'Subgraphs':>10} {'Efficiency':>12} {'FLOPs(G)':>10} {'Reduction':>10}")
        print("-" * 80)
        for row in results_table:
            print(f"{row['name']:<20} {row['subgraphs']:>10} {row['efficiency']:>11.2f}× "
                  f"{row['flops_g']:>9.2f} {row['reduction']:>9.1f}%")


def run_all_tests():
    """Run all test suites"""
    print("=" * 80)
    print("FUSION PARTITIONER UNIT TESTS")
    print("=" * 80)

    results = ResultsTracker()

    # Run test suites
    test_fusion_pattern_detection(results)
    test_metrics_calculations(results)
    test_fvcore_comparison(results)
    test_fusion_quality(results)
    test_balance_analysis(results)
    test_edge_cases(results)
    test_diverse_architectures(results)

    # Print summary
    results.print_summary()

    return results.all_passed()


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
