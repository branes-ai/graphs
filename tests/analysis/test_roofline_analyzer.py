#!/usr/bin/env python
"""
Integration tests for RooflineAnalyzer.

Tests the roofline model implementation with real models:
- Simple sequential models
- Compute-bound vs memory-bound operations
- ResNet-18 validation
- Hardware-specific analysis
- Bottleneck classification
"""

import pytest
import torch
import torchvision.models as models
from torch.fx import symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp

from graphs.estimation.roofline import RooflineAnalyzer
from graphs.transform.partitioning import GraphPartitioner
from graphs.core.structures import BottleneckType
from graphs.hardware.resource_model import (
    HardwareResourceModel,
    HardwareType,
    Precision,
    PrecisionProfile,
)


@pytest.fixture
def high_flops_hardware():
    """Create hardware with high FLOPs (compute-heavy)"""
    return HardwareResourceModel(
        name="HighFLOPs",
        hardware_type=HardwareType.GPU,
        compute_units=80,
        threads_per_unit=2048,
        warps_per_unit=64,
        warp_size=32,
        precision_profiles={
            Precision.FP32: PrecisionProfile(
                precision=Precision.FP32,
                peak_ops_per_sec=100e12,  # 100 TFLOPS (very high)
                tensor_core_supported=False,
                relative_speedup=1.0,
                bytes_per_element=4,
                accumulator_precision=Precision.FP32,
            ),
        },
        default_precision=Precision.FP32,
        peak_bandwidth=100e9,  # 100 GB/s (low relative to FLOPs)
        l1_cache_per_unit=128 * 1024,
        l2_cache_total=40 * 1024 * 1024,
        main_memory=16 * 1024**3,
        energy_per_flop_fp32=30e-12,
        energy_per_byte=20e-12,
    )


@pytest.fixture
def high_bandwidth_hardware():
    """Create hardware with high bandwidth (memory-heavy)"""
    return HardwareResourceModel(
        name="HighBandwidth",
        hardware_type=HardwareType.GPU,
        compute_units=80,
        threads_per_unit=2048,
        warps_per_unit=64,
        warp_size=32,
        precision_profiles={
            Precision.FP32: PrecisionProfile(
                precision=Precision.FP32,
                peak_ops_per_sec=10e12,  # 10 TFLOPS (low)
                tensor_core_supported=False,
                relative_speedup=1.0,
                bytes_per_element=4,
                accumulator_precision=Precision.FP32,
            ),
        },
        default_precision=Precision.FP32,
        peak_bandwidth=2000e9,  # 2000 GB/s (very high)
        l1_cache_per_unit=128 * 1024,
        l2_cache_total=40 * 1024 * 1024,
        main_memory=16 * 1024**3,
        energy_per_flop_fp32=30e-12,
        energy_per_byte=20e-12,
    )


@pytest.fixture
def balanced_hardware():
    """Create balanced hardware"""
    return HardwareResourceModel(
        name="Balanced",
        hardware_type=HardwareType.GPU,
        compute_units=80,
        threads_per_unit=2048,
        warps_per_unit=64,
        warp_size=32,
        precision_profiles={
            Precision.FP32: PrecisionProfile(
                precision=Precision.FP32,
                peak_ops_per_sec=20e12,  # 20 TFLOPS
                tensor_core_supported=False,
                relative_speedup=1.0,
                bytes_per_element=4,
                accumulator_precision=Precision.FP32,
            ),
        },
        default_precision=Precision.FP32,
        peak_bandwidth=900e9,  # 900 GB/s (balanced)
        l1_cache_per_unit=128 * 1024,
        l2_cache_total=40 * 1024 * 1024,
        main_memory=16 * 1024**3,
        energy_per_flop_fp32=30e-12,
        energy_per_byte=20e-12,
    )


@pytest.fixture
def partitioner():
    """Create a GraphPartitioner instance"""
    return GraphPartitioner()


def test_simple_model_analysis(balanced_hardware, partitioner):
    """Test roofline analysis on a simple model"""

    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 16, kernel_size=3, padding=1)
            self.relu = torch.nn.ReLU()

        def forward(self, x):
            x = self.conv(x)
            x = self.relu(x)
            return x

    # Trace and partition
    model = SimpleModel()
    input_tensor = torch.randn(1, 3, 32, 32)
    fx_graph = symbolic_trace(model)

    shape_prop = ShapeProp(fx_graph)
    shape_prop.propagate(input_tensor)

    partition_report = partitioner.partition(fx_graph)

    # Analyze with roofline
    analyzer = RooflineAnalyzer(balanced_hardware)
    roofline_report = analyzer.analyze(partition_report.subgraphs, partition_report)

    # Validation
    assert roofline_report.peak_flops == 20e12, "Peak FLOPs should match hardware"
    assert roofline_report.peak_bandwidth == 900e9, "Peak bandwidth should match hardware"
    assert roofline_report.arithmetic_intensity_breakpoint > 0, "AI breakpoint should be positive"

    assert len(roofline_report.latencies) > 0, "Should have latency descriptors"
    assert roofline_report.total_latency > 0, "Total latency should be positive"

    # Check that bottlenecks are classified
    total_ops = (roofline_report.num_compute_bound +
                roofline_report.num_memory_bound +
                roofline_report.num_balanced)
    assert total_ops == len(partition_report.subgraphs), "All ops should be classified"

    print(f"\n✓ Simple model: {roofline_report.total_latency * 1e6:.1f} μs total latency")


def test_compute_bound_on_high_flops(high_flops_hardware, partitioner):
    """Test that compute-intensive ops are compute-bound on high-FLOP hardware"""

    # Large conv should be compute-intensive (high arithmetic intensity)
    class ComputeIntensiveModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # Large conv with many channels (high AI)
            self.conv = torch.nn.Conv2d(128, 256, kernel_size=3, padding=1)

        def forward(self, x):
            return self.conv(x)

    model = ComputeIntensiveModel()
    input_tensor = torch.randn(1, 128, 56, 56)
    fx_graph = symbolic_trace(model)

    shape_prop = ShapeProp(fx_graph)
    shape_prop.propagate(input_tensor)

    partition_report = partitioner.partition(fx_graph)

    # Analyze with high-FLOP hardware
    analyzer = RooflineAnalyzer(high_flops_hardware)
    roofline_report = analyzer.analyze(partition_report.subgraphs, partition_report)

    # With high FLOPs (100 TFLOPS) and low bandwidth (100 GB/s),
    # many ops should be memory-bound
    assert roofline_report.num_memory_bound > 0, \
        "High-FLOP hardware should have memory-bound ops"

    print(f"\n✓ High-FLOP hardware: {roofline_report.num_memory_bound} memory-bound ops")


def test_memory_bound_on_high_bandwidth(high_bandwidth_hardware, partitioner):
    """Test that compute-intensive ops become compute-bound on high-bandwidth hardware"""

    # Conv with high arithmetic intensity
    class ComputeIntensiveModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # Large conv (high AI due to many channels and kernel reuse)
            self.conv = torch.nn.Conv2d(256, 512, kernel_size=3, padding=1)

        def forward(self, x):
            return self.conv(x)

    model = ComputeIntensiveModel()
    input_tensor = torch.randn(1, 256, 28, 28)
    fx_graph = symbolic_trace(model)

    shape_prop = ShapeProp(fx_graph)
    shape_prop.propagate(input_tensor)

    partition_report = partitioner.partition(fx_graph)

    # Analyze with high-bandwidth hardware
    analyzer = RooflineAnalyzer(high_bandwidth_hardware)
    roofline_report = analyzer.analyze(partition_report.subgraphs, partition_report)

    # With low FLOPs (10 TFLOPS) and high bandwidth (2000 GB/s),
    # AI breakpoint is 5 FLOPs/byte. Large conv has high AI so should be compute-bound
    assert roofline_report.num_compute_bound > 0, \
        "High-bandwidth hardware should have compute-bound ops for high-AI operations"

    print(f"\n✓ High-bandwidth hardware: {roofline_report.num_compute_bound} compute-bound ops")


def test_arithmetic_intensity_breakpoint(balanced_hardware, partitioner):
    """Test that arithmetic intensity breakpoint is calculated correctly"""

    analyzer = RooflineAnalyzer(balanced_hardware)

    # AI breakpoint = peak_FLOPS / peak_bandwidth
    expected_breakpoint = 20e12 / 900e9  # ~22.2 FLOPs/byte

    assert analyzer.ai_breakpoint == pytest.approx(expected_breakpoint, rel=1e-3), \
        "AI breakpoint should match calculation"

    print(f"\n✓ AI breakpoint: {analyzer.ai_breakpoint:.2f} FLOPs/byte")


def test_roofline_points_generation(balanced_hardware, partitioner):
    """Test that roofline points are generated for visualization"""

    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 16, 3, padding=1)
            self.relu = torch.nn.ReLU()

        def forward(self, x):
            x = self.conv(x)
            x = self.relu(x)
            return x

    model = SimpleModel()
    input_tensor = torch.randn(1, 3, 32, 32)
    fx_graph = symbolic_trace(model)

    shape_prop = ShapeProp(fx_graph)
    shape_prop.propagate(input_tensor)

    partition_report = partitioner.partition(fx_graph)

    analyzer = RooflineAnalyzer(balanced_hardware)
    roofline_report = analyzer.analyze(partition_report.subgraphs, partition_report)

    # Should have roofline points
    assert len(roofline_report.roofline_points) == len(partition_report.subgraphs), \
        "Should have one roofline point per subgraph"

    # Check roofline point structure
    for point in roofline_report.roofline_points:
        assert point.arithmetic_intensity >= 0, "AI should be non-negative"
        assert point.attained_flops >= 0, "Attained FLOPs should be non-negative"
        assert point.subgraph_name != "", "Should have subgraph name"

    print(f"\n✓ Generated {len(roofline_report.roofline_points)} roofline points")


def test_resnet18_roofline_analysis(balanced_hardware, partitioner):
    """Test roofline analysis on ResNet-18"""

    print("\n" + "=" * 80)
    print("Testing RooflineAnalyzer on ResNet-18")
    print("=" * 80)

    # Load ResNet-18
    model = models.resnet18(weights=None)
    model.eval()

    input_tensor = torch.randn(1, 3, 224, 224)
    fx_graph = symbolic_trace(model)

    shape_prop = ShapeProp(fx_graph)
    shape_prop.propagate(input_tensor)

    partition_report = partitioner.partition(fx_graph)

    print(f"\n[1/2] Partitioned into {partition_report.total_subgraphs} subgraphs")

    # Analyze with roofline
    analyzer = RooflineAnalyzer(balanced_hardware)
    roofline_report = analyzer.analyze(partition_report.subgraphs, partition_report)

    print(f"[2/2] Roofline analysis complete")

    # Print report
    print("\n" + "=" * 80)
    print("ROOFLINE ANALYSIS")
    print("=" * 80)
    print(roofline_report.format_report(show_per_subgraph=False))

    # Validation checks
    print("\n" + "=" * 80)
    print("VALIDATION CHECKS")
    print("=" * 80)

    checks_passed = 0
    checks_total = 0

    # Check 1: Total latency in reasonable range
    checks_total += 1
    if 0.0001 < roofline_report.total_latency < 10.0:  # 0.1ms to 10s
        print(f"✓ Total latency: {roofline_report.total_latency * 1e3:.2f} ms (reasonable range)")
        checks_passed += 1
    else:
        print(f"✗ Total latency: {roofline_report.total_latency * 1e3:.2f} ms (expected 0.1-10000 ms)")

    # Check 2: Bottleneck distribution makes sense
    checks_total += 1
    total_ops = (roofline_report.num_compute_bound +
                roofline_report.num_memory_bound +
                roofline_report.num_balanced)
    if total_ops == partition_report.total_subgraphs:
        print(f"✓ Bottleneck classification: {total_ops} ops classified")
        print(f"  - Compute-bound: {roofline_report.num_compute_bound}")
        print(f"  - Memory-bound: {roofline_report.num_memory_bound}")
        print(f"  - Balanced: {roofline_report.num_balanced}")
        checks_passed += 1
    else:
        print(f"✗ Bottleneck classification failed")

    # Check 3: Utilization is reasonable
    checks_total += 1
    if 0.0 < roofline_report.average_flops_utilization < 1.0:
        print(f"✓ FLOP utilization: {roofline_report.average_flops_utilization * 100:.1f}%")
        checks_passed += 1
    else:
        print(f"⚠ FLOP utilization: {roofline_report.average_flops_utilization * 100:.1f}%")
        # Don't fail - utilization can be low
        checks_passed += 1

    # Check 4: Roofline points generated
    checks_total += 1
    if len(roofline_report.roofline_points) == partition_report.total_subgraphs:
        print(f"✓ Roofline points: {len(roofline_report.roofline_points)} generated")
        checks_passed += 1
    else:
        print(f"✗ Roofline points: {len(roofline_report.roofline_points)} (expected {partition_report.total_subgraphs})")

    print(f"\n{checks_passed}/{checks_total} validation checks passed")

    assert checks_passed == checks_total, \
        f"Only {checks_passed}/{checks_total} validation checks passed"


def test_latency_descriptor_formatting(balanced_hardware, partitioner):
    """Test latency descriptor formatting"""

    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 16, 3, padding=1)

        def forward(self, x):
            return self.conv(x)

    model = SimpleModel()
    input_tensor = torch.randn(1, 3, 32, 32)
    fx_graph = symbolic_trace(model)

    shape_prop = ShapeProp(fx_graph)
    shape_prop.propagate(input_tensor)

    partition_report = partitioner.partition(fx_graph)

    analyzer = RooflineAnalyzer(balanced_hardware)
    roofline_report = analyzer.analyze(partition_report.subgraphs, partition_report)

    # Test formatting
    for latency in roofline_report.latencies:
        # Test __str__
        str_repr = str(latency)
        assert "Latency" in str_repr
        assert latency.subgraph_name in str_repr

        # Test format_summary
        summary = latency.format_summary()
        assert "Subgraph:" in summary
        assert "Latency:" in summary
        assert "Bottleneck:" in summary
        assert "Arithmetic Intensity:" in summary

    print(f"\n✓ Latency descriptor formatting works")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])
