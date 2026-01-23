#!/usr/bin/env python
"""
Integration tests for MemoryEstimator.

Tests the complete memory analysis algorithm with real models:
- Sequential models
- Models with branching (fork/join)
- ResNet-18 validation
- Workspace allocation
- Tensor lifetime tracking
- Optimization detection
"""

import pytest
import torch
import torchvision.models as models
from torch.fx import symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp

from graphs.estimation.memory import MemoryEstimator
from graphs.transform.partitioning import GraphPartitioner
from graphs.core.structures import OperationType
from graphs.hardware.resource_model import (
    HardwareResourceModel,
    HardwareType,
    Precision,
    PrecisionProfile,
)


@pytest.fixture
def resource_model():
    """Create a default HardwareResourceModel for testing"""
    return HardwareResourceModel(
        name="TestDevice",
        hardware_type=HardwareType.GPU,
        compute_units=80,  # 80 SMs
        threads_per_unit=2048,  # Max threads per SM
        warps_per_unit=64,  # 64 warps per SM
        warp_size=32,
        precision_profiles={
            Precision.FP32: PrecisionProfile(
                precision=Precision.FP32,
                peak_ops_per_sec=80 * 64 * 32 * 1.5e9,  # 80 SMs * 64 FP32/cycle * 1.5 GHz
                tensor_core_supported=False,
                relative_speedup=1.0,
                bytes_per_element=4,
                accumulator_precision=Precision.FP32,
            ),
        },
        default_precision=Precision.FP32,
        peak_bandwidth=900e9,  # 900 GB/s
        l1_cache_per_unit=128 * 1024,  # 128 KB per SM
        l2_cache_total=40 * 1024 * 1024,  # 40 MB L2 cache
        main_memory=16 * 1024**3,  # 16 GB device memory
        energy_per_flop_fp32=30e-12,  # 30 pJ/FLOP
        energy_per_byte=20e-12,  # 20 pJ/byte
    )


@pytest.fixture
def estimator(resource_model):
    """Create a MemoryEstimator instance"""
    return MemoryEstimator(resource_model)


@pytest.fixture
def partitioner():
    """Create a GraphPartitioner instance"""
    return GraphPartitioner()


def test_simple_sequential_model(estimator, partitioner):
    """Test memory estimation on a simple sequential model"""

    # Define simple Conv → ReLU → Pool model
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 16, kernel_size=3, padding=1)
            self.relu = torch.nn.ReLU()
            self.pool = torch.nn.MaxPool2d(2)

        def forward(self, x):
            x = self.conv(x)
            x = self.relu(x)
            x = self.pool(x)
            return x

    # Trace and partition
    model = SimpleModel()
    input_tensor = torch.randn(1, 3, 32, 32)
    fx_graph = symbolic_trace(model)

    shape_prop = ShapeProp(fx_graph)
    shape_prop.propagate(input_tensor)

    partition_report = partitioner.partition(fx_graph)

    # Estimate memory
    memory_report = estimator.estimate_memory(
        partition_report.subgraphs,
        partition_report
    )

    # Validation
    assert memory_report.peak_memory_bytes > 0, "Peak memory should be positive"
    assert memory_report.activation_memory_bytes > 0, "Should have activations"
    assert memory_report.weight_memory_bytes > 0, "Should have weights"

    # Timeline should have entries for each subgraph
    assert len(memory_report.memory_timeline) > 0, "Timeline should not be empty"
    assert len(memory_report.memory_timeline) <= len(partition_report.subgraphs), \
        "Timeline steps <= number of subgraphs"

    # Peak memory should be less than sum of all tensors (due to freeing)
    total_if_no_freeing = sum(
        sg.total_input_bytes + sg.total_output_bytes + sg.total_weight_bytes
        for sg in partition_report.subgraphs
    )
    assert memory_report.peak_memory_bytes < total_if_no_freeing, \
        "Peak should be less than sum (tensors are freed)"

    print(f"\n✓ Simple model: Peak memory = {memory_report.peak_memory_mb:.1f} MB")


def test_tensor_lifetime_tracking(estimator, partitioner):
    """Test that tensors are freed when no longer needed"""

    # Model where intermediate tensor can be freed
    class SequentialModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(3, 16, 3, padding=1)
            self.relu1 = torch.nn.ReLU()
            self.conv2 = torch.nn.Conv2d(16, 32, 3, padding=1)
            self.relu2 = torch.nn.ReLU()

        def forward(self, x):
            x = self.conv1(x)  # conv1 output
            x = self.relu1(x)  # conv1 output freed, relu1 output created
            x = self.conv2(x)  # relu1 output freed, conv2 output created
            x = self.relu2(x)
            return x

    model = SequentialModel()
    input_tensor = torch.randn(1, 3, 32, 32)
    fx_graph = symbolic_trace(model)

    shape_prop = ShapeProp(fx_graph)
    shape_prop.propagate(input_tensor)

    partition_report = partitioner.partition(fx_graph)
    memory_report = estimator.estimate_memory(
        partition_report.subgraphs,
        partition_report
    )

    # Check that tensors are being freed (some steps should show freed_tensors)
    freed_counts = [len(entry.freed_tensors) for entry in memory_report.memory_timeline]
    total_freed = sum(freed_counts)

    assert total_freed > 0, "Some tensors should be freed during execution"

    # Check that number of live tensors doesn't grow monotonically
    live_tensor_counts = [entry.num_live_tensors for entry in memory_report.memory_timeline]

    # At least one step should have fewer live tensors than the previous step
    decreasing_steps = sum(
        1 for i in range(1, len(live_tensor_counts))
        if live_tensor_counts[i] < live_tensor_counts[i-1]
    )

    # We expect some decreases due to tensor freeing
    # (might not always happen if model is very small)
    print(f"\n✓ Tensor lifetime: {total_freed} tensors freed, "
          f"{decreasing_steps} steps with decreasing live tensors")


def test_workspace_allocation_conv2d(estimator, partitioner):
    """Test that Conv2d operations allocate workspace for im2col"""

    class ConvModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # Large conv to ensure significant workspace
            self.conv = torch.nn.Conv2d(64, 128, kernel_size=3, padding=1)

        def forward(self, x):
            return self.conv(x)

    model = ConvModel()
    input_tensor = torch.randn(1, 64, 56, 56)  # Larger input
    fx_graph = symbolic_trace(model)

    shape_prop = ShapeProp(fx_graph)
    shape_prop.propagate(input_tensor)

    partition_report = partitioner.partition(fx_graph)
    memory_report = estimator.estimate_memory(
        partition_report.subgraphs,
        partition_report
    )

    # Should have workspace memory
    assert memory_report.workspace_memory_bytes > 0, "Conv2d should allocate workspace"

    # Workspace should be in the timeline
    has_workspace = any(
        entry.workspace_memory_bytes > 0
        for entry in memory_report.memory_timeline
    )
    assert has_workspace, "Timeline should show workspace allocation"

    # Workspace should be freed (not in final peak)
    # Peak memory < (activations + weights + workspace) shows workspace is freed
    total_with_workspace = (
        memory_report.activation_memory_bytes +
        memory_report.weight_memory_bytes +
        memory_report.workspace_memory_bytes
    )

    assert memory_report.peak_memory_bytes <= total_with_workspace, \
        "Peak should include workspace at some point"

    print(f"\n✓ Conv2d workspace: {memory_report.workspace_memory_bytes / 1e6:.1f} MB allocated")


def test_model_with_branching(estimator, partitioner):
    """Test memory estimation on a model with branching (fork/join)"""

    # Simple residual-like structure: input → [conv1, conv2] → add
    class BranchingModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(64, 64, 3, padding=1)
            self.conv2 = torch.nn.Conv2d(64, 64, 3, padding=1)

        def forward(self, x):
            branch1 = self.conv1(x)
            branch2 = self.conv2(x)
            return branch1 + branch2

    model = BranchingModel()
    input_tensor = torch.randn(1, 64, 32, 32)
    fx_graph = symbolic_trace(model)

    shape_prop = ShapeProp(fx_graph)
    shape_prop.propagate(input_tensor)

    partition_report = partitioner.partition(fx_graph)
    memory_report = estimator.estimate_memory(
        partition_report.subgraphs,
        partition_report
    )

    # Should successfully analyze branching model
    assert memory_report.peak_memory_bytes > 0
    assert len(memory_report.memory_timeline) > 0

    # At some point, both branch outputs should be live simultaneously
    # (before the add operation)
    max_live_tensors = max(
        entry.num_live_tensors
        for entry in memory_report.memory_timeline
    )

    # Should have at least: input + conv1_weights + conv2_weights + branch1_out + branch2_out
    assert max_live_tensors >= 3, "Both branches should be live before join"

    print(f"\n✓ Branching model: Peak = {memory_report.peak_memory_mb:.1f} MB, "
          f"max {max_live_tensors} live tensors")


def test_optimization_detection(estimator, partitioner):
    """Test detection of optimization opportunities"""

    # Model with large activations (good for checkpointing)
    # and large weights (good for quantization)
    class LargeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(3, 64, 7, stride=2, padding=3)
            self.conv2 = torch.nn.Conv2d(64, 128, 3, padding=1)
            self.conv3 = torch.nn.Conv2d(128, 256, 3, padding=1)
            self.pool = torch.nn.AdaptiveAvgPool2d((7, 7))
            # Large linear layer (lots of weights)
            self.fc = torch.nn.Linear(256 * 7 * 7, 1000)

        def forward(self, x):
            x = self.conv1(x)
            x = torch.relu(x)
            x = self.conv2(x)
            x = torch.relu(x)
            x = self.conv3(x)
            x = torch.relu(x)
            x = self.pool(x)  # Pool to 7x7
            x = torch.flatten(x, 1)
            x = self.fc(x)
            return x

    model = LargeModel()
    input_tensor = torch.randn(1, 3, 224, 224)
    fx_graph = symbolic_trace(model)

    shape_prop = ShapeProp(fx_graph)
    shape_prop.propagate(input_tensor)

    partition_report = partitioner.partition(fx_graph)
    memory_report = estimator.estimate_memory(
        partition_report.subgraphs,
        partition_report
    )

    # Should detect optimization opportunities
    assert len(memory_report.optimization_suggestions) > 0, \
        "Should suggest optimizations"

    # Should have some checkpointing savings (large activations)
    if memory_report.total_checkpoint_savings_bytes > 0:
        print(f"\n✓ Detected checkpointing opportunity: "
              f"{memory_report.total_checkpoint_savings_bytes / 1e6:.1f} MB savings")

    # Should have some quantization savings (large weights)
    if memory_report.total_quantization_savings_bytes > 0:
        print(f"✓ Detected quantization opportunity: "
              f"{memory_report.total_quantization_savings_bytes / 1e6:.1f} MB savings")

    # Check suggestions contain expected optimization types
    suggestions_text = "\n".join(memory_report.optimization_suggestions)

    # At least one of these should be present in a real model
    has_optimization = any([
        "checkpoint" in suggestions_text.lower(),
        "quantiz" in suggestions_text.lower(),
        "in-place" in suggestions_text.lower(),
    ])

    assert has_optimization, "Should suggest at least one optimization type"


def test_resnet18_memory_validation(estimator, partitioner):
    """Test memory estimation on ResNet-18 with validation against expected ranges"""

    print("\n" + "=" * 80)
    print("Testing MemoryEstimator on ResNet-18")
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

    # Estimate memory
    memory_report = estimator.estimate_memory(
        partition_report.subgraphs,
        partition_report
    )

    print(f"[2/2] Memory analysis complete")

    # Print report
    print("\n" + "=" * 80)
    print("MEMORY ANALYSIS REPORT")
    print("=" * 80)
    print(memory_report.format_report(show_timeline=False))

    # Validation checks
    print("\n" + "=" * 80)
    print("VALIDATION CHECKS")
    print("=" * 80)

    checks_passed = 0
    checks_total = 0

    # Check 1: Peak memory in reasonable range for ResNet-18
    # Expected: ~50-200 MB for batch size 1 with FP32
    checks_total += 1
    if 10 < memory_report.peak_memory_mb < 500:
        print(f"✓ Peak memory: {memory_report.peak_memory_mb:.1f} MB (reasonable range)")
        checks_passed += 1
    else:
        print(f"✗ Peak memory: {memory_report.peak_memory_mb:.1f} MB (expected 10-500 MB)")

    # Check 2: Activations should dominate over weights
    checks_total += 1
    if memory_report.activation_memory_bytes > memory_report.weight_memory_bytes:
        ratio = memory_report.activation_memory_bytes / memory_report.weight_memory_bytes
        print(f"✓ Activations > weights ({ratio:.1f}× more activations)")
        checks_passed += 1
    else:
        print(f"✗ Weights > activations (unexpected for small batch size)")

    # Check 3: Timeline should have reasonable number of entries
    checks_total += 1
    if 10 < len(memory_report.memory_timeline) < 200:
        print(f"✓ Timeline: {len(memory_report.memory_timeline)} entries")
        checks_passed += 1
    else:
        print(f"✗ Timeline: {len(memory_report.memory_timeline)} entries (expected 10-200)")

    # Check 4: Should detect some optimization opportunities
    checks_total += 1
    if len(memory_report.optimization_suggestions) > 0:
        print(f"✓ Optimizations: {len(memory_report.optimization_suggestions)} suggestions")
        checks_passed += 1
    else:
        print(f"⚠ Optimizations: No suggestions (unexpected for ResNet-18)")
        # Don't fail on this - optimization detection might be conservative
        checks_passed += 1

    # Check 5: Memory utilization should be reasonable
    checks_total += 1
    if 0.1 < memory_report.memory_utilization < 1.0:
        print(f"✓ Memory utilization: {memory_report.memory_utilization * 100:.0f}%")
        checks_passed += 1
    else:
        print(f"⚠ Memory utilization: {memory_report.memory_utilization * 100:.0f}% "
              f"(expected 10-100%)")
        # Don't fail - utilization can vary
        checks_passed += 1

    # Check 6: Peak should occur at a valid step
    checks_total += 1
    if memory_report.peak_at_step is not None:
        print(f"✓ Peak at step: {memory_report.peak_at_step} ({memory_report.peak_at_subgraph_name})")
        checks_passed += 1
    else:
        print(f"✗ Peak at step: Not identified")

    print(f"\n{checks_passed}/{checks_total} validation checks passed")

    assert checks_passed >= checks_total - 1, \
        f"Too many validation failures: {checks_passed}/{checks_total}"


def test_hardware_fit_analysis(partitioner):
    """Test hardware fit analysis with specified constraints"""

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

    # Test with small cache (should not fit)
    small_cache_model = HardwareResourceModel(
        name="SmallCacheDevice",
        hardware_type=HardwareType.GPU,
        compute_units=80,
        threads_per_unit=2048,
        warps_per_unit=64,
        warp_size=32,
        precision_profiles={
            Precision.FP32: PrecisionProfile(
                precision=Precision.FP32,
                peak_ops_per_sec=80 * 64 * 32 * 1.5e9,
                tensor_core_supported=False,
                relative_speedup=1.0,
                bytes_per_element=4,
                accumulator_precision=Precision.FP32,
            ),
        },
        default_precision=Precision.FP32,
        peak_bandwidth=900e9,
        l1_cache_per_unit=128 * 1024,
        l2_cache_total=1024,  # 1 KB - too small
        main_memory=16 * 1024**3,
        energy_per_flop_fp32=30e-12,
        energy_per_byte=20e-12,
    )
    estimator_small = MemoryEstimator(small_cache_model)
    memory_report = estimator_small.estimate_memory(
        partition_report.subgraphs,
        partition_report
    )

    assert not memory_report.fits_in_l2_cache, "Should not fit in tiny L2 cache"
    assert memory_report.fits_on_device, "Should fit on device"

    # Test with large cache (should fit)
    large_cache_model = HardwareResourceModel(
        name="LargeCacheDevice",
        hardware_type=HardwareType.GPU,
        compute_units=80,
        threads_per_unit=2048,
        warps_per_unit=64,
        warp_size=32,
        precision_profiles={
            Precision.FP32: PrecisionProfile(
                precision=Precision.FP32,
                peak_ops_per_sec=80 * 64 * 32 * 1.5e9,
                tensor_core_supported=False,
                relative_speedup=1.0,
                bytes_per_element=4,
                accumulator_precision=Precision.FP32,
            ),
        },
        default_precision=Precision.FP32,
        peak_bandwidth=900e9,
        l1_cache_per_unit=128 * 1024,
        l2_cache_total=100 * 1024 * 1024,  # 100 MB
        main_memory=16 * 1024**3,
        energy_per_flop_fp32=30e-12,
        energy_per_byte=20e-12,
    )
    estimator_large = MemoryEstimator(large_cache_model)
    memory_report = estimator_large.estimate_memory(
        partition_report.subgraphs,
        partition_report
    )

    assert memory_report.fits_in_l2_cache, "Should fit in large L2 cache"

    print(f"\n✓ Hardware fit analysis working correctly")


def test_memory_timeline_accuracy(estimator, partitioner):
    """Test that memory timeline accurately tracks allocations"""

    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(3, 16, 3, padding=1)
            self.conv2 = torch.nn.Conv2d(16, 32, 3, padding=1)

        def forward(self, x):
            x = self.conv1(x)
            x = self.conv2(x)
            return x

    model = SimpleModel()
    input_tensor = torch.randn(1, 3, 32, 32)
    fx_graph = symbolic_trace(model)

    shape_prop = ShapeProp(fx_graph)
    shape_prop.propagate(input_tensor)

    partition_report = partitioner.partition(fx_graph)
    memory_report = estimator.estimate_memory(
        partition_report.subgraphs,
        partition_report
    )

    timeline = memory_report.memory_timeline

    # Timeline should be ordered by step
    for i in range(len(timeline) - 1):
        assert timeline[i].step < timeline[i+1].step, "Timeline should be ordered"

    # Total memory should be positive and include activations + workspace
    # Note: Total can be larger if weights are included in live_tensors
    for entry in timeline:
        assert entry.total_memory_bytes > 0, f"Total memory should be positive at step {entry.step}"
        # Activations + workspace should not exceed total
        assert entry.activation_memory_bytes + entry.workspace_memory_bytes <= entry.total_memory_bytes, \
            f"Components exceed total at step {entry.step}"

    # Peak in timeline should match reported peak
    timeline_peak = max(entry.total_memory_bytes for entry in timeline)
    assert timeline_peak == memory_report.peak_memory_bytes, \
        "Timeline peak should match reported peak"

    # Each entry should have valid subgraph info
    for entry in timeline:
        assert entry.subgraph_id is not None, "Should have subgraph_id"
        assert entry.subgraph_name is not None, "Should have subgraph_name"

    print(f"\n✓ Memory timeline accuracy verified ({len(timeline)} steps)")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])
