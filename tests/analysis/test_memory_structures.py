#!/usr/bin/env python
"""
Unit tests for Memory Analysis data structures.

Tests the basic functionality of MemoryTimelineEntry, MemoryDescriptor,
and MemoryReport without requiring full execution simulation.
"""

import pytest
from graphs.analysis.memory import (
    MemoryTimelineEntry,
    MemoryDescriptor,
    MemoryReport,
)
from graphs.ir.structures import OperationType


def test_memory_timeline_entry_creation():
    """Test basic MemoryTimelineEntry creation"""
    entry = MemoryTimelineEntry(
        step=0,
        subgraph_id="node_1",
        subgraph_name="conv1",
        total_memory_bytes=10 * 1024 * 1024,  # 10 MB
        activation_memory_bytes=8 * 1024 * 1024,  # 8 MB
        workspace_memory_bytes=2 * 1024 * 1024,  # 2 MB
        live_tensors=["input", "conv1_output"],
        num_live_tensors=2,
        allocated_tensors=["conv1_output"],
        freed_tensors=[],
    )

    assert entry.step == 0
    assert entry.total_memory_bytes == 10 * 1024 * 1024
    assert entry.total_memory_mb() == 10.0
    assert entry.total_memory_gb() == pytest.approx(10.0 / 1024, rel=1e-3)
    assert entry.num_live_tensors == 2


def test_memory_timeline_entry_formatting():
    """Test timeline entry formatting"""
    entry = MemoryTimelineEntry(
        step=5,
        subgraph_id="node_5",
        subgraph_name="layer1_conv2",
        total_memory_bytes=50 * 1024 * 1024,  # 50 MB
        activation_memory_bytes=45 * 1024 * 1024,
        workspace_memory_bytes=5 * 1024 * 1024,
        live_tensors=["t1", "t2", "t3"],
        num_live_tensors=3,
        allocated_tensors=["layer1_conv2_output"],
        freed_tensors=["layer1_conv1_output"],
    )

    summary = entry.format_summary()
    assert "Step   5" in summary
    assert "layer1_conv2" in summary
    assert "50.0 MB" in summary
    assert "3 tensors" in summary
    assert "[+1-1]" in summary  # 1 allocated, 1 freed


def test_memory_descriptor_creation():
    """Test basic MemoryDescriptor creation"""
    desc = MemoryDescriptor(
        subgraph_id="node_1",
        subgraph_name="conv1",
        operation_type=OperationType.CONV2D,
        input_memory_bytes=5 * 1024 * 1024,   # 5 MB
        output_memory_bytes=10 * 1024 * 1024,  # 10 MB
        weight_memory_bytes=2 * 1024 * 1024,   # 2 MB
        workspace_memory_bytes=8 * 1024 * 1024,  # 8 MB (im2col)
    )

    assert desc.total_memory_bytes == (10 + 2 + 8) * 1024 * 1024  # out + weight + workspace
    assert desc.peak_memory_bytes == (5 + 10 + 2 + 8) * 1024 * 1024  # all combined
    assert desc.total_memory_mb == 20.0
    assert desc.peak_memory_mb == 25.0


def test_memory_descriptor_optimization_flags():
    """Test optimization potential flags"""
    # Large activation - good for checkpointing
    desc1 = MemoryDescriptor(
        subgraph_id="node_1",
        subgraph_name="large_conv",
        operation_type=OperationType.CONV2D,
        input_memory_bytes=50 * 1024 * 1024,
        output_memory_bytes=50 * 1024 * 1024,
        weight_memory_bytes=1 * 1024 * 1024,
        workspace_memory_bytes=10 * 1024 * 1024,
        can_checkpoint=True,
        checkpoint_savings_bytes=50 * 1024 * 1024,
    )

    assert desc1.can_checkpoint
    assert desc1.checkpoint_savings_bytes == 50 * 1024 * 1024

    # Large weights - good for quantization
    desc2 = MemoryDescriptor(
        subgraph_id="node_2",
        subgraph_name="linear",
        operation_type=OperationType.MATMUL,
        input_memory_bytes=1 * 1024 * 1024,
        output_memory_bytes=1 * 1024 * 1024,
        weight_memory_bytes=100 * 1024 * 1024,  # Large weight matrix
        workspace_memory_bytes=0,
        can_quantize=True,
        quantization_savings_bytes=75 * 1024 * 1024,  # FP32→INT8 = 4× = 75MB saved
    )

    assert desc2.can_quantize
    assert desc2.quantization_savings_bytes == 75 * 1024 * 1024


def test_memory_report_creation():
    """Test basic MemoryReport creation"""
    report = MemoryReport(
        peak_memory_bytes=100 * 1024 * 1024,  # 100 MB
        peak_memory_mb=100.0,
        peak_memory_gb=100.0 / 1024,
        activation_memory_bytes=70 * 1024 * 1024,
        weight_memory_bytes=20 * 1024 * 1024,
        workspace_memory_bytes=10 * 1024 * 1024,
        average_memory_bytes=60 * 1024 * 1024,
        memory_utilization=0.6,  # 60% average utilization
        fits_on_device=True,
        device_memory_bytes=80 * 1024**3,  # 80 GB
    )

    assert report.peak_memory_mb == 100.0
    assert report.peak_memory_gb == pytest.approx(100.0 / 1024, rel=1e-3)
    assert report.memory_utilization == 0.6
    assert report.fits_on_device


def test_memory_report_with_timeline():
    """Test MemoryReport with timeline entries"""
    timeline = [
        MemoryTimelineEntry(
            step=0,
            subgraph_id="n0",
            subgraph_name="input",
            total_memory_bytes=10 * 1024 * 1024,
            activation_memory_bytes=10 * 1024 * 1024,
            workspace_memory_bytes=0,
            num_live_tensors=1,
        ),
        MemoryTimelineEntry(
            step=1,
            subgraph_id="n1",
            subgraph_name="conv1",
            total_memory_bytes=50 * 1024 * 1024,  # Peak here
            activation_memory_bytes=45 * 1024 * 1024,
            workspace_memory_bytes=5 * 1024 * 1024,
            num_live_tensors=3,
        ),
        MemoryTimelineEntry(
            step=2,
            subgraph_id="n2",
            subgraph_name="relu",
            total_memory_bytes=45 * 1024 * 1024,  # Workspace freed
            activation_memory_bytes=45 * 1024 * 1024,
            workspace_memory_bytes=0,
            num_live_tensors=2,
        ),
    ]

    report = MemoryReport(
        peak_memory_bytes=50 * 1024 * 1024,
        peak_memory_mb=50.0,
        peak_memory_gb=50.0 / 1024,
        activation_memory_bytes=45 * 1024 * 1024,
        weight_memory_bytes=5 * 1024 * 1024,
        workspace_memory_bytes=5 * 1024 * 1024,
        average_memory_bytes=35 * 1024 * 1024,
        memory_utilization=0.7,
        fits_on_device=True,
        device_memory_bytes=16 * 1024**3,
        memory_timeline=timeline,
        peak_at_step=1,
        peak_at_subgraph="n1",
        peak_at_subgraph_name="conv1",
    )

    assert len(report.memory_timeline) == 3
    assert report.peak_at_step == 1
    assert report.peak_at_subgraph_name == "conv1"


def test_memory_report_formatting():
    """Test MemoryReport.format_report()"""
    report = MemoryReport(
        peak_memory_bytes=100 * 1024 * 1024,
        peak_memory_mb=100.0,
        peak_memory_gb=100.0 / 1024,
        activation_memory_bytes=80 * 1024 * 1024,
        weight_memory_bytes=15 * 1024 * 1024,
        workspace_memory_bytes=5 * 1024 * 1024,
        average_memory_bytes=70 * 1024 * 1024,
        memory_utilization=0.7,
        fits_on_device=True,
        device_memory_bytes=16 * 1024**3,
        l2_cache_size_bytes=50 * 1024 * 1024,
        fits_in_l2_cache=False,  # 100 MB > 50 MB
        peak_at_step=10,
        peak_at_subgraph="layer3_conv2",
        peak_at_subgraph_name="layer3_1_conv2",
        optimization_suggestions=[
            "✓ Activation checkpointing: Save ~48 MB",
            "✓ INT8 quantization: Save ~11 MB in weights",
        ],
        total_checkpoint_savings_bytes=48 * 1024 * 1024,
        total_quantization_savings_bytes=11 * 1024 * 1024,
    )

    formatted = report.format_report(show_timeline=False)

    # Check key sections present
    assert "MEMORY ANALYSIS REPORT" in formatted
    assert "Peak Memory: 100.0 MB" in formatted
    assert "Activations:" in formatted and "80.0 MB" in formatted
    assert "Weights:" in formatted and "15.0 MB" in formatted
    assert "Hardware Fit Analysis:" in formatted
    assert "Fits on device" in formatted
    assert "Peak occurs at: layer3_1_conv2" in formatted
    assert "Optimization Opportunities:" in formatted
    assert "Activation checkpointing" in formatted
    assert "Estimated Savings:" in formatted


def test_memory_report_str():
    """Test MemoryReport __str__ method"""
    report = MemoryReport(
        peak_memory_bytes=100 * 1024 * 1024,
        peak_memory_mb=100.0,
        peak_memory_gb=100.0 / 1024,
        activation_memory_bytes=80 * 1024 * 1024,
        weight_memory_bytes=20 * 1024 * 1024,
        workspace_memory_bytes=0,
        average_memory_bytes=60 * 1024 * 1024,
        memory_utilization=0.6,
        fits_on_device=True,
        device_memory_bytes=16 * 1024**3,
    )

    string_repr = str(report)
    assert "peak=100.0MB" in string_repr
    assert "avg=60.0MB" in string_repr
    assert "util=60%" in string_repr


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
