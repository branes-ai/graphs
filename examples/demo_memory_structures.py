#!/usr/bin/env python
"""
Demo: Memory Analysis Data Structures

This example demonstrates the memory analysis data structures without
requiring a full MemoryEstimator implementation. Shows how to create
and use MemoryTimelineEntry, MemoryDescriptor, and MemoryReport.

Run: python examples/demo_memory_structures.py
"""

from graphs.analysis.memory import (
    MemoryTimelineEntry,
    MemoryDescriptor,
    MemoryReport,
)
from graphs.ir.structures import OperationType


def demo_timeline_entry():
    """Demo: Creating and formatting timeline entries"""
    print("=" * 80)
    print("DEMO 1: Memory Timeline Entries")
    print("=" * 80)
    print()

    # Simulate a few execution steps
    entries = [
        MemoryTimelineEntry(
            step=0,
            subgraph_id="input",
            subgraph_name="input",
            total_memory_bytes=5 * 1024 * 1024,  # 5 MB
            activation_memory_bytes=5 * 1024 * 1024,
            workspace_memory_bytes=0,
            live_tensors=["input"],
            num_live_tensors=1,
            allocated_tensors=["input"],
            freed_tensors=[],
        ),
        MemoryTimelineEntry(
            step=1,
            subgraph_id="conv1",
            subgraph_name="conv1",
            total_memory_bytes=25 * 1024 * 1024,  # 25 MB (input + conv1_out + weights)
            activation_memory_bytes=20 * 1024 * 1024,
            workspace_memory_bytes=5 * 1024 * 1024,  # im2col buffer
            live_tensors=["input", "conv1_output", "conv1_weights"],
            num_live_tensors=3,
            allocated_tensors=["conv1_output", "conv1_weights"],
            freed_tensors=[],
        ),
        MemoryTimelineEntry(
            step=2,
            subgraph_id="relu",
            subgraph_name="relu",
            total_memory_bytes=20 * 1024 * 1024,  # workspace freed
            activation_memory_bytes=20 * 1024 * 1024,
            workspace_memory_bytes=0,
            live_tensors=["conv1_output", "conv1_weights"],  # input freed
            num_live_tensors=2,
            allocated_tensors=[],
            freed_tensors=["input", "workspace"],
        ),
    ]

    print("Execution Timeline:")
    print("-" * 80)
    for entry in entries:
        print(entry.format_summary())
    print()


def demo_memory_descriptor():
    """Demo: Per-subgraph memory descriptors"""
    print("=" * 80)
    print("DEMO 2: Subgraph Memory Descriptors")
    print("=" * 80)
    print()

    # Example 1: Typical Conv2d layer
    conv_desc = MemoryDescriptor(
        subgraph_id="layer1_conv1",
        subgraph_name="layer1_conv1",
        operation_type=OperationType.CONV2D,
        input_memory_bytes=12 * 1024 * 1024,   # 12 MB input
        output_memory_bytes=24 * 1024 * 1024,  # 24 MB output
        weight_memory_bytes=2 * 1024 * 1024,   # 2 MB weights
        workspace_memory_bytes=24 * 1024 * 1024,  # 24 MB im2col
        can_checkpoint=True,
        checkpoint_savings_bytes=24 * 1024 * 1024,
    )

    print("Conv2d Layer:")
    print(conv_desc.format_summary())
    print(f"  Total memory: {conv_desc.total_memory_mb:.1f} MB")
    print(f"  Peak memory:  {conv_desc.peak_memory_mb:.1f} MB")
    print(f"  Can checkpoint: {conv_desc.can_checkpoint}")
    print()

    # Example 2: Large linear layer (good for quantization)
    linear_desc = MemoryDescriptor(
        subgraph_id="classifier",
        subgraph_name="classifier",
        operation_type=OperationType.MATMUL,
        input_memory_bytes=5 * 1024 * 1024,    # 5 MB input
        output_memory_bytes=4 * 1024 * 1024,   # 4 MB output
        weight_memory_bytes=80 * 1024 * 1024,  # 80 MB weights (large!)
        workspace_memory_bytes=0,
        can_quantize=True,
        quantization_savings_bytes=60 * 1024 * 1024,  # FP32→INT8 = 4× = 60MB
    )

    print("Linear Layer (Classifier):")
    print(linear_desc.format_summary())
    print(f"  Total memory: {linear_desc.total_memory_mb:.1f} MB")
    print(f"  Can quantize: {linear_desc.can_quantize}")
    print(f"  Quantization savings: {linear_desc.quantization_savings_bytes/1024**2:.0f} MB")
    print()


def demo_memory_report():
    """Demo: Complete memory report"""
    print("=" * 80)
    print("DEMO 3: Complete Memory Report")
    print("=" * 80)
    print()

    # Create timeline
    timeline = []
    for i in range(10):
        # Simulate increasing memory then decreasing
        if i < 5:
            memory = (i + 1) * 20 * 1024 * 1024  # Increasing to 100 MB
        else:
            memory = (10 - i) * 20 * 1024 * 1024  # Decreasing

        timeline.append(MemoryTimelineEntry(
            step=i,
            subgraph_id=f"layer{i}",
            subgraph_name=f"layer{i}_conv",
            total_memory_bytes=memory,
            activation_memory_bytes=int(memory * 0.8),
            workspace_memory_bytes=int(memory * 0.2),
            num_live_tensors=i + 2,
        ))

    # Create report
    report = MemoryReport(
        peak_memory_bytes=100 * 1024 * 1024,  # 100 MB peak
        peak_memory_mb=100.0,
        peak_memory_gb=100.0 / 1024,
        activation_memory_bytes=80 * 1024 * 1024,
        weight_memory_bytes=15 * 1024 * 1024,
        workspace_memory_bytes=5 * 1024 * 1024,
        average_memory_bytes=60 * 1024 * 1024,
        memory_utilization=0.6,
        fits_in_l2_cache=False,
        fits_in_shared_memory=False,
        fits_on_device=True,
        l2_cache_size_bytes=50 * 1024 * 1024,  # 50 MB L2
        device_memory_bytes=16 * 1024**3,  # 16 GB
        memory_timeline=timeline,
        peak_at_step=4,
        peak_at_subgraph="layer4",
        peak_at_subgraph_name="layer4_conv",
        total_checkpoint_savings_bytes=48 * 1024 * 1024,
        total_quantization_savings_bytes=11 * 1024 * 1024,
        optimization_suggestions=[
            "✓ Activation checkpointing: Save ~48 MB by recomputing 60% of activations",
            "✓ INT8 quantization: Save ~11 MB in weights (4× compression)",
            "✓ In-place ops: 15 ReLU ops could save ~6 MB",
        ],
    )

    # Print short summary
    print("Short Summary:")
    print(report)
    print()

    # Print full report (without timeline)
    print(report.format_report(show_timeline=False))

    # Print report with timeline
    print("\nWith Timeline (first 5 steps):")
    print("=" * 80)
    print(report.format_report(show_timeline=True, timeline_steps=5))


def main():
    """Run all demos"""
    demo_timeline_entry()
    print("\n")

    demo_memory_descriptor()
    print("\n")

    demo_memory_report()

    print("\n" + "=" * 80)
    print("Data structures ready! Next: Implement MemoryEstimator algorithm")
    print("=" * 80)


if __name__ == "__main__":
    main()
