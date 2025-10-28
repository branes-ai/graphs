#!/usr/bin/env python
"""
Demo: Memory Estimator End-to-End

This example demonstrates the complete memory analysis workflow:
1. Load and trace a model (ResNet-18, MobileNet-V2)
2. Partition the graph
3. Estimate memory usage
4. Analyze timeline and optimization opportunities
5. Compare memory footprints across models

Run: python examples/demo_memory_estimator.py
"""

import torch
import torchvision.models as models
from torch.fx import symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp

from graphs.transform.partitioning import GraphPartitioner
from graphs.analysis.memory import MemoryEstimator
from graphs.hardware.resource_model import HardwareResourceModel, HardwareType, Precision, PrecisionProfile


def create_test_hardware():
    """Create a test hardware model (GPU-like)"""
    return HardwareResourceModel(
        name="TestGPU",
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
        peak_bandwidth=900e9,  # 900 GB/s
        l1_cache_per_unit=128 * 1024,
        l2_cache_total=40 * 1024 * 1024,  # 40 MB L2
        main_memory=16 * 1024**3,  # 16 GB
        energy_per_flop_fp32=30e-12,
        energy_per_byte=20e-12,
    )


def analyze_model(model_name, model, input_size, partitioner, estimator):
    """Analyze memory for a single model"""

    print("=" * 80)
    print(f"MEMORY ANALYSIS: {model_name}")
    print("=" * 80)

    # Create input
    input_tensor = torch.randn(1, *input_size)

    # FX trace
    print(f"\n[1/3] Tracing {model_name}...")
    try:
        fx_graph = symbolic_trace(model)
    except Exception as e:
        print(f"Error during FX trace: {e}")
        return None

    # Shape propagation
    print(f"[2/3] Shape propagation...")
    shape_prop = ShapeProp(fx_graph)
    shape_prop.propagate(input_tensor)

    # Partition graph
    partition_report = partitioner.partition(fx_graph)
    print(f"      Partitioned into {partition_report.total_subgraphs} subgraphs")

    # Memory estimation
    print(f"[3/3] Estimating memory...")
    memory_report = estimator.estimate_memory(
        partition_report.subgraphs,
        partition_report
    )

    return memory_report


def visualize_timeline(memory_report, max_steps=10):
    """Visualize memory timeline (text-based)"""

    print("\n" + "=" * 80)
    print("MEMORY TIMELINE (first {} steps)".format(max_steps))
    print("=" * 80)

    timeline = memory_report.memory_timeline[:max_steps]

    # Find max memory for scaling
    max_mem = max(entry.total_memory_bytes for entry in timeline)

    for entry in timeline:
        # Create bar chart
        bar_length = int(50 * entry.total_memory_bytes / max_mem)
        bar = "█" * bar_length

        # Format memory values
        total_mb = entry.total_memory_bytes / (1024 ** 2)
        act_mb = entry.activation_memory_bytes / (1024 ** 2)
        ws_mb = entry.workspace_memory_bytes / (1024 ** 2)

        print(f"\nStep {entry.step:3d} | {entry.subgraph_name:20s}")
        print(f"  {bar} {total_mb:.1f} MB")
        print(f"  └─ Activations: {act_mb:.1f} MB, Workspace: {ws_mb:.1f} MB")
        print(f"  └─ Live tensors: {entry.num_live_tensors}")

        if entry.freed_tensors:
            print(f"  └─ Freed: {len(entry.freed_tensors)} tensor(s)")


def compare_models(reports):
    """Compare memory footprints across models"""

    print("\n" + "=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)

    # Header
    print(f"\n{'Model':<20} {'Peak (MB)':>12} {'Avg (MB)':>12} {'Util':>8} {'Timeline':>10}")
    print("-" * 80)

    for model_name, report in reports.items():
        if report is None:
            print(f"{model_name:<20} {'FAILED':>12}")
            continue

        print(f"{model_name:<20} "
              f"{report.peak_memory_mb:>12.1f} "
              f"{report.average_memory_bytes / (1024**2):>12.1f} "
              f"{report.memory_utilization * 100:>7.0f}% "
              f"{len(report.memory_timeline):>10}")

    # Detailed comparison
    print("\n" + "=" * 80)
    print("OPTIMIZATION OPPORTUNITIES")
    print("=" * 80)

    for model_name, report in reports.items():
        if report is None:
            continue

        print(f"\n{model_name}:")

        if report.optimization_suggestions:
            for suggestion in report.optimization_suggestions:
                print(f"  {suggestion}")
        else:
            print("  No optimizations suggested")

        # Savings summary
        if report.total_checkpoint_savings_bytes > 0 or report.total_quantization_savings_bytes > 0:
            print(f"\n  Potential savings:")
            if report.total_checkpoint_savings_bytes > 0:
                savings_pct = 100 * report.total_checkpoint_savings_bytes / report.peak_memory_bytes
                print(f"    Checkpointing: {report.total_checkpoint_savings_bytes / (1024**2):.1f} MB "
                      f"({savings_pct:.0f}% reduction)")
            if report.total_quantization_savings_bytes > 0:
                savings_pct = 100 * report.total_quantization_savings_bytes / report.peak_memory_bytes
                print(f"    Quantization:  {report.total_quantization_savings_bytes / (1024**2):.1f} MB "
                      f"({savings_pct:.0f}% reduction)")


def main():
    """Run memory analysis demos"""

    print("\n" + "=" * 80)
    print("MEMORY ESTIMATOR END-TO-END DEMO")
    print("=" * 80)
    print("\nThis demo analyzes memory usage for popular CNN models.")
    print("Memory estimation includes:")
    print("  - Activation memory (feature maps)")
    print("  - Weight memory (model parameters)")
    print("  - Workspace memory (im2col buffers, etc.)")
    print("  - Timeline tracking (allocation/deallocation)")
    print("  - Optimization opportunities (checkpointing, quantization)")

    # Create partitioner and estimator
    partitioner = GraphPartitioner()
    hardware = create_test_hardware()
    estimator = MemoryEstimator(hardware)

    reports = {}

    # --- Demo 1: ResNet-18 ---
    print("\n\n")
    model = models.resnet18(weights=None)
    model.eval()
    reports['ResNet-18'] = analyze_model(
        'ResNet-18',
        model,
        (3, 224, 224),
        partitioner,
        estimator
    )

    if reports['ResNet-18']:
        print("\n" + reports['ResNet-18'].format_report(show_timeline=False))
        visualize_timeline(reports['ResNet-18'], max_steps=8)

    # --- Demo 2: MobileNet-V2 ---
    print("\n\n")
    model = models.mobilenet_v2(weights=None)
    model.eval()
    reports['MobileNet-V2'] = analyze_model(
        'MobileNet-V2',
        model,
        (3, 224, 224),
        partitioner,
        estimator
    )

    if reports['MobileNet-V2']:
        print("\n" + reports['MobileNet-V2'].format_report(show_timeline=False))
        visualize_timeline(reports['MobileNet-V2'], max_steps=8)

    # --- Demo 3: ResNet-50 (larger model) ---
    print("\n\n")
    model = models.resnet50(weights=None)
    model.eval()
    reports['ResNet-50'] = analyze_model(
        'ResNet-50',
        model,
        (3, 224, 224),
        partitioner,
        estimator
    )

    if reports['ResNet-50']:
        print("\n" + reports['ResNet-50'].format_report(show_timeline=False))

    # --- Model Comparison ---
    compare_models(reports)

    # --- Summary ---
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("\nKey Insights:")
    print("  1. Peak memory ≠ sum of all tensors (tensors are freed over time)")
    print("  2. Memory utilization varies by model architecture")
    print("  3. ResNet models have higher memory due to residual connections")
    print("  4. MobileNet models are more memory-efficient (depthwise separable convs)")
    print("  5. Optimization opportunities depend on model characteristics:")
    print("     - Large activations → good for checkpointing")
    print("     - Large weights → good for quantization")
    print("     - Many element-wise ops → good for in-place operations")

    print("\n" + "=" * 80)
    print("Memory Estimator is ready for use!")
    print("=" * 80)


if __name__ == "__main__":
    main()
