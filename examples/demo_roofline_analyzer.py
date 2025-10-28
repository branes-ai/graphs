#!/usr/bin/env python
"""
Demo: Roofline Analyzer End-to-End

This example demonstrates the complete roofline analysis workflow:
1. Load and trace models (ResNet-18, MobileNet-V2, ResNet-50)
2. Partition the graphs
3. Analyze with roofline model
4. Identify compute-bound vs memory-bound operations
5. Compare latency across models and hardware

The Roofline Model:
- Determines performance bottlenecks (compute vs memory bandwidth)
- Calculates realistic latency = max(compute_time, memory_time) + overhead
- Identifies optimization opportunities based on arithmetic intensity

Run: python examples/demo_roofline_analyzer.py
"""

import torch
import torchvision.models as models
from torch.fx import symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp

from graphs.transform.partitioning import GraphPartitioner
from graphs.analysis.roofline import RooflineAnalyzer
from graphs.hardware.resource_model import HardwareResourceModel, HardwareType, Precision, PrecisionProfile


def create_gpu_hardware():
    """Create a realistic GPU hardware model (A100-like)"""
    return HardwareResourceModel(
        name="GPU-A100",
        hardware_type=HardwareType.GPU,
        compute_units=108,  # 108 SMs
        threads_per_unit=2048,
        warps_per_unit=64,
        warp_size=32,
        precision_profiles={
            Precision.FP32: PrecisionProfile(
                precision=Precision.FP32,
                peak_ops_per_sec=19.5e12,  # 19.5 TFLOPS FP32
                tensor_core_supported=False,
                relative_speedup=1.0,
                bytes_per_element=4,
                accumulator_precision=Precision.FP32,
            ),
            Precision.FP16: PrecisionProfile(
                precision=Precision.FP16,
                peak_ops_per_sec=312e12,  # 312 TFLOPS FP16 with tensor cores
                tensor_core_supported=True,
                relative_speedup=16.0,
                bytes_per_element=2,
                accumulator_precision=Precision.FP32,
            ),
        },
        default_precision=Precision.FP32,
        peak_bandwidth=1555e9,  # 1555 GB/s (HBM2e)
        l1_cache_per_unit=192 * 1024,  # 192 KB per SM
        l2_cache_total=40 * 1024 * 1024,  # 40 MB L2
        main_memory=80 * 1024**3,  # 80 GB HBM2e
        energy_per_flop_fp32=20e-12,  # 20 pJ/FLOP
        energy_per_byte=10e-12,  # 10 pJ/byte
    )


def create_cpu_hardware():
    """Create a realistic CPU hardware model (Intel Xeon)"""
    return HardwareResourceModel(
        name="CPU-Xeon",
        hardware_type=HardwareType.CPU,
        compute_units=64,  # 64 cores
        threads_per_unit=2,  # 2 threads per core (hyperthreading)
        warps_per_unit=1,
        warp_size=1,
        precision_profiles={
            Precision.FP32: PrecisionProfile(
                precision=Precision.FP32,
                peak_ops_per_sec=4.0e12,  # 4 TFLOPS (64 cores × 2 × 16 FP32/cycle × 2 GHz)
                tensor_core_supported=False,
                relative_speedup=1.0,
                bytes_per_element=4,
                accumulator_precision=Precision.FP32,
            ),
        },
        default_precision=Precision.FP32,
        peak_bandwidth=200e9,  # 200 GB/s (8-channel DDR4)
        l1_cache_per_unit=80 * 1024,  # 80 KB per core
        l2_cache_total=64 * 1024 * 1024,  # 64 MB L2
        main_memory=256 * 1024**3,  # 256 GB DDR4
        energy_per_flop_fp32=50e-12,  # 50 pJ/FLOP (higher than GPU)
        energy_per_byte=30e-12,  # 30 pJ/byte
    )


def analyze_model(model_name, model, input_size, partitioner, analyzer):
    """Analyze roofline for a single model"""

    print("=" * 80)
    print(f"ROOFLINE ANALYSIS: {model_name}")
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

    # Roofline analysis
    print(f"[3/3] Roofline analysis...")
    roofline_report = analyzer.analyze(partition_report.subgraphs, partition_report)

    return roofline_report


def visualize_roofline_plot(roofline_report, max_points=15):
    """ASCII-art roofline visualization"""

    print("\n" + "=" * 80)
    print("ROOFLINE PLOT (ASCII)")
    print("=" * 80)

    # Sort points by AI
    points = sorted(roofline_report.roofline_points, key=lambda p: p.arithmetic_intensity)

    # Take representative sample
    if len(points) > max_points:
        step = len(points) // max_points
        points = points[::step]

    print(f"\nAI Breakpoint: {roofline_report.arithmetic_intensity_breakpoint:.2f} FLOPs/byte")
    print(f"(Operations left of breakpoint are memory-bound)")
    print("")

    # Create ASCII plot
    print(f"{'AI (FLOPs/byte)':<20} {'Attained (GFLOPS)':<20} {'Bottleneck':<20} {'Op'}")
    print("-" * 100)

    for point in points:
        ai = point.arithmetic_intensity
        gflops = point.attained_flops / 1e9

        # Determine bottleneck
        if ai < roofline_report.arithmetic_intensity_breakpoint * 0.9:
            bottleneck = "Memory-bound"
            marker = "◀"
        elif ai > roofline_report.arithmetic_intensity_breakpoint * 1.1:
            bottleneck = "Compute-bound"
            marker = "▶"
        else:
            bottleneck = "Balanced"
            marker = "●"

        # Create bar
        bar_length = int(min(30, ai * 2))
        bar = marker * bar_length

        print(f"{ai:>8.2f}  {bar:<30} {gflops:>10.2f} {bottleneck:<20} {point.subgraph_name[:20]}")


def compare_hardware(reports_by_hardware):
    """Compare roofline results across different hardware"""

    print("\n" + "=" * 80)
    print("HARDWARE COMPARISON")
    print("=" * 80)

    # Header
    print(f"\n{'Hardware':<15} {'Model':<15} {'Total (ms)':>12} {'Compute':>10} {'Memory':>10} {'Overhead':>10}")
    print("-" * 90)

    for hw_name, model_reports in reports_by_hardware.items():
        for model_name, report in model_reports.items():
            if report is None:
                continue

            print(f"{hw_name:<15} {model_name:<15} "
                  f"{report.total_latency * 1e3:>12.3f} "
                  f"{report.total_compute_time / report.total_latency * 100:>9.1f}% "
                  f"{report.total_memory_time / report.total_latency * 100:>9.1f}% "
                  f"{report.total_overhead / report.total_latency * 100:>9.1f}%")


def compare_bottlenecks(reports):
    """Compare bottleneck distribution across models"""

    print("\n" + "=" * 80)
    print("BOTTLENECK DISTRIBUTION")
    print("=" * 80)

    print(f"\n{'Model':<20} {'Compute-bound':>15} {'Memory-bound':>15} {'Balanced':>15}")
    print("-" * 80)

    for model_name, report in reports.items():
        if report is None:
            continue

        total = report.num_compute_bound + report.num_memory_bound + report.num_balanced
        if total == 0:
            continue

        print(f"{model_name:<20} "
              f"{report.num_compute_bound:>6} ({report.num_compute_bound / total * 100:>5.1f}%) "
              f"{report.num_memory_bound:>6} ({report.num_memory_bound / total * 100:>5.1f}%) "
              f"{report.num_balanced:>6} ({report.num_balanced / total * 100:>5.1f}%)")


def main():
    """Run roofline analysis demos"""

    print("\n" + "=" * 80)
    print("ROOFLINE ANALYZER END-TO-END DEMO")
    print("=" * 80)
    print("\nThis demo analyzes latency using the roofline model.")
    print("The roofline model determines whether operations are limited by:")
    print("  - Compute (FLOPs): High arithmetic intensity operations")
    print("  - Memory bandwidth: Low arithmetic intensity operations")
    print("\nKey Insight: Latency = max(compute_time, memory_time) + overhead")

    # Create hardware models
    gpu = create_gpu_hardware()
    cpu = create_cpu_hardware()

    print(f"\nHardware Configurations:")
    print(f"  GPU-A100:   {gpu.precision_profiles[Precision.FP32].peak_ops_per_sec / 1e12:.1f} TFLOPS, "
          f"{gpu.peak_bandwidth / 1e9:.0f} GB/s (AI breakpoint: {gpu.precision_profiles[Precision.FP32].peak_ops_per_sec / gpu.peak_bandwidth:.1f})")
    print(f"  CPU-Xeon:   {cpu.precision_profiles[Precision.FP32].peak_ops_per_sec / 1e12:.1f} TFLOPS, "
          f"{cpu.peak_bandwidth / 1e9:.0f} GB/s (AI breakpoint: {cpu.precision_profiles[Precision.FP32].peak_ops_per_sec / cpu.peak_bandwidth:.1f})")

    # Create partitioner and analyzers
    partitioner = GraphPartitioner()
    gpu_analyzer = RooflineAnalyzer(gpu)
    cpu_analyzer = RooflineAnalyzer(cpu)

    reports_gpu = {}
    reports_cpu = {}

    # --- Demo 1: ResNet-18 on GPU ---
    print("\n\n")
    model = models.resnet18(weights=None)
    model.eval()
    reports_gpu['ResNet-18'] = analyze_model(
        'ResNet-18',
        model,
        (3, 224, 224),
        partitioner,
        gpu_analyzer
    )

    if reports_gpu['ResNet-18']:
        print("\n" + reports_gpu['ResNet-18'].format_report(show_per_subgraph=True, max_subgraphs=5))
        visualize_roofline_plot(reports_gpu['ResNet-18'], max_points=10)

    # --- Demo 2: MobileNet-V2 on GPU ---
    print("\n\n")
    model = models.mobilenet_v2(weights=None)
    model.eval()
    reports_gpu['MobileNet-V2'] = analyze_model(
        'MobileNet-V2',
        model,
        (3, 224, 224),
        partitioner,
        gpu_analyzer
    )

    if reports_gpu['MobileNet-V2']:
        print("\n" + reports_gpu['MobileNet-V2'].format_report(show_per_subgraph=False))

    # --- Demo 3: ResNet-18 on CPU ---
    print("\n\n")
    model = models.resnet18(weights=None)
    model.eval()
    reports_cpu['ResNet-18'] = analyze_model(
        'ResNet-18',
        model,
        (3, 224, 224),
        partitioner,
        cpu_analyzer
    )

    if reports_cpu['ResNet-18']:
        print("\n" + reports_cpu['ResNet-18'].format_report(show_per_subgraph=False))

    # --- Hardware Comparison ---
    compare_hardware({
        'GPU-A100': reports_gpu,
        'CPU-Xeon': reports_cpu,
    })

    # --- Bottleneck Comparison ---
    compare_bottlenecks(reports_gpu)

    # --- Summary ---
    print("\n" + "=" * 80)
    print("SUMMARY & KEY INSIGHTS")
    print("=" * 80)
    print("\n1. Roofline Model Basics:")
    print("   - Arithmetic Intensity (AI) = FLOPs / bytes")
    print("   - AI Breakpoint = Peak FLOPs / Peak Bandwidth")
    print("   - If AI < breakpoint: Memory-bound (limited by bandwidth)")
    print("   - If AI > breakpoint: Compute-bound (limited by FLOPs)")

    print("\n2. Hardware Characteristics:")
    print("   - GPUs: High FLOPs, high bandwidth → higher AI breakpoint")
    print("   - CPUs: Lower FLOPs, lower bandwidth → lower AI breakpoint")
    print("   - Tensor cores: Dramatically increase peak FLOPs (shifts breakpoint)")

    print("\n3. Model Characteristics:")
    print("   - Conv layers: High AI (kernel reuse) → often compute-bound")
    print("   - Element-wise ops (ReLU, Add): Low AI → always memory-bound")
    print("   - Depthwise convs (MobileNet): Lower AI than standard convs")

    print("\n4. Optimization Strategies:")
    print("   - Memory-bound ops: Reduce data movement, improve cache reuse")
    print("   - Compute-bound ops: Use tensor cores, quantization, pruning")
    print("   - GPU kernels: Minimize kernel launch overhead")

    print("\n" + "=" * 80)
    print("Roofline Analyzer is ready for use!")
    print("=" * 80)


if __name__ == "__main__":
    main()
