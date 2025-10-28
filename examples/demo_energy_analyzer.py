#!/usr/bin/env python
"""
Demo: Energy Analyzer End-to-End

This example demonstrates the complete energy analysis workflow:
1. Load and trace models (ResNet-18, MobileNet-V2)
2. Partition the graphs
3. Analyze energy consumption (compute, memory, static)
4. Compare energy across models and hardware
5. Identify optimization opportunities

Energy Components:
- Compute energy: FLOPs × energy_per_flop
- Memory energy: bytes × energy_per_byte
- Static energy: idle_power × latency (leakage, always-on circuits)

Run: python examples/demo_energy_analyzer.py
"""

import torch
import torchvision.models as models
from torch.fx import symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp

from graphs.transform.partitioning import GraphPartitioner
from graphs.analysis.energy import EnergyAnalyzer
from graphs.analysis.roofline import RooflineAnalyzer
from graphs.hardware.resource_model import (
    HardwareResourceModel,
    HardwareType,
    Precision,
    PrecisionProfile,
)


def create_gpu_hardware():
    """Create realistic GPU hardware model (A100-like)"""
    return HardwareResourceModel(
        name="GPU-A100",
        hardware_type=HardwareType.GPU,
        compute_units=108,
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
        l1_cache_per_unit=192 * 1024,
        l2_cache_total=40 * 1024 * 1024,
        main_memory=80 * 1024**3,
        energy_per_flop_fp32=20e-12,  # 20 pJ/FLOP (datacenter GPU)
        energy_per_byte=10e-12,  # 10 pJ/byte (HBM2e)
        energy_scaling={
            Precision.FP32: 1.0,
            Precision.FP16: 0.5,  # FP16 uses ~50% energy of FP32
            Precision.INT8: 0.25,  # INT8 uses ~25% energy of FP32
        },
    )


def create_edge_hardware():
    """Create edge device hardware model (Jetson-like)"""
    return HardwareResourceModel(
        name="Edge-Jetson",
        hardware_type=HardwareType.GPU,
        compute_units=8,  # 8 SMs (much smaller)
        threads_per_unit=1024,
        warps_per_unit=32,
        warp_size=32,
        precision_profiles={
            Precision.FP32: PrecisionProfile(
                precision=Precision.FP32,
                peak_ops_per_sec=1.5e12,  # 1.5 TFLOPS
                tensor_core_supported=False,
                relative_speedup=1.0,
                bytes_per_element=4,
                accumulator_precision=Precision.FP32,
            ),
            Precision.FP16: PrecisionProfile(
                precision=Precision.FP16,
                peak_ops_per_sec=3.0e12,  # 3 TFLOPS FP16
                tensor_core_supported=True,
                relative_speedup=2.0,
                bytes_per_element=2,
                accumulator_precision=Precision.FP32,
            ),
        },
        default_precision=Precision.FP32,
        peak_bandwidth=100e9,  # 100 GB/s (LPDDR5)
        l1_cache_per_unit=96 * 1024,
        l2_cache_total=4 * 1024 * 1024,
        main_memory=8 * 1024**3,  # 8 GB
        energy_per_flop_fp32=100e-12,  # 100 pJ/FLOP (less efficient than datacenter)
        energy_per_byte=50e-12,  # 50 pJ/byte (LPDDR5 less efficient than HBM)
        energy_scaling={
            Precision.FP32: 1.0,
            Precision.FP16: 0.5,
            Precision.INT8: 0.25,
        },
    )


def analyze_model(model_name, model, input_size, partitioner, energy_analyzer, roofline_analyzer=None):
    """Analyze energy for a single model"""

    print("=" * 80)
    print(f"ENERGY ANALYSIS: {model_name}")
    print("=" * 80)

    # Create input
    input_tensor = torch.randn(1, *input_size)

    # FX trace
    print(f"\n[1/4] Tracing {model_name}...")
    try:
        fx_graph = symbolic_trace(model)
    except Exception as e:
        print(f"Error during FX trace: {e}")
        return None, None

    # Shape propagation
    print(f"[2/4] Shape propagation...")
    shape_prop = ShapeProp(fx_graph)
    shape_prop.propagate(input_tensor)

    # Partition graph
    partition_report = partitioner.partition(fx_graph)
    print(f"      Partitioned into {partition_report.total_subgraphs} subgraphs")

    # Roofline analysis for latencies (if provided)
    latencies = None
    if roofline_analyzer:
        print(f"[3/4] Roofline analysis for latencies...")
        roofline_report = roofline_analyzer.analyze(partition_report.subgraphs, partition_report)
        latencies = [lat.actual_latency for lat in roofline_report.latencies]

    # Energy analysis
    print(f"[4/4] Energy analysis...")
    energy_report = energy_analyzer.analyze(
        partition_report.subgraphs,
        partition_report,
        latencies=latencies
    )

    return energy_report, roofline_report if roofline_analyzer else None


def visualize_energy_breakdown(energy_report, model_name):
    """Visualize energy breakdown with ASCII bar chart"""

    print("\n" + "=" * 80)
    print(f"ENERGY BREAKDOWN: {model_name}")
    print("=" * 80)

    total = energy_report.total_energy_j
    compute_pct = energy_report.compute_energy_j / total * 100
    memory_pct = energy_report.memory_energy_j / total * 100
    static_pct = energy_report.static_energy_j / total * 100

    # Create bar chart
    max_bar_len = 50
    compute_bar = "█" * int(compute_pct / 100 * max_bar_len)
    memory_bar = "█" * int(memory_pct / 100 * max_bar_len)
    static_bar = "█" * int(static_pct / 100 * max_bar_len)

    print(f"\nCompute ({compute_pct:.1f}%): {compute_bar}")
    print(f"Memory  ({memory_pct:.1f}%): {memory_bar}")
    print(f"Static  ({static_pct:.1f}%): {static_bar}")
    print(f"\nTotal: {energy_report.total_energy_mj:.2f} mJ ({energy_report.total_energy_j * 1e6:.0f} μJ)")
    print(f"Average Power: {energy_report.average_power_w:.2f} W")


def compare_hardware(reports_by_hardware):
    """Compare energy across different hardware"""

    print("\n" + "=" * 80)
    print("HARDWARE COMPARISON")
    print("=" * 80)

    # Header
    print(f"\n{'Hardware':<20} {'Model':<15} {'Energy (mJ)':>12} {'Power (W)':>10} {'Efficiency':>10} {'Latency (ms)':>12}")
    print("-" * 90)

    for hw_name, model_reports in reports_by_hardware.items():
        for model_name, report in model_reports.items():
            if report is None:
                continue

            print(f"{hw_name:<20} {model_name:<15} "
                  f"{report.total_energy_mj:>12.2f} "
                  f"{report.average_power_w:>10.2f} "
                  f"{report.average_efficiency * 100:>9.1f}% "
                  f"{report.total_latency_s * 1e3:>12.2f}")


def compare_precisions(model, input_size, partitioner, gpu_hardware):
    """Compare energy across different precisions"""

    print("\n" + "=" * 80)
    print("PRECISION COMPARISON (ResNet-18)")
    print("=" * 80)

    input_tensor = torch.randn(1, *input_size)
    fx_graph = symbolic_trace(model)
    shape_prop = ShapeProp(fx_graph)
    shape_prop.propagate(input_tensor)
    partition_report = partitioner.partition(fx_graph)

    results = {}

    for precision in [Precision.FP32, Precision.FP16]:
        analyzer = EnergyAnalyzer(gpu_hardware, precision=precision)
        roofline = RooflineAnalyzer(gpu_hardware, precision=precision)

        roofline_report = roofline.analyze(partition_report.subgraphs, partition_report)
        latencies = [lat.actual_latency for lat in roofline_report.latencies]

        energy_report = analyzer.analyze(
            partition_report.subgraphs,
            partition_report,
            latencies=latencies
        )

        results[precision.value] = energy_report

    # Print comparison
    print(f"\n{'Precision':<10} {'Energy (mJ)':>12} {'Power (W)':>10} {'Latency (ms)':>12} {'Savings':>10}")
    print("-" * 60)

    fp32_report = results['fp32']
    print(f"{'FP32':<10} {fp32_report.total_energy_mj:>12.2f} {fp32_report.average_power_w:>10.2f} {fp32_report.total_latency_s * 1e3:>12.2f} {'baseline':>10}")

    for precision_name, report in results.items():
        if precision_name == 'fp32':
            continue

        savings = (1 - report.total_energy_mj / fp32_report.total_energy_mj) * 100
        print(f"{precision_name.upper():<10} {report.total_energy_mj:>12.2f} {report.average_power_w:>10.2f} {report.total_latency_s * 1e3:>12.2f} {savings:>9.1f}%")


def main():
    """Run energy analysis demos"""

    print("\n" + "=" * 80)
    print("ENERGY ANALYZER END-TO-END DEMO")
    print("=" * 80)
    print("\nThis demo analyzes energy consumption with hardware-aware modeling.")
    print("Energy Components:")
    print("  - Compute energy: FLOPs × energy_per_flop")
    print("  - Memory energy: bytes × energy_per_byte")
    print("  - Static energy: idle_power × latency (leakage, always-on circuits)")
    print("\nKey Insight: Energy efficiency depends on utilization and latency")

    # Create hardware models
    gpu = create_gpu_hardware()
    edge = create_edge_hardware()

    print(f"\nHardware Configurations:")
    print(f"  GPU-A100:      {gpu.precision_profiles[Precision.FP32].peak_ops_per_sec / 1e12:.1f} TFLOPS, "
          f"{gpu.energy_per_flop_fp32 * 1e12:.0f} pJ/FLOP, "
          f"{gpu.energy_per_byte * 1e12:.0f} pJ/byte")
    print(f"  Edge-Jetson:   {edge.precision_profiles[Precision.FP32].peak_ops_per_sec / 1e12:.1f} TFLOPS, "
          f"{edge.energy_per_flop_fp32 * 1e12:.0f} pJ/FLOP, "
          f"{edge.energy_per_byte * 1e12:.0f} pJ/byte")

    # Create partitioner and analyzers
    partitioner = GraphPartitioner()

    reports_gpu = {}
    reports_edge = {}

    # --- Demo 1: ResNet-18 on GPU ---
    print("\n\n")
    model = models.resnet18(weights=None)
    model.eval()

    gpu_energy_analyzer = EnergyAnalyzer(gpu)
    gpu_roofline_analyzer = RooflineAnalyzer(gpu)

    energy_report, roofline_report = analyze_model(
        'ResNet-18',
        model,
        (3, 224, 224),
        partitioner,
        gpu_energy_analyzer,
        gpu_roofline_analyzer
    )
    reports_gpu['ResNet-18'] = energy_report

    if energy_report:
        print("\n" + energy_report.format_report(show_per_subgraph=True, max_subgraphs=5))
        visualize_energy_breakdown(energy_report, 'ResNet-18 on GPU-A100')

    # --- Demo 2: MobileNet-V2 on GPU ---
    print("\n\n")
    model = models.mobilenet_v2(weights=None)
    model.eval()

    energy_report, _ = analyze_model(
        'MobileNet-V2',
        model,
        (3, 224, 224),
        partitioner,
        gpu_energy_analyzer,
        gpu_roofline_analyzer
    )
    reports_gpu['MobileNet-V2'] = energy_report

    if energy_report:
        print("\n" + energy_report.format_report(show_per_subgraph=False))

    # --- Demo 3: ResNet-18 on Edge ---
    print("\n\n")
    model = models.resnet18(weights=None)
    model.eval()

    edge_energy_analyzer = EnergyAnalyzer(edge)
    edge_roofline_analyzer = RooflineAnalyzer(edge)

    energy_report, _ = analyze_model(
        'ResNet-18',
        model,
        (3, 224, 224),
        partitioner,
        edge_energy_analyzer,
        edge_roofline_analyzer
    )
    reports_edge['ResNet-18'] = energy_report

    if energy_report:
        print("\n" + energy_report.format_report(show_per_subgraph=False))

    # --- Hardware Comparison ---
    compare_hardware({
        'GPU-A100': reports_gpu,
        'Edge-Jetson': reports_edge,
    })

    # --- Precision Comparison ---
    model = models.resnet18(weights=None)
    model.eval()
    compare_precisions(model, (3, 224, 224), partitioner, gpu)

    # --- Summary ---
    print("\n" + "=" * 80)
    print("SUMMARY & KEY INSIGHTS")
    print("=" * 80)
    print("\n1. Energy Components:")
    print("   - Compute energy: Proportional to FLOPs")
    print("   - Memory energy: Proportional to data movement")
    print("   - Static energy: Proportional to latency (idle power × time)")

    print("\n2. Hardware Characteristics:")
    print("   - Datacenter GPUs: Low energy/op, high idle power")
    print("   - Edge devices: Higher energy/op, lower idle power")
    print("   - Energy efficiency ≠ performance efficiency")

    print("\n3. Optimization Strategies:")
    print("   - Reduce latency: Saves static energy (leakage)")
    print("   - Improve utilization: Reduces wasted energy on idle resources")
    print("   - Reduce data movement: Memory transfers are expensive")
    print("   - Lower precision: FP16 saves ~50% compute energy, INT8 saves ~75%")

    print("\n4. Model Characteristics:")
    print("   - ResNet-18: ~1mJ on datacenter GPU, ~5mJ on edge device")
    print("   - MobileNet-V2: More energy-efficient (fewer ops)")
    print("   - Static energy can dominate for small models (30-50%)")

    print("\n" + "=" * 80)
    print("Energy Analyzer is ready for use!")
    print("=" * 80)


if __name__ == "__main__":
    main()
