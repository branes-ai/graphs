#!/usr/bin/env python
"""
Automotive Hardware Comparison

Comprehensive performance analysis for automotive ADAS workloads across
automotive-grade hardware platforms.

Target Latency: <100ms for real-time perception
"""

import sys
from pathlib import Path

# Add src to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "src"))

from graphs.estimation.unified_analyzer import UnifiedAnalyzer, AnalysisConfig
from graphs.hardware.resource_model import Precision
from graphs.reporting import ReportGenerator

def run_automotive_comparison():
    """Run comprehensive automotive hardware comparison"""

    # Configuration
    models = [
        'resnet50',           # Classification backbone
        'deeplabv3_resnet50', # Semantic segmentation
        'fcn_resnet50',       # Semantic segmentation (alternate)
        'vit_b_16',           # Vision transformer
    ]

    automotive_hardware = [
        # NVIDIA Automotive
        'Jetson-Orin-AGX',      # 275 TOPS INT8 (AGX)
        'Jetson-Thor',          # Next-gen automotive SoC

        # Qualcomm Automotive
        'Snapdragon-Ride',      # Automotive compute platform
        'SA8775P',              # Snapdragon Auto Platform

        # TI Automotive
        'TDA4VM',               # Jacinto 7 (10W)
        'TDA4VH',               # Jacinto 7 (20W, higher perf)
        'TDA4VL',               # Jacinto 7 (7W, lower power)
        'TDA4AL',               # Jacinto 7 (10W, alternate)

        # Stillwater KPU Automotive
        'KPU-T64',              # 64 tile KPU (competes with Hailo-8, Coral)
        'KPU-T256',             # 256 tile KPU
        'KPU-T768',             # 768 tile KPU (high-end)

        # Hailo Automotive
        'Hailo-10H',            # 34 TOPS INT8
        'Hailo-8',              # 26 TOPS INT8
    ]

    batch_sizes = [1]  # Real-time ADAS is always batch=1
    precisions = [Precision.FP32, Precision.FP16, Precision.INT8]

    print("="*80)
    print("AUTOMOTIVE HARDWARE COMPARISON")
    print("="*80)
    print(f"\nModels: {', '.join(models)}")
    print(f"Hardware: {len(automotive_hardware)} platforms (13 total)")
    print(f"Batch Size: {batch_sizes[0]} (real-time inference)")
    print(f"Precisions: {', '.join(p.name for p in precisions)}")
    print(f"\nTotal Analyses: {len(models)} × {len(automotive_hardware)} × {len(precisions)} = {len(models) * len(automotive_hardware) * len(precisions)}")
    print(f"Target Latency: <100ms for real-time perception")
    print("="*80)

    analyzer = UnifiedAnalyzer(verbose=False)
    generator = ReportGenerator()

    results = []
    total = len(models) * len(automotive_hardware) * len(precisions)
    count = 0

    for model_name in models:
        for hw_name in automotive_hardware:
            for precision in precisions:
                count += 1
                print(f"\n[{count}/{total}] Analyzing {model_name} on {hw_name} @ {precision.name}...")

                try:
                    result = analyzer.analyze_model(
                        model_name=model_name,
                        hardware_name=hw_name,
                        batch_size=1,
                        precision=precision,
                        config=AnalysisConfig(
                            run_roofline=True,
                            run_energy=True,
                            run_memory=True,
                            run_concurrency=False,
                            validate_consistency=True  # Enable validation for memory constraints
                        )
                    )

                    results.append(result)

                    # Check for memory constraint violations
                    has_memory_violation = any("MEMORY CONSTRAINT VIOLATION" in w for w in result.validation_warnings)

                    # Print key metrics
                    latency_ok = "✓" if result.total_latency_ms < 100 else "✗"
                    memory_status = "✗ FAIL (memory)" if has_memory_violation else "✓"

                    print(f"  Latency: {result.total_latency_ms:.2f} ms {latency_ok}")
                    print(f"  Energy: {result.total_energy_mj:.2f} mJ")
                    print(f"  Throughput: {result.throughput_fps:.1f} FPS")
                    print(f"  Peak Memory: {result.peak_memory_mb:.1f} MB {memory_status}")

                    if has_memory_violation:
                        # Print memory warning
                        for warning in result.validation_warnings:
                            if "MEMORY CONSTRAINT VIOLATION" in warning:
                                # Extract just the key info
                                import re
                                match = re.search(r'Model size \(([\d.]+) MB\) exceeds available on-chip memory \(([\d.]+) MB\)', warning)
                                if match:
                                    model_size, available = match.groups()
                                    print(f"  WARNING: Model ({model_size} MB) > On-chip ({available} MB)")

                except Exception as e:
                    print(f"  ERROR: {e}")
                    continue

    # Generate comparison report
    print("\n" + "="*80)
    print("GENERATING COMPARISON REPORT")
    print("="*80)

    comparison_report = generator.generate_comparison_report(
        results,
        comparison_dimension='hardware',
        format='markdown',
        sort_by='latency'
    )

    # Save to file
    output_file = repo_root / "automotive_hardware_comparison.md"
    with open(output_file, 'w') as f:
        f.write(comparison_report)

    print(f"\nReport saved to: {output_file}")

    # Print summary table
    print("\n" + "="*80)
    print("PERFORMANCE SUMMARY (ResNet50 @ FP16)")
    print("="*80)
    print(f"{'Hardware':<25} {'Latency (ms)':<15} {'Energy (mJ)':<15} {'FPS':<10} {'Status':<10}")
    print("-"*80)

    # Filter for ResNet50 @ FP16
    resnet50_fp16 = [r for r in results if r.model_name == 'ResNet-50' and r.precision == Precision.FP16]
    resnet50_fp16.sort(key=lambda r: r.total_latency_ms)

    for result in resnet50_fp16:
        has_memory_violation = any("MEMORY CONSTRAINT VIOLATION" in w for w in result.validation_warnings)
        if has_memory_violation:
            latency_status = "✗ FAIL (memory)"
        elif result.total_latency_ms < 100:
            latency_status = "✓ PASS"
        else:
            latency_status = "✗ FAIL (latency)"
        print(f"{result.hardware_name:<25} {result.total_latency_ms:<15.2f} "
              f"{result.total_energy_mj:<15.2f} {result.throughput_fps:<10.1f} {latency_status:<10}")

    print("\n" + "="*80)
    print("PERFORMANCE SUMMARY (DeepLabV3 @ FP16)")
    print("="*80)
    print(f"{'Hardware':<25} {'Latency (ms)':<15} {'Energy (mJ)':<15} {'FPS':<10} {'Status':<10}")
    print("-"*80)

    # Filter for DeepLabV3 @ FP16
    deeplabv3_fp16 = [r for r in results if r.model_name == 'DeepLabV3-ResNet50' and r.precision == Precision.FP16]
    deeplabv3_fp16.sort(key=lambda r: r.total_latency_ms)

    for result in deeplabv3_fp16:
        has_memory_violation = any("MEMORY CONSTRAINT VIOLATION" in w for w in result.validation_warnings)
        if has_memory_violation:
            latency_status = "✗ FAIL (memory)"
        elif result.total_latency_ms < 100:
            latency_status = "✓ PASS"
        else:
            latency_status = "✗ FAIL (latency)"
        print(f"{result.hardware_name:<25} {result.total_latency_ms:<15.2f} "
              f"{result.total_energy_mj:<15.2f} {result.throughput_fps:<10.1f} {latency_status:<10}")

    print("\n" + "="*80)
    print("PERFORMANCE SUMMARY (FCN-ResNet50 @ FP16)")
    print("="*80)
    print(f"{'Hardware':<25} {'Latency (ms)':<15} {'Energy (mJ)':<15} {'FPS':<10} {'Status':<10}")
    print("-"*80)

    # Filter for FCN @ FP16
    fcn_fp16 = [r for r in results if r.model_name == 'FCN-ResNet50' and r.precision == Precision.FP16]
    fcn_fp16.sort(key=lambda r: r.total_latency_ms)

    for result in fcn_fp16:
        has_memory_violation = any("MEMORY CONSTRAINT VIOLATION" in w for w in result.validation_warnings)
        if has_memory_violation:
            latency_status = "✗ FAIL (memory)"
        elif result.total_latency_ms < 100:
            latency_status = "✓ PASS"
        else:
            latency_status = "✗ FAIL (latency)"
        print(f"{result.hardware_name:<25} {result.total_latency_ms:<15.2f} "
              f"{result.total_energy_mj:<15.2f} {result.throughput_fps:<10.1f} {latency_status:<10}")

    print("\n" + "="*80)
    print("PERFORMANCE SUMMARY (ViT-B/16 @ FP16)")
    print("="*80)
    print(f"{'Hardware':<25} {'Latency (ms)':<15} {'Energy (mJ)':<15} {'FPS':<10} {'Status':<10}")
    print("-"*80)

    # Filter for ViT @ FP16
    vit_fp16 = [r for r in results if r.model_name == 'ViT-B/16' and r.precision == Precision.FP16]
    vit_fp16.sort(key=lambda r: r.total_latency_ms)

    for result in vit_fp16:
        has_memory_violation = any("MEMORY CONSTRAINT VIOLATION" in w for w in result.validation_warnings)
        if has_memory_violation:
            latency_status = "✗ FAIL (memory)"
        elif result.total_latency_ms < 100:
            latency_status = "✓ PASS"
        else:
            latency_status = "✗ FAIL (latency)"
        print(f"{result.hardware_name:<25} {result.total_latency_ms:<15.2f} "
              f"{result.total_energy_mj:<15.2f} {result.throughput_fps:<10.1f} {latency_status:<10}")

    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)

    # Best performers for each model
    for model_results, model_label in [
        (resnet50_fp16, "ResNet50"),
        (deeplabv3_fp16, "DeepLabV3"),
        (fcn_fp16, "FCN-ResNet50"),
        (vit_fp16, "ViT-B/16")
    ]:
        if model_results:
            best_latency = min(model_results, key=lambda r: r.total_latency_ms)
            best_energy = min(model_results, key=lambda r: r.total_energy_mj)
            best_throughput = max(model_results, key=lambda r: r.throughput_fps)

            print(f"\n{model_label}:")
            print(f"  Best Latency: {best_latency.hardware_name} @ {best_latency.total_latency_ms:.2f} ms")
            print(f"  Best Energy: {best_energy.hardware_name} @ {best_energy.total_energy_mj:.2f} mJ")
            print(f"  Best Throughput: {best_throughput.hardware_name} @ {best_throughput.throughput_fps:.1f} FPS")

    # Count platforms meeting <100ms target
    print(f"\nPlatforms Meeting <100ms Real-Time Target:")
    for model_results, model_label in [
        (resnet50_fp16, "ResNet50"),
        (deeplabv3_fp16, "DeepLabV3"),
        (fcn_fp16, "FCN-ResNet50"),
        (vit_fp16, "ViT-B/16")
    ]:
        if model_results:
            passing = sum(1 for r in model_results if r.total_latency_ms < 100)
            print(f"  {model_label}: {passing}/{len(model_results)}")

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"Total Analyses: {len(results)}")
    print(f"Output: {output_file}")


if __name__ == '__main__':
    run_automotive_comparison()
