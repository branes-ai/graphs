#!/usr/bin/env python
"""
Batch Analysis Tool

Analyzes neural networks across multiple configurations to understand
the impact of batch size, model selection, hardware choice, and precision
on performance, energy, and memory usage.

Primary Use Cases:
1. Batch Size Sweep: Understand latency/throughput/energy trade-offs
2. Model Comparison: Compare multiple models on same hardware
3. Hardware Comparison: Compare same model across different hardware
4. Precision Comparison: Analyze quantization impact (FP32/FP16/INT8)
5. Full Sweeps: Combinations of the above

Key Insights from Batch Size Analysis:
- Latency increases linearly with batch size
- Throughput increases then plateaus (hardware saturation)
- Energy per inference decreases (amortized static energy)
- Memory increases linearly with batch size
- Energy efficiency improves significantly with batching

Usage:
    # Batch size sweep (analyze batching impact)
    ./cli/analyze_batch.py --model resnet18 --hardware H100 \\
        --batch-size 1 2 4 8 16 32 64 \\
        --output batch_sweep.csv

    # Model comparison
    ./cli/analyze_batch.py \\
        --models resnet18 resnet50 mobilenet_v2 efficientnet_b0 \\
        --hardware H100 \\
        --output model_comparison.csv

    # Hardware comparison
    ./cli/analyze_batch.py --model resnet18 \\
        --hardware H100 A100 Jetson-Orin TPU-v4 \\
        --output hardware_comparison.csv

    # Precision comparison
    ./cli/analyze_batch.py --model resnet18 --hardware H100 \\
        --precision fp32 fp16 int8 \\
        --output precision_comparison.csv

    # Full sweep (combinations)
    ./cli/analyze_batch.py \\
        --models resnet18 mobilenet_v2 \\
        --hardware H100 Jetson-Orin \\
        --precision fp32 fp16 \\
        --batch-size 1 4 16 \\
        --output full_sweep.csv
"""

import argparse
import csv
import json
import sys
import itertools
from typing import List, Dict, Any
from dataclasses import dataclass, asdict

# Import the comprehensive analyzer
from analyze_comprehensive import (
    analyze_model,
    ComprehensiveAnalysisReport,
    Precision,
)


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class BatchSweepResult:
    """Result from a single configuration in the sweep"""

    # Configuration
    model: str
    hardware: str
    precision: str
    batch_size: int

    # Performance
    latency_ms: float
    throughput_fps: float
    utilization_pct: float

    # Latency per inference (for batch size analysis)
    latency_per_inference_ms: float

    # Energy
    total_energy_mj: float
    energy_per_inference_mj: float  # Total energy / batch size
    compute_energy_mj: float
    memory_energy_mj: float
    static_energy_mj: float
    average_power_w: float
    energy_efficiency_pct: float

    # Memory
    peak_memory_mb: float
    weight_memory_mb: float
    activation_memory_mb: float
    workspace_memory_mb: float

    # Roofline
    arithmetic_intensity_median: float
    ai_breakpoint: float
    memory_bound_pct: float
    compute_bound_pct: float

    # Model info
    total_flops: float
    total_bytes: float
    total_params: int
    num_subgraphs: int


def report_to_sweep_result(
    report: ComprehensiveAnalysisReport
) -> BatchSweepResult:
    """Convert comprehensive report to batch sweep result"""

    # Calculate per-inference metrics
    latency_per_inf = report.latency_ms / report.batch_size if report.batch_size > 0 else report.latency_ms
    energy_per_inf = report.total_energy_mj / report.batch_size if report.batch_size > 0 else report.total_energy_mj

    return BatchSweepResult(
        # Configuration
        model=report.model_name,
        hardware=report.hardware_name,
        precision=report.precision,
        batch_size=report.batch_size,

        # Performance
        latency_ms=report.latency_ms,
        throughput_fps=report.throughput_fps,
        utilization_pct=report.utilization_pct,
        latency_per_inference_ms=latency_per_inf,

        # Energy
        total_energy_mj=report.total_energy_mj,
        energy_per_inference_mj=energy_per_inf,
        compute_energy_mj=report.compute_energy_mj,
        memory_energy_mj=report.memory_energy_mj,
        static_energy_mj=report.static_energy_mj,
        average_power_w=report.average_power_w,
        energy_efficiency_pct=report.energy_efficiency_pct,

        # Memory
        peak_memory_mb=report.peak_memory_mb,
        weight_memory_mb=report.weight_memory_mb,
        activation_memory_mb=report.activation_memory_mb,
        workspace_memory_mb=report.workspace_memory_mb,

        # Roofline
        arithmetic_intensity_median=report.arithmetic_intensity_median,
        ai_breakpoint=report.ai_breakpoint,
        memory_bound_pct=report.memory_bound_pct,
        compute_bound_pct=report.compute_bound_pct,

        # Model info
        total_flops=report.total_flops,
        total_bytes=report.total_bytes,
        total_params=report.total_params,
        num_subgraphs=report.num_subgraphs,
    )


# =============================================================================
# Batch Sweep Engine
# =============================================================================

def run_batch_sweep(
    models: List[str],
    hardware_list: List[str],
    precisions: List[str],
    batch_sizes: List[int],
    verbose: bool = True
) -> List[BatchSweepResult]:
    """
    Run batch sweep across all configurations

    Args:
        models: List of model names
        hardware_list: List of hardware names
        precisions: List of precision strings ('fp32', 'fp16', 'int8')
        batch_sizes: List of batch sizes
        verbose: Print progress

    Returns:
        List of BatchSweepResult objects
    """

    # Generate all combinations
    configurations = list(itertools.product(
        models,
        hardware_list,
        precisions,
        batch_sizes
    ))

    total_configs = len(configurations)

    if verbose:
        print(f"\n{'='*80}")
        print(f"BATCH SWEEP ANALYSIS")
        print(f"{'='*80}")
        print(f"Total configurations: {total_configs}")
        print(f"  Models: {len(models)} ({', '.join(models)})")
        print(f"  Hardware: {len(hardware_list)} ({', '.join(hardware_list)})")
        print(f"  Precisions: {len(precisions)} ({', '.join(precisions)})")
        print(f"  Batch sizes: {len(batch_sizes)} ({', '.join(map(str, batch_sizes))})")
        print(f"{'='*80}\n")

    results = []

    # Run each configuration
    for i, (model, hardware, precision_str, batch_size) in enumerate(configurations, 1):
        if verbose:
            print(f"[{i}/{total_configs}] Analyzing: {model} on {hardware} "
                  f"(precision={precision_str}, batch={batch_size})")

        # Convert precision string to enum
        precision_map = {
            'fp32': Precision.FP32,
            'fp16': Precision.FP16,
            'int8': Precision.INT8,
        }
        precision = precision_map[precision_str.lower()]

        try:
            # Run comprehensive analysis
            report = analyze_model(
                model_name=model,
                hardware_name=hardware,
                precision=precision,
                batch_size=batch_size,
                verbose=False  # Suppress per-analysis output
            )

            # Convert to sweep result
            result = report_to_sweep_result(report)
            results.append(result)

            if verbose:
                print(f"      ✓ Latency: {result.latency_ms:.2f} ms, "
                      f"Throughput: {result.throughput_fps:.0f} fps, "
                      f"Energy: {result.total_energy_mj:.2f} mJ, "
                      f"Memory: {result.peak_memory_mb:.1f} MB")

        except Exception as e:
            if verbose:
                print(f"      ✗ Error: {e}")
            # Continue with next configuration
            continue

    if verbose:
        print(f"\n{'='*80}")
        print(f"Sweep complete! {len(results)}/{total_configs} configurations analyzed successfully")
        print(f"{'='*80}\n")

    return results


# =============================================================================
# Output Formatting
# =============================================================================

def write_csv(results: List[BatchSweepResult], output_file: str):
    """Write results to CSV file"""

    if not results:
        print("No results to write", file=sys.stderr)
        return

    # Get field names from dataclass
    fieldnames = list(asdict(results[0]).keys())

    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for result in results:
            writer.writerow(asdict(result))

    print(f"Results written to: {output_file}")


def write_json(results: List[BatchSweepResult], output_file: str):
    """Write results to JSON file"""

    if not results:
        print("No results to write", file=sys.stderr)
        return

    # Convert to list of dicts
    results_dicts = [asdict(result) for result in results]

    with open(output_file, 'w') as f:
        json.dump(results_dicts, f, indent=2)

    print(f"Results written to: {output_file}")


def print_summary_table(results: List[BatchSweepResult], analysis_type: str = "general"):
    """Print summary table to stdout"""

    if not results:
        print("No results to display", file=sys.stderr)
        return

    print(f"\n{'='*120}")
    print(f"BATCH SWEEP SUMMARY ({len(results)} configurations)")
    print(f"{'='*120}\n")

    if analysis_type == "batch_size":
        # Special formatting for batch size analysis
        print(f"{'Model':<15} {'HW':<15} {'Prec':<6} {'Batch':>5} "
              f"{'Lat(ms)':>8} {'Lat/Inf':>8} {'Thru(fps)':>10} "
              f"{'Energy(mJ)':>11} {'E/Inf(mJ)':>10} {'Eff(%)':>7} {'Mem(MB)':>9}")
        print("-" * 120)

        for result in results:
            print(f"{result.model:<15} {result.hardware:<15} {result.precision:<6} {result.batch_size:>5} "
                  f"{result.latency_ms:>8.2f} {result.latency_per_inference_ms:>8.3f} {result.throughput_fps:>10.0f} "
                  f"{result.total_energy_mj:>11.2f} {result.energy_per_inference_mj:>10.3f} "
                  f"{result.energy_efficiency_pct:>7.1f} {result.peak_memory_mb:>9.1f}")

    else:
        # General formatting
        print(f"{'Model':<15} {'HW':<15} {'Prec':<6} {'Batch':>5} "
              f"{'Lat(ms)':>8} {'Thru(fps)':>10} {'Energy(mJ)':>11} {'Mem(MB)':>9} {'Util(%)':>8}")
        print("-" * 120)

        for result in results:
            print(f"{result.model:<15} {result.hardware:<15} {result.precision:<6} {result.batch_size:>5} "
                  f"{result.latency_ms:>8.2f} {result.throughput_fps:>10.0f} "
                  f"{result.total_energy_mj:>11.2f} {result.peak_memory_mb:>9.1f} "
                  f"{result.utilization_pct:>8.1f}")

    print()


def print_batch_size_insights(results: List[BatchSweepResult]):
    """Print insights specific to batch size analysis"""

    # Filter for batch size sweep (same model, hardware, precision)
    if not results:
        return

    # Group by (model, hardware, precision)
    from collections import defaultdict
    groups = defaultdict(list)

    for result in results:
        key = (result.model, result.hardware, result.precision)
        groups[key].append(result)

    for (model, hardware, precision), group_results in groups.items():
        if len(group_results) <= 1:
            continue  # Need multiple batch sizes for analysis

        # Sort by batch size
        group_results.sort(key=lambda r: r.batch_size)

        print(f"\n{'='*80}")
        print(f"BATCH SIZE ANALYSIS: {model} on {hardware} ({precision})")
        print(f"{'='*80}")

        # Compare batch sizes
        baseline = group_results[0]  # Batch size 1

        print(f"\n{'Batch':>6} {'Latency':>10} {'Lat/Inf':>10} {'Throughput':>12} {'Energy/Inf':>12} {'Efficiency':>11} {'Memory':>10}")
        print(f"{'Size':>6} {'(ms)':>10} {'(ms)':>10} {'(fps)':>12} {'(mJ)':>12} {'(%)':>11} {'(MB)':>10}")
        print("-" * 80)

        for result in group_results:
            # Calculate speedup/improvement vs baseline
            lat_ratio = result.latency_ms / baseline.latency_ms
            thru_ratio = result.throughput_fps / baseline.throughput_fps
            energy_ratio = result.energy_per_inference_mj / baseline.energy_per_inference_mj
            mem_ratio = result.peak_memory_mb / baseline.peak_memory_mb

            marker = ""
            if result.batch_size == 1:
                marker = " (baseline)"
            elif result.energy_per_inference_mj < baseline.energy_per_inference_mj * 0.5:
                marker = " ★ Best efficiency"

            print(f"{result.batch_size:>6} "
                  f"{result.latency_ms:>10.2f} "
                  f"{result.latency_per_inference_ms:>10.3f} "
                  f"{result.throughput_fps:>12.0f} "
                  f"{result.energy_per_inference_mj:>12.3f} "
                  f"{result.energy_efficiency_pct:>11.1f} "
                  f"{result.peak_memory_mb:>10.1f}{marker}")

        print("\nKey Insights:")

        # Throughput scaling
        max_batch = group_results[-1]
        thru_improvement = max_batch.throughput_fps / baseline.throughput_fps
        print(f"  • Throughput improvement: {thru_improvement:.1f}× "
              f"(batch 1: {baseline.throughput_fps:.0f} fps → "
              f"batch {max_batch.batch_size}: {max_batch.throughput_fps:.0f} fps)")

        # Energy per inference improvement
        best_energy = min(group_results, key=lambda r: r.energy_per_inference_mj)
        energy_improvement = baseline.energy_per_inference_mj / best_energy.energy_per_inference_mj
        print(f"  • Energy/inference improvement: {energy_improvement:.1f}× "
              f"(batch 1: {baseline.energy_per_inference_mj:.3f} mJ → "
              f"batch {best_energy.batch_size}: {best_energy.energy_per_inference_mj:.3f} mJ)")

        # Static energy impact
        static_reduction = (1 - max_batch.static_energy_mj / max_batch.total_energy_mj /
                           (baseline.static_energy_mj / baseline.total_energy_mj)) * 100
        print(f"  • Static energy impact reduced: {static_reduction:.0f}% "
              f"(batch 1: {baseline.static_energy_mj/baseline.total_energy_mj*100:.0f}% static → "
              f"batch {max_batch.batch_size}: {max_batch.static_energy_mj/max_batch.total_energy_mj*100:.0f}% static)")

        # Memory growth
        mem_growth = max_batch.peak_memory_mb / baseline.peak_memory_mb
        print(f"  • Memory growth: {mem_growth:.1f}× "
              f"({baseline.peak_memory_mb:.1f} MB → {max_batch.peak_memory_mb:.1f} MB)")

        # Recommended batch size
        # Find sweet spot: good throughput, reasonable energy, acceptable memory
        recommended = None
        for result in group_results:
            thru_ratio = result.throughput_fps / max_batch.throughput_fps
            mem_ok = result.peak_memory_mb < 1000  # Arbitrary threshold
            if thru_ratio > 0.7 and mem_ok:  # 70% of max throughput
                recommended = result.batch_size
                break

        if recommended:
            rec_result = next(r for r in group_results if r.batch_size == recommended)
            print(f"\n  ★ Recommended batch size: {recommended}")
            print(f"    - Throughput: {rec_result.throughput_fps:.0f} fps "
                  f"({rec_result.throughput_fps/max_batch.throughput_fps*100:.0f}% of peak)")
            print(f"    - Energy/inference: {rec_result.energy_per_inference_mj:.3f} mJ "
                  f"({baseline.energy_per_inference_mj/rec_result.energy_per_inference_mj:.1f}× better than batch 1)")
            print(f"    - Memory: {rec_result.peak_memory_mb:.1f} MB")

        print()


# =============================================================================
# Main Function
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Batch Analysis Tool - Sweep across models, hardware, precision, and batch sizes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Batch size sweep (analyze batching impact)
  ./cli/analyze_batch.py --model resnet18 --hardware H100 \\
      --batch-size 1 2 4 8 16 32 64 \\
      --output batch_sweep.csv

  # Model comparison
  ./cli/analyze_batch.py \\
      --models resnet18 resnet50 mobilenet_v2 efficientnet_b0 \\
      --hardware H100 \\
      --output model_comparison.csv

  # Hardware comparison
  ./cli/analyze_batch.py --model resnet18 \\
      --hardware H100 A100 Jetson-Orin TPU-v4 \\
      --output hardware_comparison.csv

  # Precision comparison
  ./cli/analyze_batch.py --model resnet18 --hardware H100 \\
      --precision fp32 fp16 int8 \\
      --output precision_comparison.csv

  # Full sweep
  ./cli/analyze_batch.py \\
      --models resnet18 mobilenet_v2 \\
      --hardware H100 Jetson-Orin \\
      --precision fp32 fp16 \\
      --batch-size 1 4 16 \\
      --output full_sweep.csv
        """
    )

    # Sweep parameters (can specify multiple values for each)
    parser.add_argument('--model', '--models', type=str, nargs='+', dest='models',
                        help='Model name(s) (resnet18, resnet50, mobilenet_v2, etc.)')
    parser.add_argument('--hardware', type=str, nargs='+', required=True,
                        help='Hardware target(s) (H100, A100, Jetson-Orin, TPU-v4, etc.)')
    parser.add_argument('--precision', type=str, nargs='+', default=['fp32'],
                        choices=['fp32', 'fp16', 'int8'],
                        help='Precision(s) (default: fp32)')
    parser.add_argument('--batch-size', type=int, nargs='+', default=[1],
                        help='Batch size(s) (default: 1)')

    # Output options
    parser.add_argument('--output', '--output-file', type=str, dest='output_file', required=True,
                        help='Output file path (.csv or .json)')
    parser.add_argument('--format', type=str, choices=['csv', 'json'], default=None,
                        help='Output format (default: inferred from file extension)')

    # Display options
    parser.add_argument('--show-summary', action='store_true', default=True,
                        help='Show summary table (default: True)')
    parser.add_argument('--show-insights', action='store_true', default=False,
                        help='Show batch size insights (only for batch size sweeps)')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress progress messages')

    args = parser.parse_args()

    # Validate inputs
    if not args.models:
        print("Error: --model/--models is required", file=sys.stderr)
        sys.exit(1)

    # Infer output format from file extension if not specified
    if args.format is None:
        if args.output_file.endswith('.json'):
            output_format = 'json'
        elif args.output_file.endswith('.csv'):
            output_format = 'csv'
        else:
            print("Error: Could not infer output format from file extension. "
                  "Please use .csv or .json extension, or specify --format",
                  file=sys.stderr)
            sys.exit(1)
    else:
        output_format = args.format

    # Run batch sweep
    results = run_batch_sweep(
        models=args.models,
        hardware_list=args.hardware,
        precisions=args.precision,
        batch_sizes=args.batch_size,
        verbose=not args.quiet
    )

    if not results:
        print("No successful analyses. Exiting.", file=sys.stderr)
        sys.exit(1)

    # Write output file
    if output_format == 'csv':
        write_csv(results, args.output_file)
    elif output_format == 'json':
        write_json(results, args.output_file)

    # Show summary table
    if args.show_summary and not args.quiet:
        # Determine analysis type for better formatting
        if len(args.batch_size) > 1 and len(args.models) == 1 and len(args.hardware) == 1:
            analysis_type = "batch_size"
        else:
            analysis_type = "general"

        print_summary_table(results, analysis_type=analysis_type)

    # Show batch size insights
    if args.show_insights or (len(args.batch_size) > 1 and not args.quiet):
        print_batch_size_insights(results)


if __name__ == "__main__":
    main()
