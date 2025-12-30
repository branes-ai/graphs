#!/usr/bin/env python
"""
Batch Size Analysis Tool (Refactored with Unified Framework)

Analyzes the impact of batching on performance, energy, and efficiency.
Performs batch size sweeps to understand trade-offs between latency,
throughput, energy efficiency, and memory usage.

Supports:
- Batch size sweeps (single model/hardware)
- Model comparison (same hardware, same batch sizes)
- Hardware comparison (same model, same batch sizes)
- Verdict-first output for agentic workflows (constraint checking)

Usage:
    # Batch size sweep
    ./cli/analyze_batch.py --model resnet18 --hardware H100 \
        --batch-size 1 2 4 8 16 32 --output results.csv

    # Model comparison
    ./cli/analyze_batch.py --models resnet18 mobilenet_v2 efficientnet_b0 \
        --hardware H100 --batch-size 1 16 32 --output comparison.csv

    # Hardware comparison
    ./cli/analyze_batch.py --model resnet50 \
        --hardware H100 Jetson-Orin-AGX KPU-T256 \
        --batch-size 1 8 16 --output hardware_comp.csv

    # Verdict-first: Find batch sizes that meet latency constraint
    ./cli/analyze_batch.py --model resnet18 --hardware H100 \
        --batch-size 1 2 4 8 16 32 --check-latency 5.0

    # Verdict-first: Find batch sizes that meet memory constraint
    ./cli/analyze_batch.py --model resnet50 --hardware Jetson-Orin-AGX \
        --batch-size 1 2 4 8 --check-memory 1000
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any

# Add repo root to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from graphs.analysis.unified_analyzer import UnifiedAnalyzer, AnalysisConfig, UnifiedAnalysisResult
from graphs.reporting import ReportGenerator
from graphs.hardware.resource_model import Precision


# =============================================================================
# Verdict-First Output Generation
# =============================================================================

def get_constraint_value(result: UnifiedAnalysisResult, metric: str) -> float:
    """Get the actual value for a constraint metric from a result."""
    if metric == 'latency':
        return result.total_latency_ms
    elif metric == 'power':
        return result.energy_report.average_power_w
    elif metric == 'memory':
        return result.peak_memory_mb
    elif metric == 'energy':
        return result.energy_per_inference_mj
    else:
        raise ValueError(f"Unknown constraint metric: {metric}")


def evaluate_constraint(actual: float, threshold: float, lower_is_better: bool = True) -> Dict[str, Any]:
    """Evaluate a single constraint and return pass/fail with margin."""
    if lower_is_better:
        passes = actual <= threshold
        margin_pct = ((threshold - actual) / threshold) * 100
    else:
        passes = actual >= threshold
        margin_pct = ((actual - threshold) / threshold) * 100

    return {
        "passes": passes,
        "actual": actual,
        "threshold": threshold,
        "margin_pct": margin_pct,
    }


def generate_batch_verdict(
    results: List[UnifiedAnalysisResult],
    constraint_metric: str,
    constraint_threshold: float,
) -> str:
    """Generate verdict-first JSON output for batch sweep analysis.

    Args:
        results: List of analysis results from batch sweep
        constraint_metric: Metric to check ('latency', 'power', 'memory', 'energy')
        constraint_threshold: Threshold value for the constraint

    Returns:
        JSON string with verdict-first batch sweep output
    """
    if not results:
        return json.dumps({
            "verdict": "UNKNOWN",
            "confidence": "low",
            "summary": "No results to analyze",
            "passing_configs": [],
            "failing_configs": [],
        }, indent=2)

    # Group results by model+hardware
    groups: Dict[tuple, List[UnifiedAnalysisResult]] = {}
    for r in results:
        key = (r.model_name, r.hardware_name)
        if key not in groups:
            groups[key] = []
        groups[key].append(r)

    # Build output structure
    all_passing = []
    all_failing = []
    group_summaries = []

    for (model, hardware), group_results in groups.items():
        # Sort by batch size
        group_results.sort(key=lambda r: r.batch_size)

        passing = []
        failing = []

        for r in group_results:
            actual = get_constraint_value(r, constraint_metric)
            eval_result = evaluate_constraint(actual, constraint_threshold)

            config = {
                "model": r.model_name,
                "hardware": r.hardware_name,
                "batch_size": r.batch_size,
                "precision": r.precision.name.lower(),
                constraint_metric: round(actual, 3),
                "margin_pct": round(eval_result["margin_pct"], 1),
                "latency_ms": round(r.total_latency_ms, 3),
                "throughput_fps": round(r.throughput_fps, 1),
                "energy_per_inference_mj": round(r.energy_per_inference_mj, 2),
                "peak_memory_mb": round(r.peak_memory_mb, 1),
            }

            if eval_result["passes"]:
                passing.append(config)
            else:
                failing.append(config)

        all_passing.extend(passing)
        all_failing.extend(failing)

        # Generate recommendations for this group
        recommendations = {}
        if passing:
            # Best for latency (smallest batch that passes)
            best_latency = min(passing, key=lambda c: c["latency_ms"])
            recommendations["for_latency"] = {
                "batch_size": best_latency["batch_size"],
                "latency_ms": best_latency["latency_ms"],
            }

            # Best for throughput (highest throughput that passes)
            best_throughput = max(passing, key=lambda c: c["throughput_fps"])
            recommendations["for_throughput"] = {
                "batch_size": best_throughput["batch_size"],
                "throughput_fps": best_throughput["throughput_fps"],
            }

            # Best for energy efficiency (lowest energy/inference that passes)
            best_energy = min(passing, key=lambda c: c["energy_per_inference_mj"])
            recommendations["for_energy_efficiency"] = {
                "batch_size": best_energy["batch_size"],
                "energy_per_inference_mj": best_energy["energy_per_inference_mj"],
            }

        group_summaries.append({
            "model": model,
            "hardware": hardware,
            "total_configs": len(group_results),
            "passing_count": len(passing),
            "failing_count": len(failing),
            "recommendations": recommendations if passing else None,
        })

    # Determine overall verdict
    total_configs = len(all_passing) + len(all_failing)
    if len(all_failing) == 0:
        verdict = "PASS"
        summary = f"All {total_configs} configurations meet {constraint_metric} target of {constraint_threshold}"
    elif len(all_passing) == 0:
        verdict = "FAIL"
        summary = f"No configurations meet {constraint_metric} target of {constraint_threshold}"
    else:
        verdict = "PARTIAL"
        summary = f"{len(all_passing)} of {total_configs} configurations meet {constraint_metric} target of {constraint_threshold}"

    # Build suggestions
    suggestions = []
    if verdict == "FAIL":
        suggestions.append(f"All tested batch sizes exceed {constraint_metric} target")
        if constraint_metric == 'latency':
            suggestions.append("Try smaller batch sizes or faster hardware")
        elif constraint_metric == 'memory':
            suggestions.append("Try smaller batch sizes or hardware with more memory")
        elif constraint_metric == 'power':
            suggestions.append("Try smaller batch sizes or more power-efficient hardware")
        elif constraint_metric == 'energy':
            suggestions.append("Try larger batch sizes to amortize static energy")
    elif verdict == "PARTIAL":
        max_passing_batch = max(c["batch_size"] for c in all_passing)
        suggestions.append(f"Maximum batch size meeting constraint: {max_passing_batch}")

    # Determine confidence
    confidence = "high" if total_configs >= 3 else "medium"

    output = {
        "verdict": verdict,
        "confidence": confidence,
        "summary": summary,
        "constraint": {
            "metric": constraint_metric,
            "threshold": constraint_threshold,
        },
        "total_configs": total_configs,
        "passing_count": len(all_passing),
        "failing_count": len(all_failing),
        "passing_configs": all_passing,
        "failing_configs": all_failing,
        "group_summaries": group_summaries,
        "suggestions": suggestions,
    }

    return json.dumps(output, indent=2)


def generate_batch_verdict_no_constraint(results: List[UnifiedAnalysisResult]) -> str:
    """Generate verdict-first JSON output for batch sweep without constraint.

    When no constraint is specified, provides a summary with recommendations
    for different optimization goals.

    Args:
        results: List of analysis results from batch sweep

    Returns:
        JSON string with batch sweep summary
    """
    if not results:
        return json.dumps({
            "verdict": "UNKNOWN",
            "confidence": "low",
            "summary": "No results to analyze",
            "configs": [],
        }, indent=2)

    # Group results by model+hardware
    groups: Dict[tuple, List[UnifiedAnalysisResult]] = {}
    for r in results:
        key = (r.model_name, r.hardware_name)
        if key not in groups:
            groups[key] = []
        groups[key].append(r)

    all_configs = []
    group_summaries = []

    for (model, hardware), group_results in groups.items():
        # Sort by batch size
        group_results.sort(key=lambda r: r.batch_size)

        configs = []
        for r in group_results:
            configs.append({
                "model": r.model_name,
                "hardware": r.hardware_name,
                "batch_size": r.batch_size,
                "precision": r.precision.name.lower(),
                "latency_ms": round(r.total_latency_ms, 3),
                "throughput_fps": round(r.throughput_fps, 1),
                "energy_per_inference_mj": round(r.energy_per_inference_mj, 2),
                "peak_memory_mb": round(r.peak_memory_mb, 1),
            })

        all_configs.extend(configs)

        # Generate recommendations
        recommendations = {}
        if configs:
            best_latency = min(configs, key=lambda c: c["latency_ms"])
            recommendations["for_latency"] = {
                "batch_size": best_latency["batch_size"],
                "latency_ms": best_latency["latency_ms"],
            }

            best_throughput = max(configs, key=lambda c: c["throughput_fps"])
            recommendations["for_throughput"] = {
                "batch_size": best_throughput["batch_size"],
                "throughput_fps": best_throughput["throughput_fps"],
            }

            best_energy = min(configs, key=lambda c: c["energy_per_inference_mj"])
            recommendations["for_energy_efficiency"] = {
                "batch_size": best_energy["batch_size"],
                "energy_per_inference_mj": best_energy["energy_per_inference_mj"],
            }

        group_summaries.append({
            "model": model,
            "hardware": hardware,
            "total_configs": len(configs),
            "recommendations": recommendations,
        })

    # Build summary
    batch_sizes = sorted(set(c["batch_size"] for c in all_configs))
    summary = f"Analyzed {len(all_configs)} configurations across batch sizes {batch_sizes}"

    output = {
        "verdict": "PASS",
        "confidence": "high" if len(all_configs) >= 3 else "medium",
        "summary": summary,
        "total_configs": len(all_configs),
        "batch_sizes": batch_sizes,
        "configs": all_configs,
        "group_summaries": group_summaries,
    }

    return json.dumps(output, indent=2)


# =============================================================================
# Batch Analysis
# =============================================================================

def analyze_batch_sweep(
    model_names: List[str],
    hardware_names: List[str],
    batch_sizes: List[int],
    precision: Precision,
    verbose: bool = True
) -> List[UnifiedAnalysisResult]:
    """
    Run batch size sweep analysis.

    Args:
        model_names: Models to analyze
        hardware_names: Hardware targets
        batch_sizes: Batch sizes to test
        precision: Precision to use
        verbose: Print progress

    Returns:
        List of analysis results
    """
    analyzer = UnifiedAnalyzer(verbose=False)
    results = []

    total_configs = len(model_names) * len(hardware_names) * len(batch_sizes)
    current = 0

    for model in model_names:
        for hardware in hardware_names:
            for batch_size in batch_sizes:
                current += 1
                if verbose:
                    print(f"[{current}/{total_configs}] Analyzing {model} @ {hardware}, batch={batch_size}...")

                try:
                    result = analyzer.analyze_model(
                        model_name=model,
                        hardware_name=hardware,
                        batch_size=batch_size,
                        precision=precision,
                        config=AnalysisConfig(
                            run_roofline=True,
                            run_energy=True,
                            run_memory=True,
                            run_concurrency=False,
                            validate_consistency=False  # Skip for speed
                        )
                    )
                    results.append(result)
                except Exception as e:
                    print(f"  Error: {e}", file=sys.stderr)
                    continue

    return results


def print_batch_insights(results: List[UnifiedAnalysisResult]):
    """Print insights specific to batch size analysis"""
    if not results:
        return

    # Group by model+hardware
    groups = {}
    for r in results:
        key = (r.model_name, r.hardware_name)
        if key not in groups:
            groups[key] = []
        groups[key].append(r)

    print("\n" + "="*79)
    print(f"{'BATCH SIZE INSIGHTS':^79}")
    print("="*79 + "\n")

    for (model, hardware), group_results in groups.items():
        if len(group_results) < 2:
            continue

        # Sort by batch size
        group_results.sort(key=lambda r: r.batch_size)
        baseline = group_results[0]  # Batch size 1 (or smallest)
        max_batch = group_results[-1]

        print(f"{model} on {hardware}:")
        print("-" * 79)

        # Throughput improvement
        thru_improvement = max_batch.throughput_fps / baseline.throughput_fps
        print(f"  • Throughput improvement: {thru_improvement:.1f}× "
              f"(batch {baseline.batch_size}: {baseline.throughput_fps:.0f} fps → "
              f"batch {max_batch.batch_size}: {max_batch.throughput_fps:.0f} fps)")

        # Energy per inference improvement
        best_energy = min(group_results, key=lambda r: r.energy_per_inference_mj)
        energy_improvement = baseline.energy_per_inference_mj / best_energy.energy_per_inference_mj
        print(f"  • Energy/inference improvement: {energy_improvement:.1f}× "
              f"(batch {baseline.batch_size}: {baseline.energy_per_inference_mj:.1f} mJ → "
              f"batch {best_energy.batch_size}: {best_energy.energy_per_inference_mj:.1f} mJ)")

        # Latency trade-off
        latency_increase = max_batch.total_latency_ms / baseline.total_latency_ms
        print(f"  • Latency increase: {latency_increase:.1f}× "
              f"({baseline.total_latency_ms:.2f} ms → {max_batch.total_latency_ms:.2f} ms)")

        # Memory growth
        memory_increase = max_batch.peak_memory_mb / baseline.peak_memory_mb
        print(f"  • Memory growth: {memory_increase:.1f}× "
              f"({baseline.peak_memory_mb:.1f} MB → {max_batch.peak_memory_mb:.1f} MB)")

        # Recommendations
        print(f"\n  Recommendations:")
        if best_energy.batch_size != baseline.batch_size:
            print(f"    - For energy efficiency: Use batch {best_energy.batch_size}")
        if max_batch.batch_size > baseline.batch_size:
            print(f"    - For throughput: Use batch {max_batch.batch_size}")
        if baseline.total_latency_ms < 10:
            print(f"    - For low latency: Use batch {baseline.batch_size}")

        print()


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Batch size impact analysis tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Batch size sweep (single model/hardware)
  %(prog)s --model resnet18 --hardware H100 --batch-size 1 2 4 8 16 32

  # Model comparison (same hardware, same batch sizes)
  %(prog)s --models resnet18 mobilenet_v2 efficientnet_b0 \
      --hardware H100 --batch-size 1 16 32

  # Hardware comparison (same model, same batch sizes)
  %(prog)s --model resnet50 --hardware H100 Jetson-Orin-AGX KPU-T256 \
      --batch-size 1 8 16

  # Verdict-first: Find batch sizes meeting latency constraint
  %(prog)s --model resnet18 --hardware H100 \
      --batch-size 1 2 4 8 16 32 --check-latency 5.0

  # Verdict-first: Find batch sizes meeting memory constraint
  %(prog)s --model resnet50 --hardware Jetson-Orin-AGX \
      --batch-size 1 2 4 8 --check-memory 1000

  # Explicit verdict format output
  %(prog)s --model resnet18 --hardware H100 \
      --batch-size 1 4 16 --format verdict
        """
    )

    # Model and hardware
    parser.add_argument('--model', type=str,
                       help='Single model to analyze')
    parser.add_argument('--models', nargs='+',
                       help='Multiple models to compare')
    parser.add_argument('--hardware', nargs='+', required=True,
                       help='Hardware target(s) to analyze')

    # Batch sizes
    parser.add_argument('--batch-size', nargs='+', type=int, required=True,
                       help='Batch sizes to test')

    # Analysis configuration
    parser.add_argument('--precision', default='fp32',
                       choices=['fp32', 'fp16', 'int8'],
                       help='Precision (default: fp32)')

    # Constraint checking (verdict-first output)
    parser.add_argument('--check-latency', type=float, metavar='MS',
                       help='Check if latency is under target (in milliseconds)')
    parser.add_argument('--check-power', type=float, metavar='WATTS',
                       help='Check if average power is under budget (in watts)')
    parser.add_argument('--check-memory', type=float, metavar='MB',
                       help='Check if peak memory is under limit (in megabytes)')
    parser.add_argument('--check-energy', type=float, metavar='MJ',
                       help='Check if energy per inference is under limit (in millijoules)')

    # Output configuration
    parser.add_argument('--output', '-o', type=str,
                       help='Output file path (CSV, JSON, Markdown, HTML, or verdict JSON)')
    parser.add_argument('--format', '-f',
                       choices=['csv', 'json', 'markdown', 'html', 'verdict'],
                       help='Output format (verdict: verdict-first JSON for agentic workflows)')

    # Display options
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Suppress progress output')
    parser.add_argument('--show-insights', action='store_true', default=True,
                       help='Show batch size insights (default: true)')
    parser.add_argument('--no-insights', dest='show_insights', action='store_false',
                       help='Hide batch size insights')

    args = parser.parse_args()

    # Validate arguments
    if not args.model and not args.models:
        print("Error: Must specify either --model or --models", file=sys.stderr)
        return 1

    # Parse model list
    if args.models:
        model_names = args.models
    else:
        model_names = [args.model]

    # Parse hardware list
    hardware_names = args.hardware

    # Parse precision
    precision_map = {
        'fp32': Precision.FP32,
        'fp16': Precision.FP16,
        'int8': Precision.INT8,
    }
    precision = precision_map[args.precision.lower()]

    # Run batch sweep analysis
    try:
        if not args.quiet:
            print(f"\n{'='*79}")
            print(f"{'BATCH SIZE ANALYSIS':^79}")
            print(f"{'='*79}")
            print(f"Models: {', '.join(model_names)}")
            print(f"Hardware: {', '.join(hardware_names)}")
            print(f"Batch Sizes: {', '.join(map(str, args.batch_size))}")
            print(f"Precision: {precision.name}")
            print(f"{'='*79}\n")

        results = analyze_batch_sweep(
            model_names=model_names,
            hardware_names=hardware_names,
            batch_sizes=args.batch_size,
            precision=precision,
            verbose=not args.quiet
        )

        if not results:
            print("\nNo results generated", file=sys.stderr)
            return 1

    except Exception as e:
        print(f"\nError during analysis: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

    # Determine constraint (if any)
    constraint_metric = None
    constraint_threshold = None
    if args.check_latency is not None:
        constraint_metric = 'latency'
        constraint_threshold = args.check_latency
    elif args.check_power is not None:
        constraint_metric = 'power'
        constraint_threshold = args.check_power
    elif args.check_memory is not None:
        constraint_metric = 'memory'
        constraint_threshold = args.check_memory
    elif args.check_energy is not None:
        constraint_metric = 'energy'
        constraint_threshold = args.check_energy

    # Show insights (unless using verdict format)
    format_type = args.format
    if constraint_metric and not format_type:
        format_type = 'verdict'  # Auto-switch to verdict when constraint specified

    if args.show_insights and not args.quiet and format_type != 'verdict':
        print_batch_insights(results)

    # Generate output
    try:
        # Determine format for file output
        if args.output:
            if args.format:
                format_type = args.format
            else:
                ext = Path(args.output).suffix.lower()
                format_map = {
                    '.json': 'json',
                    '.csv': 'csv',
                    '.md': 'markdown',
                    '.html': 'html',
                }
                format_type = format_map.get(ext, 'csv')
                # Override to verdict if constraint specified and no explicit format
                if constraint_metric and not args.format:
                    format_type = 'verdict'

            # Generate output based on format
            if format_type == 'verdict':
                if constraint_metric:
                    content = generate_batch_verdict(results, constraint_metric, constraint_threshold)
                else:
                    # No constraint - generate verdict with summary only
                    content = generate_batch_verdict_no_constraint(results)
            else:
                generator = ReportGenerator()
                content = generator.generate_comparison_report(
                    results,
                    format=format_type,
                    sort_by='latency'
                )

            with open(args.output, 'w') as f:
                f.write(content)

            if not args.quiet:
                print(f"\nResults saved to: {args.output}")

        else:
            # Print to stdout
            if not format_type:
                format_type = 'csv'

            if format_type == 'verdict':
                if constraint_metric:
                    content = generate_batch_verdict(results, constraint_metric, constraint_threshold)
                else:
                    content = generate_batch_verdict_no_constraint(results)
                print(content)
            else:
                generator = ReportGenerator()
                content = generator.generate_comparison_report(
                    results,
                    format=format_type,
                    sort_by='latency'
                )
                print("\n" + content)

    except Exception as e:
        print(f"\nError generating output: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
