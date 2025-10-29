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

Usage:
    # Batch size sweep
    ./cli/analyze_batch_v2.py --model resnet18 --hardware H100 \
        --batch-size 1 2 4 8 16 32 --output results.csv

    # Model comparison
    ./cli/analyze_batch_v2.py --models resnet18 mobilenet_v2 efficientnet_b0 \
        --hardware H100 --batch-size 1 16 32 --output comparison.csv

    # Hardware comparison
    ./cli/analyze_batch_v2.py --model resnet50 \
        --hardware H100 Jetson-Orin-AGX KPU-T256 \
        --batch-size 1 8 16 --output hardware_comp.csv
"""

import argparse
import sys
from pathlib import Path
from typing import List

# Add repo root to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from graphs.analysis.unified_analyzer import UnifiedAnalyzer, AnalysisConfig, UnifiedAnalysisResult
from graphs.reporting import ReportGenerator
from graphs.hardware.resource_model import Precision


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

    # Output configuration
    parser.add_argument('--output', '-o', type=str,
                       help='Output file path (CSV or JSON)')
    parser.add_argument('--format', '-f',
                       choices=['csv', 'json', 'markdown'],
                       help='Output format (auto-detected if --output provided)')

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

    # Show insights
    if args.show_insights and not args.quiet:
        print_batch_insights(results)

    # Generate output
    try:
        generator = ReportGenerator()

        # Determine format
        if args.output:
            if args.format:
                format_type = args.format
            else:
                ext = Path(args.output).suffix.lower()
                format_map = {
                    '.json': 'json',
                    '.csv': 'csv',
                    '.md': 'markdown',
                }
                format_type = format_map.get(ext, 'csv')

            # Generate comparison report
            content = generator.generate_comparison_report(
                results,
                format=format_type,
                sort_by='latency'
            )

            with open(args.output, 'w') as f:
                f.write(content)

            if not args.quiet:
                print(f"\n✓ Results saved to: {args.output}")

        else:
            # Print to stdout (CSV by default)
            format_type = args.format or 'csv'
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
