#!/usr/bin/env python
"""
Calibration Analysis Tool

Analyzes empirical sweep results against analytical estimates to:
1. Compute error metrics (MAPE, R², max error)
2. Identify bottleneck transitions (compute vs memory bound)
3. Recommend efficiency_factor coefficient updates
4. Generate calibration report

Usage:
    python validation/empirical/calibration_analysis.py \
        --input results/mlp_sweep_cpu.csv \
        --output results/calibration_report.md
"""

import argparse
import csv
import statistics
from pathlib import Path
from typing import List, Dict, Tuple
import sys


def load_sweep_results(csv_file: str) -> List[Dict]:
    """Load sweep results from CSV file"""
    results = []
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric fields
            for key in row:
                if key.endswith('_ms') or key.endswith('_mb') or key.endswith('_pct') \
                   or key.endswith('_flops') or key.endswith('_ai') or key.endswith('_throughput') \
                   or key in ['input_dim', 'output_dim', 'batch_size']:
                    try:
                        row[key] = float(row[key])
                    except (ValueError, TypeError):
                        pass
            results.append(row)
    return results


def compute_mape(empirical: List[float], analytical: List[float]) -> float:
    """Compute Mean Absolute Percentage Error"""
    errors = [abs(e - a) / e * 100 for e, a in zip(empirical, analytical) if e > 0]
    return statistics.mean(errors) if errors else 0.0


def compute_r_squared(empirical: List[float], analytical: List[float]) -> float:
    """Compute R² coefficient of determination"""
    if not empirical or not analytical:
        return 0.0

    mean_emp = statistics.mean(empirical)
    ss_tot = sum((e - mean_emp) ** 2 for e in empirical)
    ss_res = sum((e - a) ** 2 for e, a in zip(empirical, analytical))

    if ss_tot == 0:
        return 0.0

    return 1 - (ss_res / ss_tot)


def analyze_by_bottleneck(results: List[Dict]) -> Tuple[Dict, Dict]:
    """Separate results by bottleneck type and compute statistics"""
    compute_bound = [r for r in results if r.get('analytical_ai', 0) > 10]
    memory_bound = [r for r in results if r.get('analytical_ai', 0) <= 10]

    def stats(data, key):
        values = [r[key] for r in data if key in r]
        if not values:
            return {'count': 0, 'mean': 0, 'min': 0, 'max': 0}
        return {
            'count': len(values),
            'mean': statistics.mean(values),
            'min': min(values),
            'max': max(values),
        }

    compute_stats = {
        'count': len(compute_bound),
        'time_error': stats(compute_bound, 'time_error_pct'),
        'memory_error': stats(compute_bound, 'memory_error_pct'),
    }

    memory_stats = {
        'count': len(memory_bound),
        'time_error': stats(memory_bound, 'time_error_pct'),
        'memory_error': stats(memory_bound, 'memory_error_pct'),
    }

    return compute_stats, memory_stats


def recommend_efficiency_factor(results: List[Dict]) -> float:
    """
    Recommend efficiency_factor coefficient based on compute-bound configs

    For compute-bound workloads (AI > 10), the ratio of empirical/analytical
    time gives us the actual performance vs theoretical peak.
    """
    compute_bound = [r for r in results if r.get('analytical_ai', 0) > 10]

    if not compute_bound:
        return None

    # Compute empirical/analytical ratio for each config
    ratios = []
    for r in compute_bound:
        emp = r.get('empirical_time_ms', 0)
        ana = r.get('analytical_time_ms', 0)
        if ana > 0 and emp > 0:
            # Ratio > 1 means we're slower than predicted (need to decrease derate)
            # Ratio < 1 means we're faster than predicted (increase derate)
            ratio = emp / ana
            ratios.append(ratio)

    if not ratios:
        return None

    # Mean ratio tells us how far off we are
    mean_ratio = statistics.mean(ratios)

    # Current derate can be inferred (typically 0.6-0.8 for CPUs)
    # If empirical is 50% slower than analytical (ratio=1.5),
    # and analytical assumed 70% derate, then actual is 70%/1.5 = 47%
    assumed_current_derate = 0.70  # Typical starting value

    recommended_derate = assumed_current_derate / mean_ratio

    return recommended_derate


def generate_markdown_report(results: List[Dict], output_file: str):
    """Generate comprehensive calibration report in Markdown"""

    if not results:
        print("No results to analyze!")
        return

    # Overall statistics
    time_errors = [r['time_error_pct'] for r in results if 'time_error_pct' in r]
    memory_errors = [r['memory_error_pct'] for r in results if 'memory_error_pct' in r]

    empirical_times = [r['empirical_time_ms'] for r in results if 'empirical_time_ms' in r]
    analytical_times = [r['analytical_time_ms'] for r in results if 'analytical_time_ms' in r]

    mape_time = compute_mape(empirical_times, analytical_times)
    r2_time = compute_r_squared(empirical_times, analytical_times)

    # Bottleneck analysis
    compute_stats, memory_stats = analyze_by_bottleneck(results)

    # Recommendation
    recommended_derate = recommend_efficiency_factor(results)

    # Extract device and precision from first result
    device = results[0].get('device', 'unknown')
    precision = results[0].get('precision', 'unknown')

    # Generate report
    report = f"""# Calibration Report: MLP Sweep on {device.upper()}

**Date**: {Path(output_file).stem}
**Device**: {device}
**Precision**: {precision}
**Total Configurations**: {len(results)}

---

## Executive Summary

### Overall Accuracy

| Metric | Value | Assessment |
|--------|-------|------------|
| **Time MAPE†** | {mape_time:.1f}% | {'✓ Excellent' if mape_time < 10 else '⚠ Needs tuning' if mape_time < 20 else '✗ Poor'} |
| **Time R²** | {r2_time:.3f} | {'✓ Good correlation' if r2_time > 0.9 else '⚠ Moderate' if r2_time > 0.7 else '✗ Poor correlation'} |
| **Memory MAPE†** | {statistics.mean(memory_errors) if memory_errors else 0:.1f}% | {'✓ Excellent' if statistics.mean(memory_errors) < 5 else '⚠ Good' if statistics.mean(memory_errors) < 15 else '✗ Needs work'} |

† **MAPE** = Mean Absolute Percentage Error: Average of |empirical - analytical| / empirical × 100%

### Error Distribution

| Statistic | Time Error (%) | Memory Error (%) |
|-----------|----------------|------------------|
| **Mean** | {statistics.mean(time_errors):.1f} | {statistics.mean(memory_errors) if memory_errors else 0:.1f} |
| **Median** | {statistics.median(time_errors):.1f} | {statistics.median(memory_errors) if memory_errors else 0:.1f} |
| **Std Dev** | {statistics.stdev(time_errors) if len(time_errors) > 1 else 0:.1f} | {statistics.stdev(memory_errors) if len(memory_errors) > 1 else 0:.1f} |
| **Min** | {min(time_errors):.1f} | {min(memory_errors) if memory_errors else 0:.1f} |
| **Max** | {max(time_errors):.1f} | {max(memory_errors) if memory_errors else 0:.1f} |

---

## Bottleneck Analysis

### Compute-Bound Workloads (AI > 10)

**Count**: {compute_stats['count']} configurations

| Metric | Mean Error (%) | Min | Max |
|--------|----------------|-----|-----|
| **Time** | {compute_stats['time_error']['mean']:.1f} | {compute_stats['time_error']['min']:.1f} | {compute_stats['time_error']['max']:.1f} |
| **Memory** | {compute_stats['memory_error']['mean']:.1f} | {compute_stats['memory_error']['min']:.1f} | {compute_stats['memory_error']['max']:.1f} |

**Interpretation**: Compute-bound workloads are limited by FLOPS capacity. Lower error here indicates good modeling of peak throughput and instruction efficiency.

### Memory-Bound Workloads (AI ≤ 10)

**Count**: {memory_stats['count']} configurations

| Metric | Mean Error (%) | Min | Max |
|--------|----------------|-----|-----|
| **Time** | {memory_stats['time_error']['mean']:.1f} | {memory_stats['time_error']['min']:.1f} | {memory_stats['time_error']['max']:.1f} |
| **Memory** | {memory_stats['memory_error']['mean']:.1f} | {memory_stats['memory_error']['min']:.1f} | {memory_stats['memory_error']['max']:.1f} |

**Interpretation**: Memory-bound workloads are limited by bandwidth. Higher error here suggests memory bottleneck modeling needs refinement.

---

## Calibration Recommendations

### Empirical Derate Coefficient

"""

    if recommended_derate is not None:
        report += f"""
**Current (assumed)**: `efficiency_factor = 0.70` (typical starting value)
**Recommended**: `efficiency_factor = {recommended_derate:.3f}`

**Change**: {((recommended_derate - 0.70) / 0.70 * 100):+.1f}%

#### How to Apply

Update the hardware mapper for **{device}** at precision **{precision}**:

```python
# File: src/graphs/characterize/{'cpu_mapper.py' if device == 'cpu' else 'gpu_mapper.py'}

Precision.{precision.upper()}: PerformanceCharacteristics(
    precision=Precision.{precision.upper()},
    compute_resource=compute_resource,
    efficiency_factor={recommended_derate:.3f},  # ← UPDATE THIS
    tile_utilization=0.95,
    native_acceleration=True,
),
```

**Expected improvement**: MAPE should decrease from {mape_time:.1f}% to ~{mape_time * 0.70:.1f}% (±30% reduction)
"""
    else:
        report += """
**Status**: ⚠ Insufficient compute-bound workloads to recommend derate coefficient

**Action**: Run sweep with larger models to get more compute-bound configs (AI > 10).
"""

    # Memory bottleneck factor recommendation
    if memory_stats['count'] > 0:
        memory_time_error = memory_stats['time_error']['mean']
        current_mem_factor = 0.60
        # If we're slower than predicted for memory-bound, reduce the factor
        recommended_mem_factor = current_mem_factor * (100 / (100 + memory_time_error))

        report += f"""

### Memory Bottleneck Factor

**Current (assumed)**: `memory_bottleneck_factor = 0.60`
**Recommended**: `memory_bottleneck_factor = {recommended_mem_factor:.3f}`

**Rationale**: Memory-bound configs show {memory_time_error:.1f}% error. Adjusting this factor accounts for memory bandwidth contention and cache effects.

```python
# File: src/graphs/characterize/{'cpu_mapper.py' if device == 'cpu' else 'gpu_mapper.py'}

Precision.{precision.upper()}: PerformanceCharacteristics(
    precision=Precision.{precision.upper()},
    compute_resource=compute_resource,
    efficiency_factor={recommended_derate if recommended_derate else 0.70:.3f},
    memory_bottleneck_factor={recommended_mem_factor:.3f},  # ← ADD/UPDATE THIS
    tile_utilization=0.95,
    native_acceleration=True,
),
```
"""

    # Worst offenders
    worst_time_errors = sorted(results, key=lambda r: r.get('time_error_pct', 0), reverse=True)[:5]

    report += f"""

---

## Worst Offenders (Top 5 by Time Error)

| Model | Config | Batch | Empirical (ms) | Analytical (ms) | Error (%) | AI | Bottleneck |
|-------|--------|-------|----------------|-----------------|-----------|----|-----------{chr(10)}"""

    for r in worst_time_errors:
        report += f"| {r.get('model', 'N/A')} | {r.get('hidden_dims', 'N/A')} | {r.get('batch_size', 0):.0f} | {r.get('empirical_time_ms', 0):.3f} | {r.get('analytical_time_ms', 0):.3f} | {r.get('time_error_pct', 0):.1f} | {r.get('analytical_ai', 0):.1f} | {r.get('analytical_bottleneck', 'N/A')} |\n"

    report += """

**Action**: Investigate these configs to understand systematic errors.

---

## Next Steps

1. **Apply Recommendations**: Update hardware mapper coefficients as shown above
2. **Re-run Validation**: Execute validation tests to verify improvement
   ```bash
   python validation/estimators/test_resnet18.py
   python validation/hardware/test_all_hardware.py
   ```
3. **Verify MAPE**: Target < 10% for production use, < 15% acceptable
4. **Iterate**: If errors persist, run additional sweeps with different parameter ranges

---

## Appendix: Full Results

"""

    # Add table of all results
    report += "| Model | Input | Hidden | Output | Batch | Emp Time | Ana Time | Error (%) | AI | Bottleneck |\n"
    report += "|-------|-------|--------|--------|-------|----------|----------|-----------|----|-----------|\n"

    for r in results:
        report += f"| {r.get('model', '')} | {r.get('input_dim', 0):.0f} | {r.get('hidden_dims', '')} | {r.get('output_dim', 0):.0f} | {r.get('batch_size', 0):.0f} | {r.get('empirical_time_ms', 0):.3f} | {r.get('analytical_time_ms', 0):.3f} | {r.get('time_error_pct', 0):.1f} | {r.get('analytical_ai', 0):.1f} | {r.get('analytical_bottleneck', '')} |\n"

    # Write report
    with open(output_file, 'w') as f:
        f.write(report)

    print(f"✓ Calibration report written to: {output_file}")

    # Print summary to console
    print("\n" + "=" * 80)
    print("CALIBRATION SUMMARY")
    print("=" * 80)
    print(f"Device: {device.upper()}, Precision: {precision.upper()}")
    print(f"Configs tested: {len(results)}")
    print(f"\nTime MAPE†: {mape_time:.1f}% {'✓' if mape_time < 15 else '⚠'}")
    print(f"Time R²:    {r2_time:.3f} {'✓' if r2_time > 0.8 else '⚠'}")
    print(f"\n† MAPE = Mean Absolute Percentage Error")

    if recommended_derate:
        print(f"\nRecommended efficiency_factor: {recommended_derate:.3f}")
        print(f"  (change from 0.70: {((recommended_derate - 0.70) / 0.70 * 100):+.1f}%)")

    print(f"\nBottleneck breakdown:")
    print(f"  Compute-bound: {compute_stats['count']} configs, {compute_stats['time_error']['mean']:.1f}% MAPE")
    print(f"  Memory-bound:  {memory_stats['count']} configs, {memory_stats['time_error']['mean']:.1f}% MAPE")


def main():
    parser = argparse.ArgumentParser(description="Calibration Analysis Tool")
    parser.add_argument('--input', required=True, help="Input CSV file from sweep")
    parser.add_argument('--output', default=None, help="Output Markdown report file")

    args = parser.parse_args()

    # Default output file
    if args.output is None:
        input_path = Path(args.input)
        output_file = input_path.parent / f"{input_path.stem}_calibration.md"
    else:
        output_file = args.output

    # Load results
    print(f"Loading sweep results from: {args.input}")
    results = load_sweep_results(args.input)
    print(f"  Loaded {len(results)} configurations")

    # Generate report
    generate_markdown_report(results, str(output_file))


if __name__ == "__main__":
    main()
