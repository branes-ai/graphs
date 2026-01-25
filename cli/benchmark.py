#!/usr/bin/env python
"""
Benchmark CLI Tool

Command-line interface for running microbenchmarks and collecting performance data.
Supports GEMM, Conv2d, memory bandwidth, and workload benchmarks.

Usage:
    # Run single benchmark by name
    ./cli/benchmark.py --benchmark gemm_1024x1024_fp32

    # Run benchmark suite
    ./cli/benchmark.py --suite gemm
    ./cli/benchmark.py --suite conv2d
    ./cli/benchmark.py --suite memory

    # Specify device
    ./cli/benchmark.py --suite gemm --device cuda
    ./cli/benchmark.py --benchmark gemm_1024x1024_fp32 --device cpu

    # Configure execution
    ./cli/benchmark.py --suite gemm --warmup 20 --iterations 200

    # Output formats
    ./cli/benchmark.py --suite gemm --output results.json
    ./cli/benchmark.py --suite gemm --output results.csv
    ./cli/benchmark.py --suite gemm --format table

    # Filter benchmarks
    ./cli/benchmark.py --suite gemm --tags transformer
    ./cli/benchmark.py --suite gemm --precision fp16

    # Save to registry (for calibration)
    ./cli/benchmark.py --suite gemm --save-to-registry
"""

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Add repo root to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from graphs.benchmarks.schema import (
    BenchmarkResult,
    BenchmarkSpec,
    GEMMSpec,
    Conv2dSpec,
    MemoryBenchSpec,
    ExecutionConfig,
    Precision,
    DeviceType,
)
from graphs.benchmarks.runner import PyTorchRunner, resolve_device, get_device_name
from graphs.benchmarks.microbench.gemm import (
    get_gemm_specs,
    create_gemm_spec,
    format_result_table,
)


# Registry path for saving results
REGISTRY_PATH = repo_root / "hardware_registry" / "benchmark_results"


def get_conv2d_specs(
    filter_tags: Optional[List[str]] = None,
) -> List[Conv2dSpec]:
    """Load Conv2d benchmark specifications."""
    import yaml

    specs_dir = repo_root / "src" / "graphs" / "benchmarks" / "specs" / "conv2d"
    specs = []

    for yaml_file in sorted(specs_dir.glob("*.yaml")):
        with open(yaml_file, 'r') as f:
            content = f.read()

        for doc in yaml.safe_load_all(content):
            if doc is None:
                continue

            precisions = []
            for p in doc.get('precisions', ['fp32']):
                try:
                    precisions.append(Precision(p.lower()))
                except ValueError:
                    pass

            devices = []
            for d in doc.get('devices', ['auto']):
                try:
                    devices.append(DeviceType(d.lower()))
                except ValueError:
                    pass

            exec_cfg = doc.get('execution', {})
            execution = ExecutionConfig(
                warmup_iterations=exec_cfg.get('warmup_iterations', 10),
                measurement_iterations=exec_cfg.get('measurement_iterations', 100),
                sync_before_timing=exec_cfg.get('sync_before_timing', True),
            )

            spec = Conv2dSpec(
                name=doc['name'],
                description=doc.get('description', ''),
                tags=doc.get('tags', []),
                batch_size=doc.get('batch_size', 1),
                in_channels=doc['in_channels'],
                out_channels=doc['out_channels'],
                height=doc['height'],
                width=doc['width'],
                kernel_size=doc['kernel_size'],
                stride=doc.get('stride', 1),
                padding=doc.get('padding', 0),
                groups=doc.get('groups', 1),
                precisions=precisions,
                devices=devices,
                execution=execution,
            )

            if filter_tags:
                if not any(tag in spec.tags for tag in filter_tags):
                    continue

            specs.append(spec)

    return specs


def get_memory_specs(
    filter_tags: Optional[List[str]] = None,
) -> List[MemoryBenchSpec]:
    """Load memory bandwidth benchmark specifications."""
    import yaml

    specs_dir = repo_root / "src" / "graphs" / "benchmarks" / "specs" / "memory"
    specs = []

    for yaml_file in sorted(specs_dir.glob("*.yaml")):
        with open(yaml_file, 'r') as f:
            content = f.read()

        for doc in yaml.safe_load_all(content):
            if doc is None:
                continue

            precisions = []
            for p in doc.get('precisions', ['fp32']):
                try:
                    precisions.append(Precision(p.lower()))
                except ValueError:
                    pass

            devices = []
            for d in doc.get('devices', ['auto']):
                try:
                    devices.append(DeviceType(d.lower()))
                except ValueError:
                    pass

            exec_cfg = doc.get('execution', {})
            execution = ExecutionConfig(
                warmup_iterations=exec_cfg.get('warmup_iterations', 10),
                measurement_iterations=exec_cfg.get('measurement_iterations', 100),
                sync_before_timing=exec_cfg.get('sync_before_timing', True),
            )

            spec = MemoryBenchSpec(
                name=doc['name'],
                description=doc.get('description', ''),
                tags=doc.get('tags', []),
                array_size=doc.get('array_size', 10_000_000),
                pattern=doc.get('pattern', 'triad'),
                precisions=precisions,
                devices=devices,
                execution=execution,
            )

            if filter_tags:
                if not any(tag in spec.tags for tag in filter_tags):
                    continue

            specs.append(spec)

    return specs


def find_benchmark_by_name(name: str) -> Optional[BenchmarkSpec]:
    """Find a benchmark spec by name across all suites."""
    # Search GEMM specs
    for spec in get_gemm_specs():
        if spec.name == name:
            return spec

    # Search Conv2d specs
    for spec in get_conv2d_specs():
        if spec.name == name:
            return spec

    # Search memory specs
    for spec in get_memory_specs():
        if spec.name == name:
            return spec

    return None


def list_benchmarks(suite: Optional[str] = None) -> Dict[str, List[str]]:
    """List available benchmarks by suite."""
    result = {}

    if suite is None or suite == "gemm":
        result["gemm"] = [s.name for s in get_gemm_specs()]

    if suite is None or suite == "conv2d":
        result["conv2d"] = [s.name for s in get_conv2d_specs()]

    if suite is None or suite == "memory":
        result["memory"] = [s.name for s in get_memory_specs()]

    return result


def run_benchmark_suite(
    suite: str,
    device: str,
    precision: Optional[Precision],
    config: ExecutionConfig,
    filter_tags: Optional[List[str]] = None,
    include_batched: bool = False,
    quiet: bool = False,
) -> List[BenchmarkResult]:
    """Run all benchmarks in a suite."""
    specs: List[BenchmarkSpec] = []

    if suite == "gemm":
        specs = get_gemm_specs(filter_tags=filter_tags)
        if not include_batched:
            specs = [s for s in specs if s.batch_size == 1]
    elif suite == "conv2d":
        specs = get_conv2d_specs(filter_tags=filter_tags)
    elif suite == "memory":
        specs = get_memory_specs(filter_tags=filter_tags)
    else:
        raise ValueError(f"Unknown suite: {suite}")

    runner = PyTorchRunner(config)
    results = []

    for i, spec in enumerate(specs):
        if not quiet:
            print(f"[{i+1}/{len(specs)}] {spec.name}...", end=" ", flush=True)

        try:
            p = precision if precision else spec.precisions[0]
            result = runner.run(spec, device, p)
            results.append(result)

            if not quiet:
                if result.success and result.timing:
                    if result.gflops > 0:
                        print(f"{result.timing.mean_ms:.3f} ms, {result.gflops:.1f} GFLOPS")
                    elif result.bandwidth_gbps > 0:
                        print(f"{result.timing.mean_ms:.3f} ms, {result.bandwidth_gbps:.1f} GB/s")
                    else:
                        print(f"{result.timing.mean_ms:.3f} ms")
                else:
                    print(f"FAILED: {result.error_message}")

        except Exception as e:
            if not quiet:
                print(f"ERROR: {e}")
            results.append(BenchmarkResult(
                spec_name=spec.name,
                timestamp=datetime.now().isoformat(),
                device=device,
                precision=(precision.value if precision else "fp32"),
                success=False,
                error_message=str(e),
            ))

    return results


def format_results_table(results: List[BenchmarkResult]) -> str:
    """Format results as a text table."""
    lines = []

    # Header
    lines.append(
        f"{'Name':<40} {'Device':<8} {'Prec':<6} "
        f"{'Mean (ms)':<12} {'Std (ms)':<10} {'GFLOPS':<10} {'GB/s':<10} {'Status'}"
    )
    lines.append("-" * 115)

    for r in results:
        if r.success and r.timing:
            gflops_str = f"{r.gflops:.1f}" if r.gflops > 0 else "-"
            bw_str = f"{r.bandwidth_gbps:.1f}" if r.bandwidth_gbps > 0 else "-"
            lines.append(
                f"{r.spec_name:<40} {r.device:<8} {r.precision:<6} "
                f"{r.timing.mean_ms:<12.3f} {r.timing.std_ms:<10.3f} "
                f"{gflops_str:<10} {bw_str:<10} OK"
            )
        else:
            error_msg = r.error_message[:15] if r.error_message else "Unknown"
            lines.append(
                f"{r.spec_name:<40} {r.device:<8} {r.precision:<6} "
                f"{'-':<12} {'-':<10} {'-':<10} {'-':<10} FAILED: {error_msg}"
            )

    return "\n".join(lines)


def save_results_json(results: List[BenchmarkResult], output_path: Path) -> None:
    """Save results to JSON file."""
    data = [r.to_dict() for r in results]
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)


def save_results_csv(results: List[BenchmarkResult], output_path: Path) -> None:
    """Save results to CSV file."""
    if not results:
        return

    with open(output_path, 'w', newline='') as f:
        # Flatten timing stats into the dict
        fieldnames = [
            'spec_name', 'timestamp', 'device', 'device_name', 'precision',
            'success', 'error_message',
            'mean_ms', 'std_ms', 'min_ms', 'max_ms', 'median_ms', 'p95_ms', 'p99_ms',
            'gflops', 'bandwidth_gbps', 'throughput_samples_per_sec',
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()

        for r in results:
            row = {
                'spec_name': r.spec_name,
                'timestamp': r.timestamp,
                'device': r.device,
                'device_name': r.device_name,
                'precision': r.precision,
                'success': r.success,
                'error_message': r.error_message,
                'gflops': r.gflops,
                'bandwidth_gbps': r.bandwidth_gbps,
                'throughput_samples_per_sec': r.throughput_samples_per_sec,
            }
            if r.timing:
                row.update({
                    'mean_ms': r.timing.mean_ms,
                    'std_ms': r.timing.std_ms,
                    'min_ms': r.timing.min_ms,
                    'max_ms': r.timing.max_ms,
                    'median_ms': r.timing.median_ms,
                    'p95_ms': r.timing.p95_ms,
                    'p99_ms': r.timing.p99_ms,
                })
            writer.writerow(row)


def save_to_registry(results: List[BenchmarkResult], device_name: str) -> Path:
    """Save results to benchmark registry for calibration."""
    REGISTRY_PATH.mkdir(parents=True, exist_ok=True)

    # Create filename with device and timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_device = device_name.replace(" ", "_").replace("/", "_")
    filename = f"{safe_device}_{timestamp}.json"

    output_path = REGISTRY_PATH / filename

    # Add metadata
    data = {
        "metadata": {
            "device_name": device_name,
            "timestamp": datetime.now().isoformat(),
            "num_benchmarks": len(results),
            "successful": sum(1 for r in results if r.success),
        },
        "results": [r.to_dict() for r in results],
    }

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    return output_path


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Benchmark CLI Tool - Run microbenchmarks for hardware characterization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  ./cli/benchmark.py --benchmark gemm_1024x1024_fp32
  ./cli/benchmark.py --suite gemm --device cuda
  ./cli/benchmark.py --suite gemm --output results.json
  ./cli/benchmark.py --suite memory --device cpu
  ./cli/benchmark.py --list
  ./cli/benchmark.py --list --suite gemm
        """
    )

    # Benchmark selection
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--benchmark", "-b",
        type=str,
        help="Run a specific benchmark by name"
    )
    selection.add_argument(
        "--suite", "-s",
        type=str,
        choices=["gemm", "conv2d", "memory", "all"],
        help="Run all benchmarks in a suite"
    )
    selection.add_argument(
        "--list", "-l",
        type=str,
        nargs="?",
        const="all",
        choices=["all", "gemm", "conv2d", "memory"],
        help="List available benchmarks (optionally filter by suite)"
    )

    # Device selection
    parser.add_argument(
        "--device", "-d",
        type=str,
        default="auto",
        help="Device to run on (cpu, cuda, cuda:0, auto)"
    )

    # Precision
    parser.add_argument(
        "--precision", "-p",
        type=str,
        choices=["fp32", "fp16", "bf16", "int8"],
        help="Precision to use (overrides spec default)"
    )

    # Execution config
    parser.add_argument(
        "--warmup", "-w",
        type=int,
        default=10,
        help="Number of warmup iterations"
    )
    parser.add_argument(
        "--iterations", "-i",
        type=int,
        default=100,
        help="Number of measurement iterations"
    )

    # Filtering
    parser.add_argument(
        "--tags", "-t",
        nargs="+",
        help="Filter benchmarks by tags"
    )
    parser.add_argument(
        "--include-batched",
        action="store_true",
        help="Include batched benchmarks (for GEMM suite)"
    )

    # Output
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output file (json or csv based on extension)"
    )
    parser.add_argument(
        "--format", "-f",
        type=str,
        choices=["table", "json", "csv"],
        default="table",
        help="Output format (default: table)"
    )
    parser.add_argument(
        "--save-to-registry",
        action="store_true",
        help="Save results to benchmark registry for calibration"
    )

    # Verbosity
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output with extra details"
    )

    args = parser.parse_args()

    # Handle --list
    if args.list:
        suite_filter = args.list if args.list != "all" else None
        benchmarks = list_benchmarks(suite=suite_filter)
        for suite_name, names in benchmarks.items():
            print(f"\n{suite_name.upper()} ({len(names)} benchmarks):")
            for name in names:
                print(f"  {name}")
        return 0

    # Validate selection
    if not args.benchmark and not args.suite:
        parser.error("Must specify --benchmark or --suite (or use --list to see available)")

    # Resolve device
    device = args.device
    if device == "auto":
        device = resolve_device(DeviceType.AUTO)

    device_name = get_device_name(device)

    if not args.quiet:
        print(f"Device: {device_name} ({device})")
        print()

    # Parse precision
    precision = None
    if args.precision:
        precision = Precision(args.precision)

    # Create execution config
    config = ExecutionConfig(
        warmup_iterations=args.warmup,
        measurement_iterations=args.iterations,
        sync_before_timing=True,
    )

    # Run benchmarks
    results: List[BenchmarkResult] = []

    if args.benchmark:
        # Single benchmark
        spec = find_benchmark_by_name(args.benchmark)
        if spec is None:
            print(f"Error: Benchmark '{args.benchmark}' not found", file=sys.stderr)
            print("\nUse --list to see available benchmarks", file=sys.stderr)
            return 1

        runner = PyTorchRunner(config)
        p = precision if precision else spec.precisions[0]

        if not args.quiet:
            print(f"Running {spec.name}...", flush=True)

        try:
            result = runner.run(spec, device, p)
            results.append(result)
        except Exception as e:
            print(f"Error running benchmark: {e}", file=sys.stderr)
            return 1

    elif args.suite:
        # Benchmark suite
        if args.suite == "all":
            for suite in ["gemm", "conv2d", "memory"]:
                if not args.quiet:
                    print(f"\n=== {suite.upper()} Suite ===\n")
                suite_results = run_benchmark_suite(
                    suite, device, precision, config,
                    filter_tags=args.tags,
                    include_batched=args.include_batched,
                    quiet=args.quiet,
                )
                results.extend(suite_results)
        else:
            results = run_benchmark_suite(
                args.suite, device, precision, config,
                filter_tags=args.tags,
                include_batched=args.include_batched,
                quiet=args.quiet,
            )

    # Output results
    output_path = Path(args.output) if args.output else None

    # Determine format from output path if not specified
    output_format = args.format
    if output_path:
        if output_path.suffix == ".json":
            output_format = "json"
        elif output_path.suffix == ".csv":
            output_format = "csv"

    if output_format == "json":
        if output_path:
            save_results_json(results, output_path)
            if not args.quiet:
                print(f"\nResults written to {output_path}")
        else:
            print(json.dumps([r.to_dict() for r in results], indent=2))

    elif output_format == "csv":
        if output_path:
            save_results_csv(results, output_path)
            if not args.quiet:
                print(f"\nResults written to {output_path}")
        else:
            # CSV to stdout
            for r in results:
                if r.success and r.timing:
                    print(f"{r.spec_name},{r.timing.mean_ms:.3f},{r.gflops:.1f}")

    else:  # table
        print()
        print(format_results_table(results))
        print()

        # Summary
        successful = [r for r in results if r.success]
        if successful:
            gflops_results = [r for r in successful if r.gflops > 0]
            bw_results = [r for r in successful if r.bandwidth_gbps > 0]

            summary_parts = [f"{len(successful)}/{len(results)} succeeded"]
            if gflops_results:
                avg_gflops = sum(r.gflops for r in gflops_results) / len(gflops_results)
                summary_parts.append(f"avg {avg_gflops:.1f} GFLOPS")
            if bw_results:
                avg_bw = sum(r.bandwidth_gbps for r in bw_results) / len(bw_results)
                summary_parts.append(f"avg {avg_bw:.1f} GB/s")

            print(f"Summary: {', '.join(summary_parts)}")

    # Save to registry if requested
    if args.save_to_registry:
        registry_path = save_to_registry(results, device_name)
        if not args.quiet:
            print(f"\nResults saved to registry: {registry_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
