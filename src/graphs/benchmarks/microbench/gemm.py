"""
GEMM Microbenchmark Suite

Comprehensive GEMM (General Matrix Multiply) benchmarks for measuring
hardware compute performance across different sizes, precisions, and batch dimensions.

Usage:
    # As a module
    python -m graphs.benchmarks.microbench.gemm --device cpu
    python -m graphs.benchmarks.microbench.gemm --device cuda --precision fp16
    python -m graphs.benchmarks.microbench.gemm --size 1024 --all-precisions

    # Programmatic usage
    from graphs.benchmarks.microbench.gemm import run_gemm_benchmark, get_gemm_specs
    specs = get_gemm_specs()
    result = run_gemm_benchmark(specs[0], device="cuda")
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..schema import (
    BenchmarkResult,
    GEMMSpec,
    ExecutionConfig,
    Precision,
    DeviceType,
    TimingStats,
)
from ..runner import PyTorchRunner, resolve_device


# Package location for finding spec files
SPECS_DIR = Path(__file__).parent.parent / "specs" / "gemm"


@dataclass
class GEMMBenchmark:
    """
    GEMM benchmark executor.

    Wraps the PyTorchRunner with GEMM-specific functionality.
    """

    config: ExecutionConfig
    runner: PyTorchRunner = None

    def __post_init__(self):
        if self.runner is None:
            self.runner = PyTorchRunner(self.config)

    def run(
        self,
        spec: GEMMSpec,
        device: str = "auto",
        precision: Precision = Precision.FP32,
    ) -> BenchmarkResult:
        """Run a single GEMM benchmark."""
        return self.runner.run(spec, device, precision)

    def run_all_precisions(
        self,
        spec: GEMMSpec,
        device: str = "auto",
    ) -> List[BenchmarkResult]:
        """Run benchmark for all precisions in spec."""
        return self.runner.run_all_precisions(spec, device)

    def run_suite(
        self,
        specs: List[GEMMSpec],
        device: str = "auto",
        precision: Optional[Precision] = None,
    ) -> List[BenchmarkResult]:
        """
        Run a suite of GEMM benchmarks.

        Args:
            specs: List of GEMM specifications to run
            device: Target device
            precision: If specified, override spec precision. If None, use first precision in spec.

        Returns:
            List of BenchmarkResult for each spec
        """
        results = []
        for spec in specs:
            p = precision if precision is not None else spec.precisions[0]
            try:
                result = self.run(spec, device, p)
                results.append(result)
            except Exception as e:
                from datetime import datetime
                results.append(BenchmarkResult(
                    spec_name=spec.name,
                    timestamp=datetime.now().isoformat(),
                    device=device,
                    precision=p.value,
                    success=False,
                    error_message=str(e),
                ))
        return results


def get_gemm_specs(
    spec_dir: Optional[Path] = None,
    filter_tags: Optional[List[str]] = None,
    filter_sizes: Optional[List[int]] = None,
) -> List[GEMMSpec]:
    """
    Load GEMM benchmark specifications from YAML files.

    Args:
        spec_dir: Directory containing YAML spec files. Defaults to package specs.
        filter_tags: Only include specs with these tags
        filter_sizes: Only include specs with M dimension in this list

    Returns:
        List of GEMMSpec objects
    """
    import yaml

    if spec_dir is None:
        spec_dir = SPECS_DIR

    specs = []

    for yaml_file in sorted(spec_dir.glob("*.yaml")):
        with open(yaml_file, 'r') as f:
            content = f.read()

        # Handle multi-document YAML
        for doc in yaml.safe_load_all(content):
            if doc is None:
                continue

            # Convert precision strings to enum
            precisions = []
            for p in doc.get('precisions', ['fp32']):
                try:
                    precisions.append(Precision(p.lower()))
                except ValueError:
                    pass

            # Convert device strings to enum
            devices = []
            for d in doc.get('devices', ['auto']):
                try:
                    devices.append(DeviceType(d.lower()))
                except ValueError:
                    pass

            # Parse execution config
            exec_cfg = doc.get('execution', {})
            execution = ExecutionConfig(
                warmup_iterations=exec_cfg.get('warmup_iterations', 10),
                measurement_iterations=exec_cfg.get('measurement_iterations', 100),
                sync_before_timing=exec_cfg.get('sync_before_timing', True),
            )

            spec = GEMMSpec(
                name=doc['name'],
                description=doc.get('description', ''),
                tags=doc.get('tags', []),
                M=doc['M'],
                N=doc['N'],
                K=doc['K'],
                batch_size=doc.get('batch_size', 1),
                precisions=precisions,
                devices=devices,
                execution=execution,
            )

            # Apply filters
            if filter_tags:
                if not any(tag in spec.tags for tag in filter_tags):
                    continue

            if filter_sizes:
                if spec.M not in filter_sizes:
                    continue

            specs.append(spec)

    return specs


def run_gemm_benchmark(
    spec: GEMMSpec,
    device: str = "auto",
    precision: Precision = Precision.FP32,
    config: Optional[ExecutionConfig] = None,
) -> BenchmarkResult:
    """
    Convenience function to run a single GEMM benchmark.

    Args:
        spec: GEMM specification
        device: Target device (cpu, cuda, cuda:0, etc.)
        precision: Numerical precision
        config: Optional execution config override

    Returns:
        BenchmarkResult with timing and performance metrics
    """
    cfg = config if config is not None else spec.execution
    benchmark = GEMMBenchmark(config=cfg)
    return benchmark.run(spec, device, precision)


def create_gemm_spec(
    M: int,
    N: int,
    K: int,
    batch_size: int = 1,
    name: Optional[str] = None,
    precision: Precision = Precision.FP32,
) -> GEMMSpec:
    """
    Create a simple GEMM spec programmatically.

    Args:
        M, N, K: Matrix dimensions (M x K) @ (K x N)
        batch_size: Number of independent matrix multiplies
        name: Spec name (auto-generated if None)
        precision: Numerical precision

    Returns:
        GEMMSpec ready for benchmarking
    """
    if name is None:
        if batch_size > 1:
            name = f"gemm_{M}x{N}x{K}_batch{batch_size}"
        else:
            name = f"gemm_{M}x{N}x{K}"

    return GEMMSpec(
        name=name,
        M=M,
        N=N,
        K=K,
        batch_size=batch_size,
        precisions=[precision],
    )


def format_result_table(results: List[BenchmarkResult]) -> str:
    """Format results as a text table."""
    lines = []

    # Header
    lines.append(f"{'Name':<35} {'Device':<8} {'Prec':<6} {'Mean (ms)':<12} {'Std (ms)':<10} {'GFLOPS':<12} {'Status'}")
    lines.append("-" * 100)

    for r in results:
        if r.success and r.timing is not None:
            lines.append(
                f"{r.spec_name:<35} {r.device:<8} {r.precision:<6} "
                f"{r.timing.mean_ms:<12.3f} {r.timing.std_ms:<10.3f} "
                f"{r.gflops:<12.1f} OK"
            )
        else:
            lines.append(
                f"{r.spec_name:<35} {r.device:<8} {r.precision:<6} "
                f"{'N/A':<12} {'N/A':<10} {'N/A':<12} FAILED: {r.error_message[:20]}"
            )

    return "\n".join(lines)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="GEMM Microbenchmark Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m graphs.benchmarks.microbench.gemm --device cpu
  python -m graphs.benchmarks.microbench.gemm --device cuda --precision fp16
  python -m graphs.benchmarks.microbench.gemm --size 1024 1024 1024
  python -m graphs.benchmarks.microbench.gemm --all-sizes --all-precisions
  python -m graphs.benchmarks.microbench.gemm --tags transformer --output results.json
        """
    )

    # Device options
    parser.add_argument(
        "--device", "-d",
        default="auto",
        help="Device to run on (cpu, cuda, cuda:0, auto)"
    )

    # Precision options
    parser.add_argument(
        "--precision", "-p",
        default=None,
        choices=["fp32", "fp16", "bf16", "int8"],
        help="Precision to use"
    )
    parser.add_argument(
        "--all-precisions",
        action="store_true",
        help="Run all precisions defined in each spec"
    )

    # Size filtering
    parser.add_argument(
        "--size",
        type=int,
        nargs=3,
        metavar=("M", "N", "K"),
        help="Run specific MxNxK size"
    )
    parser.add_argument(
        "--all-sizes",
        action="store_true",
        help="Run all predefined sizes"
    )
    parser.add_argument(
        "--filter-size",
        type=int,
        nargs="+",
        help="Only run specs with M dimension in this list"
    )

    # Tag filtering
    parser.add_argument(
        "--tags",
        nargs="+",
        help="Only run specs with these tags"
    )

    # Batched
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size for custom size runs"
    )
    parser.add_argument(
        "--include-batched",
        action="store_true",
        help="Include batched GEMM specs"
    )

    # Execution config
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Number of warmup iterations"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of measurement iterations"
    )

    # Output
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output file (json or csv)"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON (to stdout if no --output)"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output"
    )

    args = parser.parse_args()

    # Resolve device
    device = args.device
    if device == "auto":
        device = resolve_device(DeviceType.AUTO)

    # Determine precision
    precision = None
    if args.precision:
        precision = Precision(args.precision)

    # Load or create specs
    specs: List[GEMMSpec] = []

    if args.size:
        # Custom size
        M, N, K = args.size
        batch = args.batch_size if args.batch_size else 1
        p = precision if precision else Precision.FP32
        specs = [create_gemm_spec(M, N, K, batch_size=batch, precision=p)]
    else:
        # Load from YAML files
        specs = get_gemm_specs(
            filter_tags=args.tags,
            filter_sizes=args.filter_size,
        )

        # Filter out batched if not requested
        if not args.include_batched:
            specs = [s for s in specs if s.batch_size == 1]

    if not specs:
        print("No benchmark specs found matching filters", file=sys.stderr)
        sys.exit(1)

    # Create benchmark runner
    config = ExecutionConfig(
        warmup_iterations=args.warmup,
        measurement_iterations=args.iterations,
        sync_before_timing=True,
    )
    benchmark = GEMMBenchmark(config=config)

    # Run benchmarks
    results: List[BenchmarkResult] = []

    for i, spec in enumerate(specs):
        if not args.quiet:
            print(f"[{i+1}/{len(specs)}] Running {spec.name}...", end=" ", flush=True)

        try:
            if args.all_precisions:
                spec_results = benchmark.run_all_precisions(spec, device)
                results.extend(spec_results)
            else:
                p = precision if precision else spec.precisions[0]
                result = benchmark.run(spec, device, p)
                results.append(result)

            if not args.quiet:
                last_result = results[-1]
                if last_result.success:
                    print(f"{last_result.timing.mean_ms:.3f} ms, {last_result.gflops:.1f} GFLOPS")
                else:
                    print(f"FAILED: {last_result.error_message}")

        except Exception as e:
            if not args.quiet:
                print(f"ERROR: {e}")

    # Output results
    if args.json or (args.output and args.output.endswith('.json')):
        output_data = [r.to_dict() for r in results]
        json_str = json.dumps(output_data, indent=2)

        if args.output:
            with open(args.output, 'w') as f:
                f.write(json_str)
            if not args.quiet:
                print(f"\nResults written to {args.output}")
        else:
            print(json_str)

    elif args.output and args.output.endswith('.csv'):
        import csv

        with open(args.output, 'w', newline='') as f:
            if results:
                writer = csv.DictWriter(f, fieldnames=results[0].to_dict().keys())
                writer.writeheader()
                for r in results:
                    writer.writerow(r.to_dict())

        if not args.quiet:
            print(f"\nResults written to {args.output}")

    else:
        # Print table
        print()
        print(format_result_table(results))
        print()

        # Summary
        successful = [r for r in results if r.success]
        if successful:
            total_gflops = sum(r.gflops for r in successful)
            avg_gflops = total_gflops / len(successful)
            print(f"Summary: {len(successful)}/{len(results)} succeeded, avg {avg_gflops:.1f} GFLOPS")


if __name__ == "__main__":
    main()
