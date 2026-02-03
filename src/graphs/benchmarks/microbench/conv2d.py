"""
Conv2d Microbenchmark Suite

Comprehensive Conv2d benchmarks for measuring hardware performance
across different kernel sizes, channel counts, and spatial dimensions.

Usage:
    # As a module
    python -m graphs.benchmarks.microbench.conv2d --device cpu
    python -m graphs.benchmarks.microbench.conv2d --device cuda --precision fp16
    python -m graphs.benchmarks.microbench.conv2d --tags depthwise

    # Programmatic usage
    from graphs.benchmarks.microbench.conv2d import run_conv2d_benchmark, get_conv2d_specs
    specs = get_conv2d_specs()
    result = run_conv2d_benchmark(specs[0], device="cuda")
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from ..schema import (
    BenchmarkResult,
    Conv2dSpec,
    ExecutionConfig,
    Precision,
    DeviceType,
)
from ..runner import PyTorchRunner, resolve_device


# Package location for finding spec files
SPECS_DIR = Path(__file__).parent.parent / "specs" / "conv2d"


@dataclass
class Conv2dBenchmark:
    """
    Conv2d benchmark executor.

    Wraps the PyTorchRunner with Conv2d-specific functionality.
    """

    config: ExecutionConfig
    runner: PyTorchRunner = None

    def __post_init__(self):
        if self.runner is None:
            self.runner = PyTorchRunner(self.config)

    def run(
        self,
        spec: Conv2dSpec,
        device: str = "auto",
        precision: Precision = Precision.FP32,
    ) -> BenchmarkResult:
        """Run a single Conv2d benchmark."""
        return self.runner.run(spec, device, precision)

    def run_all_precisions(
        self,
        spec: Conv2dSpec,
        device: str = "auto",
    ) -> List[BenchmarkResult]:
        """Run benchmark for all precisions in spec."""
        return self.runner.run_all_precisions(spec, device)

    def run_suite(
        self,
        specs: List[Conv2dSpec],
        device: str = "auto",
        precision: Optional[Precision] = None,
    ) -> List[BenchmarkResult]:
        """
        Run a suite of Conv2d benchmarks.

        Args:
            specs: List of Conv2d specifications to run
            device: Target device
            precision: If specified, override spec precision.

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


def get_conv2d_specs(
    spec_dir: Optional[Path] = None,
    filter_tags: Optional[List[str]] = None,
    include_batched: bool = True,
    include_depthwise: bool = True,
) -> List[Conv2dSpec]:
    """
    Load Conv2d benchmark specifications from YAML files.

    Args:
        spec_dir: Directory containing YAML spec files. Defaults to package specs.
        filter_tags: Only include specs with these tags
        include_batched: Include batched specs (batch_size > 1)
        include_depthwise: Include depthwise specs (groups > 1)

    Returns:
        List of Conv2dSpec objects
    """
    import yaml

    if spec_dir is None:
        spec_dir = SPECS_DIR

    specs = []

    for yaml_file in sorted(spec_dir.glob("*.yaml")):
        with open(yaml_file, 'r') as f:
            content = f.read()

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
                dilation=doc.get('dilation', 1),
                groups=doc.get('groups', 1),
                bias=doc.get('bias', False),
                precisions=precisions,
                devices=devices,
                execution=execution,
            )

            # Apply filters
            if filter_tags:
                if not any(tag in spec.tags for tag in filter_tags):
                    continue

            if not include_batched and spec.batch_size > 1:
                continue

            if not include_depthwise and spec.groups > 1:
                continue

            specs.append(spec)

    return specs


def run_conv2d_benchmark(
    spec: Conv2dSpec,
    device: str = "auto",
    precision: Precision = Precision.FP32,
    config: Optional[ExecutionConfig] = None,
) -> BenchmarkResult:
    """
    Convenience function to run a single Conv2d benchmark.

    Args:
        spec: Conv2d specification
        device: Target device
        precision: Numerical precision
        config: Optional execution config override

    Returns:
        BenchmarkResult with timing and performance metrics
    """
    cfg = config if config is not None else spec.execution
    benchmark = Conv2dBenchmark(config=cfg)
    return benchmark.run(spec, device, precision)


def create_conv2d_spec(
    in_channels: int,
    out_channels: int,
    height: int,
    width: int,
    kernel_size: int = 3,
    stride: int = 1,
    padding: int = 1,
    groups: int = 1,
    batch_size: int = 1,
    name: Optional[str] = None,
    precision: Precision = Precision.FP32,
) -> Conv2dSpec:
    """
    Create a Conv2d spec programmatically.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        height, width: Input spatial dimensions
        kernel_size: Convolution kernel size
        stride: Convolution stride
        padding: Padding amount
        groups: Number of groups (>1 for grouped/depthwise conv)
        batch_size: Batch size
        name: Spec name (auto-generated if None)
        precision: Numerical precision

    Returns:
        Conv2dSpec ready for benchmarking
    """
    if name is None:
        parts = [f"conv2d_{kernel_size}x{kernel_size}"]
        if groups == in_channels:
            parts[0] = f"conv2d_dw_{kernel_size}x{kernel_size}"
        parts.append(f"{in_channels}_{height}x{width}")
        if batch_size > 1:
            parts.append(f"batch{batch_size}")
        if stride > 1:
            parts.append(f"s{stride}")
        name = "_".join(parts)

    return Conv2dSpec(
        name=name,
        in_channels=in_channels,
        out_channels=out_channels,
        height=height,
        width=width,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        groups=groups,
        batch_size=batch_size,
        precisions=[precision],
    )


def format_result_table(results: List[BenchmarkResult]) -> str:
    """Format results as a text table."""
    lines = []

    # Header
    lines.append(
        f"{'Name':<45} {'Device':<8} {'Prec':<6} "
        f"{'Mean (ms)':<12} {'Std (ms)':<10} {'GFLOPS':<12} {'Status'}"
    )
    lines.append("-" * 110)

    for r in results:
        if r.success and r.timing is not None:
            lines.append(
                f"{r.spec_name:<45} {r.device:<8} {r.precision:<6} "
                f"{r.timing.mean_ms:<12.3f} {r.timing.std_ms:<10.3f} "
                f"{r.gflops:<12.1f} OK"
            )
        else:
            lines.append(
                f"{r.spec_name:<45} {r.device:<8} {r.precision:<6} "
                f"{'N/A':<12} {'N/A':<10} {'N/A':<12} FAILED: {r.error_message[:20]}"
            )

    return "\n".join(lines)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Conv2d Microbenchmark Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m graphs.benchmarks.microbench.conv2d --device cpu
  python -m graphs.benchmarks.microbench.conv2d --device cuda --precision fp16
  python -m graphs.benchmarks.microbench.conv2d --tags depthwise
  python -m graphs.benchmarks.microbench.conv2d --no-depthwise --no-batched
  python -m graphs.benchmarks.microbench.conv2d --output results.json
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
        help="Precision to use (int8 uses quantized conv)"
    )
    parser.add_argument(
        "--all-precisions",
        action="store_true",
        help="Run all precisions defined in each spec"
    )

    # Filtering
    parser.add_argument(
        "--tags",
        nargs="+",
        help="Only run specs with these tags"
    )
    parser.add_argument(
        "--no-batched",
        action="store_true",
        help="Exclude batched benchmarks"
    )
    parser.add_argument(
        "--no-depthwise",
        action="store_true",
        help="Exclude depthwise benchmarks"
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

    # Load specs
    specs = get_conv2d_specs(
        filter_tags=args.tags,
        include_batched=not args.no_batched,
        include_depthwise=not args.no_depthwise,
    )

    if not specs:
        print("No benchmark specs found matching filters", file=sys.stderr)
        sys.exit(1)

    # Create benchmark runner
    config = ExecutionConfig(
        warmup_iterations=args.warmup,
        measurement_iterations=args.iterations,
        sync_before_timing=True,
    )
    benchmark = Conv2dBenchmark(config=config)

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
                if last_result.success and last_result.timing:
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
