"""
TorchInductor Validation and Baseline Analysis

This module demonstrates how to use TorchInductor as a validation baseline
for the graphs package characterization pipeline.

Key use cases:
1. Baseline performance measurement (what users actually get)
2. Validation: Compare predicted latency vs actual inductor performance
3. Fusion analysis: See what inductor fuses vs what you predict
4. Optimization gap: Identify differences between analysis and reality

The workflow:
1. Extract graph with custom backend (for analysis)
2. Compile with inductor (for baseline performance)
3. Benchmark both
4. Compare predictions vs reality

Usage:
    python experiments/dynamo/inductor_validation.py --model resnet18
    python experiments/dynamo/inductor_validation.py --model resnet18 --benchmark
"""

import torch
import torch._dynamo as dynamo
import time
import argparse
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
import json


# ============================================================================
# Dual Backend: Extract Graph + Get Inductor Performance
# ============================================================================

class GraphExtractorWithMetrics:
    """
    Enhanced extractor that captures both:
    1. Graph structure (for analysis)
    2. Execution metrics (for validation)
    """

    def __init__(self):
        self.graphs: List[torch.fx.Graph] = []
        self.graph_modules: List[torch.fx.GraphModule] = []
        self.subgraph_count = 0

    def __call__(self, gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
        self.subgraph_count += 1
        self.graphs.append(gm.graph)
        self.graph_modules.append(gm)

        print(f"\n  Graph partition {self.subgraph_count}:")
        print(f"    Nodes: {len(list(gm.graph.nodes))}")

        # Count operations
        ops = sum(1 for n in gm.graph.nodes if n.op == 'call_function')
        print(f"    Operations: {ops}")

        return gm.forward


@dataclass
class BenchmarkResult:
    """Results from benchmarking a model."""
    name: str
    mean_time_ms: float
    std_time_ms: float
    min_time_ms: float
    max_time_ms: float
    num_runs: int
    throughput_imgs_per_sec: Optional[float] = None


@dataclass
class ValidationReport:
    """Complete validation report comparing analysis vs reality."""
    model_name: str

    # Performance metrics
    eager_time_ms: float
    inductor_time_ms: float
    speedup: float

    # Graph structure
    num_partitions: int
    total_nodes: int
    total_operations: int

    # Analysis predictions (to be filled from graphs package)
    predicted_latency_ms: Optional[float] = None
    predicted_energy_j: Optional[float] = None

    # Validation metrics
    latency_error_percent: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON export."""
        return {
            'model_name': self.model_name,
            'performance': {
                'eager_time_ms': self.eager_time_ms,
                'inductor_time_ms': self.inductor_time_ms,
                'speedup': self.speedup,
            },
            'graph_structure': {
                'num_partitions': self.num_partitions,
                'total_nodes': self.total_nodes,
                'total_operations': self.total_operations,
            },
            'predictions': {
                'predicted_latency_ms': self.predicted_latency_ms,
                'predicted_energy_j': self.predicted_energy_j,
            },
            'validation': {
                'latency_error_percent': self.latency_error_percent,
            }
        }


# ============================================================================
# Benchmarking Utilities
# ============================================================================

def benchmark_model(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    num_warmup: int = 10,
    num_runs: int = 100,
    name: str = "model"
) -> BenchmarkResult:
    """
    Benchmark a model with proper warmup.

    Args:
        model: Model to benchmark
        input_tensor: Input tensor
        num_warmup: Number of warmup runs
        num_runs: Number of measurement runs
        name: Name for the benchmark

    Returns:
        BenchmarkResult with timing statistics
    """
    model.eval()

    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(input_tensor)

    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = model(input_tensor)
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms

    times_tensor = torch.tensor(times)

    batch_size = input_tensor.shape[0]
    throughput = (batch_size * num_runs) / (sum(times) / 1000)

    return BenchmarkResult(
        name=name,
        mean_time_ms=float(times_tensor.mean()),
        std_time_ms=float(times_tensor.std()),
        min_time_ms=float(times_tensor.min()),
        max_time_ms=float(times_tensor.max()),
        num_runs=num_runs,
        throughput_imgs_per_sec=throughput
    )


# ============================================================================
# Validation Workflow
# ============================================================================

def validate_model_with_inductor(
    model: torch.nn.Module,
    example_input: torch.Tensor,
    model_name: str = "model",
    benchmark: bool = True,
    verbose: bool = True
) -> ValidationReport:
    """
    Complete validation workflow:
    1. Extract graph for analysis
    2. Benchmark eager mode (baseline)
    3. Benchmark inductor (optimized baseline)
    4. Compare results

    Args:
        model: PyTorch model to validate
        example_input: Example input tensor
        model_name: Name of the model
        benchmark: Whether to run performance benchmarks
        verbose: Print detailed information

    Returns:
        ValidationReport with all metrics
    """
    if verbose:
        print("\n" + "="*80)
        print(f"VALIDATION: {model_name}")
        print("="*80)

    # Step 1: Extract graph structure (for analysis)
    if verbose:
        print("\n1. Extracting graph structure...")

    dynamo.reset()
    extractor = GraphExtractorWithMetrics()

    compiled_extract = torch.compile(
        model,
        backend=extractor,
        fullgraph=False
    )

    with torch.no_grad():
        _ = compiled_extract(example_input)

    total_nodes = sum(len(list(g.nodes)) for g in extractor.graphs)
    total_ops = sum(
        sum(1 for n in g.nodes if n.op == 'call_function')
        for g in extractor.graphs
    )

    if verbose:
        print(f"   ✓ Captured {extractor.subgraph_count} partition(s)")
        print(f"   ✓ Total nodes: {total_nodes}")
        print(f"   ✓ Total operations: {total_ops}")

    # Step 2: Benchmark eager mode (baseline)
    eager_result = None
    if benchmark:
        if verbose:
            print("\n2. Benchmarking eager mode (baseline)...")

        # Use the original model directly (no copy needed)
        eager_result = benchmark_model(
            model,
            example_input,
            name="eager"
        )

        if verbose:
            print(f"   ✓ Mean time: {eager_result.mean_time_ms:.2f} ± {eager_result.std_time_ms:.2f} ms")
            print(f"   ✓ Throughput: {eager_result.throughput_imgs_per_sec:.1f} img/s")

    # Step 3: Benchmark inductor (optimized)
    inductor_result = None
    if benchmark:
        if verbose:
            print("\n3. Compiling with inductor...")

        dynamo.reset()

        # Compile the model with inductor
        # Note: torch.compile creates its own compiled version, no need for copy
        compiled_inductor = torch.compile(
            model,
            backend="inductor",
            mode="default"
        )

        if verbose:
            print("   ✓ Compiled with inductor")
            print("\n4. Benchmarking inductor (optimized)...")

        inductor_result = benchmark_model(
            compiled_inductor,
            example_input,
            name="inductor"
        )

        if verbose:
            print(f"   ✓ Mean time: {inductor_result.mean_time_ms:.2f} ± {inductor_result.std_time_ms:.2f} ms")
            print(f"   ✓ Throughput: {inductor_result.throughput_imgs_per_sec:.1f} img/s")

    # Step 4: Create validation report
    speedup = 1.0
    if eager_result and inductor_result:
        speedup = eager_result.mean_time_ms / inductor_result.mean_time_ms

        if verbose:
            print("\n5. Performance comparison:")
            print(f"   Eager:     {eager_result.mean_time_ms:.2f} ms")
            print(f"   Inductor:  {inductor_result.mean_time_ms:.2f} ms")
            print(f"   Speedup:   {speedup:.2f}x")

    report = ValidationReport(
        model_name=model_name,
        eager_time_ms=eager_result.mean_time_ms if eager_result else 0.0,
        inductor_time_ms=inductor_result.mean_time_ms if inductor_result else 0.0,
        speedup=speedup,
        num_partitions=extractor.subgraph_count,
        total_nodes=total_nodes,
        total_operations=total_ops
    )

    return report


# ============================================================================
# Fusion Analysis
# ============================================================================

def analyze_inductor_optimizations(
    model: torch.nn.Module,
    example_input: torch.Tensor,
    model_name: str = "model"
) -> Dict[str, Any]:
    """
    Analyze what optimizations inductor applies.

    This extracts both:
    1. Pre-optimization graph (custom backend)
    2. Post-optimization graph (after inductor)

    Comparing these shows what inductor fused/optimized.

    Args:
        model: Model to analyze
        example_input: Example input
        model_name: Model name

    Returns:
        Dictionary with optimization analysis
    """
    print("\n" + "="*80)
    print(f"INDUCTOR OPTIMIZATION ANALYSIS: {model_name}")
    print("="*80)

    # Get pre-optimization graph
    print("\n1. Extracting pre-optimization graph...")
    dynamo.reset()

    extractor = GraphExtractorWithMetrics()
    compiled_extract = torch.compile(model, backend=extractor)

    with torch.no_grad():
        _ = compiled_extract(example_input)

    pre_ops = sum(
        sum(1 for n in g.nodes if n.op == 'call_function')
        for g in extractor.graphs
    )

    print(f"   Pre-optimization operations: {pre_ops}")

    # Note: Getting post-optimization graph from inductor is complex
    # because inductor generates code. For now, we rely on performance
    # metrics to infer fusion effectiveness.

    analysis = {
        'model_name': model_name,
        'pre_optimization': {
            'partitions': extractor.subgraph_count,
            'operations': pre_ops,
        },
        'note': 'Post-optimization graph requires inductor internals inspection'
    }

    return analysis


# ============================================================================
# Integration with graphs Package
# ============================================================================

def compare_with_graphs_analysis(
    validation_report: ValidationReport,
    predicted_latency_ms: float,
    predicted_energy_j: Optional[float] = None
) -> ValidationReport:
    """
    Add graphs package predictions to validation report for comparison.

    Args:
        validation_report: Validation report from inductor
        predicted_latency_ms: Predicted latency from graphs.analysis
        predicted_energy_j: Predicted energy from graphs.analysis

    Returns:
        Updated validation report with error metrics
    """
    validation_report.predicted_latency_ms = predicted_latency_ms
    validation_report.predicted_energy_j = predicted_energy_j

    # Calculate error (compare to inductor as ground truth)
    if validation_report.inductor_time_ms > 0:
        error_percent = abs(
            predicted_latency_ms - validation_report.inductor_time_ms
        ) / validation_report.inductor_time_ms * 100
        validation_report.latency_error_percent = error_percent

    return validation_report


# ============================================================================
# Example Models
# ============================================================================

class SimpleModel(torch.nn.Module):
    """Simple CNN for testing."""
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 32, 3, padding=1)
        self.fc = torch.nn.Linear(32 * 56 * 56, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.nn.functional.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.nn.functional.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def get_resnet18():
    """Get ResNet18 model."""
    from torchvision.models import resnet18
    return resnet18(weights=None)


# ============================================================================
# Examples
# ============================================================================

def example_simple_validation():
    """Example: Validate simple model."""
    print("\n" + "="*80)
    print("EXAMPLE 1: Simple Model Validation")
    print("="*80)

    model = SimpleModel()
    example_input = torch.randn(1, 3, 224, 224)

    report = validate_model_with_inductor(
        model,
        example_input,
        model_name="SimpleCNN",
        benchmark=True
    )

    # Simulate graphs package prediction (would come from actual analysis)
    # For demo, predict 90% of inductor time
    predicted_latency = report.inductor_time_ms * 0.9

    report = compare_with_graphs_analysis(
        report,
        predicted_latency_ms=predicted_latency,
        predicted_energy_j=None
    )

    print("\n" + "="*80)
    print("VALIDATION RESULTS")
    print("="*80)
    print(f"Model: {report.model_name}")
    print(f"\nActual (inductor): {report.inductor_time_ms:.2f} ms")
    print(f"Predicted (graphs): {report.predicted_latency_ms:.2f} ms")
    print(f"Error: {report.latency_error_percent:.1f}%")

    return report


def example_resnet_validation():
    """Example: Validate ResNet18."""
    print("\n" + "="*80)
    print("EXAMPLE 2: ResNet18 Validation")
    print("="*80)

    model = get_resnet18()
    example_input = torch.randn(1, 3, 224, 224)

    report = validate_model_with_inductor(
        model,
        example_input,
        model_name="ResNet18",
        benchmark=True
    )

    return report


def example_batch_validation():
    """Example: Validate across different batch sizes."""
    print("\n" + "="*80)
    print("EXAMPLE 3: Batch Size Validation")
    print("="*80)

    model = SimpleModel()
    batch_sizes = [1, 4, 8, 16]

    results = []
    for bs in batch_sizes:
        print(f"\n{'='*40}")
        print(f"Batch Size: {bs}")
        print(f"{'='*40}")

        example_input = torch.randn(bs, 3, 224, 224)

        report = validate_model_with_inductor(
            model,
            example_input,
            model_name=f"SimpleCNN_bs{bs}",
            benchmark=True,
            verbose=False
        )

        results.append(report)

        print(f"Eager:    {report.eager_time_ms:.2f} ms")
        print(f"Inductor: {report.inductor_time_ms:.2f} ms")
        print(f"Speedup:  {report.speedup:.2f}x")

    # Summary table
    print("\n" + "="*80)
    print("BATCH SIZE SUMMARY")
    print("="*80)
    print(f"{'Batch':<8} {'Eager (ms)':<12} {'Inductor (ms)':<15} {'Speedup':<10}")
    print("-"*80)
    for report in results:
        bs = int(report.model_name.split('_bs')[1])
        print(f"{bs:<8} {report.eager_time_ms:<12.2f} "
              f"{report.inductor_time_ms:<15.2f} {report.speedup:<10.2f}x")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Validate graphs package predictions against inductor baseline"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["simple", "resnet18", "batch"],
        default="simple",
        help="Model to validate"
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        default=True,
        help="Run performance benchmarks"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save report to JSON file"
    )

    args = parser.parse_args()

    # Run validation
    if args.model == "simple":
        report = example_simple_validation()
    elif args.model == "resnet18":
        report = example_resnet_validation()
    elif args.model == "batch":
        example_batch_validation()
        return

    # Save report
    if args.output and report:
        with open(args.output, 'w') as f:
            json.dump(report.to_dict(), f, indent=2)
        print(f"\n✓ Report saved to: {args.output}")


if __name__ == "__main__":
    main()
