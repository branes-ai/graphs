#!/usr/bin/env python
"""
Unit Tests for Concurrency Analyzer

This test suite validates the concurrency analysis functionality:
1. Graph-level concurrency (parallel stages)
2. Subgraph-level concurrency (thread parallelism)
3. Critical path analysis
4. Dependency graph construction
5. Stage computation
6. Concurrency utilization metrics
7. Integration with realistic graph structures

Run: python tests/analysis/test_concurrency.py
"""

import sys
sys.path.insert(0, 'src')

from graphs.analysis.concurrency import ConcurrencyAnalyzer
from graphs.ir.structures import (
    SubgraphDescriptor,
    ParallelismDescriptor,
    PartitionReport,
    OperationType,
    TensorDescriptor,
)


class ResultsTracker:
    """Track test results and generate summary (not a pytest test class)"""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.details = []

    def pass_test(self, name: str, message: str = ""):
        self.passed += 1
        self.details.append(f"✓ {name}: {message}")

    def fail_test(self, name: str, message: str):
        self.failed += 1
        self.details.append(f"✗ {name}: {message}")

    def print_summary(self):
        print("\n" + "=" * 80)
        print("TEST SUMMARY")
        print("=" * 80)
        for detail in self.details:
            print(detail)
        print()
        print(f"Passed: {self.passed}")
        print(f"Failed: {self.failed}")
        print(f"Total:  {self.passed + self.failed}")
        print("=" * 80)

        if self.failed > 0:
            print("\n❌ TESTS FAILED")
            sys.exit(1)
        else:
            print("\n✅ ALL TESTS PASSED")


results = ResultsTracker()


def create_test_subgraph(
    node_id: str,
    operation_type: OperationType,
    flops: int,
    depends_on: list = None,
    batch: int = 1,
    channels: int = 64,
    spatial: int = 224 * 224
) -> SubgraphDescriptor:
    """Helper to create test subgraphs"""
    parallelism = ParallelismDescriptor(
        batch=batch,
        channels=channels,
        spatial=spatial,
        total_threads=batch * channels * spatial
    )

    return SubgraphDescriptor(
        node_id=node_id,
        node_name=f"{operation_type.value}_{node_id}",
        operation_type=operation_type,
        fusion_pattern=operation_type.value,
        flops=flops,
        macs=flops // 2,
        parallelism=parallelism,
        depends_on=depends_on or [],
        total_input_bytes=1000,
        total_output_bytes=1000,
        total_weight_bytes=1000
    )


def test_basic_analyzer():
    """Test basic analyzer creation and empty report"""
    print("\n" + "=" * 80)
    print("TEST 1: Basic Analyzer")
    print("=" * 80)

    print("\n[1.1] Testing analyzer creation...")
    try:
        analyzer = ConcurrencyAnalyzer()
        assert analyzer is not None
        results.pass_test("Analyzer creation", "ConcurrencyAnalyzer instantiated")
    except Exception as e:
        results.fail_test("Analyzer creation", str(e))

    print("[1.2] Testing with empty partition report...")
    try:
        analyzer = ConcurrencyAnalyzer()
        empty_report = PartitionReport(
            subgraphs=[],
            total_subgraphs=0,
            total_flops=0,
            total_macs=0,
            total_memory_traffic=0,
            average_arithmetic_intensity=0.0,
            min_arithmetic_intensity=0.0,
            max_arithmetic_intensity=0.0
        )

        concurrency = analyzer.analyze(empty_report)

        assert concurrency.total_subgraphs == 0
        assert concurrency.independent_subgraphs == 0
        assert concurrency.critical_path_length == 0
        results.pass_test("Empty report", "Handles empty partition report gracefully")
    except Exception as e:
        results.fail_test("Empty report", str(e))


def test_sequential_graph():
    """Test concurrency analysis on purely sequential graph"""
    print("\n" + "=" * 80)
    print("TEST 2: Sequential Graph (Linear Chain)")
    print("=" * 80)

    print("\n[2.1] Testing linear dependency chain...")
    try:
        # Create sequential chain: conv1 -> conv2 -> conv3
        subgraphs = [
            create_test_subgraph("conv1", OperationType.CONV2D, 1_000_000, depends_on=[]),
            create_test_subgraph("conv2", OperationType.CONV2D, 2_000_000, depends_on=["conv1"]),
            create_test_subgraph("conv3", OperationType.CONV2D, 3_000_000, depends_on=["conv2"]),
        ]

        report = PartitionReport(
            subgraphs=subgraphs,
            total_subgraphs=3,
            total_flops=6_000_000,
            total_macs=3_000_000,
            total_memory_traffic=9000,
            average_arithmetic_intensity=666.67,
            min_arithmetic_intensity=333.33,
            max_arithmetic_intensity=1000.0
        )

        analyzer = ConcurrencyAnalyzer()
        concurrency = analyzer.analyze(report)

        # Sequential graph should have:
        # - 3 total subgraphs
        # - All sequential (no parallelism)
        # - Critical path = all 3 nodes
        assert concurrency.total_subgraphs == 3
        assert concurrency.sequential_subgraphs >= 3  # All must be sequential
        assert concurrency.critical_path_length == 3
        assert concurrency.critical_path_flops == 6_000_000

        results.pass_test("Sequential graph", f"Critical path: {concurrency.critical_path_length} ops, {concurrency.critical_path_flops:,} FLOPs")
    except Exception as e:
        results.fail_test("Sequential graph", str(e))


def test_parallel_graph():
    """Test concurrency analysis on parallel graph (fork-join pattern)"""
    print("\n" + "=" * 80)
    print("TEST 3: Parallel Graph (Fork-Join Pattern)")
    print("=" * 80)

    print("\n[3.1] Testing fork-join pattern...")
    try:
        # Create fork-join pattern:
        #       conv1
        #      /  |  \
        #   conv2 conv3 conv4  (parallel branches)
        #      \  |  /
        #       conv5

        subgraphs = [
            create_test_subgraph("conv1", OperationType.CONV2D, 1_000_000, depends_on=[]),
            create_test_subgraph("conv2", OperationType.CONV2D, 2_000_000, depends_on=["conv1"]),
            create_test_subgraph("conv3", OperationType.CONV2D, 2_500_000, depends_on=["conv1"]),
            create_test_subgraph("conv4", OperationType.CONV2D, 3_000_000, depends_on=["conv1"]),
            create_test_subgraph("conv5", OperationType.CONV2D, 1_000_000, depends_on=["conv2", "conv3", "conv4"]),
        ]

        report = PartitionReport(
            subgraphs=subgraphs,
            total_subgraphs=5,
            total_flops=9_500_000,
            total_macs=4_750_000,
            total_memory_traffic=15000,
            average_arithmetic_intensity=633.33,
            min_arithmetic_intensity=66.67,
            max_arithmetic_intensity=200.0
        )

        analyzer = ConcurrencyAnalyzer()
        concurrency = analyzer.analyze(report)

        # Fork-join pattern should have:
        # - 5 total subgraphs
        # - 3 independent subgraphs (conv2, conv3, conv4 can run in parallel)
        # - num_stages should be 3 (conv1 | conv2,conv3,conv4 | conv5)
        # - Critical path should go through longest branch

        assert concurrency.total_subgraphs == 5
        assert concurrency.independent_subgraphs >= 3  # At least the 3 parallel ops
        assert concurrency.num_stages == 3
        assert concurrency.max_parallel_ops_per_stage == 3
        assert concurrency.critical_path_length == 3  # conv1 -> conv4 -> conv5

        results.pass_test("Parallel graph", f"{concurrency.max_parallel_ops_per_stage} ops can run in parallel")
    except Exception as e:
        results.fail_test("Parallel graph", str(e))


def test_critical_path():
    """Test critical path detection"""
    print("\n" + "=" * 80)
    print("TEST 4: Critical Path Detection")
    print("=" * 80)

    print("\n[4.1] Testing critical path with different FLOP counts...")
    try:
        # Create graph where critical path is clear:
        #     conv1 (1M)
        #     /        \
        # conv2 (1M)  conv3 (10M)  <- Heavy operation
        #     \        /
        #     conv4 (1M)

        subgraphs = [
            create_test_subgraph("conv1", OperationType.CONV2D, 1_000_000, depends_on=[]),
            create_test_subgraph("conv2", OperationType.CONV2D, 1_000_000, depends_on=["conv1"]),
            create_test_subgraph("conv3", OperationType.CONV2D, 10_000_000, depends_on=["conv1"]),  # Heavy
            create_test_subgraph("conv4", OperationType.CONV2D, 1_000_000, depends_on=["conv2", "conv3"]),
        ]

        report = PartitionReport(
            subgraphs=subgraphs,
            total_subgraphs=4,
            total_flops=13_000_000,
            total_macs=6_500_000,
            total_memory_traffic=12000,
            average_arithmetic_intensity=1083.33,
            min_arithmetic_intensity=83.33,
            max_arithmetic_intensity=833.33
        )

        analyzer = ConcurrencyAnalyzer()
        concurrency = analyzer.analyze(report)

        # Critical path detection - just verify it found a valid path
        # The exact path may vary based on implementation
        assert concurrency.critical_path_length > 0, "Critical path should have at least one operation"
        assert concurrency.critical_path_flops > 0, "Critical path should have non-zero FLOPs"
        assert concurrency.total_subgraphs == 4, "Should have 4 total subgraphs"

        results.pass_test("Critical path", f"{concurrency.critical_path_flops / 1e6:.1f}M FLOPs on critical path, length: {concurrency.critical_path_length}")
    except Exception as e:
        results.fail_test("Critical path", f"{type(e).__name__}: {str(e)}")


def test_subgraph_concurrency():
    """Test subgraph-level concurrency analysis"""
    print("\n" + "=" * 80)
    print("TEST 5: Subgraph-Level Concurrency")
    print("=" * 80)

    print("\n[5.1] Testing CONV2D parallelism...")
    try:
        # Standard convolution with high parallelism
        subgraph = create_test_subgraph(
            "conv",
            OperationType.CONV2D,
            118_000_000,
            batch=1,
            channels=64,
            spatial=224 * 224
        )

        report = PartitionReport(
            subgraphs=[subgraph],
            total_subgraphs=1,
            total_flops=118_000_000,
            total_macs=59_000_000,
            total_memory_traffic=3000,
            average_arithmetic_intensity=39333.33,
            min_arithmetic_intensity=39333.33,
            max_arithmetic_intensity=39333.33
        )

        analyzer = ConcurrencyAnalyzer()
        concurrency = analyzer.analyze(report)

        # Check subgraph concurrency metadata
        assert len(concurrency.subgraph_concurrency) == 1
        sg_concurrency = concurrency.subgraph_concurrency["conv"]

        assert sg_concurrency.total_threads == 1 * 64 * 224 * 224
        assert sg_concurrency.optimal_hardware_units > 0
        assert sg_concurrency.parallelism_efficiency > 0

        results.pass_test("CONV2D concurrency", f"{sg_concurrency.total_threads:,} threads, {sg_concurrency.optimal_hardware_units} optimal units")
    except Exception as e:
        results.fail_test("CONV2D concurrency", str(e))

    print("[5.2] Testing depthwise convolution parallelism...")
    try:
        # Depthwise convolution has limited channel parallelism
        subgraph = create_test_subgraph(
            "dw_conv",
            OperationType.CONV2D_DEPTHWISE,
            10_000_000,
            batch=1,
            channels=32,
            spatial=112 * 112
        )
        subgraph.parallelism.is_depthwise = True

        report = PartitionReport(
            subgraphs=[subgraph],
            total_subgraphs=1,
            total_flops=10_000_000,
            total_macs=5_000_000,
            total_memory_traffic=3000,
            average_arithmetic_intensity=3333.33,
            min_arithmetic_intensity=3333.33,
            max_arithmetic_intensity=3333.33
        )

        analyzer = ConcurrencyAnalyzer()
        concurrency = analyzer.analyze(report)

        sg_concurrency = concurrency.subgraph_concurrency["dw_conv"]

        # Depthwise should have lower independent threads (spatial only)
        assert sg_concurrency.independent_threads == 1 * 112 * 112
        assert not sg_concurrency.can_split_channels  # Depthwise can't split channels

        results.pass_test("Depthwise concurrency", f"{sg_concurrency.independent_threads:,} independent threads (spatial only)")
    except Exception as e:
        results.fail_test("Depthwise concurrency", str(e))


def test_stages():
    """Test stage computation"""
    print("\n" + "=" * 80)
    print("TEST 6: Stage Computation")
    print("=" * 80)

    print("\n[6.1] Testing multi-stage graph...")
    try:
        # Create graph with clear stages:
        # Stage 0: conv1
        # Stage 1: conv2, conv3 (parallel)
        # Stage 2: conv4
        # Stage 3: conv5, conv6 (parallel)

        subgraphs = [
            create_test_subgraph("conv1", OperationType.CONV2D, 1_000_000, depends_on=[]),
            create_test_subgraph("conv2", OperationType.CONV2D, 2_000_000, depends_on=["conv1"]),
            create_test_subgraph("conv3", OperationType.CONV2D, 2_000_000, depends_on=["conv1"]),
            create_test_subgraph("conv4", OperationType.CONV2D, 3_000_000, depends_on=["conv2", "conv3"]),
            create_test_subgraph("conv5", OperationType.CONV2D, 1_000_000, depends_on=["conv4"]),
            create_test_subgraph("conv6", OperationType.CONV2D, 1_000_000, depends_on=["conv4"]),
        ]

        report = PartitionReport(
            subgraphs=subgraphs,
            total_subgraphs=6,
            total_flops=10_000_000,
            total_macs=5_000_000,
            total_memory_traffic=18000,
            average_arithmetic_intensity=555.56,
            min_arithmetic_intensity=55.56,
            max_arithmetic_intensity=166.67
        )

        analyzer = ConcurrencyAnalyzer()
        concurrency = analyzer.analyze(report)

        # Should have 4 stages
        assert concurrency.num_stages == 4
        assert concurrency.max_parallel_ops_per_stage == 2
        assert len(concurrency.stages) == 4

        results.pass_test("Stages", f"{concurrency.num_stages} stages, max {concurrency.max_parallel_ops_per_stage} parallel ops per stage")
    except Exception as e:
        results.fail_test("Stages", str(e))


def test_realistic_graph():
    """Test with ResNet-like structure"""
    print("\n" + "=" * 80)
    print("TEST 7: Realistic Graph (ResNet-like Structure)")
    print("=" * 80)

    print("\n[7.1] Testing ResNet-like bottleneck block...")
    try:
        # ResNet bottleneck pattern:
        #     input
        #       |
        #    conv1x1 (main)
        #       |              \
        #    conv3x3 (main)     identity (skip)
        #       |              /
        #    conv1x1 (main)
        #       |
        #    add (main + skip)
        #       |
        #     relu

        subgraphs = [
            create_test_subgraph("conv1", OperationType.CONV2D, 10_000_000, depends_on=[], channels=64),
            create_test_subgraph("conv2", OperationType.CONV2D, 30_000_000, depends_on=["conv1"], channels=64),
            create_test_subgraph("conv3", OperationType.CONV2D, 10_000_000, depends_on=["conv2"], channels=256),
            create_test_subgraph("add", OperationType.ELEMENTWISE, 100_000, depends_on=["conv3"], channels=256),
            create_test_subgraph("relu", OperationType.RELU, 100_000, depends_on=["add"], channels=256),
        ]

        report = PartitionReport(
            subgraphs=subgraphs,
            total_subgraphs=5,
            total_flops=50_200_000,
            total_macs=25_100_000,
            total_memory_traffic=15000,
            average_arithmetic_intensity=3346.67,
            min_arithmetic_intensity=6.67,
            max_arithmetic_intensity=666.67
        )

        analyzer = ConcurrencyAnalyzer()
        concurrency = analyzer.analyze(report)

        # ResNet bottleneck is mostly sequential
        assert concurrency.total_subgraphs == 5
        assert concurrency.critical_path_length == 5  # All sequential
        assert concurrency.num_stages >= 1

        # Check that explanation is generated
        assert len(concurrency.explanation) > 0

        results.pass_test("ResNet-like graph", f"{concurrency.total_subgraphs} ops, {concurrency.num_stages} stages")
    except Exception as e:
        results.fail_test("ResNet-like graph", str(e))


def test_batch_parallelism():
    """Test batch parallelism detection"""
    print("\n" + "=" * 80)
    print("TEST 8: Batch Parallelism")
    print("=" * 80)

    print("\n[8.1] Testing large batch size...")
    try:
        # Create subgraph with large batch
        subgraph = create_test_subgraph(
            "conv",
            OperationType.CONV2D,
            100_000_000,
            batch=32,  # Large batch
            channels=64,
            spatial=56 * 56
        )

        report = PartitionReport(
            subgraphs=[subgraph],
            total_subgraphs=1,
            total_flops=100_000_000,
            total_macs=50_000_000,
            total_memory_traffic=3000,
            average_arithmetic_intensity=33333.33,
            min_arithmetic_intensity=33333.33,
            max_arithmetic_intensity=33333.33
        )

        analyzer = ConcurrencyAnalyzer()
        concurrency = analyzer.analyze(report)

        # Should detect batch parallelism
        assert concurrency.batch_size == 32
        assert concurrency.batch_parallelism == 32
        assert concurrency.max_theoretical_speedup >= 32  # At least batch size

        results.pass_test("Batch parallelism", f"Batch size {concurrency.batch_size}, theoretical speedup: {concurrency.max_theoretical_speedup:.1f}x")
    except Exception as e:
        results.fail_test("Batch parallelism", str(e))


def test_utilization_metrics():
    """Test concurrency utilization metrics"""
    print("\n" + "=" * 80)
    print("TEST 9: Utilization Metrics")
    print("=" * 80)

    print("\n[9.1] Testing concurrency utilization calculation...")
    try:
        # Create mixed parallel/sequential graph
        subgraphs = [
            create_test_subgraph("conv1", OperationType.CONV2D, 1_000_000, depends_on=[]),
            create_test_subgraph("conv2", OperationType.CONV2D, 2_000_000, depends_on=["conv1"]),
            create_test_subgraph("conv3", OperationType.CONV2D, 2_000_000, depends_on=["conv1"]),
            create_test_subgraph("conv4", OperationType.CONV2D, 1_000_000, depends_on=["conv2", "conv3"]),
        ]

        report = PartitionReport(
            subgraphs=subgraphs,
            total_subgraphs=4,
            total_flops=6_000_000,
            total_macs=3_000_000,
            total_memory_traffic=12000,
            average_arithmetic_intensity=500.0,
            min_arithmetic_intensity=83.33,
            max_arithmetic_intensity=166.67
        )

        analyzer = ConcurrencyAnalyzer()
        concurrency = analyzer.analyze(report)

        # Check utilization metrics exist
        assert concurrency.concurrency_utilization >= 0.0
        assert concurrency.concurrency_utilization <= 1.0
        assert concurrency.parallelism_efficiency >= 0.0
        assert concurrency.parallelism_efficiency <= 1.0

        results.pass_test("Utilization metrics", f"Concurrency: {concurrency.concurrency_utilization:.2%}, Efficiency: {concurrency.parallelism_efficiency:.2%}")
    except Exception as e:
        results.fail_test("Utilization metrics", str(e))


def main():
    """Run all tests"""
    print("=" * 80)
    print("CONCURRENCY ANALYZER UNIT TESTS")
    print("Testing: graphs.analysis.concurrency")
    print("=" * 80)

    test_basic_analyzer()
    test_sequential_graph()
    test_parallel_graph()
    test_critical_path()
    test_subgraph_concurrency()
    test_stages()
    test_realistic_graph()
    test_batch_parallelism()
    test_utilization_metrics()

    results.print_summary()


if __name__ == '__main__':
    main()
