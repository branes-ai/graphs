#!/usr/bin/env python
"""
Unit Tests for IR Data Structures

This test suite validates the core intermediate representation data structures:
1. Enumerations (OperationType, BottleneckType, PartitionReason)
2. TensorDescriptor - Shape and memory information
3. ParallelismDescriptor - Parallelism dimensions and constraints
4. SubgraphDescriptor - Complete subgraph metadata
5. SubgraphConcurrency - Concurrency metadata
6. ConcurrencyDescriptor - Graph-level concurrency
7. PartitionReport - Complete partition statistics

Run: python tests/ir/test_structures.py
"""

import sys
sys.path.insert(0, 'src')

from graphs.core.structures import (
    OperationType,
    BottleneckType,
    PartitionReason,
    TensorDescriptor,
    ParallelismDescriptor,
    SubgraphDescriptor,
    SubgraphConcurrency,
    ConcurrencyDescriptor,
    PartitionReport,
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


def test_enumerations():
    """Test enumeration types"""
    print("\n" + "=" * 80)
    print("TEST 1: Enumerations")
    print("=" * 80)

    # Test OperationType
    print("\n[1.1] Testing OperationType enumeration...")
    try:
        assert OperationType.CONV2D.value == "conv2d"
        assert OperationType.LINEAR.value == "linear"
        assert OperationType.MULTIHEAD_ATTENTION.value == "multihead_attention"
        assert OperationType.RELU.value == "relu"
        results.pass_test("OperationType", "All operation types accessible")
    except Exception as e:
        results.fail_test("OperationType", str(e))

    # Test BottleneckType
    print("[1.2] Testing BottleneckType enumeration...")
    try:
        assert BottleneckType.COMPUTE_BOUND.value == "compute_bound"
        assert BottleneckType.MEMORY_BOUND.value == "memory_bound"
        assert BottleneckType.BANDWIDTH_BOUND.value == "bandwidth_bound"
        assert BottleneckType.BALANCED.value == "balanced"
        results.pass_test("BottleneckType", "All bottleneck types defined")
    except Exception as e:
        results.fail_test("BottleneckType", str(e))

    # Test PartitionReason
    print("[1.3] Testing PartitionReason enumeration...")
    try:
        assert PartitionReason.OPERATION_BOUNDARY.value == "operation_boundary"
        assert PartitionReason.MEMORY_LIMIT_EXCEEDED.value == "memory_limit_exceeded"
        assert PartitionReason.FUSION_INCOMPATIBLE.value == "fusion_incompatible"
        results.pass_test("PartitionReason", "All partition reasons defined")
    except Exception as e:
        results.fail_test("PartitionReason", str(e))


def test_tensor_descriptor():
    """Test TensorDescriptor dataclass"""
    print("\n" + "=" * 80)
    print("TEST 2: TensorDescriptor")
    print("=" * 80)

    print("\n[2.1] Testing basic tensor creation...")
    try:
        tensor = TensorDescriptor(
            shape=(1, 64, 224, 224),
            dtype="float32",
            size_bytes=64 * 224 * 224 * 4,
            layout="NCHW"
        )
        assert tensor.shape == (1, 64, 224, 224)
        assert tensor.dtype == "float32"
        assert tensor.size_bytes == 64 * 224 * 224 * 4
        assert tensor.layout == "NCHW"
        results.pass_test("TensorDescriptor basic", f"Created tensor with shape {tensor.shape}")
    except Exception as e:
        results.fail_test("TensorDescriptor basic", str(e))

    print("[2.2] Testing tensor validation (negative size)...")
    try:
        tensor_invalid = TensorDescriptor(
            shape=(1, 64),
            dtype="float32",
            size_bytes=-100
        )
        results.fail_test("TensorDescriptor validation", "Should reject negative size_bytes")
    except ValueError as e:
        results.pass_test("TensorDescriptor validation", "Correctly rejects negative size_bytes")
    except Exception as e:
        results.fail_test("TensorDescriptor validation", f"Unexpected error: {e}")

    print("[2.3] Testing different dtypes...")
    try:
        fp32 = TensorDescriptor(shape=(1, 10), dtype="float32", size_bytes=40)
        fp16 = TensorDescriptor(shape=(1, 10), dtype="float16", size_bytes=20)
        int8 = TensorDescriptor(shape=(1, 10), dtype="int8", size_bytes=10)
        assert fp32.dtype == "float32"
        assert fp16.dtype == "float16"
        assert int8.dtype == "int8"
        results.pass_test("TensorDescriptor dtypes", "Supports multiple dtypes")
    except Exception as e:
        results.fail_test("TensorDescriptor dtypes", str(e))


def test_parallelism_descriptor():
    """Test ParallelismDescriptor dataclass"""
    print("\n" + "=" * 80)
    print("TEST 3: ParallelismDescriptor")
    print("=" * 80)

    print("\n[3.1] Testing basic parallelism descriptor...")
    try:
        parallelism = ParallelismDescriptor(
            batch=1,
            channels=64,
            spatial=224 * 224,
            total_threads=1 * 64 * 224 * 224
        )
        assert parallelism.batch == 1
        assert parallelism.channels == 64
        assert parallelism.total_threads == 1 * 64 * 224 * 224
        results.pass_test("ParallelismDescriptor basic", f"Total threads: {parallelism.total_threads}")
    except Exception as e:
        results.fail_test("ParallelismDescriptor basic", str(e))

    print("[3.2] Testing depthwise convolution constraints...")
    try:
        depthwise = ParallelismDescriptor(
            batch=1,
            channels=32,
            spatial=112 * 112,
            total_threads=1 * 32 * 112 * 112,
            is_depthwise=True
        )
        assert depthwise.is_depthwise == True
        assert depthwise.can_split_channels == False  # Depthwise restricts channel splitting
        results.pass_test("ParallelismDescriptor depthwise", "Depthwise correctly restricts channel splitting")
    except Exception as e:
        results.fail_test("ParallelismDescriptor depthwise", str(e))

    print("[3.3] Testing validation (negative dimensions)...")
    try:
        invalid = ParallelismDescriptor(
            batch=-1,
            channels=32,
            spatial=100,
            total_threads=100
        )
        results.fail_test("ParallelismDescriptor validation", "Should reject negative dimensions")
    except ValueError:
        results.pass_test("ParallelismDescriptor validation", "Correctly rejects negative dimensions")
    except Exception as e:
        results.fail_test("ParallelismDescriptor validation", f"Unexpected error: {e}")

    print("[3.4] Testing vectorization metadata...")
    try:
        vectorizable = ParallelismDescriptor(
            batch=1,
            channels=256,
            spatial=56 * 56,
            total_threads=1 * 256 * 56 * 56,
            vectorizable_dim="channels",
            vector_width=16  # AVX-512
        )
        assert vectorizable.vectorizable_dim == "channels"
        assert vectorizable.vector_width == 16
        results.pass_test("ParallelismDescriptor vectorization", "Vectorization metadata stored")
    except Exception as e:
        results.fail_test("ParallelismDescriptor vectorization", str(e))


def test_subgraph_descriptor():
    """Test SubgraphDescriptor dataclass"""
    print("\n" + "=" * 80)
    print("TEST 4: SubgraphDescriptor")
    print("=" * 80)

    print("\n[4.1] Testing basic subgraph descriptor...")
    try:
        input_tensor = TensorDescriptor(shape=(1, 3, 224, 224), dtype="float32", size_bytes=602112)
        output_tensor = TensorDescriptor(shape=(1, 64, 224, 224), dtype="float32", size_bytes=12845056)
        weight_tensor = TensorDescriptor(shape=(64, 3, 7, 7), dtype="float32", size_bytes=37632)

        subgraph = SubgraphDescriptor(
            node_id="conv1",
            node_name="conv2d_0",
            operation_type=OperationType.CONV2D,
            fusion_pattern="conv_bn_relu",
            flops=118013952,
            macs=59006976,
            input_tensors=[input_tensor],
            output_tensors=[output_tensor],
            weight_tensors=[weight_tensor],
            total_input_bytes=602112,
            total_output_bytes=12845056,
            total_weight_bytes=37632
        )
        assert subgraph.node_id == "conv1"
        assert subgraph.operation_type == OperationType.CONV2D
        assert subgraph.flops == 118013952
        results.pass_test("SubgraphDescriptor basic", f"Created subgraph with {subgraph.flops:,} FLOPs")
    except Exception as e:
        results.fail_test("SubgraphDescriptor basic", str(e))

    print("[4.2] Testing arithmetic intensity calculation...")
    try:
        subgraph = SubgraphDescriptor(
            node_id="test",
            node_name="test",
            operation_type=OperationType.CONV2D,
            fusion_pattern="conv",
            flops=1000000,
            macs=500000,
            total_input_bytes=10000,
            total_output_bytes=10000,
            total_weight_bytes=5000
        )
        expected_ai = 1000000 / (10000 + 10000 + 5000)
        assert abs(subgraph.arithmetic_intensity - expected_ai) < 0.01
        results.pass_test("SubgraphDescriptor arithmetic_intensity", f"AI = {subgraph.arithmetic_intensity:.2f} FLOPs/byte")
    except Exception as e:
        results.fail_test("SubgraphDescriptor arithmetic_intensity", str(e))

    print("[4.3] Testing bottleneck classification...")
    try:
        # High AI -> compute bound
        compute_bound = SubgraphDescriptor(
            node_id="compute",
            node_name="compute",
            operation_type=OperationType.MATMUL,
            fusion_pattern="matmul",
            flops=1000000,
            macs=500000,
            total_input_bytes=1000,
            total_output_bytes=1000,
            total_weight_bytes=1000
        )
        assert compute_bound.recommended_bottleneck == BottleneckType.COMPUTE_BOUND

        # Low AI -> bandwidth bound
        bandwidth_bound = SubgraphDescriptor(
            node_id="bandwidth",
            node_name="bandwidth",
            operation_type=OperationType.RELU,
            fusion_pattern="relu",
            flops=1000,
            macs=0,
            total_input_bytes=10000,
            total_output_bytes=10000,
            total_weight_bytes=0
        )
        assert bandwidth_bound.recommended_bottleneck == BottleneckType.BANDWIDTH_BOUND

        results.pass_test("SubgraphDescriptor bottleneck", "Bottleneck classification working")
    except Exception as e:
        results.fail_test("SubgraphDescriptor bottleneck", str(e))

    print("[4.4] Testing partition reasoning summary...")
    try:
        subgraph = SubgraphDescriptor(
            node_id="test",
            node_name="test",
            operation_type=OperationType.CONV2D,
            fusion_pattern="conv",
            flops=1000,
            macs=500,
            partition_reason=PartitionReason.MEMORY_LIMIT_EXCEEDED,
            partition_criteria={
                "total_bytes": 100_000_000,
                "threshold_memory": 50_000_000
            }
        )
        summary = subgraph.partition_reasoning_summary()
        assert "Memory usage" in summary
        assert "exceeds fusion threshold" in summary
        results.pass_test("SubgraphDescriptor reasoning", "Partition reasoning summary generated")
    except Exception as e:
        results.fail_test("SubgraphDescriptor reasoning", str(e))


def test_subgraph_concurrency():
    """Test SubgraphConcurrency dataclass"""
    print("\n" + "=" * 80)
    print("TEST 5: SubgraphConcurrency")
    print("=" * 80)

    print("\n[5.1] Testing subgraph concurrency metadata...")
    try:
        concurrency = SubgraphConcurrency(
            total_threads=1024,
            independent_threads=512,
            independent_operations=128,
            dependency_chains=10,
            can_split_batch=True,
            can_split_spatial=True,
            can_split_channels=True,
            min_hardware_units=1,
            optimal_hardware_units=8,
            max_hardware_units=16,
            parallelism_efficiency=0.85
        )
        assert concurrency.total_threads == 1024
        assert concurrency.independent_threads == 512
        assert concurrency.optimal_hardware_units == 8
        results.pass_test("SubgraphConcurrency basic", f"Total threads: {concurrency.total_threads}, optimal units: {concurrency.optimal_hardware_units}")
    except Exception as e:
        results.fail_test("SubgraphConcurrency basic", str(e))


def test_concurrency_descriptor():
    """Test ConcurrencyDescriptor dataclass"""
    print("\n" + "=" * 80)
    print("TEST 6: ConcurrencyDescriptor")
    print("=" * 80)

    print("\n[6.1] Testing graph-level concurrency descriptor...")
    try:
        descriptor = ConcurrencyDescriptor(
            total_subgraphs=32,
            independent_subgraphs=12,
            sequential_subgraphs=20,
            critical_path_length=10,
            critical_path_flops=500_000_000,
            parallelizable_flops=3_000_000_000,
            batch_size=1,
            batch_parallelism=1,
            max_theoretical_speedup=4.5,
            num_stages=8,
            max_parallel_ops_per_stage=4
        )
        assert descriptor.total_subgraphs == 32
        assert descriptor.independent_subgraphs == 12
        assert descriptor.critical_path_length == 10
        assert descriptor.max_theoretical_speedup == 4.5
        results.pass_test("ConcurrencyDescriptor basic", f"{descriptor.total_subgraphs} total, {descriptor.independent_subgraphs} independent")
    except Exception as e:
        results.fail_test("ConcurrencyDescriptor basic", str(e))


def test_partition_report():
    """Test PartitionReport dataclass"""
    print("\n" + "=" * 80)
    print("TEST 7: PartitionReport")
    print("=" * 80)

    print("\n[7.1] Testing partition report...")
    try:
        report = PartitionReport(
            subgraphs=[],
            total_subgraphs=32,
            total_flops=3_628_000_000,
            total_macs=1_814_000_000,
            total_memory_traffic=138_700_000,  # input + output + weights
            average_arithmetic_intensity=26.2,
            min_arithmetic_intensity=0.5,
            max_arithmetic_intensity=150.0
        )
        assert report.total_subgraphs == 32
        assert report.total_flops == 3_628_000_000
        assert report.average_arithmetic_intensity == 26.2
        assert report.total_memory_traffic == 138_700_000
        results.pass_test("PartitionReport basic", f"{report.total_subgraphs} subgraphs, {report.total_flops / 1e9:.2f} GFLOPs")
    except Exception as e:
        results.fail_test("PartitionReport basic", str(e))


def test_integration():
    """Test integration of multiple data structures"""
    print("\n" + "=" * 80)
    print("TEST 8: Integration Tests")
    print("=" * 80)

    print("\n[8.1] Testing complete subgraph with all metadata...")
    try:
        # Create tensors
        input_tensor = TensorDescriptor(
            shape=(1, 64, 56, 56),
            dtype="float32",
            size_bytes=64 * 56 * 56 * 4
        )
        output_tensor = TensorDescriptor(
            shape=(1, 128, 56, 56),
            dtype="float32",
            size_bytes=128 * 56 * 56 * 4
        )
        weight_tensor = TensorDescriptor(
            shape=(128, 64, 3, 3),
            dtype="float32",
            size_bytes=128 * 64 * 3 * 3 * 4
        )

        # Create parallelism descriptor
        parallelism = ParallelismDescriptor(
            batch=1,
            channels=128,
            spatial=56 * 56,
            total_threads=1 * 128 * 56 * 56,
            can_split_batch=True,
            can_split_spatial=True,
            can_split_channels=True
        )

        # Create subgraph with all metadata
        subgraph = SubgraphDescriptor(
            node_id="conv2_1",
            node_name="layer2.0.conv1",
            operation_type=OperationType.CONV2D,
            fusion_pattern="conv_bn_relu",
            flops=231_211_008,
            macs=115_605_504,
            input_tensors=[input_tensor],
            output_tensors=[output_tensor],
            weight_tensors=[weight_tensor],
            total_input_bytes=input_tensor.size_bytes,
            total_output_bytes=output_tensor.size_bytes,
            total_weight_bytes=weight_tensor.size_bytes,
            parallelism=parallelism,
            depends_on=["conv1"],
            partition_reason=PartitionReason.FUSION_OPPORTUNITY,
            partition_criteria={"fusion_candidates": ["bn2_1", "relu2_1"]}
        )

        # Verify all fields
        assert subgraph.operation_type == OperationType.CONV2D
        assert subgraph.parallelism is not None
        assert subgraph.parallelism.total_threads == 1 * 128 * 56 * 56
        assert subgraph.arithmetic_intensity > 0
        assert len(subgraph.depends_on) == 1

        results.pass_test("Integration test", "Complete subgraph with all metadata created successfully")
    except Exception as e:
        results.fail_test("Integration test", str(e))


def main():
    """Run all tests"""
    print("=" * 80)
    print("IR STRUCTURES UNIT TESTS")
    print("Testing: graphs.ir.structures")
    print("=" * 80)

    test_enumerations()
    test_tensor_descriptor()
    test_parallelism_descriptor()
    test_subgraph_descriptor()
    test_subgraph_concurrency()
    test_concurrency_descriptor()
    test_partition_report()
    test_integration()

    results.print_summary()


if __name__ == '__main__':
    main()
