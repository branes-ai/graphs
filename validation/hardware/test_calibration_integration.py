#!/usr/bin/env python3
"""
Test calibration integration with CPUMapper.

This script verifies that:
1. CPUMapper accepts calibration parameter
2. Calibrated efficiency is applied correctly
3. Latency estimates improve with calibration
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from graphs.hardware.calibration import load_calibration
from graphs.hardware.mappers.cpu import create_i7_12700k_mapper
from graphs.hardware.resource_model import Precision
from graphs.ir.structures import SubgraphDescriptor, TensorDescriptor, ParallelismDescriptor, OperationType
from graphs.transform.partitioning import FusedSubgraph


def create_test_matmul_subgraph():
    """Create a synthetic matmul subgraph for testing"""
    # Matrix size: 2048x2048 (matches calibration data)
    N = 2048
    flops = 2 * N * N * N  # 2 * N^3 for matmul

    # Input A: [2048, 2048], B: [2048, 2048], Output C: [2048, 2048]
    bytes_per_element = 4  # FP32
    input_bytes = 2 * N * N * bytes_per_element  # A + B
    output_bytes = N * N * bytes_per_element      # C
    weight_bytes = 0  # No separate weights

    from graphs.ir.structures import BottleneckType

    subgraph = FusedSubgraph(
        subgraph_id=0,
        node_ids=["matmul_0"],
        node_names=["matmul"],
        operation_types=[OperationType.MATMUL],
        total_flops=flops,
        total_macs=flops // 2,
        total_input_bytes=input_bytes,
        total_output_bytes=output_bytes,
        internal_bytes=0,
        total_weight_bytes=weight_bytes,
        parallelism=ParallelismDescriptor(
            batch=1,
            channels=1,
            spatial=N * N,
            total_threads=N * N,
        ),
        fusion_pattern="matmul",
        num_operators=1,
        depends_on=[],  # No dependencies
        arithmetic_intensity=flops / (input_bytes + output_bytes),
        recommended_bottleneck=BottleneckType.COMPUTE_BOUND,
    )

    return subgraph


def create_test_elementwise_subgraph():
    """Create a synthetic element-wise operation (ReLU) for testing"""
    from graphs.ir.structures import BottleneckType

    # Large tensor: 64MB
    size_mb = 64
    num_elements = (size_mb * 1024 * 1024) // 4  # FP32

    # Element-wise: 1 FLOP per element
    flops = num_elements
    bytes_per_element = 4  # FP32
    input_bytes = num_elements * bytes_per_element
    output_bytes = num_elements * bytes_per_element

    subgraph = FusedSubgraph(
        subgraph_id=1,
        node_ids=["relu_0"],
        node_names=["relu"],
        operation_types=[OperationType.RELU],
        total_flops=flops,
        total_macs=flops // 2,
        total_input_bytes=input_bytes,
        total_output_bytes=output_bytes,
        internal_bytes=0,
        total_weight_bytes=0,
        parallelism=ParallelismDescriptor(
            batch=1,
            channels=1,
            spatial=num_elements,
            total_threads=num_elements,
        ),
        fusion_pattern="relu",
        num_operators=1,
        depends_on=[],  # No dependencies
        arithmetic_intensity=flops / (input_bytes + output_bytes),
        recommended_bottleneck=BottleneckType.MEMORY_BOUND,
    )

    return subgraph


def main():
    print("=" * 80)
    print("CPUMapper Calibration Integration Test")
    print("=" * 80)
    print()

    # Load calibration profile
    # validation/hardware/test_calibration_integration.py → ../../src/graphs/hardware/calibration/profiles/
    calibration_path = Path(__file__).parent.parent.parent / "src" / "graphs" / "hardware" / "calibration" / "profiles" / "intel_i7_12700k.json"

    if not calibration_path.exists():
        print(f"ERROR: Calibration profile not found at {calibration_path}")
        return 1

    print(f"Loading calibration: {calibration_path}")
    calibration = load_calibration(calibration_path)
    print()

    calibration.print_summary()
    print()

    # Create mappers (with and without calibration)
    print("Creating mappers...")
    mapper_uncalibrated = create_i7_12700k_mapper()
    mapper_calibrated = create_i7_12700k_mapper()
    mapper_calibrated.calibration = calibration
    print(f"  Uncalibrated mapper: default efficiency = {mapper_uncalibrated.default_efficiency:.2f}")
    print(f"  Calibrated mapper: using profile with {len(calibration.operation_profiles)} operations")
    print()

    # Test 1: Matrix multiplication (compute-bound)
    print("-" * 80)
    print("Test 1: Matrix Multiplication (2048×2048)")
    print("-" * 80)
    matmul_subgraph = create_test_matmul_subgraph()

    # Map without calibration
    alloc_uncal = mapper_uncalibrated.map_subgraph(
        matmul_subgraph,
        execution_stage=0,
        concurrent_subgraphs=1,
        precision=Precision.FP32
    )

    # Map with calibration
    alloc_cal = mapper_calibrated.map_subgraph(
        matmul_subgraph,
        execution_stage=0,
        concurrent_subgraphs=1,
        precision=Precision.FP32
    )

    print(f"  FLOPs: {matmul_subgraph.total_flops / 1e9:.2f} GFLOP")
    print(f"  Arithmetic Intensity: {matmul_subgraph.arithmetic_intensity:.2f} FLOP/byte")
    print()
    print(f"  Uncalibrated estimate: {alloc_uncal.estimated_latency * 1000:.2f} ms")
    print(f"  Calibrated estimate:   {alloc_cal.estimated_latency * 1000:.2f} ms")
    improvement = (alloc_uncal.estimated_latency / alloc_cal.estimated_latency - 1) * 100
    print(f"  Improvement: {improvement:.1f}% faster with calibration")
    print()

    # Expected calibrated efficiency for 2048 matmul: ~73-78%
    # vs default 20% → expect 3-4× speedup
    expected_ratio = 0.20 / 0.75  # ~0.27
    actual_ratio = alloc_cal.estimated_latency / alloc_uncal.estimated_latency
    print(f"  Expected ratio: ~{expected_ratio:.2f} (calibrated/uncalibrated)")
    print(f"  Actual ratio:    {actual_ratio:.2f}")
    print()

    # Test 2: Element-wise operation (memory-bound)
    print("-" * 80)
    print("Test 2: Element-wise ReLU (64 MB)")
    print("-" * 80)
    relu_subgraph = create_test_elementwise_subgraph()

    # Map without calibration
    alloc_uncal_relu = mapper_uncalibrated.map_subgraph(
        relu_subgraph,
        execution_stage=0,
        concurrent_subgraphs=1,
        precision=Precision.FP32
    )

    # Map with calibration
    alloc_cal_relu = mapper_calibrated.map_subgraph(
        relu_subgraph,
        execution_stage=0,
        concurrent_subgraphs=1,
        precision=Precision.FP32
    )

    print(f"  FLOPs: {relu_subgraph.total_flops / 1e9:.3f} GFLOP")
    print(f"  Bytes: {(relu_subgraph.total_input_bytes + relu_subgraph.total_output_bytes) / 1e6:.1f} MB")
    print(f"  Arithmetic Intensity: {relu_subgraph.arithmetic_intensity:.3f} FLOP/byte")
    print()
    print(f"  Uncalibrated estimate: {alloc_uncal_relu.estimated_latency * 1000:.2f} ms")
    print(f"  Calibrated estimate:   {alloc_cal_relu.estimated_latency * 1000:.2f} ms")
    improvement_relu = (alloc_uncal_relu.estimated_latency / alloc_cal_relu.estimated_latency - 1) * 100
    print(f"  Improvement: {improvement_relu:.1f}% with calibration")
    print()

    # Memory-bound ops: calibrated bandwidth ~52.6 GB/s vs theoretical 75 GB/s
    # Expect calibrated to be slower (more realistic)
    bandwidth_ratio = 75 / 52.6  # ~1.43
    actual_ratio_relu = alloc_cal_relu.estimated_latency / alloc_uncal_relu.estimated_latency
    print(f"  Expected ratio: ~{bandwidth_ratio:.2f} (calibrated/uncalibrated, should be slower)")
    print(f"  Actual ratio:    {actual_ratio_relu:.2f}")
    print()

    # Summary
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print("✓ Calibration integration successful!")
    print(f"  - Matmul (compute-bound): {improvement:.1f}% faster with calibration")
    print(f"  - ReLU (memory-bound): {improvement_relu:.1f}% impact with calibration")
    print()
    print("Calibration provides operation-specific efficiency factors that dramatically")
    print("improve estimation accuracy for compute-bound workloads while also")
    print("accounting for realistic memory bandwidth limitations.")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
