"""
Unit tests for workload characterization data structures.

Tests the core MAC/FLOP/IntOp separation design.
"""

import pytest
from graphs.analysis.workload_characterization import (
    WorkloadCharacterization,
    OperationBreakdown,
    Precision,
    create_simple_workload
)


class TestOperationBreakdown:
    """Test OperationBreakdown data structure"""

    def test_empty_breakdown(self):
        """Test empty breakdown"""
        breakdown = OperationBreakdown()
        assert breakdown.total_macs() == 0
        assert breakdown.total_flops() == 0
        assert breakdown.total_intops() == 0

    def test_mac_operations(self):
        """Test MAC operation counting"""
        breakdown = OperationBreakdown(
            matmul_macs=65536,
            conv2d_macs=1000,
            depthwise_conv_macs=500
        )
        assert breakdown.total_macs() == 67036
        assert breakdown.total_flops() == 0
        assert breakdown.total_intops() == 0

    def test_flop_operations(self):
        """Test FLOP operation counting"""
        breakdown = OperationBreakdown(
            bias_flops=256,
            relu_flops=256,
            batchnorm_flops=1024
        )
        assert breakdown.total_macs() == 0
        assert breakdown.total_flops() == 1536
        assert breakdown.total_intops() == 0

    def test_intop_operations(self):
        """Test integer operation counting"""
        breakdown = OperationBreakdown(
            quantize_intops=1000,
            dequantize_intops=1000,
            indexing_intops=500
        )
        assert breakdown.total_macs() == 0
        assert breakdown.total_flops() == 0
        assert breakdown.total_intops() == 2500

    def test_mixed_operations(self):
        """Test mixed operation types"""
        breakdown = OperationBreakdown(
            matmul_macs=65536,
            bias_flops=256,
            relu_flops=256,
            quantize_intops=100
        )
        assert breakdown.total_macs() == 65536
        assert breakdown.total_flops() == 512
        assert breakdown.total_intops() == 100


class TestWorkloadCharacterization:
    """Test WorkloadCharacterization data structure"""

    def test_simple_workload(self):
        """Test simple workload creation"""
        workload = WorkloadCharacterization(
            macs=65536,
            flops=512,
            intops=0,
            bytes_transferred=264192
        )
        assert workload.macs == 65536
        assert workload.flops == 512
        assert workload.intops == 0
        assert workload.total_ops() == 66048
        assert workload.bytes_transferred == 264192

    def test_factory_function(self):
        """Test create_simple_workload factory function"""
        workload = create_simple_workload(
            macs=65536,
            flops=512,
            bytes_transferred=264192,
            model_name="mlp_256x256"
        )
        assert workload.macs == 65536
        assert workload.flops == 512
        assert workload.intops == 0
        assert workload.model_name == "mlp_256x256"

    def test_arithmetic_intensity(self):
        """Test arithmetic intensity calculation"""
        workload = WorkloadCharacterization(
            macs=65536,
            flops=512,
            intops=0,
            bytes_transferred=264192
        )
        # MACs/byte
        assert abs(workload.arithmetic_intensity_macs() - 0.248) < 0.001

        # Total ops/byte
        assert abs(workload.arithmetic_intensity_total() - 0.250) < 0.001

    def test_operation_percentages(self):
        """Test operation percentage calculations"""
        workload = WorkloadCharacterization(
            macs=65536,
            flops=512,
            intops=100,
            bytes_transferred=264192
        )
        # MACs: 65536 / 66148 = 99.07%
        assert abs(workload.mac_percentage() - 99.07) < 0.1

        # FLOPs: 512 / 66148 = 0.77%
        assert abs(workload.flop_percentage() - 0.77) < 0.1

        # IntOps: 100 / 66148 = 0.15%
        assert abs(workload.intop_percentage() - 0.15) < 0.1

    def test_precision_configuration(self):
        """Test precision configuration"""
        workload = WorkloadCharacterization(
            macs=65536,
            flops=512,
            intops=0,
            mac_precision="fp16",
            flop_precision="fp32",
            intop_precision="int8",
            accumulator_precision="fp32",
            bytes_transferred=264192
        )
        assert workload.mac_precision == "fp16"
        assert workload.flop_precision == "fp32"
        assert workload.intop_precision == "int8"
        assert workload.accumulator_precision == "fp32"

    def test_breakdown_validation(self):
        """Test that breakdown totals match"""
        breakdown = OperationBreakdown(
            matmul_macs=65536,
            bias_flops=256,
            relu_flops=256,
            quantize_intops=100
        )
        workload = WorkloadCharacterization(
            macs=65536,
            flops=512,
            intops=100,
            bytes_transferred=264192,
            breakdown=breakdown
        )
        # Should not raise
        assert workload.breakdown.total_macs() == workload.macs
        assert workload.breakdown.total_flops() == workload.flops
        assert workload.breakdown.total_intops() == workload.intops

    def test_breakdown_validation_mismatch(self):
        """Test that mismatched breakdown raises error"""
        breakdown = OperationBreakdown(
            matmul_macs=1000,  # Doesn't match
            bias_flops=256,
            relu_flops=256
        )
        with pytest.raises(ValueError, match="Breakdown MACs"):
            WorkloadCharacterization(
                macs=65536,  # Different from breakdown
                flops=512,
                intops=0,
                bytes_transferred=264192,
                breakdown=breakdown
            )

    def test_negative_values_rejected(self):
        """Test that negative values are rejected"""
        with pytest.raises(ValueError, match="macs must be non-negative"):
            WorkloadCharacterization(
                macs=-100,
                flops=512,
                intops=0,
                bytes_transferred=264192
            )

        with pytest.raises(ValueError, match="flops must be non-negative"):
            WorkloadCharacterization(
                macs=65536,
                flops=-10,
                intops=0,
                bytes_transferred=264192
            )

        with pytest.raises(ValueError, match="bytes_transferred must be non-negative"):
            WorkloadCharacterization(
                macs=65536,
                flops=512,
                intops=0,
                bytes_transferred=-1000
            )

    def test_is_compute_bound(self):
        """Test compute-bound vs memory-bound classification"""
        # Memory-bound workload (low AI)
        mem_bound = WorkloadCharacterization(
            macs=1000,
            flops=0,
            intops=0,
            bytes_transferred=10000  # AI = 0.1 ops/byte
        )
        # Peak: 100 GFLOPS, 100 GB/s → threshold = 1 op/byte
        assert not mem_bound.is_compute_bound(
            peak_bandwidth_bytes_per_sec=100e9,
            peak_compute_ops_per_sec=100e9
        )

        # Compute-bound workload (high AI)
        compute_bound = WorkloadCharacterization(
            macs=100000,
            flops=0,
            intops=0,
            bytes_transferred=10000  # AI = 10 ops/byte
        )
        # Peak: 100 GFLOPS, 100 GB/s → threshold = 1 op/byte
        assert compute_bound.is_compute_bound(
            peak_bandwidth_bytes_per_sec=100e9,
            peak_compute_ops_per_sec=100e9
        )

    def test_string_representation(self):
        """Test __str__ method"""
        workload = WorkloadCharacterization(
            macs=65536,
            flops=512,
            intops=0,
            bytes_transferred=264192,
            model_name="mlp_256x256",
            batch_size=1
        )
        s = str(workload)
        assert "mlp_256x256" in s
        assert "65,536" in s
        assert "512" in s
        assert "MACs:" in s
        assert "FLOPs:" in s


class TestMLP256Example:
    """Test case for 256×256 MLP with bias and ReLU"""

    def test_mlp_256_workload(self):
        """
        256×256 MLP: Y = X @ W + b, ReLU(Y)

        Operations:
        - MatMul: 256×256 = 65,536 MACs
        - Bias: 256 additions = 256 FLOPs
        - ReLU: 256 comparisons = 256 FLOPs
        Total: 65,536 MACs + 512 FLOPs = 66,048 ops

        Memory:
        - Input: 1×256 FP32 = 1,024 bytes
        - Weights: 256×256 FP32 = 262,144 bytes
        - Output: 1×256 FP32 = 1,024 bytes
        Total: 264,192 bytes
        """
        breakdown = OperationBreakdown(
            matmul_macs=65536,
            bias_flops=256,
            relu_flops=256
        )

        workload = WorkloadCharacterization(
            macs=65536,
            flops=512,
            intops=0,
            mac_precision="fp32",
            flop_precision="fp32",
            bytes_transferred=264192,
            input_bytes=1024,
            weight_bytes=262144,
            output_bytes=1024,
            batch_size=1,
            model_name="mlp_256x256",
            breakdown=breakdown
        )

        # Verify totals
        assert workload.total_ops() == 66048
        assert workload.bytes_transferred == 264192

        # Verify arithmetic intensity
        # MACs/byte = 65536 / 264192 = 0.248
        assert abs(workload.arithmetic_intensity_macs() - 0.248) < 0.001

        # Total ops/byte = 66048 / 264192 = 0.250
        assert abs(workload.arithmetic_intensity_total() - 0.250) < 0.001

        # Verify breakdown
        assert workload.breakdown.matmul_macs == 65536
        assert workload.breakdown.bias_flops == 256
        assert workload.breakdown.relu_flops == 256

        # Verify it's memory-bound (AI < 1)
        # Typical GPU: 300 GB/s, 30 TFLOPS → threshold = 100 ops/byte
        assert not workload.is_compute_bound(
            peak_bandwidth_bytes_per_sec=300e9,
            peak_compute_ops_per_sec=30e12
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
