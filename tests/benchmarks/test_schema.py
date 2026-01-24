"""
Tests for benchmark specification schema.

Tests cover:
- Schema creation and validation
- Serialization (to_dict, to_json)
- Deserialization (from_dict, from_json)
- YAML loading
- Computed properties (flops, arithmetic intensity)
"""

import pytest
import json
import tempfile
from pathlib import Path

from graphs.benchmarks.schema import (
    BenchmarkSpec,
    BenchmarkCategory,
    Precision,
    DeviceType,
    ExecutionConfig,
    GEMMSpec,
    Conv2dSpec,
    MemoryBenchSpec,
    WorkloadSpec,
    BenchmarkResult,
    TimingStats,
    load_spec_from_yaml,
    save_spec_to_yaml,
    load_specs_from_directory,
)


class TestExecutionConfig:
    """Tests for ExecutionConfig"""

    def test_defaults(self):
        config = ExecutionConfig()
        assert config.warmup_iterations == 10
        assert config.measurement_iterations == 100
        assert config.sync_before_timing is True

    def test_custom_values(self):
        config = ExecutionConfig(
            warmup_iterations=5,
            measurement_iterations=50,
            min_duration_ms=200.0,
        )
        assert config.warmup_iterations == 5
        assert config.measurement_iterations == 50
        assert config.min_duration_ms == 200.0

    def test_serialization(self):
        config = ExecutionConfig(warmup_iterations=5)
        d = config.to_dict()
        assert d['warmup_iterations'] == 5

        config2 = ExecutionConfig.from_dict(d)
        assert config2.warmup_iterations == 5


class TestBenchmarkSpec:
    """Tests for base BenchmarkSpec"""

    def test_create_basic(self):
        spec = BenchmarkSpec(name="test_bench")
        assert spec.name == "test_bench"
        assert spec.category == BenchmarkCategory.MICROBENCHMARK
        assert Precision.FP32 in spec.precisions

    def test_with_tags(self):
        spec = BenchmarkSpec(
            name="test_bench",
            tags=["compute", "test"],
        )
        assert "compute" in spec.tags
        assert "test" in spec.tags

    def test_serialization_roundtrip(self):
        spec = BenchmarkSpec(
            name="test_bench",
            description="A test benchmark",
            tags=["test"],
            precisions=[Precision.FP32, Precision.FP16],
            devices=[DeviceType.CUDA],
        )

        d = spec.to_dict()
        spec2 = BenchmarkSpec.from_dict(d)

        assert spec2.name == spec.name
        assert spec2.description == spec.description
        assert spec2.tags == spec.tags
        assert spec2.precisions == spec.precisions
        assert spec2.devices == spec.devices


class TestGEMMSpec:
    """Tests for GEMM benchmark specification"""

    def test_create_default(self):
        spec = GEMMSpec(name="gemm_test")
        assert spec.M == 1024
        assert spec.N == 1024
        assert spec.K == 1024
        assert spec.category == BenchmarkCategory.MICROBENCHMARK
        assert "gemm" in spec.tags

    def test_create_custom_size(self):
        spec = GEMMSpec(
            name="gemm_large",
            M=4096,
            N=4096,
            K=4096,
        )
        assert spec.M == 4096
        assert spec.N == 4096
        assert spec.K == 4096

    def test_flops_calculation(self):
        # GEMM: 2*M*N*K
        spec = GEMMSpec(name="gemm_test", M=1024, N=1024, K=1024)
        expected_flops = 2 * 1024 * 1024 * 1024
        assert spec.flops == expected_flops

    def test_flops_batched(self):
        spec = GEMMSpec(name="gemm_batched", M=512, N=512, K=512, batch_size=4)
        expected_flops = 4 * 2 * 512 * 512 * 512
        assert spec.flops == expected_flops

    def test_arithmetic_intensity(self):
        spec = GEMMSpec(name="gemm_test", M=1024, N=1024, K=1024)
        # AI should be positive for compute-heavy GEMM
        assert spec.arithmetic_intensity > 0
        # Large GEMM should have high arithmetic intensity
        assert spec.arithmetic_intensity > 100  # Very compute-bound

    def test_serialization_roundtrip(self):
        spec = GEMMSpec(
            name="gemm_test",
            M=2048,
            N=1024,
            K=512,
            batch_size=2,
            alpha=2.0,
        )

        d = spec.to_dict()
        spec2 = GEMMSpec.from_dict(d)

        assert spec2.name == spec.name
        assert spec2.M == spec.M
        assert spec2.N == spec.N
        assert spec2.K == spec.K
        assert spec2.batch_size == spec.batch_size
        assert spec2.alpha == spec.alpha


class TestConv2dSpec:
    """Tests for Conv2d benchmark specification"""

    def test_create_default(self):
        spec = Conv2dSpec(name="conv_test")
        assert spec.in_channels == 64
        assert spec.out_channels == 64
        assert spec.kernel_size == 3
        assert "conv2d" in spec.tags

    def test_create_depthwise(self):
        spec = Conv2dSpec(
            name="conv_dw",
            in_channels=64,
            out_channels=64,
            groups=64,  # Depthwise
        )
        assert spec.groups == 64
        assert "depthwise" in spec.tags

    def test_create_pointwise(self):
        spec = Conv2dSpec(
            name="conv_pw",
            kernel_size=1,
        )
        assert spec.kernel_size == 1
        assert "pointwise" in spec.tags

    def test_output_size_calculation(self):
        spec = Conv2dSpec(
            name="conv_test",
            height=56,
            width=56,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        assert spec.output_height == 56  # Same padding
        assert spec.output_width == 56

    def test_output_size_stride2(self):
        spec = Conv2dSpec(
            name="conv_stride2",
            height=56,
            width=56,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        assert spec.output_height == 28
        assert spec.output_width == 28

    def test_flops_calculation(self):
        spec = Conv2dSpec(
            name="conv_test",
            batch_size=1,
            in_channels=64,
            out_channels=64,
            height=56,
            width=56,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        # FLOPs should be positive
        assert spec.flops > 0
        # Rough check: 2 * K^2 * C_in * C_out * H * W
        expected_order = 2 * 9 * 64 * 64 * 56 * 56
        assert abs(spec.flops - expected_order) < expected_order * 0.1

    def test_serialization_roundtrip(self):
        spec = Conv2dSpec(
            name="conv_test",
            in_channels=128,
            out_channels=256,
            kernel_size=5,
            stride=2,
        )

        d = spec.to_dict()
        spec2 = Conv2dSpec.from_dict(d)

        assert spec2.in_channels == spec.in_channels
        assert spec2.out_channels == spec.out_channels
        assert spec2.kernel_size == spec.kernel_size
        assert spec2.stride == spec.stride


class TestMemoryBenchSpec:
    """Tests for memory bandwidth benchmark specification"""

    def test_create_default(self):
        spec = MemoryBenchSpec(name="mem_test")
        assert spec.array_size == 10_000_000
        assert spec.pattern == "triad"
        assert spec.category == BenchmarkCategory.MEMORY

    def test_patterns(self):
        for pattern in ["copy", "scale", "add", "triad"]:
            spec = MemoryBenchSpec(name=f"mem_{pattern}", pattern=pattern)
            assert spec.pattern == pattern
            assert pattern in spec.tags


class TestWorkloadSpec:
    """Tests for full model workload specification"""

    def test_create_default(self):
        spec = WorkloadSpec(name="workload_test")
        assert spec.model_name == "resnet18"
        assert spec.model_source == "torchvision"
        assert spec.batch_size == 1
        assert spec.category == BenchmarkCategory.WORKLOAD

    def test_create_custom(self):
        spec = WorkloadSpec(
            name="bert_bench",
            model_name="bert-base-uncased",
            model_source="huggingface",
            batch_size=8,
            input_shape=(512,),  # Sequence length
        )
        assert spec.model_name == "bert-base-uncased"
        assert spec.batch_size == 8

    def test_serialization_roundtrip(self):
        spec = WorkloadSpec(
            name="workload_test",
            model_name="mobilenet_v2",
            batch_size=4,
            input_shape=(3, 224, 224),
        )

        d = spec.to_dict()
        spec2 = WorkloadSpec.from_dict(d)

        assert spec2.model_name == spec.model_name
        assert spec2.batch_size == spec.batch_size
        assert spec2.input_shape == spec.input_shape


class TestTimingStats:
    """Tests for timing statistics"""

    def test_create(self):
        stats = TimingStats(
            mean_ms=1.5,
            std_ms=0.1,
            min_ms=1.2,
            max_ms=2.0,
            median_ms=1.4,
            p95_ms=1.8,
            p99_ms=1.9,
            num_iterations=100,
        )
        assert stats.mean_ms == 1.5
        assert stats.num_iterations == 100

    def test_serialization_roundtrip(self):
        stats = TimingStats(
            mean_ms=1.5,
            std_ms=0.1,
            min_ms=1.2,
            max_ms=2.0,
            median_ms=1.4,
            p95_ms=1.8,
            p99_ms=1.9,
            num_iterations=100,
        )

        d = stats.to_dict()
        stats2 = TimingStats.from_dict(d)

        assert stats2.mean_ms == stats.mean_ms
        assert stats2.std_ms == stats.std_ms


class TestBenchmarkResult:
    """Tests for benchmark results"""

    def test_create_success(self):
        result = BenchmarkResult(
            spec_name="gemm_1024",
            timestamp="2026-01-23T12:00:00Z",
            device="cuda:0",
            device_name="NVIDIA H100",
            gflops=1500.0,
            success=True,
        )
        assert result.success
        assert result.gflops == 1500.0

    def test_create_with_timing(self):
        timing = TimingStats(
            mean_ms=0.5,
            std_ms=0.05,
            min_ms=0.4,
            max_ms=0.7,
            median_ms=0.48,
            p95_ms=0.6,
            p99_ms=0.65,
            num_iterations=100,
        )
        result = BenchmarkResult(
            spec_name="gemm_1024",
            timestamp="2026-01-23T12:00:00Z",
            device="cuda:0",
            timing=timing,
        )
        assert result.timing is not None
        assert result.timing.mean_ms == 0.5

    def test_create_failure(self):
        result = BenchmarkResult(
            spec_name="gemm_huge",
            timestamp="2026-01-23T12:00:00Z",
            device="cuda:0",
            success=False,
            error_message="Out of memory",
        )
        assert not result.success
        assert "memory" in result.error_message.lower()

    def test_json_roundtrip(self):
        result = BenchmarkResult(
            spec_name="gemm_1024",
            timestamp="2026-01-23T12:00:00Z",
            device="cuda:0",
            device_name="NVIDIA H100",
            precision="fp16",
            gflops=3000.0,
            bandwidth_gbps=2000.0,
            avg_power_watts=400.0,
        )

        json_str = result.to_json()
        result2 = BenchmarkResult.from_json(json_str)

        assert result2.spec_name == result.spec_name
        assert result2.gflops == result.gflops
        assert result2.precision == result.precision


class TestYAMLLoading:
    """Tests for YAML spec loading"""

    def test_save_and_load_gemm(self):
        spec = GEMMSpec(
            name="gemm_test",
            M=2048,
            N=2048,
            K=2048,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "gemm_test.yaml"
            save_spec_to_yaml(spec, path)

            loaded = load_spec_from_yaml(path)

            assert isinstance(loaded, GEMMSpec)
            assert loaded.name == spec.name
            assert loaded.M == spec.M

    def test_save_and_load_conv2d(self):
        spec = Conv2dSpec(
            name="conv_test",
            in_channels=128,
            out_channels=256,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "conv_test.yaml"
            save_spec_to_yaml(spec, path)

            loaded = load_spec_from_yaml(path)

            assert isinstance(loaded, Conv2dSpec)
            assert loaded.in_channels == spec.in_channels

    def test_load_directory(self):
        specs = [
            GEMMSpec(name="gemm_1"),
            GEMMSpec(name="gemm_2"),
            Conv2dSpec(name="conv_1"),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            for spec in specs:
                save_spec_to_yaml(spec, Path(tmpdir) / f"{spec.name}.yaml")

            registry = load_specs_from_directory(tmpdir)

            assert len(registry) == 3
            assert "gemm_1" in registry
            assert "gemm_2" in registry
            assert "conv_1" in registry


class TestSpecValidation:
    """Tests for spec validation"""

    def test_gemm_positive_dimensions(self):
        # Should work with positive dimensions
        spec = GEMMSpec(name="test", M=1024, N=1024, K=1024)
        assert spec.flops > 0

    def test_conv2d_valid_output(self):
        # Output should be positive with valid parameters
        spec = Conv2dSpec(
            name="test",
            height=224,
            width=224,
            kernel_size=7,
            stride=2,
            padding=3,
        )
        assert spec.output_height > 0
        assert spec.output_width > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
