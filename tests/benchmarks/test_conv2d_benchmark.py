"""
Tests for Conv2d Microbenchmark Suite.

Tests cover:
- Spec loading from YAML files
- Benchmark execution on CPU
- Result format validation
- Statistical aggregation
- Depthwise and batched variants
- Error handling
"""

import pytest
import torch

from graphs.benchmarks.schema import (
    Conv2dSpec,
    ExecutionConfig,
    Precision,
)
from graphs.benchmarks.microbench.conv2d import (
    Conv2dBenchmark,
    get_conv2d_specs,
    run_conv2d_benchmark,
    create_conv2d_spec,
    format_result_table,
)


class TestSpecLoading:
    """Tests for loading Conv2d specs from YAML."""

    def test_load_all_specs(self):
        specs = get_conv2d_specs()
        assert len(specs) > 0
        assert all(isinstance(s, Conv2dSpec) for s in specs)

    def test_specs_have_required_fields(self):
        specs = get_conv2d_specs()
        for spec in specs:
            assert spec.name
            assert spec.in_channels > 0
            assert spec.out_channels > 0
            assert spec.height > 0
            assert spec.width > 0
            assert spec.kernel_size > 0

    def test_filter_by_tags(self):
        all_specs = get_conv2d_specs()
        depthwise_specs = get_conv2d_specs(filter_tags=["depthwise"])

        assert len(depthwise_specs) <= len(all_specs)
        for spec in depthwise_specs:
            assert "depthwise" in spec.tags

    def test_exclude_batched(self):
        all_specs = get_conv2d_specs()
        unbatched_specs = get_conv2d_specs(include_batched=False)

        assert len(unbatched_specs) <= len(all_specs)
        for spec in unbatched_specs:
            assert spec.batch_size == 1

    def test_exclude_depthwise(self):
        all_specs = get_conv2d_specs()
        standard_specs = get_conv2d_specs(include_depthwise=False)

        assert len(standard_specs) <= len(all_specs)
        for spec in standard_specs:
            assert spec.groups == 1


class TestCreateConv2dSpec:
    """Tests for programmatic spec creation."""

    def test_create_basic_spec(self):
        spec = create_conv2d_spec(
            in_channels=64,
            out_channels=64,
            height=56,
            width=56,
            kernel_size=3,
        )
        assert spec.in_channels == 64
        assert spec.out_channels == 64
        assert spec.kernel_size == 3
        assert "conv2d_3x3" in spec.name

    def test_create_depthwise_spec(self):
        spec = create_conv2d_spec(
            in_channels=64,
            out_channels=64,
            height=56,
            width=56,
            kernel_size=3,
            groups=64,  # Depthwise
        )
        assert spec.groups == 64
        assert "dw" in spec.name

    def test_create_batched_spec(self):
        spec = create_conv2d_spec(
            in_channels=64,
            out_channels=64,
            height=56,
            width=56,
            batch_size=4,
        )
        assert spec.batch_size == 4
        assert "batch4" in spec.name

    def test_create_strided_spec(self):
        spec = create_conv2d_spec(
            in_channels=64,
            out_channels=64,
            height=56,
            width=56,
            stride=2,
        )
        assert spec.stride == 2
        assert "s2" in spec.name


class TestConv2dBenchmark:
    """Tests for Conv2dBenchmark execution."""

    @pytest.fixture
    def fast_config(self):
        return ExecutionConfig(
            warmup_iterations=2,
            measurement_iterations=5,
        )

    def test_run_simple_conv2d_cpu(self, fast_config):
        spec = create_conv2d_spec(
            in_channels=32,
            out_channels=32,
            height=28,
            width=28,
            kernel_size=3,
        )
        benchmark = Conv2dBenchmark(config=fast_config)

        result = benchmark.run(spec, device="cpu")

        assert result.success
        assert result.timing is not None
        assert result.timing.mean_ms > 0
        assert result.gflops > 0

    def test_run_depthwise_conv2d_cpu(self, fast_config):
        spec = create_conv2d_spec(
            in_channels=64,
            out_channels=64,
            height=28,
            width=28,
            kernel_size=3,
            groups=64,
        )
        benchmark = Conv2dBenchmark(config=fast_config)

        result = benchmark.run(spec, device="cpu")

        assert result.success
        # Depthwise has fewer FLOPs so GFLOPS may be lower
        assert result.timing.mean_ms > 0

    def test_run_batched_conv2d_cpu(self, fast_config):
        spec = create_conv2d_spec(
            in_channels=32,
            out_channels=32,
            height=28,
            width=28,
            batch_size=4,
        )
        benchmark = Conv2dBenchmark(config=fast_config)

        result = benchmark.run(spec, device="cpu")

        assert result.success

    def test_run_fp16_conv2d_cpu(self, fast_config):
        spec = create_conv2d_spec(
            in_channels=32,
            out_channels=32,
            height=28,
            width=28,
            precision=Precision.FP16,
        )
        benchmark = Conv2dBenchmark(config=fast_config)

        result = benchmark.run(spec, device="cpu", precision=Precision.FP16)

        assert result.success
        assert result.precision == "fp16"

    def test_run_1x1_conv2d_cpu(self, fast_config):
        spec = create_conv2d_spec(
            in_channels=64,
            out_channels=256,
            height=28,
            width=28,
            kernel_size=1,
            padding=0,
        )
        benchmark = Conv2dBenchmark(config=fast_config)

        result = benchmark.run(spec, device="cpu")

        assert result.success

    def test_run_suite(self, fast_config):
        specs = [
            create_conv2d_spec(in_channels=16, out_channels=16, height=14, width=14),
            create_conv2d_spec(in_channels=32, out_channels=32, height=14, width=14),
        ]
        benchmark = Conv2dBenchmark(config=fast_config)

        results = benchmark.run_suite(specs, device="cpu")

        assert len(results) == 2
        assert all(r.success for r in results)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_run_conv2d_cuda(self, fast_config):
        spec = create_conv2d_spec(
            in_channels=64,
            out_channels=64,
            height=56,
            width=56,
        )
        benchmark = Conv2dBenchmark(config=fast_config)

        result = benchmark.run(spec, device="cuda")

        assert result.success
        assert result.device == "cuda"


class TestRunConv2dBenchmark:
    """Tests for the convenience function."""

    def test_convenience_function(self):
        spec = create_conv2d_spec(
            in_channels=16,
            out_channels=16,
            height=14,
            width=14,
        )
        config = ExecutionConfig(warmup_iterations=2, measurement_iterations=5)

        result = run_conv2d_benchmark(spec, device="cpu", config=config)

        assert result.success
        assert result.timing is not None


class TestResultFormat:
    """Tests for result format and content."""

    @pytest.fixture
    def sample_result(self):
        spec = create_conv2d_spec(
            in_channels=32,
            out_channels=32,
            height=28,
            width=28,
        )
        config = ExecutionConfig(warmup_iterations=2, measurement_iterations=10)
        benchmark = Conv2dBenchmark(config=config)
        return benchmark.run(spec, device="cpu")

    def test_result_has_timing_stats(self, sample_result):
        assert sample_result.timing is not None
        assert sample_result.timing.mean_ms > 0
        assert sample_result.timing.std_ms >= 0
        assert sample_result.timing.num_iterations == 10

    def test_result_has_gflops(self, sample_result):
        assert sample_result.gflops > 0

    def test_result_has_extra_fields(self, sample_result):
        assert 'in_channels' in sample_result.extra
        assert 'out_channels' in sample_result.extra
        assert 'kernel_size' in sample_result.extra

    def test_result_serializable(self, sample_result):
        import json

        d = sample_result.to_dict()
        json_str = json.dumps(d)

        assert json_str
        parsed = json.loads(json_str)
        assert parsed['success']


class TestFormatResultTable:
    """Tests for result table formatting."""

    def test_format_table(self):
        spec = create_conv2d_spec(
            in_channels=32,
            out_channels=32,
            height=28,
            width=28,
        )
        config = ExecutionConfig(warmup_iterations=2, measurement_iterations=5)
        benchmark = Conv2dBenchmark(config=config)
        result = benchmark.run(spec, device="cpu")

        table = format_result_table([result])

        assert "conv2d" in table
        assert "cpu" in table
        assert "OK" in table


class TestFLOPSCalculation:
    """Tests for FLOPS calculation accuracy."""

    def test_conv2d_flops_formula(self):
        spec = create_conv2d_spec(
            in_channels=64,
            out_channels=64,
            height=56,
            width=56,
            kernel_size=3,
        )

        # Conv2d FLOPs = 2 * H_out * W_out * K^2 * C_in * C_out
        # With same padding: H_out = H_in, W_out = W_in
        expected_flops = 2 * 56 * 56 * 9 * 64 * 64
        assert spec.flops == expected_flops

    def test_depthwise_has_fewer_flops(self):
        standard = create_conv2d_spec(
            in_channels=64,
            out_channels=64,
            height=56,
            width=56,
            kernel_size=3,
        )
        depthwise = create_conv2d_spec(
            in_channels=64,
            out_channels=64,
            height=56,
            width=56,
            kernel_size=3,
            groups=64,
        )

        # Depthwise should have 64x fewer FLOPs
        assert depthwise.flops < standard.flops


class TestLoadedSpecs:
    """Tests for specs loaded from YAML files."""

    def test_resnet_specs_present(self):
        specs = get_conv2d_specs(filter_tags=["resnet"])
        assert len(specs) >= 1

    def test_depthwise_specs_present(self):
        specs = get_conv2d_specs(filter_tags=["depthwise"])
        assert len(specs) >= 1

        # All should have groups > 1
        for spec in specs:
            assert spec.groups > 1

    def test_7x7_stem_present(self):
        specs = get_conv2d_specs(filter_tags=["stem"])
        assert len(specs) >= 1

        # Should have 7x7 kernel
        assert any(s.kernel_size == 7 for s in specs)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
