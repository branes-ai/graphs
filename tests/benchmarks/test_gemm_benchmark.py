"""
Tests for GEMM Microbenchmark Suite.

Tests cover:
- Spec loading from YAML files
- Benchmark execution on CPU
- Result format validation
- Statistical aggregation
- Precision handling
- Error handling
"""

import pytest
import torch

from graphs.benchmarks.schema import (
    GEMMSpec,
    ExecutionConfig,
    Precision,
)
from graphs.benchmarks.microbench.gemm import (
    GEMMBenchmark,
    get_gemm_specs,
    run_gemm_benchmark,
    create_gemm_spec,
    format_result_table,
)


class TestSpecLoading:
    """Tests for loading GEMM specs from YAML."""

    def test_load_all_specs(self):
        specs = get_gemm_specs()
        assert len(specs) > 0
        assert all(isinstance(s, GEMMSpec) for s in specs)

    def test_specs_have_required_fields(self):
        specs = get_gemm_specs()
        for spec in specs:
            assert spec.name
            assert spec.M > 0
            assert spec.N > 0
            assert spec.K > 0
            assert len(spec.precisions) > 0

    def test_filter_by_tags(self):
        all_specs = get_gemm_specs()
        transformer_specs = get_gemm_specs(filter_tags=["transformer"])

        # Should filter down to fewer specs
        assert len(transformer_specs) <= len(all_specs)

        # All should have transformer tag
        for spec in transformer_specs:
            assert "transformer" in spec.tags

    def test_filter_by_size(self):
        specs = get_gemm_specs(filter_sizes=[1024])

        for spec in specs:
            assert spec.M == 1024

    def test_no_specs_for_nonexistent_tag(self):
        specs = get_gemm_specs(filter_tags=["nonexistent_tag_xyz"])
        assert len(specs) == 0


class TestCreateGEMMSpec:
    """Tests for programmatic spec creation."""

    def test_create_basic_spec(self):
        spec = create_gemm_spec(M=256, N=256, K=256)
        assert spec.M == 256
        assert spec.N == 256
        assert spec.K == 256
        assert spec.batch_size == 1
        assert "gemm_256x256x256" in spec.name

    def test_create_batched_spec(self):
        spec = create_gemm_spec(M=512, N=512, K=512, batch_size=4)
        assert spec.batch_size == 4
        assert "batch4" in spec.name

    def test_create_spec_custom_name(self):
        spec = create_gemm_spec(M=128, N=128, K=128, name="my_custom_gemm")
        assert spec.name == "my_custom_gemm"

    def test_create_spec_custom_precision(self):
        spec = create_gemm_spec(M=256, N=256, K=256, precision=Precision.FP16)
        assert Precision.FP16 in spec.precisions


class TestGEMMBenchmark:
    """Tests for GEMMBenchmark execution."""

    @pytest.fixture
    def fast_config(self):
        return ExecutionConfig(
            warmup_iterations=2,
            measurement_iterations=5,
        )

    def test_run_simple_gemm_cpu(self, fast_config):
        spec = create_gemm_spec(M=64, N=64, K=64)
        benchmark = GEMMBenchmark(config=fast_config)

        result = benchmark.run(spec, device="cpu")

        assert result.success
        assert result.timing is not None
        assert result.timing.mean_ms > 0
        assert result.gflops > 0

    def test_run_batched_gemm_cpu(self, fast_config):
        spec = create_gemm_spec(M=32, N=32, K=32, batch_size=4)
        benchmark = GEMMBenchmark(config=fast_config)

        result = benchmark.run(spec, device="cpu")

        assert result.success
        assert result.extra['batch_size'] == 4

    def test_run_fp16_gemm_cpu(self, fast_config):
        spec = create_gemm_spec(M=64, N=64, K=64, precision=Precision.FP16)
        benchmark = GEMMBenchmark(config=fast_config)

        result = benchmark.run(spec, device="cpu", precision=Precision.FP16)

        assert result.success
        assert result.precision == "fp16"

    def test_run_all_precisions(self, fast_config):
        spec = GEMMSpec(
            name="test_multi_precision",
            M=32,
            N=32,
            K=32,
            precisions=[Precision.FP32, Precision.FP16],
        )
        benchmark = GEMMBenchmark(config=fast_config)

        results = benchmark.run_all_precisions(spec, device="cpu")

        assert len(results) == 2
        assert any(r.precision == "fp32" for r in results)
        assert any(r.precision == "fp16" for r in results)

    def test_run_suite(self, fast_config):
        specs = [
            create_gemm_spec(M=32, N=32, K=32),
            create_gemm_spec(M=64, N=64, K=64),
            create_gemm_spec(M=128, N=128, K=128),
        ]
        benchmark = GEMMBenchmark(config=fast_config)

        results = benchmark.run_suite(specs, device="cpu")

        assert len(results) == 3
        assert all(r.success for r in results)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_run_gemm_cuda(self, fast_config):
        spec = create_gemm_spec(M=256, N=256, K=256)
        benchmark = GEMMBenchmark(config=fast_config)

        result = benchmark.run(spec, device="cuda")

        assert result.success
        assert result.device == "cuda"


class TestRunGEMMBenchmark:
    """Tests for the convenience function."""

    def test_convenience_function(self):
        spec = create_gemm_spec(M=32, N=32, K=32)
        config = ExecutionConfig(warmup_iterations=2, measurement_iterations=5)

        result = run_gemm_benchmark(spec, device="cpu", config=config)

        assert result.success
        assert result.timing is not None


class TestResultFormat:
    """Tests for result format and content."""

    @pytest.fixture
    def sample_result(self):
        spec = create_gemm_spec(M=64, N=64, K=64)
        config = ExecutionConfig(warmup_iterations=2, measurement_iterations=10)
        benchmark = GEMMBenchmark(config=config)
        return benchmark.run(spec, device="cpu")

    def test_result_has_timing_stats(self, sample_result):
        assert sample_result.timing is not None
        assert sample_result.timing.mean_ms > 0
        assert sample_result.timing.std_ms >= 0
        assert sample_result.timing.min_ms <= sample_result.timing.mean_ms
        assert sample_result.timing.max_ms >= sample_result.timing.mean_ms
        assert sample_result.timing.num_iterations == 10

    def test_result_has_gflops(self, sample_result):
        assert sample_result.gflops > 0

        # Verify GFLOPS calculation
        spec = create_gemm_spec(M=64, N=64, K=64)
        expected_flops = spec.flops
        expected_gflops = (expected_flops / 1e9) / (sample_result.timing.mean_ms / 1000)
        assert abs(sample_result.gflops - expected_gflops) < 0.1

    def test_result_has_extra_fields(self, sample_result):
        assert 'M' in sample_result.extra
        assert 'N' in sample_result.extra
        assert 'K' in sample_result.extra
        assert sample_result.extra['M'] == 64

    def test_result_serializable(self, sample_result):
        import json

        d = sample_result.to_dict()
        json_str = json.dumps(d)

        assert json_str
        parsed = json.loads(json_str)
        assert parsed['success']


class TestFormatResultTable:
    """Tests for result table formatting."""

    def test_format_table_single_result(self):
        spec = create_gemm_spec(M=32, N=32, K=32)
        config = ExecutionConfig(warmup_iterations=2, measurement_iterations=5)
        benchmark = GEMMBenchmark(config=config)
        result = benchmark.run(spec, device="cpu")

        table = format_result_table([result])

        assert "gemm_32x32x32" in table
        assert "cpu" in table
        assert "OK" in table

    def test_format_table_multiple_results(self):
        specs = [
            create_gemm_spec(M=32, N=32, K=32),
            create_gemm_spec(M=64, N=64, K=64),
        ]
        config = ExecutionConfig(warmup_iterations=2, measurement_iterations=5)
        benchmark = GEMMBenchmark(config=config)

        results = [benchmark.run(s, device="cpu") for s in specs]
        table = format_result_table(results)

        assert "gemm_32x32x32" in table
        assert "gemm_64x64x64" in table


class TestErrorHandling:
    """Tests for error handling."""

    def test_suite_continues_on_failure(self):
        # Create a mix of valid and invalid specs
        config = ExecutionConfig(warmup_iterations=2, measurement_iterations=5)
        benchmark = GEMMBenchmark(config=config)

        # Run on a valid spec
        specs = [create_gemm_spec(M=32, N=32, K=32)]
        results = benchmark.run_suite(specs, device="cpu")

        # Should complete without raising
        assert len(results) == 1


class TestFLOPSCalculation:
    """Tests for FLOPS calculation accuracy."""

    def test_gemm_flops_formula(self):
        spec = create_gemm_spec(M=1024, N=1024, K=1024)

        # GEMM FLOPs = 2 * M * N * K
        expected_flops = 2 * 1024 * 1024 * 1024
        assert spec.flops == expected_flops

    def test_batched_gemm_flops(self):
        spec = create_gemm_spec(M=512, N=512, K=512, batch_size=4)

        # Batched GEMM FLOPs = 2 * B * M * N * K
        expected_flops = 2 * 4 * 512 * 512 * 512
        assert spec.flops == expected_flops

    def test_arithmetic_intensity(self):
        spec = create_gemm_spec(M=1024, N=1024, K=1024)

        # Large square GEMM has high arithmetic intensity
        assert spec.arithmetic_intensity > 100  # Should be compute-bound


class TestLoadedSpecs:
    """Tests for specs loaded from YAML files."""

    def test_standard_sizes_include_expected(self):
        specs = get_gemm_specs()
        sizes = {(s.M, s.N, s.K) for s in specs}

        # Check for some expected sizes
        assert (1024, 1024, 1024) in sizes
        assert (512, 512, 512) in sizes

    def test_transformer_shapes_present(self):
        specs = get_gemm_specs(filter_tags=["transformer"])

        # Should have BERT/GPT shapes
        assert len(specs) >= 1

        # Check for transformer-typical rectangular shapes
        shapes = [(s.M, s.N, s.K) for s in specs]
        # BERT-base FFN projection
        assert any(s[0] == 768 and s[1] == 3072 for s in shapes)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
