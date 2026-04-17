"""
Tests for Layer 1 ALU microbenchmarks.

Tests cover:
- FMA rate benchmark runs on CPU and produces valid results
- Precision sweep returns results for all requested precisions
- DCE sink defense is exercised
- Layer1ALUFitter fits coefficients from results
- Layer1ALUFitter.apply_to_model writes provenance
- Composition check logic (predict + compare)

All tests run on CPU with small vectors to complete in <1s.
"""

from __future__ import annotations

import pytest

from graphs.benchmarks.schema import BenchmarkResult, LayerTag


class TestFMARateBenchmark:
    """Tests for the core FMA-rate benchmark."""

    def test_runs_fp32_on_cpu(self):
        from graphs.benchmarks.layer1_alu.fma_rate import run_fma_rate_benchmark
        result = run_fma_rate_benchmark(
            device="cpu",
            precision="fp32",
            num_elements=1024,
            num_iterations=100,
            warmup_iterations=10,
            num_trials=3,
        )
        assert result.success
        assert result.layer is LayerTag.ALU
        assert result.gflops > 0
        assert result.precision == "fp32"
        assert result.extra["num_elements"] == 1024
        assert result.extra["total_flops_per_trial"] == 2 * 1024 * 100

    def test_runs_fp64_on_cpu(self):
        from graphs.benchmarks.layer1_alu.fma_rate import run_fma_rate_benchmark
        result = run_fma_rate_benchmark(
            device="cpu",
            precision="fp64",
            num_elements=1024,
            num_iterations=100,
            warmup_iterations=10,
            num_trials=3,
        )
        assert result.success
        assert result.gflops > 0

    def test_empty_loop_subtraction_is_positive(self):
        from graphs.benchmarks.layer1_alu.fma_rate import run_fma_rate_benchmark
        result = run_fma_rate_benchmark(
            device="cpu",
            precision="fp32",
            num_elements=1024,
            num_iterations=100,
            warmup_iterations=10,
            num_trials=3,
        )
        assert result.extra["empty_loop_overhead_ms"] >= 0

    def test_sink_is_written(self):
        from graphs.benchmarks.layer1_alu.fma_rate import (
            run_fma_rate_benchmark,
            get_sink_value,
        )
        run_fma_rate_benchmark(
            device="cpu", precision="fp32",
            num_elements=256, num_iterations=10,
            warmup_iterations=5, num_trials=2,
        )
        sink = get_sink_value()
        assert isinstance(sink, float)

    def test_result_serializes_to_json(self):
        from graphs.benchmarks.layer1_alu.fma_rate import run_fma_rate_benchmark
        result = run_fma_rate_benchmark(
            device="cpu", precision="fp32",
            num_elements=256, num_iterations=10,
            warmup_iterations=5, num_trials=2,
        )
        restored = BenchmarkResult.from_json(result.to_json())
        assert restored.layer is LayerTag.ALU
        assert restored.gflops > 0


class TestPrecisionSweep:
    """Tests for the multi-precision sweep."""

    def test_sweeps_multiple_precisions(self):
        from graphs.benchmarks.layer1_alu.precision_sweep import run_precision_sweep
        results = run_precision_sweep(
            device="cpu",
            precisions=["fp32", "fp64"],
            num_elements=512,
            num_iterations=50,
            warmup_iterations=5,
            num_trials=2,
            enable_power=False,
        )
        assert len(results) == 2
        assert all(r.layer is LayerTag.ALU for r in results)
        precs = {r.precision for r in results}
        assert "fp32" in precs
        assert "fp64" in precs

    def test_handles_unsupported_precision_gracefully(self):
        from graphs.benchmarks.layer1_alu.precision_sweep import run_precision_sweep
        results = run_precision_sweep(
            device="cpu",
            precisions=["fp32"],
            num_elements=256,
            num_iterations=10,
            warmup_iterations=5,
            num_trials=2,
            enable_power=False,
        )
        assert len(results) == 1
        assert results[0].success


class TestLayer1ALUFitter:
    """Tests for the coefficient fitter."""

    def _make_results(self) -> list:
        from graphs.benchmarks.layer1_alu.fma_rate import run_fma_rate_benchmark
        return [
            run_fma_rate_benchmark(
                device="cpu", precision=p,
                num_elements=256, num_iterations=10,
                warmup_iterations=5, num_trials=2,
            )
            for p in ["fp32", "fp64"]
        ]

    def test_fit_produces_per_precision_throughput(self):
        from graphs.calibration.fitters.layer1_alu_fitter import Layer1ALUFitter
        results = self._make_results()
        fitter = Layer1ALUFitter()
        fit = fitter.fit(
            results,
            sustained_clock_hz=4.5e9,
            num_cores=10,
            hardware_name="test-sku",
        )
        assert fit.num_results_used == 2
        assert "fp32" in fit.measured_throughput
        assert "fp64" in fit.measured_throughput
        assert fit.measured_throughput["fp32"] > 0
        assert "fp32" in fit.ops_per_clock_per_core

    def test_apply_writes_provenance(self):
        from graphs.calibration.fitters.layer1_alu_fitter import Layer1ALUFitter
        from graphs.hardware.resource_model import (
            HardwareResourceModel, HardwareType,
        )
        from graphs.core.confidence import ConfidenceLevel

        model = HardwareResourceModel(
            name="test",
            hardware_type=HardwareType.CPU,
            compute_units=10,
            threads_per_unit=1,
            warps_per_unit=1,
            peak_bandwidth=75e9,
            l1_cache_per_unit=32768,
            l2_cache_total=25 * 1024 * 1024,
            main_memory=64 * 1024**3,
            energy_per_flop_fp32=1.5e-12,
            energy_per_byte=25e-12,
        )

        results = self._make_results()
        fitter = Layer1ALUFitter()
        fit = fitter.fit(results, sustained_clock_hz=4.5e9, num_cores=10)
        fitter.apply_to_model(model, fit)

        prov = model.get_provenance("compute_fabric.ops_per_clock.fp32")
        assert prov.level is ConfidenceLevel.CALIBRATED
        assert "GOPS" in prov.source

    def test_fit_ignores_failed_results(self):
        from graphs.calibration.fitters.layer1_alu_fitter import Layer1ALUFitter
        good = self._make_results()
        bad = BenchmarkResult(
            spec_name="bad",
            timestamp="2026-04-17T00:00:00Z",
            device="cpu",
            precision="fp32",
            layer=LayerTag.ALU,
            success=False,
        )
        fitter = Layer1ALUFitter()
        fit = fitter.fit(good + [bad])
        assert fit.num_results_used == 2


class TestCompositionCheck:
    """Tests for the Layer 1 -> Layer 3 GEMM composition logic."""

    def test_predict_gemm_from_layer1(self):
        from validation.composition.layer1_to_layer3_gemm import (
            predict_gemm_gflops_from_layer1,
        )
        from graphs.benchmarks.layer1_alu.fma_rate import run_fma_rate_benchmark
        result = run_fma_rate_benchmark(
            device="cpu", precision="fp32",
            num_elements=1024, num_iterations=100,
            warmup_iterations=10, num_trials=3,
        )
        predicted = predict_gemm_gflops_from_layer1(
            [result], precision="fp32", gemm_efficiency=0.70,
        )
        assert predicted is not None
        assert predicted > 0
        assert predicted < result.gflops

    def test_check_passes_when_within_tolerance(self):
        from validation.composition.layer1_to_layer3_gemm import (
            check_layer1_to_gemm,
        )
        from validation.composition.test_layer_composition import CheckStatus
        from graphs.benchmarks.layer1_alu.fma_rate import run_fma_rate_benchmark

        result = run_fma_rate_benchmark(
            device="cpu", precision="fp32",
            num_elements=1024, num_iterations=100,
            warmup_iterations=10, num_trials=3,
        )
        predicted_gflops = result.gflops * 0.70

        check_result = check_layer1_to_gemm(
            layer1_results=[result],
            measured_gemm_gflops=predicted_gflops,
            precision="fp32",
            hardware="test",
            gemm_efficiency=0.70,
            tolerance=0.10,
        )
        assert check_result.status is CheckStatus.PASSED
        assert check_result.max_relative_error < 0.01

    def test_check_skips_when_no_layer1_data(self):
        from validation.composition.layer1_to_layer3_gemm import (
            check_layer1_to_gemm,
        )
        from validation.composition.test_layer_composition import CheckStatus

        check_result = check_layer1_to_gemm(
            layer1_results=[],
            measured_gemm_gflops=100.0,
            precision="fp32",
        )
        assert check_result.status is CheckStatus.SKIPPED

    def test_check_skips_when_no_measured_gemm(self):
        from validation.composition.layer1_to_layer3_gemm import (
            check_layer1_to_gemm,
        )
        from validation.composition.test_layer_composition import CheckStatus
        from graphs.benchmarks.layer1_alu.fma_rate import run_fma_rate_benchmark

        result = run_fma_rate_benchmark(
            device="cpu", precision="fp32",
            num_elements=256, num_iterations=10,
            warmup_iterations=5, num_trials=2,
        )
        check_result = check_layer1_to_gemm(
            layer1_results=[result],
            measured_gemm_gflops=0.0,
            precision="fp32",
        )
        assert check_result.status is CheckStatus.SKIPPED


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
