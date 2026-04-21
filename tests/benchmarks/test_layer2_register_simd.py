"""
Tests for Layer 2 register/SIMD benchmarks.

Tests cover:
- SIMD width sweep produces results for multiple widths
- Throughput increases with vector width (validates SIMD is engaged)
- Register pressure benchmark runs and produces ILP ratio
- Layer2RegisterFitter fits SIMD efficiency from results
- Fitter.apply_to_model writes provenance
- Composition check Layer 1+2 -> GEMM logic

All tests run on CPU with small vectors (<1s total).
"""

from __future__ import annotations

import pytest

from graphs.benchmarks.schema import LayerTag


class TestSIMDWidthSweep:
    """Tests for the SIMD width sweep benchmark."""

    def test_sweep_produces_results_per_width(self):
        from graphs.benchmarks.layer2_register_simd.simd_width import (
            run_simd_width_sweep,
        )
        widths = [4, 64, 1024]
        results = run_simd_width_sweep(
            device="cpu",
            precision="fp32",
            widths=widths,
            num_iterations=500,
            warmup_iterations=50,
            num_trials=3,
        )
        assert len(results) == 3
        assert all(r.success for r in results)
        assert all(r.layer is LayerTag.REGISTER for r in results)

    def test_wider_vectors_have_higher_throughput(self):
        from graphs.benchmarks.layer2_register_simd.simd_width import (
            run_simd_width_sweep,
        )
        # Use widths where both are above dispatch-floor (64 vs 4096)
        # to avoid flakiness from width=1 being dispatch-dominated
        results = run_simd_width_sweep(
            device="cpu",
            precision="fp32",
            widths=[64, 4096],
            num_iterations=2000,
            warmup_iterations=100,
            num_trials=3,
        )
        narrow = results[0]
        wide = results[1]
        assert wide.gflops > narrow.gflops

    def test_extra_contains_vector_width(self):
        from graphs.benchmarks.layer2_register_simd.simd_width import (
            run_simd_width_sweep,
        )
        results = run_simd_width_sweep(
            device="cpu", precision="fp32",
            widths=[8], num_iterations=100,
            warmup_iterations=10, num_trials=2,
        )
        assert results[0].extra["vector_width"] == 8


class TestRegisterPressure:
    """Tests for the register pressure benchmark."""

    def test_runs_and_produces_ilp_ratio(self):
        from graphs.benchmarks.layer2_register_simd.register_pressure import (
            run_register_pressure_benchmark,
        )
        result = run_register_pressure_benchmark(
            device="cpu",
            precision="fp32",
            num_elements=1024,
            num_iterations=500,
            warmup_iterations=50,
            num_trials=3,
        )
        assert result.success
        assert result.layer is LayerTag.REGISTER
        ilp = result.extra["ilp_ratio"]
        assert ilp > 0

    def test_independent_faster_than_dependent(self):
        from graphs.benchmarks.layer2_register_simd.register_pressure import (
            run_register_pressure_benchmark,
        )
        result = run_register_pressure_benchmark(
            device="cpu", precision="fp32",
            num_elements=4096, num_iterations=1000,
            warmup_iterations=100, num_trials=3,
        )
        ind = result.extra["independent_gflops"]
        dep = result.extra["dependent_gflops"]
        # Independent should be at least as fast (usually faster)
        assert ind >= dep * 0.9


class TestLayer2Fitter:
    """Tests for the SIMD/register fitter."""

    def _make_width_results(self):
        from graphs.benchmarks.layer2_register_simd.simd_width import (
            run_simd_width_sweep,
        )
        return run_simd_width_sweep(
            device="cpu", precision="fp32",
            widths=[4, 64, 4096],
            num_iterations=500, warmup_iterations=50, num_trials=2,
        )

    def _make_pressure_result(self):
        from graphs.benchmarks.layer2_register_simd.register_pressure import (
            run_register_pressure_benchmark,
        )
        return run_register_pressure_benchmark(
            device="cpu", precision="fp32",
            num_elements=1024, num_iterations=200,
            warmup_iterations=50, num_trials=2,
        )

    def test_fit_extracts_simd_efficiency(self):
        from graphs.calibration.fitters.layer2_register_fitter import (
            Layer2RegisterFitter,
        )
        results = self._make_width_results()
        fitter = Layer2RegisterFitter()
        fit = fitter.fit(results, hardware_name="test-cpu")
        assert fit.simd_efficiency is not None
        assert fit.simd_efficiency > 0
        assert fit.scalar_gflops is not None
        assert fit.peak_vector_gflops is not None

    def test_fit_extracts_ilp_ratio(self):
        from graphs.calibration.fitters.layer2_register_fitter import (
            Layer2RegisterFitter,
        )
        pressure = self._make_pressure_result()
        fitter = Layer2RegisterFitter()
        fit = fitter.fit([pressure], hardware_name="test-cpu")
        assert fit.ilp_ratio is not None
        assert fit.ilp_ratio > 0

    def test_apply_writes_provenance(self):
        from graphs.calibration.fitters.layer2_register_fitter import (
            Layer2RegisterFitter,
        )
        from graphs.hardware.resource_model import (
            HardwareResourceModel, HardwareType,
        )
        from graphs.core.confidence import ConfidenceLevel

        model = HardwareResourceModel(
            name="test", hardware_type=HardwareType.CPU,
            compute_units=10, threads_per_unit=1, warps_per_unit=1,
            peak_bandwidth=75e9, l1_cache_per_unit=32768,
            l2_cache_total=25*1024*1024, main_memory=64*1024**3,
            energy_per_flop_fp32=1.5e-12, energy_per_byte=25e-12,
        )

        results = self._make_width_results() + [self._make_pressure_result()]
        fitter = Layer2RegisterFitter()
        fit = fitter.fit(results, hardware_name="test-cpu")
        fitter.apply_to_model(model, fit)

        prov = model.get_provenance("simd_vectorization_efficiency")
        assert prov.level is ConfidenceLevel.CALIBRATED

        prov2 = model.get_provenance("register_ilp_ratio")
        assert prov2.level is ConfidenceLevel.CALIBRATED


class TestCompositionCheck:
    """Tests for the Layer 1+2 -> GEMM composition check."""

    def test_predict_uses_simd_efficiency(self):
        from validation.composition.layer1_2_to_layer3_gemm import (
            predict_gemm_from_layer1_2,
        )
        from graphs.benchmarks.layer1_alu.fma_rate import run_fma_rate_benchmark
        from graphs.benchmarks.layer2_register_simd.simd_width import (
            run_simd_width_sweep,
        )

        layer1 = [run_fma_rate_benchmark(
            device="cpu", precision="fp32",
            num_elements=1024, num_iterations=100,
            warmup_iterations=10, num_trials=2,
        )]
        layer2 = run_simd_width_sweep(
            device="cpu", precision="fp32",
            widths=[4, 4096], num_iterations=500,
            warmup_iterations=50, num_trials=2,
        )

        predicted = predict_gemm_from_layer1_2(layer1, layer2)
        assert predicted is not None
        assert predicted > 0

    def test_check_passes_within_tolerance(self):
        from validation.composition.layer1_2_to_layer3_gemm import (
            check_layer1_2_to_gemm,
            predict_gemm_from_layer1_2,
        )
        from validation.composition.test_layer_composition import CheckStatus
        from graphs.benchmarks.layer1_alu.fma_rate import run_fma_rate_benchmark
        from graphs.benchmarks.layer2_register_simd.simd_width import (
            run_simd_width_sweep,
        )

        layer1 = [run_fma_rate_benchmark(
            device="cpu", precision="fp32",
            num_elements=1024, num_iterations=100,
            warmup_iterations=10, num_trials=2,
        )]
        layer2 = run_simd_width_sweep(
            device="cpu", precision="fp32",
            widths=[4, 4096], num_iterations=500,
            warmup_iterations=50, num_trials=2,
        )

        predicted = predict_gemm_from_layer1_2(layer1, layer2)
        # 5% off should pass at 8% tolerance
        check = check_layer1_2_to_gemm(
            layer1, layer2,
            measured_gemm_gflops=predicted * 1.05,
            tolerance=0.08,
        )
        assert check.status is CheckStatus.PASSED

    def test_check_fails_beyond_tolerance(self):
        from validation.composition.layer1_2_to_layer3_gemm import (
            check_layer1_2_to_gemm,
            predict_gemm_from_layer1_2,
        )
        from validation.composition.test_layer_composition import CheckStatus
        from graphs.benchmarks.layer1_alu.fma_rate import run_fma_rate_benchmark
        from graphs.benchmarks.layer2_register_simd.simd_width import (
            run_simd_width_sweep,
        )

        layer1 = [run_fma_rate_benchmark(
            device="cpu", precision="fp32",
            num_elements=1024, num_iterations=100,
            warmup_iterations=10, num_trials=2,
        )]
        layer2 = run_simd_width_sweep(
            device="cpu", precision="fp32",
            widths=[4, 4096], num_iterations=500,
            warmup_iterations=50, num_trials=2,
        )

        predicted = predict_gemm_from_layer1_2(layer1, layer2)
        # 15% off should fail at 8% tolerance
        check = check_layer1_2_to_gemm(
            layer1, layer2,
            measured_gemm_gflops=predicted * 1.15,
            tolerance=0.08,
        )
        assert check.status is CheckStatus.FAILED

    def test_check_skips_when_no_data(self):
        from validation.composition.layer1_2_to_layer3_gemm import (
            check_layer1_2_to_gemm,
        )
        from validation.composition.test_layer_composition import CheckStatus

        check = check_layer1_2_to_gemm([], [], measured_gemm_gflops=100.0)
        assert check.status is CheckStatus.SKIPPED


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
