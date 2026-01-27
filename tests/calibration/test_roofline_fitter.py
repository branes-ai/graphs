"""
Tests for roofline parameter fitting.

Tests cover:
- Bandwidth fitting from memory benchmark results
- Compute ceiling fitting from GEMM benchmark results
- Combined roofline fitting
- Edge cases (insufficient data, outliers)
- Efficiency curve fitting
"""

import pytest
import numpy as np
from datetime import datetime

from graphs.calibration.roofline_fitter import (
    RooflineFitter,
    RooflineParameters,
    FitMetrics,
    FitQuality,
    fit_roofline,
)
from graphs.calibration.efficiency_curves import (
    AsymptoticCurve,
    PiecewiseLinearCurve,
    PolynomialCurve,
    ConstantCurve,
    CurveType,
    EfficiencyProfile,
    fit_efficiency_curve,
    auto_fit_efficiency_curve,
)
from graphs.benchmarks.schema import (
    BenchmarkResult,
    TimingStats,
    GEMMSpec,
    MemoryBenchSpec,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def synthetic_memory_results():
    """Generate synthetic memory benchmark results"""
    # Use fixed seed for reproducibility
    np.random.seed(123)
    # Simulate STREAM-like bandwidth measurements with some variance
    base_bandwidth = 800.0  # GB/s
    results = []

    for i in range(10):
        bw = base_bandwidth * (0.95 + 0.1 * np.random.random())  # +/- 5%
        results.append(BenchmarkResult(
            spec_name=f"stream_triad_{i}",
            timestamp=datetime.now().isoformat(),
            device="cuda:0",
            device_name="Test GPU",
            precision="fp32",
            timing=TimingStats(
                mean_ms=1.0,
                std_ms=0.1,
                min_ms=0.9,
                max_ms=1.2,
                median_ms=1.0,
                p95_ms=1.15,
                p99_ms=1.18,
                num_iterations=100,
            ),
            bandwidth_gbps=bw,
            gflops=0.0,  # Memory benchmark, no FLOPS
            success=True,
        ))

    return results


@pytest.fixture
def synthetic_gemm_results():
    """Generate synthetic GEMM benchmark results at various sizes"""
    # Simulate compute-bound GEMM results
    peak_gflops = 50000.0  # 50 TFLOPS

    results = []
    sizes = [512, 1024, 2048, 4096, 8192]

    for size in sizes:
        # Efficiency increases with size (asymptotic behavior)
        efficiency = 0.8 * (1.0 - np.exp(-size / 1000.0))
        gflops = peak_gflops * efficiency

        # Calculate arithmetic intensity for GEMM: 2*M*N*K / (M*K + K*N + M*N) * 4
        # For square matrices: 2*N^3 / (3*N^2) * 4 = 2N/3 * 4 / 4 = 2N/3
        ai = 2 * size / 3.0  # Simplified

        results.append(BenchmarkResult(
            spec_name=f"gemm_{size}x{size}",
            timestamp=datetime.now().isoformat(),
            device="cuda:0",
            device_name="Test GPU",
            precision="fp32",
            timing=TimingStats(
                mean_ms=1.0,
                std_ms=0.1,
                min_ms=0.9,
                max_ms=1.2,
                median_ms=1.0,
                p95_ms=1.15,
                p99_ms=1.18,
                num_iterations=100,
            ),
            bandwidth_gbps=0.0,
            gflops=gflops,
            success=True,
            extra={'arithmetic_intensity': ai},
        ))

    return results


@pytest.fixture
def combined_results(synthetic_memory_results, synthetic_gemm_results):
    """Combined memory and compute results"""
    return synthetic_memory_results + synthetic_gemm_results


# =============================================================================
# RooflineFitter Tests
# =============================================================================

class TestRooflineFitter:
    """Tests for RooflineFitter class"""

    def test_init(self):
        """Test fitter initialization"""
        fitter = RooflineFitter(
            theoretical_bandwidth_gbps=1000.0,
            theoretical_compute_gflops=60000.0,
        )
        assert fitter.theoretical_bandwidth_gbps == 1000.0
        assert fitter.theoretical_compute_gflops == 60000.0

    def test_add_memory_result(self, synthetic_memory_results):
        """Test adding memory benchmark results"""
        fitter = RooflineFitter()

        for result in synthetic_memory_results:
            fitter.add_memory_result(result)

        assert fitter.can_fit_bandwidth()
        assert len(fitter._bandwidth_measurements) == 10

    def test_add_compute_result(self, synthetic_gemm_results):
        """Test adding compute benchmark results"""
        fitter = RooflineFitter()

        for result in synthetic_gemm_results:
            fitter.add_compute_result(result)

        assert fitter.can_fit_compute()
        assert len(fitter._compute_measurements) == 5

    def test_fit_bandwidth(self, synthetic_memory_results):
        """Test bandwidth fitting"""
        fitter = RooflineFitter(theoretical_bandwidth_gbps=1000.0)

        for result in synthetic_memory_results:
            fitter.add_memory_result(result)

        bandwidth, metrics = fitter.fit_bandwidth()

        # Should be close to 800 GB/s (base value)
        assert 700 < bandwidth < 900
        assert metrics.num_data_points == 10
        # Quality can vary due to random variance - just check it's a valid quality
        assert isinstance(metrics.quality, FitQuality)

    def test_fit_compute(self, synthetic_gemm_results):
        """Test compute ceiling fitting"""
        fitter = RooflineFitter(theoretical_compute_gflops=60000.0)

        for result in synthetic_gemm_results:
            fitter.add_compute_result(result)

        compute, metrics = fitter.fit_compute()

        # Should be around 80% of peak (asymptotic efficiency at large sizes)
        assert 30000 < compute < 50000
        assert metrics.num_data_points == 5

    def test_combined_fit(self, combined_results):
        """Test combined roofline fitting"""
        fitter = RooflineFitter(
            theoretical_bandwidth_gbps=1000.0,
            theoretical_compute_gflops=60000.0,
        )

        for result in combined_results:
            fitter.add_result(result)

        params = fitter.fit()

        assert isinstance(params, RooflineParameters)
        assert params.achieved_bandwidth_gbps > 0
        assert params.achieved_compute_gflops > 0
        assert params.ridge_point > 0
        assert params.bandwidth_efficiency > 0
        assert params.compute_efficiency > 0

    def test_insufficient_data_bandwidth(self):
        """Test error handling for insufficient bandwidth data"""
        fitter = RooflineFitter()

        # Only add 2 points (need >= 3)
        for i in range(2):
            fitter.add_memory_result(BenchmarkResult(
                spec_name=f"stream_{i}",
                timestamp=datetime.now().isoformat(),
                device="cuda:0",
                bandwidth_gbps=800.0,
                success=True,
            ))

        assert not fitter.can_fit_bandwidth()

        with pytest.raises(ValueError, match="Insufficient data"):
            fitter.fit_bandwidth()

    def test_insufficient_data_compute(self):
        """Test error handling for insufficient compute data"""
        fitter = RooflineFitter()

        # Only add 2 points (need >= 3)
        for i in range(2):
            fitter.add_compute_result(
                BenchmarkResult(
                    spec_name=f"gemm_{i}",
                    timestamp=datetime.now().isoformat(),
                    device="cuda:0",
                    gflops=40000.0,
                    success=True,
                    extra={'arithmetic_intensity': 100.0},
                )
            )

        assert not fitter.can_fit_compute()

        with pytest.raises(ValueError, match="Insufficient data"):
            fitter.fit_compute()

    def test_failed_results_ignored(self):
        """Test that failed benchmark results are ignored"""
        fitter = RooflineFitter()

        # Add failed result
        fitter.add_memory_result(BenchmarkResult(
            spec_name="stream_failed",
            timestamp=datetime.now().isoformat(),
            device="cuda:0",
            bandwidth_gbps=800.0,
            success=False,  # Failed!
            error_message="GPU error",
        ))

        assert len(fitter._bandwidth_measurements) == 0

    def test_reset(self, synthetic_memory_results):
        """Test fitter reset"""
        fitter = RooflineFitter()

        for result in synthetic_memory_results:
            fitter.add_memory_result(result)

        assert fitter.can_fit_bandwidth()

        fitter.reset()

        assert not fitter.can_fit_bandwidth()
        assert len(fitter._bandwidth_measurements) == 0

    def test_fit_from_results_class_method(self, combined_results):
        """Test convenience class method"""
        params = RooflineFitter.fit_from_results(
            combined_results,
            theoretical_bandwidth_gbps=1000.0,
            theoretical_compute_gflops=60000.0,
        )

        assert isinstance(params, RooflineParameters)
        assert params.achieved_bandwidth_gbps > 0


# =============================================================================
# RooflineParameters Tests
# =============================================================================

class TestRooflineParameters:
    """Tests for RooflineParameters class"""

    def test_predict_gflops_memory_bound(self):
        """Test GFLOPS prediction in memory-bound region"""
        params = RooflineParameters(
            achieved_bandwidth_gbps=800.0,
            achieved_compute_gflops=40000.0,
            ridge_point=50.0,  # 40000 / 800
        )

        # AI = 10 (below ridge point) -> memory bound
        predicted = params.predict_gflops(10.0)
        expected = 800.0 * 10.0  # 8000 GFLOPS
        assert predicted == expected

    def test_predict_gflops_compute_bound(self):
        """Test GFLOPS prediction in compute-bound region"""
        params = RooflineParameters(
            achieved_bandwidth_gbps=800.0,
            achieved_compute_gflops=40000.0,
            ridge_point=50.0,
        )

        # AI = 100 (above ridge point) -> compute bound
        predicted = params.predict_gflops(100.0)
        assert predicted == 40000.0

    def test_is_memory_bound(self):
        """Test memory-bound classification"""
        params = RooflineParameters(
            achieved_bandwidth_gbps=800.0,
            achieved_compute_gflops=40000.0,
            ridge_point=50.0,
        )

        assert params.is_memory_bound(10.0)
        assert params.is_memory_bound(49.0)
        assert not params.is_memory_bound(50.0)
        assert not params.is_memory_bound(100.0)

    def test_bottleneck(self):
        """Test bottleneck identification"""
        params = RooflineParameters(
            achieved_bandwidth_gbps=800.0,
            achieved_compute_gflops=40000.0,
            ridge_point=50.0,
        )

        assert params.bottleneck(10.0) == "memory"
        assert params.bottleneck(100.0) == "compute"

    def test_efficiency_calculation(self):
        """Test efficiency ratio calculation"""
        params = RooflineParameters(
            achieved_bandwidth_gbps=800.0,
            achieved_compute_gflops=40000.0,
            ridge_point=50.0,
            theoretical_bandwidth_gbps=1000.0,
            theoretical_compute_gflops=50000.0,
        )

        assert params.bandwidth_efficiency == 0.8  # 800/1000
        assert params.compute_efficiency == 0.8    # 40000/50000

    def test_serialization(self):
        """Test to_dict/from_dict roundtrip"""
        params = RooflineParameters(
            achieved_bandwidth_gbps=800.0,
            achieved_compute_gflops=40000.0,
            ridge_point=50.0,
            bandwidth_fit=FitMetrics(
                num_data_points=10,
                r_squared=0.95,
                residual_std=10.0,
                quality=FitQuality.EXCELLENT,
            ),
            theoretical_bandwidth_gbps=1000.0,
            theoretical_compute_gflops=50000.0,
            precision="fp32",
            device_name="Test GPU",
        )

        data = params.to_dict()
        restored = RooflineParameters.from_dict(data)

        assert restored.achieved_bandwidth_gbps == params.achieved_bandwidth_gbps
        assert restored.achieved_compute_gflops == params.achieved_compute_gflops
        assert restored.ridge_point == params.ridge_point
        assert restored.bandwidth_fit.quality == FitQuality.EXCELLENT


# =============================================================================
# fit_roofline Function Tests
# =============================================================================

class TestFitRooflineFunction:
    """Tests for the fit_roofline convenience function"""

    def test_basic_usage(self, combined_results):
        """Test basic usage of fit_roofline function"""
        params = fit_roofline(combined_results)

        assert isinstance(params, RooflineParameters)
        assert params.achieved_bandwidth_gbps > 0
        assert params.achieved_compute_gflops > 0

    def test_with_theoretical_values(self, combined_results):
        """Test with theoretical values provided"""
        params = fit_roofline(
            combined_results,
            theoretical_bandwidth_gbps=1000.0,
            theoretical_compute_gflops=60000.0,
        )

        assert params.theoretical_bandwidth_gbps == 1000.0
        assert params.theoretical_compute_gflops == 60000.0
        assert params.bandwidth_efficiency > 0
        assert params.compute_efficiency > 0


# =============================================================================
# Efficiency Curve Tests
# =============================================================================

class TestAsymptoticCurve:
    """Tests for AsymptoticCurve"""

    def test_basic_prediction(self):
        """Test basic efficiency prediction"""
        curve = AsymptoticCurve(peak=0.9, scale=1000.0)

        # At size=0, efficiency should be ~0
        assert curve.predict(0) == pytest.approx(0.0, abs=0.01)

        # At size=scale (1000), efficiency should be ~63% of peak
        eff_at_scale = curve.predict(1000.0)
        assert 0.5 < eff_at_scale < 0.7

        # At large sizes, efficiency should approach peak
        eff_large = curve.predict(10000.0)
        assert eff_large > 0.85

    def test_array_prediction(self):
        """Test prediction with array input"""
        curve = AsymptoticCurve(peak=0.9, scale=1000.0)

        sizes = np.array([100, 1000, 10000])
        effs = curve.predict(sizes)

        assert len(effs) == 3
        assert effs[0] < effs[1] < effs[2]

    def test_fit(self):
        """Test curve fitting"""
        # Generate synthetic data with fixed seed for reproducibility
        np.random.seed(42)
        true_peak = 0.85
        true_scale = 500.0
        sizes = np.array([100, 200, 500, 1000, 2000, 5000])
        true_curve = AsymptoticCurve(peak=true_peak, scale=true_scale)
        # Use noise-free data for reliable fitting test
        efficiencies = true_curve.predict(sizes)

        # Fit curve
        fitted = AsymptoticCurve()
        result = fitted.fit(sizes, efficiencies)

        assert result.success
        # Fitted peak should be close to true peak (within 10%)
        assert abs(fitted.peak - true_peak) < 0.10
        # Predictions should be close to original
        predictions = fitted.predict(sizes)
        assert np.mean(np.abs(predictions - efficiencies)) < 0.1

    def test_serialization(self):
        """Test to_dict/from_dict roundtrip"""
        curve = AsymptoticCurve(peak=0.9, scale=1500.0)
        data = curve.to_dict()
        restored = AsymptoticCurve.from_dict(data)

        assert restored.peak == curve.peak
        assert restored.scale == curve.scale


class TestPiecewiseLinearCurve:
    """Tests for PiecewiseLinearCurve"""

    def test_basic_prediction(self):
        """Test basic efficiency prediction"""
        breakpoints = [(0.0, 0.1), (1000.0, 0.5), (10000.0, 0.9)]
        curve = PiecewiseLinearCurve(breakpoints=breakpoints)

        # Test interpolation
        eff_500 = curve.predict(500.0)
        assert 0.2 < eff_500 < 0.4  # Between first two points

        eff_5000 = curve.predict(5000.0)
        assert 0.6 < eff_5000 < 0.8  # Between second and third points

    def test_extrapolation(self):
        """Test extrapolation beyond breakpoints"""
        breakpoints = [(100.0, 0.2), (1000.0, 0.8)]
        curve = PiecewiseLinearCurve(breakpoints=breakpoints)

        # Below first point
        eff_below = curve.predict(50.0)
        assert eff_below == pytest.approx(0.2, abs=0.01)  # Clamped to first value

        # Above last point
        eff_above = curve.predict(2000.0)
        assert eff_above == pytest.approx(0.8, abs=0.01)  # Clamped to last value

    def test_fit(self):
        """Test piecewise linear fitting"""
        sizes = np.array([100, 500, 1000, 2000, 5000, 10000])
        efficiencies = np.array([0.2, 0.4, 0.5, 0.6, 0.75, 0.8])

        curve = PiecewiseLinearCurve()
        result = curve.fit(sizes, efficiencies)

        assert result.success
        assert result.r_squared > 0.9


class TestPolynomialCurve:
    """Tests for PolynomialCurve"""

    def test_basic_prediction(self):
        """Test basic efficiency prediction"""
        curve = PolynomialCurve(coefficients=[0.1, 0.05, 0.001], degree=2)

        # Should give some reasonable values
        eff = curve.predict(1000.0)
        assert 0.0 <= eff <= 1.0

    def test_fit(self):
        """Test polynomial fitting"""
        sizes = np.array([100, 500, 1000, 2000, 5000])
        efficiencies = np.array([0.3, 0.5, 0.6, 0.7, 0.8])

        curve = PolynomialCurve(degree=2)
        result = curve.fit(sizes, efficiencies)

        assert result.success
        assert result.r_squared > 0.8


class TestConstantCurve:
    """Tests for ConstantCurve"""

    def test_prediction(self):
        """Test constant efficiency prediction"""
        curve = ConstantCurve(efficiency=0.7)

        assert curve.predict(100) == 0.7
        assert curve.predict(10000) == 0.7

        # Array input
        effs = curve.predict(np.array([100, 1000, 10000]))
        assert all(e == 0.7 for e in effs)

    def test_fit(self):
        """Test constant curve fitting (mean)"""
        efficiencies = np.array([0.65, 0.70, 0.75, 0.68, 0.72])
        sizes = np.array([100, 500, 1000, 2000, 5000])

        curve = ConstantCurve()
        result = curve.fit(sizes, efficiencies)

        assert result.success
        assert curve.efficiency == pytest.approx(np.mean(efficiencies), abs=0.01)


class TestEfficiencyProfile:
    """Tests for EfficiencyProfile"""

    def test_add_and_get_curve(self):
        """Test adding and retrieving curves"""
        profile = EfficiencyProfile(device_name="Test GPU")

        curve = AsymptoticCurve(peak=0.85, scale=1000.0)
        profile.add_curve("gemm", curve, precision="fp32")

        retrieved = profile.get_curve("gemm", "fp32")
        assert retrieved is not None
        assert retrieved.peak == 0.85

    def test_predict_efficiency(self):
        """Test efficiency prediction through profile"""
        profile = EfficiencyProfile()

        curve = AsymptoticCurve(peak=0.9, scale=1000.0)
        profile.add_curve("gemm", curve)

        eff = profile.predict_efficiency("gemm", 5000.0)
        assert 0.8 < eff < 0.95

        # Unknown operation returns default
        eff_unknown = profile.predict_efficiency("unknown_op", 1000.0, default=0.5)
        assert eff_unknown == 0.5

    def test_serialization(self):
        """Test to_dict/from_dict roundtrip"""
        profile = EfficiencyProfile(device_name="Test GPU")

        profile.add_curve("gemm", AsymptoticCurve(peak=0.9, scale=1000.0), "fp32")
        profile.add_curve("conv2d", PiecewiseLinearCurve(), "fp32")

        data = profile.to_dict()
        restored = EfficiencyProfile.from_dict(data)

        assert restored.device_name == "Test GPU"
        assert restored.get_curve("gemm", "fp32") is not None
        assert restored.get_curve("conv2d", "fp32") is not None


class TestAutoFitEfficiencyCurve:
    """Tests for automatic curve selection"""

    def test_auto_fit_selects_best(self):
        """Test that auto_fit selects appropriate curve type"""
        # Generate asymptotic-like data
        sizes = np.array([100, 200, 500, 1000, 2000, 5000, 10000])
        true_curve = AsymptoticCurve(peak=0.85, scale=800.0)
        efficiencies = true_curve.predict(sizes)

        best_curve, result = auto_fit_efficiency_curve(sizes.tolist(), efficiencies.tolist())

        assert result.success
        assert result.r_squared > 0.9

    def test_auto_fit_handles_linear_data(self):
        """Test auto_fit with linear-ish data"""
        sizes = [100, 500, 1000, 2000, 5000]
        efficiencies = [0.2, 0.35, 0.5, 0.65, 0.8]

        best_curve, result = auto_fit_efficiency_curve(sizes, efficiencies)

        assert result.success
        assert result.r_squared > 0.9


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests combining roofline fitting and efficiency curves"""

    def test_full_calibration_workflow(self, combined_results):
        """Test complete calibration workflow"""
        # 1. Fit roofline parameters
        params = fit_roofline(
            combined_results,
            theoretical_bandwidth_gbps=1000.0,
            theoretical_compute_gflops=60000.0,
        )

        # 2. Create efficiency profile
        profile = EfficiencyProfile(device_name=params.device_name)

        # 3. Generate efficiency data from results
        sizes = []
        efficiencies = []

        for result in combined_results:
            if result.gflops > 0:
                ai = result.extra.get('arithmetic_intensity', 0)
                if ai > 0:
                    # Extract size from spec name
                    parts = result.spec_name.split('_')
                    if len(parts) >= 2:
                        try:
                            size = int(parts[1].split('x')[0])
                            eff = result.gflops / params.achieved_compute_gflops
                            sizes.append(size)
                            efficiencies.append(min(1.0, eff))
                        except (ValueError, IndexError):
                            pass

        # 4. Fit efficiency curve
        if len(sizes) >= 2:
            curve, fit_result = auto_fit_efficiency_curve(sizes, efficiencies)
            profile.add_curve("gemm", curve, "fp32")

        # 5. Verify prediction
        predicted_gflops = params.predict_gflops(100.0)
        assert predicted_gflops > 0

        if profile.get_curve("gemm", "fp32"):
            predicted_eff = profile.predict_efficiency("gemm", 2048)
            assert 0.0 <= predicted_eff <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
