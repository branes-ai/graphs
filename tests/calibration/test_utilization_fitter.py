"""
Tests for Utilization Factor Calibration

Tests the UtilizationFitter and related curve models for fitting
hardware utilization from benchmark data.
"""

import pytest
import numpy as np
from datetime import datetime

from graphs.calibration.utilization_fitter import (
    UtilizationFitter,
    UtilizationFitQuality,
    UtilizationFitMetrics,
    UtilizationCurveResult,
    UtilizationProfile,
    fit_utilization,
)
from graphs.calibration.utilization_curves import (
    UtilizationCurve,
    AsymptoticUtilizationCurve,
    PiecewiseLinearUtilizationCurve,
    PolynomialUtilizationCurve,
    ConstantUtilizationCurve,
    CurveType,
    fit_utilization_curve,
    auto_fit_utilization_curve,
    create_typical_compute_curve,
    create_typical_memory_curve,
    interpolate_utilization,
)
from graphs.benchmarks.schema import (
    BenchmarkResult,
    TimingStats,
    GEMMSpec,
    Conv2dSpec,
)


# ============================================================================
# Fixtures for generating synthetic benchmark data
# ============================================================================

@pytest.fixture
def peak_compute_gflops():
    """Peak compute for a hypothetical GPU (like H100)"""
    return 50000.0  # 50 TFLOPS


@pytest.fixture
def peak_bandwidth_gbps():
    """Peak memory bandwidth"""
    return 2000.0  # 2 TB/s


def _make_timing(time_ms: float) -> TimingStats:
    """Helper to create TimingStats from mean time"""
    return TimingStats(
        mean_ms=time_ms,
        std_ms=time_ms * 0.05,
        min_ms=time_ms * 0.9,
        max_ms=time_ms * 1.1,
        median_ms=time_ms,
        p95_ms=time_ms * 1.05,
        p99_ms=time_ms * 1.08,
        num_iterations=100,
    )


@pytest.fixture
def synthetic_gemm_results(peak_compute_gflops):
    """Generate synthetic GEMM benchmark results with realistic utilization curve"""
    results = []
    specs = {}

    # Generate results for various problem sizes
    # Utilization follows asymptotic pattern: util = 0.85 * (1 - exp(-flops / 1e10))
    sizes = [
        (256, 256, 256),    # Small: ~0.01 utilization
        (512, 512, 512),    # Small: ~0.05 utilization
        (1024, 1024, 1024), # Medium: ~0.18 utilization
        (2048, 2048, 2048), # Medium: ~0.53 utilization
        (4096, 4096, 4096), # Large: ~0.81 utilization
        (8192, 8192, 8192), # Large: ~0.85 utilization (near peak)
    ]

    for m, n, k in sizes:
        flops = 2 * m * n * k
        spec = GEMMSpec(
            name=f"gemm_{m}x{n}x{k}",
            M=m, N=n, K=k,
        )

        # Calculate expected utilization
        utilization = 0.85 * (1.0 - np.exp(-flops / 1e10))
        achieved_gflops = utilization * peak_compute_gflops

        # Estimate time from achieved GFLOPS
        time_ms = (flops / 1e9) / achieved_gflops * 1000.0

        result = BenchmarkResult(
            spec_name=spec.name,
            timestamp=datetime.now().isoformat(),
            device="cuda:0",
            device_name="TestGPU",
            precision="fp32",
            timing=_make_timing(time_ms),
            gflops=achieved_gflops,
            success=True,
            extra={'flops': flops},
        )

        results.append(result)
        specs[spec.name] = spec

    return results, specs


@pytest.fixture
def synthetic_memory_results(peak_bandwidth_gbps):
    """Generate synthetic memory benchmark results"""
    results = []

    # Memory utilization curve: util = 0.80 * (1 - exp(-bytes / 1e8))
    sizes_bytes = [1e5, 1e6, 1e7, 1e8, 1e9, 1e10]

    for size in sizes_bytes:
        utilization = 0.80 * (1.0 - np.exp(-size / 1e8))
        achieved_gbps = utilization * peak_bandwidth_gbps

        time_ms = (size / 1e9) / achieved_gbps * 1000.0 if achieved_gbps > 0 else 1.0

        result = BenchmarkResult(
            spec_name=f"memory_{int(size)}",
            timestamp=datetime.now().isoformat(),
            device="cuda:0",
            device_name="TestGPU",
            precision="fp32",
            timing=_make_timing(time_ms),
            bandwidth_gbps=achieved_gbps,
            success=True,
            extra={'bytes_transferred': size},
        )

        results.append(result)

    return results


@pytest.fixture
def multi_precision_results(peak_compute_gflops):
    """Generate results for multiple precisions"""
    results = []
    specs = {}

    # FP16 has 2x peak throughput
    precisions = {
        "fp32": 1.0,
        "fp16": 2.0,
        "int8": 4.0,
    }

    m, n, k = 4096, 4096, 4096
    flops = 2 * m * n * k

    for precision, throughput_multiplier in precisions.items():
        spec = GEMMSpec(
            name=f"gemm_4k_{precision}",
            M=m, N=n, K=k,
        )

        peak = peak_compute_gflops * throughput_multiplier
        utilization = 0.85 * (1.0 - np.exp(-flops / 1e10))
        achieved_gflops = utilization * peak

        time_ms = (flops / 1e9) / achieved_gflops * 1000.0

        result = BenchmarkResult(
            spec_name=spec.name,
            timestamp=datetime.now().isoformat(),
            device="cuda:0",
            device_name="TestGPU",
            precision=precision,
            timing=_make_timing(time_ms),
            gflops=achieved_gflops,
            success=True,
            extra={'flops': flops},
        )

        results.append(result)
        specs[spec.name] = spec

    return results, specs


# ============================================================================
# Tests for UtilizationCurve classes
# ============================================================================

class TestAsymptoticUtilizationCurve:
    """Tests for asymptotic utilization curve"""

    def test_creation(self):
        curve = AsymptoticUtilizationCurve(peak=0.85, scale=1e10)
        assert curve.peak == 0.85
        assert curve.scale == 1e10
        assert curve.curve_type == CurveType.ASYMPTOTIC

    def test_prediction_small_size(self):
        curve = AsymptoticUtilizationCurve(peak=0.85, scale=1e10)
        # Small size should have low utilization
        util = curve.predict(1e6)
        assert 0 < util < 0.1

    def test_prediction_large_size(self):
        curve = AsymptoticUtilizationCurve(peak=0.85, scale=1e10)
        # Large size should approach peak
        util = curve.predict(1e12)
        assert 0.8 < util <= 0.85

    def test_prediction_at_scale(self):
        curve = AsymptoticUtilizationCurve(peak=1.0, scale=1e10)
        # At scale, utilization should be ~63% of peak
        util = curve.predict(1e10)
        assert 0.60 < util < 0.66

    def test_array_prediction(self):
        curve = AsymptoticUtilizationCurve(peak=0.85, scale=1e10)
        sizes = np.array([1e6, 1e8, 1e10, 1e12])
        utils = curve.predict(sizes)
        assert len(utils) == 4
        assert np.all(np.diff(utils) > 0)  # Monotonically increasing

    def test_serialization(self):
        curve = AsymptoticUtilizationCurve(peak=0.85, scale=1e10)
        data = curve.to_dict()
        restored = AsymptoticUtilizationCurve.from_dict(data)
        assert restored.peak == curve.peak
        assert restored.scale == curve.scale


class TestPiecewiseLinearUtilizationCurve:
    """Tests for piecewise linear utilization curve"""

    def test_creation(self):
        breakpoints = [(0.0, 0.0), (1e6, 0.3), (1e9, 0.7), (1e12, 0.85)]
        curve = PiecewiseLinearUtilizationCurve(breakpoints)
        assert curve.curve_type == CurveType.PIECEWISE_LINEAR

    def test_interpolation(self):
        breakpoints = [(0.0, 0.0), (1e6, 0.3), (1e9, 0.7), (1e12, 0.85)]
        curve = PiecewiseLinearUtilizationCurve(breakpoints)

        # At breakpoints
        assert curve.predict(1e6) == pytest.approx(0.3, rel=0.01)
        assert curve.predict(1e9) == pytest.approx(0.7, rel=0.01)

        # Between breakpoints
        util = curve.predict(1e7)
        assert 0.3 < util < 0.7

    def test_serialization(self):
        breakpoints = [(0.0, 0.0), (1e6, 0.3), (1e12, 0.85)]
        curve = PiecewiseLinearUtilizationCurve(breakpoints)
        data = curve.to_dict()
        restored = PiecewiseLinearUtilizationCurve.from_dict(data)
        assert len(restored.breakpoints) == len(curve.breakpoints)


class TestConstantUtilizationCurve:
    """Tests for constant utilization curve"""

    def test_creation(self):
        curve = ConstantUtilizationCurve(efficiency=0.7)
        assert curve.efficiency == 0.7
        assert curve.curve_type == CurveType.CONSTANT

    def test_prediction(self):
        curve = ConstantUtilizationCurve(efficiency=0.7)
        assert curve.predict(1e6) == 0.7
        assert curve.predict(1e12) == 0.7

    def test_clamping(self):
        curve = ConstantUtilizationCurve(efficiency=1.5)
        assert curve.efficiency == 1.0  # Clamped to max


# ============================================================================
# Tests for curve fitting functions
# ============================================================================

class TestCurveFitting:
    """Tests for curve fitting functions"""

    def test_fit_asymptotic(self):
        # Generate synthetic data
        sizes = np.logspace(6, 12, 20)
        true_utils = 0.85 * (1.0 - np.exp(-sizes / 1e10))
        noise = np.random.normal(0, 0.02, len(sizes))
        utils = np.clip(true_utils + noise, 0, 1)

        curve, result = fit_utilization_curve(
            sizes.tolist(), utils.tolist(), CurveType.ASYMPTOTIC
        )

        assert result.success
        assert result.r_squared > 0.9
        assert curve.peak == pytest.approx(0.85, rel=0.2)

    def test_auto_fit(self):
        sizes = np.logspace(6, 12, 20).tolist()
        utils = [0.85 * (1.0 - np.exp(-s / 1e10)) for s in sizes]

        curve, result = auto_fit_utilization_curve(sizes, utils)

        assert result.success
        assert result.r_squared > 0.8

    def test_insufficient_data(self):
        sizes = [1e6]
        utils = [0.5]

        curve, result = fit_utilization_curve(sizes, utils)
        assert not result.success


class TestConvenienceFunctions:
    """Tests for convenience curve creation functions"""

    def test_typical_compute_curve(self):
        curve = create_typical_compute_curve()
        assert curve.peak == 0.85
        assert curve.scale == 1e10

        # Should have low util for small problems
        assert curve.predict(1e6) < 0.1

        # Should approach peak for large problems
        assert curve.predict(1e14) > 0.8

    def test_typical_memory_curve(self):
        curve = create_typical_memory_curve()
        assert curve.peak == 0.80
        assert curve.scale == 1e8

    def test_interpolate_utilization(self):
        sizes = [1e6, 1e8, 1e10]
        utils = [0.1, 0.5, 0.8]

        # Interpolate within range
        result = interpolate_utilization(sizes, utils, 1e9)
        assert 0.5 < result < 0.8

        # Extrapolate beyond range
        result = interpolate_utilization(sizes, utils, 1e12)
        assert result == pytest.approx(0.8, rel=0.01)  # Clipped to last value


# ============================================================================
# Tests for UtilizationFitter
# ============================================================================

class TestUtilizationFitter:
    """Tests for the UtilizationFitter class"""

    def test_initialization(self, peak_compute_gflops, peak_bandwidth_gbps):
        fitter = UtilizationFitter(
            peak_compute_gflops=peak_compute_gflops,
            peak_bandwidth_gbps=peak_bandwidth_gbps,
        )
        assert fitter.peak_compute_gflops == peak_compute_gflops
        assert fitter.peak_bandwidth_gbps == peak_bandwidth_gbps

    def test_add_compute_results(
        self, peak_compute_gflops, peak_bandwidth_gbps, synthetic_gemm_results
    ):
        results, specs = synthetic_gemm_results

        fitter = UtilizationFitter(
            peak_compute_gflops=peak_compute_gflops,
            peak_bandwidth_gbps=peak_bandwidth_gbps,
        )

        for result in results:
            spec = specs.get(result.spec_name)
            fitter.add_compute_result(result, spec)

        assert fitter.can_fit()

    def test_add_memory_results(
        self, peak_compute_gflops, peak_bandwidth_gbps, synthetic_memory_results
    ):
        fitter = UtilizationFitter(
            peak_compute_gflops=peak_compute_gflops,
            peak_bandwidth_gbps=peak_bandwidth_gbps,
        )

        for result in synthetic_memory_results:
            fitter.add_memory_result(result)

        assert fitter.can_fit()

    def test_fit_compute(
        self, peak_compute_gflops, peak_bandwidth_gbps, synthetic_gemm_results
    ):
        results, specs = synthetic_gemm_results

        fitter = UtilizationFitter(
            peak_compute_gflops=peak_compute_gflops,
            peak_bandwidth_gbps=peak_bandwidth_gbps,
        )

        for result in results:
            spec = specs.get(result.spec_name)
            fitter.add_compute_result(result, spec)

        profile = fitter.fit()

        # Should have GEMM curve
        assert profile.get_curve("gemm", "fp32") is not None

        # Curve should predict reasonable values
        gemm_curve = profile.get_curve("gemm", "fp32")
        small_util = gemm_curve.predict(1e7)
        large_util = gemm_curve.predict(1e12)
        assert small_util < large_util

    def test_fit_memory(
        self, peak_compute_gflops, peak_bandwidth_gbps, synthetic_memory_results
    ):
        fitter = UtilizationFitter(
            peak_compute_gflops=peak_compute_gflops,
            peak_bandwidth_gbps=peak_bandwidth_gbps,
        )

        for result in synthetic_memory_results:
            fitter.add_memory_result(result)

        profile = fitter.fit()

        # Should have memory curve
        assert profile.get_curve("memory", "n/a") is not None

    def test_insufficient_data_error(self, peak_compute_gflops):
        fitter = UtilizationFitter(peak_compute_gflops=peak_compute_gflops)

        with pytest.raises(ValueError, match="Insufficient data"):
            fitter.fit()

    def test_reset(
        self, peak_compute_gflops, peak_bandwidth_gbps, synthetic_gemm_results
    ):
        results, specs = synthetic_gemm_results

        fitter = UtilizationFitter(
            peak_compute_gflops=peak_compute_gflops,
            peak_bandwidth_gbps=peak_bandwidth_gbps,
        )

        for result in results:
            spec = specs.get(result.spec_name)
            fitter.add_compute_result(result, spec)

        assert fitter.can_fit()

        fitter.reset()
        assert not fitter.can_fit()


class TestFitUtilizationFunction:
    """Tests for the fit_utilization convenience function"""

    def test_basic_usage(
        self, peak_compute_gflops, peak_bandwidth_gbps, synthetic_gemm_results
    ):
        results, specs = synthetic_gemm_results

        profile = fit_utilization(
            results,
            peak_compute_gflops=peak_compute_gflops,
            peak_bandwidth_gbps=peak_bandwidth_gbps,
            specs=specs,
        )

        assert profile.peak_compute_gflops == peak_compute_gflops
        assert profile.get_curve("gemm", "fp32") is not None


# ============================================================================
# Tests for UtilizationProfile
# ============================================================================

class TestUtilizationProfile:
    """Tests for UtilizationProfile"""

    def test_creation(self):
        profile = UtilizationProfile(
            peak_compute_gflops=50000.0,
            peak_bandwidth_gbps=2000.0,
            device_name="TestGPU",
        )
        assert profile.peak_compute_gflops == 50000.0
        assert profile.device_name == "TestGPU"

    def test_add_and_get_curve(self):
        profile = UtilizationProfile()

        curve = AsymptoticUtilizationCurve(peak=0.85, scale=1e10)
        result = UtilizationCurveResult(
            operation="gemm",
            precision="fp32",
            curve=curve,
            fit_result=curve.fit(
                np.logspace(6, 12, 10),
                0.85 * (1.0 - np.exp(-np.logspace(6, 12, 10) / 1e10))
            ),
            metrics=UtilizationFitMetrics(
                num_data_points=10,
                r_squared=0.95,
                rmse=0.02,
                min_utilization=0.01,
                max_utilization=0.85,
                mean_utilization=0.5,
                quality=UtilizationFitQuality.EXCELLENT,
            ),
        )

        profile.add_curve(result)

        retrieved = profile.get_curve("gemm", "fp32")
        assert retrieved is not None
        assert retrieved.operation == "gemm"

    def test_predict_utilization(self):
        profile = UtilizationProfile()

        curve = AsymptoticUtilizationCurve(peak=0.85, scale=1e10)
        result = UtilizationCurveResult(
            operation="gemm",
            precision="fp32",
            curve=curve,
            fit_result=curve.fit(
                np.logspace(6, 12, 10),
                0.85 * (1.0 - np.exp(-np.logspace(6, 12, 10) / 1e10))
            ),
            metrics=UtilizationFitMetrics(
                num_data_points=10,
                r_squared=0.95,
                rmse=0.02,
                min_utilization=0.01,
                max_utilization=0.85,
                mean_utilization=0.5,
                quality=UtilizationFitQuality.EXCELLENT,
            ),
        )
        profile.add_curve(result)

        # Known operation
        util = profile.predict_utilization("gemm", 1e10, "fp32")
        assert 0.0 < util < 1.0

        # Unknown operation returns default
        util = profile.predict_utilization("unknown", 1e10, "fp32", default=0.5)
        assert util == 0.5

    def test_predict_achievable_gflops(self):
        profile = UtilizationProfile(
            peak_compute_gflops=50000.0,
            peak_compute_by_precision={"fp32": 50000.0, "fp16": 100000.0},
        )

        curve = AsymptoticUtilizationCurve(peak=0.85, scale=1e10)
        result = UtilizationCurveResult(
            operation="gemm",
            precision="fp32",
            curve=curve,
            fit_result=curve.fit(
                np.logspace(6, 12, 10),
                0.85 * (1.0 - np.exp(-np.logspace(6, 12, 10) / 1e10))
            ),
            metrics=UtilizationFitMetrics(
                num_data_points=10,
                r_squared=0.95,
                rmse=0.02,
                min_utilization=0.01,
                max_utilization=0.85,
                mean_utilization=0.5,
                quality=UtilizationFitQuality.EXCELLENT,
            ),
        )
        profile.add_curve(result)

        # Large problem should achieve high GFLOPS
        achievable = profile.predict_achievable_gflops("gemm", 1e12, "fp32")
        assert achievable > 40000  # Should be close to 85% of 50000

    def test_serialization(self):
        profile = UtilizationProfile(
            peak_compute_gflops=50000.0,
            peak_bandwidth_gbps=2000.0,
            device_name="TestGPU",
        )

        curve = AsymptoticUtilizationCurve(peak=0.85, scale=1e10)
        sizes = np.logspace(6, 12, 10)
        utils = 0.85 * (1.0 - np.exp(-sizes / 1e10))

        result = UtilizationCurveResult(
            operation="gemm",
            precision="fp32",
            curve=curve,
            fit_result=curve.fit(sizes, utils),
            metrics=UtilizationFitMetrics(
                num_data_points=10,
                r_squared=0.95,
                rmse=0.02,
                min_utilization=0.01,
                max_utilization=0.85,
                mean_utilization=0.5,
                quality=UtilizationFitQuality.EXCELLENT,
            ),
            sizes=sizes.tolist(),
            utilizations=utils.tolist(),
        )
        profile.add_curve(result)

        # Serialize and restore
        data = profile.to_dict()
        restored = UtilizationProfile.from_dict(data)

        assert restored.peak_compute_gflops == profile.peak_compute_gflops
        assert restored.device_name == profile.device_name
        assert restored.get_curve("gemm", "fp32") is not None


# ============================================================================
# Tests for UtilizationFitMetrics
# ============================================================================

class TestUtilizationFitMetrics:
    """Tests for fit metrics"""

    def test_creation(self):
        metrics = UtilizationFitMetrics(
            num_data_points=20,
            r_squared=0.95,
            rmse=0.02,
            min_utilization=0.01,
            max_utilization=0.85,
            mean_utilization=0.45,
            quality=UtilizationFitQuality.EXCELLENT,
        )
        assert metrics.num_data_points == 20
        assert metrics.quality == UtilizationFitQuality.EXCELLENT

    def test_serialization(self):
        metrics = UtilizationFitMetrics(
            num_data_points=20,
            r_squared=0.95,
            rmse=0.02,
            min_utilization=0.01,
            max_utilization=0.85,
            mean_utilization=0.45,
            quality=UtilizationFitQuality.GOOD,
        )

        data = metrics.to_dict()
        restored = UtilizationFitMetrics.from_dict(data)

        assert restored.num_data_points == metrics.num_data_points
        assert restored.quality == metrics.quality


# ============================================================================
# Integration tests
# ============================================================================

class TestIntegration:
    """Integration tests for full utilization fitting workflow"""

    def test_full_workflow(
        self,
        peak_compute_gflops,
        peak_bandwidth_gbps,
        synthetic_gemm_results,
        synthetic_memory_results,
    ):
        """Test complete workflow from results to predictions"""
        gemm_results, gemm_specs = synthetic_gemm_results

        # Create fitter
        fitter = UtilizationFitter(
            peak_compute_gflops=peak_compute_gflops,
            peak_bandwidth_gbps=peak_bandwidth_gbps,
        )

        # Add all results
        for result in gemm_results:
            spec = gemm_specs.get(result.spec_name)
            fitter.add_compute_result(result, spec)

        for result in synthetic_memory_results:
            fitter.add_memory_result(result)

        # Fit profile
        profile = fitter.fit()

        # Verify curves exist
        assert profile.get_curve("gemm", "fp32") is not None
        assert profile.get_curve("memory", "n/a") is not None

        # Make predictions
        gemm_util = profile.predict_utilization("gemm", 1e11, "fp32")
        assert 0.0 < gemm_util < 1.0

        memory_util = profile.predict_utilization("memory", 1e9, "n/a")
        assert 0.0 < memory_util < 1.0

        # Verify size-dependent behavior
        small_util = profile.predict_utilization("gemm", 1e7, "fp32")
        large_util = profile.predict_utilization("gemm", 1e12, "fp32")
        assert small_util < large_util

    def test_multi_precision(
        self, peak_compute_gflops, peak_bandwidth_gbps, multi_precision_results
    ):
        """Test utilization fitting with multiple precisions"""
        results, specs = multi_precision_results

        fitter = UtilizationFitter(
            peak_compute_gflops=peak_compute_gflops,
            peak_bandwidth_gbps=peak_bandwidth_gbps,
            peak_compute_by_precision={
                "fp32": peak_compute_gflops,
                "fp16": peak_compute_gflops * 2,
                "int8": peak_compute_gflops * 4,
            },
        )

        for result in results:
            spec = specs.get(result.spec_name)
            fitter.add_compute_result(result, spec)

        # Should have enough data for each precision
        # Note: with only 1 result per precision, won't fit curves
        # But the fitter should handle gracefully
        if fitter.can_fit():
            profile = fitter.fit()
            assert profile is not None
