"""
Tests for energy coefficient fitting and power model.

Tests cover:
- Energy coefficient fitting from benchmark data
- Power model predictions
- Handling of missing/insufficient data
- Per-precision and per-memory-level coefficients
"""

import pytest
import numpy as np
from datetime import datetime

from graphs.calibration.energy_fitter import (
    EnergyFitter,
    EnergyCoefficients,
    EnergyFitMetrics,
    EnergyFitQuality,
    EnergyDataPoint,
    fit_energy_model,
)
from graphs.calibration.power_model import (
    CalibratedPowerModel,
    PowerSource,
    PowerBreakdown,
    EnergyBreakdown,
)
from graphs.benchmarks.schema import (
    BenchmarkResult,
    TimingStats,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def synthetic_compute_results():
    """Generate synthetic GEMM benchmark results with power measurements"""
    # Simulated GPU with:
    # - compute_pj_per_op = 0.5 pJ/op
    # - memory_pj_per_byte = 20 pJ/byte
    # - static_power = 50 W

    results = []
    # Different compute intensities
    configs = [
        # (gflops, bandwidth_gbps)
        (20000, 500),   # Compute-heavy
        (30000, 600),
        (40000, 700),
        (35000, 550),
        (45000, 800),
    ]

    for i, (gflops, gbps) in enumerate(configs):
        # Calculate power: P = gflops * 0.5 * 1e-3 + gbps * 20 * 1e-3 + 50
        power = gflops * 0.5 * 1e-3 + gbps * 20 * 1e-3 + 50
        # Add some noise
        power *= (0.95 + 0.1 * np.random.random())

        results.append(BenchmarkResult(
            spec_name=f"gemm_{i}",
            timestamp=datetime.now().isoformat(),
            device="cuda:0",
            device_name="Test GPU",
            precision="fp32",
            timing=TimingStats(
                mean_ms=10.0,
                std_ms=1.0,
                min_ms=9.0,
                max_ms=12.0,
                median_ms=10.0,
                p95_ms=11.5,
                p99_ms=11.8,
                num_iterations=100,
            ),
            gflops=gflops,
            bandwidth_gbps=gbps,
            avg_power_watts=power,
            peak_power_watts=power * 1.2,
            energy_joules=power * 0.01,  # 10ms
            success=True,
        ))

    return results


@pytest.fixture
def synthetic_memory_results():
    """Generate synthetic memory benchmark results with power measurements"""
    # Memory benchmarks: minimal compute, varying bandwidth

    results = []
    bandwidths = [800, 1000, 1200, 1500, 1800]

    for i, gbps in enumerate(bandwidths):
        # Power dominated by memory and static
        # P = gbps * 20 * 1e-3 + 50
        power = gbps * 20 * 1e-3 + 50
        power *= (0.95 + 0.1 * np.random.random())

        results.append(BenchmarkResult(
            spec_name=f"stream_triad_{i}",
            timestamp=datetime.now().isoformat(),
            device="cuda:0",
            device_name="Test GPU",
            precision="fp32",
            timing=TimingStats(
                mean_ms=5.0,
                std_ms=0.5,
                min_ms=4.5,
                max_ms=6.0,
                median_ms=5.0,
                p95_ms=5.5,
                p99_ms=5.8,
                num_iterations=100,
            ),
            gflops=0.0,  # Memory benchmark
            bandwidth_gbps=gbps,
            avg_power_watts=power,
            peak_power_watts=power * 1.1,
            energy_joules=power * 0.005,
            success=True,
        ))

    return results


@pytest.fixture
def combined_results(synthetic_compute_results, synthetic_memory_results):
    """Combined compute and memory benchmark results"""
    return synthetic_compute_results + synthetic_memory_results


# =============================================================================
# EnergyFitter Tests
# =============================================================================

class TestEnergyFitter:
    """Tests for EnergyFitter class"""

    def test_init(self):
        """Test fitter initialization"""
        fitter = EnergyFitter(
            theoretical_compute_pj_per_op=0.5,
            theoretical_memory_pj_per_byte=20.0,
        )
        assert fitter.theoretical_compute_pj_per_op == 0.5
        assert fitter.theoretical_memory_pj_per_byte == 20.0

    def test_add_compute_result(self, synthetic_compute_results):
        """Test adding compute benchmark results"""
        fitter = EnergyFitter()

        for result in synthetic_compute_results:
            fitter.add_compute_result(result)

        assert fitter.can_fit()
        assert len(fitter._data_points) == 5

    def test_add_memory_result(self, synthetic_memory_results):
        """Test adding memory benchmark results"""
        fitter = EnergyFitter()

        for result in synthetic_memory_results:
            fitter.add_memory_result(result)

        assert fitter.can_fit()
        assert len(fitter._data_points) == 5

    def test_add_idle_measurement(self):
        """Test adding idle power measurement"""
        fitter = EnergyFitter()
        fitter.add_idle_measurement(50.0)

        assert fitter._idle_power == 50.0

    def test_add_data_point(self):
        """Test adding raw data points"""
        fitter = EnergyFitter()

        fitter.add_data_point(
            power_watts=100.0,
            gops=30000.0,
            gbps=800.0,
            precision="fp32",
            source="test",
        )

        assert len(fitter._data_points) == 1
        assert fitter._data_points[0].power_watts == 100.0

    def test_fit_combined(self, combined_results):
        """Test fitting from combined compute and memory data"""
        np.random.seed(42)  # For reproducibility
        fitter = EnergyFitter()
        fitter.add_idle_measurement(50.0)

        for result in combined_results:
            is_memory = 'stream' in result.spec_name.lower()
            if is_memory:
                fitter.add_memory_result(result)
            else:
                fitter.add_compute_result(result)

        coeffs = fitter.fit()

        assert isinstance(coeffs, EnergyCoefficients)
        assert coeffs.compute_pj_per_op > 0
        assert coeffs.memory_pj_per_byte > 0
        assert coeffs.static_power_watts > 0

        # Check that fitted values are reasonable (within 50% of true values)
        # True values: compute=0.5 pJ/op, memory=20 pJ/byte, static=50 W
        assert 0.1 < coeffs.compute_pj_per_op < 2.0
        assert 5 < coeffs.memory_pj_per_byte < 50
        assert 20 < coeffs.static_power_watts < 80

    def test_insufficient_data(self):
        """Test error handling for insufficient data"""
        fitter = EnergyFitter()

        # Only add 2 points (need >= 3)
        fitter.add_data_point(100.0, 20000.0, 500.0)
        fitter.add_data_point(120.0, 30000.0, 600.0)

        assert not fitter.can_fit()

        with pytest.raises(ValueError, match="Insufficient data"):
            fitter.fit()

    def test_failed_results_ignored(self):
        """Test that failed benchmark results are ignored"""
        fitter = EnergyFitter()

        fitter.add_compute_result(BenchmarkResult(
            spec_name="gemm_failed",
            timestamp=datetime.now().isoformat(),
            device="cuda:0",
            gflops=30000.0,
            avg_power_watts=100.0,
            success=False,  # Failed!
            error_message="GPU error",
        ))

        assert len(fitter._data_points) == 0

    def test_missing_power_ignored(self):
        """Test that results without power data are ignored"""
        fitter = EnergyFitter()

        fitter.add_compute_result(BenchmarkResult(
            spec_name="gemm_no_power",
            timestamp=datetime.now().isoformat(),
            device="cuda:0",
            gflops=30000.0,
            avg_power_watts=None,  # No power data
            success=True,
        ))

        assert len(fitter._data_points) == 0

    def test_reset(self, synthetic_compute_results):
        """Test fitter reset"""
        fitter = EnergyFitter()

        for result in synthetic_compute_results:
            fitter.add_compute_result(result)

        assert fitter.can_fit()

        fitter.reset()

        assert not fitter.can_fit()
        assert len(fitter._data_points) == 0


# =============================================================================
# EnergyCoefficients Tests
# =============================================================================

class TestEnergyCoefficients:
    """Tests for EnergyCoefficients class"""

    def test_predict_power(self):
        """Test power prediction"""
        coeffs = EnergyCoefficients(
            compute_pj_per_op=0.5,
            memory_pj_per_byte=20.0,
            static_power_watts=50.0,
        )

        # P = 30000 * 0.5 * 1e-3 + 1000 * 20 * 1e-3 + 50
        # P = 15 + 20 + 50 = 85 W
        power = coeffs.predict_power(gops=30000, gbps=1000)
        assert power == pytest.approx(85.0, rel=0.01)

    def test_predict_energy(self):
        """Test energy prediction"""
        coeffs = EnergyCoefficients(
            compute_pj_per_op=0.5,
            memory_pj_per_byte=20.0,
            static_power_watts=50.0,
        )

        # E = 1e12 * 0.5 * 1e-12 + 1e9 * 20 * 1e-12 + 50 * 0.01
        # E = 0.5 + 0.02 + 0.5 = 1.02 J
        energy = coeffs.predict_energy(
            ops=int(1e12),
            bytes_transferred=int(1e9),
            duration_s=0.01,
        )
        assert energy == pytest.approx(1.02, rel=0.01)

    def test_per_precision_coefficients(self):
        """Test per-precision coefficient handling"""
        coeffs = EnergyCoefficients(
            compute_pj_per_op=0.5,  # FP32 default
            memory_pj_per_byte=20.0,
            static_power_watts=50.0,
            compute_pj_per_op_by_precision={
                'fp32': 0.5,
                'fp16': 0.25,  # Half the energy for FP16
                'int8': 0.125,  # Quarter energy for INT8
            },
        )

        # FP32 power
        power_fp32 = coeffs.predict_power(gops=30000, gbps=1000, precision='fp32')

        # FP16 power (lower compute energy)
        power_fp16 = coeffs.predict_power(gops=30000, gbps=1000, precision='fp16')

        assert power_fp16 < power_fp32

    def test_serialization(self):
        """Test to_dict/from_dict roundtrip"""
        coeffs = EnergyCoefficients(
            compute_pj_per_op=0.5,
            memory_pj_per_byte=20.0,
            static_power_watts=50.0,
            device_name="Test GPU",
            fit_metrics=EnergyFitMetrics(
                num_data_points=10,
                r_squared=0.95,
                residual_std=2.0,
                quality=EnergyFitQuality.EXCELLENT,
            ),
        )

        data = coeffs.to_dict()
        restored = EnergyCoefficients.from_dict(data)

        assert restored.compute_pj_per_op == coeffs.compute_pj_per_op
        assert restored.memory_pj_per_byte == coeffs.memory_pj_per_byte
        assert restored.static_power_watts == coeffs.static_power_watts
        assert restored.fit_metrics.quality == EnergyFitQuality.EXCELLENT


# =============================================================================
# fit_energy_model Function Tests
# =============================================================================

class TestFitEnergyModelFunction:
    """Tests for the fit_energy_model convenience function"""

    def test_basic_usage(self, combined_results):
        """Test basic usage of fit_energy_model function"""
        np.random.seed(42)
        coeffs = fit_energy_model(combined_results, idle_power_watts=50.0)

        assert isinstance(coeffs, EnergyCoefficients)
        assert coeffs.compute_pj_per_op > 0
        assert coeffs.static_power_watts == 50.0  # Uses provided idle power


# =============================================================================
# CalibratedPowerModel Tests
# =============================================================================

class TestCalibratedPowerModel:
    """Tests for CalibratedPowerModel class"""

    def test_init(self):
        """Test model initialization"""
        model = CalibratedPowerModel(
            compute_pj_per_op=0.5,
            memory_pj_per_byte=20.0,
            static_power_watts=50.0,
            source=PowerSource.CALIBRATED,
            device_name="Test GPU",
        )

        assert model.compute_pj_per_op == 0.5
        assert model.memory_pj_per_byte == 20.0
        assert model.static_power_watts == 50.0
        assert model.source == PowerSource.CALIBRATED

    def test_from_coefficients(self):
        """Test creating model from coefficients"""
        coeffs = EnergyCoefficients(
            compute_pj_per_op=0.5,
            memory_pj_per_byte=20.0,
            static_power_watts=50.0,
            device_name="Test GPU",
        )

        model = CalibratedPowerModel.from_coefficients(coeffs)

        assert model.compute_pj_per_op == 0.5
        assert model.source == PowerSource.CALIBRATED

    def test_from_theoretical(self):
        """Test creating model from theoretical specs"""
        model = CalibratedPowerModel.from_theoretical(
            tdp_watts=300,
            peak_gflops=50000,
            peak_bandwidth_gbps=2000,
            device_name="Theoretical GPU",
        )

        assert model.source == PowerSource.THEORETICAL
        assert model.static_power_watts > 0
        assert model.compute_pj_per_op > 0
        assert model.memory_pj_per_byte > 0

    def test_predict_power(self):
        """Test power prediction"""
        model = CalibratedPowerModel(
            compute_pj_per_op=0.5,
            memory_pj_per_byte=20.0,
            static_power_watts=50.0,
        )

        power = model.predict_power(gops=30000, gbps=1000)
        assert power == pytest.approx(85.0, rel=0.01)

    def test_predict_power_breakdown(self):
        """Test power prediction with breakdown"""
        model = CalibratedPowerModel(
            compute_pj_per_op=0.5,
            memory_pj_per_byte=20.0,
            static_power_watts=50.0,
        )

        breakdown = model.predict_power_breakdown(gops=30000, gbps=1000)

        assert isinstance(breakdown, PowerBreakdown)
        assert breakdown.compute_watts == pytest.approx(15.0, rel=0.01)
        assert breakdown.memory_watts == pytest.approx(20.0, rel=0.01)
        assert breakdown.static_watts == pytest.approx(50.0, rel=0.01)
        assert breakdown.total_watts == pytest.approx(85.0, rel=0.01)

        # Check percentages
        assert breakdown.compute_percent == pytest.approx(17.6, abs=1.0)
        assert breakdown.memory_percent == pytest.approx(23.5, abs=1.0)
        assert breakdown.static_percent == pytest.approx(58.8, abs=1.0)

    def test_predict_energy(self):
        """Test energy prediction"""
        model = CalibratedPowerModel(
            compute_pj_per_op=0.5,
            memory_pj_per_byte=20.0,
            static_power_watts=50.0,
        )

        energy = model.predict_energy(
            ops=int(1e12),
            bytes_transferred=int(1e9),
            duration_seconds=0.01,
        )
        assert energy == pytest.approx(1.02, rel=0.01)

    def test_predict_energy_breakdown(self):
        """Test energy prediction with breakdown"""
        model = CalibratedPowerModel(
            compute_pj_per_op=0.5,
            memory_pj_per_byte=20.0,
            static_power_watts=50.0,
        )

        breakdown = model.predict_energy_breakdown(
            ops=int(1e12),
            bytes_transferred=int(1e9),
            duration_seconds=0.01,
        )

        assert isinstance(breakdown, EnergyBreakdown)
        assert breakdown.compute_joules == pytest.approx(0.5, rel=0.01)
        assert breakdown.memory_joules == pytest.approx(0.02, rel=0.01)
        assert breakdown.static_joules == pytest.approx(0.5, rel=0.01)
        assert breakdown.total_joules == pytest.approx(1.02, rel=0.01)

    def test_estimate_latency_from_energy_budget(self):
        """Test latency estimation from energy budget"""
        model = CalibratedPowerModel(
            compute_pj_per_op=0.5,
            memory_pj_per_byte=20.0,
            static_power_watts=50.0,
        )

        # 1 J budget, 1e11 ops, 1e8 bytes
        # Dynamic energy = 1e11 * 0.5 * 1e-12 + 1e8 * 20 * 1e-12 = 0.05 + 0.002 = 0.052 J
        # Remaining = 1 - 0.052 = 0.948 J for static
        # Max time = 0.948 / 50 = 0.019 s
        max_time = model.estimate_latency_from_energy_budget(
            energy_budget_joules=1.0,
            ops=int(1e11),
            bytes_transferred=int(1e8),
        )
        assert max_time == pytest.approx(0.019, abs=0.001)

    def test_serialization(self):
        """Test to_dict/from_dict roundtrip"""
        model = CalibratedPowerModel(
            compute_pj_per_op=0.5,
            memory_pj_per_byte=20.0,
            static_power_watts=50.0,
            source=PowerSource.CALIBRATED,
            device_name="Test GPU",
        )

        data = model.to_dict()
        restored = CalibratedPowerModel.from_dict(data)

        assert restored.compute_pj_per_op == model.compute_pj_per_op
        assert restored.source == PowerSource.CALIBRATED


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for energy fitting and power model"""

    def test_full_workflow(self, combined_results):
        """Test complete calibration workflow"""
        np.random.seed(42)

        # 1. Fit coefficients
        fitter = EnergyFitter()
        fitter.add_idle_measurement(50.0)

        for result in combined_results:
            is_memory = 'stream' in result.spec_name.lower()
            if is_memory:
                fitter.add_memory_result(result)
            else:
                fitter.add_compute_result(result)

        coeffs = fitter.fit()

        # 2. Create power model
        model = CalibratedPowerModel.from_coefficients(coeffs)

        # 3. Make predictions
        power = model.predict_power(gops=35000, gbps=900)
        assert power > 0

        energy = model.predict_energy(
            ops=int(3.5e11),
            bytes_transferred=int(9e8),
            duration_seconds=0.01,
        )
        assert energy > 0

        # 4. Check that calibrated model gives reasonable predictions
        # Compared to theoretical model
        theoretical_model = CalibratedPowerModel.from_theoretical(
            tdp_watts=200,
            peak_gflops=50000,
            peak_bandwidth_gbps=2000,
        )

        theoretical_power = theoretical_model.predict_power(gops=35000, gbps=900)

        # Both should be in the same order of magnitude
        assert 0.1 < power / theoretical_power < 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
