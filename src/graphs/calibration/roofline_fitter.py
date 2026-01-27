"""
Roofline Parameter Fitting from Benchmark Results

Fits roofline model parameters from measured benchmark data:
- Achieved memory bandwidth (GB/s) from memory/STREAM benchmarks
- Achieved compute throughput (GFLOPS) from GEMM benchmarks
- Ridge point (arithmetic intensity where compute = memory bound)

The fitted parameters improve latency estimation accuracy from
theoretical (30-50% error) to calibrated (<15% error).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from enum import Enum

from graphs.benchmarks.schema import (
    BenchmarkResult,
    GEMMSpec,
    MemoryBenchSpec,
    BenchmarkCategory,
)


class FitQuality(Enum):
    """Quality level of the fitted parameters"""
    EXCELLENT = "excellent"  # R-squared >= 0.95, many data points
    GOOD = "good"            # R-squared >= 0.85, sufficient data
    FAIR = "fair"            # R-squared >= 0.70, limited data
    POOR = "poor"            # R-squared < 0.70 or very few points
    INSUFFICIENT = "insufficient"  # Not enough data to fit


@dataclass
class FitMetrics:
    """Metrics describing the quality of a fit"""
    num_data_points: int
    r_squared: float  # Coefficient of determination
    residual_std: float  # Standard deviation of residuals
    quality: FitQuality

    def to_dict(self) -> Dict[str, Any]:
        return {
            'num_data_points': self.num_data_points,
            'r_squared': self.r_squared,
            'residual_std': self.residual_std,
            'quality': self.quality.value,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FitMetrics':
        data = data.copy()
        data['quality'] = FitQuality(data['quality'])
        return cls(**data)


@dataclass
class RooflineParameters:
    """
    Fitted roofline model parameters.

    The roofline model bounds performance by:
    - Memory ceiling: bandwidth_gbps * arithmetic_intensity
    - Compute ceiling: compute_gflops

    Performance = min(memory_ceiling, compute_ceiling)
    Ridge point = compute_gflops / bandwidth_gbps (FLOPs/byte)
    """
    # Fitted ceilings
    achieved_bandwidth_gbps: float  # Memory bandwidth ceiling (GB/s)
    achieved_compute_gflops: float  # Compute throughput ceiling (GFLOPS)

    # Derived parameters
    ridge_point: float  # Arithmetic intensity where ceilings intersect

    # Fit quality metrics
    bandwidth_fit: Optional[FitMetrics] = None
    compute_fit: Optional[FitMetrics] = None

    # Theoretical values for comparison
    theoretical_bandwidth_gbps: Optional[float] = None
    theoretical_compute_gflops: Optional[float] = None

    # Efficiency ratios
    bandwidth_efficiency: float = 0.0  # achieved / theoretical
    compute_efficiency: float = 0.0    # achieved / theoretical

    # Metadata
    precision: str = "fp32"  # Precision these parameters apply to
    device_name: str = ""

    def __post_init__(self):
        # Calculate efficiency ratios
        if self.theoretical_bandwidth_gbps and self.theoretical_bandwidth_gbps > 0:
            self.bandwidth_efficiency = self.achieved_bandwidth_gbps / self.theoretical_bandwidth_gbps
        if self.theoretical_compute_gflops and self.theoretical_compute_gflops > 0:
            self.compute_efficiency = self.achieved_compute_gflops / self.theoretical_compute_gflops

    def predict_gflops(self, arithmetic_intensity: float) -> float:
        """
        Predict achievable GFLOPS for given arithmetic intensity.

        Args:
            arithmetic_intensity: FLOPs per byte of data movement

        Returns:
            Predicted GFLOPS (bounded by roofline)
        """
        memory_bound = self.achieved_bandwidth_gbps * arithmetic_intensity
        return min(memory_bound, self.achieved_compute_gflops)

    def is_memory_bound(self, arithmetic_intensity: float) -> bool:
        """Check if operation is memory-bound at given arithmetic intensity"""
        return arithmetic_intensity < self.ridge_point

    def bottleneck(self, arithmetic_intensity: float) -> str:
        """Return bottleneck type for given arithmetic intensity"""
        if arithmetic_intensity < self.ridge_point:
            return "memory"
        else:
            return "compute"

    def to_dict(self) -> Dict[str, Any]:
        return {
            'achieved_bandwidth_gbps': self.achieved_bandwidth_gbps,
            'achieved_compute_gflops': self.achieved_compute_gflops,
            'ridge_point': self.ridge_point,
            'bandwidth_fit': self.bandwidth_fit.to_dict() if self.bandwidth_fit else None,
            'compute_fit': self.compute_fit.to_dict() if self.compute_fit else None,
            'theoretical_bandwidth_gbps': self.theoretical_bandwidth_gbps,
            'theoretical_compute_gflops': self.theoretical_compute_gflops,
            'bandwidth_efficiency': self.bandwidth_efficiency,
            'compute_efficiency': self.compute_efficiency,
            'precision': self.precision,
            'device_name': self.device_name,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RooflineParameters':
        data = data.copy()
        if data.get('bandwidth_fit'):
            data['bandwidth_fit'] = FitMetrics.from_dict(data['bandwidth_fit'])
        if data.get('compute_fit'):
            data['compute_fit'] = FitMetrics.from_dict(data['compute_fit'])
        return cls(**data)


class RooflineFitter:
    """
    Fits roofline model parameters from benchmark measurements.

    Usage:
        fitter = RooflineFitter()

        # Add benchmark results
        for result in memory_results:
            fitter.add_memory_result(result)
        for result in gemm_results:
            fitter.add_compute_result(result)

        # Fit parameters
        params = fitter.fit()

        # Or use convenience method
        params = RooflineFitter.fit_from_results(all_results)
    """

    # Minimum data points for reliable fitting
    MIN_POINTS_BANDWIDTH = 3
    MIN_POINTS_COMPUTE = 3

    def __init__(
        self,
        theoretical_bandwidth_gbps: Optional[float] = None,
        theoretical_compute_gflops: Optional[float] = None,
    ):
        """
        Initialize the fitter.

        Args:
            theoretical_bandwidth_gbps: Theoretical peak bandwidth for comparison
            theoretical_compute_gflops: Theoretical peak GFLOPS for comparison
        """
        self.theoretical_bandwidth_gbps = theoretical_bandwidth_gbps
        self.theoretical_compute_gflops = theoretical_compute_gflops

        # Storage for measurements
        self._bandwidth_measurements: List[float] = []  # GB/s values
        self._compute_measurements: List[Tuple[float, float]] = []  # (AI, GFLOPS)

        # Metadata from results
        self._precision: str = "fp32"
        self._device_name: str = ""

    def add_memory_result(self, result: BenchmarkResult) -> None:
        """
        Add a memory benchmark result for bandwidth fitting.

        Args:
            result: BenchmarkResult from a memory/STREAM benchmark
        """
        if not result.success:
            return

        if result.bandwidth_gbps > 0:
            self._bandwidth_measurements.append(result.bandwidth_gbps)

        # Capture device name
        if result.device_name and not self._device_name:
            self._device_name = result.device_name

    def add_compute_result(
        self,
        result: BenchmarkResult,
        spec: Optional[GEMMSpec] = None,
        arithmetic_intensity: Optional[float] = None,
    ) -> None:
        """
        Add a compute benchmark result for throughput fitting.

        Args:
            result: BenchmarkResult from a GEMM benchmark
            spec: Optional GEMMSpec to calculate arithmetic intensity
            arithmetic_intensity: Override arithmetic intensity (FLOPs/byte)
        """
        if not result.success:
            return

        # Determine arithmetic intensity
        ai = arithmetic_intensity
        if ai is None and spec is not None:
            ai = spec.arithmetic_intensity
        if ai is None:
            # Estimate from result if available
            if 'arithmetic_intensity' in result.extra:
                ai = result.extra['arithmetic_intensity']
            else:
                # Cannot fit without AI
                return

        if result.gflops > 0 and ai > 0:
            self._compute_measurements.append((ai, result.gflops))

        # Capture metadata
        if result.device_name and not self._device_name:
            self._device_name = result.device_name
        if result.precision:
            self._precision = result.precision

    def add_result(
        self,
        result: BenchmarkResult,
        spec: Optional[Any] = None,
    ) -> None:
        """
        Add any benchmark result, routing to appropriate handler.

        Args:
            result: BenchmarkResult from any benchmark type
            spec: Optional spec for the benchmark
        """
        # Determine type from result or spec
        is_memory = False
        if spec is not None:
            is_memory = isinstance(spec, MemoryBenchSpec)
        elif 'memory' in result.spec_name.lower() or 'stream' in result.spec_name.lower():
            is_memory = True
        elif result.bandwidth_gbps > 0 and result.gflops == 0:
            is_memory = True

        if is_memory:
            self.add_memory_result(result)
        else:
            self.add_compute_result(result, spec if isinstance(spec, GEMMSpec) else None)

    def can_fit_bandwidth(self) -> bool:
        """Check if sufficient data for bandwidth fitting"""
        return len(self._bandwidth_measurements) >= self.MIN_POINTS_BANDWIDTH

    def can_fit_compute(self) -> bool:
        """Check if sufficient data for compute fitting"""
        return len(self._compute_measurements) >= self.MIN_POINTS_COMPUTE

    def can_fit(self) -> bool:
        """Check if sufficient data for any fitting"""
        return self.can_fit_bandwidth() or self.can_fit_compute()

    def fit_bandwidth(self) -> Tuple[float, FitMetrics]:
        """
        Fit memory bandwidth ceiling from measurements.

        Uses robust statistics (median with MAD) to handle outliers.

        Returns:
            Tuple of (fitted_bandwidth_gbps, fit_metrics)

        Raises:
            ValueError: If insufficient data for fitting
        """
        n = len(self._bandwidth_measurements)
        if n < self.MIN_POINTS_BANDWIDTH:
            raise ValueError(
                f"Insufficient data for bandwidth fitting: {n} points, "
                f"need >= {self.MIN_POINTS_BANDWIDTH}"
            )

        measurements = np.array(self._bandwidth_measurements)

        # Use median for robust estimation (handles outliers)
        fitted_value = float(np.median(measurements))

        # Calculate fit quality metrics
        residuals = measurements - fitted_value
        residual_std = float(np.std(residuals))

        # R-squared for constant model (variance explained)
        total_variance = float(np.var(measurements))
        if total_variance > 0:
            r_squared = 1.0 - (residual_std ** 2 / total_variance)
        else:
            r_squared = 1.0  # Perfect fit if no variance

        # Determine quality
        quality = self._assess_fit_quality(n, r_squared)

        metrics = FitMetrics(
            num_data_points=n,
            r_squared=max(0.0, r_squared),
            residual_std=residual_std,
            quality=quality,
        )

        return fitted_value, metrics

    def fit_compute(self) -> Tuple[float, FitMetrics]:
        """
        Fit compute throughput ceiling from measurements.

        For compute ceiling, we want the asymptotic maximum GFLOPS
        achieved at high arithmetic intensity (compute-bound region).

        Returns:
            Tuple of (fitted_compute_gflops, fit_metrics)

        Raises:
            ValueError: If insufficient data for fitting
        """
        n = len(self._compute_measurements)
        if n < self.MIN_POINTS_COMPUTE:
            raise ValueError(
                f"Insufficient data for compute fitting: {n} points, "
                f"need >= {self.MIN_POINTS_COMPUTE}"
            )

        ai_values = np.array([m[0] for m in self._compute_measurements])
        gflops_values = np.array([m[1] for m in self._compute_measurements])

        # Find the compute ceiling: maximum GFLOPS at high AI
        # Use the 90th percentile to handle outliers
        fitted_value = float(np.percentile(gflops_values, 90))

        # For points in compute-bound region (high AI), check fit
        # Estimate ridge point as where gflops starts to plateau
        if self.theoretical_bandwidth_gbps:
            estimated_ridge = fitted_value / self.theoretical_bandwidth_gbps
        else:
            # Use data to estimate - points where gflops/AI starts decreasing
            estimated_ridge = np.median(ai_values)

        # Calculate residuals for compute-bound points
        compute_bound_mask = ai_values >= estimated_ridge * 0.5
        if np.sum(compute_bound_mask) >= 2:
            compute_bound_gflops = gflops_values[compute_bound_mask]
            residuals = compute_bound_gflops - fitted_value
            residual_std = float(np.std(residuals))
            total_variance = float(np.var(compute_bound_gflops))
        else:
            residuals = gflops_values - fitted_value
            residual_std = float(np.std(residuals))
            total_variance = float(np.var(gflops_values))

        if total_variance > 0:
            r_squared = 1.0 - (residual_std ** 2 / total_variance)
        else:
            r_squared = 1.0

        quality = self._assess_fit_quality(n, r_squared)

        metrics = FitMetrics(
            num_data_points=n,
            r_squared=max(0.0, r_squared),
            residual_std=residual_std,
            quality=quality,
        )

        return fitted_value, metrics

    def _assess_fit_quality(self, n: int, r_squared: float) -> FitQuality:
        """Assess fit quality from number of points and R-squared"""
        if n < 3:
            return FitQuality.INSUFFICIENT

        if r_squared >= 0.95 and n >= 10:
            return FitQuality.EXCELLENT
        elif r_squared >= 0.85 and n >= 5:
            return FitQuality.GOOD
        elif r_squared >= 0.70 and n >= 3:
            return FitQuality.FAIR
        else:
            return FitQuality.POOR

    def fit(self) -> RooflineParameters:
        """
        Fit roofline parameters from all collected measurements.

        Returns:
            RooflineParameters with fitted ceilings

        Raises:
            ValueError: If insufficient data for both bandwidth and compute
        """
        if not self.can_fit():
            raise ValueError(
                f"Insufficient data for fitting: "
                f"{len(self._bandwidth_measurements)} bandwidth points, "
                f"{len(self._compute_measurements)} compute points"
            )

        # Fit bandwidth if possible
        bandwidth_value = 0.0
        bandwidth_metrics = None
        if self.can_fit_bandwidth():
            bandwidth_value, bandwidth_metrics = self.fit_bandwidth()
        elif self.theoretical_bandwidth_gbps:
            # Use theoretical as fallback
            bandwidth_value = self.theoretical_bandwidth_gbps

        # Fit compute if possible
        compute_value = 0.0
        compute_metrics = None
        if self.can_fit_compute():
            compute_value, compute_metrics = self.fit_compute()
        elif self.theoretical_compute_gflops:
            # Use theoretical as fallback
            compute_value = self.theoretical_compute_gflops

        # Calculate ridge point
        if bandwidth_value > 0:
            ridge_point = compute_value / bandwidth_value
        else:
            ridge_point = 0.0

        return RooflineParameters(
            achieved_bandwidth_gbps=bandwidth_value,
            achieved_compute_gflops=compute_value,
            ridge_point=ridge_point,
            bandwidth_fit=bandwidth_metrics,
            compute_fit=compute_metrics,
            theoretical_bandwidth_gbps=self.theoretical_bandwidth_gbps,
            theoretical_compute_gflops=self.theoretical_compute_gflops,
            precision=self._precision,
            device_name=self._device_name,
        )

    def reset(self) -> None:
        """Clear all stored measurements"""
        self._bandwidth_measurements.clear()
        self._compute_measurements.clear()
        self._precision = "fp32"
        self._device_name = ""

    @classmethod
    def fit_from_results(
        cls,
        results: List[BenchmarkResult],
        specs: Optional[Dict[str, Any]] = None,
        theoretical_bandwidth_gbps: Optional[float] = None,
        theoretical_compute_gflops: Optional[float] = None,
    ) -> RooflineParameters:
        """
        Convenience method to fit from a list of results.

        Args:
            results: List of BenchmarkResult objects
            specs: Optional dict mapping spec_name -> spec object
            theoretical_bandwidth_gbps: Theoretical bandwidth for comparison
            theoretical_compute_gflops: Theoretical GFLOPS for comparison

        Returns:
            Fitted RooflineParameters
        """
        fitter = cls(
            theoretical_bandwidth_gbps=theoretical_bandwidth_gbps,
            theoretical_compute_gflops=theoretical_compute_gflops,
        )

        specs = specs or {}
        for result in results:
            spec = specs.get(result.spec_name)
            fitter.add_result(result, spec)

        return fitter.fit()


def fit_roofline(
    results: List[BenchmarkResult],
    theoretical_bandwidth_gbps: Optional[float] = None,
    theoretical_compute_gflops: Optional[float] = None,
) -> RooflineParameters:
    """
    Main fitting function - fits roofline parameters from benchmark results.

    This is the primary interface for roofline fitting.

    Args:
        results: List of BenchmarkResult from benchmarks
        theoretical_bandwidth_gbps: Theoretical peak bandwidth (optional)
        theoretical_compute_gflops: Theoretical peak GFLOPS (optional)

    Returns:
        RooflineParameters with fitted ceilings and quality metrics

    Example:
        >>> from graphs.benchmarks.schema import BenchmarkResult
        >>> results = load_benchmark_results("results/")
        >>> params = fit_roofline(results, theoretical_bandwidth_gbps=2039.0)
        >>> print(f"Achieved bandwidth: {params.achieved_bandwidth_gbps:.1f} GB/s")
        >>> print(f"Efficiency: {params.bandwidth_efficiency:.1%}")
    """
    return RooflineFitter.fit_from_results(
        results,
        theoretical_bandwidth_gbps=theoretical_bandwidth_gbps,
        theoretical_compute_gflops=theoretical_compute_gflops,
    )
