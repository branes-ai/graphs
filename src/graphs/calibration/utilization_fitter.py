"""
Utilization Factor Calibration from Benchmark Results

Fits hardware utilization factors by measuring actual vs theoretical
performance across operation types and sizes:
- Compute utilization (actual GFLOPS / peak GFLOPS)
- Memory utilization (actual GB/s / peak GB/s)
- Utilization curves as function of problem size and batch size

The fitted utilization factors capture real-world effects:
- Memory system inefficiencies
- Instruction scheduling overhead
- Kernel launch latency
- Tiling and blocking losses
- Small problem size effects
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum
import numpy as np

from graphs.benchmarks.schema import (
    BenchmarkResult,
    GEMMSpec,
    Conv2dSpec,
    MemoryBenchSpec,
)
from graphs.calibration.efficiency_curves import (
    EfficiencyCurve,
    AsymptoticCurve,
    PiecewiseLinearCurve,
    PolynomialCurve,
    ConstantCurve,
    CurveType,
    CurveFitResult,
    auto_fit_efficiency_curve,
)


class UtilizationFitQuality(Enum):
    """Quality level of the fitted utilization parameters"""
    EXCELLENT = "excellent"  # R-squared >= 0.90, many data points
    GOOD = "good"            # R-squared >= 0.75, sufficient data
    FAIR = "fair"            # R-squared >= 0.50, limited data
    POOR = "poor"            # R-squared < 0.50 or very few points
    INSUFFICIENT = "insufficient"  # Not enough data to fit


@dataclass
class UtilizationFitMetrics:
    """Metrics describing the quality of a utilization fit"""
    num_data_points: int
    r_squared: float
    rmse: float  # Root mean squared error
    min_utilization: float  # Minimum observed utilization
    max_utilization: float  # Maximum observed utilization
    mean_utilization: float  # Mean observed utilization
    quality: UtilizationFitQuality

    def to_dict(self) -> Dict[str, Any]:
        return {
            'num_data_points': self.num_data_points,
            'r_squared': self.r_squared,
            'rmse': self.rmse,
            'min_utilization': self.min_utilization,
            'max_utilization': self.max_utilization,
            'mean_utilization': self.mean_utilization,
            'quality': self.quality.value,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UtilizationFitMetrics':
        data = data.copy()
        data['quality'] = UtilizationFitQuality(data['quality'])
        return cls(**data)


@dataclass
class UtilizationCurveResult:
    """Result of fitting a utilization curve for a specific operation/precision"""
    operation: str  # e.g., "gemm", "conv2d", "memory"
    precision: str  # e.g., "fp32", "fp16", "int8"
    curve: EfficiencyCurve
    fit_result: CurveFitResult
    metrics: UtilizationFitMetrics

    # Raw data used for fitting
    sizes: List[float] = field(default_factory=list)
    utilizations: List[float] = field(default_factory=list)

    def predict(self, size: Union[int, float]) -> float:
        """Predict utilization for given problem size"""
        return float(self.curve.predict(size))

    def to_dict(self) -> Dict[str, Any]:
        return {
            'operation': self.operation,
            'precision': self.precision,
            'curve': self.curve.to_dict(),
            'fit_result': self.fit_result.to_dict(),
            'metrics': self.metrics.to_dict(),
            'sizes': self.sizes,
            'utilizations': self.utilizations,
        }


@dataclass
class UtilizationProfile:
    """
    Complete utilization profile for a hardware device.

    Contains utilization curves for different operations and precisions,
    enabling prediction of real-world utilization for workloads.
    """
    # Operation -> precision -> curve result
    curves: Dict[str, Dict[str, UtilizationCurveResult]] = field(default_factory=dict)

    # Peak performance values used for normalization
    peak_compute_gflops: float = 0.0
    peak_bandwidth_gbps: float = 0.0

    # Per-precision peak compute
    peak_compute_by_precision: Dict[str, float] = field(default_factory=dict)

    # Metadata
    device_name: str = ""
    created_at: str = ""

    def add_curve(self, result: UtilizationCurveResult) -> None:
        """Add a utilization curve result"""
        if result.operation not in self.curves:
            self.curves[result.operation] = {}
        self.curves[result.operation][result.precision] = result

    def get_curve(
        self,
        operation: str,
        precision: str = "fp32",
    ) -> Optional[UtilizationCurveResult]:
        """Get utilization curve for operation/precision"""
        if operation in self.curves:
            return self.curves[operation].get(precision)
        return None

    def predict_utilization(
        self,
        operation: str,
        size: Union[int, float],
        precision: str = "fp32",
        default: float = 0.5,
    ) -> float:
        """
        Predict utilization for given operation, size, and precision.

        Args:
            operation: Operation type (e.g., "gemm", "conv2d")
            size: Problem size (FLOPs, elements, or other metric)
            precision: Numerical precision
            default: Default utilization if no curve found

        Returns:
            Predicted utilization (0.0 to 1.0)
        """
        curve_result = self.get_curve(operation, precision)
        if curve_result is None:
            # Try default precision
            curve_result = self.get_curve(operation, "fp32")
        if curve_result is None:
            return default
        return curve_result.predict(size)

    def get_peak_gflops(self, precision: str = "fp32") -> float:
        """Get peak GFLOPS for given precision"""
        return self.peak_compute_by_precision.get(precision, self.peak_compute_gflops)

    def predict_achievable_gflops(
        self,
        operation: str,
        size: Union[int, float],
        precision: str = "fp32",
    ) -> float:
        """
        Predict achievable GFLOPS based on utilization.

        Args:
            operation: Operation type
            size: Problem size
            precision: Numerical precision

        Returns:
            Predicted achievable GFLOPS
        """
        utilization = self.predict_utilization(operation, size, precision)
        peak = self.get_peak_gflops(precision)
        return utilization * peak

    def to_dict(self) -> Dict[str, Any]:
        curves_dict = {}
        for op, prec_dict in self.curves.items():
            curves_dict[op] = {}
            for prec, result in prec_dict.items():
                curves_dict[op][prec] = result.to_dict()

        return {
            'curves': curves_dict,
            'peak_compute_gflops': self.peak_compute_gflops,
            'peak_bandwidth_gbps': self.peak_bandwidth_gbps,
            'peak_compute_by_precision': self.peak_compute_by_precision,
            'device_name': self.device_name,
            'created_at': self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UtilizationProfile':
        curves = {}
        for op, prec_dict in data.get('curves', {}).items():
            curves[op] = {}
            for prec, curve_data in prec_dict.items():
                # Reconstruct the curve
                curve_type = CurveType(curve_data['curve']['curve_type'])
                if curve_type == CurveType.ASYMPTOTIC:
                    curve = AsymptoticCurve.from_dict(curve_data['curve'])
                elif curve_type == CurveType.PIECEWISE_LINEAR:
                    curve = PiecewiseLinearCurve.from_dict(curve_data['curve'])
                elif curve_type == CurveType.POLYNOMIAL:
                    curve = PolynomialCurve.from_dict(curve_data['curve'])
                else:
                    curve = ConstantCurve.from_dict(curve_data['curve'])

                curves[op][prec] = UtilizationCurveResult(
                    operation=curve_data['operation'],
                    precision=curve_data['precision'],
                    curve=curve,
                    fit_result=CurveFitResult.from_dict(curve_data['fit_result']),
                    metrics=UtilizationFitMetrics.from_dict(curve_data['metrics']),
                    sizes=curve_data.get('sizes', []),
                    utilizations=curve_data.get('utilizations', []),
                )

        return cls(
            curves=curves,
            peak_compute_gflops=data.get('peak_compute_gflops', 0.0),
            peak_bandwidth_gbps=data.get('peak_bandwidth_gbps', 0.0),
            peak_compute_by_precision=data.get('peak_compute_by_precision', {}),
            device_name=data.get('device_name', ''),
            created_at=data.get('created_at', ''),
        )


class UtilizationFitter:
    """
    Fits utilization curves from benchmark measurements.

    Utilization is defined as:
    - Compute utilization = achieved_gflops / peak_gflops
    - Memory utilization = achieved_gbps / peak_gbps

    The fitter collects benchmark results, calculates utilization,
    and fits curves that model utilization as a function of problem size.

    Usage:
        fitter = UtilizationFitter(
            peak_compute_gflops=50000.0,
            peak_bandwidth_gbps=2000.0,
        )

        # Add benchmark results
        for result in gemm_results:
            fitter.add_compute_result(result, spec)
        for result in memory_results:
            fitter.add_memory_result(result)

        # Fit utilization profile
        profile = fitter.fit()

        # Use the profile
        util = profile.predict_utilization("gemm", size=1e9)
    """

    MIN_DATA_POINTS = 3  # Minimum points for curve fitting

    def __init__(
        self,
        peak_compute_gflops: float = 0.0,
        peak_bandwidth_gbps: float = 0.0,
        peak_compute_by_precision: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize the utilization fitter.

        Args:
            peak_compute_gflops: Peak compute throughput (FP32 GFLOPS)
            peak_bandwidth_gbps: Peak memory bandwidth (GB/s)
            peak_compute_by_precision: Peak GFLOPS by precision (optional)
        """
        self.peak_compute_gflops = peak_compute_gflops
        self.peak_bandwidth_gbps = peak_bandwidth_gbps
        self.peak_compute_by_precision = peak_compute_by_precision or {}

        # Ensure fp32 is in precision map
        if "fp32" not in self.peak_compute_by_precision and peak_compute_gflops > 0:
            self.peak_compute_by_precision["fp32"] = peak_compute_gflops

        # Storage for measurements: operation -> precision -> [(size, utilization)]
        self._compute_data: Dict[str, Dict[str, List[Tuple[float, float]]]] = {}
        self._memory_data: List[Tuple[float, float]] = []  # [(size, utilization)]

        # Metadata
        self._device_name: str = ""

    def _get_peak_for_precision(self, precision: str) -> float:
        """Get peak GFLOPS for a given precision"""
        return self.peak_compute_by_precision.get(precision, self.peak_compute_gflops)

    def add_compute_result(
        self,
        result: BenchmarkResult,
        spec: Optional[Union[GEMMSpec, Conv2dSpec]] = None,
    ) -> None:
        """
        Add a compute benchmark result for utilization fitting.

        Args:
            result: BenchmarkResult from a compute benchmark
            spec: Optional spec for problem size calculation
        """
        if not result.success or result.gflops <= 0:
            return

        precision = result.precision or "fp32"
        peak = self._get_peak_for_precision(precision)

        if peak <= 0:
            return  # Cannot calculate utilization without peak

        # Calculate utilization
        utilization = min(1.0, result.gflops / peak)

        # Determine problem size (FLOPs)
        size = 0.0
        if spec is not None:
            if isinstance(spec, GEMMSpec):
                size = float(spec.flops)
            elif isinstance(spec, Conv2dSpec):
                size = float(spec.flops)
        else:
            # Try to get from result
            if 'flops' in result.extra:
                size = float(result.extra['flops'])
            elif result.gflops > 0 and result.timing and result.timing.mean_ms > 0:
                # Estimate from GFLOPS and time
                size = result.gflops * 1e9 * result.timing.mean_ms / 1000.0

        if size <= 0:
            return

        # Determine operation type
        operation = "gemm"  # Default
        if spec is not None:
            if isinstance(spec, GEMMSpec):
                operation = "gemm"
            elif isinstance(spec, Conv2dSpec):
                operation = "conv2d"
        elif 'conv' in result.spec_name.lower():
            operation = "conv2d"

        # Store data point
        if operation not in self._compute_data:
            self._compute_data[operation] = {}
        if precision not in self._compute_data[operation]:
            self._compute_data[operation][precision] = []

        self._compute_data[operation][precision].append((size, utilization))

        # Capture device name
        if result.device_name and not self._device_name:
            self._device_name = result.device_name

    def add_memory_result(self, result: BenchmarkResult) -> None:
        """
        Add a memory benchmark result for utilization fitting.

        Args:
            result: BenchmarkResult from a memory benchmark
        """
        if not result.success or result.bandwidth_gbps <= 0:
            return

        if self.peak_bandwidth_gbps <= 0:
            return  # Cannot calculate utilization without peak

        # Calculate utilization
        utilization = min(1.0, result.bandwidth_gbps / self.peak_bandwidth_gbps)

        # Determine problem size (bytes)
        size = 0.0
        if 'bytes_transferred' in result.extra:
            size = float(result.extra['bytes_transferred'])
        elif result.bandwidth_gbps > 0 and result.timing and result.timing.mean_ms > 0:
            # Estimate from bandwidth and time
            size = result.bandwidth_gbps * 1e9 * result.timing.mean_ms / 1000.0

        if size <= 0:
            return

        self._memory_data.append((size, utilization))

        # Capture device name
        if result.device_name and not self._device_name:
            self._device_name = result.device_name

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
            compute_spec = spec if isinstance(spec, (GEMMSpec, Conv2dSpec)) else None
            self.add_compute_result(result, compute_spec)

    def can_fit(self) -> bool:
        """Check if sufficient data for any fitting"""
        # Check compute data
        for op_data in self._compute_data.values():
            for prec_data in op_data.values():
                if len(prec_data) >= self.MIN_DATA_POINTS:
                    return True
        # Check memory data
        if len(self._memory_data) >= self.MIN_DATA_POINTS:
            return True
        return False

    def _assess_quality(self, n: int, r_squared: float) -> UtilizationFitQuality:
        """Assess fit quality from number of points and R-squared"""
        if n < self.MIN_DATA_POINTS:
            return UtilizationFitQuality.INSUFFICIENT

        if r_squared >= 0.90 and n >= 10:
            return UtilizationFitQuality.EXCELLENT
        elif r_squared >= 0.75 and n >= 5:
            return UtilizationFitQuality.GOOD
        elif r_squared >= 0.50 and n >= 3:
            return UtilizationFitQuality.FAIR
        else:
            return UtilizationFitQuality.POOR

    def _fit_curve(
        self,
        operation: str,
        precision: str,
        data: List[Tuple[float, float]],
    ) -> Optional[UtilizationCurveResult]:
        """Fit a utilization curve from data points"""
        if len(data) < self.MIN_DATA_POINTS:
            return None

        sizes = [d[0] for d in data]
        utilizations = [d[1] for d in data]

        # Fit curve using auto-selection
        curve, fit_result = auto_fit_efficiency_curve(sizes, utilizations)

        if not fit_result.success:
            return None

        # Calculate additional metrics
        utils_arr = np.array(utilizations)
        quality = self._assess_quality(len(data), fit_result.r_squared)

        metrics = UtilizationFitMetrics(
            num_data_points=len(data),
            r_squared=fit_result.r_squared,
            rmse=fit_result.rmse,
            min_utilization=float(np.min(utils_arr)),
            max_utilization=float(np.max(utils_arr)),
            mean_utilization=float(np.mean(utils_arr)),
            quality=quality,
        )

        return UtilizationCurveResult(
            operation=operation,
            precision=precision,
            curve=curve,
            fit_result=fit_result,
            metrics=metrics,
            sizes=sizes,
            utilizations=utilizations,
        )

    def fit(self) -> UtilizationProfile:
        """
        Fit utilization curves from all collected measurements.

        Returns:
            UtilizationProfile with fitted curves

        Raises:
            ValueError: If insufficient data for fitting
        """
        if not self.can_fit():
            total_compute = sum(
                len(prec_data)
                for op_data in self._compute_data.values()
                for prec_data in op_data.values()
            )
            raise ValueError(
                f"Insufficient data for fitting: "
                f"{total_compute} compute points, "
                f"{len(self._memory_data)} memory points "
                f"(need >= {self.MIN_DATA_POINTS})"
            )

        profile = UtilizationProfile(
            peak_compute_gflops=self.peak_compute_gflops,
            peak_bandwidth_gbps=self.peak_bandwidth_gbps,
            peak_compute_by_precision=self.peak_compute_by_precision.copy(),
            device_name=self._device_name,
        )

        # Fit compute curves
        for operation, prec_data in self._compute_data.items():
            for precision, data in prec_data.items():
                result = self._fit_curve(operation, precision, data)
                if result is not None:
                    profile.add_curve(result)

        # Fit memory curve
        if len(self._memory_data) >= self.MIN_DATA_POINTS:
            result = self._fit_curve("memory", "n/a", self._memory_data)
            if result is not None:
                profile.add_curve(result)

        return profile

    def reset(self) -> None:
        """Clear all stored measurements"""
        self._compute_data.clear()
        self._memory_data.clear()
        self._device_name = ""


def fit_utilization(
    results: List[BenchmarkResult],
    peak_compute_gflops: float,
    peak_bandwidth_gbps: float = 0.0,
    peak_compute_by_precision: Optional[Dict[str, float]] = None,
    specs: Optional[Dict[str, Any]] = None,
) -> UtilizationProfile:
    """
    Main fitting function - fits utilization profile from benchmark results.

    This is the primary interface for utilization fitting.

    Args:
        results: List of BenchmarkResult from benchmarks
        peak_compute_gflops: Peak compute throughput (FP32 GFLOPS)
        peak_bandwidth_gbps: Peak memory bandwidth (GB/s)
        peak_compute_by_precision: Peak GFLOPS by precision
        specs: Optional dict mapping spec_name -> spec object

    Returns:
        UtilizationProfile with fitted curves

    Example:
        >>> from graphs.benchmarks.schema import BenchmarkResult
        >>> results = load_benchmark_results("results/")
        >>> profile = fit_utilization(
        ...     results,
        ...     peak_compute_gflops=50000.0,
        ...     peak_bandwidth_gbps=2000.0,
        ... )
        >>> util = profile.predict_utilization("gemm", size=1e12)
        >>> print(f"Predicted utilization: {util:.1%}")
    """
    fitter = UtilizationFitter(
        peak_compute_gflops=peak_compute_gflops,
        peak_bandwidth_gbps=peak_bandwidth_gbps,
        peak_compute_by_precision=peak_compute_by_precision,
    )

    specs = specs or {}
    for result in results:
        spec = specs.get(result.spec_name)
        fitter.add_result(result, spec)

    return fitter.fit()
