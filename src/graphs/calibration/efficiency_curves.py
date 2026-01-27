"""
Efficiency Curve Fitting and Models

Models efficiency as a function of problem size for different operation types.
Efficiency typically increases with problem size due to:
- Better amortization of fixed overhead (kernel launch, etc.)
- Improved memory access patterns
- Better utilization of parallel resources

Supported curve models:
- AsymptoticCurve: efficiency = peak * (1 - exp(-size/scale))
- PiecewiseLinearCurve: Linear segments with breakpoints
- PolynomialCurve: Polynomial fit for complex behaviors
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
from enum import Enum


class CurveType(Enum):
    """Types of efficiency curve models"""
    ASYMPTOTIC = "asymptotic"
    PIECEWISE_LINEAR = "piecewise_linear"
    POLYNOMIAL = "polynomial"
    CONSTANT = "constant"


@dataclass
class CurveFitResult:
    """Result of fitting an efficiency curve"""
    success: bool
    curve_type: CurveType
    parameters: Dict[str, float]
    r_squared: float
    rmse: float  # Root mean squared error
    num_points: int
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            'success': self.success,
            'curve_type': self.curve_type.value,
            'parameters': self.parameters,
            'r_squared': self.r_squared,
            'rmse': self.rmse,
            'num_points': self.num_points,
            'message': self.message,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CurveFitResult':
        data = data.copy()
        data['curve_type'] = CurveType(data['curve_type'])
        return cls(**data)


class EfficiencyCurve(ABC):
    """
    Abstract base class for efficiency curves.

    Efficiency curves model how hardware efficiency varies with problem size.
    """

    @abstractmethod
    def predict(self, size: Union[int, float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Predict efficiency for given problem size(s).

        Args:
            size: Problem size (e.g., matrix dimension, total FLOPs)

        Returns:
            Efficiency value(s) in range [0.0, 1.0]
        """
        pass

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Serialize curve to dictionary"""
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EfficiencyCurve':
        """Deserialize curve from dictionary"""
        pass

    @abstractmethod
    def fit(self, sizes: np.ndarray, efficiencies: np.ndarray) -> CurveFitResult:
        """
        Fit curve parameters from data.

        Args:
            sizes: Array of problem sizes
            efficiencies: Array of measured efficiencies

        Returns:
            CurveFitResult with fit quality metrics
        """
        pass

    @property
    @abstractmethod
    def curve_type(self) -> CurveType:
        """Return the curve type"""
        pass


class AsymptoticCurve(EfficiencyCurve):
    """
    Asymptotic efficiency curve model.

    Models efficiency as approaching a peak value asymptotically:
        efficiency(size) = peak * (1 - exp(-size / scale))

    Parameters:
        peak: Maximum achievable efficiency (0.0 to 1.0)
        scale: Size at which efficiency reaches ~63% of peak

    This model captures the common pattern where:
    - Small problems have low efficiency (overhead-dominated)
    - Large problems approach a peak efficiency (compute-dominated)
    """

    def __init__(self, peak: float = 1.0, scale: float = 1000.0):
        """
        Initialize asymptotic curve.

        Args:
            peak: Peak efficiency (0.0 to 1.0)
            scale: Scale parameter (problem size for 63% of peak)
        """
        self.peak = max(0.0, min(1.0, peak))
        self.scale = max(1.0, scale)  # Avoid division by zero

    @property
    def curve_type(self) -> CurveType:
        return CurveType.ASYMPTOTIC

    def predict(self, size: Union[int, float, np.ndarray]) -> Union[float, np.ndarray]:
        size = np.asarray(size)
        efficiency = self.peak * (1.0 - np.exp(-size / self.scale))
        return float(efficiency) if efficiency.ndim == 0 else efficiency

    def to_dict(self) -> Dict[str, Any]:
        return {
            'curve_type': self.curve_type.value,
            'peak': self.peak,
            'scale': self.scale,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AsymptoticCurve':
        return cls(peak=data['peak'], scale=data['scale'])

    def fit(self, sizes: np.ndarray, efficiencies: np.ndarray) -> CurveFitResult:
        """
        Fit peak and scale from data using grid search optimization.

        The model is: efficiency = peak * (1 - exp(-size/scale))
        """
        sizes = np.asarray(sizes, dtype=float)
        efficiencies = np.asarray(efficiencies, dtype=float)

        n = len(sizes)
        if n < 2:
            return CurveFitResult(
                success=False,
                curve_type=self.curve_type,
                parameters={'peak': self.peak, 'scale': self.scale},
                r_squared=0.0,
                rmse=float('inf'),
                num_points=n,
                message="Insufficient data points (need >= 2)",
            )

        # Estimate peak as slightly above max observed efficiency
        self.peak = min(1.0, np.max(efficiencies) * 1.05)

        # Grid search for scale parameter
        # Scale should be roughly the size where efficiency reaches 63% of peak
        # Try a range of scales based on the data
        size_range = np.max(sizes) - np.min(sizes)
        scales_to_try = np.logspace(
            np.log10(max(1.0, np.min(sizes) / 10)),
            np.log10(np.max(sizes) * 10),
            50
        )

        best_scale = self.scale
        best_rmse = float('inf')

        for scale in scales_to_try:
            predictions = self.peak * (1.0 - np.exp(-sizes / scale))
            rmse = np.sqrt(np.mean((predictions - efficiencies) ** 2))
            if rmse < best_rmse:
                best_rmse = rmse
                best_scale = scale

        self.scale = best_scale

        # Calculate fit quality
        predictions = self.predict(sizes)
        residuals = efficiencies - predictions
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((efficiencies - np.mean(efficiencies)) ** 2)

        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        rmse = float(np.sqrt(np.mean(residuals ** 2)))

        return CurveFitResult(
            success=True,
            curve_type=self.curve_type,
            parameters={'peak': self.peak, 'scale': self.scale},
            r_squared=max(0.0, r_squared),
            rmse=rmse,
            num_points=n,
            message="Fit successful",
        )


class PiecewiseLinearCurve(EfficiencyCurve):
    """
    Piecewise linear efficiency curve.

    Models efficiency as connected linear segments with breakpoints.
    Useful for capturing distinct performance regimes (e.g., L1/L2/L3 cache).

    Parameters:
        breakpoints: List of (size, efficiency) points defining segments
    """

    def __init__(self, breakpoints: Optional[List[Tuple[float, float]]] = None):
        """
        Initialize piecewise linear curve.

        Args:
            breakpoints: List of (size, efficiency) tuples, sorted by size
        """
        if breakpoints is None:
            breakpoints = [(0.0, 0.0), (1000.0, 0.5), (1000000.0, 0.9)]

        # Ensure sorted by size
        self.breakpoints = sorted(breakpoints, key=lambda x: x[0])

    @property
    def curve_type(self) -> CurveType:
        return CurveType.PIECEWISE_LINEAR

    def predict(self, size: Union[int, float, np.ndarray]) -> Union[float, np.ndarray]:
        size = np.asarray(size)
        scalar_input = size.ndim == 0
        size = np.atleast_1d(size)

        result = np.zeros_like(size, dtype=float)

        sizes = np.array([bp[0] for bp in self.breakpoints])
        effs = np.array([bp[1] for bp in self.breakpoints])

        # Interpolate
        result = np.interp(size, sizes, effs)

        return float(result[0]) if scalar_input else result

    def to_dict(self) -> Dict[str, Any]:
        return {
            'curve_type': self.curve_type.value,
            'breakpoints': self.breakpoints,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PiecewiseLinearCurve':
        breakpoints = [tuple(bp) for bp in data['breakpoints']]
        return cls(breakpoints=breakpoints)

    def fit(self, sizes: np.ndarray, efficiencies: np.ndarray) -> CurveFitResult:
        """
        Fit piecewise linear curve from data.

        Uses quantile-based breakpoints for automatic segmentation.
        """
        sizes = np.asarray(sizes, dtype=float)
        efficiencies = np.asarray(efficiencies, dtype=float)

        n = len(sizes)
        if n < 2:
            return CurveFitResult(
                success=False,
                curve_type=self.curve_type,
                parameters={'breakpoints': self.breakpoints},
                r_squared=0.0,
                rmse=float('inf'),
                num_points=n,
                message="Insufficient data points (need >= 2)",
            )

        # Sort by size
        sort_idx = np.argsort(sizes)
        sizes_sorted = sizes[sort_idx]
        effs_sorted = efficiencies[sort_idx]

        # Create breakpoints at quantiles
        if n <= 5:
            # Use all points as breakpoints
            self.breakpoints = list(zip(sizes_sorted.tolist(), effs_sorted.tolist()))
        else:
            # Use quantiles (0%, 25%, 50%, 75%, 100%)
            quantiles = [0, 25, 50, 75, 100]
            bp_sizes = np.percentile(sizes_sorted, quantiles)
            bp_effs = np.interp(bp_sizes, sizes_sorted, effs_sorted)
            self.breakpoints = list(zip(bp_sizes.tolist(), bp_effs.tolist()))

        # Calculate fit quality
        predictions = self.predict(sizes)
        residuals = efficiencies - predictions
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((efficiencies - np.mean(efficiencies)) ** 2)

        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        rmse = float(np.sqrt(np.mean(residuals ** 2)))

        return CurveFitResult(
            success=True,
            curve_type=self.curve_type,
            parameters={'breakpoints': self.breakpoints},
            r_squared=max(0.0, r_squared),
            rmse=rmse,
            num_points=n,
            message="Fit successful",
        )


class PolynomialCurve(EfficiencyCurve):
    """
    Polynomial efficiency curve.

    Models efficiency as a polynomial of log(size):
        efficiency = sum(coeffs[i] * log(size)^i)

    Using log(size) provides better numerical stability and captures
    the typical logarithmic scaling of efficiency with problem size.

    Parameters:
        coefficients: Polynomial coefficients [c0, c1, c2, ...]
        degree: Polynomial degree
    """

    def __init__(self, coefficients: Optional[List[float]] = None, degree: int = 2):
        """
        Initialize polynomial curve.

        Args:
            coefficients: Polynomial coefficients (lowest to highest degree)
            degree: Polynomial degree (used if coefficients not provided)
        """
        if coefficients is None:
            # Default: constant efficiency of 0.5
            self.coefficients = [0.5] + [0.0] * degree
        else:
            self.coefficients = list(coefficients)

        self.degree = len(self.coefficients) - 1

    @property
    def curve_type(self) -> CurveType:
        return CurveType.POLYNOMIAL

    def predict(self, size: Union[int, float, np.ndarray]) -> Union[float, np.ndarray]:
        size = np.asarray(size)
        scalar_input = size.ndim == 0
        size = np.atleast_1d(size)

        # Use log(size) for better scaling
        log_size = np.log(np.maximum(size, 1.0))

        # Evaluate polynomial
        result = np.polyval(self.coefficients[::-1], log_size)

        # Clip to valid efficiency range
        result = np.clip(result, 0.0, 1.0)

        return float(result[0]) if scalar_input else result

    def to_dict(self) -> Dict[str, Any]:
        return {
            'curve_type': self.curve_type.value,
            'coefficients': self.coefficients,
            'degree': self.degree,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PolynomialCurve':
        return cls(
            coefficients=data['coefficients'],
            degree=data.get('degree', len(data['coefficients']) - 1),
        )

    def fit(self, sizes: np.ndarray, efficiencies: np.ndarray) -> CurveFitResult:
        """
        Fit polynomial coefficients using least squares.
        """
        sizes = np.asarray(sizes, dtype=float)
        efficiencies = np.asarray(efficiencies, dtype=float)

        n = len(sizes)
        min_points = self.degree + 1
        if n < min_points:
            return CurveFitResult(
                success=False,
                curve_type=self.curve_type,
                parameters={'coefficients': self.coefficients, 'degree': self.degree},
                r_squared=0.0,
                rmse=float('inf'),
                num_points=n,
                message=f"Insufficient data points (need >= {min_points} for degree {self.degree})",
            )

        # Transform to log scale
        log_sizes = np.log(np.maximum(sizes, 1.0))

        # Fit polynomial
        try:
            coeffs = np.polyfit(log_sizes, efficiencies, self.degree)
            self.coefficients = coeffs[::-1].tolist()  # Reverse to low-to-high order
        except (np.linalg.LinAlgError, ValueError) as e:
            return CurveFitResult(
                success=False,
                curve_type=self.curve_type,
                parameters={'coefficients': self.coefficients, 'degree': self.degree},
                r_squared=0.0,
                rmse=float('inf'),
                num_points=n,
                message=f"Fit failed: {str(e)}",
            )

        # Calculate fit quality
        predictions = self.predict(sizes)
        residuals = efficiencies - predictions
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((efficiencies - np.mean(efficiencies)) ** 2)

        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        rmse = float(np.sqrt(np.mean(residuals ** 2)))

        return CurveFitResult(
            success=True,
            curve_type=self.curve_type,
            parameters={'coefficients': self.coefficients, 'degree': self.degree},
            r_squared=max(0.0, r_squared),
            rmse=rmse,
            num_points=n,
            message="Fit successful",
        )


class ConstantCurve(EfficiencyCurve):
    """
    Constant efficiency curve (size-independent).

    Used when efficiency does not vary significantly with problem size,
    or as a simple baseline model.
    """

    def __init__(self, efficiency: float = 0.5):
        self.efficiency = max(0.0, min(1.0, efficiency))

    @property
    def curve_type(self) -> CurveType:
        return CurveType.CONSTANT

    def predict(self, size: Union[int, float, np.ndarray]) -> Union[float, np.ndarray]:
        size = np.asarray(size)
        if size.ndim == 0:
            return self.efficiency
        return np.full_like(size, self.efficiency, dtype=float)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'curve_type': self.curve_type.value,
            'efficiency': self.efficiency,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConstantCurve':
        return cls(efficiency=data['efficiency'])

    def fit(self, sizes: np.ndarray, efficiencies: np.ndarray) -> CurveFitResult:
        """Fit constant efficiency as the mean of observations"""
        efficiencies = np.asarray(efficiencies, dtype=float)
        n = len(efficiencies)

        if n == 0:
            return CurveFitResult(
                success=False,
                curve_type=self.curve_type,
                parameters={'efficiency': self.efficiency},
                r_squared=0.0,
                rmse=float('inf'),
                num_points=0,
                message="No data points",
            )

        self.efficiency = float(np.mean(efficiencies))

        # R-squared is 0 for constant model (explains no variance)
        residuals = efficiencies - self.efficiency
        rmse = float(np.sqrt(np.mean(residuals ** 2)))

        return CurveFitResult(
            success=True,
            curve_type=self.curve_type,
            parameters={'efficiency': self.efficiency},
            r_squared=0.0,  # Constant model explains no variance
            rmse=rmse,
            num_points=n,
            message="Fit successful (constant model)",
        )


@dataclass
class EfficiencyProfile:
    """
    Collection of efficiency curves by operation type.

    Groups efficiency curves for different operations (GEMM, Conv2D, etc.)
    and precisions to provide comprehensive efficiency modeling.
    """
    # Operation type -> precision -> curve
    curves: Dict[str, Dict[str, EfficiencyCurve]] = field(default_factory=dict)

    # Metadata
    device_name: str = ""
    created_at: str = ""

    def add_curve(
        self,
        operation: str,
        curve: EfficiencyCurve,
        precision: str = "fp32",
    ) -> None:
        """Add an efficiency curve for an operation/precision"""
        if operation not in self.curves:
            self.curves[operation] = {}
        self.curves[operation][precision] = curve

    def get_curve(
        self,
        operation: str,
        precision: str = "fp32",
    ) -> Optional[EfficiencyCurve]:
        """Get efficiency curve for operation/precision"""
        if operation in self.curves:
            return self.curves[operation].get(precision)
        return None

    def predict_efficiency(
        self,
        operation: str,
        size: Union[int, float],
        precision: str = "fp32",
        default: float = 0.5,
    ) -> float:
        """
        Predict efficiency for given operation, size, and precision.

        Args:
            operation: Operation type (e.g., "gemm", "conv2d")
            size: Problem size
            precision: Numerical precision
            default: Default efficiency if no curve found

        Returns:
            Predicted efficiency (0.0 to 1.0)
        """
        curve = self.get_curve(operation, precision)
        if curve is None:
            # Try default precision
            curve = self.get_curve(operation, "fp32")
        if curve is None:
            return default
        return float(curve.predict(size))

    def to_dict(self) -> Dict[str, Any]:
        curves_dict = {}
        for op, prec_dict in self.curves.items():
            curves_dict[op] = {}
            for prec, curve in prec_dict.items():
                curves_dict[op][prec] = curve.to_dict()

        return {
            'curves': curves_dict,
            'device_name': self.device_name,
            'created_at': self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EfficiencyProfile':
        curves = {}
        for op, prec_dict in data.get('curves', {}).items():
            curves[op] = {}
            for prec, curve_data in prec_dict.items():
                curve_type = CurveType(curve_data['curve_type'])
                if curve_type == CurveType.ASYMPTOTIC:
                    curves[op][prec] = AsymptoticCurve.from_dict(curve_data)
                elif curve_type == CurveType.PIECEWISE_LINEAR:
                    curves[op][prec] = PiecewiseLinearCurve.from_dict(curve_data)
                elif curve_type == CurveType.POLYNOMIAL:
                    curves[op][prec] = PolynomialCurve.from_dict(curve_data)
                elif curve_type == CurveType.CONSTANT:
                    curves[op][prec] = ConstantCurve.from_dict(curve_data)

        return cls(
            curves=curves,
            device_name=data.get('device_name', ''),
            created_at=data.get('created_at', ''),
        )


def fit_efficiency_curve(
    sizes: List[float],
    efficiencies: List[float],
    curve_type: CurveType = CurveType.ASYMPTOTIC,
) -> Tuple[EfficiencyCurve, CurveFitResult]:
    """
    Fit an efficiency curve to data.

    Args:
        sizes: Problem sizes
        efficiencies: Measured efficiencies
        curve_type: Type of curve to fit

    Returns:
        Tuple of (fitted_curve, fit_result)
    """
    sizes_arr = np.array(sizes)
    effs_arr = np.array(efficiencies)

    if curve_type == CurveType.ASYMPTOTIC:
        curve = AsymptoticCurve()
    elif curve_type == CurveType.PIECEWISE_LINEAR:
        curve = PiecewiseLinearCurve()
    elif curve_type == CurveType.POLYNOMIAL:
        curve = PolynomialCurve(degree=2)
    elif curve_type == CurveType.CONSTANT:
        curve = ConstantCurve()
    else:
        curve = AsymptoticCurve()

    result = curve.fit(sizes_arr, effs_arr)
    return curve, result


def auto_fit_efficiency_curve(
    sizes: List[float],
    efficiencies: List[float],
) -> Tuple[EfficiencyCurve, CurveFitResult]:
    """
    Automatically select and fit the best efficiency curve.

    Tries multiple curve types and returns the one with best fit.

    Args:
        sizes: Problem sizes
        efficiencies: Measured efficiencies

    Returns:
        Tuple of (best_curve, best_fit_result)
    """
    best_curve = None
    best_result = None
    best_score = -float('inf')

    for curve_type in [CurveType.ASYMPTOTIC, CurveType.PIECEWISE_LINEAR, CurveType.POLYNOMIAL]:
        curve, result = fit_efficiency_curve(sizes, efficiencies, curve_type)

        if result.success:
            # Score based on R-squared with penalty for complexity
            complexity_penalty = {
                CurveType.CONSTANT: 0.0,
                CurveType.ASYMPTOTIC: 0.01,
                CurveType.PIECEWISE_LINEAR: 0.02,
                CurveType.POLYNOMIAL: 0.03,
            }
            score = result.r_squared - complexity_penalty.get(curve_type, 0.0)

            if score > best_score:
                best_score = score
                best_curve = curve
                best_result = result

    # Fallback to constant if all fits fail
    if best_curve is None:
        best_curve, best_result = fit_efficiency_curve(
            sizes, efficiencies, CurveType.CONSTANT
        )

    return best_curve, best_result
