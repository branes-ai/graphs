"""
Utilization Curve Models

Models hardware utilization as a function of problem size.
Utilization typically increases with problem size due to:
- Better amortization of fixed overhead (kernel launch, etc.)
- Improved memory access patterns
- Better utilization of parallel resources

This module provides utilization-specific aliases and extensions
to the general-purpose efficiency curve models.

Curve Models:
- AsymptoticUtilizationCurve: util = peak * (1 - exp(-size/scale))
- PiecewiseLinearUtilizationCurve: Linear segments with breakpoints
- PolynomialUtilizationCurve: Polynomial fit for complex behaviors
- ConstantUtilizationCurve: Size-independent utilization

Usage:
    from graphs.calibration.utilization_curves import (
        AsymptoticUtilizationCurve,
        fit_utilization_curve,
        auto_fit_utilization_curve,
    )

    # Fit curve from data
    curve, result = fit_utilization_curve(sizes, utilizations)

    # Predict utilization
    util = curve.predict(problem_size)
"""

from __future__ import annotations

from typing import List, Tuple, Union
import numpy as np

# Re-export core curve classes with utilization-specific aliases
from graphs.calibration.efficiency_curves import (
    # Base class and types
    EfficiencyCurve as UtilizationCurve,
    CurveType,
    CurveFitResult,

    # Curve implementations
    AsymptoticCurve as AsymptoticUtilizationCurve,
    PiecewiseLinearCurve as PiecewiseLinearUtilizationCurve,
    PolynomialCurve as PolynomialUtilizationCurve,
    ConstantCurve as ConstantUtilizationCurve,

    # Profile container
    EfficiencyProfile,

    # Fitting functions
    fit_efficiency_curve,
    auto_fit_efficiency_curve,
)


# Utilization-specific type aliases for clarity
UtilizationCurveType = CurveType
UtilizationFitResult = CurveFitResult


def fit_utilization_curve(
    sizes: List[float],
    utilizations: List[float],
    curve_type: CurveType = CurveType.ASYMPTOTIC,
) -> Tuple[UtilizationCurve, CurveFitResult]:
    """
    Fit a utilization curve to data.

    Args:
        sizes: Problem sizes (FLOPs, bytes, elements, etc.)
        utilizations: Measured utilization values (0.0 to 1.0)
        curve_type: Type of curve to fit

    Returns:
        Tuple of (fitted_curve, fit_result)

    Example:
        >>> sizes = [1e6, 1e7, 1e8, 1e9, 1e10]
        >>> utils = [0.1, 0.3, 0.6, 0.8, 0.85]
        >>> curve, result = fit_utilization_curve(sizes, utils)
        >>> print(f"R-squared: {result.r_squared:.3f}")
        >>> print(f"Predicted at 1e11: {curve.predict(1e11):.2f}")
    """
    return fit_efficiency_curve(sizes, utilizations, curve_type)


def auto_fit_utilization_curve(
    sizes: List[float],
    utilizations: List[float],
) -> Tuple[UtilizationCurve, CurveFitResult]:
    """
    Automatically select and fit the best utilization curve.

    Tries multiple curve types and returns the one with best fit.

    Args:
        sizes: Problem sizes
        utilizations: Measured utilization values (0.0 to 1.0)

    Returns:
        Tuple of (best_curve, best_fit_result)

    Example:
        >>> sizes = [1e6, 1e7, 1e8, 1e9, 1e10]
        >>> utils = [0.1, 0.3, 0.6, 0.8, 0.85]
        >>> curve, result = auto_fit_utilization_curve(sizes, utils)
        >>> print(f"Best curve type: {result.curve_type.value}")
    """
    return auto_fit_efficiency_curve(sizes, utilizations)


def create_typical_compute_curve(
    peak_utilization: float = 0.85,
    scale_flops: float = 1e10,
) -> AsymptoticUtilizationCurve:
    """
    Create a typical compute utilization curve.

    Based on empirical observations, compute utilization typically:
    - Starts low for small problems (overhead-dominated)
    - Approaches ~85% for large problems (GPU) or ~70% (CPU)
    - Transitions around 10^10 FLOPs

    Args:
        peak_utilization: Maximum achievable utilization (0.0 to 1.0)
        scale_flops: Problem size at which utilization reaches 63% of peak

    Returns:
        AsymptoticUtilizationCurve configured with typical parameters
    """
    return AsymptoticUtilizationCurve(peak=peak_utilization, scale=scale_flops)


def create_typical_memory_curve(
    peak_utilization: float = 0.80,
    scale_bytes: float = 1e8,
) -> AsymptoticUtilizationCurve:
    """
    Create a typical memory bandwidth utilization curve.

    Based on empirical observations, memory utilization typically:
    - Is low for small transfers (latency-dominated)
    - Approaches ~80% for large transfers (bandwidth-dominated)
    - Transitions around 100 MB

    Args:
        peak_utilization: Maximum achievable utilization (0.0 to 1.0)
        scale_bytes: Transfer size at which utilization reaches 63% of peak

    Returns:
        AsymptoticUtilizationCurve configured with typical parameters
    """
    return AsymptoticUtilizationCurve(peak=peak_utilization, scale=scale_bytes)


def interpolate_utilization(
    known_sizes: List[float],
    known_utilizations: List[float],
    query_size: float,
) -> float:
    """
    Interpolate utilization for a query size from known data points.

    Uses linear interpolation in log-space for sizes, which is more
    appropriate for the typical logarithmic scaling of utilization.

    Args:
        known_sizes: Known problem sizes
        known_utilizations: Known utilization values
        query_size: Size to interpolate for

    Returns:
        Interpolated utilization value (0.0 to 1.0)
    """
    if len(known_sizes) == 0:
        return 0.5  # Default if no data

    if len(known_sizes) == 1:
        return known_utilizations[0]

    # Convert to numpy arrays
    sizes = np.array(known_sizes)
    utils = np.array(known_utilizations)

    # Sort by size
    sort_idx = np.argsort(sizes)
    sizes = sizes[sort_idx]
    utils = utils[sort_idx]

    # Interpolate in log space for sizes
    log_sizes = np.log(sizes)
    log_query = np.log(max(query_size, 1.0))

    # Linear interpolation
    result = np.interp(log_query, log_sizes, utils)

    # Clip to valid range
    return float(np.clip(result, 0.0, 1.0))


# Export list for star imports
__all__ = [
    # Base class and types
    'UtilizationCurve',
    'UtilizationCurveType',
    'CurveType',
    'CurveFitResult',
    'UtilizationFitResult',

    # Curve implementations
    'AsymptoticUtilizationCurve',
    'PiecewiseLinearUtilizationCurve',
    'PolynomialUtilizationCurve',
    'ConstantUtilizationCurve',

    # Fitting functions
    'fit_utilization_curve',
    'auto_fit_utilization_curve',

    # Convenience functions
    'create_typical_compute_curve',
    'create_typical_memory_curve',
    'interpolate_utilization',
]
