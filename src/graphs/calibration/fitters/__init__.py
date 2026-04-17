"""
Calibration fitters subpackage.

This package is the canonical home for all calibration fitters — modules
that consume benchmark measurements (``BenchmarkResult`` CSV / JSON)
and emit fitted coefficients for the hardware resource model.

Two categories of fitters live here:

1. **Legacy fitters** — existing modules
   (``roofline_fitter``, ``energy_fitter``, ``efficiency_curves``,
   ``utilization_fitter``, ``utilization_curves``) currently live at
   ``graphs.calibration.<name>`` and are re-exported here for forward
   compatibility. Their physical relocation is deferred to a follow-up
   cleanup PR so that Phase 0 changes remain additive and non-breaking.

2. **Layered fitters** (added in Phase 1 onward) — one fitter per
   hierarchy layer (ALU, register/SIMD, scratchpad, on-chip L3, DRAM,
   cluster). These land here directly as
   ``layer1_alu_fitter.py``, ``layer2_register_fitter.py``, etc.

Preferred import style (new code):

    from graphs.calibration.fitters import fit_roofline, fit_energy_model

Existing imports from ``graphs.calibration.<name>`` continue to work
unchanged.
"""

from __future__ import annotations

# Re-export legacy fitters from their current locations.
# When the physical relocation happens, only these imports change.
from graphs.calibration.roofline_fitter import (
    RooflineFitter,
    RooflineParameters,
    FitMetrics,
    FitQuality,
    fit_roofline,
)
from graphs.calibration.energy_fitter import (
    EnergyFitter,
    EnergyCoefficients,
    EnergyFitMetrics,
    EnergyFitQuality,
    fit_energy_model,
)
from graphs.calibration.efficiency_curves import (
    EfficiencyCurve,
    AsymptoticCurve,
    PiecewiseLinearCurve,
    PolynomialCurve,
    ConstantCurve,
    EfficiencyProfile,
    CurveType,
    CurveFitResult,
    fit_efficiency_curve,
    auto_fit_efficiency_curve,
)
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
    fit_utilization_curve,
    auto_fit_utilization_curve,
    create_typical_compute_curve,
    create_typical_memory_curve,
    interpolate_utilization,
)

__all__ = [
    # Roofline
    'RooflineFitter',
    'RooflineParameters',
    'FitMetrics',
    'FitQuality',
    'fit_roofline',
    # Energy
    'EnergyFitter',
    'EnergyCoefficients',
    'EnergyFitMetrics',
    'EnergyFitQuality',
    'fit_energy_model',
    # Efficiency curves
    'EfficiencyCurve',
    'AsymptoticCurve',
    'PiecewiseLinearCurve',
    'PolynomialCurve',
    'ConstantCurve',
    'EfficiencyProfile',
    'CurveType',
    'CurveFitResult',
    'fit_efficiency_curve',
    'auto_fit_efficiency_curve',
    # Utilization fitter
    'UtilizationFitter',
    'UtilizationFitQuality',
    'UtilizationFitMetrics',
    'UtilizationCurveResult',
    'UtilizationProfile',
    'fit_utilization',
    # Utilization curves
    'UtilizationCurve',
    'AsymptoticUtilizationCurve',
    'PiecewiseLinearUtilizationCurve',
    'PolynomialUtilizationCurve',
    'ConstantUtilizationCurve',
    'fit_utilization_curve',
    'auto_fit_utilization_curve',
    'create_typical_compute_curve',
    'create_typical_memory_curve',
    'interpolate_utilization',
]
