"""
Tests for the graphs.calibration.fitters subpackage.

The subpackage is the canonical home for calibration fitters going
forward. During the Phase 0 transition, it re-exports legacy fitters
from their existing module locations. Those exports must be the SAME
objects (identity, not equality) so that ``isinstance`` checks and
class registries continue to work regardless of import path.
"""

from __future__ import annotations

import pytest


class TestFittersNamespaceReExports:
    """Legacy fitter symbols re-exported via graphs.calibration.fitters."""

    def test_fit_roofline_is_same_function(self):
        from graphs.calibration.roofline_fitter import fit_roofline as legacy
        from graphs.calibration.fitters import fit_roofline as new
        assert new is legacy

    def test_fit_energy_model_is_same_function(self):
        from graphs.calibration.energy_fitter import fit_energy_model as legacy
        from graphs.calibration.fitters import fit_energy_model as new
        assert new is legacy

    def test_fit_efficiency_curve_is_same_function(self):
        from graphs.calibration.efficiency_curves import fit_efficiency_curve as legacy
        from graphs.calibration.fitters import fit_efficiency_curve as new
        assert new is legacy

    def test_fit_utilization_is_same_function(self):
        from graphs.calibration.utilization_fitter import fit_utilization as legacy
        from graphs.calibration.fitters import fit_utilization as new
        assert new is legacy

    def test_roofline_fitter_class_is_same(self):
        from graphs.calibration.roofline_fitter import RooflineFitter as legacy
        from graphs.calibration.fitters import RooflineFitter as new
        assert new is legacy

    def test_energy_coefficients_class_is_same(self):
        from graphs.calibration.energy_fitter import EnergyCoefficients as legacy
        from graphs.calibration.fitters import EnergyCoefficients as new
        assert new is legacy


class TestFittersNamespaceAllExports:
    """Every entry in __all__ must resolve to an existing attribute."""

    def test_all_symbols_resolve(self):
        from graphs.calibration import fitters as fitters_pkg
        for name in fitters_pkg.__all__:
            assert hasattr(fitters_pkg, name), f"Missing re-export: {name}"

    def test_top_level_calibration_still_works(self):
        # The pre-Phase-0 top-level calibration surface must be unchanged.
        from graphs.calibration import (
            RooflineFitter,
            EnergyCoefficients,
            EfficiencyCurve,
            UtilizationFitter,
            fit_roofline,
            fit_energy_model,
        )
        # Smoke check: all of these are usable objects, not None sentinels.
        assert RooflineFitter is not None
        assert EnergyCoefficients is not None
        assert EfficiencyCurve is not None
        assert UtilizationFitter is not None
        assert callable(fit_roofline)
        assert callable(fit_energy_model)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
