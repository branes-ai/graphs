"""
Bottom-up composition validation.

Each hierarchy layer's benchmark (see
``docs/plans/bottom-up-microbenchmark-plan.md``) fits a set of resource-
model coefficients. The composition test takes the fitted coefficients
from layers 1..N and predicts the behavior of layer N+1, then diffs
the prediction against that layer's measured values. A composition
miss catches coefficient drift that a top-down composite calibration
would silently absorb.

Preferred import style (Phase 1+)::

    from validation.composition import register_layer_check, CompositionCheck
"""

from validation.composition.test_layer_composition import (
    CheckStatus,
    CompositionCheck,
    CompositionCheckResult,
    CompositionReport,
    clear_registry,
    register_layer_check,
    registered_checks,
    run_all_checks,
)

__all__ = [
    "CheckStatus",
    "CompositionCheck",
    "CompositionCheckResult",
    "CompositionReport",
    "clear_registry",
    "register_layer_check",
    "registered_checks",
    "run_all_checks",
]
