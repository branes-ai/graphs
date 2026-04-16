"""
Bottom-up composition validation.

Each hierarchy layer's benchmark (see
``docs/plans/bottom-up-microbenchmark-plan.md``) fits a set of resource-
model coefficients. The composition test takes the fitted coefficients
from layers 1..N and predicts the behavior of layer N+1, then diffs
the prediction against that layer's measured values. A composition
miss catches coefficient drift that a top-down composite calibration
would silently absorb.
"""
