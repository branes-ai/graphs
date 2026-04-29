"""
Cache-sweep microbenchmark package (Path B PR-1).

Sweeps a streaming kernel over a logarithmic range of working-set
sizes and reports per-size effective bandwidth + measured energy.
The transition points in the bandwidth-vs-working-set curve reveal
the L1 / L2 / L3 capacity boundaries; the per-byte energy at each
plateau calibrates the per-level energy coefficients on
``HardwareResourceModel``.

Pipeline:

    cli/run_cache_sweep.py
        -> cache_sweep.working_set_sweep.run_sweep()    [ runs on hardware ]
        -> JSON file                                      [ persisted ]
        -> cache_sweep.analysis.detect_levels()           [ pure analysis ]
        -> cache_sweep_fitter.apply_to_model()            [ writes CALIBRATED ]

The runtime kernel uses NumPy memory operations rather than calling
out to a C binary -- this keeps the package portable across CPU and
Jetson SoCs while still measuring the cache hierarchy cleanly. Real
silicon shows distinct bandwidth plateaus across L1 / L2 / L3 / DRAM
even with NumPy-level overhead.
"""
from .working_set_sweep import (
    SweepConfig,
    WorkingSetPoint,
    run_sweep,
)
from .analysis import (
    CacheLevel,
    DetectedLevels,
    PerLevelEnergy,
    detect_levels,
    estimate_per_level_energy,
)

__all__ = [
    "SweepConfig",
    "WorkingSetPoint",
    "run_sweep",
    "CacheLevel",
    "DetectedLevels",
    "PerLevelEnergy",
    "detect_levels",
    "estimate_per_level_energy",
]
