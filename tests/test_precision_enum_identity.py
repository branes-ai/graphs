"""Regression tests for issue #59.

The graphs repo previously had five independent ``class Precision(Enum)``
declarations, two of which sat on the MCP analyze hot path:

    mcp/server.py             -> workload_characterization.Precision
    hardware/mappers/kpu_t64  -> hardware.resource_model.Precision

Because Python enums compare by class identity, ``precision in
self.precision_profiles`` returned False even when both enums had a
member with value ``"int8"``. The analyzer then raised a
self-contradicting "does not support int8 -- supported precisions are:
... int8 ..." error.

The fix: pick ``graphs.hardware.resource_model.Precision`` as the canonical
enum and have every other module re-export it. These tests pin that
contract so the dual-enum bug cannot regress.
"""
from graphs.hardware.resource_model import Precision as CanonicalPrecision


def test_workload_characterization_precision_is_canonical():
    from graphs.estimation.workload_characterization import Precision as P
    assert P is CanonicalPrecision


def test_baseline_alu_energy_precision_is_canonical():
    from graphs.reporting.baseline_alu_energy import Precision as P
    assert P is CanonicalPrecision


def test_benchmarks_schema_precision_is_canonical():
    from graphs.benchmarks.schema import Precision as P
    assert P is CanonicalPrecision


def test_calibration_precision_detector_precision_is_canonical():
    from graphs.calibration.precision_detector import Precision as P
    assert P is CanonicalPrecision


def test_mcp_precision_enum_returns_canonical():
    """The MCP boundary helper must yield the canonical class so that
    `precision in mapper.precision_profiles` works for KPU-T64 and every
    other catalog mapper."""
    from graphs.mcp.server import _precision_enum
    assert type(_precision_enum("int8")) is CanonicalPrecision
    assert _precision_enum("int8") is CanonicalPrecision.INT8


def test_mcp_int8_keys_kpu_t64_precision_profile():
    """Issue #59 repro at the data-structure level: bridging 'int8' through
    the MCP boundary must produce a key the KPU-T64 catalog accepts."""
    from graphs.mcp.server import _precision_enum
    from graphs.hardware.models.accelerators.kpu_t64 import (
        kpu_t64_resource_model,
    )

    model = kpu_t64_resource_model()
    precision = _precision_enum("int8")
    assert precision in model.precision_profiles
    # And the analyzer-facing accessor returns a profile, not a ValueError.
    profile = model.get_precision_profile(precision)
    assert profile.precision is CanonicalPrecision.INT8


def test_dual_enum_diagnostic_names_modules():
    """When a caller passes a same-valued member from a foreign Enum
    class, the catalog should explain the class mismatch rather than
    emit the self-contradicting "does not support X -- supported: ... X"
    message that motivated issue #59."""
    from enum import Enum
    from graphs.hardware.models.accelerators.kpu_t64 import (
        kpu_t64_resource_model,
    )

    class Precision(Enum):  # foreign, same values
        INT8 = "int8"

    model = kpu_t64_resource_model()
    try:
        model.get_precision_profile(Precision.INT8)
    except ValueError as e:
        msg = str(e)
        assert "Enum classes differ" in msg
        assert "graphs.hardware.resource_model" in msg
    else:  # pragma: no cover - defensive
        raise AssertionError("expected ValueError for dual-enum mismatch")
