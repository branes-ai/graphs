"""
Bottom-up layer composition validation.

Validates that a resource model built from layer-1..N fitted coefficients
reproduces layer-(N+1) measurements within a target tolerance. This is
the core cross-check that separates the bottom-up methodology from the
current top-down-only calibration path.

Scaffolded in Phase 0. Each subsequent phase (1..6) lands its
corresponding check via ``register_layer_check``. Until those phases
ship, this module runs, reports an empty check set, and exits cleanly
with status 0.

Runnable two ways:

    # Standalone (per the validation rules in .claude/rules/validation.md):
    python validation/composition/test_layer_composition.py

    # Via pytest (discovered automatically):
    python -m pytest validation/composition/test_layer_composition.py

Both paths exercise the same registry and produce the same pass/fail
decisions.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional

from graphs.benchmarks.schema import LayerTag


# --------------------------------------------------------------------------- #
# Check registry
# --------------------------------------------------------------------------- #

class CheckStatus(Enum):
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"  # e.g. no measurement data available for this SKU


@dataclass
class CompositionCheckResult:
    """Outcome of a single composition check."""
    name: str
    hardware: str
    predicts_layer: LayerTag
    from_layers: List[LayerTag]
    status: CheckStatus
    max_relative_error: float = 0.0
    tolerance: float = 0.15
    details: str = ""

    def summary_line(self) -> str:
        tag = {
            CheckStatus.PASSED: "PASS",
            CheckStatus.FAILED: "FAIL",
            CheckStatus.SKIPPED: "SKIP",
        }[self.status]
        layers = "+".join(l.value for l in self.from_layers) or "-"
        return (
            f"[{tag}] {self.hardware:20s} {layers:20s} -> {self.predicts_layer.value:12s} "
            f"err={self.max_relative_error*100:5.1f}% tol={self.tolerance*100:5.1f}% "
            f"{self.name}"
        )


@dataclass
class CompositionCheck:
    """
    A single bottom-up composition check.

    ``runner`` is a zero-arg callable that returns a CompositionCheckResult.
    The runner owns its data loading (from calibration_data/ or
    validation/empirical/results/) so a missing file becomes SKIPPED
    rather than a crash.
    """
    name: str
    hardware: str
    from_layers: List[LayerTag]
    predicts_layer: LayerTag
    runner: Callable[[], CompositionCheckResult]
    tolerance: float = 0.15


# Module-level registry. Phase 1 lands the first ALU -> SCRATCHPAD check.
_CHECKS: List[CompositionCheck] = []


def register_layer_check(check: CompositionCheck) -> None:
    """Add a composition check to the global registry."""
    _CHECKS.append(check)


def clear_registry() -> None:
    """Testing helper — remove all registered checks."""
    _CHECKS.clear()


def registered_checks() -> List[CompositionCheck]:
    """Read-only view of the current registry."""
    return list(_CHECKS)


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #

@dataclass
class CompositionReport:
    results: List[CompositionCheckResult] = field(default_factory=list)

    @property
    def n_passed(self) -> int:
        return sum(1 for r in self.results if r.status is CheckStatus.PASSED)

    @property
    def n_failed(self) -> int:
        return sum(1 for r in self.results if r.status is CheckStatus.FAILED)

    @property
    def n_skipped(self) -> int:
        return sum(1 for r in self.results if r.status is CheckStatus.SKIPPED)

    @property
    def all_passed(self) -> bool:
        """True when zero checks FAILED.

        SKIPPED checks are non-failing: a run where every check skips
        (e.g., measurement data missing for a SKU) reports green. An
        empty registry (Phase 0 state) also reports green. Phase 1+
        code that needs to distinguish "all healthy" from "all skipped"
        should inspect n_skipped explicitly.
        """
        return self.n_failed == 0

    def format(self) -> str:
        lines = ["=" * 78, "Bottom-Up Layer Composition Report", "=" * 78]
        if not self.results:
            lines.append(
                "(no composition checks registered; Phase 0 scaffold is live)"
            )
        else:
            for r in self.results:
                lines.append(r.summary_line())
        lines.append("-" * 78)
        lines.append(
            f"Totals: {self.n_passed} passed, "
            f"{self.n_failed} failed, {self.n_skipped} skipped "
            f"({len(self.results)} total)"
        )
        lines.append("=" * 78)
        return "\n".join(lines)


def run_all_checks() -> CompositionReport:
    """Execute every registered composition check and collect results."""
    report = CompositionReport()
    for check in _CHECKS:
        try:
            result = check.runner()
        except Exception as exc:  # noqa: BLE001 - runners should be defensive
            result = CompositionCheckResult(
                name=check.name,
                hardware=check.hardware,
                predicts_layer=check.predicts_layer,
                from_layers=check.from_layers,
                status=CheckStatus.FAILED,
                details=f"runner raised {type(exc).__name__}: {exc}",
                tolerance=check.tolerance,
            )
        report.results.append(result)
    return report


# --------------------------------------------------------------------------- #
# pytest entry point
# --------------------------------------------------------------------------- #

def test_composition_scaffold_runs() -> None:
    """The scaffold must import, run, and report cleanly before any
    Phase 1+ checks are registered."""
    saved = registered_checks()
    clear_registry()
    try:
        report = run_all_checks()
        assert report.all_passed, report.format()
        assert report.results == []
    finally:
        clear_registry()
        for c in saved:
            register_layer_check(c)


def test_registry_api_is_functional() -> None:
    """Phase 1 will register checks here; ensure the API it relies on
    behaves (add, list, clear)."""
    from graphs.benchmarks.schema import LayerTag as LT

    saved = registered_checks()
    clear_registry()
    try:
        assert registered_checks() == []

        def _runner() -> CompositionCheckResult:
            return CompositionCheckResult(
                name="phase0_self_test",
                hardware="dummy",
                from_layers=[LT.ALU],
                predicts_layer=LT.REGISTER_SIMD,
                status=CheckStatus.SKIPPED,
                details="phase 0 self-test",
            )

        register_layer_check(
            CompositionCheck(
                name="phase0_self_test",
                hardware="dummy",
                from_layers=[LT.ALU],
                predicts_layer=LT.REGISTER_SIMD,
                runner=_runner,
            )
        )
        assert len(registered_checks()) == 1

        report = run_all_checks()
        assert len(report.results) == 1
        assert report.results[0].status is CheckStatus.SKIPPED
        assert report.all_passed
    finally:
        clear_registry()
        for c in saved:
            register_layer_check(c)


# --------------------------------------------------------------------------- #
# Standalone entry point
# --------------------------------------------------------------------------- #

def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point: run the composition report and exit accordingly.

    Exit codes:
        0  - all registered checks passed (or registry is empty)
        1  - at least one registered check failed
    """
    parser = argparse.ArgumentParser(
        description="Bottom-up layer composition validation"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print per-check detail (reserved for Phase 1+)"
    )
    parser.parse_args(argv)

    report = run_all_checks()
    print(report.format())
    return 0 if report.all_passed else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
