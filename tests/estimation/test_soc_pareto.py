"""Bound-aware Pareto and the union of regimes (graphs#269 PR 4.2)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from graphs.estimation.soc.pareto import classify_front, dominates, union_of_regimes

REPO = Path(__file__).resolve().parents[2]


def _row(design="d", profile="p", area=10.0, area_lb=False, power=1.0, power_lb=False,
         feasible=True, variant="base"):
    return {"design": design, "variant": variant, "node": "n", "profile": profile,
            "efficiency": "e", "gate_idle": False, "feasible": feasible,
            "die_area_mm2": area, "die_area_is_lower_bound": area_lb,
            "power_w": power, "power_is_lower_bound": power_lb,
            "oversubscription": 1.0, "oversubscription_is_lower_bound": False}


class _Row:
    """A SweepRow stand-in: the reports only read ``to_row()``."""

    def __init__(self, **kw):
        self._row = _row(**kw)

    def to_row(self):
        return dict(self._row)


# ---------------------------------------------------------------------------
# Dominance is proven, never assumed from a bound
# ---------------------------------------------------------------------------


def test_exact_dominance():
    assert dominates(_row(area=5, power=1), _row(area=6, power=1), ["area", "power"])
    assert not dominates(_row(area=5, power=2), _row(area=6, power=1), ["area", "power"])
    assert not dominates(_row(), _row(), ["area", "power"])  # equal: not strictly better


def test_a_lower_bound_dominates_nothing():
    """A 5 mm^2 lower bound may really be 50: it proves nothing small."""
    assert not dominates(_row(area=5, area_lb=True), _row(area=6), ["area"])


def test_an_exact_point_can_dominate_a_lower_bound():
    """B's true area is at least its bound of 8, so exact 5 beats it."""
    assert dominates(_row(area=5), _row(area=8, area_lb=True), ["area"])
    # ...but not when the bound is below A: B's true value is unknown there.
    assert not dominates(_row(area=5), _row(area=4, area_lb=True), ["area"])


def test_front_classification():
    rows = [_Row(design="exact_best", area=5, power=1),
            _Row(design="exact_worse", area=6, power=2),
            _Row(design="bounded_beaten", area=7, area_lb=True, power=3),
            _Row(design="bounded_low", area=3, area_lb=True, power=0.5)]
    status = {r.to_row()["design"]: s for r, s in classify_front(rows, ["area", "power"])}
    assert status == {"exact_best": "front", "exact_worse": "dominated",
                      "bounded_beaten": "dominated", "bounded_low": "undecided"}


def test_fronts_are_per_profile():
    rows = [_Row(design="a", profile="p1", area=5), _Row(design="b", profile="p2", area=9)]
    assert [s for _, s in classify_front(rows, ["area"])] == ["front", "front"]


def test_unknown_metric_is_an_error():
    with pytest.raises(KeyError, match="unknown Pareto metrics"):
        classify_front([_Row()], ["latency"])


# ---------------------------------------------------------------------------
# Union of regimes
# ---------------------------------------------------------------------------


def test_the_minimum_is_the_smallest_proven_feasible_variant():
    rows = [_Row(design="small", profile=p, area=5) for p in ("p1", "p2")]
    rows += [_Row(design="large", profile=p, area=9) for p in ("p1", "p2")]
    rows += [_Row(design="fails", profile="p1", area=2),
             _Row(design="fails", profile="p2", area=2, feasible=False)]
    report = union_of_regimes(rows)
    assert report.minimum.variant[0] == "small"
    by = {v.variant[0]: v for v in report.verdicts}
    assert by["fails"].feasible is False and by["fails"].failing_profiles == ("p2",)
    assert not report.could_be_smaller


def test_an_open_profile_leaves_a_variant_undecided_and_possibly_smaller():
    rows = [_Row(design="proven", profile=p, area=6) for p in ("p1", "p2")]
    rows += [_Row(design="open", profile="p1", area=3, area_lb=True),
             _Row(design="open", profile="p2", area=3, area_lb=True, feasible=None)]
    report = union_of_regimes(rows)
    assert report.minimum.variant[0] == "proven"
    assert [v.variant[0] for v in report.could_be_smaller] == ["open"]


def test_a_lower_bound_area_is_never_the_minimum():
    rows = [_Row(design="bounded", profile="p", area=3, area_lb=True)]
    assert union_of_regimes(rows).minimum is None


def test_a_variant_missing_a_profile_is_not_proven():
    rows = [_Row(design="a", profile="p1"), _Row(design="a", profile="p2"),
            _Row(design="b", profile="p1")]
    by = {v.variant[0]: v for v in union_of_regimes(rows).verdicts}
    assert by["a"].feasible is True and by["b"].feasible is None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _run(*args):
    return subprocess.run([sys.executable, "cli/sweep_soc.py", *args],
                          capture_output=True, text=True, cwd=REPO, timeout=300)


def test_cli_pareto_and_union_on_orin(tmp_path):
    """Every Orin point has a bounded area and power, so nothing is on a
    front or proven feasible: the report says undecided, not a winner."""
    out = tmp_path / "s.json"
    result = _run("--study", "orin_node_scaling", "--pareto", "area,power", "--union", "-o", str(out))
    assert result.returncode == 0, result.stderr
    payload = json.loads(out.read_text())
    assert {p["pareto"] for p in payload["points"]} == {"undecided"}
    assert payload["union_of_regimes"]["minimum"] is None
    text = _run("--study", "orin_node_scaling", "--union").stdout
    assert "no variant is proven feasible for all" in text


def test_cli_plot(tmp_path):
    pytest.importorskip("matplotlib")
    png = tmp_path / "front.png"
    result = _run("--study", "orin_node_scaling", "--pareto", "area,power", "--plot", str(png))
    assert result.returncode == 0, result.stderr
    assert png.stat().st_size > 1000


def test_cli_plot_needs_two_metrics():
    result = _run("--study", "orin_node_scaling", "--plot", "x.png")
    assert result.returncode == 2, result.stderr
