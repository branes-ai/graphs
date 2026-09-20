"""The per-stage demand-and-supply view (graphs#269 Phase 7.4).

The frontier places whole configurations on a plane. This view is the level
below it: one row per stage, what it demands, and what *every* engine would
give it -- including the engine the schedule did not pick, which is what
makes the division of labour arguable rather than assumed.
"""

from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
from pathlib import Path

import pytest

from graphs.core.pipeline_workload import REACTIVE_CHAIN, load_autonomy_workload
from graphs.reporting.pipeline_view import (
    CLASS_FORMATS,
    DARK_STEPS,
    LIGHT_STEPS,
    StageRow,
    pipeline_svg,
    render,
    stage_table,
)

REPO = Path(__file__).resolve().parents[2]
WORKLOAD = load_autonomy_workload()
COBOT = "humanoid_cobot_human_adjacent_contact_rich"
TIERS = {"T1": "sensor front end", "T3": "mapping"}


@pytest.fixture(scope="module")
def cli():
    spec = importlib.util.spec_from_file_location(
        "report_pipeline_cli", REPO / "cli" / "report_pipeline.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def built(cli):
    missions, engines, soc = cli.build("kpu_t128_n7", 3, "lpddr5_phy_256b", [COBOT],
                                       "orin_nano_measured_v1")
    return missions[COBOT], engines, soc


def _rows():
    return [
        StageRow(key="tsdf", name="TSDF", tier="T3", kernel_class="raycast",
                 unit="per measured point", rate_hz=4e5, ops_per_s=1.5e9, bytes_per_s=4.8e8,
                 class_split={"A": 0.0, "B": 0.0, "C": 1.0}, on_reactive_chain=False,
                 assigned="cpu",
                 engines={"cpu": (1.03, "measured", 0.021, None),
                          "kpu": (None, "unpriced", None, "no fp32 figure, no schedule")}),
        StageRow(key="det", name="Detector", tier="T1", kernel_class="dense_conv_gemm",
                 unit="per inference", rate_hz=60, ops_per_s=8.4e12, bytes_per_s=1.8e10,
                 class_split={"A": 0.85, "B": 0.15, "C": 0.0}, on_reactive_chain=True,
                 assigned="kpu",
                 engines={"kpu": (0.052, "ceiling", 0.31, None),
                          "cpu": (None, "unpriced", None, "no int8 figure (fp16 measured)")}),
    ]


# ---------------------------------------------------------------------------
# Every stage is costed on every engine
# ---------------------------------------------------------------------------


def test_every_stage_is_costed_on_every_engine(built):
    """A row that only priced the engine the schedule picked could not show
    that the other engine was the worse choice -- or the unusable one."""
    rows, engines, _ = built
    assert len(engines) >= 2
    for row in rows:
        assert set(row.engines) == set(engines), row.key
        for share, provenance, _efficiency, why in row.engines.values():
            # Every cell says either a number or why there is none.
            assert (share is None) == (why is not None)
            assert provenance in ("measured", "ceiling", "unpriced")


def test_a_stage_is_priced_on_the_engine_it_was_not_given(built):
    """The KPU takes the dense classes; the interesting fact is what the CPU
    would have cost for them, and that has to be on the page."""
    rows, _engines, _ = built
    det = next(r for r in rows if r.key == "det")
    assert det.assigned == "kpu"
    assert det.engines["kpu"][0] is not None
    # The CPU cell is present whether or not it carries a figure.
    assert "cpu" in det.engines


def test_the_rows_are_the_missions_stages(built):
    rows, _engines, _ = built
    demanded = {d.stage.key for d in WORKLOAD.demands(
        next(p for p in WORKLOAD.profiles if p.id == COBOT))}
    assert {r.key for r in rows} == demanded
    assert all(r.tier.startswith("T") for r in rows)
    assert {r.key for r in rows if r.on_reactive_chain} == demanded & set(REACTIVE_CHAIN)


# ---------------------------------------------------------------------------
# A dead cell names the work that would fill it
# ---------------------------------------------------------------------------


def test_a_missing_measurement_names_its_format(cli, built):
    """"No figure" is not actionable. "No int8 figure (fp32 measured)" says
    exactly which benchmark run is missing."""
    rows, _engines, _ = built
    reasons = [c[3] for r in rows for c in r.engines.values() if c[3]]
    assert reasons, "this mission is expected to have unpriced cells"
    assert any(re.fullmatch(r"no \S+ figure \(\S+ measured\)", why) for why in reasons)


def test_a_missing_schedule_is_not_a_missing_measurement(built):
    """On the KPU a class the domain-flow model cannot schedule is a
    different gap from one nobody has benchmarked, and calls for different
    work, so the cell must not conflate them."""
    rows, _engines, _ = built
    kpu = [r.engines["kpu"][3] for r in rows if r.engines["kpu"][0] is None]
    assert kpu and all(why.endswith("no schedule") for why in kpu)


def test_a_priced_cell_carries_its_provenance_and_efficiency(built):
    rows, _engines, _ = built
    priced = [(r.key, e, c) for r in rows for e, c in r.engines.items() if c[0] is not None]
    assert priced
    for key, _engine, (share, provenance, efficiency, why) in priced:
        assert why is None and share > 0 and 0 < efficiency <= 1, key
        assert provenance in ("measured", "ceiling")


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------


def test_a_bar_past_one_engine_is_flagged():
    svg = pipeline_svg(_rows(), ["cpu", "kpu"], TIERS)
    assert 'class="bar over"' in svg and "1.03x" in svg
    # ...and the stage that fits is not flagged.
    assert svg.count('class="bar over"') == 1


def test_the_bar_is_shaded_by_provenance():
    svg = pipeline_svg(_rows(), ["cpu", "kpu"], TIERS)
    assert "--c:var(--prov-measured)" in svg and "--c:var(--prov-ceiling)" in svg


def test_the_rows_are_banded_by_tier():
    svg = pipeline_svg(_rows(), ["cpu", "kpu"], TIERS)
    for tier, label in TIERS.items():
        assert f">{tier} " in svg and label in svg


def test_the_schedules_choice_and_the_chain_are_marked():
    svg = pipeline_svg(_rows(), ["cpu", "kpu"], TIERS)
    assert svg.count("&#9656;") == 2      # one caret per row, on its engine
    assert svg.count("&#9679;") == 1      # one stage on the sense-to-act chain


def test_an_empty_pipeline_does_not_raise():
    assert "no stage" in pipeline_svg([], ["cpu"], TIERS)


# ---------------------------------------------------------------------------
# Reading it without the chart
# ---------------------------------------------------------------------------


def test_the_table_repeats_every_figure():
    table = stage_table(_rows(), ["cpu", "kpu"], TIERS)
    assert "<table>" in table and table.count("<tr>") == 3   # header + two rows
    assert "103% of one (measured)" in table
    assert "no fp32 figure, no schedule" in table
    assert "400 k measured point/s" in table                 # its own unit, not Hz
    assert "85% INT8 or wider" in table


def test_identity_never_rests_on_colour_alone():
    page = render({"m": _rows()}, {"m": "M"}, {"m": "s"}, ["cpu", "kpu"], TIERS, "", "today")
    for label in CLASS_FORMATS.values():
        assert label in page                                  # legend names each step
    assert "<table>" in page                                  # and a table view exists


# ---------------------------------------------------------------------------
# Both modes get their own steps
# ---------------------------------------------------------------------------


def test_dark_mode_is_selected_not_flipped():
    """The two ordinal ramps are drawn through variables, and dark mode
    redefines them -- the class ramp runs the other way there, because on a
    dark panel distance from the surface is lightness."""
    assert set(LIGHT_STEPS) == set(DARK_STEPS)
    assert not set(LIGHT_STEPS.values()) & set(DARK_STEPS.values())
    page = render({"m": _rows()}, {"m": "M"}, {"m": "s"}, ["cpu", "kpu"], TIERS, "", "today")
    assert "prefers-color-scheme: dark" in page
    for steps in (LIGHT_STEPS, DARK_STEPS):
        for name, value in steps.items():
            assert f"--{name}:{value};" in page


def _luma(hex_colour: str) -> float:
    r, g, b = (int(hex_colour[i:i + 2], 16) / 255 for i in (1, 3, 5))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def test_each_ramp_is_monotone_in_its_own_mode():
    for steps in (LIGHT_STEPS, DARK_STEPS):
        classes = [_luma(steps[f"cls-{k}"]) for k in ("a", "b", "c")]
        assert classes == sorted(classes) or classes == sorted(classes, reverse=True)
        assert all(abs(x - y) > 0.05 for x, y in zip(classes, classes[1:]))
        assert _luma(steps["prov-ceiling"]) > _luma(steps["prov-measured"])


# ---------------------------------------------------------------------------
# The CLI
# ---------------------------------------------------------------------------


def _cli(*args, expect=0):
    result = subprocess.run([sys.executable, "cli/report_pipeline.py", *args],
                            capture_output=True, text=True, cwd=REPO, timeout=900)
    assert result.returncode == expect, result.stderr
    return result


def test_cli_writes_a_standalone_page(tmp_path):
    out = tmp_path / "pipeline.html"
    _cli("--mission", COBOT, "-o", str(out))
    page = out.read_text()
    assert page.startswith("<!DOCTYPE html>") and page.rstrip().endswith("</html>")
    assert "<script src" not in page and "<link" not in page     # nothing to fetch
    assert page.count('<svg viewBox') == 1
    assert "gets from the cpu" in page and "gets from the kpu" in page


def test_cli_rejects_a_mission_it_does_not_have():
    result = _cli("--mission", "no such mission", expect=2)
    assert "no mission matches" in result.stderr
