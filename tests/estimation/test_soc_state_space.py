"""The brushable CPU + KPU state space (graphs#269 Phase 7.5).

The grid's whole job is to make a pattern that follows one dimension
visible as a shape. These tests hold the parts that decide whether it can:
a column order shared by every mission, an honest tri-state reading, and a
brush whose arithmetic matches what the table says.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from graphs.reporting.state_space import (
    BANDS,
    DARK_STEPS,
    LIGHT_STEPS,
    Configuration,
    MissionRow,
    _band,
    _shared,
    render,
    state_table,
)

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def cli():
    spec = importlib.util.spec_from_file_location(
        "report_state_space_cli", REPO / "cli" / "report_state_space.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def space(cli):
    return cli.build("orin_nano_measured_v1")


def _columns():
    return [Configuration(design=d, node=n, cpu_cores=c, memory=m)
            for c in (4, 12) for d, n in (("kpu_t64_n7", "tsmc_n7"),
                                          ("kpu_t512_n7", "tsmc_n7"))
            for m in ("lpddr5_phy_256b", "hbm3_1stack")]


def _rows():
    return [
        MissionRow("served", "Served everywhere", 15, 50,
                   factors=(1.4, 1.5, 1.6, 1.7, 2.0, 2.1, 2.2, 2.3)),
        MissionRow("cpu_bound", "CPU bound", 15, 50,
                   factors=(0.5, 0.5, 0.5, 0.5, 1.1, 1.2, 1.3, 1.4)),
        MissionRow("ruled_out", "Ruled out", 15, 50,
                   factors=(0.02, 0.2, 0.45, 0.8, 0.02, 0.2, 0.45, 0.8)),
        MissionRow("unpriced", "Nothing prices it", 15, 50,
                   factors=(None,) * 8),
    ]


# ---------------------------------------------------------------------------
# A column index means the same thing in every row
# ---------------------------------------------------------------------------


def test_every_mission_spans_the_same_configurations(space):
    """The grid only works if column i is the same configuration in every
    row; build() must prove that rather than assume it."""
    columns, rows = space
    assert len(columns) == 153
    assert len({c.key for c in columns}) == len(columns)
    for row in rows:
        assert len(row.factors) == len(columns), row.mission
        assert len(row.partial) == len(columns), row.mission


def test_the_columns_are_grouped_by_cpu_cores(space):
    """The reading "the surviving cells are the right-hand third" is only
    available if the core counts are contiguous and ascending."""
    columns, _rows = space
    cores = [c.cpu_cores for c in columns]
    assert cores == sorted(cores)
    runs = [cores[0]] + [b for a, b in zip(cores, cores[1:]) if a != b]
    assert runs == sorted(set(cores))


def test_a_mission_that_nothing_serves_says_so(space):
    """Thirteen of eighteen missions have no configuration in this space;
    that is the page's headline and it must survive a rebuild."""
    _columns, rows = space
    assert sum(1 for r in rows if r.standing() == 0) >= 1
    assert all(r.standing() <= len(r.factors) for r in rows)


def test_rows_lead_with_the_missions_that_have_options(space):
    _columns, rows = space
    standing = [r.standing() for r in rows]
    assert standing == sorted(standing, reverse=True)


# ---------------------------------------------------------------------------
# The reading is one-sided
# ---------------------------------------------------------------------------


def test_a_cell_clears_the_bar_or_is_proven_short():
    assert _band(1.0, 1.0) == "standing"
    assert _band(0.999, 1.0) == BANDS[-1][1]
    assert _band(0.05, 1.0) == BANDS[0][1]
    assert _band(None, 1.0) is None


def test_raising_the_headroom_never_adds_a_survivor():
    """The bar only moves one way: a configuration that fails at 1.0x
    cannot pass at 2.0x."""
    row = _rows()[1]
    counts = [row.standing(h) for h in (0.5, 1.0, 1.5, 2.0, 4.0)]
    assert counts == sorted(counts, reverse=True)


def test_the_bands_partition_everything_under_the_bar():
    fractions = [f for f, _step, _label in BANDS]
    assert fractions == sorted(fractions) and fractions[-1] == 1.0
    seen = {_band(f * 0.999, 1.0) for f in (0.05, 0.2, 0.45, 0.8)}
    assert seen == {step for _f, step, _l in BANDS}


# ---------------------------------------------------------------------------
# What the survivors share
# ---------------------------------------------------------------------------


def test_a_dimension_the_survivors_span_is_not_reported():
    """Naming a dimension every survivor takes every value of would be
    noise: it did not decide anything."""
    columns = _columns()
    twelve = [c for c in columns if c.cpu_cores == 12]
    assert _shared(twelve, columns) == "CPU cores: 12"
    assert _shared(columns, columns) == "every dimension: nothing narrows it"
    assert _shared([], columns) == "-"


def test_the_table_says_what_the_grid_shows():
    table = state_table(_columns(), _rows(), 1.0)
    assert "8/8" in table and "4/8" in table and "0/8" in table
    assert "CPU cores: 12" in table          # the CPU-bound mission's survivors
    assert "2.3x" in table                   # the best factor anywhere


def test_a_mission_nothing_prices_is_not_counted_as_served():
    row = _rows()[3]
    assert row.standing() == 0
    table = state_table(_columns(), [row], 1.0)
    assert "0/8" in table and ">-<" in table


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------


def test_the_grid_draws_one_cell_per_mission_and_configuration():
    page = render(_columns(), _rows(), "", "today")
    assert page.count('class="cell"') == 8 * 4


def test_every_cell_tag_is_closed():
    """An unquoted attribute swallows the closing slash and every later
    cell nests inside the one before it, which renders nothing."""
    page = render(_columns(), _rows(), "", "today")
    for fragment in page.split('<rect class="cell"')[1:]:
        head = fragment[:fragment.index(">") + 1]
        assert head.endswith("/>"), head
        assert 'data-partial="1"' in head or "data-partial" not in head


def test_an_empty_space_does_not_raise():
    assert "no configuration" in render([], [], "", "today")


def test_the_page_is_standalone_and_carries_its_data():
    page = render(_columns(), _rows(), "<h1>x</h1>", "today")
    assert page.startswith("<!DOCTYPE html>") and page.rstrip().endswith("</html>")
    assert "<script src" not in page and "<link" not in page
    data = json.loads(page.split('id="data">')[1].split("</script>")[0])
    assert len(data["columns"]) == 8 and len(data["rows"]) == 4
    assert len(data["x"]) == 8 and data["x"] == sorted(data["x"])


def test_identity_never_rests_on_colour_alone():
    page = render(_columns(), _rows(), "", "today")
    for _fraction, _step, label in BANDS:
        assert label in page                       # the legend names each step
    assert "<table>" in page                       # and a table view exists


def test_dark_mode_is_selected_not_flipped():
    assert set(LIGHT_STEPS) == set(DARK_STEPS)
    assert not set(LIGHT_STEPS.values()) & set(DARK_STEPS.values())
    page = render(_columns(), _rows(), "", "today")
    assert "prefers-color-scheme: dark" in page
    for steps in (LIGHT_STEPS, DARK_STEPS):
        for name, value in steps.items():
            assert f"--{name}:{value};" in page


def _luma(hex_colour: str) -> float:
    r, g, b = (int(hex_colour[i:i + 2], 16) / 255 for i in (1, 3, 5))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def test_the_short_ramp_is_monotone_in_each_mode():
    for steps in (LIGHT_STEPS, DARK_STEPS):
        ramp = [_luma(steps[step]) for _f, step, _l in BANDS]
        assert ramp == sorted(ramp) or ramp == sorted(ramp, reverse=True)
        assert all(abs(a - b) > 0.04 for a, b in zip(ramp, ramp[1:]))


# ---------------------------------------------------------------------------
# The CLI
# ---------------------------------------------------------------------------


def _cli(*args, expect=0):
    result = subprocess.run([sys.executable, "cli/report_state_space.py", *args],
                            capture_output=True, text=True, cwd=REPO, timeout=900)
    assert result.returncode == expect, result.stderr
    return result


def test_cli_writes_a_standalone_page(tmp_path):
    out = tmp_path / "space.html"
    _cli("-o", str(out))
    page = out.read_text()
    assert page.startswith("<!DOCTYPE html>") and page.rstrip().endswith("</html>")
    assert page.count('class="cell"') == 18 * 153


def test_cli_writes_json(tmp_path):
    out = tmp_path / "space.json"
    _cli("-o", str(out))
    data = json.loads(out.read_text())
    assert len(data["configurations"]) == 153 and len(data["missions"]) == 18
    for mission in data["missions"]:
        assert len(mission["real_time_factor"]) == 153


def test_cli_rejects_a_headroom_that_cannot_be_met():
    result = _cli("--headroom", "0", expect=2)
    assert "must be positive" in result.stderr
