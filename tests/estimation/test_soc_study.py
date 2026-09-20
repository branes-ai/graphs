"""Studies and sweeps (graphs#269 PR 4.1)."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from graphs.estimation.soc import SoCAnalyzer
from graphs.estimation.soc.study import Override, Study, expand, load_study, run_study

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def analyzer():
    return SoCAnalyzer()


def _study(**kw) -> Study:
    base = dict(id="t", name="t", workload="branes_7tier_v1", designs=["orin_class_reference"])
    base.update(kw)
    return Study.model_validate(base)


# ---------------------------------------------------------------------------
# Axes
# ---------------------------------------------------------------------------


def test_points_are_the_cross_product_of_the_axes(analyzer):
    study = _study(nodes=["samsung_8lpp", "tsmc_n7"], profiles="regimes",
                   efficiency=["annex_v1", "default_v1"], gate_idle=[False, True],
                   overrides=[{"target": "block:gpu_sm.count", "values": [8, 16, 24]}])
    points = expand(study, analyzer)
    assert len(points) == 3 * 2 * 2 * 2 * 2
    assert len({p for p, _ in points}) == len(points)


def test_an_override_changes_only_its_field(analyzer):
    study = _study(overrides=[{"target": "block:gpu_sm.count", "values": [8]},
                              {"target": "layout.whitespace_fraction", "values": [0.25]}])
    (point, design), *_ = expand(study, analyzer)
    base = analyzer.designs["orin_class_reference"]
    gpu = next(b for b in design.blocks if b.instance == "gpu_sm")
    assert gpu.count == 8 and design.layout.whitespace_fraction == 0.25
    assert [b for b in design.blocks if b.instance != "gpu_sm"] == \
        [b for b in base.blocks if b.instance != "gpu_sm"]
    assert base.blocks[0].count == 16  # the catalog design is untouched
    assert point.variant_label == "gpu_sm.count=8,layout.whitespace_fraction=0.25"


def test_a_variant_prices_its_allocation(analyzer):
    """Halving the SMs halves the GPU's dense peak and shrinks the priced die."""
    study = _study(profiles=["air superiority"],
                   overrides=[{"target": "block:gpu_sm.count", "values": [8, 16]}])
    small, full = run_study(study, analyzer)
    assert small.result.soc.peak_tops("int8") < full.result.soc.peak_tops("int8")
    assert small.result.soc.die_area_mm2 < full.result.soc.die_area_mm2


def test_the_pooled_annex_model_does_not_see_the_allocation(analyzer):
    """The finding the smoke study is for: annex_v1 is one pooled machine, so
    its oversubscription is the same at any GPU size (Phase 4 plan)."""
    study = _study(profiles=["far flight"],
                   overrides=[{"target": "block:gpu_sm.count", "values": [4, 16, 64]}])
    values = {round(r.result.schedule.oversubscription, 12) for r in run_study(study, analyzer)}
    assert len(values) == 1


@pytest.mark.parametrize("target, values, match", [
    ("block:gpu_sm.voltage", [1.0], "expected block"),
    ("layout.die_mm", [1.0], "expected block"),
    ("block:gpu_sm.count", [1.5], "positive integers"),
    ("block:gpu_sm.count", [0], "positive integers"),
    ("block:gpu_sm.count", ["four"], "takes numbers"),
    ("layout.io_ring_mm", ["wide"], "takes numbers"),
    ("block:memory.ip", [256.0], "ip takes IP template ids"),
    ("block:memory.ip", [""], "ip takes IP template ids"),
])
def test_bad_overrides_are_rejected(target, values, match):
    with pytest.raises(ValidationError, match=match):
        Override(target=target, values=values)


def test_an_override_of_a_missing_block_is_an_error(analyzer):
    with pytest.raises(KeyError, match="no block"):
        expand(_study(overrides=[{"target": "block:tpu.count", "values": [2]}]), analyzer)


def test_a_target_overridden_twice_is_rejected():
    with pytest.raises(ValidationError, match="twice"):
        _study(overrides=[{"target": "block:gpu_sm.count", "values": [8]},
                          {"target": "block:gpu_sm.count", "values": [16]}])


def test_a_study_for_another_workload_is_refused(analyzer):
    with pytest.raises(ValueError, match="workload"):
        expand(_study(workload="other_v1"), analyzer)


def test_unknown_designs_are_an_error(analyzer):
    with pytest.raises(KeyError, match="unknown designs"):
        expand(_study(designs=["nope"]), analyzer)


# ---------------------------------------------------------------------------
# Rows carry their bounds
# ---------------------------------------------------------------------------


def test_every_row_flags_its_lower_bounds(analyzer):
    rows = run_study(_study(profiles="regimes", efficiency=["annex_v1", "default_v1"]), analyzer)
    for row in (r.to_row() for r in rows):
        assert row["die_area_is_lower_bound"] is True  # Orin has unanchored silicon
        assert row["power_is_lower_bound"] is True
        assert row["useful_tops_per_w"] is None
        if row["efficiency"] == "default_v1":
            assert row["oversubscription_is_lower_bound"] is True
            assert row["unpriced_stages"] > 0


def test_the_shipped_studies_load_and_run(analyzer):
    for file in sorted((REPO / "soc_designs" / "studies").glob("*.yaml")):
        study = load_study(file)
        assert run_study(study, analyzer), file.name


def test_a_study_id_must_match_its_file(tmp_path):
    bad = tmp_path / "x.yaml"
    bad.write_text("id: y\nname: y\nworkload: branes_7tier_v1\ndesigns: [orin_class_reference]\n")
    with pytest.raises(ValueError, match="must match"):
        load_study(bad)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _run(*args):
    return subprocess.run([sys.executable, "cli/sweep_soc.py", *args],
                          capture_output=True, text=True, cwd=REPO, timeout=300)


def test_cli_runs_a_shipped_study():
    result = _run("--study", "orin_node_scaling")
    assert result.returncode == 0, result.stderr
    assert "points: 12" in result.stdout
    assert "(LB)" in result.stdout


def test_cli_ad_hoc_sweep_to_csv_and_json(tmp_path):
    out = tmp_path / "s.csv"
    result = _run("--designs", "orin_class_reference", "--nodes", "samsung_8lpp,tsmc_n7",
                  "--gate-idle", "both", "-o", str(out))
    assert result.returncode == 0, result.stderr
    rows = list(csv.DictReader(out.open(newline="")))
    assert len(rows) == 2 * 2 * 2
    assert {r["gate_idle"] for r in rows} == {"True", "False"}
    js = tmp_path / "s.json"
    result = _run("--study", "orin_node_scaling", "-o", str(js))
    assert result.returncode == 0, result.stderr
    payload = json.loads(js.read_text())
    assert payload["study"]["id"] == "orin_node_scaling" and len(payload["points"]) == 12


def test_cli_markdown(tmp_path):
    out = tmp_path / "s.md"
    result = _run("--study", "orin_node_scaling", "-o", str(out))
    assert result.returncode == 0, result.stderr
    assert out.read_text().startswith("## orin_node_scaling")


def test_cli_bad_input_exits_2():
    result = _run("--study", "nope")
    assert result.returncode == 2, result.stderr
    result = _run("--designs", "nope")
    assert result.returncode == 2, result.stderr
    result = _run("--designs", "orin_class_reference", "--efficiency", "nope")
    assert result.returncode == 2, result.stderr


@pytest.mark.parametrize("target, value", [("layout.whitespace_fraction", 1.5),
                                           ("layout.io_ring_mm", -0.1)])
def test_a_layout_override_is_validated(analyzer, target, value):
    """model_copy skips validation; an override rebuilds the design through it."""
    with pytest.raises(ValidationError):
        expand(_study(overrides=[{"target": target, "values": [value]}]), analyzer)


def test_rows_carry_the_estimation_confidence(analyzer):
    (row,) = [r.to_row() for r in run_study(_study(profiles=["far flight"]), analyzer)]
    assert row["estimation_confidence"]["level"] == row["confidence"] == "unknown"
    assert row["estimation_confidence"]["source"]


# ---------------------------------------------------------------------------
# Swapping a block's IP (graphs#269 Phase 7)
# ---------------------------------------------------------------------------


def test_an_ip_override_swaps_the_template(analyzer):
    """The memory system is an axis, so a state space costs one design
    rather than one design per combination."""
    from embodied_schemas import load_process_nodes

    from graphs.hardware.soc import compose_soc, load_designs, load_ip_library

    design = load_designs()["kpu_t256_n7"]
    override = Override(target="block:memory.ip",
                        values=["lpddr5_phy_256b", "lpddr5x_phy_512b", "hbm3_1stack"])
    library, nodes = load_ip_library(), load_process_nodes()
    bandwidths = [compose_soc(override.apply(design, v), library, nodes, None).dram_peak_gb_per_s
                  for v in override.values]
    assert bandwidths == pytest.approx([204.8, 546.1, 819.2])
    # Nothing else moved: the KPU block is the design's still.
    swapped = override.apply(design, "hbm3_1stack")
    assert [b.ip for b in swapped.blocks if b.instance != "memory"] == \
        [b.ip for b in design.blocks if b.instance != "memory"]


def test_an_ip_override_labels_its_point(analyzer):
    study = _study(designs=["kpu_t256_n7"], profiles=["far flight"],
                   overrides=[{"target": "block:memory.ip",
                               "values": ["lpddr5_phy_256b", "hbm3_1stack"]}])
    labels = {point.variant_label for point, _design in expand(study, analyzer)}
    assert labels == {"memory.ip=lpddr5_phy_256b", "memory.ip=hbm3_1stack"}


def test_wider_memory_is_what_far_flight_needs(analyzer):
    """Far flight asks 305.9 GB/s of a 204.8 GB/s part and no accelerator
    changes that; the axis exists so the study can say what does."""
    study = _study(designs=["kpu_t256_n7"], profiles=["far flight"],
                   overrides=[{"target": "block:memory.ip",
                               "values": ["lpddr5_phy_256b", "lpddr5x_phy_512b", "hbm3_1stack"]}])
    utilization = {r.point.variant_label: r.result.schedule.dram_utilization
                   for r in run_study(study, analyzer)}
    assert utilization["memory.ip=lpddr5_phy_256b"] == pytest.approx(2.30, abs=0.01)
    assert utilization["memory.ip=lpddr5x_phy_512b"] == pytest.approx(0.86, abs=0.01)
    assert utilization["memory.ip=hbm3_1stack"] < utilization["memory.ip=lpddr5x_phy_512b"]
