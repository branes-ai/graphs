"""SoC checks and their CLIs (graphs#269 PR 2.4).

The Orin design exercises the "cannot tell" paths -- incomplete area, clocks
with no speed relation, no shoreline data. Synthetic designs exercise the
paths it cannot reach: a complete composition, a pad-limited die, a reference
comparison that runs and one that fails.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from embodied_schemas import load_process_nodes

from graphs.hardware.sku_validators.framework import Severity
from graphs.hardware.soc import (
    DesignBlock,
    IPBlockTemplate,
    Layout,
    Reference,
    SoCDesign,
    compose_soc,
    load_designs,
    load_ip_library,
)
from graphs.hardware.soc.validators import (
    REFERENCE_TOLERANCE,
    area_completeness,
    clock_retargeting,
    phy_shoreline,
    reference_comparison,
    validate_soc,
)

NODES = load_process_nodes()
REPO = Path(__file__).resolve().parents[2]
ORIN = load_designs()["orin_class_reference"]
LIBRARY = load_ip_library()


def _soc(lines, *, shoreline_mm=None, count=1, reference=None, node="samsung_8lpp"):
    template = {"id": "blk", "name": "blk", "silicon": lines}
    if shoreline_mm is not None:
        template["shoreline"] = {"edge_mm": shoreline_mm, "source": "synthetic"}
    design = SoCDesign(
        id="d", name="d", process_node="samsung_8lpp",
        blocks=[DesignBlock(instance="b", ip="blk", count=count)],
        layout=Layout(whitespace_fraction=0.0, io_ring_mm=0.0),
        reference=reference,
    )
    return compose_soc(design, {"blk": IPBlockTemplate.model_validate(template)}, NODES, node)


AREA_100 = [{"name": "logic", "circuit_class": "hp_logic", "reference_area_mm2": 100.0,
             "reference_node": "samsung_8lpp", "source": "synthetic"}]
GAP = [{"name": "logic", "circuit_class": "hp_logic", "unanchored": True,
        "source": "synthetic gap, no public figure", "confidence": "unknown"}]


# ---------------------------------------------------------------------------
# Each check
# ---------------------------------------------------------------------------


def test_a_complete_design_has_no_completeness_finding():
    assert area_completeness(_soc(AREA_100)) == []


def test_an_incomplete_design_is_a_warning_that_names_its_gaps():
    (finding,) = area_completeness(_soc(AREA_100 + [
        {**GAP[0], "name": "tags"}]))
    assert finding.severity == Severity.WARNING
    assert "lower bounds" in finding.message
    assert "b: tags" in finding.message


def test_shoreline_is_not_checked_without_data():
    (finding,) = phy_shoreline(_soc(AREA_100))
    assert finding.severity == Severity.INFO
    assert "not checked" in finding.message


@pytest.mark.parametrize("edge_mm, severity", [
    (10.0, None),                 # 10 mm of 40 mm perimeter: fine
    (35.0, Severity.WARNING),     # 88% of the edge
    (50.0, Severity.ERROR),       # more edge than the die has
])
def test_shoreline_against_the_perimeter(edge_mm, severity):
    """A 100 mm^2 square die has 40 mm of edge."""
    soc = _soc(AREA_100, shoreline_mm=edge_mm)
    assert soc.die_perimeter_mm == pytest.approx(40.0)
    findings = phy_shoreline(soc)
    if severity is None:
        assert findings == []
    else:
        (finding,) = findings
        assert finding.severity == severity
        assert "pad-limited" in finding.message


def test_an_incomplete_die_qualifies_its_shoreline_verdict():
    """A lower-bound die's perimeter is a lower bound too, so a pad-limited
    verdict on it says the real die may be larger."""
    soc = _soc(AREA_100 + [{**GAP[0], "name": "tags"}], shoreline_mm=50.0)
    (finding,) = phy_shoreline(soc)
    assert "lower bound" in finding.message


@pytest.mark.parametrize("ref_area, expect", [
    (100.0, None),                 # exact
    (100.0 / 1.10, None),          # +10%: inside
    (100.0 / 1.30, Severity.WARNING),  # +30%: outside
])
def test_reference_comparison_runs_on_a_complete_design(ref_area, expect):
    ref = Reference(die_area_mm2=ref_area, process_node="samsung_8lpp", source="synthetic")
    findings = reference_comparison(_soc(AREA_100, reference=ref))
    if expect is None:
        assert findings == []
    else:
        (finding,) = findings
        assert finding.severity == expect
        assert f"{REFERENCE_TOLERANCE:.0%}" in finding.message


def test_an_incomplete_design_is_not_compared_with_its_reference():
    """Comparing a lower bound with a published die would read as a miss --
    or, worse, as a pass -- when it is neither."""
    ref = Reference(die_area_mm2=100.0, process_node="samsung_8lpp", source="synthetic")
    (finding,) = reference_comparison(_soc(GAP, reference=ref))
    assert finding.severity == Severity.INFO
    assert "Not compared" in finding.message


def test_a_reference_on_another_node_is_not_compared():
    ref = Reference(die_area_mm2=100.0, process_node="samsung_8lpp", source="synthetic")
    assert reference_comparison(_soc(AREA_100, reference=ref, node="tsmc_n7")) == []


# ---------------------------------------------------------------------------
# The Orin design
# ---------------------------------------------------------------------------


def test_orin_at_its_own_node():
    findings = validate_soc(compose_soc(ORIN, LIBRARY, NODES))
    by_name = {f.validator: f for f in findings}
    assert by_name["soc_area_completeness"].severity == Severity.WARNING
    assert by_name["soc_reference_comparison"].severity == Severity.INFO
    assert by_name["soc_phy_shoreline"].severity == Severity.INFO
    assert "soc_clock_retargeting" not in by_name
    assert not any(f.severity == Severity.ERROR for f in findings)


def test_orin_at_n7_flags_its_clocks():
    findings = clock_retargeting(compose_soc(ORIN, LIBRARY, NODES, "tsmc_n7"))
    assert sorted(f.block for f in findings) == ["cpu", "dla", "gpu_sm"]
    assert all(f.severity == Severity.WARNING for f in findings)


# ---------------------------------------------------------------------------
# CLIs
# ---------------------------------------------------------------------------


def _run(*args):
    return subprocess.run([sys.executable, *args], capture_output=True, text=True,
                          cwd=REPO, timeout=300)


def test_validate_sku_soc_mode():
    ok = _run("cli/validate_sku.py", "--soc", "orin_class_reference")
    assert ok.returncode == 0, ok.stderr
    assert "orin_class_reference@samsung_8lpp" in ok.stdout
    assert "soc_area_completeness" in ok.stdout
    strict = _run("cli/validate_sku.py", "--soc", "orin_class_reference", "--strict")
    assert strict.returncode == 1  # the completeness warning fails --strict
    unknown = _run("cli/validate_sku.py", "--soc", "no_such_design")
    assert unknown.returncode == 2


def test_show_soc_reports_a_lower_bound(tmp_path):
    text = _run("cli/show_soc.py", "--design", "orin_class_reference")
    assert text.returncode == 0, text.stderr
    assert "LOWER BOUND" in text.stdout
    assert "not comparable" in text.stdout
    out = tmp_path / "orin.json"
    as_json = _run("cli/show_soc.py", "--design", "orin_class_reference",
                   "--node", "tsmc_n7", "--output", str(out))
    assert as_json.returncode == 0, as_json.stderr
    payload = json.loads(out.read_text())
    assert payload["die_area_is_lower_bound"] is True
    assert len(payload["gaps"]) == 14
    assert set(payload["provisional_clocks"]) == {"gpu_sm", "dla", "cpu"}
    # A block with no anchored line is unknown, never zero.
    dla = next(b for b in payload["blocks"] if b["instance"] == "dla")
    assert dla["area_mm2"] == "unknown"


def test_show_soc_lists_the_library():
    result = _run("cli/show_soc.py", "--list-ip")
    assert result.returncode == 0, result.stderr
    assert "nvidia_ampere_sm_orin" in result.stdout
    assert "nvdla_v1_large" in result.stdout


def test_show_soc_lists_the_library_as_markdown(tmp_path):
    out = tmp_path / "ip.md"
    result = _run("cli/show_soc.py", "--list-ip", "--output", str(out))
    assert result.returncode == 0, result.stderr
    text = out.read_text()
    assert text.startswith("## IP library (")
    assert text.splitlines()[2].startswith("| ") and set(text.splitlines()[3]) <= set("|-")
    assert "nvdla_v1_large" in text


def _load_cli(name):
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, REPO / "cli" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _broken(*_args, **_kwargs):
    raise ValueError("ip foo.yaml: id 'bar' must match the file name")


def test_a_malformed_catalog_is_an_error_not_a_traceback(monkeypatch, capsys):
    """A catalog that fails to load exits 2 with the reason on stderr, in both
    CLIs (CodeRabbit on #304)."""
    show_soc = _load_cli("show_soc")
    monkeypatch.setattr(show_soc, "load_ip_library", _broken)
    assert show_soc.main(["--list-ip"]) == 2
    assert "must match the file name" in capsys.readouterr().err

    validate_sku = _load_cli("validate_sku")
    monkeypatch.setattr("graphs.hardware.soc.load_designs", _broken)
    monkeypatch.setattr(sys, "argv", ["validate_sku.py", "--soc", "orin_class_reference"])
    assert validate_sku.main() == 2
    assert "SoC catalog failed to load" in capsys.readouterr().err
