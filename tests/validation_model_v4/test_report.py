"""Tests for validation/model_v4/harness/report.py.

The report module is pure aggregation + formatting on a list of
ValidationRecords. These tests fabricate small synthetic record sets
and assert the format_* functions produce stable, parseable output.
"""

import json

from validation.model_v4.harness.assertions import ValidationRecord
from validation.model_v4.harness.report import (
    _aggregate,
    _cell_glyph,
    format_json,
    format_markdown,
    format_text,
)
from validation.model_v4.harness.runner import RunnerResult


def _record(*, hardware="i7_12700k", op="matmul", shape=(1024, 1024, 1024),
            dtype="fp32", regime_predicted="l2_bound",
            regime_measured="l2_bound",
            pass_regime=True, pass_latency=True, pass_energy=True,
            latency_predicted_ms=10.0, latency_measured_ms=10.0,
            energy_predicted_j=1.0, energy_measured_j=1.0) -> ValidationRecord:
    return ValidationRecord(
        hardware=hardware, op=op, shape=shape, dtype=dtype,
        regime_predicted=regime_predicted,
        latency_predicted_ms=latency_predicted_ms,
        energy_predicted_j=energy_predicted_j,
        regime_measured=regime_measured,
        latency_measured_ms=latency_measured_ms,
        energy_measured_j=energy_measured_j,
        pass_regime=pass_regime,
        pass_latency=pass_latency,
        pass_energy=pass_energy,
        tolerance_latency=0.20,
        tolerance_energy=0.25,
        bottleneck_layer="L2 / LLC capacity",
    )


# ---------------------------------------------------------------------------
# _aggregate -- pass/fail tallies per (hw, op, regime)
# ---------------------------------------------------------------------------


def test_aggregate_groups_by_hw_op_regime():
    records = [
        _record(regime_predicted="l2_bound"),
        _record(regime_predicted="l2_bound"),
        _record(regime_predicted="dram_bound"),
        _record(hardware="h100_sxm5_80gb", regime_predicted="alu_bound"),
    ]
    cells = _aggregate(records)
    assert ("i7_12700k", "matmul", "l2_bound") in cells
    assert ("i7_12700k", "matmul", "dram_bound") in cells
    assert ("h100_sxm5_80gb", "matmul", "alu_bound") in cells
    assert cells[("i7_12700k", "matmul", "l2_bound")].n_total == 2


def test_aggregate_counts_per_assertion_type():
    """A record can fail multiple assertions; each must be tallied
    independently so the report can attribute correctly."""
    records = [
        _record(pass_regime=True, pass_latency=True, pass_energy=True),     # all pass
        _record(pass_regime=False, pass_latency=True, pass_energy=True),    # regime only
        _record(pass_regime=True, pass_latency=False, pass_energy=True),    # latency only
        _record(pass_regime=False, pass_latency=False, pass_energy=False),  # all fail
    ]
    cells = _aggregate(records)
    cell = cells[("i7_12700k", "matmul", "l2_bound")]
    assert cell.n_total == 4
    assert cell.n_pass == 1
    assert cell.n_fail_regime == 2
    assert cell.n_fail_latency == 2
    assert cell.n_fail_energy == 1


def test_aggregate_treats_none_energy_as_not_failed():
    """A record where pass_energy is None (not measured) counts as
    'not failed' for energy aggregation."""
    records = [
        _record(pass_energy=None),      # not measured -> not a fail
        _record(pass_energy=True),
        _record(pass_energy=False),     # only this one is a fail
    ]
    cell = _aggregate(records)[("i7_12700k", "matmul", "l2_bound")]
    assert cell.n_fail_energy == 1


# ---------------------------------------------------------------------------
# _cell_glyph
# ---------------------------------------------------------------------------


def test_cell_glyph_all_pass():
    records = [_record() for _ in range(7)]
    cell = _aggregate(records)[("i7_12700k", "matmul", "l2_bound")]
    glyph = _cell_glyph(cell).strip()
    assert "OK" in glyph
    assert "7" in glyph


def test_cell_glyph_failures_show_breakdown():
    records = [
        _record(pass_regime=False),
        _record(pass_latency=False),
        _record(pass_latency=False),
    ]
    cell = _aggregate(records)[("i7_12700k", "matmul", "l2_bound")]
    glyph = _cell_glyph(cell)
    assert "R1" in glyph
    assert "L2" in glyph


# ---------------------------------------------------------------------------
# format_text
# ---------------------------------------------------------------------------


def _result(records, *, no_baseline=0, unsupported=0):
    return RunnerResult(
        records=records,
        skipped_no_baseline=[("matmul", (1, 1, 1), "fp32")] * no_baseline,
        skipped_unsupported=[("matmul", (1, 1, 1), "fp16")] * unsupported,
    )


def test_format_text_includes_record_skipped_counts():
    text = format_text(_result([_record()], no_baseline=2, unsupported=3),
                       op="matmul")
    assert "records: 1" in text
    assert "skipped(no baseline): 2" in text
    assert "skipped(unsupported): 3" in text


def test_format_text_renders_heatmap_columns_in_severity_order():
    """Columns must appear in firm-spec-first order (alu, l2, dram, launch)
    so a reader scanning left-to-right encounters the strongest signal
    first."""
    text = format_text(_result([_record()]), op="matmul")
    # Find the header line (one with all four regime names)
    header_idx = next(i for i, line in enumerate(text.splitlines())
                      if "alu_bound" in line and "dram_bound" in line)
    header = text.splitlines()[header_idx]
    assert header.index("alu_bound") < header.index("l2_bound")
    assert header.index("l2_bound") < header.index("dram_bound")
    assert header.index("dram_bound") < header.index("launch_bound")


def test_format_text_lists_failures_with_attribution_tags():
    """Each failure row must show R/L/E tags so the reader can tell
    which assertion drifted."""
    records = [
        _record(pass_regime=False, pass_latency=True, pass_energy=True,
                regime_measured="ambiguous"),
        _record(pass_regime=False, pass_latency=False, pass_energy=False,
                regime_measured="ambiguous"),
    ]
    text = format_text(_result(records), op="matmul")
    assert "Failures (2 of 2 records)" in text
    # First failure: only R fails
    # Second failure: R + L + E all fail
    assert "[R  ]" in text or "[R]" in text
    assert "[RLE]" in text


def test_format_text_handles_empty_record_list():
    """No records should not crash; just print an empty heatmap note."""
    text = format_text(_result([]), op="matmul")
    assert "records: 0" in text
    assert "no records produced" in text


# ---------------------------------------------------------------------------
# format_markdown
# ---------------------------------------------------------------------------


def test_format_markdown_uses_pipe_table_syntax():
    text = format_markdown(_result([_record()]), op="matmul")
    assert "| target |" in text
    assert "| --- |" in text


def test_format_markdown_failures_table_present_on_failure():
    text = format_markdown(_result([_record(pass_latency=False)]), op="matmul")
    assert "### Failures" in text
    assert "| op | shape | dtype |" in text


def test_format_markdown_failures_section_omitted_on_clean():
    text = format_markdown(_result([_record()]), op="matmul")
    assert "### Failures" not in text


# ---------------------------------------------------------------------------
# format_json
# ---------------------------------------------------------------------------


def test_format_json_is_parseable_with_expected_schema():
    records = [_record(), _record(pass_latency=False)]
    result = _result(records, no_baseline=1)
    payload = json.loads(format_json(result))
    # Top-level shape
    assert set(payload) == {"summary", "cells", "records"}
    # Summary fields
    assert payload["summary"]["records"] == 2
    assert payload["summary"]["passed"] == 1
    assert payload["summary"]["failed"] == 1
    assert payload["summary"]["skipped_no_baseline"] == 1
    # Records round-trip
    assert len(payload["records"]) == 2
    assert payload["records"][0]["hardware"] == "i7_12700k"


def test_format_json_records_have_full_field_set():
    """The JSON record must carry every ValidationRecord field so the
    output is sufficient for trend tracking without re-running."""
    payload = json.loads(format_json(_result([_record()])))
    rec = payload["records"][0]
    expected = {
        "hardware", "op", "shape", "dtype",
        "regime_predicted", "latency_predicted_ms", "energy_predicted_j",
        "regime_measured", "latency_measured_ms", "energy_measured_j",
        "pass_regime", "pass_latency", "pass_energy",
        "tolerance_latency", "tolerance_energy",
        "bottleneck_layer", "notes",
    }
    assert expected.issubset(set(rec.keys()))
