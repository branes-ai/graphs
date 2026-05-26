"""Tests for the KPU family scaling study (issue #122)."""

from __future__ import annotations

import pytest

from validation.model_v4.cli.compare_kpu_scaling import (
    KPU_FAMILY,
    kpu_scaling_rows,
    render_kpu_scaling,
    summarize_crossovers,
)
from validation.model_v4.harness.runner import SWEEP_HW_TO_MAPPER


def test_full_kpu_family_is_registered():
    """All four SKUs the scaling study sweeps must resolve to mappers."""
    for k in KPU_FAMILY:
        assert k in SWEEP_HW_TO_MAPPER, f"{k} missing from SWEEP_HW_TO_MAPPER"
    assert KPU_FAMILY == ["kpu_t64", "kpu_t128", "kpu_t256", "kpu_t768"]


@pytest.fixture(scope="module")
def rows():
    return kpu_scaling_rows(dtype="bf16")


def test_rows_cover_every_sku(rows):
    """Each SKU produces predictions (no surprise energy-model gaps -- the
    #122 'same energy-model maturity' check)."""
    seen = {r["hw_key"] for r in rows}
    assert seen == set(KPU_FAMILY)


def test_rows_carry_scaling_metrics(rows):
    """Every row has the derived scaling metrics, physically sane."""
    for r in rows:
        assert r["predicted_energy_j"] > 0
        assert r["predicted_avg_power_w"] > 0
        # avg power must respect TDP (clamp invariant holds in the study too).
        assert r["predicted_avg_power_w"] <= 105.0  # T768 100W profile + slack
        # efficiency is the reciprocal of energy-per-inference.
        assert r["efficiency_inf_per_j"] == pytest.approx(
            1.0 / r["predicted_energy_j"], rel=1e-9
        )
        assert 0.0 <= r["utilization"] <= 1.0


def test_bigger_kpu_never_slower_on_large_matmul(rows):
    """At the largest matmul, latency must be non-increasing across the family
    (more tiles cannot make a fixed large GEMM slower)."""
    big = max(r["shape"] for r in rows if r["op"] == "matmul")
    lat = {}
    for r in rows:
        if r["op"] == "matmul" and r["shape"] == big:
            lat[r["hw_key"]] = r["predicted_latency_ms"]
    seq = [lat[k] for k in KPU_FAMILY if k in lat]
    assert seq == sorted(seq, reverse=True), (
        f"latency not non-increasing across family at {big}: {seq}"
    )


def test_summarize_crossovers_returns_findings(rows):
    findings = summarize_crossovers(rows)
    # one per adjacent pair (3) + one utilization line per SKU (4).
    assert len(findings) >= 7
    assert all(isinstance(f, str) and f for f in findings)


def test_render_produces_nonempty_png(tmp_path, rows):
    # Visualization-only; CI test jobs don't install matplotlib. Skip there
    # (the local dev path / viz CI job has it). Mirrors test_compare_hardware.
    pytest.importorskip("matplotlib")
    out = tmp_path / "kpu_scaling.png"
    render_kpu_scaling(rows, out, dtype="bf16")
    assert out.exists()
    magic = out.read_bytes()[:8]
    assert magic == b"\x89PNG\r\n\x1a\n"
    assert out.stat().st_size > 50_000


def test_table_output_csv(tmp_path, rows):
    from validation.model_v4.cli.compare_kpu_scaling import _write_table
    out = tmp_path / "kpu_scaling.csv"
    _write_table(rows, out)
    text = out.read_text()
    assert "hw_key,op,shape" in text.splitlines()[0]
    assert len(text.splitlines()) == len(rows) + 1  # header + rows
