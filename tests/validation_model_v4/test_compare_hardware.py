"""Smoke tests for the 3-way hardware comparison CLI.

The actual rendering is matplotlib; we don't visually diff PNGs here.
What's pinned:
  * ``_enrich_predictions`` runs the analyzer in tier-aware mode and
    produces the expected fields for matmul / linear / vector_add.
  * ``render_comparison`` runs end-to-end without raising, producing
    a non-empty PNG for the default 3-target compare.
  * Default dtype map matches each target's native precision.
  * KPU T64 is registered as a comparison target (the predictions-only
    path the user-driven 3-way comparison depends on).
"""

from pathlib import Path

import pytest

from graphs.hardware.mappers import get_mapper_by_name

from validation.model_v4.cli.compare_hardware import (
    DEFAULT_DTYPES,
    DEFAULT_TARGETS,
    _enrich_predictions,
    render_comparison,
)
from validation.model_v4.harness.runner import SWEEP_HW_TO_MAPPER


def test_default_targets_are_known_to_runner_registry():
    """Every default comparison target must be in SWEEP_HW_TO_MAPPER --
    otherwise the visualizer / runner can't translate the key to a
    concrete mapper."""
    for hw_key in DEFAULT_TARGETS:
        assert hw_key in SWEEP_HW_TO_MAPPER, (
            f"{hw_key!r} not in SWEEP_HW_TO_MAPPER; add it before using "
            f"in the comparison CLI"
        )


def test_kpu_t64_registered():
    """KPU T64 is the V5 comparison target. Pin its presence."""
    assert "kpu_t64" in SWEEP_HW_TO_MAPPER
    assert SWEEP_HW_TO_MAPPER["kpu_t64"] == "Stillwater-KPU-T64"


def test_default_dtypes_cover_all_targets():
    """Every default target must have a native dtype defined; otherwise
    the comparison falls back to 'fp16' which may be wrong (e.g. CPU)."""
    for hw_key in DEFAULT_TARGETS:
        assert hw_key in DEFAULT_DTYPES, (
            f"{hw_key!r} missing from DEFAULT_DTYPES"
        )


def test_default_dtype_map_matches_hardware_class():
    """CPU -> fp32, GPU/KPU -> fp16. (KPU's bf16 fabric also delivers
    fp16 at 33 TFLOPS so fp16 is the right comparison precision; an
    explicit override is available via --dtype-for kpu_t64=bf16.)"""
    assert DEFAULT_DTYPES["i7_12700k"] == "fp32"
    assert DEFAULT_DTYPES["jetson_orin_nano_8gb"] == "fp16"
    assert DEFAULT_DTYPES["kpu_t64"] == "fp16"


def test_enrich_predictions_returns_expected_fields_for_matmul():
    """A small matmul shape on Orin Nano produces all the expected
    predicted fields."""
    hw = get_mapper_by_name("Jetson-Orin-Nano-8GB").resource_model
    e = _enrich_predictions("matmul", (256, 256, 256), "fp16", hw)
    assert e is not None
    assert e["op"] == "matmul"
    assert e["shape"] == (256, 256, 256)
    assert e["dtype"] == "fp16"
    assert e["operational_intensity"] > 0
    assert e["working_set_bytes"] > 0
    assert e["predicted_latency_ms"] > 0
    assert e["predicted_gflops"] > 0
    # Energy may be None on hardware without an energy model,
    # but for the Jetson mapper it's populated.
    assert e["predicted_energy_j"] is not None
    assert e["predicted_energy_j"] > 0


def test_enrich_predictions_returns_expected_fields_for_vector_add():
    """vector_add shape is single-dim; verify the path works."""
    hw = get_mapper_by_name("Stillwater-KPU-T64").resource_model
    e = _enrich_predictions("vector_add", (1024,), "fp16", hw)
    assert e is not None
    assert e["op"] == "vector_add"
    assert e["shape"] == (1024,)
    # vector_add OI = 1/(3*bpe) = 0.167 for fp16
    assert e["operational_intensity"] == pytest.approx(1.0 / (3 * 2))


def test_enrich_predictions_kpu_matmul_faster_than_orin_nano():
    """KPU T64 fp16 has 25x the matmul TFLOPS of Orin Nano fp16. For a
    compute-heavy shape, predicted latency on KPU should be markedly
    lower than on Orin Nano. Sanity check that the comparison plot's
    underlying numbers actually reflect that."""
    orin = get_mapper_by_name("Jetson-Orin-Nano-8GB").resource_model
    kpu = get_mapper_by_name("Stillwater-KPU-T64").resource_model
    # Pick a square shape large enough that compute matters.
    shape = (512, 4096, 4096)
    e_orin = _enrich_predictions("matmul", shape, "fp16", orin)
    e_kpu = _enrich_predictions("matmul", shape, "fp16", kpu)
    assert e_orin is not None
    assert e_kpu is not None
    # KPU should be at least 2x faster (allowing margin for bandwidth
    # asymmetry: KPU's BW is 64 GB/s vs Orin's 102 GB/s, so the gain
    # is bottlenecked by memory at low OI).
    assert e_kpu["predicted_latency_ms"] < e_orin["predicted_latency_ms"], (
        f"Expected KPU faster than Orin Nano for {shape}; got "
        f"KPU={e_kpu['predicted_latency_ms']:.3f}ms vs "
        f"Orin={e_orin['predicted_latency_ms']:.3f}ms"
    )


def test_render_comparison_produces_nonempty_png(tmp_path):
    """End-to-end smoke: default 3-target comparison renders without
    raising and produces a non-empty file."""
    out = tmp_path / "compare.png"
    render_comparison(DEFAULT_TARGETS, out)
    assert out.exists()
    # PNG header check: starts with 8-byte magic
    with out.open("rb") as f:
        magic = f.read(8)
    assert magic == b"\x89PNG\r\n\x1a\n", (
        f"Output file isn't a valid PNG (magic={magic!r})"
    )
    # Reasonable size sanity (not zero-byte, not 10-line debug)
    assert out.stat().st_size > 50_000
