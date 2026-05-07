"""Tests for validation/model_v4/harness/runner.py.

The runner is orchestration: most logic lives in classify, assertions,
ground_truth, and the analyzer. These tests cover:

- _build_subgraph populates the correct input/output/weight bytes
  for matmul (no weight) and linear (weight + bias).
- _resolve_precision dispatches every supported dtype.
- run_sweep with an empty cache + no measurer produces no records and
  records the skips.
- run_sweep with a populated cache produces records (regardless of
  pass/fail outcome -- that's a separate concern).
- run_sweep skips UNSUPPORTED entries cleanly.
- Unknown hardware key raises a clear error rather than silently
  producing nothing.
"""

import json
from pathlib import Path

import pytest

from validation.model_v4.ground_truth.base import Measurement
from validation.model_v4.ground_truth.cache import CacheKey, store
from validation.model_v4.harness.runner import (
    RunnerConfig,
    SWEEP_HW_TO_MAPPER,
    _build_subgraph,
    _resolve_precision,
    run_sweep,
)
from graphs.core.structures import OperationType
from graphs.hardware.resource_model import Precision


# ---------------------------------------------------------------------------
# _build_subgraph
# ---------------------------------------------------------------------------


def test_build_subgraph_matmul_has_no_weight_bytes():
    """matmul has no parameter -- both args are activations. The
    SubgraphDescriptor must reflect that or the analyzer will inflate
    bandwidth pressure."""
    sg = _build_subgraph("matmul", (128, 64, 32), "fp32")
    assert sg.total_weight_bytes == 0
    # input = (M*K + K*N) * 4 bytes
    assert sg.total_input_bytes == (128 * 64 + 64 * 32) * 4
    # output = M*N * 4 bytes
    assert sg.total_output_bytes == 128 * 32 * 4
    assert sg.total_flops == 2 * 128 * 64 * 32
    assert sg.operation_types == [OperationType.MATMUL]


def test_build_subgraph_linear_includes_weight_plus_bias():
    """linear has a (OUT, IN) weight + (OUT,) bias parameter -- both
    contribute to weight_bytes, mirroring what build_linear() actually
    allocates."""
    sg = _build_subgraph("linear", (16, 128, 64), "fp32")
    # input = B*IN * 4 bytes
    assert sg.total_input_bytes == 16 * 128 * 4
    # output = B*OUT * 4 bytes
    assert sg.total_output_bytes == 16 * 64 * 4
    # weight = (IN*OUT + OUT) * 4 bytes
    assert sg.total_weight_bytes == (128 * 64 + 64) * 4
    assert sg.total_flops == 2 * 16 * 128 * 64
    assert sg.operation_types == [OperationType.LINEAR]


def test_build_subgraph_scales_with_dtype():
    """Sub-byte/half precisions scale all byte counts proportionally."""
    sg32 = _build_subgraph("matmul", (256, 256, 256), "fp32")
    sg16 = _build_subgraph("matmul", (256, 256, 256), "fp16")
    assert sg16.total_input_bytes == sg32.total_input_bytes // 2
    assert sg16.total_output_bytes == sg32.total_output_bytes // 2
    # FLOPs do not depend on dtype
    assert sg16.total_flops == sg32.total_flops


def test_build_subgraph_rejects_unknown_op():
    with pytest.raises(ValueError, match="Unsupported op"):
        _build_subgraph("conv2d", (1, 3, 224, 224), "fp32")


# ---------------------------------------------------------------------------
# _resolve_precision
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dtype,expected",
    [
        ("fp32", Precision.FP32),
        ("fp16", Precision.FP16),
        ("bf16", Precision.BF16),
        ("int8", Precision.INT8),
        ("int4", Precision.INT4),
    ],
)
def test_resolve_precision(dtype, expected):
    assert _resolve_precision(dtype) == expected


def test_resolve_precision_rejects_unknown():
    with pytest.raises(ValueError, match="Unknown dtype"):
        _resolve_precision("complex64")


# ---------------------------------------------------------------------------
# run_sweep
# ---------------------------------------------------------------------------


def _write_test_sweep(tmp_path: Path, op: str, entries: list) -> Path:
    """Helper: write a minimal sweep JSON for a specific test scenario."""
    sweep_path = tmp_path / f"{op}_test.json"
    sweep_path.write_text(
        json.dumps(
            {
                "op": op,
                "purpose": "test",
                "generator_seed": 0,
                "generated_against_hardware": ["i7_12700k"],
                "shapes": entries,
            }
        )
    )
    return sweep_path


def test_run_sweep_with_empty_cache_records_skips(tmp_path):
    """No cache + no measurer -> every shape skipped as 'no baseline'."""
    sweep = _write_test_sweep(
        tmp_path,
        "matmul",
        [
            {
                "shape": [256, 256, 256],
                "dtype": "fp32",
                "regime_per_hw": {"i7_12700k": "l2_bound"},
            },
            {
                "shape": [1024, 1024, 1024],
                "dtype": "fp32",
                "regime_per_hw": {"i7_12700k": "l2_bound"},
            },
        ],
    )
    cfg = RunnerConfig(
        sweep_path=sweep, hardware_key="i7_12700k", baseline_dir=tmp_path
    )
    result = run_sweep(cfg)
    assert len(result.records) == 0
    assert len(result.skipped_no_baseline) == 2
    assert len(result.skipped_unsupported) == 0


def test_run_sweep_with_populated_cache_produces_records(tmp_path):
    """Cache hit -> ValidationRecord built and returned."""
    # Pre-populate cache
    store(
        CacheKey("i7_12700k", "matmul", (1024, 1024, 1024), "fp32"),
        Measurement(latency_s=4e-3, energy_j=0.5, trial_count=10),
        baseline_dir=tmp_path,
    )
    sweep = _write_test_sweep(
        tmp_path,
        "matmul",
        [
            {
                "shape": [1024, 1024, 1024],
                "dtype": "fp32",
                "regime_per_hw": {"i7_12700k": "l2_bound"},
            },
        ],
    )
    cfg = RunnerConfig(
        sweep_path=sweep, hardware_key="i7_12700k", baseline_dir=tmp_path
    )
    result = run_sweep(cfg)
    assert len(result.records) == 1
    rec = result.records[0]
    assert rec.hardware == "i7_12700k"
    assert rec.op == "matmul"
    assert rec.shape == (1024, 1024, 1024)
    assert rec.regime_predicted == "l2_bound"
    # The actual pass/fail depends on the analyzer's prediction; we
    # don't assert one outcome -- the point is that the record was built.


def test_run_sweep_handles_vector_add_with_populated_cache(tmp_path):
    """V4 vector_add validation harness: end-to-end, the runner walks
    a vector_add sweep entry, builds the SubgraphDescriptor (1-D
    a + b -> c), predicts via roofline, and produces a ValidationRecord.
    The pass/fail outcome is a separate concern surfaced in PR
    discussion -- the key here is that vector_add wires through the
    runner the same way matmul/linear do."""
    store(
        CacheKey("i7_12700k", "vector_add", (1024,), "fp32"),
        Measurement(latency_s=2e-6, energy_j=None, trial_count=11),
        baseline_dir=tmp_path,
    )
    sweep = _write_test_sweep(
        tmp_path,
        "vector_add",
        [
            {
                "shape": [1024],
                "dtype": "fp32",
                "regime_per_hw": {"i7_12700k": "launch_bound"},
            },
        ],
    )
    cfg = RunnerConfig(
        sweep_path=sweep, hardware_key="i7_12700k", baseline_dir=tmp_path
    )
    result = run_sweep(cfg)
    assert len(result.records) == 1
    rec = result.records[0]
    assert rec.op == "vector_add"
    assert rec.shape == (1024,)
    assert rec.regime_predicted == "launch_bound"


def test_run_sweep_vector_add_uses_tier_aware_path_when_opted_in(tmp_path):
    """Smoke test: with use_tier_aware_memory=True, a DRAM-bound
    vector_add subgraph produces a ValidationRecord whose
    binding_tier is 'DRAM' (V5-3b extended to ELEMENTWISE in this PR).
    Without this PR, the path declined for ELEMENTWISE and binding_tier
    came back None."""
    N = 16 * 1024 * 1024  # 192 MB > i7 LLC (25 MB) -> DRAM-bound
    store(
        CacheKey("i7_12700k", "vector_add", (N,), "fp32"),
        Measurement(latency_s=5e-3, energy_j=None, trial_count=11),
        baseline_dir=tmp_path,
    )
    sweep = _write_test_sweep(
        tmp_path,
        "vector_add",
        [
            {
                "shape": [N],
                "dtype": "fp32",
                "regime_per_hw": {"i7_12700k": "dram_bound"},
            },
        ],
    )
    cfg = RunnerConfig(
        sweep_path=sweep,
        hardware_key="i7_12700k",
        baseline_dir=tmp_path,
        use_tier_aware_memory=True,
    )
    result = run_sweep(cfg)
    assert len(result.records) == 1
    assert result.records[0].binding_tier == "DRAM"


def test_run_sweep_skips_unsupported_entries(tmp_path):
    """An entry classified as UNSUPPORTED on the target hardware is
    counted in skipped_unsupported, not silently dropped or attempted."""
    sweep = _write_test_sweep(
        tmp_path,
        "matmul",
        [
            {
                "shape": [1024, 1024, 1024],
                "dtype": "fp16",
                "regime_per_hw": {"i7_12700k": "unsupported"},
            },
            {
                "shape": [1024, 1024, 1024],
                "dtype": "fp32",
                "regime_per_hw": {"i7_12700k": "l2_bound"},
            },
        ],
    )
    # Populate cache for the second one so we can count records correctly
    store(
        CacheKey("i7_12700k", "matmul", (1024, 1024, 1024), "fp32"),
        Measurement(latency_s=4e-3, energy_j=0.5, trial_count=10),
        baseline_dir=tmp_path,
    )
    cfg = RunnerConfig(
        sweep_path=sweep, hardware_key="i7_12700k", baseline_dir=tmp_path
    )
    result = run_sweep(cfg)
    assert len(result.records) == 1
    assert len(result.skipped_unsupported) == 1
    assert result.skipped_unsupported[0] == ("matmul", (1024, 1024, 1024), "fp16")


def test_run_sweep_filters_entries_lacking_target_hw_label(tmp_path):
    """Sweep entries that don't list the requested hardware in
    regime_per_hw are silently filtered (they were generated for some
    other hw and are out of scope for this run)."""
    sweep = _write_test_sweep(
        tmp_path,
        "matmul",
        [
            # Only labeled for h100, not i7
            {
                "shape": [4096, 4096, 4096],
                "dtype": "fp16",
                "regime_per_hw": {"h100_sxm5_80gb": "dram_bound"},
            },
        ],
    )
    cfg = RunnerConfig(
        sweep_path=sweep, hardware_key="i7_12700k", baseline_dir=tmp_path
    )
    result = run_sweep(cfg)
    assert len(result.records) == 0
    assert len(result.skipped_no_baseline) == 0
    assert len(result.skipped_unsupported) == 0


def test_run_sweep_unknown_hardware_key_raises(tmp_path):
    """A typo'd hardware key must fail loudly so the user notices,
    rather than silently producing 0 records."""
    sweep = _write_test_sweep(tmp_path, "matmul", [])
    cfg = RunnerConfig(
        sweep_path=sweep, hardware_key="non_existent_hw", baseline_dir=tmp_path
    )
    with pytest.raises(ValueError, match="non_existent_hw"):
        run_sweep(cfg)


def test_sweep_hw_to_mapper_keys_resolve(tmp_path):
    """Every key in SWEEP_HW_TO_MAPPER must actually resolve to a
    real mapper -- protects against typos in the dispatch table."""
    from graphs.hardware.mappers import get_mapper_by_name

    for sweep_key, mapper_key in SWEEP_HW_TO_MAPPER.items():
        m = get_mapper_by_name(mapper_key)
        assert m is not None, f"{sweep_key} -> {mapper_key} did not resolve"
