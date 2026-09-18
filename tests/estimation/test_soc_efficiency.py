"""Efficiency tables and kernel classes for the SoC analyzer (graphs#269 PR 3.1)."""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import pytest
from embodied_schemas import load_process_nodes
from pydantic import ValidationError

from graphs.core.pipeline_workload import DEFAULT_THROUGHPUT, load_autonomy_workload
from graphs.estimation.soc import (
    CLASS_FLOOR,
    EfficiencyEntry,
    EfficiencyTable,
    KernelClass,
    execution_format,
    load_efficiency_tables,
    load_kernel_classes,
)
from graphs.hardware.soc import Confidence, EngineKind, compose_soc, load_designs, load_ip_library

REPO = Path(__file__).resolve().parents[2]
TABLES = load_efficiency_tables()
KERNELS = load_kernel_classes()
WORKLOAD = load_autonomy_workload()


# ---------------------------------------------------------------------------
# Kernel classes
# ---------------------------------------------------------------------------


def test_every_stage_has_exactly_one_kernel_class():
    KERNELS.check_against(WORKLOAD.stages)
    assert KERNELS.workload == WORKLOAD.version


def test_kernel_class_map_rejects_a_missing_or_stray_stage():
    with pytest.raises(KeyError, match="missing"):
        KERNELS.check_against(list(WORKLOAD.stages) + ["isp"])


@pytest.mark.parametrize("stage, kernel", [
    ("det", KernelClass.DENSE_CONV_GEMM),
    ("sgm", KernelClass.COST_VOLUME_DP),
    ("vlm", KernelClass.WEIGHT_STREAM_DECODE),
    ("mpc", KernelClass.SMALL_QP),
    ("esdf", KernelClass.WAVEFRONT),
])
def test_unambiguous_stages_classify_as_their_basis_says(stage, kernel):
    assert KERNELS.of(stage) == kernel


# ---------------------------------------------------------------------------
# Format selection: fail closed, never promote
# ---------------------------------------------------------------------------


def test_a_class_runs_in_the_lowest_format_at_or_above_its_floor():
    sm = {"int8": 4096.0, "fp32": 256.0}
    assert execution_format("A", sm) == "int8"
    # No FP16 on the SM template: Class B runs in FP32, above its floor.
    assert execution_format("B", sm) == "fp32"
    assert execution_format("C", sm) == "fp32"


def test_an_engine_without_a_format_at_the_floor_cannot_run_the_class():
    dla = {"int8": 16384.0}
    assert execution_format("A", dla) == "int8"
    assert execution_format("B", dla) is None
    assert execution_format("C", dla) is None


def test_a_zero_peak_is_not_support():
    assert execution_format("C", {"fp32": 0.0}) is None


def test_class_floors_match_the_workload_definition():
    """A INT8-eligible, B FP16 floor, C FP32/FP64 floor (taken as FP32)."""
    assert CLASS_FLOOR == {"A": "int8", "B": "fp16", "C": "fp32"}


# ---------------------------------------------------------------------------
# annex_v1: the annex's pooled model, carried as data
# ---------------------------------------------------------------------------


def test_annex_v1_is_the_workload_models_throughput():
    annex = TABLES["annex_v1"]
    assert annex.kind == "pooled"
    assert annex.pooled.as_effective_throughput() == DEFAULT_THROUGHPUT


def test_annex_v1_reproduces_the_annex_on_every_profile():
    """The pooled table fed back through the workload gives the annex's own
    oversubscription and stages-over: nothing added, nothing lost."""
    throughput = TABLES["annex_v1"].pooled.as_effective_throughput()
    for profile in WORKLOAD.profiles:
        ref = WORKLOAD.summary(profile)
        occupancy = sum(
            d.stage.service_time_s(throughput) * d.rate_hz for d in ref.demands
        )
        assert occupancy == pytest.approx(ref.oversubscription, rel=1e-12), profile.id


def test_a_pooled_table_has_no_per_engine_lookup():
    with pytest.raises(TypeError, match="pooled"):
        TABLES["annex_v1"].lookup(KernelClass.DENSE_CONV_GEMM, EngineKind.GPU, "int8")


# ---------------------------------------------------------------------------
# default_v1: numbers only where measured
# ---------------------------------------------------------------------------


def test_default_v1_states_numbers_only_for_measured_pairs():
    table = TABLES["default_v1"]
    known = {e.key for e in table.known_entries}
    assert known == {
        (KernelClass.DENSE_CONV_GEMM, EngineKind.GPU, "fp32"),
        (KernelClass.DENSE_CONV_GEMM, EngineKind.NPU, "int8"),
    }
    for entry in table.known_entries:
        assert entry.confidence == Confidence.INTERPOLATED
        assert "calibrations" in entry.source


def test_every_unknown_entry_says_what_is_missing():
    for entry in TABLES["default_v1"].entries:
        if not entry.known:
            assert entry.confidence == Confidence.UNKNOWN
            assert entry.source.startswith("No "), entry.key


def _orin():
    return compose_soc(load_designs()["orin_class_reference"], load_ip_library(),
                       load_process_nodes(), "samsung_8lpp")


def test_dla_int8_efficiency_reproduces_the_calibration():
    """Recompute the entry from the raw TensorRT layer runs: median attained
    GOPS of the conv2d layers that ran wholly on the DLA, over the composed
    per-DLA peak."""
    attained = []
    for file in sorted((REPO / "hardware_registry/accelerator/nvidia_dla_orin/calibrations")
                       .glob("dla*_int8_maxn_*.json")):
        for layer in json.loads(file.read_text())["layer_benchmarks"]:
            if (layer["status"] == "success" and layer["layer_type"] == "conv2d"
                    and layer["on_dla"] and layer["gpu_layer_count"] == 0):
                attained.append(layer["attained_gflops"] * 1e9)
    assert len(attained) == 18
    dla = next(b for b in _orin().blocks if b.engine_kind == EngineKind.NPU)
    per_dla_peak = dla.peak_ops_per_s("int8") / dla.count
    entry = TABLES["default_v1"].lookup(KernelClass.DENSE_CONV_GEMM, EngineKind.NPU, "int8")
    assert entry.compute_eff == pytest.approx(statistics.median(attained) / per_dla_peak, abs=5e-5)
    assert entry.eff_range[0] == pytest.approx(min(attained) / per_dla_peak, abs=5e-5)
    assert entry.eff_range[1] == pytest.approx(max(attained) / per_dla_peak, abs=5e-5)


def test_gpu_fp32_efficiency_reproduces_the_calibration():
    attained = []
    for file in sorted((REPO / "hardware_registry/gpu/jetson_orin_agx_gpu/calibrations")
                       .glob("MAXN_*.json")):
        for profile in (json.loads(file.read_text()).get("operation_profiles") or {}).values():
            if profile["operation_type"] == "blas3_gemm" and profile["extra_params"]["size"] == 2048:
                attained.append(profile["precision_results"]["fp32"]["measured_gops"] * 1e9)
    assert len(attained) == 2
    gpu = next(b for b in _orin().blocks if b.engine_kind == EngineKind.GPU)
    peak = gpu.peak_ops_per_s("fp32")
    entry = TABLES["default_v1"].lookup(KernelClass.DENSE_CONV_GEMM, EngineKind.GPU, "fp32")
    assert entry.eff_range == pytest.approx((min(attained) / peak, max(attained) / peak), abs=5e-4)
    assert entry.compute_eff == pytest.approx(statistics.mean(attained) / peak, abs=5e-4)


def test_a_missing_row_is_none_not_a_default():
    assert TABLES["default_v1"].lookup(KernelClass.GRAPH_SEARCH, EngineKind.CPU, "fp32") is None


# ---------------------------------------------------------------------------
# Schema guards
# ---------------------------------------------------------------------------


def _entry(**kw):
    base = dict(kernel_class="fft", engine_kind="gpu", precision="fp32",
                compute_eff=0.5, confidence="interpolated", source="a test")
    base.update(kw)
    return EfficiencyEntry.model_validate(base)


def test_an_unknown_efficiency_must_be_labelled_unknown():
    with pytest.raises(ValidationError, match="unknown"):
        _entry(compute_eff=None)
    with pytest.raises(ValidationError, match="confidence above unknown"):
        _entry(confidence="unknown")


def test_a_range_must_bracket_its_value():
    with pytest.raises(ValidationError, match="bracket"):
        _entry(eff_range=(0.6, 0.9))


def test_efficiency_is_a_fraction():
    with pytest.raises(ValidationError):
        _entry(compute_eff=1.5)


def test_an_unknown_precision_is_rejected():
    with pytest.raises(ValidationError, match="precision"):
        _entry(precision="int4")


def test_duplicate_entries_are_rejected():
    e = _entry().model_dump()
    with pytest.raises(ValidationError, match="duplicate"):
        EfficiencyTable.model_validate(
            dict(id="t", name="t", kind="per_engine", entries=[e, e]))


def test_a_pooled_table_cannot_also_carry_entries():
    with pytest.raises(ValidationError, match="pooled"):
        EfficiencyTable.model_validate(dict(
            id="t", name="t", kind="pooled", entries=[_entry().model_dump()],
            pooled=dict(a=1, b=1, c=1, confidence="theoretical", source="x")))
