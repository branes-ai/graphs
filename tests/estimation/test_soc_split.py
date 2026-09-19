"""Split mappings: a stage's precision classes on different engines (graphs#269 Phase 5)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from graphs.estimation.soc import EfficiencyTable, KernelClass, SoCAnalyzer, engines_of, find_mapping
from graphs.estimation.soc.mapping import SUSTAINED_DRAM_FRACTION, StageAssignment

TRANSFER = 6.4e6  # bytes of the detector's trunk-to-head intermediate, for the tests


def _all_known() -> EfficiencyTable:
    return EfficiencyTable.model_validate(dict(
        id="all_known", name="t", kind="per_engine",
        entries=[dict(kernel_class=k.value, engine_kind=e, precision=p, compute_eff=0.3,
                      confidence="theoretical", source="t")
                 for k in KernelClass for e in ("cpu", "gpu", "npu", "kpu")
                 for p in ("int8", "fp16", "fp32", "fp64")]))


@pytest.fixture(scope="module")
def analyzer():
    return SoCAnalyzer(tables={**SoCAnalyzer().tables, "all_known": _all_known()})


def _split_det(analyzer, transfers=None, split=None, node="tsmc_n7"):
    mapping = find_mapping("orin_class_reference", analyzer.workload.version).assignments()
    mapping["det"] = split or {"A": "dla", "B": "gpu_sm"}
    result = analyzer.analyze("orin_class_reference", "air superiority", node=node,
                              efficiency="all_known", mapping=mapping, transfers=transfers)
    return result, next(s for s in result.schedule.services if s.stage == "det")


# ---------------------------------------------------------------------------
# Service time and utilization
# ---------------------------------------------------------------------------


def test_the_int8_trunk_goes_to_the_dla_the_fp16_head_to_the_gpu(analyzer):
    """det is 85% Class A, 15% Class B. Whole, the DLA cannot run it; split,
    the DLA takes the INT8 trunk and the GPU the head."""
    result, det = _split_det(analyzer, {"det": TRANSFER})
    assert det.served and det.class_engines == {"A": "dla", "B": "gpu_sm"}
    assert det.formats == {"A": "int8", "B": "fp16"}  # the SM's sourced FP16 tensor rate
    assert set(det.parts) == {"dla", "gpu_sm"}
    engines = engines_of(result.soc)
    stage = result.stages["det"]
    expected_dla = stage.ops_per_call * 0.85 / (engines["dla"].server_peak_ops_per_s("int8") * 0.3)
    assert det.parts["dla"] == pytest.approx(expected_dla)


def test_the_parts_run_in_sequence_plus_the_transfer(analyzer):
    _, det = _split_det(analyzer, {"det": TRANSFER})
    supply = 204.8 * SUSTAINED_DRAM_FRACTION
    assert det.t_transfer_s == pytest.approx(2 * TRANSFER / (supply * 1e9))  # written, then read
    assert det.t_compute_s == pytest.approx(sum(det.parts.values()))
    assert det.t_service_s == pytest.approx(max(det.t_compute_s + det.t_transfer_s, det.t_memory_s))


def test_each_engine_carries_only_its_part(analyzer):
    result, det = _split_det(analyzer, {"det": TRANSFER})
    util = result.schedule.engine_utilization()
    servers = result.schedule.servers
    others_on_gpu = sum(s.t_service_s * s.rate_hz for s in result.schedule.served
                        if s.engine == "gpu_sm" and s.stage != "det")
    assert util["dla"] == pytest.approx(det.parts["dla"] * det.rate_hz / servers["dla"])
    assert util["gpu_sm"] == pytest.approx(others_on_gpu + det.parts["gpu_sm"] * det.rate_hz)


def test_the_transfer_is_dram_demand(analyzer):
    with_split, det = _split_det(analyzer, {"det": TRANSFER})
    whole = analyzer.analyze("orin_class_reference", "air superiority", node="tsmc_n7",
                             efficiency="all_known")
    extra = with_split.schedule.dram_demand_gb_per_s - whole.schedule.dram_demand_gb_per_s
    assert extra == pytest.approx(2 * TRANSFER * det.rate_hz / 1e9)


def test_dynamic_power_prices_each_class_on_its_own_engine(analyzer):
    result, det = _split_det(analyzer, {"det": TRANSFER})
    by_block = {b.name: b for b in result.power.blocks}
    assert by_block["dla"].dynamic_w > 0  # the trunk's INT8 ops, in the DLA's library
    # The head runs in FP16 on the GPU, priced at the node's FP16 figure in
    # the SM's library, not at the DLA's.
    assert by_block["gpu_sm"].dynamic_w > 0
    assert not [g for g in result.power.dynamic.gaps if g.startswith("det:")]


# ---------------------------------------------------------------------------
# An unstated transfer is a lower bound, never a free lunch
# ---------------------------------------------------------------------------


def test_without_a_transfer_size_the_split_is_a_lower_bound(analyzer):
    result, det = _split_det(analyzer)
    assert det.served and det.lower_bound and det.t_transfer_s == 0.0
    assert det.confidence.value == "unknown"
    assert "no intermediate size" in det.confidence_source
    assert result.schedule.lower_bounds == ("det",)
    assert not result.schedule.complete
    assert result.schedule.feasible() in (None, False)  # open unless something is proven over
    assert result.to_dict()["stages"][[s["stage"] for s in result.to_dict()["stages"]].index("det")][
        "lower_bound"] is True


def test_a_split_onto_one_engine_needs_no_transfer(analyzer):
    _, det = _split_det(analyzer, split={"A": "gpu_sm", "B": "gpu_sm"})
    assert not det.lower_bound and det.t_transfer_s == 0.0


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


def test_a_class_on_an_engine_that_cannot_run_it_is_a_gap(analyzer):
    _, det = _split_det(analyzer, split={"A": "gpu_sm", "B": "dla"})
    assert not det.served and det.gap == "npu cannot run Class B"


def test_a_split_must_place_every_class_the_stage_has(analyzer):
    _, det = _split_det(analyzer, split={"A": "dla"})
    assert not det.served and "exactly classes" in det.gap


def test_an_unknown_engine_in_a_split_is_an_error(analyzer):
    with pytest.raises(KeyError, match="does not have"):
        _split_det(analyzer, split={"A": "tpu", "B": "gpu_sm"})


@pytest.mark.parametrize("data, match", [
    ({"engine": "gpu_sm", "engines": {"A": "dla"}, "reason": "both forms given"}, "exactly one"),
    ({"engines": {"D": "dla"}, "reason": "an unknown class"}, "A, B or C"),
    ({"engine": "gpu_sm", "transfer_bytes": 1e6, "transfer_source": "somewhere real",
      "reason": "transfer on a whole stage"}, "only qualifies a split"),
    ({"engines": {"A": "dla", "B": "gpu_sm"}, "transfer_bytes": 1e6, "reason": "no source given"},
     "transfer_source"),
])
def test_mapping_file_assignments_are_validated(data, match):
    with pytest.raises(ValidationError, match=match):
        StageAssignment.model_validate(data)


def test_a_mapping_file_split_carries_its_transfer(tmp_path, analyzer):
    import yaml

    shipped = find_mapping("orin_class_reference", analyzer.workload.version)
    data = shipped.model_dump(exclude_none=True)
    data["stages"]["det"] = {"engines": {"A": "dla", "B": "gpu_sm"}, "transfer_bytes": TRANSFER,
                             "transfer_source": "test fixture: a 80x80x1000 INT8 feature map",
                             "reason": "INT8 trunk on the DLA, FP16 head on the GPU"}
    path = tmp_path / "orin_split.yaml"
    path.write_text(yaml.safe_dump(data))
    result = analyzer.analyze("orin_class_reference", "air superiority", node="tsmc_n7",
                              efficiency="all_known", mapping=path)
    det = next(s for s in result.schedule.services if s.stage == "det")
    assert det.class_engines == {"A": "dla", "B": "gpu_sm"} and not det.lower_bound
    assert result.schedule.complete


@pytest.mark.parametrize("bad", [0.0, -1e6])
def test_a_programmatic_transfer_must_be_positive(analyzer, bad):
    """A mapping file validates transfer_bytes > 0; a caller's dict must too,
    or a negative transfer would shorten the service time (#311 review)."""
    with pytest.raises(ValueError, match="must be positive"):
        _split_det(analyzer, {"det": bad})
