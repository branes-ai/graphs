"""Phase E2 of the KPU heterogeneous-tile refactor (graphs#268): the
tile-kind energy ladder.

The ladder answers what specialization buys, so what matters is not that
the numbers are stable but that they are *honest*: every rung carries its
provenance, a class is only priced on work it can actually run, and the
ladder says out loud when its own ops model cannot support the comparison.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from embodied_schemas import load_process_nodes

from graphs.hardware.kpu_sku_generator import KPUSKUInputSpec, generate_kpu_sku
from graphs.hardware.kpu_tile_ladder import (
    DRAM_PJ_PER_BYTE,
    OPS_MODELS,
    LadderError,
    build_ladder,
)

SPEC = (
    Path(__file__).resolve().parents[2]
    / "data" / "sku_specs" / "kpu_h64_auto1_lp5x4_16nm_tsmc_ffp.yaml"
)
NODES = load_process_nodes()
N16 = NODES["tsmc_n16"]
SKU = generate_kpu_sku(
    KPUSKUInputSpec.model_validate(yaml.safe_load(SPEC.read_text(encoding="utf-8"))),
    process_nodes=NODES,
)


def _ladder(function_id: str):
    return build_ladder(SKU, N16, function_id, nodes=NODES)


# ---------------------------------------------------------------------------
# Ops models
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("function_id", sorted(OPS_MODELS))
def test_every_ops_model_states_its_arithmetic_and_source(function_id):
    """The ops model is a workload claim, not a silicon one, so it has to
    show its working."""
    ops = OPS_MODELS[function_id]
    assert ops.ops_per_unit > 0
    assert ops.work_unit and ops.operand_formats and ops.required_ops
    assert len(ops.derivation) > 20
    assert len(ops.citation) > 20
    assert ops.confidence in {"CALIBRATED", "INTERPOLATED", "THEORETICAL"}


def test_the_sgm_ops_model_is_the_papers_own_arithmetic():
    """2.3 TOPS/W x 13440 pJ/pixel, both from Li et al. -- so the ops count
    brings in no assumption of ours, which is why it rates INTERPOLATED
    while the structural models rate THEORETICAL."""
    ops = OPS_MODELS["stereo.sgm"]
    assert ops.ops_per_unit == pytest.approx(2.3e12 * 13440e-12)
    assert ops.confidence == "INTERPOLATED"
    assert "ISSCC 2017" in ops.citation
    assert OPS_MODELS["isp.raw_to_yuv"].confidence == "THEORETICAL"


# ---------------------------------------------------------------------------
# Capability: a class is only priced on work it can run
# ---------------------------------------------------------------------------


def test_stereo_is_priced_on_the_min_plus_fabric_and_nothing_else():
    """Path aggregation is min-plus work. A MAC array cannot do it, however
    cheap its MACs look, so it must not appear as a rung."""
    kinds = {r.label.split()[0] for r in _ladder("stereo.sgm").rungs}
    assert "pe_minplus_i16" in kinds
    assert "pe_int8_mac_i32" not in kinds
    assert "systolic_int8_ws" not in kinds


def test_gemm_is_priced_on_the_mac_classes_and_not_the_min_plus_one():
    kinds = {r.label.split()[0] for r in _ladder("gemm.int8").rungs}
    assert {"systolic_int8_ws", "pe_int8_mac_i32"} <= kinds
    assert "pe_minplus_i16" not in kinds


def test_a_systolic_array_is_only_priced_on_its_declared_kernels():
    """Its MACs price well for anything, but a weight-stationary array runs
    gemm and conv2d; an image pipeline is not one of them."""
    assert OPS_MODELS["gemm.int8"].systolic_kernel == "gemm"
    assert OPS_MODELS["isp.raw_to_yuv"].systolic_kernel is None
    assert any(r.kind == "systolic" for r in _ladder("gemm.int8").rungs)
    assert not any(r.kind == "systolic" for r in _ladder("isp.raw_to_yuv").rungs)


def test_an_unknown_function_is_refused_with_the_known_ones():
    with pytest.raises(LadderError, match="no ops model"):
        _ladder("stereo.definitely_not_real")


# ---------------------------------------------------------------------------
# Rungs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("function_id", sorted(OPS_MODELS))
def test_every_rung_carries_provenance_and_confidence(function_id):
    ladder = _ladder(function_id)
    assert ladder.rungs
    for rung in ladder.rungs:
        assert rung.pj_per_unit > 0
        assert rung.confidence in {"CALIBRATED", "INTERPOLATED", "THEORETICAL"}
        assert len(rung.provenance) > 15, rung.label


@pytest.mark.parametrize("function_id", sorted(OPS_MODELS))
def test_rungs_are_cheapest_first_and_the_best_rates_1x(function_id):
    ladder = _ladder(function_id)
    energies = [r.pj_per_unit for r in ladder.rungs]
    assert energies == sorted(energies)
    assert ladder.ratio_to_best(ladder.rungs[0]) == pytest.approx(1.0)


@pytest.mark.parametrize("function_id", sorted(OPS_MODELS))
def test_the_reference_rungs_are_always_present(function_id):
    """A KPU number on its own says nothing; the GPU and CPU are what make
    it a ratio."""
    kinds = {r.kind for r in _ladder(function_id).rungs}
    assert "gpu" in kinds and "cpu" in kinds


def test_pj_per_unit_is_pj_per_op_times_the_ops_model():
    ladder = _ladder("gemm.int8")
    for rung in ladder.rungs:
        assert rung.pj_per_unit == pytest.approx(
            rung.pj_per_op * ladder.ops.ops_per_unit
        )


def test_the_systolic_array_beats_the_pe_fabric_on_gemm():
    """The reason the class exists: a weight-stationary cell has no token
    match or operand routing, so its MAC is cheaper."""
    by_kind = {r.kind: r for r in _ladder("gemm.int8").rungs}
    assert by_kind["systolic"].pj_per_op < by_kind["pe_fabric"].pj_per_op
    assert _ladder("gemm.int8").follows_specialization_order


# ---------------------------------------------------------------------------
# Honesty: the ladder reports where its own model is weak
# ---------------------------------------------------------------------------


def test_the_specialization_order_is_a_finding_not_an_invariant():
    """Sorting by energy and then asking whether the sort ascends would
    prove nothing. The check compares the resulting *kind* order against
    the expected one, so it can fail -- and on this SKU it does: a
    purpose-built min-plus fabric at n16 beats an n40 stereo ASIC
    retargeted to n16."""
    sgm = _ladder("stereo.sgm")
    assert sgm.follows_specialization_order is False
    fabric = next(r for r in sgm.rungs if r.kind == "pe_fabric")
    core = next(r for r in sgm.rungs if r.kind == "fixed_function")
    assert fabric.pj_per_unit < core.pj_per_unit


def test_a_core_that_is_not_arithmetic_bound_says_so():
    """Navion's energy is memory and control, not MACs. Its implied energy
    per op is many times the node's arithmetic op, and reading its ladder
    position as arithmetic efficiency would be wrong -- so the ladder
    refuses to let that pass silently."""
    vio = _ladder("vio.stereo_inertial")
    assert vio.arithmetic_bound is False
    assert vio.back_check_notes
    note = vio.back_check_notes[0]
    assert "memory and control" in note and "lower bound" in note


def test_a_core_that_is_arithmetic_bound_raises_no_back_check():
    sgm = _ladder("stereo.sgm")
    assert sgm.arithmetic_bound is True
    assert sgm.back_check_notes == ()


# ---------------------------------------------------------------------------
# Segment encapsulation
# ---------------------------------------------------------------------------


def test_a_stream_linked_core_avoids_a_write_and_a_read_back():
    """The chain exists so a frame does not round-trip through DRAM
    between stages: one write and one read saved per work unit."""
    ladder = _ladder("stereo.sgm")
    core = next(t for t in SKU.dies[0].blocks[0].tiles
                if getattr(t, "core", None) is not None
                and t.core.function_id == "stereo.sgm")
    out_bytes = core.core.io.output_bytes_per_unit
    assert ladder.dram_bytes_per_unit == pytest.approx(2 * out_bytes)
    assert ladder.dram_pj_per_unit == pytest.approx(
        2 * out_bytes * DRAM_PJ_PER_BYTE
    )
    assert "not written and not read back" in ladder.dram_note


def test_gemm_has_no_encapsulation_claim():
    """It has no fixed-function core, so there is nothing to encapsulate
    and the ladder claims nothing."""
    ladder = _ladder("gemm.int8")
    assert ladder.dram_bytes_per_unit is None
    assert ladder.dram_pj_per_unit is None


# ---------------------------------------------------------------------------
# Area
# ---------------------------------------------------------------------------


def test_the_cores_report_the_silicon_they_cost():
    """Energy is half the trade; the other half is area, and a
    fixed-function core buys its efficiency with silicon that does nothing
    else."""
    sgm = next(r for r in _ladder("stereo.sgm").rungs if r.kind == "fixed_function")
    fabric = next(r for r in _ladder("stereo.sgm").rungs if r.kind == "pe_fabric")
    assert sgm.area_mm2 is not None and sgm.area_mm2 > 0
    # The stereo core is far larger than one min-plus fabric tile.
    assert fabric.area_mm2 is None or sgm.area_mm2 > fabric.area_mm2


# ---------------------------------------------------------------------------
# Review fixes (CodeRabbit on #289)
# ---------------------------------------------------------------------------


def test_the_preferred_format_wins_over_declaration_order():
    """Formats are tried in the order the ops model prefers, not the order
    the tile happens to declare its modes in."""
    ladder = _ladder("vio.stereo_inertial")
    fabric = next(r for r in ladder.rungs if r.kind == "pe_fabric")
    tile = next(
        t for t in SKU.dies[0].blocks[0].tiles
        if t.tile_class_id == "pe_int8_mac_i32"
    )
    declared = [
        m.operand_format
        for u in tile.datapath.functional_units for m in u.modes
    ]
    # The class declares int8 first and has no fp32 at all, so the model's
    # fp32 -> bf16 -> fp16 preference must pick bf16, not the first mode.
    assert declared[0] == "int8" and "fp32" not in declared
    assert ladder.ops.operand_formats == ("fp32", "bf16", "fp16")
    assert "bf16" in fabric.provenance


def test_reference_rungs_are_priced_at_the_workloads_own_format():
    """Defaulting everything non-int8 to fp16 mispriced the fp32 workload by
    a factor of two."""
    gpu_vio = next(r for r in _ladder("vio.stereo_inertial").rungs if r.kind == "gpu")
    gpu_gemm = next(r for r in _ladder("gemm.int8").rungs if r.kind == "gpu")
    # fp32 is the unscaled reference; int8 is an eighth of it on this device.
    assert gpu_vio.pj_per_op == pytest.approx(8 * gpu_gemm.pj_per_op)


def test_a_function_the_kpu_cannot_run_is_refused_not_answered_by_the_gpu():
    """The reference rungs are context, not evidence that this SKU runs the
    function; without the guard a KPU with no implementation would still
    report it as supported."""
    from graphs.hardware.kpu_tile_ladder import OpsModel, OPS_MODELS

    OPS_MODELS["test.unrunnable"] = OpsModel(
        ops_per_unit=1.0, work_unit="widget",
        operand_formats=("fp64",), required_ops=("definitely_not_an_op",),
        systolic_kernel=None,
        derivation="a function no tile class in any SKU declares an op for",
        confidence="THEORETICAL",
        citation="test fixture for the no-implementation guard (#289)",
    )
    try:
        with pytest.raises(LadderError, match="could be priced"):
            _ladder("test.unrunnable")
    finally:
        del OPS_MODELS["test.unrunnable"]


def test_every_core_implementing_a_function_gets_a_rung():
    """A die may carry two cores for one function; returning the first
    would quietly drop the rest."""
    from graphs.hardware.kpu_tile_ladder import _cores_for
    from graphs.hardware.kpu_access import kpu_block_of

    block = kpu_block_of(SKU)
    assert len(_cores_for(block, "stereo.sgm")) == 1
    assert _cores_for(block, "not.a.function") == []
    # One rung per implementing core.
    ff_rungs = [r for r in _ladder("stereo.sgm").rungs if r.kind == "fixed_function"]
    assert len(ff_rungs) == len(_cores_for(block, "stereo.sgm"))
