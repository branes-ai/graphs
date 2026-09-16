"""Phase E3 of the KPU heterogeneous-tile refactor (graphs#268): tile
classes as engines.

The SoC micro-architecture study (graphs#269) maps workload stages onto
engines, and its plan says each KPU tile class is one. That analyzer does
not exist yet, so this is the export it will consume: what each class can
run, what it costs, its precision floor, and the stream-linked chains that
let a fixed-function core absorb several stages.

The regime comparison E3 also asks for needs #269's workload and its data
annex, and is deferred with it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from embodied_schemas import load_compute_products, load_process_nodes

from graphs.hardware.kpu_access import has_kpu_block
from graphs.hardware.kpu_engines import (
    describe_engines,
    describe_segments,
    format_rank,
    precision_floor_findings,
)
from graphs.hardware.kpu_sku_generator import KPUSKUInputSpec, generate_kpu_sku

SPEC = (
    Path(__file__).resolve().parents[2]
    / "data" / "sku_specs" / "kpu_h64_auto1_lp5x4_16nm_tsmc_ffp.yaml"
)
NODES = load_process_nodes()
N16 = NODES["tsmc_n16"]
HETERO = generate_kpu_sku(
    KPUSKUInputSpec.model_validate(yaml.safe_load(SPEC.read_text(encoding="utf-8"))),
    process_nodes=NODES,
)
T64 = load_compute_products()["kpu_t64_32x32_lp5x4_16nm_tsmc_ffp"]

ENGINES = describe_engines(HETERO, N16, NODES)
BY_ID = {e.engine_id: e for e in ENGINES}


# ---------------------------------------------------------------------------
# Engine descriptors
# ---------------------------------------------------------------------------


def test_every_tile_class_becomes_exactly_one_engine():
    """The SoC plan's rule: each KPU tile class is a separate engine."""
    tiles = HETERO.dies[0].blocks[0].tiles
    assert len(ENGINES) == len(tiles)
    assert [e.engine_id for e in ENGINES] == [t.tile_class_id for t in tiles]
    # Declaration order, so the export is reproducible.
    assert [e.engine_kind for e in ENGINES] == [t.tile_kind.value for t in tiles]


def test_a_programmable_engine_reports_throughput_and_energy_per_format():
    fabric = BY_ID["pe_int8_mac_i32"]
    assert fabric.is_programmable
    assert set(fabric.formats) == {"int8", "int4", "bf16", "fp16"}
    int8 = next(p for p in fabric.precisions if p.operand_format == "int8")
    assert int8.ops_per_second_per_tile == pytest.approx(
        int8.ops_per_tile_per_clock * fabric.clock_hz
    )
    assert int8.pj_per_op > 0
    assert int8.tops_per_watt == pytest.approx(1.0 / (int8.pj_per_op * 1e-12) / 1e12)


def test_a_systolic_engine_reports_the_kernels_it_runs():
    """A mapper must not hand it work it was not built for."""
    systolic = BY_ID["systolic_int8_ws"]
    assert systolic.engine_kind == "systolic"
    assert set(systolic.supported_kernels) == {"gemm", "conv2d"}
    assert systolic.is_programmable


def test_a_fixed_function_engine_is_quoted_in_its_own_work_unit():
    """Pixels are what it is published in, and it has no format choice."""
    sgm = BY_ID["ff_stereo_sgm"]
    assert not sgm.is_programmable
    assert sgm.function_id == "stereo.sgm"
    assert sgm.work_unit == "pixel"
    assert sgm.units_per_second > 0 and sgm.pj_per_unit > 0
    assert sgm.formats == ()
    assert sgm.precision_floor is None
    assert "no precision floor" in sgm.notes
    assert "ISSCC 2017" in sgm.provenance


def test_a_class_whose_ops_are_not_precision_keyed_says_so():
    """The min-plus fabric does real work a precision-indexed mapper cannot
    see. Reporting nothing with an explanation beats reporting a zero."""
    minplus = BY_ID["pe_minplus_i16"]
    assert minplus.is_programmable
    assert minplus.formats == ()
    assert minplus.precision_floor is None
    assert "not precision-keyed" in minplus.notes


def test_engines_report_the_silicon_they_cost():
    systolic = BY_ID["systolic_int8_ws"]
    assert systolic.area_mm2_per_tile > 0
    assert systolic.total_area_mm2 == pytest.approx(
        systolic.area_mm2_per_tile * systolic.num_tiles
    )
    # And the multi-site footprint travels with it, for a floorplanner.
    assert systolic.sites_per_tile == 2
    assert BY_ID["ff_vio_stereo_inertial"].sites_per_tile == 4


# ---------------------------------------------------------------------------
# Precision floors
# ---------------------------------------------------------------------------


def test_the_floor_is_the_narrowest_format_an_engine_carries():
    assert BY_ID["pe_int8_mac_i32"].precision_floor == "int4"
    assert BY_ID["systolic_int8_ws"].precision_floor == "int8"
    assert BY_ID["pe_lns16_mac"].precision_floor == "lns8"


def test_range_is_what_decides_not_speed():
    """Being fast at a narrow format does not make a wide stage runnable."""
    systolic = BY_ID["systolic_int8_ws"]
    assert systolic.supports_at_least("int8")
    assert not systolic.supports_at_least("fp32")
    assert not systolic.supports_at_least("bf16")
    # The INT8 fabric also carries bf16, so it does serve a 16-bit stage.
    assert BY_ID["pe_int8_mac_i32"].supports_at_least("bf16")


def test_an_lns_format_ranks_with_the_float_it_replaces():
    """lns16 carries roughly bf16's dynamic range, which is the whole point
    of the class, so a 16-bit stage can land on it."""
    assert format_rank("lns16") == format_rank("bf16")
    assert format_rank("lns8") == format_rank("int8")
    assert BY_ID["pe_lns16_mac"].supports_at_least("fp16")


def test_a_requirement_nothing_serves_is_an_error_naming_the_floors():
    findings = precision_floor_findings(ENGINES, {"planning_qp": "fp32"})
    assert [f.severity for f in findings] == ["ERROR"]
    assert findings[0].served_by == ()
    message = findings[0].message
    assert "fp32" in message and "no programmable engine" in message
    # It says what the die does carry, so the reader can act on it.
    assert "int4" in message and "int8" in message


def test_a_requirement_is_served_only_by_programmable_engines():
    """A fixed-function core runs its own function, not arbitrary stages, so
    it never counts as serving a precision requirement."""
    findings = precision_floor_findings(ENGINES, {"perception_trunk": "int8"})
    assert findings[0].severity == "OK"
    assert set(findings[0].served_by) == {
        "pe_int8_mac_i32", "pe_lns16_mac", "systolic_int8_ws"
    }
    assert not any(f.startswith("ff_") for f in findings[0].served_by)


def test_the_uniform_t64_serves_fp32_where_the_heterogeneous_die_does_not():
    """The trade this refactor exists to measure, in one assertion: the
    heterogeneous die spends its silicon on min-plus, LNS and systolic
    classes and loses fp32 entirely, so a QP solver that runs on the
    uniform T64 has nowhere to go on kpu_h64_auto1.

    Comparing the two on the five operating regimes needs graphs#269's
    workload and its data annex; this is the part reachable today.
    """
    requirements = {"planning_qp": "fp32", "perception_trunk": "int8"}
    hetero = {f.requirement: f for f in precision_floor_findings(ENGINES, requirements)}
    uniform = {
        f.requirement: f
        for f in precision_floor_findings(
            describe_engines(T64, N16, NODES), requirements
        )
    }
    assert hetero["planning_qp"].severity == "ERROR"
    assert uniform["planning_qp"].severity == "OK"
    assert uniform["planning_qp"].served_by == ("bf16_primary",)
    # Both serve the perception trunk; that is not what was traded away.
    assert hetero["perception_trunk"].severity == "OK"
    assert uniform["perception_trunk"].severity == "OK"


# ---------------------------------------------------------------------------
# Segments
# ---------------------------------------------------------------------------


def test_a_stream_link_becomes_a_segment():
    (segment,) = describe_segments(HETERO)
    assert segment.segment_id == "isp_sgm_vio"
    assert segment.num_stages == 3
    assert segment.engine_ids == (
        "ff_isp_raw2yuv", "ff_stereo_sgm", "ff_vio_stereo_inertial"
    )
    assert segment.function_ids == (
        "isp.raw_to_yuv", "stereo.sgm", "vio.stereo_inertial"
    )


def test_absorbed_traffic_is_keyed_by_the_producers_work_unit():
    """A chain can mix per-pixel and per-frame stages, and adding those
    byte counts together would be meaningless. Only producers contribute:
    the last engine's output leaves the chain."""
    (segment,) = describe_segments(HETERO)
    # ISP 1.5 B/px and SGM 2 B/px, each written and read back.
    assert segment.absorbed_bytes_by_unit == (("pixel", 7.0),)
    assert segment.work_units == ("pixel",)
    assert segment.absorbed_bytes("pixel") == 7.0
    assert segment.absorbed_bytes("frame") == 0.0


def test_a_uniform_sku_declares_no_segments():
    assert describe_segments(T64) == ()


# ---------------------------------------------------------------------------
# Legacy SKUs still describe
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sku_id", sorted(
    k for k, v in load_compute_products().items() if has_kpu_block(v)
))
def test_every_catalog_sku_describes_without_error(sku_id):
    """The export must work on a uniform KPU too: the SoC study composes
    those as engines as readily as a heterogeneous one."""
    cp = load_compute_products()[sku_id]
    node = NODES[cp.dies[0].process_node_id]
    engines = describe_engines(cp, node, NODES)
    assert engines
    assert all(e.engine_kind == "pe_fabric" for e in engines)
    assert all(e.formats for e in engines)
