"""SoC composition machinery (graphs#269 PRs 2.1 and 2.2).

These tests use synthetic IP so they pin the arithmetic, not the library's
numbers: an area stated at a reference node round-trips, retargeting scales
each library by its own density ratio, the die roll-up is what the docstring
says, and ``silicon_math`` prices a product without a KPU block through the
same formula as one with.

The real IP library and the Orin reconstruction are checked in
``test_soc_orin_reconstruction.py``.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import yaml
from embodied_schemas import load_compute_products, load_process_nodes
from embodied_schemas.process_node import CircuitClass
from pydantic import ValidationError

from graphs.hardware.soc import (
    DesignBlock,
    EngineKind,
    IPBlockTemplate,
    IPSilicon,
    Layout,
    SoCDesign,
    compose_soc,
    load_designs,
    load_ip_library,
)
from graphs.hardware.sku_validators import silicon_math as sm

NODES = load_process_nodes()
N8, N7, N5 = NODES["samsung_8lpp"], NODES["tsmc_n7"], NODES["tsmc_n5"]


def _ip(ip_id: str, lines, **extra) -> IPBlockTemplate:
    return IPBlockTemplate.model_validate({
        "id": ip_id, "name": ip_id, "silicon": lines, **extra,
    })


def _design(blocks, **layout) -> SoCDesign:
    return SoCDesign(
        id="d", name="d", process_node="samsung_8lpp",
        blocks=[DesignBlock(**b) for b in blocks],
        layout=Layout(**layout),
    )


LOGIC_10MM2 = {"name": "logic", "circuit_class": "hp_logic", "reference_area_mm2": 10.0,
               "reference_node": "samsung_8lpp", "source": "synthetic"}
PHY_10MM2 = {"name": "phy", "circuit_class": "analog", "reference_area_mm2": 10.0,
             "reference_node": "samsung_8lpp", "source": "synthetic"}


# ---------------------------------------------------------------------------
# The schema
# ---------------------------------------------------------------------------


def test_a_silicon_line_states_exactly_one_form():
    with pytest.raises(ValidationError, match="exactly one"):
        IPSilicon(name="x", circuit_class=CircuitClass.HP_LOGIC, source="s")
    with pytest.raises(ValidationError, match="exactly one"):
        IPSilicon(name="x", circuit_class=CircuitClass.HP_LOGIC, mtx=1.0,
                  reference_area_mm2=1.0, reference_node="samsung_8lpp", source="s")
    with pytest.raises(ValidationError, match="needs the reference_node"):
        IPSilicon(name="x", circuit_class=CircuitClass.HP_LOGIC,
                  reference_area_mm2=1.0, source="s")
    with pytest.raises(ValidationError, match="only qualifies"):
        IPSilicon(name="x", circuit_class=CircuitClass.HP_LOGIC, mtx=1.0,
                  reference_node="samsung_8lpp", source="s")


def test_every_figure_needs_a_source():
    """Third-party budgets are not public, so an unsourced number would be a
    guess presented as data."""
    with pytest.raises(ValidationError):
        IPSilicon(name="x", circuit_class=CircuitClass.HP_LOGIC, mtx=1.0, source="")


def test_compute_must_agree_with_the_block_kind():
    with pytest.raises(ValidationError, match="disagrees"):
        _ip("x", [LOGIC_10MM2], engine_kind="gpu", compute={
            "engine_kind": "cpu", "units": 4, "unit_name": "core", "source": "s",
        })


def test_a_design_names_only_known_ip():
    design = _design([{"instance": "a", "ip": "nope"}])
    with pytest.raises(KeyError, match="unknown IP"):
        design.check_against({})


def test_design_instances_are_unique():
    with pytest.raises(ValidationError, match="duplicate block instances"):
        _design([{"instance": "a", "ip": "x"}, {"instance": "a", "ip": "y"}])


# ---------------------------------------------------------------------------
# Area and retargeting
# ---------------------------------------------------------------------------


def test_an_area_round_trips_at_its_reference_node():
    """10 mm^2 measured at 8LPP is 10 mm^2 when composed at 8LPP."""
    lib = {"logic": _ip("logic", [LOGIC_10MM2]), "phy": _ip("phy", [PHY_10MM2])}
    soc = compose_soc(_design([{"instance": "l", "ip": "logic"}, {"instance": "p", "ip": "phy"}]),
                      lib, NODES)
    assert [b.area_mm2 for b in soc.blocks] == pytest.approx([10.0, 10.0], rel=1e-12)


def test_each_library_retargets_by_its_own_density_ratio():
    """The reason areas are stated per library: logic shrinks with the node,
    analog barely does, so a PHY-heavy design gets relatively bigger."""
    lib = {"logic": _ip("logic", [LOGIC_10MM2]), "phy": _ip("phy", [PHY_10MM2])}
    design = _design([{"instance": "l", "ip": "logic"}, {"instance": "p", "ip": "phy"}])
    by_node = {n: compose_soc(design, lib, NODES, n) for n in ("samsung_8lpp", "tsmc_n7", "tsmc_n5")}
    for node_id, soc in by_node.items():
        node = NODES[node_id]
        logic, phy = soc.blocks
        assert logic.area_mm2 == pytest.approx(
            10.0 * N8.density_for(CircuitClass.HP_LOGIC).mtx_per_mm2
            / node.density_for(CircuitClass.HP_LOGIC).mtx_per_mm2)
        assert phy.area_mm2 == pytest.approx(
            10.0 * N8.density_for(CircuitClass.ANALOG).mtx_per_mm2
            / node.density_for(CircuitClass.ANALOG).mtx_per_mm2)
    # Logic falls about 3x from 8LPP to N5; analog well under 2x.
    logic_shrink = by_node["samsung_8lpp"].blocks[0].area_mm2 / by_node["tsmc_n5"].blocks[0].area_mm2
    phy_shrink = by_node["samsung_8lpp"].blocks[1].area_mm2 / by_node["tsmc_n5"].blocks[1].area_mm2
    assert logic_shrink > 2.5
    assert phy_shrink < 2.0
    assert logic_shrink > phy_shrink


def test_count_multiplies_area_and_transistors():
    lib = {"logic": _ip("logic", [LOGIC_10MM2])}
    one = compose_soc(_design([{"instance": "l", "ip": "logic"}]), lib, NODES)
    four = compose_soc(_design([{"instance": "l", "ip": "logic", "count": 4}]), lib, NODES)
    assert four.blocks[0].area_mm2 == pytest.approx(4 * one.blocks[0].area_mm2)
    assert four.transistors_billion == pytest.approx(4 * one.transistors_billion)


def test_the_die_roll_up():
    """Core = blocks x (1 + whitespace); a square die adds the IO ring on
    both sides."""
    lib = {"logic": _ip("logic", [LOGIC_10MM2])}
    soc = compose_soc(
        _design([{"instance": "l", "ip": "logic", "count": 10}],
                whitespace_fraction=0.2, io_ring_mm=0.5),
        lib, NODES,
    )
    assert soc.block_area_mm2 == pytest.approx(100.0)
    assert soc.core_area_mm2 == pytest.approx(120.0)
    assert soc.die_side_mm == pytest.approx(math.sqrt(120.0) + 1.0)
    assert soc.die_area_mm2 == pytest.approx((math.sqrt(120.0) + 1.0) ** 2)
    assert soc.io_ring_area_mm2 == pytest.approx(soc.die_area_mm2 - 120.0)


def test_a_library_the_target_node_lacks_is_an_error():
    """sram_hp exists at N7 but not at 8LPP: a design using it cannot be
    priced there, and says so rather than dropping the area."""
    lib = {"x": _ip("x", [{"name": "hp", "circuit_class": "sram_hp", "mtx": 10.0, "source": "s"}])}
    with pytest.raises(sm.SiliconMathError, match="does not offer library"):
        compose_soc(_design([{"instance": "x", "ip": "x"}]), lib, NODES, "samsung_8lpp")


# ---------------------------------------------------------------------------
# Peaks and clocks
# ---------------------------------------------------------------------------


def _engine(clock_node="samsung_8lpp"):
    return _ip("gpu", [LOGIC_10MM2], engine_kind="gpu",
               compute={"engine_kind": "gpu", "units": 1, "unit_name": "SM",
                        "ops_per_clock": {"int8": 4096.0}, "source": "s"},
               clock={"fmax_ghz_ref": 1.3, "reference_node": clock_node, "source": "s"})


def test_peak_is_count_times_ops_per_clock_times_clock():
    soc = compose_soc(_design([{"instance": "g", "ip": "gpu", "count": 16}]),
                      {"gpu": _engine()}, NODES)
    assert soc.peak_ops_per_s("int8") == pytest.approx(16 * 4096 * 1.3e9)
    assert soc.peak_tops("int8", EngineKind.GPU) == pytest.approx(85.1968)
    assert soc.peak_tops("int8", EngineKind.NPU) == 0.0
    assert soc.peak_tops("fp64") == 0.0


def test_an_explicit_clock_wins():
    soc = compose_soc(_design([{"instance": "g", "ip": "gpu", "clock_ghz": 1.0}]),
                      {"gpu": _engine()}, NODES)
    assert soc.peak_ops_per_s("int8") == pytest.approx(4096 * 1.0e9)
    assert soc.off_reference_clocks == ()


def test_a_reference_clock_off_its_node_is_flagged():
    """fmax is characterized on one node. Until PR 2.3 retargets it, a peak
    computed elsewhere is provisional, and the instance says which."""
    design = _design([{"instance": "g", "ip": "gpu"}])
    assert compose_soc(design, {"gpu": _engine()}, NODES, "samsung_8lpp").off_reference_clocks == ()
    assert compose_soc(design, {"gpu": _engine()}, NODES, "tsmc_n5").off_reference_clocks == ("g",)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def test_loader_requires_id_to_match_the_file(tmp_path: Path):
    (tmp_path / "wrong_name.yaml").write_text(yaml.safe_dump(
        {"id": "logic", "name": "x", "silicon": [LOGIC_10MM2]}))
    with pytest.raises(ValueError, match="must match the file name"):
        load_ip_library(tmp_path)


def test_the_shipped_library_and_designs_load():
    library = load_ip_library()
    designs = load_designs()
    assert library, "soc_designs/ip is empty"
    assert designs, "soc_designs/designs is empty"
    for design in designs.values():
        design.check_against(library)


# ---------------------------------------------------------------------------
# silicon_math generalized
# ---------------------------------------------------------------------------


def test_one_area_formula_for_both_paths():
    """resolve_block_area (the KPU path) and area_for (composition) are the
    same arithmetic, so the two cannot disagree on how area scales."""
    cp = load_compute_products()["kpu_t64_32x32_lp5x4_16nm_tsmc_ffp"]
    node = NODES["tsmc_n16"]
    for block in cp.dies[0].silicon_bin.blocks:
        kpu = sm.resolve_block_area(block, cp, node)
        shared = sm.area_for(block.name, kpu.transistors_mtx, block.circuit_class, node)
        assert shared == kpu


def test_a_product_without_a_kpu_block_is_priced_by_the_same_code():
    """The catalog's Jetson Orin is a GPU module whose silicon_bin is all
    `fixed` lines. Before the generalization, pricing it raised because it
    has no KPU block."""
    cp = load_compute_products()["nvidia_jetson_agx_orin_64gb"]
    areas = sm.resolve_all_block_areas(cp, NODES["samsung_8lpp"])
    assert len(areas) == len(cp.dies[0].silicon_bin.blocks)
    assert sm.silicon_die(cp) is cp.dies[0]
    by_die = sm.resolve_product_block_areas(cp, NODES)
    assert [a for _, a in by_die] == areas


def test_the_kpu_die_is_still_the_kpu_die():
    cp = load_compute_products()["kpu_t64_32x32_lp5x4_16nm_tsmc_ffp"]
    assert sm.silicon_die(cp) is sm._kpu_die(cp)
