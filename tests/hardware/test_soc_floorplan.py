"""The floorplan view of a composed die (graphs#269 Phase 7).

Area, transistors, gates and SRAM capacity all come out of one figure per
line, so these tests check the arithmetic in both directions and check
that a line with no figure stays a gap rather than becoming a zero.
"""

from __future__ import annotations

import pytest
from embodied_schemas import load_process_nodes

from graphs.hardware.sku_validators.silicon_math import SRAM_MTX_PER_KIB
from graphs.hardware.soc import compose_soc, load_designs, load_ip_library
from graphs.hardware.soc.floorplan import (
    LOGIC_CLASSES,
    NAND2_TRANSISTORS,
    compose_resources,
    die_area_for,
    fit_scaling_law,
)

FAMILY = ("kpu_t64_n7", "kpu_t128_n7", "kpu_t256_n7", "kpu_t512_n7")
TILES = {"kpu_t64_n7": 64, "kpu_t128_n7": 128, "kpu_t256_n7": 256, "kpu_t512_n7": 512}


@pytest.fixture(scope="module")
def composed():
    lib, nodes, designs = load_ip_library(), load_process_nodes(), load_designs()
    return {name: compose_soc(designs[name], lib, nodes, None) for name in FAMILY}


@pytest.fixture(scope="module")
def comp(composed):
    return compose_resources(composed["kpu_t128_n7"])


def test_every_area_is_its_own_library_density_divided_into_its_transistors(comp):
    """One division, per line, with the density of the library that line
    is built in. A single die-wide density would put the SRAM and the
    logic on the same curve, and they are a factor of three apart."""
    for block in comp.blocks:
        for line in block.anchored_lines:
            assert line.mtx_per_mm2, line.name
            assert line.area_mm2 == pytest.approx(
                line.transistors_mtx / line.mtx_per_mm2), f"{block.name}.{line.name}"
    densities = {c.circuit_class: c.mtx_per_mm2 for c in comp.classes}
    assert len(set(densities.values())) > 1


def test_sram_capacity_comes_back_out_of_the_transistors(comp):
    """The catalogue builds an SRAM line at a stated Mtx per KiB, so the
    capacity is recoverable. The figures below are the ones the templates
    cite from the platform's technical brief."""
    blocks = {b.name: b for b in comp.blocks}
    lines = {ln.name: ln for ln in blocks["kpu"].lines}
    assert lines["l3_sram"].sram_kib == pytest.approx(32 * 1024)
    assert lines["l2_sram"].sram_kib == pytest.approx(4 * 1024)
    # Three clusters of four cores: 512 KiB of L1 and 2 MiB of L3 each.
    cpu = {ln.name: ln for ln in blocks["cpu"].lines}
    assert cpu["l1_caches"].sram_kib == pytest.approx(3 * 512)
    assert cpu["l3_cache"].sram_kib == pytest.approx(3 * 2048)
    assert blocks["system_cache"].sram_kib == pytest.approx(4 * 1024)
    assert comp.sram_kib == pytest.approx(
        sum(b.sram_kib for b in comp.blocks))


def test_a_gate_is_four_transistors_and_only_logic_has_one(comp):
    """No gate count exists anywhere in this repository. The figure is the
    transistor count restated at the NAND2 convention, and it is only
    meaningful for a standard-cell line."""
    assert NAND2_TRANSISTORS == 4
    seen = 0
    for block in comp.blocks:
        for line in block.lines:
            if line.circuit_class in LOGIC_CLASSES and line.transistors_mtx:
                assert line.gates_m == pytest.approx(line.transistors_mtx / 4)
                assert line.sram_kib is None
                seen += 1
            elif line.circuit_class in SRAM_MTX_PER_KIB:
                assert line.gates_m is None, line.name
    assert seen


def test_a_line_with_no_figure_stays_a_gap_and_never_becomes_a_zero(comp):
    """The CPU's core logic is the one that matters: filling it with a
    guess would turn the only honest column on the page into a number."""
    cpu = next(b for b in comp.blocks if b.name == "cpu")
    core_logic = next(ln for ln in cpu.lines if ln.name == "core_logic")
    assert core_logic.transistors_mtx is None
    assert core_logic.area_mm2 is None
    assert not core_logic.anchored
    assert core_logic in cpu.gaps
    assert cpu.area_mm2 == pytest.approx(sum(ln.area_mm2 for ln in cpu.anchored_lines))
    assert not comp.complete
    # A block with nothing anchored takes no area at all, and is listed.
    assert {b.name for b in comp.gap_blocks} == {"isp", "codec", "memory", "io", "fabric"}
    for block in comp.gap_blocks:
        assert block.area_mm2 == 0.0


def test_the_composition_totals_are_the_die_compose_soc_built(comp, composed):
    """Taking the die apart must not change it."""
    soc = composed["kpu_t128_n7"]
    assert comp.block_area_mm2 == pytest.approx(soc.block_area_mm2)
    assert comp.core_area_mm2 == pytest.approx(soc.core_area_mm2)
    assert comp.die_area_mm2 == pytest.approx(soc.die_area_mm2)
    assert comp.transistors_mtx == pytest.approx(soc.transistors_billion * 1e3)
    assert comp.whitespace_mm2 == pytest.approx(
        soc.block_area_mm2 * soc.design.layout.whitespace_fraction)
    by_class = soc.area_by_class()
    for entry in comp.classes:
        assert entry.area_mm2 == pytest.approx(by_class[entry.circuit_class.value])


def test_the_roll_up_applied_to_a_die_it_did_not_build(comp, composed):
    """``die_area_for`` is the same square-die arithmetic, used on a block
    area no catalogued design has. It has to agree on one that does."""
    soc = composed["kpu_t128_n7"]
    assert die_area_for(soc.block_area_mm2, soc.design.layout) == pytest.approx(
        soc.die_area_mm2)


def test_the_scaling_law_reproduces_the_designs_it_was_fitted_to(composed):
    """A per-tile slope is only worth quoting if it lands on the family it
    came from. It is still an extrapolation off the ends, and says so."""
    law = fit_scaling_law([(n, composed[n], TILES[n]) for n in FAMILY])
    assert law is not None
    for name in FAMILY:
        assert law.die_area_mm2(TILES[name], law.cores_per_cluster) == pytest.approx(
            composed[name].die_area_mm2, rel=0.01), name
    assert law.tiles_fitted == (64, 512)
    assert law.extrapolates(1) == "below"
    assert law.extrapolates(128) == "within"
    assert law.extrapolates(2000) == "above"
    # Smallest fabric first, so a page listing the family reads in order.
    assert law.designs == FAMILY
    assert [t for t, _ in law.points] == [64, 128, 256, 512]
    # The blocks that price at nothing are named, because they are the
    # fixed area the law is missing.
    assert set(law.unpriced_blocks) == {"isp", "codec", "memory", "io", "fabric"}


def test_a_family_that_varies_more_than_its_fabric_is_refused(composed):
    """The slope would absorb whatever else changed. Two nodes in one
    family is the easy case: the CPU cluster is a different size."""
    lib, nodes, designs = load_ip_library(), load_process_nodes(), load_designs()
    other = compose_soc(designs["kpu_t256_n16"], lib, nodes, None)
    mixed = [("kpu_t128_n7", composed["kpu_t128_n7"], 128),
             ("kpu_t256_n16", other, 256)]
    assert fit_scaling_law(mixed) is None


def test_one_point_is_not_a_line(composed):
    assert fit_scaling_law([("kpu_t128_n7", composed["kpu_t128_n7"], 128)]) is None
