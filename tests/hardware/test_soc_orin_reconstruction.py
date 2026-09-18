"""The Orin-class reference design: Phase 2's acceptance check (graphs#269).

Three criteria from the parent plan, and where each stands:

1. **Area within 15% of 455 mm^2 / 17 B -- open, by decision.** Most of the
   design's silicon has no credible public area anchor (no per-SM, per-core,
   PHY or IO figure), so those lines are declared unanchored and the
   composition is a lower bound. Of the target, 17 B is NVIDIA's own figure;
   455 mm^2 is one third-party teardown. Filling the gaps with
   estimates chosen after seeing 455 would make the check circular, so it
   stays open until anchors exist; these tests pin that the composition
   *says* it is incomplete rather than presenting a number.
2. **GPU and DLA peak INT8 TOPS match the datasheet -- passes**, from
   NVIDIA's own figures.
3. **Monotonic scaling at N7 and N5 -- passes** on everything that is priced,
   with the per-op energy check labelled ALU-only (decision P2-D6).

See ``docs/plans/soc-phase2-execution-plan.md``.
"""

from __future__ import annotations

import pytest
from embodied_schemas import load_process_nodes
from embodied_schemas.process_node import CircuitClass

from graphs.hardware.soc import EngineKind, compose_soc, load_designs, load_ip_library

NODES = load_process_nodes()
LIBRARY = load_ip_library()
DESIGN = load_designs()["orin_class_reference"]
NODE_ORDER = ("samsung_8lpp", "tsmc_n7", "tsmc_n5")


@pytest.fixture(scope="module")
def socs():
    return {n: compose_soc(DESIGN, LIBRARY, NODES, n) for n in NODE_ORDER}


# ---------------------------------------------------------------------------
# Criterion 1: the area check is open, and the composition says so
# ---------------------------------------------------------------------------


def test_the_reconstruction_declares_itself_incomplete(socs):
    soc = socs["samsung_8lpp"]
    assert not soc.complete
    gap_blocks = {block for block, _ in soc.gaps}
    # The load-bearing silicon of the die has no public anchor.
    assert {"gpu_sm", "cpu", "dla", "pva", "memory", "io"} <= gap_blocks
    # A lower bound, and far from the target -- which is why it is not
    # compared as a pass/fail.
    assert soc.die_area_mm2 < 0.1 * DESIGN.reference.die_area_mm2
    assert soc.transistors_billion < 0.2 * DESIGN.reference.transistors_billion


def test_every_gap_says_what_is_missing():
    """An unanchored line is a stated gap, not a silent zero."""
    for template in LIBRARY.values():
        for line in template.unanchored_lines:
            assert len(line.source) > 40, (template.id, line.name)
            assert line.confidence.value == "unknown"


def test_each_target_figure_carries_its_own_provenance():
    """17 B is NVIDIA's own figure (the DRIVE AGX Orin press release, 17
    December 2019). 455 mm^2 is not: NVIDIA never states the die area, and it
    traces to one third-party teardown. Conflating the two -- as this design
    first did -- overstates how weak the transistor target is and understates
    how weak the area target is (CodeRabbit on #303)."""
    source = DESIGN.reference.source
    transistors, area = source.split("Die area:", 1)
    assert "NVIDIA" in transistors and "17 billion transistors" in transistors
    assert "17 December 2019" in transistors
    assert "NOT an NVIDIA figure" in area and "TechAnaLye" in area


def test_what_is_priced_is_structural_sram_from_nvidia_capacities(socs):
    """The priced lines are the SRAM whose capacities NVIDIA publishes,
    priced as 6T cells: the part of the die that can be derived rather than
    guessed."""
    soc = socs["samsung_8lpp"]
    classes = soc.area_by_class()
    assert set(classes) == {"sram_hd"}
    kib = (16 * (192 + 256)          # per-SM L1/shared + register file
           + 4096                    # GPU L2
           + 3 * (4 * 128 + 4 * 256 + 2048)  # CPU clusters
           + 1024 + 2 * 128          # PVA
           + 4096                    # system cache
           + 3072)                   # safety island
    assert soc.transistors_billion == pytest.approx(kib * 0.052 / 1e3, rel=1e-9)


# ---------------------------------------------------------------------------
# Criterion 2: peak TOPS, from NVIDIA's own figures
# ---------------------------------------------------------------------------


def test_gpu_int8_dense_tops_matches_the_datasheet(socs):
    """NVIDIA: 275 INT8 sparse TOPS combined, 105 on the DLAs, so 170 on the
    GPU sparse and 85 dense. 16 SM x 4096 ops x 1.3 GHz = 85.2."""
    got = socs["samsung_8lpp"].peak_tops("int8", EngineKind.GPU)
    assert got == pytest.approx(16 * 4096 * 1.3e9 / 1e12)
    assert got == pytest.approx(DESIGN.reference.peak_tops["int8_gpu_dense"], rel=0.01)


def test_dla_int8_dense_tops_matches_the_datasheet(socs):
    """NVIDIA: up to 105 INT8 sparse TOPS on the two DLAs, 52.5 dense.
    2 x 16384 ops x 1.6 GHz = 52.4."""
    got = socs["samsung_8lpp"].peak_tops("int8", EngineKind.NPU)
    assert got == pytest.approx(2 * 16384 * 1.6e9 / 1e12)
    assert got == pytest.approx(DESIGN.reference.peak_tops["int8_dla_dense"], rel=0.01)


def test_sparse_totals_reproduce_the_275_tops_headline(socs):
    """The two dense peaks, doubled for 2:4 sparsity, give NVIDIA's 275."""
    soc = socs["samsung_8lpp"]
    dense = soc.peak_tops("int8", EngineKind.GPU) + soc.peak_tops("int8", EngineKind.NPU)
    assert 2 * dense == pytest.approx(275, rel=0.01)


# ---------------------------------------------------------------------------
# Criterion 3: scaling across nodes
# ---------------------------------------------------------------------------


def test_priced_area_falls_monotonically_with_the_node(socs):
    areas = [socs[n].die_area_mm2 for n in NODE_ORDER]
    assert areas[0] > areas[1] > areas[2]
    # Transistors do not change with the node; only their area does.
    counts = {round(socs[n].transistors_billion, 9) for n in NODE_ORDER}
    assert len(counts) == 1


def test_io_and_analog_scale_worse_than_logic():
    """Pinned on the node data the composition prices through: logic density
    roughly triples from 8LPP to N5, analog and IO rise by well under 2x. A
    PHY-heavy block therefore takes a growing share of a shrinking die."""
    def ratio(cc):
        return (NODES["tsmc_n5"].density_for(cc).mtx_per_mm2
                / NODES["samsung_8lpp"].density_for(cc).mtx_per_mm2)
    logic = ratio(CircuitClass.HP_LOGIC)
    assert logic > 2.5
    for cc in (CircuitClass.ANALOG, CircuitClass.IO):
        assert ratio(cc) < 2.0 and ratio(cc) < logic / 1.5, cc


@pytest.mark.parametrize("key", ["hp_logic:int8", "hp_logic:fp32", "balanced_logic:int8"])
def test_per_op_energy_falls_monotonically(key):
    """ALU-only energy per op from the node table (decision P2-D6): the
    Phase 3 power model adds the architectural overhead that makes a CPU or
    GPU cost 10-50x more per useful op, so this is a floor, not a power."""
    values = [NODES[n].energy_per_op_pj[key] for n in NODE_ORDER]
    assert values[0] > values[1] > values[2]


def test_reference_clocks_are_flagged_off_their_node(socs):
    """Orin's clocks were characterized on 8LPP. At N7 and N5 the peaks above
    are provisional until PR 2.3 retargets fmax, and the instance says so."""
    assert socs["samsung_8lpp"].off_reference_clocks == ()
    assert set(socs["tsmc_n5"].off_reference_clocks) == {"gpu_sm", "dla", "cpu"}


# ---------------------------------------------------------------------------
# The anchored references compose too
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ip_id, node, area", [
    ("nvdla_v1_large", "tsmc_n16", 3.3),
    ("isp_dsc_12bit_65nm", "tsmc_n65", 1.5),
    ("hevc_encoder_4k30_28nm", "tsmc_n28hpm", 1.96),
])
def test_a_published_area_round_trips_at_its_node(ip_id, node, area):
    """The three real anchors the research found are in the library as their
    own IPs, not passed off as Orin's blocks, and price back to their
    published areas on their own nodes."""
    template = LIBRARY[ip_id]
    assert not template.unanchored_lines
    got = sum(
        line.transistors_mtx(NODES) / NODES[node].density_for(line.circuit_class).mtx_per_mm2
        for line in template.silicon
    )
    assert got == pytest.approx(area)


def test_nvdlas_buffer_retargets_as_sram_not_logic():
    """The Primer's 3.3 mm^2 includes the 512 KB convolution buffer. Priced as
    one logic line, the buffer would retarget at logic density; split out, it
    follows SRAM density (CodeRabbit on #303).

    The direction depends on the node pair and is not assumed: in this
    catalog, SRAM density rises 6.3x from N16 to N5 and balanced logic 6.1x,
    so the split DLA comes out slightly smaller at N5 than the all-logic one.
    What the test pins is that the two are priced differently at all."""
    lines = {line.name: line for line in LIBRARY["nvdla_v1_large"].silicon}
    assert lines["conv_buffer"].circuit_class == CircuitClass.SRAM_HD
    assert lines["conv_buffer"].mtx == pytest.approx(512 * 0.052)
    assert lines["dla_logic"].circuit_class == CircuitClass.BALANCED_LOGIC

    def area_at(node, lines_):
        return sum(
            l.transistors_mtx(NODES) / NODES[node].density_for(l.circuit_class).mtx_per_mm2
            for l in lines_
        )
    split = area_at("tsmc_n5", lines.values())
    as_one_logic_line = 3.3 * (
        NODES["tsmc_n16"].density_for(CircuitClass.BALANCED_LOGIC).mtx_per_mm2
        / NODES["tsmc_n5"].density_for(CircuitClass.BALANCED_LOGIC).mtx_per_mm2
    )
    assert area_at("tsmc_n16", lines.values()) == pytest.approx(3.3)
    assert split != pytest.approx(as_one_logic_line, rel=1e-3)
    assert split < as_one_logic_line  # this catalog's N16 -> N5 densities
