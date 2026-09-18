"""Retargeting rated clocks across nodes (graphs#269 PR 2.3).

Clocks move only along foundry-stated speed relations whose source states
iso-power, carried as ranges when the foundry's own documents disagree, and refuse to move where
no relation exists. See ``hardware/soc/clocking.py`` for why the parent plan's
alpha-power law in Vdd is not modeled.
"""

from __future__ import annotations

import pytest
from embodied_schemas import load_process_nodes
from pydantic import ValidationError

from graphs.hardware.soc import (
    DesignBlock,
    IPBlockTemplate,
    Layout,
    SoCDesign,
    compose_soc,
    load_node_speed,
    retarget_fmax,
)
from graphs.hardware.soc.clocking import NodeSpeedTable

NODES = load_process_nodes()
TABLE = load_node_speed()


def test_same_node_is_the_identity():
    clock = retarget_fmax(1.3, "tsmc_n7", "tsmc_n7", TABLE)
    assert (clock.low_ghz, clock.high_ghz, clock.path, clock.sources) == (1.3, 1.3, ("tsmc_n7",), ())


def test_a_stated_relation_carries_its_range():
    """TSMC states ~35% (one document) and ~30% (another) from N16 to N7 at
    iso-power. Both ends survive, and the quoted clock is the lower."""
    clock = retarget_fmax(1.0, "tsmc_n16", "tsmc_n7", TABLE)
    assert (clock.low_ghz, clock.high_ghz) == pytest.approx((1.30, 1.35))
    assert clock.ghz == clock.low_ghz
    assert len(clock.sources) == 1 and "TSMC" in clock.sources[0]


def _table(*relations) -> NodeSpeedTable:
    return NodeSpeedTable.model_validate({"relations": [
        {"from_node": a, "to_node": b, "speed_ratio": {"low": lo, "high": hi},
         "power_condition": cond, "source": "test"}
        for a, b, lo, hi, cond in relations
    ]})


CHAIN = _table(("n1", "n2", 1.30, 1.35, "iso_power"), ("n2", "n3", 1.15, 1.15, "iso_power"))


def test_relations_chain_and_compound():
    clock = retarget_fmax(1.0, "n1", "n3", CHAIN)
    assert clock.path == ("n1", "n2", "n3")
    assert (clock.low_ghz, clock.high_ghz) == pytest.approx((1.30 * 1.15, 1.35 * 1.15))
    assert len(clock.sources) == 2


def test_going_backwards_inverts_and_swaps_the_range():
    down = retarget_fmax(1.0, "n3", "n1", CHAIN)
    assert (down.low_ghz, down.high_ghz) == pytest.approx(
        (1 / (1.35 * 1.15), 1 / (1.30 * 1.15))
    )
    # Back up from the conservative end: the top of the range recovers the
    # original clock, the bottom falls short of it by the sources'
    # disagreement. Uncertainty compounds; it does not cancel on a round trip.
    up = retarget_fmax(down.low_ghz, "n1", "n3", CHAIN)
    assert (up.low_ghz, up.high_ghz) == pytest.approx((1.30 / 1.35, 1.0))


def test_a_relation_without_a_stated_power_condition_is_not_used():
    """A speed gain that may be bought with more power is not the clock the
    block reaches in its budget, so it cannot carry a clock -- here, or as a
    link in a chain."""
    table = _table(("n1", "n2", 1.3, 1.3, "iso_power"), ("n2", "n3", 1.15, 1.15, "unstated"))
    assert retarget_fmax(1.0, "n1", "n2", table) is not None
    assert retarget_fmax(1.0, "n2", "n3", table) is None
    assert retarget_fmax(1.0, "n1", "n3", table) is None


def test_tsmcs_n5_and_n4p_gains_are_recorded_but_unstated():
    """TSMC's own text gives 15% (N7 -> N5) and 11% (N5 -> N4P) with no power
    condition (CodeRabbit on #304), so they stay in the table as data and a
    clock does not move along them. Only N16 -> N7 states "at the same power"."""
    by_pair = {(r.from_node, r.to_node): r for r in TABLE.relations}
    assert by_pair[("tsmc_n16", "tsmc_n7")].power_condition == "iso_power"
    assert "at the same power" in by_pair[("tsmc_n16", "tsmc_n7")].source
    for pair in (("tsmc_n7", "tsmc_n5"), ("tsmc_n5", "tsmc_n4p")):
        assert by_pair[pair].power_condition == "unstated"
    assert retarget_fmax(1.0, "tsmc_n7", "tsmc_n5", TABLE) is None
    assert retarget_fmax(1.0, "tsmc_n16", "tsmc_n4p", TABLE) is None


@pytest.mark.parametrize("reference, target", [
    ("samsung_8lpp", "tsmc_n7"),   # no foundry states a Samsung-vs-TSMC relation
    ("samsung_8lpp", "tsmc_n5"),
    ("tsmc_n7", "gf_12lp"),
    ("tsmc_n7", "tsmc_n5"),        # TSMC states the gain but not its power condition
])
def test_no_stated_relation_refuses_to_retarget(reference, target):
    """Samsung publishes no 10LPP-to-8LPP speed figure and no foundry
    compares itself with a competitor, so an 8LPP clock has no defensible
    value at N7. None, not an extrapolation."""
    assert retarget_fmax(1.3, reference, target, TABLE) is None


def test_every_relation_is_sourced_and_names_catalog_nodes():
    assert TABLE.relations, "node_speed.yaml is empty"
    for relation in TABLE.relations:
        assert relation.from_node in NODES, relation.from_node
        assert relation.to_node in NODES, relation.to_node
        assert "TSMC" in relation.source or "Samsung" in relation.source or "GF" in relation.source
        assert relation.speed_ratio.low > 1.0  # each is a gain to a newer node


def test_the_table_rejects_a_pair_stated_twice():
    rel = {"from_node": "tsmc_n16", "to_node": "tsmc_n7", "power_condition": "iso_power",
           "speed_ratio": {"low": 1.3, "high": 1.3}, "source": "x"}
    reverse = {**rel, "from_node": "tsmc_n7", "to_node": "tsmc_n16"}
    with pytest.raises(ValidationError, match="more than once"):
        NodeSpeedTable.model_validate({"relations": [rel, reverse]})


def test_a_range_must_be_ordered():
    with pytest.raises(ValidationError, match="exceeds"):
        NodeSpeedTable.model_validate({"relations": [{
            "from_node": "a", "to_node": "b", "power_condition": "iso_power",
            "speed_ratio": {"low": 1.4, "high": 1.3}, "source": "x"}]})


def test_a_relation_must_state_its_power_condition():
    with pytest.raises(ValidationError, match="power_condition"):
        NodeSpeedTable.model_validate({"relations": [{
            "from_node": "a", "to_node": "b",
            "speed_ratio": {"low": 1.1, "high": 1.1}, "source": "x"}]})


# ---------------------------------------------------------------------------
# Composition uses it
# ---------------------------------------------------------------------------


def _design(node_ref: str) -> tuple:
    template = IPBlockTemplate.model_validate({
        "id": "gpu", "name": "gpu", "engine_kind": "gpu",
        "silicon": [{"name": "logic", "circuit_class": "hp_logic", "mtx": 100.0, "source": "s"}],
        "compute": {"engine_kind": "gpu", "units": 1, "unit_name": "SM",
                    "ops_per_clock": {"int8": 1000.0}, "source": "s"},
        "clock": {"fmax_ghz_ref": 1.0, "reference_node": node_ref, "source": "s"},
    })
    design = SoCDesign(id="d", name="d", process_node=node_ref,
                       blocks=[DesignBlock(instance="g", ip="gpu")], layout=Layout())
    return design, {"gpu": template}


def test_a_retargeted_clock_sets_the_peak():
    design, library = _design("tsmc_n16")
    soc = compose_soc(design, library, NODES, "tsmc_n7")
    block = soc.blocks[0]
    assert block.clock_basis == "retargeted"
    assert block.clock_ghz == pytest.approx(1.30)
    assert soc.peak_ops_per_s("int8") == pytest.approx(1000 * 1.30e9)
    assert soc.off_reference_clocks == ()


def test_an_unretargetable_clock_is_flagged_not_invented():
    design, library = _design("samsung_8lpp")
    soc = compose_soc(design, library, NODES, "tsmc_n7")
    assert soc.blocks[0].clock_basis == "unretargetable"
    assert soc.blocks[0].clock_retarget is None
    assert soc.off_reference_clocks == ("g",)


def test_the_orin_clocks_cannot_move_off_8lpp():
    """The acceptance design's GPU, DLA and CPU clocks are 8LPP figures, and
    nothing connects 8LPP to TSMC. At N7 they stay the 8LPP numbers and are
    flagged, so its N7 peaks are provisional, and the instance says so."""
    from graphs.hardware.soc import load_designs, load_ip_library

    soc = compose_soc(load_designs()["orin_class_reference"], load_ip_library(), NODES, "tsmc_n7")
    assert set(soc.off_reference_clocks) == {"gpu_sm", "dla", "cpu"}
    assert {b.clock_basis for b in soc.blocks if b.name in soc.off_reference_clocks} == {
        "unretargetable"
    }
