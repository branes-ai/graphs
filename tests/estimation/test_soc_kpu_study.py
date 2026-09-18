"""KPU IP generation, the KPU-heterogeneous design and the first study
(graphs#269 PR 4.3)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from embodied_schemas import load_compute_products, load_process_nodes

from graphs.estimation.soc import SoCAnalyzer, engines_of, load_efficiency_tables, load_kernel_classes, schedule
from graphs.estimation.soc.pareto import classify_front, union_of_regimes
from graphs.estimation.soc.study import load_study, run_study
from graphs.hardware.soc import EngineKind, compose_soc, load_designs, load_ip_library

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
import generate_kpu_ip as gen  # noqa: E402

NODES = load_process_nodes()
LIBRARY = load_ip_library()
DESIGN = load_designs()["kpu_heterogeneous_h64"]


# ---------------------------------------------------------------------------
# Generated IP
# ---------------------------------------------------------------------------


def test_generated_templates_match_their_compute_products():
    assert gen.main(["--check"]) == 0


def test_the_h64s_unpriced_tiles_are_gaps_not_a_complete_block():
    """The H64's silicon_bin prices only its INT8 and min-plus PE tiles. Its
    systolic, LNS and three fixed-function tiles have no line, so they are
    unanchored -- the KPU is not presented as complete."""
    unanchored = {l.name for l in LIBRARY["kpu_h64_core"].unanchored_lines}
    assert unanchored == {"tile_pe_lns16_mac", "tile_systolic_int8_ws", "tile_ff_isp_raw2yuv",
                          "tile_ff_stereo_sgm", "tile_ff_vio_stereo_inertial"}


def test_a_fully_priced_kpu_generates_complete():
    assert not LIBRARY["kpu_t128_core"].unanchored_lines


def test_a_core_drops_the_chips_own_phys_and_pads():
    names = {l.name for l in LIBRARY["kpu_t128_core"].silicon}
    assert not names & gen.CHIP_ONLY
    assert LIBRARY["kpu_t128_core"].memory_interface is None


def test_peak_ops_per_clock_is_the_tile_sum():
    """24 INT8 PE tiles x 2048 + 4 systolic tiles x 8192; FP16 from the PE
    tiles only; BF16, INT4 and LNS are not counted as anything else."""
    compute = LIBRARY["kpu_h64_core"].compute
    assert compute.ops_per_clock == {"int8": 24 * 2048 + 4 * 8192, "fp16": 24 * 1024}
    assert compute.units == 45 and compute.architecture_class == "domain_flow"


def test_transistors_are_the_skus_own():
    cp = load_compute_products()["kpu_t128_32x32_lp5x8_7nm_tsmc_hpc"]
    from graphs.hardware.sku_validators.silicon_math import resolve_block_transistors, silicon_die

    expected = sum(resolve_block_transistors(b, cp) for b in silicon_die(cp).silicon_bin.blocks
                   if b.name not in gen.CHIP_ONLY)
    assert LIBRARY["kpu_t128_core"].transistors_mtx(NODES) == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# The design
# ---------------------------------------------------------------------------


def test_the_kpu_is_one_server_and_its_clock_follows_the_speed_table():
    at = {n: compose_soc(DESIGN, LIBRARY, NODES, n) for n in ("tsmc_n16", "tsmc_n7", "tsmc_n5")}
    kpu = {n: next(b for b in s.blocks if b.name == "kpu") for n, s in at.items()}
    assert engines_of(at["tsmc_n7"])["kpu"].servers == 1
    assert kpu["tsmc_n7"].clock_basis == "reference" and kpu["tsmc_n7"].clock_ghz == 0.75
    # Down to N16 along TSMC's iso-power relation, at its conservative end.
    assert kpu["tsmc_n16"].clock_basis == "retargeted"
    assert kpu["tsmc_n16"].clock_ghz == pytest.approx(0.75 / 1.35)
    # N7 -> N5 states no power condition: the clock is provisional there.
    assert kpu["tsmc_n5"].clock_basis == "unretargetable"


def test_class_c_cannot_run_on_the_kpu():
    """No FP32 on the H64: a stage with a Class C share fails closed."""
    soc = compose_soc(DESIGN, LIBRARY, NODES, "tsmc_n7")
    wl = SoCAnalyzer().workload
    air = wl.profile("air superiority")
    s = schedule(wl, air, soc, _all_known(), load_kernel_classes(), {"sgm": "kpu", "det": "kpu"})
    reasons = dict(s.gaps)
    assert reasons["sgm"] == "kpu cannot run Class C"
    assert "det" not in reasons


def _all_known():
    from graphs.estimation.soc import EfficiencyTable, KernelClass

    return EfficiencyTable.model_validate(dict(
        id="all_known", name="t", kind="per_engine",
        entries=[dict(kernel_class=k.value, engine_kind=e, precision=p, compute_eff=0.3,
                      confidence="theoretical", source="t")
                 for k in KernelClass for e in ("cpu", "kpu") for p in ("int8", "fp16", "fp32")]))


def test_the_explicit_mapping_sends_class_c_to_the_cpu():
    from graphs.estimation.soc import find_mapping

    wl = SoCAnalyzer().workload
    mapping = find_mapping(DESIGN.id, wl.version)
    for key, stage in wl.stages.items():
        engine = mapping.stages[key].engine
        assert engine == ("cpu" if stage.class_split[2] > 0 else "kpu"), key


# ---------------------------------------------------------------------------
# The study, and what it can decide
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def study_rows():
    return run_study(load_study("orin_vs_kpu_heterogeneous"))


def test_the_study_covers_its_axes(study_rows):
    assert len(study_rows) == 2 * 3 * 2 * 2 * 2  # designs x nodes x regimes x tables x gating


def test_no_point_is_on_a_front_and_no_design_is_proven(study_rows):
    """Both designs carry unanchored silicon and no priced dynamic power, so
    area and power are lower bounds everywhere: the comparison is undecided,
    and the report says so instead of naming a winner."""
    assert {s for _, s in classify_front(study_rows, ["area", "power"])} == {"undecided"}
    assert union_of_regimes(study_rows).minimum is None


def test_what_the_study_does_decide_far_flight_dram(study_rows):
    """Both designs keep the reference's 204.8 GB/s LPDDR5; far flight
    demands 306 GB/s, so both are proven infeasible in it whatever the
    compute turns out to be."""
    far = [r for r in study_rows if r.result.profile.regime == "far flight"]
    assert far and all(r.result.feasible() is False for r in far)
    assert all(r.result.schedule.dram_utilization > 1 for r in far)


def test_the_pooled_model_does_not_distinguish_the_designs(study_rows):
    pooled = {}
    for r in study_rows:
        if r.point.efficiency == "annex_v1":
            pooled.setdefault(r.result.profile.id, set()).add(round(r.result.schedule.oversubscription, 12))
    assert all(len(v) == 1 for v in pooled.values())
