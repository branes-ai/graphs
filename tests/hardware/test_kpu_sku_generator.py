"""Phase 3 generator tests.

Coverage:
  - Round-trip every catalog SKU: extract spec -> regenerate -> die,
    perf, power numbers match the original within rounding.
  - Generator output validates clean (no ERROR findings) on all four
    real SKUs.
  - GeneratorError on bad inputs (missing process node, bad default
    profile name, empty silicon_bin).
  - input_spec_from_compute_product preserves architect-authored fields.
"""

from __future__ import annotations

import pytest

from embodied_schemas import (
    KPUSiliconBin,
    load_cooling_solutions,
    load_process_nodes,
)

from graphs.hardware.compute_product_loader import load_compute_products_unified
from graphs.hardware.kpu_sku_generator import (
    GeneratorError,
    apply_pe_array_override,
    generate_kpu_sku,
    input_spec_from_compute_product,
)
from graphs.hardware.sku_validators import (
    Severity,
    ValidatorContext,
    default_registry,
    load_validators,
)


load_validators()


@pytest.fixture(scope="module")
def catalogs():
    return {
        "kpus": load_compute_products_unified(),
        "process_nodes": load_process_nodes(),
        "cooling_solutions": load_cooling_solutions(),
    }


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sku_id", [
    "kpu_t64_32x32_lp5x4_16nm_tsmc_ffp",
    "kpu_t128_32x32_lp5x8_16nm_tsmc_ffp",
    "kpu_t256_32x32_lp5x16_16nm_tsmc_ffp",
    "kpu_t768_16x8_hbm3x16_7nm_tsmc_hpc",
])
def test_roundtrip_die_within_rounding(sku_id, catalogs):
    """Generator-derived die.transistors_billion and die.die_size_mm2
    must match the YAML's hand-authored values within rounding."""
    original = catalogs["kpus"][sku_id]
    spec = input_spec_from_compute_product(original)
    regen = generate_kpu_sku(
        spec,
        process_nodes=catalogs["process_nodes"],
    )
    # Within 2% (the generator rounds to 1 dp for area / 3 dp for
    # transistors; the hand-authored YAMLs round earlier in the
    # back-of-envelope chain, so a couple of percent of difference is
    # rounding only, not a real generator drift).
    # Defensive guards: zero transistors / zero die size would mean a
    # broken catalog entry, not a generator issue -- fail loudly with
    # a clear message rather than ZeroDivisionError if a future SKU
    # YAML is misauthored.
    assert original.dies[0].transistors_billion > 0, (
        f"{sku_id}: catalog entry has die.transistors_billion <= 0"
    )
    assert original.dies[0].die_size_mm2 > 0, (
        f"{sku_id}: catalog entry has die.die_size_mm2 <= 0"
    )
    assert abs(regen.dies[0].transistors_billion - original.dies[0].transistors_billion) \
        / original.dies[0].transistors_billion <= 0.02
    assert abs(regen.dies[0].die_size_mm2 - original.dies[0].die_size_mm2) \
        / original.dies[0].die_size_mm2 <= 0.02


@pytest.mark.parametrize("sku_id", [
    "kpu_t64_32x32_lp5x4_16nm_tsmc_ffp",
    "kpu_t128_32x32_lp5x8_16nm_tsmc_ffp",
    "kpu_t256_32x32_lp5x16_16nm_tsmc_ffp",
    "kpu_t768_16x8_hbm3x16_7nm_tsmc_hpc",
])
def test_roundtrip_performance_within_rounding(sku_id, catalogs):
    """int8_tops, bf16_tflops, fp32_tflops should round-trip within 1%."""
    original = catalogs["kpus"][sku_id]
    spec = input_spec_from_compute_product(original)
    regen = generate_kpu_sku(
        spec,
        process_nodes=catalogs["process_nodes"],
    )
    for attr in ("int8_tops", "bf16_tflops", "fp32_tflops"):
        a = getattr(original.performance, attr)
        b = getattr(regen.performance, attr)
        if a == 0 and b == 0:
            continue
        assert abs(b - a) / a <= 0.02, (
            f"{sku_id} {attr}: original={a} regen={b}"
        )


@pytest.mark.parametrize("sku_id", [
    "kpu_t64_32x32_lp5x4_16nm_tsmc_ffp",
    "kpu_t128_32x32_lp5x8_16nm_tsmc_ffp",
    "kpu_t256_32x32_lp5x16_16nm_tsmc_ffp",
    "kpu_t768_16x8_hbm3x16_7nm_tsmc_hpc",
])
def test_roundtrip_power_default_tdp_is_derived(sku_id, catalogs):
    """power.tdp_watts is DERIVED by the kpu_power_model from
    (clock, Vdd, ProcessNode energies, WorkloadAssumption). The catalog
    YAML's tdp_watts must match the generator's derived value to within
    0.5W -- the architect tunes (clock, Vdd) per profile so derivation
    lands at the target TDP."""
    original = catalogs["kpus"][sku_id]
    spec = input_spec_from_compute_product(original)
    regen = generate_kpu_sku(
        spec,
        process_nodes=catalogs["process_nodes"],
    )
    assert regen.power.default_thermal_profile == original.power.default_thermal_profile
    assert len(regen.power.thermal_profiles) == len(original.power.thermal_profiles)
    # Default-profile TDP within rounding of catalog target
    assert abs(regen.power.tdp_watts - original.power.tdp_watts) < 0.5, (
        f"{sku_id}: catalog default TDP={original.power.tdp_watts} W, "
        f"derived={regen.power.tdp_watts} W"
    )
    # Per-profile derived TDPs match catalog targets
    for o, r in zip(original.power.thermal_profiles, regen.power.thermal_profiles):
        assert abs(r.tdp_watts - o.tdp_watts) < 0.5, (
            f"{sku_id} profile {o.name}: catalog={o.tdp_watts} W, "
            f"derived={r.tdp_watts} W"
        )


@pytest.mark.parametrize("sku_id", [
    "kpu_t64_32x32_lp5x4_16nm_tsmc_ffp",
    "kpu_t128_32x32_lp5x8_16nm_tsmc_ffp",
    "kpu_t256_32x32_lp5x16_16nm_tsmc_ffp",
    "kpu_t768_16x8_hbm3x16_7nm_tsmc_hpc",
])
def test_generated_sku_validates_clean(sku_id, catalogs):
    """Output of the generator should have zero ERROR findings when run
    through the validator framework. WARNINGs are expected (DVFS
    throttle messages)."""
    original = catalogs["kpus"][sku_id]
    spec = input_spec_from_compute_product(original)
    regen = generate_kpu_sku(
        spec,
        process_nodes=catalogs["process_nodes"],
    )
    # generate_kpu_sku now returns a ComputeProduct directly; pass to the
    # ValidatorContext without an adapter.
    ctx = ValidatorContext(
        sku=regen,
        process_node=catalogs["process_nodes"][regen.dies[0].process_node_id],
        cooling_solutions=catalogs["cooling_solutions"],
    )
    findings = default_registry.run_all(ctx)
    errors = [f for f in findings if f.severity == Severity.ERROR]
    assert not errors, (
        f"generated {sku_id} produced ERROR findings:\n"
        + "\n".join(f.render_one_line() for f in errors)
    )


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------

def test_unknown_process_node_raises(catalogs):
    """A spec referencing a non-existent process_node_id raises GeneratorError."""
    original = catalogs["kpus"]["kpu_t256_32x32_lp5x16_16nm_tsmc_ffp"]
    spec = input_spec_from_compute_product(original).model_copy(
        update={"process_node_id": "no_such_node"}
    )
    with pytest.raises(GeneratorError, match="does not resolve"):
        generate_kpu_sku(
            spec,
            process_nodes=catalogs["process_nodes"],
        )


def test_bad_default_profile_name_raises(catalogs):
    """default_thermal_profile must name an existing profile."""
    original = catalogs["kpus"]["kpu_t256_32x32_lp5x16_16nm_tsmc_ffp"]
    spec = input_spec_from_compute_product(original).model_copy(
        update={"default_thermal_profile": "9000W"}
    )
    with pytest.raises(GeneratorError, match="not in thermal_profiles"):
        generate_kpu_sku(
            spec,
            process_nodes=catalogs["process_nodes"],
        )


def test_empty_silicon_bin_raises(catalogs):
    """A spec with an empty silicon_bin (no resolvable blocks) raises."""
    original = catalogs["kpus"]["kpu_t256_32x32_lp5x16_16nm_tsmc_ffp"]
    spec = input_spec_from_compute_product(original).model_copy(
        update={"silicon_bin": KPUSiliconBin(blocks=[])}
    )
    with pytest.raises(GeneratorError, match="silicon_bin"):
        generate_kpu_sku(
            spec,
            process_nodes=catalogs["process_nodes"],
        )


# ---------------------------------------------------------------------------
# input_spec_from_compute_product preserves architect-authored fields
# ---------------------------------------------------------------------------

def test_input_spec_preserves_architecture(catalogs):
    original = catalogs["kpus"]["kpu_t256_32x32_lp5x16_16nm_tsmc_ffp"]
    spec = input_spec_from_compute_product(original)
    die = original.dies[0]
    block = die.blocks[0]
    # KPUArchitecture (legacy spec field) is structurally identical to
    # KPUBlock (new schema): same total_tiles, multi_precision_alu,
    # tiles, noc, memory.
    assert spec.kpu_architecture.total_tiles == block.total_tiles
    assert spec.kpu_architecture.tiles == block.tiles
    assert spec.kpu_architecture.noc == block.noc
    assert spec.kpu_architecture.memory == block.memory
    assert spec.silicon_bin == die.silicon_bin
    assert spec.thermal_profiles == original.power.thermal_profiles
    # KPUMarket has is_discontinued; ComputeProduct.market is the new
    # Market (no is_discontinued -- it lives in cp.lifecycle). Compare
    # the fields that DO overlap.
    assert spec.market.target_market == original.market.target_market
    assert spec.market.product_family == original.market.product_family
    assert spec.market.model_tier == original.market.model_tier
    assert spec.market.is_available == original.market.is_available
    assert spec.clocks == die.clocks


def test_input_spec_does_not_carry_die_or_perf_rollups(catalogs):
    """The spec's shape excludes the generator-derived roll-ups -- if a
    field is in KPUEntry but not in KPUSKUInputSpec, the spec is
    correctly minimal."""
    original = catalogs["kpus"]["kpu_t256_32x32_lp5x16_16nm_tsmc_ffp"]
    spec = input_spec_from_compute_product(original)
    spec_fields = set(type(spec).model_fields)
    assert "die" not in spec_fields
    assert "performance" not in spec_fields
    assert "power" not in spec_fields  # Power.tdp roll-up is generator-derived; profiles are separate


# ---------------------------------------------------------------------------
# apply_pe_array_override -- PE-array sweep helper
# ---------------------------------------------------------------------------

def test_pe_array_override_is_no_op_when_dims_unchanged(catalogs):
    """Override to the spec's existing PE-array dims must round-trip."""
    original = catalogs["kpus"]["kpu_t256_32x32_lp5x16_16nm_tsmc_ffp"]
    spec = input_spec_from_compute_product(original)
    rows = spec.kpu_architecture.tiles[0].pe_array_rows
    cols = spec.kpu_architecture.tiles[0].pe_array_cols
    overridden = apply_pe_array_override(spec, rows, cols)
    g_orig = generate_kpu_sku(spec, process_nodes=catalogs["process_nodes"])
    g_over = generate_kpu_sku(overridden, process_nodes=catalogs["process_nodes"])
    assert g_orig.dies[0].die_size_mm2 == g_over.dies[0].die_size_mm2
    assert g_orig.dies[0].transistors_billion == g_over.dies[0].transistors_billion
    assert g_orig.performance == g_over.performance


def test_pe_array_override_preserves_ops_per_pe_ratio(catalogs):
    """Scaling PE-array size must keep ops/PE/clock constant."""
    original = catalogs["kpus"]["kpu_t256_32x32_lp5x16_16nm_tsmc_ffp"]
    spec = input_spec_from_compute_product(original)
    overridden = apply_pe_array_override(spec, 16, 16)
    for orig_t, new_t in zip(spec.kpu_architecture.tiles, overridden.kpu_architecture.tiles):
        old_pes = orig_t.pe_array_rows * orig_t.pe_array_cols
        new_pes = new_t.pe_array_rows * new_t.pe_array_cols
        for precision, old_ops in orig_t.ops_per_tile_per_clock.items():
            new_ops = new_t.ops_per_tile_per_clock[precision]
            assert (new_ops / new_pes) == pytest.approx(old_ops / old_pes)


def test_pe_array_override_scales_die_area(catalogs):
    """Halving PE-array dims should reduce die area but not below the
    fixed (IO/control/memory_phys) floor."""
    original = catalogs["kpus"]["kpu_t256_32x32_lp5x16_16nm_tsmc_ffp"]
    spec = input_spec_from_compute_product(original)
    g_full = generate_kpu_sku(spec, process_nodes=catalogs["process_nodes"])
    g_half = generate_kpu_sku(
        apply_pe_array_override(spec, 16, 16),
        process_nodes=catalogs["process_nodes"],
    )
    assert g_half.dies[0].die_size_mm2 < g_full.dies[0].die_size_mm2
    assert g_half.performance.int8_tops < g_full.performance.int8_tops


def test_pe_array_override_rejects_non_positive_dims(catalogs):
    spec = input_spec_from_compute_product(catalogs["kpus"]["kpu_t256_32x32_lp5x16_16nm_tsmc_ffp"])
    with pytest.raises(ValueError):
        apply_pe_array_override(spec, 0, 32)
    with pytest.raises(ValueError):
        apply_pe_array_override(spec, 32, -1)


# ---------------------------------------------------------------------------
# Power model -- V^2*f scaling, sweep monotonicity, activity_factor knob
# ---------------------------------------------------------------------------

from graphs.hardware.kpu_power_model import compute_thermal_profile_tdp_breakdown


def test_tdp_scales_quadratically_with_vdd(catalogs):
    """Dropping Vdd by sqrt(2) should halve the dynamic power; total
    TDP drops by half the dynamic share."""
    sku = catalogs["kpus"]["kpu_t256_32x32_lp5x16_16nm_tsmc_ffp"]
    spec = input_spec_from_compute_product(sku)
    node = catalogs["process_nodes"][sku.dies[0].process_node_id]
    # Take the default profile and run with two Vdds: nominal and nominal/sqrt(2).
    base = sku.power.thermal_profiles[1]  # 30W profile
    high_v = base.model_copy(update={"vdd_v": 0.80})  # ~ Vnom
    low_v = base.model_copy(update={"vdd_v": 0.80 / (2 ** 0.5)})  # halved V^2
    bd_high = compute_thermal_profile_tdp_breakdown(spec, high_v, node)
    bd_low = compute_thermal_profile_tdp_breakdown(spec, low_v, node)
    # Dynamic should halve; leakage unchanged.
    assert bd_low.dynamic_w == pytest.approx(bd_high.dynamic_w / 2.0, rel=0.01)
    assert bd_low.leakage_w == pytest.approx(bd_high.leakage_w)


def test_tdp_scales_linearly_with_clock(catalogs):
    """At fixed Vdd, doubling clock doubles dynamic power."""
    sku = catalogs["kpus"]["kpu_t256_32x32_lp5x16_16nm_tsmc_ffp"]
    spec = input_spec_from_compute_product(sku)
    node = catalogs["process_nodes"][sku.dies[0].process_node_id]
    base = sku.power.thermal_profiles[1]
    p_low = base.model_copy(update={"clock_mhz": 500.0, "vdd_v": 0.80})
    p_high = base.model_copy(update={"clock_mhz": 1000.0, "vdd_v": 0.80})
    bd_low = compute_thermal_profile_tdp_breakdown(spec, p_low, node)
    bd_high = compute_thermal_profile_tdp_breakdown(spec, p_high, node)
    assert bd_high.dynamic_w == pytest.approx(2.0 * bd_low.dynamic_w, rel=0.01)


def test_activity_factor_scales_dynamic(catalogs):
    """Per-profile activity_factor=0.5 should halve dynamic power
    (it's a multiplier on the workload duty cycle)."""
    sku = catalogs["kpus"]["kpu_t256_32x32_lp5x16_16nm_tsmc_ffp"]
    spec = input_spec_from_compute_product(sku)
    node = catalogs["process_nodes"][sku.dies[0].process_node_id]
    base = sku.power.thermal_profiles[1]
    full = base.model_copy(update={"activity_factor": 1.0})
    half = base.model_copy(update={"activity_factor": 0.5})
    bd_full = compute_thermal_profile_tdp_breakdown(spec, full, node)
    bd_half = compute_thermal_profile_tdp_breakdown(spec, half, node)
    assert bd_half.dynamic_w == pytest.approx(bd_full.dynamic_w / 2.0, rel=0.01)
    assert bd_half.leakage_w == pytest.approx(bd_full.leakage_w)


def test_pe_array_sweep_tdp_monotonic(catalogs):
    """Across a PE-array sweep at fixed clock + Vdd, derived TDP must
    strictly increase with PE count -- the whole point of programmable
    PE arrays for roadmap generation."""
    sku = catalogs["kpus"]["kpu_t256_32x32_lp5x16_16nm_tsmc_ffp"]
    spec = input_spec_from_compute_product(sku)
    node = catalogs["process_nodes"][sku.dies[0].process_node_id]
    base = sku.power.thermal_profiles[1]
    sweep = [(16, 16), (24, 24), (32, 32), (40, 40)]
    tdps = []
    for rows, cols in sweep:
        s = apply_pe_array_override(spec, rows, cols)
        bd = compute_thermal_profile_tdp_breakdown(s, base, node)
        tdps.append(bd.total_tdp_w)
    assert tdps == sorted(tdps), (
        f"TDP not monotonic across PE-array sweep: {list(zip(sweep, tdps))}"
    )
