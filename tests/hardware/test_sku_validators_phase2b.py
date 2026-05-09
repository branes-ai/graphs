"""Phase 2b validator tests.

Each test crafts a broken or borderline SKU by mutating a real loaded
SKU in-memory and asserts the relevant validator produces (or doesn't
produce) the expected Findings. Uses ``model_copy(update=)`` to mutate
Pydantic models without touching disk.

Test fixtures:
- ``ctx``: real ValidatorContext for stillwater_kpu_t256 (a known-good
  SKU). Tests perturb it to verify catches.
"""

from __future__ import annotations

import pytest

from embodied_schemas import (
    KPUEntry,
    KPUTheoreticalPerformance,
    SiliconBinBlock,
    TransistorSource,
    TransistorSourceKind,
)
from embodied_schemas.process_node import CircuitClass

from graphs.hardware.sku_validators import (
    Severity,
    ValidatorCategory,
    ValidatorContext,
    build_context_for_kpu,
    default_registry,
    load_validators,
)


# Auto-load validators for every test in this module.
load_validators()


@pytest.fixture
def ctx() -> ValidatorContext:
    """Real context for a known-good SKU. Tests mutate sku/process_node
    via model_copy and feed the result into a perturbed context."""
    return build_context_for_kpu("stillwater_kpu_t256")


def _ctx_with_sku(ctx: ValidatorContext, **sku_updates) -> ValidatorContext:
    """Return a new context with sku replaced by the perturbed version."""
    new_sku = ctx.sku.model_copy(update=sku_updates)
    return ValidatorContext(
        sku=new_sku,
        process_node=ctx.process_node,
        cooling_solutions=ctx.cooling_solutions,
    )


def _findings_for(name: str, ctx: ValidatorContext):
    return default_registry.run_one(name, ctx)


# ---------------------------------------------------------------------------
# Sanity: real SKU passes everything
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sku_id", [
    "stillwater_kpu_t64",
    "stillwater_kpu_t128",
    "stillwater_kpu_t256",
    "stillwater_kpu_t768",
])
def test_real_skus_have_no_errors(sku_id):
    """Hand-authored SKUs should pass every validator. If a future YAML
    edit introduces an inconsistency, this test catches it in CI."""
    ctx = build_context_for_kpu(sku_id)
    findings = default_registry.run_all(ctx)
    errors = [f for f in findings if f.severity == Severity.ERROR]
    assert not errors, (
        f"{sku_id} produced ERROR findings:\n"
        + "\n".join(f.render_one_line() for f in errors)
    )


# ---------------------------------------------------------------------------
# tile_mix_consistency
# ---------------------------------------------------------------------------

def test_tile_mix_consistency_catches_bad_int8_tops(ctx):
    """A claim of 1 PetaOP @ T256 specs would be ~3.5x the real fabric
    output -- the validator should ERROR."""
    bad_perf = ctx.sku.performance.model_copy(update={"int8_tops": 1000.0})
    bad_ctx = _ctx_with_sku(ctx, performance=bad_perf)
    findings = _findings_for("tile_mix_consistency", bad_ctx)
    errors = [f for f in findings if f.severity == Severity.ERROR]
    assert any("int8_tops" in f.message for f in errors)


def test_tile_mix_consistency_catches_total_tiles_mismatch(ctx):
    """Σtile.num_tiles must equal total_tiles."""
    bad_arch = ctx.sku.kpu_architecture.model_copy(update={"total_tiles": 999})
    bad_ctx = _ctx_with_sku(ctx, kpu_architecture=bad_arch)
    findings = _findings_for("tile_mix_consistency", bad_ctx)
    assert any(
        f.severity == Severity.ERROR and "total_tiles" in f.message
        for f in findings
    )


def test_tile_mix_consistency_catches_unsupported_default_profile(ctx):
    """power.default_thermal_profile must name an actual profile."""
    bad_power = ctx.sku.power.model_copy(
        update={"default_thermal_profile": "9000W"}
    )
    bad_ctx = _ctx_with_sku(ctx, power=bad_power)
    findings = _findings_for("tile_mix_consistency", bad_ctx)
    assert any(
        f.severity == Severity.ERROR
        and "default_thermal_profile" in f.message
        for f in findings
    )


def test_tile_mix_consistency_passes_real_sku(ctx):
    findings = _findings_for("tile_mix_consistency", ctx)
    errs = [f for f in findings if f.severity == Severity.ERROR]
    assert not errs


# ---------------------------------------------------------------------------
# cross_ref_consistency
# ---------------------------------------------------------------------------

def test_cross_ref_catches_dangling_cooling_id(ctx):
    """A thermal profile pointing at a non-existent cooling solution."""
    profiles = list(ctx.sku.power.thermal_profiles)
    profiles[0] = profiles[0].model_copy(update={"cooling_solution_id": "nope"})
    bad_power = ctx.sku.power.model_copy(update={"thermal_profiles": profiles})
    bad_ctx = _ctx_with_sku(ctx, power=bad_power)
    findings = _findings_for("cross_ref_consistency", bad_ctx)
    assert any(
        f.severity == Severity.ERROR
        and "cooling_solution_id" in f.message
        and f.profile == profiles[0].name
        for f in findings
    )


def test_cross_ref_catches_unsupported_circuit_class(ctx):
    """A silicon_bin block tagged with a CircuitClass the node doesn't
    offer (e.g., SRAM_HP on a node without dual-port SRAM)."""
    blocks = list(ctx.sku.silicon_bin.blocks)
    blocks[0] = SiliconBinBlock(
        name="bogus",
        circuit_class=CircuitClass.SRAM_HP,  # tsmc_n16 doesn't list HP-SRAM
        transistor_source=TransistorSource(
            kind=TransistorSourceKind.FIXED, mtx=1.0
        ),
    )
    bad_bin = ctx.sku.silicon_bin.model_copy(update={"blocks": blocks})
    bad_ctx = _ctx_with_sku(ctx, silicon_bin=bad_bin)
    findings = _findings_for("cross_ref_consistency", bad_ctx)
    assert any(
        f.severity == Severity.ERROR
        and f.block == "bogus"
        and "sram_hp" in f.message
        for f in findings
    )


# ---------------------------------------------------------------------------
# power_profile_monotonicity
# ---------------------------------------------------------------------------

def test_power_monotonicity_catches_inverted_clocks(ctx):
    """A higher-TDP profile that runs slower than a lower-TDP one."""
    profiles = list(ctx.sku.power.thermal_profiles)
    # Sorted by TDP ascending: 15W -> 1100, 30W -> 1400, 50W -> 1650
    # Force 50W to run at 1000 MHz (slower than 30W's 1400).
    for i, p in enumerate(profiles):
        if p.name == "50W":
            profiles[i] = p.model_copy(update={"clock_mhz": 1000.0})
    bad_power = ctx.sku.power.model_copy(update={"thermal_profiles": profiles})
    bad_ctx = _ctx_with_sku(ctx, power=bad_power)
    findings = _findings_for("power_profile_monotonicity", bad_ctx)
    assert any(
        f.severity == Severity.WARNING and f.profile == "50W"
        for f in findings
    )


def test_power_monotonicity_passes_real_sku(ctx):
    findings = _findings_for("power_profile_monotonicity", ctx)
    warns = [f for f in findings if f.severity == Severity.WARNING]
    assert not warns


# ---------------------------------------------------------------------------
# block_library_validity
# ---------------------------------------------------------------------------

def test_block_library_validity_catches_unsupported_class(ctx):
    """Same as the cross_ref check, but exposed in the AREA category for
    users who --filter to AREA."""
    blocks = list(ctx.sku.silicon_bin.blocks)
    blocks.append(
        SiliconBinBlock(
            name="bogus_ull",
            circuit_class=CircuitClass.ULL_LOGIC,  # tsmc_n16 doesn't offer ULL
            transistor_source=TransistorSource(
                kind=TransistorSourceKind.FIXED, mtx=1.0
            ),
        )
    )
    bad_bin = ctx.sku.silicon_bin.model_copy(update={"blocks": blocks})
    bad_ctx = _ctx_with_sku(ctx, silicon_bin=bad_bin)
    findings = _findings_for("block_library_validity", bad_ctx)
    assert any(
        f.severity == Severity.ERROR and f.block == "bogus_ull"
        for f in findings
    )


def test_block_library_validity_passes_when_class_supported(ctx):
    """SRAM_HD is supported by tsmc_n16; no finding."""
    findings = _findings_for("block_library_validity", ctx)
    assert not findings


# ---------------------------------------------------------------------------
# area_self_consistency
# ---------------------------------------------------------------------------

def test_area_self_consistency_catches_inflated_die(ctx):
    """Doubling claimed die_size while leaving silicon_bin alone -> ERROR."""
    bad_die = ctx.sku.die.model_copy(
        update={"die_size_mm2": ctx.sku.die.die_size_mm2 * 2.0}
    )
    bad_ctx = _ctx_with_sku(ctx, die=bad_die)
    findings = _findings_for("area_self_consistency", bad_ctx)
    assert any(
        f.severity == Severity.ERROR and "die.die_size_mm2" in f.message
        for f in findings
    )


def test_area_self_consistency_catches_minor_mismatch_as_warning(ctx):
    """A 10% die_size error sits in the WARN band (5-25%)."""
    bad_die = ctx.sku.die.model_copy(
        update={"die_size_mm2": ctx.sku.die.die_size_mm2 * 1.10}
    )
    bad_ctx = _ctx_with_sku(ctx, die=bad_die)
    findings = _findings_for("area_self_consistency", bad_ctx)
    assert any(
        f.severity == Severity.WARNING and "die.die_size_mm2" in f.message
        for f in findings
    )


def test_area_self_consistency_passes_real_sku(ctx):
    findings = _findings_for("area_self_consistency", ctx)
    assert not findings


# ---------------------------------------------------------------------------
# composite_density_envelope
# ---------------------------------------------------------------------------

def test_composite_density_catches_above_node_max(ctx):
    """Claim 100 B transistors on a 50 mm^2 die at TSMC N16 -> 2000
    Mtx/mm^2 composite, way above N16's max library density (60)."""
    bad_die = ctx.sku.die.model_copy(
        update={"transistors_billion": 100.0, "die_size_mm2": 50.0}
    )
    bad_ctx = _ctx_with_sku(ctx, die=bad_die)
    findings = _findings_for("composite_density_envelope", bad_ctx)
    assert any(
        f.severity == Severity.ERROR and "exceeds" in f.message
        for f in findings
    )


def test_composite_density_passes_real_sku(ctx):
    findings = _findings_for("composite_density_envelope", ctx)
    errs = [f for f in findings if f.severity == Severity.ERROR]
    assert not errs


# ---------------------------------------------------------------------------
# tops_per_watt_envelope -- the "1 PetaOP @ 25 W" catch
# ---------------------------------------------------------------------------

def test_tops_per_watt_catches_one_petaop_at_25w(ctx):
    """The user's headline bug class. T256 on 16 nm at 25 W claiming
    1,000,000 INT8 TOPS = 40,000 TOPS/W >> 30 TOPS/W ceiling."""
    bad_perf = KPUTheoreticalPerformance(
        int8_tops=1_000_000.0, bf16_tflops=0.0, fp32_tflops=0.0,
    )
    bad_power = ctx.sku.power.model_copy(update={"tdp_watts": 25.0})
    bad_ctx = _ctx_with_sku(ctx, performance=bad_perf, power=bad_power)
    findings = _findings_for("tops_per_watt_envelope", bad_ctx)
    errs = [f for f in findings if f.severity == Severity.ERROR]
    assert errs, "expected ERROR for 1 PetaOP @ 25 W on 16 nm"
    assert "TOPS/W" in errs[0].message
    assert "ceiling" in errs[0].message


def test_tops_per_watt_warns_in_upper_band(ctx):
    """287 TOPS / 11 W on 16 nm = 26.1 TOPS/W -> in 80-100% of 30 ceiling."""
    bad_power = ctx.sku.power.model_copy(update={"tdp_watts": 11.0})
    bad_ctx = _ctx_with_sku(ctx, power=bad_power)
    findings = _findings_for("tops_per_watt_envelope", bad_ctx)
    assert any(
        f.severity == Severity.WARNING and "upper" in f.message
        for f in findings
    )


def test_tops_per_watt_warns_below_floor(ctx):
    """Very low TOPS/W suggests broken inputs (mislabeled precision)."""
    bad_perf = ctx.sku.performance.model_copy(update={"int8_tops": 1.0})
    bad_power = ctx.sku.power.model_copy(update={"tdp_watts": 100.0})
    bad_ctx = _ctx_with_sku(ctx, performance=bad_perf, power=bad_power)
    findings = _findings_for("tops_per_watt_envelope", bad_ctx)
    assert any(
        f.severity == Severity.WARNING and "below" in f.message
        for f in findings
    )


def test_tops_per_watt_passes_real_sku(ctx):
    findings = _findings_for("tops_per_watt_envelope", ctx)
    errs = [f for f in findings if f.severity == Severity.ERROR]
    assert not errs


# ---------------------------------------------------------------------------
# Cross-validator sanity
# ---------------------------------------------------------------------------

def test_run_all_returns_no_errors_for_real_sku(ctx):
    """With every validator registered, the real T256 SKU produces no
    ERROR findings."""
    findings = default_registry.run_all(ctx)
    errs = [f for f in findings if f.severity == Severity.ERROR]
    assert not errs


def test_run_category_filters_to_area_only(ctx):
    findings = default_registry.run_category(ctx, ValidatorCategory.AREA)
    assert all(f.category == ValidatorCategory.AREA for f in findings)
