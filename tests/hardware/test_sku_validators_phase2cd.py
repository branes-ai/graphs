"""Phase 2c + 2d validator tests: thermal_hotspot, electromigration.

Like Phase 2b's tests, these mutate a real SKU in-memory to exercise
each branch of the new validators.
"""

from __future__ import annotations

import pytest

from embodied_schemas import KPUTheoreticalPerformance

from graphs.hardware.sku_validators import (
    Severity,
    ValidatorCategory,
    ValidatorContext,
    build_context_for_kpu,
    default_registry,
    load_validators,
)


load_validators()


@pytest.fixture
def ctx() -> ValidatorContext:
    return build_context_for_kpu("stillwater_kpu_t256")


def _ctx_with(ctx, **sku_updates) -> ValidatorContext:
    new_sku = ctx.sku.model_copy(update=sku_updates)
    return ValidatorContext(
        sku=new_sku,
        process_node=ctx.process_node,
        cooling_solutions=ctx.cooling_solutions,
    )


def _findings_for(name: str, ctx: ValidatorContext):
    return default_registry.run_one(name, ctx)


# ---------------------------------------------------------------------------
# thermal_hotspot
# ---------------------------------------------------------------------------

def test_thermal_hotspot_catches_tdp_exceeding_cooling_envelope(ctx):
    """A profile with TDP > cooling.max_total_w."""
    profiles = list(ctx.sku.power.thermal_profiles)
    # Set 50W profile's TDP to 999W -- exceeds active_fan's 100W envelope.
    for i, p in enumerate(profiles):
        if p.name == "50W":
            profiles[i] = p.model_copy(update={"tdp_watts": 999.0})
    bad_power = ctx.sku.power.model_copy(update={"thermal_profiles": profiles})
    bad_ctx = _ctx_with(ctx, power=bad_power)
    findings = _findings_for("thermal_hotspot", bad_ctx)
    assert any(
        f.severity == Severity.ERROR
        and "exceeds cooling solution" in f.message
        and f.profile == "50W"
        for f in findings
    )


def test_thermal_hotspot_catches_dangerous_density(ctx):
    """A profile paired with grossly-undersized cooling triggers the
    'fundamentally hot' ERROR (>5x ceiling)."""
    profiles = list(ctx.sku.power.thermal_profiles)
    for i, p in enumerate(profiles):
        if p.name == "30W":
            profiles[i] = p.model_copy(
                update={"cooling_solution_id": "passive_fanless"}
            )
    bad_power = ctx.sku.power.model_copy(update={"thermal_profiles": profiles})
    bad_ctx = _ctx_with(ctx, power=bad_power)
    findings = _findings_for("thermal_hotspot", bad_ctx)
    errs = [
        f for f in findings
        if f.severity == Severity.ERROR
        and "fundamentally hot" in f.message
        and f.profile == "30W"
    ]
    assert errs, "expected ERROR for 30W on passive_fanless"


def test_thermal_hotspot_emits_dvfs_throttle_warning(ctx):
    """The actionable WARNING the user asked for: 'DVFS throttle to N%'."""
    findings = _findings_for("thermal_hotspot", ctx)
    warns = [f for f in findings if f.severity == Severity.WARNING]
    assert warns, "expected at least one WARNING with DVFS throttle"
    assert all("DVFS" in f.message for f in warns)
    assert all("throttling to" in f.message for f in warns)


def test_thermal_hotspot_petaop_at_25w_fires(ctx):
    """The headline catch: a profile claiming PetaOP at very low TDP
    triggers thermal_hotspot from the PE-block density side."""
    bad_perf = KPUTheoreticalPerformance(
        int8_tops=1_000_000.0, bf16_tflops=0.0, fp32_tflops=0.0,
    )
    bad_power = ctx.sku.power.model_copy(update={"tdp_watts": 25.0})
    bad_ctx = _ctx_with(ctx, performance=bad_perf, power=bad_power)
    # tops_per_watt_envelope catches this independently; verify thermal
    # also flags the PE-block density at peak.
    findings = _findings_for("thermal_hotspot", bad_ctx)
    block_warns = [
        f for f in findings
        if f.block and f.block.startswith("pe_")
        and f.severity in (Severity.WARNING, Severity.ERROR)
    ]
    assert block_warns


# ---------------------------------------------------------------------------
# electromigration
# ---------------------------------------------------------------------------

def test_em_skips_when_node_has_no_em_data(ctx):
    """A node without em_j_max_by_temp_c emits one INFO and skips."""
    bad_node = ctx.process_node.model_copy(update={"em_j_max_by_temp_c": {}})
    bad_ctx = ValidatorContext(
        sku=ctx.sku, process_node=bad_node,
        cooling_solutions=ctx.cooling_solutions,
    )
    findings = _findings_for("electromigration", bad_ctx)
    assert len(findings) == 1
    assert findings[0].severity == Severity.INFO
    assert "no em_j_max_by_temp_c" in findings[0].message


def test_em_passes_real_sku(ctx):
    """Real SKUs with sustained-current model + 12-layer stack should
    have no ERROR / WARNING findings."""
    findings = _findings_for("electromigration", ctx)
    bad = [f for f in findings if f.severity != Severity.INFO]
    assert not bad, f"unexpected EM findings: {[f.render_one_line() for f in bad]}"


def test_em_catches_outrageous_design(ctx):
    """A SKU claiming TDP many times what the metal stack can carry
    triggers EM ERROR. Set TDP to 100,000 W to force a violation."""
    bad_power = ctx.sku.power.thermal_profiles[-1].model_copy(
        update={"tdp_watts": 100_000.0}
    )
    profiles = list(ctx.sku.power.thermal_profiles)
    profiles[-1] = bad_power
    new_power = ctx.sku.power.model_copy(update={"thermal_profiles": profiles})
    bad_ctx = _ctx_with(ctx, power=new_power)
    findings = _findings_for("electromigration", bad_ctx)
    assert any(
        f.severity == Severity.ERROR
        and "EM lifetime" in f.message
        for f in findings
    )


# ---------------------------------------------------------------------------
# Cross-validator sanity for all 4 real SKUs
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sku_id", [
    "stillwater_kpu_t64",
    "stillwater_kpu_t128",
    "stillwater_kpu_t256",
    "stillwater_kpu_t768",
])
def test_real_skus_have_no_errors_after_phase2cd(sku_id):
    """All four hand-authored SKUs should have zero ERROR findings
    once Phase 2c+d validators are added."""
    ctx = build_context_for_kpu(sku_id)
    findings = default_registry.run_all(ctx)
    errors = [f for f in findings if f.severity == Severity.ERROR]
    assert not errors, (
        f"{sku_id} produced ERROR findings:\n"
        + "\n".join(f.render_one_line() for f in errors)
    )


def test_run_category_thermal_only(ctx):
    findings = default_registry.run_category(ctx, ValidatorCategory.THERMAL)
    assert all(f.category == ValidatorCategory.THERMAL for f in findings)


def test_run_category_reliability_only(ctx):
    findings = default_registry.run_category(
        ctx, ValidatorCategory.RELIABILITY
    )
    assert all(f.category == ValidatorCategory.RELIABILITY for f in findings)


def test_total_validator_count_is_nine_after_phase2cd():
    """Phase 2c added thermal_hotspot, Phase 2d added electromigration --
    we should have 9 validators total (7 from Phase 2b + 2 from 2c+d)."""
    assert len(default_registry) == 9
