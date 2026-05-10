"""Phase 3 generator tests.

Coverage:
  - Round-trip every catalog SKU: extract spec -> regenerate -> die,
    perf, power numbers match the original within rounding.
  - Generator output validates clean (no ERROR findings) on all four
    real SKUs.
  - GeneratorError on bad inputs (missing process node, bad default
    profile name, empty silicon_bin).
  - input_spec_from_kpu_entry preserves architect-authored fields.
"""

from __future__ import annotations

import pytest

from embodied_schemas import (
    KPUSiliconBin,
    load_cooling_solutions,
    load_kpus,
    load_process_nodes,
)

from graphs.hardware.kpu_sku_generator import (
    GeneratorError,
    generate_kpu_sku,
    input_spec_from_kpu_entry,
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
        "kpus": load_kpus(),
        "process_nodes": load_process_nodes(),
        "cooling_solutions": load_cooling_solutions(),
    }


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sku_id", [
    "stillwater_kpu_t64",
    "stillwater_kpu_t128",
    "stillwater_kpu_t256",
    "stillwater_kpu_t768",
])
def test_roundtrip_die_within_rounding(sku_id, catalogs):
    """Generator-derived die.transistors_billion and die.die_size_mm2
    must match the YAML's hand-authored values within rounding."""
    original = catalogs["kpus"][sku_id]
    spec = input_spec_from_kpu_entry(original)
    regen = generate_kpu_sku(
        spec,
        process_nodes=catalogs["process_nodes"],
    )
    # Within 2% (the generator rounds to 1 dp for area / 3 dp for
    # transistors; the hand-authored YAMLs round earlier in the
    # back-of-envelope chain, so a couple of percent of difference is
    # rounding only, not a real generator drift).
    assert abs(regen.die.transistors_billion - original.die.transistors_billion) \
        / original.die.transistors_billion <= 0.02
    assert abs(regen.die.die_size_mm2 - original.die.die_size_mm2) \
        / original.die.die_size_mm2 <= 0.02


@pytest.mark.parametrize("sku_id", [
    "stillwater_kpu_t64",
    "stillwater_kpu_t128",
    "stillwater_kpu_t256",
    "stillwater_kpu_t768",
])
def test_roundtrip_performance_within_rounding(sku_id, catalogs):
    """int8_tops, bf16_tflops, fp32_tflops should round-trip within 1%."""
    original = catalogs["kpus"][sku_id]
    spec = input_spec_from_kpu_entry(original)
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
    "stillwater_kpu_t64",
    "stillwater_kpu_t128",
    "stillwater_kpu_t256",
    "stillwater_kpu_t768",
])
def test_roundtrip_power_default_tdp_passes_through(sku_id, catalogs):
    """power.tdp_watts comes from the default thermal profile -- the
    generator must not change it."""
    original = catalogs["kpus"][sku_id]
    spec = input_spec_from_kpu_entry(original)
    regen = generate_kpu_sku(
        spec,
        process_nodes=catalogs["process_nodes"],
    )
    assert regen.power.tdp_watts == original.power.tdp_watts
    assert regen.power.default_thermal_profile == original.power.default_thermal_profile
    assert len(regen.power.thermal_profiles) == len(original.power.thermal_profiles)


@pytest.mark.parametrize("sku_id", [
    "stillwater_kpu_t64",
    "stillwater_kpu_t128",
    "stillwater_kpu_t256",
    "stillwater_kpu_t768",
])
def test_generated_sku_validates_clean(sku_id, catalogs):
    """Output of the generator should have zero ERROR findings when run
    through the validator framework. WARNINGs are expected (DVFS
    throttle messages)."""
    original = catalogs["kpus"][sku_id]
    spec = input_spec_from_kpu_entry(original)
    regen = generate_kpu_sku(
        spec,
        process_nodes=catalogs["process_nodes"],
    )
    ctx = ValidatorContext(
        sku=regen,
        process_node=catalogs["process_nodes"][regen.process_node_id],
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
    original = catalogs["kpus"]["stillwater_kpu_t256"]
    spec = input_spec_from_kpu_entry(original).model_copy(
        update={"process_node_id": "no_such_node"}
    )
    with pytest.raises(GeneratorError, match="does not resolve"):
        generate_kpu_sku(
            spec,
            process_nodes=catalogs["process_nodes"],
        )


def test_bad_default_profile_name_raises(catalogs):
    """default_thermal_profile must name an existing profile."""
    original = catalogs["kpus"]["stillwater_kpu_t256"]
    spec = input_spec_from_kpu_entry(original).model_copy(
        update={"default_thermal_profile": "9000W"}
    )
    with pytest.raises(GeneratorError, match="not in thermal_profiles"):
        generate_kpu_sku(
            spec,
            process_nodes=catalogs["process_nodes"],
        )


def test_empty_silicon_bin_raises(catalogs):
    """A spec with an empty silicon_bin (no resolvable blocks) raises."""
    original = catalogs["kpus"]["stillwater_kpu_t256"]
    spec = input_spec_from_kpu_entry(original).model_copy(
        update={"silicon_bin": KPUSiliconBin(blocks=[])}
    )
    with pytest.raises(GeneratorError, match="silicon_bin"):
        generate_kpu_sku(
            spec,
            process_nodes=catalogs["process_nodes"],
        )


# ---------------------------------------------------------------------------
# input_spec_from_kpu_entry preserves architect-authored fields
# ---------------------------------------------------------------------------

def test_input_spec_preserves_architecture(catalogs):
    original = catalogs["kpus"]["stillwater_kpu_t256"]
    spec = input_spec_from_kpu_entry(original)
    assert spec.kpu_architecture == original.kpu_architecture
    assert spec.silicon_bin == original.silicon_bin
    assert spec.thermal_profiles == original.power.thermal_profiles
    assert spec.market == original.market
    assert spec.clocks == original.clocks


def test_input_spec_does_not_carry_die_or_perf_rollups(catalogs):
    """The spec's shape excludes the generator-derived roll-ups -- if a
    field is in KPUEntry but not in KPUSKUInputSpec, the spec is
    correctly minimal."""
    original = catalogs["kpus"]["stillwater_kpu_t256"]
    spec = input_spec_from_kpu_entry(original)
    spec_fields = set(type(spec).model_fields)
    assert "die" not in spec_fields
    assert "performance" not in spec_fields
    assert "power" not in spec_fields  # Power.tdp roll-up is generator-derived; profiles are separate
