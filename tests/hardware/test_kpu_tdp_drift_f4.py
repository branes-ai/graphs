"""Phase F4 of the KPU heterogeneous-tile refactor (graphs#268): the
declared TDP must be what the power model computes.

Nothing checked this, and it drifted. Every ``7nm_tsmc_hpc`` SKU's ``lp``
and ``default`` profiles declared more than the model computed -- the
T512's ``lp`` claimed 10.3 W against a computed 8.9 W -- because their Vdd
values were never re-tuned after leakage gained its Vdd scaling
(``leakage_vdd_exponent``, 4.5 on tsmc_n7).

The tell was which profile looked healthy: ``boost`` sits exactly at the
node's nominal Vdd, where that scaling is a no-op, so the one profile that
could not reveal the problem was the one that passed.
"""

from __future__ import annotations

import pytest
from embodied_schemas import (
    ComputeProduct,
    load_compute_products,
    load_cooling_solutions,
    load_process_nodes,
)

from graphs.hardware.kpu_access import kpu_die_of
from graphs.hardware.kpu_power_model import compute_thermal_profile_tdp_w
from graphs.hardware.kpu_sku_generator import input_spec_from_compute_product
from graphs.hardware.sku_validators import (
    Severity,
    build_context_for_kpu,
    default_registry,
    load_validators,
)

from hardware.test_kpu_catalog_ids import ALL_KPU_SKU_IDS

NODES = load_process_nodes()
CATALOG = load_compute_products()
VALIDATOR = "declared_tdp_matches_model"

#: The four SKUs F4 found drifted, and the profiles that drifted on each.
DRIFTED = {
    "kpu_t64_32x32_lp5x4_7nm_tsmc_hpc": ("lp", "default"),
    "kpu_t128_32x32_lp5x8_7nm_tsmc_hpc": ("lp", "default"),
    "kpu_t256_32x32_lp5x16_7nm_tsmc_hpc": ("lp", "default"),
    "kpu_t512_32x32_lp5x32_7nm_tsmc_hpc": ("lp", "default"),
}


@pytest.fixture(scope="module", autouse=True)
def _validators():
    load_validators()


def _context(sku_id: str, cp=None):
    return build_context_for_kpu(
        sku_id,
        kpus={sku_id: cp or CATALOG[sku_id]},
        process_nodes=NODES,
        cooling_solutions=load_cooling_solutions(),
    )


def _check(sku_id: str, cp=None):
    return default_registry._validators[VALIDATOR].check(_context(sku_id, cp))


@pytest.mark.parametrize("sku_id", ALL_KPU_SKU_IDS)
def test_every_catalog_sku_declares_the_tdp_the_model_computes(sku_id):
    """The contract F4 restores, across the whole catalog."""
    cp = CATALOG[sku_id]
    spec = input_spec_from_compute_product(cp)
    node = NODES[kpu_die_of(cp).process_node_id]
    for profile in cp.power.thermal_profiles:
        computed = compute_thermal_profile_tdp_w(spec, profile, node)
        assert computed == pytest.approx(profile.tdp_watts, abs=0.05), (
            f"{sku_id} {profile.name}: declared {profile.tdp_watts}, "
            f"computed {computed}"
        )


@pytest.mark.parametrize("sku_id", ALL_KPU_SKU_IDS)
def test_the_validator_is_silent_on_the_catalog(sku_id):
    assert _check(sku_id) == []


@pytest.mark.parametrize("sku_id", sorted(DRIFTED))
def test_the_validator_catches_the_drift_it_was_written_for(sku_id):
    """Put the untuned round Vdd values back and the validator must fire.

    Without this the validator could be silently inert -- it passes on the
    catalog either way.
    """
    data = CATALOG[sku_id].model_dump(mode="json")
    for profile in data["power"]["thermal_profiles"]:
        profile["vdd_v"] = {"lp": 0.5, "default": 0.65, "boost": 0.75}[profile["name"]]
    findings = _check(sku_id, ComputeProduct.model_validate(data))

    assert {f.profile for f in findings} == set(DRIFTED[sku_id])
    # boost sits at nominal Vdd, where the leakage scaling is a no-op, so
    # it stays clean -- which is exactly why the drift went unnoticed.
    assert "boost" not in {f.profile for f in findings}
    # The model computes *less* than declared, and the message says so.
    for finding in findings:
        assert "below" in finding.message
        assert "re-tune vdd_v" in finding.message
    assert Severity.ERROR in {f.severity for f in findings}


def test_the_retuned_vdds_sit_below_nominal():
    """The re-tune raised Vdd toward nominal, not past it: these are
    low-power profiles and must stay under the node's nominal 0.75 V."""
    node = NODES["tsmc_n7"]
    assert node.nominal_vdd_v == 0.75
    assert node.leakage_vdd_exponent == 4.5
    for sku_id, names in DRIFTED.items():
        profiles = {p.name: p for p in CATALOG[sku_id].power.thermal_profiles}
        for name in names:
            vdd = profiles[name].vdd_v
            assert 0.50 < vdd < node.nominal_vdd_v, (sku_id, name, vdd)
        assert profiles["boost"].vdd_v == node.nominal_vdd_v
