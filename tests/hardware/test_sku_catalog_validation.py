"""Phase 6 CI gate: every SKU YAML in the catalog must validate cleanly.

The Phase 2 validator framework + Phase 4b factories tested specific
SKUs by id (T64/T128/T256/T768) -- a future ``stillwater_kpu_t1024``
or third-party KPU SKU added to the catalog wouldn't trip those tests.

This module parametrizes over ``load_kpus().keys()`` so every SKU in
the embodied-schemas catalog is automatically gated. A new SKU lands
without touching the test list; a regression in a new SKU surfaces
on its own commit.

What's enforced:
  * Every SKU resolves: ``process_node_id`` -> a known ProcessNode,
    ``cooling_solution_id`` (per thermal profile) -> a known
    CoolingSolution.
  * Every validator in ``default_registry`` runs cleanly: no ERROR
    findings.
  * KPUMapper construction succeeds (the live mapper-factory path,
    so e.g. mapper-specific attributes the loader doesn't set are
    still populated).

WARNING-severity findings are allowed (most are the actionable "DVFS
throttle to N%" messages from the thermal-hotspot validator). The
gate fails only on ERROR. Use ``--strict`` on the CLI to also fail
on warnings if you need that locally.
"""

from __future__ import annotations

import pytest

from embodied_schemas import load_kpus

from graphs.hardware.sku_validators import (
    Severity,
    build_context_for_kpu,
    default_registry,
    load_validators,
)


# Auto-load validators once per test session.
load_validators()


def _all_kpu_sku_ids() -> list[str]:
    """Discover every KPU SKU id in the embodied-schemas catalog.

    Cached at module import so the parametrize list is stable across
    a single test session.
    """
    return sorted(load_kpus().keys())


@pytest.mark.parametrize("sku_id", _all_kpu_sku_ids())
def test_kpu_catalog_entry_has_no_validator_errors(sku_id):
    """Every KPU YAML in ``embodied-schemas:data/kpus/**`` must pass
    every registered validator with zero ERROR findings.

    The Phase 6 catalog gate. New SKU YAMLs auto-discover; the test
    list does not need maintenance.
    """
    ctx = build_context_for_kpu(sku_id)
    findings = default_registry.run_all(ctx)
    errors = [f for f in findings if f.severity == Severity.ERROR]
    assert not errors, (
        f"{sku_id} produced ERROR findings (Phase 6 catalog gate):\n"
        + "\n".join(f.render_one_line() for f in errors)
    )


@pytest.mark.parametrize("sku_id", _all_kpu_sku_ids())
def test_kpu_catalog_entry_constructs_mapper(sku_id):
    """Every catalog entry must produce a working KPUMapper. Catches
    cases where the YAML validates against the schema but a downstream
    consumer (the mapper, the energy model, the validator framework's
    ``ValidatorContext``) still chokes on it.
    """
    from graphs.hardware.mappers.accelerators.kpu import KPUMapper
    from graphs.hardware.models.accelerators.kpu_yaml_loader import (
        load_kpu_resource_model_from_yaml,
    )

    rm = load_kpu_resource_model_from_yaml(sku_id)
    mapper = KPUMapper(rm)
    assert mapper.num_tiles > 0
    assert mapper.scratchpad_per_tile > 0


def test_kpu_catalog_is_non_empty():
    """Sanity check: at least one SKU YAML lives in the catalog. If
    this fails the parametrized tests above run zero cases and
    silently pass (pytest treats empty parametrize as a skip)."""
    sku_ids = _all_kpu_sku_ids()
    assert sku_ids, "embodied-schemas catalog has zero KPU SKUs"
    # Sanity: the four Stillwater SKUs Phase 4b authored should be present.
    for required in (
        "stillwater_kpu_t64", "stillwater_kpu_t128",
        "stillwater_kpu_t256", "stillwater_kpu_t768",
    ):
        assert required in sku_ids, (
            f"{required!r} missing from catalog -- Phase 4b SKUs must "
            f"remain in the catalog. Available: {sku_ids}"
        )
