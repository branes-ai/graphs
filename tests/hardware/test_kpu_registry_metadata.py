"""KPU mapper-registry metadata must match the models the factories build.

The registry's ``description`` / ``default_tdp_w`` / ``memory_gb`` for the KPU
entries are hand-written, and they drifted from the catalog. T128 and T256
were described as 24x24 / 16x16 PE arrays, and T256 was listed at 10 W / 8 GB
against a 30 W / 32 GB catalog SKU. ``default_tdp_w`` is functional:
``list_mappers_by_tdp_range`` filters on it. These tests pin every KPU entry
to the resource model and PhysicalSpec its factory produces (graphs#268 A3).
"""

from __future__ import annotations

import pytest

from graphs.hardware.mappers import (
    get_mapper_by_name,
    get_mapper_info,
    list_mappers_by_category,
    list_mappers_by_tdp_range,
)

KPU_MAPPERS = sorted(list_mappers_by_category("kpu"))


def _built(name: str):
    mapper = get_mapper_by_name(name)
    rm = mapper.resource_model
    profile = rm.thermal_operating_points[rm.default_thermal_profile]
    specs = next(iter(profile.performance_specs.values())).compute_resource.tile_specializations
    return mapper, rm, profile, specs


def test_registry_has_the_kpu_family():
    assert KPU_MAPPERS == sorted(
        f"Stillwater-KPU-T{n}" for n in (64, 128, 256, 768)
    )


@pytest.mark.parametrize("name", KPU_MAPPERS)
def test_default_tdp_and_memory_match_the_model(name):
    info = get_mapper_info(name)
    _, rm, profile, _ = _built(name)
    assert info["default_tdp_w"] == pytest.approx(profile.tdp_watts), (
        f"{name}: registry default_tdp_w={info['default_tdp_w']} W, model's "
        f"default profile {rm.default_thermal_profile!r} is {profile.tdp_watts} W"
    )
    assert info["memory_gb"] == pytest.approx(rm.main_memory / 2**30), name


@pytest.mark.parametrize("name", KPU_MAPPERS)
def test_description_matches_the_model(name):
    description = get_mapper_info(name)["description"]
    mapper, rm, _, specs = _built(name)
    assert f"{rm.compute_units} tiles" in description, description
    for spec in specs:
        rows, cols = spec.array_dimensions
        assert f"{rows}x{cols}" in description, (
            f"{name}: tile class {spec.tile_type} is {rows}x{cols}; description "
            f"{description!r} does not mention it"
        )
    assert mapper.physical_spec.process_node_name in description, description


def test_tdp_range_filter_uses_catalog_tdps():
    """A <=15 W query must not return the 30 W T256 or the 60 W T768."""
    low_power = set(list_mappers_by_tdp_range(0, 15)) & set(KPU_MAPPERS)
    assert low_power == {"Stillwater-KPU-T64", "Stillwater-KPU-T128"}
