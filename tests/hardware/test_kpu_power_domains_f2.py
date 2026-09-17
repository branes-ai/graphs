"""Phase F2 of the KPU heterogeneous-tile refactor (graphs#268): the
default per-cluster DVFS partition on the uniform SKUs.

From ``docs/designs/kpu-cluster-organization-for-dvfs-and-floorsweeping.md``:
k x k clusters of compute sites, each its own DVFS domain with its own
regulator and PLL, plus an uncore domain. The default declares no operating
points, so it adds structure without moving TDP, performance or area.
"""

from __future__ import annotations

import pytest
import yaml
from embodied_schemas import load_compute_products, load_process_nodes
from embodied_schemas.power_domain import PowerDomain

from graphs.hardware.kpu_access import kpu_block_of, kpu_die_of
from graphs.hardware.kpu_power_domains import (
    PowerDomainDefaultError,
    default_cluster_edge,
    default_power_domains,
    default_power_domains_for,
)
from graphs.hardware.kpu_power_model import (
    compute_thermal_profile_tdp_breakdown,
    is_legacy_shaped,
)
from graphs.hardware.kpu_sku_generator import (
    KPUSKUInputSpec,
    apply_default_power_domains,
    generate_kpu_sku,
    input_spec_from_compute_product,
)
from hardware.test_kpu_catalog_ids import LEGACY_KPU_SKU_IDS

NODES = load_process_nodes()
CATALOG = load_compute_products()


# ---------------------------------------------------------------------------
# Cluster size: the design's table
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rows, cols, edge, clusters", [
    (8, 8, 2, 16),    # T64: 4x4 would leave 4 clusters, too coarse
    (16, 8, 4, 8),    # T128
    (16, 16, 4, 16),  # T256
    (32, 16, 4, 32),  # T512
    (32, 24, 4, 48),  # T768, absent from the design's table
])
def test_cluster_edge_follows_the_design_table(rows, cols, edge, clusters):
    assert default_cluster_edge(rows, cols) == edge
    assert (rows // edge) * (cols // edge) == clusters


def test_the_table_wins_over_the_prose_rule_for_the_t128():
    """The design's prose rule ("the smallest cluster such that ... 8-64
    clusters") would pick 2x2 for the T128 -- 32 clusters, in range -- but
    its table and its "recommended k=4" say 4x4. The table wins."""
    assert (16 // 2) * (8 // 2) == 32  # the prose rule's pick is in range...
    assert default_cluster_edge(16, 8) == 4  # ...and is still not chosen


def test_a_grid_too_fine_for_4x4_grows_the_cluster():
    """The design's note for future large SKUs: more than 64 clusters is too
    many rails and PLLs, so the cluster grows."""
    assert (64 // 4) * (64 // 4) == 256
    assert default_cluster_edge(64, 64) == 8  # 64 clusters


@pytest.mark.parametrize("rows, cols, edge", [
    (10, 10, 2),  # 4 does not divide; 2x2 gives 25
    (18, 18, 6),  # 4 does not divide; 2x2 gives 81, too many; 6x6 gives 9
])
def test_a_grid_4_does_not_divide_tries_every_other_edge(rows, cols, edge):
    """When 4x4 does not divide the grid its count says nothing about the
    direction to go, so both finer and coarser edges are tried, finer first
    (CodeRabbit on #294: 18x18 was refused although 6x6 fits)."""
    assert rows % 4 or cols % 4
    assert default_cluster_edge(rows, cols) == edge


def test_a_grid_no_edge_divides_is_refused():
    with pytest.raises(PowerDomainDefaultError, match="divides"):
        default_cluster_edge(7, 7)


# ---------------------------------------------------------------------------
# The partition
# ---------------------------------------------------------------------------


def test_the_partition_tiles_the_mesh_exactly_once():
    domains = [PowerDomain.model_validate(d) for d in default_power_domains(16, 8, [])]
    clusters = [d for d in domains if d.kind.value == "cluster"]
    covered = [site for d in clusters for site in d.sites()]
    assert len(covered) == len(set(covered)) == 16 * 8


def test_every_cluster_has_its_own_rail_and_pll():
    domains = default_power_domains(8, 8, [])
    clusters = [d for d in domains if d["kind"] == "cluster"]
    assert len({d["rail_id"] for d in clusters}) == len(clusters)
    assert len({d["clock_domain_id"] for d in clusters}) == len(clusters)
    assert [d["domain_id"] for d in clusters][:3] == ["c_0_0", "c_0_1", "c_0_2"]


def test_the_uncore_holds_exactly_the_silicon_outside_the_mesh():
    t64 = CATALOG["kpu_t64_32x32_lp5x4_16nm_tsmc_ffp"]
    blocks = kpu_die_of(t64).silicon_bin.blocks
    (uncore,) = [d for d in default_power_domains(8, 8, blocks) if d["kind"] == "uncore"]
    # Mesh-scaled blocks (per PE, per KiB, per router) are the clusters'.
    assert uncore["members"] == ["memory_phys", "io_pads", "control_logic"]
    assert uncore["rail_id"] == "vdd_uncore"


# ---------------------------------------------------------------------------
# The catalog carries exactly the default, and it changes no model output
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sku_id", LEGACY_KPU_SKU_IDS)
def test_the_catalog_carries_exactly_the_default(sku_id):
    """embodied-schemas 0.13.0 was authored from this helper; if the two
    diverge, one of them was edited by hand."""
    cp = CATALOG[sku_id]
    block = kpu_block_of(cp)
    want = [
        PowerDomain.model_validate(d)
        for d in default_power_domains(
            block.noc.mesh_rows, block.noc.mesh_cols, kpu_die_of(cp).silicon_bin.blocks
        )
    ]
    assert list(block.power_domains) == want


@pytest.mark.parametrize("sku_id", LEGACY_KPU_SKU_IDS)
def test_the_default_changes_no_model_output(sku_id):
    """No operating points, so every cluster runs at its profile's Vdd and
    clock: the power model keeps its original formula, and TDP is compared
    unrounded -- the convenience wrapper rounds to 0.1 W, which would hide a
    small change."""
    cp = CATALOG[sku_id]
    with_domains = input_spec_from_compute_product(cp)
    data = with_domains.model_dump(mode="json")
    data["kpu_architecture"]["power_domains"] = None
    without = KPUSKUInputSpec.model_validate(data)
    node = NODES[kpu_die_of(cp).process_node_id]

    a = generate_kpu_sku(without, process_nodes=NODES)
    b = generate_kpu_sku(with_domains, process_nodes=NODES)
    assert a.dies[0].die_size_mm2 == b.dies[0].die_size_mm2
    assert a.dies[0].transistors_billion == b.dies[0].transistors_billion
    assert a.performance == b.performance
    assert a.power == b.power
    for profile in with_domains.thermal_profiles:
        assert is_legacy_shaped(with_domains, profile)
        assert compute_thermal_profile_tdp_breakdown(
            with_domains, profile, node
        ).total_tdp_w == compute_thermal_profile_tdp_breakdown(
            without, profile, node
        ).total_tdp_w


# ---------------------------------------------------------------------------
# Scope: uniform KPUs only
# ---------------------------------------------------------------------------


def _uniform_spec_without_domains():
    spec = input_spec_from_compute_product(CATALOG["kpu_t64_32x32_lp5x4_16nm_tsmc_ffp"])
    data = spec.model_dump(mode="json")
    data["kpu_architecture"]["power_domains"] = None
    return KPUSKUInputSpec.model_validate(data)


def test_apply_default_power_domains_on_a_uniform_spec():
    spec = apply_default_power_domains(_uniform_spec_without_domains())
    assert len(spec.kpu_architecture.power_domains) == 17  # 16 clusters + uncore


def test_the_default_refuses_an_architecture_that_already_has_domains():
    spec = input_spec_from_compute_product(CATALOG["kpu_t64_32x32_lp5x4_16nm_tsmc_ffp"])
    with pytest.raises(ValueError, match="already declares power domains"):
        apply_default_power_domains(spec)


@pytest.mark.parametrize("sku_id", [
    "kpu_h64_auto1_lp5x4_16nm_tsmc_ffp", "kpu_h64_auto1_lp5x4_7nm_tsmc_hpc",
])
def test_the_default_refuses_a_heterogeneous_kpu(sku_id):
    """A heterogeneous checkerboard partitions differently: the refactor
    plan gives each fixed-function class its own gateable domain."""
    cp = CATALOG[sku_id]
    with pytest.raises(PowerDomainDefaultError, match="uniform"):
        default_power_domains_for(kpu_block_of(cp), kpu_die_of(cp).silicon_bin.blocks)


def test_generate_cli_reproduces_the_catalog_partition(cli_runner, tmp_path):
    """The flag is how the embodied-schemas data was authored, so running it
    on a domain-less T64 must give back the catalog's partition."""
    from pathlib import Path

    spec_path = tmp_path / "t64.yaml"
    spec_path.write_text(
        yaml.safe_dump(_uniform_spec_without_domains().model_dump(mode="json"), sort_keys=False),
        encoding="utf-8",
    )
    out = tmp_path / "out.yaml"
    cli = Path(__file__).resolve().parents[2] / "cli" / "generate_kpu_sku.py"
    rc, _, err = cli_runner(
        cli, ["--input", str(spec_path), "--default-power-domains", "--output", str(out)]
    )
    assert rc == 0, err
    got = yaml.safe_load(out.read_text(encoding="utf-8"))["dies"][0]["blocks"][0]["power_domains"]
    catalog = kpu_block_of(CATALOG["kpu_t64_32x32_lp5x4_16nm_tsmc_ffp"]).power_domains
    assert [PowerDomain.model_validate(d) for d in got] == list(catalog)


# ---------------------------------------------------------------------------
# The implicit site plan: what makes the partition visible
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sku_id", LEGACY_KPU_SKU_IDS)
def test_implicit_plan_is_the_legacy_row_major_fill(sku_id):
    """A view, not a placement: place_tiles still declines a uniform SKU so
    the floorplan keeps its geometry, but the implicit plan gives the site
    views something to draw the clusters on."""
    from graphs.hardware.kpu_checkerboard_placer import (
        implicit_site_plan,
        place_tiles,
        site_power_domains,
    )

    block = kpu_block_of(CATALOG[sku_id])
    assert place_tiles(block) is None
    plan = implicit_site_plan(block)
    assert plan.mode == "implicit"
    assert (plan.rows, plan.cols) == (block.noc.mesh_rows, block.noc.mesh_cols)
    assert len(plan.placements) == block.total_tiles and plan.spare_sites == ()

    # Row-major in declaration order, like the legacy floorplan's walk.
    first = block.tiles[0]
    assert all(
        p.tile_class_id == first.tile_class_id
        for p in plan.placements[: first.num_tiles]
    )
    assert [(p.row, p.col) for p in plan.placements[:3]] == [(0, 0), (0, 1), (0, 2)]

    # Every site lands in exactly one cluster domain.
    domains = site_power_domains(block, plan)
    assert set(domains) == set(plan.site_owner())
    assert all(d.startswith("c_") for d in domains.values())


def test_implicit_plan_declines_a_heterogeneous_block():
    from graphs.hardware.kpu_checkerboard_placer import implicit_site_plan

    block = kpu_block_of(CATALOG["kpu_h64_auto1_lp5x4_16nm_tsmc_ffp"])
    assert implicit_site_plan(block) is None  # it has a checkerboard to place


def test_the_heterogeneous_fixture_does_not_inherit_the_t64_partition():
    """build_heterogeneous_kpu starts from the T64 catalog entry. When the
    T64 gained its default partition the fixture silently inherited 16 2x2
    clusters that happened to validate over its 8x8 checkerboard; hundreds
    of fixture tests passed over the changed fixture. It must stay exactly
    what it was."""
    from graphs.hardware.kpu_hetero_fixture import BASE_SKU_ID, build_heterogeneous_kpu

    assert kpu_block_of(CATALOG[BASE_SKU_ID]).power_domains  # the chassis has them
    assert kpu_block_of(build_heterogeneous_kpu()).power_domains is None
