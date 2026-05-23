"""PhysicalSpec regression pins for the 3 AMD chiplet datacenter SKUs.

Sprint #245 PR 6 (graphs side). Pins downstream byte-identical
PhysicalSpec headline values across the embodied-schemas v13 IOBlock
multi-die rewrites:

  - EPYC 9654 Genoa  -- embodied-schemas#80, sprint #245 PR 3
  - EPYC 9754 Bergamo -- embodied-schemas#81, sprint #245 PR 4
  - EPYC 9965 Turin Dense -- embodied-schemas#82, sprint #245 PR 5

The three SKUs went from a "single virtual die" representation to
proper multi-die layouts (compute die + IO die joined by IFOP
interconnects, with the IO die populating the new v13 IOBlock kind).
This is invisible to the downstream graphs ``PhysicalSpec`` loader,
which sums ``die_size_mm2`` and ``transistors_billion`` across
``dies[]`` -- but a regression test makes the invariance explicit and
catches any future drift in the loader's aggregation behavior.

The tests also exercise the now-visible per-die ``process_node_id``
distinction at the schema level: prior to sprint #245, every chiplet
YAML had a single virtual die with one process node (collapsing the
AMD CCDs/IOD process-node split into the CCD node). Post-sprint, the
IO die carries its own ``process_node_id`` (TSMC N6 for the IOD vs
TSMC N5 / N4P for the CCDs). The schema-level per-die assertions
pin this distinction.

See ``.claude/decisions/DECISION-2026-05-21-001.yaml`` Question A
for the architectural context.
"""

from __future__ import annotations

import pytest

from graphs.hardware.mappers import get_mapper_by_name


# (mapper_name, expected_die_size_mm2, expected_transistors_billion,
#  expected_num_dies, expected_chiplet)
EPYC_PHYSICAL_SPEC_PINS = [
    # EPYC 9654 Genoa: 12 Zen 4 CCDs (792 mm^2 / 78.84 B) + 1 Genoa IOD
    # (397 mm^2 / 12 B) = 1189 mm^2 / 90.84 B. num_dies = 13 chiplets
    # (12 CCDs + 1 IOD). Byte-identical across the multi-die rewrite.
    ("AMD-EPYC-9654", 1189.0, 90.84, 13, True),
    # EPYC 9754 Bergamo: 8 Zen 4c CCDs (581.6 mm^2 / 77.6 B) + 1 Genoa
    # IOD (397 mm^2 / 12 B) = 978.6 mm^2 / 89.6 B. num_dies = 9
    # (8 CCDs + 1 shared Genoa IOD). Same physical IOD as 9654.
    ("AMD-EPYC-9754", 978.6, 89.6, 9, True),
    # EPYC 9965 Turin Dense: 12 Zen 5c CCDs (876 mm^2 / 132 B) + 1
    # *new* Turin IOD (400 mm^2 / 13 B) = 1276 mm^2 / 145 B. num_dies =
    # 13. Distinct Turin IOD silicon (DDR5-6000, CXL 2.0).
    ("AMD-EPYC-Turin", 1276.0, 145.0, 13, True),
]


@pytest.mark.parametrize(
    "mapper_name,expected_area,expected_tx,expected_num_dies,expected_chiplet",
    EPYC_PHYSICAL_SPEC_PINS,
    ids=[entry[0] for entry in EPYC_PHYSICAL_SPEC_PINS],
)
def test_epyc_physical_spec_headline_values_byte_identical(
    mapper_name: str,
    expected_area: float,
    expected_tx: float,
    expected_num_dies: int,
    expected_chiplet: bool,
):
    """PhysicalSpec headline values must remain byte-identical across
    the v13 multi-die rewrites (embodied-schemas#80 / #81 / #82).

    The graphs ``PhysicalSpec`` loader sums ``die_size_mm2`` and
    ``transistors_billion`` across ``dies[]``, so multi-die rewrites
    that preserve their sums are invisible at the PhysicalSpec level.
    This test pins that the sums are preserved.
    """
    mapper = get_mapper_by_name(mapper_name)
    assert mapper is not None, f"mapper {mapper_name!r} not in registry"

    ps = mapper.physical_spec
    assert ps is not None, f"{mapper_name} has no physical_spec"

    assert ps.die_size_mm2 == pytest.approx(expected_area), (
        f"{mapper_name}: die_size_mm2 drift -- expected {expected_area} "
        f"(sum across dies[]), got {ps.die_size_mm2}"
    )
    assert ps.transistors_billion == pytest.approx(expected_tx), (
        f"{mapper_name}: transistors_billion drift -- expected "
        f"{expected_tx} (sum across dies[]), got {ps.transistors_billion}"
    )
    assert ps.num_dies == expected_num_dies, (
        f"{mapper_name}: num_dies (physical chiplet count) drift -- "
        f"expected {expected_num_dies}, got {ps.num_dies}"
    )
    assert ps.is_chiplet is expected_chiplet, (
        f"{mapper_name}: is_chiplet drift -- expected {expected_chiplet}, "
        f"got {ps.is_chiplet}"
    )


@pytest.mark.parametrize(
    "mapper_name",
    [entry[0] for entry in EPYC_PHYSICAL_SPEC_PINS],
)
def test_epyc_density_in_plausible_range(mapper_name: str):
    """Sanity: transistor density falls in the chiplet-package envelope.

    Bergamo's Zen 4c density (133 Mtx/mm^2 on the CCD die only) drops
    to ~91 at the package level once the lower-density Genoa IOD (30
    Mtx/mm^2 on N6) is averaged in. Turin Dense lands at ~114 thanks
    to the denser Zen 5c CCDs. All three should clear ~60 (well above
    the >5 floor that the loader's validator checks).
    """
    mapper = get_mapper_by_name(mapper_name)
    assert mapper is not None
    ps = mapper.physical_spec
    assert ps is not None

    density = ps.transistor_density_mtx_mm2
    assert density is not None
    # Empirical package-level envelope across the 3 AMD chiplet SKUs:
    # 9654 = 76.4, 9754 = 91.6, 9965 = 113.6. Loose bound catches gross
    # drift without false-positives on small refinements.
    assert 60.0 < density < 160.0, (
        f"{mapper_name}: package-level density {density:.1f} Mtx/mm^2 "
        f"is outside the chiplet envelope [60, 160]"
    )


# ---------------------------------------------------------------------------
# Schema-level per-die distinction (now visible after v13 multi-die rewrite)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "sku_id,compute_node,io_node,compute_die_id_suffix,io_die_id",
    [
        # EPYC 9654 Genoa: CCDs on N5, IOD on N6 (Genoa IOD shared with Bergamo)
        ("amd_epyc_9654_sp5", "tsmc_n5", "tsmc_n6",
         "epyc_9654_compute_aggregate", "genoa_iod"),
        # EPYC 9754 Bergamo: Zen 4c CCDs on N5, IOD on N6 (reused Genoa IOD)
        ("amd_epyc_9754_sp5", "tsmc_n5", "tsmc_n6",
         "epyc_9754_compute_aggregate", "genoa_iod"),
        # EPYC 9965 Turin Dense: Zen 5c CCDs on N4P (placeholder; N3E
        # pending in catalog), Turin IOD (new silicon, not shared) on N6
        ("amd_epyc_9965_sp5", "tsmc_n4p", "tsmc_n6",
         "epyc_9965_compute_aggregate", "turin_iod"),
    ],
    ids=["genoa", "bergamo", "turin_dense"],
)
def test_epyc_per_die_process_node_distinction_visible(
    sku_id: str,
    compute_node: str,
    io_node: str,
    compute_die_id_suffix: str,
    io_die_id: str,
):
    """v13 multi-die rewrite makes the per-die process_node distinction
    visible at the schema level. Prior to sprint #245, every chiplet
    YAML collapsed to a single virtual die with one process node
    (the compute die's). After sprint #245, the IO die carries its
    own process_node_id (TSMC N6 for all three IODs in this catalog).

    This pin catches any future drift that would re-collapse the
    representation -- e.g., a downstream loader bug that summed the
    dies[] list and lost the per-die node distinction.
    """
    from embodied_schemas.loaders import load_compute_products

    cps = load_compute_products()
    cp = cps.get(sku_id)
    assert cp is not None, f"{sku_id} not in embodied-schemas catalog"

    # Multi-die structure: 1 compute + 1 IO die
    assert len(cp.dies) == 2, (
        f"{sku_id}: expected 2 dies (compute + IO) after sprint #245 "
        f"rewrite, got {len(cp.dies)}"
    )
    die_by_role = {d.die_role.value: d for d in cp.dies}
    assert set(die_by_role.keys()) == {"compute", "io"}, (
        f"{sku_id}: die_roles {sorted(die_by_role)} != expected "
        f"{{'compute', 'io'}}"
    )

    compute_die = die_by_role["compute"]
    io_die = die_by_role["io"]

    # Per-die process node distinction
    assert compute_die.process_node_id == compute_node, (
        f"{sku_id}: compute die process_node_id drift -- expected "
        f"{compute_node}, got {compute_die.process_node_id}"
    )
    assert io_die.process_node_id == io_node, (
        f"{sku_id}: IO die process_node_id drift -- expected "
        f"{io_node}, got {io_die.process_node_id}"
    )

    # Canonical die_ids (cross-SKU IOD-reuse traceability)
    assert compute_die.die_id == compute_die_id_suffix
    assert io_die.die_id == io_die_id


def test_bergamo_shares_genoa_iod_die_id():
    """Cross-SKU shared-IOD invariant: Bergamo (9754) reuses Genoa IOD
    silicon unchanged from Genoa (9654). Both should reference the
    same canonical ``genoa_iod`` die_id, with byte-identical area and
    transistor count.

    This regression test catches accidental divergence -- e.g., if a
    future YAML edit renamed Bergamo's IOD die without renaming
    Genoa's, the shared-silicon invariant would silently break.
    """
    from embodied_schemas import DieRole
    from embodied_schemas.loaders import load_compute_products

    cps = load_compute_products()
    genoa = cps["amd_epyc_9654_sp5"]
    bergamo = cps["amd_epyc_9754_sp5"]

    genoa_iod = next(d for d in genoa.dies if d.die_role == DieRole.IO)
    bergamo_iod = next(d for d in bergamo.dies if d.die_role == DieRole.IO)

    assert genoa_iod.die_id == bergamo_iod.die_id == "genoa_iod"
    assert genoa_iod.die_size_mm2 == bergamo_iod.die_size_mm2
    assert genoa_iod.transistors_billion == bergamo_iod.transistors_billion
    assert genoa_iod.process_node_id == bergamo_iod.process_node_id


def test_turin_iod_is_distinct_from_genoa_iod():
    """Cross-SKU distinct-IOD invariant: Turin Dense (9965) uses a
    *new* Turin IOD silicon distinct from the Genoa IOD shared by
    9654/9754. The two IODs share the same Infinity-Fabric family
    but differ on memory (DDR5-6000 vs 4800), CXL (2.0 vs 1.1), and
    transistor count (~13 B vs ~12 B).

    Regression catches accidental re-unification (e.g., a copy-paste
    that pointed Turin at the Genoa IOD die_id and lost the new IOD
    silicon).
    """
    from embodied_schemas import DieRole, IOBlock
    from embodied_schemas.loaders import load_compute_products

    cps = load_compute_products()
    genoa = cps["amd_epyc_9654_sp5"]
    turin = cps["amd_epyc_9965_sp5"]

    genoa_iod = next(d for d in genoa.dies if d.die_role == DieRole.IO)
    turin_iod = next(d for d in turin.dies if d.die_role == DieRole.IO)

    # Distinct die_ids (different physical silicon)
    assert genoa_iod.die_id == "genoa_iod"
    assert turin_iod.die_id == "turin_iod"
    assert turin_iod.die_id != genoa_iod.die_id

    # Distinct transistor budgets (Turin IOD ~13 B vs Genoa IOD ~12 B)
    assert turin_iod.transistors_billion > genoa_iod.transistors_billion

    # IOBlock-level deltas: DDR5-6000 vs 4800, CXL 2.0 vs 1.1
    genoa_block = next(b for b in genoa_iod.blocks if isinstance(b, IOBlock))
    turin_block = next(b for b in turin_iod.blocks if isinstance(b, IOBlock))

    assert turin_block.memory.memory_bandwidth_gbps > genoa_block.memory.memory_bandwidth_gbps
    assert turin_block.cxl_version == "2.0"
    assert genoa_block.cxl_version == "1.1"
