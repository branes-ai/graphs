"""The default per-cluster DVFS partition for a uniform KPU (graphs#268 F2).

``docs/designs/kpu-cluster-organization-for-dvfs-and-floorsweeping.md``
recommends organizing a KPU mesh into clusters of k x k tiles, each its own
DVFS domain with its own supply rail and PLL, sized so a cluster sits in the
millimetre range where process variation is spatially correlated and the
per-chip cluster count stays in 8-64 -- enough for meaningful DVFS and
harvest decisions, few enough that regulator and PLL overhead does not
dominate. B4 gave the schema generic ``PowerDomain``s to express that; this
module is the default partition for the SKUs that shipped without one.

**What the default changes, and what it does not.** It adds structure: one
``cluster`` domain per k x k block of compute sites, each naming its own
``rail_id`` and ``clock_domain_id``, plus one ``uncore`` domain for the
silicon outside the mesh. It declares no ``domain_operating_points``, so
every cluster runs at its thermal profile's own Vdd and clock, the power
model keeps its original formula (``is_legacy_shaped``), and TDP,
performance and area are unchanged. What DVFS policy then does with the
clusters -- running a hot quadrant slower, dropping a slow-corner cluster's
Vdd headroom -- is a per-profile choice layered on top.

**Cluster size.** The design's table gives 2x2 for the T64 and 4x4 for the
T128, T256 and T512, and says "recommended k=4". Its prose rule of thumb
("the smallest cluster size such that ... the count is in 8-64") would
instead pick 2x2 for the T128 (32 clusters), contradicting its own table.
This module follows the table: prefer 4x4, drop to 2x2 when 4x4 yields fewer
than 8 clusters, grow to 6x6 or 8x8 when it yields more than 64 (the
design's note for future large SKUs), and try 2, 6 and 8 in that order when
4 does not divide the grid at all. The T768, absent from the table,
gets 4x4 and 48 clusters by the same rule.

**Not modeled here** (each needs a schema field or data the catalog does
not carry): the quadrant level, per-cluster regulator and PLL silicon and
power, per-cluster harvest (``is_disabled``), and process-variation bins.
Clusters are not ``gateable``: in the design the tile is the gating unit
and the quadrant the region-gating unit; the cluster is the DVFS and
floorsweeping unit.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

#: The design's recommended cluster edge, in compute sites.
PREFERRED_CLUSTER_EDGE = 4

#: Per-chip cluster count the design targets.
CLUSTER_COUNT_RANGE = (8, 64)

#: Tried when the preferred edge gives too few clusters (smaller), too many
#: (larger), or does not divide the grid (all of them, smallest first).
_SMALLER_EDGES = (2,)
_LARGER_EDGES = (6, 8)

#: silicon_bin transistor-source kinds that scale with the mesh, and so
#: belong to the clusters rather than the uncore.
_MESH_SCALED_KINDS = frozenset({"per_pe", "per_kib", "per_router"})


class PowerDomainDefaultError(ValueError):
    """The default partition cannot be applied to this architecture."""


def _count(rows: int, cols: int, edge: int) -> Optional[int]:
    """Clusters of ``edge`` x ``edge`` sites on the grid, or None if the grid
    does not divide evenly -- a ragged edge cluster would not be the one
    repeated shape the design calls for."""
    if rows % edge or cols % edge:
        return None
    return (rows // edge) * (cols // edge)


def default_cluster_edge(rows: int, cols: int) -> int:
    """The cluster edge the design's table and recommendation give for a
    ``rows`` x ``cols`` compute-site grid.

    Raises:
        PowerDomainDefaultError: no candidate edge divides the grid into a
            cluster count in ``CLUSTER_COUNT_RANGE``.
    """
    lo, hi = CLUSTER_COUNT_RANGE
    preferred = _count(rows, cols, PREFERRED_CLUSTER_EDGE)
    if preferred is not None and lo <= preferred <= hi:
        return PREFERRED_CLUSTER_EDGE
    if preferred is None:
        # 4 does not divide the grid, so the count says nothing about which
        # direction to go: try every other edge, smallest first -- the
        # design's prose rule. An 18x18 mesh fails 2 (81 clusters) and
        # takes 6 (9); only trying the finer side made that an error
        # (CodeRabbit on #294).
        fallbacks = tuple(sorted(_SMALLER_EDGES + _LARGER_EDGES))
    elif preferred < lo:
        fallbacks = _SMALLER_EDGES  # too few clusters: go finer
    else:
        fallbacks = _LARGER_EDGES  # too many: go coarser
    for edge in fallbacks:
        count = _count(rows, cols, edge)
        if count is not None and lo <= count <= hi:
            return edge
    raise PowerDomainDefaultError(
        f"no cluster edge in {(PREFERRED_CLUSTER_EDGE,) + _SMALLER_EDGES + _LARGER_EDGES} "
        f"divides a {rows}x{cols} compute-site grid into {lo}-{hi} clusters"
    )


def cluster_domain_id(row: int, col: int) -> str:
    """The design's cluster naming, ``c_<row>_<col>`` in cluster coordinates."""
    return f"c_{row}_{col}"


def default_cluster_domains(rows: int, cols: int) -> List[Dict[str, Any]]:
    """One ``cluster`` power domain per k x k block of compute sites, row
    major, each with its own rail and PLL."""
    edge = default_cluster_edge(rows, cols)
    domains: List[Dict[str, Any]] = []
    for crow in range(rows // edge):
        for ccol in range(cols // edge):
            did = cluster_domain_id(crow, ccol)
            domains.append({
                "domain_id": did,
                "kind": "cluster",
                "site_ranges": [{
                    "row_min": crow * edge,
                    "row_max": (crow + 1) * edge - 1,
                    "col_min": ccol * edge,
                    "col_max": (ccol + 1) * edge - 1,
                }],
                # The design: one rail and one PLL per cluster.
                "rail_id": f"vdd_{did}",
                "clock_domain_id": f"clk_{did}",
            })
    return domains


def uncore_block_names(silicon_bin_blocks: Sequence[Any]) -> List[str]:
    """The silicon_bin blocks that do not scale with the mesh -- memory PHYs,
    IO, control -- in declaration order."""
    names = []
    for block in silicon_bin_blocks:
        kind = block.transistor_source.kind
        kind = getattr(kind, "value", kind)
        if kind not in _MESH_SCALED_KINDS:
            names.append(block.name)
    return names


def default_power_domains(
    rows: int, cols: int, silicon_bin_blocks: Sequence[Any]
) -> List[Dict[str, Any]]:
    """The full default: the cluster partition plus one uncore domain on the
    shared chip-wide rail, as the design's generator pass describes."""
    return default_cluster_domains(rows, cols) + [{
        "domain_id": "uncore",
        "kind": "uncore",
        "members": uncore_block_names(silicon_bin_blocks),
        "rail_id": "vdd_uncore",
        "clock_domain_id": "clk_uncore",
    }]


def _grid(arch: Any) -> Tuple[int, int]:
    return arch.noc.mesh_rows, arch.noc.mesh_cols


def check_uniform(arch: Any) -> None:
    """The default is for uniform KPUs only.

    A heterogeneous checkerboard partitions differently -- the refactor plan
    gives each fixed-function class its own gateable domain -- and a SKU
    that already declares domains has made its own choice.
    """
    if getattr(arch, "power_domains", None):
        raise PowerDomainDefaultError("the architecture already declares power domains")
    if getattr(arch, "checkerboard", None) is not None:
        raise PowerDomainDefaultError(
            "the default cluster partition is for uniform KPUs; this one has an "
            "explicit checkerboard"
        )
    kinds = {getattr(t.tile_kind, "value", t.tile_kind) for t in arch.tiles}
    if kinds != {"pe_fabric"}:
        raise PowerDomainDefaultError(
            f"the default cluster partition is for uniform pe_fabric KPUs; this one "
            f"has tile kinds {sorted(kinds)}"
        )
    rows, cols = _grid(arch)
    if rows * cols != arch.total_tiles:
        raise PowerDomainDefaultError(
            f"the {rows}x{cols} mesh holds {rows * cols} sites but the architecture "
            f"declares {arch.total_tiles} tiles, so sites and tiles do not correspond"
        )


def default_power_domains_for(arch: Any, silicon_bin_blocks: Sequence[Any]) -> List[Dict[str, Any]]:
    """``default_power_domains`` for an architecture, after ``check_uniform``."""
    check_uniform(arch)
    rows, cols = _grid(arch)
    return default_power_domains(rows, cols, silicon_bin_blocks)
