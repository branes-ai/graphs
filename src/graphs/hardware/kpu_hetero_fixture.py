"""Synthetic heterogeneous KPU for the Phase C acceptance tests (graphs#268).

``build_heterogeneous_kpu()`` returns a ComputeProduct with one tile class of
every kind, built from the embodied-schemas tile-class library:

    pe_fabric       pe_int8_mac_i32 x24 (with a row-broadcast overlay)
                    pe_lns16_mac x8 (datapath with a transistor count)
                    pe_minplus_i16 x6
    systolic        systolic_int8_ws x4 (cell transistor count)
    fixed_function  ff_isp_raw2yuv x1, ff_stereo_sgm x1,
                    ff_vio_stereo_inertial x1 (2x2 footprint, absorbs its
                    memory cells)

on the Stillwater T64 (TSMC N16) chassis: an 8x8 checkerboard, 48 sites
used and 16 spare, and an ISP -> SGM -> VIO stream link. The chip silicon_bin
counts the INT8 and min-plus PEs (by tile_class_id and by tile_type label,
respectively); the LNS datapath, the systolic cells and the fixed-function
cores carry their own silicon.

The figures are illustrative, not a product. Each Phase C step makes the
fixture flow through one more consumer (silicon math, power model,
generator, validators, loader, mapper) without special cases.
"""

from __future__ import annotations

from embodied_schemas import (
    ComputeProduct,
    derive_kpu_performance,
    load_compute_products,
    load_kpu_tile_classes,
)

HETERO_FIXTURE_ID = "kpu_h64_fixture_16nm_tsmc_ffp"
BASE_SKU_ID = "kpu_t64_32x32_lp5x4_16nm_tsmc_ffp"

# Illustrative transistor counts for the tile-carried silicon (Mtx).
LNS_DATAPATH_MTX_PER_PE = 0.009
SYSTOLIC_CELL_MTX = 0.004
ROW_BROADCAST_MTX_PER_INSTANCE = 0.002
STREAM_LINK_MTX = 0.5


def _tiles(lib) -> list:
    lns = lib["pe_lns16_mac"].tile
    lns_datapath = {
        **lns.datapath.model_dump(mode="json", exclude_none=True),
        "mtx_per_pe": LNS_DATAPATH_MTX_PER_PE,
    }
    sys_mac = {
        **lib["systolic_int8_ws"].tile.mac.model_dump(mode="json", exclude_none=True),
        "mtx": SYSTOLIC_CELL_MTX,
    }
    row_broadcast = {
        "base": "nearest_neighbor_4",
        "link_bits": 32,
        "overlays": [
            {
                "overlay_id": "row_bcast",
                "kind": "row_broadcast",
                "instances_per": "row",
                "width_bits": 32,
                "mtx_per_instance": ROW_BROADCAST_MTX_PER_INSTANCE,
            }
        ],
    }
    return [
        lib["pe_int8_mac_i32"].instantiate(24, interconnect=row_broadcast),
        lib["pe_lns16_mac"].instantiate(8, datapath=lns_datapath),
        lib["pe_minplus_i16"].instantiate(6),
        lib["systolic_int8_ws"].instantiate(4, mac=sys_mac),
        lib["ff_isp_raw2yuv"].instantiate(1),
        lib["ff_stereo_sgm"].instantiate(1),
        lib["ff_vio_stereo_inertial"].instantiate(1),
    ]


def build_heterogeneous_kpu() -> ComputeProduct:
    """The synthetic heterogeneous KPU ComputeProduct (see module docstring)."""
    lib = load_kpu_tile_classes()
    base = load_compute_products()[BASE_SKU_ID]
    data = base.model_dump(mode="json")
    tiles = _tiles(lib)
    die = data["dies"][0]
    block = die["blocks"][0]

    block["tiles"] = [t.model_dump(mode="json") for t in tiles]
    block["total_tiles"] = sum(t.num_tiles for t in tiles)
    # The T64 chassis carries the uniform default DVFS partition (graphs#268
    # F2): 2x2 clusters over a mesh of identical PE-fabric tiles. That
    # partition means nothing over this heterogeneous checkerboard, and the
    # fixture predates it, so it does not inherit it. Without this line the
    # fixture silently gained 16 cluster domains that happened to validate
    # on the same 8x8 grid.
    block["power_domains"] = None
    block["checkerboard"] = {
        "compute_sites": {"rows": 8, "cols": 8},
        "placement": "auto",
        "spare_sites": 64 - sum(t.total_sites for t in tiles),
    }
    block["noc"]["overlays"] = [
        {
            "overlay_id": "isp_sgm_vio",
            "kind": "stream_link",
            "endpoints": ["ff_isp_raw2yuv", "ff_stereo_sgm", "ff_vio_stereo_inertial"],
            "width_bytes": 32,
            "mtx_per_instance": STREAM_LINK_MTX,
        }
    ]

    # Chip-level PE blocks: INT8 by tile_class_id, min-plus by tile_type label.
    # The LNS, systolic and fixed-function classes carry their own silicon.
    keep = [b for b in die["silicon_bin"]["blocks"] if not b["name"].startswith("pe_")]
    pe_blocks = [
        {
            "name": "pe_int8",
            "circuit_class": "balanced_logic",
            "transistor_source": {
                "kind": "per_pe", "per_unit_mtx": 0.006, "count_ref": "tile.pe_int8_mac_i32",
            },
        },
        {
            "name": "pe_minplus",
            "circuit_class": "balanced_logic",
            "transistor_source": {
                "kind": "per_pe", "per_unit_mtx": 0.004, "count_ref": "tile.MINPLUS-I16",
            },
        },
    ]
    die["silicon_bin"]["blocks"] = pe_blocks + keep

    default = next(
        p for p in data["power"]["thermal_profiles"]
        if p["name"] == data["power"]["default_thermal_profile"]
    )
    data["performance"] = derive_kpu_performance(tiles, default["clock_mhz"]).model_dump(
        mode="json"
    )
    data.update(
        id=HETERO_FIXTURE_ID,
        name="KPU heterogeneous-tile fixture (synthetic)",
        notes="Synthetic Phase C fixture (graphs#268); illustrative, not a product.",
    )
    return ComputeProduct.model_validate(data)
