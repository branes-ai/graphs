"""KPU golden snapshot gate (heterogeneous-tile refactor, Phase A0).

Every KPU SKU in the embodied-schemas catalog has a golden snapshot in
``tests/hardware/golden/kpu/<sku_id>.json`` capturing what the modeling
layers produce for it: generator round-trip + TDP breakdown, silicon
areas / power, both floorplans, PhysicalSpec, resource model, mapper
results on synthetic subgraphs, and validator findings.

The gate fails on ANY numerical or structural change. That is intended:
the KPU refactor must be zero-diff for the catalog SKUs unless a PR is a
declared model change. When a diff is expected, review it, then run

    python cli/kpu_golden_snapshot.py --update

and commit the regenerated JSON together with the change that caused it.
"""

from __future__ import annotations

import copy
import math

import pytest

from graphs.hardware import kpu_golden as kg

_MAX_REPORTED_DIFFS = 25


@pytest.fixture(scope="module")
def catalogs():
    return kg.load_catalogs()


def _catalog_ids() -> list[str]:
    return kg.catalog_kpu_sku_ids()


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sku_id", _catalog_ids())
def test_kpu_sku_matches_golden(sku_id, catalogs):
    path = kg.golden_path(sku_id)
    assert path.exists(), (
        f"no golden snapshot for catalog SKU {sku_id!r} at {path}. "
        f"Generate it with: python cli/kpu_golden_snapshot.py --update --sku {sku_id}"
    )
    diffs = kg.compare_snapshots(kg.load_snapshot(sku_id), kg.build_snapshot(sku_id, catalogs))
    if diffs:
        shown = "\n  ".join(diffs[:_MAX_REPORTED_DIFFS])
        more = len(diffs) - _MAX_REPORTED_DIFFS
        pytest.fail(
            f"{sku_id}: {len(diffs)} difference(s) vs golden snapshot:\n  {shown}"
            + (f"\n  ... {more} more" if more > 0 else "")
            + "\nIf this change is intended (declared model change or catalog "
            "data change), review the diff and regenerate with:\n"
            f"  python cli/kpu_golden_snapshot.py --update --sku {sku_id}"
        )


def test_golden_set_matches_catalog():
    """No catalog SKU without a golden; no golden for a retired SKU."""
    catalog = set(_catalog_ids())
    golden = set(kg.golden_sku_ids())
    assert catalog, "embodied-schemas catalog has zero KPU SKUs"
    assert catalog - golden == set(), f"catalog SKUs missing goldens: {sorted(catalog - golden)}"
    assert golden - catalog == set(), f"stale goldens (SKU not in catalog): {sorted(golden - catalog)}"


@pytest.mark.parametrize("sku_id", _catalog_ids())
def test_golden_file_is_canonical(sku_id):
    """Golden files are exactly what dumps_snapshot writes (not hand-edited),
    so a regeneration diff in git only shows real changes."""
    path = kg.golden_path(sku_id)
    text = path.read_text()
    assert kg.dumps_snapshot(kg.load_snapshot(sku_id)) == text


def test_golden_schema_version_is_current():
    for sku_id in kg.golden_sku_ids():
        meta = kg.load_snapshot(sku_id)["_meta"]
        assert meta["golden_schema_version"] == kg.GOLDEN_SCHEMA_VERSION, (
            f"{sku_id}: golden written by schema v{meta['golden_schema_version']}, "
            f"code is v{kg.GOLDEN_SCHEMA_VERSION}; regenerate with --update"
        )


# ---------------------------------------------------------------------------
# The gate actually catches changes
# ---------------------------------------------------------------------------

def test_snapshot_is_deterministic(catalogs):
    sku_id = "kpu_t64_32x32_lp5x4_16nm_tsmc_ffp"
    a = kg.build_snapshot(sku_id, catalogs)
    b = kg.build_snapshot(sku_id, catalogs)
    assert kg.compare_snapshots(a, b) == []
    assert kg.dumps_snapshot(a) == kg.dumps_snapshot(b)


def test_snapshot_detects_silicon_perturbation(catalogs):
    """A 1% change to one PE transistor coefficient must surface in the
    generator (die area, TDP), silicon, and floorplan sections."""
    sku_id = "kpu_t256_32x32_lp5x16_16nm_tsmc_ffp"
    baseline = kg.build_snapshot(sku_id, catalogs)

    perturbed = dict(catalogs)
    perturbed["kpus"] = dict(catalogs["kpus"])
    cp = catalogs["kpus"][sku_id].model_copy(deep=True)
    block = next(b for b in cp.dies[0].silicon_bin.blocks if b.name == "pe_int8")
    block.transistor_source.per_unit_mtx *= 1.01
    perturbed["kpus"][sku_id] = cp

    diffs = kg.compare_snapshots(baseline, kg.build_snapshot(sku_id, perturbed))
    sections = {d.split(".")[1].split("[")[0].split(":")[0] for d in diffs}
    assert {"input", "generator", "silicon", "floorplan"} <= sections, sections
    assert any(".generator.die_size_mm2" in d for d in diffs)
    assert any("tdp_breakdown_by_profile" in d and "leakage_w" in d for d in diffs)


# ---------------------------------------------------------------------------
# compare_snapshots / to_jsonable / dumps_snapshot units
# ---------------------------------------------------------------------------

def test_compare_float_tolerance():
    assert kg.compare_snapshots({"x": 1.0}, {"x": 1.0 + 1e-12}) == []
    assert kg.compare_snapshots({"x": 1.0}, {"x": 1.0 + 1e-6}) != []
    assert kg.compare_snapshots({"x": 2}, {"x": 2.0}) == []
    assert kg.compare_snapshots({"x": 0.0}, {"x": 1e-15}) == []


def test_compare_integers_exactly():
    """Large counts (bytes, transistors) must not hide behind rel_tol."""
    assert kg.compare_snapshots({"n": 2_000_000_000}, {"n": 2_000_000_001}) == [
        ".n: 2000000000 -> 2000000001"
    ]
    assert kg.compare_snapshots({"n": 8589934592}, {"n": 8589934592}) == []


# ---------------------------------------------------------------------------
# Snapshot coverage (regressions for gaps found in review of #267)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sku_id", _catalog_ids())
def test_sub_byte_precisions_carry_memory_traffic(sku_id):
    """INT4 is 0.5 bytes/element; truncating it to 0 made every INT4 mapping
    memory-free and blinded the gate to INT4 memory/tiling changes."""
    mappings = kg.load_snapshot(sku_id)["mapper"]["map_subgraph"]
    for precision, per_sg in mappings.items():
        for sg_name, alloc in per_sg.items():
            assert alloc["memory_time"] > 0, (sku_id, precision, sg_name)
    if "int4" in mappings and "int8" in mappings:
        ratio = (
            mappings["int4"]["gemm_1024"]["memory_time"]
            / mappings["int8"]["gemm_1024"]["memory_time"]
        )
        assert ratio == pytest.approx(0.5)


@pytest.mark.parametrize("sku_id", _catalog_ids())
def test_dynamic_power_covers_every_supported_precision(sku_id):
    snap = kg.load_snapshot(sku_id)
    tiles = snap["input"]["dies"][0]["blocks"][0]["tiles"]
    supported = {p for t in tiles for p in t["ops_per_tile_per_clock"]}
    profiles = {p["name"] for p in snap["input"]["power"]["thermal_profiles"]}
    for block in snap["silicon"]["blocks"]:
        by_profile = block["peak_dynamic_w_by_profile"]
        assert set(by_profile) == profiles, block["name"]
        for profile, by_prec in by_profile.items():
            assert set(by_prec) == supported, (block["name"], profile)


def test_compare_structure():
    assert kg.compare_snapshots({"a": 1}, {"a": 1, "b": 2}) == [".b: unexpected in actual"]
    assert kg.compare_snapshots({"a": 1, "b": 2}, {"a": 1}) == [".b: missing in actual"]
    assert kg.compare_snapshots({"l": [1, 2]}, {"l": [1]}) == [".l: list length 2 -> 1"]
    assert kg.compare_snapshots({"s": "x"}, {"s": "y"}) == [".s: 'x' -> 'y'"]
    assert kg.compare_snapshots({"b": True}, {"b": 1}) != []  # bool is not a number here
    assert kg.compare_snapshots({"n": None}, {"n": 0.0}) != []


def test_compare_ignores_meta():
    assert kg.compare_snapshots({"_meta": {"v": 1}, "x": 1}, {"_meta": {"v": 2}, "x": 1}) == []


def test_to_jsonable_enums_and_specials():
    import enum
    from dataclasses import dataclass

    class StrEnum(str, enum.Enum):
        A = "a"

    class IntEnum(enum.Enum):
        B = 2

    @dataclass
    class D:
        e: StrEnum
        k: dict
        f: float
        t: tuple
        s: set

    out = kg.to_jsonable(
        D(e=StrEnum.A, k={StrEnum.A: 1, IntEnum.B: 2}, f=math.inf, t=(1, 2), s={3, 1})
    )
    assert out == {"e": "a", "k": {"a": 1, "2": 2}, "f": "inf", "t": [1, 2], "s": [1, 3]}
    assert type(out["e"]) is str


def test_to_jsonable_rejects_unknown_types():
    class Opaque:
        pass

    with pytest.raises(TypeError):
        kg.to_jsonable({"x": Opaque()})


def test_dumps_round_trip_and_leaf_compaction():
    import json

    snap = {"_meta": {"v": 1}, "blocks": [{"x": 1.5, "n": "a"}, {"x": 2.0, "n": "b"}], "e": []}
    text = kg.dumps_snapshot(snap)
    assert json.loads(text) == snap
    # Each flat record is on a single line.
    assert '{"n": "a", "x": 1.5}' in text
    assert kg.dumps_snapshot(copy.deepcopy(snap)) == text
