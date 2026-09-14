"""
KPU golden snapshot -- the zero-diff safety net for the KPU refactor.

The heterogeneous-tile refactor (docs/plans/kpu-heterogeneous-tile-refactor-plan.md)
touches the KPU schema, generator, power model, silicon math, floorplanner,
validators, resource-model loader, and mapper. Every PR in that sprint must
leave the 12 catalog SKUs *numerically identical* unless it is a declared
model change. This module captures everything those layers produce for one
SKU into a JSON-able dict, and compares two such dicts structurally.

Snapshot sections (one dict per SKU):

  input            The ComputeProduct as loaded from embodied-schemas. A diff
                   here means the *catalog data* changed, not the model.
  generator        Round-trip through input_spec_from_compute_product ->
                   generate_kpu_sku: die area, transistors, performance,
                   power roll-up, plus the per-profile 5-term TDP breakdown.
  silicon          Per-block transistors / density / area / leakage / peak
                   dynamic power (at every profile clock), plus totals.
  floorplan        Circuit-class and architectural floorplans (every placed
                   block) plus their derived metrics.
  physical_spec    The PhysicalSpec the mappers attach.
  resource_model   The full HardwareResourceModel from the YAML loader.
  mapper           KPUMapper attributes plus map_subgraph() on a fixed set
                   of synthetic subgraphs at every supported precision.
  validators       Every finding from the default ValidatorRegistry.

Comparison is structural: dict keys and list lengths must match exactly,
strings / ints / bools exactly, floats within a tight relative tolerance
(so a refactor that reorders a floating-point sum does not fail the gate,
but any real modeling change does).

Regenerate / check with ``cli/kpu_golden_snapshot.py``.
"""

from __future__ import annotations

import dataclasses
import enum
import json
import math
from pathlib import Path
from typing import Any, Iterable, Optional

GOLDEN_SCHEMA_VERSION = 1

DEFAULT_REL_TOL = 1e-9
DEFAULT_ABS_TOL = 1e-12

# Keys excluded from comparison (provenance only).
_META_KEY = "_meta"

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_GOLDEN_DIR = _REPO_ROOT / "tests" / "hardware" / "golden" / "kpu"


# ---------------------------------------------------------------------------
# Generic conversion to JSON-able values
# ---------------------------------------------------------------------------

def _key_to_str(key: Any) -> str:
    if isinstance(key, enum.Enum):  # before str: str-mixin enums are str too
        return str(key.value)
    if isinstance(key, (str, int, float, bool)) or key is None:
        return str(key)
    if isinstance(key, tuple):
        return ",".join(_key_to_str(k) for k in key)
    raise TypeError(f"unsupported dict key type {type(key).__name__}: {key!r}")


def _float_to_jsonable(x: float) -> Any:
    if math.isnan(x):
        return "nan"
    if math.isinf(x):
        return "inf" if x > 0 else "-inf"
    return x


def to_jsonable(obj: Any, _stack: Optional[set[int]] = None) -> Any:
    """Convert dataclasses / pydantic models / enums / containers into
    plain JSON-able values with deterministic structure.

    Raises TypeError on an object it cannot represent faithfully, so a new
    field type surfaces as a loud failure rather than a silently unstable
    ``repr`` (which could embed memory addresses).
    """
    if _stack is None:
        _stack = set()

    # Enum first: ``class X(str, Enum)`` members are also ``str`` instances
    # and must serialize as their value, not pass through as enum objects.
    if isinstance(obj, enum.Enum):
        return to_jsonable(obj.value, _stack)
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        return _float_to_jsonable(obj)

    oid = id(obj)
    if oid in _stack:
        raise TypeError(f"reference cycle through {type(obj).__name__}")

    # pydantic v2 models
    if hasattr(obj, "model_dump") and callable(obj.model_dump):
        return to_jsonable(obj.model_dump(mode="json"), _stack)

    _stack.add(oid)
    try:
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            return {
                f.name: to_jsonable(getattr(obj, f.name), _stack)
                for f in dataclasses.fields(obj)
            }
        if isinstance(obj, dict):
            out: dict[str, Any] = {}
            for k, v in obj.items():
                ks = _key_to_str(k)
                if ks in out:
                    raise TypeError(f"dict key collision after str(): {ks!r}")
                out[ks] = to_jsonable(v, _stack)
            return out
        if isinstance(obj, (list, tuple)):
            return [to_jsonable(v, _stack) for v in obj]
        if isinstance(obj, (set, frozenset)):
            return sorted((to_jsonable(v, _stack) for v in obj), key=repr)
    finally:
        _stack.discard(oid)

    raise TypeError(
        f"to_jsonable: unsupported type {type(obj).__module__}."
        f"{type(obj).__qualname__}"
    )


# ---------------------------------------------------------------------------
# Snapshot construction
# ---------------------------------------------------------------------------

def load_catalogs() -> dict[str, Any]:
    """Load the three catalogs a snapshot needs, once."""
    from embodied_schemas import load_cooling_solutions, load_process_nodes

    from graphs.hardware.compute_product_loader import load_compute_products_unified

    return {
        "kpus": load_compute_products_unified(),
        "process_nodes": load_process_nodes(),
        "cooling_solutions": load_cooling_solutions(),
    }


def catalog_kpu_sku_ids() -> list[str]:
    """Every KPU SKU in the embodied-schemas catalog (same set as the
    Phase 6 catalog validation gate)."""
    from embodied_schemas import load_kpus

    return sorted(load_kpus().keys())


def _generator_section(cp, node, process_nodes) -> dict:
    from graphs.hardware.kpu_power_model import compute_thermal_profile_tdp_breakdown
    from graphs.hardware.kpu_sku_generator import (
        generate_kpu_sku,
        input_spec_from_compute_product,
    )

    spec = input_spec_from_compute_product(cp)
    regen = generate_kpu_sku(spec, process_nodes=process_nodes)
    die = regen.dies[0]

    breakdowns = {}
    for profile in spec.thermal_profiles:
        bd = compute_thermal_profile_tdp_breakdown(spec, profile, node)
        entry = to_jsonable(bd)
        entry["dynamic_w"] = bd.dynamic_w
        entry["total_tdp_w"] = bd.total_tdp_w
        breakdowns[profile.name] = entry

    return {
        "die_size_mm2": die.die_size_mm2,
        "transistors_billion": die.transistors_billion,
        "performance": to_jsonable(regen.performance),
        "power": to_jsonable(regen.power),
        "tdp_breakdown_by_profile": breakdowns,
    }


def _silicon_section(cp, node) -> dict:
    from graphs.hardware.sku_validators import silicon_math as sm

    profiles = cp.power.thermal_profiles
    blocks = []
    for block in cp.dies[0].silicon_bin.blocks:
        entry: dict[str, Any] = {
            "name": block.name,
            "circuit_class": block.circuit_class.value,
        }
        try:
            ba = sm.resolve_block_area(block, cp, node)
            entry.update(
                transistors_mtx=ba.transistors_mtx,
                density_mtx_per_mm2=ba.density_mtx_per_mm2,
                area_mm2=ba.area_mm2,
            )
        except sm.SiliconMathError as exc:
            entry["error"] = str(exc)
        entry["leakage_w"] = sm.estimate_block_leakage_w(block, cp, node)
        entry["peak_dynamic_w_int8_by_profile"] = {
            p.name: sm.estimate_block_peak_dynamic_w(
                block, cp, node, clock_mhz=p.clock_mhz, precision="int8"
            )
            for p in profiles
        }
        blocks.append(entry)

    return {
        "blocks": blocks,
        "total_pe_count": sm.total_pe_count(cp),
        "total_l1_kib": sm.total_l1_kib(cp),
        "total_l2_kib": sm.total_l2_kib(cp),
        "total_l3_kib": sm.total_l3_kib(cp),
        "num_tiles_by_type": sm.num_tiles_by_type(cp),
        "total_pes_by_tile_type": sm.total_pes_by_tile_type(cp),
        "total_chip_leakage_w": sm.total_chip_leakage_w(cp, node),
    }


def _floorplan_section(cp, node) -> dict:
    from graphs.hardware.silicon_floorplan import (
        derive_kpu_architectural_floorplan,
        derive_kpu_floorplan,
    )

    circuit = derive_kpu_floorplan(cp, node)
    arch = derive_kpu_architectural_floorplan(cp, node)
    return {
        "circuit": {
            **to_jsonable(circuit),
            "derived": {
                "die_area_mm2": circuit.die_area_mm2,
                "total_block_area_mm2": circuit.total_block_area_mm2(),
                "whitespace_fraction": circuit.whitespace_fraction(),
            },
        },
        "architectural": {
            **to_jsonable(arch),
            "derived": {
                "die_area_mm2": arch.die_area_mm2,
                "compute_memory_pitch_ratio": _float_to_jsonable(
                    arch.compute_memory_pitch_ratio
                ),
            },
        },
    }


def _synthetic_subgraphs(bpe: int) -> list:
    """Fixed, representative subgraphs used to pin mapper behavior."""
    from graphs.core.structures import (
        OperationType,
        ParallelismDescriptor,
        SubgraphDescriptor,
    )

    def matmul(sid: int, name: str, M: int, K: int, N: int) -> SubgraphDescriptor:
        return SubgraphDescriptor(
            subgraph_id=sid,
            node_ids=[name],
            node_names=[name],
            operation_types=[OperationType.MATMUL],
            fusion_pattern="matmul",
            total_flops=2 * M * K * N,
            total_macs=M * K * N,
            total_input_bytes=M * K * bpe,
            total_output_bytes=M * N * bpe,
            total_weight_bytes=K * N * bpe,
            parallelism=ParallelismDescriptor(
                batch=1, channels=N, spatial=M, total_threads=M * N
            ),
        )

    return [
        matmul(0, "gemm_1024", 1024, 1024, 1024),
        matmul(1, "gemv_4096", 1, 4096, 4096),
        matmul(2, "conv_im2col_56x56x64", 3136, 576, 64),
        matmul(3, "gemm_tall_8192x256", 8192, 256, 256),
    ]


def _mapper_section(rm) -> dict:
    from graphs.hardware.mappers.accelerators.kpu import KPUMapper

    mapper = KPUMapper(rm)
    attrs = {
        name: to_jsonable(getattr(mapper, name))
        for name in ("num_tiles", "scratchpad_per_tile", "threads_per_tile")
        if hasattr(mapper, name)
    }
    mappings: dict[str, dict] = {}
    for precision in sorted(rm.precision_profiles, key=lambda p: p.value):
        bpe = int(rm.precision_profiles[precision].bytes_per_element or 1)
        per_sg = {}
        for sg in _synthetic_subgraphs(bpe):
            alloc = mapper.map_subgraph(
                sg, execution_stage=0, concurrent_subgraphs=1, precision=precision
            )
            per_sg[sg.node_names[0]] = to_jsonable(alloc)
        mappings[precision.value] = per_sg
    return {"attributes": attrs, "map_subgraph": mappings}


def _validators_section(sku_id, catalogs) -> list[dict]:
    from graphs.hardware.sku_validators import (
        build_context_for_kpu,
        default_registry,
        load_validators,
    )

    load_validators()
    ctx = build_context_for_kpu(
        sku_id,
        kpus=catalogs["kpus"],
        process_nodes=catalogs["process_nodes"],
        cooling_solutions=catalogs["cooling_solutions"],
    )
    findings = [to_jsonable(f) for f in default_registry.run_all(ctx)]
    return sorted(
        findings,
        key=lambda f: (
            f["validator"], f["severity"], f.get("profile") or "",
            f.get("block") or "", f["message"],
        ),
    )


def _meta() -> dict:
    meta: dict[str, Any] = {"golden_schema_version": GOLDEN_SCHEMA_VERSION}
    try:
        from importlib.metadata import version

        meta["embodied_schemas_version"] = version("embodied-schemas")
    except Exception:  # pragma: no cover - provenance only
        meta["embodied_schemas_version"] = "unknown"
    return meta


def build_snapshot(sku_id: str, catalogs: Optional[dict] = None) -> dict:
    """Build the golden snapshot for one catalog KPU SKU."""
    from graphs.hardware.models.accelerators.kpu_yaml_loader import (
        load_kpu_resource_model_from_yaml,
    )
    from graphs.hardware.physical_spec_loader import (
        load_physical_spec_from_compute_product,
    )

    catalogs = catalogs or load_catalogs()
    cp = catalogs["kpus"][sku_id]
    node = catalogs["process_nodes"][cp.dies[0].process_node_id]
    rm = load_kpu_resource_model_from_yaml(sku_id)

    return {
        _META_KEY: _meta(),
        "sku_id": sku_id,
        "input": to_jsonable(cp),
        "generator": _generator_section(cp, node, catalogs["process_nodes"]),
        "silicon": _silicon_section(cp, node),
        "floorplan": _floorplan_section(cp, node),
        "physical_spec": to_jsonable(load_physical_spec_from_compute_product(sku_id)),
        "resource_model": to_jsonable(rm),
        "mapper": _mapper_section(rm),
        "validators": _validators_section(sku_id, catalogs),
    }


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def golden_path(sku_id: str, golden_dir: Path = DEFAULT_GOLDEN_DIR) -> Path:
    return Path(golden_dir) / f"{sku_id}.json"


def _is_leaf_container(obj: Any) -> bool:
    """A dict/list whose values are all scalars -- rendered on one line."""
    values = obj.values() if isinstance(obj, dict) else obj
    return all(not isinstance(v, (dict, list)) for v in values)


def _dumps_compact_leaves(obj: Any, indent: int) -> str:
    pad = " " * indent
    if isinstance(obj, (dict, list)) and (not obj or _is_leaf_container(obj)):
        return json.dumps(obj, sort_keys=True, allow_nan=False, separators=(", ", ": "))
    inner = " " * (indent + 1)
    if isinstance(obj, dict):
        items = [
            f"{inner}{json.dumps(k)}: {_dumps_compact_leaves(obj[k], indent + 1)}"
            for k in sorted(obj)
        ]
        return "{\n" + ",\n".join(items) + "\n" + pad + "}"
    if isinstance(obj, list):
        items = [f"{inner}{_dumps_compact_leaves(v, indent + 1)}" for v in obj]
        return "[\n" + ",\n".join(items) + "\n" + pad + "]"
    return json.dumps(obj, allow_nan=False)


def dumps_snapshot(snapshot: dict) -> str:
    """Deterministic serialization: sorted keys, full float precision.

    Containers holding only scalars (a floorplan block, a per-precision
    table) are written on one line, so a golden regeneration shows up in
    ``git diff`` as one changed line per changed record while keeping the
    ~2000-block floorplans of the large SKUs compact.
    """
    return _dumps_compact_leaves(snapshot, 0) + "\n"


def write_snapshot(snapshot: dict, golden_dir: Path = DEFAULT_GOLDEN_DIR) -> Path:
    path = golden_path(snapshot["sku_id"], golden_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dumps_snapshot(snapshot))
    return path


def load_snapshot(sku_id: str, golden_dir: Path = DEFAULT_GOLDEN_DIR) -> dict:
    return json.loads(golden_path(sku_id, golden_dir).read_text())


def golden_sku_ids(golden_dir: Path = DEFAULT_GOLDEN_DIR) -> list[str]:
    return sorted(p.stem for p in Path(golden_dir).glob("*.json"))


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def compare_snapshots(
    expected: Any,
    actual: Any,
    *,
    rel_tol: float = DEFAULT_REL_TOL,
    abs_tol: float = DEFAULT_ABS_TOL,
    path: str = "",
) -> list[str]:
    """Return a list of human-readable differences (empty = identical).

    ``_meta`` keys are ignored at every level. An int and a float that are
    numerically equal compare equal (JSON round-trips ``2.0`` as ``2.0`` but
    a refactor may legitimately change int-vs-float typing of a count).
    """
    diffs: list[str] = []
    here = path or "<root>"

    if isinstance(expected, dict) and isinstance(actual, dict):
        ek = {k for k in expected if k != _META_KEY}
        ak = {k for k in actual if k != _META_KEY}
        for k in sorted(ek - ak):
            diffs.append(f"{path}.{k}: missing in actual")
        for k in sorted(ak - ek):
            diffs.append(f"{path}.{k}: unexpected in actual")
        for k in sorted(ek & ak):
            diffs.extend(
                compare_snapshots(
                    expected[k], actual[k],
                    rel_tol=rel_tol, abs_tol=abs_tol, path=f"{path}.{k}",
                )
            )
        return diffs

    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            diffs.append(
                f"{here}: list length {len(expected)} -> {len(actual)}"
            )
        for i, (e, a) in enumerate(zip(expected, actual)):
            diffs.extend(
                compare_snapshots(
                    e, a, rel_tol=rel_tol, abs_tol=abs_tol, path=f"{path}[{i}]"
                )
            )
        return diffs

    if _is_number(expected) and _is_number(actual):
        if not math.isclose(expected, actual, rel_tol=rel_tol, abs_tol=abs_tol):
            diffs.append(f"{here}: {expected!r} -> {actual!r}")
        return diffs

    if type(expected) is not type(actual) or expected != actual:
        diffs.append(f"{here}: {expected!r} -> {actual!r}")
    return diffs


def check_skus(
    sku_ids: Optional[Iterable[str]] = None,
    *,
    golden_dir: Path = DEFAULT_GOLDEN_DIR,
    catalogs: Optional[dict] = None,
) -> dict[str, list[str]]:
    """Build current snapshots and diff them against the goldens.

    Returns ``{sku_id: [diff, ...]}`` for every requested SKU (empty list =
    identical). A SKU with no golden file reports a single diff saying so.
    """
    catalogs = catalogs or load_catalogs()
    ids = list(sku_ids) if sku_ids is not None else catalog_kpu_sku_ids()
    results: dict[str, list[str]] = {}
    for sku_id in ids:
        if not golden_path(sku_id, golden_dir).exists():
            results[sku_id] = [f"no golden snapshot at {golden_path(sku_id, golden_dir)}"]
            continue
        results[sku_id] = compare_snapshots(
            load_snapshot(sku_id, golden_dir), build_snapshot(sku_id, catalogs)
        )
    return results
