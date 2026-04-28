"""
Layer 4 L2 cache panel builder.

Surfaces per-SKU L2 capacity (per unit + total), L2 read energy
from the SKU's TechnologyProfile, and the topology classification
that distinguishes private per-core L2 (CPUs, KPU tile-local L2)
from shared L2 (discrete GPUs) and shared-LLC (Ampere SoCs, TPU
unified buffer collapse, Hailo on-chip SRAM).

Crucially, the schema's legacy ``l2_cache_total`` field carries the
LLC (== L3 on x86), not the physical L2. Layer 4 reads the new
``l2_cache_per_unit`` field instead, so the panel reports physical
L2 across architectures consistently. The legacy LLC field is
reserved for the future Layer 5 panel.

For cache-based SKUs (CPU, GPU L2) the panel surfaces a per-op-type
hit-rate table from ``DataParallelEnergyModel.l2_hit_rate_by_op``.
For scratchpad-based SKUs (KPU tile-local L2, Hailo on-chip, TPU
unified buffer collapse) the hit rate is deterministic 1.0.

KPU SKUs: panel reads ``l2_cache_per_unit`` from the resource model,
which the SKU factories populate from
``KPUTileEnergyModel.l2_size_per_tile`` -- no double-counting.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from graphs.benchmarks.schema import LayerTag
from graphs.core.confidence import ConfidenceLevel
from graphs.hardware.architectural_energy import DataParallelEnergyModel
from graphs.hardware.resource_model import HardwareResourceModel
from graphs.reporting.microarch_schema import LayerPanel
from graphs.reporting.layer_panels.layer1_alu import resolve_sku_resource_model
from graphs.reporting.layer_panels.layer2_register import (
    _process_node_nm,
    _tech_profile_for,
)


# Valid topology values for the L2 layer.
VALID_L2_TOPOLOGIES = {"per-unit", "shared", "shared-llc"}


def _normalize_topology(raw_topo: Optional[str], sku_id: str) -> str:
    """Normalize ``l2_topology`` to canonical lowercase form.

    Defaults to ``"per-unit"`` (CPU-style) when unset; raises on
    unknown values so a typo doesn't silently misclassify the layer.
    """
    topo = (raw_topo or "per-unit").strip().lower()
    if topo not in VALID_L2_TOPOLOGIES:
        raise ValueError(
            f"Invalid l2_topology '{raw_topo}' for SKU '{sku_id}'. "
            f"Expected one of: {sorted(VALID_L2_TOPOLOGIES)}."
        )
    return topo


def _l2_value(model: HardwareResourceModel) -> Optional[int]:
    """Read the M4 physical L2 per-unit value, treating None as 'absent'.

    A value of 0 is a valid datasheet point (TPU collapses L2 into UB)
    and should not be confused with 'unset'. The panel rendering
    differentiates the two cases.
    """
    return model.l2_cache_per_unit


def _has_l2_field(model: HardwareResourceModel) -> bool:
    """The M4 field is set (including 0) on this SKU."""
    return model.l2_cache_per_unit is not None


def _default_hit_rate_table() -> Dict[str, float]:
    """Pull the default per-op-type L2 hit-rate table from the
    DataParallelEnergyModel dataclass field default."""
    fld = next(
        f for f in DataParallelEnergyModel.__dataclass_fields__.values()
        if f.name == "l2_hit_rate_by_op"
    )
    return dict(fld.default_factory())


def _format_kib(bytes_value: int) -> str:
    if bytes_value >= 1024 * 1024:
        return f"{bytes_value / (1024*1024):.1f}"
    return f"{bytes_value / 1024:.0f}"


def _kib_unit(bytes_value: int) -> str:
    return "MiB" if bytes_value >= 1024 * 1024 else "KiB"


def _format_capacity(bytes_per_unit: int, units: int) -> str:
    total = bytes_per_unit * units
    if total >= 1024 * 1024:
        return f"{total / (1024*1024):.1f} MiB total"
    return f"{total / 1024:.0f} KiB total"


def _provenance_or_theoretical(
    model: HardwareResourceModel, key: str,
) -> str:
    conf = model.get_provenance(key)
    if conf.level is ConfidenceLevel.UNKNOWN:
        return ConfidenceLevel.THEORETICAL.value.upper()
    return conf.level.value.upper()


def _l2_energy_pj_per_byte(model: HardwareResourceModel,
                           sku_id: str) -> float:
    """L2 energy per byte from the SKU's TechnologyProfile."""
    tech = _tech_profile_for(model, sku_id)
    return tech.l2_cache_energy_per_byte_pj


# --------------------------------------------------------------------
# Panel construction
# --------------------------------------------------------------------

def build_layer4_l2_cache_panel(
    sku_id: str,
    model: Optional[HardwareResourceModel] = None,
) -> LayerPanel:
    """Build the populated Layer 4 (L2 cache) panel."""
    if model is None:
        model = resolve_sku_resource_model(sku_id)

    title = "Layer 4: L2 Cache"

    if model is None or not _has_l2_field(model):
        return LayerPanel(
            layer=LayerTag.L2_CACHE,
            title=title,
            status="not_populated",
            summary=f"L2 capacity not reported for {sku_id}.",
        )

    bytes_per_unit = model.l2_cache_per_unit  # may be 0 (collapsed)
    units = model.compute_units or 1
    topology = _normalize_topology(model.l2_topology, sku_id)
    energy_pj_per_byte = _l2_energy_pj_per_byte(model, sku_id)

    metrics: Dict[str, Dict] = {}

    if bytes_per_unit == 0:
        # L2 explicitly collapsed (e.g., TPU UB).
        metrics["L2 status"] = {
            "value": "collapsed into LLC",
            "unit":  "",
            "provenance": _provenance_or_theoretical(
                model, "l2_cache_per_unit"
            ),
        }
    else:
        metrics["L2 per unit"] = {
            "value": _format_kib(bytes_per_unit),
            "unit":  _kib_unit(bytes_per_unit),
            "provenance": _provenance_or_theoretical(
                model, "l2_cache_per_unit"
            ),
        }
        metrics["L2 total"] = {
            "value": _format_capacity(bytes_per_unit, units),
            "unit":  "",
            "provenance": "n/a",
        }

    metrics["Topology"] = {
        "value": topology,
        "unit":  "",
        "provenance": _provenance_or_theoretical(model, "l2_topology"),
    }

    metrics["L2 read energy"] = {
        "value": f"{energy_pj_per_byte:.3f}",
        "unit":  "pJ/byte",
        "provenance": "THEORETICAL",
    }

    metrics["Process / market"] = {
        "value": f"{_process_node_nm(model)} nm",
        "unit":  "",
        "provenance": "n/a",
    }

    notes: List[str] = []

    # L2-as-LLC annotation for shared-llc topology
    if topology == "shared-llc":
        notes.append(
            "On this SKU L2 is the last-level cache: there is no "
            "distinct L3 layer between L2 and DRAM. The Layer 5 panel "
            "(LLC, future M5) will surface zero capacity here."
        )

    # Hit rate table: cache-managed (per-unit / shared) vs deterministic
    storage_kind = (model.l1_storage_kind or "").strip().lower()
    is_cache_managed = (storage_kind == "cache")

    if is_cache_managed and bytes_per_unit > 0:
        for op_kind, rate in _default_hit_rate_table().items():
            metrics[f"Hit rate ({op_kind})"] = {
                "value": f"{rate * 100:.0f}",
                "unit":  "%",
                "provenance": "THEORETICAL",
            }
        notes.append(
            "Per-op-type L2 hit rates from "
            "DataParallelEnergyModel.l2_hit_rate_by_op (M4 lift). "
            "Matrix workloads with weight reuse retain L2 locality "
            "(~95% of L1 misses); elementwise streams that miss L1 "
            "have little reuse left (~80%)."
        )
    elif bytes_per_unit > 0:
        # Scratchpad-managed: deterministic
        metrics["Hit rate (deterministic)"] = {
            "value": "100",
            "unit":  "%",
            "provenance": "n/a",
        }
        notes.append(
            "Software-managed L2 SRAM: the host compiler stages "
            "data through this layer, so the hit rate is deterministic "
            "1.0 by construction. Matches the M0.5 dataflow-tile "
            "abstraction for KPU SKUs."
        )

    summary_pieces = [f"L2 layer for {sku_id}: "]
    if bytes_per_unit == 0:
        summary_pieces.append(
            "no distinct L2 layer (collapsed into LLC). "
        )
    else:
        summary_pieces.append(
            f"{_format_kib(bytes_per_unit)} {_kib_unit(bytes_per_unit)} "
            f"per unit ({_format_capacity(bytes_per_unit, units)}) at "
            f"{energy_pj_per_byte:.3f} pJ/byte. "
        )
    summary_pieces.append(f"Topology: {topology}.")
    summary = "".join(summary_pieces)

    return LayerPanel(
        layer=LayerTag.L2_CACHE,
        title=title,
        status="theoretical",
        summary=summary,
        metrics=metrics,
        notes=notes,
    )


def build_layer4_panels_for_skus(
    sku_ids: List[str],
) -> Dict[str, LayerPanel]:
    return {sku: build_layer4_l2_cache_panel(sku) for sku in sku_ids}


# --------------------------------------------------------------------
# Cross-SKU comparison
# --------------------------------------------------------------------

@dataclass
class CrossSKULayer4Chart:
    """Per-SKU L2 capacity, energy, topology classification."""
    skus: List[str]
    l2_per_unit_bytes: Dict[str, int]
    l2_total_bytes: Dict[str, int]
    topology: Dict[str, str]
    energy_pj_per_byte: Dict[str, float]
    provenance: Dict[str, str]


def cross_sku_layer4_chart(sku_ids: List[str]) -> CrossSKULayer4Chart:
    chart = CrossSKULayer4Chart(
        skus=list(sku_ids),
        l2_per_unit_bytes={},
        l2_total_bytes={},
        topology={},
        energy_pj_per_byte={},
        provenance={},
    )

    models: Dict[str, Optional[HardwareResourceModel]] = {
        sku: resolve_sku_resource_model(sku) for sku in sku_ids
    }

    for sku in sku_ids:
        m = models.get(sku)
        if m is None or not _has_l2_field(m):
            continue
        per_unit = m.l2_cache_per_unit  # may be 0
        units = m.compute_units or 1
        chart.l2_per_unit_bytes[sku] = per_unit
        chart.l2_total_bytes[sku] = per_unit * units
        chart.topology[sku] = _normalize_topology(m.l2_topology, sku)
        chart.energy_pj_per_byte[sku] = _l2_energy_pj_per_byte(m, sku)
        chart.provenance[sku] = _provenance_or_theoretical(
            m, "l2_cache_per_unit"
        )

    return chart


__all__ = [
    "build_layer4_l2_cache_panel",
    "build_layer4_panels_for_skus",
    "cross_sku_layer4_chart",
    "CrossSKULayer4Chart",
    "VALID_L2_TOPOLOGIES",
]
