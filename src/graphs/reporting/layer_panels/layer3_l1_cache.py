"""
Layer 3 L1 cache / scratchpad panel builder.

Surfaces per-SKU L1 capacity (per unit + total), L1 read / write
energy from the SKU's TechnologyProfile, and the storage-kind badge
that distinguishes hardware-managed caches (CPU L1, GPU shared mem)
from software-managed scratchpads (KPU tile-local SRAM, TPU unified
buffer, Hailo on-chip SRAM).

For cache-based SKUs the panel renders a per-op-type hit-rate table
(matrix / elementwise / default), pulling from
``DataParallelEnergyModel.shared_mem_l1_hit_rate_by_op``. For
scratchpad-based SKUs the hit rate is deterministic 1.0 by design
(software-managed staging), so the panel substitutes a note instead.

KPU SKUs surface tile-local SRAM as the L1-equivalent, consistent
with the M0.5 dataflow-tile abstraction. The Layer 3 panel reads
their ``l1_cache_per_unit`` directly; it does not double-count
against the M0.5 tile model.
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


# Per-op-type hit rates exposed in the panel for cache-based SKUs.
# Pulled from DataParallelEnergyModel defaults so the table stays in
# sync with the analytical model. This keeps the constant in one
# place even if the M3 lookup expands later.
def _default_hit_rate_table() -> Dict[str, float]:
    """Pull the default per-op-type hit-rate table without instantiating
    a model with a real TechnologyProfile (we only want defaults)."""
    # Read the default_factory from the dataclass field
    fld = next(
        f for f in DataParallelEnergyModel.__dataclass_fields__.values()
        if f.name == "shared_mem_l1_hit_rate_by_op"
    )
    return dict(fld.default_factory())


# --------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------

def _format_capacity_kib(bytes_per_unit: int, units: int) -> str:
    total_kib = (bytes_per_unit * units) / 1024
    if total_kib >= 1024:
        return f"{total_kib / 1024:.1f} MiB total"
    return f"{total_kib:.0f} KiB total"


def _format_kib(bytes_value: int) -> str:
    if bytes_value >= 1024 * 1024:
        return f"{bytes_value / (1024*1024):.1f}"
    return f"{bytes_value / 1024:.0f}"


def _kib_unit(bytes_value: int) -> str:
    return "MiB" if bytes_value >= 1024 * 1024 else "KiB"


def _provenance_or_theoretical(
    model: HardwareResourceModel, key: str,
) -> str:
    """Read provenance; default to THEORETICAL when unannotated and
    populated."""
    conf = model.get_provenance(key)
    if conf.level is ConfidenceLevel.UNKNOWN:
        return ConfidenceLevel.THEORETICAL.value.upper()
    return conf.level.value.upper()


def _l1_energy_pj_per_byte(model: HardwareResourceModel,
                           sku_id: str) -> float:
    """L1 energy per byte from the SKU's TechnologyProfile."""
    tech = _tech_profile_for(model, sku_id)
    return tech.l1_cache_energy_per_byte_pj


# --------------------------------------------------------------------
# Panel construction
# --------------------------------------------------------------------

def build_layer3_l1_cache_panel(
    sku_id: str,
    model: Optional[HardwareResourceModel] = None,
) -> LayerPanel:
    """Build the populated Layer 3 (L1 cache / scratchpad) panel."""
    if model is None:
        model = resolve_sku_resource_model(sku_id)

    title = "Layer 3: L1 Cache / Scratchpad"

    if model is None or not model.l1_cache_per_unit:
        return LayerPanel(
            layer=LayerTag.L1_CACHE,
            title=title,
            status="not_populated",
            summary=f"No L1 capacity reported for {sku_id}.",
        )

    bytes_per_unit = model.l1_cache_per_unit
    units = model.compute_units or 1
    storage_kind = model.l1_storage_kind or "cache"
    energy_pj_per_byte = _l1_energy_pj_per_byte(model, sku_id)

    metrics: Dict[str, Dict] = {
        "L1 per unit": {
            "value": _format_kib(bytes_per_unit),
            "unit":  _kib_unit(bytes_per_unit),
            "provenance": _provenance_or_theoretical(
                model, "l1_cache_per_unit"
            ),
        },
        "L1 total": {
            "value": _format_capacity_kib(bytes_per_unit, units),
            "unit":  "",
            "provenance": "n/a",
        },
        "Storage kind": {
            "value": storage_kind,
            "unit":  "",
            "provenance": _provenance_or_theoretical(
                model, "l1_storage_kind"
            ),
        },
        "L1 read energy": {
            "value": f"{energy_pj_per_byte:.3f}",
            "unit":  "pJ/byte",
            "provenance": "THEORETICAL",
        },
        "Process / market": {
            "value": f"{_process_node_nm(model)} nm",
            "unit":  "",
            "provenance": "n/a",
        },
    }

    notes: List[str] = []

    if storage_kind == "cache":
        # Cache-based SKU: surface per-op-type hit rates.
        hit_rates = _default_hit_rate_table()
        for op_kind, rate in hit_rates.items():
            metrics[f"Hit rate ({op_kind})"] = {
                "value": f"{rate * 100:.0f}",
                "unit":  "%",
                "provenance": "THEORETICAL",
            }
        notes.append(
            "Per-op-type L1 hit rates from "
            "DataParallelEnergyModel.shared_mem_l1_hit_rate_by_op. "
            "Matrix workloads with weight reuse hit ~95%; elementwise "
            "streams with poor locality drop to ~85%. "
            "M3 lifts this from a single-rate constant."
        )
    else:
        # Scratchpad: deterministic by design.
        metrics["Hit rate (deterministic)"] = {
            "value": "100",
            "unit":  "%",
            "provenance": "n/a",
        }
        notes.append(
            "Software-managed scratchpad: the host compiler stages "
            "tiles in and out, so the hit rate is deterministic 1.0 "
            "by construction. The cost lives in the compiler's tiling "
            "decisions and is accounted for in the M0.5 dataflow-tile "
            "abstraction (KPU) or the systolic-fill model (TPU)."
        )

    summary = (
        f"L1-equivalent storage for {sku_id}: "
        f"{_format_kib(bytes_per_unit)} {_kib_unit(bytes_per_unit)} "
        f"per unit ({_format_capacity_kib(bytes_per_unit, units)}). "
        f"{storage_kind.capitalize()}-managed at "
        f"{energy_pj_per_byte:.3f} pJ/byte read."
    )

    return LayerPanel(
        layer=LayerTag.L1_CACHE,
        title=title,
        status="theoretical",
        summary=summary,
        metrics=metrics,
        notes=notes,
    )


def build_layer3_panels_for_skus(
    sku_ids: List[str],
) -> Dict[str, LayerPanel]:
    """Build Layer 3 panels for a batch of SKUs."""
    return {sku: build_layer3_l1_cache_panel(sku) for sku in sku_ids}


# --------------------------------------------------------------------
# Cross-SKU comparison
# --------------------------------------------------------------------

@dataclass
class CrossSKULayer3Chart:
    """Per-SKU L1 capacity, energy, and storage-kind classification."""
    skus: List[str]
    l1_per_unit_bytes: Dict[str, int]
    l1_total_bytes: Dict[str, int]
    storage_kind: Dict[str, str]
    energy_pj_per_byte: Dict[str, float]
    provenance: Dict[str, str]


def cross_sku_layer3_chart(sku_ids: List[str]) -> CrossSKULayer3Chart:
    """Build the Layer 3 cross-SKU comparison chart data."""
    chart = CrossSKULayer3Chart(
        skus=list(sku_ids),
        l1_per_unit_bytes={},
        l1_total_bytes={},
        storage_kind={},
        energy_pj_per_byte={},
        provenance={},
    )

    models: Dict[str, Optional[HardwareResourceModel]] = {
        sku: resolve_sku_resource_model(sku) for sku in sku_ids
    }

    for sku in sku_ids:
        m = models.get(sku)
        if m is None or not m.l1_cache_per_unit:
            continue
        per_unit = m.l1_cache_per_unit
        units = m.compute_units or 1
        chart.l1_per_unit_bytes[sku] = per_unit
        chart.l1_total_bytes[sku] = per_unit * units
        chart.storage_kind[sku] = m.l1_storage_kind or "cache"
        chart.energy_pj_per_byte[sku] = _l1_energy_pj_per_byte(m, sku)
        chart.provenance[sku] = _provenance_or_theoretical(
            m, "l1_cache_per_unit"
        )

    return chart


__all__ = [
    "build_layer3_l1_cache_panel",
    "build_layer3_panels_for_skus",
    "cross_sku_layer3_chart",
    "CrossSKULayer3Chart",
]
