"""
Layer 5 L3 / LLC panel builder.

Surfaces per-SKU L3 capacity, L3 read energy from the SKU's
TechnologyProfile, the cache-coherence protocol classifier, and an
honest annotation when the SKU has *no* L3 layer (GPU L2-as-LLC,
TPU UB collapse, KPU mesh routing, Hailo dataflow). In those cases
the panel renders an explicit "no L3 by design" note rather than
fabricating a phantom capacity.

Coherence-protocol energy lives at this layer (PROTOCOL cost: snoop
messages, state transitions, directory broadcasts). The TRANSPORT
cost (NoC hop energy) belongs to Layer 6 and is intentionally out
of scope here.

For SKUs with a distinct L3 (CPUs only in the M5 catalog), the
panel surfaces a per-op-type hit-rate table from
``StoredProgramEnergyModel.l3_hit_rate_by_op``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from graphs.benchmarks.schema import LayerTag
from graphs.core.confidence import ConfidenceLevel
from graphs.hardware.architectural_energy import StoredProgramEnergyModel
from graphs.hardware.resource_model import HardwareResourceModel
from graphs.reporting.microarch_schema import LayerPanel
from graphs.reporting.layer_panels.layer1_alu import resolve_sku_resource_model
from graphs.reporting.layer_panels.layer2_register import (
    _process_node_nm,
    _tech_profile_for,
)


# Valid coherence-protocol classes; gates panel rendering against typos.
VALID_COHERENCE_PROTOCOLS = {"snoopy_mesi", "directory", "none"}


def _normalize_coherence(raw: Optional[str], sku_id: str) -> str:
    """Normalize coherence_protocol to canonical form; raise on typo."""
    val = (raw or "none").strip().lower()
    if val not in VALID_COHERENCE_PROTOCOLS:
        raise ValueError(
            f"Invalid coherence_protocol '{raw}' for SKU '{sku_id}'. "
            f"Expected one of: {sorted(VALID_COHERENCE_PROTOCOLS)}."
        )
    return val


def _has_l3(model: HardwareResourceModel) -> bool:
    """The SKU explicitly carries a distinct L3 layer."""
    return bool(model.l3_present) and (model.l3_cache_total or 0) > 0


def _provenance_or_theoretical(
    model: HardwareResourceModel, key: str,
) -> str:
    conf = model.get_provenance(key)
    if conf.level is ConfidenceLevel.UNKNOWN:
        return ConfidenceLevel.THEORETICAL.value.upper()
    return conf.level.value.upper()


def _l3_energy_pj_per_byte(model: HardwareResourceModel,
                           sku_id: str) -> float:
    tech = _tech_profile_for(model, sku_id)
    return tech.l3_cache_energy_per_byte_pj


def _coherence_protocol_pj(
    model: HardwareResourceModel, sku_id: str,
) -> float:
    """Coherence-protocol overhead in pJ per request, from the
    SKU's TechnologyProfile."""
    tech = _tech_profile_for(model, sku_id)
    return tech.coherence_energy_per_request_pj


def _default_l3_hit_rate_table() -> Dict[str, float]:
    """Pull the StoredProgramEnergyModel default per-op L3 hit rates."""
    fld = next(
        f for f in StoredProgramEnergyModel.__dataclass_fields__.values()
        if f.name == "l3_hit_rate_by_op"
    )
    return dict(fld.default_factory())


def _format_capacity(bytes_value: int) -> str:
    if bytes_value >= 1024 * 1024:
        return f"{bytes_value / (1024 * 1024):.1f}"
    if bytes_value > 0:
        return f"{bytes_value / 1024:.0f}"
    return "0"


def _capacity_unit(bytes_value: int) -> str:
    return "MiB" if bytes_value >= 1024 * 1024 else "KiB"


# Per-architecture explanatory note for SKUs without an L3 layer.
_NO_L3_NOTES: Dict[str, str] = {
    "shared-llc": (
        "L2 is the last-level cache on this SKU. There is no distinct "
        "L3 layer; demand misses go directly from L2 to DRAM."
    ),
    "kpu_dataflow": (
        "Domain-flow fabric: cross-tile data movement is routed via the "
        "M0.5 mesh-token mechanism rather than an inter-cluster cache. "
        "The transport energy lives at Layer 6, not here."
    ),
}


# --------------------------------------------------------------------
# Panel construction
# --------------------------------------------------------------------

def build_layer5_l3_cache_panel(
    sku_id: str,
    model: Optional[HardwareResourceModel] = None,
) -> LayerPanel:
    """Build the populated Layer 5 (L3 / LLC) panel for one SKU."""
    if model is None:
        model = resolve_sku_resource_model(sku_id)

    title = "Layer 5: L3 / Last-Level Cache"

    if model is None:
        return LayerPanel(
            layer=LayerTag.L3_CACHE,
            title=title,
            status="not_populated",
            summary=f"Resource model unavailable for {sku_id}.",
        )

    coherence = _normalize_coherence(model.coherence_protocol, sku_id)
    coh_pj = _coherence_protocol_pj(model, sku_id)
    metrics: Dict[str, Dict] = {}
    notes: List[str] = []

    if _has_l3(model):
        l3_total = model.l3_cache_total
        energy = _l3_energy_pj_per_byte(model, sku_id)
        metrics["L3 total"] = {
            "value": _format_capacity(l3_total),
            "unit":  _capacity_unit(l3_total),
            "provenance": _provenance_or_theoretical(model, "l3_cache_total"),
        }
        metrics["L3 read energy"] = {
            "value": f"{energy:.3f}",
            "unit":  "pJ/byte",
            "provenance": "THEORETICAL",
        }

        # Per-op hit rates apply only to coherent caches. Scratchpad-
        # managed L3 (KPU distributed SRAM) is software-staged so
        # the hit rate is deterministic 1.0 by construction.
        is_coherent = (model.coherence_protocol or "none").lower() != "none"
        if is_coherent:
            for op_kind, rate in _default_l3_hit_rate_table().items():
                metrics[f"Hit rate ({op_kind})"] = {
                    "value": f"{rate * 100:.0f}",
                    "unit":  "%",
                    "provenance": "THEORETICAL",
                }
            notes.append(
                "Per-op-type L3 hit rates from "
                "StoredProgramEnergyModel.l3_hit_rate_by_op (M5 lift). "
                "Matrix workloads with weight reuse retain L3 locality "
                "(~95% of L2 misses); elementwise streams that miss "
                "L1 + L2 have little reuse left (~70%)."
            )
        else:
            metrics["Hit rate (deterministic)"] = {
                "value": "100",
                "unit":  "%",
                "provenance": "n/a",
            }
            notes.append(
                "Software-managed distributed L3 SRAM: the host "
                "compiler stages data through this layer, so the hit "
                "rate is deterministic 1.0 by construction. Matches "
                "the M0.5 KPUTileEnergyModel abstraction."
            )
    else:
        # No L3: route via topology to pick the right note
        topology = (model.l2_topology or "").strip().lower()
        if topology == "shared-llc":
            kind = "shared-llc"
        else:
            # Fall through: KPU mesh / dataflow architectures
            kind = "kpu_dataflow"
        metrics["L3 status"] = {
            "value": "no L3 by design",
            "unit":  "",
            "provenance": _provenance_or_theoretical(model, "l3_present"),
        }
        notes.append(_NO_L3_NOTES[kind])

    metrics["Coherence protocol"] = {
        "value": coherence,
        "unit":  "",
        "provenance": _provenance_or_theoretical(model, "coherence_protocol"),
    }

    if coherence != "none":
        metrics["Coherence pJ / request"] = {
            "value": f"{coh_pj:.3f}",
            "unit":  "pJ",
            "provenance": "THEORETICAL",
        }
        notes.append(
            "Coherence-PROTOCOL energy at Layer 5: cost of snoop / "
            "directory messages and state-transition logic. "
            "TRANSPORT cost (NoC hops, ring traversal) is out of "
            "scope here -- it lives at Layer 6 (M6)."
        )
    else:
        notes.append(
            "No inter-core coherence: SIMT / dataflow / systolic "
            "architectures route data through shared memory or the "
            "mesh fabric rather than a coherence protocol. Layer 5 "
            "PROTOCOL cost is zero by design."
        )

    metrics["Process / market"] = {
        "value": f"{_process_node_nm(model)} nm",
        "unit":  "",
        "provenance": "n/a",
    }

    if _has_l3(model):
        summary = (
            f"L3 / LLC layer for {sku_id}: "
            f"{_format_capacity(model.l3_cache_total)} "
            f"{_capacity_unit(model.l3_cache_total)} shared at "
            f"{_l3_energy_pj_per_byte(model, sku_id):.3f} pJ/byte. "
            f"Coherence: {coherence}."
        )
    else:
        summary = (
            f"L3 / LLC layer for {sku_id}: no distinct L3 by design. "
            f"Coherence: {coherence}."
        )

    return LayerPanel(
        layer=LayerTag.L3_CACHE,
        title=title,
        status="theoretical",
        summary=summary,
        metrics=metrics,
        notes=notes,
    )


def build_layer5_panels_for_skus(
    sku_ids: List[str],
) -> Dict[str, LayerPanel]:
    return {sku: build_layer5_l3_cache_panel(sku) for sku in sku_ids}


# --------------------------------------------------------------------
# Cross-SKU comparison
# --------------------------------------------------------------------

@dataclass
class CrossSKULayer5Chart:
    skus: List[str]
    l3_present: Dict[str, bool]
    l3_total_bytes: Dict[str, int]
    coherence_protocol: Dict[str, str]
    coherence_pj_per_request: Dict[str, float]
    energy_pj_per_byte: Dict[str, float]
    provenance: Dict[str, str]


def cross_sku_layer5_chart(sku_ids: List[str]) -> CrossSKULayer5Chart:
    chart = CrossSKULayer5Chart(
        skus=list(sku_ids),
        l3_present={},
        l3_total_bytes={},
        coherence_protocol={},
        coherence_pj_per_request={},
        energy_pj_per_byte={},
        provenance={},
    )

    models: Dict[str, Optional[HardwareResourceModel]] = {
        sku: resolve_sku_resource_model(sku) for sku in sku_ids
    }

    for sku in sku_ids:
        m = models.get(sku)
        if m is None:
            continue
        # Gate L3 capacity / energy on _has_l3 so scratchpad SKUs
        # don't show phantom hardware-cache numbers, and gate
        # coherence energy on coherence != "none" so SIMT / dataflow
        # SKUs don't show phantom snoop costs. Keeps the chart
        # consistent with the per-SKU panel's classification.
        has_l3 = _has_l3(m)
        coherence = _normalize_coherence(m.coherence_protocol, sku)

        chart.l3_present[sku] = has_l3
        chart.l3_total_bytes[sku] = (m.l3_cache_total or 0) if has_l3 else 0
        chart.coherence_protocol[sku] = coherence
        chart.coherence_pj_per_request[sku] = (
            _coherence_protocol_pj(m, sku) if coherence != "none" else 0.0
        )
        chart.energy_pj_per_byte[sku] = (
            _l3_energy_pj_per_byte(m, sku) if has_l3 else 0.0
        )
        chart.provenance[sku] = _provenance_or_theoretical(m, "l3_present")

    return chart


__all__ = [
    "build_layer5_l3_cache_panel",
    "build_layer5_panels_for_skus",
    "cross_sku_layer5_chart",
    "CrossSKULayer5Chart",
    "VALID_COHERENCE_PROTOCOLS",
]
