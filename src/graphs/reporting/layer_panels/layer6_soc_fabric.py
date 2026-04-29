"""
Layer 6 SoC fabric / on-chip data movement panel builder.

Surfaces per-SKU NoC topology, hop count, per-hop latency / energy,
bisection bandwidth, and controller count. The Layer 6 panel
captures the TRANSPORT cost of moving packets across the chip --
distinct from the coherence PROTOCOL cost (Layer 5) and from
external memory bandwidth (Layer 7).

Topology classes:
- ``crossbar``  -- single-hop full bipartite (GPU SM-to-L2, trivial
  single-tile fabrics)
- ``ring``      -- bidirectional ring bus (CPU multi-core)
- ``mesh_2d``   -- 2D mesh of tiles / PEs (KPU, Hailo)
- ``clos``      -- multi-stage Clos network (datacenter scale-out)
- ``full_mesh`` -- direct point-to-point (small clusters)
- ``unknown``   -- topology not classified

Per the M6 issue constraint, SKUs with thin datasheet sources ship
with the ``low_confidence`` flag set on their SoCFabricModel. The
panel renders an explicit warning rather than hiding the
uncertainty.

KPU SKUs: panel reads the fabric attached to ``model.soc_fabric``,
which the KPU factories populate with ``routing_distance_factor=1.2``
matching the legacy ``KPUTileEnergyModel.l3_routing_distance_factor``
constant. The energy model now reads from the same source -- single
source of truth.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from graphs.benchmarks.schema import LayerTag
from graphs.core.confidence import ConfidenceLevel
from graphs.hardware.fabric_model import SoCFabricModel, Topology
from graphs.hardware.resource_model import HardwareResourceModel
from graphs.reporting.microarch_schema import LayerPanel
from graphs.reporting.layer_panels.layer1_alu import resolve_sku_resource_model


# Per-topology narrative for the panel notes.
_TOPOLOGY_NOTES: Dict[Topology, str] = {
    Topology.CROSSBAR: (
        "Crossbar topology: every source has a direct dedicated path "
        "to every destination. Latency is constant (1 hop) regardless "
        "of source / destination pair. Used for GPU SM-to-L2 fabrics "
        "and small single-tile accelerators."
    ),
    Topology.RING: (
        "Ring topology: stops on a bidirectional bus. Average path "
        "length scales linearly with stop count (N/4). Used for "
        "x86 multi-core CPUs (Alder Lake double ring, AMD Infinity "
        "Fabric intra-CCX)."
    ),
    Topology.MESH_2D: (
        "2D-mesh topology: tiles arranged on a grid with XY routing. "
        "Average Manhattan distance is (W + H) / 3 hops; bisection "
        "bandwidth scales with the smaller mesh side. Used for "
        "tile-array accelerators (KPU domain-flow, Hailo dataflow)."
    ),
    Topology.CLOS: (
        "Clos network: multi-stage non-blocking switch. Avg path "
        "length scales as O(log N). Used in datacenter-scale fabrics."
    ),
    Topology.FULL_MESH: (
        "Full mesh: every node has a dedicated link to every other. "
        "Constant 1-hop latency but quadratic edge cost; only "
        "practical for small clusters."
    ),
    Topology.UNKNOWN: (
        "Topology not classified for this SKU."
    ),
}


def _has_fabric(model: HardwareResourceModel) -> bool:
    return model.soc_fabric is not None


def _provenance_or_theoretical(
    model: HardwareResourceModel, key: str,
) -> str:
    conf = model.get_provenance(key)
    if conf.level is ConfidenceLevel.UNKNOWN:
        return ConfidenceLevel.THEORETICAL.value.upper()
    return conf.level.value.upper()


def _topology_summary(fabric: SoCFabricModel) -> str:
    """One-word topology summary including mesh dims when applicable."""
    base = fabric.topology.value
    if fabric.topology is Topology.MESH_2D and fabric.mesh_dimensions:
        w, h = fabric.mesh_dimensions
        return f"{base} ({w}x{h})"
    if fabric.controller_count > 1:
        return f"{base} ({fabric.controller_count} ports)"
    return base


# --------------------------------------------------------------------
# Panel construction
# --------------------------------------------------------------------

def build_layer6_soc_fabric_panel(
    sku_id: str,
    model: Optional[HardwareResourceModel] = None,
) -> LayerPanel:
    """Build the Layer 6 (SoC data movement) panel for one SKU."""
    if model is None:
        model = resolve_sku_resource_model(sku_id)

    title = "Layer 6: SoC Data Movement"

    if model is None or not _has_fabric(model):
        return LayerPanel(
            layer=LayerTag.SOC_DATA_MOVEMENT,
            title=title,
            status="not_populated",
            summary=f"No SoC fabric model attached for {sku_id}.",
        )

    f = model.soc_fabric
    avg_hops = f.hop_count_avg()
    topo_str = _topology_summary(f)

    metrics: Dict[str, Dict] = {
        "Topology": {
            "value": topo_str,
            "unit":  "",
            "provenance": _provenance_or_theoretical(model, "soc_fabric"),
        },
        "Avg hop count": {
            "value": f"{avg_hops:.2f}",
            "unit":  "hops",
            "provenance": "THEORETICAL",
        },
        "Hop latency": {
            "value": f"{f.hop_latency_ns:.2f}",
            "unit":  "ns",
            "provenance": "THEORETICAL",
        },
        "Energy / flit / hop": {
            "value": f"{f.pj_per_flit_per_hop:.2f}",
            "unit":  "pJ",
            "provenance": "THEORETICAL",
        },
        "Bisection bandwidth": {
            "value": f"{f.bisection_bandwidth_gbps:.0f}",
            "unit":  "Gbps",
            "provenance": "THEORETICAL",
        },
        "Controller count": {
            "value": str(f.controller_count),
            "unit":  "",
            "provenance": "n/a",
        },
        "Flit size": {
            "value": str(f.flit_size_bytes),
            "unit":  "B",
            "provenance": "n/a",
        },
    }

    notes: List[str] = [_TOPOLOGY_NOTES.get(f.topology, "")]

    if f.routing_distance_factor != 1.0:
        notes.append(
            f"Routing distance factor of {f.routing_distance_factor:.2f} "
            "applied to shortest-path hop counts (accounts for "
            "non-Manhattan routes in the real NoC)."
        )

    if f.low_confidence:
        notes.append(
            "LOW CONFIDENCE: vendor does not publish per-hop NoC "
            "details; this panel ships analytical estimates with "
            "the assumption documented in the fabric provenance "
            "string. Per M6 issue constraint."
        )

    summary = (
        f"On-chip fabric for {sku_id}: {topo_str}. "
        f"Avg path = {avg_hops:.2f} hops at "
        f"{f.hop_latency_ns:.1f} ns / hop and "
        f"{f.pj_per_flit_per_hop:.1f} pJ / flit / hop."
    )
    if f.low_confidence:
        summary += " (low-confidence datasheet)"

    return LayerPanel(
        layer=LayerTag.SOC_DATA_MOVEMENT,
        title=title,
        status="theoretical",
        summary=summary,
        metrics=metrics,
        notes=notes,
    )


def build_layer6_panels_for_skus(
    sku_ids: List[str],
) -> Dict[str, LayerPanel]:
    return {sku: build_layer6_soc_fabric_panel(sku) for sku in sku_ids}


# --------------------------------------------------------------------
# Cross-SKU comparison
# --------------------------------------------------------------------

@dataclass
class CrossSKULayer6Chart:
    skus: List[str]
    topology: Dict[str, str]
    avg_hop_count: Dict[str, float]
    hop_latency_ns: Dict[str, float]
    pj_per_flit_per_hop: Dict[str, float]
    bisection_bandwidth_gbps: Dict[str, float]
    low_confidence: Dict[str, bool]
    provenance: Dict[str, str]


def cross_sku_layer6_chart(sku_ids: List[str]) -> CrossSKULayer6Chart:
    chart = CrossSKULayer6Chart(
        skus=list(sku_ids),
        topology={},
        avg_hop_count={},
        hop_latency_ns={},
        pj_per_flit_per_hop={},
        bisection_bandwidth_gbps={},
        low_confidence={},
        provenance={},
    )

    models: Dict[str, Optional[HardwareResourceModel]] = {
        sku: resolve_sku_resource_model(sku) for sku in sku_ids
    }

    for sku in sku_ids:
        m = models.get(sku)
        if m is None or not _has_fabric(m):
            continue
        f = m.soc_fabric
        chart.topology[sku] = _topology_summary(f)
        chart.avg_hop_count[sku] = f.hop_count_avg()
        chart.hop_latency_ns[sku] = f.hop_latency_ns
        chart.pj_per_flit_per_hop[sku] = f.pj_per_flit_per_hop
        chart.bisection_bandwidth_gbps[sku] = f.bisection_bandwidth_gbps
        chart.low_confidence[sku] = f.low_confidence
        chart.provenance[sku] = _provenance_or_theoretical(m, "soc_fabric")

    return chart


__all__ = [
    "build_layer6_soc_fabric_panel",
    "build_layer6_panels_for_skus",
    "cross_sku_layer6_chart",
    "CrossSKULayer6Chart",
]
