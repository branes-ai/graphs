"""
Layer 7 external memory panel builder.

Surfaces per-SKU external (DRAM) memory characteristics: technology
class, peak bandwidth, read / write energy with explicit asymmetry,
and access-pattern overhead (sequential / strided / random) from
the energy model.

The cross-SKU view also surfaces a bandwidth-vs-pJ/byte data table --
the visually striking comparison that motivates the M7 milestone:
modern DDR5/LPDDR5 sit at ~10-25 pJ/B; older LPDDR4 climbs to 20+;
HBM3 (not currently in the catalog) drops below 10 pJ/B at the cost
of higher die area.

For SKUs whose deployment is on-chip-dominated (Hailo), the
``memory_technology`` string is annotated to reflect the host-vs-
on-chip split rather than papering over it.
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


def _provenance_or_theoretical(
    model: HardwareResourceModel, key: str,
) -> str:
    conf = model.get_provenance(key)
    if conf.level is ConfidenceLevel.UNKNOWN:
        return ConfidenceLevel.THEORETICAL.value.upper()
    return conf.level.value.upper()


def _read_energy_pj(model: HardwareResourceModel) -> float:
    """Return the read energy per byte in pJ. Falls back to the legacy
    symmetric ``energy_per_byte`` when no asymmetric value is set."""
    if model.memory_read_energy_per_byte_pj is not None:
        return model.memory_read_energy_per_byte_pj
    return model.energy_per_byte * 1e12


def _write_energy_pj(model: HardwareResourceModel) -> float:
    """Return the write energy per byte in pJ. Falls back to the legacy
    symmetric ``energy_per_byte`` when no asymmetric value is set."""
    if model.memory_write_energy_per_byte_pj is not None:
        return model.memory_write_energy_per_byte_pj
    return model.energy_per_byte * 1e12


def _default_access_pattern_table() -> Dict[str, float]:
    """Pull the default access-pattern multipliers from the energy
    model's dataclass field default."""
    fld = next(
        f for f in StoredProgramEnergyModel.__dataclass_fields__.values()
        if f.name == "memory_access_pattern_multiplier"
    )
    return dict(fld.default_factory())


# --------------------------------------------------------------------
# Panel construction
# --------------------------------------------------------------------

def build_layer7_external_memory_panel(
    sku_id: str,
    model: Optional[HardwareResourceModel] = None,
) -> LayerPanel:
    """Build the Layer 7 (external memory) panel for one SKU."""
    if model is None:
        model = resolve_sku_resource_model(sku_id)

    title = "Layer 7: External Memory"

    if model is None or not model.peak_bandwidth:
        return LayerPanel(
            layer=LayerTag.EXTERNAL_MEMORY,
            title=title,
            status="not_populated",
            summary=f"No external memory model attached for {sku_id}.",
        )

    tech = model.memory_technology or "(unspecified)"
    bw_gbs = model.peak_bandwidth / 1e9
    rd = _read_energy_pj(model)
    wr = _write_energy_pj(model)
    asymmetry = wr / rd if rd > 0 else 1.0

    metrics: Dict[str, Dict] = {
        "Memory technology": {
            "value": tech,
            "unit":  "",
            "provenance": _provenance_or_theoretical(model, "memory_technology"),
        },
        "Peak bandwidth": {
            "value": f"{bw_gbs:.1f}",
            "unit":  "GB/s",
            "provenance": _provenance_or_theoretical(model, "peak_bandwidth"),
        },
        "Read energy": {
            "value": f"{rd:.2f}",
            "unit":  "pJ/B",
            "provenance": _provenance_or_theoretical(
                model, "memory_read_energy_per_byte_pj"
            ),
        },
        "Write energy": {
            "value": f"{wr:.2f}",
            "unit":  "pJ/B",
            "provenance": _provenance_or_theoretical(
                model, "memory_write_energy_per_byte_pj"
            ),
        },
        "W/R asymmetry": {
            "value": f"{asymmetry:.2f}x",
            "unit":  "ratio",
            "provenance": "n/a",
        },
    }

    # Access-pattern multipliers from the energy-model defaults
    for pattern, mult in _default_access_pattern_table().items():
        metrics[f"Access pattern ({pattern})"] = {
            "value": f"{mult:.2f}x",
            "unit":  "ratio",
            "provenance": "THEORETICAL",
        }

    notes: List[str] = [
        "Layer 7 captures the cost of moving bytes between the chip "
        "and external memory. Access-pattern multipliers model "
        "row-buffer locality: sequential streams amortize row "
        "activation across many bytes; random access pays full "
        "row-activation energy per request."
    ]

    if "on-chip" in tech.lower():
        notes.append(
            "This SKU's deployment loads model weights into on-chip "
            "SRAM at initialization and runs steady-state inference "
            "without DRAM traffic. The Layer 7 numbers describe the "
            "host-side memory used during weight loading, not the "
            "steady-state inference loop."
        )

    summary = (
        f"External memory for {sku_id}: {tech}, "
        f"{bw_gbs:.1f} GB/s peak. "
        f"Reads at {rd:.2f} pJ/B; writes at {wr:.2f} pJ/B "
        f"({asymmetry:.2f}x asymmetry)."
    )

    return LayerPanel(
        layer=LayerTag.EXTERNAL_MEMORY,
        title=title,
        status="theoretical",
        summary=summary,
        metrics=metrics,
        notes=notes,
    )


def build_layer7_panels_for_skus(
    sku_ids: List[str],
) -> Dict[str, LayerPanel]:
    return {sku: build_layer7_external_memory_panel(sku) for sku in sku_ids}


# --------------------------------------------------------------------
# Cross-SKU comparison: bandwidth vs pJ/byte
# --------------------------------------------------------------------

@dataclass
class CrossSKULayer7Chart:
    skus: List[str]
    memory_technology: Dict[str, str]
    peak_bandwidth_gbps: Dict[str, float]
    read_energy_pj: Dict[str, float]
    write_energy_pj: Dict[str, float]
    asymmetry: Dict[str, float]
    provenance: Dict[str, str]


def cross_sku_layer7_chart(sku_ids: List[str]) -> CrossSKULayer7Chart:
    chart = CrossSKULayer7Chart(
        skus=list(sku_ids),
        memory_technology={},
        peak_bandwidth_gbps={},
        read_energy_pj={},
        write_energy_pj={},
        asymmetry={},
        provenance={},
    )

    models: Dict[str, Optional[HardwareResourceModel]] = {
        sku: resolve_sku_resource_model(sku) for sku in sku_ids
    }

    for sku in sku_ids:
        m = models.get(sku)
        if m is None or not m.peak_bandwidth:
            continue
        chart.memory_technology[sku] = (
            m.memory_technology or "(unspecified)"
        )
        chart.peak_bandwidth_gbps[sku] = m.peak_bandwidth / 1e9
        rd = _read_energy_pj(m)
        wr = _write_energy_pj(m)
        chart.read_energy_pj[sku] = rd
        chart.write_energy_pj[sku] = wr
        chart.asymmetry[sku] = wr / rd if rd > 0 else 1.0
        chart.provenance[sku] = _provenance_or_theoretical(
            m, "memory_technology"
        )

    return chart


__all__ = [
    "build_layer7_external_memory_panel",
    "build_layer7_panels_for_skus",
    "cross_sku_layer7_chart",
    "CrossSKULayer7Chart",
]
