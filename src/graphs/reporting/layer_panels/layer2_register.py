"""
Layer 2 Register-File panel builder.

Surfaces per-SKU register-file read/write energy plus the two
analytical coefficients M2 lifts out of hard-coded constants:

  - SIMD-vectorization efficiency (CPU/DSP). When the SKU's resource
    model carries ``simd_efficiency`` populated by M2, this panel
    reports the per-op-kind values with their provenance.
  - Systolic / wavefront pipeline-fill overhead (TPU). When the
    resource model carries ``pipeline_fill_overhead`` populated by
    M2, this panel reports the value with its provenance.
  - KPU fill / drain. M0.5 already accounts for these via
    ``TileSpecialization.pipeline_fill_cycles`` and
    ``pipeline_drain_cycles``. The Layer 2 panel reads them
    directly; it never recomputes them.

Register read / write energies come from the ``TechnologyProfile``
keyed by ``_process_node_nm(model)`` and the SKU's deployment market
(datacenter / edge / mobile). The panel surfaces the read energy as
a fraction of the SKU's representative ALU energy so the dataflow
advantage of the KPU shows up directly: domain-flow tiles avoid
the register fetch entirely, so the ratio is effectively zero.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from graphs.benchmarks.schema import LayerTag
from graphs.core.confidence import ConfidenceLevel
from graphs.hardware.resource_model import (
    HardwareResourceModel,
    HardwareType,
)
from graphs.hardware.technology_profile import (
    MemoryType,
    TechnologyProfile,
)
from graphs.reporting.microarch_schema import LayerPanel
from graphs.reporting.layer_panels.layer1_alu import resolve_sku_resource_model


# Per-SKU mapping to the deployment market used by TechnologyProfile.
# Drives register-energy scaling (datacenter cores have more rename
# registers + more ports => higher access energy).
_SKU_MARKET: Dict[str, str] = {
    "jetson_orin_agx_64gb":  "edge",
    "intel_core_i7_12700k":  "datacenter",  # consumer desktop ~ datacenter
    "ryzen_9_8945hs":        "mobile",      # laptop SoC
    "kpu_t64":               "edge",
    "kpu_t128":              "edge",
    "kpu_t256":              "edge",
    "coral_edge_tpu":        "edge",
    "hailo8":                "edge",
    "hailo10h":              "edge",
}

# Per-SKU memory type, used as a tie-breaker in TechnologyProfile.create.
_SKU_MEMORY: Dict[str, MemoryType] = {
    "jetson_orin_agx_64gb":  MemoryType.LPDDR5,
    "intel_core_i7_12700k":  MemoryType.DDR5,
    "ryzen_9_8945hs":        MemoryType.LPDDR5X,
    "kpu_t64":               MemoryType.LPDDR5,
    "kpu_t128":              MemoryType.LPDDR5,
    "kpu_t256":              MemoryType.LPDDR5,
    "coral_edge_tpu":        MemoryType.LPDDR4,
    "hailo8":                MemoryType.LPDDR4,
    "hailo10h":              MemoryType.LPDDR5,
}


def _process_node_nm(model: HardwareResourceModel) -> int:
    """
    Pick a representative process node for the SKU.

    HardwareResourceModel carries process node per ComputeFabric, not
    on the model itself. Use the highest-throughput fabric's node so
    the dominant compute path drives Layer 2 energy. Falls back to 8 nm
    when the model has no fabrics.
    """
    fabrics = list(model.compute_fabrics or [])
    if not fabrics:
        return 8
    # Pick the fabric with the largest unit count as a proxy for
    # "primary path" (matches how ALU peak is dominated).
    primary = max(fabrics, key=lambda f: f.num_units)
    return primary.process_node_nm or 8


def _tech_profile_for(model: HardwareResourceModel,
                      sku_id: str) -> TechnologyProfile:
    """Build a TechnologyProfile for the SKU's process node + market."""
    market = _SKU_MARKET.get(sku_id, "edge")
    mem = _SKU_MEMORY.get(sku_id, MemoryType.LPDDR5)
    return TechnologyProfile.create(
        process_node_nm=_process_node_nm(model),
        memory_type=mem,
        target_market=market,
    )


def _representative_alu_energy_pj(model: HardwareResourceModel) -> float:
    """
    Pick a representative ALU energy (pJ per FP32 op) for the
    register-as-fraction-of-ALU ratio.

    Uses the model's ``energy_per_flop_fp32`` if positive, otherwise
    the first compute fabric's energy. Never zero; falls back to the
    process-node baseline.
    """
    if model.energy_per_flop_fp32 and model.energy_per_flop_fp32 > 0:
        return model.energy_per_flop_fp32 * 1e12  # J -> pJ
    fabrics = list(model.compute_fabrics or [])
    if fabrics and fabrics[0].energy_per_flop_fp32 > 0:
        return fabrics[0].energy_per_flop_fp32 * 1e12
    return 1.9  # 8nm baseline pJ/FP32op


def _kpu_fill_drain_overhead(
    model: HardwareResourceModel,
) -> Optional[Tuple[int, int]]:
    """
    Read fill / drain cycle counts from the M0.5 KPU tile abstraction.

    Returns (fill_cycles, drain_cycles) for the first tile
    specialization, or None when the model doesn't carry one
    (non-KPU SKUs -- GPU / CPU / NPU / etc.). Hailo dataflow NPUs
    are architecturally similar to KPUs but don't populate
    tile_specializations on their thermal_operating_points, so this
    KPU-gated branch correctly excludes them.
    """
    if model.hardware_type is not HardwareType.KPU:
        return None
    # The M0.5 KPU resource models attach KPUComputeResource via
    # thermal operating points. Read the first tile spec we find.
    for tp in (model.thermal_operating_points or {}).values():
        for spec in tp.performance_specs.values():
            cr = getattr(spec, "compute_resource", None)
            if cr is None:
                continue
            tile_specs = getattr(cr, "tile_specializations", None) or []
            if tile_specs:
                ts = tile_specs[0]
                return (ts.pipeline_fill_cycles, ts.pipeline_drain_cycles)
    return None


# --------------------------------------------------------------------
# Panel construction
# --------------------------------------------------------------------

def _format_energy_pj(value: float) -> str:
    if value < 0.01:
        return f"{value:.4f}"
    if value < 1.0:
        return f"{value:.3f}"
    return f"{value:.2f}"


def _provenance_or_theoretical(
    model: HardwareResourceModel, key: str,
) -> str:
    """
    Read provenance for ``key``; if UNKNOWN but the field is populated
    (caller already checked), report THEORETICAL by convention.
    """
    conf = model.get_provenance(key)
    if conf.level is ConfidenceLevel.UNKNOWN:
        return ConfidenceLevel.THEORETICAL.value.upper()
    return conf.level.value.upper()


def build_layer2_register_panel(
    sku_id: str,
    model: Optional[HardwareResourceModel] = None,
) -> LayerPanel:
    """Build the populated Layer 2 (Register File) panel for one SKU."""
    if model is None:
        model = resolve_sku_resource_model(sku_id)

    title = "Layer 2: Register File"

    if model is None:
        return LayerPanel(
            layer=LayerTag.REGISTER,
            title=title,
            status="not_populated",
            summary=f"Resource model unavailable for {sku_id}.",
        )

    tech = _tech_profile_for(model, sku_id)
    rread = tech.register_read_energy_pj
    rwrite = tech.register_write_energy_pj
    alu_pj = _representative_alu_energy_pj(model)
    ratio = rread / alu_pj if alu_pj > 0 else 0.0

    metrics: Dict[str, Dict] = {
        "Register read": {
            "value": _format_energy_pj(rread),
            "unit":  "pJ/access",
            "provenance": "THEORETICAL",
        },
        "Register write": {
            "value": _format_energy_pj(rwrite),
            "unit":  "pJ/access",
            "provenance": "THEORETICAL",
        },
        "Read / ALU ratio": {
            "value": f"{ratio:.2f}",
            "unit":  "pJ/pJ",
            "provenance": "THEORETICAL",
        },
        "Process / market": {
            "value": f"{_process_node_nm(model)} nm / {tech.target_market}",
            "unit":  "",
            "provenance": "n/a",
        },
    }

    notes: List[str] = []

    # SIMD efficiency (CPU/DSP)
    if model.simd_efficiency:
        for op_kind, eff in model.simd_efficiency.items():
            metrics[f"SIMD eff ({op_kind})"] = {
                "value": f"{eff:.2f}",
                "unit":  "ratio",
                "provenance": _provenance_or_theoretical(
                    model, f"simd_efficiency.{op_kind}"
                ),
            }
        notes.append(
            "SIMD-vectorization efficiency is the fraction of theoretical "
            "vector throughput that survives ISA overhead, alignment, and "
            "tail-loop costs. Lifted from CPUMapper analytical defaults."
        )

    # TPU pipeline-fill overhead
    if model.pipeline_fill_overhead is not None:
        metrics["Pipeline fill overhead"] = {
            "value": f"{model.pipeline_fill_overhead:.3f}",
            "unit":  "fraction",
            "provenance": _provenance_or_theoretical(
                model, "pipeline_fill_overhead"
            ),
        }
        notes.append(
            "Pipeline fill overhead = fraction of cycles spent filling "
            "and draining the systolic array before sustained throughput. "
            "Formula in TPUMapper averaged over typical edge-AI matrices."
        )

    # KPU fill / drain (read from M0.5; do not recompute)
    fd = _kpu_fill_drain_overhead(model)
    if fd is not None:
        fill, drain = fd
        metrics["Tile fill cycles"] = {
            "value": str(fill),
            "unit":  "cycles",
            "provenance": "THEORETICAL",
        }
        metrics["Tile drain cycles"] = {
            "value": str(drain),
            "unit":  "cycles",
            "provenance": "THEORETICAL",
        }
        notes.append(
            "KPU tile fill / drain cycles read directly from "
            "TileSpecialization (M0.5). The dataflow fabric has no "
            "register file in the conventional sense -- token payloads "
            "land in PE accumulators rather than a renamed RF."
        )

    summary = (
        f"Register-file read costs {_format_energy_pj(rread)} pJ per "
        f"access at {_process_node_nm(model)} nm ({tech.target_market} "
        f"market), or {ratio:.2f} pJ per pJ of representative ALU work. "
        f"Domain-flow architectures (KPU) avoid this fetch entirely."
    )

    # Aggregate status: all M2 fields land THEORETICAL.
    status = "theoretical"

    return LayerPanel(
        layer=LayerTag.REGISTER,
        title=title,
        status=status,
        summary=summary,
        metrics=metrics,
        notes=notes,
    )


def build_layer2_panels_for_skus(
    sku_ids: List[str],
) -> Dict[str, LayerPanel]:
    """Build Layer 2 panels for a batch of SKU ids."""
    return {sku: build_layer2_register_panel(sku) for sku in sku_ids}


# --------------------------------------------------------------------
# Cross-SKU register-energy comparison
# --------------------------------------------------------------------

@dataclass
class CrossSKULayer2Chart:
    """Per-SKU register-as-fraction-of-ALU plus raw read/write energies."""
    skus: List[str]
    register_read_pj: Dict[str, float]
    register_write_pj: Dict[str, float]
    read_alu_ratio: Dict[str, float]
    provenance: Dict[str, str]


def cross_sku_layer2_chart(
    sku_ids: List[str],
) -> CrossSKULayer2Chart:
    """Build the Layer 2 cross-SKU register-energy chart data."""
    chart = CrossSKULayer2Chart(
        skus=list(sku_ids),
        register_read_pj={},
        register_write_pj={},
        read_alu_ratio={},
        provenance={},
    )

    # Resolve once per SKU; tech profile is cheap to rebuild.
    models: Dict[str, Optional[HardwareResourceModel]] = {
        sku: resolve_sku_resource_model(sku) for sku in sku_ids
    }

    for sku in sku_ids:
        model = models.get(sku)
        if model is None:
            continue
        tech = _tech_profile_for(model, sku)
        rread = tech.register_read_energy_pj
        rwrite = tech.register_write_energy_pj
        alu = _representative_alu_energy_pj(model)
        chart.register_read_pj[sku] = rread
        chart.register_write_pj[sku] = rwrite
        chart.read_alu_ratio[sku] = rread / alu if alu > 0 else 0.0
        # Per-SKU register provenance is always THEORETICAL at M2.
        chart.provenance[sku] = ConfidenceLevel.THEORETICAL.value.upper()

    return chart


__all__ = [
    "build_layer2_register_panel",
    "build_layer2_panels_for_skus",
    "cross_sku_layer2_chart",
    "CrossSKULayer2Chart",
]
