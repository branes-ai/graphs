"""
Layer 1 ALU panel builder.

Produces a populated ``LayerPanel`` (with title, status, summary, and
per-precision metrics) from a ``HardwareResourceModel`` whose
``compute_fabrics`` field has been populated. The panel surfaces:

  - peak ops/s per precision (across all fabrics)
  - per-fabric ops/clock/unit and unit count
  - per-precision provenance tag (CALIBRATED / INTERPOLATED /
    THEORETICAL / UNKNOWN), driven by ``model.field_provenance``
  - the dominant process node (used for energy normalization)

Also exposes a ``cross_sku_layer1_chart`` helper for the per-precision
cross-SKU peak-TOPS comparison rendered on the comparison page.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from graphs.benchmarks.schema import LayerTag
from graphs.core.confidence import ConfidenceLevel
from graphs.hardware.resource_model import (
    ComputeFabric,
    HardwareResourceModel,
    Precision,
)
from graphs.reporting.microarch_schema import LayerPanel


# Precisions exposed in the Layer 1 panel, in display order.
LAYER1_PRECISIONS: Tuple[Precision, ...] = (
    Precision.FP64,
    Precision.FP32,
    Precision.TF32,
    Precision.BF16,
    Precision.FP16,
    Precision.FP8,
    Precision.INT8,
    Precision.INT4,
)


# --------------------------------------------------------------------
# SKU id -> resource model resolution
# --------------------------------------------------------------------
# The micro-arch report's SKU IDs (lowercase, snake_case) map to the
# resource-model factory functions in graphs.hardware.models. Keeping
# the table here avoids a hard dependency on the mapper registry,
# which uses display names ("Intel-i7-12700K") rather than SKU IDs.

def _build_sku_factory_map() -> Dict[str, Callable[[], HardwareResourceModel]]:
    """
    Build the SKU id -> resource-model factory map.

    Imports are deferred to the function body so importing this module
    does not transitively load every model in the catalog (each model
    file pulls in heavy resource-model machinery).
    """
    from graphs.hardware.models import (
        jetson_orin_agx_64gb_resource_model,
        intel_core_i7_12700k_resource_model,
        ryzen_9_8945hs_resource_model,
        coral_edge_tpu_resource_model,
    )
    from graphs.hardware.models.accelerators.kpu_t64 import (
        kpu_t64_resource_model,
    )
    from graphs.hardware.models.accelerators.kpu_t128 import (
        kpu_t128_resource_model,
    )
    from graphs.hardware.models.accelerators.kpu_t256 import (
        kpu_t256_resource_model,
    )

    factory: Dict[str, Callable[[], HardwareResourceModel]] = {
        "jetson_orin_agx_64gb":  jetson_orin_agx_64gb_resource_model,
        "intel_core_i7_12700k":  intel_core_i7_12700k_resource_model,
        "ryzen_9_8945hs":        ryzen_9_8945hs_resource_model,
        "kpu_t64":               kpu_t64_resource_model,
        "kpu_t128":              kpu_t128_resource_model,
        "kpu_t256":              kpu_t256_resource_model,
        "coral_edge_tpu":        coral_edge_tpu_resource_model,
    }

    # Hailo SKUs are optional (model files exist but may fail to import
    # in trimmed environments). Fall back to None on import error.
    try:
        from graphs.hardware.models.edge.hailo8 import hailo8_resource_model
        factory["hailo8"] = hailo8_resource_model
    except Exception:
        pass
    try:
        from graphs.hardware.models.edge.hailo10h import hailo10h_resource_model
        factory["hailo10h"] = hailo10h_resource_model
    except Exception:
        pass

    return factory


def resolve_sku_resource_model(
    sku_id: str,
) -> Optional[HardwareResourceModel]:
    """
    Resolve a SKU id (e.g., ``'kpu_t128'``) to a ``HardwareResourceModel``.

    Returns ``None`` if the SKU id is not registered.
    """
    factory_map = _build_sku_factory_map()
    factory = factory_map.get(sku_id)
    return factory() if factory else None


# --------------------------------------------------------------------
# Per-fabric / per-precision aggregation
# --------------------------------------------------------------------

def _peak_ops_per_sec(
    model: HardwareResourceModel,
    precision: Precision,
) -> float:
    """
    Sum peak ops/sec across all fabrics for one precision.

    Falls back to 0.0 if no fabric advertises this precision.
    """
    fabrics: List[ComputeFabric] = list(model.compute_fabrics or [])
    return sum(f.get_peak_ops_per_sec(precision) for f in fabrics)


def _provenance_tag(
    model: HardwareResourceModel,
    precision: Precision,
) -> str:
    """
    Read the field-provenance tag for a precision's ALU rate.

    Default: when no provenance is recorded, return THEORETICAL --
    a populated ComputeFabric is, by construction, derived from the
    datasheet. Returning UNKNOWN would mis-represent these as
    ungrounded.
    """
    conf = model.get_provenance(
        f"compute_fabric.ops_per_clock.{precision.value}"
    )
    if conf.level is ConfidenceLevel.UNKNOWN:
        # Populated ComputeFabric without an explicit tag is, by the
        # populating-from-datasheet convention, theoretical.
        return ConfidenceLevel.THEORETICAL.value.upper()
    return conf.level.value.upper()


def _format_ops(value: float) -> str:
    """Pretty-format a peak-ops-per-sec value with appropriate unit."""
    if value >= 1e12:
        return f"{value / 1e12:.2f}"
    if value >= 1e9:
        return f"{value / 1e9:.1f}"
    return f"{value / 1e6:.0f}"


def _ops_unit(value: float) -> str:
    if value >= 1e12:
        return "TOPS"
    if value >= 1e9:
        return "GOPS"
    return "MOPS"


def _summarize_fabrics(model: HardwareResourceModel) -> str:
    """Build a short fabric-list summary, e.g. 'CUDA cores + tensor cores'."""
    fabrics = list(model.compute_fabrics or [])
    if not fabrics:
        return "no compute fabrics declared"
    parts = []
    for f in fabrics:
        parts.append(
            f"{f.fabric_type} ({f.num_units} units @ "
            f"{f.core_frequency_hz / 1e9:.2f} GHz, {f.process_node_nm} nm)"
        )
    return "; ".join(parts)


def _aggregate_status(
    model: HardwareResourceModel,
    precisions: List[Precision],
) -> str:
    """
    Pick the worst (most pessimistic) provenance level across the
    populated precisions. Drives the panel badge.
    """
    levels = []
    for p in precisions:
        # Only include precisions that are actually populated
        if _peak_ops_per_sec(model, p) > 0:
            levels.append(_provenance_tag(model, p))
    if not levels:
        return "not_populated"
    rank = {"CALIBRATED": 3, "INTERPOLATED": 2, "THEORETICAL": 1, "UNKNOWN": 0}
    worst = min(levels, key=lambda x: rank.get(x, 0))
    return worst.lower()


# --------------------------------------------------------------------
# Panel construction
# --------------------------------------------------------------------

def build_layer1_panel(
    sku_id: str,
    model: Optional[HardwareResourceModel] = None,
) -> LayerPanel:
    """
    Build the populated Layer 1 (ALU) panel for one SKU.

    If ``model`` is None, attempts to resolve it via ``sku_id``. When
    no resource model is available, returns a 'not_populated' panel.
    """
    if model is None:
        model = resolve_sku_resource_model(sku_id)

    title = "Layer 1: ALU / MAC / Tensor Core"

    if model is None or not (model.compute_fabrics or []):
        return LayerPanel(
            layer=LayerTag.ALU,
            title=title,
            status="not_populated",
            summary=f"No compute_fabrics populated for {sku_id}.",
        )

    metrics: Dict[str, Dict] = {}
    for prec in LAYER1_PRECISIONS:
        peak = _peak_ops_per_sec(model, prec)
        if peak <= 0:
            continue  # skip precisions this SKU does not advertise
        metrics[f"{prec.value.upper()} peak"] = {
            "value": _format_ops(peak),
            "unit":  _ops_unit(peak),
            "provenance": _provenance_tag(model, prec),
        }

    # Per-fabric breakdown (one metric per fabric, ops/clock for the
    # default precision). Helps readers see how the peak is composed.
    for f in (model.compute_fabrics or []):
        # Pick the highest-throughput precision this fabric supports
        if not f.ops_per_unit_per_clock:
            continue
        primary = max(
            f.ops_per_unit_per_clock.items(),
            key=lambda kv: kv[1],
        )
        prec, ops = primary
        metrics[f"{f.fabric_type} ({prec.value})"] = {
            "value": f"{f.num_units} x {ops}",
            "unit":  "ops/clk",
            "provenance": "n/a",
        }

    summary = (
        f"Per-precision peak throughput from {len(model.compute_fabrics)} "
        f"compute fabric(s): {_summarize_fabrics(model)}. Provenance "
        f"reflects whether the per-precision ops/clock came from a "
        f"datasheet (THEORETICAL), an interpolated calibration "
        f"(INTERPOLATED), or a measured Layer-1 FMA-rate sweep "
        f"(CALIBRATED)."
    )

    status = _aggregate_status(model, list(LAYER1_PRECISIONS))

    panel = LayerPanel(
        layer=LayerTag.ALU,
        title=title,
        status=status,
        summary=summary,
        metrics=metrics,
        notes=[
            "Layer 1 panel sums peak ops/sec across every populated "
            "ComputeFabric. The Layer 1 fitter "
            "(graphs.calibration.fitters.layer1_alu_fitter) replaces "
            "datasheet rates with measured values when available.",
        ],
    )
    return panel


def build_layer1_panels_for_skus(
    sku_ids: List[str],
) -> Dict[str, LayerPanel]:
    """Build Layer 1 panels for a batch of SKU ids."""
    return {sku: build_layer1_panel(sku) for sku in sku_ids}


# --------------------------------------------------------------------
# Cross-SKU comparison chart (per-precision peak TOPS)
# --------------------------------------------------------------------

@dataclass
class CrossSKUChart:
    """
    Chart-ready data for the cross-SKU Layer 1 comparison.

    Per-precision rows of (sku, peak_ops_per_sec, provenance_tag).
    Consumers (HTML + PPTX renderers) translate this into the actual
    plot.
    """
    precisions: List[str]
    skus: List[str]
    # Map (precision, sku) -> peak ops/sec
    peak_ops: Dict[Tuple[str, str], float]
    # Map (precision, sku) -> provenance tag
    provenance: Dict[Tuple[str, str], str]


def cross_sku_layer1_chart(
    sku_ids: List[str],
) -> CrossSKUChart:
    """
    Build the Layer 1 cross-SKU peak-TOPS chart data.

    Skips precisions that no SKU advertises.
    """
    chart = CrossSKUChart(precisions=[], skus=list(sku_ids),
                          peak_ops={}, provenance={})

    for prec in LAYER1_PRECISIONS:
        any_value = False
        for sku in sku_ids:
            model = resolve_sku_resource_model(sku)
            if model is None:
                continue
            peak = _peak_ops_per_sec(model, prec)
            if peak <= 0:
                continue
            chart.peak_ops[(prec.value, sku)] = peak
            chart.provenance[(prec.value, sku)] = _provenance_tag(model, prec)
            any_value = True
        if any_value:
            chart.precisions.append(prec.value)

    return chart
