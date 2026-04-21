"""
JSON data contract for the micro-architectural validation report.

One ``MicroarchReport`` is emitted per SKU per run of
``cli/microarch_validation_report.py``. The HTML template and PPT
generator consume the same JSON.

M0 ships the scaffolding with empty panels. M1-M7 populate layer content.
Layer 8 and Layer 9 panels remain placeholders during the M1-M7 delivery
push (system-level validation is out of scope for this epic).

See ``docs/plans/microarch-model-delivery-plan.md``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List
import json
from pathlib import Path

from graphs.benchmarks.schema import LayerTag


CONFIDENCE_LADDER = ["CALIBRATED", "INTERPOLATED", "THEORETICAL", "UNKNOWN"]


@dataclass
class LayerPanel:
    """
    Per-layer content for one SKU.

    Fields are intentionally empty for M0. M1-M7 populate them as the
    layers are modeled.

    Attributes:
        layer: One of the nine validation layers (plus COMPOSITE).
        title: Human-readable layer title for the panel header.
        status: One of 'not_populated', 'theoretical', 'interpolated',
            'calibrated'. Governs the confidence badge in the HTML.
        summary: One-paragraph summary shown at the top of the panel.
        metrics: Dict of named metrics (ops/s, pJ/op, GB/s, etc.).
            Each metric is a dict with 'value', 'unit', 'provenance'.
        charts: List of chart descriptors consumed by the HTML template.
        notes: Free-form notes (architecture-specific annotations).
    """
    layer: LayerTag = LayerTag.COMPOSITE
    title: str = ""
    status: str = "not_populated"
    summary: str = ""
    metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    charts: List[Dict[str, Any]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "layer": self.layer.value,
            "title": self.title,
            "status": self.status,
            "summary": self.summary,
            "metrics": self.metrics,
            "charts": self.charts,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LayerPanel":
        data = dict(data)
        if "layer" in data and isinstance(data["layer"], str):
            data["layer"] = LayerTag(data["layer"])
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class MicroarchReport:
    """
    Full micro-architectural report for one SKU.

    Attributes:
        sku: Canonical hardware ID (e.g., 'jetson_orin_agx_64gb').
        display_name: Human-readable SKU name.
        generated_at: ISO timestamp when the report was generated.
        plan_version: Reference to the plan this report implements.
        layers: Seven LayerPanel entries for Layers 1-7 in order
            (ALU, REGISTER, L1_CACHE, L2_CACHE, L3_CACHE,
            SOC_DATA_MOVEMENT, EXTERNAL_MEMORY). Layers 8 and 9 are
            placeholders in this delivery epic.
        archetype: 'simt', 'systolic', 'dataflow', 'cpu', 'dsp'.
            Used by compare_archetypes.html to cluster SKUs.
        overall_confidence: Derived from the weakest layer (min rule).
    """
    sku: str = ""
    display_name: str = ""
    generated_at: str = ""
    plan_version: str = "microarch-model-delivery-plan.md"
    layers: List[LayerPanel] = field(default_factory=list)
    archetype: str = ""
    overall_confidence: str = "UNKNOWN"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sku": self.sku,
            "display_name": self.display_name,
            "generated_at": self.generated_at,
            "plan_version": self.plan_version,
            "layers": [p.to_dict() for p in self.layers],
            "archetype": self.archetype,
            "overall_confidence": self.overall_confidence,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MicroarchReport":
        data = dict(data)
        if "layers" in data:
            data["layers"] = [LayerPanel.from_dict(p) for p in data["layers"]]
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Path) -> "MicroarchReport":
        return cls.from_dict(json.loads(Path(path).read_text()))


# The seven layers rendered in the per-SKU HTML, in display order.
# Layers 8 and 9 are present in LayerTag but are not shown as panels
# during the M1-M7 delivery push.
REPORT_LAYERS_IN_ORDER: List[tuple] = [
    (LayerTag.ALU, "Layer 1: ALU / MAC / Tensor Core"),
    (LayerTag.REGISTER, "Layer 2: Register File"),
    (LayerTag.L1_CACHE, "Layer 3: L1 Cache / Scratchpad"),
    (LayerTag.L2_CACHE, "Layer 4: L2 Cache"),
    (LayerTag.L3_CACHE, "Layer 5: L3 / Last-Level Cache"),
    (LayerTag.SOC_DATA_MOVEMENT, "Layer 6: SoC Data Movement"),
    (LayerTag.EXTERNAL_MEMORY, "Layer 7: External Memory"),
]


def empty_report(sku: str, display_name: str = "") -> MicroarchReport:
    """
    Build an unpopulated MicroarchReport for a SKU.

    All seven layer panels start in status 'not_populated'. M1-M7
    replace these in place.
    """
    now = datetime.now(timezone.utc).isoformat()
    panels = [
        LayerPanel(layer=tag, title=title, status="not_populated",
                   summary="NOT YET POPULATED")
        for tag, title in REPORT_LAYERS_IN_ORDER
    ]
    return MicroarchReport(
        sku=sku,
        display_name=display_name or sku,
        generated_at=now,
        layers=panels,
    )
