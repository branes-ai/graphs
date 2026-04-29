"""
SKU id resolution for the validation pipeline.

Three identifier spaces collide here:

- **Microarch report SKU id** (``intel_core_i7_12700k``, ``kpu_t128``)
  used by the M1-M7 panels and ``HardwareResourceModel`` factories.
- **Calibration data directory name** (``intel_core_i7_12700k``,
  ``jetson_orin_agx_30w``) under ``calibration_data/`` -- includes
  thermal profile in the name for Jetson SKUs.
- **UnifiedAnalyzer mapper name** (``i7-12700k``,
  ``jetson-orin-agx-64gb``) used by ``UnifiedAnalyzer.analyze_model``.

The three tables below resolve any one to the others. Missing
entries return None, which the validation pipeline treats as
"no measurement data for this SKU."
"""
from __future__ import annotations

from typing import Optional


# Microarch SKU id -> calibration_data/<dir> name.
# When a SKU has multiple thermal profiles in calibration_data
# (Jetson at 15W / 30W / 50W / MAXN), pick the one that matches the
# thermal_profile reported by the resource model factory.
_SKU_TO_CALIBRATION_DIR = {
    "intel_core_i7_12700k":  "intel_core_i7_12700k",
    "ryzen_9_8945hs":        "ryzen_9_8945hs",
    "jetson_orin_agx_64gb":  "jetson_orin_agx_30w",  # 30W matches M1-M7 baseline
    # KPU / TPU / Hailo: no measurement data
    "kpu_t64":               None,
    "kpu_t128":              None,
    "kpu_t256":              None,
    "coral_edge_tpu":        None,
    "hailo8":                None,
    "hailo10h":              None,
}


# Microarch SKU id -> UnifiedAnalyzer mapper name.
_SKU_TO_MAPPER_NAME = {
    "intel_core_i7_12700k":  "i7-12700k",
    "ryzen_9_8945hs":        "ryzen",
    "jetson_orin_agx_64gb":  "jetson-orin-agx-64gb",
    "kpu_t64":               "kpu-t64",
    "kpu_t128":              None,                   # not in mapper registry
    "kpu_t256":              "kpu-t256",
    "coral_edge_tpu":        "coral-edge-tpu",
    "hailo8":                "hailo-8",
    "hailo10h":              "hailo-10h",
}


# For Jetson, the thermal_profile string passed to UnifiedAnalyzer.
# Other SKUs default to None (use the resource model's default).
_SKU_TO_THERMAL_PROFILE = {
    "jetson_orin_agx_64gb":  "30W",
}


def sku_id_to_calibration_dir(sku_id: str) -> Optional[str]:
    """Return the calibration_data subdirectory for a SKU, or None
    when no measurement data is registered."""
    return _SKU_TO_CALIBRATION_DIR.get(sku_id)


def sku_id_to_mapper_name(sku_id: str) -> Optional[str]:
    """Return the UnifiedAnalyzer mapper name for a SKU, or None when
    the SKU is not in the mapper registry (used to detect SKUs we can
    populate in M1-M7 but not yet drive end-to-end through
    UnifiedAnalyzer)."""
    return _SKU_TO_MAPPER_NAME.get(sku_id)


def sku_id_to_thermal_profile(sku_id: str) -> Optional[str]:
    """Return the thermal_profile string to pass to
    UnifiedAnalyzer.analyze_model, or None to use the SKU default."""
    return _SKU_TO_THERMAL_PROFILE.get(sku_id)


__all__ = [
    "sku_id_to_calibration_dir",
    "sku_id_to_mapper_name",
    "sku_id_to_thermal_profile",
]
