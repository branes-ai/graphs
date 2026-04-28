"""
Layer-panel builders for the micro-architectural validation report.

Each module in this package builds a populated ``LayerPanel`` for one
of the nine validation layers, given a ``HardwareResourceModel``. The
``cli/microarch_validation_report.py`` wires the panel-builders into
each per-SKU report.

M1 ships ``layer1_alu``. M2-M7 land the remaining layers.
"""
from .layer1_alu import (
    build_layer1_panel,
    build_layer1_panels_for_skus,
    resolve_sku_resource_model,
    cross_sku_layer1_chart,
)
from .layer2_register import (
    build_layer2_register_panel,
    build_layer2_panels_for_skus,
    cross_sku_layer2_chart,
    CrossSKULayer2Chart,
)
from .layer3_l1_cache import (
    build_layer3_l1_cache_panel,
    build_layer3_panels_for_skus,
    cross_sku_layer3_chart,
    CrossSKULayer3Chart,
)
from .layer4_l2_cache import (
    build_layer4_l2_cache_panel,
    build_layer4_panels_for_skus,
    cross_sku_layer4_chart,
    CrossSKULayer4Chart,
)

__all__ = [
    "build_layer1_panel",
    "build_layer1_panels_for_skus",
    "resolve_sku_resource_model",
    "cross_sku_layer1_chart",
    "build_layer2_register_panel",
    "build_layer2_panels_for_skus",
    "cross_sku_layer2_chart",
    "CrossSKULayer2Chart",
    "build_layer3_l1_cache_panel",
    "build_layer3_panels_for_skus",
    "cross_sku_layer3_chart",
    "CrossSKULayer3Chart",
    "build_layer4_l2_cache_panel",
    "build_layer4_panels_for_skus",
    "cross_sku_layer4_chart",
    "CrossSKULayer4Chart",
]
