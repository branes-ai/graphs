"""Which KPU ComputeProduct each generated core template comes from.

``tools/generate_kpu_ip.py`` writes the templates in ``soc_designs/ip/``
from these SKUs; analyses that need the SKU behind a design's KPU block
(the domain-flow ceilings, for one) resolve it here rather than parsing a
template's source line.
"""

from __future__ import annotations

from typing import Dict, Tuple

#: template id -> (ComputeProduct id, core_only). A *core* is the KPU as an
#: on-die block of an SoC: it drops the chip's own memory PHYs and IO pads,
#: because the SoC's memory system and pad ring are the design's.
KPU_CORES: Dict[str, Tuple[str, bool]] = {
    "kpu_h64_core": ("kpu_h64_auto1_lp5x4_7nm_tsmc_hpc", True),
    "kpu_t64_core": ("kpu_t64_32x32_lp5x4_7nm_tsmc_hpc", True),
    "kpu_t128_core": ("kpu_t128_32x32_lp5x8_7nm_tsmc_hpc", True),
    "kpu_t256_core": ("kpu_t256_32x32_lp5x16_7nm_tsmc_hpc", True),
}


def sku_of(template_id: str) -> str:
    """The ComputeProduct a generated core template was built from."""
    if template_id not in KPU_CORES:
        raise KeyError(f"{template_id!r} is not a generated KPU core; "
                       f"have {', '.join(sorted(KPU_CORES))}")
    return KPU_CORES[template_id][0]


__all__ = ["KPU_CORES", "sku_of"]
