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
    # The 7 nm cores, which the Phase 6 ladder uses.
    "kpu_h64_core": ("kpu_h64_auto1_lp5x4_7nm_tsmc_hpc", True),
    "kpu_t64_core": ("kpu_t64_32x32_lp5x4_7nm_tsmc_hpc", True),
    "kpu_t128_core": ("kpu_t128_32x32_lp5x8_7nm_tsmc_hpc", True),
    "kpu_t256_core": ("kpu_t256_32x32_lp5x16_7nm_tsmc_hpc", True),
    "kpu_t512_core": ("kpu_t512_32x32_lp5x32_7nm_tsmc_hpc", True),
    # The same fabrics at the other nodes the catalog states a SKU for, so
    # a design at that node carries a clock sourced there rather than a
    # provisional one (graphs#269 Phase 7).
    "kpu_h64_core_n16": ("kpu_h64_auto1_lp5x4_16nm_tsmc_ffp", True),
    "kpu_t64_core_n16": ("kpu_t64_32x32_lp5x4_16nm_tsmc_ffp", True),
    "kpu_t128_core_n16": ("kpu_t128_32x32_lp5x8_16nm_tsmc_ffp", True),
    "kpu_t256_core_n16": ("kpu_t256_32x32_lp5x16_16nm_tsmc_ffp", True),
    "kpu_t64_core_12fdx": ("kpu_t64_32x32_lp5x4_12nm_gf_fdx", True),
    "kpu_t128_core_12fdx": ("kpu_t128_32x32_lp5x8_12nm_gf_fdx", True),
    "kpu_t256_core_12fdx": ("kpu_t256_32x32_lp5x16_12nm_gf_fdx", True),
    "kpu_t512_core_12fdx": ("kpu_t512_32x32_lp5x32_12nm_gf_fdx", True),
}

#: The process node each core's SKU is stated at, for the designs that
#: place it: a core composed at another node carries a provisional clock.
CORE_NODES: Dict[str, str] = {
    "kpu_h64_core": "tsmc_n7", "kpu_t64_core": "tsmc_n7", "kpu_t128_core": "tsmc_n7",
    "kpu_t256_core": "tsmc_n7", "kpu_t512_core": "tsmc_n7",
    "kpu_h64_core_n16": "tsmc_n16", "kpu_t64_core_n16": "tsmc_n16",
    "kpu_t128_core_n16": "tsmc_n16", "kpu_t256_core_n16": "tsmc_n16",
    "kpu_t64_core_12fdx": "gf_12fdx", "kpu_t128_core_12fdx": "gf_12fdx",
    "kpu_t256_core_12fdx": "gf_12fdx", "kpu_t512_core_12fdx": "gf_12fdx",
}


def sku_of(template_id: str) -> str:
    """The ComputeProduct a generated core template was built from."""
    if template_id not in KPU_CORES:
        raise KeyError(f"{template_id!r} is not a generated KPU core; "
                       f"have {', '.join(sorted(KPU_CORES))}")
    return KPU_CORES[template_id][0]


__all__ = ["CORE_NODES", "KPU_CORES", "sku_of"]
