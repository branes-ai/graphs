"""
Silicon speed-of-light (SoL) analysis with design-space trade-offs.

This module answers two linked questions:

1. What is the theoretical upper bound on INT8 compute density /
   efficiency at a given process node and die size?

2. Where do shipping products sit inside that upper bound, and
   which design-space axes explain the gap?

The ALU trade-off space has three largely-orthogonal axes:

  Axis 1 - ALU width W (MACs per instance per clock)
    W=1  : bare FMA (KPU PE, CUDA core, TPU MXU cell)
    W=4  : small dot-product (Volta/Turing TC lane)
    W=16 : medium dot-product (Ampere TC lane)
    W=32+: wide dot-product (Hopper TC lane)

  Axis 2 - accumulator precision and rounding (AccumMode)
    LOSSLESS            : INT8 x INT8 -> INT32 with no bits dropped
    TREE_ROUNDED        : FP16 x FP16 -> FP16 with rounding at each
                          reduction-tree level (log2(W) roundings)
    MIXED_PRECISION     : FP16 operand with FP32 accumulator
                          (Ampere HMMA default)
    AGGRESSIVE_TRUNCATE : accumulator fits into operand width;
                          used in edge accelerators for density

  Axis 3 - operand-reuse topology (ReuseTopology)
    ISOLATED            : each MAC reads its own operands from RF
                          (2 bytes/MAC at INT8)
    INTRA_ALU_BROADCAST : within one W-wide ALU, multiplications share
                          two operand buses (2 bytes/ALU-clock = 2/W
                          bytes/MAC in steady state on a matched
                          matmul)
    MESH_STREAMING      : 2D mesh forwards operands between PEs;
                          edge RF reads only. NxN mesh = 2/N bytes/MAC.
    SYSTOLIC_STATIONARY : weight-stationary systolic, activations
                          stream; similar 2/N bytes/MAC scaling.

The design-space produces three distinct ceilings that a shipping
product can bump into:

  Density ceiling  (MACs/mm^2)   : favors wider W up to ~W=8-16
                                   where reduction-tree bit-width
                                   growth outpaces multiplier sharing.
  Energy ceiling   (TOPS/W)      : 2 / pJ_per_MAC; sensitive to
                                   reuse topology more than W.
  Accuracy ceiling (bits kept)   : favors low W with lossless
                                   accumulation.

Use `build_default_sol_report()` to generate the analysis with the
default catalog (KPU PE, CUDA FP32, Volta/Turing TC, Ampere TC,
Hopper TC, TPU MXU) + an analytical curve at W=1,2,4,8,16,32,64.
"""
from __future__ import annotations

import html
import json
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Tuple


# ======================================================================
# Design-space enums
# ======================================================================


class AccumMode(Enum):
    """Qualitative label for accumulator precision and rounding."""
    LOSSLESS = "lossless (INT32 accum from INT8 mult)"
    TREE_ROUNDED = "tree-rounded (loses bits at each tree level)"
    MIXED_PRECISION = "mixed (FP16 operand, FP32 accum)"
    AGGRESSIVE_TRUNCATE = "aggressive (accum fits operand width)"


class ReuseTopology(Enum):
    """Qualitative label for operand-reuse pattern."""
    ISOLATED = "isolated (2 B/MAC, no reuse)"
    INTRA_ALU_BROADCAST = "intra-ALU broadcast (2/W B/MAC)"
    MESH_STREAMING = "2D mesh streaming (2/N B/MAC)"
    SYSTOLIC_STATIONARY = "systolic weight-stationary (2/N B/MAC)"


# ======================================================================
# Tunable constants
# ======================================================================


DEFAULT_TDP_TARGETS_W: Tuple[float, ...] = (
    5.0, 15.0, 30.0, 60.0, 150.0, 300.0,
)

DEFAULT_DIE_AREA_MM2: float = 250.0
DEFAULT_SILICON_CLOCK_GHZ: float = 1.5

# Canonical transistor density per process node (million transistors
# per mm^2) for dense compute layout. These set the INVARIANT:
#
#   die_area / ALU_area  ==  die_transistors / ALU_transistors
#
# which requires every ALU at a given process node to share the same
# transistor-to-area ratio (i.e., the density here).
#
# Real layouts actually vary by +/- 2x across logic (wire-limited),
# dense MAC arrays (routing-friendly), and SRAM (cell-limited). The
# SoL analysis deliberately collapses this to a single dense-compute
# density per node so the "fill a die with one ALU type" idealization
# gives consistent die-transistor counts regardless of which ALU type
# is chosen. The density values below are anchored to public die-shot
# data for dense compute regions on each node (NVIDIA / TPU / AMD
# whitepaper area breakdowns + published full-chip transistor counts).
PROCESS_DENSITY_MT_PER_MM2: Dict[int, float] = {
    16: 25.0,    # TSMC 16FF+ / Samsung 14LPP (A9, Volta GV100 scaled)
    12: 40.0,    # TSMC 12FFN (Turing)
    10: 55.0,    # Intel 10nm / TSMC N10
    8: 80.0,     # Samsung 8LPP dense compute (Ampere GA10x TC region)
    7: 100.0,    # TSMC N7 (A100 scaled; TPU v4)
    5: 140.0,    # TSMC N5 (M1 Max / H100 pre-variant)
    4: 180.0,    # TSMC N4 dense compute (H100 TC region)
    3: 220.0,    # TSMC N3 (forward-looking)
}

# Legacy aliases retained for callers that still reference them.
DENSITY_MT_PER_MM2_LOGIC: float = 45.0
DENSITY_MT_PER_MM2_DENSE_MAC: float = PROCESS_DENSITY_MT_PER_MM2[8]


def process_density_mt_per_mm2(process_nm: int) -> float:
    """Canonical dense-compute transistor density (MT/mm^2) for
    `process_nm`. If the node is not in the table, snap to the
    nearest known node."""
    if process_nm in PROCESS_DENSITY_MT_PER_MM2:
        return PROCESS_DENSITY_MT_PER_MM2[process_nm]
    nearest = min(
        PROCESS_DENSITY_MT_PER_MM2.keys(),
        key=lambda n: abs(n - process_nm),
    )
    return PROCESS_DENSITY_MT_PER_MM2[nearest]


# ======================================================================
# DotProductALU: the fundamental unit of replication
# ======================================================================


@dataclass
class DotProductALU:
    """
    One ALU instance that produces `W` MACs per clock.

    All physical numbers are PER INSTANCE - the area, transistor count,
    and energy are for the whole ALU as a single unit of replication
    on the die. Per-MAC numbers are derived properties.

    For W = 1 this degenerates to a bare FMA (as in BareALU below).

    Field order keeps backwards compatibility with the original
    BareALU: required fields come first, design-space axes are
    defaulted. A W=1 / ISOLATED / LOSSLESS ALU with a user-supplied
    area/transistors/pJ is identical to the old BareALU contract.

    Operand-bandwidth has TWO levels:

    * `bytes_per_mac_alu`: bandwidth THE ALU ITSELF demands from its
      operand feeders (RF, operand collector, edge injector, etc.).
      For ANY dot-product ALU this is 2 * bytes_per_operand - each
      MAC needs its own A operand and B operand, regardless of W.
      This number does NOT go down with ALU width.

    * `bytes_per_mac_die`: bandwidth the DIE AS A WHOLE demands from
      main memory (or the next cache level) in steady state. This
      IS sensitive to topology: mesh / systolic / broadcast-across-
      ALUs all recover operand reuse at levels above the single ALU.
      For an NxN mesh the floor is 2/N B/MAC. For a Wm x Wn matmul
      tile it is 1/Wm + 1/Wn B/MAC. For a fully-isolated FMA it
      stays at 2 * bytes_per_operand (no reuse anywhere).
    """
    name: str
    precision: str                 # "INT8", "FP16", "BF16", "FP32", "FP8"
    process_nm: int
    # Per-instance silicon numbers (hand-tuned for real products,
    # analytically derived for parametric curve points).
    area_mm2: float                # area of one ALU instance
    transistor_count_k: float      # K transistors per instance
    pj_per_clock: float            # energy per instance per clock
    # Design-space axes: default to the simplest archetype (bare FMA,
    # isolated, lossless) so legacy BareALU(**kwargs) still works.
    W: int = 1                     # MACs per ALU per clock (== native width)
    accum_mode: AccumMode = AccumMode.LOSSLESS
    reuse: ReuseTopology = ReuseTopology.ISOLATED
    # Per-MAC operand bandwidth. See class docstring for the ALU vs
    # die-level distinction.
    bytes_per_mac_alu: float = 2.0   # 2 * bytes_per_op for INT8 dot products
    bytes_per_mac_die: float = 2.0   # topology-level; same as ALU if isolated
    ops_per_mac: int = 2           # 1 FMA = 2 ops (multiply + add)
    is_parametric: bool = False    # True for analytically-derived curve points
    citation: str = ""

    @property
    def bytes_per_mac(self) -> float:
        """Deprecated alias for bytes_per_mac_die - the number most
        consumers actually care about (memory-bandwidth demand).
        Prefer the explicit names in new code."""
        return self.bytes_per_mac_die

    def __post_init__(self) -> None:
        """Enforce the SoL density invariant: every ALU at a given
        process node shares one transistor-to-area ratio, so that

            die_area / ALU_area == die_transistors / ALU_transistors

        holds across archetypes. If `area_mm2` is 0 or not provided,
        derive it from transistors and process density. If it IS
        provided and differs materially from the canonical value,
        recompute it (with a warning via the `is_parametric` flag
        semantics) so the invariant holds. Callers who need a
        layout-specific density should change PROCESS_DENSITY_MT_
        PER_MM2[process_nm] instead.
        """
        density = process_density_mt_per_mm2(self.process_nm)
        canonical = self.transistor_count_k / 1000.0 / density
        # If caller passed a non-zero area that differs from canonical
        # by more than 5%, silently replace it so the invariant holds.
        # Hand-tuned archetypes typically omit area_mm2 (=0.0) and get
        # the canonical value. Explicit area is preserved only when it
        # already matches canonical to within rounding.
        if self.area_mm2 <= 0.0:
            self.area_mm2 = canonical
        else:
            # If the provided area disagrees with canonical by > 5 %,
            # replace it to preserve the density invariant. Otherwise
            # keep the caller's value (rounding differences).
            delta = abs(self.area_mm2 - canonical) / canonical
            if delta > 0.05:
                self.area_mm2 = canonical

    # ------------------------------------------------------------------
    # Per-MAC derived properties
    # ------------------------------------------------------------------

    @property
    def area_per_mac_mm2(self) -> float:
        return self.area_mm2 / self.W if self.W > 0 else 0.0

    @property
    def transistor_count_per_mac_k(self) -> float:
        return self.transistor_count_k / self.W if self.W > 0 else 0.0

    @property
    def pj_per_mac(self) -> float:
        return self.pj_per_clock / self.W if self.W > 0 else 0.0

    @property
    def tops_per_watt_ceiling(self) -> float:
        """Clock- and count-independent silicon ceiling. TOPS/W =
        ops_per_MAC / pJ_per_MAC."""
        return self.ops_per_mac / self.pj_per_mac if self.pj_per_mac > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "precision": self.precision,
            "process_nm": self.process_nm,
            "W": self.W,
            "accum_mode": self.accum_mode.value,
            "reuse": self.reuse.value,
            "area_mm2": self.area_mm2,
            "area_per_mac_mm2": self.area_per_mac_mm2,
            "transistor_count_k": self.transistor_count_k,
            "transistor_count_per_mac_k": self.transistor_count_per_mac_k,
            "pj_per_clock": self.pj_per_clock,
            "pj_per_mac": self.pj_per_mac,
            "bytes_per_mac_alu": self.bytes_per_mac_alu,
            "bytes_per_mac_die": self.bytes_per_mac_die,
            "ops_per_mac": self.ops_per_mac,
            "tops_per_watt_ceiling": self.tops_per_watt_ceiling,
            "is_parametric": self.is_parametric,
            "citation": self.citation,
        }


# Backwards-compatible alias for the old BareALU name. Prior code
# expected per-MAC semantics; since a BareALU is a W=1 DotProductALU,
# per-instance and per-MAC coincide and the old contract still holds.
BareALU = DotProductALU


# ======================================================================
# Parametric cost model
# ======================================================================

# Multiplier transistor cost in K transistors, keyed by precision.
# These are rough estimates from standard-cell synthesis data at 8 nm:
#   INT8  mult: ~4 K   (8x8 Wallace tree + Booth encoding)
#   INT4  mult: ~1.5 K
#   FP16  mult: ~10 K  (mantissa mult + sign/exponent logic)
#   BF16  mult: ~8 K   (8-bit mantissa is cheaper than FP16's 10-bit)
#   FP32  mult: ~35 K  (23-bit mantissa multiply dominates)
#   FP8   mult: ~3.5 K (4-bit mantissa)
_MULT_K_BY_PRECISION: Dict[str, float] = {
    "INT4": 1.5,
    "INT8": 4.0,
    "FP8": 3.5,
    "BF16": 8.0,
    "FP16": 10.0,
    "FP32": 35.0,
    "FP64": 140.0,
}

# Base INT32 adder cost in K transistors (carry-lookahead at 8 nm).
_ADDER_K_BASE: float = 2.0

# Multiplier energy (pJ/clock) keyed by precision. Again rough
# first-principles at 8 nm.
_MULT_PJ_BY_PRECISION: Dict[str, float] = {
    "INT4": 0.020,
    "INT8": 0.045,   # matches microarch_accounting KPU FMA (0.050 incl. carry tree)
    "FP8": 0.055,
    "BF16": 0.110,
    "FP16": 0.140,
    "FP32": 0.600,
    "FP64": 2.500,
}


def _process_scaling(process_nm: int) -> float:
    """Scale factor relative to 8 nm for density and energy. A rough
    power-of-two model: halving the node doubles density and halves
    dynamic energy per op."""
    return 8.0 / process_nm


def parametric_dot_product_alu(
    W: int,
    precision: str = "INT8",
    process_nm: int = 8,
    reuse: ReuseTopology = ReuseTopology.INTRA_ALU_BROADCAST,
    accum_mode: AccumMode = AccumMode.LOSSLESS,
) -> DotProductALU:
    """
    Synthesize a DotProductALU from first-principles at width `W`.

    Cost model:

      C_mult(precision)     : multiplier transistor count (table lookup)
      C_tree(W)             : (W - 1) adders with bit-widths growing
                              by log2(W); modelled with per-level
                              fan-in factor
      C_accum               : 1 K (INT32 accumulator register)
      C_pipe(W)             : 0.2 * W K (pipeline flops per stage)
      C_route(W)            : 0.3 * W K (operand-distribution wiring)

    Energy model:

      E = W * E_mult(precision) + E_tree(W) + E_overhead(W)

    where E_tree ~ tree transistor count * 0.0015 pJ/transistor/switch.

    Bytes/MAC depends on reuse topology:

      ISOLATED           : 2 * bytes_per_operand
      INTRA_ALU_BROADCAST: 2/W * bytes_per_operand (in its target matmul)
      MESH_STREAMING     : 2/sqrt(W) for an NxN mesh
      SYSTOLIC_STATIONARY: 1/sqrt(W) (weights don't refresh each clock)

    Process-node scaling is a rough 8 nm / process_nm multiplier on
    density; energy scales by the same factor.
    """
    if W < 1:
        raise ValueError("W must be >= 1")
    if precision not in _MULT_K_BY_PRECISION:
        raise ValueError(f"Unknown precision {precision}")

    scale = _process_scaling(process_nm)
    c_mult_k = _MULT_K_BY_PRECISION[precision]
    e_mult_pj = _MULT_PJ_BY_PRECISION[precision] / scale

    # Reduction-tree cost: for W MACs fed into a single sum, we need
    # W-1 adders arranged in log2(W) levels. The bit widths grow by
    # about 1 bit per level as partial sums widen, so use a modest
    # per-level inflation factor.
    tree_k = 0.0
    if W > 1:
        levels = max(1, int(math.log2(W)))
        nodes_at_level = W // 2
        for lv in range(levels):
            width_factor = 1.0 + 0.10 * lv  # ~10% growth per level
            tree_k += nodes_at_level * _ADDER_K_BASE * width_factor
            nodes_at_level = max(1, nodes_at_level // 2)

    c_accum_k = 1.0
    c_pipe_k = 0.2 * W
    c_route_k = 0.3 * W

    total_k = W * c_mult_k + tree_k + c_accum_k + c_pipe_k + c_route_k

    # Area: use the canonical dense-compute density per process node
    # (PROCESS_DENSITY_MT_PER_MM2) so all ALUs at the same node share
    # the SoL density invariant.
    density_mt_per_mm2 = process_density_mt_per_mm2(process_nm)
    area_mm2 = (total_k / 1000.0) / density_mt_per_mm2

    # Energy: multipliers dominate; tree contributes ~0.0015 pJ per
    # K-transistor per switch (rough standard-cell number).
    e_tree_pj = tree_k * 0.0015
    e_overhead_pj = (c_accum_k + c_pipe_k + c_route_k) * 0.0010
    pj_per_clock = W * e_mult_pj + e_tree_pj + e_overhead_pj

    # Operand bandwidth.
    #
    # AT THE ALU LEVEL: a dot-product ALU of width W reads W A-
    # operands + W B-operands per clock and produces W MACs. So at
    # the ALU level, bytes_per_mac = 2 * bytes_per_op ALWAYS, for
    # any W. There is NO reuse inside a single dot-product ALU.
    #
    # AT THE DIE LEVEL: topology-level reuse (mesh, systolic,
    # broadcast-across-ALUs) drives memory-side bandwidth much
    # lower. We only know that for a SPECIFIC topology + workload;
    # the parametric curve does not know how many instances tile
    # together, so we default die-level to ALU-level (no reuse
    # assumed). Real-product archetypes override bytes_per_mac_die.
    bytes_per_op = _precision_byte_width(precision)
    bytes_per_mac_alu = 2.0 * bytes_per_op
    bytes_per_mac_die = bytes_per_mac_alu  # no topology-level reuse assumed

    return DotProductALU(
        name=f"Parametric {precision} W={W}",
        precision=precision,
        process_nm=process_nm,
        W=W,
        accum_mode=accum_mode,
        reuse=reuse,
        area_mm2=area_mm2,
        transistor_count_k=total_k,
        pj_per_clock=pj_per_clock,
        bytes_per_mac_alu=bytes_per_mac_alu,
        bytes_per_mac_die=bytes_per_mac_die,
        is_parametric=True,
        citation=(
            f"Analytical: {W} * {c_mult_k} K mult + "
            f"{tree_k:.1f} K tree + {c_accum_k:.1f} K accum + "
            f"{c_pipe_k:.1f} K pipe + {c_route_k:.1f} K route. "
            f"Density {density_mt_per_mm2:.0f} MT/mm^2 at {process_nm} nm. "
            f"ALU-level bandwidth is 2*{bytes_per_op:.1f} = "
            f"{bytes_per_mac_alu:.1f} B/MAC (constant across W); die-"
            f"level bandwidth equals ALU-level for the parametric "
            f"curve since no topology is assumed."
        ),
    )


def _precision_byte_width(precision: str) -> float:
    """Bytes per operand. INT4 = 0.5 B; INT8/FP8 = 1 B; BF16/FP16 = 2 B;
    FP32 = 4 B; FP64 = 8 B."""
    table = {
        "INT4": 0.5,
        "INT8": 1.0, "FP8": 1.0,
        "BF16": 2.0, "FP16": 2.0,
        "FP32": 4.0, "FP64": 8.0,
    }
    return table.get(precision, 1.0)


def generate_parametric_curve(
    widths: Tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64),
    precision: str = "INT8",
    process_nm: int = 8,
    reuse: ReuseTopology = ReuseTopology.INTRA_ALU_BROADCAST,
) -> List[DotProductALU]:
    """Generate a parametric DotProductALU at each W for curve plots."""
    return [
        parametric_dot_product_alu(W=w, precision=precision,
                                   process_nm=process_nm, reuse=reuse)
        for w in widths
    ]


# ======================================================================
# Hand-tuned real-product archetypes
# ======================================================================
#
# Each entry below is a real-product ALU whose numbers come from this
# repo's microarch_accounting.py / building_block_energy.py (for KPU
# and Ampere) or from published die-shot analyses / whitepapers (for
# other architectures). The "is_parametric=False" flag marks them as
# cited data points rather than curve-derived estimates.
# ======================================================================


def default_alu_catalog() -> List[DotProductALU]:
    """Hand-tuned ALU archetypes - purely the multiply-add datapath,
    not composite PE/Tile blocks. Area is DERIVED from transistor
    count + process density (see PROCESS_DENSITY_MT_PER_MM2), so
    every ALU at a given process node has the same density and the
    SoL invariant `die_area * density == die_transistors` holds
    across the catalog.

    Composite building blocks (PE, Tile, Cluster, SoC) live in the
    silicon_composition module - that is the SoC-floorplanning view.
    Do not add PE-or-above entries here.
    """
    return [
        # --------------- KPU INT8 FMA (W=1, mesh) -----------------
        DotProductALU(
            name="KPU INT8 FMA",
            precision="INT8",
            process_nm=8,
            W=1,
            accum_mode=AccumMode.LOSSLESS,
            reuse=ReuseTopology.MESH_STREAMING,
            area_mm2=0.0,           # derived from 7 K / 80 MT/mm^2
            transistor_count_k=7.0,
            pj_per_clock=0.050,
            bytes_per_mac_alu=2.0,
            bytes_per_mac_die=0.0625,
            citation=(
                "Bare INT8 multiplier (~4 K trans) + INT32 adder "
                "(~2 K) + pipeline reg (~1 K). Matches "
                "microarch_accounting.build_kpu_tile_accounting() "
                "'FMA unit' row at 0.050 pJ/clock. Die-level "
                "bandwidth reflects the 32x32 mesh that surrounds "
                "this ALU in a KPU tile."
            ),
        ),

        # --------------- CUDA cores (W=1, FP32, isolated) -----------------
        DotProductALU(
            name="Ampere CUDA-core FMA (FP32)",
            precision="FP32",
            process_nm=8,
            W=1,
            accum_mode=AccumMode.LOSSLESS,
            reuse=ReuseTopology.ISOLATED,
            area_mm2=0.0,           # derived from 78 K / 80 MT/mm^2
            transistor_count_k=78.0,
            pj_per_clock=2.5,
            bytes_per_mac_alu=8.0,
            bytes_per_mac_die=8.0,
            citation=(
                "10 M transistors across 128 FP32 FMA lanes per SM. "
                "Per-lane ~78 K trans; area derived from 8 nm dense-"
                "compute density (80 MT/mm^2). Energy 2.5 pJ/clock "
                "matches building_block_energy at full activity. "
                "CUDA path has no systolic / mesh reuse."
            ),
        ),

        # --------------- Volta / Turing TC lane (W=4, FP16, tree-rounded) -----
        DotProductALU(
            name="Volta/Turing TC lane (W=4 FP16)",
            precision="FP16",
            process_nm=12,
            W=4,
            accum_mode=AccumMode.MIXED_PRECISION,
            reuse=ReuseTopology.INTRA_ALU_BROADCAST,
            area_mm2=0.0,           # derived from 35 K / 40 MT/mm^2
            transistor_count_k=35.0,
            pj_per_clock=0.55,
            bytes_per_mac_alu=4.0,
            bytes_per_mac_die=0.5,
            citation=(
                "Volta V100 / Turing T4 first-generation Tensor Core. "
                "Per lane: 4 FP16 multiplies into a 3-adder reduction "
                "tree with FP32 accumulator. Hand-tuned from NVIDIA "
                "Volta whitepaper area breakdowns + public die-shot "
                "analyses. Die-level bandwidth from 8x8x4 HMMA: "
                "fragment reuse drives 4/8 = 0.5 B/MAC."
            ),
        ),

        # --------------- Ampere TC lane (W=16, INT8, tree-rounded) -----
        DotProductALU(
            name="Ampere TC lane (W=16 INT8)",
            precision="INT8",
            process_nm=8,
            W=16,
            accum_mode=AccumMode.MIXED_PRECISION,
            reuse=ReuseTopology.INTRA_ALU_BROADCAST,
            area_mm2=0.0,           # derived from 110 K / 80 MT/mm^2
            transistor_count_k=110.0,
            pj_per_clock=1.28,
            bytes_per_mac_alu=2.0,
            bytes_per_mac_die=0.125,
            citation=(
                "Ampere GA10x 3rd-gen Tensor Core lane. Each SM has "
                "4 TCs aggregating ~40 M MAC-array transistors for "
                "4096 INT8 MACs/clock; per-lane W=16, ~110 K trans, "
                "0.08 pJ/MAC. Die-level bandwidth from 16x16x16 "
                "HMMA: 512 B frags / 4096 MACs = 0.125 B/MAC."
            ),
        ),

        # --------------- Hopper TC lane (W=32, FP8, tree-rounded) -----
        DotProductALU(
            name="Hopper TC lane (W=32 FP8)",
            precision="FP8",
            process_nm=4,
            W=32,
            accum_mode=AccumMode.MIXED_PRECISION,
            reuse=ReuseTopology.INTRA_ALU_BROADCAST,
            area_mm2=0.0,           # derived from 190 K / 180 MT/mm^2
            transistor_count_k=190.0,
            pj_per_clock=1.95,
            bytes_per_mac_alu=2.0,
            bytes_per_mac_die=0.0625,
            citation=(
                "Hopper H100 4th-gen Tensor Core with FP8 Transformer "
                "Engine. W=32, FP8 operand with FP32 accumulator. "
                "Hand-tuned from NVIDIA H100 whitepaper TC specs "
                "(4x throughput vs Ampere for new precisions). Area "
                "derived from 4 nm dense-compute density (180 MT/mm^2)."
            ),
        ),

        # --------------- TPU MXU cell (W=1, BF16 systolic) -----
        DotProductALU(
            name="TPU v4 MXU cell (BF16 systolic)",
            precision="BF16",
            process_nm=7,
            W=1,
            accum_mode=AccumMode.LOSSLESS,
            reuse=ReuseTopology.SYSTOLIC_STATIONARY,
            area_mm2=0.0,           # derived from 18 K / 100 MT/mm^2
            transistor_count_k=18.0,
            pj_per_clock=0.18,
            bytes_per_mac_alu=4.0,
            bytes_per_mac_die=0.0156,
            citation=(
                "TPU v4 MXU: 128x128 BF16 systolic array. Per-cell: "
                "BF16 multiplier + FP32 accumulator + stationary-"
                "weight latch. Hand-tuned from Jouppi et al. 2023. "
                "Die-level bandwidth: weights stationary (1-time "
                "load amortized), activations stream -> 2/128 B/MAC."
            ),
        ),
    ]


# ======================================================================
# SoL analysis over a die
# ======================================================================


@dataclass
class SoLAnalysis:
    """One ALU archetype tiled across a die, swept across clocks and TDPs."""
    alu: DotProductALU
    die_area_mm2: float = DEFAULT_DIE_AREA_MM2
    silicon_clock_ghz: float = DEFAULT_SILICON_CLOCK_GHZ
    tdp_targets_w: Tuple[float, ...] = DEFAULT_TDP_TARGETS_W

    @property
    def num_alus(self) -> int:
        return int(self.die_area_mm2 / self.alu.area_mm2)

    @property
    def num_macs_on_die(self) -> int:
        """Total MACs per clock across all ALU instances."""
        return self.num_alus * self.alu.W

    @property
    def die_transistor_count_m(self) -> float:
        return self.num_alus * self.alu.transistor_count_k / 1000.0

    @property
    def die_energy_pj_per_clock(self) -> float:
        return self.num_alus * self.alu.pj_per_clock

    def peak_tops(self, clock_ghz: float) -> float:
        """Peak TOPS at a given clock - every ALU firing its W MACs."""
        macs_per_sec = self.num_macs_on_die * clock_ghz * 1e9
        return macs_per_sec * self.alu.ops_per_mac / 1e12

    def die_power_w(self, clock_ghz: float) -> float:
        return self.die_energy_pj_per_clock * clock_ghz * 1e-3

    def clock_for_tdp(self, tdp_w: float) -> float:
        denom = self.die_energy_pj_per_clock * 1e-3
        return tdp_w / denom if denom > 0 else 0.0

    def tdp_sweep(self) -> List[Dict[str, float]]:
        out: List[Dict[str, float]] = []
        for tdp in self.tdp_targets_w:
            f = self.clock_for_tdp(tdp)
            silicon_limited = f > self.silicon_clock_ghz
            effective_f = min(f, self.silicon_clock_ghz)
            out.append({
                "tdp_w": tdp,
                "clock_for_tdp_ghz": f,
                "effective_clock_ghz": effective_f,
                "silicon_limited": silicon_limited,
                "peak_tops": self.peak_tops(effective_f),
                "effective_power_w": self.die_power_w(effective_f),
            })
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alu": self.alu.to_dict(),
            "die_area_mm2": self.die_area_mm2,
            "num_alus": self.num_alus,
            "num_macs_on_die": self.num_macs_on_die,
            "die_transistor_count_m": self.die_transistor_count_m,
            "die_energy_pj_per_clock": self.die_energy_pj_per_clock,
            "silicon_clock_ghz": self.silicon_clock_ghz,
            "peak_tops_at_silicon_clock": self.peak_tops(
                self.silicon_clock_ghz
            ),
            "die_power_at_silicon_clock_w": self.die_power_w(
                self.silicon_clock_ghz
            ),
            "tops_per_watt_ceiling": self.alu.tops_per_watt_ceiling,
            "tdp_sweep": self.tdp_sweep(),
        }


@dataclass
class ProductReference:
    name: str
    process_nm: int
    die_area_mm2: float
    peak_int8_tops: float
    tdp_w: float

    @property
    def tops_per_watt(self) -> float:
        return self.peak_int8_tops / self.tdp_w if self.tdp_w > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "process_nm": self.process_nm,
            "die_area_mm2": self.die_area_mm2,
            "peak_int8_tops": self.peak_int8_tops,
            "tdp_w": self.tdp_w,
            "tops_per_watt": self.tops_per_watt,
        }


def default_product_references() -> List[ProductReference]:
    """Shipping and target products for gap-to-SoL comparison in the
    embedded-AI space (8 nm, 5-60 W TDP class).

    Scope is deliberately embedded: all references are on the same
    8 nm process and target the same deployment envelope so the
    Actual / SoL ratios are directly comparable. Dense INT8 unless
    labelled sparse. NVIDIA's 275 TOPS Orin AGX headline is sparse
    INT8 at MAXN across GPU + DLA + PVA; dense numbers below come
    from NVIDIA published specs scaled by the sustained clock each
    TDP mode permits.

    Datacenter parts (H100 etc.) deliberately are NOT included:
    different market, different process, different TDP class, and
    different customer-competition dynamics. Mixing them into this
    table makes the edge-AI narrative less clean and introduces
    apples-vs-oranges process comparisons.
    """
    return [
        ProductReference(
            name="Jetson AGX Orin MAXN (60 W, dense INT8)",
            process_nm=8, die_area_mm2=180.0,
            peak_int8_tops=137.0, tdp_w=60.0,
        ),
        ProductReference(
            name="Jetson AGX Orin (50 W, dense INT8)",
            process_nm=8, die_area_mm2=180.0,
            peak_int8_tops=105.0, tdp_w=50.0,
        ),
        ProductReference(
            name="Jetson AGX Orin (30 W, dense INT8)",
            process_nm=8, die_area_mm2=180.0,
            peak_int8_tops=68.0, tdp_w=30.0,
        ),
        ProductReference(
            name="Jetson AGX Orin (15 W, dense INT8)",
            process_nm=8, die_area_mm2=180.0,
            peak_int8_tops=34.0, tdp_w=15.0,
        ),
        ProductReference(
            name="Jetson AGX Orin MAXN (60 W, sparse INT8 marketing)",
            process_nm=8, die_area_mm2=180.0,
            peak_int8_tops=275.0, tdp_w=60.0,
        ),
        ProductReference(
            name="KPU T128 (hypothetical, 12 W)",
            process_nm=8, die_area_mm2=80.0,
            peak_int8_tops=262.0, tdp_w=12.0,
        ),
    ]


@dataclass
class SoLReport:
    alus: List[DotProductALU]
    analyses: List[SoLAnalysis]
    parametric_curve: List[DotProductALU]
    products: List[ProductReference]
    die_area_mm2: float = DEFAULT_DIE_AREA_MM2
    silicon_clock_ghz: float = DEFAULT_SILICON_CLOCK_GHZ
    generated_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alus": [a.to_dict() for a in self.alus],
            "analyses": [a.to_dict() for a in self.analyses],
            "parametric_curve": [p.to_dict() for p in self.parametric_curve],
            "products": [p.to_dict() for p in self.products],
            "die_area_mm2": self.die_area_mm2,
            "silicon_clock_ghz": self.silicon_clock_ghz,
            "generated_at": self.generated_at,
        }


def build_default_sol_report(
    die_area_mm2: float = DEFAULT_DIE_AREA_MM2,
    silicon_clock_ghz: float = DEFAULT_SILICON_CLOCK_GHZ,
) -> SoLReport:
    from datetime import datetime, timezone
    alus = default_alu_catalog()
    analyses = [
        SoLAnalysis(alu=a, die_area_mm2=die_area_mm2,
                    silicon_clock_ghz=silicon_clock_ghz)
        for a in alus
    ]
    curve = generate_parametric_curve()
    return SoLReport(
        alus=alus,
        analyses=analyses,
        parametric_curve=curve,
        products=default_product_references(),
        die_area_mm2=die_area_mm2,
        silicon_clock_ghz=silicon_clock_ghz,
        generated_at=datetime.now(timezone.utc).isoformat(),
    )


# ======================================================================
# HTML rendering
# ======================================================================


def render_alu_instance_table(alus: List[DotProductALU]) -> str:
    """Per-instance table: shows the physical unit of replication.

    Columns are pure ALU attributes. Die-level operand bandwidth
    (bytes/MAC at the die boundary) is a topology / composition
    property, not an ALU property - it belongs in the silicon-
    composition view, not here.
    """
    rows = []
    for alu in alus:
        rows.append(
            f'<tr>'
            f'<td class="name">{html.escape(alu.name)}</td>'
            f'<td class="num">{alu.W}</td>'
            f'<td>{html.escape(alu.precision)}</td>'
            f'<td class="num">{alu.process_nm}</td>'
            f'<td>{html.escape(alu.accum_mode.value)}</td>'
            f'<td class="num">{alu.area_mm2*1e6:.0f}</td>'
            f'<td class="num">{alu.transistor_count_k:.0f}</td>'
            f'<td class="num"><strong>{alu.pj_per_clock:.3f}</strong></td>'
            f'<td class="num">{alu.bytes_per_mac_alu:.2f}</td>'
            f'<td class="src"><em>{html.escape(alu.citation)}</em></td>'
            f'</tr>'
        )
    return (
        '<table class="blocks">'
        '<thead><tr>'
        '<th>ALU archetype</th>'
        '<th>ALU Width (MAC/clk)</th>'
        '<th>Precision</th>'
        '<th>Process (nm)</th>'
        '<th>Accumulator</th>'
        '<th>Area / instance (um^2)</th>'
        '<th>Trans / instance (K)</th>'
        '<th>pJ / clock / instance</th>'
        '<th>B / MAC (ALU)</th>'
        '<th>Derivation</th>'
        '</tr></thead>'
        f'<tbody>{"".join(rows)}</tbody>'
        '</table>'
    )


def render_alu_per_mac_table(alus: List[DotProductALU]) -> str:
    """Per-MAC derivation for TOPS/W ceiling comparisons. Pure ALU
    attributes only; topology-dependent die bandwidth lives in the
    silicon-composition view."""
    rows = []
    for alu in alus:
        rows.append(
            f'<tr>'
            f'<td class="name">{html.escape(alu.name)}</td>'
            f'<td class="num">{alu.W}</td>'
            f'<td class="num">{alu.area_per_mac_mm2*1e6:.1f}</td>'
            f'<td class="num">{alu.transistor_count_per_mac_k:.1f}</td>'
            f'<td class="num"><strong>{alu.pj_per_mac:.3f}</strong></td>'
            f'<td class="num">{alu.bytes_per_mac_alu:.2f}</td>'
            f'<td class="num"><strong>{alu.tops_per_watt_ceiling:.1f}</strong></td>'
            f'</tr>'
        )
    return (
        '<table class="blocks">'
        '<thead><tr>'
        '<th>ALU archetype</th>'
        '<th>ALU Width</th>'
        '<th>Area / MAC (um^2)</th>'
        '<th>Trans / MAC (K)</th>'
        '<th>pJ / MAC</th>'
        '<th>B / MAC (ALU)</th>'
        '<th>TOPS / W ceiling</th>'
        '</tr></thead>'
        f'<tbody>{"".join(rows)}</tbody>'
        '</table>'
    )


def render_parametric_curve_table(curve: List[DotProductALU]) -> str:
    """Parametric cost curve at each W, same precision and process."""
    rows = []
    for alu in curve:
        rows.append(
            f'<tr>'
            f'<td class="num"><strong>{alu.W}</strong></td>'
            f'<td class="num">{alu.transistor_count_k:.1f}</td>'
            f'<td class="num">{alu.area_mm2*1e6:.0f}</td>'
            f'<td class="num">{alu.pj_per_clock:.3f}</td>'
            f'<td class="num">{alu.transistor_count_per_mac_k:.2f}</td>'
            f'<td class="num">{alu.area_per_mac_mm2*1e6:.1f}</td>'
            f'<td class="num">{alu.pj_per_mac:.4f}</td>'
            f'<td class="num">{alu.bytes_per_mac_alu:.2f}</td>'
            f'<td class="num"><strong>{alu.tops_per_watt_ceiling:.0f}</strong></td>'
            f'</tr>'
        )
    header = (
        '<table class="blocks">'
        '<thead><tr>'
        '<th>ALU Width</th>'
        '<th colspan="3" style="text-align:center;">Per instance</th>'
        '<th colspan="4" style="text-align:center;">Per MAC</th>'
        '<th>Ceiling</th>'
        '</tr><tr>'
        '<th></th>'
        '<th>Trans (K)</th><th>Area (um^2)</th><th>pJ/clock</th>'
        '<th>Trans/MAC (K)</th><th>Area/MAC (um^2)</th>'
        '<th>pJ/MAC</th><th>B/MAC (ALU)</th>'
        '<th>TOPS/W</th>'
        '</tr></thead>'
    )
    return header + f'<tbody>{"".join(rows)}</tbody></table>'


def render_sol_summary_table(
    analyses: List[SoLAnalysis], die_area_mm2: float,
) -> str:
    rows = []
    for a in analyses:
        f = a.silicon_clock_ghz
        rows.append(
            f'<tr>'
            f'<td class="name">{html.escape(a.alu.name)}</td>'
            f'<td class="num">{a.alu.W}</td>'
            f'<td class="num">{a.num_alus:,}</td>'
            f'<td class="num">{a.num_macs_on_die:,}</td>'
            f'<td class="num">{a.die_transistor_count_m:.0f}</td>'
            f'<td class="num"><strong>{a.peak_tops(f):,.0f}</strong></td>'
            f'<td class="num">{a.die_power_w(f):.0f}</td>'
            f'<td class="num"><strong>'
            f'{a.alu.tops_per_watt_ceiling:.1f}</strong></td>'
            f'</tr>'
        )
    return (
        '<table class="blocks">'
        '<thead><tr>'
        f'<th>ALU archetype (on {die_area_mm2:.0f} mm^2 die)</th>'
        '<th>ALU Width</th>'
        '<th># ALU instances</th>'
        '<th># MACs on die</th>'
        '<th>Die trans (M)</th>'
        '<th>Peak TOPS @ 1.5 GHz</th>'
        '<th>Die power (W)</th>'
        '<th>TOPS / W ceiling</th>'
        '</tr></thead>'
        f'<tbody>{"".join(rows)}</tbody>'
        '</table>'
    )


def render_tdp_sweep_table(
    analyses: List[SoLAnalysis], tdp_targets: Tuple[float, ...],
) -> str:
    header_cols = "".join(
        f'<th class="num" colspan="2">{html.escape(a.alu.name)}</th>'
        for a in analyses
    )
    sub_cols = "".join(
        '<th class="num">Clock (GHz)</th>'
        '<th class="num">Peak TOPS</th>'
        for _ in analyses
    )
    rows = []
    for tdp in tdp_targets:
        cells = []
        for a in analyses:
            swept = a.tdp_sweep()
            rec = next(r for r in swept if abs(r["tdp_w"] - tdp) < 1e-9)
            clock_text = f'{rec["effective_clock_ghz"]:.2f}'
            if rec["silicon_limited"]:
                clock_text += " *"
            cells.append(
                f'<td class="num">{clock_text}</td>'
                f'<td class="num"><strong>'
                f'{rec["peak_tops"]:,.0f}</strong></td>'
            )
        rows.append(
            f'<tr><td class="name">{tdp:.0f} W</td>{"".join(cells)}</tr>'
        )
    return (
        '<table class="blocks">'
        f'<thead><tr><th rowspan="2">TDP target</th>{header_cols}</tr>'
        f'<tr>{sub_cols}</tr></thead>'
        f'<tbody>{"".join(rows)}</tbody>'
        '</table>'
    )


def render_gap_to_products_table(
    products: List[ProductReference], analyses: List[SoLAnalysis],
) -> str:
    alu_cols = "".join(
        f'<th class="num" colspan="2">{html.escape(a.alu.name)}<br/>'
        f'<span style="font-weight:normal; font-size:10px; color:#586374;">'
        f'SoL @ product TDP</span></th>'
        for a in analyses
    )
    sub_cols = "".join(
        '<th class="num">SoL TOPS</th><th class="num">Actual / SoL</th>'
        for _ in analyses
    )
    rows = []
    for p in products:
        cells = []
        for a in analyses:
            f_for_tdp = a.clock_for_tdp(p.tdp_w)
            effective_f = min(f_for_tdp, a.silicon_clock_ghz)
            sol_tops = a.peak_tops(effective_f)
            ratio = p.peak_int8_tops / sol_tops if sol_tops > 0 else 0.0
            cells.append(
                f'<td class="num">{sol_tops:,.0f}</td>'
                f'<td class="num"><strong>{ratio*100:.0f}%</strong></td>'
            )
        rows.append(
            f'<tr>'
            f'<td class="name">{html.escape(p.name)}</td>'
            f'<td class="num">{p.process_nm}</td>'
            f'<td class="num">{p.die_area_mm2:.0f}</td>'
            f'<td class="num">{p.tdp_w:.0f} W</td>'
            f'<td class="num"><strong>{p.peak_int8_tops:,.0f}</strong></td>'
            f'<td class="num">{p.tops_per_watt:.1f}</td>'
            f'{"".join(cells)}'
            f'</tr>'
        )
    return (
        '<table class="blocks">'
        '<thead><tr>'
        '<th rowspan="2">Product</th><th rowspan="2">Process (nm)</th>'
        '<th rowspan="2">Die (mm^2)</th>'
        '<th rowspan="2">TDP (W)</th>'
        '<th rowspan="2">Actual TOPS</th>'
        '<th rowspan="2">Actual TOPS/W</th>'
        f'{alu_cols}'
        '</tr>'
        f'<tr>{sub_cols}</tr></thead>'
        f'<tbody>{"".join(rows)}</tbody>'
        '</table>'
    )


def render_tradeoff_chart_js(
    curve: List[DotProductALU],
    alus: List[DotProductALU],
) -> str:
    """Plotly chart: parametric curve vs real-product points along W.

    Three panels sharing an x-axis (W):
      - transistors per MAC
      - pJ per MAC
      - bytes per MAC
    Real products appear as labelled markers.
    """
    # Curve series (from parametric model). Parametric curve assumes
    # no topology-level reuse, so bytes_per_mac_alu == bytes_per_mac_die.
    curve_x = [p.W for p in curve]
    curve_y_trans_per_mac = [p.transistor_count_per_mac_k for p in curve]
    curve_y_pj_per_mac = [p.pj_per_mac for p in curve]
    curve_y_bytes_per_mac_alu = [p.bytes_per_mac_alu for p in curve]

    # Product points (non-parametric). Real products report BOTH ALU-
    # level demand (what the ALU itself reads) and die-level demand
    # (what the die as a whole reads from memory after topology-level
    # reuse).
    real = [a for a in alus if not a.is_parametric]
    real_x = [a.W for a in real]
    real_labels = [a.name for a in real]
    real_y_trans = [a.transistor_count_per_mac_k for a in real]
    real_y_pj = [a.pj_per_mac for a in real]
    real_y_bytes_alu = [a.bytes_per_mac_alu for a in real]
    real_y_bytes_die = [a.bytes_per_mac_die for a in real]

    def panel(y_curve: List[float], y_real: List[float],
              title: str, ytitle: str, log_y: bool = False) -> Dict[str, Any]:
        traces = [
            {
                "type": "scatter", "mode": "lines+markers",
                "name": "Parametric curve (INT8, 8 nm)",
                "x": curve_x, "y": y_curve,
                "line": {"color": "#5b8ff9"},
                "marker": {"size": 6, "color": "#5b8ff9"},
            },
            {
                "type": "scatter", "mode": "markers+text",
                "name": "Real products",
                "x": real_x, "y": y_real,
                "text": real_labels,
                "textposition": "top right",
                "textfont": {"size": 10, "color": "#0a2540"},
                "marker": {
                    "size": 11, "color": "#d4860b",
                    "symbol": "diamond",
                    "line": {"width": 1, "color": "#0a2540"},
                },
            },
        ]
        layout: Dict[str, Any] = {
            "title": title,
            "xaxis": {
                "title": "ALU Width (MACs per instance per clock)",
                "type": "log", "tickvals": [1, 2, 4, 8, 16, 32, 64],
            },
            "yaxis": {"title": ytitle},
            "margin": {"t": 50, "b": 60, "l": 70, "r": 20},
            "showlegend": True,
            "legend": {"orientation": "h", "y": -0.2},
        }
        if log_y:
            layout["yaxis"]["type"] = "log"
        return {"data": traces, "layout": layout}

    # Bandwidth panel is special: the ALU-level bandwidth is a
    # constant (per precision) across W - the curve is flat. The
    # die-level bandwidth drops steeply for mesh / systolic / tiled-
    # matmul topologies but has nothing to do with W directly; it is
    # a topology-level property. Show both so the reader sees:
    #   - the flat ALU-level line (a dot-product ALU always reads
    #     2 * bytes_per_op per MAC, no matter how wide it is)
    #   - the scattered die-level points (orange diamonds at lower
    #     B/MAC values where topology reuse kicks in)
    bandwidth_panel = {
        "data": [
            {
                "type": "scatter", "mode": "lines+markers",
                "name": "Parametric ALU-level (= 2 x bytes/operand)",
                "x": curve_x, "y": curve_y_bytes_per_mac_alu,
                "line": {"color": "#5b8ff9"},
                "marker": {"size": 6, "color": "#5b8ff9"},
            },
            {
                "type": "scatter", "mode": "markers+text",
                "name": "Real products - ALU-level",
                "x": real_x, "y": real_y_bytes_alu,
                "text": [n for n in real_labels],
                "textposition": "top right",
                "textfont": {"size": 9, "color": "#586374"},
                "marker": {
                    "size": 9, "color": "#9db4e0",
                    "symbol": "circle-open",
                    "line": {"width": 1.5, "color": "#5b8ff9"},
                },
            },
            {
                "type": "scatter", "mode": "markers+text",
                "name": "Real products - die-level (after topology reuse)",
                "x": real_x, "y": real_y_bytes_die,
                "text": real_labels,
                "textposition": "bottom right",
                "textfont": {"size": 10, "color": "#0a2540"},
                "marker": {
                    "size": 11, "color": "#d4860b",
                    "symbol": "diamond",
                    "line": {"width": 1, "color": "#0a2540"},
                },
            },
        ],
        "layout": {
            "title": (
                "Operand bandwidth per MAC: ALU-level (constant) "
                "vs die-level (topology-dependent)"
            ),
            "xaxis": {
                "title": "ALU Width (MACs per instance per clock)",
                "type": "log", "tickvals": [1, 2, 4, 8, 16, 32, 64],
            },
            "yaxis": {"title": "Bytes / MAC", "type": "log"},
            "margin": {"t": 50, "b": 60, "l": 70, "r": 20},
            "showlegend": True,
            "legend": {"orientation": "h", "y": -0.2},
        },
    }

    payload = {
        "chart_sol_trans_per_mac": panel(
            curve_y_trans_per_mac, real_y_trans,
            "Transistors per MAC vs ALU width",
            "K transistors / MAC",
        ),
        "chart_sol_pj_per_mac": panel(
            curve_y_pj_per_mac, real_y_pj,
            "Energy per MAC vs ALU width",
            "pJ / MAC", log_y=True,
        ),
        "chart_sol_bytes_per_mac": bandwidth_panel,
    }
    return (
        f"const SOL_CHARTS = {json.dumps(payload)};\n"
        "for (const [id, spec] of Object.entries(SOL_CHARTS)) {\n"
        "  const el = document.getElementById(id);\n"
        "  if (el) Plotly.newPlot(id, spec.data, spec.layout, "
        "{displayModeBar: false, responsive: true});\n"
        "}\n"
    )
