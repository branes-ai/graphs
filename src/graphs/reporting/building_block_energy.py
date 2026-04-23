"""
Building-block (per-clock) energy accounting for the two foundational
compute engines - NVIDIA SM and KPU compute tile - as they are deployed
on the SoC.

This complements ``microarch_accounting.py`` (per-MAC view). The two
views reconcile through:

    pJ/MAC = total_pJ_per_clock / native_MACs_per_clock

Use cases:

- **Per-MAC view** (microarch_accounting.py): understand performance
  under a fixed power constraint. Answers "how much compute can I get
  per watt given workload X on architecture Y?"

- **Per-clock view** (this module): drives SoC composition and
  super-cluster designs. Answers "if I deploy N building blocks at
  frequency F with utilization U, how much power and how much compute
  will the chip deliver?"

Each component carries its absolute per-clock energy contribution AND
an estimated transistor count so circuit-designer intuition (SRAM,
ALU, crossbar, clock tree) can sanity-check both numbers.

Process technology: all three building blocks are reported at 8 nm
(matched to the Ampere Jetson baseline) so direct comparisons do not
require a process-normalization chart.
"""
from __future__ import annotations

import html
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from graphs.reporting.microarch_accounting import StructureCategory
from graphs.reporting.microarch_html_template import (
    _CSS,
    _load_logo,
    _render_brand_footer,
    _render_brand_header,
)


@dataclass
class EngineComponent:
    """
    One major component inside a compute engine (e.g. register file,
    tensor cores, L1 scratchpad).

    Each component reports THREE independent estimates so circuit
    designers can sanity-check the model:
      - transistor_count_m: estimated silicon, millions of transistors
      - area_mm2: estimated silicon area at the engine's process node
      - per_clock_pj: steady-state energy per clock

    Typical density rules at 8 nm:
      - Dense SRAM (6T cells + peripheral): ~100 M transistors / mm^2
      - Logic (mixed std cells + flops):    ~45 M transistors / mm^2
      - Dense MAC arrays (regular tiling):  ~80 M transistors / mm^2
    """
    name: str
    category: StructureCategory
    count: int                    # number of instances inside the engine
    size_or_spec: str             # "256 KB, 4 banks" / "1024 PE (32x32)"
    transistor_count_m: float     # estimated silicon, millions of transistors
    area_mm2: float               # estimated silicon area (mm^2)
    per_clock_pj: float           # absolute energy contribution per clock
    activity_note: str            # steady-state activity assumption
    citation: str                 # derivation / source

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category.value,
            "count": self.count,
            "size_or_spec": self.size_or_spec,
            "transistor_count_m": self.transistor_count_m,
            "area_mm2": self.area_mm2,
            "per_clock_pj": self.per_clock_pj,
            "activity_note": self.activity_note,
            "citation": self.citation,
        }


@dataclass
class BuildingBlock:
    """
    One replicated compute engine on a SoC (SM, KPU tile, etc.).
    Aggregates components and exposes engine-level metrics.
    """
    name: str
    process_nm: int
    clock_ghz: float
    native_macs_per_clock: int
    native_op_precision: str       # "INT8 HMMA", "FP32 FMA", "INT8 MAC"
    components: List[EngineComponent] = field(default_factory=list)
    color: str = "#0a2540"
    notes: str = ""

    @property
    def total_pj_per_clock(self) -> float:
        return sum(c.per_clock_pj for c in self.components)

    @property
    def total_transistor_count_m(self) -> float:
        return sum(c.transistor_count_m for c in self.components)

    @property
    def total_area_mm2(self) -> float:
        return sum(c.area_mm2 for c in self.components)

    @property
    def execute_pj_per_clock(self) -> float:
        return sum(c.per_clock_pj for c in self.components
                   if c.category == StructureCategory.EXECUTE)

    @property
    def execute_fraction(self) -> float:
        tot = self.total_pj_per_clock
        return self.execute_pj_per_clock / tot if tot > 0 else 0.0

    @property
    def power_mw(self) -> float:
        return self.total_pj_per_clock * self.clock_ghz

    @property
    def derived_pj_per_mac(self) -> float:
        return (self.total_pj_per_clock / self.native_macs_per_clock
                if self.native_macs_per_clock > 0 else 0.0)

    @property
    def pj_per_mtransistor_per_clock(self) -> float:
        tc = self.total_transistor_count_m
        return self.total_pj_per_clock / tc if tc > 0 else 0.0

    def by_category(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for c in self.components:
            out[c.category.value] = out.get(c.category.value, 0.0) + c.per_clock_pj
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "process_nm": self.process_nm,
            "clock_ghz": self.clock_ghz,
            "native_macs_per_clock": self.native_macs_per_clock,
            "native_op_precision": self.native_op_precision,
            "total_pj_per_clock": self.total_pj_per_clock,
            "total_transistor_count_m": self.total_transistor_count_m,
            "total_area_mm2": self.total_area_mm2,
            "execute_pj_per_clock": self.execute_pj_per_clock,
            "execute_fraction": self.execute_fraction,
            "power_mw": self.power_mw,
            "derived_pj_per_mac": self.derived_pj_per_mac,
            "pj_per_mtransistor_per_clock": self.pj_per_mtransistor_per_clock,
            "components": [c.to_dict() for c in self.components],
            "by_category": self.by_category(),
            "color": self.color,
            "notes": self.notes,
        }


@dataclass
class SocComposition:
    """
    Composes a SoC by replicating a building block N times plus a
    process-dependent overhead (memory controller, L3, IO, etc.).
    """
    block: BuildingBlock
    block_count: int
    utilization: float             # fraction of blocks active per clock
    overhead_mw: float             # fixed SoC overhead at this freq

    @property
    def active_blocks(self) -> float:
        return self.block_count * self.utilization

    @property
    def compute_power_mw(self) -> float:
        return self.active_blocks * self.block.power_mw

    @property
    def total_power_w(self) -> float:
        return (self.compute_power_mw + self.overhead_mw) / 1000.0

    @property
    def peak_tops_int8(self) -> float:
        """Peak INT8 TOPS assuming 2 ops per MAC (multiply + add)."""
        macs_per_sec = (self.block_count * self.block.native_macs_per_clock
                        * self.block.clock_ghz * 1e9)
        return macs_per_sec * 2 / 1e12

    @property
    def sustained_tops_int8(self) -> float:
        return self.peak_tops_int8 * self.utilization

    @property
    def sustained_tops_per_watt(self) -> float:
        tot_w = self.total_power_w
        return self.sustained_tops_int8 / tot_w if tot_w > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "block_name": self.block.name,
            "block_count": self.block_count,
            "utilization": self.utilization,
            "overhead_mw": self.overhead_mw,
            "compute_power_mw": self.compute_power_mw,
            "total_power_w": self.total_power_w,
            "peak_tops_int8": self.peak_tops_int8,
            "sustained_tops_int8": self.sustained_tops_int8,
            "sustained_tops_per_watt": self.sustained_tops_per_watt,
        }


# ----------------------------------------------------------------------
# NVIDIA Ampere SM (GA10x @ 8 nm, ~0.65 GHz sustained on Orin AGX)
# ----------------------------------------------------------------------
#
# Two workloads produce two "building block" views of the SAME silicon:
#
#   build_nvidia_sm_building_block()         -> Tensor-Core path
#         Native throughput: 4096 INT8 MACs/clock (4 TC x 1024 MAC/clk)
#
#   build_nvidia_sm_cuda_path_building_block() -> CUDA-core path
#         Native throughput: 128 FP32 MACs/clock (128 FMA lanes)
#
# Both views use the same per-component transistor counts; only the
# per_clock_pj values differ to reflect which functional units are
# active vs. clock-gated.
# ----------------------------------------------------------------------

def build_nvidia_sm_building_block(
    clock_ghz: float = 1.5,
) -> BuildingBlock:
    """
    Ampere SM running an INT8 HMMA (Tensor-Core) workload at steady
    state. CUDA cores, SFUs, RT cores are clock-gated in this mode
    and contribute only static + clock-tree energy.

    Default clock 1.5 GHz reflects the silicon capability at 8 nm
    (comparable to desktop/mobile GA106 at 1.3-1.8 GHz boost). The
    Jetson AGX Orin 30 W sustain of ~0.65 GHz is TDP-limited, not
    silicon-limited - DVFS clamps the clock so 16 SMs fit the 30 W
    SoC envelope. For a deployed-product view, call with
    clock_ghz=0.65.

    Transistor counts reference public GA10x die-area breakdowns:
    the full GA10x SM lands at ~240-260 M transistors, dominated by
    the 4 Tensor Cores (~100 M) and the on-SM caches + RF (~60 M).
    """
    components = [
        EngineComponent(
            name="Register file (main)",
            category=StructureCategory.MEMORY,
            count=1,
            size_or_spec="64K x 32b = 256 KB, 4-bank",
            transistor_count_m=20.0,
            area_mm2=0.20,  # SRAM density ~100 MT/mm^2
            per_clock_pj=100.0,
            activity_note=(
                "HMMA frag A+B reads ~512 B / issue; 1 MMA issue / clock "
                "distributed across 4 TC partitions"
            ),
            citation=(
                "256 KB 6T SRAM array ~13 M + peripheral ~7 M = 20 M. "
                "Area @ 8 nm SRAM density ~100 MT/mm^2 -> 0.20 mm^2. "
                "Energy: 512 B x 0.014 pJ/B = 7 pJ/issue dynamic + "
                "array static + clock ~95 pJ/clock."
            ),
        ),
        EngineComponent(
            name="Instruction pipeline (L0 I$ + decode + IB)",
            category=StructureCategory.FETCH,
            count=4,
            size_or_spec="4 sub-partitions, 1 inst/clk each",
            transistor_count_m=8.0,
            area_mm2=0.12,  # mixed SRAM + logic
            per_clock_pj=50.0,
            activity_note="4 schedulers issue 1 instruction/clock each",
            citation=(
                "2 KB L0 I$ x 4 + decoder + IB latches ~2 M each = 8 M. "
                "Area: mixed SRAM + logic ~65 MT/mm^2 -> 0.12 mm^2. "
                "Energy: ~8 pJ fetch + ~4 pJ decode per issue x 4."
            ),
        ),
        EngineComponent(
            name="Warp scheduler + scoreboard",
            category=StructureCategory.SCHEDULE,
            count=4,
            size_or_spec="1 scheduler + scoreboard per sub-partition",
            transistor_count_m=5.0,
            area_mm2=0.11,  # logic ~45 MT/mm^2
            per_clock_pj=40.0,
            activity_note="Ready-queue check + dependency tracking per issue",
            citation=(
                "4 schedulers x ~1 M each + ~1 M scoreboard = 5 M. Area "
                "at logic density 45 MT/mm^2 -> 0.11 mm^2. Energy: "
                "~10 pJ per issued instruction x 4."
            ),
        ),
        EngineComponent(
            name="Operand collector + bypass crossbar",
            category=StructureCategory.REGISTER,
            count=4,
            size_or_spec="1 collector per sub-partition, 4-bank xbar",
            transistor_count_m=10.0,
            area_mm2=0.25,  # wire-heavy, lower effective density
            per_clock_pj=90.0,
            activity_note="All issues pass through operand collector",
            citation=(
                "4 collectors + 4-bank crossbar ~10 M. Area is wire-"
                "heavy (effective ~40 MT/mm^2) -> 0.25 mm^2. Energy "
                "dominated by crossbar static capacitance (~22 pJ x 4)."
            ),
        ),
        EngineComponent(
            name="CUDA cores (FP32/INT32 FMA)",
            category=StructureCategory.EXECUTE,
            count=128,
            size_or_spec="128 lanes = 4 partitions x 32",
            transistor_count_m=10.0,
            area_mm2=0.22,  # logic
            per_clock_pj=15.0,
            activity_note="Clock-gated during HMMA; only clock-tree + leakage",
            citation=(
                "128 FP32 FMA units x ~78 K transistors each = 10 M. "
                "Area at logic density 45 MT/mm^2 -> 0.22 mm^2. Idle "
                "HMMA activity: ~5% duty yields 15 pJ/clock."
            ),
        ),
        EngineComponent(
            name="SFUs (special function units)",
            category=StructureCategory.EXECUTE,
            count=16,
            size_or_spec="4 SFUs per partition",
            transistor_count_m=2.0,
            area_mm2=0.04,
            per_clock_pj=10.0,
            activity_note="Idle on pure GEMM",
            citation="2 M at logic density -> 0.04 mm^2. Clock+leak only.",
        ),
        EngineComponent(
            name="Tensor Cores",
            category=StructureCategory.EXECUTE,
            count=4,
            size_or_spec="16x16x16 HMMA, 1024 INT8 MAC/clock each",
            transistor_count_m=100.0,
            area_mm2=1.25,  # dense MAC array ~80 MT/mm^2
            per_clock_pj=580.0,
            activity_note="All 4 TCs firing; 4096 INT8 MACs/clock aggregate",
            citation=(
                "Per TC ~25 M: 1024 INT8 MAC units + local operand regs "
                "+ 16x16 INT32 accum + fragment broadcast. Dense MAC "
                "array density ~80 MT/mm^2 -> 1.25 mm^2 for 4 TCs. "
                "Energy: 4096 MAC x ~0.12 pJ + ~90 pJ internal."
            ),
        ),
        EngineComponent(
            name="L1 data cache / shared memory",
            category=StructureCategory.MEMORY,
            count=1,
            size_or_spec="128 KB, unified L1/SMEM",
            transistor_count_m=20.0,
            area_mm2=0.20,  # SRAM
            per_clock_pj=30.0,
            activity_note="LDSM fragment loads feed Tensor Cores",
            citation=(
                "128 KB 6T SRAM ~13 M + peripheral ~7 M = 20 M. "
                "Area at SRAM density ~100 MT/mm^2 -> 0.20 mm^2. "
                "Modest HMMA LDSM traffic ~30 pJ/clock."
            ),
        ),
        EngineComponent(
            name="RT cores (ray-tracing)",
            category=StructureCategory.EXECUTE,
            count=1,
            size_or_spec="Ampere 2nd-gen, ray-triangle intersect",
            transistor_count_m=20.0,
            area_mm2=0.40,  # logic-heavy
            per_clock_pj=10.0,
            activity_note="Idle on inference workload",
            citation=(
                "~20 M logic at 50 MT/mm^2 -> 0.40 mm^2. Clock-tree + "
                "leakage only on GEMM."
            ),
        ),
        EngineComponent(
            name="Texture / surface unit",
            category=StructureCategory.MEMORY,
            count=1,
            size_or_spec="4 texture filters + LSU",
            transistor_count_m=20.0,
            area_mm2=0.40,  # mixed logic + small caches
            per_clock_pj=20.0,
            activity_note="Mostly idle on GEMM; light LSU traffic for weights",
            citation=(
                "~20 M mixed logic + small filter caches at 50 MT/mm^2 "
                "-> 0.40 mm^2. Texture gated; LSU light."
            ),
        ),
        EngineComponent(
            name="Clock tree + leakage (SM-wide)",
            category=StructureCategory.CONTROL,
            count=1,
            size_or_spec="H-tree + rail decoupling + unaccounted infra",
            transistor_count_m=25.0,
            area_mm2=0.65,  # distributed/structural
            per_clock_pj=80.0,
            activity_note="Always-on; includes non-itemized local infrastructure",
            citation=(
                "~25 M distributed transistors (clock buffers, decap, "
                "rail muxes) -> 0.65 mm^2. ~8% of dynamic engine power "
                "at 8 nm HPM."
            ),
        ),
    ]

    return BuildingBlock(
        name=(
            f"NVIDIA Streaming Multiprocessor (Ampere, TC path - "
            f"INT8 HMMA @ {clock_ghz:.2f} GHz)"
        ),
        process_nm=8,
        clock_ghz=clock_ghz,
        native_macs_per_clock=4096,  # 4 TCs x 1024 MACs/clock
        native_op_precision="INT8 HMMA (4 TCs x 16x16x16)",
        components=components,
        color="#5b8ff9",
        notes=(
            "SM-level budget during INT8 HMMA. Excludes L2 cache, memory "
            "controller, and DRAM (SoC-level). Silicon footprint "
            "~240 M transistors, dominated by the 4 Tensor Cores "
            "(~100 M) and local caches/RF (~60 M). At the silicon-"
            "capability clock of 1.5 GHz, one SM draws ~1.54 W; the "
            "Orin AGX's 0.65 GHz sustain is a TDP-limited operating "
            "point (16 SMs x 1.54 W at 1.5 GHz = 24 W compute alone, "
            "past the 30 W SoC envelope once L2, memory, CPU, etc. "
            "are included)."
        ),
    )


def build_nvidia_sm_cuda_path_building_block(
    clock_ghz: float = 1.5,
) -> BuildingBlock:
    """
    Ampere SM running an FP32 GEMM through the 128 CUDA-core FMA lanes.
    Tensor Cores, RT cores, SFUs are clock-gated; the CUDA cores are
    active and draw their full FP32 FMA energy. This is the native
    path for FP32 workloads (Kalman, SLAM, HPC) that cannot use TCs.

    Same silicon as the TC path - only the activity-induced pJ/clock
    values differ to reflect which units are firing. Default clock
    1.5 GHz reflects silicon capability; call with clock_ghz=0.65
    for the Orin AGX deployed view.
    """
    components = [
        EngineComponent(
            name="Register file (main)",
            category=StructureCategory.MEMORY,
            count=1,
            size_or_spec="64K x 32b = 256 KB, 4-bank",
            transistor_count_m=20.0,
            area_mm2=0.20,
            per_clock_pj=90.0,
            activity_note=(
                "128 lanes x ~3 operands (4 B each, FP32) per FMA; "
                "~1.5 KB/clock RF read + ~0.5 KB writeback"
            ),
            citation=(
                "Same silicon as TC path. Energy: 128 x 12 B reads x "
                "0.014 pJ/B = 21 pJ dynamic + array static ~70 pJ."
            ),
        ),
        EngineComponent(
            name="Instruction pipeline (L0 I$ + decode + IB)",
            category=StructureCategory.FETCH,
            count=4,
            size_or_spec="4 sub-partitions, 1 inst/clk each",
            transistor_count_m=8.0,
            area_mm2=0.12,
            per_clock_pj=50.0,
            activity_note="4 schedulers issue 1 instruction/clock each",
            citation="Same silicon as TC path (instruction stream same).",
        ),
        EngineComponent(
            name="Warp scheduler + scoreboard",
            category=StructureCategory.SCHEDULE,
            count=4,
            size_or_spec="1 scheduler + scoreboard per sub-partition",
            transistor_count_m=5.0,
            area_mm2=0.11,
            per_clock_pj=40.0,
            activity_note="Ready-queue check + dependency tracking per issue",
            citation="Same silicon as TC path.",
        ),
        EngineComponent(
            name="Operand collector + bypass crossbar",
            category=StructureCategory.REGISTER,
            count=4,
            size_or_spec="1 collector per sub-partition, 4-bank xbar",
            transistor_count_m=10.0,
            area_mm2=0.25,
            per_clock_pj=90.0,
            activity_note="All CUDA-core issues pass through operand collector",
            citation="Same silicon; comparable crossbar traffic ~22 pJ x 4.",
        ),
        EngineComponent(
            name="CUDA cores (FP32/INT32 FMA)",
            category=StructureCategory.EXECUTE,
            count=128,
            size_or_spec="128 lanes = 4 partitions x 32, FP32 FMA",
            transistor_count_m=10.0,
            area_mm2=0.22,
            per_clock_pj=320.0,
            activity_note="All 128 lanes firing one FP32 FMA per clock",
            citation=(
                "Same 10 M silicon. FP32 FMA ~2.5 pJ at 8 nm (Horowitz; "
                "~23 FA for mantissa multiply + 24-bit add); 128 x 2.5 "
                "= 320 pJ."
            ),
        ),
        EngineComponent(
            name="SFUs (special function units)",
            category=StructureCategory.EXECUTE,
            count=16,
            size_or_spec="4 SFUs per partition",
            transistor_count_m=2.0,
            area_mm2=0.04,
            per_clock_pj=10.0,
            activity_note="Idle on FP32 GEMM",
            citation="Same silicon; clock-tree + leakage only.",
        ),
        EngineComponent(
            name="Tensor Cores",
            category=StructureCategory.EXECUTE,
            count=4,
            size_or_spec="16x16x16 HMMA, 1024 INT8 MAC/clock each",
            transistor_count_m=100.0,
            area_mm2=1.25,
            per_clock_pj=50.0,
            activity_note="Clock-gated; only clock-tree + leakage",
            citation="Same 100 M silicon; 8 nm HPM leakage ~50 pJ/clock.",
        ),
        EngineComponent(
            name="L1 data cache / shared memory",
            category=StructureCategory.MEMORY,
            count=1,
            size_or_spec="128 KB, unified L1/SMEM",
            transistor_count_m=20.0,
            area_mm2=0.20,
            per_clock_pj=20.0,
            activity_note="Light LSU traffic for FP32 GEMM tiles",
            citation="Same SRAM silicon; less fragment-heavy ~20 pJ/clock.",
        ),
        EngineComponent(
            name="RT cores (ray-tracing)",
            category=StructureCategory.EXECUTE,
            count=1,
            size_or_spec="Ampere 2nd-gen, ray-triangle intersect",
            transistor_count_m=20.0,
            area_mm2=0.40,
            per_clock_pj=10.0,
            activity_note="Idle on GEMM workload",
            citation="Same silicon; clock-tree + leakage only.",
        ),
        EngineComponent(
            name="Texture / surface unit",
            category=StructureCategory.MEMORY,
            count=1,
            size_or_spec="4 texture filters + LSU",
            transistor_count_m=20.0,
            area_mm2=0.40,
            per_clock_pj=20.0,
            activity_note="Mostly idle on GEMM",
            citation="Same silicon; texture gated; LSU light.",
        ),
        EngineComponent(
            name="Clock tree + leakage (SM-wide)",
            category=StructureCategory.CONTROL,
            count=1,
            size_or_spec="H-tree + rail decoupling + unaccounted infra",
            transistor_count_m=25.0,
            area_mm2=0.65,
            per_clock_pj=80.0,
            activity_note="Always-on",
            citation="Same silicon; ~8% of dynamic engine power.",
        ),
    ]

    return BuildingBlock(
        name=(
            f"NVIDIA Streaming Multiprocessor (Ampere, CUDA-core path - "
            f"FP32 FMA @ {clock_ghz:.2f} GHz)"
        ),
        process_nm=8,
        clock_ghz=clock_ghz,
        native_macs_per_clock=128,       # 128 FP32 FMA lanes
        native_op_precision="FP32 FMA (128 CUDA-core lanes)",
        components=components,
        color="#9db4e0",
        notes=(
            "Same SM silicon, different workload: the 128 CUDA-core "
            "lanes execute one FP32 FMA per clock while Tensor Cores "
            "leak. FP32 GEMM, Kalman filters, and other non-DNN "
            "workloads cannot use TC and pay the per-lane FMA cost. "
            "Per-MAC energy is ~20x the TC path - the quantitative "
            "reason Tensor Cores exist."
        ),
    )


# ----------------------------------------------------------------------
# KPU compute tile at 8 nm, 1.5 GHz sustained
# ----------------------------------------------------------------------
#
# Moved to 8 nm to match the Ampere Jetson baseline so direct per-clock
# comparisons are immediately feasible without a process-normalization
# chart. Clock bumped to 1.5 GHz (process-appropriate; 8 nm supports
# higher sustained clocks than 16 nm for this kind of streaming
# datapath).
# ----------------------------------------------------------------------

def build_kpu_tile_building_block(
    clock_ghz: float = 1.5,
) -> BuildingBlock:
    """
    One KPU compute tile: a 2D mesh of 1024 FMAs (32x32), fed by a
    tile-local L1 scratchpad at its edges.

    Process: 8 nm (matched to Ampere). Default clock 1.5 GHz is the
    silicon-capability target; a deployed T128 product inside a 12 W
    envelope may clamp lower (e.g., 1.0 GHz).

    Silicon footprint is dominated by the PE array itself (INT8 MAC
    + 2 operand regs + accumulator + mesh wire + token-match per PE),
    roughly 15 K transistors/PE x 1024 = 15.3 M. With the L1 scratchpad
    (~6 M) and local control (~1.5 M), the whole tile lands near 23 M
    transistors - still about 1/10 the silicon of one Ampere SM.
    """
    PE_ROWS = 32
    PE_COLS = 32
    PE_COUNT = PE_ROWS * PE_COLS  # 1024

    # Per-PE per-clock fabric cost at 8 nm (scaled from the per-MAC
    # view in microarch_accounting.build_kpu_tile_accounting):
    #   FMA       ~0.050  (8 FA @ 8nm x 0.005 pJ + carry-tree)
    #   2 op regs  0.028  (2 bytes x 0.014 pJ/byte)
    #   accumulator 0.015 (pipeline reg, ~32 FF toggles)
    #   forward    0.006  (short wire, no router)
    #   token mtch 0.004  (valid-bit + coord check)
    #   clock      0.008  (H-tree)
    # Total: 0.111 pJ/PE/clock
    pe_per_clock_pj = 0.111 * PE_COUNT  # ~113.7 pJ/clock

    components = [
        EngineComponent(
            name="L1 scratchpad (tile-local, edge-feed)",
            category=StructureCategory.MEMORY,
            count=1,
            size_or_spec="96 KB SRAM, banked",
            transistor_count_m=6.0,
            area_mm2=0.06,   # SRAM density ~100 MT/mm^2
            per_clock_pj=14.0,
            activity_note=(
                f"{2 * PE_ROWS} edge byte-reads/clock (A-edge + B-edge) "
                f"plus periodic output writeback"
            ),
            citation=(
                "96 KB 6T SRAM ~4.5 M + peripheral ~1.5 M = 6 M (sized "
                "up from 64 KB to feed 64 B/clock edge reads). Area at "
                "SRAM density ~100 MT/mm^2 -> 0.06 mm^2. Energy: 64 "
                "reads x 0.15 pJ/B + writeback ~4 pJ = ~14 pJ/clock."
            ),
        ),
        EngineComponent(
            name="2D mesh of FMAs",
            category=StructureCategory.EXECUTE,
            count=PE_COUNT,
            size_or_spec=f"{PE_ROWS}x{PE_COLS} PE grid, 1 INT8 MAC/PE/clock",
            transistor_count_m=15.3,
            area_mm2=0.19,   # dense regular MAC array ~80 MT/mm^2
            per_clock_pj=pe_per_clock_pj,
            activity_note=f"All {PE_COUNT} PEs fire in steady-state wavefront",
            citation=(
                "~15 K transistors/PE (INT8 MAC + 2 op regs + INT32 "
                "accum + mesh wire + token match) x 1024 = 15.3 M. "
                "Area at dense MAC density ~80 MT/mm^2 -> 0.19 mm^2. "
                "Energy: 0.111 pJ/PE/clock x 1024 = 113.7 pJ. See "
                "microarch_accounting.build_kpu_tile_accounting()."
            ),
        ),
        EngineComponent(
            name="Token / coordinate generators (mesh edges)",
            category=StructureCategory.CONTROL,
            count=2 * PE_ROWS,
            size_or_spec=f"{2 * PE_ROWS} injectors at A-edge + B-edge",
            transistor_count_m=0.2,
            area_mm2=0.005,
            per_clock_pj=3.3,
            activity_note="Generate + dispatch one token per edge per clock",
            citation=(
                f"Edge-only control: {2 * PE_ROWS} injectors x ~2 K "
                f"transistors = 0.2 M. Area ~0.005 mm^2. Energy: "
                f"{2 * PE_ROWS} x ~0.05 pJ token generation at 8 nm."
            ),
        ),
        EngineComponent(
            name="Clock tree + leakage (tile-wide)",
            category=StructureCategory.CONTROL,
            count=1,
            size_or_spec="H-tree + rail decoupling",
            transistor_count_m=0.7,
            area_mm2=0.03,
            per_clock_pj=13.0,
            activity_note="Simple H-tree; no OOO machinery, no schedulers",
            citation=(
                "~0.7 M distributed clock buffers + decap (scales with "
                "1024-PE mesh). Area ~0.03 mm^2. ~10% of dynamic fabric "
                "power at 8 nm."
            ),
        ),
    ]

    return BuildingBlock(
        name=f"KPU Compute Tile (32x32 FMA mesh @ {clock_ghz:.2f} GHz)",
        process_nm=8,
        clock_ghz=clock_ghz,
        native_macs_per_clock=PE_COUNT,  # 1024
        native_op_precision="INT8 MAC (1024 PEs)",
        components=components,
        color="#3fc98a",
        notes=(
            "Tile-level budget in steady-state mesh throughput at 8 nm, "
            "1.5 GHz. Silicon footprint ~22 M transistors - roughly "
            "1/10 the size of one Ampere SM. At 1.5 GHz sustained, "
            "~215 mW/tile; a T128 deploying 128 tiles at ~55% utilization "
            "draws ~15 W compute (128 x 215 mW x 0.55 + ~3 W overhead = "
            "~18 W), which exceeds the previous 12 W T128 target. Either "
            "bump the TDP envelope to match or rebalance via clock / "
            "tile count. See cli/check_tdp_feasibility.py."
        ),
    )


# ----------------------------------------------------------------------
# Default SoC compositions (illustrative, used in the HTML page)
# ----------------------------------------------------------------------

def default_soc_compositions() -> List[SocComposition]:
    """Illustrative SoC compositions at two clock domains.

    Silicon-capability clocks (1.5 GHz SM, 1.5 GHz tile) show what the
    raw silicon can deliver; deployed-product clocks (0.65 GHz SM on
    Orin AGX, 1.0 GHz tile on a T128 envelope) show what customers
    actually experience. The gap between the two is pure thermal
    constraint, not architectural limit.
    """
    sm_sol = build_nvidia_sm_building_block(clock_ghz=1.5)
    sm_orin = build_nvidia_sm_building_block(clock_ghz=0.65)
    tile_sol = build_kpu_tile_building_block(clock_ghz=1.5)
    tile_deployed = build_kpu_tile_building_block(clock_ghz=1.0)
    return [
        # Orin AGX as deployed: 16 SMs @ 0.65 GHz, 30 W envelope
        SocComposition(
            block=sm_orin, block_count=16, utilization=0.50,
            overhead_mw=7000.0,
        ),
        # Hypothetical: same SM silicon at capability clock - blows TDP
        SocComposition(
            block=sm_sol, block_count=16, utilization=0.50,
            overhead_mw=7000.0,
        ),
        # KPU T64 at deployed 1.0 GHz
        SocComposition(
            block=tile_deployed, block_count=64, utilization=0.55,
            overhead_mw=1500.0,
        ),
        # KPU T128 at deployed 1.0 GHz
        SocComposition(
            block=tile_deployed, block_count=128, utilization=0.55,
            overhead_mw=2500.0,
        ),
        # KPU T128 at silicon-capability 1.5 GHz
        SocComposition(
            block=tile_sol, block_count=128, utilization=0.55,
            overhead_mw=2500.0,
        ),
        # KPU T256 at deployed 1.0 GHz
        SocComposition(
            block=tile_deployed, block_count=256, utilization=0.55,
            overhead_mw=4500.0,
        ),
    ]


@dataclass
class BuildingBlockReport:
    blocks: List[BuildingBlock]
    socs: List[SocComposition]
    generated_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "blocks": [b.to_dict() for b in self.blocks],
            "socs": [s.to_dict() for s in self.socs],
            "generated_at": self.generated_at,
        }


def build_default_report() -> BuildingBlockReport:
    from datetime import datetime, timezone
    return BuildingBlockReport(
        blocks=[
            build_nvidia_sm_building_block(),
            build_nvidia_sm_cuda_path_building_block(),
            build_kpu_tile_building_block(),
        ],
        socs=default_soc_compositions(),
        generated_at=datetime.now(timezone.utc).isoformat(),
    )


# ----------------------------------------------------------------------
# HTML rendering
# ----------------------------------------------------------------------

_PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.35.2.min.js"

_CATEGORY_COLORS = {
    StructureCategory.FETCH.value: "#7f3b8d",
    StructureCategory.DECODE.value: "#a04bdd",
    StructureCategory.SCHEDULE.value: "#5b8ff9",
    StructureCategory.REGISTER.value: "#d4860b",
    StructureCategory.EXECUTE.value: "#3fc98a",
    StructureCategory.ACCUMULATE.value: "#67d8a6",
    StructureCategory.MEMORY.value: "#e98c3f",
    StructureCategory.INTERCONNECT.value: "#8bbafc",
    StructureCategory.CONTROL.value: "#586374",
    StructureCategory.STATIC.value: "#b0b8c2",
}


def _render_chart_js(report: BuildingBlockReport) -> str:
    bds = [b.to_dict() for b in report.blocks]
    all_cats: List[str] = []
    for b in bds:
        for cat in b["by_category"].keys():
            if cat not in all_cats:
                all_cats.append(cat)

    traces = []
    for cat in all_cats:
        ys = [b["name"] for b in bds]
        xs = [b["by_category"].get(cat, 0.0) for b in bds]
        traces.append({
            "type": "bar", "orientation": "h",
            "name": cat, "y": ys, "x": xs,
            "marker": {"color": _CATEGORY_COLORS.get(cat, "#888")},
            "text": [f"{v:.0f}" if v > 1 else "" for v in xs],
            "textposition": "inside",
        })
    chart_per_clock = {
        "data": traces,
        "layout": {
            "title": "Engine energy per clock by component category (pJ/clock)",
            "xaxis": {"title": "pJ / clock"},
            "yaxis": {"title": "", "automargin": True},
            "barmode": "stack",
            "margin": {"t": 50, "b": 50, "l": 340, "r": 20},
            "legend": {"orientation": "h", "y": -0.2},
        },
    }

    # Transistor footprint chart
    tr_traces = [{
        "type": "bar", "orientation": "h",
        "name": "Transistors (M)",
        "y": [b["name"] for b in bds],
        "x": [b["total_transistor_count_m"] for b in bds],
        "marker": {"color": [b["color"] for b in bds]},
        "text": [f"{b['total_transistor_count_m']:.0f} M"
                 for b in bds],
        "textposition": "outside",
    }]
    chart_transistors = {
        "data": tr_traces,
        "layout": {
            "title": "Silicon footprint (millions of transistors)",
            "xaxis": {"title": "Transistors (millions)"},
            "yaxis": {"title": "", "automargin": True},
            "margin": {"t": 50, "b": 50, "l": 340, "r": 60},
            "showlegend": False,
        },
    }

    # SoC composition bar: total power per SoC
    soc_labels = [f"{s.block.name.split('(')[0].strip()} x{s.block_count} "
                  f"@ U={s.utilization:.0%}" for s in report.socs]
    soc_power = [s.total_power_w for s in report.socs]
    soc_tops = [s.sustained_tops_int8 for s in report.socs]
    chart_soc = {
        "data": [
            {
                "type": "bar", "name": "Power (W)",
                "x": soc_labels, "y": soc_power,
                "marker": {"color": "#586374"},
                "text": [f"{p:.1f} W" for p in soc_power],
                "textposition": "outside",
            },
            {
                "type": "bar", "name": "Sustained TOPS (INT8)",
                "x": soc_labels, "y": soc_tops,
                "yaxis": "y2",
                "marker": {"color": "#3fc98a"},
                "text": [f"{t:.0f}" for t in soc_tops],
                "textposition": "outside",
            },
        ],
        "layout": {
            "title": "SoC composition: power & sustained INT8 TOPS",
            "xaxis": {"title": "Configuration"},
            "yaxis": {"title": "Power (W)", "side": "left"},
            "yaxis2": {"title": "Sustained TOPS",
                       "overlaying": "y", "side": "right"},
            "barmode": "group",
            "margin": {"t": 50, "b": 120, "l": 60, "r": 60},
        },
    }

    payload = {
        "chart_per_clock": chart_per_clock,
        "chart_transistors": chart_transistors,
        "chart_soc": chart_soc,
    }
    return (
        f"const CHARTS = {json.dumps(payload)};\n"
        "for (const [id, spec] of Object.entries(CHARTS)) {\n"
        "  Plotly.newPlot(id, spec.data, spec.layout, "
        "{displayModeBar: false, responsive: true});\n"
        "}\n"
    )


def _render_side_by_side_table(blocks: List[BuildingBlock]) -> str:
    """Structure x building-block matrix so numbers can be read directly."""
    # Collect every unique component name across all blocks.
    names: List[str] = []
    for b in blocks:
        for c in b.components:
            if c.name not in names:
                names.append(c.name)

    col_headers = "".join(
        f'<th colspan="3" style="text-align:center; '
        f'border-bottom:2px solid {b.color};">'
        f'{html.escape(b.name)}<br/>'
        f'<span style="font-weight:normal; font-size:10px; color:#586374;">'
        f'{b.process_nm} nm, {b.clock_ghz:.2f} GHz, '
        f'{b.native_macs_per_clock:,} MACs/clk</span></th>'
        for b in blocks
    )
    sub_headers = "".join(
        '<th class="num">Trans. (M)</th>'
        '<th class="num">Area (mm²)</th>'
        '<th class="num">pJ/clk</th>'
        for _ in blocks
    )

    rows = []
    for name in names:
        cells = []
        for b in blocks:
            match = next((c for c in b.components if c.name == name), None)
            if match is None:
                cells.append('<td class="num">&mdash;</td>'
                             '<td class="num">&mdash;</td>'
                             '<td class="num">&mdash;</td>')
            else:
                cells.append(
                    f'<td class="num">{match.transistor_count_m:.1f}</td>'
                    f'<td class="num">{match.area_mm2:.3f}</td>'
                    f'<td class="num"><strong>{match.per_clock_pj:.1f}</strong></td>'
                )
        rows.append(
            f'<tr><td class="name">{html.escape(name)}</td>{"".join(cells)}</tr>'
        )

    # Totals row
    tot_cells = []
    for b in blocks:
        tot_cells.append(
            f'<td class="num"><strong>{b.total_transistor_count_m:.1f}</strong></td>'
            f'<td class="num"><strong>{b.total_area_mm2:.2f}</strong></td>'
            f'<td class="num"><strong>{b.total_pj_per_clock:.0f}</strong></td>'
        )
    rows.append(
        f'<tr class="total-row"><td><strong>Engine total</strong></td>'
        f'{"".join(tot_cells)}</tr>'
    )
    # Derived pJ/MAC row
    derived_cells = []
    for b in blocks:
        derived_cells.append(
            f'<td class="num" colspan="3"><strong>'
            f'{b.derived_pj_per_mac:.3f} pJ/MAC</strong> '
            f'(native {b.native_macs_per_clock:,})</td>'
        )
    rows.append(
        f'<tr class="total-row"><td><strong>Derived pJ/MAC</strong></td>'
        f'{"".join(derived_cells)}</tr>'
    )
    # Power row
    power_cells = []
    for b in blocks:
        power_cells.append(
            f'<td class="num" colspan="3"><strong>'
            f'{b.power_mw:.0f} mW</strong> '
            f'at {b.clock_ghz:.2f} GHz</td>'
        )
    rows.append(
        f'<tr class="total-row"><td><strong>Engine power</strong></td>'
        f'{"".join(power_cells)}</tr>'
    )

    return (
        '<table class="blocks side-by-side">'
        f'<thead><tr><th rowspan="2">Component</th>{col_headers}</tr>'
        f'<tr>{sub_headers}</tr></thead>'
        f'<tbody>{"".join(rows)}</tbody>'
        '</table>'
    )


def _render_block_table(b: BuildingBlock) -> str:
    rows = []
    tot = b.total_pj_per_clock
    for c in b.components:
        pct = (c.per_clock_pj / tot * 100.0) if tot > 0 else 0.0
        rows.append(
            f'<tr>'
            f'<td class="name">{html.escape(c.name)}</td>'
            f'<td><span class="cat" style="background:'
            f'{_CATEGORY_COLORS.get(c.category.value, "#888")};'
            f'">{html.escape(c.category.value)}</span></td>'
            f'<td class="num">{c.count}</td>'
            f'<td class="spec">{html.escape(c.size_or_spec)}</td>'
            f'<td class="num">{c.transistor_count_m:.1f}</td>'
            f'<td class="num">{c.area_mm2:.3f}</td>'
            f'<td class="num"><strong>{c.per_clock_pj:.1f}</strong></td>'
            f'<td class="num">{pct:.1f}%</td>'
            f'<td class="src">{html.escape(c.activity_note)}<br/>'
            f'<em>{html.escape(c.citation)}</em></td>'
            f'</tr>'
        )
    exec_pct = b.execute_fraction * 100.0
    rows.append(
        f'<tr class="total-row">'
        f'<td colspan="4"><strong>Engine total per clock</strong></td>'
        f'<td class="num"><strong>{b.total_transistor_count_m:.1f}</strong></td>'
        f'<td class="num"><strong>{b.total_area_mm2:.2f}</strong></td>'
        f'<td class="num"><strong>{tot:.1f} pJ/clock</strong></td>'
        f'<td class="num"><strong>100%</strong></td>'
        f'<td class="src">'
        f'Execute fraction: <strong>{exec_pct:.1f}%</strong>. '
        f'Power at {b.clock_ghz:.2f} GHz: <strong>{b.power_mw:.0f} mW</strong>. '
        f'Derived pJ/MAC (cross-check): <strong>'
        f'{b.derived_pj_per_mac:.3f}</strong> (native throughput '
        f'{b.native_macs_per_clock:,} MACs/clock).'
        f'</td>'
        f'</tr>'
    )
    return (
        '<table class="blocks">'
        '<thead><tr>'
        '<th>Component</th><th>Category</th><th>Count</th>'
        '<th>Size / spec</th>'
        '<th>Trans. (M)</th>'
        '<th>Area (mm²)</th>'
        '<th>pJ / clock</th><th>% of engine</th>'
        '<th>Activity assumption / source</th>'
        '</tr></thead>'
        f'<tbody>{"".join(rows)}</tbody>'
        '</table>'
    )


def _render_block_header(b: BuildingBlock) -> str:
    return f"""
<section class="block-header"
         style="border-left: 6px solid {b.color};">
  <h3>{html.escape(b.name)}</h3>
  <div class="meta">
    <strong>Process:</strong> {b.process_nm} nm
    | <strong>Clock:</strong> {b.clock_ghz:.2f} GHz
    | <strong>Native throughput:</strong>
        {b.native_macs_per_clock:,} MACs/clock ({b.native_op_precision})
    | <strong>Silicon:</strong>
        {b.total_transistor_count_m:.0f} M transistors
        ({b.total_area_mm2:.2f} mm²)
    | <strong>Power:</strong> {b.power_mw:.0f} mW
    | <strong>pJ/MAC:</strong> {b.derived_pj_per_mac:.3f}
  </div>
  <p class="block-notes">{html.escape(b.notes)}</p>
</section>
"""


def _render_soc_table(report: BuildingBlockReport) -> str:
    rows = []
    for s in report.socs:
        rows.append(
            f'<tr>'
            f'<td>{html.escape(s.block.name)}</td>'
            f'<td class="num">{s.block_count}</td>'
            f'<td class="num">{s.utilization:.0%}</td>'
            f'<td class="num">{s.block.clock_ghz:.2f} GHz</td>'
            f'<td class="num">{s.compute_power_mw/1000:.2f} W</td>'
            f'<td class="num">{s.overhead_mw/1000:.2f} W</td>'
            f'<td class="num"><strong>{s.total_power_w:.2f} W</strong></td>'
            f'<td class="num">{s.peak_tops_int8:.1f}</td>'
            f'<td class="num">{s.sustained_tops_int8:.1f}</td>'
            f'<td class="num"><strong>{s.sustained_tops_per_watt:.2f}</strong></td>'
            f'</tr>'
        )
    return (
        '<table class="blocks">'
        '<thead><tr>'
        '<th>Building block</th>'
        '<th># blocks</th><th>Utilization</th><th>Clock</th>'
        '<th>Compute power</th><th>SoC overhead</th>'
        '<th>Total SoC power</th>'
        '<th>Peak TOPS</th><th>Sustained TOPS</th>'
        '<th>Sustained TOPS/W</th>'
        '</tr></thead>'
        f'<tbody>{"".join(rows)}</tbody>'
        '</table>'
    )


def render_building_block_page(
    report: BuildingBlockReport, repo_root: Path,
) -> str:
    assets = _load_logo(repo_root)
    header = _render_brand_header(
        assets,
        "Building-block energy accounting (per clock)",
        f"Engine-level budget for SM vs. KPU tile "
        f"| generated {report.generated_at}",
    )
    footer = _render_brand_footer("microarch-model-delivery-plan.md")

    extra_css = """
table.blocks { width: 100%; border-collapse: collapse; background: #fff;
               margin-bottom: 18px; font-size: 13px; }
table.blocks th, table.blocks td { padding: 7px 10px;
                                     border-bottom: 1px solid #e3e6eb;
                                     vertical-align: top; }
table.blocks th { font-size: 11px; text-transform: uppercase;
                  color: #586374; background: #f3f5f8; text-align: left; }
table.blocks td.name { font-weight: 600; color: #0a2540; }
table.blocks td.num { text-align: right; font-variant-numeric: tabular-nums; }
table.blocks td.spec { color: #3a4452; font-size: 12px; }
table.blocks td.src { color: #586374; font-size: 11px; max-width: 320px; }
table.blocks tr.total-row td { background: #eef6ea;
                                border-top: 2px solid #3fc98a;
                                border-bottom: 2px solid #3fc98a;
                                padding: 12px 10px; font-size: 14px; }
table.side-by-side th.num { text-align: right; }
table.side-by-side tbody tr td:not(.name) { text-align: right; }
span.cat { display: inline-block; padding: 2px 8px; border-radius: 3px;
           color: white; font-size: 11px; font-weight: 600; }
section.block-header { background: #fff; padding: 14px 20px;
                        border-radius: 0 6px 6px 0;
                        margin-bottom: 12px;
                        box-shadow: 0 1px 3px rgba(0,0,0,0.04); }
section.block-header h3 { margin: 0 0 6px; color: #0a2540; }
section.block-header .meta { color: #586374; font-size: 13px;
                              margin-bottom: 8px; }
section.block-header p.block-notes { color: #3a4452; font-size: 13px;
                                       margin: 0; }
.chart-section { background: #fff; padding: 18px 22px; border-radius: 6px;
  margin-bottom: 18px; box-shadow: 0 1px 3px rgba(0,0,0,0.04); }
.chart-section h3 { margin: 0 0 4px; color: #0a2540; }
.chart-section .chart-desc { color: #586374; font-size: 13px; margin: 0 0 12px; }
.plot { width: 100%; min-height: 360px; }
section.method-note { background: #eef2f7; padding: 14px 18px;
                      border-left: 3px solid #0a2540; border-radius: 4px;
                      margin: 18px 0; }
a.nav-back { display: inline-block; color: #0a2540; text-decoration: none;
             font-weight: 600; margin-bottom: 10px; }
a.nav-back:hover { text-decoration: underline; }
"""

    # Cross-validation between per-clock and per-MAC views
    from graphs.reporting.microarch_accounting import build_default_report as mar
    mar_report = mar()
    mar_blocks = {b.building_block: b.total_pj_per_mac for b in mar_report.blocks}
    rows_cv = []
    for b in report.blocks:
        per_clock = b.total_pj_per_clock
        derived = b.derived_pj_per_mac
        # Match on substring - names differ slightly
        matched_val = None
        for mname, mval in mar_blocks.items():
            if ("KPU" in b.name and "KPU" in mname):
                matched_val = mval
                break
            # Only the TC-path SM has an independent per-MAC counterpart
            # in microarch_accounting; the CUDA-path SM intentionally
            # stays on the n/a branch.
            if ("Multiprocessor" in b.name and "TC path" in b.name
                    and "Multiprocessor" in mname):
                matched_val = mval
                break
        if matched_val is None:
            rows_cv.append(
                f'<tr><td>{html.escape(b.name)}</td>'
                f'<td class="num">{per_clock:.1f}</td>'
                f'<td class="num">{b.native_macs_per_clock:,}</td>'
                f'<td class="num"><strong>{derived:.3f}</strong></td>'
                f'<td class="num">&mdash;</td>'
                f'<td class="num">n/a</td></tr>'
            )
            continue
        delta = (abs(derived - matched_val) / matched_val * 100.0
                 if matched_val > 0 else 0.0)
        rows_cv.append(
            f'<tr>'
            f'<td>{html.escape(b.name)}</td>'
            f'<td class="num">{per_clock:.1f}</td>'
            f'<td class="num">{b.native_macs_per_clock:,}</td>'
            f'<td class="num"><strong>{derived:.3f}</strong></td>'
            f'<td class="num">{matched_val:.3f}</td>'
            f'<td class="num">{delta:.0f}%</td>'
            f'</tr>'
        )
    cv_table = (
        '<table class="blocks">'
        '<thead><tr>'
        '<th>Engine</th><th>Per-clock total (pJ)</th>'
        '<th>Native MACs/clock</th>'
        '<th>Derived pJ/MAC</th>'
        '<th>Independent pJ/MAC (microarch_accounting)</th>'
        '<th>Delta</th>'
        '</tr></thead>'
        f'<tbody>{"".join(rows_cv)}</tbody>'
        '</table>'
    )

    side_by_side = _render_side_by_side_table(report.blocks)
    blocks_html = "\n".join(
        _render_block_header(b) + _render_block_table(b)
        for b in report.blocks
    )

    # Speed-of-light analysis (silicon-capability view)
    from graphs.reporting.silicon_speed_of_light import (
        DEFAULT_TDP_TARGETS_W,
        build_default_sol_report,
        render_alu_instance_table,
        render_alu_per_mac_table,
        render_gap_to_products_table,
        render_parametric_curve_table,
        render_sol_summary_table,
        render_tdp_sweep_table,
        render_tradeoff_chart_js,
    )
    sol = build_default_sol_report()
    sol_instance_table = render_alu_instance_table(sol.alus)
    sol_per_mac_table = render_alu_per_mac_table(sol.alus)
    sol_curve_table = render_parametric_curve_table(sol.parametric_curve)
    sol_summary_table = render_sol_summary_table(
        sol.analyses, sol.die_area_mm2
    )
    sol_tdp_table = render_tdp_sweep_table(
        sol.analyses, DEFAULT_TDP_TARGETS_W
    )
    sol_gap_table = render_gap_to_products_table(
        sol.products, sol.analyses
    )
    sol_tradeoff_js = render_tradeoff_chart_js(
        sol.parametric_curve, sol.alus,
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>Building-block energy accounting</title>
  <script src="{_PLOTLY_CDN}"></script>
  <style>{_CSS}
{extra_css}
  </style>
</head>
<body>
{header}
<main>
  <p><a class="nav-back" href="index.html">&larr; Back to index</a></p>
  <section class="page-header">
    <h2>Engine-level energy budget: SM vs. KPU tile</h2>
    <div class="meta">Per-clock energy of every major component as it
      is deployed on the SoC - all reported at 8 nm to match the
      Jetson Ampere baseline so direct comparisons do not need a
      process-normalization chart. For each component we report
      <strong>transistor count, silicon area (mm²), and energy per
      clock</strong> so circuit-designer intuition can audit the
      numbers. Complements the
      <a href="microarch_accounting.html">per-MAC view</a>; the two
      reconcile via <code>pJ/MAC = pJ_per_clock /
      MACs_per_clock</code>.</div>
  </section>

  <section class="method-note">
    <h3 style="margin-top:0;">Quantitative component tables</h3>
    <p>One table per building block. Columns are
      <strong>Transistors (M)</strong>,
      <strong>Area (mm²)</strong>, and
      <strong>pJ / clock</strong> - the three independent grounding
      estimates. Energy totals are <em>not</em> normalized to pJ/MAC
      here; this is the raw silicon budget the SoC must supply every
      cycle. The bar graphs below visualize the same data.</p>
  </section>

  {blocks_html}

  <section class="chart-section">
    <h3>Bar-graph view: engine energy per clock by category</h3>
    <p class="chart-desc">Absolute pJ/clock at 8 nm and steady-state
      clock. Stacked by component category. Same data as the per-block
      tables above.</p>
    <div id="chart_per_clock" class="plot"></div>
  </section>

  <section class="chart-section">
    <h3>Bar-graph view: silicon footprint</h3>
    <p class="chart-desc">Estimated transistor count for each building
      block. Energy should scale roughly with active silicon; large
      deviations indicate either an architectural advantage or an
      estimation error.</p>
    <div id="chart_transistors" class="plot" style="min-height:260px;"></div>
  </section>

  <section class="chart-section">
    <h3>Side-by-side component comparison</h3>
    <p class="chart-desc">Same data as the per-block tables, laid out
      with one row per component and two columns per building block
      (Trans. M / pJ/clk) so differences can be read column-to-column.</p>
    {side_by_side}
  </section>

  <section class="method-note" style="background:#fff8e6;
    border-left-color:#d4860b;">
    <h3 style="margin-top:0;">Silicon speed-of-light</h3>
    <p>The <em>silicon ceiling</em> is what a process node can do if
      you fill a die with nothing but bare ALUs and clock them at the
      silicon's capability frequency. It is the upper bound against
      which every shipping product must be measured. Incumbents ship
      well below their own silicon ceiling because TDP envelopes,
      packaging, and architectural scaffolding all consume power that
      bare ALUs do not.</p>
    <p>The ALU design space has three orthogonal axes that each
      touch a different ceiling:</p>
    <ul>
      <li><strong>ALU width W</strong> (MACs per instance per clock):
        W=1 bare FMA (KPU PE), W=4 small dot product (Volta/Turing
        TC), W=16 medium (Ampere TC), W=32+ wide (Hopper TC).</li>
      <li><strong>Accumulator mode</strong>: lossless (INT32 accum
        from INT8), tree-rounded (FP16 at each reduction level),
        mixed-precision (FP16 operand → FP32 accum), aggressive-
        truncate. Accuracy ceiling drops as W and rounding increase.</li>
      <li><strong>Reuse topology</strong>: isolated (2 B/MAC),
        intra-ALU broadcast (2/W B/MAC), 2D-mesh streaming
        (2/N B/MAC), systolic-stationary (1/N B/MAC). This axis
        dominates the operand-bandwidth ceiling.</li>
    </ul>
    <p>Three metrics - density, energy, bandwidth - each have their
      own ceiling curve vs W, and real products sit at different
      positions on each curve.</p>

    <h4>Parametric cost curve (INT8 at 8 nm, intra-ALU broadcast)</h4>
    <p class="chart-desc">Analytical first-principles cost: W INT8
      multipliers into a log2(W)-deep reduction tree with INT32
      accumulator. Shows the slow rise of transistors/MAC as tree
      bit-width growth outpaces multiplier sharing past W~8.</p>
    {sol_curve_table}

    <h4>Real-product ALU archetypes (per-instance)</h4>
    <p class="chart-desc">The physical unit of replication for each
      architecture. Numbers are for the whole ALU as one instance -
      per-MAC values derive by dividing by W.</p>
    {sol_instance_table}

    <h4>Real-product ALU archetypes (per-MAC)</h4>
    <p class="chart-desc">Same archetypes, normalized per MAC to
      compare ceilings directly. TOPS/W = ops / pJ_per_MAC is the
      clock-independent silicon ceiling.</p>
    {sol_per_mac_table}

    <h4>Trade-off curve with real products overlaid</h4>
    <p class="chart-desc">Three views of the trade-off space. The
      blue curve is the parametric first-principles model at each W;
      the orange diamonds are real products. The gap between a
      product and the curve shows where that product's architecture
      is above or below the first-principles expectation.</p>
    <div id="chart_sol_trans_per_mac" class="plot" style="min-height:340px;"></div>
    <div id="chart_sol_pj_per_mac" class="plot" style="min-height:340px;"></div>
    <div id="chart_sol_bytes_per_mac" class="plot" style="min-height:340px;"></div>

    <h4>On a 250 mm² die at silicon-capability clock (1.5 GHz)</h4>
    {sol_summary_table}

    <h4>TDP-constrained peak performance on 250 mm² (* = silicon-limited)</h4>
    {sol_tdp_table}

    <h4>Gap to shipping products</h4>
    <p class="chart-desc" style="margin-bottom:8px;">For each product,
      the "Actual / SoL" ratio is the shipping TOPS divided by the
      silicon-ceiling TOPS at the product's own TDP. A 30% ratio means
      3x of headroom is available before the silicon ceiling binds.</p>
    {sol_gap_table}
  </section>

  <section class="method-note">
    <h3 style="margin-top:0;">SoC composition (illustrative)</h3>
    <p>Aggregating building blocks into a SoC: <strong>total power =
      count x utilization x power-per-block + SoC overhead</strong>.
      The overhead term covers L2/L3 caches, memory controller,
      NVLink/PCIe, display, and other non-compute IP. Utilization is
      the fraction of blocks active per clock on realistic workloads.</p>
    {_render_soc_table(report)}
    <div id="chart_soc" class="plot" style="margin-top: 12px;"></div>
  </section>

  <section class="method-note">
    <h3 style="margin-top:0;">Cross-validation with per-MAC view</h3>
    <p>The derived pJ/MAC from this per-clock view must agree with the
      independent per-MAC accounting in
      <code>microarch_accounting.py</code>. Agreement validates that
      both analytical models converge on the same silicon cost.</p>
    {cv_table}
  </section>

  <section class="method-note">
    <strong>Which view for which question?</strong>
    <ul>
      <li><strong>pJ/MAC view</strong> (microarch_accounting.py):
        performance under a fixed power budget. "At 12 W, how much
        compute will this engine deliver on workload X?"</li>
      <li><strong>pJ/clock view</strong> (this page): SoC composition
        and super-cluster design. "If I deploy N engines at F GHz with
        U utilization, what is the total power and sustained TOPS?"</li>
      <li><strong>Transistor count</strong>: circuit-designer sanity
        check. Silicon footprint should roughly track energy per clock;
        large departures indicate either an architectural advantage
        or an estimation error.</li>
    </ul>
  </section>
</main>
{footer}
<script>
{_render_chart_js(report)}
{sol_tradeoff_js}
</script>
</body>
</html>
"""
