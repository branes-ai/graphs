"""
Silicon composition analysis: hierarchical building blocks (ALU -> PE
-> Tile -> Cluster -> SoC) and the way silicon efficiency decays as
architectural scaffolding is added.

This is the SoC-floorplanning companion to silicon_speed_of_light.py.
Where the SoL module reports the irreducible ALU ceiling, this
module shows what happens once you wrap the ALU in operand regs,
schedulers, caches, fabrics, and memory controllers.

For each architecture we enumerate the canonical composition
hierarchy:

  ALU      = bare multiply-add datapath
  PE       = ALU + local operand regs + accumulator + intra-PE routing
  Tile     = PE array + L1 scratchpad + tile controller
  Cluster  = Tile array + L2 interconnect
  SoC      = Cluster(s) + L3 + memory controllers + host I/O

At each level we report:

  transistor_count_m  (millions of transistors per instance)
  area_mm2            (derived at canonical process density)
  macs_per_clock      (native INT8 MAC throughput per instance)
  pj_per_clock        (dynamic energy per instance per clock)
  tops_per_watt       (silicon efficiency = ops/pJ, clock-
                       independent)
  active_fraction     (fraction of transistors contributing MACs)

The point: silicon efficiency DROPS monotonically from ALU to SoC as
each composition level adds silicon that serves coordination rather
than compute. This drop quantifies the "scaffolding tax" for each
architecture, and is the primary input to SoC-specialization
planning (picking how many CPU cores vs DSP lanes vs NPU tiles to
match a workload's op mix).
"""
from __future__ import annotations

import html
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from graphs.reporting.silicon_speed_of_light import (
    process_density_mt_per_mm2,
)


# ======================================================================
# Enums
# ======================================================================


class BlockLevel(Enum):
    ALU = "ALU"
    PE = "PE"
    TILE = "Tile"
    CLUSTER = "Cluster"
    SOC = "SoC"


class ArchitectureCategory(Enum):
    CPU = "CPU"
    DSP = "DSP"
    GPU = "GPU"
    NPU = "NPU"


# ======================================================================
# Data model
# ======================================================================


@dataclass
class CompositionBlock:
    """
    One level in a composition hierarchy.

    `macs_per_clock` is the INT8-equivalent MAC throughput per single
    INSTANCE at the native workload (e.g., for a GPU SM, this is the
    Tensor-Core-path INT8 throughput at full TC activity; for a CPU
    core this is the SIMD INT8 dot-product rate).

    `active_fraction` is an estimate of what fraction of this block's
    transistors actually participates in MAC computation on a dense
    workload. CPUs have low active fractions (lots of control +
    out-of-order logic); systolic NPUs are closer to 1.0.
    """
    name: str
    level: BlockLevel
    architecture: ArchitectureCategory
    process_nm: int
    children_per_instance: int     # e.g., SM has 4 sub-partitions
    child_name: str = ""           # name of child block (""=no child)
    transistor_count_m: float = 0.0
    macs_per_clock: int = 0
    pj_per_clock: float = 0.0
    active_fraction: float = 1.0   # portion contributing to MACs
    clock_ghz: float = 1.0
    citation: str = ""

    @property
    def area_mm2(self) -> float:
        density = process_density_mt_per_mm2(self.process_nm)
        return self.transistor_count_m / density

    @property
    def ops_per_mac(self) -> int:
        return 2  # 1 FMA = 1 multiply + 1 add

    @property
    def tops_at_clock(self) -> float:
        return (self.macs_per_clock * self.ops_per_mac
                * self.clock_ghz / 1000.0)

    @property
    def power_mw(self) -> float:
        return self.pj_per_clock * self.clock_ghz

    @property
    def tops_per_watt(self) -> float:
        """Silicon efficiency at this composition level.
        Clock-independent because numerator and denominator both
        scale with clock."""
        if self.pj_per_clock <= 0:
            return 0.0
        return self.ops_per_mac * self.macs_per_clock / self.pj_per_clock

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "level": self.level.value,
            "architecture": self.architecture.value,
            "process_nm": self.process_nm,
            "children_per_instance": self.children_per_instance,
            "child_name": self.child_name,
            "transistor_count_m": self.transistor_count_m,
            "area_mm2": self.area_mm2,
            "macs_per_clock": self.macs_per_clock,
            "pj_per_clock": self.pj_per_clock,
            "active_fraction": self.active_fraction,
            "clock_ghz": self.clock_ghz,
            "tops_at_clock": self.tops_at_clock,
            "power_mw": self.power_mw,
            "tops_per_watt": self.tops_per_watt,
            "citation": self.citation,
        }


@dataclass
class ArchitectureHierarchy:
    """Canonical composition hierarchy for one architecture family."""
    name: str
    architecture: ArchitectureCategory
    process_nm: int
    blocks: List[CompositionBlock] = field(default_factory=list)
    notes: str = ""

    def by_level(self, level: BlockLevel) -> Optional[CompositionBlock]:
        for b in self.blocks:
            if b.level is level:
                return b
        return None

    @property
    def soc(self) -> Optional[CompositionBlock]:
        return self.by_level(BlockLevel.SOC)

    @property
    def alu(self) -> Optional[CompositionBlock]:
        return self.by_level(BlockLevel.ALU)

    @property
    def efficiency_decay(self) -> float:
        """Ratio alu_tops_per_watt / soc_tops_per_watt. Large values
        indicate most silicon goes to non-compute scaffolding."""
        alu, soc = self.alu, self.soc
        if alu is None or soc is None or soc.tops_per_watt <= 0:
            return 0.0
        return alu.tops_per_watt / soc.tops_per_watt

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "architecture": self.architecture.value,
            "process_nm": self.process_nm,
            "blocks": [b.to_dict() for b in self.blocks],
            "efficiency_decay": self.efficiency_decay,
            "notes": self.notes,
        }


# ======================================================================
# Canonical hierarchies
# ======================================================================
#
# Numbers below are hand-tuned. Real sources are cited per entry.
# Where the underlying data comes from this repo's own accounting
# modules (microarch_accounting / building_block_energy), those
# modules are named as the primary citation. External numbers are
# cited to whitepapers or published die-shot analyses.
# ======================================================================


def build_nvidia_ampere_orin_hierarchy() -> ArchitectureHierarchy:
    """NVIDIA Ampere (Orin AGX class) GPU composition at 8 nm.

    At INT8 HMMA workload: each SM has 4 TCs x 1024 MAC/clk = 4096
    MAC/clk. Orin AGX has 16 SMs grouped into 2 GPCs.
    """
    return ArchitectureHierarchy(
        name="NVIDIA Ampere GA10B (Jetson AGX Orin)",
        architecture=ArchitectureCategory.GPU,
        process_nm=8,
        notes=(
            "INT8 Tensor-Core path. CUDA-core (FP32) path has ~10x "
            "lower TOPS/W at the SM level - covered separately in "
            "building_block_energy.py."
        ),
        blocks=[
            CompositionBlock(
                name="Ampere TC lane (W=16 INT8)",
                level=BlockLevel.ALU,
                architecture=ArchitectureCategory.GPU,
                process_nm=8,
                children_per_instance=1,
                transistor_count_m=0.110,       # 110 K trans
                macs_per_clock=16,
                pj_per_clock=1.28,
                active_fraction=1.0,
                clock_ghz=1.5,
                citation=(
                    "silicon_speed_of_light 'Ampere TC lane': "
                    "110 K trans, 1.28 pJ/clk, W=16 INT8 MAC."
                ),
            ),
            CompositionBlock(
                name="Warp sub-partition (W=16 TC + 32 CUDA lanes idle)",
                level=BlockLevel.PE,
                architecture=ArchitectureCategory.GPU,
                process_nm=8,
                children_per_instance=16,        # 16 TC lanes per TC
                child_name="Ampere TC lane",
                transistor_count_m=30.0,         # 1 TC + 32 CUDA idle + RF bank + operand collector + scheduler
                macs_per_clock=256,               # 1 TC * 16 lanes * 16 MAC
                pj_per_clock=220.0,               # 1 TC (1.28*16=20.5) + scaffolding (~200)
                active_fraction=0.20,
                clock_ghz=1.5,
                citation=(
                    "1 Tensor Core (the 16 lanes above) + 32 idle "
                    "CUDA lanes + RF bank (5 M trans) + operand "
                    "collector + 1 warp scheduler. Hand-tuned from "
                    "microarch_accounting + GA10x die-shot analyses."
                ),
            ),
            CompositionBlock(
                name="Streaming Multiprocessor (TC path)",
                level=BlockLevel.TILE,
                architecture=ArchitectureCategory.GPU,
                process_nm=8,
                children_per_instance=4,         # 4 sub-partitions
                child_name="Warp sub-partition",
                transistor_count_m=240.0,
                macs_per_clock=4096,
                pj_per_clock=1025.0,
                active_fraction=0.50,
                clock_ghz=1.5,
                citation=(
                    "building_block_energy.build_nvidia_sm_building_"
                    "block(): 240 M trans, 1025 pJ/clk, 4096 MAC/clk."
                ),
            ),
            CompositionBlock(
                name="GPC (Graphics Processing Cluster)",
                level=BlockLevel.CLUSTER,
                architecture=ArchitectureCategory.GPU,
                process_nm=8,
                children_per_instance=8,         # 8 SMs per GPC
                child_name="Streaming Multiprocessor",
                transistor_count_m=2100.0,       # 8 SMs + raster + polymorph
                macs_per_clock=32768,
                pj_per_clock=9500.0,              # 8 SMs + fabric overhead
                active_fraction=0.55,
                clock_ghz=1.5,
                citation=(
                    "8 SMs + raster engine + polymorph engine + GPC-"
                    "level interconnect. Hand-tuned from GA102 die "
                    "analysis (~28 B trans / 84 SMs scales to Orin "
                    "16 SMs in 2 GPCs)."
                ),
            ),
            CompositionBlock(
                name="Orin AGX SoC (GPU + CPU + system)",
                level=BlockLevel.SOC,
                architecture=ArchitectureCategory.GPU,
                process_nm=8,
                children_per_instance=2,          # 2 GPCs
                child_name="GPC",
                transistor_count_m=14400.0,       # 250 mm^2 * 80 MT/mm^2
                macs_per_clock=65536,
                pj_per_clock=91000.0,             # 30 W at 0.65 GHz -> 46 nJ/clk; scale to 1.5 GHz
                active_fraction=0.15,
                clock_ghz=0.65,                   # MAXN sustained (TDP-limited)
                citation=(
                    "Orin AGX full SoC: 2 GPCs (16 SMs) + 12 ARM "
                    "cores + L2 cache + memory controllers + DLA + "
                    "PVA + display + video engines. Peak 137 TOPS "
                    "dense INT8 at MAXN (60 W, 0.65 GHz sustained "
                    "gives the published throughput; at 1.5 GHz "
                    "silicon ceiling the SoC would blow past 150 W)."
                ),
            ),
        ],
    )


def build_kpu_t128_hierarchy() -> ArchitectureHierarchy:
    """KPU T128 NPU composition at 8 nm.

    Direct mapping of the KPU design: bare FMA -> PE (with token/
    mesh scaffolding) -> 32x32 Tile -> Cluster of 32 tiles ->
    T128 SoC (128 tiles = 4 clusters)."""
    return ArchitectureHierarchy(
        name="KPU T128 (Knowledge Processing Unit, hypothetical 8 nm SoC)",
        architecture=ArchitectureCategory.NPU,
        process_nm=8,
        notes=(
            "Domain-flow NPU. Compute lives in the 2D mesh of FMAs; "
            "scaffolding is minimal (no instruction fetch/decode, "
            "no scheduler, no cache coherence)."
        ),
        blocks=[
            CompositionBlock(
                name="KPU INT8 FMA",
                level=BlockLevel.ALU,
                architecture=ArchitectureCategory.NPU,
                process_nm=8,
                children_per_instance=1,
                transistor_count_m=0.007,
                macs_per_clock=1,
                pj_per_clock=0.050,
                active_fraction=1.0,
                clock_ghz=1.5,
                citation="silicon_speed_of_light 'KPU INT8 FMA'.",
            ),
            CompositionBlock(
                name="KPU PE (FMA + regs + accum + mesh mux + token)",
                level=BlockLevel.PE,
                architecture=ArchitectureCategory.NPU,
                process_nm=8,
                children_per_instance=1,
                child_name="KPU INT8 FMA",
                transistor_count_m=0.015,
                macs_per_clock=1,
                pj_per_clock=0.111,
                active_fraction=0.47,           # 7 K / 15 K
                clock_ghz=1.5,
                citation=(
                    "microarch_accounting per-PE breakdown: 7 K FMA "
                    "+ 2 K op regs + 0.4 K accum + 0.5 K mesh mux "
                    "+ 0.2 K token match + clock latch = 15 K."
                ),
            ),
            CompositionBlock(
                name="KPU Compute Tile (32x32 mesh)",
                level=BlockLevel.TILE,
                architecture=ArchitectureCategory.NPU,
                process_nm=8,
                children_per_instance=1024,
                child_name="KPU PE",
                transistor_count_m=22.0,
                macs_per_clock=1024,
                pj_per_clock=144.0,
                active_fraction=0.70,            # 15.3 M mesh / 22 M total
                clock_ghz=1.5,
                citation=(
                    "building_block_energy.build_kpu_tile_building_"
                    "block(): 22 M trans, 144 pJ/clk, 1024 MAC/clk."
                ),
            ),
            CompositionBlock(
                name="KPU tile cluster (32 tiles + L2 scratchpad)",
                level=BlockLevel.CLUSTER,
                architecture=ArchitectureCategory.NPU,
                process_nm=8,
                children_per_instance=32,
                child_name="KPU Compute Tile",
                transistor_count_m=720.0,        # 32 tiles (704 M) + 10 M L2 + 6 M DMA
                macs_per_clock=32768,
                pj_per_clock=4650.0,              # 32*144 + L2 + DMA overhead
                active_fraction=0.68,
                clock_ghz=1.5,
                citation=(
                    "Hypothetical cluster: 32 tiles + shared L2 "
                    "scratchpad (256 KB) + cluster DMA + mesh "
                    "interconnect. ~95% tile + 5% cluster overhead."
                ),
            ),
            CompositionBlock(
                name="KPU T128 SoC (4 clusters + L3 + memory)",
                level=BlockLevel.SOC,
                architecture=ArchitectureCategory.NPU,
                process_nm=8,
                children_per_instance=4,
                child_name="KPU tile cluster",
                transistor_count_m=6400.0,       # 4 clusters (2880) + 3.5 B host/mem/fabric overhead
                macs_per_clock=131072,
                pj_per_clock=21000.0,
                active_fraction=0.45,
                clock_ghz=1.0,                   # deployed sustained
                citation=(
                    "T128 full SoC: 4 clusters = 128 tiles + "
                    "distributed L3 + LPDDR5 controller + host PCIe "
                    "+ clock trees. Target 12-18 W TDP at 1.0 GHz."
                ),
            ),
        ],
    )


def build_arm_a78_cpu_hierarchy() -> ArchitectureHierarchy:
    """ARM Cortex-A78 CPU cluster composition at 8 nm.

    MAC throughput comes from the NEON SIMD unit (128-bit vectors).
    A78 supports SDOT (signed INT8 dot product) which gives 32 INT8
    MACs per clock per core on a dual-issue pipe. At the ALU level
    we model one NEON MAC lane; at the core level we model the
    full SDOT throughput."""
    return ArchitectureHierarchy(
        name="ARM Cortex-A78 quad-core cluster",
        architecture=ArchitectureCategory.CPU,
        process_nm=8,
        notes=(
            "Scalar control + OOO + cache hierarchy dominate "
            "transistor budget; dense-MAC compute is a thin slice. "
            "INT8 SDOT throughput assumed: 32 INT8 MAC/clk/core "
            "(dual-issue NEON with SDOT)."
        ),
        blocks=[
            CompositionBlock(
                name="NEON SDOT lane (1/32 of SDOT unit)",
                level=BlockLevel.ALU,
                architecture=ArchitectureCategory.CPU,
                process_nm=8,
                children_per_instance=1,
                transistor_count_m=0.010,
                macs_per_clock=1,
                pj_per_clock=0.070,
                active_fraction=1.0,
                clock_ghz=3.0,
                citation=(
                    "One INT8 MAC lane within the A78 NEON SDOT "
                    "unit. ~10 K trans per lane (INT8 mult + 32-bit "
                    "add share). Hand-tuned from ARM A78 PPA data."
                ),
            ),
            CompositionBlock(
                name="NEON SDOT unit (32 INT8 MAC lanes)",
                level=BlockLevel.PE,
                architecture=ArchitectureCategory.CPU,
                process_nm=8,
                children_per_instance=32,
                child_name="NEON SDOT lane",
                transistor_count_m=0.60,          # 32*10 K + routing + FP fallback
                macs_per_clock=32,
                pj_per_clock=3.0,
                active_fraction=0.53,
                clock_ghz=3.0,
                citation=(
                    "A78 NEON 128-bit vector unit with SDOT. One "
                    "SDOT inst produces 32 INT8 MACs/clock across "
                    "the vector register pair."
                ),
            ),
            CompositionBlock(
                name="A78 core (NEON + scalar + L1 + OOO + branch)",
                level=BlockLevel.TILE,
                architecture=ArchitectureCategory.CPU,
                process_nm=8,
                children_per_instance=1,
                child_name="NEON SDOT unit",
                transistor_count_m=130.0,         # A78 core ~130 M trans
                macs_per_clock=32,                 # same as NEON unit
                pj_per_clock=330.0,                # ~1 W at 3 GHz
                active_fraction=0.005,             # 0.6 M / 130 M
                clock_ghz=3.0,
                citation=(
                    "A78 core area ~1.5 mm^2 at 7 nm -> ~2.5 mm^2 "
                    "at 8 nm, ~130 M transistors. Most silicon is "
                    "decode/rename/scheduler/L1$/branch - NEON is "
                    "under 0.5% of the core transistor budget."
                ),
            ),
            CompositionBlock(
                name="A78 DSU-110 cluster (4 cores + L3)",
                level=BlockLevel.CLUSTER,
                architecture=ArchitectureCategory.CPU,
                process_nm=8,
                children_per_instance=4,
                child_name="A78 core",
                transistor_count_m=650.0,          # 4*130 + 100 M L3 + 30 M DSU
                macs_per_clock=128,                 # 4 cores * 32 MAC
                pj_per_clock=1550.0,
                active_fraction=0.004,
                clock_ghz=3.0,
                citation=(
                    "DSU-110 cluster: 4 A78 cores + 4 MB shared L3 "
                    "+ snoop filter + coherent fabric."
                ),
            ),
            CompositionBlock(
                name="Embedded CPU SoC (1 cluster + fabric + I/O)",
                level=BlockLevel.SOC,
                architecture=ArchitectureCategory.CPU,
                process_nm=8,
                children_per_instance=1,
                child_name="A78 cluster",
                transistor_count_m=1200.0,         # 650 cluster + 550 M system
                macs_per_clock=128,
                pj_per_clock=2800.0,                # 8-9 W typical
                active_fraction=0.002,
                clock_ghz=3.0,
                citation=(
                    "Typical single-cluster A78 SoC: cluster + "
                    "system fabric + DDR controller + USB/PCIe/"
                    "Ethernet + boot ROM + PMIC interface."
                ),
            ),
        ],
    )


def build_qualcomm_hexagon_hierarchy() -> ArchitectureHierarchy:
    """Qualcomm Hexagon DSP with HVX (1024-bit vector) at 8 nm.

    HVX is a 1024-bit vector unit; an HVX 'VMAC' INT8 inst produces
    128 INT8 MAC/clk. Each Hexagon core has 4 HW contexts each able
    to issue HVX; in dense compute one context's HVX is the bottle-
    neck, so we model per-context (128 MAC/clk)."""
    return ArchitectureHierarchy(
        name="Qualcomm Hexagon DSP with HVX",
        architecture=ArchitectureCategory.DSP,
        process_nm=8,
        notes=(
            "DSP has wider vectors than CPU SIMD but still carries "
            "scalar + VLIW control overhead. Better than CPU on "
            "dense MACs, worse than NPU on scaffolding."
        ),
        blocks=[
            CompositionBlock(
                name="HVX INT8 MAC lane",
                level=BlockLevel.ALU,
                architecture=ArchitectureCategory.DSP,
                process_nm=8,
                children_per_instance=1,
                transistor_count_m=0.009,
                macs_per_clock=1,
                pj_per_clock=0.060,
                active_fraction=1.0,
                clock_ghz=1.0,
                citation=(
                    "One INT8 MAC slice of the 1024-bit HVX vector "
                    "unit (128 lanes total per VMAC inst)."
                ),
            ),
            CompositionBlock(
                name="HVX vector unit (1024-bit, 128 INT8 lanes)",
                level=BlockLevel.PE,
                architecture=ArchitectureCategory.DSP,
                process_nm=8,
                children_per_instance=128,
                child_name="HVX INT8 MAC lane",
                transistor_count_m=2.5,
                macs_per_clock=128,
                pj_per_clock=11.0,
                active_fraction=0.46,
                clock_ghz=1.0,
                citation=(
                    "HVX 1024-bit vector unit with VMAC. Per-cycle "
                    "throughput 128 INT8 MAC (packed) or 64 INT16."
                ),
            ),
            CompositionBlock(
                name="Hexagon core (4 HW contexts, scalar + HVX + VMEM)",
                level=BlockLevel.TILE,
                architecture=ArchitectureCategory.DSP,
                process_nm=8,
                children_per_instance=1,           # 1 HVX unit per core
                child_name="HVX vector unit",
                transistor_count_m=30.0,           # ~3 mm^2 at 8 nm
                macs_per_clock=128,
                pj_per_clock=80.0,
                active_fraction=0.08,
                clock_ghz=1.0,
                citation=(
                    "Hexagon core: 4 HW contexts + 4-wide VLIW "
                    "scalar + HVX + VMEM (vector memory) + L1. "
                    "HVX is the compute bottleneck for dense MACs."
                ),
            ),
            CompositionBlock(
                name="Hexagon cluster (4 cores + L2 + TCM)",
                level=BlockLevel.CLUSTER,
                architecture=ArchitectureCategory.DSP,
                process_nm=8,
                children_per_instance=4,
                child_name="Hexagon core",
                transistor_count_m=170.0,
                macs_per_clock=512,
                pj_per_clock=380.0,
                active_fraction=0.06,
                clock_ghz=1.0,
                citation=(
                    "4-core Hexagon cluster with shared L2 and "
                    "tightly-coupled memory (TCM). Matches typical "
                    "smartphone AI-DSP sub-system."
                ),
            ),
            CompositionBlock(
                name="Sensing / always-on SoC (Hexagon + memory + I/O)",
                level=BlockLevel.SOC,
                architecture=ArchitectureCategory.DSP,
                process_nm=8,
                children_per_instance=1,
                child_name="Hexagon cluster",
                transistor_count_m=480.0,
                macs_per_clock=512,
                pj_per_clock=950.0,
                active_fraction=0.021,
                clock_ghz=1.0,
                citation=(
                    "Always-on sensing SoC with Hexagon compute + "
                    "DDR + sensor I/O + audio/video codec IP. "
                    "Typical 2-3 W TDP."
                ),
            ),
        ],
    )


def default_hierarchies() -> List[ArchitectureHierarchy]:
    return [
        build_nvidia_ampere_orin_hierarchy(),
        build_kpu_t128_hierarchy(),
        build_arm_a78_cpu_hierarchy(),
        build_qualcomm_hexagon_hierarchy(),
    ]


@dataclass
class CompositionReport:
    hierarchies: List[ArchitectureHierarchy]
    generated_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hierarchies": [h.to_dict() for h in self.hierarchies],
            "generated_at": self.generated_at,
        }


def build_default_composition_report() -> CompositionReport:
    from datetime import datetime, timezone
    return CompositionReport(
        hierarchies=default_hierarchies(),
        generated_at=datetime.now(timezone.utc).isoformat(),
    )


# ======================================================================
# HTML rendering
# ======================================================================


_PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.35.2.min.js"

_LEVEL_COLORS = {
    BlockLevel.ALU.value: "#3fc98a",
    BlockLevel.PE.value: "#5b8ff9",
    BlockLevel.TILE.value: "#d4860b",
    BlockLevel.CLUSTER.value: "#7f3b8d",
    BlockLevel.SOC.value: "#586374",
}


def _render_hierarchy_table(h: ArchitectureHierarchy) -> str:
    rows = []
    alu_tops_per_watt = h.alu.tops_per_watt if h.alu else 0.0
    for b in h.blocks:
        ratio_vs_alu = (
            b.tops_per_watt / alu_tops_per_watt if alu_tops_per_watt > 0 else 0.0
        )
        color = _LEVEL_COLORS.get(b.level.value, "#888")
        rows.append(
            f'<tr>'
            f'<td><span class="cat" style="background:{color};">'
            f'{html.escape(b.level.value)}</span></td>'
            f'<td class="name">{html.escape(b.name)}</td>'
            f'<td class="num">{b.children_per_instance}</td>'
            f'<td class="num">{b.transistor_count_m:.2f}</td>'
            f'<td class="num">{b.area_mm2:.3f}</td>'
            f'<td class="num">{b.macs_per_clock:,}</td>'
            f'<td class="num">{b.pj_per_clock:.2f}</td>'
            f'<td class="num"><strong>{b.tops_per_watt:.2f}</strong></td>'
            f'<td class="num">{b.active_fraction*100:.1f}%</td>'
            f'<td class="num">{ratio_vs_alu*100:.1f}%</td>'
            f'<td class="src"><em>{html.escape(b.citation)}</em></td>'
            f'</tr>'
        )
    return (
        '<table class="blocks">'
        '<thead><tr>'
        '<th>Level</th>'
        '<th>Block</th>'
        '<th># children / instance</th>'
        '<th>Trans (M)</th>'
        '<th>Area (mm²)</th>'
        '<th>MACs / clock</th>'
        '<th>pJ / clock</th>'
        '<th>TOPS / W</th>'
        '<th>Active %</th>'
        '<th>Efficiency vs ALU</th>'
        '<th>Derivation</th>'
        '</tr></thead>'
        f'<tbody>{"".join(rows)}</tbody>'
        '</table>'
    )


def _render_efficiency_chart_js(report: CompositionReport) -> str:
    """Bar chart comparing TOPS/W at each level across architectures."""
    levels = [bl.value for bl in BlockLevel]
    traces = []
    for h in report.hierarchies:
        ys = []
        xs = []
        for lvl in BlockLevel:
            b = h.by_level(lvl)
            if b is not None:
                xs.append(lvl.value)
                ys.append(b.tops_per_watt)
        traces.append({
            "type": "bar",
            "name": html.escape(h.name),
            "x": xs,
            "y": ys,
            "text": [f"{v:.1f}" for v in ys],
            "textposition": "outside",
        })
    chart = {
        "data": traces,
        "layout": {
            "title": (
                "Silicon efficiency (TOPS/W) at each composition "
                "level across architecture families"
            ),
            "xaxis": {"title": "Composition level"},
            "yaxis": {"title": "TOPS / W (INT8, dense)", "type": "log"},
            "barmode": "group",
            "margin": {"t": 50, "b": 70, "l": 70, "r": 20},
            "legend": {"orientation": "h", "y": -0.25},
        },
    }
    payload = {"chart_composition_efficiency": chart}
    return (
        f"const COMP_CHARTS = {json.dumps(payload)};\n"
        "for (const [id, spec] of Object.entries(COMP_CHARTS)) {\n"
        "  const el = document.getElementById(id);\n"
        "  if (el) Plotly.newPlot(id, spec.data, spec.layout, "
        "{displayModeBar: false, responsive: true});\n"
        "}\n"
    )


def render_composition_page(report: CompositionReport,
                            repo_root: Path) -> str:
    from graphs.reporting.microarch_html_template import (
        _CSS, _load_logo,
        _render_brand_footer, _render_brand_header,
    )
    assets = _load_logo(repo_root)
    header = _render_brand_header(
        assets,
        "Silicon composition hierarchy",
        f"ALU → PE → Tile → Cluster → SoC | generated {report.generated_at}",
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
table.blocks td.src { color: #586374; font-size: 11px; max-width: 360px; }
span.cat { display: inline-block; padding: 2px 8px; border-radius: 3px;
           color: white; font-size: 11px; font-weight: 600; }
section.arch-header { background: #fff; padding: 14px 20px;
                       border-radius: 6px; margin-bottom: 12px;
                       box-shadow: 0 1px 3px rgba(0,0,0,0.04); }
section.arch-header h3 { margin: 0 0 6px; color: #0a2540; }
section.arch-header .meta { color: #586374; font-size: 13px; }
.chart-section { background: #fff; padding: 18px 22px; border-radius: 6px;
                 margin-bottom: 18px;
                 box-shadow: 0 1px 3px rgba(0,0,0,0.04); }
.chart-section h3 { margin: 0 0 4px; color: #0a2540; }
.chart-section .chart-desc { color: #586374; font-size: 13px;
                             margin: 0 0 12px; }
.plot { width: 100%; min-height: 400px; }
section.method-note { background: #eef2f7; padding: 14px 18px;
                      border-left: 3px solid #0a2540; border-radius: 4px;
                      margin: 18px 0; }
a.nav-back { display: inline-block; color: #0a2540; text-decoration: none;
             font-weight: 600; margin-bottom: 10px; }
a.nav-back:hover { text-decoration: underline; }
"""

    arch_sections = []
    for h in report.hierarchies:
        decay = h.efficiency_decay
        arch_sections.append(
            f"""
<section class="arch-header" style="border-left:6px solid #0a2540;">
  <h3>{html.escape(h.name)}</h3>
  <div class="meta">
    <strong>Category:</strong> {html.escape(h.architecture.value)}
    | <strong>Process:</strong> {h.process_nm} nm
    | <strong>Levels:</strong> {len(h.blocks)}
    | <strong>ALU→SoC efficiency decay:</strong>
        <strong>{decay:.0f}×</strong>
  </div>
  <p style="color:#3a4452; font-size:13px; margin:8px 0 0;">
    {html.escape(h.notes)}
  </p>
</section>
{_render_hierarchy_table(h)}
"""
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>Silicon composition hierarchy</title>
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
    <h2>Silicon composition: ALU → PE → Tile → Cluster → SoC</h2>
    <div class="meta">Companion to the
      <a href="building_block_energy.html">building-block energy</a>
      and
      <a href="microarch_accounting.html">per-MAC accounting</a>
      views. Where those report the silicon-capability ceiling at
      the ALU level, this page follows the scaffolding tax as you
      wrap an ALU in successively larger composition levels. The
      drop in TOPS/W from ALU to SoC is the "overhead for
      coordination" that SoC specialization tries to minimize
      against a given workload's op mix.</div>
  </section>

  <section class="method-note">
    <h3 style="margin-top:0;">How to read these tables</h3>
    <ul>
      <li><strong>TOPS/W</strong> is silicon efficiency at that
        level (ops_per_MAC / pJ_per_MAC). It drops as you climb the
        hierarchy because more transistors serve coordination rather
        than compute.</li>
      <li><strong>Active %</strong> is the fraction of the level's
        transistors that participate in MAC computation on a dense
        INT8 workload. A CPU core sits near 0.5%; a KPU tile is
        above 70%.</li>
      <li><strong>Efficiency vs ALU</strong> shows how far the
        block's TOPS/W is below its own ALU's ceiling.</li>
      <li><strong>Decay</strong> in the arch header is the ratio
        ALU TOPS/W / SoC TOPS/W. A 50× decay means the SoC delivers
        1/50 the silicon efficiency of its own bare ALU.</li>
    </ul>
  </section>

  <section class="chart-section">
    <h3>Cross-architecture efficiency decay</h3>
    <p class="chart-desc">TOPS/W at each composition level, log-scale
      y-axis. Architectures that keep their TOPS/W flat from ALU
      to SoC are less wasteful of silicon on coordination overhead.</p>
    <div id="chart_composition_efficiency" class="plot"></div>
  </section>

  {''.join(arch_sections)}

  <section class="method-note">
    <strong>SoC specialization pointer:</strong>
    The composition hierarchy is the input to workload-driven SoC
    design. Given a workload's mix of (matmul / scatter-gather /
    sequential-control) operations, the right SoC combines CPU
    cores, DSP lanes, GPU SMs, and NPU tiles in proportions that
    match the op mix. This view provides the per-level silicon
    efficiency that feeds that trade-off analysis.
  </section>
</main>
{footer}
<script>
{_render_efficiency_chart_js(report)}
</script>
</body>
</html>
"""
