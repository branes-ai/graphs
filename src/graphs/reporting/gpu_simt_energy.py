"""
Energy accounting for one SIMT instruction on an Ampere-class GPU.

Models one SM-cycle issuing the same instruction across all 4
subpartitions of an Ampere SM (4 x 32-lane SIMT = 128 lanes total).
The data is assumed already resident in the per-subpartition register
file -- no L1, no shared memory, no off-chip traffic. The goal is to
expose the per-stage energy decomposition of the in-SM SIMT control
overhead vs the actual ALU work.

Pipeline stages (per subpartition; 1-4 fire 4x at SM level because
each subpartition runs its own L0 I-cache + decoder + warp scheduler
+ dispatch):

  1. Fetch       (L0 I-cache read)
  2. Decode      (instruction decoder)
  3. Schedule    (warp scheduler -- one per subpartition)
  4. Dispatch    (subpartition control wires)
  5. RF read     (banked register-file read, sources_per_op x lanes)
  6. OpCollect   (operand collector flop write/read)
  7. ALU disp    (operand wires from OC to CUDA cores)
  8. Compute     (FADD / FMUL / FMA in the lane ALUs)
  9. Writeback   (banked register-file write, lanes worth of dest)

For each (op_kind, precision) we report:
- per-stage energy (1..9)
- per-row energy (instruction-control / operand-A / operand-B
  / [operand-C] / compute / writeback) so the table reads like a
  pipeline diagram with energy in place of occupancy
- aggregate per-instruction energy
- normalized per-op energy using industry-standard FLOP counts
  (FMA = 2 FLOPS; packed narrow precision = 2x or 4x ops per
  instruction)

The model takes its primitive energies from
``graphs.hardware.technology_profile.TechnologyProfile`` -- the
documented source-of-truth for process-node-derived energy in this
repo.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from graphs.core.confidence import ConfidenceLevel, EstimationConfidence
from graphs.hardware.technology_profile import TechnologyProfile
from graphs.reporting.baseline_alu_energy import (
    OpKind,
    Precision,
    _alu_energies_pj as _baseline_alu_energies_pj,
)
from graphs.reporting.gpu_register_file import (
    GPURegisterFileBankModel,
    default_ampere_subpartition_rf,
)


# --------------------------------------------------------------------
# Pipeline stage taxonomy
# --------------------------------------------------------------------

# 9 stages, ordered. The labels are short to fit in the report
# header row. Long names are in STAGE_DESCRIPTIONS for the doc.
STAGE_LABELS: List[str] = [
    "Fch",       # 1. Fetch (L0 I$ read)
    "Dec",       # 2. Decode
    "Sch",       # 3. Warp schedule (per subpart)
    "Dsp",       # 4. Dispatch (control to subpart)
    "Rd",        # 5. RF read (banked)
    "OC",        # 6. Operand collector
    "Disp",      # 7. ALU dispatch (operand wires)
    "Exe",       # 8. Compute (lane ALUs)
    "WB",        # 9. Writeback (RF write)
]

STAGE_DESCRIPTIONS: Dict[str, str] = {
    "Fch":  "Fetch -- L0 instruction cache read; one per subpartition",
    "Dec":  "Decode -- instruction decoder; one per subpartition",
    "Sch":  "Warp schedule -- 16-warp scoreboard + priority arbitration; one per subpartition",
    "Dsp":  "Dispatch -- subpartition control wires energized",
    "Rd":   "Register-file read -- banked 32-bit reads, one per source operand per lane",
    "OC":   "Operand collector -- flop write + read to align operands across cycles",
    "Disp": "ALU dispatch -- operand wires from OC to lane ALUs",
    "Exe":  "Compute -- FADD / FMUL / FMA in the per-lane ALU",
    "WB":   "Writeback -- banked RF write, one per lane (dest only)",
}


# Default Ampere SM topology (Jetson Orin GA10B uses this layout).
DEFAULT_SUBPARTITIONS = 4
DEFAULT_LANES_PER_SUBPART = 32

# Sources per op (mirrors the baseline model).
_SOURCES_PER_OP: Dict[OpKind, int] = {
    OpKind.FADD: 2,
    OpKind.FMUL: 2,
    OpKind.FMA:  3,
}

# FLOPs per op (FMA = 2 FLOPS by convention).
_FLOPS_PER_OP: Dict[OpKind, int] = {
    OpKind.FADD: 1,
    OpKind.FMUL: 1,
    OpKind.FMA:  2,
}

# Packing factor: how many useful narrow-precision ops a single
# 32-bit-wide SIMT lane executes per instruction. fp16 packs 2x
# (HFMA2/HADD2/HMUL2), int8 packs 4x (DP4A and friends).
_PACKING_FACTOR: Dict[Precision, int] = {
    Precision.FP32: 1,
    Precision.FP16: 2,
    Precision.INT8: 4,
}


# Operand collector energy is poorly characterized in public
# literature. Treat it as a flop write + flop read per operand.
# Use the same per-flop figure as the baseline model: 5% of an RF
# read. Overhead vs RF traffic is small enough that the order-of-
# magnitude doesn't shift.
_OC_FLOP_FRACTION = 0.05


# Wire energy for a 32-bit operand traversing from the operand
# collector to a lane ALU. On modern GPUs this is on the order of
# a fraction of a register read (short interconnect); we model as
# 25% of register_read.
_ALU_DISPATCH_WIRE_FRACTION = 0.25


# --------------------------------------------------------------------
# Output dataclasses
# --------------------------------------------------------------------

@dataclass
class StageEnergy:
    """One pipeline-stage worth of energy.

    Activity count = how many times the unit fires during one
    SM-cycle (e.g. RF reads = 256 for FMUL/FADD, 384 for FMA).
    """
    label: str
    description: str
    pj_each: float
    activity_count: int

    @property
    def total_pj(self) -> float:
        return self.pj_each * self.activity_count


@dataclass
class SIMTInstructionEnergy:
    """Energy report for one SIMT instruction at one (op, precision).

    Provides:
    - ``stages``: 9 StageEnergy entries in pipeline order
    - ``rows``: row x stage matrix (operation -> {stage label -> pJ})
                so a caller can render the gantt-style energy table
    - ``rf_model``: the banked-SRAM register-file model in use
    - aggregate totals + per-op normalization
    """
    op_kind: OpKind
    precision: Precision
    sm_subpartitions: int
    lanes_per_subpartition: int
    sources_per_op: int
    flops_per_op: int
    packing_factor: int

    rf_model: GPURegisterFileBankModel

    stages: List[StageEnergy]
    rows: Dict[str, Dict[str, float]]

    confidence: EstimationConfidence

    @property
    def lanes(self) -> int:
        return self.sm_subpartitions * self.lanes_per_subpartition

    @property
    def total_pj(self) -> float:
        return sum(s.total_pj for s in self.stages)

    @property
    def ops_executed(self) -> int:
        """How many useful narrow-precision ops this instruction did
        (with packing). Headline counts for the 3 precisions:
        fp32 -> 128, fp16 packed -> 256, int8 packed -> 512."""
        return self.lanes * self.packing_factor

    @property
    def flops_executed(self) -> int:
        """Headline FLOP count: ops_executed * flops_per_op.
        FMA fp32 -> 256 FLOPS; FMA fp16 packed -> 512 FLOPS;
        FMA int8 packed -> 1024 IntOPS (counted via flops_per_op=2)."""
        return self.ops_executed * self.flops_per_op

    @property
    def pj_per_op(self) -> float:
        return self.total_pj / self.ops_executed

    @property
    def pj_per_flop(self) -> float:
        return self.total_pj / self.flops_executed


# --------------------------------------------------------------------
# Builder
# --------------------------------------------------------------------

def simt_instruction_energy(
    profile: TechnologyProfile,
    op_kind: OpKind,
    precision: Precision,
    sm_subpartitions: int = DEFAULT_SUBPARTITIONS,
    lanes_per_subpartition: int = DEFAULT_LANES_PER_SUBPART,
    rf_model: GPURegisterFileBankModel | None = None,
) -> SIMTInstructionEnergy:
    """Build the 9-stage energy report for one SIMT instruction.

    The register file is modelled as banked SRAM (the key SIMT
    architectural overhead -- many threads in flight require a large
    multi-banked RF, not flip-flops). Stage 5 (Rd) and stage 9 (WB)
    are wide-bank accesses, NOT individual register-per-thread reads.

    Args:
        profile: TechnologyProfile (canonical case:
            ``EDGE_8NM_LPDDR5`` for Jetson Orin GA10B).
        op_kind: FADD / FMUL / FMA.
        precision: FP32 / FP16 / INT8 (narrow precisions are
            assumed to be packed: HFMA2 for fp16, DP4A for int8).
        sm_subpartitions: typically 4 on Ampere (GA100/GA10B).
        lanes_per_subpartition: typically 32.
        rf_model: banked-SRAM register file. Defaults to the Ampere
            subpartition layout (64 KiB / 4 banks / 1024-bit wide)
            parameterised by the profile.

    Returns:
        ``SIMTInstructionEnergy`` with per-stage + per-row breakdowns.
    """
    sources = _SOURCES_PER_OP[op_kind]
    flops = _FLOPS_PER_OP[op_kind]
    packing = _PACKING_FACTOR[precision]
    lanes = sm_subpartitions * lanes_per_subpartition

    if rf_model is None:
        rf_model = default_ampere_subpartition_rf(profile)

    # Banked SRAM RF: per-access energies + concurrency counts.
    bank_read_pj = rf_model.bank_read_energy_pj()
    bank_write_pj = rf_model.bank_write_energy_pj()
    rf_read_count = rf_model.sm_bank_reads_per_instruction(
        sm_subpartitions, sources,
    )
    rf_write_count = rf_model.sm_bank_writes_per_instruction(
        sm_subpartitions,
    )
    reads_per_warp_source = rf_model.reads_per_warp_source()

    # Pipeline-front-end energies (per subpartition).
    fetch_pj = profile.instruction_fetch_energy_pj
    decode_pj = profile.instruction_decode_energy_pj
    dispatch_pj = profile.instruction_dispatch_energy_pj
    sched_pj = decode_pj * 0.5  # scoreboard + arbitration

    # Operand collector: per source-operand-warp it buffers one wide
    # operand (a flop write + read at warp width). Scale by the same
    # bytes_per_bank_access since OC entries match the RF bank width.
    oc_per_op_pj = bank_read_pj * _OC_FLOP_FRACTION * 2

    # ALU dispatch wire: per-lane per-source operand wire drive from
    # the operand collector to the ALU. Scaled relative to a wide-
    # bank read since the wires fan out from the OC at bank width.
    alu_disp_pj = bank_read_pj * _ALU_DISPATCH_WIRE_FRACTION / lanes_per_subpartition
    # (The /lanes_per_subpartition splits the bank-wide energy across
    # the per-thread fanout wires it represents.)

    # ALU compute: per-lane FADD/FMUL/FMA energy from the baseline
    # model, scaled by the precision packing so per-instruction
    # compute energy stays approximately constant across precisions
    # (same 32-bit datapath, doing 1/2/4 useful ops).
    mul_pj, add_pj = _baseline_alu_energies_pj(profile, op_kind, precision)
    alu_per_lane_pj = ((mul_pj or 0.0) + (add_pj or 0.0)) * packing

    # Stages -- 1-4 fire per subpartition; 5-9 fire at the bank /
    # lane / wire activity counts derived above.
    oc_count = sm_subpartitions * sources * reads_per_warp_source
    alu_disp_count = sources * lanes
    stages: List[StageEnergy] = [
        StageEnergy("Fch",  STAGE_DESCRIPTIONS["Fch"],
                    fetch_pj,    sm_subpartitions),
        StageEnergy("Dec",  STAGE_DESCRIPTIONS["Dec"],
                    decode_pj,   sm_subpartitions),
        StageEnergy("Sch",  STAGE_DESCRIPTIONS["Sch"],
                    sched_pj,    sm_subpartitions),
        StageEnergy("Dsp",  STAGE_DESCRIPTIONS["Dsp"],
                    dispatch_pj, sm_subpartitions),
        # Stage 5: banked SRAM RF reads -- the SIMT energy story.
        StageEnergy("Rd",   STAGE_DESCRIPTIONS["Rd"],
                    bank_read_pj, rf_read_count),
        StageEnergy("OC",   STAGE_DESCRIPTIONS["OC"],
                    oc_per_op_pj, oc_count),
        StageEnergy("Disp", STAGE_DESCRIPTIONS["Disp"],
                    alu_disp_pj, alu_disp_count),
        StageEnergy("Exe",  STAGE_DESCRIPTIONS["Exe"],
                    alu_per_lane_pj, lanes),
        # Stage 9: banked SRAM RF writes.
        StageEnergy("WB",   STAGE_DESCRIPTIONS["WB"],
                    bank_write_pj, rf_write_count),
    ]

    # Build the row x stage matrix. Each source operand row owns
    # `reads_per_warp_source` wide-bank reads (1 for a perfect-fit
    # Ampere RF; more for narrower banks).
    rows: Dict[str, Dict[str, float]] = {}
    rows["Instruction control"] = {
        "Fch":  stages[0].total_pj,
        "Dec":  stages[1].total_pj,
        "Sch":  stages[2].total_pj,
        "Dsp":  stages[3].total_pj,
    }
    src_labels = ["Src operand A", "Src operand B"]
    if sources == 3:
        src_labels.append("Src operand C")
    for src in src_labels:
        # Per source operand: sm_subpartitions wide-bank reads
        # (one per subpartition's warp), each at bank_read_pj.
        per_src_read_count = sm_subpartitions * reads_per_warp_source
        per_src_oc_count = sm_subpartitions * reads_per_warp_source
        per_src_disp_count = lanes
        rows[src] = {
            "Rd":   bank_read_pj * per_src_read_count,
            "OC":   oc_per_op_pj * per_src_oc_count,
            "Disp": alu_disp_pj * per_src_disp_count,
        }
    rows["ALU compute"] = {
        "Exe":  stages[7].total_pj,
    }
    rows["Dest writeback"] = {
        "WB":   stages[8].total_pj,
    }

    confidence = EstimationConfidence(
        level=ConfidenceLevel.THEORETICAL,
        score=0.55,
        source=(
            f"gpu_simt: TechnologyProfile['{profile.name}'] "
            f"on {sm_subpartitions}x{lanes_per_subpartition} SIMT; "
            f"RF banked SRAM "
            f"({rf_model.bytes_per_subpartition // 1024} KiB/subpart, "
            f"{rf_model.num_banks} banks, "
            f"{rf_model.bank_width_bits}-bit wide), "
            f"per-bank-read {bank_read_pj:.2f} pJ; "
            f"per-subpartition control fanout x{sm_subpartitions}"
        ),
    )

    return SIMTInstructionEnergy(
        op_kind=op_kind,
        precision=precision,
        sm_subpartitions=sm_subpartitions,
        lanes_per_subpartition=lanes_per_subpartition,
        sources_per_op=sources,
        flops_per_op=flops,
        packing_factor=packing,
        rf_model=rf_model,
        stages=stages,
        rows=rows,
        confidence=confidence,
    )


__all__ = [
    "STAGE_LABELS",
    "STAGE_DESCRIPTIONS",
    "DEFAULT_SUBPARTITIONS",
    "DEFAULT_LANES_PER_SUBPART",
    "StageEnergy",
    "SIMTInstructionEnergy",
    "simt_instruction_energy",
]
