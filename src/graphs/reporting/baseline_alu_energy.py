"""
Baseline ALU energy model -- the "irreducible compute floor".

Models a single ALU stripped of all SIMT overhead: input flip-flops
wired directly to the ALU inputs, and an output flip-flop on the
result. No register file. No operand collector. No instruction fetch
/ decode / scheduler / dispatch. Just the gates and clocked storage
needed to do *one* arithmetic operation.

This is the absolute minimum energy a fp32 ALU op can cost on a
given process technology. Real GPU SIMT instructions execute the
same arithmetic but pay 20-150x more because of the SIMT overhead
(banked register file, banked operand collector, per-subpartition
warp schedulers, inter-stage wires). The ratio between SIMT and
this baseline is the architectural "tax" you pay for parallel
control.

Three operator types are modeled, mirroring the basic linear-algebra
arithmetic set:

- FADD: 2 source flops, 1 ALU (32-bit floating add), 1 dest flop
- FMUL: 2 source flops, 1 ALU (32-bit floating multiply), 1 dest flop
- FMA:  3 source flops, 1 multiplier + 1 adder, 1 dest flop
        (the canonical multiply-accumulate; one FMA = 2 FLOPS)

Three precisions:

- FP32 (1 op per ALU)
- FP16 (1 op per ALU; ALU is half-width so ~0.5x energy)
- INT8 (1 op per ALU; ALU is quarter-width so ~0.2x energy)

Note: the baseline does NOT model packing. A baseline FP16 op
runs through one half-width ALU; HFMA2 packing only matters in the
SIMT pipeline where the 32-bit datapath gets reused for two fp16
elements in parallel.

All energy primitives are sourced from
``graphs.hardware.technology_profile.TechnologyProfile``. The
relative ratios between FADD/FMUL/FMA come from Horowitz's
canonical 45nm energy table (FP add 0.9 pJ, FP mul 3.7 pJ,
FMA ~ 1.13x FMUL); they are stable across process nodes within
~10-15%.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict

from graphs.core.confidence import ConfidenceLevel, EstimationConfidence
from graphs.hardware.technology_profile import TechnologyProfile

# Re-export the canonical Precision enum (issue #59). The baseline
# model only treats FP32/FP16/INT8, but it must use the same enum
# class as the hardware catalog so that callers passing the canonical
# member find their entry in this module's scaling dict.
from graphs.hardware.resource_model import Precision  # noqa: F401


# --------------------------------------------------------------------
# Op kinds, precisions
# --------------------------------------------------------------------

class OpKind(Enum):
    FADD = "FADD"   # 2-source floating add
    FMUL = "FMUL"   # 2-source floating multiply
    FMA  = "FMA"    # 3-source fused multiply-add (1 FMA = 2 FLOPS)


# Sources per op kind (for FF read count in baseline).
_SOURCES_PER_OP: Dict[OpKind, int] = {
    OpKind.FADD: 2,
    OpKind.FMUL: 2,
    OpKind.FMA:  3,
}


# FLOPs per op for headline conversion (FMA = 2 FLOPS by industry
# convention; FADD/FMUL = 1 FLOP each).
_FLOPS_PER_OP: Dict[OpKind, int] = {
    OpKind.FADD: 1,
    OpKind.FMUL: 1,
    OpKind.FMA:  2,
}


# Energy ratio of FADD : FMUL : FMA at 45nm (Horowitz 2014, ISSCC
# tutorial "Computing's energy problem"; FP add ~0.9 pJ, FP mul
# ~3.7 pJ, FMA ~ FP add + FP mul). Ratios are stable across process
# nodes; absolute values come from the technology profile.
_FMUL_OVER_FMA = 0.88   # FMUL is ~88% of FMA (most of FMA is the MUL)
_FADD_OVER_FMA = 0.22   # FADD is ~22% of FMA (adders are small)


# Precision ALU scaling: a multiplier is roughly bits**2 in area /
# active capacitance. Mantissa-bit ratios:
#   fp32 mantissa 24 bits, fp16 mantissa 11 bits, int8 8 bits.
#   (24/24)^2 = 1.00, (11/24)^2 = 0.21, (8/24)^2 = 0.11.
# Real implementations carry overhead (sign / exponent / round logic),
# so the practical ratios converge to the rule of thumb:
#   fp16 ~= 0.5 x fp32 (HFMA2 unit area is approximately half)
#   int8 ~= 0.2 x fp32 (DP4A int8 multiplier is 4-5x cheaper)
_PRECISION_ALU_SCALE: Dict[Precision, float] = {
    Precision.FP32: 1.00,
    Precision.FP16: 0.50,
    Precision.INT8: 0.20,
}


# --------------------------------------------------------------------
# Component energies (pJ)
# --------------------------------------------------------------------

@dataclass
class ALUEnergyComponents:
    """Per-component energies for one baseline ALU op.

    All values in picojoules. ``mul_pj`` is None for FADD; ``add_pj``
    is None for FMUL.
    """
    ff_read_each_pj: float       # per-flop read
    ff_write_each_pj: float      # per-flop write
    mul_pj: float | None          # multiplier energy (FMUL or FMA)
    add_pj: float | None          # adder energy (FADD or FMA)
    sources: int                  # 2 for FADD/FMUL, 3 for FMA
    flops_per_op: int             # 1 for FADD/FMUL, 2 for FMA

    @property
    def ff_total_read_pj(self) -> float:
        return self.ff_read_each_pj * self.sources

    @property
    def alu_total_pj(self) -> float:
        return (self.mul_pj or 0.0) + (self.add_pj or 0.0)

    @property
    def total_pj(self) -> float:
        return self.ff_total_read_pj + self.alu_total_pj + self.ff_write_each_pj

    @property
    def pj_per_flop(self) -> float:
        return self.total_pj / self.flops_per_op


@dataclass
class BaselineALUEnergy:
    """Energy report for a single baseline-ALU op of one (op, prec).

    Holds the per-component breakdown plus the aggregate total and a
    confidence record.
    """
    op_kind: OpKind
    precision: Precision
    components: ALUEnergyComponents
    confidence: EstimationConfidence

    @property
    def total_pj(self) -> float:
        return self.components.total_pj

    @property
    def pj_per_flop(self) -> float:
        return self.components.pj_per_flop

    def as_row(self) -> Dict[str, float]:
        """Return the row for the per-stage table.

        Stages for the baseline are:
          (1) FF_read  (2) MUL  (3) ADD  (4) FF_write
        """
        c = self.components
        return {
            "FF_read":  c.ff_total_read_pj,
            "MUL":      c.mul_pj or 0.0,
            "ADD":      c.add_pj or 0.0,
            "FF_write": c.ff_write_each_pj,
            "Total":    c.total_pj,
        }


# --------------------------------------------------------------------
# Builders
# --------------------------------------------------------------------

def _ff_energy_pj(profile: TechnologyProfile) -> float:
    """Per-flop active energy.

    A directly-clocked flip-flop is far smaller than a register file
    cell (no addressing, no decoder, no banked sense). Empirically
    a 32-bit edge-triggered FF burns ~3-5% of a register-file read;
    we take a fixed 5% of the profile's register_read energy as the
    per-flop figure.
    """
    return profile.register_read_energy_pj * 0.05


def _alu_energies_pj(
    profile: TechnologyProfile,
    op_kind: OpKind,
    precision: Precision,
) -> tuple[float | None, float | None]:
    """Return (mul_pj, add_pj) for a single ALU op at this precision.

    The technology profile provides ``base_alu_energy_pj`` -- we
    interpret that as the canonical FMA-class compute energy at fp32
    (the profile's MAC-oriented numbers like simd_mac, tensor_core_mac
    are derived from this baseline). FADD and FMUL are scaled down
    using the Horowitz ratios; precision narrows the ALU width.
    """
    base_fma_fp32 = profile.base_alu_energy_pj  # canonical fp32 FMA
    if precision not in _PRECISION_ALU_SCALE:
        supported = sorted(p.name for p in _PRECISION_ALU_SCALE)
        raise ValueError(
            f"baseline_alu_energy only models {', '.join(supported)}; "
            f"got {precision.name}"
        )
    prec_scale = _PRECISION_ALU_SCALE[precision]

    if op_kind is OpKind.FADD:
        return (None, base_fma_fp32 * _FADD_OVER_FMA * prec_scale)
    if op_kind is OpKind.FMUL:
        return (base_fma_fp32 * _FMUL_OVER_FMA * prec_scale, None)
    # FMA = full multiply + accumulator-add
    return (
        base_fma_fp32 * _FMUL_OVER_FMA * prec_scale,
        base_fma_fp32 * (1.0 - _FMUL_OVER_FMA) * prec_scale,
    )


def baseline_alu_energy(
    profile: TechnologyProfile,
    op_kind: OpKind,
    precision: Precision,
) -> BaselineALUEnergy:
    """Compute baseline ALU energy for one op.

    Args:
        profile: technology source-of-truth (Samsung 8nm Orin profile
            for the canonical case; any profile supported).
        op_kind: FADD / FMUL / FMA.
        precision: FP32 / FP16 / INT8.

    Returns:
        ``BaselineALUEnergy`` with components + total + confidence.
    """
    ff_each = _ff_energy_pj(profile)
    mul_pj, add_pj = _alu_energies_pj(profile, op_kind, precision)
    sources = _SOURCES_PER_OP[op_kind]
    flops = _FLOPS_PER_OP[op_kind]

    components = ALUEnergyComponents(
        ff_read_each_pj=ff_each,
        ff_write_each_pj=ff_each,
        mul_pj=mul_pj,
        add_pj=add_pj,
        sources=sources,
        flops_per_op=flops,
    )

    conf = EstimationConfidence(
        level=ConfidenceLevel.THEORETICAL,
        score=0.55,
        source=(
            f"baseline_alu: TechnologyProfile['{profile.name}'] "
            f"base_alu={profile.base_alu_energy_pj:.2f} pJ, "
            f"precision_scale={_PRECISION_ALU_SCALE[precision]:.2f}, "
            f"horowitz ratios FADD/FMA={_FADD_OVER_FMA}, FMUL/FMA={_FMUL_OVER_FMA}"
        ),
    )
    return BaselineALUEnergy(
        op_kind=op_kind,
        precision=precision,
        components=components,
        confidence=conf,
    )


__all__ = [
    "OpKind",
    "Precision",
    "ALUEnergyComponents",
    "BaselineALUEnergy",
    "baseline_alu_energy",
]
