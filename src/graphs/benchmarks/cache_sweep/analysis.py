"""
Analyse a working-set-size sweep to recover per-cache-level energy.

A bandwidth-vs-working-set curve has plateaus at L1, L2, L3, and
DRAM with steep transitions at the cache capacity boundaries. The
plateau bandwidth gives the *effective* per-level throughput; the
plateau energy (when RAPL is available) divided by bytes streamed
gives the per-byte energy.

Algorithm (deliberately simple to keep the test surface small):

1. Sort points by working-set size ascending.
2. Compute log-scale derivative of bandwidth: drops above a
   threshold mark transition steps.
3. Plateaus between transitions identify L1 / L2 / L3 / DRAM.
4. For each plateau, average the bandwidth and the per-byte energy.

Real silicon shows clean plateaus on x86 CPUs; on Jetson SoCs the
L1-vs-L2 transition is sometimes blurred by the unified L2/SLC.
The detector reports the plateaus it found and leaves missing
levels as ``None`` rather than fabricating boundaries.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence

from graphs.benchmarks.cache_sweep.working_set_sweep import WorkingSetPoint


# Per-level bandwidth ratio (prev / curr) that qualifies as a real
# transition. Tuned for a NumPy in-place XOR kernel on a modern x86
# CPU: empirically the L2->L3 step is ~1.4x (clear), L3->DRAM is
# only ~1.2x because the in-place kernel's write traffic stays
# coherent at the cache controller. 1.2 captures both transitions
# without firing on plateau noise (intra-plateau ratios stay below
# 1.1 in practice).
_TRANSITION_RATIO = 1.2


class CacheLevel(Enum):
    L1 = "L1"
    L2 = "L2"
    L3 = "L3"
    DRAM = "DRAM"


@dataclass
class DetectedLevels:
    """Working-set bounds for each detected plateau.

    Each entry is ``(min_bytes, max_bytes)`` covering the points
    assigned to the level. None means "no plateau identified" --
    for example, on an SoC where L1 is too small to fit any sweep
    point, or when the operator cut the max range below the L3
    boundary.
    """
    levels: Dict[CacheLevel, Optional[tuple]] = field(default_factory=dict)


@dataclass
class PerLevelEnergy:
    """Per-cache-level effective bandwidth + per-byte energy."""
    bandwidth_gbps: Dict[CacheLevel, Optional[float]] = field(
        default_factory=dict
    )
    energy_per_byte_pj: Dict[CacheLevel, Optional[float]] = field(
        default_factory=dict
    )

    def has_calibrated_energy(self, level: CacheLevel) -> bool:
        """True when the sweep captured a non-None per-byte energy
        for the level (RAPL was available + a plateau was found)."""
        return self.energy_per_byte_pj.get(level) is not None


# --------------------------------------------------------------------
# Plateau detection
# --------------------------------------------------------------------

def _find_transition_indices(
    points: Sequence[WorkingSetPoint],
) -> List[int]:
    """Indices where bandwidth drops by at least ``_TRANSITION_RATIO``
    relative to the previous point.

    A transition is the FIRST index in the new (lower-bandwidth)
    plateau, so a 4-point sequence [L1, L1, L2, L2] returns [2].
    """
    transitions: List[int] = []
    for i in range(1, len(points)):
        prev = points[i - 1].bandwidth_gbps
        curr = points[i].bandwidth_gbps
        if prev <= 0 or curr <= 0:
            continue
        if prev / curr >= _TRANSITION_RATIO:
            transitions.append(i)
    return transitions


def detect_levels(points: Sequence[WorkingSetPoint]) -> DetectedLevels:
    """Partition the sweep into L1 / L2 / L3 / DRAM plateaus.

    Returns plateaus assigned in order: the first plateau is L1,
    the next is L2, and so on. When fewer than 4 plateaus are
    detected the missing levels are mapped to ``None`` so the
    caller can fall back to analytical defaults rather than
    fabricating a boundary.
    """
    sorted_points = sorted(points, key=lambda p: p.bytes_resident)
    if len(sorted_points) < 2:
        return DetectedLevels()

    transitions = _find_transition_indices(sorted_points)
    # Plateau boundaries: [0, t1, t2, ..., tn, len]
    bounds = [0] + transitions + [len(sorted_points)]
    plateau_ranges = [
        (bounds[i], bounds[i + 1])
        for i in range(len(bounds) - 1)
        if bounds[i + 1] > bounds[i]
    ]

    levels = DetectedLevels()
    level_order = [CacheLevel.L1, CacheLevel.L2, CacheLevel.L3, CacheLevel.DRAM]

    # Initialise every level to None
    for lvl in level_order:
        levels.levels[lvl] = None

    for level, (start, end) in zip(level_order, plateau_ranges):
        plateau_points = sorted_points[start:end]
        if not plateau_points:
            continue
        levels.levels[level] = (
            plateau_points[0].bytes_resident,
            plateau_points[-1].bytes_resident,
        )
    return levels


# --------------------------------------------------------------------
# Per-level energy estimation
# --------------------------------------------------------------------

def estimate_per_level_energy(
    points: Sequence[WorkingSetPoint],
    levels: Optional[DetectedLevels] = None,
) -> PerLevelEnergy:
    """Average bandwidth and per-byte energy across each plateau.

    When ``levels`` is omitted, runs ``detect_levels`` first.
    """
    if levels is None:
        levels = detect_levels(points)

    sorted_points = sorted(points, key=lambda p: p.bytes_resident)
    out = PerLevelEnergy()
    for lvl in CacheLevel:
        out.bandwidth_gbps[lvl] = None
        out.energy_per_byte_pj[lvl] = None

    for level, byte_range in levels.levels.items():
        if byte_range is None:
            continue
        lo, hi = byte_range
        plateau = [
            p for p in sorted_points if lo <= p.bytes_resident <= hi
        ]
        if not plateau:
            continue

        bws = [p.bandwidth_gbps for p in plateau if p.bandwidth_gbps > 0]
        if bws:
            out.bandwidth_gbps[level] = sum(bws) / len(bws)

        epbs = [
            p.energy_per_byte_pj for p in plateau
            if p.energy_per_byte_pj is not None
        ]
        if epbs:
            out.energy_per_byte_pj[level] = sum(epbs) / len(epbs)

    return out


__all__ = [
    "CacheLevel",
    "DetectedLevels",
    "PerLevelEnergy",
    "detect_levels",
    "estimate_per_level_energy",
]
