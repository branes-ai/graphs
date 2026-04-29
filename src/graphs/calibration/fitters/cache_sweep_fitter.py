"""
Cache-sweep fitter (Path B PR-1).

Consumes the output of ``graphs.benchmarks.cache_sweep.run_sweep``
and writes per-cache-level calibrated coefficients into a
``HardwareResourceModel``'s ``field_provenance`` map.

Fields written (when the corresponding plateau was detected with a
non-None per-byte energy):

  - ``l1_cache_energy_per_byte`` -> CALIBRATED
  - ``l2_cache_energy_per_byte`` -> CALIBRATED
  - ``l3_cache_energy_per_byte`` -> CALIBRATED

Fields stay at THEORETICAL when:
  - the corresponding plateau wasn't found
  - RAPL was unavailable (energy column is None)

In both cases the panel renders the analytical TechnologyProfile
default and surfaces the THEORETICAL badge -- matching the pattern
established in M3-M5.

Design notes:

- The fitter does NOT mutate the energy model directly. It writes
  to ``field_provenance`` so the existing panel readers can pick
  up the upgrade. Future PRs may add a dedicated apply-to-tech-
  profile path; for now the calibrated values surface in the panel
  notes / metrics.

- Bandwidth values are recorded in ``provenance.source`` rather
  than overwriting any field, since the resource model has no
  per-level effective-bandwidth field today.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

from graphs.benchmarks.cache_sweep.analysis import (
    CacheLevel,
    DetectedLevels,
    PerLevelEnergy,
    detect_levels,
    estimate_per_level_energy,
)
from graphs.benchmarks.cache_sweep.working_set_sweep import WorkingSetPoint
from graphs.core.confidence import EstimationConfidence


# Map cache-sweep CacheLevel -> resource_model field name we tag.
_LEVEL_TO_FIELD = {
    CacheLevel.L1: "l1_cache_energy_per_byte",
    CacheLevel.L2: "l2_cache_energy_per_byte",
    CacheLevel.L3: "l3_cache_energy_per_byte",
}


@dataclass
class CacheFitResult:
    """Fitted per-cache-level coefficients ready to apply."""

    sku_name: str = ""

    # Calibrated per-byte energies from the sweep (pJ/byte).
    # None when the level had no plateau or no energy capture.
    l1_energy_per_byte_pj: Optional[float] = None
    l2_energy_per_byte_pj: Optional[float] = None
    l3_energy_per_byte_pj: Optional[float] = None

    # Effective bandwidth per level (GB/s); kept for reporting.
    l1_bandwidth_gbps: Optional[float] = None
    l2_bandwidth_gbps: Optional[float] = None
    l3_bandwidth_gbps: Optional[float] = None

    # Plateaus actually detected (range of working-set sizes).
    detected: Optional[DetectedLevels] = None

    # Number of sweep points consumed.
    num_points: int = 0


class CacheSweepFitter:
    """
    Fit per-cache-level coefficients from a working-set sweep.

    Mirrors the M1 / M2 fitter shape: ``fit`` produces a
    ``CacheFitResult``; ``apply_to_model`` writes CALIBRATED
    provenance entries into the resource model's
    ``field_provenance``.
    """

    def fit(
        self,
        points: Sequence[WorkingSetPoint],
        sku_name: str = "",
    ) -> CacheFitResult:
        levels = detect_levels(points)
        per_level = estimate_per_level_energy(points, levels=levels)

        result = CacheFitResult(
            sku_name=sku_name,
            num_points=len(points),
            detected=levels,
        )

        # Bandwidth (always available when a plateau exists)
        result.l1_bandwidth_gbps = per_level.bandwidth_gbps.get(CacheLevel.L1)
        result.l2_bandwidth_gbps = per_level.bandwidth_gbps.get(CacheLevel.L2)
        result.l3_bandwidth_gbps = per_level.bandwidth_gbps.get(CacheLevel.L3)

        # Energy (None when RAPL was unavailable)
        result.l1_energy_per_byte_pj = (
            per_level.energy_per_byte_pj.get(CacheLevel.L1)
        )
        result.l2_energy_per_byte_pj = (
            per_level.energy_per_byte_pj.get(CacheLevel.L2)
        )
        result.l3_energy_per_byte_pj = (
            per_level.energy_per_byte_pj.get(CacheLevel.L3)
        )
        return result

    @staticmethod
    def apply_to_model(
        resource_model: object,
        fit_result: CacheFitResult,
    ) -> None:
        """Write CALIBRATED provenance for every level the fit
        returned a non-None per-byte energy for. Levels without an
        energy capture stay at their pre-existing provenance.
        """
        if not hasattr(resource_model, "set_provenance"):
            return

        per_level_energies = {
            CacheLevel.L1: fit_result.l1_energy_per_byte_pj,
            CacheLevel.L2: fit_result.l2_energy_per_byte_pj,
            CacheLevel.L3: fit_result.l3_energy_per_byte_pj,
        }
        per_level_bandwidth = {
            CacheLevel.L1: fit_result.l1_bandwidth_gbps,
            CacheLevel.L2: fit_result.l2_bandwidth_gbps,
            CacheLevel.L3: fit_result.l3_bandwidth_gbps,
        }

        for level, pj_per_byte in per_level_energies.items():
            if pj_per_byte is None:
                continue
            field_name = _LEVEL_TO_FIELD[level]
            bw = per_level_bandwidth.get(level)
            source_parts = [
                f"cache_sweep_fitter/{fit_result.sku_name or 'unknown'}",
                f"{pj_per_byte:.3f} pJ/byte",
            ]
            if bw is not None:
                source_parts.append(f"{bw:.1f} GB/s effective")
            source_parts.append(f"{fit_result.num_points} sweep points")
            resource_model.set_provenance(
                field_name,
                EstimationConfidence.calibrated(
                    score=0.85,
                    source="; ".join(source_parts),
                ),
            )


__all__ = [
    "CacheFitResult",
    "CacheSweepFitter",
]
