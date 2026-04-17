"""
Layer 2 Register/SIMD Fitter - Fit SIMD Efficiency and Register Overhead

Consumes BenchmarkResult objects from Layer 2 benchmarks (SIMD width
sweep and register pressure) and fits:
  - simd_efficiency: ratio of wide-vector throughput to scalar throughput,
    comparable to the 0.70 constant in CPUMapper._analyze_vectorization
  - ilp_ratio: independent-vs-dependent FMA throughput ratio, indicating
    register-delivery overhead
  - simd_packed_multiplier: measured energy scaling for packed SIMD ops,
    comparable to the 0.90 in CIRCUIT_TYPE_MULTIPLIER['simd_packed']

Writes EstimationConfidence(CALIBRATED) into
HardwareResourceModel.field_provenance for the fitted fields.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from graphs.benchmarks.schema import BenchmarkResult, LayerTag
from graphs.core.confidence import EstimationConfidence


@dataclass
class SIMDFitResult:
    """Result of fitting Layer 2 SIMD/register coefficients."""

    hardware_name: str = ""

    # Per-width measured throughput (GFLOPS)
    width_throughput: Dict[int, float] = field(default_factory=dict)

    # Derived SIMD efficiency: wide / scalar throughput
    # Comparable to CPUMapper's 0.70 vectorization_efficiency
    simd_efficiency: Optional[float] = None

    # Scalar throughput (width=1) as baseline
    scalar_gflops: Optional[float] = None

    # Peak vector throughput (largest width with plateau)
    peak_vector_gflops: Optional[float] = None

    # ILP ratio from register-pressure benchmark
    ilp_ratio: Optional[float] = None

    # Metadata
    num_results_used: int = 0


class Layer2RegisterFitter:
    """
    Fit SIMD efficiency and register-delivery overhead from Layer 2
    measurements.
    """

    def fit(
        self,
        results: List[BenchmarkResult],
        hardware_name: str = "",
    ) -> SIMDFitResult:
        """
        Fit SIMD efficiency from width-sweep and register-pressure results.

        Args:
            results: BenchmarkResults from Layer 2 benchmarks
                     (layer==REGISTER_SIMD, success==True)
            hardware_name: SKU name for the fit record

        Returns:
            SIMDFitResult with derived coefficients
        """
        layer2_results = [
            r for r in results
            if r.layer is LayerTag.REGISTER_SIMD and r.success
        ]

        fit = SIMDFitResult(
            hardware_name=hardware_name,
            num_results_used=len(layer2_results),
        )

        # Extract width-sweep data
        for r in layer2_results:
            width = r.extra.get("vector_width")
            if width is not None and r.gflops > 0:
                fit.width_throughput[width] = r.gflops

        # Extract register-pressure data
        for r in layer2_results:
            ilp = r.extra.get("ilp_ratio")
            if ilp is not None:
                fit.ilp_ratio = ilp

        # Derive SIMD efficiency
        if fit.width_throughput:
            sorted_widths = sorted(fit.width_throughput.keys())

            # Scalar baseline: smallest width
            fit.scalar_gflops = fit.width_throughput[sorted_widths[0]]

            # Peak vector: largest width (plateau region)
            fit.peak_vector_gflops = fit.width_throughput[sorted_widths[-1]]

            if fit.scalar_gflops and fit.scalar_gflops > 0:
                # Efficiency = how much of the SIMD width is effectively used
                # Normalize by width ratio to get per-lane efficiency
                max_width = sorted_widths[-1]
                min_width = sorted_widths[0]
                width_ratio = max_width / min_width if min_width > 0 else 1.0
                throughput_ratio = (
                    fit.peak_vector_gflops / fit.scalar_gflops
                )
                # SIMD efficiency = actual speedup / theoretical speedup
                # Clamp to [0, 1]: dispatch overhead on small widths
                # can make the ratio exceed 1.0 or go near 0.
                raw_eff = throughput_ratio / width_ratio
                fit.simd_efficiency = max(0.0, min(1.0, raw_eff))

        return fit

    @staticmethod
    def apply_to_model(
        resource_model: object,
        fit_result: SIMDFitResult,
    ) -> None:
        """
        Write fitted SIMD coefficients into field_provenance as CALIBRATED.
        """
        if not hasattr(resource_model, 'set_provenance'):
            return

        if fit_result.simd_efficiency is not None:
            source_parts = [f"layer2_register_fitter/{fit_result.hardware_name}"]
            source_parts.append(
                f"simd_eff={fit_result.simd_efficiency:.3f}"
            )
            if fit_result.scalar_gflops:
                source_parts.append(f"scalar={fit_result.scalar_gflops:.1f} GFLOPS")
            if fit_result.peak_vector_gflops:
                source_parts.append(f"peak={fit_result.peak_vector_gflops:.1f} GFLOPS")
            source = ", ".join(source_parts)

            resource_model.set_provenance(
                "simd_vectorization_efficiency",
                EstimationConfidence.calibrated(
                    score=0.85,
                    source=source,
                ),
            )

        if fit_result.ilp_ratio is not None:
            resource_model.set_provenance(
                "register_ilp_ratio",
                EstimationConfidence.calibrated(
                    score=0.80,
                    source=(
                        f"layer2_register_fitter/{fit_result.hardware_name}, "
                        f"ilp_ratio={fit_result.ilp_ratio:.2f}"
                    ),
                ),
            )
