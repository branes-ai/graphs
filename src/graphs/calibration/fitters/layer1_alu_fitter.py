"""
Layer 1 ALU Fitter - Fit ComputeFabric Coefficients from FMA Benchmarks

Consumes BenchmarkResult objects from the Layer 1 FMA-rate benchmark
and fits:
  - ops_per_clock: measured throughput / sustained_clock_hz / num_cores
  - pJ_per_op: measured energy / total_ops (from RAPL or other PowerMeter)

Writes EstimationConfidence(CALIBRATED) into
HardwareResourceModel.field_provenance for the fitted fields.

Usage:
    from graphs.calibration.fitters.layer1_alu_fitter import Layer1ALUFitter
    fitter = Layer1ALUFitter()
    fit_result = fitter.fit(benchmark_results)
    fitter.apply_to_model(resource_model, fit_result)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from graphs.benchmarks.schema import BenchmarkResult, LayerTag
from graphs.core.confidence import EstimationConfidence


@dataclass
class ALUFitResult:
    """Result of fitting Layer 1 ALU coefficients from measurement."""

    hardware_name: str = ""

    # Per-precision measured throughput (ops/sec)
    measured_throughput: Dict[str, float] = field(default_factory=dict)

    # Per-precision measured pJ/op (None if PowerMeter was unavailable)
    measured_pj_per_op: Dict[str, Optional[float]] = field(default_factory=dict)

    # Derived: ops per clock per core (throughput / clock / cores)
    ops_per_clock_per_core: Dict[str, float] = field(default_factory=dict)

    # Metadata
    num_results_used: int = 0
    precisions_fitted: List[str] = field(default_factory=list)


class Layer1ALUFitter:
    """
    Fit ComputeFabric coefficients from Layer 1 FMA-rate measurements.

    The fitter takes a list of BenchmarkResults (from
    ``run_precision_sweep`` or individual ``run_fma_rate_benchmark``
    calls), all tagged ``LayerTag.ALU``, and produces an
    ``ALUFitResult`` containing per-precision throughput and energy
    coefficients.

    ``apply_to_model`` then writes these into a
    ``HardwareResourceModel``'s ``field_provenance`` as CALIBRATED.
    """

    def fit(
        self,
        results: List[BenchmarkResult],
        sustained_clock_hz: float = 4.5e9,
        num_cores: int = 10,
        hardware_name: str = "",
    ) -> ALUFitResult:
        """
        Fit per-precision ALU coefficients from benchmark results.

        Args:
            results: BenchmarkResults from Layer 1 benchmarks
                     (must have layer==ALU and success==True)
            sustained_clock_hz: all-core sustained clock for ops_per_clock
            num_cores: effective core count (P + 0.6*E for hybrid)
            hardware_name: SKU name for the fit record

        Returns:
            ALUFitResult with per-precision coefficients
        """
        alu_results = [
            r for r in results
            if r.layer is LayerTag.ALU and r.success
        ]

        fit = ALUFitResult(
            hardware_name=hardware_name,
            num_results_used=len(alu_results),
        )

        for r in alu_results:
            prec = r.precision
            throughput = r.throughput_ops_per_sec
            if throughput <= 0:
                continue

            fit.measured_throughput[prec] = throughput
            fit.precisions_fitted.append(prec)

            if sustained_clock_hz > 0 and num_cores > 0:
                fit.ops_per_clock_per_core[prec] = (
                    throughput / sustained_clock_hz / num_cores
                )

            pj = r.extra.get("pj_per_op")
            fit.measured_pj_per_op[prec] = pj

        return fit

    @staticmethod
    def apply_to_model(
        resource_model: object,
        fit_result: ALUFitResult,
    ) -> None:
        """
        Write fitted coefficients into a HardwareResourceModel's
        field_provenance as CALIBRATED.

        Does NOT overwrite the actual ComputeFabric fields (those are
        used by the mapper for peak calculation). Instead, records the
        measured values in provenance so callers can compare measured
        vs. analytical and the aggregate_confidence reflects Layer 1
        calibration status.
        """
        if not hasattr(resource_model, 'set_provenance'):
            return

        for prec in fit_result.precisions_fitted:
            throughput = fit_result.measured_throughput.get(prec)
            ops_clk = fit_result.ops_per_clock_per_core.get(prec)
            pj = fit_result.measured_pj_per_op.get(prec)

            source_parts = [f"layer1_alu_fitter/{fit_result.hardware_name}"]
            if throughput:
                source_parts.append(f"{throughput/1e9:.1f} GOPS")
            if ops_clk:
                source_parts.append(f"{ops_clk:.1f} ops/clk/core")
            if pj:
                source_parts.append(f"{pj:.2f} pJ/op")
            source = ", ".join(source_parts)

            resource_model.set_provenance(
                f"compute_fabric.ops_per_clock.{prec}",
                EstimationConfidence.calibrated(
                    score=0.90,
                    source=source,
                ),
            )

            if pj is not None:
                resource_model.set_provenance(
                    f"energy_per_op.{prec}",
                    EstimationConfidence.calibrated(
                        score=0.85,
                        source=source,
                    ),
                )
