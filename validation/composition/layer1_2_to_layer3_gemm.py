"""
Composition Check: Layer 1+2 -> Layer 3 GEMM Throughput Prediction

Refines the Layer 1 -> GEMM prediction (10% tolerance) by
incorporating the Layer 2 SIMD efficiency measurement. The
prediction becomes:

    predicted = layer1_fma_gflops * simd_efficiency * gemm_overhead

where simd_efficiency comes from the Layer 2 width sweep (replacing
the hardcoded 0.70) and gemm_overhead accounts for GEMM-specific
overhead beyond SIMD (loop control, address calculation, ~0.85-0.95
for well-optimized BLAS).

Tolerance: 8% (tighter than Layer 1's 10% because Layer 2 captures
more of the real operand-delivery cost).
"""

from __future__ import annotations

from typing import List, Optional

from graphs.benchmarks.schema import BenchmarkResult, LayerTag


def predict_gemm_from_layer1_2(
    layer1_results: List[BenchmarkResult],
    layer2_results: List[BenchmarkResult],
    precision: str = "fp32",
    gemm_overhead: float = 0.90,
) -> Optional[float]:
    """
    Predict large-GEMM GFLOPS from Layer 1 FMA rate + Layer 2 SIMD efficiency.

    Args:
        layer1_results: FMA-rate results (LayerTag.ALU)
        layer2_results: SIMD width sweep results (LayerTag.REGISTER_SIMD)
        precision: which precision to predict for
        gemm_overhead: GEMM-specific overhead beyond SIMD (loop control, etc.)

    Returns:
        Predicted GFLOPS, or None if insufficient data
    """
    # Get Layer 1 FMA throughput
    layer1_gflops = None
    for r in layer1_results:
        if r.layer is LayerTag.ALU and r.success and r.precision == precision:
            layer1_gflops = r.gflops
            break

    if layer1_gflops is None:
        return None

    # Get SIMD efficiency from Layer 2 via the fitter (single source of truth)
    simd_efficiency = _extract_simd_efficiency(layer2_results)
    if simd_efficiency is None:
        simd_efficiency = 0.70  # fallback to analytical default

    return layer1_gflops * simd_efficiency * gemm_overhead


def _extract_simd_efficiency(
    layer2_results: List[BenchmarkResult],
) -> Optional[float]:
    """Extract SIMD efficiency from Layer 2 results via the fitter."""
    from graphs.calibration.fitters.layer2_register_fitter import Layer2RegisterFitter
    fitter = Layer2RegisterFitter()
    fit = fitter.fit(layer2_results)
    return fit.simd_efficiency


def check_layer1_2_to_gemm(
    layer1_results: List[BenchmarkResult],
    layer2_results: List[BenchmarkResult],
    measured_gemm_gflops: float,
    precision: str = "fp32",
    hardware: str = "unknown",
    gemm_overhead: float = 0.90,
    tolerance: float = 0.08,
) -> "CompositionCheckResult":
    """
    Compare Layer 1+2 GEMM prediction against measured baseline.

    Args:
        layer1_results: Layer 1 FMA-rate results
        layer2_results: Layer 2 SIMD width-sweep results
        measured_gemm_gflops: actual large-GEMM GFLOPS on same SKU
        precision: precision to check
        hardware: SKU name
        gemm_overhead: GEMM overhead factor
        tolerance: max allowed relative error (0.08 = 8%)

    Returns:
        CompositionCheckResult
    """
    from validation.composition.test_layer_composition import (
        CheckStatus,
        CompositionCheckResult,
    )

    predicted = predict_gemm_from_layer1_2(
        layer1_results, layer2_results, precision, gemm_overhead,
    )

    if predicted is None:
        return CompositionCheckResult(
            name=f"layer1_2_to_gemm_{precision}",
            hardware=hardware,
            predicts_layer=LayerTag.COMPOSITE,
            from_layers=[LayerTag.ALU, LayerTag.REGISTER_SIMD],
            status=CheckStatus.SKIPPED,
            tolerance=tolerance,
            details="Insufficient Layer 1 or Layer 2 data",
        )

    if measured_gemm_gflops <= 0:
        return CompositionCheckResult(
            name=f"layer1_2_to_gemm_{precision}",
            hardware=hardware,
            predicts_layer=LayerTag.COMPOSITE,
            from_layers=[LayerTag.ALU, LayerTag.REGISTER_SIMD],
            status=CheckStatus.SKIPPED,
            tolerance=tolerance,
            details="No measured GEMM baseline available",
        )

    relative_error = abs(predicted - measured_gemm_gflops) / measured_gemm_gflops
    passed = relative_error <= tolerance

    return CompositionCheckResult(
        name=f"layer1_2_to_gemm_{precision}",
        hardware=hardware,
        predicts_layer=LayerTag.COMPOSITE,
        from_layers=[LayerTag.ALU, LayerTag.REGISTER_SIMD],
        status=CheckStatus.PASSED if passed else CheckStatus.FAILED,
        max_relative_error=relative_error,
        tolerance=tolerance,
        details=(
            f"predicted={predicted:.1f} GFLOPS, "
            f"measured={measured_gemm_gflops:.1f} GFLOPS, "
            f"error={relative_error*100:.1f}%"
        ),
    )
