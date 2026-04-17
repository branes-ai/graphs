"""
Composition Check: Layer 1 ALU -> Layer 3 GEMM Throughput Prediction

Validates that the per-ALU throughput measured in Layer 1 (FMA-rate
benchmark) predicts the large-GEMM throughput (Layer 3) within
tolerance on the same SKU.

Prediction model:
    predicted_gemm_gflops = ops_per_clock_per_core * clock_hz * num_cores
                          * gemm_efficiency_factor

    gemm_efficiency_factor accounts for the fact that a real GEMM
    has overhead beyond pure FMA (address calculation, loop control,
    cache management). For large GEMMs (M=N=K>=2048) on modern CPUs,
    this factor is typically 0.60-0.80.

Registration:
    This module is imported by the composition test runner. To
    register the check, import this module after Layer 1 results
    exist on disk.

Usage (manual):
    python validation/composition/layer1_to_layer3_gemm.py
"""

from __future__ import annotations

from typing import List, Optional

from graphs.benchmarks.schema import BenchmarkResult, LayerTag


def predict_gemm_gflops_from_layer1(
    layer1_results: List[BenchmarkResult],
    precision: str = "fp32",
    gemm_efficiency: float = 0.70,
) -> Optional[float]:
    """
    Predict large-GEMM GFLOPS from Layer 1 FMA throughput.

    Args:
        layer1_results: FMA-rate benchmark results (LayerTag.ALU)
        precision: which precision to use for the prediction
        gemm_efficiency: ratio of GEMM throughput to raw FMA throughput
                        (0.70 is a reasonable default for large GEMMs
                        on AVX2 CPUs with good BLAS libraries)

    Returns:
        Predicted GFLOPS for large GEMM, or None if no matching result
    """
    for r in layer1_results:
        if (
            r.layer is LayerTag.ALU
            and r.success
            and r.precision == precision
            and r.gflops > 0
        ):
            return r.gflops * gemm_efficiency
    return None


def check_layer1_to_gemm(
    layer1_results: List[BenchmarkResult],
    measured_gemm_gflops: float,
    precision: str = "fp32",
    hardware: str = "unknown",
    gemm_efficiency: float = 0.70,
    tolerance: float = 0.10,
) -> "CompositionCheckResult":
    """
    Compare predicted GEMM throughput against measured.

    Args:
        layer1_results: Layer 1 FMA-rate results
        measured_gemm_gflops: actual large-GEMM GFLOPS on same SKU
        precision: precision to check
        hardware: SKU name for reporting
        gemm_efficiency: FMA-to-GEMM efficiency factor
        tolerance: max allowed relative error (0.10 = 10%)

    Returns:
        CompositionCheckResult
    """
    from validation.composition.test_layer_composition import (
        CheckStatus,
        CompositionCheckResult,
    )

    predicted = predict_gemm_gflops_from_layer1(
        layer1_results, precision, gemm_efficiency,
    )

    if predicted is None:
        return CompositionCheckResult(
            name=f"layer1_to_gemm_{precision}",
            hardware=hardware,
            predicts_layer=LayerTag.COMPOSITE,
            from_layers=[LayerTag.ALU],
            status=CheckStatus.SKIPPED,
            tolerance=tolerance,
            details=f"No Layer 1 result for precision={precision}",
        )

    if measured_gemm_gflops <= 0:
        return CompositionCheckResult(
            name=f"layer1_to_gemm_{precision}",
            hardware=hardware,
            predicts_layer=LayerTag.COMPOSITE,
            from_layers=[LayerTag.ALU],
            status=CheckStatus.SKIPPED,
            tolerance=tolerance,
            details="No measured GEMM baseline available",
        )

    relative_error = abs(predicted - measured_gemm_gflops) / measured_gemm_gflops
    passed = relative_error <= tolerance

    return CompositionCheckResult(
        name=f"layer1_to_gemm_{precision}",
        hardware=hardware,
        predicts_layer=LayerTag.COMPOSITE,
        from_layers=[LayerTag.ALU],
        status=CheckStatus.PASSED if passed else CheckStatus.FAILED,
        max_relative_error=relative_error,
        tolerance=tolerance,
        details=(
            f"predicted={predicted:.1f} GFLOPS, "
            f"measured={measured_gemm_gflops:.1f} GFLOPS, "
            f"error={relative_error*100:.1f}%"
        ),
    )


if __name__ == "__main__":
    print("Layer 1 -> Layer 3 GEMM composition check")
    print("Run via: python validation/composition/test_layer_composition.py")
    print("This module provides check_layer1_to_gemm() for programmatic use.")
