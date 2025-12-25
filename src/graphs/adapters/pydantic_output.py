"""Convert UnifiedAnalysisResult to Pydantic schemas for agentic workflows.

This module provides adapters that convert internal analysis results
to verdict-first Pydantic models from embodied-schemas.

The verdict-first pattern ensures LLMs can trust tool outputs:
- verdict: PASS/FAIL/PARTIAL/UNKNOWN
- confidence: high/medium/low
- summary: One sentence explaining what was checked

Usage:
    from graphs.analysis.unified_analyzer import UnifiedAnalyzer
    from graphs.adapters import convert_to_pydantic

    analyzer = UnifiedAnalyzer()
    result = analyzer.analyze_model('resnet18', 'H100')

    # Convert to Pydantic with constraint checking
    pydantic_result = convert_to_pydantic(
        result,
        constraint_metric='latency',
        constraint_threshold=10.0  # ms
    )

    print(pydantic_result.verdict)  # PASS or FAIL
    print(pydantic_result.summary)  # Human-readable explanation
"""

from datetime import datetime
from typing import Optional, Literal, TYPE_CHECKING

# Lazy imports to make embodied-schemas optional
if TYPE_CHECKING:
    from embodied_schemas import (
        Verdict,
        Confidence,
        Bottleneck,
        RooflineResult,
        EnergyResult,
        MemoryResult,
        ConcurrencyResult,
        SubgraphBreakdown,
        GraphAnalysisResult,
    )

from graphs.analysis.unified_analyzer import UnifiedAnalysisResult
from graphs.analysis.roofline import RooflineReport
from graphs.analysis.energy import EnergyReport
from graphs.analysis.memory import MemoryReport


def _import_schemas():
    """Import embodied-schemas (raises ImportError if not installed)."""
    try:
        from embodied_schemas import (
            Verdict,
            Confidence,
            Bottleneck,
            RooflineResult,
            EnergyResult,
            MemoryResult,
            ConcurrencyResult,
            SubgraphBreakdown,
            GraphAnalysisResult,
        )
        return {
            'Verdict': Verdict,
            'Confidence': Confidence,
            'Bottleneck': Bottleneck,
            'RooflineResult': RooflineResult,
            'EnergyResult': EnergyResult,
            'MemoryResult': MemoryResult,
            'ConcurrencyResult': ConcurrencyResult,
            'SubgraphBreakdown': SubgraphBreakdown,
            'GraphAnalysisResult': GraphAnalysisResult,
        }
    except ImportError:
        raise ImportError(
            "embodied-schemas is required for Pydantic output. "
            "Install with: pip install embodied-schemas "
            "or: pip install graphs[schemas]"
        )


def make_verdict(
    actual: float,
    threshold: float,
    metric: str,
    lower_is_better: bool = True
) -> tuple:
    """Determine verdict and margin from actual vs threshold.

    Args:
        actual: Actual measured/estimated value
        threshold: Required threshold value
        metric: Metric name (for summary generation)
        lower_is_better: If True, actual < threshold is PASS (e.g., latency)
                        If False, actual > threshold is PASS (e.g., throughput)

    Returns:
        Tuple of (verdict, margin_pct, summary)
    """
    schemas = _import_schemas()
    Verdict = schemas['Verdict']

    if lower_is_better:
        passes = actual <= threshold
        margin_pct = ((threshold - actual) / threshold) * 100
    else:
        passes = actual >= threshold
        margin_pct = ((actual - threshold) / threshold) * 100

    if passes:
        verdict = Verdict.PASS
        if margin_pct > 20:
            summary = f"{metric} of {actual:.1f} is well under {threshold:.1f} target ({margin_pct:.0f}% headroom)"
        else:
            summary = f"{metric} of {actual:.1f} meets {threshold:.1f} target ({margin_pct:.0f}% headroom)"
    else:
        verdict = Verdict.FAIL
        summary = f"{metric} of {actual:.1f} exceeds {threshold:.1f} target by {abs(margin_pct):.0f}%"

    return verdict, margin_pct, summary


def convert_roofline_to_pydantic(
    roofline: RooflineReport,
    total_latency_ms: float
) -> "RooflineResult":
    """Convert RooflineReport to Pydantic RooflineResult.

    Args:
        roofline: Internal RooflineReport from graphs
        total_latency_ms: Total latency in milliseconds

    Returns:
        Pydantic RooflineResult for embodied-schemas
    """
    schemas = _import_schemas()
    RooflineResult = schemas['RooflineResult']
    Bottleneck = schemas['Bottleneck']

    # Determine dominant bottleneck
    if roofline.num_compute_bound > roofline.num_memory_bound:
        bottleneck = Bottleneck.COMPUTE_BOUND
    elif roofline.num_memory_bound > roofline.num_compute_bound:
        bottleneck = Bottleneck.MEMORY_BOUND
    else:
        bottleneck = Bottleneck.BALANCED

    latency_s = total_latency_ms / 1000.0

    # Calculate achieved performance from attained_flops in latency descriptors
    # or estimate from compute_time
    total_attained_flops = sum(
        lat.attained_flops for lat in roofline.latencies
        if hasattr(lat, 'attained_flops') and lat.attained_flops > 0
    )

    if total_attained_flops > 0 and latency_s > 0:
        achieved_flops = total_attained_flops
    else:
        # Fallback: estimate from utilization
        achieved_flops = roofline.peak_flops * roofline.average_flops_utilization

    # Estimate bandwidth from memory_time
    total_memory_time = sum(lat.memory_time for lat in roofline.latencies)
    if total_memory_time > 0:
        achieved_bandwidth = roofline.peak_bandwidth * roofline.average_bandwidth_utilization
    else:
        achieved_bandwidth = 0.0

    # Use average arithmetic intensity from roofline points if available
    if hasattr(roofline, 'roofline_points') and roofline.roofline_points:
        avg_ai = sum(p[0] for p in roofline.roofline_points) / len(roofline.roofline_points)
    elif roofline.latencies:
        avg_ai = sum(lat.arithmetic_intensity for lat in roofline.latencies) / len(roofline.latencies)
    else:
        avg_ai = roofline.arithmetic_intensity_breakpoint

    return RooflineResult(
        latency_ms=total_latency_ms,
        bottleneck=bottleneck,
        utilization_pct=roofline.average_flops_utilization * 100,
        arithmetic_intensity=avg_ai,
        peak_flops=roofline.peak_flops,
        peak_bandwidth_gbps=roofline.peak_bandwidth / 1e9,
        achieved_flops=achieved_flops,
        achieved_bandwidth_gbps=achieved_bandwidth / 1e9,
        ridge_point=roofline.arithmetic_intensity_breakpoint,
    )


def convert_energy_to_pydantic(
    energy: EnergyReport,
    hardware_tdp_w: Optional[float] = None
) -> "EnergyResult":
    """Convert EnergyReport to Pydantic EnergyResult.

    Args:
        energy: Internal EnergyReport from graphs
        hardware_tdp_w: Hardware TDP in watts (optional)

    Returns:
        Pydantic EnergyResult for embodied-schemas
    """
    schemas = _import_schemas()
    EnergyResult = schemas['EnergyResult']

    # Calculate efficiency if we have latency
    efficiency = None
    if energy.total_latency_s > 0 and energy.average_power_w > 0:
        # Approximate GFLOPS from energy if we had FLOPs data
        # For now, use efficiency from the report
        efficiency = energy.average_efficiency

    return EnergyResult(
        total_energy_mj=energy.total_energy_mj,
        compute_energy_mj=energy.compute_energy_j * 1000,
        memory_energy_mj=energy.memory_energy_j * 1000,
        static_energy_mj=energy.static_energy_j * 1000,
        average_power_w=energy.average_power_w,
        peak_power_w=energy.peak_power_w if energy.peak_power_w > 0 else None,
        tdp_w=hardware_tdp_w,
        energy_efficiency_gflops_per_w=efficiency,
        power_gating_enabled=energy.power_gating_enabled,
        power_gating_savings_mj=(
            energy.total_power_gating_savings_j * 1000
            if energy.power_gating_enabled else None
        ),
    )


def convert_memory_to_pydantic(
    memory: MemoryReport,
) -> "MemoryResult":
    """Convert MemoryReport to Pydantic MemoryResult.

    Args:
        memory: Internal MemoryReport from graphs

    Returns:
        Pydantic MemoryResult for embodied-schemas
    """
    schemas = _import_schemas()
    MemoryResult = schemas['MemoryResult']

    # Calculate workspace (total - weights - activations)
    workspace_bytes = max(
        0,
        memory.peak_memory_bytes - memory.weight_memory_bytes - memory.activation_memory_bytes
    )

    # Memory utilization
    utilization = None
    if memory.device_memory_bytes > 0:
        utilization = (memory.peak_memory_bytes / memory.device_memory_bytes) * 100

    return MemoryResult(
        peak_memory_mb=memory.peak_memory_mb,
        weights_mb=memory.weight_memory_bytes / (1024 * 1024),
        activations_mb=memory.activation_memory_bytes / (1024 * 1024),
        workspace_mb=workspace_bytes / (1024 * 1024) if workspace_bytes > 0 else None,
        fits_in_l2=memory.fits_in_l2_cache,
        fits_in_device_memory=memory.fits_on_device,
        l2_cache_mb=memory.l2_cache_size_bytes / (1024 * 1024) if memory.l2_cache_size_bytes > 0 else None,
        device_memory_mb=memory.device_memory_bytes / (1024 * 1024) if memory.device_memory_bytes > 0 else None,
        memory_utilization_pct=utilization,
    )


def convert_to_pydantic(
    result: UnifiedAnalysisResult,
    constraint_metric: Optional[str] = None,
    constraint_threshold: Optional[float] = None,
) -> "GraphAnalysisResult":
    """Convert UnifiedAnalysisResult to verdict-first Pydantic GraphAnalysisResult.

    This is the main entry point for converting analysis results to
    Pydantic schemas for use in agentic workflows.

    Args:
        result: Internal UnifiedAnalysisResult from UnifiedAnalyzer
        constraint_metric: Optional metric to check: 'latency', 'power', 'memory', 'energy'
        constraint_threshold: Required threshold for the constraint metric

    Returns:
        Pydantic GraphAnalysisResult with verdict-first output

    Raises:
        ImportError: If embodied-schemas is not installed
        ValueError: If required reports are missing
    """
    schemas = _import_schemas()
    Verdict = schemas['Verdict']
    Confidence = schemas['Confidence']
    GraphAnalysisResult = schemas['GraphAnalysisResult']
    ConcurrencyResult = schemas['ConcurrencyResult']
    SubgraphBreakdown = schemas['SubgraphBreakdown']
    Bottleneck = schemas['Bottleneck']

    # Validate required reports exist
    if result.roofline_report is None:
        raise ValueError("RooflineReport is required for Pydantic conversion")
    if result.energy_report is None:
        raise ValueError("EnergyReport is required for Pydantic conversion")
    if result.memory_report is None:
        raise ValueError("MemoryReport is required for Pydantic conversion")

    # Convert sub-reports
    roofline = convert_roofline_to_pydantic(
        result.roofline_report,
        result.total_latency_ms
    )
    energy = convert_energy_to_pydantic(
        result.energy_report,
        hardware_tdp_w=result.hardware.tdp if hasattr(result.hardware, 'tdp') else None
    )
    memory = convert_memory_to_pydantic(result.memory_report)

    # Convert concurrency if available
    concurrency = None
    if result.concurrency_report is not None:
        cr = result.concurrency_report
        concurrency = ConcurrencyResult(
            data_parallelism=cr.batch_parallelism if hasattr(cr, 'batch_parallelism') else None,
            tensor_parallelism=cr.spatial_parallelism if hasattr(cr, 'spatial_parallelism') else None,
            pipeline_parallelism=cr.pipeline_parallelism if hasattr(cr, 'pipeline_parallelism') else None,
        )

    # Convert subgraph breakdown if available
    subgraphs = None
    total_subgraphs = None
    fusion_ratio = None

    if result.partition_report is not None:
        pr = result.partition_report
        total_subgraphs = len(pr.subgraphs)

        # Compute fusion ratio if we have original op count
        if hasattr(pr, 'original_op_count') and pr.original_op_count > 0:
            fused_ops = sum(len(sg.fused_ops) for sg in pr.subgraphs if hasattr(sg, 'fused_ops'))
            fusion_ratio = fused_ops / pr.original_op_count if fused_ops > 0 else 0.0

        # Build subgraph breakdown if we have latency info
        if result.roofline_report and result.energy_report:
            subgraphs = []
            for i, sg in enumerate(pr.subgraphs):
                if i < len(result.roofline_report.latencies) and i < len(result.energy_report.energy_descriptors):
                    lat = result.roofline_report.latencies[i]
                    eng = result.energy_report.energy_descriptors[i]

                    # Determine bottleneck
                    if lat.bottleneck == 'compute':
                        bn = Bottleneck.COMPUTE_BOUND
                    elif lat.bottleneck == 'memory':
                        bn = Bottleneck.MEMORY_BOUND
                    else:
                        bn = Bottleneck.BALANCED

                    subgraphs.append(SubgraphBreakdown(
                        subgraph_id=f"sg_{i}",
                        op_types=sg.fused_ops if hasattr(sg, 'fused_ops') else [sg.dominant_op],
                        flops=sg.flops,
                        bytes_transferred=sg.memory_traffic,
                        latency_ms=lat.actual_latency * 1000,
                        energy_mj=eng.total_energy_j * 1000,
                        bottleneck=bn,
                    ))

    # Determine verdict based on constraint
    verdict = Verdict.UNKNOWN
    confidence = Confidence.MEDIUM
    summary = f"Analysis of {result.display_name} on {result.hardware_display_name}"
    constraint_actual = None
    margin_pct = None
    suggestions = []
    warnings = list(result.validation_warnings)

    if constraint_metric and constraint_threshold is not None:
        if constraint_metric == 'latency':
            actual = result.total_latency_ms
            verdict, margin_pct, summary = make_verdict(
                actual, constraint_threshold, f"Latency", lower_is_better=True
            )
            constraint_actual = actual
            confidence = Confidence.HIGH if result.hardware_allocation else Confidence.MEDIUM

            if verdict == Verdict.FAIL:
                suggestions.append(f"Consider faster hardware or smaller model")
                if result.precision.name == 'FP32':
                    suggestions.append("Try FP16 precision for ~2x speedup")

        elif constraint_metric == 'power':
            actual = result.energy_report.average_power_w
            verdict, margin_pct, summary = make_verdict(
                actual, constraint_threshold, f"Average power", lower_is_better=True
            )
            constraint_actual = actual
            confidence = Confidence.MEDIUM

            if verdict == Verdict.FAIL:
                suggestions.append("Consider lower-power hardware or power gating")

        elif constraint_metric == 'memory':
            actual = result.peak_memory_mb
            verdict, margin_pct, summary = make_verdict(
                actual, constraint_threshold, f"Peak memory", lower_is_better=True
            )
            constraint_actual = actual
            confidence = Confidence.HIGH

            if verdict == Verdict.FAIL:
                suggestions.append("Consider gradient checkpointing or smaller batch size")

        elif constraint_metric == 'energy':
            actual = result.energy_per_inference_mj
            verdict, margin_pct, summary = make_verdict(
                actual, constraint_threshold, f"Energy per inference", lower_is_better=True
            )
            constraint_actual = actual
            confidence = Confidence.MEDIUM

            if verdict == Verdict.FAIL:
                suggestions.append("Consider more efficient hardware or quantization")
    else:
        # No constraint - just report analysis complete
        verdict = Verdict.PASS
        confidence = Confidence.HIGH if result.hardware_allocation else Confidence.MEDIUM
        summary = (
            f"{result.display_name} on {result.hardware_display_name}: "
            f"{result.total_latency_ms:.1f}ms latency, "
            f"{result.energy_per_inference_mj:.1f}mJ/inference"
        )

    # Add recommendations from the result
    recommendations = result._generate_recommendations()
    suggestions.extend(recommendations)

    return GraphAnalysisResult(
        # Verdict
        verdict=verdict,
        confidence=confidence,
        summary=summary,

        # Metadata
        model_id=result.model_name,
        hardware_id=result.hardware_name,
        batch_size=result.batch_size,
        precision=result.precision.name.lower(),
        timestamp=datetime.fromisoformat(result.analysis_timestamp),
        analyzer_version="1.0.0",

        # Key metrics
        latency_ms=result.total_latency_ms,
        throughput_fps=result.throughput_fps,
        energy_per_inference_mj=result.energy_per_inference_mj,
        peak_memory_mb=result.peak_memory_mb,

        # Detailed breakdowns
        roofline=roofline,
        energy=energy,
        memory=memory,
        concurrency=concurrency,

        # Subgraph breakdown
        subgraphs=subgraphs,
        total_subgraphs=total_subgraphs,
        fusion_ratio=fusion_ratio,

        # Constraint checking
        constraint_metric=constraint_metric,
        constraint_threshold=constraint_threshold,
        constraint_actual=constraint_actual,
        constraint_margin_pct=margin_pct,

        # Recommendations
        suggestions=suggestions,
        warnings=warnings,
    )
