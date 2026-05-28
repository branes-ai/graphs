"""Latency + memory estimates carry an explicit EstimationConfidence.

The #79 follow-up audit: LatencyDescriptor (roofline) and MemoryDescriptor had
the same UNKNOWN-default gap that #79 fixed for energy. RooflineAnalyzer reports
THEORETICAL by default and CALIBRATED when constructed with measured calibration
(is_calibrated); memory footprint is analytical (computed from tensor shapes),
so it is THEORETICAL. Both reports aggregate worst-case across descriptors.
"""

from __future__ import annotations

import contextlib
import io

import torch
import torch.nn as nn

from graphs.core.confidence import ConfidenceLevel, EstimationConfidence
from graphs.core.structures import OperationType, SubgraphDescriptor
from graphs.estimation.roofline import RooflineAnalyzer
from graphs.estimation.unified_analyzer import UnifiedAnalyzer
from graphs.hardware.mappers.cpu import create_i7_12700k_mapper
from graphs.hardware.resource_model import Precision

P = Precision.INT8


def _matmul_sg(M: int, K: int, N: int, bpe: int = 1) -> SubgraphDescriptor:
    return SubgraphDescriptor(
        subgraph_id=0,
        node_ids=["mm"], node_names=["mm"],
        operation_types=[OperationType.MATMUL],
        fusion_pattern="matmul",
        total_flops=2 * M * K * N,
        total_macs=M * K * N,
        total_input_bytes=(M * K + K * N) * bpe,
        total_output_bytes=M * N * bpe,
        total_weight_bytes=0,
    )


# ---------------------------------------------------------------------------
# Roofline / latency
# ---------------------------------------------------------------------------

def test_latency_default_is_theoretical():
    hw = create_i7_12700k_mapper().resource_model
    r = RooflineAnalyzer(hw, precision=P)
    lat = r._analyze_subgraph(_matmul_sg(1, 512, 512))
    assert lat.confidence.level == ConfidenceLevel.THEORETICAL
    assert lat.confidence.source


def test_latency_is_calibrated_when_calibration_supplied():
    hw = create_i7_12700k_mapper().resource_model
    r = RooflineAnalyzer(hw, precision=P, efficiency_factor=0.8)
    assert r.is_calibrated
    lat = r._analyze_subgraph(_matmul_sg(1, 512, 512))
    assert lat.confidence.level == ConfidenceLevel.CALIBRATED


def test_roofline_report_confidence_is_worst_case():
    """Mixed per-subgraph confidences -> report resolves to the worst."""
    hw = create_i7_12700k_mapper().resource_model
    r = RooflineAnalyzer(hw, precision=P)
    mixed = iter([
        EstimationConfidence.calibrated(source="measured"),
        EstimationConfidence.theoretical(source="specs"),
    ])
    r._resolve_confidence = lambda: next(mixed)
    report = r.analyze(subgraphs=[_matmul_sg(1, 512, 512), _matmul_sg(1, 1024, 1024)])
    levels = {l.confidence.level for l in report.latencies}
    assert levels == {ConfidenceLevel.CALIBRATED, ConfidenceLevel.THEORETICAL}
    assert report.confidence.level == ConfidenceLevel.THEORETICAL


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------

def test_memory_reports_are_theoretical_end_to_end():
    """Both roofline and memory reports are THEORETICAL (no UNKNOWN) via the
    full UnifiedAnalyzer path with an uncalibrated mapper."""
    model = nn.Sequential(nn.Linear(512, 512)).eval()
    a = UnifiedAnalyzer()
    with contextlib.redirect_stdout(io.StringIO()):
        res = a.analyze_model_with_custom_hardware(
            model=model, input_tensor=torch.randn(1, 512), model_name="lin",
            hardware_mapper=create_i7_12700k_mapper(), precision=P,
        )
    assert res.roofline_report.confidence.level == ConfidenceLevel.THEORETICAL
    assert res.memory_report.confidence.level == ConfidenceLevel.THEORETICAL
    # every per-subgraph memory descriptor is tagged too
    assert all(
        d.confidence.level == ConfidenceLevel.THEORETICAL
        for d in res.memory_report.subgraph_descriptors
    )
