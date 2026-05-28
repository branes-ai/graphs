"""Energy estimates carry an explicit EstimationConfidence (issue #79).

EnergyDescriptor / EnergyReport used to default to UNKNOWN confidence, violating
the estimation rule that every estimate must carry a confidence level. The
EnergyAnalyzer now reports THEORETICAL for its analytical model (process-node
energy coefficients + datasheet/derived TDP), and CALIBRATED only when the
caller supplies a confidence backed by a measured calibration profile -- which
UnifiedAnalyzer derives from the mapper's `.calibration` (the V4 RAPL/NVML
baseline validates a measured SKU's energy end-to-end).
"""

from __future__ import annotations

import contextlib
import io
from types import SimpleNamespace

import torch
import torch.nn as nn

from graphs.core.confidence import ConfidenceLevel, EstimationConfidence
from graphs.core.structures import OperationType, SubgraphDescriptor
from graphs.estimation.energy import EnergyAnalyzer
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


def test_default_confidence_is_theoretical_not_unknown():
    """The analytical energy estimate is THEORETICAL -- never the UNKNOWN
    default that violated the estimation rule."""
    hw = create_i7_12700k_mapper().resource_model
    e = EnergyAnalyzer(hw, precision=P)
    report = e.analyze(subgraphs=[_matmul_sg(1, 512, 512)], latencies=[0.001])
    assert report.confidence.level == ConfidenceLevel.THEORETICAL
    assert report.energy_descriptors[0].confidence.level == ConfidenceLevel.THEORETICAL
    assert report.confidence.source  # non-empty, describes the model basis


def test_supplied_calibrated_confidence_is_honored():
    """A caller that knows the SKU is measured can pass CALIBRATED."""
    hw = create_i7_12700k_mapper().resource_model
    e = EnergyAnalyzer(
        hw, precision=P,
        confidence=EstimationConfidence.calibrated(source="i7 RAPL baseline"),
    )
    report = e.analyze(subgraphs=[_matmul_sg(1, 512, 512)], latencies=[0.001])
    assert report.confidence.level == ConfidenceLevel.CALIBRATED
    assert report.energy_descriptors[0].confidence.level == ConfidenceLevel.CALIBRATED


def test_report_confidence_is_worst_case_across_descriptors():
    """With genuinely mixed per-subgraph confidences, the report resolves to the
    worst (lowest-score) one. Force a CALIBRATED then THEORETICAL descriptor via
    a per-call _resolve_confidence override to exercise the wired aggregation."""
    hw = create_i7_12700k_mapper().resource_model
    e = EnergyAnalyzer(hw, precision=P)
    mixed = iter([
        EstimationConfidence.calibrated(source="measured"),
        EstimationConfidence.theoretical(source="model"),
    ])
    e._resolve_confidence = lambda: next(mixed)  # 1st sg CALIBRATED, 2nd THEORETICAL
    report = e.analyze(
        subgraphs=[_matmul_sg(1, 512, 512), _matmul_sg(1, 1024, 1024)],
        latencies=[0.001, 0.002],
    )
    levels = {d.confidence.level for d in report.energy_descriptors}
    assert levels == {ConfidenceLevel.CALIBRATED, ConfidenceLevel.THEORETICAL}
    assert report.confidence.level == ConfidenceLevel.THEORETICAL  # worst-case wins


def test_unified_analyzer_end_to_end_is_theoretical():
    """End-to-end energy report (no calibration on the mapper) is THEORETICAL,
    not UNKNOWN -- the headline #79 fix."""
    model = nn.Sequential(nn.Linear(512, 512)).eval()
    inp = torch.randn(1, 512)
    a = UnifiedAnalyzer()
    with contextlib.redirect_stdout(io.StringIO()):
        r = a.analyze_model_with_custom_hardware(
            model=model, input_tensor=inp, model_name="lin",
            hardware_mapper=create_i7_12700k_mapper(), precision=P,
        )
    assert r.energy_report.confidence.level == ConfidenceLevel.THEORETICAL


class _StubCalibration:
    """Minimal HardwareCalibration stand-in: enough for the CPU mapper's
    calibration path (get_efficiency / measured_bandwidth_gbps) plus the
    metadata.hardware_name the confidence source reads."""
    measured_bandwidth_gbps = 60.0
    metadata = SimpleNamespace(hardware_name="i7-12700K")

    def get_efficiency(self, *_args, **_kwargs):
        return 0.5


def test_calibration_profile_yields_calibrated():
    """When the mapper carries a measured calibration profile, UnifiedAnalyzer
    reports CALIBRATED (its energy is validated against the measured baseline)."""
    model = nn.Sequential(nn.Linear(512, 512)).eval()
    inp = torch.randn(1, 512)
    mapper = create_i7_12700k_mapper()
    mapper.calibration = _StubCalibration()
    a = UnifiedAnalyzer()
    with contextlib.redirect_stdout(io.StringIO()):
        r = a.analyze_model_with_custom_hardware(
            model=model, input_tensor=inp, model_name="lin",
            hardware_mapper=mapper, precision=P,
        )
    assert r.energy_report.confidence.level == ConfidenceLevel.CALIBRATED
    assert "i7-12700K" in r.energy_report.confidence.source
