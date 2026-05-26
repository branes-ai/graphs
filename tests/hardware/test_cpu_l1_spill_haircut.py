"""CPU L1-spill compute-utilization haircut (issue #178 fix #2).

An un-blocked matvec whose working set far exceeds per-core L1 can't sustain
peak vector throughput; the model previously reported ~90% utilization, letting
a 5 W single ARM core out-rank a 30 W Jetson on a batch=1 Linear. The haircut
(`RooflineAnalyzer._cpu_l1_fit_scale`) degrades utilization for L1-spilling
operators, but only on uncalibrated theoretical references (opt-in via
`cpu_l1_spill_haircut`), leaving calibrated/measured CPUs (i7) untouched.
"""

from __future__ import annotations

import contextlib
import io

import pytest
import torch
import torch.nn as nn

from graphs.estimation.roofline import RooflineAnalyzer
from graphs.estimation.unified_analyzer import UnifiedAnalyzer
from graphs.hardware.mappers.cpu import (
    create_ampere_ampereone_192_mapper,
    create_ampere_ampereone_1core_reference_mapper,
    create_arm_neoverse_n2_mapper,
    create_i7_12700k_mapper,
)
from graphs.hardware.mappers.gpu import (
    create_jetson_orin_agx_64gb_mapper,
    create_jetson_orin_nano_8gb_mapper,
)
from graphs.hardware.resource_model import Precision
from validation.model_v4.invariants.kpu import _MatmulShape, _bpe

P = Precision.INT8


def _scale(mapper, shape: _MatmulShape) -> float:
    r = RooflineAnalyzer(mapper.resource_model, precision=P)
    return r._cpu_l1_fit_scale(shape.to_subgraph(_bpe(P)))


def test_opt_in_flag_set_on_references_not_on_i7():
    for mk in (create_ampere_ampereone_192_mapper,
               create_ampere_ampereone_1core_reference_mapper,
               create_arm_neoverse_n2_mapper):
        assert mk().resource_model.cpu_l1_spill_haircut is True
    assert create_i7_12700k_mapper().resource_model.cpu_l1_spill_haircut is False


def test_l1_spilling_matvec_haircut_lands_in_band():
    """1-core ARM matvec (4 MB weights, 64x L1) -> ~30-50% utilization."""
    mapper = create_ampere_ampereone_1core_reference_mapper()
    scale = _scale(mapper, _MatmulShape(1, 2048, 2048))
    assert scale < 1.0
    r = RooflineAnalyzer(mapper.resource_model, precision=P)
    lat = r._analyze_subgraph(_MatmulShape(1, 2048, 2048).to_subgraph(_bpe(P)))
    assert 0.30 <= lat.flops_utilization <= 0.50


def test_l1_resident_op_not_haircut():
    """A tiny matmul whose working set fits in L1 keeps full utilization."""
    mapper = create_ampere_ampereone_1core_reference_mapper()
    assert _scale(mapper, _MatmulShape(1, 64, 64)) == pytest.approx(1.0)


def test_i7_opt_out_never_haircut():
    """i7 (calibrated/measured baseline) is not haircut even on a big L1-spill."""
    mapper = create_i7_12700k_mapper()
    assert _scale(mapper, _MatmulShape(1, 2048, 2048)) == pytest.approx(1.0)


def test_non_cpu_not_haircut():
    mapper = create_jetson_orin_nano_8gb_mapper()
    assert _scale(mapper, _MatmulShape(1, 2048, 2048)) == pytest.approx(1.0)


def test_jetson_beats_single_arm_core_on_batch1_linear():
    """#178's headline: with the haircut, a 30 W Jetson out-ranks the 5 W single
    ARM core on batch=1 Linear(2048,2048)+atan -- it lost before the fix."""
    class Atan(nn.Module):
        def forward(self, x):
            return torch.atan(x)

    model = nn.Sequential(nn.Linear(2048, 2048), Atan()).eval()
    inp = torch.randn(1, 2048)
    a = UnifiedAnalyzer()

    def tp(mapper):
        with contextlib.redirect_stdout(io.StringIO()):
            r = a.analyze_model_with_custom_hardware(
                model=model, input_tensor=inp, model_name="lin",
                hardware_mapper=mapper, precision=P,
            )
        return 1000.0 / r.total_latency_ms

    arm1 = tp(create_ampere_ampereone_1core_reference_mapper())
    agx = tp(create_jetson_orin_agx_64gb_mapper())
    nano = tp(create_jetson_orin_nano_8gb_mapper())
    assert agx > arm1, f"AGX Orin ({agx:.0f}) should beat 1-core ARM ({arm1:.0f})"
    assert nano > arm1, f"Orin Nano ({nano:.0f}) should beat 1-core ARM ({arm1:.0f})"
