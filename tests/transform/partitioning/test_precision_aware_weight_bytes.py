"""Regression tests for issue #52: precision-aware weight/activation byte counts.

The partitioner used to hardcode 4 bytes per parameter (fp32), so requesting
analysis at fp16 or int8 silently dropped the precision flag. These tests
assert that:

1. ``FusionBasedPartitioner`` weight bytes scale with precision.
2. ``trace_and_partition`` honors the ``precision`` keyword argument.
3. ``GraphPartitioner`` (legacy path referenced in the issue) does the same.
"""

import pytest
import torch
import torch.nn as nn
import torchvision.models as tvm
from torch.fx import symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp

from graphs.frontends import trace_and_partition
from graphs.hardware.resource_model import Precision, precision_bytes_per_element
from graphs.transform.partitioning import FusionBasedPartitioner, GraphPartitioner


@pytest.mark.parametrize(
    "precision,expected_bpe",
    [
        (Precision.FP32, 4),
        (Precision.FP16, 2),
        (Precision.BF16, 2),
        (Precision.INT8, 1),
        (Precision.FP8, 1),
        (Precision.FP8_E4M3, 1),
        (Precision.FP8_E5M2, 1),
        (Precision.INT4, 0.5),
        (Precision.FP4, 0.5),
    ],
)
def test_precision_bytes_per_element(precision, expected_bpe):
    assert precision_bytes_per_element(precision) == expected_bpe


def _trace(model: nn.Module, x: torch.Tensor):
    fx = symbolic_trace(model)
    ShapeProp(fx).propagate(x)
    return fx


def test_fusion_partitioner_weight_bytes_scale_with_precision():
    """ViT-B/16 has ~86M parameters; weight bytes must scale with precision."""
    model = tvm.vit_b_16(weights=None).eval()
    x = torch.randn(1, 3, 224, 224)
    fx = _trace(model, x)

    sizes = {}
    for p in [Precision.FP32, Precision.FP16, Precision.INT8, Precision.INT4]:
        partitioner = FusionBasedPartitioner(precision=p)
        report = partitioner.partition(fx)
        sizes[p] = sum(sg.total_weight_bytes for sg in report.subgraphs)

    # Match the orders of magnitude called out in issue #52
    assert 300e6 <= sizes[Precision.FP32] <= 360e6, sizes
    assert 150e6 <= sizes[Precision.FP16] <= 180e6, sizes
    assert 75e6 <= sizes[Precision.INT8] <= 95e6, sizes
    assert 35e6 <= sizes[Precision.INT4] <= 50e6, sizes

    # Exact ratios w.r.t. fp32 within rounding noise
    assert sizes[Precision.FP16] == pytest.approx(sizes[Precision.FP32] / 2, rel=1e-3)
    assert sizes[Precision.INT8] == pytest.approx(sizes[Precision.FP32] / 4, rel=1e-3)
    assert sizes[Precision.INT4] == pytest.approx(sizes[Precision.FP32] / 8, rel=1e-3)


def test_trace_and_partition_threads_precision():
    """The frontend entry point must forward precision into the partitioner."""
    model = tvm.resnet18(weights=None).eval()
    x = torch.randn(1, 3, 224, 224)

    _, fp32_report = trace_and_partition(model, x, precision=Precision.FP32)
    _, int8_report = trace_and_partition(model, x, precision=Precision.INT8)

    fp32_w = sum(sg.total_weight_bytes for sg in fp32_report.subgraphs)
    int8_w = sum(sg.total_weight_bytes for sg in int8_report.subgraphs)

    assert fp32_w > 0
    assert int8_w == pytest.approx(fp32_w / 4, rel=1e-3)


def test_graph_partitioner_weight_bytes_scale_with_precision():
    """Legacy partitioner referenced directly in the issue must also honor precision."""

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=True)
            self.linear = nn.Linear(64 * 32 * 32, 10, bias=True)

        def forward(self, x):
            x = self.conv(x)
            return self.linear(x.flatten(1))

    model = Tiny().eval()
    x = torch.randn(1, 3, 32, 32)
    fx = _trace(model, x)

    fp32 = GraphPartitioner(precision=Precision.FP32).partition(fx)
    int8 = GraphPartitioner(precision=Precision.INT8).partition(fx)

    fp32_w = sum(sg.total_weight_bytes for sg in fp32.subgraphs)
    int8_w = sum(sg.total_weight_bytes for sg in int8.subgraphs)

    assert fp32_w > 0
    # int8 weights occupy 1 byte per element vs 4 for fp32 -> exact 4x scaling
    assert int8_w * 4 == fp32_w


def test_subbyte_precision_uses_ceil_not_round():
    """A sub-byte precision must not floor a single-element tensor to 0 bytes.

    Banker's rounding (Python's ``round``) maps 0.5 -> 0, which would silently
    lose tensors at int4/fp4. The partitioner uses ``math.ceil`` to model
    packed-storage padding (a 1-element int4 tensor still occupies 1 byte).
    """

    class OneParamConv(nn.Module):
        def __init__(self):
            super().__init__()
            # 1 input channel, 1 output channel, 1x1 kernel, no bias -> 1 param
            self.conv = nn.Conv2d(1, 1, kernel_size=1, bias=False)

        def forward(self, x):
            return self.conv(x)

    model = OneParamConv().eval()
    x = torch.randn(1, 1, 8, 8)
    fx = _trace(model, x)

    int4 = GraphPartitioner(precision=Precision.INT4).partition(fx)
    weight_bytes = sum(sg.total_weight_bytes for sg in int4.subgraphs)
    # 1 param * 0.5 bytes/elem = 0.5 -> ceil = 1, round() would give 0.
    assert weight_bytes >= 1


def test_default_precision_is_fp32_backward_compatible():
    """No-arg constructors must still produce fp32 byte counts."""
    model = tvm.resnet18(weights=None).eval()
    x = torch.randn(1, 3, 224, 224)
    fx = _trace(model, x)

    default = FusionBasedPartitioner().partition(fx)
    explicit_fp32 = FusionBasedPartitioner(precision=Precision.FP32).partition(fx)

    default_w = sum(sg.total_weight_bytes for sg in default.subgraphs)
    explicit_w = sum(sg.total_weight_bytes for sg in explicit_fp32.subgraphs)

    assert default_w == explicit_w
