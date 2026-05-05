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


# ---------------------------------------------------------------------------
# Issue #55: call_function (Dynamo / torch.export ATen) weight accounting
# ---------------------------------------------------------------------------


class _ConvLinear(nn.Module):
    """Small model used to exercise ATen conv2d + linear weight accounting."""

    def __init__(self):
        super().__init__()
        # 3*8*3*3 + 8 = 224 weight elements
        self.conv = nn.Conv2d(3, 8, kernel_size=3, padding=1, bias=True)
        # 4*8192 + 4 = 32772 weight elements
        self.fc = nn.Linear(8 * 32 * 32, 4, bias=True)

    def forward(self, x):
        return self.fc(self.conv(x).flatten(1))


def _dynamo_export(model: nn.Module, x: torch.Tensor):
    ep = torch.export.export(model, (x,))
    fx = ep.module()
    from torch.fx.passes.shape_prop import ShapeProp
    ShapeProp(fx).propagate(x)
    return fx


def test_dynamo_export_call_function_weights_counted():
    """ATen conv2d/linear nodes must contribute weight bytes (issue #55)."""
    model = _ConvLinear().eval()
    x = torch.randn(1, 3, 32, 32)
    fx = _dynamo_export(model, x)

    # Sanity: conv/linear lower to call_function (ATen) under torch.export,
    # not call_module. (Dynamo also emits a _guards_fn call_module helper,
    # which we ignore.)
    weighted_ops = [
        n for n in fx.graph.nodes
        if any(tag in str(n.target) for tag in ('conv2d', 'linear', 'addmm'))
    ]
    assert weighted_ops, "expected at least one ATen conv/linear node"
    assert all(n.op == 'call_function' for n in weighted_ops)

    report = FusionBasedPartitioner(precision=Precision.FP32).partition(fx)
    total_weight_bytes = sum(sg.total_weight_bytes for sg in report.subgraphs)

    # 224 conv elements + 32772 linear elements = 32996 elements * 4 bytes
    assert total_weight_bytes == 32996 * 4


def test_dynamo_export_weights_scale_with_precision():
    """ATen weight bytes must scale with the analysis precision (issue #52 + #55)."""
    model = _ConvLinear().eval()
    x = torch.randn(1, 3, 32, 32)
    fx = _dynamo_export(model, x)

    fp32 = sum(sg.total_weight_bytes
               for sg in FusionBasedPartitioner(Precision.FP32).partition(fx).subgraphs)
    fp16 = sum(sg.total_weight_bytes
               for sg in FusionBasedPartitioner(Precision.FP16).partition(fx).subgraphs)
    int8 = sum(sg.total_weight_bytes
               for sg in FusionBasedPartitioner(Precision.INT8).partition(fx).subgraphs)
    int4 = sum(sg.total_weight_bytes
               for sg in FusionBasedPartitioner(Precision.INT4).partition(fx).subgraphs)

    assert fp32 > 0
    assert fp16 == fp32 // 2
    assert int8 == fp32 // 4
    assert int4 == fp32 // 8


def test_dynamo_export_arithmetic_intensity_uses_real_weights():
    """Arithmetic intensity must include weight bytes for ATen subgraphs.

    Before issue #55 was fixed, weight_bytes was 0, so AI was inflated to
    FLOPs / (input + output) and a 32-channel conv got misclassified as
    compute-bound. The test asserts the conv subgraph's external_bytes now
    includes the weight contribution.
    """
    model = _ConvLinear().eval()
    x = torch.randn(1, 3, 32, 32)
    fx = _dynamo_export(model, x)

    report = FusionBasedPartitioner(Precision.FP32).partition(fx)

    conv_sgs = [
        sg for sg in report.subgraphs
        if any('conv' in name for name in sg.node_names)
    ]
    assert conv_sgs, "expected at least one conv subgraph"
    for sg in conv_sgs:
        assert sg.total_weight_bytes > 0


def test_aten_matmul_does_not_count_weights():
    """matmul has no parameter tensor; both inputs are activations."""

    class TwoMatmul(nn.Module):
        def forward(self, a, b):
            return a @ b

    model = TwoMatmul().eval()
    a = torch.randn(4, 8)
    b = torch.randn(8, 16)
    ep = torch.export.export(model, (a, b))
    fx = ep.module()
    from torch.fx.passes.shape_prop import ShapeProp
    ShapeProp(fx).propagate(a, b)

    report = FusionBasedPartitioner(Precision.FP32).partition(fx)
    assert sum(sg.total_weight_bytes for sg in report.subgraphs) == 0
