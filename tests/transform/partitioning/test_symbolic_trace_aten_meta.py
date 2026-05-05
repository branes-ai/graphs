"""Regression tests for issue #63.

The ATen call_function flop/byte helpers in fusion_partitioner.py used to
read only ``arg.meta['val']`` (set by torch.export / Dynamo). When a model
was traced via the default ``symbolic_trace + ShapeProp`` path -- which
sets ``arg.meta['tensor_meta']`` instead -- those helpers silently
returned 0. Single-Linear and single-matmul models then reported 0 FLOPs
and 0 weight bytes, leading to nonsense roofline predictions.

These tests lock in:

1. ``_meta_shape`` reads both Dynamo-style ``meta['val']`` and
   symbolic_trace-style ``meta['tensor_meta']``.
2. A single ``nn.Linear`` traced via ``symbolic_trace`` reports the
   correct FLOPs and weight bytes through ``FusionBasedPartitioner``.
3. A single ``torch.matmul`` traced via ``symbolic_trace`` reports the
   correct FLOPs (and zero weights, because matmul has no parameters).
4. ``aten.addmm`` and ``aten.conv2d`` style nodes work too.
"""

import pytest
import torch
import torch.nn as nn
from torch.fx import symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp

from graphs.hardware.resource_model import Precision
from graphs.transform.partitioning import FusionBasedPartitioner


# ---------------------------------------------------------------------------
# _meta_shape unit tests
# ---------------------------------------------------------------------------


def _trace(model: nn.Module, *inputs) -> "torch.fx.GraphModule":
    fx = symbolic_trace(model)
    ShapeProp(fx).propagate(*inputs)
    return fx


def test_meta_shape_reads_tensor_meta_from_symbolic_trace():
    """symbolic_trace + ShapeProp populates meta['tensor_meta']; the helper
    must read it."""
    m = nn.Linear(64, 32).eval()
    fx = _trace(m, torch.randn(8, 64))

    weight_node = next(n for n in fx.graph.nodes if n.op == "get_attr" and "weight" in n.name)
    shape = FusionBasedPartitioner._meta_shape(weight_node)
    assert shape == (32, 64)


def test_meta_shape_returns_none_when_meta_absent():
    """Plain int/literal arguments have no meta; helper must return None
    rather than crashing."""

    class Dummy:
        pass

    arg = Dummy()  # no .meta attribute
    assert FusionBasedPartitioner._meta_shape(arg) is None


def test_meta_numel_multiplies_correctly():
    m = nn.Linear(128, 64).eval()
    fx = _trace(m, torch.randn(4, 128))
    weight_node = next(n for n in fx.graph.nodes if "weight" in n.name and n.op == "get_attr")
    assert FusionBasedPartitioner._meta_numel(weight_node) == 128 * 64


# ---------------------------------------------------------------------------
# End-to-end: single-Linear / single-matmul through symbolic_trace
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("B,IN,OUT", [
    (64, 1024, 1024),
    (1, 4096, 4096),
    (256, 768, 3072),
    (8, 512, 256),
])
def test_single_linear_symbolic_trace_reports_correct_flops_and_weights(B, IN, OUT):
    """Issue #63: single nn.Linear traced via symbolic_trace was reporting
    0 FLOPs and 0 weight bytes. Lock the correct values in."""
    m = nn.Linear(IN, OUT).eval()
    fx = _trace(m, torch.randn(B, IN))

    report = FusionBasedPartitioner(precision=Precision.FP32).partition(fx)
    sg = report.subgraphs[0]

    expected_flops = 2 * B * IN * OUT
    # Weight = OUT*IN floats, bias = OUT floats; 4 bytes/elem (fp32)
    expected_weight_bytes = 4 * (IN * OUT + OUT)

    assert sg.total_flops == expected_flops, (
        f"Linear({IN},{OUT}) on B={B}: flops={sg.total_flops} expected={expected_flops}"
    )
    assert sg.total_weight_bytes == expected_weight_bytes


def test_single_linear_no_bias_symbolic_trace():
    """Linear without bias: weight bytes only count the weight matrix."""
    m = nn.Linear(512, 256, bias=False).eval()
    fx = _trace(m, torch.randn(8, 512))

    report = FusionBasedPartitioner(precision=Precision.FP32).partition(fx)
    sg = report.subgraphs[0]

    assert sg.total_flops == 2 * 8 * 512 * 256
    assert sg.total_weight_bytes == 4 * 512 * 256  # no bias


@pytest.mark.parametrize("M,K,N", [
    (128, 256, 64),
    (1, 1024, 1024),
    (512, 512, 512),
])
def test_single_matmul_symbolic_trace_reports_correct_flops(M, K, N):
    """torch.matmul via symbolic_trace: FLOPs nonzero, weights zero
    (both inputs are activations, no parameter)."""

    class MM(nn.Module):
        def forward(self, a, b):
            return torch.matmul(a, b)

    fx = _trace(MM().eval(), torch.randn(M, K), torch.randn(K, N))
    report = FusionBasedPartitioner(precision=Precision.FP32).partition(fx)
    sg = report.subgraphs[0]

    assert sg.total_flops == 2 * M * K * N
    assert sg.total_weight_bytes == 0


def test_precision_scaling_still_works_after_fix():
    """The fix preserves PR #54's precision-aware byte counts: weight
    bytes scale with the analysis precision, not just fp32."""
    m = nn.Linear(1024, 1024).eval()
    fx = _trace(m, torch.randn(64, 1024))

    fp32 = sum(sg.total_weight_bytes for sg in
               FusionBasedPartitioner(Precision.FP32).partition(fx).subgraphs)
    fp16 = sum(sg.total_weight_bytes for sg in
               FusionBasedPartitioner(Precision.FP16).partition(fx).subgraphs)
    int8 = sum(sg.total_weight_bytes for sg in
               FusionBasedPartitioner(Precision.INT8).partition(fx).subgraphs)

    assert fp32 > 0
    assert fp16 == fp32 // 2
    assert int8 == fp32 // 4


def test_dynamo_export_path_still_works_after_fix():
    """Negative control: the fix must not regress the Dynamo / torch.export
    path that PR #56 added. Re-trace via torch.export.export and verify
    weight bytes are still counted via meta['val']."""

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 8, 3, padding=1)
            self.fc = nn.Linear(8 * 32 * 32, 4)

        def forward(self, x):
            return self.fc(self.conv(x).flatten(1))

    m = Net().eval()
    x = torch.randn(1, 3, 32, 32)
    ep = torch.export.export(m, (x,))
    fx = ep.module()
    ShapeProp(fx).propagate(x)

    report = FusionBasedPartitioner(precision=Precision.FP32).partition(fx)
    total_w = sum(sg.total_weight_bytes for sg in report.subgraphs)
    # Conv: 3*8*9 + 8 = 224 elems; Linear: 4*8192 + 4 = 32772 elems;
    # 32996 elems * 4 bytes = 131,984 bytes (matches PR #56 hand-verified number)
    assert total_w == 32996 * 4
