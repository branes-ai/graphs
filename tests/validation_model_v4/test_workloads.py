"""Workload-factory tests for matmul.py and linear.py."""

import pytest
import torch

from validation.model_v4.workloads.linear import build_linear
from validation.model_v4.workloads.matmul import build_matmul


# ---------------------------------------------------------------------------
# matmul
# ---------------------------------------------------------------------------


def test_matmul_runs_and_shapes_match():
    w = build_matmul(M=128, K=64, N=32, dtype="fp32")
    assert w.op == "matmul"
    assert w.shape == (128, 64, 32)
    assert w.dtype == "fp32"
    assert len(w.inputs) == 2
    assert tuple(w.inputs[0].shape) == (128, 64)
    assert tuple(w.inputs[1].shape) == (64, 32)
    out = w.model(*w.inputs)
    assert tuple(out.shape) == (128, 32)


@pytest.mark.parametrize("dtype,torch_dtype", [
    ("fp32", torch.float32),
    ("fp16", torch.float16),
    ("bf16", torch.bfloat16),
    ("fp64", torch.float64),
])
def test_matmul_float_dtypes(dtype, torch_dtype):
    w = build_matmul(64, 64, 64, dtype)
    assert w.inputs[0].dtype == torch_dtype
    out = w.model(*w.inputs)
    assert out.dtype == torch_dtype


def test_matmul_rejects_subbyte_dtypes():
    """fp4/int4/fp8 require quantized paths -- not silently accepted."""
    for dtype in ["fp4", "int4", "fp8", "fp8_e4m3"]:
        with pytest.raises(ValueError, match="Unsupported torch dtype"):
            build_matmul(64, 64, 64, dtype)


def test_matmul_rejects_zero_dim():
    with pytest.raises(ValueError, match="must be positive"):
        build_matmul(0, 64, 64)


def test_matmul_seed_is_reproducible():
    w1 = build_matmul(32, 32, 32, "fp32", seed=42)
    w2 = build_matmul(32, 32, 32, "fp32", seed=42)
    assert torch.equal(w1.inputs[0], w2.inputs[0])
    assert torch.equal(w1.inputs[1], w2.inputs[1])


def test_matmul_name_is_descriptive():
    w = build_matmul(1024, 768, 256, "fp16")
    assert w.name == "matmul_M1024_K768_N256_fp16"


# ---------------------------------------------------------------------------
# linear
# ---------------------------------------------------------------------------


def test_linear_runs_and_shapes_match():
    w = build_linear(B=16, IN=128, OUT=64, dtype="fp32")
    assert w.op == "linear"
    assert w.shape == (16, 128, 64)
    assert w.dtype == "fp32"
    assert len(w.inputs) == 1
    assert tuple(w.inputs[0].shape) == (16, 128)
    out = w.model(*w.inputs)
    assert tuple(out.shape) == (16, 64)


@pytest.mark.parametrize("dtype,torch_dtype", [
    ("fp32", torch.float32),
    ("fp16", torch.float16),
    ("bf16", torch.bfloat16),
    ("fp64", torch.float64),
])
def test_linear_float_dtypes(dtype, torch_dtype):
    w = build_linear(8, 64, 32, dtype)
    assert w.inputs[0].dtype == torch_dtype
    out = w.model(*w.inputs)
    assert out.dtype == torch_dtype


def test_linear_rejects_int_dtypes():
    """Quantized Linear needs a different path -- not silently accepted."""
    for dtype in ["int8", "int16", "int32", "fp4", "int4"]:
        with pytest.raises(ValueError, match="Unsupported torch dtype"):
            build_linear(8, 64, 32, dtype)


def test_linear_rejects_zero_dim():
    with pytest.raises(ValueError, match="must be positive"):
        build_linear(0, 64, 64)


def test_linear_seed_is_reproducible():
    w1 = build_linear(16, 64, 32, "fp32", seed=42)
    w2 = build_linear(16, 64, 32, "fp32", seed=42)
    assert torch.equal(w1.inputs[0], w2.inputs[0])
    assert torch.equal(w1.model.weight, w2.model.weight)
    assert torch.equal(w1.model.bias, w2.model.bias)


def test_linear_bias_can_be_disabled():
    w = build_linear(8, 16, 32, "fp32", bias=False)
    assert w.model.bias is None


def test_linear_name_is_descriptive():
    w = build_linear(64, 1024, 256, "bf16")
    assert w.name == "linear_B64_IN1024_OUT256_bf16"


# ---------------------------------------------------------------------------
# vector_add (V5-2a)
# ---------------------------------------------------------------------------


from validation.model_v4.workloads.vector_add import build_vector_add


def test_vector_add_runs_and_shapes_match():
    w = build_vector_add(1024, "fp32")
    assert w.op == "vector_add"
    assert w.shape == (1024,)
    assert w.dtype == "fp32"
    assert len(w.inputs) == 2
    a, b = w.inputs
    assert a.shape == (1024,)
    assert b.shape == (1024,)
    out = w.model(*w.inputs)
    assert out.shape == (1024,)
    # Forward should equal a + b elementwise
    assert torch.allclose(out, a + b)


@pytest.mark.parametrize("dtype,torch_dtype", [
    ("fp32", torch.float32),
    ("fp16", torch.float16),
    ("bf16", torch.bfloat16),
    ("fp64", torch.float64),
])
def test_vector_add_float_dtypes(dtype, torch_dtype):
    w = build_vector_add(64, dtype)
    assert w.inputs[0].dtype == torch_dtype


@pytest.mark.parametrize("dtype,torch_dtype", [
    ("int8", torch.int8),
    ("int16", torch.int16),
    ("int32", torch.int32),
])
def test_vector_add_int_dtypes(dtype, torch_dtype):
    """Unlike matmul / linear, vector_add over int dtypes is a normal
    elementwise op -- no quantized path needed."""
    w = build_vector_add(64, dtype)
    assert w.inputs[0].dtype == torch_dtype
    out = w.model(*w.inputs)
    assert torch.equal(out, w.inputs[0] + w.inputs[1])


def test_vector_add_rejects_subbyte_dtypes():
    for dtype in ["fp4", "int4"]:
        with pytest.raises(ValueError, match="Unsupported torch dtype"):
            build_vector_add(64, dtype)


def test_vector_add_rejects_zero_or_negative_n():
    with pytest.raises(ValueError, match="must be positive"):
        build_vector_add(0)
    with pytest.raises(ValueError, match="must be positive"):
        build_vector_add(-1)


def test_vector_add_seed_is_reproducible():
    w1 = build_vector_add(128, "fp32", seed=42)
    w2 = build_vector_add(128, "fp32", seed=42)
    assert torch.equal(w1.inputs[0], w2.inputs[0])
    assert torch.equal(w1.inputs[1], w2.inputs[1])


def test_vector_add_name_is_descriptive():
    w = build_vector_add(65536, "fp16")
    assert w.name == "vector_add_N65536_fp16"
