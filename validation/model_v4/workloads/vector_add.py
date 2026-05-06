"""Vector add workload factory for the v4 validation harness (V5-2a).

The zero-reuse ground truth: ``c[i] = a[i] + b[i]`` for N elements.
Each input byte is read once, each output byte is written once,
no operand reuse. This makes vector add the cleanest tier-bandwidth
microbenchmark there is -- no tile-size choices, no kernel-tuning
variance, just bytes-per-second.

Working set: ``3 * N * bytes_per_element`` (a, b, c).
FLOPs: ``N`` (one add per element).
Operational intensity: ``1 / (3 * bpe)`` -- always memory-bound.

Why this exists (V5 plan):
    The V4 sweep is matmul + linear -- both have rich operand reuse
    that confounds bandwidth measurement (the achieved BW depends on
    the kernel's tile size and how well cuBLAS / oneDNN tunes it).
    Vector add has *no* reuse, so the measured latency / N directly
    yields the binding tier's effective bandwidth. Sweeping N across
    the cache hierarchy isolates each tier's BW one at a time.

Same artifact contract as matmul.py / linear.py: a WorkloadArtifact
flows through the ground-truth measurer, the analyzer, and the
classifier.

Shape convention: ``(N,)`` -- a 1-D length tuple. The factory accepts
N as a Python int for ergonomics; the artifact stores it as ``(N,)``
so the rest of the harness can treat it uniformly with matmul's
``(M, K, N)`` and linear's ``(B, IN, OUT)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn


_TORCH_DTYPE: dict[str, torch.dtype] = {
    "fp64": torch.float64,
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "int32": torch.int32,
    "int16": torch.int16,
    "int8": torch.int8,
}


@dataclass(frozen=True)
class WorkloadArtifact:
    """A self-describing workload bundle (parallels matmul / linear)."""
    name: str
    op: str                # "vector_add"
    shape: Tuple[int, ...]  # (N,)
    dtype: str
    model: nn.Module
    inputs: Tuple[torch.Tensor, ...]


class _VectorAdd(nn.Module):
    """Single-op nn.Module that performs ``a + b`` elementwise.

    Wrapping ``torch.add`` in an nn.Module rather than calling it
    directly means it traces through ``frontend.trace_and_partition``
    the same way a real layer would, exercising the same partitioner /
    mapper code path the production analyzer uses."""

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return a + b


def _resolve_dtype(dtype: str) -> torch.dtype:
    key = dtype.lower()
    if key not in _TORCH_DTYPE:
        raise ValueError(
            f"Unsupported torch dtype {dtype!r}; supported: {sorted(_TORCH_DTYPE)}"
        )
    return _TORCH_DTYPE[key]


def build_vector_add(
    N: int, dtype: str = "fp32", *, seed: int = 0,
) -> WorkloadArtifact:
    """Build a single-op vector-add workload.

    Args:
        N: Number of elements per vector. The two input vectors and
            the output vector each have shape ``(N,)``.
        dtype: One of fp32, fp16, bf16, fp64, int8, int16, int32.
        seed: PyTorch RNG seed for reproducible inputs.

    Returns:
        A WorkloadArtifact bundling the nn.Module and the two inputs.
    """
    if N <= 0:
        raise ValueError(f"vector_add shape must be positive, got N={N}")

    torch_dtype = _resolve_dtype(dtype)
    g = torch.Generator().manual_seed(seed)

    if torch_dtype.is_floating_point:
        a = torch.randn(N, generator=g, dtype=torch_dtype)
        b = torch.randn(N, generator=g, dtype=torch_dtype)
    else:
        # Integer dtypes: bounded random values so a + b doesn't overflow.
        a = torch.randint(-8, 8, (N,), generator=g, dtype=torch_dtype)
        b = torch.randint(-8, 8, (N,), generator=g, dtype=torch_dtype)

    name = f"vector_add_N{N}_{dtype}"
    return WorkloadArtifact(
        name=name,
        op="vector_add",
        shape=(N,),
        dtype=dtype,
        model=_VectorAdd().eval(),
        inputs=(a, b),
    )
