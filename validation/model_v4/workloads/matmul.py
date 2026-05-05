"""Matmul workload factory for the v4 validation harness.

Produces a single-op nn.Module plus its example input. The same artifact
flows through three consumers:

1. PyTorch ground-truth measurer (run on real silicon, capture latency/energy)
2. The graphs analyzer (trace via frontend.trace_and_partition, predict)
3. The classifier (compute working_set / FLOPs / OI from the shape)

Keeping all three on one factory means the analyzer's prediction and the
real measurement are about the same computation by construction.

Shape convention: ``(M, K, N)`` such that the model computes ``A @ B`` with
``A`` of shape ``(M, K)`` and ``B`` of shape ``(K, N)``, output ``(M, N)``.
This is rank-2 matmul; for batched matmul use the ``linear`` factory or
extend with a leading batch dim.
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
    # int4/fp4/fp8 are not first-class torch dtypes for general ops; the
    # validation harness will need a quantized path or a different
    # measurer for those. Excluded here so a typo doesn't silently fall
    # back to fp32.
}


@dataclass(frozen=True)
class WorkloadArtifact:
    """A self-describing workload bundle."""
    name: str                    # e.g., "matmul_M1024_K1024_N1024_fp16"
    op: str                      # "matmul"
    shape: Tuple[int, int, int]  # (M, K, N)
    dtype: str                   # "fp16", "fp32", ...
    model: nn.Module
    inputs: Tuple[torch.Tensor, ...]


class _Matmul(nn.Module):
    """Single-op nn.Module that performs ``A @ B``.

    Wrapping torch.matmul in an nn.Module rather than calling it directly
    means it traces through ``frontend.trace_and_partition`` the same way
    a model layer would, exercising the same partitioner/mapper code path
    the production analyzer uses.
    """

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return torch.matmul(a, b)


def _resolve_dtype(dtype: str) -> torch.dtype:
    key = dtype.lower()
    if key not in _TORCH_DTYPE:
        raise ValueError(
            f"Unsupported torch dtype {dtype!r}; supported: {sorted(_TORCH_DTYPE)}"
        )
    return _TORCH_DTYPE[key]


def build_matmul(
    M: int, K: int, N: int, dtype: str = "fp32", *, seed: int = 0,
) -> WorkloadArtifact:
    """Build a single-op matmul workload.

    Args:
        M, K, N: Matrix dimensions for ``(M, K) @ (K, N) -> (M, N)``.
        dtype: One of fp32, fp16, bf16, fp64, int8, int16, int32. Sub-byte
            types (int4, fp4, fp8) require a quantized measurement path
            and are intentionally not supported by this factory.
        seed: PyTorch RNG seed for reproducible inputs.

    Returns:
        A WorkloadArtifact bundling the nn.Module and example inputs.

    Inputs use a fixed seed so two runs against the same shape see the
    same data (matters for measurement variance: pathological data can
    trigger different cache-line patterns).
    """
    if M <= 0 or K <= 0 or N <= 0:
        raise ValueError(f"matmul shape must be positive, got ({M}, {K}, {N})")

    torch_dtype = _resolve_dtype(dtype)
    g = torch.Generator().manual_seed(seed)

    if torch_dtype.is_floating_point:
        a = torch.randn(M, K, generator=g, dtype=torch_dtype)
        b = torch.randn(K, N, generator=g, dtype=torch_dtype)
    else:
        # Integer dtypes: bounded random values so accumulation stays in range.
        a = torch.randint(-8, 8, (M, K), generator=g, dtype=torch_dtype)
        b = torch.randint(-8, 8, (K, N), generator=g, dtype=torch_dtype)

    name = f"matmul_M{M}_K{K}_N{N}_{dtype}"
    return WorkloadArtifact(
        name=name,
        op="matmul",
        shape=(M, K, N),
        dtype=dtype,
        model=_Matmul().eval(),
        inputs=(a, b),
    )
