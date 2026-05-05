"""Linear workload factory for the v4 validation harness.

Same contract as matmul.py: builds a single-op nn.Module that the
PyTorch ground-truth measurer, the graphs analyzer, and the regime
classifier all consume from one artifact.

Difference from matmul.py: ``nn.Linear`` is parameterised -- the second
matrix is a learned weight, not an activation. That matters for two
reasons:

1. The forward call takes one input tensor (the activation), so this
   workload traces cleanly through ``frontend.trace_and_partition``
   without the multi-input frontend extension that matmul.py needs.
2. The partitioner sees the weight via ``module.parameters()`` and
   exercises the call_module weight-counting path (post-#54 +
   post-#56), not the call_function ATen weight extractor.

Shape convention: ``(B, IN, OUT)`` such that the model computes
``input @ W.T + b`` with ``input`` of shape ``(B, IN)``, ``W`` of shape
``(OUT, IN)`` (PyTorch's nn.Linear convention), and bias of shape
``(OUT,)``. Output: ``(B, OUT)``.
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
    # nn.Linear with int dtypes requires a quantized variant
    # (torch.ao.nn.quantized.Linear). Excluded from this factory --
    # the harness will need a separate quantized path.
}


@dataclass(frozen=True)
class WorkloadArtifact:
    """A self-describing workload bundle (parallels matmul.WorkloadArtifact)."""
    name: str
    op: str                      # "linear"
    shape: Tuple[int, int, int]  # (B, IN, OUT)
    dtype: str
    model: nn.Module
    inputs: Tuple[torch.Tensor, ...]


def _resolve_dtype(dtype: str) -> torch.dtype:
    key = dtype.lower()
    if key not in _TORCH_DTYPE:
        raise ValueError(
            f"Unsupported torch dtype {dtype!r}; supported: {sorted(_TORCH_DTYPE)}"
        )
    return _TORCH_DTYPE[key]


def build_linear(
    B: int, IN: int, OUT: int, dtype: str = "fp32",
    *, bias: bool = True, seed: int = 0,
) -> WorkloadArtifact:
    """Build a single-op ``nn.Linear`` workload.

    Args:
        B: Batch dimension (input shape is ``(B, IN)``).
        IN: Input feature dimension.
        OUT: Output feature dimension.
        dtype: One of fp32, fp16, bf16, fp64. Quantized integer dtypes
            require a different code path and are intentionally excluded.
        bias: Whether to include a bias term (default True, matching
            most real layers).
        seed: PyTorch RNG seed for reproducible inputs and parameters.

    Returns:
        WorkloadArtifact bundling the nn.Linear module and the input.

    Notes:
        - The module is set to ``eval()`` so dropout/batchnorm semantics
          (none here, but defensive) match inference behavior.
        - Parameters use the same RNG seed so the ground-truth measurer
          and the analyzer's prediction see the same exact tensor data.
    """
    if B <= 0 or IN <= 0 or OUT <= 0:
        raise ValueError(f"linear shape must be positive, got (B={B}, IN={IN}, OUT={OUT})")

    torch_dtype = _resolve_dtype(dtype)
    g = torch.Generator().manual_seed(seed)

    # Build the module first, then cast to the target dtype, then
    # re-init parameters with the seeded generator so the cast doesn't
    # disturb reproducibility.
    model = nn.Linear(IN, OUT, bias=bias).to(torch_dtype).eval()
    with torch.no_grad():
        model.weight.copy_(torch.randn(model.weight.shape, generator=g, dtype=torch_dtype))
        if model.bias is not None:
            model.bias.copy_(torch.randn(model.bias.shape, generator=g, dtype=torch_dtype))

    x = torch.randn(B, IN, generator=g, dtype=torch_dtype)

    name = f"linear_B{B}_IN{IN}_OUT{OUT}_{dtype}"
    return WorkloadArtifact(
        name=name,
        op="linear",
        shape=(B, IN, OUT),
        dtype=dtype,
        model=model,
        inputs=(x,),
    )
