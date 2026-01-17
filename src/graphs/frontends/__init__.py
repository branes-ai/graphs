"""
Model Frontends

This package provides frontends that convert various model formats into
the internal graph representation used by the graphs framework.

Currently supported frontends:
- dynamo: PyTorch models via torch.export (Dynamo)

Planned frontends:
- onnx: ONNX model import
- jax: JAX function tracing
- tflite: TensorFlow Lite models

Usage:
    from graphs.frontends import trace_and_partition

    model = torchvision.models.resnet18()
    input_tensor = torch.randn(1, 3, 224, 224)

    fx_graph, partition_report = trace_and_partition(model, input_tensor)
"""

from .dynamo import (
    trace_and_partition,
    trace_only,
    get_model_stats,
)

__all__ = [
    'trace_and_partition',
    'trace_only',
    'get_model_stats',
]
