"""
TensorRT DLA Benchmarks

Benchmarks for NVIDIA Deep Learning Accelerator (DLA) via TensorRT.
Requires TensorRT Python bindings (available on Jetson via JetPack).

Provides:
- Synthetic single-layer benchmarks (Conv2D, FC, depthwise conv)
- Reference model benchmarks (ResNet-18, MobileNetV2)
- Per-layer DLA vs GPU placement profiling
"""

__all__ = []
