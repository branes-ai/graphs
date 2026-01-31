"""
Synthetic single-layer DLA benchmarks.

Tests individual DLA-supported operations (Conv2D, FC, depthwise conv)
at various sizes to characterize DLA throughput and latency.
"""

import tempfile
import os
from typing import Dict, List, Any, Optional, Tuple

from .trt_utils import (
    check_trt_available,
    build_engine_from_onnx,
    time_engine,
    get_layer_info,
    export_pytorch_to_onnx,
    get_dla_peak_gflops,
    TRT_AVAILABLE,
)


# ---- Synthetic layer definitions ----

# Conv2D configurations: (in_channels, out_channels, kernel_size, stride, padding, input_hw)
CONV2D_CONFIGS = [
    # Small feature maps, varying channels
    (3, 64, 3, 1, 1, 224),       # First conv (ResNet-style)
    (64, 64, 3, 1, 1, 56),       # Early stage
    (64, 128, 3, 2, 1, 56),      # Stride-2 downsample
    (128, 256, 3, 2, 1, 28),     # Mid stage
    (256, 512, 3, 2, 1, 14),     # Late stage
    # 1x1 pointwise convolutions
    (64, 256, 1, 1, 0, 56),
    (256, 64, 1, 1, 0, 56),
    (512, 2048, 1, 1, 0, 7),
    # 5x5 kernels
    (32, 64, 5, 1, 2, 112),
]

# Depthwise conv: (channels, kernel_size, stride, padding, input_hw)
DEPTHWISE_CONFIGS = [
    (32, 3, 1, 1, 112),
    (64, 3, 2, 1, 56),
    (128, 3, 1, 1, 28),
    (256, 5, 1, 2, 14),
    (512, 3, 1, 1, 7),
]

# FC (Linear) configurations: (in_features, out_features)
FC_CONFIGS = [
    (512, 512),
    (512, 1000),
    (1024, 1024),
    (2048, 1000),
    (2048, 2048),
]


def _make_conv2d_model(in_ch, out_ch, kernel, stride, padding):
    """Create a simple Conv2D + ReLU PyTorch model."""
    import torch
    import torch.nn as nn

    class ConvModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride, padding, bias=False)
            self.bn = nn.BatchNorm2d(out_ch)
            self.relu = nn.ReLU()

        def forward(self, x):
            return self.relu(self.bn(self.conv(x)))

    return ConvModel()


def _make_depthwise_model(channels, kernel, stride, padding):
    """Create a depthwise separable conv model."""
    import torch
    import torch.nn as nn

    class DWModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.dw = nn.Conv2d(channels, channels, kernel, stride, padding,
                                groups=channels, bias=False)
            self.bn = nn.BatchNorm2d(channels)
            self.relu = nn.ReLU()

        def forward(self, x):
            return self.relu(self.bn(self.dw(x)))

    return DWModel()


def _make_fc_model(in_features, out_features):
    """Create a simple FC + ReLU model."""
    import torch
    import torch.nn as nn

    class FCModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(in_features, out_features)
            self.relu = nn.ReLU()

        def forward(self, x):
            return self.relu(self.fc(x))

    return FCModel()


def _compute_conv2d_flops(in_ch, out_ch, kernel, stride, padding, input_hw):
    """Compute FLOPs for Conv2D (multiply-add counted as 2 ops)."""
    out_hw = (input_hw + 2 * padding - kernel) // stride + 1
    flops_per_output = 2 * in_ch * kernel * kernel
    total_outputs = out_ch * out_hw * out_hw
    return flops_per_output * total_outputs


def _compute_depthwise_flops(channels, kernel, stride, padding, input_hw):
    """Compute FLOPs for depthwise conv."""
    out_hw = (input_hw + 2 * padding - kernel) // stride + 1
    flops_per_output = 2 * kernel * kernel
    total_outputs = channels * out_hw * out_hw
    return flops_per_output * total_outputs


def _compute_fc_flops(in_features, out_features):
    """Compute FLOPs for FC layer (2 * in * out for multiply-add)."""
    return 2 * in_features * out_features


def benchmark_conv2d(
    dla_core: int = 0,
    precision: str = "fp16",
    gpu_fallback: bool = True,
    batch_size: int = 1,
    warmup: int = 10,
    iterations: int = 100,
    configs: Optional[List] = None,
) -> List[Dict[str, Any]]:
    """
    Benchmark Conv2D layers on DLA.

    Returns list of result dicts with latency, throughput, and layer placement.
    """
    check_trt_available()
    import torch

    if configs is None:
        configs = CONV2D_CONFIGS

    results = []
    for in_ch, out_ch, kernel, stride, padding, input_hw in configs:
        config_name = f"Conv2d({in_ch},{out_ch},k{kernel},s{stride})"
        input_shape = (batch_size, in_ch, input_hw, input_hw)
        flops = _compute_conv2d_flops(in_ch, out_ch, kernel, stride, padding, input_hw) * batch_size

        result = {
            'layer_type': 'conv2d',
            'config': config_name,
            'params': {
                'in_channels': in_ch, 'out_channels': out_ch,
                'kernel_size': kernel, 'stride': stride,
                'padding': padding, 'input_hw': input_hw,
            },
            'input_shape': list(input_shape),
            'batch_size': batch_size,
            'flops': flops,
            'precision': precision,
            'dla_core': dla_core,
        }

        try:
            model = _make_conv2d_model(in_ch, out_ch, kernel, stride, padding)
            with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
                onnx_path = f.name

            export_pytorch_to_onnx(model, input_shape, onnx_path)
            engine = build_engine_from_onnx(
                onnx_path, dla_core=dla_core, precision=precision,
                gpu_fallback=gpu_fallback, input_shape=input_shape,
            )

            timing = time_engine(engine, warmup=warmup, iterations=iterations)
            layers = get_layer_info(engine)

            dla_layers = sum(1 for l in layers if 'DLA' in str(l.get('device', '')))
            gpu_layers = len(layers) - dla_layers

            result.update({
                'status': 'success',
                'latency_ms': timing['median_ms'],
                'mean_ms': timing['mean_ms'],
                'min_ms': timing['min_ms'],
                'max_ms': timing['max_ms'],
                'std_ms': timing['std_ms'],
                'tflops': (flops / timing['median_ms'] / 1e9) if timing['median_ms'] > 0 else 0,
                'dla_layer_count': dla_layers,
                'gpu_layer_count': gpu_layers,
                'on_dla': dla_layers > 0,
                'layer_info': layers,
            })
            # XUE metrics
            if timing['median_ms'] > 0:
                attained = flops / (timing['median_ms'] / 1000.0) / 1e9
                peak = get_dla_peak_gflops(precision)
                result['attained_gflops'] = attained
                result['peak_gflops'] = peak
                result['efficiency'] = (attained / peak) if peak > 0 else 0.0
        except Exception as e:
            result.update({
                'status': 'failed',
                'error': str(e),
                'on_dla': False,
            })
        finally:
            if 'onnx_path' in locals() and os.path.exists(onnx_path):
                os.unlink(onnx_path)

        results.append(result)

    return results


def benchmark_depthwise(
    dla_core: int = 0,
    precision: str = "fp16",
    gpu_fallback: bool = True,
    batch_size: int = 1,
    warmup: int = 10,
    iterations: int = 100,
    configs: Optional[List] = None,
) -> List[Dict[str, Any]]:
    """Benchmark depthwise conv layers on DLA."""
    check_trt_available()
    import torch

    if configs is None:
        configs = DEPTHWISE_CONFIGS

    results = []
    for channels, kernel, stride, padding, input_hw in configs:
        config_name = f"DWConv({channels},k{kernel},s{stride})"
        input_shape = (batch_size, channels, input_hw, input_hw)
        flops = _compute_depthwise_flops(channels, kernel, stride, padding, input_hw) * batch_size

        result = {
            'layer_type': 'depthwise_conv',
            'config': config_name,
            'params': {
                'channels': channels, 'kernel_size': kernel,
                'stride': stride, 'padding': padding, 'input_hw': input_hw,
            },
            'input_shape': list(input_shape),
            'batch_size': batch_size,
            'flops': flops,
            'precision': precision,
            'dla_core': dla_core,
        }

        try:
            model = _make_depthwise_model(channels, kernel, stride, padding)
            with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
                onnx_path = f.name

            export_pytorch_to_onnx(model, input_shape, onnx_path)
            engine = build_engine_from_onnx(
                onnx_path, dla_core=dla_core, precision=precision,
                gpu_fallback=gpu_fallback, input_shape=input_shape,
            )

            timing = time_engine(engine, warmup=warmup, iterations=iterations)
            layers = get_layer_info(engine)

            dla_layers = sum(1 for l in layers if 'DLA' in str(l.get('device', '')))
            gpu_layers = len(layers) - dla_layers

            result.update({
                'status': 'success',
                'latency_ms': timing['median_ms'],
                'mean_ms': timing['mean_ms'],
                'min_ms': timing['min_ms'],
                'max_ms': timing['max_ms'],
                'std_ms': timing['std_ms'],
                'tflops': (flops / timing['median_ms'] / 1e9) if timing['median_ms'] > 0 else 0,
                'dla_layer_count': dla_layers,
                'gpu_layer_count': gpu_layers,
                'on_dla': dla_layers > 0,
                'layer_info': layers,
            })
            # XUE metrics
            if timing['median_ms'] > 0:
                attained = flops / (timing['median_ms'] / 1000.0) / 1e9
                peak = get_dla_peak_gflops(precision)
                result['attained_gflops'] = attained
                result['peak_gflops'] = peak
                result['efficiency'] = (attained / peak) if peak > 0 else 0.0
        except Exception as e:
            result.update({
                'status': 'failed',
                'error': str(e),
                'on_dla': False,
            })
        finally:
            if 'onnx_path' in locals() and os.path.exists(onnx_path):
                os.unlink(onnx_path)

        results.append(result)

    return results


def benchmark_fc(
    dla_core: int = 0,
    precision: str = "fp16",
    gpu_fallback: bool = True,
    batch_size: int = 1,
    warmup: int = 10,
    iterations: int = 100,
    configs: Optional[List] = None,
) -> List[Dict[str, Any]]:
    """Benchmark fully-connected layers on DLA."""
    check_trt_available()
    import torch

    if configs is None:
        configs = FC_CONFIGS

    results = []
    for in_features, out_features in configs:
        config_name = f"FC({in_features},{out_features})"
        input_shape = (batch_size, in_features)
        flops = _compute_fc_flops(in_features, out_features) * batch_size

        result = {
            'layer_type': 'fc',
            'config': config_name,
            'params': {
                'in_features': in_features,
                'out_features': out_features,
            },
            'input_shape': list(input_shape),
            'batch_size': batch_size,
            'flops': flops,
            'precision': precision,
            'dla_core': dla_core,
        }

        try:
            model = _make_fc_model(in_features, out_features)
            with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
                onnx_path = f.name

            export_pytorch_to_onnx(model, input_shape, onnx_path)
            engine = build_engine_from_onnx(
                onnx_path, dla_core=dla_core, precision=precision,
                gpu_fallback=gpu_fallback, input_shape=input_shape,
            )

            timing = time_engine(engine, warmup=warmup, iterations=iterations)
            layers = get_layer_info(engine)

            dla_layers = sum(1 for l in layers if 'DLA' in str(l.get('device', '')))
            gpu_layers = len(layers) - dla_layers

            result.update({
                'status': 'success',
                'latency_ms': timing['median_ms'],
                'mean_ms': timing['mean_ms'],
                'min_ms': timing['min_ms'],
                'max_ms': timing['max_ms'],
                'std_ms': timing['std_ms'],
                'tflops': (flops / timing['median_ms'] / 1e9) if timing['median_ms'] > 0 else 0,
                'dla_layer_count': dla_layers,
                'gpu_layer_count': gpu_layers,
                'on_dla': dla_layers > 0,
                'layer_info': layers,
            })
            # XUE metrics
            if timing['median_ms'] > 0:
                attained = flops / (timing['median_ms'] / 1000.0) / 1e9
                peak = get_dla_peak_gflops(precision)
                result['attained_gflops'] = attained
                result['peak_gflops'] = peak
                result['efficiency'] = (attained / peak) if peak > 0 else 0.0
        except Exception as e:
            result.update({
                'status': 'failed',
                'error': str(e),
                'on_dla': False,
            })
        finally:
            if 'onnx_path' in locals() and os.path.exists(onnx_path):
                os.unlink(onnx_path)

        results.append(result)

    return results


def run_all_synthetic(
    dla_core: int = 0,
    precision: str = "fp16",
    gpu_fallback: bool = True,
    batch_size: int = 1,
    warmup: int = 10,
    iterations: int = 100,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Run all synthetic benchmarks on DLA.

    Returns dict keyed by layer type with lists of results.
    """
    return {
        'conv2d': benchmark_conv2d(
            dla_core=dla_core, precision=precision,
            gpu_fallback=gpu_fallback, batch_size=batch_size,
            warmup=warmup, iterations=iterations,
        ),
        'depthwise_conv': benchmark_depthwise(
            dla_core=dla_core, precision=precision,
            gpu_fallback=gpu_fallback, batch_size=batch_size,
            warmup=warmup, iterations=iterations,
        ),
        'fc': benchmark_fc(
            dla_core=dla_core, precision=precision,
            gpu_fallback=gpu_fallback, batch_size=batch_size,
            warmup=warmup, iterations=iterations,
        ),
    }
