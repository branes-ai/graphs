"""
Reference model DLA benchmarks.

Tests complete models (ResNet-18, MobileNetV2) on DLA to measure
end-to-end inference latency and DLA vs GPU layer placement.
"""

import tempfile
import os
from typing import Dict, List, Any, Optional

from .trt_utils import (
    check_trt_available,
    build_engine_from_onnx,
    time_engine,
    get_layer_info,
    export_pytorch_to_onnx,
    get_dla_peak_gflops,
)


# Reference models: (name, factory_fn_name, input_shape_without_batch, approx_flops)
REFERENCE_MODELS = [
    ('resnet18', 'torchvision.models.resnet18', (3, 224, 224), 1.8e9),
    ('mobilenet_v2', 'torchvision.models.mobilenet_v2', (3, 224, 224), 0.3e9),
]


def _load_model(model_name: str):
    """Load a torchvision model by name."""
    import torch
    import torch.nn as nn

    # Build models directly to avoid torchvision import issues
    # (torchvision.models may pull in 'requests' even with pretrained=False)
    if model_name == 'resnet18':
        try:
            import torchvision.models as models
            model = models.resnet18(weights=None)
        except (ImportError, TypeError):
            # TypeError: weights= not supported in older torchvision
            try:
                import torchvision.models as models
                model = models.resnet18(pretrained=False)
            except ImportError:
                raise RuntimeError(
                    "torchvision not available. Install with: pip install torchvision"
                )
    elif model_name == 'mobilenet_v2':
        try:
            import torchvision.models as models
            model = models.mobilenet_v2(weights=None)
        except (ImportError, TypeError):
            try:
                import torchvision.models as models
                model = models.mobilenet_v2(pretrained=False)
            except ImportError:
                raise RuntimeError(
                    "torchvision not available. Install with: pip install torchvision"
                )
    else:
        raise ValueError(
            f"Unknown model: {model_name}. Available: ['resnet18', 'mobilenet_v2']"
        )

    model.eval()
    return model


def benchmark_model(
    model_name: str,
    dla_core: int = 0,
    precision: str = "fp16",
    gpu_fallback: bool = True,
    batch_size: int = 1,
    warmup: int = 10,
    iterations: int = 100,
) -> Dict[str, Any]:
    """
    Benchmark a reference model on DLA.

    Args:
        model_name: Name of model ('resnet18', 'mobilenet_v2').
        dla_core: DLA core to use (-1 for GPU-only baseline).
        precision: "fp16" or "int8".
        gpu_fallback: Allow GPU fallback for unsupported layers.
        batch_size: Inference batch size.
        warmup: Warmup iterations.
        iterations: Timed iterations.

    Returns:
        Dict with latency, throughput, layer placement breakdown.
    """
    check_trt_available()
    import torch

    # Find model config
    model_config = None
    for name, _, input_shape_no_batch, approx_flops in REFERENCE_MODELS:
        if name == model_name:
            model_config = (name, input_shape_no_batch, approx_flops)
            break

    if model_config is None:
        raise ValueError(f"Unknown model: {model_name}")

    name, input_shape_no_batch, approx_flops = model_config
    input_shape = (batch_size,) + input_shape_no_batch
    flops = approx_flops * batch_size

    result = {
        'model': model_name,
        'precision': precision,
        'dla_core': dla_core,
        'gpu_fallback': gpu_fallback,
        'batch_size': batch_size,
        'input_shape': list(input_shape),
        'approx_flops': flops,
    }

    onnx_path = None
    try:
        model = _load_model(model_name)

        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
            onnx_path = f.name

        export_pytorch_to_onnx(model, input_shape, onnx_path)

        engine = build_engine_from_onnx(
            onnx_path, dla_core=dla_core, precision=precision,
            gpu_fallback=gpu_fallback, input_shape=input_shape,
        )

        timing = time_engine(engine, warmup=warmup, iterations=iterations)
        layers = get_layer_info(engine)

        dla_layers = [l for l in layers if 'DLA' in str(l.get('device', ''))]
        gpu_layers = [l for l in layers if 'DLA' not in str(l.get('device', ''))]

        result.update({
            'status': 'success',
            'latency_ms': timing['median_ms'],
            'mean_ms': timing['mean_ms'],
            'min_ms': timing['min_ms'],
            'max_ms': timing['max_ms'],
            'std_ms': timing['std_ms'],
            'throughput_fps': (1000.0 / timing['median_ms'] * batch_size) if timing['median_ms'] > 0 else 0,
            'tflops': (flops / timing['median_ms'] / 1e9) if timing['median_ms'] > 0 else 0,
            'total_layers': len(layers),
            'dla_layer_count': len(dla_layers),
            'gpu_layer_count': len(gpu_layers),
            'dla_percentage': (len(dla_layers) / len(layers) * 100) if layers else 0,
            'layer_info': layers,
            'dla_layer_types': list(set(l['type'] for l in dla_layers)),
            'gpu_fallback_types': list(set(l['type'] for l in gpu_layers)),
        })

        # XUE metrics
        if timing['median_ms'] > 0:
            attained_gflops = flops / (timing['median_ms'] / 1000.0) / 1e9
            peak = get_dla_peak_gflops(precision)
            result['attained_gflops'] = attained_gflops
            result['peak_gflops'] = peak
            result['efficiency'] = (attained_gflops / peak) if peak > 0 else 0.0

    except Exception as e:
        result.update({
            'status': 'failed',
            'error': str(e),
        })
    finally:
        if onnx_path and os.path.exists(onnx_path):
            os.unlink(onnx_path)

    return result


def benchmark_all_models(
    dla_core: int = 0,
    precision: str = "fp16",
    gpu_fallback: bool = True,
    batch_size: int = 1,
    warmup: int = 10,
    iterations: int = 100,
) -> List[Dict[str, Any]]:
    """Benchmark all reference models on DLA."""
    results = []
    for name, _, _, _ in REFERENCE_MODELS:
        print(f"  Benchmarking {name} on DLA core {dla_core} ({precision})...")
        result = benchmark_model(
            name, dla_core=dla_core, precision=precision,
            gpu_fallback=gpu_fallback, batch_size=batch_size,
            warmup=warmup, iterations=iterations,
        )
        results.append(result)

        if result['status'] == 'success':
            print(f"    {result['latency_ms']:.2f} ms, "
                  f"{result['throughput_fps']:.1f} FPS, "
                  f"DLA: {result['dla_layer_count']}/{result['total_layers']} layers")
        else:
            print(f"    FAILED: {result.get('error', 'unknown')}")

    return results


def benchmark_model_comparison(
    model_name: str,
    precision: str = "fp16",
    batch_size: int = 1,
    warmup: int = 10,
    iterations: int = 100,
) -> Dict[str, Any]:
    """
    Compare DLA vs GPU-only performance for a model.

    Returns dict with both DLA and GPU-only results plus speedup ratio.
    """
    from .trt_utils import get_dla_core_count

    # GPU-only baseline
    print(f"  {model_name} GPU-only baseline ({precision})...")
    gpu_result = benchmark_model(
        model_name, dla_core=-1, precision=precision,
        batch_size=batch_size, warmup=warmup, iterations=iterations,
    )

    # DLA core 0
    print(f"  {model_name} DLA core 0 ({precision})...")
    dla_result = benchmark_model(
        model_name, dla_core=0, precision=precision,
        gpu_fallback=True, batch_size=batch_size,
        warmup=warmup, iterations=iterations,
    )

    comparison = {
        'model': model_name,
        'precision': precision,
        'batch_size': batch_size,
        'gpu_only': gpu_result,
        'dla_core0': dla_result,
    }

    # Compute speedup
    if (gpu_result['status'] == 'success' and dla_result['status'] == 'success'):
        gpu_ms = gpu_result['latency_ms']
        dla_ms = dla_result['latency_ms']
        if dla_ms > 0:
            comparison['speedup'] = gpu_ms / dla_ms
            comparison['latency_reduction_pct'] = (1 - dla_ms / gpu_ms) * 100

    return comparison
