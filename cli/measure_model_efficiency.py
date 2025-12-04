#!/usr/bin/env python3
"""
Model Efficiency Measurement Tool

Measures actual MAC throughput efficiency for neural network models by:
1. Tracing the model to count theoretical FLOPs/MACs
2. Running timed inference on the target device
3. Comparing delivered MACs/sec to peak hardware capability

This reveals the gap between marketing peak TOPS and real workload performance.

Usage:
    # Measure ViT efficiency on GPU
    ./cli/measure_model_efficiency.py --model vit_b_16 --device cuda

    # Measure ResNet18 efficiency on GPU
    ./cli/measure_model_efficiency.py --model resnet18 --device cuda

    # Compare multiple models
    ./cli/measure_model_efficiency.py --model resnet18,resnet50,vit_b_16 --device cuda

    # Specify batch size
    ./cli/measure_model_efficiency.py --model vit_b_16 --device cuda --batch-size 32

    # Use hardware spec for peak (instead of measured)
    ./cli/measure_model_efficiency.py --model vit_b_16 --device cuda --hardware H100-SXM5-80GB

    # Sweep batch sizes
    ./cli/measure_model_efficiency.py --model vit_b_16 --device cuda --batch-size 1,4,16,64
"""

import argparse
import sys
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    import torch
    import torch.nn as nn
    from torch.fx import symbolic_trace
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("ERROR: PyTorch is required for this tool")
    sys.exit(1)

try:
    import torchvision.models as tv_models
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False


# Known model configurations
MODEL_CONFIGS = {
    # Vision Transformers
    'vit_b_16': {'input_shape': (3, 224, 224), 'factory': 'torchvision'},
    'vit_b_32': {'input_shape': (3, 224, 224), 'factory': 'torchvision'},
    'vit_l_16': {'input_shape': (3, 224, 224), 'factory': 'torchvision'},
    'vit_l_32': {'input_shape': (3, 224, 224), 'factory': 'torchvision'},
    'vit_h_14': {'input_shape': (3, 224, 224), 'factory': 'torchvision'},

    # ResNets
    'resnet18': {'input_shape': (3, 224, 224), 'factory': 'torchvision'},
    'resnet34': {'input_shape': (3, 224, 224), 'factory': 'torchvision'},
    'resnet50': {'input_shape': (3, 224, 224), 'factory': 'torchvision'},
    'resnet101': {'input_shape': (3, 224, 224), 'factory': 'torchvision'},
    'resnet152': {'input_shape': (3, 224, 224), 'factory': 'torchvision'},

    # EfficientNets
    'efficientnet_b0': {'input_shape': (3, 224, 224), 'factory': 'torchvision'},
    'efficientnet_b4': {'input_shape': (3, 380, 380), 'factory': 'torchvision'},
    'efficientnet_b7': {'input_shape': (3, 600, 600), 'factory': 'torchvision'},
    'efficientnet_v2_s': {'input_shape': (3, 384, 384), 'factory': 'torchvision'},
    'efficientnet_v2_m': {'input_shape': (3, 480, 480), 'factory': 'torchvision'},
    'efficientnet_v2_l': {'input_shape': (3, 480, 480), 'factory': 'torchvision'},

    # MobileNets
    'mobilenet_v2': {'input_shape': (3, 224, 224), 'factory': 'torchvision'},
    'mobilenet_v3_small': {'input_shape': (3, 224, 224), 'factory': 'torchvision'},
    'mobilenet_v3_large': {'input_shape': (3, 224, 224), 'factory': 'torchvision'},

    # ConvNeXt
    'convnext_tiny': {'input_shape': (3, 224, 224), 'factory': 'torchvision'},
    'convnext_small': {'input_shape': (3, 224, 224), 'factory': 'torchvision'},
    'convnext_base': {'input_shape': (3, 224, 224), 'factory': 'torchvision'},
    'convnext_large': {'input_shape': (3, 224, 224), 'factory': 'torchvision'},

    # Swin Transformers
    'swin_t': {'input_shape': (3, 224, 224), 'factory': 'torchvision'},
    'swin_s': {'input_shape': (3, 224, 224), 'factory': 'torchvision'},
    'swin_b': {'input_shape': (3, 224, 224), 'factory': 'torchvision'},
    'swin_v2_t': {'input_shape': (3, 256, 256), 'factory': 'torchvision'},
    'swin_v2_s': {'input_shape': (3, 256, 256), 'factory': 'torchvision'},
    'swin_v2_b': {'input_shape': (3, 256, 256), 'factory': 'torchvision'},
}


@dataclass
class EfficiencyResult:
    """Results from efficiency measurement."""
    model_name: str
    batch_size: int
    device: str
    precision: str

    # Theoretical
    theoretical_flops: int  # Total FLOPs (multiply + add counted separately)
    theoretical_macs: int   # MACs = FLOPs / 2

    # Measured
    warmup_runs: int
    timed_runs: int
    mean_latency_ms: float
    std_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float

    # Derived
    delivered_macs_per_sec: float  # MACs / mean_latency
    delivered_tflops: float        # TFLOPS (counting MACs as 2 ops)

    # Peak reference
    peak_tflops: Optional[float] = None  # From hardware spec or calibration
    peak_source: Optional[str] = None    # 'spec', 'calibration', 'cublas'

    # Efficiency
    efficiency: Optional[float] = None   # delivered / peak

    def __str__(self) -> str:
        lines = [
            f"Model: {self.model_name}",
            f"Batch size: {self.batch_size}",
            f"Device: {self.device}",
            f"Precision: {self.precision}",
            f"",
            f"Theoretical:",
            f"  FLOPs: {self.theoretical_flops:,} ({self.theoretical_flops/1e9:.2f} GFLOPs)",
            f"  MACs:  {self.theoretical_macs:,} ({self.theoretical_macs/1e9:.2f} GMACs)",
            f"",
            f"Measured ({self.timed_runs} runs after {self.warmup_runs} warmup):",
            f"  Mean latency: {self.mean_latency_ms:.3f} ms",
            f"  Std latency:  {self.std_latency_ms:.3f} ms",
            f"  Min latency:  {self.min_latency_ms:.3f} ms",
            f"  Max latency:  {self.max_latency_ms:.3f} ms",
            f"",
            f"Throughput:",
            f"  Delivered: {self.delivered_tflops:.2f} TFLOPS",
        ]

        if self.peak_tflops and self.efficiency is not None:
            lines.extend([
                f"  Peak ({self.peak_source}): {self.peak_tflops:.2f} TFLOPS",
                f"  Efficiency: {self.efficiency*100:.1f}%",
            ])

        return "\n".join(lines)


def get_model(model_name: str) -> Tuple[nn.Module, Tuple[int, ...]]:
    """
    Load a model by name.

    Returns:
        (model, input_shape) where input_shape is (C, H, W)
    """
    if not TORCHVISION_AVAILABLE:
        raise RuntimeError(f"torchvision required for model '{model_name}'")

    config = MODEL_CONFIGS.get(model_name)
    if not config:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_CONFIGS.keys())}")

    # Get model from torchvision
    if hasattr(tv_models, model_name):
        model = getattr(tv_models, model_name)(weights=None)
    else:
        raise ValueError(f"Model '{model_name}' not found in torchvision.models")

    return model, config['input_shape']


def count_flops_fvcore(model: nn.Module, input_shape: Tuple[int, ...], batch_size: int = 1) -> int:
    """
    Count FLOPs using fvcore (if available).

    Returns total FLOPs (multiply and add counted as 2 ops).
    """
    try:
        from fvcore.nn import FlopCountAnalysis
        import logging

        # Suppress fvcore warnings about unsupported ops
        logging.getLogger('fvcore').setLevel(logging.ERROR)

        dummy_input = torch.randn(batch_size, *input_shape)
        flops = FlopCountAnalysis(model, dummy_input)
        # Silence the unsupported op warnings
        flops.unsupported_ops_warnings(False)
        return flops.total()
    except ImportError:
        return 0


def count_flops_manual(model: nn.Module, input_shape: Tuple[int, ...], batch_size: int = 1) -> int:
    """
    Count FLOPs by tracing the model and analyzing operations.

    This is a fallback when fvcore is not available.
    """
    # Create dummy input
    dummy_input = torch.randn(batch_size, *input_shape)

    total_flops = 0

    # Hook to count operations
    def count_hook(module, input, output):
        nonlocal total_flops

        if isinstance(module, nn.Linear):
            # Linear: 2 * in_features * out_features * batch
            batch = input[0].shape[0]
            total_flops += 2 * module.in_features * module.out_features * batch

        elif isinstance(module, nn.Conv2d):
            # Conv2d: 2 * Cout * Hout * Wout * Cin * Kh * Kw / groups
            batch = input[0].shape[0]
            out_h, out_w = output.shape[2], output.shape[3]
            kernel_ops = module.kernel_size[0] * module.kernel_size[1] * (module.in_channels // module.groups)
            total_flops += 2 * batch * module.out_channels * out_h * out_w * kernel_ops

        elif isinstance(module, nn.BatchNorm2d):
            # BN: 2 ops per element (normalize + scale/shift)
            total_flops += 2 * output.numel()

        elif isinstance(module, nn.LayerNorm):
            # LN: 2 ops per element
            total_flops += 2 * output.numel()

        elif isinstance(module, nn.MultiheadAttention):
            # Attention: Q*K^T + softmax + V
            # Approximate: 4 * seq_len^2 * embed_dim for QK^T and V matmuls
            # This is a rough estimate
            if len(input) >= 1 and input[0] is not None:
                seq_len = input[0].shape[0]
                embed_dim = module.embed_dim
                batch = input[0].shape[1] if len(input[0].shape) > 1 else 1
                total_flops += 4 * batch * seq_len * seq_len * embed_dim

    # Register hooks
    hooks = []
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.BatchNorm2d, nn.LayerNorm, nn.MultiheadAttention)):
            hooks.append(module.register_forward_hook(count_hook))

    # Run forward pass
    model.eval()
    with torch.no_grad():
        model(dummy_input)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    return total_flops


def count_flops(model: nn.Module, input_shape: Tuple[int, ...], batch_size: int = 1) -> int:
    """
    Count FLOPs for a model. Tries fvcore first, falls back to manual counting.
    """
    # Try fvcore first (more accurate)
    flops = count_flops_fvcore(model, input_shape, batch_size)
    if flops > 0:
        return flops

    # Fallback to manual counting
    return count_flops_manual(model, input_shape, batch_size)


def measure_inference_time(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    batch_size: int,
    device: str,
    precision: str = 'fp32',
    warmup_runs: int = 10,
    timed_runs: int = 100,
) -> Dict[str, float]:
    """
    Measure inference time with proper warmup and synchronization.

    Returns dict with mean, std, min, max latency in milliseconds.
    """
    model = model.to(device)
    model.eval()

    # Create input tensor
    dummy_input = torch.randn(batch_size, *input_shape, device=device)

    # Handle precision
    if precision == 'fp16':
        model = model.half()
        dummy_input = dummy_input.half()
    elif precision == 'bf16':
        model = model.to(torch.bfloat16)
        dummy_input = dummy_input.to(torch.bfloat16)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(dummy_input)
            if device == 'cuda':
                torch.cuda.synchronize()

    # Timed runs
    latencies = []
    with torch.no_grad():
        for _ in range(timed_runs):
            if device == 'cuda':
                torch.cuda.synchronize()

            start = time.perf_counter()
            _ = model(dummy_input)

            if device == 'cuda':
                torch.cuda.synchronize()

            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # Convert to ms

    import statistics
    return {
        'mean': statistics.mean(latencies),
        'std': statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
        'min': min(latencies),
        'max': max(latencies),
        'warmup_runs': warmup_runs,
        'timed_runs': timed_runs,
    }


def get_cpu_peak_tflops(precision: str = 'fp32') -> Tuple[Optional[float], Optional[str]]:
    """
    Get peak TFLOPS for the current CPU.

    Tries to detect CPU and look up known specs.
    Returns (peak_tflops, source) where source describes the lookup method.
    """
    try:
        import platform
        cpu_name = platform.processor()

        # Try py-cpuinfo for better detection
        try:
            import cpuinfo
            info = cpuinfo.get_cpu_info()
            cpu_name = info.get('brand_raw', cpu_name)
            hz_actual = info.get('hz_actual', (0, 0))
            if isinstance(hz_actual, tuple):
                freq_ghz = hz_actual[0] / 1e9 if hz_actual[0] > 1e6 else hz_actual[0]
            else:
                freq_ghz = float(hz_actual) / 1e9 if hz_actual > 1e6 else float(hz_actual)
        except ImportError:
            freq_ghz = None

        # Known CPU specs (approximate FP32 peak TFLOPS)
        # Peak = cores x freq x FLOPs/cycle (AVX2=16, AVX-512=32 for FP32)
        CPU_SPECS = {
            # Intel Desktop/Server
            'i9-14900K': {'cores': 24, 'freq': 5.8, 'flops_per_cycle': 64},  # AVX-512
            'i9-13900K': {'cores': 24, 'freq': 5.8, 'flops_per_cycle': 64},
            'i7-12700K': {'cores': 12, 'freq': 5.0, 'flops_per_cycle': 32},  # AVX2 (E-cores lower)
            'i7-12700': {'cores': 12, 'freq': 4.9, 'flops_per_cycle': 32},
            'Xeon Platinum 8380': {'cores': 40, 'freq': 3.0, 'flops_per_cycle': 64},
            'Xeon Gold 6248': {'cores': 20, 'freq': 2.5, 'flops_per_cycle': 64},

            # AMD
            'Ryzen 9 7950X': {'cores': 16, 'freq': 5.7, 'flops_per_cycle': 32},
            'Ryzen 9 5950X': {'cores': 16, 'freq': 4.9, 'flops_per_cycle': 32},
            'EPYC 7742': {'cores': 64, 'freq': 2.25, 'flops_per_cycle': 32},
            'EPYC 9654': {'cores': 96, 'freq': 2.4, 'flops_per_cycle': 32},

            # Apple Silicon (Neural Engine not counted, CPU only)
            'M1': {'cores': 8, 'freq': 3.2, 'flops_per_cycle': 16},
            'M1 Pro': {'cores': 10, 'freq': 3.2, 'flops_per_cycle': 16},
            'M1 Max': {'cores': 10, 'freq': 3.2, 'flops_per_cycle': 16},
            'M2': {'cores': 8, 'freq': 3.5, 'flops_per_cycle': 16},
            'M3 Max': {'cores': 16, 'freq': 4.0, 'flops_per_cycle': 16},
        }

        # Try to match CPU name
        for known_cpu, specs in CPU_SPECS.items():
            if known_cpu.lower() in cpu_name.lower():
                # Calculate peak TFLOPS
                peak = specs['cores'] * specs['freq'] * specs['flops_per_cycle'] / 1000
                return peak, f"spec ({known_cpu})"

        # Fallback: try to calculate from detected info
        if freq_ghz and freq_ghz > 0:
            import os
            try:
                cores = os.cpu_count() or 4
                # Assume AVX2 (16 FLOPs/cycle for FP32 FMA)
                # This is conservative - many CPUs have 2 FMA units = 32 FLOPs/cycle
                peak = cores * freq_ghz * 16 / 1000
                return peak, f"estimated ({cores} cores @ {freq_ghz:.1f} GHz, AVX2)"
            except Exception:
                pass

        return None, None

    except Exception:
        return None, None


def measure_cpu_blas_peak(precision: str = 'fp32', size: int = 2048) -> Tuple[float, str]:
    """
    Measure peak TFLOPS using CPU BLAS GEMM as a reference.
    """
    if precision == 'fp16':
        dtype = torch.float16
    elif precision == 'bf16':
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    a = torch.randn(size, size, dtype=dtype)
    b = torch.randn(size, size, dtype=dtype)

    # Warmup
    for _ in range(3):
        c = torch.matmul(a, b)

    # Measure
    start = time.perf_counter()

    num_runs = 10
    for _ in range(num_runs):
        c = torch.matmul(a, b)

    end = time.perf_counter()

    elapsed = (end - start) / num_runs
    flops = 2 * size * size * size  # GEMM FLOPs
    tflops = flops / elapsed / 1e12

    return tflops, f'cpu_blas (N={size})'


def get_gpu_peak_tflops(device: str, precision: str = 'fp32') -> Tuple[Optional[float], Optional[str]]:
    """
    Get peak TFLOPS for the current GPU.

    Tries multiple methods:
    1. Hardware database lookup
    2. nvidia-smi detection + known specs
    3. None if unknown

    Returns (peak_tflops, source) where source is 'spec', 'detected', etc.
    """
    if device != 'cuda':
        return None, None

    if not torch.cuda.is_available():
        return None, None

    gpu_name = torch.cuda.get_device_name(0)

    # Known GPU specs (FP32 Tensor Core TFLOPS for datacenter, CUDA for consumer)
    # Using FP16 Tensor Core as primary for transformer workloads
    GPU_SPECS = {
        # Datacenter - FP16 Tensor Core peaks
        'H100': {'fp16_tc': 989.4, 'fp32': 66.9, 'fp32_tc': 494.7},
        'H100 SXM': {'fp16_tc': 989.4, 'fp32': 66.9, 'fp32_tc': 494.7},
        'H100 PCIe': {'fp16_tc': 756.0, 'fp32': 51.2, 'fp32_tc': 378.0},
        'A100': {'fp16_tc': 312.0, 'fp32': 19.5, 'fp32_tc': 156.0},
        'A100-SXM': {'fp16_tc': 312.0, 'fp32': 19.5, 'fp32_tc': 156.0},
        'A100-PCIE': {'fp16_tc': 312.0, 'fp32': 19.5, 'fp32_tc': 156.0},
        'A100-SXM4-80GB': {'fp16_tc': 312.0, 'fp32': 19.5, 'fp32_tc': 156.0},
        'A100-SXM4-40GB': {'fp16_tc': 312.0, 'fp32': 19.5, 'fp32_tc': 156.0},
        'V100': {'fp16_tc': 125.0, 'fp32': 15.7, 'fp32_tc': 125.0},
        'V100-SXM2': {'fp16_tc': 125.0, 'fp32': 15.7, 'fp32_tc': 125.0},
        'V100S': {'fp16_tc': 130.0, 'fp32': 16.4},
        'T4': {'fp16_tc': 65.0, 'fp32': 8.1, 'int8_tc': 130.0},

        # Consumer - mostly CUDA core peaks (no proper TC support for training)
        'RTX 4090': {'fp16_tc': 330.0, 'fp32': 82.6},
        'RTX 4080': {'fp16_tc': 242.0, 'fp32': 48.7},
        'RTX 3090': {'fp16_tc': 142.0, 'fp32': 35.6},
        'RTX 3080': {'fp16_tc': 119.0, 'fp32': 29.8},
        'RTX 3070': {'fp16_tc': 81.0, 'fp32': 20.3},
        'RTX 2080 Ti': {'fp16': 26.9, 'fp32': 13.4},
        'RTX 2080': {'fp16': 20.1, 'fp32': 10.1},
    }

    # Try to match GPU name
    for known_gpu, specs in GPU_SPECS.items():
        if known_gpu.lower() in gpu_name.lower():
            # Select appropriate peak based on precision
            if precision == 'fp16':
                peak = specs.get('fp16_tc', specs.get('fp16', specs.get('fp32', 0)))
            elif precision == 'bf16':
                peak = specs.get('bf16_tc', specs.get('fp16_tc', specs.get('fp32', 0)))
            else:  # fp32
                # For FP32, use TC if available (TF32 mode), otherwise CUDA
                peak = specs.get('fp32_tc', specs.get('fp32', 0))

            return peak, f'spec ({known_gpu})'

    return None, None


def measure_cublas_peak(device: str, precision: str = 'fp32', size: int = 4096) -> Tuple[float, str]:
    """
    Measure peak TFLOPS using cuBLAS GEMM as a reference.

    This gives a realistic upper bound that the hardware can actually achieve.
    """
    if device != 'cuda':
        return 0.0, 'N/A'

    # Create matrices
    if precision == 'fp16':
        dtype = torch.float16
    elif precision == 'bf16':
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    a = torch.randn(size, size, device=device, dtype=dtype)
    b = torch.randn(size, size, device=device, dtype=dtype)

    # Warmup
    for _ in range(5):
        c = torch.matmul(a, b)
        torch.cuda.synchronize()

    # Measure
    torch.cuda.synchronize()
    start = time.perf_counter()

    num_runs = 20
    for _ in range(num_runs):
        c = torch.matmul(a, b)

    torch.cuda.synchronize()
    end = time.perf_counter()

    elapsed = (end - start) / num_runs
    flops = 2 * size * size * size  # GEMM FLOPs
    tflops = flops / elapsed / 1e12

    return tflops, f'cublas (N={size})'


def measure_model_efficiency(
    model_name: str,
    batch_size: int = 1,
    device: str = 'cuda',
    precision: str = 'fp32',
    warmup_runs: int = 10,
    timed_runs: int = 100,
    use_cublas_peak: bool = False,
    hardware_spec: Optional[str] = None,
    cached_peak: Optional[Tuple[float, str]] = None,  # (peak_tflops, source) to reuse
) -> EfficiencyResult:
    """
    Measure efficiency for a model.

    Args:
        model_name: Name of the model (e.g., 'vit_b_16', 'resnet50')
        batch_size: Batch size for inference
        device: 'cuda' or 'cpu'
        precision: 'fp32', 'fp16', or 'bf16'
        warmup_runs: Number of warmup iterations
        timed_runs: Number of timed iterations
        use_cublas_peak: If True, measure peak using cuBLAS GEMM
        hardware_spec: Optional hardware spec name for peak lookup
        cached_peak: Optional (peak_tflops, source) tuple to reuse instead of measuring

    Returns:
        EfficiencyResult with all measurements
    """
    # Load model
    model, input_shape = get_model(model_name)

    # Count FLOPs
    theoretical_flops = count_flops(model, input_shape, batch_size)
    theoretical_macs = theoretical_flops // 2

    # Measure inference time
    timing = measure_inference_time(
        model, input_shape, batch_size, device, precision,
        warmup_runs=warmup_runs, timed_runs=timed_runs
    )

    # Calculate throughput
    mean_latency_sec = timing['mean'] / 1000
    delivered_macs_per_sec = theoretical_macs / mean_latency_sec
    delivered_tflops = theoretical_flops / mean_latency_sec / 1e12

    # Get peak reference
    peak_tflops = None
    peak_source = None

    # Use cached peak if provided (avoids variability between measurements)
    if cached_peak is not None:
        peak_tflops, peak_source = cached_peak
    elif device == 'cuda':
        if use_cublas_peak:
            peak_tflops, peak_source = measure_cublas_peak(device, precision)
        else:
            peak_tflops, peak_source = get_gpu_peak_tflops(device, precision)
    else:  # CPU
        if use_cublas_peak:
            peak_tflops, peak_source = measure_cpu_blas_peak(precision)
        else:
            peak_tflops, peak_source = get_cpu_peak_tflops(precision)
            # If spec lookup failed, fall back to BLAS measurement
            if peak_tflops is None:
                peak_tflops, peak_source = measure_cpu_blas_peak(precision)

    # Calculate efficiency
    efficiency = None
    if peak_tflops and peak_tflops > 0:
        efficiency = delivered_tflops / peak_tflops

    return EfficiencyResult(
        model_name=model_name,
        batch_size=batch_size,
        device=device,
        precision=precision,
        theoretical_flops=theoretical_flops,
        theoretical_macs=theoretical_macs,
        warmup_runs=timing['warmup_runs'],
        timed_runs=timing['timed_runs'],
        mean_latency_ms=timing['mean'],
        std_latency_ms=timing['std'],
        min_latency_ms=timing['min'],
        max_latency_ms=timing['max'],
        delivered_macs_per_sec=delivered_macs_per_sec,
        delivered_tflops=delivered_tflops,
        peak_tflops=peak_tflops,
        peak_source=peak_source,
        efficiency=efficiency,
    )


def print_comparison_table(results: List[EfficiencyResult]):
    """Print a comparison table for multiple results."""
    print()
    print("=" * 120)
    print("MODEL EFFICIENCY COMPARISON")
    print("=" * 120)
    print()

    # Header
    print(f"{'Model':<20} {'Batch':>6} {'Prec':>6} {'GFLOPs':>10} {'Latency':>10} "
          f"{'Delivered':>10} {'Peak':>10} {'Efficiency':>10}")
    print(f"{'':20} {'Size':>6} {'':>6} {'':>10} {'(ms)':>10} "
          f"{'TFLOPS':>10} {'TFLOPS':>10} {'':>10}")
    print("-" * 120)

    for r in results:
        peak_str = f"{r.peak_tflops:.1f}" if r.peak_tflops else "N/A"
        eff_str = f"{r.efficiency*100:.1f}%" if r.efficiency else "N/A"

        print(f"{r.model_name:<20} {r.batch_size:>6} {r.precision:>6} "
              f"{r.theoretical_flops/1e9:>10.1f} {r.mean_latency_ms:>10.2f} "
              f"{r.delivered_tflops:>10.2f} {peak_str:>10} {eff_str:>10}")

    print("-" * 120)
    print()

    # Analysis
    if results and results[0].efficiency is not None:
        avg_eff = sum(r.efficiency for r in results if r.efficiency) / len([r for r in results if r.efficiency])
        print(f"Average Efficiency: {avg_eff*100:.1f}%")
        print()

        if avg_eff < 0.30:
            print("FINDING: Low efficiency (<30%) suggests peak TFLOPS specs are not achievable")
            print("         for real workloads. This supports the hypothesis that marketing")
            print("         numbers exceed sustainable compute capacity.")
        elif avg_eff < 0.50:
            print("FINDING: Moderate efficiency (30-50%) - typical for transformer workloads")
            print("         which are often memory-bound rather than compute-bound.")
        else:
            print("FINDING: Good efficiency (>50%) - workload is well-matched to hardware.")


def main():
    parser = argparse.ArgumentParser(
        description="Measure neural network model efficiency",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument("--model", type=str, required=False, default=None,
                       help="Model name(s), comma-separated (e.g., 'vit_b_16' or 'resnet18,resnet50')")
    parser.add_argument("--device", type=str, default='cuda', choices=['cuda', 'cpu'],
                       help="Device to run on (default: cuda)")
    parser.add_argument("--batch-size", type=str, default='1',
                       help="Batch size(s), comma-separated (default: 1)")
    parser.add_argument("--precision", type=str, default='fp32', choices=['fp32', 'fp16', 'bf16'],
                       help="Precision (default: fp32)")
    parser.add_argument("--warmup", type=int, default=10,
                       help="Number of warmup runs (default: 10)")
    parser.add_argument("--runs", type=int, default=100,
                       help="Number of timed runs (default: 100)")
    parser.add_argument("--cublas-peak", "--blas-peak", action='store_true',
                       dest='blas_peak',
                       help="Measure peak using BLAS GEMM instead of spec lookup")
    parser.add_argument("--hardware", type=str, default=None,
                       help="Hardware spec name for peak lookup (optional)")
    parser.add_argument("--list-models", action='store_true',
                       help="List available models and exit")

    args = parser.parse_args()

    if args.list_models:
        print("Available models:")
        for name in sorted(MODEL_CONFIGS.keys()):
            config = MODEL_CONFIGS[name]
            print(f"  {name:<25} input: {config['input_shape']}")
        return 0

    if not args.model:
        parser.error("--model is required unless --list-models is specified")

    # Check CUDA availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("ERROR: CUDA requested but not available")
        print("  torch.cuda.is_available() = False")
        return 1

    # Parse models and batch sizes
    model_names = [m.strip() for m in args.model.split(',')]
    batch_sizes = [int(b.strip()) for b in args.batch_size.split(',')]

    # Print header
    print()
    print("=" * 80)
    print("MODEL EFFICIENCY MEASUREMENT")
    print("=" * 80)
    print()

    if args.device == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU: {gpu_name}")

        # Show memory
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"Memory: {total_mem:.1f} GB")
    else:
        print("Device: CPU")

    print(f"Precision: {args.precision}")
    print(f"Warmup runs: {args.warmup}")
    print(f"Timed runs: {args.runs}")
    print()

    # Measure peak ONCE at the start and cache it for all measurements
    # This avoids variability from CPU frequency scaling between runs
    print("Measuring peak performance (BLAS GEMM)...")
    if args.device == 'cuda':
        if args.blas_peak:
            cached_peak = measure_cublas_peak(args.device, args.precision)
        else:
            cached_peak = get_gpu_peak_tflops(args.device, args.precision)
            if cached_peak[0] is None:
                cached_peak = measure_cublas_peak(args.device, args.precision)
    else:  # CPU
        if args.blas_peak:
            cached_peak = measure_cpu_blas_peak(args.precision)
        else:
            cached_peak = get_cpu_peak_tflops(args.precision)
            if cached_peak[0] is None:
                cached_peak = measure_cpu_blas_peak(args.precision)

    print(f"Peak: {cached_peak[0]:.2f} TFLOPS ({cached_peak[1]})")
    print()

    # Run measurements
    results = []

    for model_name in model_names:
        for batch_size in batch_sizes:
            print(f"Measuring {model_name} (batch={batch_size})...")

            try:
                result = measure_model_efficiency(
                    model_name=model_name,
                    batch_size=batch_size,
                    device=args.device,
                    precision=args.precision,
                    warmup_runs=args.warmup,
                    timed_runs=args.runs,
                    use_cublas_peak=args.blas_peak,
                    hardware_spec=args.hardware,
                    cached_peak=cached_peak,  # Use the same peak for all measurements
                )
                results.append(result)

                # Print individual result
                eff_str = f"{result.efficiency*100:.1f}%" if result.efficiency else "N/A"
                print(f"  -> {result.delivered_tflops:.2f} TFLOPS, efficiency: {eff_str}")

            except Exception as e:
                print(f"  -> ERROR: {e}")
                import traceback
                traceback.print_exc()

    # Print comparison table
    if results:
        print_comparison_table(results)

    return 0


if __name__ == "__main__":
    sys.exit(main())
