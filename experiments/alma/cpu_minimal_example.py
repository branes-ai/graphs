#!/usr/bin/env python3
"""
Minimal Alma Benchmarking Example for CPU-Only Environments

This example demonstrates how to use Alma for benchmarking PyTorch models on CPU.
It includes explicit CPU configuration and validation to ensure proper platform detection.

Key Configuration Points:
1. Device selection: Explicitly set to CPU with allow_cuda=False
2. Thread control: Set optimal thread count for your CPU
3. Memory management: Use small batch sizes and sample counts
4. Multiprocessing: Disabled to avoid hangs on some systems

Hardware: Intel Core i7-12700K (12 cores, 20 threads with HT)
Target Conversions: CPU-optimized backends only (EAGER, COMPILE_INDUCTOR, ONNX_CPU, OPENVINO)

Usage:
    python3 experiments/alma/cpu_minimal_example.py --model resnet18
    python3 experiments/alma/cpu_minimal_example.py --model resnet50 --batch-size 4
    python3 experiments/alma/cpu_minimal_example.py --model vit_b_16 --samples 64
"""

import torch
import torch.nn as nn
from typing import Dict, Any
import time
import os
import sys
import argparse

# ============================================================================
# CPU Configuration - CRITICAL for performance and stability
# ============================================================================

def configure_cpu_environment():
    """
    Configure PyTorch for optimal CPU performance.

    This function sets thread counts and validates the CPU environment.
    Call this BEFORE any model loading or benchmarking.
    """
    # Get CPU information
    import multiprocessing
    cpu_count = multiprocessing.cpu_count()

    print("\n" + "="*80)
    print("CPU ENVIRONMENT CONFIGURATION")
    print("="*80)
    print(f"System CPU count: {cpu_count}")

    # Set intra-op parallelism threads (for operations like matrix multiply)
    # Recommendation: Use physical cores, not logical threads
    # For i7-12700K: 12 cores (8 P-cores + 4 E-cores), but PyTorch sees 20 threads
    num_threads = min(12, cpu_count)  # Use physical cores
    torch.set_num_threads(num_threads)

    # Also set environment variables for underlying libraries
    os.environ['OMP_NUM_THREADS'] = str(num_threads)
    os.environ['MKL_NUM_THREADS'] = str(num_threads)
    os.environ['OPENBLAS_NUM_THREADS'] = str(num_threads)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(num_threads)
    os.environ['NUMEXPR_NUM_THREADS'] = str(num_threads)

    print(f"PyTorch threads set to: {torch.get_num_threads()}")
    print(f"OMP_NUM_THREADS: {os.environ['OMP_NUM_THREADS']}")
    print(f"MKL_NUM_THREADS: {os.environ['MKL_NUM_THREADS']}")

    # Validate CPU-only environment
    cuda_available = torch.cuda.is_available()
    print(f"\nCUDA available: {cuda_available}")
    if cuda_available:
        print("WARNING: CUDA is available but we'll force CPU-only for this example")

    # Validate device
    device = torch.device('cpu')
    test_tensor = torch.randn(10, 10, device=device)
    print(f"Test tensor device: {test_tensor.device}")
    print(f"Test tensor dtype: {test_tensor.dtype}")

    print("="*80)
    print("✓ CPU environment configured successfully")
    print("="*80 + "\n")

    return device, num_threads


# ============================================================================
# Model Factory
# ============================================================================

class SimpleCNN(nn.Module):
    """
    Minimal CNN for quick benchmarking.

    Architecture:
    - Conv2d (3->16, 3x3) + ReLU + MaxPool
    - Conv2d (16->32, 3x3) + ReLU + MaxPool
    - Flatten + Linear (32*56*56 -> 10)

    Input: (B, 3, 224, 224)
    Output: (B, 10)
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(32 * 56 * 56, 10)

    def forward(self, x):
        # Input: (B, 3, 224, 224)
        x = self.pool(torch.relu(self.conv1(x)))  # -> (B, 16, 112, 112)
        x = self.pool(torch.relu(self.conv2(x)))  # -> (B, 32, 56, 56)
        x = torch.flatten(x, 1)                   # -> (B, 32*56*56)
        x = self.fc(x)                            # -> (B, 10)
        return x


def get_model(model_name: str):
    """
    Get model by name.

    Supported models:
    - simple: SimpleCNN (1M params, fast)
    - resnet18, resnet34, resnet50, resnet101, resnet152
    - mobilenet_v2, mobilenet_v3_small, mobilenet_v3_large
    - efficientnet_b0, efficientnet_b1, efficientnet_b2
    - vit_b_16, vit_b_32, vit_l_16 (Vision Transformer)
    - convnext_tiny, convnext_small

    Args:
        model_name: Name of the model (case-insensitive, underscores or hyphens)

    Returns:
        Tuple of (model, model_display_name)
    """
    model_name_clean = model_name.lower().replace('-', '_')

    if model_name_clean == 'simple':
        return SimpleCNN(), "SimpleCNN"

    # Try torchvision models
    try:
        import torchvision.models as models

        # ResNet family
        if model_name_clean == 'resnet18':
            return models.resnet18(weights=None), "ResNet-18"
        elif model_name_clean == 'resnet34':
            return models.resnet34(weights=None), "ResNet-34"
        elif model_name_clean == 'resnet50':
            return models.resnet50(weights=None), "ResNet-50"
        elif model_name_clean == 'resnet101':
            return models.resnet101(weights=None), "ResNet-101"
        elif model_name_clean == 'resnet152':
            return models.resnet152(weights=None), "ResNet-152"

        # MobileNet family
        elif model_name_clean == 'mobilenet_v2':
            return models.mobilenet_v2(weights=None), "MobileNet-V2"
        elif model_name_clean == 'mobilenet_v3_small':
            return models.mobilenet_v3_small(weights=None), "MobileNet-V3-Small"
        elif model_name_clean == 'mobilenet_v3_large':
            return models.mobilenet_v3_large(weights=None), "MobileNet-V3-Large"

        # EfficientNet family
        elif model_name_clean == 'efficientnet_b0':
            return models.efficientnet_b0(weights=None), "EfficientNet-B0"
        elif model_name_clean == 'efficientnet_b1':
            return models.efficientnet_b1(weights=None), "EfficientNet-B1"
        elif model_name_clean == 'efficientnet_b2':
            return models.efficientnet_b2(weights=None), "EfficientNet-B2"

        # Vision Transformer family
        elif model_name_clean == 'vit_b_16':
            return models.vit_b_16(weights=None), "ViT-B/16"
        elif model_name_clean == 'vit_b_32':
            return models.vit_b_32(weights=None), "ViT-B/32"
        elif model_name_clean == 'vit_l_16':
            return models.vit_l_16(weights=None), "ViT-L/16"
        elif model_name_clean == 'vit_l_32':
            return models.vit_l_32(weights=None), "ViT-L/32"

        # ConvNeXt family
        elif model_name_clean == 'convnext_tiny':
            return models.convnext_tiny(weights=None), "ConvNeXt-Tiny"
        elif model_name_clean == 'convnext_small':
            return models.convnext_small(weights=None), "ConvNeXt-Small"
        elif model_name_clean == 'convnext_base':
            return models.convnext_base(weights=None), "ConvNeXt-Base"

    except ImportError as e:
        raise ImportError(f"torchvision not available: {e}")

    raise ValueError(
        f"Unknown model: '{model_name}'\n\n"
        f"Supported models:\n"
        f"  Lightweight:\n"
        f"    - simple (SimpleCNN, 1M params)\n"
        f"    - mobilenet_v2, mobilenet_v3_small, mobilenet_v3_large\n"
        f"    - efficientnet_b0, efficientnet_b1, efficientnet_b2\n"
        f"  Medium:\n"
        f"    - resnet18, resnet34, resnet50\n"
        f"    - convnext_tiny, convnext_small\n"
        f"  Large:\n"
        f"    - resnet101, resnet152\n"
        f"    - vit_b_16, vit_b_32, vit_l_16, vit_l_32\n"
        f"    - convnext_base\n"
    )


# ============================================================================
# Baseline Benchmarking (without Alma)
# ============================================================================

def benchmark_baseline(model: nn.Module, input_tensor: torch.Tensor,
                      num_runs: int = 100, warmup: int = 10) -> Dict[str, float]:
    """
    Baseline PyTorch eager mode benchmarking.

    This provides a reference point before trying Alma conversions.
    """
    print("\n" + "="*80)
    print("BASELINE BENCHMARKING (PyTorch Eager Mode)")
    print("="*80)

    model.eval()

    # Warmup
    print(f"Warmup: {warmup} runs...")
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_tensor)

    # Benchmark
    print(f"Benchmarking: {num_runs} runs...")
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()
            output = model(input_tensor)
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms

    # Statistics
    import statistics
    mean_time = statistics.mean(times)
    median_time = statistics.median(times)
    std_time = statistics.stdev(times) if len(times) > 1 else 0
    min_time = min(times)
    max_time = max(times)

    print(f"\nResults (over {num_runs} runs):")
    print(f"  Mean:   {mean_time:.3f} ms")
    print(f"  Median: {median_time:.3f} ms")
    print(f"  Std:    {std_time:.3f} ms")
    print(f"  Min:    {min_time:.3f} ms")
    print(f"  Max:    {max_time:.3f} ms")
    print(f"  Throughput: {1000 / mean_time:.2f} inferences/sec")

    return {
        'mean_ms': mean_time,
        'median_ms': median_time,
        'std_ms': std_time,
        'min_ms': min_time,
        'max_ms': max_time,
        'throughput': 1000 / mean_time
    }


# ============================================================================
# Alma Benchmarking (CPU-specific conversions)
# ============================================================================

def benchmark_with_alma_cpu(model: nn.Module, input_tensor: torch.Tensor,
                           n_samples: int = 128) -> Dict[str, Any]:
    """
    Benchmark model using Alma with CPU-specific conversions.

    CPU-Compatible Conversions:
    - EAGER: Standard PyTorch eager execution
    - COMPILE_INDUCTOR_DEFAULT: torch.compile with inductor backend
    - ONNX_CPU: ONNX Runtime CPU execution provider
    - COMPILE_OPENVINO: Intel OpenVINO (optimized for Intel CPUs)

    Configuration Notes:
    - device: Explicitly set to CPU
    - allow_cuda: Set to False to prevent GPU attempts
    - multiprocessing: Set to False to avoid hangs
    - n_samples: Configurable sample count (default: 128)
    - batch_size: Match input batch size

    Args:
        model: PyTorch model to benchmark
        input_tensor: Example input tensor
        n_samples: Number of samples for benchmarking (default: 128)

    Returns:
        Dictionary of results per conversion
    """
    try:
        from alma import benchmark_model
        from alma.benchmark import BenchmarkConfig
        from alma.benchmark.log import display_all_results
    except ImportError:
        print("\n⚠️  Alma not installed. Install with: pip install alma-torch")
        return {}

    print("\n" + "="*80)
    print("ALMA BENCHMARKING (CPU-Optimized Conversions)")
    print("="*80)

    # CPU-specific conversions that work without GPU
    # Note: Alma uses specific conversion names (see MODEL_CONVERSION_OPTIONS)
    cpu_conversions = [
        "EAGER",                      # Baseline PyTorch (always available)
        "COMPILE_INDUCTOR_DEFAULT",   # torch.compile with inductor backend
        "ONNX_CPU",                   # ONNX Runtime CPU (requires: pip install onnxruntime)
        "COMPILE_OPENVINO",           # OpenVINO backend (requires: pip install openvino)
    ]

    print(f"\nTesting {len(cpu_conversions)} CPU conversions:")
    for conv in cpu_conversions:
        print(f"  - {conv}")

    # ========================================================================
    # CRITICAL: Alma CPU Configuration
    # ========================================================================
    print("\nConfiguring Alma for CPU:")

    batch_size = input_tensor.shape[0]
    # n_samples passed as parameter

    config = BenchmarkConfig(
        n_samples=n_samples,  # Use parameter value
        batch_size=batch_size,
        device=torch.device('cpu'),     # ← EXPLICIT CPU DEVICE
        allow_cuda=False,                # ← DISABLE CUDA (critical!)
        allow_mps=False,                 # ← DISABLE MPS (macOS GPU)
        multiprocessing=False,           # ← DISABLE MP (avoid hangs)
        fail_on_error=False,             # ← CONTINUE ON ERROR (graceful)
        allow_device_override=False      # ← PREVENT AUTO DEVICE SELECTION
    )

    print(f"  ✓ Device: {config.device}")
    print(f"  ✓ CUDA disabled: allow_cuda={config.allow_cuda}")
    print(f"  ✓ MPS disabled: allow_mps={config.allow_mps}")
    print(f"  ✓ Multiprocessing: {config.multiprocessing}")
    print(f"  ✓ Samples: {config.n_samples}")
    print(f"  ✓ Batch size: {config.batch_size}")
    print(f"  ✓ Device override: {config.allow_device_override}")

    # Prepare benchmark data using DataLoader (Alma's preferred method)
    # Create a simple dataset
    from torch.utils.data import TensorDataset, DataLoader

    # Generate dataset (n_samples of individual inputs)
    dataset_inputs = input_tensor.repeat(n_samples, 1, 1, 1)  # (128, 3, 224, 224)
    dataset_labels = torch.zeros(n_samples, dtype=torch.long)  # dummy labels

    dataset = TensorDataset(dataset_inputs, dataset_labels)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    print(f"\nDataLoader setup:")
    print(f"  Dataset size: {len(dataset)}")
    print(f"  Batch size: {batch_size}")
    print(f"  Number of batches: {len(data_loader)}")
    print(f"  Sample input shape: {dataset_inputs[0].shape}")

    # ========================================================================
    # Run Alma Benchmark
    # ========================================================================
    print("\n" + "-"*80)
    print("Running Alma benchmark...")
    print("-"*80)

    try:
        results = benchmark_model(
            model=model,
            config=config,
            conversions=cpu_conversions,
            data_loader=data_loader  # Use data_loader instead of data
        )

        print("\n" + "="*80)
        print("ALMA RESULTS")
        print("="*80)

        # Display Alma's built-in results table
        display_all_results(results)

        # Parse results for comparison
        parsed_results = {}
        for conv_name, result_dict in results.items():
            if 'error' in result_dict:
                print(f"\n⚠️  {conv_name} failed: {result_dict.get('error', 'Unknown error')}")
                continue

            # Extract metrics
            total_time_s = result_dict.get('total_inf_time', 0)
            total_samples = result_dict.get('total_samples', 1)
            avg_latency_ms = (total_time_s / total_samples) * 1000
            throughput = result_dict.get('throughput', 0)

            parsed_results[conv_name] = {
                'latency_ms': avg_latency_ms,
                'throughput': throughput,
                'total_time_s': total_time_s,
                'total_samples': total_samples
            }

        # Print comparison table
        if parsed_results:
            print("\n" + "="*80)
            print("PERFORMANCE COMPARISON")
            print("="*80)
            print(f"{'Conversion':<30} {'Latency (ms)':<15} {'Throughput (inf/s)':<20}")
            print("-"*80)

            # Sort by latency (fastest first)
            sorted_results = sorted(parsed_results.items(),
                                   key=lambda x: x[1]['latency_ms'])

            eager_latency = parsed_results.get('EAGER', {}).get('latency_ms', 0)

            for conv_name, metrics in sorted_results:
                latency = metrics['latency_ms']
                throughput = metrics['throughput']

                # Calculate speedup vs eager
                speedup_str = ""
                if eager_latency > 0 and conv_name != 'EAGER':
                    speedup = eager_latency / latency
                    speedup_str = f" ({speedup:.2f}x vs EAGER)"

                print(f"{conv_name:<30} {latency:<15.3f} {throughput:<20.2f}{speedup_str}")

            # Recommendations
            print("\n" + "="*80)
            print("RECOMMENDATIONS FOR CPU DEPLOYMENT")
            print("="*80)

            best_conv = sorted_results[0][0]
            best_latency = sorted_results[0][1]['latency_ms']

            print(f"✓ Best performing: {best_conv} ({best_latency:.3f} ms)")

            if 'COMPILE_INDUCTOR' in parsed_results:
                inductor_speedup = eager_latency / parsed_results['COMPILE_INDUCTOR']['latency_ms']
                print(f"✓ torch.compile speedup: {inductor_speedup:.2f}x (easy to deploy)")

            if 'OPENVINO' in parsed_results:
                print(f"✓ OpenVINO available: Optimized for Intel CPUs")

            if 'ONNX_CPU' in parsed_results:
                print(f"✓ ONNX CPU available: Cross-platform deployment")

        return parsed_results

    except Exception as e:
        print(f"\n❌ Alma benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return {}


# ============================================================================
# Main Example
# ============================================================================

def main():
    """
    Main entry point demonstrating CPU-optimized Alma benchmarking.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Alma CPU Benchmarking - Multi-backend performance comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # SimpleCNN (default, fast)
  python3 %(prog)s

  # ResNet models
  python3 %(prog)s --model resnet18
  python3 %(prog)s --model resnet50 --batch-size 4

  # Vision Transformer
  python3 %(prog)s --model vit-b-16 --samples 64

  # MobileNet with larger batch
  python3 %(prog)s --model mobilenet-v2 --batch-size 8

Supported models:
  Lightweight: simple, mobilenet_v2, mobilenet_v3_small, efficientnet_b0
  Medium: resnet18, resnet34, resnet50, convnext_tiny
  Large: resnet101, vit_b_16, vit_l_16, convnext_base
        """
    )
    parser.add_argument(
        "--model",
        type=str,
        default="simple",
        help="Model to benchmark (default: simple). Use hyphens or underscores."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for inference (default: 1)"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=128,
        help="Number of samples for Alma benchmark (default: 128)"
    )
    parser.add_argument(
        "--baseline-runs",
        type=int,
        default=100,
        help="Number of runs for baseline benchmark (default: 100)"
    )

    args = parser.parse_args()

    print("\n" + "="*80)
    print("ALMA CPU BENCHMARKING")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Model: {args.model}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Alma samples: {args.samples}")
    print(f"  Baseline runs: {args.baseline_runs}")

    # ========================================================================
    # Step 1: Configure CPU Environment
    # ========================================================================
    device, num_threads = configure_cpu_environment()

    # ========================================================================
    # Step 2: Create Model and Input
    # ========================================================================
    print("\n" + "="*80)
    print("MODEL SETUP")
    print("="*80)

    try:
        model, model_display_name = get_model(args.model)
    except (ValueError, ImportError) as e:
        print(f"\n❌ Error loading model: {e}")
        return 1

    model.eval()
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {model_display_name}")
    print(f"Parameters: {total_params:,}")
    print(f"Device: {device}")

    # Create input
    input_tensor = torch.randn(args.batch_size, 3, 224, 224, device=device)
    print(f"\nInput shape: {input_tensor.shape}")
    print(f"Input device: {input_tensor.device}")
    print(f"Input dtype: {input_tensor.dtype}")

    # Validate forward pass
    try:
        with torch.no_grad():
            output = model(input_tensor)
        print(f"Output shape: {output.shape}")
        print("✓ Forward pass successful")
    except Exception as e:
        print(f"\n❌ Forward pass failed: {e}")
        return 1

    # ========================================================================
    # Step 3: Baseline Benchmarking
    # ========================================================================
    baseline_results = benchmark_baseline(model, input_tensor, num_runs=args.baseline_runs)

    # ========================================================================
    # Step 4: Alma Benchmarking
    # ========================================================================
    alma_results = benchmark_with_alma_cpu(model, input_tensor, n_samples=args.samples)

    # ========================================================================
    # Step 5: Final Summary
    # ========================================================================
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"✓ CPU threads: {num_threads}")
    print(f"✓ Model parameters: {total_params:,}")
    print(f"✓ Input shape: {input_tensor.shape}")
    print(f"✓ Baseline latency: {baseline_results['mean_ms']:.3f} ms")

    if alma_results:
        best_alma = min(alma_results.items(), key=lambda x: x[1]['latency_ms'])
        best_name, best_metrics = best_alma
        speedup = baseline_results['mean_ms'] / best_metrics['latency_ms']

        print(f"✓ Best Alma conversion: {best_name}")
        print(f"✓ Best Alma latency: {best_metrics['latency_ms']:.3f} ms")
        print(f"✓ Speedup vs baseline: {speedup:.2f}x")
    else:
        print("\n⚠️  Alma benchmarking was not available or failed")
        print("   This could be due to:")
        print("   - Alma not installed (pip install alma-torch)")
        print("   - Missing optional dependencies (onnxruntime, openvino)")
        print("   - Conversion backend errors")

    print("\n" + "="*80)
    print("✓ Example completed successfully")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
