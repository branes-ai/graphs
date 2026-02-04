#!/usr/bin/env python3
"""
Run golden reference validation for all supported models.

Outputs a summary table with estimated vs measured latency and error percentage.

Usage:
    ./cli/validate_all_models.py --hardware Jetson-Orin-AGX --thermal-profile 50W
"""

import argparse
import sys
import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Any
import time

# Add project root to path
sys.path.insert(0, '/mnt/nvme/dev/branes/clones/graphs/src')

from graphs.estimation.unified_analyzer import UnifiedAnalyzer
from graphs.estimation.roofline import RooflineAnalyzer
from graphs.transform.partitioning.fusion_partitioner import FusionBasedPartitioner
from torch.fx.passes.shape_prop import ShapeProp


# All supported models
SUPPORTED_MODELS = [
    # ResNet family
    'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
    # MobileNet family
    'mobilenet_v2', 'mobilenet_v3_small', 'mobilenet_v3_large',
    # EfficientNet family
    'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2',
    # VGG family
    'vgg11', 'vgg16', 'vgg19',
    # ViT family
    'vit_b_16', 'vit_b_32', 'vit_l_16', 'vit_l_32', 'vit_h_14', 'maxvit_t',
    # Segmentation
    'deeplabv3_resnet50', 'fcn_resnet50',
]


def measure_full_model_inference(model: nn.Module, input_tensor: torch.Tensor,
                                  device: str = 'cuda',
                                  warmup_runs: int = 50,
                                  timing_runs: int = 100) -> Tuple[float, float, float]:
    """Measure full model inference time with CUDA events (gold standard)."""
    if device == 'cuda':
        model = model.cuda()
        input_tensor = input_tensor.cuda()

    model.eval()

    # Warmup
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(input_tensor)

    if device == 'cuda':
        torch.cuda.synchronize()

    # Timing runs
    times = []
    with torch.no_grad():
        for _ in range(timing_runs):
            if device == 'cuda':
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                _ = model(input_tensor)
                end_event.record()
                torch.cuda.synchronize()
                times.append(start_event.elapsed_time(end_event))
            else:
                start = time.perf_counter()
                _ = model(input_tensor)
                end = time.perf_counter()
                times.append((end - start) * 1000)

    times.sort()
    median_time = times[len(times) // 2]
    min_time = times[0]
    max_time = times[-1]

    return median_time, min_time, max_time


def validate_model(model_name: str, hardware_name: str, thermal_profile: str,
                   device: str, batch_size: int, quiet: bool = True) -> Dict[str, Any]:
    """Validate a single model and return results."""
    analyzer = UnifiedAnalyzer(verbose=False)

    try:
        # Create model
        model, input_tensor, display_name = analyzer._create_model(model_name, batch_size)
        model.eval()

        # Create hardware mapper
        hardware_mapper = analyzer._create_hardware_mapper(hardware_name, thermal_profile=thermal_profile)

        # Trace model
        try:
            from torch.fx import symbolic_trace
            traced = symbolic_trace(model)
        except Exception as e:
            return {
                'model': model_name,
                'display_name': display_name,
                'status': 'TRACE_FAILED',
                'error': str(e)[:80],
            }

        # Shape propagation
        shape_prop = ShapeProp(traced)
        shape_prop.propagate(input_tensor)

        # Partition into subgraphs
        partitioner = FusionBasedPartitioner()
        partition_report = partitioner.partition(traced)

        # Get roofline estimation
        hardware = hardware_mapper.resource_model
        roofline_analyzer = RooflineAnalyzer(hardware, thermal_profile=thermal_profile)
        roofline_report = roofline_analyzer.analyze(partition_report.subgraphs, partition_report)
        estimated_ms = roofline_report.total_latency * 1000

        # Measure actual inference
        measured_ms, min_ms, max_ms = measure_full_model_inference(
            model, input_tensor, device=device,
            warmup_runs=50, timing_runs=100
        )

        # Calculate error
        error_pct = (estimated_ms - measured_ms) / measured_ms * 100

        # Determine rating
        abs_error = abs(error_pct)
        if abs_error < 10:
            rating = 'EXCELLENT'
        elif abs_error < 25:
            rating = 'GOOD'
        elif abs_error < 50:
            rating = 'FAIR'
        else:
            rating = 'POOR'

        return {
            'model': model_name,
            'display_name': display_name,
            'status': 'OK',
            'estimated_ms': estimated_ms,
            'measured_ms': measured_ms,
            'min_ms': min_ms,
            'max_ms': max_ms,
            'error_pct': error_pct,
            'rating': rating,
        }

    except Exception as e:
        import traceback
        return {
            'model': model_name,
            'display_name': model_name,
            'status': 'FAILED',
            'error': str(e)[:80],
            'traceback': traceback.format_exc(),
        }


def main():
    parser = argparse.ArgumentParser(
        description='Run golden reference validation for all supported models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--hardware', required=True,
                        help='Target hardware (e.g., Jetson-Orin-AGX, H100)')
    parser.add_argument('--thermal-profile', default=None,
                        help='Thermal/power profile (e.g., 50W, 30W)')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'],
                        help='Device to run on')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size')
    parser.add_argument('--models', nargs='+', default=None,
                        help='Specific models to test (default: all)')
    parser.add_argument('--skip', nargs='+', default=[],
                        help='Models to skip')
    args = parser.parse_args()

    if not torch.cuda.is_available() and args.device == 'cuda':
        print("CUDA not available, switching to CPU")
        args.device = 'cpu'

    # Determine which models to run
    models_to_run = args.models if args.models else SUPPORTED_MODELS
    models_to_run = [m for m in models_to_run if m not in args.skip]

    # Header
    profile_str = f" ({args.thermal_profile})" if args.thermal_profile else ""
    print("=" * 95)
    print(f"  GOLDEN REFERENCE VALIDATION - {args.hardware}{profile_str}")
    print("=" * 95)
    print(f"  Device: {torch.cuda.get_device_name() if args.device == 'cuda' else 'CPU'}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Models: {len(models_to_run)}")
    print("=" * 95)
    print()

    # Results table header
    print(f"  {'Model':<22} {'Est (ms)':>10} {'Meas (ms)':>10} {'Error':>10} {'Rating':<10}")
    print("  " + "-" * 70)

    results = []
    import warnings
    warnings.filterwarnings('ignore')

    for model_name in models_to_run:
        result = validate_model(
            model_name, args.hardware, args.thermal_profile,
            args.device, args.batch_size
        )
        results.append(result)

        if result['status'] == 'OK':
            print(f"  {model_name:<22} {result['estimated_ms']:>10.2f} {result['measured_ms']:>10.2f} "
                  f"{result['error_pct']:>+9.1f}% {result['rating']:<10}")
        else:
            error_msg = result.get('error', 'Unknown error')
            print(f"  {model_name:<22} {'--':>10} {'--':>10} {'--':>10} {result['status']:<10} ({error_msg})")

        # Clear CUDA cache between models
        if args.device == 'cuda':
            torch.cuda.empty_cache()

    print("  " + "-" * 70)
    print()

    # Summary statistics
    successful = [r for r in results if r['status'] == 'OK']
    if successful:
        errors = [abs(r['error_pct']) for r in successful]
        avg_error = sum(errors) / len(errors)
        max_error = max(errors)
        excellent = sum(1 for r in successful if r['rating'] == 'EXCELLENT')
        good = sum(1 for r in successful if r['rating'] == 'GOOD')
        fair = sum(1 for r in successful if r['rating'] == 'FAIR')
        poor = sum(1 for r in successful if r['rating'] == 'POOR')

        print("  SUMMARY")
        print("  " + "-" * 40)
        print(f"  Total models:     {len(models_to_run)}")
        print(f"  Successful:       {len(successful)}")
        print(f"  Failed/Skipped:   {len(results) - len(successful)}")
        print()
        print(f"  Mean |error|:     {avg_error:.1f}%")
        print(f"  Max |error|:      {max_error:.1f}%")
        print()
        print(f"  EXCELLENT (<10%): {excellent}")
        print(f"  GOOD (10-25%):    {good}")
        print(f"  FAIR (25-50%):    {fair}")
        print(f"  POOR (>50%):      {poor}")
    print()


if __name__ == '__main__':
    main()
