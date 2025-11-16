"""
Alma Integration for graphs Package

This module integrates Alma (https://github.com/saifhaq/alma) into the
graphs package validation workflow, providing multi-backend validation
and deployment optimization guidance.

Three-tier validation strategy:
1. Tier 1 (Quick): Inductor only - validate prediction accuracy (seconds)
2. Tier 2 (Core): Key deployment options - TensorRT, ONNX, OpenVINO (minutes)
3. Tier 3 (Extended): All 90+ options - comprehensive analysis (hours)

Requirements:
    pip install alma-torch

Usage:
    python experiments/alma/alma_integration.py --model resnet18 --tier 2
    python experiments/alma/alma_integration.py --model resnet18 --tier 3
    python experiments/alma/alma_integration.py --model simple --conversions TENSORRT ONNX_GPU
"""

import torch
from typing import Dict, Any, List, Optional, Tuple
import argparse
from dataclasses import dataclass
import json
import sys
from pathlib import Path

# Add dynamo experiments to path for inductor_validation import
sys.path.append(str(Path(__file__).parent.parent / "dynamo"))

# ============================================================================
# Monkey-patch to make ONNX Runtime optional for ARM64/Jetson compatibility
# ============================================================================
import importlib
import types

# Create a mock onnxruntime module if it's not available
def _create_mock_onnxruntime():
    """Create a mock onnxruntime module that raises errors on use."""
    mock_module = types.ModuleType('onnxruntime')

    # Create a mock ModuleSpec for compatibility with PyTorch dynamo
    import importlib.machinery
    mock_spec = importlib.machinery.ModuleSpec(
        name='onnxruntime',
        loader=None,
        origin='mock',
        is_package=False
    )
    mock_module.__spec__ = mock_spec
    mock_module.__file__ = '<mock>'
    mock_module.__package__ = None

    class MockInferenceSession:
        def __init__(self, *args, **kwargs):
            raise ImportError("onnxruntime not available on this platform (ARM64/Jetson)")

    mock_module.InferenceSession = MockInferenceSession
    return mock_module

# Try to import onnxruntime, if it fails, use mock
try:
    import onnxruntime
    ONNX_AVAILABLE = True
except (ImportError, OSError) as e:
    # ONNX Runtime not available (common on Jetson ARM64)
    # Install mock so alma's import doesn't crash
    sys.modules['onnxruntime'] = _create_mock_onnxruntime()
    ONNX_AVAILABLE = False
    print("⚠️  ONNX Runtime not available (ARM64/Jetson compatibility issue)")
    print("   ONNX-based conversions will be disabled")

# Try to import Alma (graceful degradation if not installed)
ALMA_AVAILABLE = False
try:
    from alma import benchmark_model
    from alma.benchmark import BenchmarkConfig
    from alma.benchmark.log import display_all_results
    ALMA_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Alma not installed or import failed: {e}")
    print("   Install with: pip install alma-torch")
    print("   Continuing with inductor-only validation...")

# Import our inductor validation
try:
    from inductor_validation import (
        validate_model_with_inductor,
        ValidationReport,
        compare_with_graphs_analysis
    )
except ImportError:
    print("⚠️  inductor_validation not found in dynamo experiments")
    print("   Some features may be limited...")
    ALMA_AVAILABLE = False


# ============================================================================
# Alma Integration Classes
# ============================================================================

@dataclass
class AlmaValidationResult:
    """Results from Alma multi-backend validation."""
    model_name: str
    hardware: str

    # Prediction
    predicted_latency_ms: float
    predicted_energy_j: Optional[float]

    # Quick validation (Tier 1)
    inductor_latency_ms: float
    inductor_error_percent: float

    # Alma results (Tier 2/3)
    alma_results: Dict[str, Any]  # conversion_name -> {latency, throughput, ...}

    # Analysis
    best_conversion: str
    best_latency_ms: float
    best_speedup_vs_eager: float
    best_speedup_vs_inductor: float

    # Recommendations
    deployment_recommendations: Dict[str, str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON export."""
        return {
            'model_name': self.model_name,
            'hardware': self.hardware,
            'prediction': {
                'latency_ms': self.predicted_latency_ms,
                'energy_j': self.predicted_energy_j,
            },
            'validation': {
                'inductor': {
                    'latency_ms': self.inductor_latency_ms,
                    'error_percent': self.inductor_error_percent,
                },
                'alma': {
                    'best_conversion': self.best_conversion,
                    'best_latency_ms': self.best_latency_ms,
                    'speedup_vs_eager': self.best_speedup_vs_eager,
                    'speedup_vs_inductor': self.best_speedup_vs_inductor,
                    'all_results': self.alma_results,
                }
            },
            'deployment_recommendations': self.deployment_recommendations
        }


# ============================================================================
# Conversion Presets
# ============================================================================

# Tier 1: Inductor only (inductor_validation.py)
TIER1_CONVERSIONS = ["EAGER", "COMPILE_INDUCTOR"]

# Tier 2: Core deployment options (~10 options, ~5 minutes)
TIER2_CONVERSIONS_GPU = [
    "EAGER",
    "COMPILE_INDUCTOR",
    "COMPILE_INDUCTOR_MAX_AUTOTUNE",
    "TENSORRT",
    "ONNX_GPU",
    "FP16+COMPILE_CUDAGRAPHS",
    "TORCHAO_QUANT_INT8+COMPILE_INDUCTOR",
]

TIER2_CONVERSIONS_CPU = [
    "EAGER",
    "COMPILE_INDUCTOR",
    "ONNX_CPU",
    "OPENVINO",
    "TORCHAO_QUANT_INT8",
]

# Tier 3: Extended analysis (~90 options, comprehensive)
TIER3_CONVERSIONS_GPU = TIER2_CONVERSIONS_GPU + [
    "COMPILE_OPENXLA",
    "COMPILE_TVM",
    "TENSORRT_FP16",
    "OPENVINO_FP16",
    "TORCHAO_QUANT_I4_WEIGHT_ONLY",
    "TORCHAO_QUANT_FP8",
    "TORCHAO_AUTOQUANT",
    "FP16",
    "BF16",
    "FP16+COMPILE_INDUCTOR",
    "BF16+TENSORRT",
    "EXPORT+COMPILE_INDUCTOR",
    "EXPORT+COMPILE_TENSORRT",
]


def _filter_onnx_conversions(conversions: List[str]) -> List[str]:
    """Filter out ONNX conversions if ONNX Runtime is not available."""
    if ONNX_AVAILABLE:
        return conversions

    # Remove any conversion with "ONNX" in the name
    filtered = [c for c in conversions if 'ONNX' not in c]

    if len(filtered) < len(conversions):
        removed = set(conversions) - set(filtered)
        print(f"   Filtered out ONNX conversions: {', '.join(sorted(removed))}")

    return filtered


def get_conversions_for_tier(tier: int, hardware: str) -> List[str]:
    """Get appropriate conversion list for validation tier and hardware."""
    if tier == 1:
        conversions = TIER1_CONVERSIONS
    elif tier == 2:
        if 'GPU' in hardware.upper() or 'Jetson-Orin-AGX' in hardware or 'A100' in hardware:
            conversions = TIER2_CONVERSIONS_GPU
        else:
            conversions = TIER2_CONVERSIONS_CPU
    elif tier == 3:
        conversions = TIER3_CONVERSIONS_GPU
    else:
        raise ValueError(f"Invalid tier: {tier}. Must be 1, 2, or 3")

    # Filter out ONNX conversions if ONNX Runtime is not available
    return _filter_onnx_conversions(conversions)


# ============================================================================
# Main Validation Functions
# ============================================================================

def validate_with_alma(
    model: torch.nn.Module,
    example_input: torch.Tensor,
    model_name: str,
    hardware: str = 'Jetson-Orin-AGX',
    tier: int = 2,
    conversions: Optional[List[str]] = None,
    predicted_latency_ms: Optional[float] = None,
    predicted_energy_j: Optional[float] = None,
    verbose: bool = True
) -> AlmaValidationResult:
    """
    Complete validation workflow with Alma multi-backend testing.

    Args:
        model: PyTorch model to validate
        example_input: Example input tensor
        model_name: Name of the model
        hardware: Hardware target (e.g., 'Jetson-Orin-AGX', 'Intel-i7-12700k', 'Coral-Edge-TPU', 'KPU-T256')
        tier: Validation tier (1=inductor only, 2=core options, 3=all options)
        conversions: Optional explicit list of conversions (overrides tier)
        predicted_latency_ms: Predicted latency from graphs.analysis
        predicted_energy_j: Predicted energy from graphs.analysis
        verbose: Print detailed output

    Returns:
        AlmaValidationResult with complete validation data
    """
    if not ALMA_AVAILABLE and tier > 1:
        print("\n⚠️  Alma not available, falling back to Tier 1 (inductor only)")
        tier = 1

    if verbose:
        print("\n" + "="*80)
        print(f"ALMA VALIDATION: {model_name} on {hardware}")
        print(f"Tier: {tier}")
        print("="*80)

    # Step 1: Quick validation with inductor (Tier 1)
    if verbose:
        print("\n" + "="*80)
        print("Tier 1: Inductor Validation (Quick)")
        print("="*80)

    inductor_result = validate_model_with_inductor(
        model, example_input, model_name, benchmark=True, verbose=verbose
    )

    inductor_error = 0.0
    if predicted_latency_ms:
        inductor_error = abs(predicted_latency_ms - inductor_result.inductor_time_ms) / \
                        inductor_result.inductor_time_ms * 100

        if verbose:
            print(f"\nPrediction vs Inductor:")
            print(f"  Predicted: {predicted_latency_ms:.2f} ms")
            print(f"  Actual:    {inductor_result.inductor_time_ms:.2f} ms")
            print(f"  Error:     {inductor_error:.1f}%")

    # If Tier 1 only, return early
    if tier == 1 or not ALMA_AVAILABLE:
        return AlmaValidationResult(
            model_name=model_name,
            hardware=hardware,
            predicted_latency_ms=predicted_latency_ms or 0.0,
            predicted_energy_j=predicted_energy_j,
            inductor_latency_ms=inductor_result.inductor_time_ms,
            inductor_error_percent=inductor_error,
            alma_results={'INDUCTOR': inductor_result.inductor_time_ms},
            best_conversion='INDUCTOR',
            best_latency_ms=inductor_result.inductor_time_ms,
            best_speedup_vs_eager=inductor_result.speedup,
            best_speedup_vs_inductor=1.0,
            deployment_recommendations={'quick_deploy': 'INDUCTOR'}
        )

    # Step 2: Alma multi-backend validation (Tier 2/3)
    if verbose:
        print("\n" + "="*80)
        print(f"Tier {tier}: Alma Multi-Backend Validation")
        print("="*80)

    # Get conversions list
    if conversions is None:
        conversions = get_conversions_for_tier(tier, hardware)
    else:
        # Filter user-provided conversions as well
        conversions = _filter_onnx_conversions(conversions)

    if verbose:
        print(f"\nTesting {len(conversions)} conversions:")
        for conv in conversions:
            print(f"  - {conv}")

    # Configure Alma
    device = torch.device("cuda" if torch.cuda.is_available() and 'GPU' in hardware.upper() else "cpu")
    config = BenchmarkConfig(
        n_samples=2048,
        batch_size=example_input.shape[0],
        device=device,
        multiprocessing=False,  # Disable multiprocessing to avoid hangs
    )

    # Prepare data for Alma
    # Alma expects data tensor directly (not dataloader for simplicity)
    # Repeat input to create multiple samples for benchmarking
    benchmark_data = example_input.repeat(config.n_samples, 1, 1, 1)

    # Run Alma benchmark
    if verbose:
        print("\nRunning Alma benchmark...")
        print(f"Benchmarking {benchmark_data.shape[0]} samples...")

    try:
        alma_results = benchmark_model(
            model, config, conversions, data=benchmark_data
        )

        if verbose:
            print("\nBenchmark complete")
            display_all_results(alma_results)

    except Exception as e:
        print(f"\n⚠️  Alma benchmark failed: {e}")
        print("Falling back to inductor-only results")
        return AlmaValidationResult(
            model_name=model_name,
            hardware=hardware,
            predicted_latency_ms=predicted_latency_ms or 0.0,
            predicted_energy_j=predicted_energy_j,
            inductor_latency_ms=inductor_result.inductor_time_ms,
            inductor_error_percent=inductor_error,
            alma_results={'INDUCTOR': inductor_result.inductor_time_ms},
            best_conversion='INDUCTOR',
            best_latency_ms=inductor_result.inductor_time_ms,
            best_speedup_vs_eager=inductor_result.speedup,
            best_speedup_vs_inductor=1.0,
            deployment_recommendations={'quick_deploy': 'INDUCTOR'}
        )

    # Step 3: Analyze results and generate recommendations
    if verbose:
        print("\n" + "="*80)
        print("Analysis & Recommendations")
        print("="*80)

    # Extract metrics
    # Convert Alma results format: total_inf_time is in seconds, convert to ms
    def get_latency_ms(result_dict):
        """Extract latency in ms from Alma result dict."""
        # total_inf_time is total time for all samples, divide by n_samples
        total_time_s = result_dict.get('total_inf_time', 0)
        n_samples = result_dict.get('total_samples', 1)
        return (total_time_s / n_samples) * 1000  # Convert to ms per sample

    eager_time = get_latency_ms(alma_results['EAGER']) if 'EAGER' in alma_results \
                 else inductor_result.eager_time_ms

    # Find best conversion
    best_name = min(alma_results.keys(),
                   key=lambda k: get_latency_ms(alma_results[k]))
    best_result = alma_results[best_name]
    best_latency = get_latency_ms(best_result)

    best_speedup_eager = eager_time / best_latency if best_latency > 0 else 0
    best_speedup_inductor = inductor_result.inductor_time_ms / best_latency if best_latency > 0 else 0

    # Print summary table
    if verbose:
        print(f"\n{'Conversion':<45} {'Latency (ms)':<15} {'Speedup (vs eager)':<20}")
        print("-"*80)

        for name in sorted(alma_results.keys(),
                          key=lambda k: get_latency_ms(alma_results[k])):
            result = alma_results[name]
            latency = get_latency_ms(result)
            speedup = eager_time / latency if latency > 0 else 0

            marker = " ← BEST" if name == best_name else ""
            if 'INDUCTOR' in name:
                marker += " ← PREDICTED"

            print(f"{name:<45} {latency:<15.2f} {speedup:<20.2f}x{marker}")

    # Generate deployment recommendations
    recommendations = _generate_deployment_recommendations(alma_results, hardware)

    if verbose:
        print("\n" + "="*80)
        print("Deployment Recommendations")
        print("="*80)
        for category, rec in recommendations.items():
            print(f"  {category}: {rec}")

        print(f"\nOptimization Headroom:")
        print(f"  {best_speedup_inductor:.2f}x improvement possible beyond inductor baseline")

    # Create result object
    alma_results_dict = {
        name: {
            'latency_ms': get_latency_ms(result),
            'throughput': result.get('throughput', 0),
            'speedup_vs_eager': eager_time / get_latency_ms(result) if get_latency_ms(result) > 0 else 0
        }
        for name, result in alma_results.items()
    }

    return AlmaValidationResult(
        model_name=model_name,
        hardware=hardware,
        predicted_latency_ms=predicted_latency_ms or 0.0,
        predicted_energy_j=predicted_energy_j,
        inductor_latency_ms=inductor_result.inductor_time_ms,
        inductor_error_percent=inductor_error,
        alma_results=alma_results_dict,
        best_conversion=best_name,
        best_latency_ms=best_latency,
        best_speedup_vs_eager=best_speedup_eager,
        best_speedup_vs_inductor=best_speedup_inductor,
        deployment_recommendations=recommendations
    )


def _generate_deployment_recommendations(
    alma_results: Dict[str, Any],
    hardware: str
) -> Dict[str, str]:
    """Generate deployment recommendations based on Alma results."""
    recommendations = {}

    # Helper to get latency from result dict
    def get_latency(result_dict):
        total_time_s = result_dict.get('total_inf_time', 0)
        n_samples = result_dict.get('total_samples', 1)
        return (total_time_s / n_samples) * 1000

    # GPU production
    if any('TENSORRT' in k for k in alma_results.keys()):
        trt_results = {k: v for k, v in alma_results.items() if 'TENSORRT' in k}
        best_trt = min(trt_results.items(), key=lambda x: get_latency(x[1]))
        recommendations['GPU Production'] = f"{best_trt[0]} (best GPU performance)"

    # Cross-platform
    if any('ONNX' in k for k in alma_results.keys()):
        onnx_results = {k: v for k, v in alma_results.items() if 'ONNX' in k}
        best_onnx = min(onnx_results.items(), key=lambda x: get_latency(x[1]))
        recommendations['Cross-platform'] = f"{best_onnx[0]} (portable)"

    # Edge devices
    if any('QUANT' in k for k in alma_results.keys()):
        quant_results = {k: v for k, v in alma_results.items() if 'QUANT' in k}
        best_quant = min(quant_results.items(), key=lambda x: get_latency(x[1]))
        recommendations['Edge devices'] = f"{best_quant[0]} (low memory)"

    # Intel hardware
    if 'OPENVINO' in alma_results:
        recommendations['Intel hardware'] = "OPENVINO (optimized for Intel)"

    # Quick deploy
    if 'COMPILE_INDUCTOR' in alma_results:
        recommendations['Quick deploy'] = "COMPILE_INDUCTOR (torch.compile, easy)"

    return recommendations


# ============================================================================
# Example Models
# ============================================================================

class SimpleModel(torch.nn.Module):
    """Simple CNN for testing."""
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 32, 3, padding=1)
        self.fc = torch.nn.Linear(32 * 56 * 56, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.nn.functional.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.nn.functional.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def get_model(model_name: str):
    """Get model by name."""
    if model_name == 'simple':
        return SimpleModel()
    elif model_name == 'resnet18':
        from torchvision.models import resnet18
        return resnet18(weights=None)
    elif model_name == 'resnet50':
        from torchvision.models import resnet50
        return resnet50(weights=None)
    else:
        raise ValueError(f"Unknown model: {model_name}")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Alma integration for graphs package validation"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="simple",
        choices=["simple", "resnet18", "resnet50"],
        help="Model to validate"
    )
    parser.add_argument(
        "--hardware",
        type=str,
        default="Jetson-Orin-AGX",
        help="Hardware target (e.g., Jetson-Orin-AGX, Intel-i7-12700k, Qualcomm-Hexagon, Coral-Edge-TPU, KPU-T256)"
    )
    parser.add_argument(
        "--tier",
        type=int,
        default=2,
        choices=[1, 2, 3],
        help="Validation tier (1=inductor, 2=core options, 3=all options)"
    )
    parser.add_argument(
        "--conversions",
        type=str,
        nargs='+',
        default=None,
        help="Explicit list of conversions to test"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save results to JSON file"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for inference"
    )

    args = parser.parse_args()

    # Check Alma availability
    if not ALMA_AVAILABLE and args.tier > 1:
        print("\n" + "="*80)
        print("ERROR: Alma not installed")
        print("="*80)
        print("\nTo use Tier 2 or 3 validation, install Alma:")
        print("  pip install alma-torch")
        print("\nFalling back to Tier 1 (inductor only)...")
        args.tier = 1

    # Get model
    print(f"\nLoading model: {args.model}")
    model = get_model(args.model)
    model.eval()

    example_input = torch.randn(args.batch_size, 3, 224, 224)

    # Run validation
    result = validate_with_alma(
        model,
        example_input,
        model_name=args.model,
        hardware=args.hardware,
        tier=args.tier,
        conversions=args.conversions,
        predicted_latency_ms=None,  # Would come from graphs.analysis in production
        verbose=True
    )

    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
