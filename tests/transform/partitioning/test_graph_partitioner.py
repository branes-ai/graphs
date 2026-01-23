#!/usr/bin/env python
"""
Generalized Graph Partitioner Test

This test validates GraphPartitioner and ConcurrencyAnalyzer on any model.
It uses architecture-specific expected values but validates universal properties.
"""

import torch
import torchvision.models as models
from torch.fx import symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp
from dataclasses import dataclass
from typing import Optional, Dict, Any

import sys
sys.path.insert(0, '/home/stillwater/dev/branes/clones/graphs/src')

from graphs.transform.partitioning import GraphPartitioner
from graphs.estimation.concurrency import ConcurrencyAnalyzer


@dataclass
class ModelProfile:
    """Expected characteristics for a model architecture"""
    name: str
    model_fn: Any  # function that creates the model
    input_shape: tuple

    # Expected ranges (for validation)
    expected_flops_range: tuple  # (min, max) in GFLOPs
    expected_subgraph_range: tuple  # (min, max) subgraphs
    expected_avg_arithmetic_intensity_range: tuple  # (min, max) FLOPs/byte

    # Architectural characteristics
    has_depthwise_conv: bool = False
    has_grouped_conv: bool = False
    has_residual_connections: bool = False
    has_squeeze_excite: bool = False

    # Expected dominant operations
    dominant_op_types: list = None  # e.g., ['conv2d', 'batchnorm']

    # Parallelism expectations
    expected_max_threads_range: tuple = (10000, 10000000)

    def __post_init__(self):
        if self.dominant_op_types is None:
            self.dominant_op_types = []


# Model profiles for common architectures
MODEL_PROFILES = {
    "resnet18": ModelProfile(
        name="ResNet-18",
        model_fn=lambda: models.resnet18(weights=None),
        input_shape=(1, 3, 224, 224),
        expected_flops_range=(3.5, 5.0),  # GFLOPs
        expected_subgraph_range=(40, 80),
        expected_avg_arithmetic_intensity_range=(20, 50),
        has_residual_connections=True,
        dominant_op_types=['conv2d', 'batchnorm', 'relu']
    ),

    "resnet50": ModelProfile(
        name="ResNet-50",
        model_fn=lambda: models.resnet50(weights=None),
        input_shape=(1, 3, 224, 224),
        expected_flops_range=(8.0, 12.0),
        expected_subgraph_range=(120, 200),
        expected_avg_arithmetic_intensity_range=(20, 50),
        has_residual_connections=True,
        dominant_op_types=['conv2d', 'batchnorm', 'relu']
    ),

    "mobilenet_v2": ModelProfile(
        name="MobileNet-V2",
        model_fn=lambda: models.mobilenet_v2(weights=None),
        input_shape=(1, 3, 224, 224),
        expected_flops_range=(1.5, 2.5),
        expected_subgraph_range=(100, 200),
        expected_avg_arithmetic_intensity_range=(5, 30),
        has_depthwise_conv=True,
        has_residual_connections=True,
        dominant_op_types=['conv2d_depthwise', 'conv2d_pointwise', 'batchnorm']
    ),

    "mobilenet_v3_small": ModelProfile(
        name="MobileNet-V3-Small",
        model_fn=lambda: models.mobilenet_v3_small(weights=None),
        input_shape=(1, 3, 224, 224),
        expected_flops_range=(0.2, 0.5),
        expected_subgraph_range=(80, 150),
        expected_avg_arithmetic_intensity_range=(3, 25),
        has_depthwise_conv=True,
        has_squeeze_excite=True,
        has_residual_connections=True,
        dominant_op_types=['conv2d_depthwise', 'conv2d_pointwise']
    ),

    "mobilenet_v3_large": ModelProfile(
        name="MobileNet-V3-Large",
        model_fn=lambda: models.mobilenet_v3_large(weights=None),
        input_shape=(1, 3, 224, 224),
        expected_flops_range=(1.0, 1.5),
        expected_subgraph_range=(100, 180),
        expected_avg_arithmetic_intensity_range=(5, 30),
        has_depthwise_conv=True,
        has_squeeze_excite=True,
        has_residual_connections=True,
        dominant_op_types=['conv2d_depthwise', 'conv2d_pointwise']
    ),

    "efficientnet_b0": ModelProfile(
        name="EfficientNet-B0",
        model_fn=lambda: models.efficientnet_b0(weights=None),
        input_shape=(1, 3, 224, 224),
        expected_flops_range=(2.0, 3.0),
        expected_subgraph_range=(150, 250),
        expected_avg_arithmetic_intensity_range=(5, 30),
        has_depthwise_conv=True,
        has_squeeze_excite=True,
        has_residual_connections=True,
        dominant_op_types=['conv2d_depthwise', 'conv2d_pointwise']
    ),

    "efficientnet_b2": ModelProfile(
        name="EfficientNet-B2",
        model_fn=lambda: models.efficientnet_b2(weights=None),
        input_shape=(1, 3, 260, 260),  # EfficientNet uses different input sizes
        expected_flops_range=(3.5, 5.0),
        expected_subgraph_range=(150, 250),
        expected_avg_arithmetic_intensity_range=(5, 30),
        has_depthwise_conv=True,
        has_squeeze_excite=True,
        has_residual_connections=True,
        dominant_op_types=['conv2d_depthwise', 'conv2d_pointwise']
    ),

    "vgg16": ModelProfile(
        name="VGG-16",
        model_fn=lambda: models.vgg16(weights=None),
        input_shape=(1, 3, 224, 224),
        expected_flops_range=(15.0, 20.0),
        expected_subgraph_range=(30, 60),
        expected_avg_arithmetic_intensity_range=(40, 80),
        has_residual_connections=False,
        dominant_op_types=['conv2d', 'relu', 'maxpool']
    ),
}


class GeneralizedPartitionerTest:
    """Generalized test framework for graph partitioner"""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.results = {}

    def test_model(self, model_profile: ModelProfile) -> Dict[str, Any]:
        """Test partitioner on a specific model"""

        if self.verbose:
            print("=" * 80)
            print(f"Testing {model_profile.name}")
            print("=" * 80)

        try:
            # Load model
            if self.verbose:
                print(f"\n[1/5] Loading {model_profile.name}...")
            model = model_profile.model_fn()
            model.eval()

            # FX trace
            if self.verbose:
                print("[2/5] FX tracing...")
            input_tensor = torch.randn(*model_profile.input_shape)

            try:
                fx_graph = symbolic_trace(model)
            except Exception as e:
                return {
                    'success': False,
                    'error': f"FX tracing failed: {e}",
                    'model': model_profile.name
                }

            # Shape propagation
            if self.verbose:
                print("[3/5] Shape propagation...")
            shape_prop = ShapeProp(fx_graph)
            shape_prop.propagate(input_tensor)

            # Partition
            if self.verbose:
                print("[4/5] Partitioning graph...")
            partitioner = GraphPartitioner()
            partition_report = partitioner.partition(fx_graph)

            # Concurrency analysis
            if self.verbose:
                print("[5/5] Analyzing concurrency...")
            analyzer = ConcurrencyAnalyzer()
            concurrency = analyzer.analyze(partition_report)
            partition_report.concurrency = concurrency

            # Validate
            validation_results = self._validate(model_profile, partition_report, concurrency)

            return {
                'success': True,
                'model': model_profile.name,
                'partition_report': partition_report,
                'concurrency': concurrency,
                'validation': validation_results
            }

        except Exception as e:
            import traceback
            return {
                'success': False,
                'error': f"Unexpected error: {e}",
                'traceback': traceback.format_exc(),
                'model': model_profile.name
            }

    def _validate(self, profile: ModelProfile, report, concurrency) -> Dict[str, Any]:
        """Validate partition results against expected values"""

        checks = []

        # Universal checks (apply to all models)

        # Check 1: Non-zero subgraphs
        checks.append({
            'name': 'Non-zero subgraphs',
            'passed': report.total_subgraphs > 0,
            'value': report.total_subgraphs,
            'expected': '> 0',
            'universal': True
        })

        # Check 2: Non-zero FLOPs
        checks.append({
            'name': 'Non-zero FLOPs',
            'passed': report.total_flops > 0,
            'value': f"{report.total_flops / 1e9:.2f} G",
            'expected': '> 0',
            'universal': True
        })

        # Check 3: Parallelism detected
        avg_threads = (sum(sg.parallelism.total_threads for sg in report.subgraphs
                          if sg.parallelism) / max(1, report.total_subgraphs))
        checks.append({
            'name': 'Thread-level parallelism',
            'passed': avg_threads > 100,
            'value': f"{avg_threads:,.0f} threads avg",
            'expected': '> 100',
            'universal': True
        })

        # Check 4: Concurrency analysis exists
        checks.append({
            'name': 'Concurrency analysis',
            'passed': concurrency.total_subgraphs > 0,
            'value': f"{concurrency.num_stages} stages, {concurrency.max_parallel_ops_per_stage} max parallel",
            'expected': 'Valid stages',
            'universal': True
        })

        # Check 5: Critical path exists
        checks.append({
            'name': 'Critical path',
            'passed': concurrency.critical_path_length > 0,
            'value': f"{concurrency.critical_path_length} ops, {concurrency.critical_path_flops / 1e9:.2f} GFLOPs",
            'expected': '> 0',
            'universal': True
        })

        # Architecture-specific checks

        # Check 6: FLOPs in expected range
        flops_gflops = report.total_flops / 1e9
        in_range = profile.expected_flops_range[0] <= flops_gflops <= profile.expected_flops_range[1]
        checks.append({
            'name': 'FLOPs in expected range',
            'passed': in_range,
            'value': f"{flops_gflops:.2f} G",
            'expected': f"{profile.expected_flops_range[0]:.1f}-{profile.expected_flops_range[1]:.1f} G",
            'universal': False
        })

        # Check 7: Subgraph count in expected range
        in_range = profile.expected_subgraph_range[0] <= report.total_subgraphs <= profile.expected_subgraph_range[1]
        checks.append({
            'name': 'Subgraph count in range',
            'passed': in_range,
            'value': report.total_subgraphs,
            'expected': f"{profile.expected_subgraph_range[0]}-{profile.expected_subgraph_range[1]}",
            'universal': False
        })

        # Check 8: Arithmetic intensity in range
        in_range = (profile.expected_avg_arithmetic_intensity_range[0] <=
                   report.average_arithmetic_intensity <=
                   profile.expected_avg_arithmetic_intensity_range[1])
        checks.append({
            'name': 'Arithmetic intensity in range',
            'passed': in_range,
            'value': f"{report.average_arithmetic_intensity:.1f} FLOPs/byte",
            'expected': f"{profile.expected_avg_arithmetic_intensity_range[0]}-{profile.expected_avg_arithmetic_intensity_range[1]}",
            'universal': False
        })

        # Check 9: Depthwise convolutions detected (if expected)
        if profile.has_depthwise_conv:
            has_depthwise = 'conv2d_depthwise' in report.operation_type_counts
            checks.append({
                'name': 'Depthwise conv detected',
                'passed': has_depthwise,
                'value': report.operation_type_counts.get('conv2d_depthwise', 0),
                'expected': '> 0',
                'universal': False
            })

        # Check 10: Dominant operation types present
        if profile.dominant_op_types:
            dominant_found = sum(1 for op in profile.dominant_op_types
                               if op in report.operation_type_counts)
            checks.append({
                'name': 'Dominant ops present',
                'passed': dominant_found >= len(profile.dominant_op_types) * 0.5,
                'value': f"{dominant_found}/{len(profile.dominant_op_types)} types found",
                'expected': f">= {len(profile.dominant_op_types) // 2}",
                'universal': False
            })

        # Compute pass rate
        passed = sum(1 for c in checks if c['passed'])
        total = len(checks)
        universal_passed = sum(1 for c in checks if c['passed'] and c['universal'])
        universal_total = sum(1 for c in checks if c['universal'])

        return {
            'checks': checks,
            'passed': passed,
            'total': total,
            'pass_rate': passed / total if total > 0 else 0,
            'universal_passed': universal_passed,
            'universal_total': universal_total,
            'universal_pass_rate': universal_passed / universal_total if universal_total > 0 else 0
        }

    def print_results(self, result: Dict[str, Any]):
        """Print detailed results"""

        if not result['success']:
            print(f"\n✗ {result['model']} FAILED")
            print(f"  Error: {result['error']}")
            if 'traceback' in result:
                print(f"\n{result['traceback']}")
            return

        report = result['partition_report']
        concurrency = result['concurrency']
        validation = result['validation']

        # Summary
        print(f"\n{'=' * 80}")
        print("PARTITION SUMMARY")
        print('=' * 80)
        print(report.summary_stats())

        # Concurrency
        print(f"\n{'=' * 80}")
        print("CONCURRENCY SUMMARY")
        print('=' * 80)
        print(concurrency.explanation)

        # Validation results
        print(f"\n{'=' * 80}")
        print("VALIDATION RESULTS")
        print('=' * 80)

        for check in validation['checks']:
            status = '✓' if check['passed'] else '✗'
            universal = ' [UNIVERSAL]' if check['universal'] else ''
            print(f"{status} {check['name']}{universal}")
            print(f"    Value: {check['value']}")
            print(f"    Expected: {check['expected']}")

        print(f"\nOverall: {validation['passed']}/{validation['total']} checks passed "
              f"({validation['pass_rate']*100:.0f}%)")
        print(f"Universal checks: {validation['universal_passed']}/{validation['universal_total']} passed "
              f"({validation['universal_pass_rate']*100:.0f}%)")

    def test_all(self, model_names: list = None):
        """Test multiple models"""

        if model_names is None:
            model_names = list(MODEL_PROFILES.keys())

        results = []

        for model_name in model_names:
            if model_name not in MODEL_PROFILES:
                print(f"Warning: Unknown model '{model_name}', skipping...")
                continue

            profile = MODEL_PROFILES[model_name]
            result = self.test_model(profile)
            results.append(result)

            if self.verbose:
                self.print_results(result)

        # Summary across all models
        print(f"\n{'=' * 80}")
        print("SUMMARY ACROSS ALL MODELS")
        print('=' * 80)

        successful = [r for r in results if r['success']]
        failed = [r for r in results if not r['success']]

        print(f"Tested: {len(results)} models")
        print(f"Successful: {len(successful)}")
        print(f"Failed: {len(failed)}")

        if failed:
            print("\nFailed models:")
            for r in failed:
                print(f"  - {r['model']}: {r['error']}")

        if successful:
            print("\nValidation summary:")
            for r in successful:
                v = r['validation']
                print(f"  {r['model']}: {v['passed']}/{v['total']} checks "
                     f"({v['universal_passed']}/{v['universal_total']} universal)")

        return results


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Test graph partitioner on models')
    parser.add_argument('models', nargs='*',
                       help='Models to test (default: all)',
                       default=None)
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress verbose output')

    args = parser.parse_args()

    tester = GeneralizedPartitionerTest(verbose=not args.quiet)

    if args.models:
        results = tester.test_all(args.models)
    else:
        # Test a representative subset
        results = tester.test_all(['resnet18', 'mobilenet_v2', 'efficientnet_b0'])

    # Exit with error if any test failed
    failed = [r for r in results if not r['success']]
    exit(1 if failed else 0)


if __name__ == "__main__":
    main()
