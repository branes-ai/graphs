#!/usr/bin/env python
"""
Model Discovery Tool
====================

Tests which torchvision models are FX-traceable and can be added to the registry.
"""

import torch
import torchvision.models as models
from torch.fx import symbolic_trace
import warnings
import argparse


def test_model_traceability(model_name: str, verbose: bool = False) -> bool:
    """Test if a model is FX-traceable"""
    try:
        model_fn = getattr(models, model_name, None)
        if model_fn is None:
            if verbose:
                print(f"✗ {model_name:<30} - No constructor found")
            return False

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = model_fn(weights=None)
            model.eval()
            symbolic_trace(model)

        if verbose:
            print(f"✓ {model_name:<30} - FX-traceable")
        return True

    except Exception as e:
        if verbose:
            error_msg = str(e)[:50]
            print(f"✗ {model_name:<30} - {error_msg}")
        return False


def discover_models(verbose: bool = False, skip_patterns: list = None):
    """Discover all FX-traceable models"""

    # Default skip patterns
    if skip_patterns is None:
        skip_patterns = [
            'rcnn', 'retinanet', 'fcos', 'ssd',  # Detection
            'deeplabv3', 'fcn', 'lraspp',         # Segmentation
            'raft',                                # Optical flow
            'r3d', 'r2plus1d', 'mc3', 's3d',      # Video (5D input)
            'mvit', 'swin3d',                      # Video transformers
            'quantized',                           # Quantized models
        ]

    all_models = models.list_models()
    traceable = []
    skipped = []
    failed = []

    print("=" * 80)
    print("TORCHVISION MODEL DISCOVERY")
    print("=" * 80)
    print(f"\nTotal models in torchvision: {len(all_models)}")
    print(f"Skip patterns: {', '.join(skip_patterns)}\n")

    if verbose:
        print("Testing models:\n")

    for model_name in sorted(all_models):
        # Check skip patterns
        if any(pattern in model_name for pattern in skip_patterns):
            skipped.append(model_name)
            continue

        # Test traceability
        if test_model_traceability(model_name, verbose=verbose):
            traceable.append(model_name)
        else:
            failed.append(model_name)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\nFX-traceable:  {len(traceable)} models")
    print(f"  Failed:        {len(failed)} models")
    print(f"  Skipped:       {len(skipped)} models (detection/segmentation/video/quantized)")

    # Show traceable models by family
    print("\n" + "=" * 80)
    print("FX-TRACEABLE MODELS BY FAMILY")
    print("=" * 80)

    families = {}
    for name in traceable:
        family = name.split('_')[0]
        families.setdefault(family, []).append(name)

    for family, model_names in sorted(families.items()):
        print(f"\n{family.upper()} ({len(model_names)}):")
        for name in model_names:
            print(f"  {name}")

    # Show failed models if requested
    if failed and verbose:
        print("\n" + "=" * 80)
        print("FAILED MODELS")
        print("=" * 80)
        for name in failed:
            print(f"  {name}")

    return traceable, failed, skipped


def generate_registry_code(traceable_models: list):
    """Generate Python code for MODEL_REGISTRY"""
    print("\n" + "=" * 80)
    print("GENERATED REGISTRY CODE")
    print("=" * 80)
    print("\nCopy this to replace MODEL_REGISTRY in profile_graph.py:\n")
    print("MODEL_REGISTRY = {")

    # Group by family
    families = {}
    for name in traceable_models:
        family = name.split('_')[0]
        families.setdefault(family, []).append(name)

    for family, model_names in sorted(families.items()):
        print(f"    # {family.capitalize()} family")
        for name in sorted(model_names):
            print(f"    '{name}': models.{name},")

    print("}")


def main():
    parser = argparse.ArgumentParser(
        description='Discover FX-traceable models in torchvision',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick summary
  python cli/discover_models.py

  # Verbose output showing each model test
  python cli/discover_models.py --verbose

  # Generate registry code
  python cli/discover_models.py --generate-code

  # Test a specific model
  python cli/discover_models.py --test-model resnet18

  # Scan registery
  python cli/discover_models.py --skip-patterns fcos vit ssd
        """
    )

    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Show detailed output for each model')

    parser.add_argument('--generate-code', '-g', action='store_true',
                       help='Generate MODEL_REGISTRY code')

    parser.add_argument('--test-model', type=str,
                       help='Test a specific model')

    parser.add_argument('--skip-patterns', type=str, nargs='*',
                       help='Space-separated skip patterns to filter out torchvision models')

    args = parser.parse_args()

    if args.test_model:
        # Test single model
        print(f"Testing {args.test_model}...")
        if test_model_traceability(args.test_model, verbose=True):
            print(f"\n✓ {args.test_model} is FX-traceable!")
        else:
            print(f"\n✗ {args.test_model} is NOT FX-traceable")
    else:
        # Full discovery
        traceable, failed, skipped = discover_models(verbose=args.verbose, skip_patterns=args.skip_patterns)

        if args.generate_code:
            generate_registry_code(traceable)


if __name__ == "__main__":
    main()
