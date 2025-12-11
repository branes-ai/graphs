#!/usr/bin/env python
"""
Shape Collection Tool

Collect tensor shapes from all traceable TorchVision and HuggingFace models.

Usage:
    python cli/research/collect_shapes.py --output shapes.parquet
    python cli/research/collect_shapes.py --output shapes.csv --format csv
    python cli/research/collect_shapes.py --models resnet18 mobilenet_v2 vit_b_16
    python cli/research/collect_shapes.py --class CNN --batch-sizes 1 8 32
"""

import argparse
import sys
from pathlib import Path

# Add repo root to path
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / 'src'))

import torch
import torchvision.models as models
import warnings

from graphs.research.shape_collection import (
    TensorShapeRecord,
    ShapeExtractor,
    DNNClassifier,
    ShapeDatabase,
)


# Default models to analyze
DEFAULT_MODELS = [
    # ResNet family
    'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
    # MobileNet family
    'mobilenet_v2', 'mobilenet_v3_small', 'mobilenet_v3_large',
    # EfficientNet family
    'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3',
    # Vision Transformer
    'vit_b_16', 'vit_b_32', 'vit_l_16',
    # Swin Transformer
    'swin_t', 'swin_s', 'swin_b',
    # Other CNNs
    'vgg16', 'vgg19', 'densenet121', 'densenet161',
    'shufflenet_v2_x1_0', 'squeezenet1_0',
    # ConvNeXt
    'convnext_tiny', 'convnext_small', 'convnext_base',
    # RegNet
    'regnet_y_400mf', 'regnet_y_800mf', 'regnet_y_1_6gf',
]

# Model-specific input shapes
MODEL_INPUT_SHAPES = {
    'efficientnet_b1': (1, 3, 240, 240),
    'efficientnet_b2': (1, 3, 260, 260),
    'efficientnet_b3': (1, 3, 300, 300),
    'efficientnet_b4': (1, 3, 380, 380),
    'efficientnet_b5': (1, 3, 456, 456),
}


def get_input_shape(model_name: str, batch_size: int = 1):
    """Get input shape for a model."""
    if model_name in MODEL_INPUT_SHAPES:
        _, c, h, w = MODEL_INPUT_SHAPES[model_name]
        return (batch_size, c, h, w)
    return (batch_size, 3, 224, 224)


def load_model(model_name: str):
    """Load a torchvision model."""
    model_fn = getattr(models, model_name, None)
    if model_fn is None:
        return None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = model_fn(weights=None)
        model.eval()
    return model


def collect_shapes(
    model_names: list,
    batch_sizes: list = [1],
    verbose: bool = False,
) -> ShapeDatabase:
    """
    Collect shapes from specified models.

    Args:
        model_names: List of model names to analyze
        batch_sizes: List of batch sizes to test
        verbose: Print progress

    Returns:
        ShapeDatabase with all collected shapes
    """
    extractor = ShapeExtractor()
    classifier = DNNClassifier()
    database = ShapeDatabase()

    total = len(model_names) * len(batch_sizes)
    count = 0

    for model_name in model_names:
        model = load_model(model_name)
        if model is None:
            if verbose:
                print(f"  Skipping {model_name} (not found)")
            continue

        model_class = classifier.classify(model_name)

        for batch_size in batch_sizes:
            count += 1
            if verbose:
                print(f"[{count}/{total}] {model_name} (batch={batch_size})")

            input_shape = get_input_shape(model_name, batch_size)
            input_tensor = torch.randn(*input_shape)

            try:
                records = extractor.extract_from_model(
                    model, input_tensor, model_name, model_class
                )
                database.add_all(records)

                if verbose:
                    print(f"    Extracted {len(records)} layers")

            except Exception as e:
                if verbose:
                    print(f"    Failed: {str(e)[:50]}")

    return database


def main():
    parser = argparse.ArgumentParser(
        description='Collect tensor shapes from DNN models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Collect shapes from all default models
    python cli/research/collect_shapes.py --output shapes.parquet

    # Collect shapes for specific models
    python cli/research/collect_shapes.py --models resnet18 vit_b_16 --output shapes.csv

    # Collect shapes with multiple batch sizes
    python cli/research/collect_shapes.py --batch-sizes 1 8 32 64 --output shapes.parquet
        """,
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Output file path (.parquet, .csv, or .json)',
    )
    parser.add_argument(
        '--models', '-m',
        nargs='+',
        default=None,
        help='Specific models to analyze (default: all)',
    )
    parser.add_argument(
        '--batch-sizes', '-b',
        type=int,
        nargs='+',
        default=[1],
        help='Batch sizes to test (default: 1)',
    )
    parser.add_argument(
        '--class', '-c',
        dest='dnn_class',
        choices=['CNN', 'Encoder', 'Decoder', 'FullTransformer'],
        default=None,
        help='Filter to specific DNN class',
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Print progress',
    )
    parser.add_argument(
        '--list-models',
        action='store_true',
        help='List available models and exit',
    )

    args = parser.parse_args()

    if args.list_models:
        print("Available models:")
        print("=" * 60)
        classifier = DNNClassifier()
        for name in sorted(DEFAULT_MODELS):
            cls = classifier.classify(name)
            print(f"  {name:<30} ({cls})")
        return 0

    # Determine models to analyze
    if args.models:
        model_names = args.models
    else:
        model_names = DEFAULT_MODELS

    # Filter by class if specified
    if args.dnn_class:
        classifier = DNNClassifier()
        model_names = [m for m in model_names
                      if classifier.classify(m) == args.dnn_class]

    print(f"Collecting shapes from {len(model_names)} models")
    print(f"Batch sizes: {args.batch_sizes}")
    print()

    # Collect shapes
    database = collect_shapes(
        model_names,
        batch_sizes=args.batch_sizes,
        verbose=args.verbose,
    )

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    database.save(args.output)

    # Print summary
    stats = database.get_statistics()
    print()
    print("=" * 60)
    print("COLLECTION SUMMARY")
    print("=" * 60)
    print(f"Total records: {stats['total_records']}")
    print(f"Matmul operations: {stats['matmul_ops']}")
    print(f"Unique models: {stats['unique_models']}")
    print()
    print("Class distribution:")
    for cls, count in stats.get('class_distribution', {}).items():
        print(f"  {cls}: {count}")
    print()
    print(f"Output saved to: {args.output}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
