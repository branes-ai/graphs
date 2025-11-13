#!/usr/bin/env python3
"""
Model Factory for DNN Energy Analysis Tools

Provides unified model loading, tracing, and partitioning for architecture-specific
energy analysis CLI tools.

Supports:
- Built-in torchvision models (ResNet, MobileNet, EfficientNet, ViT, etc.)
- Custom PyTorch models from file paths
- Automatic Dynamo tracing and shape propagation
- Fusion-based graph partitioning
"""

import sys
from pathlib import Path
from typing import Optional, Tuple
import importlib.util

import torch
import torchvision.models as models
from torch.fx import GraphModule
from torch.fx.passes.shape_prop import ShapeProp

# Add repo root to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from graphs.transform.partitioning import FusionBasedPartitioner
from graphs.ir.structures import PartitionReport


# Built-in model definitions from torchvision
AVAILABLE_MODELS = {
    # ResNet family
    'resnet18': lambda: models.resnet18(weights=None),
    'resnet34': lambda: models.resnet34(weights=None),
    'resnet50': lambda: models.resnet50(weights=None),
    'resnet101': lambda: models.resnet101(weights=None),
    'resnet152': lambda: models.resnet152(weights=None),

    # MobileNet family
    'mobilenet_v2': lambda: models.mobilenet_v2(weights=None),
    'mobilenet_v3_small': lambda: models.mobilenet_v3_small(weights=None),
    'mobilenet_v3_large': lambda: models.mobilenet_v3_large(weights=None),

    # EfficientNet family
    'efficientnet_b0': lambda: models.efficientnet_b0(weights=None),
    'efficientnet_b1': lambda: models.efficientnet_b1(weights=None),
    'efficientnet_b2': lambda: models.efficientnet_b2(weights=None),
    'efficientnet_b3': lambda: models.efficientnet_b3(weights=None),
    'efficientnet_b4': lambda: models.efficientnet_b4(weights=None),
    'efficientnet_b5': lambda: models.efficientnet_b5(weights=None),

    # Vision Transformer (ViT)
    'vit_b_16': lambda: models.vit_b_16(weights=None),
    'vit_b_32': lambda: models.vit_b_32(weights=None),
    'vit_l_16': lambda: models.vit_l_16(weights=None),
    'vit_l_32': lambda: models.vit_l_32(weights=None),

    # Other CNNs
    'vgg16': lambda: models.vgg16(weights=None),
    'vgg19': lambda: models.vgg19(weights=None),
    'densenet121': lambda: models.densenet121(weights=None),
    'densenet161': lambda: models.densenet161(weights=None),
    'shufflenet_v2_x1_0': lambda: models.shufflenet_v2_x1_0(weights=None),
    'squeezenet1_0': lambda: models.squeezenet1_0(weights=None),
    'squeezenet1_1': lambda: models.squeezenet1_1(weights=None),

    # ConvNeXt
    'convnext_tiny': lambda: models.convnext_tiny(weights=None),
    'convnext_small': lambda: models.convnext_small(weights=None),
    'convnext_base': lambda: models.convnext_base(weights=None),
    'convnext_large': lambda: models.convnext_large(weights=None),
}


# Model-specific input shapes (most use 224x224, some need custom sizes)
MODEL_INPUT_SHAPES = {
    'efficientnet_b1': (1, 3, 240, 240),
    'efficientnet_b2': (1, 3, 260, 260),
    'efficientnet_b3': (1, 3, 300, 300),
    'efficientnet_b4': (1, 3, 380, 380),
    'efficientnet_b5': (1, 3, 456, 456),
    'vit_l_16': (1, 3, 224, 224),
    'vit_l_32': (1, 3, 224, 224),
}


def get_input_shape(model_name: str, batch_size: int = 1) -> Tuple[int, int, int, int]:
    """
    Get input shape for a model.

    Args:
        model_name: Name of the model
        batch_size: Batch size for inference

    Returns:
        (batch, channels, height, width) tuple
    """
    if model_name in MODEL_INPUT_SHAPES:
        _, c, h, w = MODEL_INPUT_SHAPES[model_name]
        return (batch_size, c, h, w)
    else:
        # Default: ImageNet 224x224
        return (batch_size, 3, 224, 224)


def load_builtin_model(model_name: str) -> torch.nn.Module:
    """
    Load a built-in torchvision model.

    Args:
        model_name: Name from AVAILABLE_MODELS

    Returns:
        PyTorch model in eval mode

    Raises:
        ValueError: If model name not recognized
    """
    if model_name not in AVAILABLE_MODELS:
        available = ', '.join(sorted(AVAILABLE_MODELS.keys()))
        raise ValueError(
            f"Unknown model '{model_name}'. Available models:\n{available}"
        )

    model = AVAILABLE_MODELS[model_name]()
    model.eval()
    return model


def load_custom_model(
    model_path: str,
    model_class: str,
    *args,
    **kwargs
) -> torch.nn.Module:
    """
    Load a custom PyTorch model from a file path.

    Args:
        model_path: Path to Python file containing the model
        model_class: Name of the model class to instantiate
        *args, **kwargs: Arguments to pass to model constructor

    Returns:
        PyTorch model in eval mode

    Raises:
        FileNotFoundError: If model file not found
        AttributeError: If model class not found in file
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Load module from file
    spec = importlib.util.spec_from_file_location("custom_model", model_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {model_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Get model class
    if not hasattr(module, model_class):
        raise AttributeError(
            f"Model class '{model_class}' not found in {model_path}"
        )

    ModelClass = getattr(module, model_class)
    model = ModelClass(*args, **kwargs)
    model.eval()
    return model


def trace_and_partition(
    model: torch.nn.Module,
    input_shape: Tuple[int, int, int, int],
    verbose: bool = False
) -> Tuple[GraphModule, PartitionReport]:
    """
    Trace model with Dynamo and partition using FusionBasedPartitioner.

    Args:
        model: PyTorch model to trace
        input_shape: (batch, channels, height, width)
        verbose: Print tracing and partitioning details

    Returns:
        (traced_model, partition_report) tuple
    """
    if verbose:
        print(f"Tracing model with input shape {input_shape}...")

    # Create example input
    example_input = torch.randn(*input_shape)

    # Dynamo tracing (symbolic trace)
    try:
        traced = torch.fx.symbolic_trace(model)
    except Exception as e:
        raise RuntimeError(f"Failed to trace model: {e}")

    # Shape propagation
    if verbose:
        print("Running shape propagation...")
    ShapeProp(traced).propagate(example_input)

    # Fusion-based partitioning
    if verbose:
        print("Partitioning graph with FusionBasedPartitioner...")

    partitioner = FusionBasedPartitioner()
    partition_report = partitioner.partition(traced)

    if verbose:
        print(f"Partitioned into {len(partition_report.subgraphs)} subgraphs")
        for i, subgraph in enumerate(partition_report.subgraphs):
            total_bytes = (subgraph.total_input_bytes + subgraph.total_output_bytes +
                          subgraph.total_weight_bytes + subgraph.internal_bytes)
            print(f"  Subgraph {i}: {subgraph.num_operators} ops, "
                  f"{subgraph.total_flops / 1e9:.2f} GFLOPs, "
                  f"{total_bytes / 1e6:.2f} MB")

    return traced, partition_report


def load_and_prepare_model(
    model_name: Optional[str] = None,
    model_path: Optional[str] = None,
    model_class: Optional[str] = None,
    batch_size: int = 1,
    verbose: bool = False,
    **model_kwargs
) -> Tuple[torch.nn.Module, GraphModule, PartitionReport, Tuple[int, int, int, int]]:
    """
    Load, trace, and partition a model (all-in-one).

    Args:
        model_name: Built-in model name (e.g., 'resnet18')
        model_path: Path to custom model file (alternative to model_name)
        model_class: Class name for custom model (required if model_path provided)
        batch_size: Batch size for inference
        verbose: Print details
        **model_kwargs: Additional kwargs for custom model constructor

    Returns:
        (original_model, traced_model, partition_report, input_shape) tuple

    Raises:
        ValueError: If neither model_name nor model_path provided
    """
    # Load model
    if model_name:
        if verbose:
            print(f"Loading built-in model: {model_name}")
        model = load_builtin_model(model_name)
        input_shape = get_input_shape(model_name, batch_size)
    elif model_path and model_class:
        if verbose:
            print(f"Loading custom model: {model_class} from {model_path}")
        model = load_custom_model(model_path, model_class, **model_kwargs)
        # For custom models, default to 224x224 ImageNet input
        input_shape = (batch_size, 3, 224, 224)
    else:
        raise ValueError(
            "Must provide either model_name or (model_path + model_class)"
        )

    # Trace and partition
    traced, partition_report = trace_and_partition(model, input_shape, verbose)

    return model, traced, partition_report, input_shape


def list_available_models() -> None:
    """Print all available built-in models."""
    print("Available Built-in Models:")
    print("=" * 60)

    categories = {
        'ResNet': ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'],
        'MobileNet': ['mobilenet_v2', 'mobilenet_v3_small', 'mobilenet_v3_large'],
        'EfficientNet': [k for k in AVAILABLE_MODELS if k.startswith('efficientnet_')],
        'Vision Transformer': [k for k in AVAILABLE_MODELS if k.startswith('vit_')],
        'ConvNeXt': [k for k in AVAILABLE_MODELS if k.startswith('convnext_')],
        'Other CNNs': ['vgg16', 'vgg19', 'densenet121', 'densenet161',
                       'shufflenet_v2_x1_0', 'squeezenet1_0', 'squeezenet1_1'],
    }

    for category, model_list in categories.items():
        print(f"\n{category}:")
        for model_name in model_list:
            if model_name in AVAILABLE_MODELS:
                print(f"  - {model_name}")


if __name__ == '__main__':
    # Demo usage
    import argparse

    parser = argparse.ArgumentParser(description='Model Factory Demo')
    parser.add_argument('--model', default='resnet18', help='Model name')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('--list', action='store_true', help='List available models')
    args = parser.parse_args()

    if args.list:
        list_available_models()
    else:
        print(f"Loading and tracing {args.model}...")
        model, traced, partition_report, input_shape = load_and_prepare_model(
            model_name=args.model,
            batch_size=args.batch_size,
            verbose=True
        )
        print(f"\nModel loaded successfully!")
        print(f"Input shape: {input_shape}")
        print(f"Partitioned into {len(partition_report.subgraphs)} subgraphs")
