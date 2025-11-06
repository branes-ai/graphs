#!/usr/bin/env python
"""
Graph Profiler - Computational Graph Characterization
======================================================

Profiles PyTorch models by characterizing each operator in the computational graph:
- Execution order and hierarchical structure
- Computational requirements (MACs, FLOPs)
- Memory bandwidth demands
- Parameter counts and tensor shapes
- Operation-level resource analysis
"""

import torch
import torchvision.models as models
from torch.fx import symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp
import argparse

from graphs.transform.partitioning import GraphPartitioner
from graphs.hardware.table_formatter import format_partition_table


# Model registry - maps model names to constructors
#
# To discover new models in torchvision:
#   python -c "import torchvision.models as m; print('\n'.join(sorted(m.list_models())))"
#
# To test if a model is FX-traceable:
#   python -c "from torch.fx import symbolic_trace; import torchvision.models as m; \
#              model = m.MODEL_NAME(weights=None); symbolic_trace(model); print('✓')"
#
# Note: Not all torchvision models are suitable for FX tracing:
#   - Detection models (Faster R-CNN, RetinaNet, etc.) have complex outputs
#   - Segmentation models (DeepLab, FCN) may need special handling
#   - Quantized models use different ops
#   - Some video models (R3D, etc.) need 5D inputs
#
# This registry is curated for classification models that work well with FX tracing.
# For automatic discovery, see the _build_model_registry_auto() function below.
#
MODEL_REGISTRY = {
    # ResNet family
    'resnet18': models.resnet18,
    'resnet34': models.resnet34,
    'resnet50': models.resnet50,
    'resnet101': models.resnet101,
    'resnet152': models.resnet152,
    # MobileNet family
    'mobilenet_v2': models.mobilenet_v2,
    'mobilenet_v3_large': models.mobilenet_v3_large,
    'mobilenet_v3_small': models.mobilenet_v3_small,
    # EfficientNet family
    'efficientnet_b0': models.efficientnet_b0,
    'efficientnet_b1': models.efficientnet_b1,
    'efficientnet_b4': models.efficientnet_b4,
    'efficientnet_b7': models.efficientnet_b7,
    # DenseNet family
    'densenet121': models.densenet121,
    'densenet161': models.densenet161,
    'densenet201': models.densenet201,
    # VGG family
    'vgg16': models.vgg16,
    'vgg19': models.vgg19,
    # Classic architectures
    'alexnet': models.alexnet,
    'squeezenet1_0': models.squeezenet1_0,
    # Modern CNNs
    'convnext_tiny': models.convnext_tiny,
    'convnext_small': models.convnext_small,
    'convnext_base': models.convnext_base,
    # Vision Transformers
    'vit_b_16': models.vit_b_16,
    'vit_b_32': models.vit_b_32,
    'vit_l_16': models.vit_l_16,
    # Swin Transformers
    'swin_t': models.swin_t,
    'swin_s': models.swin_s,
    'swin_b': models.swin_b,
    # RegNets
    'regnet_y_400mf': models.regnet_y_400mf,
    'regnet_y_800mf': models.regnet_y_800mf,
    'regnet_y_1_6gf': models.regnet_y_1_6gf,
    'regnet_y_3_2gf': models.regnet_y_3_2gf,
    'regnet_y_8gf': models.regnet_y_8gf,
}


def _build_model_registry_auto():
    """
    Automatically discover all FX-traceable models from torchvision.

    This function attempts to trace all available models and returns
    a registry of those that successfully trace.

    Note: This is slower than the manual registry (tests ~120 models)
    but ensures you have access to all compatible models.

    Usage:
        To use automatic discovery, replace MODEL_REGISTRY with:
        MODEL_REGISTRY = _build_model_registry_auto()
    """
    from torch.fx import symbolic_trace
    import warnings

    # Models to skip (known to be incompatible or not useful)
    SKIP_PATTERNS = [
        'rcnn',        # Detection models (complex outputs)
        'retinanet',   # Detection models
        'fcos',        # Detection models
        'ssd',         # Detection models
        'deeplabv3',   # Segmentation models
        'fcn',         # Segmentation models
        'lraspp',      # Segmentation models
        'raft',        # Optical flow models
        'r3d',         # Video models (need 5D input)
        'r2plus1d',    # Video models
        'mc3',         # Video models
        's3d',         # Video models
        'mvit',        # Video models
        'swin3d',      # Video models
        'quantized',   # Quantized models (different ops)
    ]

    registry = {}
    all_models = models.list_models()

    for model_name in sorted(all_models):
        # Skip known incompatible patterns
        if any(pattern in model_name for pattern in SKIP_PATTERNS):
            continue

        try:
            # Try to get the model constructor
            model_fn = getattr(models, model_name, None)
            if model_fn is None:
                continue

            # Try to instantiate and trace
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = model_fn(weights=None)
                model.eval()
                symbolic_trace(model)

            # Success! Add to registry
            registry[model_name] = model_fn

        except Exception:
            # Skip models that fail to trace
            pass

    return registry


def show_table(model_name: str, show_shapes: bool = False, input_shape=(1, 3, 224, 224)):
    """Show hierarchical table for a model"""

    print("=" * 100)
    print(f"HIERARCHICAL MODULE TABLE: {model_name}")
    print("=" * 100)

    # Load model
    if model_name not in MODEL_REGISTRY:
        print(f"Unknown model: {model_name}")
        print(f"Available models: {', '.join(sorted(MODEL_REGISTRY.keys()))}")
        return

    model = MODEL_REGISTRY[model_name](weights=None)
    model.eval()
    input_tensor = torch.randn(*input_shape)

    # Trace
    print("\n[1/3] Tracing with PyTorch FX...")
    fx_graph = symbolic_trace(model)
    ShapeProp(fx_graph).propagate(input_tensor)

    # Partition
    print("[2/3] Running graph partitioner...")
    partitioner = GraphPartitioner()
    report = partitioner.partition(fx_graph)

    # Format table
    print("[3/3] Formatting hierarchical table...")
    table = format_partition_table(fx_graph, report, show_shapes=show_shapes)

    # Display
    print("\n" + "=" * 100)
    print("GRAPH PROFILE")
    print("=" * 100)
    print()
    print(table)

    # Summary
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params / 1e6:.2f}M ({total_params:,})")
    print(f"Total FLOPs: {report.total_flops / 1e9:.3f} GFLOPs ({report.total_flops:,})")
    print(f"Total MACs: {report.total_macs / 1e9:.3f} GMACs ({report.total_macs:,})")

    total_memory = sum(sg.total_input_bytes + sg.total_output_bytes + sg.total_weight_bytes
                      for sg in report.subgraphs)
    print(f"Total memory: {total_memory / 1e6:.2f} MB ({total_memory:,} bytes)")

    print(f"\nSubgraphs: {len(report.subgraphs)}")
    print(f"Average AI: {report.average_arithmetic_intensity:.2f} FLOPs/byte")


def main():
    parser = argparse.ArgumentParser(
        description='Profile computational graphs: characterize execution order, compute (MACs/FLOPs), and memory demands',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Profile ResNet-18
  python cli/profile_graph.py --model resnet18

  # Show tensor shapes and parameter details
  python cli/profile_graph.py --model resnet18 --showshape

  # Profile MobileNet V2
  python cli/profile_graph.py --model mobilenet_v2
        """
    )

    parser.add_argument('--model', type=str, default='resnet18',
                       choices=sorted(MODEL_REGISTRY.keys()),
                       help='Model to analyze (40+ models available)')

    parser.add_argument('--showshape', action='store_true',
                       help='Show parameter shapes (weight/bias dimensions)')

    parser.add_argument('--input-shape', type=int, nargs=4, default=[1, 3, 224, 224],
                       metavar=('B', 'C', 'H', 'W'),
                       help='Input tensor shape')

    args = parser.parse_args()

    show_table(args.model, show_shapes=args.showshape, input_shape=tuple(args.input_shape))


if __name__ == "__main__":
    main()
