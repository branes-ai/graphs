#!/usr/bin/env python
"""
FVCore Table Display - Reference FLOP Analysis
===============================================

Displays fvcore's hierarchical table showing module structure and FLOP counts.
This serves as the reference implementation that our profile_graph.py is based on.

Note: fvcore reports MACs but labels them as "FLOPs"
"""

import torch
import torchvision.models as models
import sys
import argparse

try:
    from fvcore.nn import FlopCountAnalysis, flop_count_table
    FVCORE_AVAILABLE = True
except ImportError:
    FVCORE_AVAILABLE = False
    print("ERROR: fvcore not installed")
    print("Install with: pip install fvcore")
    sys.exit(1)


# Model registry - maps model names to constructors
#
# To discover new models in torchvision:
#   python -c "import torchvision.models as m; print('\n'.join(sorted(m.list_models())))"
#
# To find FX-traceable models automatically:
#   python cli/discover_models.py
#
# See profile_graph.py for automatic discovery via _build_model_registry_auto()
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


def show_fvcore_table(model_name: str, max_depth: int = 4, input_shape=(1, 3, 224, 224)):
    """Display fvcore's hierarchical FLOP table for a model"""

    print("=" * 100)
    print(f"FVCORE HIERARCHICAL TABLE: {model_name}")
    print("=" * 100)

    # Load model
    if model_name not in MODEL_REGISTRY:
        print(f"Unknown model: {model_name}")
        print(f"Available models: {', '.join(sorted(MODEL_REGISTRY.keys()))}")
        return

    model = MODEL_REGISTRY[model_name](weights=None)
    model.eval()
    input_tensor = torch.randn(*input_shape)

    print(f"\n[1/2] Loading {model_name}...")
    print(f"[2/2] Analyzing with fvcore (max_depth={max_depth})...")

    # Generate fvcore analysis
    flops_analysis = FlopCountAnalysis(model, input_tensor)

    # Display table
    print("\n" + "=" * 100)
    print("FVCORE TABLE")
    print("=" * 100)
    print()

    table = flop_count_table(
        flops_analysis,
        max_depth=max_depth
    )
    print(table)

    # Summary
    total_flops = flops_analysis.total()
    total_params = sum(p.numel() for p in model.parameters())

    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"\nTotal parameters: {total_params / 1e6:.2f}M ({total_params:,})")
    print(f"Total FLOPs (MACs): {total_flops / 1e9:.3f} GFLOPs ({total_flops:,})")
    print("\nNote: fvcore counts MACs (multiply-accumulate operations) but labels them as 'FLOPs'")
    print("      In standard definitions: 1 MAC = 2 FLOPs")


def main():
    parser = argparse.ArgumentParser(
        description='Display fvcore hierarchical FLOP table (reference implementation)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show ResNet-18 with default depth
  python cli/show_fvcore_table.py --model resnet18

  # Show with more hierarchy levels
  python cli/show_fvcore_table.py --model resnet18 --max-depth 5

  # Show MobileNet V2
  python cli/show_fvcore_table.py --model mobilenet_v2
        """
    )

    parser.add_argument('--model', type=str, default='resnet18',
                       choices=sorted(MODEL_REGISTRY.keys()),
                       help='Model to analyze (40+ models available)')

    parser.add_argument('--max-depth', type=int, default=4,
                       help='Maximum depth of hierarchy to display (default: 4)')

    parser.add_argument('--input-shape', type=int, nargs=4, default=[1, 3, 224, 224],
                       metavar=('B', 'C', 'H', 'W'),
                       help='Input tensor shape')

    args = parser.parse_args()

    show_fvcore_table(args.model, max_depth=args.max_depth, input_shape=tuple(args.input_shape))


if __name__ == "__main__":
    main()
