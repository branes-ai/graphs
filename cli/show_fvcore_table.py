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


def show_fvcore_table(model_name: str, max_depth: int = 4, input_shape=(1, 3, 224, 224)):
    """Display fvcore's hierarchical FLOP table for a model"""

    print("=" * 100)
    print(f"FVCORE HIERARCHICAL TABLE: {model_name}")
    print("=" * 100)

    # Load model
    if model_name == 'resnet18':
        model = models.resnet18(weights=None)
    elif model_name == 'resnet34':
        model = models.resnet34(weights=None)
    elif model_name == 'resnet50':
        model = models.resnet50(weights=None)
    elif model_name == 'mobilenet_v2':
        model = models.mobilenet_v2(weights=None)
    else:
        print(f"Unknown model: {model_name}")
        return

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
                       choices=['resnet18', 'resnet34', 'resnet50', 'mobilenet_v2'],
                       help='Model to analyze')

    parser.add_argument('--max-depth', type=int, default=4,
                       help='Maximum depth of hierarchy to display (default: 4)')

    parser.add_argument('--input-shape', type=int, nargs=4, default=[1, 3, 224, 224],
                       metavar=('B', 'C', 'H', 'W'),
                       help='Input tensor shape')

    args = parser.parse_args()

    show_fvcore_table(args.model, max_depth=args.max_depth, input_shape=tuple(args.input_shape))


if __name__ == "__main__":
    main()
