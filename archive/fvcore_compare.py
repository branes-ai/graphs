#!/usr/bin/env python
"""
FVCore FLOP Comparison CLI
===========================

Cross-validates our partitioner's FLOP counting against Facebook's fvcore library.
Shows per-layer comparisons and identifies discrepancies.

Usage:
    # Compare ResNet-18
    python cli/fvcore_compare.py --model resnet18

    # Detailed per-layer comparison
    python cli/fvcore_compare.py --model mobilenet_v2 --detailed

    # Show only discrepancies
    python cli/fvcore_compare.py --model resnet50 --show-discrepancies

Requirements:
    pip install fvcore
"""

import torch
import torchvision.models as models
from torch.fx import symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp
import sys
import argparse
from typing import Dict, List, Tuple
sys.path.insert(0, 'src')

from graphs.characterize.graph_partitioner import GraphPartitioner

try:
    from fvcore.nn import FlopCountAnalysis, flop_count_table
    FVCORE_AVAILABLE = True
except ImportError:
    FVCORE_AVAILABLE = False
    print("ERROR: fvcore not installed")
    print("Install with: pip install fvcore")


class FVCoreComparator:
    """Compare our FLOP counting against fvcore"""

    SUPPORTED_MODELS = {
        'resnet18': models.resnet18,
        'resnet34': models.resnet34,
        'resnet50': models.resnet50,
        'mobilenet_v2': models.mobilenet_v2,
        'efficientnet_b0': models.efficientnet_b0,
    }

    def __init__(self):
        self.model_name = None
        self.model = None
        self.input_tensor = None
        self.fx_graph = None

    def load_model(self, model_name: str, input_shape=(1, 3, 224, 224)):
        """Load and prepare model"""
        print("=" * 80)
        print(f"FVCore FLOP Comparison: {model_name}")
        print("=" * 80)

        if model_name not in self.SUPPORTED_MODELS:
            print(f"Error: Unknown model '{model_name}'")
            return False

        print(f"\n[1/3] Loading {model_name}...")
        self.model = self.SUPPORTED_MODELS[model_name](weights=None)
        self.model.eval()
        self.model_name = model_name

        print("[2/3] Creating input tensor...")
        self.input_tensor = torch.randn(*input_shape)

        print("[3/3] Tracing with PyTorch FX...")
        try:
            self.fx_graph = symbolic_trace(self.model)
            ShapeProp(self.fx_graph).propagate(self.input_tensor)
        except Exception as e:
            print(f"Error tracing model: {e}")
            return False

        return True

    def count_flops_our_tool(self) -> Tuple[int, int, Dict]:
        """Count FLOPs using our partitioner"""
        print("\nCounting FLOPs with our partitioner...")
        partitioner = GraphPartitioner()
        report = partitioner.partition(self.fx_graph)

        # Build per-operation breakdown
        per_op = {}
        for sg in report.subgraphs:
            per_op[sg.node_name] = {
                'flops': sg.flops,
                'macs': sg.macs,
                'type': sg.operation_type.value,
                'ai': sg.arithmetic_intensity
            }

        return report.total_flops, report.total_macs, per_op

    def count_flops_fvcore(self) -> Tuple[int, Dict]:
        """Count FLOPs using fvcore"""
        print("Counting FLOPs with fvcore...")
        flops_analysis = FlopCountAnalysis(self.model, self.input_tensor)
        total_flops = flops_analysis.total()

        # Get per-module breakdown
        per_module = flops_analysis.by_module()

        return total_flops, per_module

    def compare_totals(self, our_flops: int, our_macs: int, fvcore_total: int):
        """Compare total FLOP counts"""
        print("\n" + "=" * 80)
        print("TOTAL FLOP/MAC COMPARISON")
        print("=" * 80)

        print(f"\nOur Tool:")
        print(f"  FLOPs (2×MACs): {our_flops / 1e9:.3f} GFLOPs ({our_flops:,})")
        print(f"  MACs:           {our_macs / 1e9:.3f} GMACs  ({our_macs:,})")

        print(f"\nFVCore:")
        print(f"  Total (MACs):   {fvcore_total / 1e9:.3f} GFLOPs* ({fvcore_total:,})")
        print(f"                  *FVCore counts MACs but labels them as 'FLOPs'")

        diff_flops = abs(our_flops - fvcore_total)
        diff_pct_flops = (diff_flops / fvcore_total) * 100 if fvcore_total > 0 else 0

        diff_macs = abs(our_macs - fvcore_total)
        diff_pct_macs = (diff_macs / fvcore_total) * 100 if fvcore_total > 0 else 0

        print(f"\nComparison:")
        print(f"  Our FLOPs vs FVCore: {diff_flops / 1e9:.3f} GFLOPs ({diff_pct_flops:.2f}% difference)")
        print(f"  Our MACs vs FVCore:  {diff_macs / 1e6:.1f} MMACs ({diff_pct_macs:.2f}% difference)")

        if diff_pct_macs < 1.0:
            print("\nStatus: ✓ EXCELLENT MATCH (< 1% difference in MACs)")
            print("Note: Our FLOPs are 2× MACs (standard definition), while FVCore reports MACs as 'FLOPs'")
        elif diff_pct_macs < 5.0:
            print("\nStatus: ✓ GOOD MATCH (< 5% difference in MACs)")
        elif diff_pct_macs < 10.0:
            print("\nStatus: ACCEPTABLE (< 10% difference in MACs)")
        else:
            print("\nStatus: ✗ NEEDS INVESTIGATION (> 10% difference in MACs)")

    def compare_per_layer(self, our_ops: Dict, fvcore_modules: Dict,
                         detailed: bool = False, show_discrepancies: bool = False):
        """Compare per-layer FLOP counts"""
        print("\n" + "=" * 80)
        print("PER-LAYER COMPARISON (Conv2d/Linear only)")
        print("=" * 80)

        # Filter fvcore_modules to remove only the empty string (total) entry
        # Keep everything else including top-level modules (conv1, fc, etc.)
        fvcore_filtered = {name: flops for name, flops in fvcore_modules.items()
                          if name}  # Skip only empty string

        # Match our operations to fvcore modules
        # Our FX names use underscores: layer1_0_conv1
        # FVCore uses dots: layer1.0.conv1
        matches = []
        unmatched_ours = []

        def normalize_name(name: str) -> str:
            """Normalize name for matching: convert underscores to dots"""
            return name.replace('_', '.')

        # Try to match by name similarity
        for our_name, our_data in our_ops.items():
            # Only compare Conv2d and Linear (skip activations, pooling, etc.)
            if our_data['type'] not in ['conv2d', 'conv2d_pointwise', 'conv2d_depthwise', 'linear']:
                continue

            # Normalize our name for matching
            our_name_normalized = normalize_name(our_name)

            # Try to find matching fvcore module
            matched = False
            our_macs = our_data['macs']  # Use MACs for comparison

            # First try exact match
            if our_name_normalized in fvcore_filtered:
                fv_flops = fvcore_filtered[our_name_normalized]
                diff = abs(our_macs - fv_flops)
                diff_pct = (diff / fv_flops * 100) if fv_flops > 0 else 0

                matches.append({
                    'our_name': our_name,
                    'fv_name': our_name_normalized,
                    'our_macs': our_macs,
                    'fv_macs': fv_flops,
                    'diff': diff,
                    'diff_pct': diff_pct,
                    'type': our_data['type']
                })
                matched = True

            if not matched and our_macs > 0:
                unmatched_ours.append((our_name, our_data))

        # Show matches
        if detailed or not show_discrepancies:
            print(f"\n{'Layer':<35} {'Our MACs':<15} {'FVCore MACs':<15} {'Diff %':<10} {'Status':<10}")
            print("-" * 95)

        match_count = 0
        mismatch_count = 0

        for match in sorted(matches, key=lambda x: x['fv_macs'], reverse=True):
            is_close = match['diff_pct'] < 1.0  # 1% tolerance
            status = "✓ MATCH" if is_close else "✗ DIFF"

            if is_close:
                match_count += 1
            else:
                mismatch_count += 1

            # Skip if only showing discrepancies and this matches
            if show_discrepancies and is_close:
                continue

            if detailed or not show_discrepancies:
                our_str = f"{match['our_macs'] / 1e6:.2f}M"
                fv_str = f"{match['fv_macs'] / 1e6:.2f}M"
                name = match['our_name'][:33]

                print(f"{name:<35} {our_str:<15} {fv_str:<15} {match['diff_pct']:<10.2f} {status:<10}")

        print(f"\nMatched layers: {match_count} close (< 1%), {mismatch_count} different (> 1%)")

        if unmatched_ours:
            print(f"\nLayers in our tool but not matched to fvcore: {len(unmatched_ours)}")
            if detailed:
                for name, data in unmatched_ours[:10]:
                    print(f"  {name}: {data['flops'] / 1e6:.1f}M FLOPs ({data['type']})")

    def show_fvcore_table(self):
        """Show fvcore's detailed table"""
        print("\n" + "=" * 80)
        print("FVCORE DETAILED TABLE")
        print("=" * 80)
        print()

        table = flop_count_table(
            FlopCountAnalysis(self.model, self.input_tensor),
            max_depth=3
        )
        print(table)

    def analyze_arithmetic_intensity(self, our_ops: Dict):
        """Show arithmetic intensity distribution"""
        print("\n" + "=" * 80)
        print("ARITHMETIC INTENSITY ANALYSIS")
        print("=" * 80)

        # Group by AI ranges
        ai_ranges = {
            'Compute-bound (AI > 50)': [],
            'Balanced (10 < AI <= 50)': [],
            'Memory-bound (1 < AI <= 10)': [],
            'Bandwidth-bound (AI <= 1)': []
        }

        for name, data in our_ops.items():
            ai = data['ai']
            if ai > 50:
                ai_ranges['Compute-bound (AI > 50)'].append((name, ai, data['flops']))
            elif ai > 10:
                ai_ranges['Balanced (10 < AI <= 50)'].append((name, ai, data['flops']))
            elif ai > 1:
                ai_ranges['Memory-bound (1 < AI <= 10)'].append((name, ai, data['flops']))
            else:
                ai_ranges['Bandwidth-bound (AI <= 1)'].append((name, ai, data['flops']))

        for category, ops in ai_ranges.items():
            if ops:
                total_flops = sum(flops for _, _, flops in ops)
                print(f"\n{category}: {len(ops)} operations, {total_flops / 1e9:.2f} GFLOPs")

                # Show top 3 by FLOPs
                for name, ai, flops in sorted(ops, key=lambda x: x[2], reverse=True)[:3]:
                    print(f"  {name[:40]:<40} AI: {ai:6.1f}, FLOPs: {flops / 1e6:8.1f}M")

    def run(self, args):
        """Main execution"""
        if not FVCORE_AVAILABLE:
            return 1

        # Load model
        if not self.load_model(args.model, args.input_shape):
            return 1

        # Count FLOPs with both tools
        our_flops, our_macs, our_ops = self.count_flops_our_tool()
        fvcore_total, fvcore_modules = self.count_flops_fvcore()

        # Compare
        self.compare_totals(our_flops, our_macs, fvcore_total)
        self.compare_per_layer(our_ops, fvcore_modules,
                              detailed=args.detailed,
                              show_discrepancies=args.show_discrepancies)

        # Additional analysis
        if args.show_ai:
            self.analyze_arithmetic_intensity(our_ops)

        if args.show_fvcore_table:
            self.show_fvcore_table()

        return 0


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description='Cross-validate FLOP counting against fvcore',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic comparison
  python cli/fvcore_compare.py --model resnet18

  # Detailed per-layer analysis
  python cli/fvcore_compare.py --model mobilenet_v2 --detailed

  # Show only mismatches
  python cli/fvcore_compare.py --model resnet50 --show-discrepancies

  # Full analysis with AI breakdown
  python cli/fvcore_compare.py --model resnet18 --detailed --show-ai --show-fvcore-table
        """
    )

    parser.add_argument('--model', type=str, default='resnet18',
                       choices=list(FVCoreComparator.SUPPORTED_MODELS.keys()),
                       help='Model to analyze')

    parser.add_argument('--detailed', action='store_true',
                       help='Show detailed per-layer comparison')

    parser.add_argument('--show-discrepancies', action='store_true',
                       help='Show only layers with FLOP count differences')

    parser.add_argument('--show-ai', action='store_true',
                       help='Show arithmetic intensity analysis')

    parser.add_argument('--show-fvcore-table', action='store_true',
                       help='Show fvcore detailed FLOP table')

    parser.add_argument('--input-shape', type=int, nargs=4, default=[1, 3, 224, 224],
                       metavar=('B', 'C', 'H', 'W'),
                       help='Input tensor shape')

    return parser.parse_args()


def main():
    args = parse_args()
    comparator = FVCoreComparator()
    return comparator.run(args)


if __name__ == "__main__":
    sys.exit(main())
