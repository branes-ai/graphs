#!/usr/bin/env python3
"""
Architecture Comparison Tool - Compare DNN models across hardware architectures

This tool analyzes a DNN model and compares energy, latency, and memory usage
across multiple hardware architectures, providing hierarchical drill-down
for discovery workflows.

Usage Examples:
    # Quick summary comparison
    ./cli/compare_architectures.py --model resnet18

    # Compare specific architectures
    ./cli/compare_architectures.py --model resnet18 \\
        --architectures CPU GPU TPU KPU

    # Detailed breakdown for one architecture
    ./cli/compare_architectures.py --model resnet18 \\
        --level detailed --architecture GPU

    # Subgraph-level comparison
    ./cli/compare_architectures.py --model resnet18 \\
        --level subgraph

    # Explain energy difference
    ./cli/compare_architectures.py --model resnet18 \\
        --explain-difference GPU TPU --metric energy

    # Different batch size and precision
    ./cli/compare_architectures.py --model resnet18 \\
        --batch-size 32 --precision FP16
"""

import argparse
import sys
from typing import Dict, List

# Add parent directory to path for imports
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from graphs.hardware.resource_model import Precision, HardwareMapper
from graphs.analysis.architecture_comparator import ArchitectureComparator, ComparisonLevel

# Import available mappers
from graphs.hardware.mappers.cpu import create_intel_xeon_platinum_8490h_mapper
from graphs.hardware.mappers.gpu import create_h100_mapper
from graphs.hardware.mappers.accelerators.tpu import create_tpu_v4_mapper
from graphs.hardware.mappers.accelerators.kpu import create_kpu_t256_mapper
from graphs.hardware.mappers.research.dfm import create_dfm_128_mapper


# Available architectures and their factory functions
AVAILABLE_ARCHITECTURES = {
    'CPU': ('Intel Xeon Platinum 8490H', create_intel_xeon_platinum_8490h_mapper),
    'GPU': ('NVIDIA H100', create_h100_mapper),
    'TPU': ('Google TPU v4', create_tpu_v4_mapper),
    'KPU': ('Stillwater KPU-T256', create_kpu_t256_mapper),
    'DFM': ('Data Flow Machine DFM-128', create_dfm_128_mapper),
}


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description='Compare DNN model performance across hardware architectures',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Model specification
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Model name (e.g., resnet18, mobilenet_v2)'
    )

    # Architecture selection
    parser.add_argument(
        '--architectures',
        type=str,
        nargs='+',
        choices=list(AVAILABLE_ARCHITECTURES.keys()),
        default=None,
        help='Architectures to compare (default: all available)'
    )

    # Comparison level
    parser.add_argument(
        '--level',
        type=str,
        choices=['summary', 'detailed', 'subgraph'],
        default='summary',
        help='Comparison detail level (default: summary)'
    )

    # For detailed level: specify which architecture
    parser.add_argument(
        '--architecture',
        type=str,
        choices=list(AVAILABLE_ARCHITECTURES.keys()),
        help='Specific architecture for detailed view'
    )

    # Explain difference mode
    parser.add_argument(
        '--explain-difference',
        type=str,
        nargs=2,
        metavar=('ARCH1', 'ARCH2'),
        help='Explain why ARCH1 differs from ARCH2'
    )

    parser.add_argument(
        '--metric',
        type=str,
        choices=['energy', 'latency', 'memory'],
        default='energy',
        help='Metric to explain (default: energy)'
    )

    # Batch size and precision
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help='Batch size for analysis (default: 1)'
    )

    parser.add_argument(
        '--precision',
        type=str,
        choices=[p.value for p in Precision],
        default='FP32',
        help='Numerical precision (default: FP32)'
    )

    # Output options
    parser.add_argument(
        '--output',
        type=str,
        help='Output file (extension determines format: .json, .csv, .html)'
    )

    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress progress output'
    )

    return parser.parse_args()


def create_mappers(arch_names: List[str]) -> Dict[str, HardwareMapper]:
    """Create hardware mappers for specified architectures"""
    mappers = {}

    for name in arch_names:
        if name not in AVAILABLE_ARCHITECTURES:
            print(f"Warning: Unknown architecture '{name}', skipping")
            continue

        desc, factory = AVAILABLE_ARCHITECTURES[name]
        try:
            mappers[name] = factory()
        except Exception as e:
            print(f"Warning: Failed to create mapper for {name}: {e}")

    return mappers


def main():
    """Main entry point"""
    args = parse_args()

    # Determine which architectures to use
    if args.architectures:
        arch_names = args.architectures
    else:
        # Default: use all available
        arch_names = list(AVAILABLE_ARCHITECTURES.keys())

    if not args.quiet:
        print(f"Comparing {args.model} across architectures: {', '.join(arch_names)}")
        print()

    # Create hardware mappers
    mappers = create_mappers(arch_names)

    if not mappers:
        print("Error: No valid architectures specified")
        sys.exit(1)

    # Parse precision
    precision = Precision[args.precision]

    # Create comparator
    comparator = ArchitectureComparator(
        model_name=args.model,
        architectures=mappers,
        batch_size=args.batch_size,
        precision=precision
    )

    # Run analysis
    try:
        comparator.analyze_all()
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Generate output based on mode
    if args.explain_difference:
        # Explain difference mode
        arch1, arch2 = args.explain_difference

        if arch1 not in mappers or arch2 not in mappers:
            print(f"Error: Architecture not found: {arch1} or {arch2}")
            sys.exit(1)

        output = comparator.explain_difference(arch1, arch2, args.metric)
        print(output)

    elif args.level == 'summary':
        # Summary level
        output = comparator.generate_summary()
        print(output)

    elif args.level == 'detailed':
        # Detailed level
        if not args.architecture:
            print("Error: --architecture required for detailed level")
            print("Available architectures:", ', '.join(mappers.keys()))
            sys.exit(1)

        if args.architecture not in mappers:
            print(f"Error: Architecture '{args.architecture}' not analyzed")
            sys.exit(1)

        output = comparator.generate_detailed(args.architecture)
        print(output)

    elif args.level == 'subgraph':
        # Subgraph level
        output = comparator.generate_subgraph_comparison()
        print(output)

    # Save to file if requested
    if args.output:
        ext = os.path.splitext(args.output)[1].lower()

        if ext == '.json':
            # TODO: Implement JSON export
            print(f"JSON export not yet implemented")
        elif ext == '.csv':
            # TODO: Implement CSV export
            print(f"CSV export not yet implemented")
        elif ext == '.html':
            # TODO: Implement HTML export
            print(f"HTML export not yet implemented")
        else:
            # Plain text
            with open(args.output, 'w') as f:
                f.write(output)
            print(f"Output saved to {args.output}")


if __name__ == '__main__':
    main()
