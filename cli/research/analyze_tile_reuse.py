#!/usr/bin/env python3
"""
Tile Reuse Analysis CLI

Analyze tile reuse for tiled matrix operations with explicit 2D tile tracking.

Key insight: Tiles are 2D submatrices, not 3D objects.
- A_tile: shape (Tm, Tk) - input activation
- B_tile: shape (Tk, Tn) - weight
- C_tile: shape (Tm, Tn) - output/accumulator

Usage:
    # Basic analysis with 2D tile shapes
    python cli/research/analyze_tile_reuse.py --M 1024 --K 512 --N 1024 \
        --Tm 128 --Tk 64 --Tn 128

    # Compare loop orderings
    python cli/research/analyze_tile_reuse.py --M 1024 --K 512 --N 1024 \
        --Tm 128 --Tk 64 --Tn 128 --compare-loop-orders

    # Specify loop order
    python cli/research/analyze_tile_reuse.py --M 1024 --K 512 --N 1024 \
        --Tm 128 --Tk 64 --Tn 128 --loop-order NKM

    # JSON output
    python cli/research/analyze_tile_reuse.py --M 1024 --K 512 --N 1024 \
        --Tm 128 --Tk 64 --Tn 128 --output results.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from graphs.research.tiling import (
    MatmulTiling,
    TileSchedule,
    LoopOrder,
    TileReuseAnalyzer,
    TileReuseAnalysis,
    analyze_tile_reuse,
    print_tile_analysis,
    analyze_memory_traffic,
    DistributedMatmulAnalysis,
    create_mesh_l3,
    compare_algorithms,
)


def analyze_single(args) -> TileReuseAnalysis:
    """Analyze single configuration."""
    analyzer = TileReuseAnalyzer()
    return analyzer.analyze(
        M=args.M, K=args.K, N=args.N,
        Tm=args.Tm, Tk=args.Tk, Tn=args.Tn,
        loop_order=LoopOrder(args.loop_order),
    )


def compare_loop_orders(args) -> Dict[str, Dict]:
    """Compare all loop orderings."""
    analyzer = TileReuseAnalyzer()
    comparisons = analyzer.compare_loop_orders(
        M=args.M, K=args.K, N=args.N,
        Tm=args.Tm, Tk=args.Tk, Tn=args.Tn,
    )

    results = {
        'problem': {'M': args.M, 'K': args.K, 'N': args.N},
        'tiles': {
            'A': (args.Tm, args.Tk),
            'B': (args.Tk, args.Tn),
            'C': (args.Tm, args.Tn),
        },
        'loop_orders': {},
    }

    for name, analysis in comparisons.items():
        results['loop_orders'][name] = {
            'A_reuse': analysis.a_metrics.reuse_factor,
            'B_reuse': analysis.b_metrics.reuse_factor,
            'C_accumulations': analysis.c_metrics.reuse_factor,
            'arithmetic_intensity': analysis.arithmetic_intensity,
            'peak_working_set_bytes': analysis.working_set.peak_total_bytes,
            'A_reuse_efficiency': analysis.a_metrics.reuse_efficiency,
            'B_reuse_efficiency': analysis.b_metrics.reuse_efficiency,
        }

    # Find best
    best_name = max(comparisons, key=lambda x: comparisons[x].arithmetic_intensity)
    results['best_loop_order'] = best_name
    results['best_arithmetic_intensity'] = comparisons[best_name].arithmetic_intensity

    return results


def distributed_analysis(args) -> Dict:
    """Analyze tile distribution on distributed L3."""
    analysis = DistributedMatmulAnalysis(
        M=args.M, K=args.K, N=args.N,
        Tm=args.Tm, Tk=args.Tk, Tn=args.Tn,
    )

    return {
        'problem': {'M': args.M, 'K': args.K, 'N': args.N},
        'tiles': {
            'A': (args.Tm, args.Tk),
            'B': (args.Tk, args.Tn),
            'C': (args.Tm, args.Tn),
        },
        'communication': analysis.analyze_communication(),
        'topology_comparison': analysis.compare_topologies(),
    }


def rotation_analysis(args) -> Dict:
    """Compare tile rotation algorithms."""
    return {
        'problem': {'M': args.M, 'K': args.K, 'N': args.N},
        'tiles': {
            'A': (args.Tm, args.Tk),
            'B': (args.Tk, args.Tn),
            'C': (args.Tm, args.Tn),
        },
        'num_procs': args.num_procs,
        'algorithms': compare_algorithms(
            args.M, args.K, args.N,
            args.Tm, args.Tk, args.Tn,
            num_procs=args.num_procs,
        ),
    }


def print_comparison_results(results: Dict):
    """Print loop order comparison in human-readable format."""
    print("\n" + "=" * 80)
    print("LOOP ORDER COMPARISON")
    print("=" * 80)

    print(f"\nProblem: C({results['problem']['M']}, {results['problem']['N']}) = "
          f"A({results['problem']['M']}, {results['problem']['K']}) @ "
          f"B({results['problem']['K']}, {results['problem']['N']})")

    t = results['tiles']
    print(f"\nTile shapes (2D):")
    print(f"  A_tile: {t['A']}")
    print(f"  B_tile: {t['B']}")
    print(f"  C_tile: {t['C']}")

    print("\n" + "-" * 80)
    print(f"{'Loop Order':<12} {'A Reuse':<10} {'B Reuse':<10} {'C Accum':<10} "
          f"{'AI':<10} {'Peak WS':<15}")
    print("-" * 80)

    for name, data in results['loop_orders'].items():
        ws_mb = data['peak_working_set_bytes'] / 1024 / 1024
        print(f"{name:<12} {data['A_reuse']:<10.1f} {data['B_reuse']:<10.1f} "
              f"{data['C_accumulations']:<10.1f} {data['arithmetic_intensity']:<10.2f} "
              f"{ws_mb:<15.2f} MB")

    print("-" * 80)
    print(f"\nBest loop order: {results['best_loop_order']} "
          f"(AI = {results['best_arithmetic_intensity']:.2f})")

    # Explain what each loop order optimizes
    print("\nLoop order meanings:")
    print("  MNK: Output-stationary (C stays, good for large K)")
    print("  NKM: Weight-stationary (B stays, good for large M)")
    print("  MKN: Input-stationary (A stays, good for large N)")


def print_distributed_results(results: Dict):
    """Print distributed L3 analysis."""
    print("\n" + "=" * 70)
    print("DISTRIBUTED L3 ANALYSIS")
    print("=" * 70)

    t = results['tiles']
    print(f"\nTile shapes (2D):")
    print(f"  A_tile: {t['A']}")
    print(f"  B_tile: {t['B']}")
    print(f"  C_tile: {t['C']}")

    print("\nTopology comparison:")
    print("-" * 60)
    print(f"{'Topology':<15} {'Transfers':<12} {'Bytes':<15} {'Avg Hops':<10}")
    print("-" * 60)

    for name, data in results['topology_comparison'].items():
        print(f"{name:<15} {data['num_transfers']:<12} {data['total_bytes']:<15,} "
              f"{data['avg_hops_per_transfer']:<10.2f}")


def print_rotation_results(results: Dict):
    """Print rotation algorithm comparison."""
    print("\n" + "=" * 70)
    print("TILE ROTATION ALGORITHM COMPARISON")
    print("=" * 70)

    t = results['tiles']
    print(f"\nTile shapes (2D):")
    print(f"  A_tile: {t['A']}")
    print(f"  B_tile: {t['B']}")
    print(f"  C_tile: {t['C']}")
    print(f"  Processors: {results['num_procs']}")

    print("\n" + "-" * 70)
    print(f"{'Algorithm':<12} {'Steps':<10} {'Transfer Bytes':<18} {'Comm/Compute':<12}")
    print("-" * 70)

    for name, data in results['algorithms'].items():
        print(f"{name:<12} {data['num_steps']:<10} {data['total_transfer_bytes']:<18,} "
              f"{data['communication_to_compute_ratio']:<12.4f}")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze tile reuse with explicit 2D tile shapes',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Problem dimensions
    parser.add_argument('--M', type=int, default=1024,
                        help='M dimension (A rows, C rows)')
    parser.add_argument('--K', type=int, default=512,
                        help='K dimension (A cols, B rows - reduction)')
    parser.add_argument('--N', type=int, default=1024,
                        help='N dimension (B cols, C cols)')

    # Tile dimensions (explicit 2D)
    parser.add_argument('--Tm', type=int, default=128,
                        help='A tile rows, C tile rows')
    parser.add_argument('--Tk', type=int, default=64,
                        help='A tile cols, B tile rows')
    parser.add_argument('--Tn', type=int, default=128,
                        help='B tile cols, C tile cols')

    # Loop order
    parser.add_argument('--loop-order', type=str, default='MNK',
                        choices=['MNK', 'NKM', 'MKN', 'KMN', 'KNM', 'NMK'],
                        help='Loop order (default: MNK = output-stationary)')

    # Analysis modes
    parser.add_argument('--compare-loop-orders', action='store_true',
                        help='Compare all loop orderings')
    parser.add_argument('--distributed-l3', action='store_true',
                        help='Analyze distributed L3 topology')
    parser.add_argument('--rotation-algorithms', action='store_true',
                        help='Compare tile rotation algorithms')

    # Hardware config
    parser.add_argument('--num-procs', type=int, default=16,
                        help='Number of processors for distributed analysis')

    # Output
    parser.add_argument('--output', '-o', type=str,
                        help='Output file (JSON if .json extension)')
    parser.add_argument('--format', type=str, default='text',
                        choices=['text', 'json'],
                        help='Output format')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Suppress verbose output')

    args = parser.parse_args()

    # Determine output format from file extension
    if args.output and args.output.endswith('.json'):
        args.format = 'json'

    # Run appropriate analysis
    if args.compare_loop_orders:
        results = compare_loop_orders(args)
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"Results written to {args.output}")
        elif args.format == 'json':
            print(json.dumps(results, indent=2, default=str))
        else:
            print_comparison_results(results)

    elif args.distributed_l3:
        results = distributed_analysis(args)
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"Results written to {args.output}")
        elif args.format == 'json':
            print(json.dumps(results, indent=2, default=str))
        else:
            print_distributed_results(results)

    elif args.rotation_algorithms:
        results = rotation_analysis(args)
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"Results written to {args.output}")
        elif args.format == 'json':
            print(json.dumps(results, indent=2, default=str))
        else:
            print_rotation_results(results)

    else:
        # Single analysis
        analysis = analyze_single(args)

        if args.output:
            with open(args.output, 'w') as f:
                json.dump(analysis.summary(), f, indent=2, default=str)
            print(f"Results written to {args.output}")
        elif args.format == 'json':
            print(json.dumps(analysis.summary(), indent=2, default=str))
        else:
            print_tile_analysis(analysis)


if __name__ == '__main__':
    main()
