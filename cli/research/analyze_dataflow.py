#!/usr/bin/env python
"""
Dataflow Analysis Tool

Analyze tiling schedules and dataflow patterns for systolic arrays.

Usage:
    python cli/research/analyze_dataflow.py --model resnet18 --array-size 128
    python cli/research/analyze_dataflow.py --input shapes.parquet --dataflow weight_stationary
    python cli/research/analyze_dataflow.py --M 1024 --K 512 --N 1024 --compare-dataflows
"""

import argparse
import sys
import json
from pathlib import Path

# Add repo root to path
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / 'src'))

import torch
import torchvision.models as models
import warnings

from graphs.research.shape_collection import (
    ShapeExtractor,
    DNNClassifier,
    ShapeDatabase,
)
from graphs.research.dataflow import (
    TileSchedule,
    TileScheduler,
    DataflowType,
    LoopNest,
    DataMovementBreakdown,
    DataMovementAnalyzer,
    generate_weight_stationary_loop_nest,
    generate_output_stationary_loop_nest,
    generate_row_stationary_loop_nest,
)
from graphs.research.dataflow.loop_nests import create_loop_nest_from_schedule


def analyze_single_operation(
    M: int,
    K: int,
    N: int,
    array_size: int,
    compare_dataflows: bool = True,
    batch_size: int = 1,
) -> dict:
    """
    Analyze a single matrix operation.

    Args:
        M, K, N: Matrix dimensions
        array_size: Systolic array size
        compare_dataflows: Compare all three dataflow strategies
        batch_size: Batch size for reuse calculation

    Returns:
        Dictionary with analysis results
    """
    scheduler = TileScheduler(
        array_rows=array_size,
        array_cols=array_size,
    )
    analyzer = DataMovementAnalyzer()

    results = {}

    if compare_dataflows:
        dataflows = [
            DataflowType.WEIGHT_STATIONARY,
            DataflowType.OUTPUT_STATIONARY,
            DataflowType.ROW_STATIONARY,
        ]
    else:
        dataflows = [DataflowType.WEIGHT_STATIONARY]

    for dataflow in dataflows:
        schedule = scheduler.schedule(M, K, N, dataflow)
        loop_nest = create_loop_nest_from_schedule(schedule)
        movement = analyzer.analyze(schedule, batch_size)

        results[dataflow.value] = {
            'schedule': schedule.to_dict(),
            'loop_nest': loop_nest.to_dict(),
            'data_movement': movement.to_dict(),
        }

    return results


def analyze_model(
    model_name: str,
    array_size: int,
    dataflow: str = 'weight_stationary',
    batch_size: int = 1,
    verbose: bool = False,
) -> dict:
    """
    Analyze dataflow for all layers in a model.

    Args:
        model_name: TorchVision model name
        array_size: Systolic array size
        dataflow: Dataflow strategy
        batch_size: Batch size
        verbose: Print progress

    Returns:
        Dictionary with per-layer analysis
    """
    # Load model
    model_fn = getattr(models, model_name, None)
    if model_fn is None:
        raise ValueError(f"Model {model_name} not found in torchvision")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = model_fn(weights=None)
        model.eval()

    # Extract shapes
    extractor = ShapeExtractor()
    classifier = DNNClassifier()
    model_class = classifier.classify(model_name)

    input_tensor = torch.randn(batch_size, 3, 224, 224)
    records = extractor.extract_from_model(model, input_tensor, model_name, model_class)

    # Analyze each matmul-like layer
    scheduler = TileScheduler(array_rows=array_size, array_cols=array_size)
    analyzer = DataMovementAnalyzer()

    dataflow_type = DataflowType(dataflow)

    results = {
        'model_name': model_name,
        'model_class': model_class,
        'array_size': array_size,
        'dataflow': dataflow,
        'batch_size': batch_size,
        'layers': [],
    }

    total_flops = 0
    total_energy_pj = 0
    total_dram_bytes = 0

    for record in records:
        if record.M > 0 and record.K > 0 and record.N > 0:
            schedule = scheduler.schedule(record.M, record.K, record.N, dataflow_type)
            movement = analyzer.analyze(schedule, batch_size)

            layer_result = {
                'layer_name': record.layer_name,
                'op_type': record.op_type,
                'M': record.M,
                'K': record.K,
                'N': record.N,
                'flops': record.flops,
                'schedule': {
                    'Tm': schedule.Tm,
                    'Tk': schedule.Tk,
                    'Tn': schedule.Tn,
                    'total_tiles': schedule.total_tiles,
                    'arithmetic_intensity': schedule.arithmetic_intensity,
                },
                'data_movement': {
                    'dram_bytes': movement.dram_bytes,
                    'total_energy_pj': movement.total_energy_pj,
                    'input_reuse': movement.input_reuse_factor,
                    'weight_reuse': movement.weight_reuse_factor,
                    'output_reuse': movement.output_reuse_factor,
                },
            }

            results['layers'].append(layer_result)
            total_flops += record.flops
            total_energy_pj += movement.total_energy_pj
            total_dram_bytes += movement.dram_bytes

            if verbose:
                print(f"  {record.layer_name}: M={record.M}, K={record.K}, N={record.N}")
                print(f"    Tiles: {schedule.total_tiles}, Energy: {movement.total_energy_pj/1e6:.2f} uJ")

    results['summary'] = {
        'total_flops': total_flops,
        'total_energy_pj': total_energy_pj,
        'total_energy_mj': total_energy_pj / 1e9,
        'total_dram_bytes': total_dram_bytes,
        'num_layers': len(results['layers']),
        'energy_per_flop_pj': total_energy_pj / total_flops if total_flops > 0 else 0,
    }

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Analyze tiling schedules and dataflow patterns',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Analyze a specific operation
    python cli/research/analyze_dataflow.py --M 1024 --K 512 --N 1024 --array-size 128

    # Compare all dataflows for an operation
    python cli/research/analyze_dataflow.py --M 1024 --K 512 --N 1024 --compare-dataflows

    # Analyze all layers in a model
    python cli/research/analyze_dataflow.py --model resnet18 --array-size 128

    # Generate loop nest pseudocode
    python cli/research/analyze_dataflow.py --M 256 --K 256 --N 256 --show-loop-nest
        """,
    )

    # Operation dimensions
    parser.add_argument('--M', type=int, help='Output rows')
    parser.add_argument('--K', type=int, help='Reduction dimension')
    parser.add_argument('--N', type=int, help='Output columns')

    # Model analysis
    parser.add_argument('--model', type=str, help='TorchVision model name')
    parser.add_argument('--input', type=str, help='Input shape database')

    # Array configuration
    parser.add_argument('--array-size', '-a', type=int, default=128,
                       help='Systolic array size (default: 128)')

    # Dataflow options
    parser.add_argument('--dataflow', '-d', type=str, default='weight_stationary',
                       choices=['weight_stationary', 'output_stationary', 'row_stationary'],
                       help='Dataflow strategy (default: weight_stationary)')
    parser.add_argument('--compare-dataflows', action='store_true',
                       help='Compare all three dataflow strategies')

    # Output options
    parser.add_argument('--output', '-o', type=str, help='Output JSON file')
    parser.add_argument('--show-loop-nest', action='store_true',
                       help='Show loop nest pseudocode')
    parser.add_argument('--batch-size', '-b', type=int, default=1,
                       help='Batch size (default: 1)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Print detailed output')

    args = parser.parse_args()

    # Analyze single operation
    if args.M and args.K and args.N:
        print(f"Analyzing operation: M={args.M}, K={args.K}, N={args.N}")
        print(f"Array size: {args.array_size}x{args.array_size}")
        print()

        results = analyze_single_operation(
            args.M, args.K, args.N,
            args.array_size,
            compare_dataflows=args.compare_dataflows,
            batch_size=args.batch_size,
        )

        # Print results
        for dataflow_name, result in results.items():
            print(f"{'=' * 60}")
            print(f"DATAFLOW: {dataflow_name.upper()}")
            print(f"{'=' * 60}")
            print()

            schedule = result['schedule']
            print("Tiling Schedule:")
            print(f"  Tile sizes: Tm={schedule['Tm']}, Tk={schedule['Tk']}, Tn={schedule['Tn']}")
            print(f"  Tile counts: {schedule['num_m_tiles']} x {schedule['num_k_tiles']} x {schedule['num_n_tiles']}")
            print(f"  Total tiles: {schedule['total_tiles']}")
            print(f"  Arithmetic intensity: {schedule['arithmetic_intensity']:.2f} ops/byte")
            print()

            movement = result['data_movement']
            print("Data Movement:")
            print(f"  DRAM traffic: {movement['dram_bytes'] / 1e6:.2f} MB")
            print(f"  Total energy: {movement['total_energy_pj'] / 1e6:.2f} uJ")
            print(f"  Energy breakdown:")
            print(f"    RF:   {movement['rf_energy_pj'] / movement['total_energy_pj'] * 100:.1f}%")
            print(f"    L1:   {movement['l1_energy_pj'] / movement['total_energy_pj'] * 100:.1f}%")
            print(f"    DRAM: {movement['dram_energy_pj'] / movement['total_energy_pj'] * 100:.1f}%")
            print()

            print("Reuse Factors:")
            print(f"  Input:  {movement['input_reuse_factor']:.1f}x")
            print(f"  Weight: {movement['weight_reuse_factor']:.1f}x")
            print(f"  Output: {movement['output_reuse_factor']:.1f}x")
            print()

            if args.show_loop_nest:
                loop_nest = result['loop_nest']
                print("Loop Nest:")
                for i, loop in enumerate(loop_nest['loops']):
                    indent = "  " * i
                    print(f"{indent}for {loop['variable']} in range({loop['bound']}):")
                print()

        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to: {args.output}")

    # Analyze model
    elif args.model:
        print(f"Analyzing model: {args.model}")
        print(f"Array size: {args.array_size}x{args.array_size}")
        print(f"Dataflow: {args.dataflow}")
        print()

        results = analyze_model(
            args.model,
            args.array_size,
            args.dataflow,
            args.batch_size,
            verbose=args.verbose,
        )

        print("=" * 60)
        print("MODEL SUMMARY")
        print("=" * 60)
        print()

        summary = results['summary']
        print(f"Model: {results['model_name']} ({results['model_class']})")
        print(f"Matmul layers: {summary['num_layers']}")
        print(f"Total FLOPs: {summary['total_flops'] / 1e9:.2f} GFLOPs")
        print(f"Total DRAM traffic: {summary['total_dram_bytes'] / 1e6:.2f} MB")
        print(f"Total energy: {summary['total_energy_mj']:.4f} mJ")
        print(f"Energy/FLOP: {summary['energy_per_flop_pj']:.2f} pJ")

        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to: {args.output}")

    else:
        parser.print_help()
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
