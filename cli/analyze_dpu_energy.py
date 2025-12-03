#!/usr/bin/env python
"""
DPU Energy Breakdown Analysis Tool

Analyzes energy consumption of DNN models mapped to Xilinx Vitis AI DPU (FPGA-based).
Shows detailed breakdown of DPU-specific energy events (AIE tiles, reconfiguration, etc.).

Usage:
    # Basic usage
    ./cli/analyze_dpu_energy.py --model resnet18

    # With custom batch size and precision
    ./cli/analyze_dpu_energy.py --model mobilenet_v2 --batch-size 8 --precision int8

    # JSON output
    ./cli/analyze_dpu_energy.py --model resnet50 --output dpu_energy.json

    # List available models
    ./cli/analyze_dpu_energy.py --list-models
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, Optional
import json

# Add repo root to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from graphs.hardware.resource_model import Precision
from graphs.hardware.mappers.accelerators.dpu import create_dpu_vitis_ai_mapper

# Import model factory
from model_factory import load_and_prepare_model, list_available_models

# Import export utilities
from energy_breakdown_utils import export_energy_results


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='DPU Energy Breakdown Analysis for DNN Models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Model selection
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument(
        '--model',
        help='Built-in model name (e.g., resnet18, mobilenet_v2)'
    )
    model_group.add_argument(
        '--model-path',
        help='Path to custom PyTorch model file'
    )

    parser.add_argument(
        '--model-class',
        help='Class name for custom model (required with --model-path)'
    )

    # Analysis options
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help='Batch size for inference (default: 1)'
    )

    parser.add_argument(
        '--precision',
        choices=['fp32', 'fp16', 'bf16', 'int8'],
        default='int8',
        help='Numerical precision (default: int8 for DPU)'
    )

    # Output options
    parser.add_argument(
        '--output',
        help='Output file path (.json, .csv, or .txt)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show detailed tracing and partitioning logs'
    )

    # List options
    parser.add_argument(
        '--list-models',
        action='store_true',
        help='List available built-in models'
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Handle list commands
    if args.list_models:
        list_available_models()
        return

    # Validate required arguments
    if not args.model and not args.model_path:
        print("ERROR: Either --model or --model-path is required (or use --list-models)")
        sys.exit(1)

    if args.model_path and not args.model_class:
        print("ERROR: --model-class is required when using --model-path")
        sys.exit(1)

    # Parse precision
    precision_map = {
        'fp32': Precision.FP32,
        'fp16': Precision.FP16,
        'bf16': Precision.BF16,
        'int8': Precision.INT8,
    }
    precision = precision_map[args.precision]

    print("=" * 80)
    print("DPU ENERGY BREAKDOWN ANALYSIS")
    print("=" * 80)
    print(f"DPU: Xilinx Vitis AI (B4096 on Versal VE2302)")
    print(f"Model: {args.model or args.model_path}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Precision: {args.precision.upper()}")
    print("=" * 80)

    # Step 1: Load and prepare model
    print("\n[1/4] Loading and tracing model...")
    try:
        model, traced, partition_report, input_shape = load_and_prepare_model(
            model_name=args.model,
            model_path=args.model_path,
            model_class=args.model_class,
            batch_size=args.batch_size,
            verbose=args.verbose
        )
        print(f"✓ Model loaded: {len(partition_report.subgraphs)} subgraphs")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        sys.exit(1)

    # Step 2: Create DPU mapper
    print(f"\n[2/4] Creating DPU mapper...")
    try:
        dpu_mapper = create_dpu_vitis_ai_mapper()

        print(f"✓ DPU mapper created: {dpu_mapper.resource_model.name}")
        print(f"  Tiles: {dpu_mapper.resource_model.compute_units}")
        print(f"  Peak Performance (INT8): {dpu_mapper.resource_model.precision_profiles[Precision.INT8].peak_ops_per_sec / 1e12:.2f} TOPS")
    except Exception as e:
        print(f"✗ Failed to create DPU mapper: {e}")
        sys.exit(1)

    # Step 3: Map model to DPU
    print(f"\n[3/4] Mapping model to DPU...")
    try:
        # Create sequential execution stages (one stage per subgraph)
        execution_stages = [[i] for i in range(len(partition_report.fused_subgraphs))]

        # Map to DPU hardware
        mapping_result = dpu_mapper.map_graph(
            partition_report,
            execution_stages,
            batch_size=args.batch_size,
            precision=precision
        )
        print(f"✓ Model mapped")
        print(f"  Tile utilization: {mapping_result.average_utilization * 100:.1f}%")
        print(f"  Estimated latency: {mapping_result.total_latency * 1000:.2f} ms")
        print(f"  Total energy: {mapping_result.total_energy * 1000:.2f} mJ")
    except Exception as e:
        print(f"✗ Failed to map model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Step 4: Extract and display energy breakdown
    print(f"\n[4/4] Energy Breakdown")
    print("=" * 80)

    # Extract architectural energy breakdown
    arch_events = None
    if dpu_mapper.resource_model.architecture_energy_model:
        # Aggregate ops and bytes across all subgraphs
        total_ops = sum(alloc.compute_time * dpu_mapper.resource_model.get_peak_ops(precision)
                       for alloc in mapping_result.subgraph_allocations)
        total_bytes = sum(sg.total_input_bytes + sg.total_output_bytes + sg.total_weight_bytes
                         for sg in partition_report.fused_subgraphs)

        # Get baseline energies from mapping result
        total_compute_energy = sum(a.compute_energy for a in mapping_result.subgraph_allocations)
        total_memory_energy = sum(a.memory_energy for a in mapping_result.subgraph_allocations)

        # Compute architectural energy breakdown
        execution_context = {
            'batch_size': args.batch_size,
            'num_tiles': dpu_mapper.resource_model.compute_units,
        }

        try:
            arch_breakdown = dpu_mapper.resource_model.architecture_energy_model.compute_architectural_energy(
                ops=int(total_ops),
                bytes_transferred=int(total_bytes),
                compute_energy_baseline=total_compute_energy,
                data_movement_energy_baseline=total_memory_energy,
                execution_context=execution_context
            )
            # Extract events from extra_details
            arch_events = arch_breakdown.extra_details
        except Exception as e:
            if args.verbose:
                print(f"Warning: Failed to compute architectural breakdown: {e}")
            arch_events = None

    # Print hierarchical breakdown if available
    if arch_events:
        from energy_breakdown_utils import print_dpu_hierarchical_breakdown

        total_compute_energy = sum(a.compute_energy for a in mapping_result.subgraph_allocations)
        total_memory_energy = sum(a.memory_energy for a in mapping_result.subgraph_allocations)

        print_dpu_hierarchical_breakdown(
            arch_specific_events=arch_events,
            total_energy_j=mapping_result.total_energy,
            compute_energy_j=total_compute_energy,
            memory_energy_j=total_memory_energy,
            latency_s=mapping_result.total_latency
        )
    else:
        # Fallback: basic breakdown
        print(f"\nTotal Energy: {mapping_result.total_energy * 1000:.3f} mJ")
        print(f"Energy per Inference: {mapping_result.total_energy / args.batch_size * 1000:.3f} mJ")
        print(f"Latency: {mapping_result.total_latency * 1000:.2f} ms")
        print(f"Throughput: {args.batch_size / mapping_result.total_latency:.2f} inferences/sec")

        if not dpu_mapper.resource_model.architecture_energy_model:
            print(f"\nNote: Detailed architectural breakdown not available")
            print(f"      (no DPUEnergyModel configured)")

    # Export to JSON/CSV if requested
    if args.output:
        # Calculate energies
        total_compute_energy = sum(a.compute_energy for a in mapping_result.subgraph_allocations)
        total_memory_energy = sum(a.memory_energy for a in mapping_result.subgraph_allocations)
        static_energy = mapping_result.total_energy - total_compute_energy - total_memory_energy

        export_energy_results(
            output_path=args.output,
            architecture='DPU',
            hardware_name=dpu_mapper.resource_model.name,
            model_name=args.model or args.model_path,
            batch_size=args.batch_size,
            precision=args.precision.upper(),
            total_energy_j=mapping_result.total_energy,
            latency_s=mapping_result.total_latency,
            throughput_inf_per_s=args.batch_size / mapping_result.total_latency,
            compute_energy_j=total_compute_energy,
            memory_energy_j=total_memory_energy,
            static_energy_j=static_energy,
            utilization=mapping_result.average_utilization,
            num_subgraphs=len(partition_report.subgraphs),
            arch_specific_events=arch_events,
            subgraph_allocations=mapping_result.subgraph_allocations
        )

    print("\n" + "=" * 80)
    print("Analysis complete!")


if __name__ == '__main__':
    main()
