#!/usr/bin/env python
"""
TPU Energy Breakdown Analysis Tool

Analyzes energy consumption of DNN models mapped to specific TPU architectures.
Shows detailed breakdown of TPU-specific energy events (systolic array control, etc.).

Usage:
    # Basic usage
    ./cli/analyze_tpu_energy.py --tpu tpu_v4 --model resnet18

    # With custom batch size and precision
    ./cli/analyze_tpu_energy.py --tpu coral_edge --model mobilenet_v2 --batch-size 8 --precision int8

    # JSON output
    ./cli/analyze_tpu_energy.py --tpu tpu_v5p --model resnet50 --output tpu_energy.json

    # List available TPUs
    ./cli/analyze_tpu_energy.py --list-tpus

    # List available models
    ./cli/analyze_tpu_energy.py --list-models
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
from graphs.hardware.mappers.accelerators.tpu import (
    create_tpu_v1_mapper,
    create_tpu_v3_mapper,
    create_tpu_v4_mapper,
    create_tpu_v5p_mapper,
    create_coral_edge_tpu_mapper,
    create_tpu_edge_pro_mapper,
)

# Import model factory
from model_factory import load_and_prepare_model, list_available_models

# Import export utilities
from energy_breakdown_utils import export_energy_results


# TPU configurations: name -> (factory_function, description, default_thermal_profile)
TPU_CONFIGS = {
    'tpu_v1': (
        create_tpu_v1_mapper,
        'Google TPU v1 (256×256 systolic array, 28nm, 40W)',
        '40W'
    ),
    'tpu_v3': (
        create_tpu_v3_mapper,
        'Google TPU v3 (128×128 systolic array, 16nm, 450W)',
        '450W'
    ),
    'tpu_v4': (
        create_tpu_v4_mapper,
        'Google TPU v4 (128×128 systolic array, 7nm, 300W)',
        '300W'
    ),
    'tpu_v5p': (
        create_tpu_v5p_mapper,
        'Google TPU v5p (improved systolic array, 4nm, 350W)',
        '350W'
    ),
    'coral_edge': (
        create_coral_edge_tpu_mapper,
        'Google Coral Edge TPU (8×8 systolic array, 16nm, 2W)',
        '2W'
    ),
    'tpu_edge_pro': (
        create_tpu_edge_pro_mapper,
        'Google TPU Edge Pro (16×16 systolic array, 7nm, 15W)',
        '15W'
    ),
}


def list_available_tpus():
    """Print all available TPU configurations."""
    print("\nAvailable TPU Configurations:")
    print("=" * 80)
    for tpu_name, (_, description, _) in TPU_CONFIGS.items():
        print(f"  {tpu_name:<20} {description}")
    print()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='TPU Energy Breakdown Analysis for DNN Models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # TPU selection
    parser.add_argument(
        '--tpu',
        choices=list(TPU_CONFIGS.keys()),
        help='TPU architecture to analyze'
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
        help='Numerical precision (default: int8 for TPU)'
    )

    parser.add_argument(
        '--thermal-profile',
        help='Thermal/power profile (e.g., "2W", "300W"). Uses hardware default if not specified.'
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
        '--list-tpus',
        action='store_true',
        help='List available TPU configurations'
    )

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
    if args.list_tpus:
        list_available_tpus()
        return

    if args.list_models:
        list_available_models()
        return

    # Validate required arguments
    if not args.tpu:
        print("ERROR: --tpu is required (or use --list-tpus)")
        sys.exit(1)

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
    print("TPU ENERGY BREAKDOWN ANALYSIS")
    print("=" * 80)
    print(f"TPU: {TPU_CONFIGS[args.tpu][1]}")
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

    # Step 2: Create TPU mapper
    print(f"\n[2/4] Creating TPU mapper for {args.tpu}...")
    try:
        factory_fn, description, default_thermal = TPU_CONFIGS[args.tpu]
        # TODO: Support thermal profiles once TPU mappers have them properly configured
        # For now, use default from factory
        tpu_mapper = factory_fn()

        print(f"✓ TPU mapper created: {tpu_mapper.resource_model.name}")
        print(f"  Systolic Array: {tpu_mapper.resource_model.compute_units} units")
        print(f"  Peak Performance (INT8): {tpu_mapper.resource_model.precision_profiles[Precision.INT8].peak_ops_per_sec / 1e12:.2f} TOPS")
    except Exception as e:
        print(f"✗ Failed to create TPU mapper: {e}")
        sys.exit(1)

    # Step 3: Map model to TPU
    print(f"\n[3/4] Mapping model to TPU...")
    try:
        # Create sequential execution stages (one stage per subgraph)
        execution_stages = [[i] for i in range(len(partition_report.fused_subgraphs))]

        # Map to TPU hardware
        mapping_result = tpu_mapper.map_graph(
            partition_report,
            execution_stages,
            batch_size=args.batch_size,
            precision=precision
        )
        print(f"✓ Model mapped")
        print(f"  Systolic array utilization: {mapping_result.average_utilization * 100:.1f}%")
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
    if tpu_mapper.resource_model.architecture_energy_model:
        # Aggregate ops and bytes across all subgraphs
        total_ops = sum(alloc.compute_time * tpu_mapper.resource_model.get_peak_ops(precision)
                       for alloc in mapping_result.subgraph_allocations)
        total_bytes = sum(sg.total_input_bytes + sg.total_output_bytes + sg.total_weight_bytes
                         for sg in partition_report.fused_subgraphs)

        # Get baseline energies from mapping result
        total_compute_energy = sum(a.compute_energy for a in mapping_result.subgraph_allocations)
        total_memory_energy = sum(a.memory_energy for a in mapping_result.subgraph_allocations)

        # Compute architectural energy breakdown
        execution_context = {
            'batch_size': args.batch_size,
            'array_dimension': 128,  # Will be overridden by model-specific value
        }

        try:
            arch_breakdown = tpu_mapper.resource_model.architecture_energy_model.compute_architectural_energy(
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
        from energy_breakdown_utils import print_tpu_hierarchical_breakdown

        total_compute_energy = sum(a.compute_energy for a in mapping_result.subgraph_allocations)
        total_memory_energy = sum(a.memory_energy for a in mapping_result.subgraph_allocations)

        print_tpu_hierarchical_breakdown(
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

        if not tpu_mapper.resource_model.architecture_energy_model:
            print(f"\nNote: Detailed architectural breakdown not available for {args.tpu}")
            print(f"      (no SystolicArrayEnergyModel configured)")

    # Export to JSON/CSV if requested
    if args.output:
        # Calculate energies
        total_compute_energy = sum(a.compute_energy for a in mapping_result.subgraph_allocations)
        total_memory_energy = sum(a.memory_energy for a in mapping_result.subgraph_allocations)
        static_energy = mapping_result.total_energy - total_compute_energy - total_memory_energy

        export_energy_results(
            output_path=args.output,
            architecture='TPU',
            hardware_name=tpu_mapper.resource_model.name,
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
