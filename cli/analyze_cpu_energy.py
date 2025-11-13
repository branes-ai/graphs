#!/usr/bin/env python3
"""
CPU Energy Breakdown Analysis Tool

Analyzes energy consumption of DNN models mapped to specific CPU architectures.
Shows detailed breakdown of CPU-specific energy events.

Usage:
    # Basic usage
    ./cli/analyze_cpu_energy.py --cpu xeon_emerald_rapids --model resnet18

    # With custom batch size and precision
    ./cli/analyze_cpu_energy.py --cpu jetson_orin_agx_cpu --model mobilenet_v2 --batch-size 8 --precision fp32

    # JSON output
    ./cli/analyze_cpu_energy.py --cpu epyc_genoa --model resnet50 --output cpu_energy.json

    # List available CPUs
    ./cli/analyze_cpu_energy.py --list-cpus

    # List available models
    ./cli/analyze_cpu_energy.py --list-models
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
from graphs.hardware.mappers.cpu import (
    create_jetson_orin_agx_cpu_mapper,
    create_intel_xeon_platinum_8490h_mapper,
    create_intel_xeon_platinum_8592plus_mapper,
    create_intel_granite_rapids_mapper,
    create_amd_epyc_9654_mapper,
    create_amd_epyc_9754_mapper,
    create_amd_epyc_turin_mapper,
    create_ampere_ampereone_192_mapper,
    create_ampere_ampereone_128_mapper,
)

# Import model factory
from model_factory import load_and_prepare_model, list_available_models


# CPU configurations: name -> (factory_function, description, default_thermal_profile)
CPU_CONFIGS = {
    'jetson_orin_agx_cpu': (
        lambda: create_jetson_orin_agx_cpu_mapper('30W'),
        'ARM Cortex-A78AE (12 cores, 8nm, 30W)',
        '30W'
    ),
    'xeon_emerald_rapids': (
        create_intel_xeon_platinum_8490h_mapper,
        'Intel Xeon Platinum 8490H (60 cores, 7nm Intel 4)',
        None  # Uses default from mapper
    ),
    'xeon_sapphire_rapids': (
        create_intel_xeon_platinum_8592plus_mapper,
        'Intel Xeon Platinum 8592+ (64 cores, 7nm Intel 4)',
        None
    ),
    'xeon_granite_rapids': (
        create_intel_granite_rapids_mapper,
        'Intel Granite Rapids (128 cores, 4nm Intel 3)',
        None
    ),
    'epyc_genoa': (
        create_amd_epyc_9654_mapper,
        'AMD EPYC 9654 (96 cores, 5nm)',
        None
    ),
    'epyc_bergamo': (
        create_amd_epyc_9754_mapper,
        'AMD EPYC 9754 (128 cores, 5nm)',
        None
    ),
    'epyc_turin': (
        create_amd_epyc_turin_mapper,
        'AMD EPYC Turin (192 cores, 4nm)',
        None
    ),
    'ampere_one_m192': (
        create_ampere_ampereone_192_mapper,
        'Ampere One M192 (192 cores, 5nm)',
        None
    ),
    'ampere_altra_max_m128': (
        create_ampere_ampereone_128_mapper,
        'Ampere Altra Max M128 (128 cores, 7nm)',
        None
    ),
}


def list_available_cpus():
    """Print all available CPU configurations."""
    print("\nAvailable CPU Configurations:")
    print("=" * 80)
    for cpu_name, (_, description, _) in CPU_CONFIGS.items():
        print(f"  {cpu_name:<30} {description}")
    print()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='CPU Energy Breakdown Analysis for DNN Models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # CPU selection
    parser.add_argument(
        '--cpu',
        choices=list(CPU_CONFIGS.keys()),
        help='CPU architecture to analyze'
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
        default='fp32',
        help='Numerical precision (default: fp32)'
    )

    parser.add_argument(
        '--thermal-profile',
        help='Thermal/power profile (e.g., "30W", "350W"). Uses hardware default if not specified.'
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
        '--list-cpus',
        action='store_true',
        help='List available CPU configurations'
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
    if args.list_cpus:
        list_available_cpus()
        return

    if args.list_models:
        list_available_models()
        return

    # Validate required arguments
    if not args.cpu:
        print("ERROR: --cpu is required (or use --list-cpus)")
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
    print("CPU ENERGY BREAKDOWN ANALYSIS")
    print("=" * 80)
    print(f"CPU: {CPU_CONFIGS[args.cpu][1]}")
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

    # Step 2: Create CPU mapper
    print(f"\n[2/4] Creating CPU mapper for {args.cpu}...")
    try:
        factory_fn, description, default_thermal = CPU_CONFIGS[args.cpu]
        thermal_profile = args.thermal_profile or default_thermal

        if thermal_profile:
            # TODO: Pass thermal profile to factory (some mappers accept it, some don't)
            # For now, use default from factory
            cpu_mapper = factory_fn()
        else:
            cpu_mapper = factory_fn()

        print(f"✓ CPU mapper created: {cpu_mapper.resource_model.name}")
        print(f"  Cores: {cpu_mapper.resource_model.compute_units}")
        print(f"  Peak Performance (FP32): {cpu_mapper.resource_model.precision_profiles[Precision.FP32].peak_ops_per_sec / 1e12:.2f} TFLOPS")
    except Exception as e:
        print(f"✗ Failed to create CPU mapper: {e}")
        sys.exit(1)

    # Step 3: Map model to CPU
    print(f"\n[3/4] Mapping model to CPU...")
    try:
        # Create sequential execution stages (one stage per subgraph)
        execution_stages = [[i] for i in range(len(partition_report.fused_subgraphs))]

        # Map to CPU hardware
        mapping_result = cpu_mapper.map_graph(
            partition_report,  # PartitionReport has .fused_subgraphs alias
            execution_stages,
            batch_size=args.batch_size,
            precision=precision
        )
        print(f"✓ Model mapped")
        print(f"  Peak cores used: {mapping_result.peak_compute_units_used} / {cpu_mapper.resource_model.compute_units}")
        print(f"  Avg utilization: {mapping_result.average_utilization * 100:.1f}%")
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
    if cpu_mapper.resource_model.architecture_energy_model:
        # Aggregate ops and bytes across all subgraphs
        total_ops = sum(alloc.compute_time * cpu_mapper.resource_model.get_peak_ops(precision)
                       for alloc in mapping_result.subgraph_allocations)
        total_bytes = sum(sg.total_input_bytes + sg.total_output_bytes + sg.total_weight_bytes
                         for sg in partition_report.fused_subgraphs)

        # Get baseline energies from mapping result
        total_compute_energy = sum(a.compute_energy for a in mapping_result.subgraph_allocations)
        total_memory_energy = sum(a.memory_energy for a in mapping_result.subgraph_allocations)

        # Compute architectural energy breakdown
        execution_context = {
            'batch_size': args.batch_size,
            'num_threads': cpu_mapper.resource_model.compute_units,
        }

        try:
            arch_breakdown = cpu_mapper.resource_model.architecture_energy_model.compute_architectural_energy(
                ops=int(total_ops),
                bytes_transferred=int(total_bytes),
                compute_energy_baseline=total_compute_energy,
                memory_energy_baseline=total_memory_energy,
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
        from energy_breakdown_utils import print_cpu_hierarchical_breakdown

        total_compute_energy = sum(a.compute_energy for a in mapping_result.subgraph_allocations)
        total_memory_energy = sum(a.memory_energy for a in mapping_result.subgraph_allocations)

        print_cpu_hierarchical_breakdown(
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

        if not cpu_mapper.resource_model.architecture_energy_model:
            print(f"\nNote: Detailed architectural breakdown not available for {args.cpu}")
            print(f"      (no StoredProgramEnergyModel configured)")

    # TODO: Export to JSON/CSV if requested

    print("\n" + "=" * 80)
    print("Analysis complete!")


if __name__ == '__main__':
    main()
