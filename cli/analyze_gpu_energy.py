#!/usr/bin/env python
"""
GPU Energy Breakdown Analysis Tool

Analyzes energy consumption of DNN models mapped to specific GPU architectures.
Shows detailed breakdown of GPU-specific energy events (SIMT control, coherence, etc.).

Usage:
    # Basic usage
    ./cli/analyze_gpu_energy.py --gpu h100_sxm5 --model resnet18

    # With custom batch size and precision
    ./cli/analyze_gpu_energy.py --gpu jetson_orin_agx --model mobilenet_v2 --batch-size 8 --precision fp16

    # JSON output
    ./cli/analyze_gpu_energy.py --gpu a100_sxm4 --model resnet50 --output gpu_energy.json

    # List available GPUs
    ./cli/analyze_gpu_energy.py --list-gpus

    # List available models
    ./cli/analyze_gpu_energy.py --list-models
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
from graphs.hardware.mappers.gpu import (
    create_h100_sxm5_80gb_mapper,
    create_h100_pcie_80gb_mapper,
    create_b100_sxm6_192gb_mapper,
    create_a100_sxm4_80gb_mapper,
    create_v100_sxm3_32gb_mapper,
    create_t4_pcie_16gb_mapper,
    create_jetson_orin_agx_64gb_mapper,
    create_jetson_orin_nano_8gb_mapper,
    create_jetson_thor_128gb_mapper,
    create_arm_mali_g78_mp20_mapper,
)

# Import model factory
from model_factory import load_and_prepare_model, list_available_models

# Import export utilities
from energy_breakdown_utils import export_energy_results


# GPU configurations: name -> (factory_function, description, default_thermal_profile)
GPU_CONFIGS = {
    'h100_sxm5': (
        create_h100_sxm5_80gb_mapper,
        'NVIDIA H100 SXM5 (132 SMs, 80GB HBM3, 700W)',
        '700W'
    ),
    'h100_pcie': (
        create_h100_pcie_80gb_mapper,
        'NVIDIA H100 PCIe (114 SMs, 80GB HBM2e, 350W)',
        '350W'
    ),
    'b100_sxm6': (
        create_b100_sxm6_192gb_mapper,
        'NVIDIA B100 SXM6 (156 SMs, 192GB HBM3e, 1000W)',
        '1000W'
    ),
    'a100_sxm4': (
        create_a100_sxm4_80gb_mapper,
        'NVIDIA A100 SXM4 (108 SMs, 80GB HBM2e, 400W)',
        '400W'
    ),
    'v100_sxm3': (
        create_v100_sxm3_32gb_mapper,
        'NVIDIA V100 SXM3 (80 SMs, 32GB HBM2, 300W)',
        '300W'
    ),
    't4_pcie': (
        create_t4_pcie_16gb_mapper,
        'NVIDIA T4 PCIe (40 SMs, 16GB GDDR6, 70W)',
        '70W'
    ),
    'jetson_orin_agx': (
        create_jetson_orin_agx_64gb_mapper,
        'NVIDIA Jetson Orin AGX (2048 CUDA cores, 64GB LPDDR5, 50W)',
        '50W'
    ),
    'jetson_orin_nano': (
        create_jetson_orin_nano_8gb_mapper,
        'NVIDIA Jetson Orin Nano (1024 CUDA cores, 8GB LPDDR5, 15W)',
        '15W'
    ),
    'jetson_thor': (
        create_jetson_thor_128gb_mapper,
        'NVIDIA Jetson Thor (4096 CUDA cores, 128GB LPDDR5x, 75W)',
        '75W'
    ),
    'arm_mali_g78': (
        create_arm_mali_g78_mp20_mapper,
        'ARM Mali-G78 MP20 (20 cores, 8GB, 15W)',
        '15W'
    ),
}


def list_available_gpus():
    """Print all available GPU configurations."""
    print("\nAvailable GPU Configurations:")
    print("=" * 80)
    for gpu_name, (_, description, _) in GPU_CONFIGS.items():
        print(f"  {gpu_name:<25} {description}")
    print()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='GPU Energy Breakdown Analysis for DNN Models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # GPU selection
    parser.add_argument(
        '--gpu',
        choices=list(GPU_CONFIGS.keys()),
        help='GPU architecture to analyze'
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
        help='Thermal/power profile (e.g., "50W", "700W"). Uses hardware default if not specified.'
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
        '--list-gpus',
        action='store_true',
        help='List available GPU configurations'
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
    if args.list_gpus:
        list_available_gpus()
        return

    if args.list_models:
        list_available_models()
        return

    # Validate required arguments
    if not args.gpu:
        print("ERROR: --gpu is required (or use --list-gpus)")
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
    print("GPU ENERGY BREAKDOWN ANALYSIS")
    print("=" * 80)
    print(f"GPU: {GPU_CONFIGS[args.gpu][1]}")
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

    # Step 2: Create GPU mapper
    print(f"\n[2/4] Creating GPU mapper for {args.gpu}...")
    try:
        factory_fn, description, default_thermal = GPU_CONFIGS[args.gpu]
        # TODO: Support thermal profiles once GPU mappers have them properly configured
        # For now, use default from factory
        gpu_mapper = factory_fn()

        print(f"✓ GPU mapper created: {gpu_mapper.resource_model.name}")
        print(f"  SMs/Cores: {gpu_mapper.resource_model.compute_units}")
        print(f"  Peak Performance (FP32): {gpu_mapper.resource_model.precision_profiles[Precision.FP32].peak_ops_per_sec / 1e12:.2f} TFLOPS")
    except Exception as e:
        print(f"✗ Failed to create GPU mapper: {e}")
        sys.exit(1)

    # Step 3: Map model to GPU
    print(f"\n[3/4] Mapping model to GPU...")
    try:
        # Create sequential execution stages (one stage per subgraph)
        execution_stages = [[i] for i in range(len(partition_report.fused_subgraphs))]

        # Map to GPU hardware
        mapping_result = gpu_mapper.map_graph(
            partition_report,
            execution_stages,
            batch_size=args.batch_size,
            precision=precision
        )
        print(f"✓ Model mapped")
        print(f"  Peak SMs used: {mapping_result.peak_compute_units_used} / {gpu_mapper.resource_model.compute_units}")
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
    if gpu_mapper.resource_model.architecture_energy_model:
        # Aggregate ops and bytes across all subgraphs
        total_ops = sum(alloc.compute_time * gpu_mapper.resource_model.get_peak_ops(precision)
                       for alloc in mapping_result.subgraph_allocations)
        total_bytes = sum(sg.total_input_bytes + sg.total_output_bytes + sg.total_weight_bytes
                         for sg in partition_report.fused_subgraphs)

        # Get baseline energies from mapping result
        total_compute_energy = sum(a.compute_energy for a in mapping_result.subgraph_allocations)
        total_memory_energy = sum(a.memory_energy for a in mapping_result.subgraph_allocations)

        # Compute architectural energy breakdown
        execution_context = {
            'batch_size': args.batch_size,
            'num_sms': gpu_mapper.resource_model.compute_units,
        }

        try:
            arch_breakdown = gpu_mapper.resource_model.architecture_energy_model.compute_architectural_energy(
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
        from energy_breakdown_utils import print_gpu_hierarchical_breakdown

        total_compute_energy = sum(a.compute_energy for a in mapping_result.subgraph_allocations)
        total_memory_energy = sum(a.memory_energy for a in mapping_result.subgraph_allocations)

        print_gpu_hierarchical_breakdown(
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

        if not gpu_mapper.resource_model.architecture_energy_model:
            print(f"\nNote: Detailed architectural breakdown not available for {args.gpu}")
            print(f"      (no SIMTEnergyModel configured)")

    # Export to JSON/CSV if requested
    if args.output:
        # Calculate energies
        total_compute_energy = sum(a.compute_energy for a in mapping_result.subgraph_allocations)
        total_memory_energy = sum(a.memory_energy for a in mapping_result.subgraph_allocations)
        static_energy = mapping_result.total_energy - total_compute_energy - total_memory_energy

        export_energy_results(
            output_path=args.output,
            architecture='GPU',
            hardware_name=gpu_mapper.resource_model.name,
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
