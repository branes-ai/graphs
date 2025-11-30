#!/usr/bin/env python
"""
Architecture Energy Comparison: CPU vs GPU vs TPU vs KPU

Compares energy consumption and architectural overhead for batched MLP inference
across four fundamental architecture classes:
- Stored-Program (CPU): Sequential MIMD with instruction fetch
- Data-Parallel (GPU): SIMT with massive coherence machinery
- Systolic-Array (TPU): Systolic array with weight-stationary dataflow
- Domain-Flow (KPU): Programmable spatial dataflow with token routing

Focus: Understanding WHY energy differs through architectural energy events.

This tool performs apples-to-apples comparison at the same power budget (30W)
using clean MLP dimensions (256x256, 512x512, 1024x1024) that map efficiently
to hardware resources.

Usage:
    # Basic comparison (default: all 3 MLPs @ batch 1,8,16)
    ./cli/compare_architectures_energy.py

    # Custom MLP dimensions and batch sizes
    ./cli/compare_architectures_energy.py --mlp-dims 512 1024 --batch-sizes 1 4 16

    # Generate plots
    ./cli/compare_architectures_energy.py --plot --plot-dir ./energy_plots

    # JSON output
    ./cli/compare_architectures_energy.py --output comparison.json

    # CSV output
    ./cli/compare_architectures_energy.py --output comparison.csv
"""

import argparse
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional
import json

# Add repo root to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from graphs.hardware.resource_model import Precision
from graphs.hardware.mappers.cpu import create_jetson_orin_agx_cpu_mapper
from graphs.hardware.mappers.gpu import create_jetson_orin_agx_64gb_mapper
from graphs.hardware.mappers.accelerators.tpu import create_tpu_edge_pro_mapper
from graphs.hardware.mappers.accelerators.kpu import create_kpu_t256_mapper

from graphs.transform.partitioning import FusionBasedPartitioner
from graphs.hardware.architectural_energy import (
    ArchitecturalEnergyBreakdown as ArchEnergyBreakdown  # Aliased
)

# Try importing matplotlib for plotting (optional)
try:
    import matplotlib.pyplot as plt
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


@dataclass
class MLPConfig:
    """MLP linear operator configuration"""
    name: str
    dim: int  # Square matrix: dim × dim
    total_macs: int
    total_flops: int
    weight_params: int
    input_size: int  # Bytes for input tensor [batch, dim]
    weight_size: int  # Bytes for weight tensor [dim, dim]
    output_size: int  # Bytes for output tensor [batch, dim]

    @classmethod
    def from_dim(cls, dim: int, precision: Precision = Precision.FP32):
        """Create config from dimension"""
        macs = dim * dim  # Matrix multiplication: dim×dim
        flops = 2 * dim  # Bias addition (dim) + activation (dim)
        params = dim * dim + dim  # Weights + bias

        bytes_per_element = {
            Precision.FP32: 4,
            Precision.FP16: 2,
            Precision.INT8: 1,
        }.get(precision, 4)

        input_bytes = dim * bytes_per_element  # [batch, dim] per sample
        weight_bytes = dim * dim * bytes_per_element  # [dim, dim]
        output_bytes = dim * bytes_per_element  # [batch, dim] per sample

        return cls(
            name=f"{dim}x{dim}",
            dim=dim,
            total_macs=macs,
            total_flops=flops,
            weight_params=params,
            input_size=input_bytes,
            weight_size=weight_bytes,
            output_size=output_bytes,
        )

    def total_bytes_per_inference(self) -> int:
        """Total bytes transferred per inference (input + weight + output)"""
        return self.input_size + self.weight_size + self.output_size


@dataclass
class ArchitecturalEnergyBreakdown:
    """
    Detailed architectural energy event breakdown for a single architecture.
    """
    architecture: str  # "CPU", "GPU", "TPU", "KPU"
    architecture_class: str  # "Stored-Program", "Data-Parallel", "Systolic-Array", "Domain-Flow" 
    hardware_name: str  # "Jetson-Orin-AGX-64GB", "KPU-T256", etc.
    batch_size: int
    mlp_config: MLPConfig

    # Total energy and performance
    total_energy_j: float
    latency_s: float
    throughput_inferences_per_sec: float
    power_w: float

    # Three-component energy model
    compute_energy_j: float
    memory_energy_j: float
    static_energy_j: float

    # Architectural overhead breakdown (from architectural_energy.py)
    architectural_compute_overhead_j: float
    architectural_data_movement_overhead_j: float
    architectural_control_overhead_j: float

    # Architecture-specific energy events (dict for flexibility)
    arch_specific_events: Dict[str, float]  # Event name → energy (J)

    # Derived metrics
    energy_per_inference_j: float
    energy_per_mac_pj: float  # Picojoules per MAC

    # Hardware utilization
    compute_units_total: int
    compute_units_allocated: int
    peak_utilization: float

    def get_total_architectural_overhead(self) -> float:
        """Sum of all architectural overhead"""
        return (self.architectural_compute_overhead_j +
                self.architectural_data_movement_overhead_j +
                self.architectural_control_overhead_j)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        d = asdict(self)
        # Convert MLPConfig to dict
        d['mlp_config'] = asdict(self.mlp_config)
        return d


class ArchitectureEnergyComparator:
    """
    Compare energy across CPU/GPU/TPU/KPU for batched MLP workloads.

    This class orchestrates the complete comparison:
    1. Create simple MLP models with specified dimensions
    2. Trace and partition using FusionPartitioner
    3. Map to each architecture (GPU, KPU, CPU) at same power budget
    4. Extract architectural energy breakdowns
    5. Calculate derived metrics and comparisons
    """

    def __init__(
        self,
        mlp_configs: List[MLPConfig],
        batch_sizes: List[int],
        precision: Precision = Precision.FP32,
        thermal_profile: str = "30W"
    ):
        """
        Initialize comparator.

        Args:
            mlp_configs: List of MLP configurations to test
            batch_sizes: List of batch sizes to sweep
            precision: Numerical precision for operations
            thermal_profile: Power budget (e.g., "30W")
        """
        self.mlp_configs = mlp_configs
        self.batch_sizes = batch_sizes
        self.precision = precision
        self.thermal_profile = thermal_profile

        # Create hardware mappers
        print(f"Initializing hardware mappers @ {thermal_profile}...")
        self.cpu_mapper = create_jetson_orin_agx_cpu_mapper(thermal_profile)
        self.gpu_mapper = create_jetson_orin_agx_64gb_mapper(thermal_profile)
        self.tpu_mapper = create_tpu_edge_pro_mapper(thermal_profile)  # TPU Edge Pro @ 30W
        self.kpu_mapper = create_kpu_t256_mapper(thermal_profile)

        # For micro-benchmarks: disable kernel/program launch overhead
        # This gives a pure compute comparison without one-time setup costs
        self.gpu_mapper.disable_launch_overhead = True
        # TPU and KPU use execution_context, which we'll pass through _extract_architectural_energy


    def compare_all(self) -> Dict[str, List[ArchitecturalEnergyBreakdown]]:
        """
        Run comparison for all configs and batch sizes.

        Returns:
            {
                'cpu': [breakdown1, breakdown2, ...],
                'gpu': [breakdown1, breakdown2, ...],
                'tpu': [breakdown1, breakdown2, ...],
                'kpu': [breakdown1, breakdown2, ...],
            }
        """
        results = {'cpu': [], 'gpu': [], 'tpu': [], 'kpu': []}

        total_runs = len(self.mlp_configs) * len(self.batch_sizes) * 4  # 4 architectures
        current_run = 0

        for mlp_config in self.mlp_configs:
            for batch_size in self.batch_sizes:
                print(f"\n{'='*80}")
                print(f"Analyzing {mlp_config.name} MLP @ batch={batch_size}")
                print(f"{'='*80}")

                # CPU
                current_run += 1
                print(f"[{current_run}/{total_runs}] CPU (Stored-Program)...")
                cpu_breakdown = self._analyze_architecture(
                    'cpu', mlp_config, batch_size, self.cpu_mapper
                )
                results['cpu'].append(cpu_breakdown)

                # GPU
                current_run += 1
                print(f"[{current_run}/{total_runs}] GPU (Data-Parallel)...")
                gpu_breakdown = self._analyze_architecture(
                    'gpu', mlp_config, batch_size, self.gpu_mapper
                )
                results['gpu'].append(gpu_breakdown)

                # TPU
                current_run += 1
                print(f"[{current_run}/{total_runs}] TPU (Systolic-Array)...")
                tpu_breakdown = self._analyze_architecture(
                    'tpu', mlp_config, batch_size, self.tpu_mapper
                )
                results['tpu'].append(tpu_breakdown)

                # KPU
                current_run += 1
                print(f"[{current_run}/{total_runs}] KPU (Domain-Flow)...")
                kpu_breakdown = self._analyze_architecture(
                    'kpu', mlp_config, batch_size, self.kpu_mapper
                )
                results['kpu'].append(kpu_breakdown)

        return results

    def _analyze_architecture(
        self,
        arch_type: str,
        mlp_config: MLPConfig,
        batch_size: int,
        mapper
    ) -> ArchitecturalEnergyBreakdown:
        """
        Analyze single architecture for given MLP config and batch size.

        Args:
            arch_type: 'cpu', 'gpu', 'tpu', or 'kpu'
            mlp_config: MLP configuration
            batch_size: Batch size
            mapper: Hardware mapper instance

        Returns:
            ArchitecturalEnergyBreakdown with complete analysis
        """
        # Create synthetic PyTorch model
        model = self._create_mlp_model(mlp_config.dim)

        # Trace and partition
        from torch.fx import symbolic_trace
        from torch.fx.passes.shape_prop import ShapeProp
        import torch

        # Create example input
        example_input = torch.randn(batch_size, mlp_config.dim)

        # Trace model
        traced = symbolic_trace(model)

        # Run shape propagation
        ShapeProp(traced).propagate(example_input)

        # Partition with fusion
        partitioner = FusionBasedPartitioner()
        fusion_report = partitioner.partition(traced)

        # Create execution stages (sequential for single linear layer)
        execution_stages = [[i] for i in range(len(fusion_report.fused_subgraphs))]

        # Map to hardware
        hw_allocation = mapper.map_graph(
            fusion_report,
            execution_stages,
            batch_size=batch_size,
            precision=self.precision
        )

        # Extract architectural energy
        arch_breakdown = self._extract_architectural_energy(
            arch_type, mlp_config, batch_size, hw_allocation, mapper
        )

        return arch_breakdown

    def _create_mlp_model(self, dim: int):
        """Create simple PyTorch MLP model (single linear layer)"""
        import torch.nn as nn

        class SimpleMLP(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.linear = nn.Linear(dim, dim)

            def forward(self, x):
                return self.linear(x)

        return SimpleMLP(dim)

    def _extract_architectural_energy(
        self,
        arch_type: str,
        mlp_config: MLPConfig,
        batch_size: int,
        hw_allocation,
        mapper
    ) -> ArchitecturalEnergyBreakdown:
        """
        Extract architectural energy breakdown from hardware allocation.

        This is where we compute the architectural overhead using the
        architectural energy models from architectural_energy.py.
        """
        # Get architectural energy model from mapper
        arch_energy_model = mapper.resource_model.architecture_energy_model

        # Total ops and bytes
        total_ops = (mlp_config.total_macs + mlp_config.total_flops) * batch_size
        total_bytes = (mlp_config.input_size * batch_size +
                      mlp_config.weight_size +
                      mlp_config.output_size * batch_size)

        # Compute baseline energy (what pure ops and memory would cost)
        compute_energy_baseline = total_ops * mapper.resource_model.energy_per_flop_fp32
        data_movement_energy_baseline = total_bytes * mapper.resource_model.energy_per_byte

        # Build execution context for architectural model
        execution_context = {
            'batch_size': batch_size,
            'cache_line_size': 64,  # Typical cache line
            'disable_launch_overhead': True,  # For micro-benchmarks: pure compute comparison
        }

        if arch_type == 'gpu':
            # GPU-specific context
            execution_context.update({
                'concurrent_threads': hw_allocation.peak_compute_units_used * 2048,  # SMs × threads/SM
                'warp_size': 32,
            })
        elif arch_type == 'tpu':
            # TPU-specific context
            execution_context.update({
                'kernel_changes': 1,  # Single linear layer
                'mlp_input_dim': mlp_config.dim,
                'mlp_output_dim': mlp_config.dim,
            })
        elif arch_type == 'kpu':
            # KPU-specific context
            execution_context.update({
                'kernel_changes': 1,  # Single linear layer
                'mlp_input_dim': mlp_config.dim,  # For KPUTileEnergyAdapter
                'mlp_output_dim': mlp_config.dim,  # For KPUTileEnergyAdapter
            })

        # Compute architectural overhead
        if arch_energy_model:
            arch_breakdown_raw = arch_energy_model.compute_architectural_energy(
                ops=total_ops,
                bytes_transferred=total_bytes,
                compute_energy_baseline=compute_energy_baseline,
                data_movement_energy_baseline=data_movement_energy_baseline,
                execution_context=execution_context,
            )
        else:
            # Fallback if no architectural model
            arch_breakdown_raw = ArchEnergyBreakdown(
                compute_overhead=0.0,
                data_movement_overhead=0.0,
                control_overhead=0.0,
                extra_details={},
                explanation="No architectural energy model available"
            )

        # Extract architecture-specific events
        arch_specific = self._extract_arch_specific_events(
            arch_type, arch_breakdown_raw
        )

        # Calculate total energy
        total_energy = hw_allocation.total_energy

        # Calculate latency (sum of all stage latencies)
        total_latency = hw_allocation.total_latency

        # Derived metrics
        energy_per_inference = total_energy / batch_size
        energy_per_mac_pj = (total_energy * 1e12) / (mlp_config.total_macs * batch_size)
        throughput = batch_size / total_latency if total_latency > 0 else 0
        power = total_energy / total_latency if total_latency > 0 else 0

        # Get architecture class name
        arch_class_name = {
            'cpu': 'Stored-Program',
            'gpu': 'Data-Parallel',
            'tpu': 'Systolic-Array',
            'kpu': 'Domain-Flow',
        }[arch_type]

        return ArchitecturalEnergyBreakdown(
            architecture=arch_type.upper(),
            architecture_class=arch_class_name,
            hardware_name=mapper.resource_model.name,
            batch_size=batch_size,
            mlp_config=mlp_config,
            total_energy_j=total_energy,
            latency_s=total_latency,
            throughput_inferences_per_sec=throughput,
            power_w=power,
            compute_energy_j=sum(a.compute_energy for a in hw_allocation.subgraph_allocations),
            memory_energy_j=sum(a.memory_energy for a in hw_allocation.subgraph_allocations),
            static_energy_j=0.0,  # Will be calculated from idle power
            architectural_compute_overhead_j=arch_breakdown_raw.compute_overhead,
            architectural_data_movement_overhead_j=arch_breakdown_raw.data_movement_overhead,
            architectural_control_overhead_j=arch_breakdown_raw.control_overhead,
            arch_specific_events=arch_specific,
            energy_per_inference_j=energy_per_inference,
            energy_per_mac_pj=energy_per_mac_pj,
            compute_units_total=mapper.resource_model.compute_units,
            compute_units_allocated=hw_allocation.peak_compute_units_used,
            peak_utilization=hw_allocation.peak_utilization,
        )

    def _extract_arch_specific_events(
        self,
        arch_type: str,
        arch_breakdown: ArchEnergyBreakdown
    ) -> Dict[str, float]:
        """Extract architecture-specific energy events from breakdown"""
        if arch_type == 'gpu':
            return {
                # Compute units (Phase 3: separate MAC/FLOP energy)
                'Tensor Core Operations': arch_breakdown.extra_details.get('tensor_core_mac_energy', 0),
                'tensor_core_ops': arch_breakdown.extra_details.get('tensor_core_ops', 0),
                'tensor_core_macs': arch_breakdown.extra_details.get('tensor_core_macs', 0),
                'CUDA Core Operations': (
                    arch_breakdown.extra_details.get('cuda_core_mac_energy', 0) +
                    arch_breakdown.extra_details.get('cuda_core_flop_energy', 0)
                ),
                'cuda_core_ops': (
                    arch_breakdown.extra_details.get('cuda_core_macs', 0) +
                    arch_breakdown.extra_details.get('cuda_core_flops', 0)
                ),
                'cuda_core_macs': arch_breakdown.extra_details.get('cuda_core_macs', 0),
                'cuda_core_flops': arch_breakdown.extra_details.get('cuda_core_flops', 0),
                'Register File Access': arch_breakdown.extra_details.get('register_file_energy', 0),
                'num_register_accesses': arch_breakdown.extra_details.get('num_register_accesses', 0),

                # Energy model parameters (for hardware config display)
                'cuda_core_mac_energy': arch_breakdown.extra_details.get('cuda_core_mac_energy', 0),
                'tensor_core_mac_energy': arch_breakdown.extra_details.get('tensor_core_mac_energy', 0),
                'register_file_energy_per_access': arch_breakdown.extra_details.get('register_file_energy_per_access', 0),

                # Instruction pipeline
                'Instruction Fetch': arch_breakdown.extra_details.get('instruction_fetch_energy', 0),
                'Instruction Decode': arch_breakdown.extra_details.get('instruction_decode_energy', 0),
                'Instruction Execute': arch_breakdown.extra_details.get('instruction_execute_energy', 0),
                'num_instructions': arch_breakdown.extra_details.get('num_instructions', 0),

                # Memory hierarchy (NVIDIA Ampere nomenclature)
                'Shared Memory/L1 Unified': arch_breakdown.extra_details.get('shared_mem_l1_unified_energy', 0),
                'shared_mem_l1_accesses': arch_breakdown.extra_details.get('shared_mem_l1_accesses', 0),
                'shared_mem_bytes': arch_breakdown.extra_details.get('shared_mem_bytes', 0),
                'l1_bytes': arch_breakdown.extra_details.get('l1_bytes', 0),
                'L2 Cache': arch_breakdown.extra_details.get('l2_cache_energy', 0),
                'l2_accesses': arch_breakdown.extra_details.get('l2_accesses', 0),
                'l2_bytes': arch_breakdown.extra_details.get('l2_bytes', 0),
                'DRAM': arch_breakdown.extra_details.get('dram_energy', 0),
                'dram_accesses': arch_breakdown.extra_details.get('dram_accesses', 0),
                'dram_bytes': arch_breakdown.extra_details.get('dram_bytes', 0),

                # Workload data movement (for consistent AI)
                'bytes_transferred': arch_breakdown.extra_details.get('bytes_transferred', 0),

                # SIMT control overheads
                'Coherence Machinery': arch_breakdown.extra_details.get('coherence_energy', 0),
                'num_concurrent_warps': arch_breakdown.extra_details.get('num_concurrent_warps', 0),
                'num_memory_ops': arch_breakdown.extra_details.get('num_memory_ops', 0),
                'Thread Scheduling': arch_breakdown.extra_details.get('scheduling_energy', 0),
                'concurrent_threads': arch_breakdown.extra_details.get('concurrent_threads', 0),
                'Warp Divergence': arch_breakdown.extra_details.get('divergence_energy', 0),
                'num_divergent_ops': arch_breakdown.extra_details.get('num_divergent_ops', 0),
                'Memory Coalescing': arch_breakdown.extra_details.get('coalescing_energy', 0),
                'num_uncoalesced': arch_breakdown.extra_details.get('num_uncoalesced', 0),
                'Synchronization Barriers': arch_breakdown.extra_details.get('barrier_energy', 0),
                'num_barriers': arch_breakdown.extra_details.get('num_barriers', 0),
            }
        elif arch_type == 'kpu':
            # Check if we have detailed tile energy model (8 components)
            if 'dram_energy' in arch_breakdown.extra_details:
                return arch_breakdown.extra_details  # Return all components directly
            else:
                # Fallback: simple domain-flow model
                return {
                    'Domain Flow': arch_breakdown.extra_details.get('domain_tracking_energy', 0),
                    'Kernel Load': arch_breakdown.extra_details.get('kernel_load_energy', 0),
                    'Domain Injection': arch_breakdown.extra_details.get('injection_energy', 0),
                    'Domain Extraction': arch_breakdown.extra_details.get('extraction_energy', 0),
                }
        elif arch_type == 'tpu':
            # Check if we have detailed systolic array energy model (SystolicArrayEnergyModel)
            if 'instruction_decode' in arch_breakdown.extra_details:
                return arch_breakdown.extra_details  # Return all detailed components directly
            else:
                # Fallback: simple systolic array model
                return {
                    'Systolic Array Operations': arch_breakdown.extra_details.get('systolic_array_mac_energy', 0),
                    'systolic_array_ops': arch_breakdown.extra_details.get('systolic_array_ops', 0),
                    'On-Chip Buffer Access': arch_breakdown.extra_details.get('on_chip_buffer_energy', 0),
                    'on_chip_buffer_accesses': arch_breakdown.extra_details.get('on_chip_buffer_accesses', 0),
                    'Off-Chip DRAM Access': arch_breakdown.extra_details.get('dram_energy', 0),
                    'dram_accesses': arch_breakdown.extra_details.get('dram_accesses', 0),
                    'bytes_transferred': arch_breakdown.extra_details.get('bytes_transferred', 0),
                }
        else:  # cpu
            # Return all CPU components directly
            return arch_breakdown.extra_details


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Compare energy consumption across CPU/GPU/TPU/KPU architectures',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic comparison (default: 256x256, 512x512, 1024x1024 @ batch 1,8,16)
  %(prog)s

  # Custom MLP dimensions
  %(prog)s --mlp-dims 512 1024

  # Custom batch sizes
  %(prog)s --batch-sizes 1 4 16 64

  # Generate plots
  %(prog)s --plot --plot-dir ./energy_plots

  # JSON output
  %(prog)s --output comparison.json

  # CSV output
  %(prog)s --output comparison.csv

  # Detailed architecture energy breakdowns
  %(prog)s --print-arch cpu gpu tpu kpu
  %(prog)s --print-arch gpu              # Only GPU details

  # All options
  %(prog)s --mlp-dims 256 512 --batch-sizes 1 8 16 --plot --output results.json

Power Budget:
  All architectures compared at 30W TDP for fairness:
  - CPU: ARM Cortex-A78AE @ 30W (all-core sustained)
  - GPU: Jetson Orin AGX @ 30W (active cooling)
  - TPU: Hypothetical TPU Pro @ 30W (Scaled up from Edge TPU)
  - KPU: Stillwater KPU-T256 @ 30W (balanced mode)

        """
    )

    parser.add_argument(
        '--mlp-dims', nargs='+', type=int, default=[256, 512, 1024],
        help='MLP dimensions to test (default: 256 512 1024)'
    )
    parser.add_argument(
        '--batch-sizes', nargs='+', type=int, default=[1, 8, 16],
        help='Batch sizes to test (default: 1 8 16)'
    )
    parser.add_argument(
        '--precision', type=str, default='fp32', choices=['int4', 'int8', 'int16', 'bf16', 'fp4', 'fp8', 'fp16', 'fp32', 'fp64'],
        help='Numerical precision (default: fp32)'
    )
    parser.add_argument(
        '--output', type=str, help='Output file (JSON or CSV based on extension)'
    )
    parser.add_argument(
        '--plot', action='store_true', help='Generate bar charts'
    )
    parser.add_argument(
        '--plot-dir', type=str, default='./energy_plots',
        help='Directory for plots (default: ./energy_plots)'
    )
    parser.add_argument(
        '--print-arch', nargs='+', type=str, default=[],
        choices=['cpu', 'gpu', 'tpu', 'kpu'],
        help='Architectures to print detailed breakdowns for (default: none, summary only)'
    )
    parser.add_argument(
        '--thermal-profile', type=str, default='30W',
        help='Power budget for all architectures (default: 30W)'
    )

    args = parser.parse_args()

    # Map precision string to enum
    precision_map = {
        'int4': Precision.INT4,
        'int8': Precision.INT8,
        'int16': Precision.INT16,
        'bf16': Precision.BF16,
        'fp4': Precision.FP4,
        'fp8': Precision.FP8,
        'fp16': Precision.FP16,
        'fp32': Precision.FP32,
        'fp64': Precision.FP64,
    }
    precision = precision_map[args.precision.lower()]

    # Create MLP configs
    mlp_configs = [MLPConfig.from_dim(dim, precision) for dim in args.mlp_dims]

    print(f"\n{'='*80}")
    print(f"ARCHITECTURE ENERGY COMPARISON: CPU vs GPU vs TPU vs KPU")
    print(f"{'='*80}")
    print(f"Power Budget: {args.thermal_profile} (all architectures)")
    print(f"Precision: {args.precision.upper()}")
    print(f"MLP Dimensions: {args.mlp_dims}")
    print(f"Batch Sizes: {args.batch_sizes}")
    print(f"{'='*80}\n")

    # Run comparison
    comparator = ArchitectureEnergyComparator(
        mlp_configs, args.batch_sizes, precision, args.thermal_profile
    )
    results = comparator.compare_all()

    print(f"\n{'='*80}")
    print(f"COMPARISON COMPLETE")
    print(f"{'='*80}\n")

    # Generate report
    # TODO: Implement EnergyComparisonReporter
    # For now, just print summary
    _print_summary(results, mlp_configs, args.batch_sizes, comparator, args.print_arch)

    # Save output
    if args.output:
        _save_results(results, args.output)
        print(f"\nResults saved to: {args.output}")

    # Generate plots
    if args.plot:
        if not MATPLOTLIB_AVAILABLE:
            print("\nWARNING: matplotlib not available, skipping plots")
            print("Install with: pip install matplotlib")
        else:
            print(f"\nGenerating plots...")
            _generate_plots(results, mlp_configs, args.batch_sizes, args.plot_dir)
            print(f"Plots saved to: {args.plot_dir}/")


def _generate_plots(results, mlp_configs, batch_sizes, plot_dir):
    """Generate matplotlib bar charts for energy breakdown"""
    from pathlib import Path

    # Create output directory
    Path(plot_dir).mkdir(parents=True, exist_ok=True)

    # Generate plot for each configuration
    for mlp_config in mlp_configs:
        for batch_size in batch_sizes:
            # Find results for this config
            cpu_result = None
            gpu_result = None
            tpu_result = None
            kpu_result = None

            for r in results['cpu']:
                if r.mlp_config.name == mlp_config.name and r.batch_size == batch_size:
                    cpu_result = r
                    break

            for r in results['gpu']:
                if r.mlp_config.name == mlp_config.name and r.batch_size == batch_size:
                    gpu_result = r
                    break

            for r in results['tpu']:
                if r.mlp_config.name == mlp_config.name and r.batch_size == batch_size:
                    tpu_result = r
                    break

            for r in results['kpu']:
                if r.mlp_config.name == mlp_config.name and r.batch_size == batch_size:
                    kpu_result = r
                    break

            if not (gpu_result and kpu_result and cpu_result):
                continue

            # Generate stacked bar chart
            output_path = Path(plot_dir) / f'energy_comparison_{mlp_config.name}_batch{batch_size}.png'
            _plot_energy_breakdown(gpu_result, kpu_result, cpu_result, str(output_path))


def _plot_energy_breakdown(gpu_result, kpu_result, cpu_result, output_path):
    """
    Create stacked bar chart showing energy breakdown by architecture.

    Chart structure:
    - X-axis: CPU, GPU, TPU, KPU
    - Y-axis: Energy (μJ)
    - Stacked bars: Compute, Memory, Static, Architectural Overhead
    """
    fig, ax = plt.subplots(figsize=(14, 10))

    architectures = ['CPU\n(Stored-Program)', 'GPU\n(Data-Parallel)', 'TPU\n(Systolic-Array)', 'KPU\n(Domain-Flow)']
    data = [cpu_result, gpu_result, tpu_result, kpu_result]

    # Energy components (convert to μJ)
    compute_energy = [d.compute_energy_j * 1e6 for d in data]
    memory_energy = [d.memory_energy_j * 1e6 for d in data]
    static_energy = [d.static_energy_j * 1e6 for d in data]

    # Architectural overhead (sum of all arch-specific events)
    arch_overhead = [sum(d.arch_specific_events.values()) * 1e6 for d in data]

    # Create stacked bar chart
    x = np.arange(len(architectures))
    width = 0.6

    # Stack bars
    p1 = ax.bar(x, compute_energy, width, label='Compute Energy', color='#3498db')
    p2 = ax.bar(x, memory_energy, width, bottom=compute_energy,
                label='Memory Energy', color='#e74c3c')

    bottom2 = [compute_energy[i] + memory_energy[i] for i in range(3)]
    p3 = ax.bar(x, static_energy, width, bottom=bottom2,
                label='Static/Idle Energy', color='#95a5a6')

    bottom3 = [bottom2[i] + static_energy[i] for i in range(3)]
    p4 = ax.bar(x, arch_overhead, width, bottom=bottom3,
                label='Architectural Overhead', color='#f39c12', hatch='///')

    # Labels and formatting
    ax.set_ylabel('Energy per Inference (μJ)', fontsize=14, fontweight='bold')
    ax.set_title(
        f'Energy Breakdown: {gpu_result.mlp_config.name} MLP @ Batch={gpu_result.batch_size}\n' +
        f'All architectures @ 30W TDP',
        fontsize=16, fontweight='bold', pad=20
    )
    ax.set_xticks(x)
    ax.set_xticklabels(architectures, fontsize=12)
    ax.legend(loc='upper left', fontsize=11, framealpha=0.9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add total energy labels on top of bars
    for i, d in enumerate(data):
        total = d.total_energy_j * 1e6
        ax.text(i, total + total * 0.05,
                f'{total:.1f} μJ\n({d.energy_per_mac_pj:.1f} pJ/MAC)',
                ha='center', va='bottom', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))

    # Add efficiency comparison annotation
    min_energy = min(d.total_energy_j for d in data)
    winner_idx = [d.total_energy_j for d in data].index(min_energy)
    winner_name = ['CPU', 'GPU', 'TPU', 'KPU'][winner_idx]

    efficiency_text = f'Energy Efficiency (vs best):\n'
    for i, d in enumerate(data):
        ratio = d.total_energy_j / min_energy
        arch_name = ['CPU', 'GPU', 'TPU', 'KPU'][i]
        efficiency_text += f'{arch_name}: {ratio:.2f}×\n'

    ax.text(0.02, 0.98, efficiency_text.strip(),
            transform=ax.transAxes, fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7, pad=1))

    # Add throughput annotation
    throughput_text = f'Throughput:\n'
    for i, d in enumerate(data):
        arch_name = ['CPU', 'GPU', 'TPU', 'KPU'][i]
        throughput_text += f'{arch_name}: {d.throughput_inferences_per_sec:,.0f} infer/s\n'

    ax.text(0.98, 0.98, throughput_text.strip(),
            transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7, pad=1))

    # Add architectural overhead details as text annotation below chart
    overhead_details = _format_overhead_details(gpu_result, kpu_result, cpu_result)
    fig.text(0.5, 0.02, overhead_details, ha='center', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

    plt.tight_layout(rect=[0, 0.08, 1, 1])  # Leave space for bottom annotation

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def _format_overhead_details(cpu_result, gpu_result, tpu_result, kpu_result):
    """Format architectural overhead details for plot annotation"""
    details = "Architectural Overhead Breakdown:\n"

    # CPU events
    cpu_events = [f"{k}: {v*1e6:.2f} μJ" for k, v in cpu_result.arch_specific_events.items() if v > 0]
    if cpu_events:
        details += f"| CPU: {', '.join(cpu_events[:3])}"

    # GPU events
    gpu_events = [f"{k}: {v*1e6:.2f} μJ" for k, v in gpu_result.arch_specific_events.items() if v > 0]
    if gpu_events:
        details += f"GPU: {', '.join(gpu_events[:3])}  "  # Show first 3

    # TPU events
    tpu_events = [f"{k}: {v*1e6:.2f} μJ" for k, v in tpu_result.arch_specific_events.items() if v > 0]
    if tpu_events:
        details += f"| TPU: {', '.join(tpu_events[:3])}"

    # KPU events
    kpu_events = [f"{k}: {v*1e6:.2f} μJ" for k, v in kpu_result.arch_specific_events.items() if v > 0]
    if kpu_events:
        details += f"| KPU: {', '.join(kpu_events[:3])}  "

    return details


def _print_summary(results, mlp_configs, batch_sizes, comparator, print_arch_selection: List[str]):
    """Print detailed text summary with formatted tables"""

    print("\n" + "="*80)
    print("DETAILED ENERGY COMPARISON RESULTS")
    print("="*80)

    # Group results by MLP config and batch size
    for mlp_config in mlp_configs:
        for batch_size in batch_sizes:
            # Find results for this config
            cpu_result = None
            gpu_result = None
            tpu_result = None
            kpu_result = None

            for r in results['cpu']:
                if r.mlp_config.name == mlp_config.name and r.batch_size == batch_size:
                    cpu_result = r
                    break

            for r in results['gpu']:
                if r.mlp_config.name == mlp_config.name and r.batch_size == batch_size:
                    gpu_result = r
                    break

            for r in results['tpu']:
                if r.mlp_config.name == mlp_config.name and r.batch_size == batch_size:
                    tpu_result = r
                    break

            for r in results['kpu']:
                if r.mlp_config.name == mlp_config.name and r.batch_size == batch_size:
                    kpu_result = r
                    break

            if not (cpu_result and gpu_result and tpu_result and kpu_result):
                continue

            _print_config_comparison(mlp_config, batch_size, cpu_result, gpu_result, tpu_result, kpu_result, comparator, print_arch_selection)

    # Print overall summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total configurations tested: {len(mlp_configs)} MLPs × {len(batch_sizes)} batches = {len(mlp_configs) * len(batch_sizes)}")
    print(f"Total runs: {len(results['cpu']) + len(results['gpu']) + len(results['tpu']) + len(results['kpu'])}")
    print(f"\nArchitectures compared:")
    print(f"  • CPU (Stored-Program): ARM Cortex-A78AE 12-core @ 30W")
    print(f"  • GPU (Data-Parallel): NVIDIA Jetson Orin AGX @ 30W")
    print(f"  • TPU (Systolic-Array): Google Edge TPU Pro @ 30W")
    print(f"  • KPU (Domain-Flow): Stillwater KPU-T256 @ 30W")


def _print_gpu_hierarchical_breakdown(gpu_result):
    """Print hierarchical energy breakdown for GPU (Data-Parallel/SIMT)"""
    events = gpu_result.arch_specific_events

    print(f"\n{'─'*80}")
    print(f"GPU (DATA-PARALLEL SIMT) ENERGY BREAKDOWN")
    print(f"{'─'*80}")

    # Category 1: Compute Units
    print(f"\n  1. COMPUTE UNITS (Tensor Cores vs CUDA Cores)")
    tensor_core = events.get('Tensor Core Operations', 0) * 1e6
    tensor_core_ops = events.get('tensor_core_ops', 0)
    cuda_core = events.get('CUDA Core Operations', 0) * 1e6
    cuda_core_ops = events.get('cuda_core_ops', 0)
    register_file = events.get('Register File Access', 0) * 1e6
    num_register_accesses = events.get('num_register_accesses', 0)
    compute_total = tensor_core + cuda_core + register_file

    print(f"     • Tensor Core Operations:        {tensor_core:8.3f} μJ  ({tensor_core/compute_total*100:5.1f}%)  [{tensor_core_ops:,} ops]")
    print(f"     • CUDA Core Operations:          {cuda_core:8.3f} μJ  ({cuda_core/compute_total*100:5.1f}%)  [{cuda_core_ops:,} ops]")
    print(f"     • Register File Access:          {register_file:8.3f} μJ  ({register_file/compute_total*100:5.1f}%)  [{num_register_accesses:,} accesses]")
    print(f"     └─ Subtotal:                     {compute_total:8.3f} μJ")

    # Category 2: Instruction Pipeline
    print(f"\n  2. INSTRUCTION PIPELINE (Fetch → Decode → Execute)")
    inst_fetch = events.get('Instruction Fetch', 0) * 1e6
    inst_decode = events.get('Instruction Decode', 0) * 1e6
    inst_execute = events.get('Instruction Execute', 0) * 1e6
    num_instructions = events.get('num_instructions', 0)
    pipeline_total = inst_fetch + inst_decode + inst_execute

    print(f"     • Instruction Fetch:             {inst_fetch:8.3f} μJ  ({inst_fetch/pipeline_total*100:5.1f}%)  [{num_instructions:,} instructions]")
    print(f"     • Instruction Decode:            {inst_decode:8.3f} μJ  ({inst_decode/pipeline_total*100:5.1f}%)")
    print(f"     • Instruction Execute:           {inst_execute:8.3f} μJ  ({inst_execute/pipeline_total*100:5.1f}%)")
    print(f"     └─ Subtotal:                     {pipeline_total:8.3f} μJ")

    # Category 3: Memory Hierarchy (NVIDIA Ampere nomenclature)
    print(f"\n  3. MEMORY HIERARCHY (Register File → Shared Mem/L1 → L2 → DRAM)")
    # Try new nomenclature first, fall back to old for backward compatibility
    shared_mem_l1 = events.get('Shared Memory/L1 Unified', events.get('Shared Memory', 0) + events.get('L1 Cache', 0)) * 1e6
    shared_mem_l1_accesses = events.get('shared_mem_l1_accesses', 0)
    l2_cache = events.get('L2 Cache', 0) * 1e6
    l2_accesses = events.get('l2_accesses', 0)
    dram = events.get('DRAM', 0) * 1e6
    dram_accesses = events.get('dram_accesses', 0)
    memory_total = shared_mem_l1 + l2_cache + dram

    print(f"     • Shared Memory/L1 (unified):    {shared_mem_l1:8.3f} μJ  ({shared_mem_l1/memory_total*100 if memory_total > 0 else 0:5.1f}%)  [{shared_mem_l1_accesses:,} accesses]")
    print(f"     • L2 Cache (shared across SMs):  {l2_cache:8.3f} μJ  ({l2_cache/memory_total*100 if memory_total > 0 else 0:5.1f}%)  [{l2_accesses:,} accesses]")
    print(f"     • DRAM (HBM2e/LPDDR5):           {dram:8.3f} μJ  ({dram/memory_total*100 if memory_total > 0 else 0:5.1f}%)  [{dram_accesses:,} accesses]")
    print(f"     └─ Subtotal:                     {memory_total:8.3f} μJ")

    # Category 4: SIMT Control Overheads
    print(f"\n  4. SIMT CONTROL OVERHEADS (GPU-Specific)")
    coherence = events.get('Coherence Machinery', 0) * 1e6
    num_concurrent_warps = events.get('num_concurrent_warps', 0)
    num_memory_ops = events.get('num_memory_ops', 0)
    scheduling = events.get('Thread Scheduling', 0) * 1e6
    concurrent_threads = events.get('concurrent_threads', 0)
    divergence = events.get('Warp Divergence', 0) * 1e6
    num_divergent_ops = events.get('num_divergent_ops', 0)
    coalescing = events.get('Memory Coalescing', 0) * 1e6
    num_uncoalesced = events.get('num_uncoalesced', 0)
    barriers = events.get('Synchronization Barriers', 0) * 1e6
    num_barriers = events.get('num_barriers', 0)
    simt_total = coherence + scheduling + divergence + coalescing + barriers

    print(f"     • Coherence Machinery:           {coherence:8.3f} μJ  ({coherence/simt_total*100:5.1f}%)  [{num_concurrent_warps:,} warps × {num_memory_ops:,} mem ops] ← DOMINANT!")
    print(f"     • Thread Scheduling:             {scheduling:8.3f} μJ  ({scheduling/simt_total*100:5.1f}%)  [{concurrent_threads:,} threads]")
    print(f"     • Warp Divergence:               {divergence:8.3f} μJ  ({divergence/simt_total*100:5.1f}%)  [{num_divergent_ops:,} divergent ops]")
    print(f"     • Memory Coalescing:             {coalescing:8.3f} μJ  ({coalescing/simt_total*100:5.1f}%)  [{num_uncoalesced:,} uncoalesced]")
    print(f"     • Synchronization Barriers:      {barriers:8.3f} μJ  ({barriers/simt_total*100:5.1f}%)  [{num_barriers:,} barriers]")
    print(f"     └─ Subtotal:                     {simt_total:8.3f} μJ")

    # Total architectural overhead
    arch_total = compute_total + pipeline_total + memory_total + simt_total
    dynamic_energy_total = arch_total + gpu_result.compute_energy_j*1e6 + gpu_result.memory_energy_j*1e6
    idle_leakage_energy = gpu_result.total_energy_j*1e6 - dynamic_energy_total

    # Store idle energy back in result for summary table
    gpu_result.static_energy_j = idle_leakage_energy * 1e-6

    print(f"\n  TOTAL GPU ARCHITECTURAL OVERHEAD:  {arch_total:8.3f} μJ")
    print(f"  Base Compute Energy:               {gpu_result.compute_energy_j*1e6:8.3f} μJ")
    print(f"  Base Memory Energy:                {gpu_result.memory_energy_j*1e6:8.3f} μJ")
    print(f"  {'─'*80}")
    print(f"  SUBTOTAL DYNAMIC ENERGY:           {dynamic_energy_total:8.3f} μJ  ({dynamic_energy_total/gpu_result.total_energy_j/1e6*100:.1f}%)")
    print(f"  Idle/Leakage Energy (15W × latency): {idle_leakage_energy:8.3f} μJ  ({idle_leakage_energy/gpu_result.total_energy_j/1e6*100:.1f}%)")
    print(f"  {'─'*80}")
    print(f"  TOTAL GPU ENERGY:                  {gpu_result.total_energy_j*1e6:8.3f} μJ")

    # Efficiency metrics
    print(f"\n  EFFICIENCY METRICS:")
    print(f"  • Energy per MAC:                     {gpu_result.energy_per_mac_pj:.2f} pJ")

    # Calculate arithmetic intensity (ops/byte) - ROOFLINE MODEL DEFINITION
    # AI = total_ops / bytes_transferred (workload-level, consistent across architectures)
    # NOTE: Use MACs + FLOPs, not tensor_core_ops (which counts Tensor Core instructions)
    total_ops = (events.get('tensor_core_macs', 0) + events.get('cuda_core_macs', 0) +
                 events.get('cuda_core_flops', 0))
    total_bytes = events.get('bytes_transferred', 1)  # Workload bytes (avoid div by zero)
    arithmetic_intensity = total_ops / total_bytes if total_bytes > 0 else 0
    print(f"  • Arithmetic Intensity:               {arithmetic_intensity:.2f} ops/byte")

    # Compute vs Data Movement breakdown
    compute_energy = gpu_result.compute_energy_j * 1e6  # Convert to μJ
    data_movement_energy = gpu_result.memory_energy_j * 1e6  # Convert to μJ
    total_dynamic = compute_energy + data_movement_energy
    compute_efficiency = (compute_energy / total_dynamic * 100) if total_dynamic > 0 else 100.0
    print(f"  • Compute Energy:                     {compute_energy:.3f} μJ")
    print(f"  • Data Movement Energy:               {data_movement_energy:.3f} μJ")
    print(f"  • Compute Efficiency:                 {compute_efficiency:.1f}%")

    # Performance metrics
    print(f"\n  PERFORMANCE:")
    print(f"  • Latency per inference:              {gpu_result.latency_s*1e6:.2f} μs")
    print(f"  • Throughput:                         {gpu_result.throughput_inferences_per_sec:,.0f} infer/sec")
    print(f"  • Total energy per inference:         {gpu_result.total_energy_j*1e6:.3f} μJ")


def _print_tpu_hierarchical_breakdown(tpu_result):
    """Print hierarchical energy breakdown for TPU (Systolic-Array)"""
    events = tpu_result.arch_specific_events

    print(f"\n{'─'*80}")
    print(f"TPU (SYSTOLIC-ARRAY) ENERGY BREAKDOWN")
    print(f"{'─'*80}")

    # Check if we have detailed systolic array energy model data
    if 'instruction_decode' in events:
        # Component 1: Instruction Decode (per matrix operation, not per MAC!)
        print(f"\n  1. INSTRUCTION DECODE (Per matrix operation)")
        inst_decode = events.get('instruction_decode', 0) * 1e6
        num_matrix_ops = events.get('num_matrix_ops', 0)

        print(f"     • Matrix Operation Decode:          {inst_decode:8.3f} μJ  [{num_matrix_ops:,} matrix ops]")
        print(f"     └─ Subtotal:                        {inst_decode:8.3f} μJ")

        # Component 2: DMA Controller
        print(f"\n  2. DMA CONTROLLER (Off-chip data transfers)")
        dma_setup = events.get('dma_setup', 0) * 1e6
        dma_addr_gen = events.get('dma_address_gen', 0) * 1e6
        num_dma_transfers = events.get('num_dma_transfers', 0)
        num_cache_lines = events.get('num_cache_lines', 0)
        dma_total = dma_setup + dma_addr_gen

        print(f"     • Descriptor Setup:                 {dma_setup:8.3f} μJ  ({dma_setup/dma_total*100 if dma_total > 0 else 0:5.1f}%)  [{num_dma_transfers:,} transfers]")
        print(f"     • Address Generation:               {dma_addr_gen:8.3f} μJ  ({dma_addr_gen/dma_total*100 if dma_total > 0 else 0:5.1f}%)  [{num_cache_lines:,} cache lines]")
        print(f"     └─ Subtotal:                        {dma_total:8.3f} μJ")

        # Component 3: Weight Loading Sequencer
        print(f"\n  3. WEIGHT LOADING SEQUENCER (Shift into systolic array)")
        weight_shift = events.get('weight_shift_control', 0) * 1e6
        weight_column = events.get('weight_column_select', 0) * 1e6
        num_cycles = events.get('num_systolic_cycles', 0)
        num_weight_elements = events.get('num_weight_elements', 0)
        weight_total = weight_shift + weight_column

        print(f"     • Weight Shift Control:             {weight_shift:8.3f} μJ  ({weight_shift/weight_total*100 if weight_total > 0 else 0:5.1f}%)  [{num_weight_elements:,} elements]")
        print(f"     • Column Select:                    {weight_column:8.3f} μJ  ({weight_column/weight_total*100 if weight_total > 0 else 0:5.1f}%)  [{num_cycles:,} cycles]")
        print(f"     └─ Subtotal:                        {weight_total:8.3f} μJ")

        # Component 4: Unified Buffer Controller
        print(f"\n  4. UNIFIED BUFFER CONTROLLER (Activation scratchpad)")
        ub_addr_gen = events.get('ub_address_gen', 0) * 1e6
        ub_arbitration = events.get('ub_arbitration', 0) * 1e6
        num_ub_accesses = events.get('num_ub_accesses', 0)
        ub_total = ub_addr_gen + ub_arbitration

        print(f"     • Address Generation:               {ub_addr_gen:8.3f} μJ  ({ub_addr_gen/ub_total*100 if ub_total > 0 else 0:5.1f}%)  [{num_ub_accesses:,} accesses]")
        print(f"     • Arbitration:                      {ub_arbitration:8.3f} μJ  ({ub_arbitration/ub_total*100 if ub_total > 0 else 0:5.1f}%)  [{num_ub_accesses:,} requests]")
        print(f"     └─ Subtotal:                        {ub_total:8.3f} μJ")

        # Component 5: Accumulator Controller
        print(f"\n  5. ACCUMULATOR CONTROLLER (Partial sum staging)")
        acc_read = events.get('accumulator_read', 0) * 1e6
        acc_write = events.get('accumulator_write', 0) * 1e6
        acc_addr = events.get('accumulator_address', 0) * 1e6
        num_accumulator_ops = events.get('num_accumulator_ops', 0)
        acc_total = acc_read + acc_write + acc_addr

        print(f"     • Read Control:                     {acc_read:8.3f} μJ  ({acc_read/acc_total*100 if acc_total > 0 else 0:5.1f}%)  [{num_accumulator_ops:,} reads]")
        print(f"     • Write Control:                    {acc_write:8.3f} μJ  ({acc_write/acc_total*100 if acc_total > 0 else 0:5.1f}%)  [{num_accumulator_ops:,} writes]")
        print(f"     • Address Generation:               {acc_addr:8.3f} μJ  ({acc_addr/acc_total*100 if acc_total > 0 else 0:5.1f}%)  [{num_accumulator_ops:,} addresses]")
        print(f"     └─ Subtotal:                        {acc_total:8.3f} μJ")

        # Component 6: Tile Loop Control
        print(f"\n  6. TILE LOOP CONTROL (Tiled matrix operations)")
        tile_loop = events.get('tile_loop_control', 0) * 1e6
        num_tiles = events.get('num_tiles', 0)

        print(f"     • Tile Iteration:                   {tile_loop:8.3f} μJ  [{num_tiles:,} tiles]")
        print(f"     └─ Subtotal:                        {tile_loop:8.3f} μJ")

        # Component 7: Data Injection/Extraction
        print(f"\n  7. DATA INJECTION/EXTRACTION (Spatial array interface)")
        injection = events.get('injection_energy', 0) * 1e6
        extraction = events.get('extraction_energy', 0) * 1e6
        num_elements = events.get('num_elements', 0)
        data_move_total = injection + extraction

        print(f"     • Data Injection:                   {injection:8.3f} μJ  ({injection/data_move_total*100 if data_move_total > 0 else 0:5.1f}%)  [{num_elements:,} elements]")
        print(f"     • Data Extraction:                  {extraction:8.3f} μJ  ({extraction/data_move_total*100 if data_move_total > 0 else 0:5.1f}%)  [{num_elements:,} elements]")
        print(f"     └─ Subtotal:                        {data_move_total:8.3f} μJ")

        # Total Control Overhead
        total_control = inst_decode + dma_total + weight_total + ub_total + acc_total + tile_loop + data_move_total
        control_per_mac = events.get('control_overhead_per_mac_pj', 0)
        array_dim = events.get('array_dimension', 128)

        print(f"\n  TOTAL TPU CONTROL OVERHEAD:        {total_control:8.3f} μJ")
        print(f"  Control per MAC:                     {control_per_mac:.4f} pJ")
        print(f"  Systolic Array Dimension:            {array_dim} × {array_dim} MACs")

        # Summary
        arch_total = total_control
        dynamic_energy_total = arch_total + tpu_result.compute_energy_j*1e6 + tpu_result.memory_energy_j*1e6
        idle_leakage_energy = tpu_result.total_energy_j*1e6 - dynamic_energy_total

        # Store idle energy back in result for summary table
        tpu_result.static_energy_j = idle_leakage_energy * 1e-6

        print(f"\n  TOTAL TPU ARCHITECTURAL OVERHEAD:  {arch_total:8.3f} μJ")
        print(f"  Base Compute Energy:               {tpu_result.compute_energy_j*1e6:8.3f} μJ")
        print(f"  Base Memory Energy:                {tpu_result.memory_energy_j*1e6:8.3f} μJ")
        print(f"  {'─'*80}")
        print(f"  SUBTOTAL DYNAMIC ENERGY:           {dynamic_energy_total:8.3f} μJ  ({dynamic_energy_total/tpu_result.total_energy_j/1e6*100:.1f}%)")
        print(f"  Idle/Leakage Energy (30W x latency): {idle_leakage_energy:8.3f} μJ  ({idle_leakage_energy/tpu_result.total_energy_j/1e6*100:.1f}%)")
        print(f"  {'─'*80}")
        print(f"  TOTAL TPU ENERGY:                  {tpu_result.total_energy_j*1e6:8.3f} μJ")

        # Efficiency metrics
        print(f"\n  EFFICIENCY METRICS:")
        print(f"  • Energy per MAC:                     {tpu_result.energy_per_mac_pj:.2f} pJ")

        # Calculate arithmetic intensity (ops/byte) - ROOFLINE MODEL DEFINITION
        # AI = total_ops / bytes_transferred (workload-level, consistent across architectures)
        total_ops = events.get('total_macs', 0) * 2  # MAC = 2 ops
        total_bytes = events.get('bytes_transferred', 1)  # Workload bytes (avoid div by zero)
        arithmetic_intensity = total_ops / total_bytes if total_bytes > 0 else 0
        print(f"  • Arithmetic Intensity:               {arithmetic_intensity:.2f} ops/byte")

        # Compute vs Data Movement breakdown
        compute_energy = tpu_result.compute_energy_j * 1e6  # Convert to μJ
        data_movement_energy = tpu_result.memory_energy_j * 1e6  # Convert to μJ
        total_dynamic = compute_energy + data_movement_energy
        compute_efficiency = (compute_energy / total_dynamic * 100) if total_dynamic > 0 else 100.0
        print(f"  • Compute Energy:                     {compute_energy:.3f} μJ")
        print(f"  • Data Movement Energy:               {data_movement_energy:.3f} μJ")
        print(f"  • Compute Efficiency:                 {compute_efficiency:.1f}%")

        # Performance metrics
        print(f"\n  PERFORMANCE:")
        print(f"  • Latency per inference:              {tpu_result.latency_s*1e6:.2f} μs")
        print(f"  • Throughput:                         {tpu_result.throughput_inferences_per_sec:,.0f} infer/sec")
        print(f"  • Total energy per inference:         {tpu_result.total_energy_j*1e6:.3f} μJ")
    else:
        # Fallback: simple systolic array model
        print(f"\n  (Simplified systolic array model - no detailed breakdown available)")
        systolic_mac = events.get('systolic_array_mac_energy', 0) * 1e6
        on_chip = events.get('on_chip_buffer_energy', 0) * 1e6
        dram = events.get('dram_energy', 0) * 1e6

        if systolic_mac > 0 or on_chip > 0 or dram > 0:
            print(f"  • Systolic Array Operations:        {systolic_mac:8.3f} μJ")
            print(f"  • On-Chip Buffer Access:            {on_chip:8.3f} μJ")
            print(f"  • Off-Chip DRAM Access:             {dram:8.3f} μJ")

        # Summary
        arch_total = tpu_result.get_total_architectural_overhead() * 1e6
        dynamic_energy_total = arch_total + tpu_result.compute_energy_j*1e6 + tpu_result.memory_energy_j*1e6
        idle_leakage_energy = tpu_result.total_energy_j*1e6 - dynamic_energy_total

        # Store idle energy back in result for summary table
        tpu_result.static_energy_j = idle_leakage_energy * 1e-6

        print(f"\n  TOTAL TPU ARCHITECTURAL OVERHEAD:  {arch_total:8.3f} μJ")
        print(f"  Base Compute Energy:               {tpu_result.compute_energy_j*1e6:8.3f} μJ")
        print(f"  Base Memory Energy:                {tpu_result.memory_energy_j*1e6:8.3f} μJ")
        print(f"  {'─'*80}")
        print(f"  SUBTOTAL DYNAMIC ENERGY:           {dynamic_energy_total:8.3f} μJ  ({dynamic_energy_total/tpu_result.total_energy_j/1e6*100:.1f}%)")
        print(f"  Idle/Leakage Energy (2W × latency): {idle_leakage_energy:8.3f} μJ  ({idle_leakage_energy/tpu_result.total_energy_j/1e6*100:.1f}%)")
        print(f"  {'─'*80}")
        print(f"  TOTAL TPU ENERGY:                  {tpu_result.total_energy_j*1e6:8.3f} μJ")

        # Efficiency metrics (fallback path)
        print(f"\n  EFFICIENCY METRICS:")
        print(f"  • Energy per MAC:                     {tpu_result.energy_per_mac_pj:.2f} pJ")

        # Compute vs Data Movement breakdown
        compute_energy = tpu_result.compute_energy_j * 1e6  # Convert to μJ
        data_movement_energy = tpu_result.memory_energy_j * 1e6  # Convert to μJ
        total_dynamic = compute_energy + data_movement_energy
        compute_efficiency = (compute_energy / total_dynamic * 100) if total_dynamic > 0 else 100.0
        print(f"  • Compute Energy:                     {compute_energy:.3f} μJ")
        print(f"  • Data Movement Energy:               {data_movement_energy:.3f} μJ")
        print(f"  • Compute Efficiency:                 {compute_efficiency:.1f}%")

        # Performance metrics
        print(f"\n  PERFORMANCE:")
        print(f"  • Latency per inference:              {tpu_result.latency_s*1e6:.2f} μs")
        print(f"  • Throughput:                         {tpu_result.throughput_inferences_per_sec:,.0f} infer/sec")
        print(f"  • Total energy per inference:         {tpu_result.total_energy_j*1e6:.3f} μJ")


def _print_kpu_hierarchical_breakdown(kpu_result):
    """Print hierarchical energy breakdown for KPU (Domain-Flow)"""
    events = kpu_result.arch_specific_events

    print(f"\n{'─'*80}")
    print(f"KPU (DOMAIN-FLOW) ENERGY BREAKDOWN")
    print(f"{'─'*80}")

    # Extract 8-component breakdown from KPUTileEnergyAdapter
    # Check if we have detailed tile energy model data
    if 'dram_energy' in events:
        
        # Component 1: SURE Program Loading
        print(f"\n  1. SURE PROGRAM LOADING (Spatial dataflow configuration)")
        program_load = events.get('program_load_energy', 0) * 1e6
        cache_hit_rate = events.get('cache_hit_rate', 0.9)

        print(f"     • Program Load/Broadcast:           {program_load:8.3f} μJ")
        print(f"     • Cache Hit Rate:                   {cache_hit_rate*100:5.1f}%")
        print(f"     └─ Subtotal:                        {program_load:8.3f} μJ")

        # Component 2: 4-Stage Memory Hierarchy
        print(f"\n  2. MEMORY HIERARCHY (4-Stage: DRAM → L3 → L2 → L1)")
        dram = events.get('dram_energy', 0) * 1e6
        l3 = events.get('l3_energy', 0) * 1e6
        l2 = events.get('l2_energy', 0) * 1e6
        l1 = events.get('l1_energy', 0) * 1e6
        dram_accesses = events.get('dram_accesses', 0)
        l3_accesses = events.get('l3_accesses', 0)
        l2_accesses = events.get('l2_accesses', 0)
        l1_accesses = events.get('l1_accesses', 0)
        total_bytes = events.get('total_bytes', 0)
        mem_total = dram + l3 + l2 + l1

        print(f"     • DRAM (off-chip):                  {dram:8.3f} μJ  ({dram/mem_total*100 if mem_total > 0 else 0:5.1f}%)  [{dram_accesses:,} accesses]")
        print(f"     • L3 Cache (distributed scratchpad):{l3:8.3f} μJ  ({l3/mem_total*100 if mem_total > 0 else 0:5.1f}%)  [{l3_accesses:,} accesses]")
        print(f"     • L2 Cache (tile-local):            {l2:8.3f} μJ  ({l2/mem_total*100 if mem_total > 0 else 0:5.1f}%)  [{l2_accesses:,} accesses]")
        print(f"     • L1 Cache (PE-local):              {l1:8.3f} μJ  ({l1/mem_total*100 if mem_total > 0 else 0:5.1f}%)  [{l1_accesses:,} accesses]")
        print(f"     • Total Data Transferred:           [{total_bytes/1024:.1f} KB]")
        print(f"     └─ Subtotal:                        {mem_total:8.3f} μJ")

        # Component 3: Data Movement Engines
        print(f"\n  3. DATA MOVEMENT ENGINES (3 Specialized Engines)")
        dma = events.get('dma_energy', 0) * 1e6
        blockmover = events.get('blockmover_energy', 0) * 1e6
        streamer = events.get('streamer_energy', 0) * 1e6
        dma_bytes = events.get('dma_bytes', 0)
        blockmover_bytes = events.get('blockmover_bytes', 0)
        streamer_bytes = events.get('streamer_bytes', 0)
        dme_total = dma + blockmover + streamer

        print(f"     • DMA Engine (DRAM ↔ L3):           {dma:8.3f} μJ  ({dma/dme_total*100 if dme_total > 0 else 0:5.1f}%)  [{dma_bytes/1024:.1f} KB]")
        print(f"     • BlockMover (L3 ↔ L2):             {blockmover:8.3f} μJ  ({blockmover/dme_total*100 if dme_total > 0 else 0:5.1f}%)  [{blockmover_bytes/1024:.1f} KB]")
        print(f"     • Streamer (L2 ↔ L1:                {streamer:8.3f} μJ  ({streamer/dme_total*100 if dme_total > 0 else 0:5.1f}%)  [{streamer_bytes/1024:.1f} KB]")
        print(f"     └─ Subtotal:                        {dme_total:8.3f} μJ")

        # Component 4: Distributed L3 Scratchpad Routing
        print(f"\n  4. DISTRIBUTED L3 SCRATCHPAD (NoC routing)")
        l3_routing = events.get('l3_routing_energy', 0) * 1e6
        avg_hops = events.get('average_l3_hops', 0)
        l3_routing_accesses = events.get('l3_routing_accesses', 0)

        print(f"     • NoC Routing Energy:               {l3_routing:8.3f} μJ  [{l3_routing_accesses:,} accesses]")
        print(f"     • Average Hops:                     {avg_hops:5.1f}")
        print(f"     └─ Subtotal:                        {l3_routing:8.3f} μJ")

        # Component 5: Instruction Token Signature Matching & Dispatch
        print(f"\n  5. INSTRUCTION TOKEN MATCHING & DISPATCH (Dataflow execution)")
        token_match = events.get('token_matching_energy', 0) * 1e6
        signature = events.get('signature_matching_energy', 0) * 1e6
        dispatch = events.get('dispatch_energy', 0) * 1e6
        num_signature_matches = events.get('num_signature_matches', 0)
        num_tokens = events.get('num_tokens', 0)

        print(f"     • Signature Matching:               {signature:8.3f} μJ  [{num_signature_matches:,} matches]")
        print(f"     • Instruction Token Dispatch:       {dispatch:8.3f} μJ  [{num_tokens:,} tokens fired]")
        print(f"     └─ Subtotal:                        {token_match:8.3f} μJ")

        # Component 6: Operator Fusion
        print(f"\n  6. OPERATOR FUSION (Hardware fusion)")
        fusion_net = events.get('fusion_net_energy', 0) * 1e6

        print(f"     • Fusion Coordination:              {fusion_net:8.3f} μJ")
        print(f"     └─ Subtotal:                        {fusion_net:8.3f} μJ")

        # Component 7: Token Routing
        print(f"\n  7. TOKEN ROUTING (Mesh routing)")
        token_routing = events.get('token_routing_energy', 0) * 1e6
        routing_dist = events.get('average_routing_distance', 0)
        num_tokens = events.get('num_tokens', 0)

        print(f"     • Token Routing Hops:               {token_routing:8.3f} μJ  [{num_tokens:,} tokens]")
        print(f"     • Average Distance:                 {routing_dist:5.1f} hops")
        print(f"     └─ Subtotal:                        {token_routing:8.3f} μJ")

        # Component 8: Computation
        print(f"\n  8. COMPUTATION (Compute Fabric operators)")
        compute = events.get('compute_energy', 0) * 1e6
        total_ops = events.get('total_ops', 0)

        print(f"     • MAC Operations:                   {compute:8.3f} μJ  [{total_ops:,} ops]")
        print(f"     └─ Subtotal:                        {compute:8.3f} μJ")

        # Total
        arch_overhead = mem_total + dme_total + token_match + program_load + l3_routing + fusion_net + token_routing
        dynamic_energy_total = arch_overhead + compute
        idle_leakage_energy = kpu_result.total_energy_j*1e6 - dynamic_energy_total

        # Store idle energy back in result for summary table
        kpu_result.static_energy_j = idle_leakage_energy * 1e-6

        print(f"\n  TOTAL KPU ARCHITECTURAL OVERHEAD:     {arch_overhead:8.3f} μJ")
        print(f"  Base Compute Energy (from above):     {compute:8.3f} μJ")
        print(f"  {'─'*80}")
        print(f"  SUBTOTAL DYNAMIC ENERGY:              {dynamic_energy_total:8.3f} μJ  ({dynamic_energy_total/kpu_result.total_energy_j/1e6*100:.1f}%)")
        print(f"  Idle/Leakage Energy (15W × latency):  {idle_leakage_energy:8.3f} μJ  ({idle_leakage_energy/kpu_result.total_energy_j/1e6*100:.1f}%)")
        print(f"  {'─'*80}")
        print(f"  TOTAL KPU ENERGY:                     {kpu_result.total_energy_j*1e6:8.3f} μJ")

        # Metrics
        if 'energy_per_mac_pj' in events:
            print(f"\n  EFFICIENCY METRICS:")
            print(f"  • Energy per MAC:                     {events['energy_per_mac_pj']:.2f} pJ")
            print(f"  • Arithmetic Intensity:               {events.get('arithmetic_intensity', 0):.2f} ops/byte")

            # Compute vs Data Movement breakdown
            compute_energy = kpu_result.compute_energy_j * 1e6  # Convert to μJ
            data_movement_energy = kpu_result.memory_energy_j * 1e6  # Convert to μJ
            total_dynamic = compute_energy + data_movement_energy
            compute_efficiency = (compute_energy / total_dynamic * 100) if total_dynamic > 0 else 100.0
            print(f"  • Compute Energy:                     {compute_energy:.3f} μJ")
            print(f"  • Data Movement Energy:               {data_movement_energy:.3f} μJ")
            print(f"  • Compute Efficiency:                 {compute_efficiency:.1f}%")

            # Performance metrics
            print(f"\n  PERFORMANCE:")
            print(f"  • Latency per inference:              {kpu_result.latency_s*1e6:.2f} μs")
            print(f"  • Throughput:                         {kpu_result.throughput_inferences_per_sec:,.0f} infer/sec")
            print(f"  • Total energy per inference:         {kpu_result.total_energy_j*1e6:.3f} μJ")

        print(f"\n  WHY SO EFFICIENT? Token-based distributed dataflow:")
        print(f"  • No instruction fetch/decode (dataflow, not stored-program)")
        print(f"  • No coherence machinery (explicit spatial token routing vs random cache coherence)")
        print(f"  • 4-stage memory hierarchy reduces DRAM traffic")
        print(f"  • 3 specialized data movement engines (DMA, BlockMover, Streamer) to implement system level execution schedules")
    else:
        # ERROR
        print(f"\nERROR: No detailed KPU energy breakdown data available.")


def _print_cpu_hierarchical_breakdown(cpu_result):
    """Print hierarchical energy breakdown for CPU (Stored-Program)"""
    events = cpu_result.arch_specific_events

    print(f"\n{'─'*80}")
    print(f"CPU (STORED-PROGRAM MULTICORE) ENERGY BREAKDOWN")
    print(f"{'─'*80}")

    # Component 1: Instruction Pipeline
    print(f"\n  1. INSTRUCTION PIPELINE (Fetch → Decode → Dispatch)")
    inst_fetch = events.get('instruction_fetch_energy', 0) * 1e6
    inst_decode = events.get('instruction_decode_energy', 0) * 1e6
    inst_dispatch = events.get('instruction_dispatch_energy', 0) * 1e6
    num_instructions = events.get('num_instructions', 0)
    pipeline_total = inst_fetch + inst_decode + inst_dispatch

    print(f"     • Instruction Fetch (I-cache):      {inst_fetch:8.3f} μJ  ({inst_fetch/pipeline_total*100 if pipeline_total > 0 else 0:5.1f}%)  [{num_instructions:,} instructions]")
    print(f"     • Instruction Decode:               {inst_decode:8.3f} μJ  ({inst_decode/pipeline_total*100 if pipeline_total > 0 else 0:5.1f}%)")
    print(f"     • Instruction Dispatch:             {inst_dispatch:8.3f} μJ  ({inst_dispatch/pipeline_total*100 if pipeline_total > 0 else 0:5.1f}%)")
    print(f"     └─ Subtotal:                        {pipeline_total:8.3f} μJ")
    print(f"        NOTE: Dispatch writes control signals; actual ALU execution tracked separately")

    # Component 2: Register File Operations
    print(f"\n  2. REGISTER FILE OPERATIONS (2 reads + 1 write per instruction)")
    reg_read = events.get('register_read_energy', 0) * 1e6
    reg_write = events.get('register_write_energy', 0) * 1e6
    num_reg_reads = events.get('num_register_reads', 0)
    num_reg_writes = events.get('num_register_writes', 0)
    regfile_total = reg_read + reg_write

    print(f"     • Register Reads:                   {reg_read:8.3f} μJ  ({reg_read/regfile_total*100 if regfile_total > 0 else 0:5.1f}%) [{num_reg_reads:,} reads]")
    print(f"     • Register Writes:                  {reg_write:8.3f} μJ  ({reg_write/regfile_total*100 if regfile_total > 0 else 0:5.1f}%) [{num_reg_writes:,} writes]")
    print(f"     └─ Subtotal:                        {regfile_total:8.3f} μJ")
    print(f"        NOTE: Register energy ≈ ALU energy (both ~0.6-0.8 pJ per op)")

    # Component 3: Memory Hierarchy (4-Stage)
    print(f"\n  3. MEMORY HIERARCHY (4-Stage: L1 → L2 → L3 → DRAM)")
    l1 = events.get('l1_cache_energy', 0) * 1e6
    l2 = events.get('l2_cache_energy', 0) * 1e6
    l3 = events.get('l3_cache_energy', 0) * 1e6
    dram = events.get('dram_energy', 0) * 1e6
    l1_accesses = events.get('l1_accesses', 0)
    l2_accesses = events.get('l2_accesses', 0)
    l3_accesses = events.get('l3_accesses', 0)
    dram_accesses = events.get('dram_accesses', 0)
    mem_total = l1 + l2 + l3 + dram

    print(f"     • L1 Cache (per-core, 32 KB):      {l1:8.3f} μJ  ({l1/mem_total*100 if mem_total > 0 else 0:5.1f}%)  [{l1_accesses:,} accesses]")
    print(f"     • L2 Cache (per-core, 256 KB):     {l2:8.3f} μJ  ({l2/mem_total*100 if mem_total > 0 else 0:5.1f}%)  [{l2_accesses:,} accesses]")
    print(f"     • L3 Cache (shared LLC, 8 MB):     {l3:8.3f} μJ  ({l3/mem_total*100 if mem_total > 0 else 0:5.1f}%)  [{l3_accesses:,} accesses]")
    print(f"     • DRAM (off-chip DDR4):            {dram:8.3f} μJ  ({dram/mem_total*100 if mem_total > 0 else 0:5.1f}%)  [{dram_accesses:,} accesses]")
    print(f"     └─ Subtotal:                        {mem_total:8.3f} μJ")

    # Component 4: ALU Operations
    print(f"\n  4. ALU OPERATIONS (Floating-point arithmetic)")
    alu = events.get('alu_energy', 0) * 1e6
    num_instructions = events.get('num_instructions', 0)

    print(f"     • ALU Energy:                       {alu:8.3f} μJ  [{num_instructions:,} ops]")
    print(f"     └─ Subtotal:                        {alu:8.3f} μJ")

    # Component 5: Branch Prediction
    print(f"\n  5. BRANCH PREDICTION (Control flow)")
    branch = events.get('branch_energy', 0) * 1e6
    num_branches = events.get('num_branches', 0)
    num_mispredicted = events.get('num_mispredicted_branches', 0)
    prediction_rate = events.get('branch_prediction_success_rate', 0.95) * 100

    print(f"     • Branch Prediction:                {branch:8.3f} μJ  [{num_branches:,} branches, {num_mispredicted:,} mispredicted @ {prediction_rate:.0f}% success]")
    print(f"     └─ Subtotal:                        {branch:8.3f} μJ")

    # Total
    arch_overhead = pipeline_total + regfile_total + mem_total + alu + branch
    dynamic_energy_total = arch_overhead + cpu_result.compute_energy_j*1e6 + cpu_result.memory_energy_j*1e6
    idle_leakage_energy = cpu_result.total_energy_j*1e6 - dynamic_energy_total

    # Store idle energy back in result for summary table
    cpu_result.static_energy_j = idle_leakage_energy * 1e-6

    print(f"\n  TOTAL CPU ARCHITECTURAL OVERHEAD:     {arch_overhead:8.3f} μJ")
    print(f"  Base Compute Energy (from mapper):    {cpu_result.compute_energy_j*1e6:8.3f} μJ")
    print(f"  Base Memory Energy (from mapper):     {cpu_result.memory_energy_j*1e6:8.3f} μJ")
    print(f"  {'─'*80}")
    print(f"  SUBTOTAL DYNAMIC ENERGY:              {dynamic_energy_total:8.3f} μJ  ({dynamic_energy_total/cpu_result.total_energy_j/1e6*100:.1f}%)")
    print(f"  Idle/Leakage Energy (15W × latency):  {idle_leakage_energy:8.3f} μJ  ({idle_leakage_energy/cpu_result.total_energy_j/1e6*100:.1f}%)")
    print(f"  {'─'*80}")
    print(f"  TOTAL CPU ENERGY:                     {cpu_result.total_energy_j*1e6:8.3f} μJ")

    # Efficiency metrics
    print(f"\n  EFFICIENCY METRICS:")
    print(f"  • Energy per MAC:                     {cpu_result.energy_per_mac_pj:.2f} pJ")

    # Calculate arithmetic intensity (ops/byte) - ROOFLINE MODEL DEFINITION
    # AI = total_ops / bytes_transferred (workload-level, consistent across architectures)
    total_ops = events.get('alu_ops', 0) + events.get('fpu_ops', 0)
    total_bytes = events.get('bytes_transferred', 1)  # Workload bytes (avoid div by zero)
    arithmetic_intensity = total_ops / total_bytes if total_bytes > 0 else 0
    print(f"  • Arithmetic Intensity:               {arithmetic_intensity:.2f} ops/byte")

    # Compute vs Data Movement breakdown
    compute_energy = cpu_result.compute_energy_j * 1e6  # Convert to μJ
    data_movement_energy = cpu_result.memory_energy_j * 1e6  # Convert to μJ
    total_dynamic = compute_energy + data_movement_energy
    compute_efficiency = (compute_energy / total_dynamic * 100) if total_dynamic > 0 else 100.0
    print(f"  • Compute Energy:                     {compute_energy:.3f} μJ")
    print(f"  • Data Movement Energy:               {data_movement_energy:.3f} μJ")
    print(f"  • Compute Efficiency:                 {compute_efficiency:.1f}%")

    # Performance metrics
    print(f"\n  PERFORMANCE:")
    print(f"  • Latency per inference:              {cpu_result.latency_s*1e6:.2f} μs")
    print(f"  • Throughput:                         {cpu_result.throughput_inferences_per_sec:,.0f} infer/sec")
    print(f"  • Total energy per inference:         {cpu_result.total_energy_j*1e6:.3f} μJ")

    print(f"\n  CPU CHARACTERISTICS:")
    print(f"  • Instruction fetch overhead: {inst_fetch:.3f} μJ ({num_instructions:,} instructions)")
    print(f"  • Register file energy: {regfile_total:.3f} μJ (comparable to ALU: {alu:.3f} μJ)")
    print(f"  • Memory hierarchy: {mem_total:.3f} μJ (L1: {l1/mem_total*100:.0f}%, DRAM: {dram/mem_total*100:.0f}%)")
    print(f"  • Lower than GPU (no massive coherence machinery)")
    print(f"  • Higher than KPU (dataflow eliminates instruction fetch)")


def _print_workload_characteristics(mlp_config, batch_size):
    """Print detailed workload characteristics"""
    print(f"\n{'─'*80}")
    print(f"WORKLOAD CHARACTERISTICS")
    print(f"{'─'*80}")

    # Compute requirements
    total_macs = mlp_config.total_macs * batch_size
    total_flops = mlp_config.total_flops * batch_size
    total_ops = total_macs + total_flops

    print(f"\nCompute Requirements:")
    print(f"  • Total MACs:                    {total_macs:,} multiply-accumulates")
    print(f"  • Total FLOPs:                   {total_flops:,} floating-point operations")
    print(f"  • Total operations:              {total_ops:,} (MACs + FLOPs)")
    print(f"  • Per inference:                 {mlp_config.total_macs:,} MACs, {mlp_config.total_flops:,} FLOPs")

    # Memory requirements
    input_bytes = mlp_config.input_size * batch_size
    weight_bytes = mlp_config.weight_size
    output_bytes = mlp_config.output_size * batch_size
    total_bytes = input_bytes + weight_bytes + output_bytes

    print(f"\nMemory Requirements:")
    print(f"  • Input activations:             {input_bytes:,} bytes ({input_bytes/1024:.1f} KB)")
    print(f"  • Weights (shared):              {weight_bytes:,} bytes ({weight_bytes/1024:.1f} KB)")
    print(f"  • Output activations:            {output_bytes:,} bytes ({output_bytes/1024:.1f} KB)")
    print(f"  • Total data movement:           {total_bytes:,} bytes ({total_bytes/1024:.1f} KB)")

    # Arithmetic intensity
    arithmetic_intensity = total_ops / total_bytes if total_bytes > 0 else 0
    print(f"\nArithmetic Intensity:")
    print(f"  • Operations per byte:           {arithmetic_intensity:.2f} ops/byte")
    if arithmetic_intensity < 1.0:
        print(f"  • Classification:                MEMORY-BOUND (AI < 1)")
    elif arithmetic_intensity < 10.0:
        print(f"  • Classification:                BALANCED (1 ≤ AI < 10)")
    else:
        print(f"  • Classification:                COMPUTE-BOUND (AI ≥ 10)")


def _print_hardware_energy_config(gpu_result, kpu_result, cpu_result, comparator):
    """Print hardware energy configuration for all architectures"""
    print(f"\n{'─'*80}")
    print(f"HARDWARE ENERGY CONFIGURATION")
    print(f"{'─'*80}")

    # Extract base ALU energy from resource models
    cpu_base_alu = comparator.cpu_mapper.resource_model.energy_per_flop_fp32 * 1e12  # pJ    
    gpu_base_alu = comparator.gpu_mapper.resource_model.energy_per_flop_fp32 * 1e12  # pJ
    tpu_base_alu = comparator.tpu_mapper.resource_model.energy_per_flop_fp32 * 1e12  # pJ    
    kpu_base_alu = comparator.kpu_mapper.resource_model.energy_per_flop_fp32 * 1e12  # pJ

    # Get architectural energy model for CPU overhead
    print(f"\nCPU (ARM Cortex-A78AE @ 30W, 12 cores @ 2.2 GHz):")
    print(f"  • Base ALU energy (FP32):        {cpu_base_alu:.2f} pJ")

    # Calculate MAC energies from energy_scaling
    cpu_model = comparator.cpu_mapper.resource_model
    if hasattr(cpu_model, 'energy_scaling'):
        int8_energy = cpu_model.energy_per_flop_fp32 * cpu_model.energy_scaling.get(Precision.INT8, 0.25) * 1e12
        fp16_energy = cpu_model.energy_per_flop_fp32 * cpu_model.energy_scaling.get(Precision.FP16, 0.5) * 1e12
        fp32_energy = cpu_model.energy_per_flop_fp32 * cpu_model.energy_scaling.get(Precision.FP32, 1.0) * 1e12
        print(f"  • MAC operation:                 INT8: {int8_energy:.2f} pJ, FP16: {fp16_energy:.2f} pJ, FP32: {fp32_energy:.2f} pJ")

    if hasattr(comparator.cpu_mapper.resource_model, 'architecture_energy_model'):
        arch_model = comparator.cpu_mapper.resource_model.architecture_energy_model
        if arch_model:
            inst_fetch = arch_model.instruction_fetch_energy * 1e12  # pJ
            reg_read = arch_model.register_file_read_energy * 1e12  # pJ
            reg_write = arch_model.register_file_write_energy * 1e12  # pJ

            print(f"  • Instruction fetch:             {inst_fetch:.2f} pJ per instruction")
            print(f"  • Register read:                 {reg_read:.2f} pJ per read")
            print(f"  • Register write:                {reg_write:.2f} pJ per write")
    print(f"  • Memory hierarchy:              4-stage (L1 → L2 → L3 → DRAM)")

    # GPU architectural energy model
    print(f"\nGPU (Jetson Orin AGX @ 30W, Ampere SMs @ 650 MHz):")
    print(f"  • Base ALU energy (FP32):        {gpu_base_alu:.2f} pJ")

    # Calculate MAC energies from energy_scaling (CUDA cores)
    gpu_model = comparator.gpu_mapper.resource_model
    if hasattr(gpu_model, 'energy_scaling'):
        int8_energy = gpu_model.energy_per_flop_fp32 * gpu_model.energy_scaling.get(Precision.INT8, 0.125) * 1e12
        bf16_energy = gpu_model.energy_per_flop_fp32 * gpu_model.energy_scaling.get(Precision.BF16, 0.5) * 1e12
        fp32_energy = gpu_model.energy_per_flop_fp32 * gpu_model.energy_scaling.get(Precision.FP32, 1.0) * 1e12
        print(f"  • MAC operation (CUDA Core):     INT8: {int8_energy:.2f} pJ, BF16: {bf16_energy:.2f} pJ, FP32: {fp32_energy:.2f} pJ")

    # Get architectural energy model for specialized units
    if hasattr(comparator.gpu_mapper.resource_model, 'architecture_energy_model'):
        arch_model = comparator.gpu_mapper.resource_model.architecture_energy_model
        if arch_model:
            cuda_core_mac = arch_model.cuda_core_mac_energy * 1e12  # pJ per MAC
            tensor_core_mac = arch_model.tensor_core_mac_energy * 1e12  # pJ per MAC
            register_access = arch_model.register_file_energy_per_access * 1e12  # pJ

            print(f"  • Tensor Core MAC:               {tensor_core_mac:.2f} pJ (64 MACs per clock)")
            print(f"  • Register file access:          {register_access:.2f} pJ")
            print(f"  • Coherence per request:         {arch_model.coherence_energy_per_request * 1e12:.2f} pJ")
    print(f"  • Memory hierarchy:              Register File → Shared Mem/L1 (unified) → L2 → DRAM")

    # TPU architectural energy model
    print(f"\nTPU (TPU Edge Pro @ 30W, 128×128 systolic @ 850 MHz):")
    print(f"  • Base ALU energy (FP32):        {tpu_base_alu:.2f} pJ")
    print(f"  • MAC operation:                 INT8: 0.4 pJ, BF16: 0.6 pJ, FP32: 1.2 pJ")
    print(f"  • Systolic array:                128 × 128 PEs (16,384 MACs)")
    print(f"  • Memory hierarchy:              4-stage (DRAM → L2 SRAM → Scratchpad → Accumulator)")
    print(f"  • Data movement engines:         Static systolic dataflow")
    
    # KPU architectural energy model
    print(f"\nKPU (Stillwater T256 @ 30W, 256 tiles @ 1.2 GHz):")
    print(f"  • Base ALU energy (FP32):        {kpu_base_alu:.2f} pJ")

    # Calculate MAC energies from energy_scaling
    kpu_model = comparator.kpu_mapper.resource_model
    if hasattr(kpu_model, 'energy_scaling'):
        int8_energy = kpu_model.energy_per_flop_fp32 * kpu_model.energy_scaling.get(Precision.INT8, 0.125) * 1e12
        bf16_energy = kpu_model.energy_per_flop_fp32 * kpu_model.energy_scaling.get(Precision.BF16, 0.5) * 1e12
        fp32_energy = kpu_model.energy_per_flop_fp32 * kpu_model.energy_scaling.get(Precision.FP32, 1.0) * 1e12
        print(f"  • MAC operation:                 INT8: {int8_energy:.2f} pJ, BF16: {bf16_energy:.2f} pJ, FP32: {fp32_energy:.2f} pJ")

    print(f"  • Token matching:                ~0.6 pJ per token")
    print(f"  • Memory hierarchy:              4-stage (DRAM → L3 → L2 → L1)")
    print(f"  • Data movement engines:         3 specialized (DMA, BlockMover, Streamer)")


    print(f"\n⚠️  BASE ALU ENERGY COMPARISON (FP32, unencumbered):")
    print(f"  • CPU:  {cpu_base_alu:.2f} pJ (HIGHEST - high frequency + complex pipeline)")
    print(f"  • GPU:  {gpu_base_alu:.2f} pJ (CUDA core, no coherence/scheduling)")
    print(f"  • TPU:  {tpu_base_alu:.2f} pJ (LOWEST - static dataflow, no instruction fetch)")
    print(f"  • KPU:  {kpu_base_alu:.2f} pJ (PROGRAMMABLE - domainflow, no instruction fetch)")

    # Add specialized unit comparison
    if hasattr(comparator.gpu_mapper.resource_model, 'architecture_energy_model'):
        arch_model = comparator.gpu_mapper.resource_model.architecture_energy_model
        if arch_model:
            tensor_core_mac = arch_model.tensor_core_mac_energy * 1e12
            print(f"\n⚠️  SPECIALIZED UNITS:")
            print(f"  • GPU Tensor Core: {tensor_core_mac:.2f} pJ per MAC (64 MACs/cycle, {tensor_core_mac*64:.1f} pJ per clock)")
            print(f"    └─ 64 FP16 MACs + 12 FP32 accumulates = massive functional unit")
            print(f"  • Future: Intel AMX (16×16 systolic array) will be added")


def _print_config_comparison(mlp_config, batch_size, cpu_result, gpu_result, tpu_result, kpu_result, comparator, print_arch_selection: List[str]):
    """Print detailed comparison for a single configuration"""

    print(f"\n{'='*80}")
    print(f"Configuration: {mlp_config.name} MLP @ Batch={batch_size}")
    print(f"{'='*80}")
    print(f"Workload: {mlp_config.total_macs:,} MACs ({mlp_config.total_flops:,} FLOPs) per inference")
    print(f"Memory: {mlp_config.total_bytes_per_inference():,} bytes per inference")

    # Add workload characteristics
    _print_workload_characteristics(mlp_config, batch_size)

    # Add hardware energy configuration
    _print_hardware_energy_config(gpu_result, kpu_result, cpu_result, comparator)

    # Energy breakdown table
    print(f"\n{'─'*100}")
    print(f"ENERGY BREAKDOWN")
    print(f"{'─'*100}")

    # Header
    print(f"{'Component':<35} {'CPU (μJ)':<15} {'GPU (μJ)':<15} {'TPU (μJ)':<15} {'KPU (μJ)':<15}")
    print(f"{'-'*100}")

    # Base Compute Energy (unencumbered ALU operations from hardware mapper)
    cpu_base_compute = cpu_result.compute_energy_j * 1e6
    gpu_base_compute = gpu_result.compute_energy_j * 1e6
    tpu_base_compute = tpu_result.compute_energy_j * 1e6
    kpu_base_compute = kpu_result.compute_energy_j * 1e6

    # Control Overhead (resource contention management: instruction fetch, coherence, token routing, etc.)
    # For CPU/GPU/KPU: includes both compute overhead (register file, etc.) AND control overhead
    # For TPU: only control overhead (since compute_overhead is negative efficiency savings)
    cpu_control = (cpu_result.architectural_compute_overhead_j + cpu_result.architectural_control_overhead_j) * 1e6
    gpu_control = (gpu_result.architectural_compute_overhead_j + gpu_result.architectural_control_overhead_j) * 1e6
    tpu_control = tpu_result.architectural_control_overhead_j * 1e6  # TPU control overhead only (positive)
    kpu_control = (kpu_result.architectural_compute_overhead_j + kpu_result.architectural_control_overhead_j) * 1e6

    # Memory Energy (data movement)
    cpu_memory = cpu_result.memory_energy_j * 1e6
    gpu_memory = gpu_result.memory_energy_j * 1e6
    tpu_memory = tpu_result.memory_energy_j * 1e6
    kpu_memory = kpu_result.memory_energy_j * 1e6

    # Calculate and store idle/leakage energy BEFORE printing the table
    # (This is also done in the hierarchical breakdown functions, but we need it here too)

    # CPU idle energy
    cpu_dynamic_total = cpu_base_compute + cpu_control + cpu_memory
    cpu_idle = cpu_result.total_energy_j * 1e6 - cpu_dynamic_total
    cpu_result.static_energy_j = cpu_idle * 1e-6

    # GPU idle energy
    gpu_dynamic_total = gpu_base_compute + gpu_control + gpu_memory
    gpu_idle = gpu_result.total_energy_j * 1e6 - gpu_dynamic_total
    gpu_result.static_energy_j = gpu_idle * 1e-6

    # TPU idle energy
    tpu_dynamic_total = tpu_base_compute + tpu_control + tpu_memory
    tpu_idle = tpu_result.total_energy_j * 1e6 - tpu_dynamic_total
    tpu_result.static_energy_j = tpu_idle * 1e-6

    # KPU idle energy
    kpu_dynamic_total = kpu_base_compute + kpu_control + kpu_memory
    kpu_idle = kpu_result.total_energy_j * 1e6 - kpu_dynamic_total
    kpu_result.static_energy_j = kpu_idle * 1e-6

    # Static/Idle Energy (leakage) - now properly calculated above
    cpu_static = cpu_result.static_energy_j * 1e6
    gpu_static = gpu_result.static_energy_j * 1e6
    tpu_static = tpu_result.static_energy_j * 1e6
    kpu_static = kpu_result.static_energy_j * 1e6

    print(f"{'Base Compute Energy':<35} {cpu_base_compute:<15.3f} {gpu_base_compute:<15.3f} {tpu_base_compute:<15.3f} {kpu_base_compute:<15.3f}")
    print(f"{'Control Overhead':<35} {cpu_control:<15.3f} {gpu_control:<15.3f} {tpu_control:<15.3f} {kpu_control:<15.3f}")
    print(f"{'Memory Energy':<35} {cpu_memory:<15.3f} {gpu_memory:<15.3f} {tpu_memory:<15.3f} {kpu_memory:<15.3f}")
    print(f"{'Static/Idle Energy':<35} {cpu_static:<15.3f} {gpu_static:<15.3f} {tpu_static:<15.3f} {kpu_static:<15.3f}")

    # Totals
    print(f"{'-'*100}")
    cpu_total = cpu_result.total_energy_j * 1e6
    gpu_total = gpu_result.total_energy_j * 1e6
    tpu_total = tpu_result.total_energy_j * 1e6
    kpu_total = kpu_result.total_energy_j * 1e6


    print(f"{'TOTAL ENERGY per inference':<35}  {cpu_total:<15.3f} {gpu_total:<15.3f} {tpu_total:<15.3f} {kpu_total:<15.3f}")

    # Latency and throughput
    cpu_latency_us = cpu_result.latency_s * 1e6
    gpu_latency_us = gpu_result.latency_s * 1e6
    tpu_latency_us = tpu_result.latency_s * 1e6
    kpu_latency_us = kpu_result.latency_s * 1e6

    print(f"{'Latency per inference (μs)':<35} {cpu_latency_us:<15.2f} {gpu_latency_us:<15.2f} {tpu_latency_us:<15.2f} {kpu_latency_us:<15.2f}")

    # Energy efficiency (inferences per Joule)
    # inferences/J = 1 / (energy_per_inference in J) = 1e6 / (energy_per_inference in μJ)
    cpu_infer_per_joule = 1e6 / cpu_total if cpu_total > 0 else 0
    gpu_infer_per_joule = 1e6 / gpu_total if gpu_total > 0 else 0
    tpu_infer_per_joule = 1e6 / tpu_total if tpu_total > 0 else 0
    kpu_infer_per_joule = 1e6 / kpu_total if kpu_total > 0 else 0

    print(f"{'Energy Efficiency (infer/J)':<35} {cpu_infer_per_joule:<15,.0f} {gpu_infer_per_joule:<15,.0f} {tpu_infer_per_joule:<15,.0f} {kpu_infer_per_joule:<15,.0f}")

    # Energy per MAC
    print(f"{'Energy per MAC (pJ)':<35} {cpu_result.energy_per_mac_pj:<15.2f} {gpu_result.energy_per_mac_pj:<15.2f} {tpu_result.energy_per_mac_pj:<15.2f} {kpu_result.energy_per_mac_pj:<15.2f}")

    # Efficiency metrics
    print(f"\n{'─'*100}")
    print(f"EFFICIENCY METRICS (Control Overhead as % of Dynamic Energy)")
    print(f"{'─'*100}")

    # Calculate dynamic energy (excluding static/idle)
    cpu_dynamic = cpu_base_compute + cpu_control + cpu_memory
    gpu_dynamic = gpu_base_compute + gpu_control + gpu_memory
    tpu_dynamic = tpu_base_compute + tpu_control + tpu_memory
    kpu_dynamic = kpu_base_compute + kpu_control + kpu_memory

    cpu_control_pct = (cpu_control / cpu_dynamic * 100) if cpu_dynamic > 0 else 0
    gpu_control_pct = (gpu_control / gpu_dynamic * 100) if gpu_dynamic > 0 else 0
    tpu_control_pct = (tpu_control / tpu_dynamic * 100) if tpu_dynamic > 0 else 0
    kpu_control_pct = (kpu_control / kpu_dynamic * 100) if kpu_dynamic > 0 else 0

    print(f"{'Metric':<35} {'CPU':<15} {'GPU':<15} {'TPU':<15} {'KPU':<15}")
    print(f"{'-'*100}")
    print(f"{'Control Overhead (%)':<35} {cpu_control_pct:<15.1f} {gpu_control_pct:<15.1f} {tpu_control_pct:<15.1f} {kpu_control_pct:<15.1f}")
    print(f"{'Compute Efficiency (%)':<35} {(cpu_base_compute/cpu_dynamic*100):<15.1f} {(gpu_base_compute/gpu_dynamic*100):<15.1f} {(tpu_base_compute/tpu_dynamic*100):<15.1f} {(kpu_base_compute/kpu_dynamic*100):<15.1f}")
    print(f"{'Memory Overhead (%)':<35} {(cpu_memory/cpu_dynamic*100):<15.1f} {(gpu_memory/gpu_dynamic*100):<15.1f} {(tpu_memory/tpu_dynamic*100):<15.1f} {(kpu_memory/kpu_dynamic*100):<15.1f}")

    # Architecture-specific hierarchical breakdowns
    if 'cpu' in print_arch_selection:
        _print_cpu_hierarchical_breakdown(cpu_result)
    if 'gpu' in print_arch_selection:
        _print_gpu_hierarchical_breakdown(gpu_result)
    if 'tpu' in print_arch_selection:
        _print_tpu_hierarchical_breakdown(tpu_result)
    if 'kpu' in print_arch_selection:
        _print_kpu_hierarchical_breakdown(kpu_result)

    # Performance metrics
    print(f"\n{'─'*100}")
    print(f"PERFORMANCE METRICS")
    print(f"{'─'*100}")

    print(f"{'Metric':<35} {'CPU':<15} {'GPU':<15} {'TPU':<15} {'KPU':<15}")
    print(f"{'-'*100}")

    print(f"{'Latency per inference (μs)':<35} {cpu_result.latency_s*1e6:<15.2f} {gpu_result.latency_s*1e6:<15.2f} {tpu_result.latency_s*1e6:<15.2f} {kpu_result.latency_s*1e6:<15.2f}")
    print(f"{'Throughput (infer/sec)':<35} {cpu_result.throughput_inferences_per_sec:<15,.0f} {gpu_result.throughput_inferences_per_sec:<15,.0f} {tpu_result.throughput_inferences_per_sec:<15,.0f} {kpu_result.throughput_inferences_per_sec:<15,.0f}")
    print(f"{'Average Power (W)':<35} {cpu_result.power_w:<15.2f} {gpu_result.power_w:<15.2f} {tpu_result.power_w:<15.2f} {kpu_result.power_w:<15.2f}")

    # Hardware utilization
    print(f"\n{'─'*100}")
    print(f"HARDWARE UTILIZATION")
    print(f"{'─'*100}")

    print(f"{'Compute Units (total)':<35} {cpu_result.compute_units_total:<15} {gpu_result.compute_units_total:<15} {tpu_result.compute_units_total:<15} {kpu_result.compute_units_total:<15}")
    print(f"{'Compute Units (allocated)':<35} {cpu_result.compute_units_allocated:<15} {gpu_result.compute_units_allocated:<15} {tpu_result.compute_units_allocated:<15} {kpu_result.compute_units_allocated:<15}")
    print(f"{'Peak Utilization (%)':<35} {cpu_result.peak_utilization*100:<15.1f} {gpu_result.peak_utilization*100:<15.1f} {tpu_result.peak_utilization*100:<15.1f} {kpu_result.peak_utilization*100:<15.1f}")

    # Winner analysis
    print(f"\n{'─'*80}")
    print(f"ANALYSIS")
    print(f"{'─'*80}")

    # Find most efficient
    min_energy = min(cpu_total, gpu_total, tpu_total, kpu_total)

    if kpu_total == min_energy:
        winner = "KPU"
        gpu_ratio = gpu_total / kpu_total
        cpu_ratio = cpu_total / kpu_total
        print(f"🏆 WINNER: KPU (Domain-Flow)")
        print(f"   KPU is {gpu_ratio:.2f}× more energy efficient than GPU")
        print(f"   KPU is {cpu_ratio:.2f}× more energy efficient than CPU")
        print(f"\n   WHY? Domain-flow spatial dataflow eliminates:")
        print(f"   • GPU architectural overhead ({gpu_result.get_total_architectural_overhead()*1e6:.1f} μJ)")
        print(f"   • CPU architectural overhead ({cpu_result.get_total_architectural_overhead()*1e6:.1f} μJ)")
        print(f"   • Token-based execution requires only {kpu_result.get_total_architectural_overhead()*1e6:.1f} μJ overhead")
    elif tpu_total == min_energy:
        winner = "TPU"
        gpu_ratio = gpu_total / tpu_total
        cpu_ratio = cpu_total / tpu_total
        print(f"🏆 WINNER: TPU (Systolic Array)")
        print(f"   TPU is {gpu_ratio:.2f}× more energy efficient than GPU")
        print(f"   TPU is {cpu_ratio:.2f}× more energy efficient than CPU")
        print(f"\n   WHY? Systolic array dataflow with minimal overhead:")
        print(f"   • No instruction fetch/decode (dataflow, not stored-program)")
        print(f"   • No coherence machinery (static dataflow, no cache coherence)")
        print(f"   • Extremely high PE utilization for MLPs")
    elif gpu_total == min_energy:
        winner = "GPU"
        kpu_ratio = kpu_total / gpu_total
        cpu_ratio = cpu_total / gpu_total
        print(f"🏆 WINNER: GPU (Data-Parallel)")
        print(f"   GPU is {kpu_ratio:.2f}× more energy efficient than KPU")
        print(f"   GPU is {cpu_ratio:.2f}× more energy efficient than CPU")
        print(f"\n   WHY? At large batch sizes, GPU amortizes coherence overhead:")
        print(f"   • Architectural overhead per sample: {gpu_result.get_total_architectural_overhead()*1e6/batch_size:.1f} μJ")
        print(f"   • Massive parallelism wins when overhead is amortized")
    else:
        winner = "CPU"
        gpu_ratio = gpu_total / cpu_total
        kpu_ratio = kpu_total / cpu_total
        print(f"🏆 WINNER: CPU (Stored-Program)")
        print(f"   CPU is {gpu_ratio:.2f}× more energy efficient than GPU")
        print(f"   CPU is {kpu_ratio:.2f}× more energy efficient than KPU")

    # Throughput analysis
    max_throughput = max(cpu_result.throughput_inferences_per_sec,
                        gpu_result.throughput_inferences_per_sec,
                        tpu_result.throughput_inferences_per_sec,
                        kpu_result.throughput_inferences_per_sec)

    if gpu_result.throughput_inferences_per_sec == max_throughput:
        throughput_winner = "GPU"
    elif tpu_result.throughput_inferences_per_sec == max_throughput:
        throughput_winner = "TPU"        
    elif kpu_result.throughput_inferences_per_sec == max_throughput:
        throughput_winner = "KPU"
    else:
        throughput_winner = "CPU"

    print(f"\n   THROUGHPUT WINNER: {throughput_winner} ({max_throughput:,.0f} infer/sec)")

    if winner != throughput_winner:
        print(f"   Note: Energy efficiency vs throughput trade-off!")
        print(f"         {winner} wins on energy, {throughput_winner} wins on throughput")

    # Print note about detailed breakdowns if none were requested
    if not print_arch_selection:
        print(f"\n{'─'*80}")
        print(f"NOTE: Detailed architecture energy breakdowns not shown.")
        print(f"      To see detailed breakdowns, use:")
        print(f"      --print-arch cpu gpu tpu kpu   (for all architectures)")
        print(f"      --print-arch cpu gpu            (for specific architectures)")
        print(f"{'─'*80}")


def _save_results(results: Dict[str, List[ArchitecturalEnergyBreakdown]], output_path: str):
    """Save results to JSON or CSV"""
    ext = Path(output_path).suffix.lower()

    if ext == '.json':
        # Convert to JSON-serializable format
        json_data = {
            'cpu': [b.to_dict() for b in results['cpu']],
            'gpu': [b.to_dict() for b in results['gpu']],
            'tpu': [b.to_dict() for b in results['tpu']],
            'kpu': [b.to_dict() for b in results['kpu']],
        }

        with open(output_path, 'w') as f:
            json.dump(json_data, f, indent=2)

    elif ext == '.csv':
        import csv

        # Flatten all results
        rows = []
        for arch_name, breakdowns in results.items():
            for bd in breakdowns:
                row = {
                    'architecture': bd.architecture,
                    'architecture_class': bd.architecture_class,
                    'hardware': bd.hardware_name,
                    'mlp_config': bd.mlp_config.name,
                    'batch_size': bd.batch_size,
                    'total_energy_j': bd.total_energy_j,
                    'latency_s': bd.latency_s,
                    'energy_per_inference_j': bd.energy_per_inference_j,
                    'energy_per_mac_pj': bd.energy_per_mac_pj,
                    'throughput_infer_per_sec': bd.throughput_inferences_per_sec,
                    'power_w': bd.power_w,
                    'compute_units_used': bd.compute_units_allocated,
                    'utilization': bd.peak_utilization,
                }
                rows.append(row)

        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    else:
        print(f"ERROR: Unsupported output format: {ext}")
        print(f"Supported: .json, .csv")


if __name__ == '__main__':
    main()
