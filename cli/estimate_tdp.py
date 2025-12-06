#!/usr/bin/env python3
"""
TDP Power Estimation Tool

Estimates Thermal Design Power (TDP) given:
- Precision (FP32, FP16, INT8, etc.)
- Number of ALUs (compute units)
- Process technology (3nm, 5nm, 7nm, etc.)
- Circuit approach (x86_performance, tensor_core, systolic_mac, domain_flow, etc.)

Supports sweeping TDP as a function of ALU count and generating matplotlib
visualizations comparing process technologies and circuit approaches.

Physics basis:
    TDP = Power = Energy/time = energy_per_op x ops_per_second

    Where:
    - energy_per_op = base_energy_pj(process_node) x circuit_multiplier x precision_scale
    - ops_per_second = num_ALUs x frequency_hz x ops_per_cycle

    And energy_per_op is independent of frequency (Energy = C x V^2 per switch)

Usage:
    # Single estimate
    ./cli/estimate_tdp.py --alus 16384 --process 4 --circuit tensor_core --precision FP16

    # Sweep ALU count
    ./cli/estimate_tdp.py --sweep-alus 1024 65536 --process 4 --circuit tensor_core

    # Compare process technologies
    ./cli/estimate_tdp.py --alus 16384 --compare-process --circuit tensor_core --plot

    # Compare circuit approaches
    ./cli/estimate_tdp.py --alus 16384 --process 4 --compare-circuits --plot

    # Full sweep with plot
    ./cli/estimate_tdp.py --sweep-alus 1024 65536 --compare-process --compare-circuits --plot
"""

import argparse
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from graphs.hardware.technology_profile import (
    PROCESS_NODE_BASE_ENERGY_PJ,
    CIRCUIT_TYPE_MULTIPLIER,
    get_process_base_energy_pj,
    TechnologyProfile,
    ARCH_COMPARISON_8NM_X86,
)
from graphs.hardware.operand_fetch import (
    CPUOperandFetchModel,
    GPUOperandFetchModel,
    TPUOperandFetchModel,
    KPUOperandFetchModel,
)
from graphs.hardware.soc_infrastructure import (
    SoCInfrastructureModel,
    InterconnectTopology,
    InfrastructurePowerBreakdown,
    estimate_infrastructure_power,
    get_topology_from_string,
    SRAM_BYTES_PER_ALU,
    COMPUTE_GRANULARITY_PRESETS,
    get_compute_granularity,
)


# =============================================================================
# Precision Scaling Factors
# =============================================================================
# These represent the relative energy cost of different precisions.
# FP32 is baseline (1.0). Lower precision = less energy per op.

PRECISION_ENERGY_SCALE = {
    'FP64': 2.0,    # Double precision: 2x FP32 energy
    'FP32': 1.0,    # Single precision: baseline
    'TF32': 0.6,    # TensorFloat-32: reduced mantissa
    'BF16': 0.5,    # BFloat16: reduced mantissa, same exponent
    'FP16': 0.5,    # Half precision: 16-bit
    'FP8':  0.25,   # 8-bit float (E4M3 or E5M2)
    'INT8': 0.25,   # 8-bit integer
    'INT4': 0.125,  # 4-bit integer
    'INT2': 0.0625, # 2-bit integer (binary/ternary)
}

# Ops per cycle for different precisions (relative to FP32 baseline of 2 ops/cycle for FMA)
# This captures how tensor cores can do more ops per cycle at lower precision
PRECISION_OPS_PER_CYCLE = {
    'FP64': 1,      # FP64: 1 FMA/cycle typical
    'FP32': 2,      # FP32: 2 ops/cycle (1 FMA = 2 ops)
    'TF32': 4,      # TF32: 2x FP32 throughput on Tensor Cores
    'BF16': 4,      # BF16: 2x FP32 throughput
    'FP16': 4,      # FP16: 2x FP32 throughput
    'FP8':  8,      # FP8: 4x FP32 throughput
    'INT8': 8,      # INT8: 4x FP32 throughput
    'INT4': 16,     # INT4: 8x FP32 throughput
    'INT2': 32,     # INT2: 16x FP32 throughput
}

# Default frequency by circuit type (GHz)
DEFAULT_FREQUENCY_GHZ = {
    'x86_performance': 4.5,
    'x86_efficiency': 2.5,
    'arm_performance': 3.5,
    'arm_efficiency': 2.0,
    'standard_cell': 2.0,
    'cuda_core': 1.5,
    'simd_packed': 3.0,
    'tensor_core': 1.5,
    'systolic_mac': 1.0,
    'domain_flow': 1.0,
}


# =============================================================================
# TDP Estimation Model
# =============================================================================

@dataclass
class TDPEstimate:
    """Result of a TDP estimation."""
    num_alus: int
    process_node_nm: int
    circuit_type: str
    precision: str
    frequency_ghz: float

    # Derived values
    base_energy_pj: float       # Base FP32 energy for this process node
    circuit_multiplier: float   # Circuit type energy multiplier
    precision_scale: float      # Precision energy scaling
    energy_per_op_pj: float     # Final energy per operation (ALU only)

    # NEW: Operand fetch energy breakdown
    pure_alu_energy_pj: float = 0.0         # Pure ALU circuit energy
    operand_fetch_energy_pj: float = 0.0    # Operand fetch (reg-to-ALU) energy
    operand_reuse_factor: float = 1.0       # Spatial reuse factor (1.0 for CPU/GPU, >1 for TPU/KPU)
    total_energy_per_op_pj: float = 0.0     # ALU + operand fetch energy

    ops_per_cycle: int = 1                  # Operations per cycle at this precision
    ops_per_second: float = 0.0             # Total ops/sec (TOPS)

    tdp_watts: float = 0.0                  # Estimated TDP

    # Efficiency metrics
    tops: float = 0.0                       # Tera-ops per second
    tops_per_watt: float = 0.0              # TOPS/W efficiency

    # Derived property
    @property
    def alu_fetch_ratio(self) -> float:
        """Ratio of ALU energy to operand fetch energy (>1 = ALU dominated = efficient)."""
        if self.operand_fetch_energy_pj > 0:
            return self.pure_alu_energy_pj / self.operand_fetch_energy_pj
        return float('inf')


def estimate_tdp(
    num_alus: int,
    process_node_nm: int,
    circuit_type: str,
    precision: str = 'FP32',
    frequency_ghz: Optional[float] = None,
    include_operand_fetch: bool = True,
) -> TDPEstimate:
    """
    Estimate TDP for a given configuration.

    Args:
        num_alus: Number of ALUs (compute units)
        process_node_nm: Process technology in nanometers
        circuit_type: Circuit design approach (from CIRCUIT_TYPE_MULTIPLIER)
        precision: Compute precision (FP32, FP16, INT8, etc.)
        frequency_ghz: Clock frequency in GHz (defaults based on circuit type)
        include_operand_fetch: Include operand fetch energy in TDP (default: True)

    Returns:
        TDPEstimate with all derived values including operand fetch breakdown
    """
    # Get base energy for process node
    base_energy_pj = get_process_base_energy_pj(process_node_nm)

    # Get circuit multiplier
    if circuit_type not in CIRCUIT_TYPE_MULTIPLIER:
        raise ValueError(f"Unknown circuit type: {circuit_type}. "
                        f"Available: {list(CIRCUIT_TYPE_MULTIPLIER.keys())}")
    circuit_multiplier = CIRCUIT_TYPE_MULTIPLIER[circuit_type]

    # Get precision scaling
    precision = precision.upper()
    if precision not in PRECISION_ENERGY_SCALE:
        raise ValueError(f"Unknown precision: {precision}. "
                        f"Available: {list(PRECISION_ENERGY_SCALE.keys())}")
    precision_scale = PRECISION_ENERGY_SCALE[precision]
    ops_per_cycle = PRECISION_OPS_PER_CYCLE[precision]

    # Calculate pure ALU energy per operation (pJ) - the circuit energy
    pure_alu_energy_pj = base_energy_pj * circuit_multiplier * precision_scale

    # Calculate operand fetch energy based on circuit type
    # Use 8nm comparison profile as baseline (we scale by process node below)
    tech_profile = ARCH_COMPARISON_8NM_X86.cpu_profile
    operand_fetch_energy_pj = 0.0
    operand_reuse_factor = 1.0

    if include_operand_fetch:
        # Map circuit types to operand fetch models
        if circuit_type in ['x86_performance', 'x86_efficiency', 'arm_performance', 'arm_efficiency', 'simd_packed']:
            # CPU-style architectures: no spatial reuse
            model = CPUOperandFetchModel(tech_profile=tech_profile)
            breakdown = model.compute_operand_fetch_energy(num_operations=1)
            operand_fetch_energy_pj = breakdown.energy_per_operation * 1e12
            operand_reuse_factor = 1.0

        elif circuit_type in ['cuda_core', 'tensor_core']:
            # GPU-style architectures: no spatial reuse (per-thread register files)
            model = GPUOperandFetchModel(tech_profile=tech_profile)
            breakdown = model.compute_operand_fetch_energy(num_operations=1)
            operand_fetch_energy_pj = breakdown.energy_per_operation * 1e12
            operand_reuse_factor = 1.0

        elif circuit_type == 'systolic_mac':
            # TPU-style systolic array: massive spatial reuse
            model = TPUOperandFetchModel(tech_profile=tech_profile)
            # For systolic arrays, assume 128x128 tile = 16,384 reuse factor
            breakdown = model.compute_operand_fetch_energy(
                num_operations=16384,
                spatial_reuse_factor=128.0
            )
            operand_fetch_energy_pj = breakdown.energy_per_operation * 1e12
            operand_reuse_factor = breakdown.operand_reuse_factor

        elif circuit_type == 'domain_flow':
            # KPU-style domain flow: moderate spatial reuse
            model = KPUOperandFetchModel(tech_profile=tech_profile)
            # For domain flow, assume 64x reuse (configurable)
            breakdown = model.compute_operand_fetch_energy(
                num_operations=256,
                spatial_reuse_factor=64.0
            )
            operand_fetch_energy_pj = breakdown.energy_per_operation * 1e12
            operand_reuse_factor = breakdown.operand_reuse_factor

        else:
            # Other circuit types: assume CPU-like operand fetch
            model = CPUOperandFetchModel(tech_profile=tech_profile)
            breakdown = model.compute_operand_fetch_energy(num_operations=1)
            operand_fetch_energy_pj = breakdown.energy_per_operation * 1e12
            operand_reuse_factor = 1.0

        # Scale operand fetch energy by process node (relative to 8nm baseline)
        process_scale = process_node_nm / 8.0
        operand_fetch_energy_pj *= process_scale

    # Total energy = ALU + operand fetch
    total_energy_per_op_pj = pure_alu_energy_pj + operand_fetch_energy_pj

    # Get frequency
    if frequency_ghz is None:
        frequency_ghz = DEFAULT_FREQUENCY_GHZ.get(circuit_type, 1.5)

    # Calculate ops per second
    # Each ALU does ops_per_cycle operations per clock cycle
    ops_per_second = num_alus * ops_per_cycle * frequency_ghz * 1e9  # ops/sec
    tops = ops_per_second / 1e12  # TOPS

    # Calculate TDP using TOTAL energy (ALU + operand fetch)
    # Power = Energy/time = energy_per_op * ops_per_second
    energy_per_op_j = total_energy_per_op_pj * 1e-12  # Convert pJ to J
    tdp_watts = energy_per_op_j * ops_per_second

    # Calculate efficiency
    tops_per_watt = tops / tdp_watts if tdp_watts > 0 else 0

    return TDPEstimate(
        num_alus=num_alus,
        process_node_nm=process_node_nm,
        circuit_type=circuit_type,
        precision=precision,
        frequency_ghz=frequency_ghz,
        base_energy_pj=base_energy_pj,
        circuit_multiplier=circuit_multiplier,
        precision_scale=precision_scale,
        energy_per_op_pj=pure_alu_energy_pj,  # For backward compat, this is ALU energy
        pure_alu_energy_pj=pure_alu_energy_pj,
        operand_fetch_energy_pj=operand_fetch_energy_pj,
        operand_reuse_factor=operand_reuse_factor,
        total_energy_per_op_pj=total_energy_per_op_pj,
        ops_per_cycle=ops_per_cycle,
        ops_per_second=ops_per_second,
        tdp_watts=tdp_watts,
        tops=tops,
        tops_per_watt=tops_per_watt,
    )


# =============================================================================
# Sweep Functions
# =============================================================================

def sweep_alus(
    alu_range: Tuple[int, int],
    process_node_nm: int,
    circuit_type: str,
    precision: str = 'FP32',
    frequency_ghz: Optional[float] = None,
    num_points: int = 20,
) -> List[TDPEstimate]:
    """
    Sweep TDP over a range of ALU counts.

    Args:
        alu_range: (min_alus, max_alus)
        process_node_nm: Process technology
        circuit_type: Circuit design approach
        precision: Compute precision
        frequency_ghz: Clock frequency
        num_points: Number of sweep points

    Returns:
        List of TDPEstimate for each ALU count
    """
    min_alus, max_alus = alu_range

    # Generate logarithmic sweep points
    import math
    log_min = math.log2(min_alus)
    log_max = math.log2(max_alus)
    step = (log_max - log_min) / (num_points - 1)

    results = []
    for i in range(num_points):
        num_alus = int(2 ** (log_min + i * step))
        # Round to nice numbers
        num_alus = round(num_alus / 64) * 64
        if num_alus < min_alus:
            num_alus = min_alus
        if num_alus > max_alus:
            num_alus = max_alus

        estimate = estimate_tdp(
            num_alus=num_alus,
            process_node_nm=process_node_nm,
            circuit_type=circuit_type,
            precision=precision,
            frequency_ghz=frequency_ghz,
        )
        results.append(estimate)

    return results


def compare_process_nodes(
    num_alus: int,
    circuit_type: str,
    precision: str = 'FP32',
    frequency_ghz: Optional[float] = None,
    process_nodes: Optional[List[int]] = None,
) -> Dict[int, TDPEstimate]:
    """
    Compare TDP across different process nodes.

    Args:
        num_alus: Number of ALUs
        circuit_type: Circuit design approach
        precision: Compute precision
        frequency_ghz: Clock frequency
        process_nodes: List of process nodes to compare (default: all)

    Returns:
        Dict mapping process_node -> TDPEstimate
    """
    if process_nodes is None:
        process_nodes = sorted(PROCESS_NODE_BASE_ENERGY_PJ.keys())

    results = {}
    for node in process_nodes:
        estimate = estimate_tdp(
            num_alus=num_alus,
            process_node_nm=node,
            circuit_type=circuit_type,
            precision=precision,
            frequency_ghz=frequency_ghz,
        )
        results[node] = estimate

    return results


def compare_circuit_types(
    num_alus: int,
    process_node_nm: int,
    precision: str = 'FP32',
    circuit_types: Optional[List[str]] = None,
) -> Dict[str, TDPEstimate]:
    """
    Compare TDP across different circuit types.

    Args:
        num_alus: Number of ALUs
        process_node_nm: Process technology
        precision: Compute precision
        circuit_types: List of circuit types to compare (default: all)

    Returns:
        Dict mapping circuit_type -> TDPEstimate
    """
    if circuit_types is None:
        circuit_types = list(CIRCUIT_TYPE_MULTIPLIER.keys())
        # Remove legacy alias
        if 'custom_datacenter' in circuit_types:
            circuit_types.remove('custom_datacenter')

    results = {}
    for ct in circuit_types:
        estimate = estimate_tdp(
            num_alus=num_alus,
            process_node_nm=process_node_nm,
            circuit_type=ct,
            precision=precision,
        )
        results[ct] = estimate

    return results


def compare_precisions(
    num_alus: int,
    process_node_nm: int,
    circuit_type: str,
    precisions: Optional[List[str]] = None,
) -> Dict[str, TDPEstimate]:
    """
    Compare TDP across different precisions.

    Args:
        num_alus: Number of ALUs
        process_node_nm: Process technology
        circuit_type: Circuit design approach
        precisions: List of precisions to compare (default: all)

    Returns:
        Dict mapping precision -> TDPEstimate
    """
    if precisions is None:
        precisions = list(PRECISION_ENERGY_SCALE.keys())

    results = {}
    for prec in precisions:
        estimate = estimate_tdp(
            num_alus=num_alus,
            process_node_nm=process_node_nm,
            circuit_type=circuit_type,
            precision=prec,
        )
        results[prec] = estimate

    return results


# =============================================================================
# Realistic TDP with SoC Infrastructure
# =============================================================================

def estimate_realistic_tdp(
    num_alus: int,
    process_node_nm: int,
    circuit_type: str,
    precision: str = 'FP32',
    frequency_ghz: Optional[float] = None,
    topology: InterconnectTopology = InterconnectTopology.MESH_2D,
    compute_granularity: Optional[str] = None,
) -> InfrastructurePowerBreakdown:
    """
    Estimate TDP with realistic SoC infrastructure overhead.

    Unlike the linear model, this accounts for:
    1. On-chip SRAM to keep GEMM compute-bound
    2. Interconnect power (scales with clusters, not individual ALUs)
    3. Control overhead (schedulers, DMAs, coherence)
    4. Idle/leakage power (~50% of TDP envelope)

    Args:
        num_alus: Number of ALUs
        process_node_nm: Process technology in nanometers
        circuit_type: Circuit design approach
        precision: Compute precision
        frequency_ghz: Clock frequency (defaults based on circuit type)
        topology: Interconnect topology (mesh, htree, crossbar, clos)
        compute_granularity: Preset for compute building blocks
            Options: nvidia_sm, nvidia_tc, tpu_mxu, cpu_core, kpu_tile

    Returns:
        InfrastructurePowerBreakdown with full component analysis
    """
    if frequency_ghz is None:
        frequency_ghz = DEFAULT_FREQUENCY_GHZ.get(circuit_type, 1.5)

    return estimate_infrastructure_power(
        num_alus=num_alus,
        process_node_nm=process_node_nm,
        circuit_type=circuit_type,
        topology=topology,
        frequency_ghz=frequency_ghz,
        precision=precision,
        compute_granularity=compute_granularity,
    )


def sweep_realistic_tdp(
    alu_range: Tuple[int, int],
    process_node_nm: int,
    circuit_type: str,
    precision: str = 'FP32',
    frequency_ghz: Optional[float] = None,
    topology: InterconnectTopology = InterconnectTopology.MESH_2D,
    num_points: int = 20,
    compute_granularity: Optional[str] = None,
) -> List[InfrastructurePowerBreakdown]:
    """
    Sweep realistic TDP over a range of ALU counts.

    Args:
        alu_range: (min_alus, max_alus)
        process_node_nm: Process technology
        circuit_type: Circuit design approach
        precision: Compute precision
        frequency_ghz: Clock frequency
        topology: Interconnect topology
        num_points: Number of sweep points
        compute_granularity: Preset for compute building blocks

    Returns:
        List of InfrastructurePowerBreakdown for each ALU count
    """
    import math

    min_alus, max_alus = alu_range

    # Generate logarithmic sweep points
    log_min = math.log2(min_alus)
    log_max = math.log2(max_alus)
    step = (log_max - log_min) / (num_points - 1)

    results = []
    for i in range(num_points):
        num_alus = int(2 ** (log_min + i * step))
        # Round to nice numbers
        num_alus = round(num_alus / 64) * 64
        if num_alus < min_alus:
            num_alus = min_alus
        if num_alus > max_alus:
            num_alus = max_alus

        breakdown = estimate_realistic_tdp(
            num_alus=num_alus,
            process_node_nm=process_node_nm,
            circuit_type=circuit_type,
            precision=precision,
            frequency_ghz=frequency_ghz,
            topology=topology,
            compute_granularity=compute_granularity,
        )
        results.append(breakdown)

    return results


def compare_topologies(
    num_alus: int,
    process_node_nm: int,
    circuit_type: str,
    precision: str = 'FP32',
    frequency_ghz: Optional[float] = None,
    compute_granularity: Optional[str] = None,
) -> Dict[str, InfrastructurePowerBreakdown]:
    """
    Compare TDP across different interconnect topologies.

    Args:
        num_alus: Number of ALUs
        process_node_nm: Process technology
        circuit_type: Circuit design approach
        precision: Compute precision
        frequency_ghz: Clock frequency
        compute_granularity: Preset for compute building blocks

    Returns:
        Dict mapping topology name -> InfrastructurePowerBreakdown
    """
    results = {}
    for topo in InterconnectTopology:
        breakdown = estimate_realistic_tdp(
            num_alus=num_alus,
            process_node_nm=process_node_nm,
            circuit_type=circuit_type,
            precision=precision,
            frequency_ghz=frequency_ghz,
            topology=topo,
            compute_granularity=compute_granularity,
        )
        results[topo.value] = breakdown

    return results


# =============================================================================
# Plotting Functions
# =============================================================================

def plot_tdp_sweep(
    results: List[TDPEstimate],
    title: str = "TDP vs ALU Count",
    output_file: Optional[str] = None,
):
    """Plot TDP sweep results."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Install with: pip install matplotlib")
        return

    alus = [r.num_alus for r in results]
    tdps = [r.tdp_watts for r in results]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(alus, tdps, 'b-o', linewidth=2, markersize=6)

    ax.set_xlabel('Number of ALUs', fontsize=12)
    ax.set_ylabel('TDP (Watts)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xscale('log', base=2)
    ax.grid(True, alpha=0.3)

    # Add annotations for first and last points
    ax.annotate(f'{tdps[0]:.1f}W', (alus[0], tdps[0]), textcoords="offset points",
                xytext=(10, 5), fontsize=10)
    ax.annotate(f'{tdps[-1]:.1f}W', (alus[-1], tdps[-1]), textcoords="offset points",
                xytext=(-40, 5), fontsize=10)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150)
        print(f"Saved plot to {output_file}")
    else:
        plt.show()


def plot_process_comparison(
    alu_range: Tuple[int, int],
    circuit_type: str,
    precision: str = 'FP32',
    process_nodes: Optional[List[int]] = None,
    output_file: Optional[str] = None,
):
    """Plot TDP vs ALU count for different process nodes."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not installed. Install with: pip install matplotlib")
        return

    if process_nodes is None:
        process_nodes = [3, 4, 5, 7, 14, 28]

    fig, ax = plt.subplots(figsize=(12, 7))

    colors = plt.cm.viridis(np.linspace(0, 0.9, len(process_nodes)))

    for node, color in zip(process_nodes, colors):
        results = sweep_alus(alu_range, node, circuit_type, precision)
        alus = [r.num_alus for r in results]
        tdps = [r.tdp_watts for r in results]
        ax.plot(alus, tdps, '-o', color=color, linewidth=2, markersize=5,
                label=f'{node}nm')

    ax.set_xlabel('Number of ALUs', fontsize=12)
    ax.set_ylabel('TDP (Watts)', fontsize=12)
    ax.set_title(f'TDP vs ALU Count by Process Node\n({circuit_type}, {precision})', fontsize=14)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(title='Process Node', loc='upper left')

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150)
        print(f"Saved plot to {output_file}")
    else:
        plt.show()


def plot_circuit_comparison(
    alu_range: Tuple[int, int],
    process_node_nm: int,
    precision: str = 'FP32',
    circuit_types: Optional[List[str]] = None,
    output_file: Optional[str] = None,
):
    """Plot TDP vs ALU count for different circuit types."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not installed. Install with: pip install matplotlib")
        return

    if circuit_types is None:
        circuit_types = [
            'x86_performance', 'arm_efficiency', 'cuda_core',
            'tensor_core', 'systolic_mac', 'domain_flow'
        ]

    fig, ax = plt.subplots(figsize=(12, 7))

    colors = plt.cm.tab10(np.linspace(0, 1, len(circuit_types)))

    for ct, color in zip(circuit_types, colors):
        results = sweep_alus(alu_range, process_node_nm, ct, precision)
        alus = [r.num_alus for r in results]
        tdps = [r.tdp_watts for r in results]
        mult = CIRCUIT_TYPE_MULTIPLIER[ct]
        ax.plot(alus, tdps, '-o', color=color, linewidth=2, markersize=5,
                label=f'{ct} ({mult:.2f}x)')

    ax.set_xlabel('Number of ALUs', fontsize=12)
    ax.set_ylabel('TDP (Watts)', fontsize=12)
    ax.set_title(f'TDP vs ALU Count by Circuit Type\n({process_node_nm}nm, {precision})', fontsize=14)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(title='Circuit Type (multiplier)', loc='upper left', fontsize=9)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150)
        print(f"Saved plot to {output_file}")
    else:
        plt.show()


def plot_precision_comparison(
    alu_range: Tuple[int, int],
    process_node_nm: int,
    circuit_type: str,
    precisions: Optional[List[str]] = None,
    output_file: Optional[str] = None,
):
    """Plot TDP vs ALU count for different precisions."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not installed. Install with: pip install matplotlib")
        return

    if precisions is None:
        precisions = ['FP64', 'FP32', 'FP16', 'INT8', 'INT4']

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(precisions)))

    # Plot 1: TDP vs ALUs
    ax = axes[0]
    for prec, color in zip(precisions, colors):
        results = sweep_alus(alu_range, process_node_nm, circuit_type, prec)
        alus = [r.num_alus for r in results]
        tdps = [r.tdp_watts for r in results]
        energy_scale = PRECISION_ENERGY_SCALE[prec]
        ax.plot(alus, tdps, '-o', color=color, linewidth=2, markersize=5,
                label=f'{prec} ({energy_scale:.2f}x)')

    ax.set_xlabel('Number of ALUs', fontsize=12)
    ax.set_ylabel('TDP (Watts)', fontsize=12)
    ax.set_title(f'TDP vs ALU Count by Precision\n({process_node_nm}nm, {circuit_type})', fontsize=14)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(title='Precision (energy scale)', loc='upper left', fontsize=9)

    # Plot 2: TOPS vs TDP (efficiency frontier)
    ax = axes[1]
    for prec, color in zip(precisions, colors):
        results = sweep_alus(alu_range, process_node_nm, circuit_type, prec)
        tdps = [r.tdp_watts for r in results]
        tops = [r.tops for r in results]
        ax.plot(tdps, tops, '-o', color=color, linewidth=2, markersize=5,
                label=f'{prec}')

    ax.set_xlabel('TDP (Watts)', fontsize=12)
    ax.set_ylabel('TOPS', fontsize=12)
    ax.set_title(f'Performance vs Power by Precision\n({process_node_nm}nm, {circuit_type})', fontsize=14)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(title='Precision', loc='upper left', fontsize=9)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150)
        print(f"Saved plot to {output_file}")
    else:
        plt.show()


def plot_efficiency_comparison(
    num_alus: int,
    precision: str = 'FP32',
    output_file: Optional[str] = None,
):
    """Plot TOPS/W efficiency across process nodes and circuit types."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not installed. Install with: pip install matplotlib")
        return

    process_nodes = [3, 4, 5, 7, 14, 28]
    circuit_types = ['x86_performance', 'tensor_core', 'systolic_mac', 'domain_flow']

    fig, ax = plt.subplots(figsize=(12, 7))

    x = np.arange(len(process_nodes))
    width = 0.2

    colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']

    for i, (ct, color) in enumerate(zip(circuit_types, colors)):
        efficiencies = []
        for node in process_nodes:
            est = estimate_tdp(num_alus, node, ct, precision)
            efficiencies.append(est.tops_per_watt)

        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, efficiencies, width, label=ct, color=color, alpha=0.8)

        # Add value labels on bars
        for bar, eff in zip(bars, efficiencies):
            ax.annotate(f'{eff:.1f}',
                       xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=8, rotation=45)

    ax.set_xlabel('Process Node', fontsize=12)
    ax.set_ylabel('TOPS/W', fontsize=12)
    ax.set_title(f'Energy Efficiency by Process & Circuit\n({num_alus:,} ALUs, {precision})', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{n}nm' for n in process_nodes])
    ax.legend(title='Circuit Type', loc='upper right')
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150)
        print(f"Saved plot to {output_file}")
    else:
        plt.show()


def plot_full_matrix(
    alu_range: Tuple[int, int],
    precision: str = 'FP32',
    output_file: Optional[str] = None,
):
    """Generate a 2x2 matrix of plots showing all comparisons."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not installed. Install with: pip install matplotlib")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    process_nodes = [3, 4, 5, 7, 14, 28]
    circuit_types = ['x86_performance', 'tensor_core', 'systolic_mac', 'domain_flow']

    # Plot 1: TDP vs ALUs for different process nodes (fixed circuit)
    ax = axes[0, 0]
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(process_nodes)))
    for node, color in zip(process_nodes, colors):
        results = sweep_alus(alu_range, node, 'tensor_core', precision)
        alus = [r.num_alus for r in results]
        tdps = [r.tdp_watts for r in results]
        ax.plot(alus, tdps, '-', color=color, linewidth=2, label=f'{node}nm')
    ax.set_xlabel('Number of ALUs')
    ax.set_ylabel('TDP (Watts)')
    ax.set_title(f'TDP by Process Node (tensor_core, {precision})')
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(title='Process', fontsize=8)

    # Plot 2: TDP vs ALUs for different circuit types (fixed process)
    ax = axes[0, 1]
    colors = plt.cm.tab10(np.linspace(0, 0.8, len(circuit_types)))
    for ct, color in zip(circuit_types, colors):
        results = sweep_alus(alu_range, 5, ct, precision)
        alus = [r.num_alus for r in results]
        tdps = [r.tdp_watts for r in results]
        ax.plot(alus, tdps, '-', color=color, linewidth=2, label=ct)
    ax.set_xlabel('Number of ALUs')
    ax.set_ylabel('TDP (Watts)')
    ax.set_title(f'TDP by Circuit Type (5nm, {precision})')
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(title='Circuit', fontsize=8)

    # Plot 3: Energy per op vs process node
    ax = axes[1, 0]
    for ct, color in zip(circuit_types, colors):
        energies = []
        for node in process_nodes:
            est = estimate_tdp(1024, node, ct, precision)
            energies.append(est.energy_per_op_pj)
        ax.plot(process_nodes, energies, '-o', color=color, linewidth=2, label=ct)
    ax.set_xlabel('Process Node (nm)')
    ax.set_ylabel('Energy per Op (pJ)')
    ax.set_title(f'Energy per Operation ({precision})')
    ax.grid(True, alpha=0.3)
    ax.legend(title='Circuit', fontsize=8)

    # Plot 4: TOPS/W efficiency
    ax = axes[1, 1]
    x = np.arange(len(process_nodes))
    width = 0.2
    bar_colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']

    for i, (ct, color) in enumerate(zip(circuit_types, bar_colors)):
        efficiencies = []
        for node in process_nodes:
            est = estimate_tdp(16384, node, ct, precision)
            efficiencies.append(est.tops_per_watt)
        offset = (i - 1.5) * width
        ax.bar(x + offset, efficiencies, width, label=ct, color=color, alpha=0.8)

    ax.set_xlabel('Process Node')
    ax.set_ylabel('TOPS/W')
    ax.set_title(f'Energy Efficiency (16k ALUs, {precision})')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{n}nm' for n in process_nodes])
    ax.legend(title='Circuit', fontsize=8, loc='upper right')
    ax.grid(True, axis='y', alpha=0.3)

    plt.suptitle(f'TDP Power Estimation Analysis ({precision})', fontsize=16, y=1.02)
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {output_file}")
    else:
        plt.show()


def plot_realistic_tdp_breakdown(
    alu_range: Tuple[int, int],
    process_node_nm: int,
    circuit_type: str,
    precision: str = 'FP32',
    topology: InterconnectTopology = InterconnectTopology.MESH_2D,
    output_file: Optional[str] = None,
    compute_granularity: Optional[str] = None,
):
    """
    Plot stacked area chart showing TDP component breakdown.

    Shows how compute, SRAM, interconnect, control, and idle power
    scale differently with ALU count.
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not installed. Install with: pip install matplotlib")
        return

    results = sweep_realistic_tdp(alu_range, process_node_nm, circuit_type,
                                   precision, topology=topology, num_points=20,
                                   compute_granularity=compute_granularity)

    alus = [r.num_alus for r in results]
    compute = [r.compute_power_w for r in results]
    sram = [r.sram_power_w for r in results]
    interconnect = [r.interconnect_power_w for r in results]
    control = [r.control_power_w for r in results]
    idle = [r.idle_power_w for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Stacked area chart
    ax = axes[0]
    ax.stackplot(alus, compute, sram, interconnect, control, idle,
                 labels=['Compute', 'SRAM', 'Interconnect', 'Control', 'Idle'],
                 colors=['#2ecc71', '#3498db', '#e74c3c', '#9b59b6', '#95a5a6'],
                 alpha=0.8)
    ax.set_xlabel('Number of ALUs', fontsize=12)
    ax.set_ylabel('Power (Watts)', fontsize=12)
    ax.set_title(f'TDP Breakdown by Component\n({process_node_nm}nm, {circuit_type}, {topology.value})', fontsize=12)
    ax.set_xscale('log', base=2)
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Plot 2: Overhead multiplier and compute fraction
    ax2 = axes[1]
    overheads = [r.infrastructure_overhead for r in results]
    compute_fracs = [r.compute_fraction * 100 for r in results]

    ax2.plot(alus, overheads, 'b-o', linewidth=2, markersize=5, label='Overhead Multiplier')
    ax2.set_xlabel('Number of ALUs', fontsize=12)
    ax2.set_ylabel('Infrastructure Overhead (x)', fontsize=12, color='b')
    ax2.tick_params(axis='y', labelcolor='b')
    ax2.set_xscale('log', base=2)

    ax3 = ax2.twinx()
    ax3.plot(alus, compute_fracs, 'r-s', linewidth=2, markersize=5, label='Compute Fraction')
    ax3.set_ylabel('Compute Fraction (%)', fontsize=12, color='r')
    ax3.tick_params(axis='y', labelcolor='r')

    ax2.set_title(f'Infrastructure Overhead vs ALU Count\n({process_node_nm}nm, {circuit_type})', fontsize=12)
    ax2.grid(True, alpha=0.3)

    # Combined legend
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax3.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='center right', fontsize=9)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {output_file}")
    else:
        plt.show()


def plot_linear_vs_realistic(
    alu_range: Tuple[int, int],
    process_node_nm: int,
    circuit_type: str,
    precision: str = 'FP32',
    topology: InterconnectTopology = InterconnectTopology.MESH_2D,
    output_file: Optional[str] = None,
    compute_granularity: Optional[str] = None,
):
    """
    Plot comparison of linear vs realistic TDP models.

    Shows how the linear model underestimates TDP at scale.
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not installed. Install with: pip install matplotlib")
        return

    # Get both models' results
    linear_results = sweep_alus(alu_range, process_node_nm, circuit_type, precision, num_points=20)
    realistic_results = sweep_realistic_tdp(alu_range, process_node_nm, circuit_type,
                                             precision, topology=topology, num_points=20,
                                             compute_granularity=compute_granularity)

    alus_linear = [r.num_alus for r in linear_results]
    tdp_linear = [r.tdp_watts for r in linear_results]

    alus_realistic = [r.num_alus for r in realistic_results]
    tdp_realistic = [r.total_tdp_w for r in realistic_results]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Absolute TDP comparison
    ax = axes[0]
    ax.plot(alus_linear, tdp_linear, 'b--', linewidth=2, label='Linear Model', alpha=0.7)
    ax.plot(alus_realistic, tdp_realistic, 'r-', linewidth=2.5, label='Realistic Model')
    ax.fill_between(alus_realistic, tdp_linear, tdp_realistic, alpha=0.2, color='red',
                    label='Infrastructure Overhead')
    ax.set_xlabel('Number of ALUs', fontsize=12)
    ax.set_ylabel('TDP (Watts)', fontsize=12)
    ax.set_title(f'Linear vs Realistic TDP Model\n({process_node_nm}nm, {circuit_type}, {precision})', fontsize=12)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Plot 2: Overhead ratio
    ax = axes[1]
    # Match up ALU counts (they should be similar but let's be safe)
    overhead_ratios = []
    for i, r in enumerate(realistic_results):
        if i < len(linear_results):
            ratio = r.total_tdp_w / linear_results[i].tdp_watts
            overhead_ratios.append(ratio)
    ax.plot(alus_realistic[:len(overhead_ratios)], overhead_ratios, 'g-o', linewidth=2, markersize=6)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Linear baseline')
    ax.set_xlabel('Number of ALUs', fontsize=12)
    ax.set_ylabel('Realistic / Linear TDP Ratio', fontsize=12)
    ax.set_title(f'Infrastructure Overhead Growth\n({topology.value} interconnect)', fontsize=12)
    ax.set_xscale('log', base=2)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=10)

    # Add annotation for key insight
    if overhead_ratios:
        max_overhead = max(overhead_ratios)
        ax.annotate(f'Max: {max_overhead:.1f}x',
                   xy=(alus_realistic[overhead_ratios.index(max_overhead)], max_overhead),
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=10, fontweight='bold',
                   arrowprops=dict(arrowstyle='->', color='green'))

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {output_file}")
    else:
        plt.show()


def plot_topology_comparison(
    alu_range: Tuple[int, int],
    process_node_nm: int,
    circuit_type: str,
    precision: str = 'FP32',
    output_file: Optional[str] = None,
    compute_granularity: Optional[str] = None,
):
    """Plot TDP vs ALU count for different interconnect topologies."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not installed. Install with: pip install matplotlib")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    topologies = list(InterconnectTopology)
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']

    # Plot 1: TDP vs ALUs
    ax = axes[0]
    for topo, color in zip(topologies, colors):
        results = sweep_realistic_tdp(alu_range, process_node_nm, circuit_type,
                                       precision, topology=topo, num_points=15,
                                       compute_granularity=compute_granularity)
        alus = [r.num_alus for r in results]
        tdps = [r.total_tdp_w for r in results]
        ax.plot(alus, tdps, '-o', color=color, linewidth=2, markersize=4, label=topo.value)

    ax.set_xlabel('Number of ALUs', fontsize=12)
    ax.set_ylabel('TDP (Watts)', fontsize=12)
    ax.set_title(f'TDP by Interconnect Topology\n({process_node_nm}nm, {circuit_type})', fontsize=12)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.legend(title='Topology', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Plot 2: Interconnect power fraction
    ax = axes[1]
    for topo, color in zip(topologies, colors):
        results = sweep_realistic_tdp(alu_range, process_node_nm, circuit_type,
                                       precision, topology=topo, num_points=15,
                                       compute_granularity=compute_granularity)
        alus = [r.num_alus for r in results]
        interconnect_frac = [r.interconnect_power_w / r.total_tdp_w * 100 for r in results]
        ax.plot(alus, interconnect_frac, '-o', color=color, linewidth=2, markersize=4, label=topo.value)

    ax.set_xlabel('Number of ALUs', fontsize=12)
    ax.set_ylabel('Interconnect Power (%)', fontsize=12)
    ax.set_title(f'Interconnect Power Fraction\n({process_node_nm}nm, {circuit_type})', fontsize=12)
    ax.set_xscale('log', base=2)
    ax.legend(title='Topology', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {output_file}")
    else:
        plt.show()


def plot_full_matrix_realistic(
    alu_range: Tuple[int, int],
    precision: str = 'FP32',
    topology: InterconnectTopology = InterconnectTopology.MESH_2D,
    output_file: Optional[str] = None,
    compute_granularity: Optional[str] = None,
):
    """
    Generate a 2x2 matrix of plots showing realistic TDP with infrastructure.

    All plots use TOPS (Tera Operations Per Second) as the X-axis for
    user-meaningful performance metrics.

    Row 1: Power Analysis
      Plot 1: TDP breakdown stacked area (compute, SRAM, interconnect, control, idle)
      Plot 2: TDP vs TOPS for different circuit types

    Row 2: Efficiency Analysis
      Plot 3: TOPS/W vs TOPS (energy efficiency scaling)
      Plot 4: Infrastructure overhead vs TOPS by process node
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not installed. Install with: pip install matplotlib")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    process_nodes = [3, 4, 5, 7, 14, 28]
    circuit_types = ['x86_performance', 'tensor_core', 'systolic_mac', 'domain_flow']
    circuit_labels = ['x86 CPU', 'GPU Tensor', 'TPU Systolic', 'KPU Domain']
    circuit_colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']

    # Precision multipliers for TOPS calculation
    # ops_per_alu_per_cycle varies by precision
    precision_ops_mult = {
        'FP64': 1, 'FP32': 2, 'TF32': 4, 'BF16': 4,
        'FP16': 4, 'FP8': 8, 'INT8': 8, 'INT4': 16, 'INT2': 32,
    }
    ops_mult = precision_ops_mult.get(precision.upper(), 2)
    freq_ghz = 1.5  # Reference frequency

    def alus_to_tops(num_alus: int) -> float:
        """Convert ALU count to TOPS at reference frequency."""
        return num_alus * ops_mult * freq_ghz * 1e9 / 1e12

    # =========================================================================
    # Row 1: Power Analysis
    # =========================================================================

    # Plot 1: TDP breakdown stacked area (5nm, tensor_core) vs TOPS
    ax = axes[0, 0]
    results = sweep_realistic_tdp(alu_range, 5, 'tensor_core', precision,
                                   topology=topology, num_points=20,
                                   compute_granularity=compute_granularity)
    tops_vals = [alus_to_tops(r.num_alus) for r in results]
    compute = [r.compute_power_w for r in results]
    sram = [r.sram_power_w for r in results]
    interconnect = [r.interconnect_power_w for r in results]
    control = [r.control_power_w for r in results]
    idle = [r.idle_power_w for r in results]

    ax.stackplot(tops_vals, compute, sram, interconnect, control, idle,
                 labels=['Compute', 'SRAM', 'Interconnect', 'Control', 'Idle'],
                 colors=['#2ecc71', '#3498db', '#e74c3c', '#9b59b6', '#95a5a6'],
                 alpha=0.8)
    ax.set_xlabel(f'Peak Throughput (TOPS {precision})')
    ax.set_ylabel('Power (Watts)')
    ax.set_title(f'TDP Component Breakdown\n(5nm, tensor_core, {topology.value})')
    ax.set_xscale('log')
    ax.legend(loc='upper left', fontsize=7)
    ax.grid(True, alpha=0.3)

    # Plot 2: TDP vs TOPS for different circuit types
    ax = axes[0, 1]
    for ct, label, color in zip(circuit_types, circuit_labels, circuit_colors):
        results = sweep_realistic_tdp(alu_range, 5, ct, precision,
                                       topology=topology, num_points=15)
        tops_vals = [alus_to_tops(r.num_alus) for r in results]
        tdps = [r.total_tdp_w for r in results]
        ax.plot(tops_vals, tdps, '-', color=color, linewidth=2, label=label)

    # Add reference points for known hardware
    ax.axvline(x=1.98, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.text(2.1, 50, 'H100\n1980 TFLOPS', fontsize=7, color='gray')

    ax.set_xlabel(f'Peak Throughput (TOPS {precision})')
    ax.set_ylabel('TDP (Watts)')
    ax.set_title(f'TDP by Circuit Type\n(5nm, {precision}, {topology.value})')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(title='Circuit', fontsize=8)

    # =========================================================================
    # Row 2: Efficiency Analysis
    # =========================================================================

    # Plot 3: TOPS/W vs TOPS (energy efficiency)
    ax = axes[1, 0]
    for ct, label, color in zip(circuit_types, circuit_labels, circuit_colors):
        results = sweep_realistic_tdp(alu_range, 5, ct, precision,
                                       topology=topology, num_points=15)
        tops_vals = [alus_to_tops(r.num_alus) for r in results]
        tops_per_w = [alus_to_tops(r.num_alus) / r.total_tdp_w for r in results]
        ax.plot(tops_vals, tops_per_w, '-o', color=color, linewidth=2, markersize=3, label=label)

    ax.set_xlabel(f'Peak Throughput (TOPS {precision})')
    ax.set_ylabel(f'Energy Efficiency (TOPS/W)')
    ax.set_title(f'Energy Efficiency vs Scale\n(5nm, {freq_ghz} GHz)')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(title='Circuit', fontsize=8, loc='upper right')

    # Plot 4: Infrastructure overhead vs TOPS by process node
    ax = axes[1, 1]
    node_colors = plt.cm.viridis(np.linspace(0, 0.9, len(process_nodes)))
    for node, color in zip(process_nodes, node_colors):
        results = sweep_realistic_tdp(alu_range, node, 'tensor_core', precision,
                                       topology=topology, num_points=15,
                                       compute_granularity=compute_granularity)
        tops_vals = [alus_to_tops(r.num_alus) for r in results]
        overheads = [r.infrastructure_overhead for r in results]
        ax.plot(tops_vals, overheads, '-o', color=color, linewidth=2, markersize=3, label=f'{node}nm')

    ax.set_xlabel(f'Peak Throughput (TOPS {precision})')
    ax.set_ylabel('Infrastructure Overhead (x)')
    ax.set_title(f'Infrastructure Overhead by Process\n(tensor_core, {topology.value})')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(title='Process', fontsize=8, loc='upper left')

    plt.suptitle(f'Realistic TDP Analysis with SoC Infrastructure ({precision})',
                 fontsize=16, y=1.02)
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {output_file}")
    else:
        plt.show()


# =============================================================================
# CLI Output Functions
# =============================================================================

def print_estimate(est: TDPEstimate):
    """Print a single TDP estimate."""
    print("\n" + "="*70)
    print("TDP ESTIMATION RESULT")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  ALUs:           {est.num_alus:,}")
    print(f"  Process:        {est.process_node_nm}nm")
    print(f"  Circuit:        {est.circuit_type}")
    print(f"  Precision:      {est.precision}")
    print(f"  Frequency:      {est.frequency_ghz:.2f} GHz")

    print(f"\nALU Energy Breakdown:")
    print(f"  Base energy:    {est.base_energy_pj:.2f} pJ (from process node)")
    print(f"  x Circuit mult: {est.circuit_multiplier:.2f}x")
    print(f"  x Precision:    {est.precision_scale:.2f}x")
    print(f"  = ALU energy:   {est.pure_alu_energy_pj:.3f} pJ")

    print(f"\nOperand Fetch Energy (register-to-ALU delivery):")
    print(f"  Fetch energy:   {est.operand_fetch_energy_pj:.3f} pJ")
    print(f"  Reuse factor:   {est.operand_reuse_factor:.1f}x")
    print(f"  ALU/Fetch:      {est.alu_fetch_ratio:.2f} {'(ALU-dominated)' if est.alu_fetch_ratio > 1 else '(Fetch-dominated)'}")

    print(f"\nTotal Energy per Operation:")
    print(f"  ALU:            {est.pure_alu_energy_pj:.3f} pJ ({est.pure_alu_energy_pj/est.total_energy_per_op_pj*100:.1f}%)")
    print(f"  Operand Fetch:  {est.operand_fetch_energy_pj:.3f} pJ ({est.operand_fetch_energy_pj/est.total_energy_per_op_pj*100:.1f}%)")
    print(f"  -----------------------------------------")
    print(f"  TOTAL:          {est.total_energy_per_op_pj:.3f} pJ")

    print(f"\nThroughput:")
    print(f"  Ops/cycle:      {est.ops_per_cycle}")
    print(f"  Max Throughput: {est.tops:.2f} TOPS")

    print(f"\nPower:")
    print(f"  TDP:            {est.tdp_watts:.1f} W")
    print(f"  TOPS/W:         {est.tops_per_watt:.2f}")
    print()


def print_sweep_table(results: List[TDPEstimate]):
    """Print sweep results as a table."""
    print("\n" + "="*80)
    print("ALU SWEEP RESULTS")
    print("="*80)
    print(f"\nProcess: {results[0].process_node_nm}nm | Circuit: {results[0].circuit_type} | "
          f"Precision: {results[0].precision}")
    print()
    print(f"{'ALUs':>10} {'TDP (W)':>10} {'TOPS':>10} {'TOPS/W':>10} {'pJ/op':>10}")
    print("-"*55)
    for r in results:
        print(f"{r.num_alus:>10,} {r.tdp_watts:>10.1f} {r.tops:>10.2f} "
              f"{r.tops_per_watt:>10.2f} {r.energy_per_op_pj:>10.3f}")
    print()


def print_process_comparison(results: Dict[int, TDPEstimate]):
    """Print process node comparison table."""
    print("\n" + "="*80)
    print("PROCESS NODE COMPARISON")
    print("="*80)
    first = list(results.values())[0]
    print(f"\nALUs: {first.num_alus:,} | Circuit: {first.circuit_type} | "
          f"Precision: {first.precision}")
    print()
    print(f"{'Node':>8} {'Base (pJ)':>10} {'Final (pJ)':>12} {'TDP (W)':>10} "
          f"{'TOPS':>10} {'TOPS/W':>10}")
    print("-"*65)
    for node, r in sorted(results.items()):
        print(f"{node:>6}nm {r.base_energy_pj:>10.2f} {r.energy_per_op_pj:>12.3f} "
              f"{r.tdp_watts:>10.1f} {r.tops:>10.2f} {r.tops_per_watt:>10.2f}")
    print()


def print_circuit_comparison(results: Dict[str, TDPEstimate]):
    """Print circuit type comparison table."""
    print("\n" + "="*110)
    print("CIRCUIT TYPE COMPARISON (with Operand Fetch Energy)")
    print("="*110)
    first = list(results.values())[0]
    print(f"\nALUs: {first.num_alus:,} | Process: {first.process_node_nm}nm | "
          f"Precision: {first.precision}")
    print()
    print(f"{'Circuit':>18} {'ALU(pJ)':>8} {'Fetch(pJ)':>10} {'Total(pJ)':>10} {'Reuse':>8} {'TDP (W)':>10} {'TOPS/W':>10}")
    print("-"*90)
    for ct, r in sorted(results.items(), key=lambda x: x[1].total_energy_per_op_pj, reverse=True):
        reuse_str = f"{r.operand_reuse_factor:.0f}x" if r.operand_reuse_factor > 1 else "1x"
        print(f"{ct:>18} {r.pure_alu_energy_pj:>8.3f} {r.operand_fetch_energy_pj:>10.3f} "
              f"{r.total_energy_per_op_pj:>10.3f} {reuse_str:>8} {r.tdp_watts:>10.1f} {r.tops_per_watt:>10.2f}")

    print()
    print("KEY INSIGHT: ALU energy is similar across circuits; operand fetch makes the difference!")
    print("  - CPU/GPU: Fetch-dominated (every op reads from register file)")
    print("  - TPU/KPU: ALU-dominated (spatial reuse amortizes operand fetch)")
    print()


def print_realistic_estimate(breakdown: InfrastructurePowerBreakdown):
    """Print a realistic TDP estimate with infrastructure breakdown."""
    print("\n" + "="*80)
    print("REALISTIC TDP ESTIMATION (with SoC Infrastructure)")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  ALUs:           {breakdown.num_alus:,}")
    print(f"  Process:        {breakdown.process_node_nm}nm")
    print(f"  Circuit:        {breakdown.circuit_type}")
    print(f"  Topology:       {breakdown.topology.value}")
    print(f"  Frequency:      {breakdown.frequency_ghz:.2f} GHz")

    print(f"\nOn-Chip Resources:")
    print(f"  SRAM:           {breakdown.sram_mib:.1f} MiB ({SRAM_BYTES_PER_ALU} bytes/ALU)")
    print(f"\nDie Area Breakdown:")
    print(f"  Compute Area:   {breakdown.compute_area_mm2:>8.2f} mm^2  ({breakdown.compute_area_fraction*100:>5.1f}%)")
    print(f"  SRAM Area:      {breakdown.sram_area_mm2:>8.2f} mm^2  ({breakdown.sram_area_mm2/breakdown.total_area_mm2*100:>5.1f}%)")
    print(f"  Control Area:   {breakdown.control_area_mm2:>8.2f} mm^2  ({breakdown.control_area_mm2/breakdown.total_area_mm2*100:>5.1f}%)")
    print(f"  Total Area:     {breakdown.total_area_mm2:>8.2f} mm^2")

    print(f"\nPower Breakdown:")
    total = breakdown.total_tdp_w
    print(f"  Compute:        {breakdown.compute_power_w:>8.1f} W  ({breakdown.compute_power_w/total*100:>5.1f}%)")
    print(f"  SRAM:           {breakdown.sram_power_w:>8.1f} W  ({breakdown.sram_power_w/total*100:>5.1f}%)")
    print(f"  Interconnect:   {breakdown.interconnect_power_w:>8.1f} W  ({breakdown.interconnect_power_w/total*100:>5.1f}%)")
    print(f"  Control:        {breakdown.control_power_w:>8.1f} W  ({breakdown.control_power_w/total*100:>5.1f}%)")
    print(f"  Idle/Leakage:   {breakdown.idle_power_w:>8.1f} W  ({breakdown.idle_power_w/total*100:>5.1f}%)")
    print(f"  ─────────────────────────────────────────")
    print(f"  Dynamic Total:  {breakdown.total_dynamic_power_w:>8.1f} W")
    print(f"  TOTAL TDP:      {breakdown.total_tdp_w:>8.1f} W")

    print(f"\nEfficiency Metrics:")
    print(f"  Compute Power Fraction:  {breakdown.compute_fraction*100:.1f}% (power to ALU switching)")
    print(f"  Compute Area Fraction:   {breakdown.compute_area_fraction*100:.1f}% (die area to compute)")
    print(f"  Infrastructure Overhead: {breakdown.infrastructure_overhead:.2f}x (total TDP / compute power)")

    # Show empirical calibration target if available
    from graphs.hardware.soc_infrastructure import EMPIRICAL_COMPUTE_FRACTION
    empirical_target = EMPIRICAL_COMPUTE_FRACTION.get(breakdown.circuit_type)
    if empirical_target:
        print(f"\nEmpirical Calibration:")
        print(f"  Target compute fraction: {empirical_target*100:.1f}% (from real hardware measurements)")
        model_vs_empirical = breakdown.compute_fraction / empirical_target
        if model_vs_empirical > 1.5:
            print(f"  Model predicts {model_vs_empirical:.1f}x higher - model may over-estimate compute")
        elif model_vs_empirical < 0.67:
            print(f"  Model predicts {model_vs_empirical:.1f}x lower - model may under-estimate compute")
        else:
            print(f"  Model vs empirical: {model_vs_empirical:.2f}x (within calibration range)")

    print()
    print("KEY INSIGHT: Only {:.1f}% of TDP is actual compute!".format(breakdown.compute_fraction * 100))
    print("  Infrastructure (SRAM + interconnect + control + idle) dominates at scale.")
    print("  Compute area fraction ({:.1f}%) shows physical silicon allocation.".format(breakdown.compute_area_fraction * 100))
    print()


def print_realistic_sweep_table(results: List[InfrastructurePowerBreakdown]):
    """Print realistic sweep results as a table."""
    print("\n" + "="*120)
    print("REALISTIC TDP SWEEP RESULTS")
    print("="*120)
    print(f"\nProcess: {results[0].process_node_nm}nm | Circuit: {results[0].circuit_type} | "
          f"Topology: {results[0].topology.value}")
    print()
    print(f"{'ALUs':>10} {'Compute':>10} {'SRAM':>10} {'Intercon':>10} {'Control':>10} "
          f"{'Idle':>10} {'TDP':>10} {'Overhead':>10}")
    print(f"{'':>10} {'(W)':>10} {'(W)':>10} {'(W)':>10} {'(W)':>10} "
          f"{'(W)':>10} {'(W)':>10} {'(x)':>10}")
    print("-"*100)
    for r in results:
        print(f"{r.num_alus:>10,} {r.compute_power_w:>10.1f} {r.sram_power_w:>10.1f} "
              f"{r.interconnect_power_w:>10.1f} {r.control_power_w:>10.1f} "
              f"{r.idle_power_w:>10.1f} {r.total_tdp_w:>10.1f} {r.infrastructure_overhead:>10.2f}")
    print()


def print_topology_comparison(results: Dict[str, InfrastructurePowerBreakdown]):
    """Print topology comparison table."""
    print("\n" + "="*100)
    print("INTERCONNECT TOPOLOGY COMPARISON")
    print("="*100)
    first = list(results.values())[0]
    print(f"\nALUs: {first.num_alus:,} | Process: {first.process_node_nm}nm | "
          f"Circuit: {first.circuit_type}")
    print()
    print(f"{'Topology':>12} {'Compute':>10} {'SRAM':>10} {'Intercon':>10} {'Control':>10} "
          f"{'Idle':>10} {'TDP':>10} {'Intercon%':>10}")
    print("-"*95)
    for topo, r in sorted(results.items(), key=lambda x: x[1].total_tdp_w):
        intercon_pct = r.interconnect_power_w / r.total_tdp_w * 100
        print(f"{topo:>12} {r.compute_power_w:>10.1f} {r.sram_power_w:>10.1f} "
              f"{r.interconnect_power_w:>10.1f} {r.control_power_w:>10.1f} "
              f"{r.idle_power_w:>10.1f} {r.total_tdp_w:>10.1f} {intercon_pct:>9.1f}%")
    print()
    print("NOTE: Interconnect power varies significantly by topology!")
    print("  - Mesh: Good for spatial locality, moderate wire length")
    print("  - H-tree: Good for broadcast, longer average wires")
    print("  - Crossbar: Only practical for small N, highest connectivity")
    print("  - CLOS: Balanced, rearrangeably non-blocking")
    print()


def print_precision_comparison(results: Dict[str, TDPEstimate]):
    """Print precision comparison table."""
    print("\n" + "="*90)
    print("PRECISION COMPARISON")
    print("="*90)
    first = list(results.values())[0]
    print(f"\nALUs: {first.num_alus:,} | Process: {first.process_node_nm}nm | "
          f"Circuit: {first.circuit_type}")
    print()
    print(f"{'Precision':>10} {'E-Scale':>8} {'Ops/Cyc':>8} {'pJ/op':>8} {'TDP (W)':>10} "
          f"{'TOPS':>10} {'TOPS/W':>10}")
    print("-"*75)
    # Sort by energy scale (highest first = FP64, then FP32, etc.)
    for prec, r in sorted(results.items(), key=lambda x: x[1].precision_scale, reverse=True):
        print(f"{prec:>10} {r.precision_scale:>8.3f} {r.ops_per_cycle:>8} {r.energy_per_op_pj:>8.3f} "
              f"{r.tdp_watts:>10.1f} {r.tops:>10.2f} {r.tops_per_watt:>10.2f}")
    print()


# =============================================================================
# Main CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Estimate TDP power for accelerator configurations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single estimate (linear model)
  %(prog)s --alus 16384 --process 4 --circuit tensor_core --precision FP16

  # Single estimate (REALISTIC model with infrastructure)
  %(prog)s --alus 16384 --process 4 --circuit tensor_core --realistic

  # Sweep ALU count (realistic)
  %(prog)s --sweep-alus 1024 65536 --process 4 --circuit tensor_core --realistic

  # Compare linear vs realistic models
  %(prog)s --sweep-alus 1024 65536 --compare-models --plot

  # Compare interconnect topologies
  %(prog)s --alus 16384 --process 4 --compare-topologies

  # TDP breakdown stacked chart
  %(prog)s --sweep-alus 1024 65536 --realistic --breakdown --plot

  # Compare process technologies
  %(prog)s --alus 16384 --compare-process --circuit tensor_core --plot

  # Full matrix plot (linear model)
  %(prog)s --sweep-alus 1024 65536 --full-matrix --plot --output tdp_linear.png

  # Full matrix plot (REALISTIC model with infrastructure breakdown)
  %(prog)s --sweep-alus 1024 65536 --full-matrix-realistic --plot --output tdp_realistic.png
        """
    )

    # Required/main args
    parser.add_argument('--alus', type=int, default=16384,
                       help='Number of ALUs (default: 16384)')
    parser.add_argument('--process', type=int, default=5,
                       help='Process node in nm (default: 5)')
    parser.add_argument('--circuit', type=str, default='tensor_core',
                       choices=list(CIRCUIT_TYPE_MULTIPLIER.keys()),
                       help='Circuit type (default: tensor_core)')
    parser.add_argument('--precision', type=str, default='FP32',
                       choices=list(PRECISION_ENERGY_SCALE.keys()),
                       help='Compute precision (default: FP32)')
    parser.add_argument('--frequency', type=float, default=None,
                       help='Clock frequency in GHz (default: auto)')

    # Sweep options
    parser.add_argument('--sweep-alus', type=int, nargs=2, metavar=('MIN', 'MAX'),
                       help='Sweep ALU count from MIN to MAX')
    parser.add_argument('--num-points', type=int, default=15,
                       help='Number of sweep points (default: 15)')

    # Comparison options
    parser.add_argument('--compare-process', action='store_true',
                       help='Compare across process nodes')
    parser.add_argument('--compare-circuits', action='store_true',
                       help='Compare across circuit types')
    parser.add_argument('--compare-precisions', action='store_true',
                       help='Compare across precisions (FP64, FP32, FP16, INT8, etc.)')
    parser.add_argument('--full-matrix', action='store_true',
                       help='Generate full 2x2 comparison matrix (linear model)')
    parser.add_argument('--full-matrix-realistic', action='store_true',
                       help='Generate full 2x2 comparison matrix (realistic model with infrastructure)')

    # Output options
    parser.add_argument('--plot', action='store_true',
                       help='Generate matplotlib plots')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file for plots (PNG, PDF, etc.)')

    # Realistic model options
    parser.add_argument('--realistic', action='store_true',
                       help='Use realistic TDP model with infrastructure overhead')
    parser.add_argument('--topology', type=str, default='mesh',
                       choices=['mesh', 'htree', 'crossbar', 'clos'],
                       help='Interconnect topology (default: mesh)')
    parser.add_argument('--compute-granularity', type=str, default=None,
                       choices=list(COMPUTE_GRANULARITY_PRESETS.keys()),
                       help='Compute building block preset (auto-selected if not specified)')
    parser.add_argument('--breakdown', action='store_true',
                       help='Show/plot TDP breakdown by component')
    parser.add_argument('--compare-models', action='store_true',
                       help='Compare linear vs realistic TDP models')
    parser.add_argument('--compare-topologies', action='store_true',
                       help='Compare across interconnect topologies')

    # List options
    parser.add_argument('--list-circuits', action='store_true',
                       help='List available circuit types')
    parser.add_argument('--list-process', action='store_true',
                       help='List available process nodes')
    parser.add_argument('--list-precisions', action='store_true',
                       help='List available precisions')
    parser.add_argument('--list-topologies', action='store_true',
                       help='List available interconnect topologies')
    parser.add_argument('--list-granularities', action='store_true',
                       help='List available compute granularity presets')

    args = parser.parse_args()

    # Parse topology
    topology = get_topology_from_string(args.topology)

    # Get compute granularity (may be None for auto-select)
    granularity = args.compute_granularity

    # Handle list options
    if args.list_circuits:
        print("\nAvailable Circuit Types:")
        print("-" * 50)
        for ct, mult in sorted(CIRCUIT_TYPE_MULTIPLIER.items(), key=lambda x: x[1], reverse=True):
            if ct != 'custom_datacenter':  # Skip legacy alias
                print(f"  {ct:20} {mult:.2f}x energy multiplier")
        return

    if args.list_process:
        print("\nAvailable Process Nodes:")
        print("-" * 40)
        for node, energy in sorted(PROCESS_NODE_BASE_ENERGY_PJ.items()):
            print(f"  {node:3}nm  {energy:.2f} pJ base energy")
        return

    if args.list_precisions:
        print("\nAvailable Precisions:")
        print("-" * 50)
        for prec, scale in sorted(PRECISION_ENERGY_SCALE.items(), key=lambda x: x[1], reverse=True):
            ops = PRECISION_OPS_PER_CYCLE[prec]
            print(f"  {prec:6}  {scale:.3f}x energy  {ops:2}x throughput")
        return

    if args.list_topologies:
        print("\nAvailable Interconnect Topologies:")
        print("-" * 60)
        print(f"  {'mesh':12} 2D mesh network (spatial locality, O(N) wires)")
        print(f"  {'htree':12} H-tree (broadcast/reduction, O(N log N) wires)")
        print(f"  {'crossbar':12} Crossbar (all-to-all, O(N^2) wires, small N only)")
        print(f"  {'clos':12} CLOS network (non-blocking, O(N log N) wires)")
        return

    if args.list_granularities:
        print("\nAvailable Compute Granularity Presets:")
        print("-" * 80)
        print(f"{'Name':15} {'ALUs/Cluster':>14} {'Clusters/Group':>16} {'Groups/Chip':>13}")
        print("-" * 80)
        for name, g in COMPUTE_GRANULARITY_PRESETS.items():
            print(f"{name:15} {g.alus_per_cluster:>14,} {g.clusters_per_group:>16} {g.groups_per_chip:>13}")
        print()
        print("Presets model hierarchical building blocks:")
        print()
        print("  nvidia_sm    - NVIDIA SM (Streaming Multiprocessor)")
        print("                 128 CUDA cores + 4 Tensor Cores (256 MACs) = 384 ALUs")
        print("                 Hierarchy: GPU -> GPC (Graphics Processing Cluster) -> SM")
        print()
        print("  nvidia_tc    - NVIDIA Tensor Cores only (AI workloads)")
        print("                 4 Tensor Cores x 64 MACs = 256 ALUs per SM")
        print()
        print("  tpu_mxu      - Google TPU MXU (Matrix Multiply Unit)")
        print("                 128x128 systolic array = 16,384 MACs")
        print()
        print("  cpu_core     - Intel/AMD CPU core")
        print("                 8 FP32 ALUs (2 x 256-bit FMA units)")
        print("                 Hierarchy: Chip -> CCX (Core Complex) -> Core")
        print()
        print("  kpu_tile     - KPU/Domain Flow tile")
        print("                 16x16 PE (Processing Element) array = 256 MACs")
        print()
        print("  generic_tile - Generic accelerator tile (configurable baseline)")
        return

    # =========================================================================
    # Realistic Model Handlers
    # =========================================================================

    # Full matrix realistic plot
    if args.full_matrix_realistic:
        alu_range = args.sweep_alus if args.sweep_alus else (1024, 65536)
        if args.plot:
            plot_full_matrix_realistic(alu_range, args.precision, topology, args.output,
                                       compute_granularity=granularity)
        else:
            print("Use --plot to generate the full matrix realistic visualization")
        return

    # Compare linear vs realistic models
    if args.compare_models:
        alu_range = args.sweep_alus if args.sweep_alus else (1024, 65536)
        if args.plot:
            plot_linear_vs_realistic(alu_range, args.process, args.circuit,
                                     args.precision, topology, args.output,
                                     compute_granularity=granularity)
        else:
            print("\nLinear vs Realistic Model Comparison")
            print("="*80)
            linear = sweep_alus(alu_range, args.process, args.circuit, args.precision, num_points=10)
            realistic = sweep_realistic_tdp(alu_range, args.process, args.circuit,
                                            args.precision, topology=topology, num_points=10,
                                            compute_granularity=granularity)
            print(f"\n{'ALUs':>10} {'Linear TDP':>12} {'Realistic TDP':>14} {'Overhead':>10}")
            print("-"*50)
            for i, (l, r) in enumerate(zip(linear, realistic)):
                overhead = r.total_tdp_w / l.tdp_watts
                print(f"{l.num_alus:>10,} {l.tdp_watts:>12.1f} W {r.total_tdp_w:>14.1f} W {overhead:>9.2f}x")
            print("\nUse --plot to visualize the difference")
        return

    # Compare topologies
    if args.compare_topologies:
        if args.sweep_alus and args.plot:
            plot_topology_comparison(tuple(args.sweep_alus), args.process, args.circuit,
                                    args.precision, args.output,
                                    compute_granularity=granularity)
        else:
            results = compare_topologies(args.alus, args.process, args.circuit,
                                        args.precision, args.frequency,
                                        compute_granularity=granularity)
            print_topology_comparison(results)
            if args.sweep_alus:
                print("Use --plot to visualize topology comparison across ALU counts")
        return

    # Realistic breakdown plot
    if args.breakdown and args.plot:
        alu_range = args.sweep_alus if args.sweep_alus else (1024, 65536)
        plot_realistic_tdp_breakdown(alu_range, args.process, args.circuit,
                                     args.precision, topology, args.output,
                                     compute_granularity=granularity)
        return

    # Realistic sweep
    if args.realistic and args.sweep_alus:
        results = sweep_realistic_tdp(
            tuple(args.sweep_alus),
            args.process,
            args.circuit,
            args.precision,
            args.frequency,
            topology,
            args.num_points,
            compute_granularity=granularity,
        )
        print_realistic_sweep_table(results)
        if args.plot:
            plot_realistic_tdp_breakdown(tuple(args.sweep_alus), args.process, args.circuit,
                                         args.precision, topology, args.output,
                                         compute_granularity=granularity)
        return

    # Single realistic estimate
    if args.realistic:
        breakdown = estimate_realistic_tdp(
            args.alus,
            args.process,
            args.circuit,
            args.precision,
            args.frequency,
            topology,
            compute_granularity=granularity,
        )
        print_realistic_estimate(breakdown)
        return

    # =========================================================================
    # Original Linear Model Handlers
    # =========================================================================

    # Full matrix plot
    if args.full_matrix:
        alu_range = args.sweep_alus if args.sweep_alus else (1024, 65536)
        if args.plot:
            plot_full_matrix(alu_range, args.precision, args.output)
        else:
            print("Use --plot to generate the full matrix visualization")
        return

    # Process and circuit comparison
    if args.compare_process and args.compare_circuits:
        # Both comparisons requested
        if args.sweep_alus:
            if args.plot:
                plot_process_comparison(tuple(args.sweep_alus), args.circuit, args.precision,
                                       output_file=args.output.replace('.', '_process.') if args.output else None)
                plot_circuit_comparison(tuple(args.sweep_alus), args.process, args.precision,
                                       output_file=args.output.replace('.', '_circuit.') if args.output else None)
            else:
                # Text output for both
                for node in [3, 4, 5, 7, 14, 28]:
                    results = sweep_alus(tuple(args.sweep_alus), node, args.circuit, args.precision, num_points=8)
                    print(f"\n--- {node}nm ---")
                    print_sweep_table(results)
        else:
            results_p = compare_process_nodes(args.alus, args.circuit, args.precision, args.frequency)
            print_process_comparison(results_p)
            results_c = compare_circuit_types(args.alus, args.process, args.precision)
            print_circuit_comparison(results_c)
        return

    # Process comparison only
    if args.compare_process:
        if args.sweep_alus and args.plot:
            plot_process_comparison(tuple(args.sweep_alus), args.circuit, args.precision,
                                   output_file=args.output)
        else:
            results = compare_process_nodes(args.alus, args.circuit, args.precision, args.frequency)
            print_process_comparison(results)
            if args.plot:
                plot_efficiency_comparison(args.alus, args.precision, args.output)
        return

    # Circuit comparison only
    if args.compare_circuits:
        if args.sweep_alus and args.plot:
            plot_circuit_comparison(tuple(args.sweep_alus), args.process, args.precision,
                                   output_file=args.output)
        else:
            results = compare_circuit_types(args.alus, args.process, args.precision)
            print_circuit_comparison(results)
        return

    # Precision comparison
    if args.compare_precisions:
        if args.sweep_alus and args.plot:
            plot_precision_comparison(tuple(args.sweep_alus), args.process, args.circuit,
                                     output_file=args.output)
        else:
            results = compare_precisions(args.alus, args.process, args.circuit)
            print_precision_comparison(results)
            if args.sweep_alus and not args.plot:
                print("Use --plot to generate precision comparison visualization")
        return

    # ALU sweep
    if args.sweep_alus:
        results = sweep_alus(
            tuple(args.sweep_alus),
            args.process,
            args.circuit,
            args.precision,
            args.frequency,
            args.num_points,
        )
        print_sweep_table(results)
        if args.plot:
            title = f"TDP vs ALU Count ({args.process}nm, {args.circuit}, {args.precision})"
            plot_tdp_sweep(results, title, args.output)
        return

    # Single estimate (default)
    estimate = estimate_tdp(
        args.alus,
        args.process,
        args.circuit,
        args.precision,
        args.frequency,
    )
    print_estimate(estimate)


if __name__ == '__main__':
    main()
