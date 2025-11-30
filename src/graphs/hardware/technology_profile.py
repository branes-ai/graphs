"""
Technology Profile - Process and Memory Technology Configuration

This module provides dataclasses that capture the confluence of process technology,
circuit design, and memory technology. These profiles enable energy models to be
parameterized by technology rather than using hardcoded values.

Key Concepts:
- ProcessTechnology: Foundry process characteristics (node, voltage, base energy)
- MemoryTechnology: Memory type characteristics (DDR, LPDDR, GDDR, HBM)
- TechnologyProfile: Complete configuration combining process + memory + market

Usage:
    # Use preset profiles
    from graphs.hardware.technology_profile import (
        DATACENTER_4NM_HBM3,
        EDGE_8NM_LPDDR5,
        AUTOMOTIVE_16NM_LPDDR4,
    )

    model = StoredProgramEnergyModel(tech_profile=EDGE_8NM_LPDDR5)

    # Or create custom profiles
    profile = TechnologyProfile.create(
        process_node_nm=7,
        memory_type=MemoryType.LPDDR5,
        target_market="edge"
    )
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional


# =============================================================================
# Process Node Energy Scaling
# =============================================================================
# Based on: Energy = Capacitance x Voltage^2 per switch
# Frequency does NOT affect energy per operation (only power)

PROCESS_NODE_BASE_ENERGY_PJ = {
    # Process node (nm) -> Base FP32 ALU energy (pJ)
    3:   1.2,   # Intel 18A, TSMC N3, AMD Zen 5
    4:   1.3,   # TSMC N4/N4P, NVIDIA Hopper/Blackwell
    5:   1.5,   # TSMC N5, Samsung 5LPE, Apple M1/M2
    6:   1.65,  # TSMC N6 (7nm extension)
    7:   1.8,   # TSMC N7, Samsung 7LPP, AMD Zen 2
    8:   1.9,   # Samsung 8LPP, NVIDIA Ampere (Jetson Orin)
    10:  2.1,   # Intel 10nm/7
    12:  2.5,   # TSMC 12FFC, GlobalFoundries 12LP
    14:  2.6,   # Intel 14nm, Samsung 14LPP
    16:  2.7,   # TSMC 16FFC, older edge devices
    22:  3.2,   # Intel 22FFL, automotive
    28:  4.0,   # TSMC 28HPC+, mature node for cost-sensitive
}


# =============================================================================
# Circuit Type Multipliers
# =============================================================================
# These multipliers capture the energy cost relative to a baseline standard cell
# FP32 ALU. The baseline (1.0) represents a simple standard-cell ALU at nominal
# voltage/frequency.
#
# CPUs trade energy for performance through:
#   - Higher voltage for faster switching (V^2 scaling)
#   - Deeper pipelines for higher frequency
#   - Wider superscalar for ILP (more transistors active)
#   - Full IEEE-754 compliance (denormals, exceptions, rounding modes)
#
# Accelerators trade flexibility for efficiency through:
#   - Eliminating instruction fetch per operation
#   - Fixed datapath (no dynamic scheduling)
#   - Reduced precision support
#   - Streaming/spatial execution

CIRCUIT_TYPE_MULTIPLIER = {
    # ==========================================================================
    # CPU Circuit Types (performance-optimized, higher energy)
    # ==========================================================================
    # x86 performance cores prioritize single-thread speed:
    # - 5+ GHz boost clocks require higher voltage (1.3-1.5V vs 0.7-0.9V)
    # - Deep OoO pipelines (14-20 stages) with large rename buffers
    # - Wide superscalar (6-8 way decode/issue)
    # - Full IEEE-754 FP with all rounding modes, denormals, exceptions
    #
    # Energy scales as V^2, plus extra switching for complex control logic

    'x86_performance':   2.50,  # Intel i7/i9, AMD Ryzen 9 (5GHz, full IEEE-754)
    'x86_efficiency':    1.50,  # Intel E-cores, lower clocks, simpler pipeline
    'arm_performance':   1.80,  # ARM Cortex-X4, Apple Firestorm (high-perf ARM)
    'arm_efficiency':    1.00,  # ARM Cortex-A520, Apple Icestorm (efficiency cores)

    # ==========================================================================
    # Standard Baseline
    # ==========================================================================
    'standard_cell':     1.00,  # Baseline: Standard cell library ALU at nominal V/f

    # ==========================================================================
    # GPU Circuit Types
    # ==========================================================================
    # GPUs amortize control across many threads but have SIMT overhead

    'cuda_core':         0.95,  # CUDA cores: simpler than CPU, but still flexible
    'simd_packed':       0.90,  # Packed SIMD ops (AVX-512, NEON) - control amortized
    'tensor_core':       0.85,  # Tensor Cores: fused 4x4x4 MAC, minimal control

    # ==========================================================================
    # Accelerator Circuit Types (efficiency-optimized, lower energy)
    # ==========================================================================
    # These eliminate instruction fetch entirely and use fixed/spatial datapaths

    'systolic_mac':      0.80,  # TPU systolic: no instruction fetch, weight-stationary
    'domain_flow':       0.75,  # KPU domain flow: streaming spatial, no fetch

    # ==========================================================================
    # Legacy alias (deprecated - use x86_performance instead)
    # ==========================================================================
    'custom_datacenter': 2.50,  # Alias for x86_performance
}


def get_process_base_energy_pj(process_node_nm: int) -> float:
    """
    Get base FP32 ALU energy for a process node.

    Interpolates for nodes not in the table.

    Args:
        process_node_nm: Process node in nanometers

    Returns:
        Base energy in picojoules
    """
    if process_node_nm in PROCESS_NODE_BASE_ENERGY_PJ:
        return PROCESS_NODE_BASE_ENERGY_PJ[process_node_nm]

    # Interpolate for missing nodes
    nodes = sorted(PROCESS_NODE_BASE_ENERGY_PJ.keys())
    if process_node_nm < nodes[0]:
        return PROCESS_NODE_BASE_ENERGY_PJ[nodes[0]]
    elif process_node_nm > nodes[-1]:
        return PROCESS_NODE_BASE_ENERGY_PJ[nodes[-1]]
    else:
        # Linear interpolation
        for i in range(len(nodes) - 1):
            if nodes[i] <= process_node_nm <= nodes[i+1]:
                e1 = PROCESS_NODE_BASE_ENERGY_PJ[nodes[i]]
                e2 = PROCESS_NODE_BASE_ENERGY_PJ[nodes[i+1]]
                t = (process_node_nm - nodes[i]) / (nodes[i+1] - nodes[i])
                return e1 + t * (e2 - e1)

    return 2.0  # Fallback


# =============================================================================
# Memory Technology
# =============================================================================

class MemoryType(Enum):
    """
    Memory technology types with their characteristics.

    Each type has different energy, bandwidth, and cost trade-offs:
    - DDR: Standard server/desktop memory (highest capacity, lowest cost/GB)
    - LPDDR: Low-power mobile/edge memory (lower power, higher cost/GB)
    - GDDR: Graphics memory (high bandwidth, moderate power)
    - HBM: High Bandwidth Memory (highest bandwidth, highest cost)
    """
    # DDR variants (standard DIMM, server/desktop)
    DDR4 = "ddr4"
    DDR5 = "ddr5"

    # LPDDR variants (mobile/edge, on-package or PoP)
    LPDDR4 = "lpddr4"
    LPDDR4X = "lpddr4x"
    LPDDR5 = "lpddr5"
    LPDDR5X = "lpddr5x"

    # GDDR variants (discrete graphics)
    GDDR6 = "gddr6"
    GDDR6X = "gddr6x"
    GDDR7 = "gddr7"

    # HBM variants (high-end accelerators)
    HBM2 = "hbm2"
    HBM2E = "hbm2e"
    HBM3 = "hbm3"
    HBM3E = "hbm3e"


@dataclass
class MemoryTechnologySpec:
    """
    Detailed specifications for a memory technology.

    Energy values are in picojoules per byte for data transfer.
    These include the energy for:
    - Row activation (amortized across burst)
    - Column access
    - Data transfer (I/O)
    - PHY/controller overhead
    """
    memory_type: MemoryType

    # Energy per byte (pJ/byte)
    energy_per_byte_pj: float

    # Bandwidth characteristics
    peak_bandwidth_gbps_per_channel: float  # GB/s per channel
    typical_channels: int                    # Typical channel count

    # Latency (ns)
    access_latency_ns: float

    # Cost tier (relative, 1.0 = DDR baseline)
    relative_cost: float

    # Target markets
    typical_markets: tuple  # ("datacenter", "edge", "mobile", etc.)

    # Notes
    description: str = ""


# Memory technology specifications
# Energy values based on literature and vendor specs
MEMORY_SPECS: Dict[MemoryType, MemoryTechnologySpec] = {
    # ==========================================================================
    # DDR (Standard Server/Desktop Memory)
    # ==========================================================================
    MemoryType.DDR4: MemoryTechnologySpec(
        memory_type=MemoryType.DDR4,
        energy_per_byte_pj=25.0,  # ~25 pJ/byte (3200 MT/s)
        peak_bandwidth_gbps_per_channel=25.6,  # 3200 MT/s x 8 bytes
        typical_channels=8,  # Server: 8 channels
        access_latency_ns=15.0,  # CAS latency ~15ns
        relative_cost=1.0,  # Baseline
        typical_markets=("server", "desktop", "workstation"),
        description="DDR4-3200: Standard server/desktop memory"
    ),
    MemoryType.DDR5: MemoryTechnologySpec(
        memory_type=MemoryType.DDR5,
        energy_per_byte_pj=20.0,  # ~20 pJ/byte (more efficient than DDR4)
        peak_bandwidth_gbps_per_channel=38.4,  # 4800 MT/s x 8 bytes
        typical_channels=8,  # Server: 8 channels
        access_latency_ns=14.0,  # Slightly lower latency
        relative_cost=1.3,  # ~30% premium over DDR4
        typical_markets=("server", "desktop", "workstation"),
        description="DDR5-4800: Next-gen server/desktop memory"
    ),

    # ==========================================================================
    # LPDDR (Low-Power Mobile/Edge Memory)
    # ==========================================================================
    MemoryType.LPDDR4: MemoryTechnologySpec(
        memory_type=MemoryType.LPDDR4,
        energy_per_byte_pj=12.0,  # ~12 pJ/byte (optimized for power)
        peak_bandwidth_gbps_per_channel=4.27,  # 4267 MT/s x 2 bytes (x16)
        typical_channels=4,  # Mobile: 4 channels
        access_latency_ns=20.0,
        relative_cost=1.5,  # Premium for low power
        typical_markets=("mobile", "edge", "automotive"),
        description="LPDDR4-4267: Mobile/edge low-power memory"
    ),
    MemoryType.LPDDR4X: MemoryTechnologySpec(
        memory_type=MemoryType.LPDDR4X,
        energy_per_byte_pj=10.0,  # ~10 pJ/byte (lower voltage than LPDDR4)
        peak_bandwidth_gbps_per_channel=4.27,
        typical_channels=4,
        access_latency_ns=20.0,
        relative_cost=1.6,
        typical_markets=("mobile", "edge", "automotive"),
        description="LPDDR4X-4267: Lower voltage variant of LPDDR4"
    ),
    MemoryType.LPDDR5: MemoryTechnologySpec(
        memory_type=MemoryType.LPDDR5,
        energy_per_byte_pj=8.0,  # ~8 pJ/byte (significant improvement)
        peak_bandwidth_gbps_per_channel=6.4,  # 6400 MT/s x 2 bytes
        typical_channels=4,
        access_latency_ns=18.0,
        relative_cost=2.0,
        typical_markets=("mobile", "edge", "automotive"),
        description="LPDDR5-6400: High-performance low-power memory"
    ),
    MemoryType.LPDDR5X: MemoryTechnologySpec(
        memory_type=MemoryType.LPDDR5X,
        energy_per_byte_pj=7.0,  # ~7 pJ/byte (best LPDDR efficiency)
        peak_bandwidth_gbps_per_channel=8.5,  # 8533 MT/s x 2 bytes
        typical_channels=4,
        access_latency_ns=17.0,
        relative_cost=2.5,
        typical_markets=("mobile", "edge", "premium_automotive"),
        description="LPDDR5X-8533: Flagship mobile/edge memory"
    ),

    # ==========================================================================
    # GDDR (Graphics Memory)
    # ==========================================================================
    MemoryType.GDDR6: MemoryTechnologySpec(
        memory_type=MemoryType.GDDR6,
        energy_per_byte_pj=15.0,  # ~15 pJ/byte
        peak_bandwidth_gbps_per_channel=16.0,  # 16 Gbps per pin, x32 bus
        typical_channels=8,  # 8 chips x 32-bit = 256-bit bus
        access_latency_ns=25.0,  # Higher latency than DDR
        relative_cost=2.0,
        typical_markets=("gaming_gpu", "professional_gpu", "edge_gpu"),
        description="GDDR6-16Gbps: Standard graphics memory"
    ),
    MemoryType.GDDR6X: MemoryTechnologySpec(
        memory_type=MemoryType.GDDR6X,
        energy_per_byte_pj=18.0,  # ~18 pJ/byte (PAM4 signaling overhead)
        peak_bandwidth_gbps_per_channel=21.0,  # 21 Gbps per pin
        typical_channels=8,
        access_latency_ns=25.0,
        relative_cost=2.5,
        typical_markets=("gaming_gpu", "professional_gpu"),
        description="GDDR6X-21Gbps: High-bandwidth graphics memory (PAM4)"
    ),
    MemoryType.GDDR7: MemoryTechnologySpec(
        memory_type=MemoryType.GDDR7,
        energy_per_byte_pj=14.0,  # ~14 pJ/byte (improved efficiency)
        peak_bandwidth_gbps_per_channel=32.0,  # 32 Gbps per pin
        typical_channels=8,
        access_latency_ns=22.0,
        relative_cost=3.0,
        typical_markets=("gaming_gpu", "professional_gpu"),
        description="GDDR7-32Gbps: Next-gen graphics memory"
    ),

    # ==========================================================================
    # HBM (High Bandwidth Memory)
    # ==========================================================================
    MemoryType.HBM2: MemoryTechnologySpec(
        memory_type=MemoryType.HBM2,
        energy_per_byte_pj=7.0,  # ~7 pJ/byte (very efficient due to wide bus)
        peak_bandwidth_gbps_per_channel=256.0,  # 256 GB/s per stack
        typical_channels=4,  # 4 stacks typical
        access_latency_ns=100.0,  # Higher latency, but massive bandwidth
        relative_cost=8.0,  # Very expensive
        typical_markets=("datacenter_gpu", "hpc", "ai_accelerator"),
        description="HBM2: First-gen high-bandwidth stacked memory"
    ),
    MemoryType.HBM2E: MemoryTechnologySpec(
        memory_type=MemoryType.HBM2E,
        energy_per_byte_pj=6.5,  # ~6.5 pJ/byte
        peak_bandwidth_gbps_per_channel=410.0,  # 410 GB/s per stack
        typical_channels=5,  # 5 stacks (A100)
        access_latency_ns=95.0,
        relative_cost=10.0,
        typical_markets=("datacenter_gpu", "hpc", "ai_accelerator"),
        description="HBM2e: Enhanced HBM2, used in A100"
    ),
    MemoryType.HBM3: MemoryTechnologySpec(
        memory_type=MemoryType.HBM3,
        energy_per_byte_pj=5.5,  # ~5.5 pJ/byte (most efficient)
        peak_bandwidth_gbps_per_channel=665.0,  # 665 GB/s per stack
        typical_channels=5,  # 5 stacks (H100)
        access_latency_ns=90.0,
        relative_cost=12.0,
        typical_markets=("datacenter_gpu", "hpc", "ai_accelerator"),
        description="HBM3: High-bandwidth memory for H100/MI300"
    ),
    MemoryType.HBM3E: MemoryTechnologySpec(
        memory_type=MemoryType.HBM3E,
        energy_per_byte_pj=5.0,  # ~5 pJ/byte (best HBM efficiency)
        peak_bandwidth_gbps_per_channel=820.0,  # 820 GB/s per stack
        typical_channels=6,  # 6 stacks (B100)
        access_latency_ns=85.0,
        relative_cost=15.0,
        typical_markets=("datacenter_gpu", "hpc", "ai_accelerator"),
        description="HBM3e: Latest HBM for B100/MI350"
    ),
}


def get_memory_spec(memory_type: MemoryType) -> MemoryTechnologySpec:
    """Get the specification for a memory type."""
    return MEMORY_SPECS[memory_type]


# =============================================================================
# On-Chip Memory (SRAM) Scaling
# =============================================================================
# SRAM energy scales with process node (smaller transistors = less capacitance)

def get_sram_energy_per_byte_pj(process_node_nm: int, sram_type: str = 'cache') -> float:
    """
    Get SRAM energy per byte based on process node and type.

    Args:
        process_node_nm: Process node in nanometers
        sram_type: 'cache' (tag + data), 'scratchpad' (data only), 'register_file'

    Returns:
        Energy per byte in picojoules
    """
    # Base SRAM energy at 7nm (reference point)
    base_sram_energy_7nm = {
        'register_file': 0.3,   # ~0.3 pJ/byte (small, fast, many ports)
        'scratchpad': 0.4,      # ~0.4 pJ/byte (no tags, simpler)
        'l1_cache': 0.8,        # ~0.8 pJ/byte (tags + data, high frequency)
        'l2_cache': 1.5,        # ~1.5 pJ/byte (larger, lower frequency)
        'l3_cache': 3.0,        # ~3.0 pJ/byte (very large, shared)
        'cache': 1.0,           # Generic cache
    }

    base_energy = base_sram_energy_7nm.get(sram_type, 1.0)

    # Scale with process node (roughly linear with node size)
    # 7nm is reference, scale up for larger nodes, down for smaller
    scale_factor = process_node_nm / 7.0

    # Apply diminishing returns for very small nodes (leakage increases)
    if process_node_nm < 5:
        scale_factor = scale_factor * 1.1  # 10% penalty for sub-5nm

    return base_energy * scale_factor


# =============================================================================
# Technology Profile (Main Configuration Class)
# =============================================================================

@dataclass
class TechnologyProfile:
    """
    Complete technology configuration combining process, memory, and market.

    This dataclass captures all the technology-dependent energy parameters
    needed by architectural energy models. Instead of hardcoding energy values,
    models can accept a TechnologyProfile and derive all energies from it.

    Example:
        profile = TechnologyProfile.create(
            process_node_nm=5,
            memory_type=MemoryType.HBM3,
            target_market="datacenter"
        )

        model = StoredProgramEnergyModel(tech_profile=profile)
    """
    # Identity
    name: str
    process_node_nm: int
    memory_type: MemoryType
    target_market: str  # "datacenter", "edge", "mobile", "automotive", "gaming"

    # Process-derived energies (pJ)
    base_alu_energy_pj: float           # FP32 ALU operation

    # Circuit-type specific compute energies (pJ)
    simd_mac_energy_pj: float           # SIMD packed MAC
    tensor_core_mac_energy_pj: float    # Tensor Core / Matrix unit MAC
    systolic_mac_energy_pj: float       # Systolic array MAC
    domain_flow_mac_energy_pj: float    # Domain flow MAC

    # Instruction pipeline energies (pJ)
    instruction_fetch_energy_pj: float
    instruction_decode_energy_pj: float
    instruction_dispatch_energy_pj: float

    # Register file energies (pJ per access)
    register_read_energy_pj: float
    register_write_energy_pj: float

    # On-chip memory energies (pJ per byte)
    sram_energy_per_byte_pj: float      # Generic SRAM / scratchpad
    l1_cache_energy_per_byte_pj: float
    l2_cache_energy_per_byte_pj: float
    l3_cache_energy_per_byte_pj: float

    # Off-chip memory energy (pJ per byte) - from MemoryTechnologySpec
    offchip_energy_per_byte_pj: float

    # Memory specification (for bandwidth/latency info)
    memory_spec: MemoryTechnologySpec = field(repr=False)

    # Control overhead energies (pJ)
    coherence_energy_per_request_pj: float = 5.0   # Cache coherence
    barrier_sync_energy_pj: float = 10.0            # Synchronization barrier
    warp_divergence_energy_pj: float = 2.0          # GPU warp divergence
    branch_mispredict_energy_pj: float = 50.0       # Branch misprediction

    # Market-specific characteristics
    typical_frequency_ghz: float = 2.0
    typical_tdp_w: float = 100.0

    @classmethod
    def create(
        cls,
        process_node_nm: int,
        memory_type: MemoryType,
        target_market: str,
        name: Optional[str] = None,
        frequency_ghz: Optional[float] = None,
        tdp_w: Optional[float] = None,
    ) -> 'TechnologyProfile':
        """
        Factory method to create a TechnologyProfile from process node and memory type.

        All energy values are derived from the process node using physics-based scaling.

        Args:
            process_node_nm: Process node in nanometers (3, 4, 5, 7, 8, 16, etc.)
            memory_type: Memory technology (DDR5, LPDDR5, HBM3, etc.)
            target_market: Target market ("datacenter", "edge", "mobile", "automotive")
            name: Optional profile name (auto-generated if None)
            frequency_ghz: Override typical frequency
            tdp_w: Override typical TDP

        Returns:
            TechnologyProfile with all energies derived from technology parameters
        """
        # Get base ALU energy from process node
        base_alu = get_process_base_energy_pj(process_node_nm)

        # Get memory specification
        mem_spec = get_memory_spec(memory_type)

        # Market-specific adjustments
        market_config = {
            'datacenter': {
                'freq': 3.5, 'tdp': 400,
                'pipeline_scale': 1.2,      # More complex pipeline
                'register_scale': 1.3,      # More ports, rename registers
                'coherence_scale': 1.5,     # Multi-socket coherence
            },
            'edge': {
                'freq': 1.5, 'tdp': 30,
                'pipeline_scale': 0.8,      # Simpler pipeline
                'register_scale': 0.9,
                'coherence_scale': 0.5,     # Limited coherence
            },
            'mobile': {
                'freq': 2.5, 'tdp': 10,
                'pipeline_scale': 0.7,      # Simple in-order cores
                'register_scale': 0.7,
                'coherence_scale': 0.3,     # Minimal coherence
            },
            'automotive': {
                'freq': 1.2, 'tdp': 25,
                'pipeline_scale': 0.9,      # Safety-critical, simpler
                'register_scale': 0.8,
                'coherence_scale': 0.6,
            },
            'gaming': {
                'freq': 2.5, 'tdp': 300,
                'pipeline_scale': 1.0,
                'register_scale': 1.0,
                'coherence_scale': 0.8,     # Single GPU, less coherence
            },
        }.get(target_market, {
            'freq': 2.0, 'tdp': 100,
            'pipeline_scale': 1.0,
            'register_scale': 1.0,
            'coherence_scale': 1.0,
        })

        # Derive instruction pipeline energies (scale with base ALU and market)
        pipeline_scale = market_config['pipeline_scale']
        instruction_fetch = base_alu * 0.375 * pipeline_scale    # ~37.5% of ALU
        instruction_decode = base_alu * 0.200 * pipeline_scale   # ~20% of ALU
        instruction_dispatch = base_alu * 0.125 * pipeline_scale # ~12.5% of ALU

        # Derive register file energies
        register_scale = market_config['register_scale']
        register_read = base_alu * 0.625 * register_scale   # ~62.5% of ALU
        register_write = base_alu * 0.750 * register_scale  # ~75% of ALU

        # Derive compute unit energies using circuit multipliers
        simd_mac = base_alu * CIRCUIT_TYPE_MULTIPLIER['simd_packed']
        tensor_mac = base_alu * CIRCUIT_TYPE_MULTIPLIER['tensor_core']
        systolic_mac = base_alu * CIRCUIT_TYPE_MULTIPLIER['systolic_mac']
        domain_flow_mac = base_alu * CIRCUIT_TYPE_MULTIPLIER['domain_flow']

        # Derive on-chip memory energies from process node
        l1_energy = get_sram_energy_per_byte_pj(process_node_nm, 'l1_cache')
        l2_energy = get_sram_energy_per_byte_pj(process_node_nm, 'l2_cache')
        l3_energy = get_sram_energy_per_byte_pj(process_node_nm, 'l3_cache')
        sram_energy = get_sram_energy_per_byte_pj(process_node_nm, 'scratchpad')

        # Control overhead energies
        coherence_scale = market_config['coherence_scale']
        coherence_energy = 5.0 * coherence_scale * (process_node_nm / 7.0)
        barrier_energy = 10.0 * (process_node_nm / 7.0)
        divergence_energy = 2.0 * (process_node_nm / 7.0)
        branch_mispredict = 50.0 * pipeline_scale * (process_node_nm / 7.0)

        # Generate name if not provided
        if name is None:
            mem_short = memory_type.value.upper().replace('_', '')
            name = f"{target_market.title()}-{process_node_nm}nm-{mem_short}"

        return cls(
            name=name,
            process_node_nm=process_node_nm,
            memory_type=memory_type,
            target_market=target_market,
            base_alu_energy_pj=base_alu,
            simd_mac_energy_pj=simd_mac,
            tensor_core_mac_energy_pj=tensor_mac,
            systolic_mac_energy_pj=systolic_mac,
            domain_flow_mac_energy_pj=domain_flow_mac,
            instruction_fetch_energy_pj=instruction_fetch,
            instruction_decode_energy_pj=instruction_decode,
            instruction_dispatch_energy_pj=instruction_dispatch,
            register_read_energy_pj=register_read,
            register_write_energy_pj=register_write,
            sram_energy_per_byte_pj=sram_energy,
            l1_cache_energy_per_byte_pj=l1_energy,
            l2_cache_energy_per_byte_pj=l2_energy,
            l3_cache_energy_per_byte_pj=l3_energy,
            offchip_energy_per_byte_pj=mem_spec.energy_per_byte_pj,
            memory_spec=mem_spec,
            coherence_energy_per_request_pj=coherence_energy,
            barrier_sync_energy_pj=barrier_energy,
            warp_divergence_energy_pj=divergence_energy,
            branch_mispredict_energy_pj=branch_mispredict,
            typical_frequency_ghz=frequency_ghz or market_config['freq'],
            typical_tdp_w=tdp_w or market_config['tdp'],
        )

    # =========================================================================
    # Convenience methods for energy model integration
    # =========================================================================

    def get_compute_energy_j(self, circuit_type: str = 'standard_cell') -> float:
        """Get compute energy per operation in Joules."""
        energies = {
            'standard_cell': self.base_alu_energy_pj,
            'simd_packed': self.simd_mac_energy_pj,
            'tensor_core': self.tensor_core_mac_energy_pj,
            'systolic_mac': self.systolic_mac_energy_pj,
            'domain_flow': self.domain_flow_mac_energy_pj,
        }
        return energies.get(circuit_type, self.base_alu_energy_pj) * 1e-12

    def get_memory_energy_j(self, level: str) -> float:
        """Get memory energy per byte in Joules."""
        levels = {
            'register': self.register_read_energy_pj / 8,  # Per byte (8 bytes per access)
            'sram': self.sram_energy_per_byte_pj,
            'scratchpad': self.sram_energy_per_byte_pj,
            'l1': self.l1_cache_energy_per_byte_pj,
            'l2': self.l2_cache_energy_per_byte_pj,
            'l3': self.l3_cache_energy_per_byte_pj,
            'dram': self.offchip_energy_per_byte_pj,
            'hbm': self.offchip_energy_per_byte_pj,
            'offchip': self.offchip_energy_per_byte_pj,
        }
        return levels.get(level, self.offchip_energy_per_byte_pj) * 1e-12

    def get_instruction_energy_j(self) -> float:
        """Get total instruction pipeline energy in Joules."""
        total_pj = (self.instruction_fetch_energy_pj +
                   self.instruction_decode_energy_pj +
                   self.instruction_dispatch_energy_pj)
        return total_pj * 1e-12

    def get_register_access_energy_j(self, is_write: bool = False) -> float:
        """Get register access energy in Joules."""
        energy_pj = self.register_write_energy_pj if is_write else self.register_read_energy_pj
        return energy_pj * 1e-12

    def summary(self) -> str:
        """Generate human-readable summary of the profile."""
        lines = [
            f"Technology Profile: {self.name}",
            f"  Process: {self.process_node_nm}nm",
            f"  Memory: {self.memory_type.value} ({self.memory_spec.description})",
            f"  Market: {self.target_market}",
            f"",
            f"  Compute Energy (pJ):",
            f"    Base ALU (FP32):    {self.base_alu_energy_pj:.2f}",
            f"    SIMD MAC:           {self.simd_mac_energy_pj:.2f}",
            f"    Tensor Core MAC:    {self.tensor_core_mac_energy_pj:.2f}",
            f"    Systolic MAC:       {self.systolic_mac_energy_pj:.2f}",
            f"    Domain Flow MAC:    {self.domain_flow_mac_energy_pj:.2f}",
            f"",
            f"  Instruction Pipeline (pJ):",
            f"    Fetch:              {self.instruction_fetch_energy_pj:.2f}",
            f"    Decode:             {self.instruction_decode_energy_pj:.2f}",
            f"    Dispatch:           {self.instruction_dispatch_energy_pj:.2f}",
            f"",
            f"  Memory Hierarchy (pJ/byte):",
            f"    SRAM/Scratchpad:    {self.sram_energy_per_byte_pj:.2f}",
            f"    L1 Cache:           {self.l1_cache_energy_per_byte_pj:.2f}",
            f"    L2 Cache:           {self.l2_cache_energy_per_byte_pj:.2f}",
            f"    L3 Cache:           {self.l3_cache_energy_per_byte_pj:.2f}",
            f"    Off-chip:           {self.offchip_energy_per_byte_pj:.2f}",
            f"",
            f"  Typical: {self.typical_frequency_ghz:.1f} GHz, {self.typical_tdp_w:.0f}W TDP",
        ]
        return "\n".join(lines)


# =============================================================================
# Preset Technology Profiles
# =============================================================================

# Datacenter profiles
DATACENTER_4NM_HBM3 = TechnologyProfile.create(
    process_node_nm=4,
    memory_type=MemoryType.HBM3,
    target_market="datacenter",
    name="Datacenter-4nm-HBM3 (H100-class)",
    frequency_ghz=2.0,
    tdp_w=700,
)

DATACENTER_4NM_HBM3E = TechnologyProfile.create(
    process_node_nm=4,
    memory_type=MemoryType.HBM3E,
    target_market="datacenter",
    name="Datacenter-4nm-HBM3e (B100-class)",
    frequency_ghz=2.1,
    tdp_w=1000,
)

DATACENTER_5NM_HBM2E = TechnologyProfile.create(
    process_node_nm=5,
    memory_type=MemoryType.HBM2E,
    target_market="datacenter",
    name="Datacenter-5nm-HBM2e (A100-class)",
    frequency_ghz=1.4,
    tdp_w=400,
)

DATACENTER_7NM_DDR5 = TechnologyProfile.create(
    process_node_nm=7,
    memory_type=MemoryType.DDR5,
    target_market="datacenter",
    name="Datacenter-7nm-DDR5 (Server CPU)",
    frequency_ghz=3.5,
    tdp_w=350,
)

# Edge profiles
EDGE_8NM_LPDDR5 = TechnologyProfile.create(
    process_node_nm=8,
    memory_type=MemoryType.LPDDR5,
    target_market="edge",
    name="Edge-8nm-LPDDR5 (Jetson Orin-class)",
    frequency_ghz=1.3,
    tdp_w=60,
)

EDGE_8NM_LPDDR5X = TechnologyProfile.create(
    process_node_nm=8,
    memory_type=MemoryType.LPDDR5X,
    target_market="edge",
    name="Edge-8nm-LPDDR5X (Jetson Thor-class)",
    frequency_ghz=1.5,
    tdp_w=100,
)

EDGE_16NM_LPDDR4 = TechnologyProfile.create(
    process_node_nm=16,
    memory_type=MemoryType.LPDDR4,
    target_market="edge",
    name="Edge-16nm-LPDDR4 (Jetson Xavier-class)",
    frequency_ghz=1.0,
    tdp_w=30,
)

# Mobile profiles
MOBILE_5NM_LPDDR5X = TechnologyProfile.create(
    process_node_nm=5,
    memory_type=MemoryType.LPDDR5X,
    target_market="mobile",
    name="Mobile-5nm-LPDDR5X (Snapdragon 8 Gen3-class)",
    frequency_ghz=3.3,
    tdp_w=12,
)

MOBILE_4NM_LPDDR5X = TechnologyProfile.create(
    process_node_nm=4,
    memory_type=MemoryType.LPDDR5X,
    target_market="mobile",
    name="Mobile-4nm-LPDDR5X (Apple M3-class)",
    frequency_ghz=4.0,
    tdp_w=15,
)

# Automotive profiles
AUTOMOTIVE_7NM_LPDDR5 = TechnologyProfile.create(
    process_node_nm=7,
    memory_type=MemoryType.LPDDR5,
    target_market="automotive",
    name="Automotive-7nm-LPDDR5 (Drive Orin-class)",
    frequency_ghz=1.4,
    tdp_w=45,
)

AUTOMOTIVE_16NM_LPDDR4 = TechnologyProfile.create(
    process_node_nm=16,
    memory_type=MemoryType.LPDDR4,
    target_market="automotive",
    name="Automotive-16nm-LPDDR4 (Drive Xavier-class)",
    frequency_ghz=1.0,
    tdp_w=30,
)

AUTOMOTIVE_22NM_DDR4 = TechnologyProfile.create(
    process_node_nm=22,
    memory_type=MemoryType.DDR4,
    target_market="automotive",
    name="Automotive-22nm-DDR4 (Safety MCU)",
    frequency_ghz=0.4,
    tdp_w=5,
)

# Gaming profiles (discrete GPU)
GAMING_8NM_GDDR6 = TechnologyProfile.create(
    process_node_nm=8,
    memory_type=MemoryType.GDDR6,
    target_market="gaming",
    name="Gaming-8nm-GDDR6 (RTX 3080-class)",
    frequency_ghz=1.7,
    tdp_w=320,
)

GAMING_5NM_GDDR6X = TechnologyProfile.create(
    process_node_nm=5,
    memory_type=MemoryType.GDDR6X,
    target_market="gaming",
    name="Gaming-5nm-GDDR6X (RTX 4090-class)",
    frequency_ghz=2.5,
    tdp_w=450,
)


# Profile registry for easy lookup
TECHNOLOGY_PROFILES = {
    # Datacenter
    'datacenter-4nm-hbm3': DATACENTER_4NM_HBM3,
    'datacenter-4nm-hbm3e': DATACENTER_4NM_HBM3E,
    'datacenter-5nm-hbm2e': DATACENTER_5NM_HBM2E,
    'datacenter-7nm-ddr5': DATACENTER_7NM_DDR5,

    # Edge
    'edge-8nm-lpddr5': EDGE_8NM_LPDDR5,
    'edge-8nm-lpddr5x': EDGE_8NM_LPDDR5X,
    'edge-16nm-lpddr4': EDGE_16NM_LPDDR4,

    # Mobile
    'mobile-5nm-lpddr5x': MOBILE_5NM_LPDDR5X,
    'mobile-4nm-lpddr5x': MOBILE_4NM_LPDDR5X,

    # Automotive
    'automotive-7nm-lpddr5': AUTOMOTIVE_7NM_LPDDR5,
    'automotive-16nm-lpddr4': AUTOMOTIVE_16NM_LPDDR4,
    'automotive-22nm-ddr4': AUTOMOTIVE_22NM_DDR4,

    # Gaming
    'gaming-8nm-gddr6': GAMING_8NM_GDDR6,
    'gaming-5nm-gddr6x': GAMING_5NM_GDDR6X,
}


# =============================================================================
# Architecture Comparison Sets
# =============================================================================
# For fair comparison of architectural energy efficiency, we need to:
# 1. Fix the process node (same transistor technology)
# 2. Fix the memory technology (same off-chip access cost)
# 3. Apply architecture-appropriate circuit multipliers
#
# This isolates the energy differences due to:
# - Resource contention mechanisms (instruction fetch, coherence, scheduling)
# - Datapath organization (von Neumann vs systolic vs spatial)
# - Control overhead (per-op instruction vs amortized vs none)
#
# Goal: Answer "Which architecture is most energy-efficient for this workload?"

@dataclass
class ArchitectureComparisonSet:
    """
    A set of technology profiles for fair architecture comparison.

    All architectures share the same process node and memory technology.
    Each architecture uses its appropriate circuit type multiplier:
    - CPU: x86_performance (2.5x) or arm_efficiency (1.0x)
    - GPU: tensor_core (0.85x)
    - TPU: systolic_mac (0.80x)
    - KPU: domain_flow (0.75x)

    This enables apples-to-apples comparison of architectural efficiency.
    """
    name: str
    description: str
    process_node_nm: int
    memory_type: MemoryType

    # CPU circuit type (determines energy multiplier)
    cpu_circuit_type: str  # 'x86_performance', 'x86_efficiency', 'arm_performance', 'arm_efficiency'

    # Architecture-specific profiles (computed)
    cpu_profile: TechnologyProfile = field(init=False)
    gpu_profile: TechnologyProfile = field(init=False)
    tpu_profile: TechnologyProfile = field(init=False)
    kpu_profile: TechnologyProfile = field(init=False)

    # Base energy at this process node
    base_alu_energy_pj: float = field(init=False)

    def __post_init__(self):
        """Compute architecture profiles from process node and memory type."""
        self.base_alu_energy_pj = get_process_base_energy_pj(self.process_node_nm)
        mem_spec = get_memory_spec(self.memory_type)

        # CPU profile with specified circuit type
        cpu_multiplier = CIRCUIT_TYPE_MULTIPLIER.get(self.cpu_circuit_type, 1.0)
        self.cpu_profile = self._create_profile(
            name=f"CPU-{self.cpu_circuit_type}",
            circuit_multiplier=cpu_multiplier,
            mem_spec=mem_spec,
            pipeline_scale=1.2 if 'performance' in self.cpu_circuit_type else 0.8,
            register_scale=1.2 if 'performance' in self.cpu_circuit_type else 0.9,
        )

        # GPU profile with tensor core
        self.gpu_profile = self._create_profile(
            name="GPU-TensorCore",
            circuit_multiplier=CIRCUIT_TYPE_MULTIPLIER['tensor_core'],
            mem_spec=mem_spec,
            pipeline_scale=1.0,
            register_scale=1.0,
        )

        # TPU profile with systolic MAC
        self.tpu_profile = self._create_profile(
            name="TPU-Systolic",
            circuit_multiplier=CIRCUIT_TYPE_MULTIPLIER['systolic_mac'],
            mem_spec=mem_spec,
            pipeline_scale=0.3,  # Minimal instruction pipeline
            register_scale=0.5,  # Weight buffers simpler than register files
        )

        # KPU profile with domain flow
        self.kpu_profile = self._create_profile(
            name="KPU-DomainFlow",
            circuit_multiplier=CIRCUIT_TYPE_MULTIPLIER['domain_flow'],
            mem_spec=mem_spec,
            pipeline_scale=0.2,  # Configuration only
            register_scale=0.4,  # Tile buffers
        )

    def _create_profile(
        self,
        name: str,
        circuit_multiplier: float,
        mem_spec: MemoryTechnologySpec,
        pipeline_scale: float,
        register_scale: float,
    ) -> TechnologyProfile:
        """Create a technology profile with specified circuit multiplier."""
        base = self.base_alu_energy_pj
        alu_energy = base * circuit_multiplier

        return TechnologyProfile(
            name=name,
            process_node_nm=self.process_node_nm,
            memory_type=self.memory_type,
            target_market="architecture_comparison",
            base_alu_energy_pj=alu_energy,
            simd_mac_energy_pj=alu_energy * 0.90,
            tensor_core_mac_energy_pj=base * CIRCUIT_TYPE_MULTIPLIER['tensor_core'],
            systolic_mac_energy_pj=base * CIRCUIT_TYPE_MULTIPLIER['systolic_mac'],
            domain_flow_mac_energy_pj=base * CIRCUIT_TYPE_MULTIPLIER['domain_flow'],
            instruction_fetch_energy_pj=alu_energy * 0.375 * pipeline_scale,
            instruction_decode_energy_pj=alu_energy * 0.200 * pipeline_scale,
            instruction_dispatch_energy_pj=alu_energy * 0.125 * pipeline_scale,
            register_read_energy_pj=alu_energy * 0.625 * register_scale,
            register_write_energy_pj=alu_energy * 0.750 * register_scale,
            sram_energy_per_byte_pj=get_sram_energy_per_byte_pj(self.process_node_nm, 'scratchpad'),
            l1_cache_energy_per_byte_pj=get_sram_energy_per_byte_pj(self.process_node_nm, 'l1_cache'),
            l2_cache_energy_per_byte_pj=get_sram_energy_per_byte_pj(self.process_node_nm, 'l2_cache'),
            l3_cache_energy_per_byte_pj=get_sram_energy_per_byte_pj(self.process_node_nm, 'l3_cache'),
            offchip_energy_per_byte_pj=mem_spec.energy_per_byte_pj,
            memory_spec=mem_spec,
            coherence_energy_per_request_pj=5.0 * (self.process_node_nm / 7.0),
            barrier_sync_energy_pj=10.0 * (self.process_node_nm / 7.0),
            warp_divergence_energy_pj=2.0 * (self.process_node_nm / 7.0),
            branch_mispredict_energy_pj=50.0 * pipeline_scale * (self.process_node_nm / 7.0),
            typical_frequency_ghz=1.0,
            typical_tdp_w=50,
        )

    def summary(self) -> str:
        """Generate summary showing architecture energy comparison."""
        cpu_mult = CIRCUIT_TYPE_MULTIPLIER.get(self.cpu_circuit_type, 1.0)
        lines = [
            f"Architecture Comparison Set: {self.name}",
            f"  {self.description}",
            f"",
            f"  Process: {self.process_node_nm}nm",
            f"  Memory: {self.memory_type.value}",
            f"  Base ALU Energy: {self.base_alu_energy_pj:.2f} pJ",
            f"",
            f"  Circuit Type Multipliers:",
            f"    CPU ({self.cpu_circuit_type}): {cpu_mult:.2f}x -> {self.cpu_profile.base_alu_energy_pj:.2f} pJ/op",
            f"    GPU (tensor_core):    {CIRCUIT_TYPE_MULTIPLIER['tensor_core']:.2f}x -> {self.gpu_profile.tensor_core_mac_energy_pj:.2f} pJ/op",
            f"    TPU (systolic_mac):   {CIRCUIT_TYPE_MULTIPLIER['systolic_mac']:.2f}x -> {self.tpu_profile.systolic_mac_energy_pj:.2f} pJ/op",
            f"    KPU (domain_flow):    {CIRCUIT_TYPE_MULTIPLIER['domain_flow']:.2f}x -> {self.kpu_profile.domain_flow_mac_energy_pj:.2f} pJ/op",
        ]
        return "\n".join(lines)


# Backward compatibility alias
ProductCategoryProfiles = ArchitectureComparisonSet


# =============================================================================
# Predefined Architecture Comparison Sets
# =============================================================================
# These provide ready-to-use comparison sets for common scenarios.

# 8nm LPDDR5 comparison with x86 performance CPU (Intel i7 NUC-class)
ARCH_COMPARISON_8NM_X86 = ArchitectureComparisonSet(
    name="8nm-x86-lpddr5",
    description="8nm process with x86 performance CPU (i7-class) vs GPU/TPU/KPU",
    process_node_nm=8,
    memory_type=MemoryType.LPDDR5,
    cpu_circuit_type='x86_performance',
)

# 8nm LPDDR5 comparison with ARM efficiency CPU (Cortex-A class)
ARCH_COMPARISON_8NM_ARM = ArchitectureComparisonSet(
    name="8nm-arm-lpddr5",
    description="8nm process with ARM efficiency CPU (Cortex-A class) vs GPU/TPU/KPU",
    process_node_nm=8,
    memory_type=MemoryType.LPDDR5,
    cpu_circuit_type='arm_efficiency',
)

# 4nm HBM3 datacenter comparison
ARCH_COMPARISON_4NM_DC = ArchitectureComparisonSet(
    name="4nm-datacenter-hbm3",
    description="4nm datacenter with x86 performance CPU vs GPU/TPU/KPU",
    process_node_nm=4,
    memory_type=MemoryType.HBM3,
    cpu_circuit_type='x86_performance',
)

# 4nm LPDDR5X mobile comparison with ARM performance CPU
ARCH_COMPARISON_4NM_MOBILE = ArchitectureComparisonSet(
    name="4nm-mobile-lpddr5x",
    description="4nm mobile with ARM performance CPU vs GPU/TPU/KPU",
    process_node_nm=4,
    memory_type=MemoryType.LPDDR5X,
    cpu_circuit_type='arm_performance',
)

# Registry of architecture comparison sets
ARCHITECTURE_COMPARISON_SETS = {
    '8nm-x86': ARCH_COMPARISON_8NM_X86,
    '8nm-arm': ARCH_COMPARISON_8NM_ARM,
    '4nm-datacenter': ARCH_COMPARISON_4NM_DC,
    '4nm-mobile': ARCH_COMPARISON_4NM_MOBILE,
}


def get_architecture_comparison_set(name: str) -> ArchitectureComparisonSet:
    """
    Get a predefined architecture comparison set.

    Args:
        name: Set name ('8nm-x86', '8nm-arm', '4nm-datacenter', '4nm-mobile')

    Returns:
        ArchitectureComparisonSet

    Raises:
        KeyError if not found
    """
    key = name.lower().replace(' ', '-').replace('_', '-')
    if key not in ARCHITECTURE_COMPARISON_SETS:
        available = ', '.join(sorted(ARCHITECTURE_COMPARISON_SETS.keys()))
        raise KeyError(f"Comparison set '{name}' not found. Available: {available}")
    return ARCHITECTURE_COMPARISON_SETS[key]


def create_architecture_comparison_set(
    process_node_nm: int,
    memory_type: MemoryType,
    cpu_circuit_type: str = 'x86_performance',
    name: str = None,
) -> ArchitectureComparisonSet:
    """
    Create a custom architecture comparison set.

    Args:
        process_node_nm: Process node in nanometers (3, 4, 5, 7, 8, etc.)
        memory_type: Memory technology (LPDDR5, HBM3, etc.)
        cpu_circuit_type: CPU circuit type ('x86_performance', 'arm_efficiency', etc.)
        name: Optional name for the set

    Returns:
        ArchitectureComparisonSet for fair architecture comparison
    """
    if name is None:
        name = f"{process_node_nm}nm-{cpu_circuit_type}-{memory_type.value}"

    return ArchitectureComparisonSet(
        name=name,
        description=f"{process_node_nm}nm with {cpu_circuit_type} CPU",
        process_node_nm=process_node_nm,
        memory_type=memory_type,
        cpu_circuit_type=cpu_circuit_type,
    )


# =============================================================================
# Legacy Product Category Support (deprecated - use ArchitectureComparisonSet)
# =============================================================================

def _create_category_cpu_profile(
    process_node_nm: int,
    memory_type: MemoryType,
    target_market: str,
    is_datacenter: bool = False,
    name: str = "",
    frequency_ghz: float = None,
    tdp_w: float = None,
) -> TechnologyProfile:
    """
    Create CPU profile with appropriate circuit design.

    DEPRECATED: Use ArchitectureComparisonSet instead for fair comparisons.
    """
    base_alu = get_process_base_energy_pj(process_node_nm)
    mem_spec = get_memory_spec(memory_type)

    # Use new circuit type multipliers
    if is_datacenter:
        alu_energy = base_alu * CIRCUIT_TYPE_MULTIPLIER['x86_performance']
        pipeline_scale = 1.2  # Complex pipeline
        register_scale = 1.3  # Many rename registers
        coherence_scale = 1.5  # Multi-socket coherence
    else:
        alu_energy = base_alu  # Standard cells
        pipeline_scale = 0.8  # Simpler pipeline
        register_scale = 0.9
        coherence_scale = 0.5

    # Derive all other energies from base
    return TechnologyProfile(
        name=name or f"CPU-{process_node_nm}nm-{target_market}",
        process_node_nm=process_node_nm,
        memory_type=memory_type,
        target_market=target_market,
        base_alu_energy_pj=alu_energy,
        simd_mac_energy_pj=alu_energy * 0.90,  # SIMD is 10% better
        tensor_core_mac_energy_pj=alu_energy * 0.85,  # If CPU had tensor
        systolic_mac_energy_pj=alu_energy * 0.80,
        domain_flow_mac_energy_pj=alu_energy * 0.75,
        instruction_fetch_energy_pj=alu_energy * 0.375 * pipeline_scale,
        instruction_decode_energy_pj=alu_energy * 0.200 * pipeline_scale,
        instruction_dispatch_energy_pj=alu_energy * 0.125 * pipeline_scale,
        register_read_energy_pj=alu_energy * 0.625 * register_scale,
        register_write_energy_pj=alu_energy * 0.750 * register_scale,
        sram_energy_per_byte_pj=get_sram_energy_per_byte_pj(process_node_nm, 'scratchpad'),
        l1_cache_energy_per_byte_pj=get_sram_energy_per_byte_pj(process_node_nm, 'l1_cache'),
        l2_cache_energy_per_byte_pj=get_sram_energy_per_byte_pj(process_node_nm, 'l2_cache'),
        l3_cache_energy_per_byte_pj=get_sram_energy_per_byte_pj(process_node_nm, 'l3_cache'),
        offchip_energy_per_byte_pj=mem_spec.energy_per_byte_pj,
        memory_spec=mem_spec,
        coherence_energy_per_request_pj=5.0 * coherence_scale * (process_node_nm / 7.0),
        barrier_sync_energy_pj=10.0 * (process_node_nm / 7.0),
        warp_divergence_energy_pj=2.0 * (process_node_nm / 7.0),
        branch_mispredict_energy_pj=50.0 * pipeline_scale * (process_node_nm / 7.0),
        typical_frequency_ghz=frequency_ghz or (3.5 if is_datacenter else 1.5),
        typical_tdp_w=tdp_w or (350 if is_datacenter else 30),
    )


def _create_category_gpu_profile(
    process_node_nm: int,
    memory_type: MemoryType,
    target_market: str,
    name: str = "",
    frequency_ghz: float = None,
    tdp_w: float = None,
) -> TechnologyProfile:
    """
    Create GPU profile with Tensor Core circuit design.

    GPUs use tensor_core multiplier (0.85) for their primary compute.
    The SIMT overhead (warp scheduling, coherence) is handled separately
    in the energy model.
    """
    base_alu = get_process_base_energy_pj(process_node_nm)
    mem_spec = get_memory_spec(memory_type)

    return TechnologyProfile(
        name=name or f"GPU-{process_node_nm}nm-{target_market}",
        process_node_nm=process_node_nm,
        memory_type=memory_type,
        target_market=target_market,
        base_alu_energy_pj=base_alu,
        simd_mac_energy_pj=base_alu * CIRCUIT_TYPE_MULTIPLIER['simd_packed'],
        tensor_core_mac_energy_pj=base_alu * CIRCUIT_TYPE_MULTIPLIER['tensor_core'],
        systolic_mac_energy_pj=base_alu * CIRCUIT_TYPE_MULTIPLIER['systolic_mac'],
        domain_flow_mac_energy_pj=base_alu * CIRCUIT_TYPE_MULTIPLIER['domain_flow'],
        instruction_fetch_energy_pj=base_alu * 0.375,
        instruction_decode_energy_pj=base_alu * 0.200,
        instruction_dispatch_energy_pj=base_alu * 0.125,
        register_read_energy_pj=base_alu * 0.625,
        register_write_energy_pj=base_alu * 0.750,
        sram_energy_per_byte_pj=get_sram_energy_per_byte_pj(process_node_nm, 'scratchpad'),
        l1_cache_energy_per_byte_pj=get_sram_energy_per_byte_pj(process_node_nm, 'l1_cache'),
        l2_cache_energy_per_byte_pj=get_sram_energy_per_byte_pj(process_node_nm, 'l2_cache'),
        l3_cache_energy_per_byte_pj=get_sram_energy_per_byte_pj(process_node_nm, 'l3_cache'),
        offchip_energy_per_byte_pj=mem_spec.energy_per_byte_pj,
        memory_spec=mem_spec,
        coherence_energy_per_request_pj=5.0 * (process_node_nm / 7.0),
        barrier_sync_energy_pj=10.0 * (process_node_nm / 7.0),
        warp_divergence_energy_pj=2.0 * (process_node_nm / 7.0),
        branch_mispredict_energy_pj=50.0 * (process_node_nm / 7.0),
        typical_frequency_ghz=frequency_ghz or 2.0,
        typical_tdp_w=tdp_w or 400,
    )


def _create_category_tpu_profile(
    process_node_nm: int,
    memory_type: MemoryType,
    target_market: str,
    name: str = "",
    frequency_ghz: float = None,
    tdp_w: float = None,
) -> TechnologyProfile:
    """
    Create TPU profile with systolic array circuit design.

    TPUs eliminate instruction fetch entirely and use highly optimized
    systolic MAC units (systolic_mac multiplier = 0.80).
    """
    base_alu = get_process_base_energy_pj(process_node_nm)
    mem_spec = get_memory_spec(memory_type)

    return TechnologyProfile(
        name=name or f"TPU-{process_node_nm}nm-{target_market}",
        process_node_nm=process_node_nm,
        memory_type=memory_type,
        target_market=target_market,
        base_alu_energy_pj=base_alu,
        simd_mac_energy_pj=base_alu * CIRCUIT_TYPE_MULTIPLIER['simd_packed'],
        tensor_core_mac_energy_pj=base_alu * CIRCUIT_TYPE_MULTIPLIER['tensor_core'],
        systolic_mac_energy_pj=base_alu * CIRCUIT_TYPE_MULTIPLIER['systolic_mac'],
        domain_flow_mac_energy_pj=base_alu * CIRCUIT_TYPE_MULTIPLIER['domain_flow'],
        # TPU has minimal instruction pipeline (no per-op fetch)
        instruction_fetch_energy_pj=base_alu * 0.1,  # Minimal control
        instruction_decode_energy_pj=base_alu * 0.1,
        instruction_dispatch_energy_pj=base_alu * 0.05,
        register_read_energy_pj=base_alu * 0.5,  # Weight buffers
        register_write_energy_pj=base_alu * 0.5,
        sram_energy_per_byte_pj=get_sram_energy_per_byte_pj(process_node_nm, 'scratchpad'),
        l1_cache_energy_per_byte_pj=get_sram_energy_per_byte_pj(process_node_nm, 'l1_cache'),
        l2_cache_energy_per_byte_pj=get_sram_energy_per_byte_pj(process_node_nm, 'l2_cache'),
        l3_cache_energy_per_byte_pj=get_sram_energy_per_byte_pj(process_node_nm, 'l3_cache'),
        offchip_energy_per_byte_pj=mem_spec.energy_per_byte_pj,
        memory_spec=mem_spec,
        coherence_energy_per_request_pj=1.0 * (process_node_nm / 7.0),  # Minimal coherence
        barrier_sync_energy_pj=5.0 * (process_node_nm / 7.0),
        warp_divergence_energy_pj=0.0,  # No warps
        branch_mispredict_energy_pj=0.0,  # No branches
        typical_frequency_ghz=frequency_ghz or 1.0,
        typical_tdp_w=tdp_w or 200,
    )


def _create_category_kpu_profile(
    process_node_nm: int,
    memory_type: MemoryType,
    target_market: str,
    name: str = "",
    frequency_ghz: float = None,
    tdp_w: float = None,
) -> TechnologyProfile:
    """
    Create KPU profile with domain flow circuit design.

    KPUs use the most efficient MAC design (domain_flow multiplier = 0.75)
    as they eliminate all instruction overhead and use streaming dataflow.
    """
    base_alu = get_process_base_energy_pj(process_node_nm)
    mem_spec = get_memory_spec(memory_type)

    return TechnologyProfile(
        name=name or f"KPU-{process_node_nm}nm-{target_market}",
        process_node_nm=process_node_nm,
        memory_type=memory_type,
        target_market=target_market,
        base_alu_energy_pj=base_alu,
        simd_mac_energy_pj=base_alu * CIRCUIT_TYPE_MULTIPLIER['simd_packed'],
        tensor_core_mac_energy_pj=base_alu * CIRCUIT_TYPE_MULTIPLIER['tensor_core'],
        systolic_mac_energy_pj=base_alu * CIRCUIT_TYPE_MULTIPLIER['systolic_mac'],
        domain_flow_mac_energy_pj=base_alu * CIRCUIT_TYPE_MULTIPLIER['domain_flow'],
        # KPU has minimal instruction overhead (domain program loaded once)
        instruction_fetch_energy_pj=base_alu * 0.05,  # Configuration only
        instruction_decode_energy_pj=base_alu * 0.05,
        instruction_dispatch_energy_pj=base_alu * 0.02,
        register_read_energy_pj=base_alu * 0.4,  # Tile buffers
        register_write_energy_pj=base_alu * 0.4,
        sram_energy_per_byte_pj=get_sram_energy_per_byte_pj(process_node_nm, 'scratchpad'),
        l1_cache_energy_per_byte_pj=get_sram_energy_per_byte_pj(process_node_nm, 'l1_cache'),
        l2_cache_energy_per_byte_pj=get_sram_energy_per_byte_pj(process_node_nm, 'l2_cache'),
        l3_cache_energy_per_byte_pj=get_sram_energy_per_byte_pj(process_node_nm, 'l3_cache'),
        offchip_energy_per_byte_pj=mem_spec.energy_per_byte_pj,
        memory_spec=mem_spec,
        coherence_energy_per_request_pj=0.5 * (process_node_nm / 7.0),  # Minimal
        barrier_sync_energy_pj=2.0 * (process_node_nm / 7.0),  # Domain barriers
        warp_divergence_energy_pj=0.0,  # No warps
        branch_mispredict_energy_pj=0.0,  # No branches
        typical_frequency_ghz=frequency_ghz or 1.0,
        typical_tdp_w=tdp_w or 30,
    )


# =============================================================================
# Legacy Product Categories (using new ArchitectureComparisonSet)
# =============================================================================
# These map old category names to new ArchitectureComparisonSets for
# backward compatibility.

DATACENTER_CATEGORY = ARCH_COMPARISON_4NM_DC

EMBODIED_AI_CATEGORY = ARCH_COMPARISON_8NM_X86

EDGE_AI_CATEGORY = ARCH_COMPARISON_8NM_ARM

MOBILE_CATEGORY = ARCH_COMPARISON_4NM_MOBILE



# Product category registry
PRODUCT_CATEGORIES = {
    'datacenter': DATACENTER_CATEGORY,
    'embodied_ai': EMBODIED_AI_CATEGORY,
    'edge_ai': EDGE_AI_CATEGORY,
    'mobile': MOBILE_CATEGORY,
}


def get_product_category(name: str) -> ProductCategoryProfiles:
    """
    Get a product category by name.

    Args:
        name: Category name (datacenter, embodied_ai, edge_ai, mobile)

    Returns:
        ProductCategoryProfiles

    Raises:
        KeyError if category not found
    """
    key = name.lower().replace(' ', '_').replace('-', '_')
    if key not in PRODUCT_CATEGORIES:
        available = ', '.join(sorted(PRODUCT_CATEGORIES.keys()))
        raise KeyError(f"Category '{name}' not found. Available: {available}")
    return PRODUCT_CATEGORIES[key]


def list_product_categories() -> Dict[str, ProductCategoryProfiles]:
    """Return all available product categories."""
    return PRODUCT_CATEGORIES.copy()


# =============================================================================
# DEFAULT PROFILE
# =============================================================================
# This profile should be used when no specific profile is specified.
# It matches the original hardcoded defaults in the energy models (7nm process,
# datacenter-class memory bandwidth).
#
# All energy models REQUIRE a TechnologyProfile - use this as the default.

DEFAULT_PROFILE = DATACENTER_7NM_DDR5


def get_profile(name: str) -> TechnologyProfile:
    """
    Get a preset technology profile by name.

    Args:
        name: Profile name (case-insensitive)

    Returns:
        TechnologyProfile

    Raises:
        KeyError if profile not found
    """
    key = name.lower().replace(' ', '-').replace('_', '-')
    if key not in TECHNOLOGY_PROFILES:
        available = ', '.join(sorted(TECHNOLOGY_PROFILES.keys()))
        raise KeyError(f"Profile '{name}' not found. Available: {available}")
    return TECHNOLOGY_PROFILES[key]


def list_profiles() -> Dict[str, TechnologyProfile]:
    """Return all available preset profiles."""
    return TECHNOLOGY_PROFILES.copy()
