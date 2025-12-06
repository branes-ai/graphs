"""
SoC Infrastructure Model - Non-Linear TDP Components

This module models the infrastructure overhead required to support ALUs in a
real System-on-Chip (SoC). The key insight is that TDP does NOT scale linearly
with ALU count because:

1. On-chip SRAM must scale to keep matmul compute-bound
2. Interconnect wires get longer as chip area grows
3. Control overhead exists per cluster/tile
4. Idle/leakage power sets a floor regardless of utilization

Physics basis:
    TDP_total = P_compute + P_sram + P_interconnect + P_control + P_idle

    Where:
    - P_compute scales linearly with ALUs
    - P_sram scales linearly with ALUs (plus ~30% leakage)
    - P_interconnect scales SUPER-linearly (wire length grows)
    - P_control scales sub-linearly (amortized over clusters)
    - P_idle = TDP_envelope x 0.5 (constant floor)

Usage:
    from graphs.hardware.soc_infrastructure import (
        SoCInfrastructureModel,
        InterconnectTopology,
        estimate_infrastructure_power,
    )

    model = SoCInfrastructureModel(
        num_alus=16384,
        process_node_nm=4,
        topology=InterconnectTopology.MESH_2D,
    )
    breakdown = model.compute_power_breakdown(frequency_ghz=1.5)
"""

from dataclasses import dataclass
from enum import Enum
from math import sqrt, log2
from typing import Dict, Optional

from .technology_profile import (
    PROCESS_NODE_BASE_ENERGY_PJ,
    CIRCUIT_TYPE_MULTIPLIER,
    get_process_base_energy_pj,
)


# =============================================================================
# Interconnect Topology
# =============================================================================

class InterconnectTopology(Enum):
    """
    Network-on-Chip (NoC) topology options.

    Each topology has different trade-offs:
    - Bisection bandwidth: How much data can cross the chip midpoint
    - Wire cost: Total wire length (affects power and area)
    - Latency: Hops to traverse the network
    - Best use case: What traffic patterns it handles well
    """
    HTREE = "h_tree"        # Broadcast/reduction, O(N log N) wires
    CROSSBAR = "crossbar"   # All-to-all, O(N^2) wires, only for small N
    MESH_2D = "mesh_2d"     # Spatial locality, O(N) wires, O(sqrt(N)) latency
    CLOS = "clos"           # Rearrangeably non-blocking, O(N log N) wires


# =============================================================================
# Constants
# =============================================================================

# SRAM sizing: bytes of on-chip memory per ALU to keep GEMM compute-bound
# Based on: TPU v4 (32 MiB / 32K MACs = 1KB), H100 (256KB / 512 TC = 512B)
SRAM_BYTES_PER_ALU = 768  # Conservative middle ground

# Wire energy at 5nm reference node (pJ per mm of wire traversal)
# Source: Horowitz ISSCC 2014, scaled to modern nodes
WIRE_ENERGY_PJ_PER_MM_5NM = 0.5

# Base ALU area at 5nm reference (mm^2 per ALU for minimal MAC unit)
ALU_AREA_MM2_5NM = 0.0001  # 0.0001 mm^2 = 100 um^2 per minimal ALU

# Area multiplier by circuit type (relative to minimal ALU)
# This captures the "effective area per ALU" including local resources.
# CPUs have large cores with OoO, register files, L1 cache per core.
# GPUs have simpler cores with shared resources.
# Systolic arrays have very dense MAC packing.
#
# Calibration target: ~30% compute area fraction for x86 at 16K ALUs
# With 16K ALUs * 0.0001mm^2 = 1.64 mm^2 base area, SRAM = 2.0 mm^2
# For 30% fraction: compute / (compute + sram) = 0.30
# compute = 0.30 * (compute + 2.0) => compute = 0.857 mm^2
# But we need to account for control/interconnect area too.
# More realistically: compute_area should be ~2-3 mm^2 at 5nm for 16K ALUs
CIRCUIT_AREA_MULTIPLIER = {
    'x86_performance': 8.0,    # ~800 um^2 per FP ALU (core slice with caches)
    'x86_efficiency': 5.0,     # Smaller in-order core
    'arm_performance': 6.0,    # Simpler than x86 but still complex
    'arm_efficiency': 3.0,     # Much simpler
    'cuda_core': 2.5,          # Simple SIMT core with shared register file
    'tensor_core': 2.0,        # Dense matrix units with shared resources
    'simd_packed': 4.0,        # SIMD with control overhead
    'systolic_mac': 1.0,       # Minimal: just MAC + registers
    'domain_flow': 1.2,        # Slightly more than systolic (domain routing)
    'standard_cell': 2.0,      # Generic
}

# SRAM area density by process node (Mb/mm^2)
# Source: docs/chip_area_estimates.md
SRAM_DENSITY_MB_PER_MM2 = {
    3: 38.0,   # GAA nanosheets
    4: 38.0,   # FinFET
    5: 48.0,   # Peak SRAM density
    7: 45.0,
    8: 40.0,
    12: 32.0,
    14: 20.0,
    16: 18.0,
    28: 12.0,
}

# SRAM energy per byte by process node (pJ/byte for read)
# Source: docs/chip_area_estimates.md
SRAM_ENERGY_PJ_PER_BYTE = {
    3: 0.25,
    4: 0.28,
    5: 0.30,
    7: 0.35,
    8: 0.38,
    12: 0.45,
    14: 0.50,
    16: 0.55,
    28: 0.65,
}

# Control energy per ALU per cycle by circuit type (pJ)
# Captures scheduling, coherence, and coordination overhead
CONTROL_ENERGY_PJ_PER_ALU = {
    'x86_performance': 2.0,    # Heavy OoO, branch prediction, coherence
    'x86_efficiency': 1.0,     # Simpler in-order
    'arm_performance': 1.5,
    'arm_efficiency': 0.5,
    'cuda_core': 1.5,          # Warp scheduling, coherence
    'tensor_core': 1.0,        # Shared warp scheduler
    'simd_packed': 0.8,
    'systolic_mac': 0.2,       # Fixed schedule, minimal control
    'domain_flow': 0.3,        # Domain tracking
    'standard_cell': 0.5,
}

# ALUs per control cluster by circuit type
# Larger clusters = more amortization of control overhead
ALUS_PER_CLUSTER = {
    'x86_performance': 4,      # 4 ALUs per core
    'x86_efficiency': 2,
    'arm_performance': 4,
    'arm_efficiency': 2,
    'cuda_core': 32,           # 32 per warp
    'tensor_core': 64,         # 64 MACs per tensor core
    'simd_packed': 8,          # 8-wide SIMD
    'systolic_mac': 256,       # 16x16 array
    'domain_flow': 256,        # 16x16 tile
    'standard_cell': 16,
}

# =============================================================================
# Compute Granularity Presets
# =============================================================================
# Real hardware uses hierarchical building blocks, not individual ALUs.
# The interconnect scales with the number of clusters, not ALUs.
#
# NVIDIA GPU Hierarchy (H100 Hopper Architecture):
#   GPU -> GPC (8) -> TPC (9/GPC) -> SM (2/TPC) -> CUDA cores (128) + Tensor Cores (4x64)
#
#   Acronyms:
#     GPC = Graphics Processing Cluster (highest-level compute partition)
#     TPC = Texture Processing Cluster (contains 2 SMs, handles texture sampling)
#     SM  = Streaming Multiprocessor (basic execution unit with warps)
#
#   Interconnect: GPC-to-L2 partitioned crossbar, SM-to-SM network within GPC
#
# Google TPU Hierarchy (v4):
#   Chip -> TensorCore (2) -> MXU (128x128 systolic array)
#
#   Acronyms:
#     MXU = Matrix Multiply Unit (systolic array for matrix operations)
#
#   Interconnect: On-chip buffer to MXU, ICI (Inter-Chip Interconnect) for pod
#
# Intel/AMD CPU Hierarchy:
#   Die -> CCX/Tile -> Core (many) -> ALUs (4-8 per core)
#
#   Acronyms:
#     CCX = Core Complex (AMD Zen, group of cores sharing L3)
#     CCD = Core Chiplet Die (AMD, contains CCXs)
#
#   Interconnect: Ring bus or mesh connecting cores to LLC (Last Level Cache)

@dataclass
class ComputeGranularity:
    """
    Defines the hierarchical building block for compute.

    The key insight is that interconnect scales with clusters, not ALUs.
    A GPU SM has 128+256 ALUs but counts as 1 node on the GPC network.
    """
    name: str
    alus_per_cluster: int           # Total ALUs in the cluster
    clusters_per_group: int         # Clusters per higher-level group (e.g., SMs per GPC)
    groups_per_chip: int            # Groups per chip (e.g., GPCs per GPU)
    intra_cluster_overhead: float   # Energy overhead within cluster (amortized)
    inter_cluster_overhead: float   # Energy overhead between clusters (per hop)
    description: str = ""


# Preset compute granularities for common architectures
COMPUTE_GRANULARITY_PRESETS = {
    # NVIDIA GPU: SM (Streaming Multiprocessor) is the building block
    # SM has 128 CUDA cores + 4 Tensor Cores (256 MACs) = 384 effective ALUs
    # H100 config: 132 SMs total, organized as 8 GPCs x ~16.5 SMs/GPC
    'nvidia_sm': ComputeGranularity(
        name='nvidia_sm',
        alus_per_cluster=384,       # 128 CUDA cores + 4 Tensor Cores x 64 MACs
        clusters_per_group=18,      # SMs per GPC (Graphics Processing Cluster)
        groups_per_chip=8,          # GPCs per GPU
        intra_cluster_overhead=0.1, # Low: shared register file, warp scheduler
        inter_cluster_overhead=0.3, # Moderate: SM-to-SM network within GPC
        description="NVIDIA SM (Streaming Multiprocessor): 128 CUDA + 4 TCs, 18 SMs/GPC, 8 GPCs",
    ),

    # NVIDIA Tensor Core focus (for AI workloads)
    # Count only tensor cores: 4 TCs x 64 MACs = 256 per SM
    'nvidia_tc': ComputeGranularity(
        name='nvidia_tc',
        alus_per_cluster=256,       # 4 Tensor Cores x 64 MACs per SM
        clusters_per_group=18,      # SMs per GPC (Graphics Processing Cluster)
        groups_per_chip=8,          # GPCs per GPU
        intra_cluster_overhead=0.1,
        inter_cluster_overhead=0.3,
        description="NVIDIA Tensor Cores only: 256 MACs/SM (4 TCs x 64), 18 SMs/GPC, 8 GPCs",
    ),

    # Google TPU: MXU (Matrix Multiply Unit) is the building block
    # TPU v4: 2 TensorCores per chip, each with 128x128 MXU = 16384 MACs
    'tpu_mxu': ComputeGranularity(
        name='tpu_mxu',
        alus_per_cluster=16384,     # 128x128 MXU (Matrix Multiply Unit) systolic array
        clusters_per_group=2,       # 2 MXUs per TensorCore
        groups_per_chip=1,          # 1 TensorCore per chip (v4 has 2 chips per board)
        intra_cluster_overhead=0.02,# Very low: systolic dataflow, fixed schedule
        inter_cluster_overhead=0.1, # Low: on-chip HBM buffer
        description="TPU MXU (Matrix Multiply Unit): 128x128 systolic, 2 MXUs/chip",
    ),

    # Intel/AMD CPU: Core is the building block
    # Each core has 4-8 FP ALUs (2 FMA units x 2 = 4 FP64 ops, 8 FP32)
    # AMD Zen: 8 cores per CCX (Core Complex), multiple CCXs per CCD (Core Chiplet Die)
    'cpu_core': ComputeGranularity(
        name='cpu_core',
        alus_per_cluster=8,         # 8 FP32 ALUs per core (2 x 256-bit FMA units)
        clusters_per_group=8,       # Cores per CCX (Core Complex, shares L3)
        groups_per_chip=8,          # CCXs per chip (or CCDs for chiplet designs)
        intra_cluster_overhead=0.5, # High: OoO execution, register file, branch predictor
        inter_cluster_overhead=0.4, # High: LLC (Last Level Cache) coherence traffic
        description="CPU Core: 8 FP32 ALUs, 8 cores/CCX (Core Complex), 8 CCXs/chip",
    ),

    # KPU/Domain Flow: Tile is the building block
    # 16x16 PE (Processing Element) array = 256 MACs per tile
    'kpu_tile': ComputeGranularity(
        name='kpu_tile',
        alus_per_cluster=256,       # 16x16 PE (Processing Element) array
        clusters_per_group=16,      # Tiles per row/column
        groups_per_chip=16,         # Rows of tiles (16x16 = 256 tiles total)
        intra_cluster_overhead=0.05,# Low: streaming dataflow
        inter_cluster_overhead=0.15,# Low: 2D mesh with spatial locality
        description="KPU Tile: 16x16 PE (Processing Element) array, 256 tiles in mesh",
    ),

    # Generic accelerator tile (configurable baseline)
    'generic_tile': ComputeGranularity(
        name='generic_tile',
        alus_per_cluster=256,
        clusters_per_group=16,
        groups_per_chip=4,
        intra_cluster_overhead=0.1,
        inter_cluster_overhead=0.2,
        description="Generic accelerator tile (configurable)",
    ),
}


def get_compute_granularity(name: str) -> ComputeGranularity:
    """Get a compute granularity preset by name."""
    if name not in COMPUTE_GRANULARITY_PRESETS:
        available = ', '.join(COMPUTE_GRANULARITY_PRESETS.keys())
        raise ValueError(f"Unknown compute granularity: {name}. Available: {available}")
    return COMPUTE_GRANULARITY_PRESETS[name]

# Idle power fraction (leakage floor as fraction of TDP envelope)
# At modern nodes (7nm and below), leakage is significant
IDLE_POWER_FRACTION = 0.50

# =============================================================================
# Empirically-Derived Compute Fractions (Calibration Targets)
# =============================================================================
# From first-principles analysis using empirical throughput and energy models:
#
# Hardware               | TDP    | Compute | Infra   | Compute%
# -----------------------|--------|---------|---------|----------
# Intel i7-12700K (CPU)  | 125 W  |   4.8 W | 120.2 W |    3.9%
# Jetson Orin Nano (GPU) |  15 W  |   1.4 W |  13.6 W |    9.6%
#
# Key insight: GPU has ~2.5x higher compute fraction than CPU!
# This is because:
#   - GPU cores are simpler (no OoO, no branch prediction)
#   - Shared control logic amortizes overhead across many ALUs
#   - Higher spatial/temporal locality in memory access
#   - Less power wasted on speculation and coherence
#
# The empirical compute fractions are MUCH lower than area fractions because:
#   1. Real workloads have memory stalls, cache misses
#   2. Clock distribution consumes ~10-15% of chip power
#   3. I/O, PHY, and voltage regulators consume power
#   4. SRAM leakage is significant at modern nodes
#
# These calibration targets help validate model predictions.
EMPIRICAL_COMPUTE_FRACTION = {
    'x86_performance': 0.039,   # 3.9% (i7-12700K @ 125W)
    'x86_efficiency': 0.05,     # Estimated: simpler cores = slightly better
    'arm_performance': 0.045,   # Estimated: between x86 and efficiency
    'arm_efficiency': 0.06,     # Estimated: more efficient than x86
    'cuda_core': 0.08,          # Estimated: between CPU and tensor core
    'tensor_core': 0.096,       # 9.6% (Jetson Orin Nano @ 15W)
    'systolic_mac': 0.15,       # TPU-style: very high efficiency
    'domain_flow': 0.12,        # KPU: high efficiency
    'simd_packed': 0.06,        # DSP-style: moderate efficiency
    'standard_cell': 0.07,      # Generic
}


# =============================================================================
# Power Breakdown Result
# =============================================================================

@dataclass
class InfrastructurePowerBreakdown:
    """Complete power breakdown for SoC infrastructure."""
    # Configuration
    num_alus: int
    process_node_nm: int
    circuit_type: str
    topology: InterconnectTopology
    frequency_ghz: float

    # Component powers (Watts)
    compute_power_w: float          # ALU switching power
    sram_power_w: float             # On-chip memory power
    interconnect_power_w: float     # NoC/wire power
    control_power_w: float          # Scheduler/DMA/coherence power
    idle_power_w: float             # Leakage floor

    # Derived totals
    total_dynamic_power_w: float    # Sum of dynamic components
    total_tdp_w: float              # Total including idle

    # Area estimates
    compute_area_mm2: float
    sram_area_mm2: float
    control_area_mm2: float         # Control logic area (scheduler, DMA, etc.)
    total_area_mm2: float

    # Efficiency metrics
    @property
    def compute_fraction(self) -> float:
        """Fraction of TDP that is 'useful' compute."""
        return self.compute_power_w / self.total_tdp_w if self.total_tdp_w > 0 else 0

    @property
    def compute_area_fraction(self) -> float:
        """Fraction of die area dedicated to compute (ALUs)."""
        return self.compute_area_mm2 / self.total_area_mm2 if self.total_area_mm2 > 0 else 0

    @property
    def infrastructure_overhead(self) -> float:
        """Overhead multiplier: total TDP / compute-only power."""
        return self.total_tdp_w / self.compute_power_w if self.compute_power_w > 0 else float('inf')

    @property
    def sram_bytes(self) -> int:
        """Total on-chip SRAM bytes."""
        return self.num_alus * SRAM_BYTES_PER_ALU

    @property
    def sram_mib(self) -> float:
        """Total on-chip SRAM in MiB."""
        return self.sram_bytes / (1024 * 1024)


# =============================================================================
# SoC Infrastructure Model
# =============================================================================

class SoCInfrastructureModel:
    """
    Models SoC infrastructure overhead for realistic TDP estimation.

    This model captures the non-linear relationship between ALU count and
    total power consumption due to:

    1. SRAM: On-chip memory scales linearly but has leakage overhead
    2. Interconnect: Wire length scales with sqrt(area), power super-linear
    3. Control: Scheduler/DMA power scales sub-linearly (clustering)
    4. Idle: Leakage floor is ~50% of TDP envelope

    Key improvement: Uses compute granularity to model hierarchical building blocks.
    Interconnect scales with clusters (e.g., SMs, MXUs), not individual ALUs.

    Example:
        model = SoCInfrastructureModel(
            num_alus=16384,
            process_node_nm=4,
            circuit_type='tensor_core',
            topology=InterconnectTopology.MESH_2D,
            compute_granularity='nvidia_tc',  # Use NVIDIA tensor core preset
        )
        breakdown = model.compute_power_breakdown(frequency_ghz=1.5)
        print(f"Total TDP: {breakdown.total_tdp_w:.1f} W")
        print(f"Compute fraction: {breakdown.compute_fraction:.1%}")
    """

    def __init__(
        self,
        num_alus: int,
        process_node_nm: int,
        circuit_type: str = 'tensor_core',
        topology: InterconnectTopology = InterconnectTopology.MESH_2D,
        compute_granularity: Optional[str] = None,
    ):
        self.num_alus = num_alus
        self.process_node_nm = process_node_nm
        self.circuit_type = circuit_type
        self.topology = topology

        # Set compute granularity (determines cluster size for interconnect)
        if compute_granularity is not None:
            self.granularity = get_compute_granularity(compute_granularity)
        else:
            # Auto-select based on circuit type
            self.granularity = self._auto_select_granularity()

        # Validate inputs
        if circuit_type not in CIRCUIT_TYPE_MULTIPLIER:
            raise ValueError(f"Unknown circuit type: {circuit_type}")

    def _auto_select_granularity(self) -> ComputeGranularity:
        """Auto-select compute granularity based on circuit type."""
        circuit_to_granularity = {
            'x86_performance': 'cpu_core',
            'x86_efficiency': 'cpu_core',
            'arm_performance': 'cpu_core',
            'arm_efficiency': 'cpu_core',
            'cuda_core': 'nvidia_sm',
            'tensor_core': 'nvidia_tc',
            'systolic_mac': 'tpu_mxu',
            'domain_flow': 'kpu_tile',
            'simd_packed': 'cpu_core',
            'standard_cell': 'generic_tile',
        }
        granularity_name = circuit_to_granularity.get(self.circuit_type, 'generic_tile')
        return get_compute_granularity(granularity_name)

    @property
    def num_clusters(self) -> int:
        """Number of compute clusters (e.g., SMs, MXUs, cores)."""
        return max(1, self.num_alus // self.granularity.alus_per_cluster)

    @property
    def num_groups(self) -> int:
        """Number of cluster groups (e.g., GPCs, CCXs)."""
        clusters = self.num_clusters
        return max(1, clusters // self.granularity.clusters_per_group)

    def _get_sram_energy_pj_per_byte(self) -> float:
        """Get SRAM energy per byte for this process node."""
        if self.process_node_nm in SRAM_ENERGY_PJ_PER_BYTE:
            return SRAM_ENERGY_PJ_PER_BYTE[self.process_node_nm]
        # Interpolate for missing nodes
        nodes = sorted(SRAM_ENERGY_PJ_PER_BYTE.keys())
        if self.process_node_nm < nodes[0]:
            return SRAM_ENERGY_PJ_PER_BYTE[nodes[0]]
        if self.process_node_nm > nodes[-1]:
            return SRAM_ENERGY_PJ_PER_BYTE[nodes[-1]]
        # Linear interpolation
        for i in range(len(nodes) - 1):
            if nodes[i] <= self.process_node_nm <= nodes[i + 1]:
                e1 = SRAM_ENERGY_PJ_PER_BYTE[nodes[i]]
                e2 = SRAM_ENERGY_PJ_PER_BYTE[nodes[i + 1]]
                t = (self.process_node_nm - nodes[i]) / (nodes[i + 1] - nodes[i])
                return e1 + t * (e2 - e1)
        return 0.40  # Fallback

    def _get_sram_density_mb_per_mm2(self) -> float:
        """Get SRAM density for this process node."""
        if self.process_node_nm in SRAM_DENSITY_MB_PER_MM2:
            return SRAM_DENSITY_MB_PER_MM2[self.process_node_nm]
        # Fallback with interpolation
        nodes = sorted(SRAM_DENSITY_MB_PER_MM2.keys())
        if self.process_node_nm < nodes[0]:
            return SRAM_DENSITY_MB_PER_MM2[nodes[0]]
        if self.process_node_nm > nodes[-1]:
            return SRAM_DENSITY_MB_PER_MM2[nodes[-1]]
        for i in range(len(nodes) - 1):
            if nodes[i] <= self.process_node_nm <= nodes[i + 1]:
                d1 = SRAM_DENSITY_MB_PER_MM2[nodes[i]]
                d2 = SRAM_DENSITY_MB_PER_MM2[nodes[i + 1]]
                t = (self.process_node_nm - nodes[i]) / (nodes[i + 1] - nodes[i])
                return d1 + t * (d2 - d1)
        return 30.0  # Fallback

    def compute_sram_power(self, frequency_ghz: float, utilization: float = 1.0) -> float:
        """
        Compute SRAM power in Watts.

        At full utilization, each ALU streams operands from SRAM every cycle.
        Static leakage adds ~30% to dynamic power for modern SRAM.
        """
        sram_bytes = self.num_alus * SRAM_BYTES_PER_ALU
        sram_energy_pj = self._get_sram_energy_pj_per_byte()

        # Access rate: assume 4 bytes per ALU per cycle (one FP32 operand)
        # In practice, tensor cores access more but with higher reuse
        bytes_per_cycle = self.num_alus * 4 * utilization
        bytes_per_second = bytes_per_cycle * frequency_ghz * 1e9

        # Dynamic power from SRAM accesses
        dynamic_power_w = bytes_per_second * sram_energy_pj * 1e-12

        # Static leakage adds ~30% for modern SRAM
        static_leakage_fraction = 0.30
        total_power_w = dynamic_power_w * (1 + static_leakage_fraction)

        return total_power_w

    def compute_sram_area(self) -> float:
        """Compute SRAM area in mm^2."""
        sram_bytes = self.num_alus * SRAM_BYTES_PER_ALU
        sram_mb = sram_bytes / (1024 * 1024)
        density = self._get_sram_density_mb_per_mm2()
        return (sram_mb * 8) / density  # Convert MiB to Mb (bits)

    def compute_alu_area(self) -> float:
        """Compute ALU/compute area in mm^2."""
        # Base ALU area scales with process node squared (area scales with feature size^2)
        base_alu_area = ALU_AREA_MM2_5NM * (self.process_node_nm / 5.0) ** 2

        # Apply circuit-specific area multiplier
        # CPUs have much larger "per ALU" area due to OoO, caches, etc.
        # Systolic arrays pack MACs very densely
        circuit_mult = CIRCUIT_AREA_MULTIPLIER.get(self.circuit_type, 4.0)

        return self.num_alus * base_alu_area * circuit_mult

    def compute_control_area(self) -> float:
        """
        Compute control/infrastructure area in mm^2.

        This includes:
        - Scheduler/dispatch logic
        - DMA controllers
        - Coherence/synchronization logic
        - NoC routers and switches
        - I/O, PLL, power management

        Control area scales with sqrt(clusters) because many resources are shared
        and don't need to replicate 1:1 with clusters. A chip with 64 clusters
        doesn't need 64x the scheduler area of a chip with 1 cluster.
        """
        # Base control area at 5nm for a reference design (mm^2)
        # A complete scheduler + router + coherence block is roughly 1-5 mm^2
        base_control_area = 1.0

        # Scale with process node
        process_scale = (self.process_node_nm / 5.0) ** 2

        # Control overhead varies by architecture:
        # - CPUs need complex control (OoO, coherence, speculation)
        # - GPUs need moderate control (warp scheduling, barrier sync)
        # - TPUs need minimal control (systolic has fixed schedule)
        control_overhead_mult = {
            'x86_performance': 5.0,   # Complex OoO, coherence
            'x86_efficiency': 2.5,
            'arm_performance': 3.0,
            'arm_efficiency': 1.5,
            'cuda_core': 2.0,
            'tensor_core': 1.5,
            'simd_packed': 2.0,
            'systolic_mac': 0.3,      # Fixed schedule, minimal control
            'domain_flow': 0.5,
            'standard_cell': 1.0,
        }
        overhead = control_overhead_mult.get(self.circuit_type, 1.0)

        num_clusters = self.num_clusters
        num_groups = self.num_groups

        # Control area scales sub-linearly with clusters (sqrt scaling)
        # This models shared resources like top-level scheduler, coherence directory
        from math import sqrt, log2
        cluster_factor = 1 + log2(max(1, num_clusters)) / 4
        group_factor = 1 + log2(max(1, num_groups)) / 6

        control_area = base_control_area * process_scale * overhead * cluster_factor * group_factor

        return control_area

    def compute_interconnect_power(self, frequency_ghz: float, data_width_bits: int = 32) -> float:
        """
        Compute interconnect power in Watts.

        Key insight: Interconnect scales with CLUSTERS, not individual ALUs.
        A GPU SM has 384 ALUs but counts as 1 node on the GPC-to-L2 crossbar.

        The model uses two levels:
        1. Intra-cluster: Low overhead within a cluster (shared resources)
        2. Inter-cluster: Higher overhead between clusters (network traffic)
        """
        # Wire energy scales with process node
        wire_energy_pj = WIRE_ENERGY_PJ_PER_MM_5NM * (self.process_node_nm / 5.0)

        # Compute chip dimensions from total area
        compute_area = self.compute_alu_area()
        sram_area = self.compute_sram_area()
        total_area = compute_area + sram_area
        chip_side_mm = sqrt(total_area)

        # Use clusters (not ALUs) for interconnect scaling
        num_clusters = self.num_clusters
        num_groups = self.num_groups

        # Base wire length between clusters
        cluster_grid_side = sqrt(num_clusters)
        base_wire_mm = chip_side_mm / max(1, cluster_grid_side) * 2

        # Topology-dependent wire factors (now applied to clusters, not ALUs)
        if self.topology == InterconnectTopology.HTREE:
            avg_wire_mm = base_wire_mm * 2.0
            effective_wires_per_cluster = 2

        elif self.topology == InterconnectTopology.CROSSBAR:
            # Crossbar between groups (e.g., GPC-to-L2 crossbar)
            avg_wire_mm = base_wire_mm * 3.0
            effective_wires_per_cluster = min(8, 2 + log2(max(2, num_groups)))

        elif self.topology == InterconnectTopology.MESH_2D:
            avg_wire_mm = base_wire_mm
            # Modest scaling with cluster count
            hop_factor = 1 + log2(max(2, cluster_grid_side)) / 15
            effective_wires_per_cluster = 4 * hop_factor

        elif self.topology == InterconnectTopology.CLOS:
            avg_wire_mm = base_wire_mm * 1.5
            effective_wires_per_cluster = 3

        else:
            avg_wire_mm = base_wire_mm
            effective_wires_per_cluster = 4

        # Energy per bit transmission over wire
        energy_per_bit_pj = wire_energy_pj * avg_wire_mm

        # Bandwidth: clusters share bandwidth, not individual ALUs
        # Each cluster needs data for all its ALUs, but with high reuse
        alus_per_cluster = self.granularity.alus_per_cluster
        reuse_factor = sqrt(alus_per_cluster)  # Systolic/spatial reuse
        bits_per_cluster_per_cycle = data_width_bits * alus_per_cluster / reuse_factor
        total_bits_per_second = num_clusters * bits_per_cluster_per_cycle * frequency_ghz * 1e9

        # Inter-cluster wire power
        inter_cluster_power = (energy_per_bit_pj * 1e-12 *
                               total_bits_per_second *
                               effective_wires_per_cluster / num_clusters *
                               num_clusters)

        # Intra-cluster overhead (amortized across ALUs in cluster)
        intra_cluster_overhead = self.granularity.intra_cluster_overhead
        intra_cluster_power = self.num_alus * frequency_ghz * 1e9 * wire_energy_pj * 0.01 * intra_cluster_overhead * 1e-12

        # Router/switch overhead (scales with groups, not clusters)
        router_overhead = 1 + 0.1 * log2(max(2, num_groups))

        total_interconnect = (inter_cluster_power + intra_cluster_power) * router_overhead

        return total_interconnect

    def compute_control_power(self, frequency_ghz: float) -> float:
        """
        Compute control overhead power in Watts.

        Uses compute granularity to properly scale control overhead.
        Control is per-cluster (e.g., per SM, per MXU), not per-ALU.
        """
        # Get control energy for this circuit type
        control_energy_pj = CONTROL_ENERGY_PJ_PER_ALU.get(self.circuit_type, 0.5)

        # Scale with process node (larger nodes = more energy)
        control_energy_pj *= (self.process_node_nm / 7.0)

        # Use granularity for cluster sizing
        num_clusters = self.num_clusters
        alus_per_cluster = self.granularity.alus_per_cluster

        # Control overhead is per-cluster, amortized over ALUs
        # Clusters with more ALUs have lower per-ALU control overhead
        amortization_factor = sqrt(alus_per_cluster) / 16  # Normalized to 256-ALU cluster
        effective_control_energy = control_energy_pj / max(1, amortization_factor)

        # Power = clusters x frequency x energy per cluster
        energy_per_cluster_pj = effective_control_energy * alus_per_cluster
        cluster_power_w = num_clusters * frequency_ghz * 1e9 * energy_per_cluster_pj * 1e-12

        # DMA controllers scale with groups (e.g., GPCs), not ALUs
        num_groups = self.num_groups
        dma_power_per_unit = 0.1 * (7.0 / self.process_node_nm)  # 0.1W at 7nm
        dma_power_w = num_groups * dma_power_per_unit

        return cluster_power_w + dma_power_w

    def compute_idle_power(self, dynamic_power_w: float) -> float:
        """
        Compute idle/leakage power in Watts.

        Idle power is ~50% of TDP envelope at modern process nodes.
        This represents subthreshold leakage, gate leakage, and clock tree power.
        """
        # Estimate TDP envelope from dynamic power with headroom
        # Typical designs have ~30% thermal headroom
        estimated_envelope = dynamic_power_w * 1.3
        return estimated_envelope * IDLE_POWER_FRACTION

    def compute_power_breakdown(
        self,
        frequency_ghz: float,
        precision: str = 'FP32',
        utilization: float = 1.0,
    ) -> InfrastructurePowerBreakdown:
        """
        Compute complete power breakdown for the SoC.

        Args:
            frequency_ghz: Operating frequency
            precision: Compute precision (affects ops per cycle)
            utilization: ALU utilization (0-1)

        Returns:
            InfrastructurePowerBreakdown with all component powers
        """
        # Import here to avoid circular dependency
        from .technology_profile import get_process_base_energy_pj

        # Precision scaling factors (from estimate_tdp.py)
        precision_energy_scale = {
            'FP64': 2.0, 'FP32': 1.0, 'TF32': 0.6, 'BF16': 0.5,
            'FP16': 0.5, 'FP8': 0.25, 'INT8': 0.25, 'INT4': 0.125, 'INT2': 0.0625,
        }
        precision_ops_per_cycle = {
            'FP64': 1, 'FP32': 2, 'TF32': 4, 'BF16': 4,
            'FP16': 4, 'FP8': 8, 'INT8': 8, 'INT4': 16, 'INT2': 32,
        }

        precision = precision.upper()
        energy_scale = precision_energy_scale.get(precision, 1.0)
        ops_per_cycle = precision_ops_per_cycle.get(precision, 2)

        # 1. Compute power (pure ALU switching)
        base_energy_pj = get_process_base_energy_pj(self.process_node_nm)
        circuit_mult = CIRCUIT_TYPE_MULTIPLIER.get(self.circuit_type, 1.0)
        energy_per_op_pj = base_energy_pj * circuit_mult * energy_scale

        ops_per_second = self.num_alus * ops_per_cycle * frequency_ghz * 1e9 * utilization
        compute_power_w = energy_per_op_pj * 1e-12 * ops_per_second

        # 2. SRAM power
        sram_power_w = self.compute_sram_power(frequency_ghz, utilization)

        # 3. Interconnect power
        interconnect_power_w = self.compute_interconnect_power(frequency_ghz)

        # 4. Control power
        control_power_w = self.compute_control_power(frequency_ghz)

        # 5. Total dynamic power
        total_dynamic_w = compute_power_w + sram_power_w + interconnect_power_w + control_power_w

        # 6. Idle power (based on dynamic power envelope)
        idle_power_w = self.compute_idle_power(total_dynamic_w)

        # 7. Total TDP
        total_tdp_w = total_dynamic_w + idle_power_w

        # Area estimates
        compute_area = self.compute_alu_area()
        sram_area = self.compute_sram_area()
        control_area = self.compute_control_area()
        total_area = compute_area + sram_area + control_area

        return InfrastructurePowerBreakdown(
            num_alus=self.num_alus,
            process_node_nm=self.process_node_nm,
            circuit_type=self.circuit_type,
            topology=self.topology,
            frequency_ghz=frequency_ghz,
            compute_power_w=compute_power_w,
            sram_power_w=sram_power_w,
            interconnect_power_w=interconnect_power_w,
            control_power_w=control_power_w,
            idle_power_w=idle_power_w,
            total_dynamic_power_w=total_dynamic_w,
            total_tdp_w=total_tdp_w,
            compute_area_mm2=compute_area,
            sram_area_mm2=sram_area,
            control_area_mm2=control_area,
            total_area_mm2=total_area,
        )


# =============================================================================
# Convenience Functions
# =============================================================================

def estimate_infrastructure_power(
    num_alus: int,
    process_node_nm: int,
    circuit_type: str = 'tensor_core',
    topology: InterconnectTopology = InterconnectTopology.MESH_2D,
    frequency_ghz: float = 1.5,
    precision: str = 'FP32',
    compute_granularity: Optional[str] = None,
) -> InfrastructurePowerBreakdown:
    """
    Convenience function to estimate infrastructure power.

    Args:
        num_alus: Number of ALUs/MACs
        process_node_nm: Process node in nanometers
        circuit_type: Circuit design type
        topology: Interconnect topology
        frequency_ghz: Operating frequency
        precision: Compute precision
        compute_granularity: Preset name for compute building block
            Options: nvidia_sm, nvidia_tc, tpu_mxu, cpu_core, kpu_tile, generic_tile

    Returns:
        InfrastructurePowerBreakdown with complete power analysis
    """
    model = SoCInfrastructureModel(
        num_alus=num_alus,
        process_node_nm=process_node_nm,
        circuit_type=circuit_type,
        topology=topology,
        compute_granularity=compute_granularity,
    )
    return model.compute_power_breakdown(frequency_ghz, precision)


def get_topology_from_string(name: str) -> InterconnectTopology:
    """Convert string to InterconnectTopology enum."""
    name_map = {
        'mesh': InterconnectTopology.MESH_2D,
        'mesh_2d': InterconnectTopology.MESH_2D,
        '2d_mesh': InterconnectTopology.MESH_2D,
        'htree': InterconnectTopology.HTREE,
        'h_tree': InterconnectTopology.HTREE,
        'h-tree': InterconnectTopology.HTREE,
        'crossbar': InterconnectTopology.CROSSBAR,
        'xbar': InterconnectTopology.CROSSBAR,
        'clos': InterconnectTopology.CLOS,
    }
    key = name.lower().replace('-', '_').replace(' ', '_')
    if key not in name_map:
        available = ', '.join(sorted(set(name_map.keys())))
        raise ValueError(f"Unknown topology: {name}. Available: {available}")
    return name_map[key]
