"""
Distributed L3 Topology Model

Models distributed L3 cache architectures for tile orchestration.
Supports checkerboard tilings where tiles rotate to adjacent L3 slices.

Key concepts:
- L3Slice: Single L3 cache slice with position and capacity
- DistributedL3: Complete distributed L3 topology (mesh, ring, etc.)
- CheckerboardPattern: Checkerboard assignment of tiles to slices
- TilePlacement: Where tiles are placed in the distributed topology
- TransferCost: Cost model for inter-slice communication

Use cases:
- Branes.AI distributed L3 SoCs
- Intel mesh L3 (L3 slices per core)
- AMD CCD L3 (per chiplet)
- Multi-chip module L3
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Set
from math import ceil
import itertools


class TopologyType(Enum):
    """Network topology types."""
    MESH_2D = "mesh_2d"       # 2D mesh (row/col)
    RING = "ring"             # 1D ring
    TORUS_2D = "torus_2d"     # 2D torus (mesh with wraparound)
    CROSSBAR = "crossbar"     # Full crossbar (any-to-any)
    HYPERCUBE = "hypercube"   # Hypercube topology


class SliceRole(Enum):
    """Role of a slice in checkerboard pattern."""
    COMPUTE = "compute"       # Executing computation
    MEMORY = "memory"         # Staging data for neighbors
    IDLE = "idle"             # Not currently active


@dataclass
class Position:
    """2D position in topology."""
    row: int
    col: int

    def __hash__(self):
        return hash((self.row, self.col))

    def __eq__(self, other):
        if isinstance(other, Position):
            return self.row == other.row and self.col == other.col
        return False

    def manhattan_distance(self, other: 'Position') -> int:
        """Manhattan distance to another position."""
        return abs(self.row - other.row) + abs(self.col - other.col)


@dataclass
class L3Slice:
    """
    Single L3 cache slice in distributed configuration.

    Models a single L3 slice with its position, capacity, and state.
    """
    slice_id: int
    position: Position
    capacity_bytes: int

    # Performance characteristics
    read_bandwidth_gbps: float = 100.0
    write_bandwidth_gbps: float = 100.0
    access_latency_cycles: int = 20

    # Current state
    role: SliceRole = SliceRole.IDLE
    current_tiles: List[str] = field(default_factory=list)
    used_bytes: int = 0

    def __hash__(self):
        return hash(self.slice_id)

    def available_bytes(self) -> int:
        """Available capacity."""
        return self.capacity_bytes - self.used_bytes

    def can_fit(self, bytes_needed: int) -> bool:
        """Check if slice can fit given bytes."""
        return bytes_needed <= self.available_bytes()

    def allocate(self, tile_id: str, bytes_needed: int) -> bool:
        """Allocate space for tile. Returns False if no space."""
        if not self.can_fit(bytes_needed):
            return False
        self.current_tiles.append(tile_id)
        self.used_bytes += bytes_needed
        return True

    def deallocate(self, tile_id: str, bytes_freed: int):
        """Free space from tile."""
        if tile_id in self.current_tiles:
            self.current_tiles.remove(tile_id)
            self.used_bytes -= bytes_freed


@dataclass
class TransferCost:
    """Cost of transferring data between slices."""
    src_slice: int
    dst_slice: int
    bytes_transferred: int

    # Cost metrics
    latency_cycles: int = 0
    energy_pj: float = 0.0
    hops: int = 0

    @property
    def bandwidth_limited_cycles(self) -> int:
        """Cycles if bandwidth-limited (at 100 GB/s, 2 GHz)."""
        # 100 GB/s = 50 bytes/cycle at 2 GHz
        return ceil(self.bytes_transferred / 50)


@dataclass
class DistributedL3:
    """
    Distributed L3 cache topology.

    Models multi-slice L3 configurations with configurable topology.
    """
    slices: List[L3Slice]
    topology: TopologyType = TopologyType.MESH_2D

    # Topology dimensions (for mesh/torus)
    mesh_rows: int = 2
    mesh_cols: int = 2

    # Inter-slice communication costs
    hop_latency_cycles: int = 5
    hop_energy_pj_per_byte: float = 0.5
    wire_energy_pj_per_byte: float = 0.1

    # Slice mapping
    _position_to_slice: Dict[Position, L3Slice] = field(default_factory=dict)
    _id_to_slice: Dict[int, L3Slice] = field(default_factory=dict)

    def __post_init__(self):
        self._build_mappings()

    def _build_mappings(self):
        """Build position and ID mappings."""
        self._position_to_slice = {s.position: s for s in self.slices}
        self._id_to_slice = {s.slice_id: s for s in self.slices}

    def get_slice(self, slice_id: int) -> Optional[L3Slice]:
        """Get slice by ID."""
        return self._id_to_slice.get(slice_id)

    def get_slice_at(self, row: int, col: int) -> Optional[L3Slice]:
        """Get slice at position."""
        return self._position_to_slice.get(Position(row, col))

    def distance(self, src_id: int, dst_id: int) -> int:
        """
        Number of hops between slices.

        Depends on topology.
        """
        if src_id == dst_id:
            return 0

        src = self._id_to_slice[src_id]
        dst = self._id_to_slice[dst_id]

        if self.topology == TopologyType.MESH_2D:
            return src.position.manhattan_distance(dst.position)

        elif self.topology == TopologyType.TORUS_2D:
            # Wrap-around: min of direct and wrapped distance
            dr = abs(src.position.row - dst.position.row)
            dc = abs(src.position.col - dst.position.col)
            dr = min(dr, self.mesh_rows - dr)
            dc = min(dc, self.mesh_cols - dc)
            return dr + dc

        elif self.topology == TopologyType.RING:
            # 1D ring distance
            n = len(self.slices)
            direct = abs(src_id - dst_id)
            return min(direct, n - direct)

        elif self.topology == TopologyType.CROSSBAR:
            return 1  # Direct connection

        elif self.topology == TopologyType.HYPERCUBE:
            # Hamming distance in binary representation
            return bin(src_id ^ dst_id).count('1')

        return src.position.manhattan_distance(dst.position)

    def transfer_cost(self, src_id: int, dst_id: int,
                      bytes_to_transfer: int) -> TransferCost:
        """
        Calculate cost to transfer data between slices.

        Includes latency (hops * hop_latency) and energy.
        """
        hops = self.distance(src_id, dst_id)

        latency = hops * self.hop_latency_cycles
        energy = bytes_to_transfer * (
            hops * self.hop_energy_pj_per_byte +
            self.wire_energy_pj_per_byte
        )

        return TransferCost(
            src_slice=src_id,
            dst_slice=dst_id,
            bytes_transferred=bytes_to_transfer,
            latency_cycles=latency,
            energy_pj=energy,
            hops=hops,
        )

    def neighbors(self, slice_id: int) -> List[int]:
        """Get IDs of neighboring slices (distance 1)."""
        return [
            s.slice_id for s in self.slices
            if self.distance(slice_id, s.slice_id) == 1
        ]

    def total_capacity(self) -> int:
        """Total L3 capacity across all slices."""
        return sum(s.capacity_bytes for s in self.slices)

    @property
    def num_slices(self) -> int:
        return len(self.slices)


def create_mesh_l3(
    rows: int,
    cols: int,
    slice_capacity_mb: int = 4,
    hop_latency_cycles: int = 5,
) -> DistributedL3:
    """
    Create 2D mesh distributed L3 configuration.

    Args:
        rows: Number of rows in mesh
        cols: Number of columns in mesh
        slice_capacity_mb: Capacity per slice in MB
        hop_latency_cycles: Latency per hop

    Returns:
        DistributedL3 with mesh topology
    """
    slices = []
    slice_bytes = slice_capacity_mb * 1024 * 1024

    for r in range(rows):
        for c in range(cols):
            slice_id = r * cols + c
            slices.append(L3Slice(
                slice_id=slice_id,
                position=Position(row=r, col=c),
                capacity_bytes=slice_bytes,
            ))

    return DistributedL3(
        slices=slices,
        topology=TopologyType.MESH_2D,
        mesh_rows=rows,
        mesh_cols=cols,
        hop_latency_cycles=hop_latency_cycles,
    )


def create_ring_l3(
    num_slices: int,
    slice_capacity_mb: int = 4,
    hop_latency_cycles: int = 5,
) -> DistributedL3:
    """Create ring distributed L3 configuration."""
    slices = [
        L3Slice(
            slice_id=i,
            position=Position(row=0, col=i),
            capacity_bytes=slice_capacity_mb * 1024 * 1024,
        )
        for i in range(num_slices)
    ]

    return DistributedL3(
        slices=slices,
        topology=TopologyType.RING,
        mesh_rows=1,
        mesh_cols=num_slices,
        hop_latency_cycles=hop_latency_cycles,
    )


@dataclass
class TilePlacement:
    """
    Placement of a tile on a specific L3 slice.

    Tracks which tiles are where and their data.
    """
    tile_id: str
    slice_id: int
    bytes_used: int
    tile_type: str = "data"  # "data", "input", "weight", "output"

    # Matrix block coordinates
    block_i: int = 0  # M dimension block
    block_j: int = 0  # N dimension block
    block_k: int = 0  # K dimension block


@dataclass
class CheckerboardPattern:
    """
    Checkerboard assignment of tiles to L3 slices.

    Alternates between compute and memory roles:
    - Compute slices: Execute matrix operations
    - Memory slices: Stage data for adjacent compute slices

    Pattern enables:
    1. Local data reuse (compute uses adjacent memory)
    2. Tile rotation (memory tiles move to become compute)
    3. Reduced DRAM traffic
    """
    topology: DistributedL3
    tile_assignments: Dict[Tuple[int, int], int] = field(default_factory=dict)
    slice_roles: Dict[int, SliceRole] = field(default_factory=dict)

    # Tile placements
    placements: List[TilePlacement] = field(default_factory=list)

    def __post_init__(self):
        self._assign_roles()

    def _assign_roles(self):
        """Assign compute/memory roles in checkerboard pattern."""
        for s in self.topology.slices:
            # Checkerboard: (row + col) % 2 determines role
            if (s.position.row + s.position.col) % 2 == 0:
                self.slice_roles[s.slice_id] = SliceRole.COMPUTE
            else:
                self.slice_roles[s.slice_id] = SliceRole.MEMORY

    def compute_slices(self) -> List[int]:
        """Get IDs of compute slices."""
        return [
            sid for sid, role in self.slice_roles.items()
            if role == SliceRole.COMPUTE
        ]

    def memory_slices(self) -> List[int]:
        """Get IDs of memory slices."""
        return [
            sid for sid, role in self.slice_roles.items()
            if role == SliceRole.MEMORY
        ]

    def assign_tile(self, block_i: int, block_j: int, slice_id: int):
        """Assign output tile (i,j) to slice."""
        self.tile_assignments[(block_i, block_j)] = slice_id

    def get_tile_slice(self, block_i: int, block_j: int) -> Optional[int]:
        """Get slice assigned to tile (i,j)."""
        return self.tile_assignments.get((block_i, block_j))

    def create_mapping(
        self,
        num_m_blocks: int,
        num_n_blocks: int,
    ):
        """
        Create checkerboard mapping for output tiles.

        Maps (i,j) output blocks to compute slices.
        """
        compute_slices = self.compute_slices()
        num_compute = len(compute_slices)

        for i in range(num_m_blocks):
            for j in range(num_n_blocks):
                # Round-robin assignment to compute slices
                slice_idx = (i * num_n_blocks + j) % num_compute
                self.assign_tile(i, j, compute_slices[slice_idx])

    def adjacent_memory_slices(self, compute_slice: int) -> List[int]:
        """Get memory slices adjacent to a compute slice."""
        neighbors = self.topology.neighbors(compute_slice)
        return [
            n for n in neighbors
            if self.slice_roles.get(n) == SliceRole.MEMORY
        ]

    def swap_roles(self):
        """
        Swap compute and memory roles.

        Used for tile rotation: memory slices become compute,
        compute becomes memory.
        """
        for slice_id in self.slice_roles:
            if self.slice_roles[slice_id] == SliceRole.COMPUTE:
                self.slice_roles[slice_id] = SliceRole.MEMORY
            elif self.slice_roles[slice_id] == SliceRole.MEMORY:
                self.slice_roles[slice_id] = SliceRole.COMPUTE


@dataclass
class DataDistribution:
    """
    Distribution of input/weight/output data across L3 slices.

    For matrix multiply C = A @ B:
    - A blocks distributed along M dimension
    - B blocks distributed along N dimension
    - C blocks at intersection of A row and B column owners
    """
    topology: DistributedL3
    pattern: CheckerboardPattern

    # Distribution of blocks
    a_distribution: Dict[Tuple[int, int], int] = field(default_factory=dict)
    b_distribution: Dict[Tuple[int, int], int] = field(default_factory=dict)
    c_distribution: Dict[Tuple[int, int], int] = field(default_factory=dict)

    def distribute_for_matmul(
        self,
        num_m_blocks: int,
        num_k_blocks: int,
        num_n_blocks: int,
    ):
        """
        Create data distribution for tiled matrix multiply.

        Distributes:
        - A blocks: M x K grid, assigned by row
        - B blocks: K x N grid, assigned by column
        - C blocks: M x N grid, at compute slices
        """
        compute_slices = self.pattern.compute_slices()
        memory_slices = self.pattern.memory_slices()

        # Distribute A by row (M dimension)
        for i in range(num_m_blocks):
            for k in range(num_k_blocks):
                # A row i goes to slices handling C row i
                slice_idx = i % len(memory_slices)
                self.a_distribution[(i, k)] = memory_slices[slice_idx]

        # Distribute B by column (N dimension)
        for k in range(num_k_blocks):
            for j in range(num_n_blocks):
                # B column j goes to slices handling C column j
                slice_idx = j % len(memory_slices)
                self.b_distribution[(k, j)] = memory_slices[slice_idx]

        # C blocks at compute slices
        for i in range(num_m_blocks):
            for j in range(num_n_blocks):
                slice_id = self.pattern.get_tile_slice(i, j)
                if slice_id is not None:
                    self.c_distribution[(i, j)] = slice_id

    def data_movement_for_op(
        self,
        block_i: int,
        block_j: int,
        block_k: int,
        a_block_bytes: int,
        b_block_bytes: int,
    ) -> List[TransferCost]:
        """
        Calculate data movement for single block operation.

        C[i,j] += A[i,k] @ B[k,j]

        Returns transfers needed to bring A and B to C's compute slice.
        """
        transfers = []

        # Where is C[i,j] computed?
        c_slice = self.c_distribution.get((block_i, block_j))
        if c_slice is None:
            return transfers

        # Where is A[i,k]?
        a_slice = self.a_distribution.get((block_i, block_k))
        if a_slice is not None and a_slice != c_slice:
            transfers.append(self.topology.transfer_cost(
                a_slice, c_slice, a_block_bytes
            ))

        # Where is B[k,j]?
        b_slice = self.b_distribution.get((block_k, block_j))
        if b_slice is not None and b_slice != c_slice:
            transfers.append(self.topology.transfer_cost(
                b_slice, c_slice, b_block_bytes
            ))

        return transfers


@dataclass
class DistributedMatmulAnalysis:
    """
    Analysis of distributed matrix multiply on L3 topology.

    Computes communication costs for different distributions.
    """
    M: int
    K: int
    N: int
    Tm: int
    Tk: int
    Tn: int
    dtype_bytes: int = 2

    topology: DistributedL3 = None
    pattern: CheckerboardPattern = None
    distribution: DataDistribution = None

    def __post_init__(self):
        if self.topology is None:
            # Default: 2x2 mesh
            self.topology = create_mesh_l3(2, 2)

        if self.pattern is None:
            self.pattern = CheckerboardPattern(topology=self.topology)
            self.pattern.create_mapping(
                ceil(self.M / self.Tm),
                ceil(self.N / self.Tn)
            )

        if self.distribution is None:
            self.distribution = DataDistribution(
                topology=self.topology,
                pattern=self.pattern
            )
            self.distribution.distribute_for_matmul(
                ceil(self.M / self.Tm),
                ceil(self.K / self.Tk),
                ceil(self.N / self.Tn)
            )

    @property
    def num_m_blocks(self) -> int:
        return ceil(self.M / self.Tm)

    @property
    def num_k_blocks(self) -> int:
        return ceil(self.K / self.Tk)

    @property
    def num_n_blocks(self) -> int:
        return ceil(self.N / self.Tn)

    @property
    def a_block_bytes(self) -> int:
        return self.Tm * self.Tk * self.dtype_bytes

    @property
    def b_block_bytes(self) -> int:
        return self.Tk * self.Tn * self.dtype_bytes

    @property
    def c_block_bytes(self) -> int:
        return self.Tm * self.Tn * 4  # FP32 accumulator

    def analyze_communication(self) -> Dict:
        """
        Analyze total communication cost.

        Returns breakdown of transfers by type.
        """
        total_transfers = []
        total_a_bytes = 0
        total_b_bytes = 0
        total_hops = 0

        for i in range(self.num_m_blocks):
            for j in range(self.num_n_blocks):
                for k in range(self.num_k_blocks):
                    transfers = self.distribution.data_movement_for_op(
                        i, j, k,
                        self.a_block_bytes,
                        self.b_block_bytes
                    )
                    for t in transfers:
                        total_transfers.append(t)
                        total_hops += t.hops
                        if "A" in str(t.src_slice):  # Approximation
                            total_a_bytes += t.bytes_transferred
                        else:
                            total_b_bytes += t.bytes_transferred

        total_energy = sum(t.energy_pj for t in total_transfers)
        total_latency = sum(t.latency_cycles for t in total_transfers)
        total_bytes = sum(t.bytes_transferred for t in total_transfers)

        return {
            'num_transfers': len(total_transfers),
            'total_bytes': total_bytes,
            'total_energy_pj': total_energy,
            'total_latency_cycles': total_latency,
            'total_hops': total_hops,
            'avg_hops_per_transfer': total_hops / len(total_transfers) if total_transfers else 0,
            'energy_per_byte_pj': total_energy / total_bytes if total_bytes else 0,
        }

    def compare_topologies(self) -> Dict:
        """Compare different topologies for this workload."""
        results = {}

        # Test different topologies
        topologies = [
            ("mesh_2x2", create_mesh_l3(2, 2)),
            ("mesh_4x4", create_mesh_l3(4, 4)),
            ("ring_4", create_ring_l3(4)),
            ("ring_8", create_ring_l3(8)),
        ]

        for name, topo in topologies:
            self.topology = topo
            self.pattern = CheckerboardPattern(topology=topo)
            self.pattern.create_mapping(self.num_m_blocks, self.num_n_blocks)
            self.distribution = DataDistribution(
                topology=topo, pattern=self.pattern
            )
            self.distribution.distribute_for_matmul(
                self.num_m_blocks, self.num_k_blocks, self.num_n_blocks
            )

            results[name] = self.analyze_communication()

        return results
