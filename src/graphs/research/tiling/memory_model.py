"""
Memory Hierarchy Model

Models memory hierarchy with capacity constraints for tile scheduling.
Supports multi-level hierarchies with configurable capacities, bandwidths,
and energy costs.

Key concepts:
- MemoryLevel: Single level in hierarchy (L1, L2, L3, DRAM)
- MemoryHierarchy: Complete hierarchy with inter-level transfers
- MemoryBudget: Capacity constraints for tile scheduling
- WorkingSet: Tracks live data at each level
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple
from math import ceil


class MemoryLevelType(Enum):
    """Types of memory levels in hierarchy."""
    REGISTER_FILE = "RF"
    L1_CACHE = "L1"
    L1_SCRATCHPAD = "L1_SPM"  # Software-managed
    L2_CACHE = "L2"
    L2_SCRATCHPAD = "L2_SPM"
    L3_CACHE = "L3"
    L3_DISTRIBUTED = "L3_DIST"  # Distributed L3 slices
    HBM = "HBM"
    DRAM = "DRAM"
    LPDDR = "LPDDR"


@dataclass
class MemoryLevel:
    """
    Single level in memory hierarchy.

    Models capacity, bandwidth, latency, and energy characteristics.
    """
    name: str
    level_type: MemoryLevelType
    capacity_bytes: int

    # Bandwidth (bytes/cycle at this level)
    read_bandwidth_bytes_per_cycle: float
    write_bandwidth_bytes_per_cycle: float

    # Latency (cycles)
    read_latency_cycles: int
    write_latency_cycles: int

    # Energy (pJ per byte)
    read_energy_pj_per_byte: float
    write_energy_pj_per_byte: float

    # Cache-specific
    line_size_bytes: int = 64
    associativity: int = 8

    # Scratchpad-specific (software-managed)
    is_software_managed: bool = False

    # Distributed memory
    is_distributed: bool = False
    num_slices: int = 1
    slice_capacity_bytes: Optional[int] = None

    def __post_init__(self):
        if self.is_distributed and self.slice_capacity_bytes is None:
            self.slice_capacity_bytes = self.capacity_bytes // self.num_slices

    @property
    def effective_capacity(self) -> int:
        """Effective capacity accounting for overhead."""
        if self.is_software_managed:
            return self.capacity_bytes  # No tag overhead
        else:
            # ~5% overhead for tags in caches
            return int(self.capacity_bytes * 0.95)

    def transfer_time_cycles(self, bytes_to_transfer: int, is_write: bool = False) -> int:
        """Time to transfer given bytes."""
        bw = self.write_bandwidth_bytes_per_cycle if is_write else self.read_bandwidth_bytes_per_cycle
        transfer_cycles = ceil(bytes_to_transfer / bw)
        latency = self.write_latency_cycles if is_write else self.read_latency_cycles
        return latency + transfer_cycles

    def transfer_energy_pj(self, bytes_to_transfer: int, is_write: bool = False) -> float:
        """Energy to transfer given bytes."""
        energy_per_byte = self.write_energy_pj_per_byte if is_write else self.read_energy_pj_per_byte
        return bytes_to_transfer * energy_per_byte


@dataclass
class MemoryHierarchy:
    """
    Complete memory hierarchy model.

    Ordered from fastest/smallest (index 0) to slowest/largest.
    """
    levels: List[MemoryLevel]
    name: str = "default"

    def __post_init__(self):
        # Build name-to-level mapping
        self._level_map: Dict[str, MemoryLevel] = {
            level.name: level for level in self.levels
        }
        self._type_map: Dict[MemoryLevelType, MemoryLevel] = {
            level.level_type: level for level in self.levels
        }

    def __getitem__(self, key) -> MemoryLevel:
        """Get level by name or type."""
        if isinstance(key, str):
            return self._level_map[key]
        elif isinstance(key, MemoryLevelType):
            return self._type_map[key]
        elif isinstance(key, int):
            return self.levels[key]
        raise KeyError(f"Unknown key type: {type(key)}")

    def get_level(self, name: str) -> Optional[MemoryLevel]:
        """Get level by name, returns None if not found."""
        return self._level_map.get(name)

    def capacity(self, level_name: str) -> int:
        """Get capacity of named level."""
        return self._level_map[level_name].capacity_bytes

    def transfer_cost(self, src_level: str, dst_level: str,
                      bytes_to_transfer: int) -> Tuple[int, float]:
        """
        Cost to transfer data between levels.

        Returns (cycles, energy_pj).
        """
        src = self._level_map[src_level]
        dst = self._level_map[dst_level]

        # Read from source
        read_cycles = src.transfer_time_cycles(bytes_to_transfer, is_write=False)
        read_energy = src.transfer_energy_pj(bytes_to_transfer, is_write=False)

        # Write to destination
        write_cycles = dst.transfer_time_cycles(bytes_to_transfer, is_write=True)
        write_energy = dst.transfer_energy_pj(bytes_to_transfer, is_write=True)

        # Transfers can overlap, take max of read/write time
        total_cycles = max(read_cycles, write_cycles)
        total_energy = read_energy + write_energy

        return total_cycles, total_energy

    @property
    def total_on_chip_capacity(self) -> int:
        """Total on-chip memory (excluding DRAM/HBM)."""
        on_chip_types = {
            MemoryLevelType.REGISTER_FILE,
            MemoryLevelType.L1_CACHE,
            MemoryLevelType.L1_SCRATCHPAD,
            MemoryLevelType.L2_CACHE,
            MemoryLevelType.L2_SCRATCHPAD,
            MemoryLevelType.L3_CACHE,
            MemoryLevelType.L3_DISTRIBUTED,
        }
        return sum(
            level.capacity_bytes
            for level in self.levels
            if level.level_type in on_chip_types
        )


@dataclass
class MemoryBudget:
    """
    Memory capacity constraints for tile scheduling.

    Specifies how much memory is available at each level for
    tile data (inputs, weights, outputs, accumulators).
    """
    # Capacity budgets (bytes available for tiles)
    l1_bytes: int = 256 * 1024           # 256 KB
    l2_bytes: int = 4 * 1024 * 1024      # 4 MB
    l3_bytes: int = 32 * 1024 * 1024     # 32 MB

    # Double-buffering settings
    double_buffer_l1: bool = False       # Usually too small
    double_buffer_l2: bool = True        # Standard practice
    double_buffer_l3: bool = False       # Architecture dependent

    # Element size (bytes)
    input_dtype_bytes: int = 2           # BF16 inputs
    weight_dtype_bytes: int = 2          # BF16 weights
    accumulator_dtype_bytes: int = 4     # FP32 accumulators

    # Allocation strategy
    l1_allocation: str = "static"        # "static" or "dynamic"
    l2_allocation: str = "static"
    l3_allocation: str = "shared"        # "shared", "partitioned", "distributed"

    @property
    def effective_l1_bytes(self) -> int:
        """L1 capacity accounting for double-buffering."""
        return self.l1_bytes // 2 if self.double_buffer_l1 else self.l1_bytes

    @property
    def effective_l2_bytes(self) -> int:
        """L2 capacity accounting for double-buffering."""
        return self.l2_bytes // 2 if self.double_buffer_l2 else self.l2_bytes

    @property
    def effective_l3_bytes(self) -> int:
        """L3 capacity accounting for double-buffering."""
        return self.l3_bytes // 2 if self.double_buffer_l3 else self.l3_bytes

    def tile_working_set(self, Tm: int, Tk: int, Tn: int) -> int:
        """
        Working set size for a tile (Tm, Tk, Tn).

        Includes:
        - Input tile: Tm x Tk
        - Weight tile: Tk x Tn
        - Output tile: Tm x Tn (accumulator precision)
        """
        input_bytes = Tm * Tk * self.input_dtype_bytes
        weight_bytes = Tk * Tn * self.weight_dtype_bytes
        output_bytes = Tm * Tn * self.accumulator_dtype_bytes
        return input_bytes + weight_bytes + output_bytes

    def fits_in_l1(self, Tm: int, Tk: int, Tn: int) -> bool:
        """Check if tile fits in L1."""
        return self.tile_working_set(Tm, Tk, Tn) <= self.effective_l1_bytes

    def fits_in_l2(self, Tm: int, Tk: int, Tn: int) -> bool:
        """Check if tile fits in L2."""
        return self.tile_working_set(Tm, Tk, Tn) <= self.effective_l2_bytes

    def fits_in_l3(self, Tm: int, Tk: int, Tn: int) -> bool:
        """Check if tile fits in L3."""
        return self.tile_working_set(Tm, Tk, Tn) <= self.effective_l3_bytes

    def max_tile_for_level(self, level: str, aspect_ratio: float = 1.0) -> Tuple[int, int, int]:
        """
        Find maximum tile size that fits in given level.

        Args:
            level: "L1", "L2", or "L3"
            aspect_ratio: Tm/Tn ratio (1.0 = square tiles)

        Returns:
            (Tm, Tk, Tn) maximum tile dimensions
        """
        if level == "L1":
            budget = self.effective_l1_bytes
        elif level == "L2":
            budget = self.effective_l2_bytes
        elif level == "L3":
            budget = self.effective_l3_bytes
        else:
            raise ValueError(f"Unknown level: {level}")

        # Solve for maximum dimension D where:
        # D^2 * (input_bytes + weight_bytes) + D^2 * acc_bytes <= budget
        # Assuming square tiles (Tm = Tk = Tn = D) for simplicity
        bytes_per_tile_element = (
            self.input_dtype_bytes +
            self.weight_dtype_bytes +
            self.accumulator_dtype_bytes
        )

        # D^2 * bytes_per_element <= budget
        max_d_squared = budget / bytes_per_tile_element
        max_d = int(max_d_squared ** 0.5)

        # Round down to power of 2 for alignment
        max_d = 2 ** (max_d.bit_length() - 1) if max_d > 0 else 1

        Tm = int(max_d * (aspect_ratio ** 0.5))
        Tn = int(max_d / (aspect_ratio ** 0.5))
        Tk = max_d

        return Tm, Tk, Tn


@dataclass
class WorkingSetState:
    """
    Current state of working set at each memory level.

    Tracks which tiles/data are currently resident at each level.
    """
    # Bytes currently in use at each level
    l1_used: int = 0
    l2_used: int = 0
    l3_used: int = 0

    # Tile identifiers currently resident
    l1_tiles: List[str] = field(default_factory=list)
    l2_tiles: List[str] = field(default_factory=list)
    l3_tiles: List[str] = field(default_factory=list)

    def add_tile(self, level: str, tile_id: str, bytes_size: int):
        """Add tile to specified level."""
        if level == "L1":
            self.l1_used += bytes_size
            self.l1_tiles.append(tile_id)
        elif level == "L2":
            self.l2_used += bytes_size
            self.l2_tiles.append(tile_id)
        elif level == "L3":
            self.l3_used += bytes_size
            self.l3_tiles.append(tile_id)

    def remove_tile(self, level: str, tile_id: str, bytes_size: int):
        """Remove tile from specified level."""
        if level == "L1":
            self.l1_used -= bytes_size
            self.l1_tiles.remove(tile_id)
        elif level == "L2":
            self.l2_used -= bytes_size
            self.l2_tiles.remove(tile_id)
        elif level == "L3":
            self.l3_used -= bytes_size
            self.l3_tiles.remove(tile_id)

    def exceeds_capacity(self, budget: MemoryBudget) -> Dict[str, bool]:
        """Check which levels exceed capacity."""
        return {
            "L1": self.l1_used > budget.effective_l1_bytes,
            "L2": self.l2_used > budget.effective_l2_bytes,
            "L3": self.l3_used > budget.effective_l3_bytes,
        }


# =============================================================================
# Predefined Memory Hierarchies
# =============================================================================

def create_tpu_v4_hierarchy() -> MemoryHierarchy:
    """Google TPU v4 memory hierarchy."""
    return MemoryHierarchy(
        name="TPU_v4",
        levels=[
            MemoryLevel(
                name="VMEM",
                level_type=MemoryLevelType.L1_SCRATCHPAD,
                capacity_bytes=16 * 1024 * 1024,  # 16 MB VMEM
                read_bandwidth_bytes_per_cycle=256,
                write_bandwidth_bytes_per_cycle=256,
                read_latency_cycles=1,
                write_latency_cycles=1,
                read_energy_pj_per_byte=0.5,
                write_energy_pj_per_byte=0.5,
                is_software_managed=True,
            ),
            MemoryLevel(
                name="CMEM",
                level_type=MemoryLevelType.L2_SCRATCHPAD,
                capacity_bytes=32 * 1024 * 1024,  # 32 MB CMEM (shared)
                read_bandwidth_bytes_per_cycle=128,
                write_bandwidth_bytes_per_cycle=128,
                read_latency_cycles=10,
                write_latency_cycles=10,
                read_energy_pj_per_byte=2.0,
                write_energy_pj_per_byte=2.0,
                is_software_managed=True,
            ),
            MemoryLevel(
                name="HBM",
                level_type=MemoryLevelType.HBM,
                capacity_bytes=32 * 1024 * 1024 * 1024,  # 32 GB HBM
                read_bandwidth_bytes_per_cycle=64,  # ~1.2 TB/s at 1.5 GHz
                write_bandwidth_bytes_per_cycle=64,
                read_latency_cycles=200,
                write_latency_cycles=200,
                read_energy_pj_per_byte=5.5,
                write_energy_pj_per_byte=5.5,
            ),
        ]
    )


def create_h100_hierarchy() -> MemoryHierarchy:
    """NVIDIA H100 memory hierarchy."""
    return MemoryHierarchy(
        name="H100",
        levels=[
            MemoryLevel(
                name="SMEM",
                level_type=MemoryLevelType.L1_SCRATCHPAD,
                capacity_bytes=228 * 1024,  # 228 KB shared memory per SM
                read_bandwidth_bytes_per_cycle=128,
                write_bandwidth_bytes_per_cycle=128,
                read_latency_cycles=20,
                write_latency_cycles=20,
                read_energy_pj_per_byte=0.8,
                write_energy_pj_per_byte=0.8,
                is_software_managed=True,
            ),
            MemoryLevel(
                name="L2",
                level_type=MemoryLevelType.L2_CACHE,
                capacity_bytes=50 * 1024 * 1024,  # 50 MB L2
                read_bandwidth_bytes_per_cycle=64,
                write_bandwidth_bytes_per_cycle=64,
                read_latency_cycles=100,
                write_latency_cycles=100,
                read_energy_pj_per_byte=1.5,
                write_energy_pj_per_byte=1.5,
            ),
            MemoryLevel(
                name="HBM3",
                level_type=MemoryLevelType.HBM,
                capacity_bytes=80 * 1024 * 1024 * 1024,  # 80 GB HBM3
                read_bandwidth_bytes_per_cycle=128,  # ~3.35 TB/s
                write_bandwidth_bytes_per_cycle=128,
                read_latency_cycles=300,
                write_latency_cycles=300,
                read_energy_pj_per_byte=5.5,
                write_energy_pj_per_byte=5.5,
            ),
        ]
    )


def create_distributed_l3_hierarchy(
    num_slices: int = 8,
    slice_capacity_mb: int = 4,
    l2_per_cluster_kb: int = 512,
) -> MemoryHierarchy:
    """
    Distributed L3 hierarchy for Branes.AI style SoCs.

    Models checkerboard topology with multiple L3 slices.
    """
    total_l3_bytes = num_slices * slice_capacity_mb * 1024 * 1024

    return MemoryHierarchy(
        name=f"DistributedL3_{num_slices}slices",
        levels=[
            MemoryLevel(
                name="L1_SPM",
                level_type=MemoryLevelType.L1_SCRATCHPAD,
                capacity_bytes=256 * 1024,  # 256 KB per tile
                read_bandwidth_bytes_per_cycle=64,
                write_bandwidth_bytes_per_cycle=64,
                read_latency_cycles=1,
                write_latency_cycles=1,
                read_energy_pj_per_byte=0.4,
                write_energy_pj_per_byte=0.4,
                is_software_managed=True,
            ),
            MemoryLevel(
                name="L2_SPM",
                level_type=MemoryLevelType.L2_SCRATCHPAD,
                capacity_bytes=l2_per_cluster_kb * 1024,
                read_bandwidth_bytes_per_cycle=32,
                write_bandwidth_bytes_per_cycle=32,
                read_latency_cycles=5,
                write_latency_cycles=5,
                read_energy_pj_per_byte=1.0,
                write_energy_pj_per_byte=1.0,
                is_software_managed=True,
            ),
            MemoryLevel(
                name="L3_DIST",
                level_type=MemoryLevelType.L3_DISTRIBUTED,
                capacity_bytes=total_l3_bytes,
                read_bandwidth_bytes_per_cycle=16,  # Per slice
                write_bandwidth_bytes_per_cycle=16,
                read_latency_cycles=20,
                write_latency_cycles=20,
                read_energy_pj_per_byte=3.0,
                write_energy_pj_per_byte=3.0,
                is_distributed=True,
                num_slices=num_slices,
                slice_capacity_bytes=slice_capacity_mb * 1024 * 1024,
            ),
            MemoryLevel(
                name="LPDDR5",
                level_type=MemoryLevelType.LPDDR,
                capacity_bytes=16 * 1024 * 1024 * 1024,  # 16 GB
                read_bandwidth_bytes_per_cycle=8,  # ~50 GB/s
                write_bandwidth_bytes_per_cycle=8,
                read_latency_cycles=500,
                write_latency_cycles=500,
                read_energy_pj_per_byte=8.0,
                write_energy_pj_per_byte=8.0,
            ),
        ]
    )


def create_kpu_hierarchy() -> MemoryHierarchy:
    """Stillwater KPU memory hierarchy (EDDO - Explicit Data Distribution)."""
    return MemoryHierarchy(
        name="KPU_EDDO",
        levels=[
            MemoryLevel(
                name="TileSPM",
                level_type=MemoryLevelType.L1_SCRATCHPAD,
                capacity_bytes=256 * 1024,  # 256 KB per tile
                read_bandwidth_bytes_per_cycle=64,
                write_bandwidth_bytes_per_cycle=64,
                read_latency_cycles=1,
                write_latency_cycles=1,
                read_energy_pj_per_byte=0.2,  # Lower due to no tags
                write_energy_pj_per_byte=0.2,
                is_software_managed=True,
            ),
            MemoryLevel(
                name="GlobalSPM",
                level_type=MemoryLevelType.L2_SCRATCHPAD,
                capacity_bytes=16 * 1024 * 1024,  # 16 MB global scratchpad
                read_bandwidth_bytes_per_cycle=32,
                write_bandwidth_bytes_per_cycle=32,
                read_latency_cycles=10,
                write_latency_cycles=10,
                read_energy_pj_per_byte=0.75,  # ~50% of cache due to no tags
                write_energy_pj_per_byte=0.75,
                is_software_managed=True,
            ),
            MemoryLevel(
                name="StreamBuffer",
                level_type=MemoryLevelType.L3_CACHE,
                capacity_bytes=64 * 1024 * 1024,  # 64 MB streaming buffer
                read_bandwidth_bytes_per_cycle=16,
                write_bandwidth_bytes_per_cycle=16,
                read_latency_cycles=30,
                write_latency_cycles=30,
                read_energy_pj_per_byte=1.2,
                write_energy_pj_per_byte=1.2,
            ),
            MemoryLevel(
                name="LPDDR5",
                level_type=MemoryLevelType.LPDDR,
                capacity_bytes=32 * 1024 * 1024 * 1024,  # 32 GB
                read_bandwidth_bytes_per_cycle=8,
                write_bandwidth_bytes_per_cycle=8,
                read_latency_cycles=500,
                write_latency_cycles=500,
                read_energy_pj_per_byte=8.0,
                write_energy_pj_per_byte=8.0,
            ),
        ]
    )


# =============================================================================
# Predefined Memory Budgets
# =============================================================================

TPU_V4_BUDGET = MemoryBudget(
    l1_bytes=16 * 1024 * 1024,   # VMEM
    l2_bytes=32 * 1024 * 1024,   # CMEM
    l3_bytes=0,                   # No L3
    double_buffer_l1=True,
    double_buffer_l2=True,
    double_buffer_l3=False,
)

H100_BUDGET = MemoryBudget(
    l1_bytes=228 * 1024,          # Shared memory per SM
    l2_bytes=50 * 1024 * 1024,    # L2 cache
    l3_bytes=0,                   # No L3
    double_buffer_l1=True,
    double_buffer_l2=False,       # L2 is cache, not software-managed
    double_buffer_l3=False,
)

DISTRIBUTED_L3_BUDGET = MemoryBudget(
    l1_bytes=256 * 1024,          # Per-tile SPM
    l2_bytes=512 * 1024,          # Per-cluster SPM
    l3_bytes=4 * 1024 * 1024,     # Per-slice L3
    double_buffer_l1=False,
    double_buffer_l2=True,
    double_buffer_l3=True,        # Enable for tile rotation
)

KPU_BUDGET = MemoryBudget(
    l1_bytes=256 * 1024,          # Tile SPM
    l2_bytes=16 * 1024 * 1024,    # Global SPM
    l3_bytes=64 * 1024 * 1024,    # Stream buffer
    double_buffer_l1=False,
    double_buffer_l2=True,
    double_buffer_l3=True,
)
