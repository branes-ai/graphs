"""
Enhanced Tile Schedule with Hierarchical Blocking

Provides comprehensive tile scheduling with memory constraints, hierarchical
blocking, and integration with memory models and block algebra.

Key concepts:
- TileConfig: Configuration for tile dimensions and memory constraints
- TileSchedule: Single-level tile schedule with reuse analysis
- HierarchicalTileSchedule: Multi-level blocking for large problems
- TileScheduler: Generate schedules respecting memory constraints
- ScheduleComparison: Compare different schedules on energy/latency

This module bridges the gap between:
- block_algebra.py (mathematical block decomposition)
- memory_model.py (hardware capacity constraints)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Iterator
from math import ceil
import json

from .block_algebra import (
    BlockMatrix, BlockSchedule, BlockScheduleGenerator,
    HierarchicalBlockSchedule, DataType, OperandRole,
    analyze_reuse_from_schedule
)
from .memory_model import (
    MemoryBudget, MemoryHierarchy, MemoryLevel,
    WorkingSetState
)


class DataflowType(Enum):
    """Dataflow strategies for systolic arrays."""
    WEIGHT_STATIONARY = "WS"      # TPU style - weights stay in array
    OUTPUT_STATIONARY = "OS"      # Outputs stay in accumulators
    INPUT_STATIONARY = "IS"       # Inputs reused across outputs
    ROW_STATIONARY = "RS"         # Eyeriss style - balanced reuse
    NO_LOCAL_REUSE = "NLR"        # Baseline - no reuse


@dataclass
class TileConfig:
    """
    Configuration for tile scheduling.

    Captures problem dimensions, tile sizes, and memory constraints.
    """
    # Problem dimensions
    M: int
    K: int
    N: int

    # Tile dimensions
    Tm: int
    Tk: int
    Tn: int

    # Data types
    input_dtype: DataType = DataType.BF16
    weight_dtype: DataType = DataType.BF16
    output_dtype: DataType = DataType.FP32

    # Dataflow strategy
    dataflow: DataflowType = DataflowType.WEIGHT_STATIONARY

    # Hardware constraints
    array_size: int = 128       # Systolic array dimension
    num_arrays: int = 1         # Parallel arrays

    def __post_init__(self):
        # Validate tile sizes
        if self.Tm <= 0 or self.Tk <= 0 or self.Tn <= 0:
            raise ValueError("Tile dimensions must be positive")

    @property
    def num_m_tiles(self) -> int:
        return ceil(self.M / self.Tm)

    @property
    def num_k_tiles(self) -> int:
        return ceil(self.K / self.Tk)

    @property
    def num_n_tiles(self) -> int:
        return ceil(self.N / self.Tn)

    @property
    def total_tiles(self) -> int:
        return self.num_m_tiles * self.num_k_tiles * self.num_n_tiles

    @property
    def input_tile_bytes(self) -> int:
        """Bytes for one input tile A[Tm, Tk]."""
        return self.Tm * self.Tk * self.input_dtype.value

    @property
    def weight_tile_bytes(self) -> int:
        """Bytes for one weight tile B[Tk, Tn]."""
        return self.Tk * self.Tn * self.weight_dtype.value

    @property
    def output_tile_bytes(self) -> int:
        """Bytes for one output tile C[Tm, Tn]."""
        return self.Tm * self.Tn * self.output_dtype.value

    @property
    def working_set_bytes(self) -> int:
        """Total working set for one tile operation."""
        return (
            self.input_tile_bytes +
            self.weight_tile_bytes +
            self.output_tile_bytes
        )

    @property
    def total_flops(self) -> int:
        """Total FLOPs for complete matmul."""
        return 2 * self.M * self.K * self.N

    @property
    def flops_per_tile(self) -> int:
        """FLOPs per tile operation."""
        return 2 * self.Tm * self.Tk * self.Tn

    def utilization(self) -> float:
        """
        Compute utilization for systolic array.

        Accounts for:
        - Edge tiles (partial fill)
        - Array dimension matching
        """
        # M dimension utilization
        m_util = min(1.0, self.Tm / self.array_size)
        # N dimension utilization
        n_util = min(1.0, self.Tn / self.array_size)
        # Spatial utilization
        spatial_util = m_util * n_util

        # Pipeline fill/drain overhead
        # depth / (K + depth) inefficiency
        pipeline_overhead = self.array_size / (self.Tk + self.array_size)
        effective_util = spatial_util * (1 - pipeline_overhead)

        return effective_util

    def loop_order_from_dataflow(self) -> str:
        """Get loop order string from dataflow type."""
        mapping = {
            DataflowType.WEIGHT_STATIONARY: "NKM",
            DataflowType.OUTPUT_STATIONARY: "MNK",
            DataflowType.INPUT_STATIONARY: "MKN",
            DataflowType.ROW_STATIONARY: "KMN",
            DataflowType.NO_LOCAL_REUSE: "MNK",
        }
        return mapping.get(self.dataflow, "MNK")


@dataclass
class TileEvent:
    """
    Single event in tile execution timeline.

    Events track data movement and computation.
    """
    cycle: int
    event_type: str  # "load", "store", "compute", "prefetch", "stall"
    tile_id: str     # e.g., "A[0,1]", "B[1,2]"
    level: str       # "L1", "L2", "L3", "DRAM"
    bytes: int = 0
    duration_cycles: int = 0
    description: str = ""


@dataclass
class TileSchedule:
    """
    Complete tile schedule with execution timeline.

    Provides:
    - Ordered tile operations
    - Data movement events
    - Reuse analysis
    - Working set tracking
    """
    config: TileConfig
    block_schedule: BlockSchedule = None

    # Timeline events
    events: List[TileEvent] = field(default_factory=list)

    # Reuse statistics
    input_reuse: float = 0.0
    weight_reuse: float = 0.0
    output_reuse: float = 0.0

    # Memory traffic
    dram_reads_bytes: int = 0
    dram_writes_bytes: int = 0
    l3_reads_bytes: int = 0
    l3_writes_bytes: int = 0
    l2_reads_bytes: int = 0
    l2_writes_bytes: int = 0

    # Timing
    total_cycles: int = 0
    compute_cycles: int = 0
    memory_cycles: int = 0
    stall_cycles: int = 0

    def __post_init__(self):
        if self.block_schedule is None:
            self._generate_block_schedule()
        self._analyze_reuse()

    def _generate_block_schedule(self):
        """Generate block schedule from config."""
        generator = BlockScheduleGenerator(
            M=self.config.M,
            K=self.config.K,
            N=self.config.N,
            Tm=self.config.Tm,
            Tk=self.config.Tk,
            Tn=self.config.Tn,
            dtype=self.config.input_dtype
        )
        loop_order = self.config.loop_order_from_dataflow()
        self.block_schedule = generator.generate(loop_order)

    def _analyze_reuse(self):
        """Analyze reuse from block schedule."""
        if self.block_schedule is None:
            return

        reuse = analyze_reuse_from_schedule(self.block_schedule)
        self.input_reuse = reuse['input_reuse']
        self.weight_reuse = reuse['weight_reuse']
        self.output_reuse = reuse['output_reuse']

    def add_event(self, event: TileEvent):
        """Add event to timeline."""
        self.events.append(event)

    def compute_memory_traffic(self) -> Dict[str, int]:
        """
        Compute actual memory traffic based on reuse.

        Traffic depends on:
        - Problem size
        - Tile sizes
        - Dataflow (determines which operand stays in cache)
        """
        cfg = self.config

        # Minimum traffic (perfect reuse)
        min_a_bytes = cfg.M * cfg.K * cfg.input_dtype.value
        min_b_bytes = cfg.K * cfg.N * cfg.weight_dtype.value
        min_c_bytes = cfg.M * cfg.N * cfg.output_dtype.value  # write only

        # Actual traffic depends on dataflow and cache capacity
        # For weight-stationary: B loaded once, A streamed per K tile
        # For output-stationary: C stays, A and B streamed

        if cfg.dataflow == DataflowType.WEIGHT_STATIONARY:
            # B loaded once per (k,n) tile, reused across M
            b_loads = cfg.num_k_tiles * cfg.num_n_tiles
            a_loads = cfg.num_m_tiles * cfg.num_k_tiles * cfg.num_n_tiles
            c_writes = cfg.num_m_tiles * cfg.num_n_tiles
        elif cfg.dataflow == DataflowType.OUTPUT_STATIONARY:
            # C stays, accumulated across K
            a_loads = cfg.num_m_tiles * cfg.num_k_tiles
            b_loads = cfg.num_k_tiles * cfg.num_n_tiles
            c_writes = cfg.num_m_tiles * cfg.num_n_tiles
        else:
            # Assume no reuse for other cases
            a_loads = cfg.total_tiles
            b_loads = cfg.total_tiles
            c_writes = cfg.num_m_tiles * cfg.num_n_tiles

        return {
            'A_bytes': a_loads * cfg.input_tile_bytes,
            'B_bytes': b_loads * cfg.weight_tile_bytes,
            'C_bytes': c_writes * cfg.output_tile_bytes,
            'total_read_bytes': (
                a_loads * cfg.input_tile_bytes +
                b_loads * cfg.weight_tile_bytes
            ),
            'total_write_bytes': c_writes * cfg.output_tile_bytes,
            'min_a_bytes': min_a_bytes,
            'min_b_bytes': min_b_bytes,
            'min_c_bytes': min_c_bytes,
        }

    def arithmetic_intensity(self) -> float:
        """
        Compute arithmetic intensity (FLOPs / byte).

        Higher is better - means more compute per memory access.
        """
        traffic = self.compute_memory_traffic()
        total_bytes = traffic['total_read_bytes'] + traffic['total_write_bytes']
        if total_bytes == 0:
            return float('inf')
        return self.config.total_flops / total_bytes

    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            'config': {
                'M': self.config.M,
                'K': self.config.K,
                'N': self.config.N,
                'Tm': self.config.Tm,
                'Tk': self.config.Tk,
                'Tn': self.config.Tn,
                'dataflow': self.config.dataflow.value,
                'array_size': self.config.array_size,
            },
            'reuse': {
                'input': self.input_reuse,
                'weight': self.weight_reuse,
                'output': self.output_reuse,
            },
            'traffic': self.compute_memory_traffic(),
            'arithmetic_intensity': self.arithmetic_intensity(),
            'utilization': self.config.utilization(),
            'total_flops': self.config.total_flops,
            'total_tiles': self.config.total_tiles,
        }


@dataclass
class HierarchicalTileSchedule:
    """
    Multi-level hierarchical tile schedule.

    When single-level tiles exceed L2 capacity, we introduce blocking:

    Level 0 (innermost): L1-resident micro-tiles (systolic array)
    Level 1: L2-resident tiles (double-buffered)
    Level 2: L3-resident blocks (tile rotation candidates)
    Level 3 (outermost): DRAM-resident super-blocks

    Example for large matmul:
        Problem: (16384, 16384, 16384)
        L1 micro-tile: (128, 128, 128) - fits in registers/L1
        L2 tile: (512, 512, 512) - fits in L2 with double-buffering
        L3 block: (2048, 2048, 2048) - fits in L3 slice
        DRAM super-block: (8192, 8192, 8192) - streaming from DRAM
    """
    # Problem dimensions
    M: int
    K: int
    N: int

    # Hierarchical tile sizes (inner to outer)
    l1_tile: Tuple[int, int, int]     # (Tm_l1, Tk_l1, Tn_l1)
    l2_tile: Tuple[int, int, int]     # (Tm_l2, Tk_l2, Tn_l2)
    l3_block: Optional[Tuple[int, int, int]] = None
    dram_superblock: Optional[Tuple[int, int, int]] = None

    # Dataflows at each level
    l1_dataflow: DataflowType = DataflowType.WEIGHT_STATIONARY
    l2_dataflow: DataflowType = DataflowType.OUTPUT_STATIONARY
    l3_dataflow: DataflowType = DataflowType.OUTPUT_STATIONARY

    # Data types
    input_dtype: DataType = DataType.BF16
    weight_dtype: DataType = DataType.BF16
    output_dtype: DataType = DataType.FP32

    # Schedules at each level
    l1_schedule: Optional[TileSchedule] = None
    l2_schedule: Optional[TileSchedule] = None
    l3_schedule: Optional[TileSchedule] = None

    def __post_init__(self):
        self._validate()
        self._generate_schedules()

    def _validate(self):
        """Validate hierarchical tile configuration."""
        # L1 tile must divide L2 tile evenly (or nearly)
        for i, (l1, l2) in enumerate(zip(self.l1_tile, self.l2_tile)):
            if l2 < l1:
                raise ValueError(
                    f"L2 tile dimension {i} ({l2}) smaller than L1 ({l1})"
                )

        # L3 block must contain L2 tiles
        if self.l3_block is not None:
            for i, (l2, l3) in enumerate(zip(self.l2_tile, self.l3_block)):
                if l3 < l2:
                    raise ValueError(
                        f"L3 block dimension {i} ({l3}) smaller than L2 ({l2})"
                    )

    def _generate_schedules(self):
        """Generate schedules at each level."""
        # L2 schedule (main tile schedule)
        Tm2, Tk2, Tn2 = self.l2_tile
        l2_config = TileConfig(
            M=self.M, K=self.K, N=self.N,
            Tm=Tm2, Tk=Tk2, Tn=Tn2,
            input_dtype=self.input_dtype,
            weight_dtype=self.weight_dtype,
            output_dtype=self.output_dtype,
            dataflow=self.l2_dataflow,
        )
        self.l2_schedule = TileSchedule(config=l2_config)

        # L1 schedule (within each L2 tile)
        Tm1, Tk1, Tn1 = self.l1_tile
        l1_config = TileConfig(
            M=Tm2, K=Tk2, N=Tn2,  # L1 tiles within L2 tile
            Tm=Tm1, Tk=Tk1, Tn=Tn1,
            input_dtype=self.input_dtype,
            weight_dtype=self.weight_dtype,
            output_dtype=self.output_dtype,
            dataflow=self.l1_dataflow,
        )
        self.l1_schedule = TileSchedule(config=l1_config)

    @property
    def num_levels(self) -> int:
        """Number of blocking levels."""
        levels = 2  # Always have L1 and L2
        if self.l3_block is not None:
            levels += 1
        if self.dram_superblock is not None:
            levels += 1
        return levels

    def working_set_at_level(self, level: str, dtype_bytes: int = 2) -> int:
        """
        Working set size at specified level.

        Args:
            level: "L1", "L2", "L3"
            dtype_bytes: Element size (default BF16)
        """
        if level == "L1":
            Tm, Tk, Tn = self.l1_tile
        elif level == "L2":
            Tm, Tk, Tn = self.l2_tile
        elif level == "L3" and self.l3_block is not None:
            Tm, Tk, Tn = self.l3_block
        else:
            return 0

        input_bytes = Tm * Tk * dtype_bytes
        weight_bytes = Tk * Tn * dtype_bytes
        output_bytes = Tm * Tn * 4  # FP32 accumulator
        return input_bytes + weight_bytes + output_bytes

    def fits_in_budget(self, budget: MemoryBudget) -> Dict[str, bool]:
        """Check if schedule fits in memory budget at each level."""
        return {
            'L1': self.working_set_at_level('L1') <= budget.effective_l1_bytes,
            'L2': self.working_set_at_level('L2') <= budget.effective_l2_bytes,
            'L3': (
                self.l3_block is None or
                self.working_set_at_level('L3') <= budget.effective_l3_bytes
            ),
        }

    def total_l1_operations(self) -> int:
        """Total number of L1 micro-tile operations."""
        # L1 ops within each L2 tile
        Tm1, Tk1, Tn1 = self.l1_tile
        Tm2, Tk2, Tn2 = self.l2_tile

        l1_per_l2 = (
            ceil(Tm2 / Tm1) *
            ceil(Tk2 / Tk1) *
            ceil(Tn2 / Tn1)
        )

        # L2 tiles in problem
        l2_tiles = (
            ceil(self.M / Tm2) *
            ceil(self.K / Tk2) *
            ceil(self.N / Tn2)
        )

        return l1_per_l2 * l2_tiles

    def summary(self) -> Dict:
        """Generate summary of hierarchical schedule."""
        return {
            'problem': {'M': self.M, 'K': self.K, 'N': self.N},
            'levels': self.num_levels,
            'l1_tile': self.l1_tile,
            'l2_tile': self.l2_tile,
            'l3_block': self.l3_block,
            'working_sets': {
                'L1': self.working_set_at_level('L1'),
                'L2': self.working_set_at_level('L2'),
                'L3': self.working_set_at_level('L3') if self.l3_block else 0,
            },
            'total_l1_ops': self.total_l1_operations(),
            'l2_schedule_reuse': {
                'input': self.l2_schedule.input_reuse if self.l2_schedule else 0,
                'weight': self.l2_schedule.weight_reuse if self.l2_schedule else 0,
                'output': self.l2_schedule.output_reuse if self.l2_schedule else 0,
            },
        }


class TileScheduler:
    """
    Generate tile schedules respecting memory constraints.

    Automatically determines:
    - Tile sizes for each memory level
    - Number of blocking levels needed
    - Dataflow at each level
    """

    def __init__(self, budget: MemoryBudget, hierarchy: Optional[MemoryHierarchy] = None):
        self.budget = budget
        self.hierarchy = hierarchy

    def schedule_matmul(
        self,
        M: int, K: int, N: int,
        array_size: int = 128,
        dataflow: DataflowType = DataflowType.WEIGHT_STATIONARY,
    ) -> HierarchicalTileSchedule:
        """
        Generate optimal tile schedule for matrix multiply.

        Args:
            M, K, N: Problem dimensions
            array_size: Systolic array size
            dataflow: Primary dataflow strategy

        Returns:
            HierarchicalTileSchedule with optimal blocking
        """
        # Find L1 tile (systolic array native size)
        l1_tile = self._find_l1_tile(array_size)

        # Find L2 tile that fits with double-buffering
        l2_tile = self._find_l2_tile(M, K, N, array_size)

        # Check if we need L3 blocking
        l3_block = None
        if self._needs_l3_blocking(M, K, N, l2_tile):
            l3_block = self._find_l3_block(M, K, N, l2_tile)

        return HierarchicalTileSchedule(
            M=M, K=K, N=N,
            l1_tile=l1_tile,
            l2_tile=l2_tile,
            l3_block=l3_block,
            l1_dataflow=dataflow,
            l2_dataflow=DataflowType.OUTPUT_STATIONARY,
        )

    def _find_l1_tile(self, array_size: int) -> Tuple[int, int, int]:
        """
        Find L1 micro-tile (systolic array native).

        For most systolic arrays, this is (array_size, array_size, array_size).
        """
        return (array_size, array_size, array_size)

    def _find_l2_tile(
        self,
        M: int, K: int, N: int,
        array_size: int,
    ) -> Tuple[int, int, int]:
        """
        Find largest L2 tile that fits with double-buffering.

        Tile size should be:
        - Multiple of array_size for utilization
        - Working set fits in effective L2
        """
        effective_l2 = self.budget.effective_l2_bytes

        # Working set: Tm*Tk*2 + Tk*Tn*2 + Tm*Tn*4 (FP32 acc)
        # For square tiles T: T^2 * (2 + 2 + 4) = 8*T^2
        max_t = int((effective_l2 / 8) ** 0.5)

        # Round down to multiple of array_size
        tile_dim = (max_t // array_size) * array_size
        if tile_dim == 0:
            tile_dim = array_size  # Minimum

        # Cap at problem dimensions
        Tm = min(tile_dim, M)
        Tk = min(tile_dim, K)
        Tn = min(tile_dim, N)

        return (Tm, Tk, Tn)

    def _needs_l3_blocking(
        self,
        M: int, K: int, N: int,
        l2_tile: Tuple[int, int, int],
    ) -> bool:
        """Check if problem needs L3-level blocking."""
        Tm, Tk, Tn = l2_tile

        # Need L3 blocking if many L2 tiles
        num_l2_tiles = ceil(M / Tm) * ceil(K / Tk) * ceil(N / Tn)

        # If more than 64 L2 tiles, benefit from L3 blocking
        return num_l2_tiles > 64

    def _find_l3_block(
        self,
        M: int, K: int, N: int,
        l2_tile: Tuple[int, int, int],
    ) -> Tuple[int, int, int]:
        """Find L3 block size (multiple L2 tiles)."""
        effective_l3 = self.budget.effective_l3_bytes
        Tm2, Tk2, Tn2 = l2_tile

        # Working set for L3 block
        max_t = int((effective_l3 / 8) ** 0.5)

        # Must be multiple of L2 tile
        tiles_per_dim = max(1, max_t // Tm2)

        Tm3 = min(tiles_per_dim * Tm2, M)
        Tk3 = min(tiles_per_dim * Tk2, K)
        Tn3 = min(tiles_per_dim * Tn2, N)

        return (Tm3, Tk3, Tn3)


@dataclass
class ScheduleComparison:
    """
    Compare multiple tile schedules.

    Useful for:
    - Comparing dataflow strategies
    - Comparing tile sizes
    - Finding Pareto-optimal configurations
    """
    schedules: List[TileSchedule] = field(default_factory=list)

    def add(self, schedule: TileSchedule, name: str = None):
        """Add schedule to comparison."""
        self.schedules.append(schedule)

    def compare_reuse(self) -> List[Dict]:
        """Compare reuse characteristics."""
        results = []
        for sched in self.schedules:
            results.append({
                'dataflow': sched.config.dataflow.value,
                'tile': (sched.config.Tm, sched.config.Tk, sched.config.Tn),
                'input_reuse': sched.input_reuse,
                'weight_reuse': sched.weight_reuse,
                'output_reuse': sched.output_reuse,
                'arithmetic_intensity': sched.arithmetic_intensity(),
            })
        return results

    def compare_traffic(self) -> List[Dict]:
        """Compare memory traffic."""
        results = []
        for sched in self.schedules:
            traffic = sched.compute_memory_traffic()
            results.append({
                'dataflow': sched.config.dataflow.value,
                'tile': (sched.config.Tm, sched.config.Tk, sched.config.Tn),
                'total_read': traffic['total_read_bytes'],
                'total_write': traffic['total_write_bytes'],
                'A_bytes': traffic['A_bytes'],
                'B_bytes': traffic['B_bytes'],
                'C_bytes': traffic['C_bytes'],
            })
        return results

    def best_by_metric(self, metric: str) -> TileSchedule:
        """Find best schedule by specified metric."""
        if metric == 'arithmetic_intensity':
            return max(self.schedules, key=lambda s: s.arithmetic_intensity())
        elif metric == 'utilization':
            return max(self.schedules, key=lambda s: s.config.utilization())
        elif metric == 'traffic':
            return min(
                self.schedules,
                key=lambda s: sum(s.compute_memory_traffic().values())
            )
        else:
            raise ValueError(f"Unknown metric: {metric}")


def compare_dataflows(
    M: int, K: int, N: int,
    Tm: int, Tk: int, Tn: int,
) -> ScheduleComparison:
    """
    Compare all dataflow strategies for given problem and tile size.

    Returns comparison object with schedules for each dataflow.
    """
    comparison = ScheduleComparison()

    for dataflow in DataflowType:
        if dataflow == DataflowType.NO_LOCAL_REUSE:
            continue  # Skip baseline

        config = TileConfig(
            M=M, K=K, N=N,
            Tm=Tm, Tk=Tk, Tn=Tn,
            dataflow=dataflow,
        )
        schedule = TileSchedule(config=config)
        comparison.add(schedule)

    return comparison


def sweep_tile_sizes(
    M: int, K: int, N: int,
    tile_sizes: List[int],
    budget: MemoryBudget,
    dataflow: DataflowType = DataflowType.WEIGHT_STATIONARY,
) -> List[TileSchedule]:
    """
    Sweep tile sizes and return schedules that fit in budget.

    Args:
        M, K, N: Problem dimensions
        tile_sizes: List of tile sizes to try (assumes square tiles)
        budget: Memory budget constraints
        dataflow: Dataflow strategy

    Returns:
        List of valid schedules sorted by arithmetic intensity
    """
    valid_schedules = []

    for T in tile_sizes:
        # Check if fits
        working_set = budget.tile_working_set(T, T, T)
        if working_set > budget.effective_l2_bytes:
            continue

        config = TileConfig(
            M=M, K=K, N=N,
            Tm=T, Tk=T, Tn=T,
            dataflow=dataflow,
        )
        schedule = TileSchedule(config=config)
        valid_schedules.append(schedule)

    # Sort by arithmetic intensity (higher is better)
    valid_schedules.sort(key=lambda s: s.arithmetic_intensity(), reverse=True)

    return valid_schedules
