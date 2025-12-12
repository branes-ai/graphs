"""
Block Algebra for Tile Scheduling

Represents tiled matrix operations in terms of actual 2D tile objects
that move through the memory hierarchy.

Key insight: There is no such thing as a 3D tile. For C = A @ B:
- A_tile[i,k]: 2D submatrix of shape (Tm, Tk)
- B_tile[k,j]: 2D submatrix of shape (Tk, Tn)
- C_tile[i,j]: 2D submatrix of shape (Tm, Tn)

The (Tm, Tk, Tn) notation describes loop bounds, not tile shapes.
This module explicitly tracks each tile type as a distinct 2D object
with its own memory footprint, reuse pattern, and lifetime.

Block Matrix Multiply:
    For C = A @ B with shapes:
        A: (M, K) -> partitioned into (M/Tm, K/Tk) tiles of shape (Tm, Tk)
        B: (K, N) -> partitioned into (K/Tk, N/Tn) tiles of shape (Tk, Tn)
        C: (M, N) -> partitioned into (M/Tm, N/Tn) tiles of shape (Tm, Tn)

    C[i,j] = sum_k(A[i,k] @ B[k,j])  where i,j,k are tile indices
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Iterator
from math import ceil


class DataType(Enum):
    """Data types with byte sizes."""
    FP32 = 4
    FP16 = 2
    BF16 = 2
    INT8 = 1
    INT4 = 0.5


@dataclass
class TileShape:
    """
    Shape of a 2D tile.

    This is the fundamental unit - tiles are always 2D submatrices.
    """
    rows: int
    cols: int
    dtype: DataType = DataType.BF16

    @property
    def elements(self) -> int:
        """Number of elements in tile."""
        return self.rows * self.cols

    @property
    def bytes(self) -> int:
        """Memory footprint in bytes."""
        return int(self.rows * self.cols * self.dtype.value)

    def __repr__(self):
        return f"({self.rows}, {self.cols}) @ {self.dtype.name} = {self.bytes} bytes"


@dataclass
class ATile:
    """
    Input activation tile: submatrix of A with shape (Tm, Tk).

    A_tile[i,k] is used for computing C[i,j] for all j.
    Reuse factor = N/Tn (number of output columns).
    """
    shape: TileShape  # (Tm, Tk)
    block_i: int      # Row block index (0 to M/Tm - 1)
    block_k: int      # Column block index (0 to K/Tk - 1)

    @property
    def tile_id(self) -> str:
        return f"A[{self.block_i},{self.block_k}]"

    @property
    def bytes(self) -> int:
        return self.shape.bytes

    def reuse_factor(self, num_n_tiles: int) -> int:
        """
        How many times this tile is used before it can be evicted.

        A[i,k] is used for all j in range(num_n_tiles).
        """
        return num_n_tiles


@dataclass
class BTile:
    """
    Weight tile: submatrix of B with shape (Tk, Tn).

    B_tile[k,j] is used for computing C[i,j] for all i.
    Reuse factor = M/Tm (number of output rows).
    """
    shape: TileShape  # (Tk, Tn)
    block_k: int      # Row block index (0 to K/Tk - 1)
    block_j: int      # Column block index (0 to N/Tn - 1)

    @property
    def tile_id(self) -> str:
        return f"B[{self.block_k},{self.block_j}]"

    @property
    def bytes(self) -> int:
        return self.shape.bytes

    def reuse_factor(self, num_m_tiles: int) -> int:
        """
        How many times this tile is used before it can be evicted.

        B[k,j] is used for all i in range(num_m_tiles).
        """
        return num_m_tiles


@dataclass
class CTile:
    """
    Output/accumulator tile: submatrix of C with shape (Tm, Tn).

    C_tile[i,j] is accumulated over all k.
    Accumulation count = K/Tk (number of reduction steps).

    Note: Accumulators typically use higher precision (FP32) than inputs.
    """
    shape: TileShape  # (Tm, Tn) - note: dtype is typically FP32
    block_i: int      # Row block index (0 to M/Tm - 1)
    block_j: int      # Column block index (0 to N/Tn - 1)

    @property
    def tile_id(self) -> str:
        return f"C[{self.block_i},{self.block_j}]"

    @property
    def bytes(self) -> int:
        return self.shape.bytes

    def accumulation_count(self, num_k_tiles: int) -> int:
        """
        How many partial products are accumulated into this tile.

        C[i,j] += A[i,k] @ B[k,j] for all k in range(num_k_tiles).
        """
        return num_k_tiles


@dataclass
class TileSet:
    """
    The set of tiles involved in one tiled matmul operation.

    This is the working set that must fit in cache for computation.
    """
    a_tile: ATile
    b_tile: BTile
    c_tile: CTile

    @property
    def working_set_bytes(self) -> int:
        """Total bytes for all three tiles."""
        return self.a_tile.bytes + self.b_tile.bytes + self.c_tile.bytes

    @property
    def input_bytes(self) -> int:
        """Bytes that must be loaded (A and B)."""
        return self.a_tile.bytes + self.b_tile.bytes

    @property
    def output_bytes(self) -> int:
        """Bytes that must be written (C)."""
        return self.c_tile.bytes

    def summary(self) -> Dict:
        return {
            'A_tile': {'shape': (self.a_tile.shape.rows, self.a_tile.shape.cols),
                       'bytes': self.a_tile.bytes},
            'B_tile': {'shape': (self.b_tile.shape.rows, self.b_tile.shape.cols),
                       'bytes': self.b_tile.bytes},
            'C_tile': {'shape': (self.c_tile.shape.rows, self.c_tile.shape.cols),
                       'bytes': self.c_tile.bytes},
            'working_set_bytes': self.working_set_bytes,
        }


@dataclass
class MatmulTiling:
    """
    Complete tiling specification for C = A @ B.

    Specifies how the three matrices are partitioned into tiles.
    This is the correct representation: three 2D tile shapes, not one 3D shape.
    """
    # Problem dimensions
    M: int  # A rows, C rows
    K: int  # A cols, B rows (reduction dimension)
    N: int  # B cols, C cols

    # Tile dimensions (these define the 2D tile shapes)
    Tm: int  # A tile rows, C tile rows
    Tk: int  # A tile cols, B tile rows
    Tn: int  # B tile cols, C tile cols

    # Data types
    input_dtype: DataType = DataType.BF16
    weight_dtype: DataType = DataType.BF16
    accum_dtype: DataType = DataType.FP32

    def __post_init__(self):
        # Create the three tile shapes
        self.a_tile_shape = TileShape(self.Tm, self.Tk, self.input_dtype)
        self.b_tile_shape = TileShape(self.Tk, self.Tn, self.weight_dtype)
        self.c_tile_shape = TileShape(self.Tm, self.Tn, self.accum_dtype)

    @property
    def num_m_tiles(self) -> int:
        """Number of tiles along M dimension."""
        return ceil(self.M / self.Tm)

    @property
    def num_k_tiles(self) -> int:
        """Number of tiles along K dimension (reduction)."""
        return ceil(self.K / self.Tk)

    @property
    def num_n_tiles(self) -> int:
        """Number of tiles along N dimension."""
        return ceil(self.N / self.Tn)

    @property
    def total_a_tiles(self) -> int:
        """Total number of unique A tiles."""
        return self.num_m_tiles * self.num_k_tiles

    @property
    def total_b_tiles(self) -> int:
        """Total number of unique B tiles."""
        return self.num_k_tiles * self.num_n_tiles

    @property
    def total_c_tiles(self) -> int:
        """Total number of unique C tiles."""
        return self.num_m_tiles * self.num_n_tiles

    @property
    def a_reuse_factor(self) -> int:
        """Each A tile is used this many times."""
        return self.num_n_tiles

    @property
    def b_reuse_factor(self) -> int:
        """Each B tile is used this many times."""
        return self.num_m_tiles

    @property
    def c_accumulation_count(self) -> int:
        """Each C tile is accumulated this many times."""
        return self.num_k_tiles

    @property
    def working_set_bytes(self) -> int:
        """
        Minimum working set: one tile of each type.

        This must fit in the innermost cache level for computation.
        """
        return (self.a_tile_shape.bytes +
                self.b_tile_shape.bytes +
                self.c_tile_shape.bytes)

    @property
    def total_flops(self) -> int:
        """Total floating point operations."""
        return 2 * self.M * self.K * self.N

    @property
    def flops_per_tile_op(self) -> int:
        """FLOPs for one tile multiply-accumulate."""
        return 2 * self.Tm * self.Tk * self.Tn

    def get_a_tile(self, i: int, k: int) -> ATile:
        """Get A tile at block position (i, k)."""
        return ATile(
            shape=self.a_tile_shape,
            block_i=i,
            block_k=k,
        )

    def get_b_tile(self, k: int, j: int) -> BTile:
        """Get B tile at block position (k, j)."""
        return BTile(
            shape=self.b_tile_shape,
            block_k=k,
            block_j=j,
        )

    def get_c_tile(self, i: int, j: int) -> CTile:
        """Get C tile at block position (i, j)."""
        return CTile(
            shape=self.c_tile_shape,
            block_i=i,
            block_j=j,
        )

    def get_tile_set(self, i: int, j: int, k: int) -> TileSet:
        """Get the three tiles involved in computing C[i,j] += A[i,k] @ B[k,j]."""
        return TileSet(
            a_tile=self.get_a_tile(i, k),
            b_tile=self.get_b_tile(k, j),
            c_tile=self.get_c_tile(i, j),
        )

    def iterate_a_tiles(self) -> Iterator[ATile]:
        """Iterate over all unique A tiles."""
        for i in range(self.num_m_tiles):
            for k in range(self.num_k_tiles):
                yield self.get_a_tile(i, k)

    def iterate_b_tiles(self) -> Iterator[BTile]:
        """Iterate over all unique B tiles."""
        for k in range(self.num_k_tiles):
            for j in range(self.num_n_tiles):
                yield self.get_b_tile(k, j)

    def iterate_c_tiles(self) -> Iterator[CTile]:
        """Iterate over all unique C tiles."""
        for i in range(self.num_m_tiles):
            for j in range(self.num_n_tiles):
                yield self.get_c_tile(i, j)

    def summary(self) -> Dict:
        """Summary of tiling configuration."""
        return {
            'problem': {'M': self.M, 'K': self.K, 'N': self.N},
            'A_tile': {
                'shape': (self.Tm, self.Tk),
                'bytes': self.a_tile_shape.bytes,
                'count': self.total_a_tiles,
                'reuse_factor': self.a_reuse_factor,
            },
            'B_tile': {
                'shape': (self.Tk, self.Tn),
                'bytes': self.b_tile_shape.bytes,
                'count': self.total_b_tiles,
                'reuse_factor': self.b_reuse_factor,
            },
            'C_tile': {
                'shape': (self.Tm, self.Tn),
                'bytes': self.c_tile_shape.bytes,
                'count': self.total_c_tiles,
                'accumulation_count': self.c_accumulation_count,
            },
            'working_set_bytes': self.working_set_bytes,
            'total_flops': self.total_flops,
        }


@dataclass
class TileOperation:
    """
    Single tile operation: C[i,j] += A[i,k] @ B[k,j]

    Represents one step in the tiled matmul where three specific
    2D tiles interact.
    """
    tile_set: TileSet
    is_first_accumulation: bool = False  # True if k=0 (C = A@B vs C += A@B)

    @property
    def flops(self) -> int:
        """FLOPs for this tile operation."""
        a = self.tile_set.a_tile.shape
        b = self.tile_set.b_tile.shape
        return 2 * a.rows * a.cols * b.cols

    def __repr__(self):
        op = "=" if self.is_first_accumulation else "+="
        return (f"{self.tile_set.c_tile.tile_id} {op} "
                f"{self.tile_set.a_tile.tile_id} @ {self.tile_set.b_tile.tile_id}")


class LoopOrder(Enum):
    """
    Loop ordering for tiled matmul.

    The loop order determines which tiles stay in cache (innermost loops)
    and which are streamed (outermost loops).
    """
    # Output-stationary: C stays, A and B stream
    MNK = "MNK"  # for i: for j: for k: C[i,j] += A[i,k] @ B[k,j]

    # Weight-stationary: B stays, A and C stream
    NKM = "NKM"  # for j: for k: for i: C[i,j] += A[i,k] @ B[k,j]

    # Input-stationary: A stays, B and C stream
    MKN = "MKN"  # for i: for k: for j: C[i,j] += A[i,k] @ B[k,j]

    # Other orderings
    KMN = "KMN"
    KNM = "KNM"
    NMK = "NMK"


@dataclass
class TileSchedule:
    """
    Ordered sequence of tile operations.

    Determined by loop ordering. Tracks which tiles are live at each step.
    """
    tiling: MatmulTiling
    loop_order: LoopOrder
    operations: List[TileOperation] = field(default_factory=list)

    def __post_init__(self):
        if not self.operations:
            self._generate_operations()

    def _generate_operations(self):
        """Generate operations according to loop order."""
        loops = list(self.loop_order.value)
        ranges = {
            'M': range(self.tiling.num_m_tiles),
            'N': range(self.tiling.num_n_tiles),
            'K': range(self.tiling.num_k_tiles),
        }

        # Track first k for each (i,j) to mark first accumulation
        first_k: Dict[Tuple[int, int], bool] = {}

        for idx0 in ranges[loops[0]]:
            for idx1 in ranges[loops[1]]:
                for idx2 in ranges[loops[2]]:
                    indices = {loops[0]: idx0, loops[1]: idx1, loops[2]: idx2}
                    i, j, k = indices['M'], indices['N'], indices['K']

                    is_first = (i, j) not in first_k
                    if is_first:
                        first_k[(i, j)] = True

                    tile_set = self.tiling.get_tile_set(i, j, k)
                    op = TileOperation(
                        tile_set=tile_set,
                        is_first_accumulation=is_first,
                    )
                    self.operations.append(op)

    def __iter__(self):
        return iter(self.operations)

    def __len__(self):
        return len(self.operations)

    @property
    def total_flops(self) -> int:
        return sum(op.flops for op in self.operations)

    def analyze_tile_lifetimes(self) -> Dict:
        """
        Analyze when each tile is first used and last used.

        This determines how long tiles must stay in cache.
        """
        a_first_use: Dict[str, int] = {}
        a_last_use: Dict[str, int] = {}
        b_first_use: Dict[str, int] = {}
        b_last_use: Dict[str, int] = {}
        c_first_use: Dict[str, int] = {}
        c_last_use: Dict[str, int] = {}

        for step, op in enumerate(self.operations):
            a_id = op.tile_set.a_tile.tile_id
            b_id = op.tile_set.b_tile.tile_id
            c_id = op.tile_set.c_tile.tile_id

            if a_id not in a_first_use:
                a_first_use[a_id] = step
            a_last_use[a_id] = step

            if b_id not in b_first_use:
                b_first_use[b_id] = step
            b_last_use[b_id] = step

            if c_id not in c_first_use:
                c_first_use[c_id] = step
            c_last_use[c_id] = step

        def lifetime_stats(first: Dict, last: Dict) -> Dict:
            lifetimes = [last[k] - first[k] for k in first]
            return {
                'count': len(first),
                'min_lifetime': min(lifetimes) if lifetimes else 0,
                'max_lifetime': max(lifetimes) if lifetimes else 0,
                'avg_lifetime': sum(lifetimes) / len(lifetimes) if lifetimes else 0,
            }

        return {
            'A_tiles': lifetime_stats(a_first_use, a_last_use),
            'B_tiles': lifetime_stats(b_first_use, b_last_use),
            'C_tiles': lifetime_stats(c_first_use, c_last_use),
        }

    def live_tiles_at_step(self, step: int) -> Dict[str, List[str]]:
        """
        Which tiles must be in cache at a given step.

        A tile is live from first use to last use.
        """
        # This is expensive - cache the lifetime analysis
        if not hasattr(self, '_lifetimes'):
            self._lifetimes = self._compute_lifetimes()

        live_a = [tid for tid, (first, last) in self._lifetimes['A'].items()
                  if first <= step <= last]
        live_b = [tid for tid, (first, last) in self._lifetimes['B'].items()
                  if first <= step <= last]
        live_c = [tid for tid, (first, last) in self._lifetimes['C'].items()
                  if first <= step <= last]

        return {'A': live_a, 'B': live_b, 'C': live_c}

    def _compute_lifetimes(self) -> Dict:
        """Compute (first_use, last_use) for each tile."""
        lifetimes = {'A': {}, 'B': {}, 'C': {}}

        for step, op in enumerate(self.operations):
            a_id = op.tile_set.a_tile.tile_id
            b_id = op.tile_set.b_tile.tile_id
            c_id = op.tile_set.c_tile.tile_id

            if a_id not in lifetimes['A']:
                lifetimes['A'][a_id] = [step, step]
            lifetimes['A'][a_id][1] = step

            if b_id not in lifetimes['B']:
                lifetimes['B'][b_id] = [step, step]
            lifetimes['B'][b_id][1] = step

            if c_id not in lifetimes['C']:
                lifetimes['C'][c_id] = [step, step]
            lifetimes['C'][c_id][1] = step

        return lifetimes

    def peak_live_tiles(self) -> Dict[str, int]:
        """
        Maximum number of tiles of each type live at any point.

        This determines minimum cache capacity needed.
        """
        max_a = max_b = max_c = 0

        for step in range(len(self.operations)):
            live = self.live_tiles_at_step(step)
            max_a = max(max_a, len(live['A']))
            max_b = max(max_b, len(live['B']))
            max_c = max(max_c, len(live['C']))

        return {'A': max_a, 'B': max_b, 'C': max_c}

    def peak_working_set_bytes(self) -> int:
        """
        Maximum bytes of tiles live at any point.
        """
        peak = self.peak_live_tiles()
        return (peak['A'] * self.tiling.a_tile_shape.bytes +
                peak['B'] * self.tiling.b_tile_shape.bytes +
                peak['C'] * self.tiling.c_tile_shape.bytes)


def analyze_memory_traffic(
    tiling: MatmulTiling,
    loop_order: LoopOrder,
) -> Dict:
    """
    Analyze memory traffic for different loop orderings.

    Returns bytes loaded/stored for each tile type, accounting for reuse.
    """
    # Minimum traffic (each tile loaded exactly once)
    min_a_bytes = tiling.total_a_tiles * tiling.a_tile_shape.bytes
    min_b_bytes = tiling.total_b_tiles * tiling.b_tile_shape.bytes
    min_c_bytes = tiling.total_c_tiles * tiling.c_tile_shape.bytes

    # Actual traffic depends on loop order and cache capacity
    # Here we assume tiles can stay in cache for their reuse window

    if loop_order == LoopOrder.MNK:
        # Output-stationary: C[i,j] stays for all k
        # for i: for j: for k: need A[i,*] and B[*,j] for each (i,j)
        a_loads = tiling.num_m_tiles * tiling.num_k_tiles  # A[i,k] per row i
        b_loads = tiling.total_b_tiles  # B[k,j] loaded once per j,k
        c_loads = 0  # C stays in accumulators
        c_stores = tiling.total_c_tiles

    elif loop_order == LoopOrder.NKM:
        # Weight-stationary: B[k,j] stays for all i
        # for j: for k: for i: B[k,j] loaded once, A[i,k] for each i
        a_loads = tiling.total_a_tiles * tiling.num_n_tiles  # A reloaded for each j
        b_loads = tiling.total_b_tiles  # B loaded once
        c_loads = tiling.total_c_tiles * (tiling.num_k_tiles - 1)  # C read-modify-write
        c_stores = tiling.total_c_tiles * tiling.num_k_tiles

    elif loop_order == LoopOrder.MKN:
        # Input-stationary: A[i,k] stays for all j
        # for i: for k: for j: A[i,k] loaded once, B[k,j] for each j
        a_loads = tiling.total_a_tiles  # A loaded once
        b_loads = tiling.total_b_tiles * tiling.num_m_tiles  # B reloaded for each i
        c_loads = tiling.total_c_tiles * (tiling.num_k_tiles - 1)
        c_stores = tiling.total_c_tiles * tiling.num_k_tiles

    else:
        # Default: assume no reuse
        a_loads = tiling.num_m_tiles * tiling.num_k_tiles * tiling.num_n_tiles
        b_loads = tiling.num_m_tiles * tiling.num_k_tiles * tiling.num_n_tiles
        c_loads = tiling.total_c_tiles * tiling.num_k_tiles
        c_stores = tiling.total_c_tiles * tiling.num_k_tiles

    actual_a_bytes = a_loads * tiling.a_tile_shape.bytes
    actual_b_bytes = b_loads * tiling.b_tile_shape.bytes
    actual_c_read_bytes = c_loads * tiling.c_tile_shape.bytes
    actual_c_write_bytes = c_stores * tiling.c_tile_shape.bytes

    total_read = actual_a_bytes + actual_b_bytes + actual_c_read_bytes
    total_write = actual_c_write_bytes

    return {
        'A_tile': {
            'shape': (tiling.Tm, tiling.Tk),
            'bytes_per_tile': tiling.a_tile_shape.bytes,
            'unique_tiles': tiling.total_a_tiles,
            'loads': a_loads,
            'minimum_bytes': min_a_bytes,
            'actual_bytes': actual_a_bytes,
            'reuse_achieved': min_a_bytes / actual_a_bytes if actual_a_bytes > 0 else 1,
        },
        'B_tile': {
            'shape': (tiling.Tk, tiling.Tn),
            'bytes_per_tile': tiling.b_tile_shape.bytes,
            'unique_tiles': tiling.total_b_tiles,
            'loads': b_loads,
            'minimum_bytes': min_b_bytes,
            'actual_bytes': actual_b_bytes,
            'reuse_achieved': min_b_bytes / actual_b_bytes if actual_b_bytes > 0 else 1,
        },
        'C_tile': {
            'shape': (tiling.Tm, tiling.Tn),
            'bytes_per_tile': tiling.c_tile_shape.bytes,
            'unique_tiles': tiling.total_c_tiles,
            'reads': c_loads,
            'writes': c_stores,
            'minimum_bytes': min_c_bytes,
            'actual_read_bytes': actual_c_read_bytes,
            'actual_write_bytes': actual_c_write_bytes,
        },
        'total': {
            'minimum_bytes': min_a_bytes + min_b_bytes + min_c_bytes,
            'actual_read_bytes': total_read,
            'actual_write_bytes': total_write,
            'actual_total_bytes': total_read + total_write,
        },
        'arithmetic_intensity': tiling.total_flops / (total_read + total_write),
    }
