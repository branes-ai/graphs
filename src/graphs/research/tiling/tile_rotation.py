"""
Tile Rotation Algorithms for Distributed Memory

Implements tile rotation algorithms for distributed L3 architectures:
- Cannon's Algorithm: Systolic rotation for square decompositions
- SUMMA: Broadcast-based for rectangular decompositions
- 2.5D Algorithm: Uses extra memory to reduce communication

Key insight: After computing C_ij = sum_k(A_ik @ B_kj), partial results
can rotate to adjacent L3 slices where they become inputs for the next
computation, avoiding DRAM round-trips.

References:
- Cannon (1969): A cellular computer to implement the Kalman Filter Algorithm
- Van De Geijn & Watts (1997): SUMMA: Scalable Universal Matrix Multiplication
- Solomonik & Demmel (2011): Communication-optimal parallel 2.5D matrix multiplication
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Iterator
from math import ceil, sqrt
import copy

from .distributed_l3 import (
    DistributedL3, L3Slice, Position, TransferCost,
    CheckerboardPattern, create_mesh_l3
)


class RotationDirection(Enum):
    """Direction of tile rotation."""
    LEFT = "left"
    RIGHT = "right"
    UP = "up"
    DOWN = "down"


@dataclass
class TileState:
    """
    State of a tile in the rotation algorithm.

    Tracks which data block is at which processor/slice.
    """
    tile_type: str      # "A", "B", or "C"
    block_i: int        # Row index in block matrix
    block_j: int        # Column index in block matrix
    current_slice: int  # Current L3 slice ID
    bytes_size: int

    def copy(self) -> 'TileState':
        return TileState(
            tile_type=self.tile_type,
            block_i=self.block_i,
            block_j=self.block_j,
            current_slice=self.current_slice,
            bytes_size=self.bytes_size,
        )


@dataclass
class RotationStep:
    """
    Single step in rotation algorithm.

    Describes data movement in one rotation phase.
    """
    step_number: int
    transfers: List[Tuple[TileState, int, int]]  # (tile, src, dst)

    # Compute operations after rotation
    compute_ops: List[Tuple[int, int, int]]  # (i, j, k) block operations

    # Cost metrics
    transfer_bytes: int = 0
    transfer_cycles: int = 0
    transfer_energy_pj: float = 0.0
    compute_cycles: int = 0

    def add_transfer(self, tile: TileState, src: int, dst: int):
        self.transfers.append((tile, src, dst))

    def add_compute(self, i: int, j: int, k: int):
        self.compute_ops.append((i, j, k))


@dataclass
class RotationSchedule:
    """
    Complete rotation schedule for distributed matmul.

    Sequence of rotation steps with data movement and compute.
    """
    algorithm: str  # "cannon", "summa", "2.5d"
    steps: List[RotationStep] = field(default_factory=list)

    # Problem parameters
    M: int = 0
    K: int = 0
    N: int = 0
    Tm: int = 0
    Tk: int = 0
    Tn: int = 0
    num_procs: int = 0

    # Summary metrics
    total_transfer_bytes: int = 0
    total_transfer_cycles: int = 0
    total_transfer_energy_pj: float = 0.0
    total_compute_cycles: int = 0

    def add_step(self, step: RotationStep):
        self.steps.append(step)
        self.total_transfer_bytes += step.transfer_bytes
        self.total_transfer_cycles += step.transfer_cycles
        self.total_transfer_energy_pj += step.transfer_energy_pj
        self.total_compute_cycles += step.compute_cycles

    def __iter__(self):
        return iter(self.steps)

    def summary(self) -> Dict:
        return {
            'algorithm': self.algorithm,
            'problem': {'M': self.M, 'K': self.K, 'N': self.N},
            'tile': {'Tm': self.Tm, 'Tk': self.Tk, 'Tn': self.Tn},
            'num_procs': self.num_procs,
            'num_steps': len(self.steps),
            'total_transfer_bytes': self.total_transfer_bytes,
            'total_transfer_cycles': self.total_transfer_cycles,
            'total_transfer_energy_pj': self.total_transfer_energy_pj,
            'total_compute_cycles': self.total_compute_cycles,
            'communication_to_compute_ratio': (
                self.total_transfer_cycles / self.total_compute_cycles
                if self.total_compute_cycles > 0 else float('inf')
            ),
        }


class CannonAlgorithm:
    """
    Cannon's Algorithm for distributed matrix multiply.

    For P = p x p processors computing C = A @ B:
    1. Initial skew: A[i,:] left-shifted by i, B[:,j] up-shifted by j
    2. For k = 0 to p-1:
       a. Each processor computes C_local += A_local @ B_local
       b. A tiles shift left (wrap around)
       c. B tiles shift up (wrap around)

    Properties:
    - Requires square processor grid
    - O(n^2 / sqrt(P)) communication per processor
    - Memory: O(n^2 / P) per processor
    - Perfectly load-balanced
    """

    def __init__(
        self,
        topology: DistributedL3,
        M: int, K: int, N: int,
        Tm: int, Tk: int, Tn: int,
        dtype_bytes: int = 2,
    ):
        self.topology = topology
        self.M = M
        self.K = K
        self.N = N
        self.Tm = Tm
        self.Tk = Tk
        self.Tn = Tn
        self.dtype_bytes = dtype_bytes

        # Processor grid dimensions
        self.p = int(sqrt(topology.num_slices))
        if self.p * self.p != topology.num_slices:
            raise ValueError(
                f"Cannon's algorithm requires square grid, got {topology.num_slices} slices"
            )

        # Block counts
        self.num_m = ceil(M / Tm)
        self.num_k = ceil(K / Tk)
        self.num_n = ceil(N / Tn)

        # Current tile positions
        self.a_tiles: Dict[Tuple[int, int], TileState] = {}
        self.b_tiles: Dict[Tuple[int, int], TileState] = {}
        self.c_tiles: Dict[Tuple[int, int], TileState] = {}

    def _proc_id(self, row: int, col: int) -> int:
        """Get processor ID from grid position."""
        return row * self.p + col

    def _proc_pos(self, proc_id: int) -> Tuple[int, int]:
        """Get grid position from processor ID."""
        return (proc_id // self.p, proc_id % self.p)

    def _left_neighbor(self, proc_id: int) -> int:
        """Get processor to the left (with wraparound)."""
        row, col = self._proc_pos(proc_id)
        return self._proc_id(row, (col - 1) % self.p)

    def _right_neighbor(self, proc_id: int) -> int:
        """Get processor to the right (with wraparound)."""
        row, col = self._proc_pos(proc_id)
        return self._proc_id(row, (col + 1) % self.p)

    def _up_neighbor(self, proc_id: int) -> int:
        """Get processor above (with wraparound)."""
        row, col = self._proc_pos(proc_id)
        return self._proc_id((row - 1) % self.p, col)

    def _down_neighbor(self, proc_id: int) -> int:
        """Get processor below (with wraparound)."""
        row, col = self._proc_pos(proc_id)
        return self._proc_id((row + 1) % self.p, col)

    def initial_distribution(self) -> RotationStep:
        """
        Create initial data distribution with skewing.

        A[i,:] is skewed left by i positions.
        B[:,j] is skewed up by j positions.
        """
        step = RotationStep(step_number=0, transfers=[], compute_ops=[])

        a_bytes = self.Tm * self.Tk * self.dtype_bytes
        b_bytes = self.Tk * self.Tn * self.dtype_bytes
        c_bytes = self.Tm * self.Tn * 4  # FP32 accumulator

        # Distribute tiles with initial skew
        for i in range(self.p):
            for j in range(self.p):
                proc = self._proc_id(i, j)

                # A[i, (i+j) mod p] goes to proc (i, j) - skewed left by i
                a_k = (i + j) % self.p
                self.a_tiles[(i, a_k)] = TileState(
                    tile_type="A",
                    block_i=i % self.num_m,
                    block_j=a_k % self.num_k,
                    current_slice=proc,
                    bytes_size=a_bytes,
                )

                # B[(i+j) mod p, j] goes to proc (i, j) - skewed up by j
                b_i = (i + j) % self.p
                self.b_tiles[(b_i, j)] = TileState(
                    tile_type="B",
                    block_i=b_i % self.num_k,
                    block_j=j % self.num_n,
                    current_slice=proc,
                    bytes_size=b_bytes,
                )

                # C[i, j] at proc (i, j)
                self.c_tiles[(i, j)] = TileState(
                    tile_type="C",
                    block_i=i % self.num_m,
                    block_j=j % self.num_n,
                    current_slice=proc,
                    bytes_size=c_bytes,
                )

        return step

    def rotation_step(self, k: int) -> RotationStep:
        """
        Perform one rotation step.

        1. Compute C += A @ B at each processor
        2. Shift A left
        3. Shift B up
        """
        step = RotationStep(step_number=k + 1, transfers=[], compute_ops=[])

        a_bytes = self.Tm * self.Tk * self.dtype_bytes
        b_bytes = self.Tk * self.Tn * self.dtype_bytes

        # Compute at each processor
        for i in range(self.p):
            for j in range(self.p):
                step.add_compute(i % self.num_m, j % self.num_n, k % self.num_k)

        # Shift A tiles left
        new_a_tiles = {}
        for (i, orig_k), tile in self.a_tiles.items():
            src = tile.current_slice
            dst = self._left_neighbor(src)
            new_tile = tile.copy()
            new_tile.current_slice = dst
            new_a_tiles[(i, orig_k)] = new_tile
            step.add_transfer(tile, src, dst)
            step.transfer_bytes += a_bytes

            cost = self.topology.transfer_cost(src, dst, a_bytes)
            step.transfer_cycles += cost.latency_cycles
            step.transfer_energy_pj += cost.energy_pj

        self.a_tiles = new_a_tiles

        # Shift B tiles up
        new_b_tiles = {}
        for (orig_i, j), tile in self.b_tiles.items():
            src = tile.current_slice
            dst = self._up_neighbor(src)
            new_tile = tile.copy()
            new_tile.current_slice = dst
            new_b_tiles[(orig_i, j)] = new_tile
            step.add_transfer(tile, src, dst)
            step.transfer_bytes += b_bytes

            cost = self.topology.transfer_cost(src, dst, b_bytes)
            step.transfer_cycles += cost.latency_cycles
            step.transfer_energy_pj += cost.energy_pj

        self.b_tiles = new_b_tiles

        # Estimate compute cycles (simplified)
        macs_per_proc = self.Tm * self.Tk * self.Tn
        step.compute_cycles = macs_per_proc  # 1 cycle per MAC

        return step

    def generate_schedule(self) -> RotationSchedule:
        """Generate complete Cannon's algorithm schedule."""
        schedule = RotationSchedule(
            algorithm="cannon",
            M=self.M, K=self.K, N=self.N,
            Tm=self.Tm, Tk=self.Tk, Tn=self.Tn,
            num_procs=self.p * self.p,
        )

        # Initial distribution
        init_step = self.initial_distribution()
        schedule.add_step(init_step)

        # P rotation steps
        for k in range(self.p):
            step = self.rotation_step(k)
            schedule.add_step(step)

        return schedule


class SUMMAAlgorithm:
    """
    SUMMA (Scalable Universal Matrix Multiplication Algorithm).

    For P processors computing C = A @ B:
    1. For each k-panel of A and B:
       a. Owner of A[:,k] broadcasts to its row
       b. Owner of B[k,:] broadcasts to its column
       c. All processors compute local update: C += A_panel @ B_panel

    Properties:
    - Works with rectangular processor grids
    - Communication: broadcasts instead of point-to-point
    - More flexible than Cannon's
    - Slightly higher communication volume
    """

    def __init__(
        self,
        topology: DistributedL3,
        M: int, K: int, N: int,
        Tm: int, Tk: int, Tn: int,
        dtype_bytes: int = 2,
    ):
        self.topology = topology
        self.M = M
        self.K = K
        self.N = N
        self.Tm = Tm
        self.Tk = Tk
        self.Tn = Tn
        self.dtype_bytes = dtype_bytes

        # Use mesh dimensions from topology
        self.proc_rows = topology.mesh_rows
        self.proc_cols = topology.mesh_cols

        self.num_m = ceil(M / Tm)
        self.num_k = ceil(K / Tk)
        self.num_n = ceil(N / Tn)

    def _proc_id(self, row: int, col: int) -> int:
        return row * self.proc_cols + col

    def _proc_pos(self, proc_id: int) -> Tuple[int, int]:
        return (proc_id // self.proc_cols, proc_id % self.proc_cols)

    def _row_procs(self, row: int) -> List[int]:
        """Get all processors in a row."""
        return [self._proc_id(row, c) for c in range(self.proc_cols)]

    def _col_procs(self, col: int) -> List[int]:
        """Get all processors in a column."""
        return [self._proc_id(r, col) for r in range(self.proc_rows)]

    def broadcast_step(self, k: int) -> RotationStep:
        """
        One SUMMA broadcast step.

        1. Owner of A[:,k] broadcasts to row
        2. Owner of B[k,:] broadcasts to column
        3. All compute local C update
        """
        step = RotationStep(step_number=k + 1, transfers=[], compute_ops=[])

        a_panel_bytes = self.Tm * self.Tk * self.dtype_bytes
        b_panel_bytes = self.Tk * self.Tn * self.dtype_bytes

        # Which column owns A panel k?
        a_owner_col = k % self.proc_cols

        # Broadcast A panel to each row
        for row in range(self.proc_rows):
            src = self._proc_id(row, a_owner_col)
            for col in range(self.proc_cols):
                if col != a_owner_col:
                    dst = self._proc_id(row, col)
                    tile = TileState(
                        tile_type="A",
                        block_i=row % self.num_m,
                        block_j=k % self.num_k,
                        current_slice=src,
                        bytes_size=a_panel_bytes,
                    )
                    step.add_transfer(tile, src, dst)
                    cost = self.topology.transfer_cost(src, dst, a_panel_bytes)
                    step.transfer_bytes += a_panel_bytes
                    step.transfer_cycles += cost.latency_cycles
                    step.transfer_energy_pj += cost.energy_pj

        # Which row owns B panel k?
        b_owner_row = k % self.proc_rows

        # Broadcast B panel to each column
        for col in range(self.proc_cols):
            src = self._proc_id(b_owner_row, col)
            for row in range(self.proc_rows):
                if row != b_owner_row:
                    dst = self._proc_id(row, col)
                    tile = TileState(
                        tile_type="B",
                        block_i=k % self.num_k,
                        block_j=col % self.num_n,
                        current_slice=src,
                        bytes_size=b_panel_bytes,
                    )
                    step.add_transfer(tile, src, dst)
                    cost = self.topology.transfer_cost(src, dst, b_panel_bytes)
                    step.transfer_bytes += b_panel_bytes
                    step.transfer_cycles += cost.latency_cycles
                    step.transfer_energy_pj += cost.energy_pj

        # All processors compute
        for i in range(self.proc_rows):
            for j in range(self.proc_cols):
                step.add_compute(
                    i % self.num_m,
                    j % self.num_n,
                    k % self.num_k
                )

        # Compute cycles
        macs_per_proc = self.Tm * self.Tk * self.Tn
        step.compute_cycles = macs_per_proc

        return step

    def generate_schedule(self) -> RotationSchedule:
        """Generate complete SUMMA schedule."""
        schedule = RotationSchedule(
            algorithm="summa",
            M=self.M, K=self.K, N=self.N,
            Tm=self.Tm, Tk=self.Tk, Tn=self.Tn,
            num_procs=self.proc_rows * self.proc_cols,
        )

        # K broadcast steps
        num_k_steps = max(self.proc_rows, self.proc_cols, self.num_k)
        for k in range(num_k_steps):
            step = self.broadcast_step(k)
            schedule.add_step(step)

        return schedule


class Algorithm25D:
    """
    2.5D Matrix Multiplication Algorithm.

    Uses c copies of data to reduce communication by factor of c^(1/2).

    For n x n matrices on P = c * p^2 processors:
    - c layers of p x p processor grids
    - Each layer holds 1/c of the A and B matrices
    - Communication: O(n^2 / (c^(1/2) * P^(1/2)))

    Trade-off: More memory for less communication.
    """

    def __init__(
        self,
        topology: DistributedL3,
        M: int, K: int, N: int,
        Tm: int, Tk: int, Tn: int,
        replication_factor: int = 2,
        dtype_bytes: int = 2,
    ):
        self.topology = topology
        self.M = M
        self.K = K
        self.N = N
        self.Tm = Tm
        self.Tk = Tk
        self.Tn = Tn
        self.c = replication_factor  # Data replication factor
        self.dtype_bytes = dtype_bytes

        # Need P = c * p^2 processors
        self.p = int(sqrt(topology.num_slices / self.c))

        self.num_m = ceil(M / Tm)
        self.num_k = ceil(K / Tk)
        self.num_n = ceil(N / Tn)

    def generate_schedule(self) -> RotationSchedule:
        """Generate 2.5D schedule."""
        schedule = RotationSchedule(
            algorithm="2.5d",
            M=self.M, K=self.K, N=self.N,
            Tm=self.Tm, Tk=self.Tk, Tn=self.Tn,
            num_procs=self.topology.num_slices,
        )

        # Simplified: model as Cannon with reduced communication
        # Communication reduced by factor of sqrt(c)
        reduction_factor = sqrt(self.c)

        a_bytes = self.Tm * self.Tk * self.dtype_bytes
        b_bytes = self.Tk * self.Tn * self.dtype_bytes

        for k in range(self.p):
            step = RotationStep(step_number=k + 1, transfers=[], compute_ops=[])

            # Communication (reduced by replication)
            per_layer_transfers = self.p * self.p * (a_bytes + b_bytes)
            step.transfer_bytes = int(per_layer_transfers / reduction_factor)

            # Compute remains the same
            macs_per_proc = self.Tm * self.Tk * self.Tn
            step.compute_cycles = macs_per_proc

            schedule.add_step(step)

        return schedule


def compare_algorithms(
    M: int, K: int, N: int,
    Tm: int, Tk: int, Tn: int,
    num_procs: int = 16,
) -> Dict:
    """
    Compare rotation algorithms for given problem.

    Returns comparison of Cannon vs SUMMA.
    """
    # Create topology
    p = int(sqrt(num_procs))
    topology = create_mesh_l3(p, p)

    results = {}

    # Cannon's algorithm (requires square grid)
    if p * p == num_procs:
        cannon = CannonAlgorithm(topology, M, K, N, Tm, Tk, Tn)
        cannon_schedule = cannon.generate_schedule()
        results['cannon'] = cannon_schedule.summary()

    # SUMMA
    summa = SUMMAAlgorithm(topology, M, K, N, Tm, Tk, Tn)
    summa_schedule = summa.generate_schedule()
    results['summa'] = summa_schedule.summary()

    # 2.5D with c=2
    if num_procs >= 8:
        algo_25d = Algorithm25D(topology, M, K, N, Tm, Tk, Tn, replication_factor=2)
        schedule_25d = algo_25d.generate_schedule()
        results['2.5d_c2'] = schedule_25d.summary()

    return results


def optimal_algorithm_for_problem(
    M: int, K: int, N: int,
    num_procs: int,
    memory_per_proc_bytes: int,
) -> str:
    """
    Recommend optimal algorithm based on problem characteristics.

    Considerations:
    - Cannon: Best for square grids, minimal communication
    - SUMMA: More flexible, works with rectangular grids
    - 2.5D: When extra memory available and communication-bound
    """
    p = int(sqrt(num_procs))
    is_square_grid = (p * p == num_procs)

    # Problem size per processor
    data_per_proc = (M * N + M * max(M, N) + max(M, N) * N) / num_procs

    # Can we afford 2.5D replication?
    can_25d = data_per_proc * 2 < memory_per_proc_bytes

    # Choose algorithm
    if is_square_grid and M == N and N == max(M, N):
        # Square problem on square grid - Cannon is optimal
        return "cannon"
    elif can_25d and num_procs >= 8:
        # Have memory, communication-bound - use 2.5D
        return "2.5d"
    else:
        # Default to SUMMA for flexibility
        return "summa"
