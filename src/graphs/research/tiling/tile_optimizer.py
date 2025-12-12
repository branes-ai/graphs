"""
Tile Size Optimizer

Determines optimal tile sizes based on memory capacity constraints.
This is the critical link between problem size and memory hierarchy that
triggers additional blocking when working sets exceed capacity.

Key insight: When the working set for a tile doesn't fit in a memory level,
we must introduce additional blocking (smaller tiles) to reduce the working set.

Hierarchical blocking example:
    Problem: C[4096, 4096] = A[4096, 2048] @ B[2048, 4096]

    L2 budget: 4 MB -> tiles (256, 128, 256) = 384 KB working set
    But we need to iterate:
        - 16 x 16 C tiles
        - 16 K tiles per C tile accumulation

    L1 budget: 256 KB -> tiles (64, 64, 64) = 24 KB working set
    Within each L2 tile, we further block:
        - 4 x 4 L1 tiles per L2 tile

    This creates the classic 6-loop tiled matmul:
        for i2 in range(M // Tm2):      # L2 M tiles
          for j2 in range(N // Tn2):    # L2 N tiles
            for k2 in range(K // Tk2):  # L2 K tiles (accumulation)
              for i1 in range(Tm2 // Tm1):  # L1 M tiles within L2
                for j1 in range(Tn2 // Tn1):  # L1 N tiles within L2
                  for k1 in range(Tk2 // Tk1):  # L1 K tiles within L2
                    # Compute on (Tm1, Tk1, Tn1) tile
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from math import ceil, sqrt, log2

from .memory_model import MemoryBudget, MemoryHierarchy
from .block_algebra import DataType, MatmulTiling, LoopOrder


@dataclass
class TileConstraint:
    """
    Constraint on tile dimensions from a memory level.

    Captures what happens when working set exceeds capacity:
    we must reduce tile dimensions until it fits.
    """
    level_name: str
    capacity_bytes: int
    effective_capacity_bytes: int  # After double-buffering

    # Maximum tile dimensions that fit
    max_Tm: int
    max_Tk: int
    max_Tn: int

    # Working set at max dimensions
    working_set_bytes: int

    # Does the unconstrained tile fit?
    requires_blocking: bool

    # Utilization of this level
    capacity_utilization: float


@dataclass
class HierarchicalBlocking:
    """
    Multi-level blocking scheme derived from memory constraints.

    For each memory level, specifies the tile dimensions and
    iteration counts.
    """
    # Problem dimensions
    M: int
    K: int
    N: int

    # L1 tiles (innermost, compute on these)
    l1_Tm: int
    l1_Tk: int
    l1_Tn: int

    # L2 tiles (middle level)
    l2_Tm: int
    l2_Tk: int
    l2_Tn: int

    # L3 tiles (outermost, if applicable)
    l3_Tm: Optional[int] = None
    l3_Tk: Optional[int] = None
    l3_Tn: Optional[int] = None

    # Derived: iteration counts at each level
    @property
    def l1_iterations_per_l2_tile(self) -> int:
        """Number of L1 tile operations per L2 tile."""
        m_iters = ceil(self.l2_Tm / self.l1_Tm)
        k_iters = ceil(self.l2_Tk / self.l1_Tk)
        n_iters = ceil(self.l2_Tn / self.l1_Tn)
        return m_iters * k_iters * n_iters

    @property
    def l2_iterations(self) -> int:
        """Total number of L2 tiles."""
        m_iters = ceil(self.M / self.l2_Tm)
        k_iters = ceil(self.K / self.l2_Tk)
        n_iters = ceil(self.N / self.l2_Tn)
        return m_iters * k_iters * n_iters

    @property
    def l3_iterations(self) -> Optional[int]:
        """Total number of L3 tiles (if L3 blocking used)."""
        if self.l3_Tm is None:
            return None
        m_iters = ceil(self.M / self.l3_Tm)
        k_iters = ceil(self.K / self.l3_Tk)
        n_iters = ceil(self.N / self.l3_Tn)
        return m_iters * k_iters * n_iters

    @property
    def total_l1_operations(self) -> int:
        """Total L1 tile operations (innermost compute)."""
        if self.l3_Tm is not None:
            l2_per_l3 = (
                ceil(self.l3_Tm / self.l2_Tm) *
                ceil(self.l3_Tk / self.l2_Tk) *
                ceil(self.l3_Tn / self.l2_Tn)
            )
            return self.l3_iterations * l2_per_l3 * self.l1_iterations_per_l2_tile
        return self.l2_iterations * self.l1_iterations_per_l2_tile

    def working_set_bytes(
        self,
        level: str,
        input_dtype: DataType = DataType.BF16,
        weight_dtype: DataType = DataType.BF16,
        accum_dtype: DataType = DataType.FP32,
    ) -> int:
        """Working set size at given level."""
        if level == "L1":
            Tm, Tk, Tn = self.l1_Tm, self.l1_Tk, self.l1_Tn
        elif level == "L2":
            Tm, Tk, Tn = self.l2_Tm, self.l2_Tk, self.l2_Tn
        elif level == "L3":
            if self.l3_Tm is None:
                return 0
            Tm, Tk, Tn = self.l3_Tm, self.l3_Tk, self.l3_Tn
        else:
            raise ValueError(f"Unknown level: {level}")

        a_bytes = Tm * Tk * input_dtype.value
        b_bytes = Tk * Tn * weight_dtype.value
        c_bytes = Tm * Tn * accum_dtype.value
        return a_bytes + b_bytes + c_bytes

    def summary(self) -> Dict:
        """Return blocking summary."""
        result = {
            'problem': {'M': self.M, 'K': self.K, 'N': self.N},
            'L1_tile': {'Tm': self.l1_Tm, 'Tk': self.l1_Tk, 'Tn': self.l1_Tn},
            'L2_tile': {'Tm': self.l2_Tm, 'Tk': self.l2_Tk, 'Tn': self.l2_Tn},
            'iterations': {
                'L1_per_L2': self.l1_iterations_per_l2_tile,
                'L2_total': self.l2_iterations,
                'L1_total': self.total_l1_operations,
            },
        }
        if self.l3_Tm is not None:
            result['L3_tile'] = {'Tm': self.l3_Tm, 'Tk': self.l3_Tk, 'Tn': self.l3_Tn}
            result['iterations']['L3_total'] = self.l3_iterations
        return result


class TileSizeOptimizer:
    """
    Optimizes tile sizes based on memory capacity constraints.

    The core algorithm:
    1. Start with problem dimensions as "tile" (no blocking)
    2. Check if working set fits in each memory level
    3. If not, reduce tile dimensions until it fits
    4. Apply this recursively for hierarchical blocking

    Tile size selection considers:
    - Memory capacity at each level
    - Data type sizes for A, B, C tiles
    - Double-buffering requirements
    - Alignment constraints (powers of 2)
    - Loop order for maximizing reuse
    """

    def __init__(
        self,
        budget: Optional[MemoryBudget] = None,
        input_dtype: DataType = DataType.BF16,
        weight_dtype: DataType = DataType.BF16,
        accum_dtype: DataType = DataType.FP32,
    ):
        self.budget = budget or MemoryBudget()
        self.input_dtype = input_dtype
        self.weight_dtype = weight_dtype
        self.accum_dtype = accum_dtype

    def working_set_bytes(self, Tm: int, Tk: int, Tn: int) -> int:
        """
        Compute working set for tile dimensions.

        Working set = A_tile + B_tile + C_tile
        where:
            A_tile: (Tm, Tk) @ input_dtype
            B_tile: (Tk, Tn) @ weight_dtype
            C_tile: (Tm, Tn) @ accum_dtype
        """
        a_bytes = Tm * Tk * self.input_dtype.value
        b_bytes = Tk * Tn * self.weight_dtype.value
        c_bytes = Tm * Tn * self.accum_dtype.value
        return a_bytes + b_bytes + c_bytes

    def check_constraint(
        self,
        M: int, K: int, N: int,
        level: str,
    ) -> TileConstraint:
        """
        Check memory constraint for given problem at specified level.

        Returns constraint analysis showing if blocking is required.
        """
        if level == "L1":
            capacity = self.budget.l1_bytes
            effective = self.budget.effective_l1_bytes
        elif level == "L2":
            capacity = self.budget.l2_bytes
            effective = self.budget.effective_l2_bytes
        elif level == "L3":
            capacity = self.budget.l3_bytes
            effective = self.budget.effective_l3_bytes
        else:
            raise ValueError(f"Unknown level: {level}")

        # Working set if we use full problem dimensions (no blocking)
        full_ws = self.working_set_bytes(M, K, N)
        requires_blocking = full_ws > effective

        if requires_blocking:
            # Find max tile that fits
            max_Tm, max_Tk, max_Tn = self.find_max_tile(effective)
            actual_ws = self.working_set_bytes(max_Tm, max_Tk, max_Tn)
        else:
            max_Tm, max_Tk, max_Tn = M, K, N
            actual_ws = full_ws

        utilization = actual_ws / effective if effective > 0 else 0

        return TileConstraint(
            level_name=level,
            capacity_bytes=capacity,
            effective_capacity_bytes=effective,
            max_Tm=max_Tm,
            max_Tk=max_Tk,
            max_Tn=max_Tn,
            working_set_bytes=actual_ws,
            requires_blocking=requires_blocking,
            capacity_utilization=utilization,
        )

    def find_max_tile(
        self,
        capacity_bytes: int,
        aspect_ratio: float = 1.0,
        alignment: int = 16,
    ) -> Tuple[int, int, int]:
        """
        Find maximum tile dimensions that fit in given capacity.

        Solves: A_bytes + B_bytes + C_bytes <= capacity
        where A=(Tm,Tk), B=(Tk,Tn), C=(Tm,Tn)

        For square tiles (Tm=Tk=Tn=D):
            D^2 * (input_dtype + weight_dtype + accum_dtype) <= capacity

        Args:
            capacity_bytes: Available memory
            aspect_ratio: Tm/Tn ratio (1.0 = square)
            alignment: Round down to multiple of this

        Returns:
            (Tm, Tk, Tn) maximum dimensions
        """
        # Bytes per element for each tile type
        input_bytes = self.input_dtype.value
        weight_bytes = self.weight_dtype.value
        accum_bytes = self.accum_dtype.value

        # For square tiles: 3 * D^2 * avg_bytes <= capacity
        # Actually: D^2 * (input + weight) + D^2 * accum = D^2 * total
        total_bytes_per_element = input_bytes + weight_bytes + accum_bytes

        max_d_squared = capacity_bytes / total_bytes_per_element
        max_d = int(sqrt(max_d_squared))

        # Round down to alignment
        max_d = (max_d // alignment) * alignment
        max_d = max(alignment, max_d)  # At least one aligned tile

        # Apply aspect ratio
        Tm = int(max_d * sqrt(aspect_ratio))
        Tn = int(max_d / sqrt(aspect_ratio))
        Tk = max_d

        # Ensure alignment
        Tm = (Tm // alignment) * alignment
        Tn = (Tn // alignment) * alignment
        Tk = (Tk // alignment) * alignment

        Tm = max(alignment, Tm)
        Tn = max(alignment, Tn)
        Tk = max(alignment, Tk)

        return Tm, Tk, Tn

    def optimize_for_level(
        self,
        M: int, K: int, N: int,
        level: str,
        loop_order: LoopOrder = LoopOrder.MNK,
    ) -> Tuple[int, int, int]:
        """
        Optimize tile size for specific memory level.

        Considers loop order to maximize reuse:
        - MNK (output-stationary): favor large Tm, Tn (C stays)
        - NKM (weight-stationary): favor large Tk, Tn (B stays)
        - MKN (input-stationary): favor large Tm, Tk (A stays)

        Args:
            M, K, N: Problem dimensions
            level: "L1", "L2", or "L3"
            loop_order: Loop ordering for reuse optimization

        Returns:
            (Tm, Tk, Tn) optimized tile dimensions
        """
        constraint = self.check_constraint(M, K, N, level)

        if not constraint.requires_blocking:
            # No blocking needed, use full dimensions
            return M, K, N

        # Get base max tile
        Tm, Tk, Tn = constraint.max_Tm, constraint.max_Tk, constraint.max_Tn

        # Adjust based on loop order to maximize reuse
        capacity = constraint.effective_capacity_bytes

        if loop_order == LoopOrder.MNK:
            # Output-stationary: C tile stays, maximize Tm * Tn
            # C accumulates over K, so Tk can be smaller
            # Allocate more to Tm, Tn
            Tm, Tk, Tn = self._optimize_output_stationary(capacity, M, K, N)

        elif loop_order == LoopOrder.NKM:
            # Weight-stationary: B tile stays, maximize Tk * Tn
            # B is reused across M, so Tm can iterate
            Tm, Tk, Tn = self._optimize_weight_stationary(capacity, M, K, N)

        elif loop_order == LoopOrder.MKN:
            # Input-stationary: A tile stays, maximize Tm * Tk
            # A is reused across N, so Tn can iterate
            Tm, Tk, Tn = self._optimize_input_stationary(capacity, M, K, N)

        # Ensure tiles don't exceed problem dimensions
        Tm = min(Tm, M)
        Tk = min(Tk, K)
        Tn = min(Tn, N)

        return Tm, Tk, Tn

    def _optimize_output_stationary(
        self,
        capacity: int,
        M: int, K: int, N: int,
    ) -> Tuple[int, int, int]:
        """Optimize for output-stationary: maximize C tile, smaller Tk."""
        # C tile dominates (FP32), minimize its reloads
        # Strategy: Large Tm, Tn; smaller Tk for K accumulation

        c_bytes = self.accum_dtype.value
        a_bytes = self.input_dtype.value
        b_bytes = self.weight_dtype.value

        # Allocate ~60% to C, ~40% to A+B
        c_budget = int(capacity * 0.6)
        ab_budget = capacity - c_budget

        # C tile: Tm * Tn * c_bytes <= c_budget
        c_dim = int(sqrt(c_budget / c_bytes))
        c_dim = (c_dim // 16) * 16
        c_dim = max(16, c_dim)

        Tm = min(c_dim, M)
        Tn = min(c_dim, N)

        # A+B tiles must fit in remaining budget
        # A: Tm * Tk, B: Tk * Tn
        # Tk * (Tm * a_bytes + Tn * b_bytes) <= ab_budget
        ab_factor = Tm * a_bytes + Tn * b_bytes
        Tk = int(ab_budget / ab_factor) if ab_factor > 0 else 16
        Tk = (Tk // 16) * 16
        Tk = max(16, min(Tk, K))

        return Tm, Tk, Tn

    def _optimize_weight_stationary(
        self,
        capacity: int,
        M: int, K: int, N: int,
    ) -> Tuple[int, int, int]:
        """Optimize for weight-stationary: maximize B tile, smaller Tm."""
        # B tile stays in cache, reused across M dimension
        # Strategy: Large Tk, Tn; smaller Tm

        b_bytes = self.weight_dtype.value
        a_bytes = self.input_dtype.value
        c_bytes = self.accum_dtype.value

        # Allocate ~50% to B, ~50% to A+C
        b_budget = int(capacity * 0.5)
        ac_budget = capacity - b_budget

        # B tile: Tk * Tn * b_bytes <= b_budget
        b_dim = int(sqrt(b_budget / b_bytes))
        b_dim = (b_dim // 16) * 16
        b_dim = max(16, b_dim)

        Tk = min(b_dim, K)
        Tn = min(b_dim, N)

        # A+C tiles must fit
        # A: Tm * Tk, C: Tm * Tn
        # Tm * (Tk * a_bytes + Tn * c_bytes) <= ac_budget
        ac_factor = Tk * a_bytes + Tn * c_bytes
        Tm = int(ac_budget / ac_factor) if ac_factor > 0 else 16
        Tm = (Tm // 16) * 16
        Tm = max(16, min(Tm, M))

        return Tm, Tk, Tn

    def _optimize_input_stationary(
        self,
        capacity: int,
        M: int, K: int, N: int,
    ) -> Tuple[int, int, int]:
        """Optimize for input-stationary: maximize A tile, smaller Tn."""
        # A tile stays in cache, reused across N dimension
        # Strategy: Large Tm, Tk; smaller Tn

        a_bytes = self.input_dtype.value
        b_bytes = self.weight_dtype.value
        c_bytes = self.accum_dtype.value

        # Allocate ~50% to A, ~50% to B+C
        a_budget = int(capacity * 0.5)
        bc_budget = capacity - a_budget

        # A tile: Tm * Tk * a_bytes <= a_budget
        a_dim = int(sqrt(a_budget / a_bytes))
        a_dim = (a_dim // 16) * 16
        a_dim = max(16, a_dim)

        Tm = min(a_dim, M)
        Tk = min(a_dim, K)

        # B+C tiles must fit
        # B: Tk * Tn, C: Tm * Tn
        # Tn * (Tk * b_bytes + Tm * c_bytes) <= bc_budget
        bc_factor = Tk * b_bytes + Tm * c_bytes
        Tn = int(bc_budget / bc_factor) if bc_factor > 0 else 16
        Tn = (Tn // 16) * 16
        Tn = max(16, min(Tn, N))

        return Tm, Tk, Tn

    def create_hierarchical_blocking(
        self,
        M: int, K: int, N: int,
        loop_order: LoopOrder = LoopOrder.MNK,
    ) -> HierarchicalBlocking:
        """
        Create complete hierarchical blocking scheme.

        Determines tile sizes at each memory level based on capacity.

        Args:
            M, K, N: Problem dimensions
            loop_order: Loop ordering (affects tile shape optimization)

        Returns:
            HierarchicalBlocking with L1, L2, (optionally L3) tiles
        """
        # Start from outermost (L3 if available) and work inward

        # L3 tiles (if L3 is configured)
        if self.budget.l3_bytes > 0:
            l3_Tm, l3_Tk, l3_Tn = self.optimize_for_level(
                M, K, N, "L3", loop_order
            )
        else:
            l3_Tm = l3_Tk = l3_Tn = None

        # L2 tiles (constrained by L2 capacity)
        if l3_Tm is not None:
            # L2 tiles must fit within L3 tiles
            l2_Tm, l2_Tk, l2_Tn = self.optimize_for_level(
                l3_Tm, l3_Tk, l3_Tn, "L2", loop_order
            )
        else:
            l2_Tm, l2_Tk, l2_Tn = self.optimize_for_level(
                M, K, N, "L2", loop_order
            )

        # L1 tiles (innermost, constrained by L1 capacity)
        l1_Tm, l1_Tk, l1_Tn = self.optimize_for_level(
            l2_Tm, l2_Tk, l2_Tn, "L1", loop_order
        )

        return HierarchicalBlocking(
            M=M, K=K, N=N,
            l1_Tm=l1_Tm, l1_Tk=l1_Tk, l1_Tn=l1_Tn,
            l2_Tm=l2_Tm, l2_Tk=l2_Tk, l2_Tn=l2_Tn,
            l3_Tm=l3_Tm, l3_Tk=l3_Tk, l3_Tn=l3_Tn,
        )

    def analyze_blocking_decision(
        self,
        M: int, K: int, N: int,
    ) -> Dict[str, TileConstraint]:
        """
        Analyze why blocking is needed at each level.

        Returns constraints showing:
        - Full problem working set vs capacity
        - Whether blocking is required
        - Resulting tile dimensions
        """
        return {
            "L1": self.check_constraint(M, K, N, "L1"),
            "L2": self.check_constraint(M, K, N, "L2"),
            "L3": self.check_constraint(M, K, N, "L3") if self.budget.l3_bytes > 0 else None,
        }


def print_blocking_analysis(
    M: int, K: int, N: int,
    budget: Optional[MemoryBudget] = None,
    loop_order: LoopOrder = LoopOrder.MNK,
):
    """Print human-readable blocking analysis."""
    optimizer = TileSizeOptimizer(budget=budget)

    print("\n" + "=" * 70)
    print("MEMORY-CONSTRAINED TILE SIZE OPTIMIZATION")
    print("=" * 70)

    print(f"\nProblem: C[{M}, {N}] = A[{M}, {K}] @ B[{K}, {N}]")
    print(f"Loop order: {loop_order.value}")

    # Full problem working set
    full_ws = optimizer.working_set_bytes(M, K, N)
    print(f"\nFull problem working set: {full_ws:,} bytes ({full_ws/1024/1024:.2f} MB)")
    print(f"  A[{M}, {K}] @ BF16 = {M*K*2:,} bytes")
    print(f"  B[{K}, {N}] @ BF16 = {K*N*2:,} bytes")
    print(f"  C[{M}, {N}] @ FP32 = {M*N*4:,} bytes")

    print("\n" + "-" * 70)
    print("MEMORY CONSTRAINTS")
    print("-" * 70)

    constraints = optimizer.analyze_blocking_decision(M, K, N)
    for level, constraint in constraints.items():
        if constraint is None:
            continue
        print(f"\n{level}:")
        print(f"  Capacity: {constraint.capacity_bytes:,} bytes "
              f"(effective: {constraint.effective_capacity_bytes:,} bytes)")
        print(f"  Requires blocking: {constraint.requires_blocking}")
        if constraint.requires_blocking:
            print(f"  Max tile: Tm={constraint.max_Tm}, Tk={constraint.max_Tk}, Tn={constraint.max_Tn}")
            print(f"  Working set: {constraint.working_set_bytes:,} bytes "
                  f"({constraint.capacity_utilization*100:.1f}% utilization)")

    print("\n" + "-" * 70)
    print("HIERARCHICAL BLOCKING SCHEME")
    print("-" * 70)

    blocking = optimizer.create_hierarchical_blocking(M, K, N, loop_order)

    print(f"\nL1 tiles (innermost compute):")
    print(f"  A_tile: ({blocking.l1_Tm}, {blocking.l1_Tk}) = {blocking.l1_Tm * blocking.l1_Tk * 2:,} bytes")
    print(f"  B_tile: ({blocking.l1_Tk}, {blocking.l1_Tn}) = {blocking.l1_Tk * blocking.l1_Tn * 2:,} bytes")
    print(f"  C_tile: ({blocking.l1_Tm}, {blocking.l1_Tn}) = {blocking.l1_Tm * blocking.l1_Tn * 4:,} bytes")
    print(f"  Total working set: {blocking.working_set_bytes('L1'):,} bytes")

    print(f"\nL2 tiles:")
    print(f"  A_tile: ({blocking.l2_Tm}, {blocking.l2_Tk}) = {blocking.l2_Tm * blocking.l2_Tk * 2:,} bytes")
    print(f"  B_tile: ({blocking.l2_Tk}, {blocking.l2_Tn}) = {blocking.l2_Tk * blocking.l2_Tn * 2:,} bytes")
    print(f"  C_tile: ({blocking.l2_Tm}, {blocking.l2_Tn}) = {blocking.l2_Tm * blocking.l2_Tn * 4:,} bytes")
    print(f"  Total working set: {blocking.working_set_bytes('L2'):,} bytes")

    if blocking.l3_Tm is not None:
        print(f"\nL3 tiles:")
        print(f"  A_tile: ({blocking.l3_Tm}, {blocking.l3_Tk})")
        print(f"  B_tile: ({blocking.l3_Tk}, {blocking.l3_Tn})")
        print(f"  C_tile: ({blocking.l3_Tm}, {blocking.l3_Tn})")
        print(f"  Total working set: {blocking.working_set_bytes('L3'):,} bytes")

    print("\n" + "-" * 70)
    print("ITERATION COUNTS")
    print("-" * 70)
    print(f"  L1 operations per L2 tile: {blocking.l1_iterations_per_l2_tile:,}")
    print(f"  Total L2 tiles: {blocking.l2_iterations:,}")
    print(f"  Total L1 operations: {blocking.total_l1_operations:,}")

    if blocking.l3_Tm is not None:
        print(f"  Total L3 tiles: {blocking.l3_iterations:,}")

    print("=" * 70)

    return blocking


def analyze_with_memory_constraints(
    M: int, K: int, N: int,
    budget: Optional[MemoryBudget] = None,
    loop_order: LoopOrder = LoopOrder.MNK,
    verbose: bool = True,
) -> Dict:
    """
    Complete analysis: memory-constrained tile selection + reuse statistics.

    This is the main entry point that:
    1. Determines tile sizes based on memory constraints
    2. Analyzes reuse for those tile sizes
    3. Shows which level triggers blocking
    """
    from .reuse_analyzer import TileReuseAnalyzer, print_tile_analysis

    optimizer = TileSizeOptimizer(budget=budget)
    blocking = optimizer.create_hierarchical_blocking(M, K, N, loop_order)
    constraints = optimizer.analyze_blocking_decision(M, K, N)

    # Use L1 tiles for reuse analysis (innermost compute)
    analyzer = TileReuseAnalyzer(budget=budget)
    reuse = analyzer.analyze(
        M=M, K=K, N=N,
        Tm=blocking.l1_Tm, Tk=blocking.l1_Tk, Tn=blocking.l1_Tn,
        loop_order=loop_order,
    )

    if verbose:
        print("\n" + "=" * 70)
        print("MEMORY-CONSTRAINED TILE REUSE ANALYSIS")
        print("=" * 70)

        print(f"\nProblem: C[{M}, {N}] = A[{M}, {K}] @ B[{K}, {N}]")
        print(f"Loop order: {loop_order.value}")

        # Show why blocking is needed
        print("\n" + "-" * 70)
        print("MEMORY CONSTRAINTS (why blocking is needed)")
        print("-" * 70)
        full_ws = optimizer.working_set_bytes(M, K, N)
        print(f"Full problem working set: {full_ws:,} bytes ({full_ws/1024/1024:.2f} MB)")

        for level, constraint in constraints.items():
            if constraint is None:
                continue
            status = "BLOCKING REQUIRED" if constraint.requires_blocking else "fits"
            print(f"  {level}: {constraint.effective_capacity_bytes:,} bytes -> {status}")

        # Show selected tile sizes
        print("\n" + "-" * 70)
        print("SELECTED TILE SIZES (from memory constraints)")
        print("-" * 70)
        print(f"L1 tiles (compute): Tm={blocking.l1_Tm}, Tk={blocking.l1_Tk}, Tn={blocking.l1_Tn}")
        print(f"  A_tile: ({blocking.l1_Tm}, {blocking.l1_Tk}) = {blocking.l1_Tm * blocking.l1_Tk * 2:,} bytes")
        print(f"  B_tile: ({blocking.l1_Tk}, {blocking.l1_Tn}) = {blocking.l1_Tk * blocking.l1_Tn * 2:,} bytes")
        print(f"  C_tile: ({blocking.l1_Tm}, {blocking.l1_Tn}) = {blocking.l1_Tm * blocking.l1_Tn * 4:,} bytes")
        print(f"  Working set: {blocking.working_set_bytes('L1'):,} bytes")

        print(f"\nL2 tiles: Tm={blocking.l2_Tm}, Tk={blocking.l2_Tk}, Tn={blocking.l2_Tn}")
        print(f"  Working set: {blocking.working_set_bytes('L2'):,} bytes")

        if blocking.l3_Tm:
            print(f"\nL3 tiles: Tm={blocking.l3_Tm}, Tk={blocking.l3_Tk}, Tn={blocking.l3_Tn}")
            print(f"  Working set: {blocking.working_set_bytes('L3'):,} bytes")

        # Show reuse statistics
        print("\n" + "-" * 70)
        print("TILE REUSE STATISTICS (for L1 tiles)")
        print("-" * 70)
        print(f"{'Tile':<8} {'Shape':<15} {'Count':<10} {'Reuse':<10} {'Bytes':<15}")
        print("-" * 70)

        a = reuse.a_metrics
        b = reuse.b_metrics
        c = reuse.c_metrics

        print(f"{'A':<8} {str(a.shape):<15} {a.unique_tiles:<10} {a.reuse_factor:<10.1f} {a.actual_bytes:<15,}")
        print(f"{'B':<8} {str(b.shape):<15} {b.unique_tiles:<10} {b.reuse_factor:<10.1f} {b.actual_bytes:<15,}")
        print(f"{'C':<8} {str(c.shape):<15} {c.unique_tiles:<10} {c.reuse_factor:<10.1f} {c.actual_bytes:<15,}")

        # Show iteration counts
        print("\n" + "-" * 70)
        print("ITERATION STRUCTURE")
        print("-" * 70)
        print(f"  L1 operations per L2 tile: {blocking.l1_iterations_per_l2_tile:,}")
        print(f"  Total L2 tiles: {blocking.l2_iterations:,}")
        print(f"  Total L1 operations: {blocking.total_l1_operations:,}")

        # Summary
        print("\n" + "-" * 70)
        print("SUMMARY")
        print("-" * 70)
        print(f"  Total FLOPs:            {reuse.total_flops:,}")
        print(f"  Arithmetic intensity:   {reuse.arithmetic_intensity:.2f} FLOPs/byte")
        print(f"  A reuse factor:         {a.reuse_factor:.1f}x")
        print(f"  B reuse factor:         {b.reuse_factor:.1f}x")
        print(f"  C accumulations:        {c.reuse_factor:.1f}x")

        print("=" * 70)

    return {
        'blocking': blocking,
        'constraints': constraints,
        'reuse': reuse,
    }
