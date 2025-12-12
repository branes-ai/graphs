"""
Tile Reuse Analysis

Analyzes data reuse for tiled matrix operations in terms of actual 2D tiles
that move through the memory hierarchy.

Key insight: Reuse analysis must track three distinct 2D tile types:
- A_tile (Tm, Tk): Input activation tiles
- B_tile (Tk, Tn): Weight tiles
- C_tile (Tm, Tn): Output/accumulator tiles

Each tile type has different:
- Memory footprint (shape * dtype)
- Reuse pattern (which loop provides reuse)
- Lifetime (how long it must stay in cache)
- Movement through memory hierarchy

The analysis answers:
1. How many times is each tile reused?
2. How long must each tile stay in cache?
3. What is the peak working set (tiles simultaneously live)?
4. What is the actual memory traffic vs minimum possible?
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple
from math import ceil

from .block_algebra import (
    DataType, TileShape, ATile, BTile, CTile, TileSet,
    MatmulTiling, TileSchedule, TileOperation, LoopOrder,
    analyze_memory_traffic
)
from .memory_model import MemoryBudget


@dataclass
class TileReuseMetrics:
    """
    Reuse metrics for a single tile type (A, B, or C).

    Tracks the 2D tile's journey through the memory hierarchy.
    """
    tile_type: str  # "A", "B", or "C"
    shape: Tuple[int, int]  # 2D shape (rows, cols)
    dtype_bytes: int
    bytes_per_tile: int

    # Tile counts
    unique_tiles: int       # Number of distinct tiles
    total_uses: int         # Total times tiles are used (with reuse)

    # Reuse factors
    reuse_factor: float     # total_uses / unique_tiles

    # Memory traffic
    minimum_bytes: int      # If each tile loaded exactly once
    actual_bytes: int       # Actual bytes transferred
    reuse_efficiency: float # minimum / actual (1.0 = perfect)

    # Lifetime (in schedule steps)
    min_lifetime_steps: int
    max_lifetime_steps: int
    avg_lifetime_steps: float

    def summary(self) -> Dict:
        return {
            'tile_type': self.tile_type,
            'shape': self.shape,
            'bytes_per_tile': self.bytes_per_tile,
            'unique_tiles': self.unique_tiles,
            'reuse_factor': self.reuse_factor,
            'minimum_bytes': self.minimum_bytes,
            'actual_bytes': self.actual_bytes,
            'reuse_efficiency': self.reuse_efficiency,
            'avg_lifetime_steps': self.avg_lifetime_steps,
        }


@dataclass
class WorkingSetAnalysis:
    """
    Analysis of working set (tiles simultaneously live in cache).

    This determines the cache capacity required.
    """
    # Peak counts of each tile type simultaneously live
    peak_a_tiles: int
    peak_b_tiles: int
    peak_c_tiles: int

    # Bytes at peak
    peak_a_bytes: int
    peak_b_bytes: int
    peak_c_bytes: int

    @property
    def peak_total_bytes(self) -> int:
        return self.peak_a_bytes + self.peak_b_bytes + self.peak_c_bytes

    def fits_in_cache(self, cache_bytes: int) -> bool:
        return self.peak_total_bytes <= cache_bytes

    def summary(self) -> Dict:
        return {
            'peak_tiles': {
                'A': self.peak_a_tiles,
                'B': self.peak_b_tiles,
                'C': self.peak_c_tiles,
            },
            'peak_bytes': {
                'A': self.peak_a_bytes,
                'B': self.peak_b_bytes,
                'C': self.peak_c_bytes,
                'total': self.peak_total_bytes,
            },
        }


@dataclass
class TileReuseAnalysis:
    """
    Complete reuse analysis for a tiled matmul.

    Provides per-tile-type metrics and overall summary.
    """
    # Tiling configuration
    tiling: MatmulTiling
    loop_order: LoopOrder

    # Per-tile-type metrics
    a_metrics: TileReuseMetrics
    b_metrics: TileReuseMetrics
    c_metrics: TileReuseMetrics

    # Working set analysis
    working_set: WorkingSetAnalysis

    # Overall metrics
    total_flops: int
    total_minimum_bytes: int
    total_actual_bytes: int
    arithmetic_intensity: float

    def summary(self) -> Dict:
        return {
            'problem': {
                'M': self.tiling.M,
                'K': self.tiling.K,
                'N': self.tiling.N,
            },
            'tiles': {
                'A': {
                    'shape': (self.tiling.Tm, self.tiling.Tk),
                    'bytes': self.tiling.a_tile_shape.bytes,
                },
                'B': {
                    'shape': (self.tiling.Tk, self.tiling.Tn),
                    'bytes': self.tiling.b_tile_shape.bytes,
                },
                'C': {
                    'shape': (self.tiling.Tm, self.tiling.Tn),
                    'bytes': self.tiling.c_tile_shape.bytes,
                },
            },
            'loop_order': self.loop_order.value,
            'A_reuse': self.a_metrics.summary(),
            'B_reuse': self.b_metrics.summary(),
            'C_reuse': self.c_metrics.summary(),
            'working_set': self.working_set.summary(),
            'total_flops': self.total_flops,
            'arithmetic_intensity': self.arithmetic_intensity,
            'overall_reuse_efficiency': (
                self.total_minimum_bytes / self.total_actual_bytes
                if self.total_actual_bytes > 0 else 1.0
            ),
        }


class TileReuseAnalyzer:
    """
    Analyzer for tile reuse in tiled matrix operations.

    Tracks each 2D tile type (A, B, C) independently through the
    memory hierarchy.
    """

    def __init__(self, budget: Optional[MemoryBudget] = None):
        self.budget = budget or MemoryBudget()

    def analyze(
        self,
        M: int, K: int, N: int,
        Tm: int, Tk: int, Tn: int,
        loop_order: LoopOrder = LoopOrder.MNK,
        input_dtype: DataType = DataType.BF16,
        weight_dtype: DataType = DataType.BF16,
        accum_dtype: DataType = DataType.FP32,
    ) -> TileReuseAnalysis:
        """
        Analyze tile reuse for given configuration.

        Args:
            M, K, N: Problem dimensions
            Tm, Tk, Tn: Tile dimensions (define 2D tile shapes)
            loop_order: Loop ordering (determines reuse pattern)
            input_dtype: Data type for A tiles
            weight_dtype: Data type for B tiles
            accum_dtype: Data type for C tiles (accumulator)

        Returns:
            TileReuseAnalysis with per-tile-type metrics
        """
        # Create tiling specification
        tiling = MatmulTiling(
            M=M, K=K, N=N,
            Tm=Tm, Tk=Tk, Tn=Tn,
            input_dtype=input_dtype,
            weight_dtype=weight_dtype,
            accum_dtype=accum_dtype,
        )

        # Generate schedule and analyze traffic
        schedule = TileSchedule(tiling=tiling, loop_order=loop_order)
        traffic = analyze_memory_traffic(tiling, loop_order)

        # Analyze lifetimes
        lifetimes = schedule.analyze_tile_lifetimes()

        # Build per-tile-type metrics
        a_metrics = TileReuseMetrics(
            tile_type="A",
            shape=(Tm, Tk),
            dtype_bytes=input_dtype.value,
            bytes_per_tile=tiling.a_tile_shape.bytes,
            unique_tiles=tiling.total_a_tiles,
            total_uses=tiling.total_a_tiles * tiling.a_reuse_factor,
            reuse_factor=tiling.a_reuse_factor,
            minimum_bytes=traffic['A_tile']['minimum_bytes'],
            actual_bytes=traffic['A_tile']['actual_bytes'],
            reuse_efficiency=traffic['A_tile']['reuse_achieved'],
            min_lifetime_steps=lifetimes['A_tiles']['min_lifetime'],
            max_lifetime_steps=lifetimes['A_tiles']['max_lifetime'],
            avg_lifetime_steps=lifetimes['A_tiles']['avg_lifetime'],
        )

        b_metrics = TileReuseMetrics(
            tile_type="B",
            shape=(Tk, Tn),
            dtype_bytes=weight_dtype.value,
            bytes_per_tile=tiling.b_tile_shape.bytes,
            unique_tiles=tiling.total_b_tiles,
            total_uses=tiling.total_b_tiles * tiling.b_reuse_factor,
            reuse_factor=tiling.b_reuse_factor,
            minimum_bytes=traffic['B_tile']['minimum_bytes'],
            actual_bytes=traffic['B_tile']['actual_bytes'],
            reuse_efficiency=traffic['B_tile']['reuse_achieved'],
            min_lifetime_steps=lifetimes['B_tiles']['min_lifetime'],
            max_lifetime_steps=lifetimes['B_tiles']['max_lifetime'],
            avg_lifetime_steps=lifetimes['B_tiles']['avg_lifetime'],
        )

        c_metrics = TileReuseMetrics(
            tile_type="C",
            shape=(Tm, Tn),
            dtype_bytes=accum_dtype.value,
            bytes_per_tile=tiling.c_tile_shape.bytes,
            unique_tiles=tiling.total_c_tiles,
            total_uses=tiling.total_c_tiles * tiling.c_accumulation_count,
            reuse_factor=tiling.c_accumulation_count,
            minimum_bytes=traffic['C_tile']['minimum_bytes'],
            actual_bytes=(traffic['C_tile']['actual_read_bytes'] +
                          traffic['C_tile']['actual_write_bytes']),
            reuse_efficiency=(
                traffic['C_tile']['minimum_bytes'] /
                (traffic['C_tile']['actual_read_bytes'] +
                 traffic['C_tile']['actual_write_bytes'])
                if traffic['C_tile']['actual_write_bytes'] > 0 else 1.0
            ),
            min_lifetime_steps=lifetimes['C_tiles']['min_lifetime'],
            max_lifetime_steps=lifetimes['C_tiles']['max_lifetime'],
            avg_lifetime_steps=lifetimes['C_tiles']['avg_lifetime'],
        )

        # Working set analysis
        peak = schedule.peak_live_tiles()
        working_set = WorkingSetAnalysis(
            peak_a_tiles=peak['A'],
            peak_b_tiles=peak['B'],
            peak_c_tiles=peak['C'],
            peak_a_bytes=peak['A'] * tiling.a_tile_shape.bytes,
            peak_b_bytes=peak['B'] * tiling.b_tile_shape.bytes,
            peak_c_bytes=peak['C'] * tiling.c_tile_shape.bytes,
        )

        return TileReuseAnalysis(
            tiling=tiling,
            loop_order=loop_order,
            a_metrics=a_metrics,
            b_metrics=b_metrics,
            c_metrics=c_metrics,
            working_set=working_set,
            total_flops=tiling.total_flops,
            total_minimum_bytes=traffic['total']['minimum_bytes'],
            total_actual_bytes=traffic['total']['actual_total_bytes'],
            arithmetic_intensity=traffic['arithmetic_intensity'],
        )

    def compare_loop_orders(
        self,
        M: int, K: int, N: int,
        Tm: int, Tk: int, Tn: int,
    ) -> Dict[str, TileReuseAnalysis]:
        """
        Compare reuse across all loop orderings.

        Returns dict mapping loop order name to analysis.
        """
        results = {}

        for loop_order in LoopOrder:
            analysis = self.analyze(M, K, N, Tm, Tk, Tn, loop_order)
            results[loop_order.value] = analysis

        return results

    def find_best_loop_order(
        self,
        M: int, K: int, N: int,
        Tm: int, Tk: int, Tn: int,
        metric: str = "arithmetic_intensity",
    ) -> Tuple[LoopOrder, TileReuseAnalysis]:
        """
        Find best loop order for given problem.

        Args:
            M, K, N: Problem dimensions
            Tm, Tk, Tn: Tile dimensions
            metric: "arithmetic_intensity", "a_reuse", "b_reuse", "working_set"

        Returns:
            (best_loop_order, analysis)
        """
        comparisons = self.compare_loop_orders(M, K, N, Tm, Tk, Tn)

        if metric == "arithmetic_intensity":
            best_name = max(comparisons, key=lambda x: comparisons[x].arithmetic_intensity)
        elif metric == "a_reuse":
            best_name = max(comparisons, key=lambda x: comparisons[x].a_metrics.reuse_efficiency)
        elif metric == "b_reuse":
            best_name = max(comparisons, key=lambda x: comparisons[x].b_metrics.reuse_efficiency)
        elif metric == "working_set":
            best_name = min(comparisons, key=lambda x: comparisons[x].working_set.peak_total_bytes)
        else:
            raise ValueError(f"Unknown metric: {metric}")

        return LoopOrder(best_name), comparisons[best_name]


def print_tile_analysis(analysis: TileReuseAnalysis):
    """Print human-readable analysis."""
    print("\n" + "=" * 70)
    print("TILE REUSE ANALYSIS")
    print("=" * 70)

    t = analysis.tiling
    print(f"\nProblem: C({t.M}, {t.N}) = A({t.M}, {t.K}) @ B({t.K}, {t.N})")
    print(f"Loop order: {analysis.loop_order.value}")

    print("\n" + "-" * 70)
    print("TILE SHAPES (2D submatrices)")
    print("-" * 70)
    print(f"  A_tile: ({t.Tm}, {t.Tk}) @ {t.input_dtype.name:5} = {t.a_tile_shape.bytes:,} bytes")
    print(f"  B_tile: ({t.Tk}, {t.Tn}) @ {t.weight_dtype.name:5} = {t.b_tile_shape.bytes:,} bytes")
    print(f"  C_tile: ({t.Tm}, {t.Tn}) @ {t.accum_dtype.name:5} = {t.c_tile_shape.bytes:,} bytes")

    print("\n" + "-" * 70)
    print("TILE COUNTS AND REUSE")
    print("-" * 70)
    print(f"{'Tile':<8} {'Count':<10} {'Reuse':<10} {'Min Bytes':<15} {'Actual Bytes':<15} {'Efficiency':<10}")
    print("-" * 70)

    for m in [analysis.a_metrics, analysis.b_metrics, analysis.c_metrics]:
        print(f"{m.tile_type:<8} {m.unique_tiles:<10} {m.reuse_factor:<10.1f} "
              f"{m.minimum_bytes:<15,} {m.actual_bytes:<15,} {m.reuse_efficiency*100:<10.1f}%")

    print("\n" + "-" * 70)
    print("WORKING SET (peak tiles simultaneously live)")
    print("-" * 70)
    ws = analysis.working_set
    print(f"  Peak A tiles: {ws.peak_a_tiles} = {ws.peak_a_bytes:,} bytes")
    print(f"  Peak B tiles: {ws.peak_b_tiles} = {ws.peak_b_bytes:,} bytes")
    print(f"  Peak C tiles: {ws.peak_c_tiles} = {ws.peak_c_bytes:,} bytes")
    print(f"  Total peak:   {ws.peak_total_bytes:,} bytes ({ws.peak_total_bytes/1024/1024:.2f} MB)")

    print("\n" + "-" * 70)
    print("SUMMARY")
    print("-" * 70)
    print(f"  Total FLOPs:            {analysis.total_flops:,}")
    print(f"  Minimum bytes:          {analysis.total_minimum_bytes:,}")
    print(f"  Actual bytes:           {analysis.total_actual_bytes:,}")
    print(f"  Arithmetic intensity:   {analysis.arithmetic_intensity:.2f} FLOPs/byte")
    overall_eff = analysis.total_minimum_bytes / analysis.total_actual_bytes if analysis.total_actual_bytes > 0 else 1
    print(f"  Overall reuse eff:      {overall_eff*100:.1f}%")

    print("=" * 70)


def analyze_tile_reuse(
    M: int, K: int, N: int,
    Tm: int, Tk: int, Tn: int,
    loop_order: str = "MNK",
    verbose: bool = True,
) -> TileReuseAnalysis:
    """
    Convenience function to analyze tile reuse.

    Args:
        M, K, N: Problem dimensions
        Tm, Tk, Tn: Tile dimensions (define the 2D tile shapes)
        loop_order: Loop order string (e.g., "MNK", "NKM", "MKN")
        verbose: Print detailed analysis

    Returns:
        TileReuseAnalysis
    """
    analyzer = TileReuseAnalyzer()
    analysis = analyzer.analyze(
        M, K, N, Tm, Tk, Tn,
        loop_order=LoopOrder(loop_order)
    )

    if verbose:
        print_tile_analysis(analysis)

    return analysis
