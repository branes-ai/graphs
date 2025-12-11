"""
Systolic Array Utilization Calculator

Calculate utilization for matrix operations on systolic arrays of various sizes.
Based on TPU-style utilization model from hardware/mappers/accelerators/tpu.py.
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple

from graphs.research.shape_collection.extractor import TensorShapeRecord


@dataclass
class SystolicArrayConfig:
    """
    Configuration for a systolic array.

    Attributes:
        rows: Array height (number of rows)
        cols: Array width (number of columns)
        precision: Data precision (FP32, BF16, INT8)
        clock_hz: Clock frequency in Hz
    """
    rows: int
    cols: int
    precision: str = 'BF16'
    clock_hz: float = 1.0e9  # 1 GHz default

    @property
    def size(self) -> int:
        """Total PEs in array (rows * cols)."""
        return self.rows * self.cols

    @property
    def peak_ops_per_cycle(self) -> int:
        """Peak operations per cycle (2 ops per MAC)."""
        return self.rows * self.cols * 2

    @property
    def peak_tops(self) -> float:
        """Peak TOPS at configured clock frequency."""
        return self.peak_ops_per_cycle * self.clock_hz / 1e12

    @property
    def is_square(self) -> bool:
        """Check if array is square."""
        return self.rows == self.cols

    def __str__(self) -> str:
        return f"{self.rows}x{self.cols} {self.precision}"


@dataclass
class UtilizationResult:
    """
    Utilization analysis for a single shape on a single systolic array.

    Captures spatial utilization (how much of the array is active),
    pipeline efficiency, and tile counts.
    """
    # Configuration
    array_config: SystolicArrayConfig
    M: int  # Output rows
    K: int  # Reduction dimension
    N: int  # Output columns

    # Dimension utilization (fraction of array dimension used)
    m_utilization: float  # min(M, rows) / rows
    n_utilization: float  # min(N, cols) / cols

    # Combined spatial utilization
    spatial_utilization: float  # m_util * n_util

    # Pipeline efficiency
    pipeline_depth: int  # Depth of systolic pipeline (typically = rows)
    pipeline_fill_overhead: float  # depth / (K + depth)
    pipeline_efficiency: float  # 1 - pipeline_fill_overhead

    # Effective utilization (spatial * pipeline)
    effective_utilization: float

    # Tile counts (for large matrices)
    m_tiles: int  # ceil(M / rows)
    n_tiles: int  # ceil(N / cols)
    k_tiles: int  # For very large K (typically 1)
    total_tiles: int

    # Source shape record (optional)
    shape_record: Optional[TensorShapeRecord] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'array_size': self.array_config.size,
            'array_rows': self.array_config.rows,
            'array_cols': self.array_config.cols,
            'precision': self.array_config.precision,
            'M': self.M,
            'K': self.K,
            'N': self.N,
            'm_utilization': self.m_utilization,
            'n_utilization': self.n_utilization,
            'spatial_utilization': self.spatial_utilization,
            'pipeline_efficiency': self.pipeline_efficiency,
            'effective_utilization': self.effective_utilization,
            'm_tiles': self.m_tiles,
            'n_tiles': self.n_tiles,
            'total_tiles': self.total_tiles,
        }


class SystolicUtilizationCalculator:
    """
    Calculate systolic array utilization for matrix operations.

    Uses TPU-style utilization model:
    - Spatial utilization: min(M/rows, 1.0) * min(N/cols, 1.0)
    - Pipeline overhead: rows / (K + rows)
    - Effective utilization: spatial * (1 - pipeline_overhead)
    """

    def __init__(self, array_config: SystolicArrayConfig):
        """
        Initialize calculator with array configuration.

        Args:
            array_config: Systolic array configuration
        """
        self.config = array_config

    def calculate_utilization(
        self,
        M: int,
        K: int,
        N: int,
        shape_record: Optional[TensorShapeRecord] = None,
    ) -> UtilizationResult:
        """
        Calculate utilization for a single (M, K, N) operation.

        The systolic array performs matrix multiply C = A @ B where:
        - A is M x K
        - B is K x N
        - C is M x N

        For a weight-stationary dataflow:
        - M dimension maps to rows (outputs computed per column)
        - N dimension maps to columns (different output channels)
        - K dimension is the reduction dimension (pipelined through)

        Args:
            M: Output rows (batch * spatial or batch * seq_len)
            K: Reduction dimension (input features)
            N: Output columns (output features)
            shape_record: Optional source shape record

        Returns:
            UtilizationResult with detailed breakdown
        """
        rows = self.config.rows
        cols = self.config.cols

        # Dimension utilization
        # If M < rows, only M rows of the array are active
        m_utilization = min(1.0, M / rows) if rows > 0 else 0.0
        n_utilization = min(1.0, N / cols) if cols > 0 else 0.0

        # Combined spatial utilization
        spatial_utilization = m_utilization * n_utilization

        # Pipeline efficiency
        # Systolic array has a pipeline fill/drain overhead
        # For weight-stationary, pipeline depth = rows
        pipeline_depth = rows

        # Pipeline overhead: depth / (K + depth)
        # Small K means more time spent filling/draining
        if K > 0:
            pipeline_fill_overhead = pipeline_depth / (K + pipeline_depth)
        else:
            pipeline_fill_overhead = 1.0  # Degenerate case

        pipeline_efficiency = 1.0 - pipeline_fill_overhead

        # Effective utilization combines spatial and pipeline effects
        effective_utilization = spatial_utilization * pipeline_efficiency

        # Tile counts for matrices larger than array
        m_tiles = max(1, math.ceil(M / rows)) if rows > 0 else 1
        n_tiles = max(1, math.ceil(N / cols)) if cols > 0 else 1

        # K tiles only needed for very large K (typically not tiled)
        # Some accelerators tile K to fit in accumulator registers
        k_tiles = 1  # Default: K fits in pipeline

        total_tiles = m_tiles * n_tiles * k_tiles

        return UtilizationResult(
            array_config=self.config,
            M=M,
            K=K,
            N=N,
            m_utilization=m_utilization,
            n_utilization=n_utilization,
            spatial_utilization=spatial_utilization,
            pipeline_depth=pipeline_depth,
            pipeline_fill_overhead=pipeline_fill_overhead,
            pipeline_efficiency=pipeline_efficiency,
            effective_utilization=effective_utilization,
            m_tiles=m_tiles,
            n_tiles=n_tiles,
            k_tiles=k_tiles,
            total_tiles=total_tiles,
            shape_record=shape_record,
        )

    def calculate_batch_utilization(
        self,
        records: List[TensorShapeRecord],
    ) -> List[UtilizationResult]:
        """
        Calculate utilization for multiple shape records.

        Args:
            records: List of TensorShapeRecord with M, K, N values

        Returns:
            List of UtilizationResult
        """
        results = []
        for record in records:
            if record.M > 0 and record.K > 0 and record.N > 0:
                result = self.calculate_utilization(
                    record.M, record.K, record.N, record
                )
                results.append(result)
        return results

    def calculate_weighted_utilization(
        self,
        records: List[TensorShapeRecord],
    ) -> Tuple[float, float]:
        """
        Calculate FLOPs-weighted average utilization.

        Larger operations contribute more to the weighted average.

        Args:
            records: List of TensorShapeRecord

        Returns:
            (mean_utilization, weighted_mean_utilization)
        """
        if not records:
            return 0.0, 0.0

        total_util = 0.0
        weighted_util = 0.0
        total_weight = 0.0
        count = 0

        for record in records:
            if record.M > 0 and record.K > 0 and record.N > 0:
                result = self.calculate_utilization(record.M, record.K, record.N)
                total_util += result.effective_utilization
                count += 1

                # Weight by FLOPs
                weight = record.flops if record.flops > 0 else record.M * record.K * record.N * 2
                weighted_util += result.effective_utilization * weight
                total_weight += weight

        mean_util = total_util / count if count > 0 else 0.0
        weighted_mean = weighted_util / total_weight if total_weight > 0 else 0.0

        return mean_util, weighted_mean


def calculate_optimal_array_size(
    M: int,
    K: int,
    N: int,
    candidate_sizes: List[int],
) -> Tuple[int, float]:
    """
    Find optimal array size for a given operation.

    Considers the trade-off between utilization and absolute throughput.
    Larger arrays have higher peak throughput but may have lower utilization.

    Args:
        M: Output rows
        K: Reduction dimension
        N: Output columns
        candidate_sizes: List of array sizes to consider

    Returns:
        (optimal_size, utilization) tuple
    """
    best_size = candidate_sizes[0]
    best_score = 0.0

    for size in candidate_sizes:
        config = SystolicArrayConfig(rows=size, cols=size)
        calc = SystolicUtilizationCalculator(config)
        result = calc.calculate_utilization(M, K, N)

        # Score = utilization * throughput_factor
        # throughput_factor accounts for larger arrays being faster
        throughput_factor = min(1.0, M / size) * min(1.0, N / size)
        score = result.effective_utilization * (1.0 + 0.5 * math.log2(size / candidate_sizes[0]))

        if score > best_score:
            best_score = score
            best_size = size

    # Return utilization at optimal size
    config = SystolicArrayConfig(rows=best_size, cols=best_size)
    calc = SystolicUtilizationCalculator(config)
    result = calc.calculate_utilization(M, K, N)

    return best_size, result.effective_utilization


def estimate_execution_cycles(
    M: int,
    K: int,
    N: int,
    array_config: SystolicArrayConfig,
) -> int:
    """
    Estimate execution cycles for a matrix operation.

    For weight-stationary systolic array:
    - Load weights: K cycles (to fill pipeline)
    - Compute: M * ceil(N / cols) cycles
    - Drain: rows cycles

    Args:
        M: Output rows
        K: Reduction dimension
        N: Output columns
        array_config: Array configuration

    Returns:
        Estimated cycles
    """
    rows = array_config.rows
    cols = array_config.cols

    # Number of N tiles
    n_tiles = math.ceil(N / cols)

    # Number of M tiles
    m_tiles = math.ceil(M / rows)

    # For each tile:
    # - K cycles to stream through reduction dimension
    # - rows cycles for pipeline fill/drain overhead
    cycles_per_tile = K + rows

    total_cycles = m_tiles * n_tiles * cycles_per_tile

    return total_cycles
