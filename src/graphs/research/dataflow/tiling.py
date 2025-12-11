"""
Tiling Schedule Generation

Generate tiling schedules for matrix operations on systolic arrays.
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from enum import Enum


class DataflowType(Enum):
    """Dataflow strategy for systolic array."""
    WEIGHT_STATIONARY = "weight_stationary"  # TPU-style
    OUTPUT_STATIONARY = "output_stationary"  # Maximize output reuse
    ROW_STATIONARY = "row_stationary"        # Eyeriss-style


@dataclass
class TileSchedule:
    """
    Complete tiling schedule for a matrix operation.

    For matrix multiply C = A @ B where:
    - A is (M, K) - input activations
    - B is (K, N) - weights
    - C is (M, N) - output activations
    """
    # Original matrix dimensions
    M: int  # Output rows (batch * spatial)
    K: int  # Reduction dimension
    N: int  # Output columns (output features)

    # Tile sizes
    Tm: int  # M tile size (maps to array rows)
    Tk: int  # K tile size (reduction tile)
    Tn: int  # N tile size (maps to array cols)

    # Tile counts
    num_m_tiles: int
    num_k_tiles: int
    num_n_tiles: int
    total_tiles: int

    # Memory footprint per tile (bytes)
    input_tile_bytes: int   # (Tm, Tk) tile of A
    weight_tile_bytes: int  # (Tk, Tn) tile of B
    output_tile_bytes: int  # (Tm, Tn) tile of C

    # Reuse factors (how many times each tile is accessed)
    input_reuse: int   # Times input tile is reused (across N tiles)
    weight_reuse: int  # Times weight tile is reused (across M tiles)
    output_reuse: int  # Times output tile is updated (across K tiles)

    # Dataflow type
    dataflow: DataflowType = DataflowType.WEIGHT_STATIONARY

    # Element size in bytes
    element_size: int = 2  # BF16 default

    @property
    def total_input_bytes(self) -> int:
        """Total input bytes loaded (accounting for reuse)."""
        # Each input tile loaded once per (M_tile, K_tile) combination
        return self.input_tile_bytes * self.num_m_tiles * self.num_k_tiles

    @property
    def total_weight_bytes(self) -> int:
        """Total weight bytes loaded (accounting for reuse)."""
        # Each weight tile loaded once per (K_tile, N_tile) combination
        return self.weight_tile_bytes * self.num_k_tiles * self.num_n_tiles

    @property
    def total_output_bytes(self) -> int:
        """Total output bytes written."""
        # Output written once per (M_tile, N_tile) combination
        return self.output_tile_bytes * self.num_m_tiles * self.num_n_tiles

    @property
    def total_memory_traffic(self) -> int:
        """Total memory traffic (input + weight + output)."""
        return self.total_input_bytes + self.total_weight_bytes + self.total_output_bytes

    @property
    def compute_ops(self) -> int:
        """Total compute operations (2 * M * K * N for FMA)."""
        return 2 * self.M * self.K * self.N

    @property
    def arithmetic_intensity(self) -> float:
        """Ops per byte of memory traffic."""
        if self.total_memory_traffic == 0:
            return 0.0
        return self.compute_ops / self.total_memory_traffic

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'M': self.M,
            'K': self.K,
            'N': self.N,
            'Tm': self.Tm,
            'Tk': self.Tk,
            'Tn': self.Tn,
            'num_m_tiles': self.num_m_tiles,
            'num_k_tiles': self.num_k_tiles,
            'num_n_tiles': self.num_n_tiles,
            'total_tiles': self.total_tiles,
            'input_tile_bytes': self.input_tile_bytes,
            'weight_tile_bytes': self.weight_tile_bytes,
            'output_tile_bytes': self.output_tile_bytes,
            'input_reuse': self.input_reuse,
            'weight_reuse': self.weight_reuse,
            'output_reuse': self.output_reuse,
            'dataflow': self.dataflow.value,
            'total_memory_traffic': self.total_memory_traffic,
            'arithmetic_intensity': self.arithmetic_intensity,
        }


class TileScheduler:
    """
    Generate tiling schedules for target hardware.

    Supports multiple dataflow strategies:
    - Weight-stationary: Load weights once, stream activations
    - Output-stationary: Keep partial sums on-chip
    - Row-stationary: Balance all three operand reuses
    """

    def __init__(
        self,
        array_rows: int,
        array_cols: int,
        l1_size_bytes: int = 32 * 1024 * 1024,  # 32 MiB (TPU Unified Buffer)
        l2_size_bytes: int = 0,  # L2 cache (if present)
        element_size: int = 2,   # BF16 default
    ):
        """
        Initialize scheduler with hardware parameters.

        Args:
            array_rows: Systolic array rows
            array_cols: Systolic array columns
            l1_size_bytes: L1/Unified Buffer size
            l2_size_bytes: L2 cache size (0 if not present)
            element_size: Bytes per element
        """
        self.array_rows = array_rows
        self.array_cols = array_cols
        self.l1_size = l1_size_bytes
        self.l2_size = l2_size_bytes
        self.element_size = element_size

    def schedule(
        self,
        M: int,
        K: int,
        N: int,
        dataflow: DataflowType = DataflowType.WEIGHT_STATIONARY,
    ) -> TileSchedule:
        """
        Generate optimal tiling schedule.

        Args:
            M: Output rows
            K: Reduction dimension
            N: Output columns
            dataflow: Dataflow strategy

        Returns:
            TileSchedule with complete tiling information
        """
        if dataflow == DataflowType.WEIGHT_STATIONARY:
            return self._schedule_weight_stationary(M, K, N)
        elif dataflow == DataflowType.OUTPUT_STATIONARY:
            return self._schedule_output_stationary(M, K, N)
        elif dataflow == DataflowType.ROW_STATIONARY:
            return self._schedule_row_stationary(M, K, N)
        else:
            raise ValueError(f"Unknown dataflow: {dataflow}")

    def _schedule_weight_stationary(self, M: int, K: int, N: int) -> TileSchedule:
        """
        Generate weight-stationary schedule (TPU-style).

        Weight-stationary maximizes weight reuse by keeping weights
        in the array while streaming activations through.

        Loop order (outer to inner):
        for n_tile in [0, N/Tn):
            for k_tile in [0, K/Tk):
                load weights[k_tile, n_tile] -> Array registers
                for m_tile in [0, M/Tm):
                    load inputs[m_tile, k_tile] -> Input buffer
                    compute (inputs @ weights) -> Accumulators
                    if k_tile == last:
                        store outputs[m_tile, n_tile] -> Output buffer
        """
        # Tile sizes: Array dimensions determine Tm and Tn
        Tm = min(M, self.array_rows)
        Tn = min(N, self.array_cols)

        # K tile: Full K fits in pipeline (typical for weight-stationary)
        # Could tile K if accumulator size is limited
        Tk = K

        # Tile counts
        num_m_tiles = math.ceil(M / Tm)
        num_k_tiles = math.ceil(K / Tk)
        num_n_tiles = math.ceil(N / Tn)
        total_tiles = num_m_tiles * num_k_tiles * num_n_tiles

        # Tile memory footprints
        input_tile_bytes = Tm * Tk * self.element_size
        weight_tile_bytes = Tk * Tn * self.element_size
        output_tile_bytes = Tm * Tn * self.element_size

        # Reuse factors
        # Input: reused across all N tiles (once per M,K combination)
        input_reuse = num_n_tiles

        # Weight: reused across all M tiles (once per K,N combination)
        weight_reuse = num_m_tiles

        # Output: updated K times (partial sums accumulated)
        output_reuse = num_k_tiles

        return TileSchedule(
            M=M, K=K, N=N,
            Tm=Tm, Tk=Tk, Tn=Tn,
            num_m_tiles=num_m_tiles,
            num_k_tiles=num_k_tiles,
            num_n_tiles=num_n_tiles,
            total_tiles=total_tiles,
            input_tile_bytes=input_tile_bytes,
            weight_tile_bytes=weight_tile_bytes,
            output_tile_bytes=output_tile_bytes,
            input_reuse=input_reuse,
            weight_reuse=weight_reuse,
            output_reuse=output_reuse,
            dataflow=DataflowType.WEIGHT_STATIONARY,
            element_size=self.element_size,
        )

    def _schedule_output_stationary(self, M: int, K: int, N: int) -> TileSchedule:
        """
        Generate output-stationary schedule.

        Output-stationary keeps partial sums on-chip, streaming both
        inputs and weights through the array.

        Loop order (outer to inner):
        for m_tile in [0, M/Tm):
            for n_tile in [0, N/Tn):
                init outputs[m_tile, n_tile] = 0
                for k_tile in [0, K/Tk):
                    load inputs[m_tile, k_tile]
                    load weights[k_tile, n_tile]
                    outputs += inputs @ weights
                store outputs[m_tile, n_tile]
        """
        Tm = min(M, self.array_rows)
        Tn = min(N, self.array_cols)

        # Tile K to fit in buffer alongside output
        max_k_tile = self.l1_size // (2 * self.element_size * max(Tm, Tn))
        Tk = min(K, max(1, max_k_tile))

        num_m_tiles = math.ceil(M / Tm)
        num_k_tiles = math.ceil(K / Tk)
        num_n_tiles = math.ceil(N / Tn)
        total_tiles = num_m_tiles * num_k_tiles * num_n_tiles

        input_tile_bytes = Tm * Tk * self.element_size
        weight_tile_bytes = Tk * Tn * self.element_size
        output_tile_bytes = Tm * Tn * self.element_size

        # Output-stationary: each output tile stays in accumulators
        # while K dimension is reduced
        input_reuse = num_n_tiles
        weight_reuse = num_m_tiles
        output_reuse = 1  # Output stays in place

        return TileSchedule(
            M=M, K=K, N=N,
            Tm=Tm, Tk=Tk, Tn=Tn,
            num_m_tiles=num_m_tiles,
            num_k_tiles=num_k_tiles,
            num_n_tiles=num_n_tiles,
            total_tiles=total_tiles,
            input_tile_bytes=input_tile_bytes,
            weight_tile_bytes=weight_tile_bytes,
            output_tile_bytes=output_tile_bytes,
            input_reuse=input_reuse,
            weight_reuse=weight_reuse,
            output_reuse=output_reuse,
            dataflow=DataflowType.OUTPUT_STATIONARY,
            element_size=self.element_size,
        )

    def _schedule_row_stationary(self, M: int, K: int, N: int) -> TileSchedule:
        """
        Generate row-stationary schedule (Eyeriss-style).

        Row-stationary balances reuse of all three operands by
        processing rows of the operation together.

        Each PE handles a row of C, accumulating partial sums locally.
        """
        # For row-stationary, tile sizes balance all dimensions
        Tm = min(M, self.array_rows)
        Tn = min(N, self.array_cols)

        # K tiles sized to maximize reuse within on-chip buffer
        buffer_per_operand = self.l1_size // 3
        max_tk = buffer_per_operand // (max(Tm, Tn) * self.element_size)
        Tk = min(K, max(1, max_tk))

        num_m_tiles = math.ceil(M / Tm)
        num_k_tiles = math.ceil(K / Tk)
        num_n_tiles = math.ceil(N / Tn)
        total_tiles = num_m_tiles * num_k_tiles * num_n_tiles

        input_tile_bytes = Tm * Tk * self.element_size
        weight_tile_bytes = Tk * Tn * self.element_size
        output_tile_bytes = Tm * Tn * self.element_size

        # Row-stationary attempts to balance reuse
        # Each row of output is computed by a PE
        input_reuse = num_n_tiles
        weight_reuse = num_m_tiles
        output_reuse = num_k_tiles

        return TileSchedule(
            M=M, K=K, N=N,
            Tm=Tm, Tk=Tk, Tn=Tn,
            num_m_tiles=num_m_tiles,
            num_k_tiles=num_k_tiles,
            num_n_tiles=num_n_tiles,
            total_tiles=total_tiles,
            input_tile_bytes=input_tile_bytes,
            weight_tile_bytes=weight_tile_bytes,
            output_tile_bytes=output_tile_bytes,
            input_reuse=input_reuse,
            weight_reuse=weight_reuse,
            output_reuse=output_reuse,
            dataflow=DataflowType.ROW_STATIONARY,
            element_size=self.element_size,
        )

    def compare_dataflows(
        self,
        M: int,
        K: int,
        N: int,
    ) -> Dict[DataflowType, TileSchedule]:
        """
        Compare all dataflow strategies for a given operation.

        Args:
            M: Output rows
            K: Reduction dimension
            N: Output columns

        Returns:
            Dictionary mapping dataflow type to schedule
        """
        return {
            DataflowType.WEIGHT_STATIONARY: self.schedule(M, K, N, DataflowType.WEIGHT_STATIONARY),
            DataflowType.OUTPUT_STATIONARY: self.schedule(M, K, N, DataflowType.OUTPUT_STATIONARY),
            DataflowType.ROW_STATIONARY: self.schedule(M, K, N, DataflowType.ROW_STATIONARY),
        }
