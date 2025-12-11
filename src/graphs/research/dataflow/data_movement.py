"""
Data Movement Analysis

Analyze data movement through memory hierarchy for tiled computations.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, List
import math

from graphs.research.dataflow.tiling import TileSchedule, DataflowType
from graphs.research.dataflow.loop_nests import LoopNest, create_loop_nest_from_schedule


# Default energy per access at each memory level (picoJoules)
# Based on typical values from literature (Eyeriss, TPU papers)
DEFAULT_ENERGY_PJ = {
    'RF': 0.5,      # Register file: ~0.5 pJ
    'L1': 5.0,      # L1/Scratchpad: ~5 pJ
    'L2': 50.0,     # L2 cache: ~50 pJ
    'DRAM': 200.0,  # DRAM: ~200 pJ
}


@dataclass
class DataMovementBreakdown:
    """
    Complete data movement analysis.

    Tracks bytes transferred and energy consumed at each memory level.
    """
    # Bytes transferred at each level
    rf_reads: int = 0     # Register file reads
    rf_writes: int = 0
    l1_reads: int = 0     # L1/Unified Buffer reads
    l1_writes: int = 0
    l2_reads: int = 0     # L2 cache reads (if present)
    l2_writes: int = 0
    dram_reads: int = 0   # DRAM reads
    dram_writes: int = 0

    # Reuse factors achieved
    input_reuse_factor: float = 1.0   # Times each input byte reused from on-chip
    weight_reuse_factor: float = 1.0  # Times each weight byte reused
    output_reuse_factor: float = 1.0  # Times output accumulated before writeback

    # Energy per access (pJ)
    energy_per_rf_access: float = DEFAULT_ENERGY_PJ['RF']
    energy_per_l1_access: float = DEFAULT_ENERGY_PJ['L1']
    energy_per_l2_access: float = DEFAULT_ENERGY_PJ['L2']
    energy_per_dram_access: float = DEFAULT_ENERGY_PJ['DRAM']

    @property
    def rf_energy_pj(self) -> float:
        """Total RF energy in picoJoules."""
        return (self.rf_reads + self.rf_writes) * self.energy_per_rf_access

    @property
    def l1_energy_pj(self) -> float:
        """Total L1 energy in picoJoules."""
        return (self.l1_reads + self.l1_writes) * self.energy_per_l1_access

    @property
    def l2_energy_pj(self) -> float:
        """Total L2 energy in picoJoules."""
        return (self.l2_reads + self.l2_writes) * self.energy_per_l2_access

    @property
    def dram_energy_pj(self) -> float:
        """Total DRAM energy in picoJoules."""
        return (self.dram_reads + self.dram_writes) * self.energy_per_dram_access

    @property
    def total_energy_pj(self) -> float:
        """Total energy across all memory levels."""
        return self.rf_energy_pj + self.l1_energy_pj + self.l2_energy_pj + self.dram_energy_pj

    @property
    def total_energy_mj(self) -> float:
        """Total energy in milliJoules."""
        return self.total_energy_pj / 1e9

    @property
    def total_bytes_moved(self) -> int:
        """Total bytes moved across all levels."""
        return (self.rf_reads + self.rf_writes +
                self.l1_reads + self.l1_writes +
                self.l2_reads + self.l2_writes +
                self.dram_reads + self.dram_writes)

    @property
    def dram_bytes(self) -> int:
        """Total DRAM traffic."""
        return self.dram_reads + self.dram_writes

    def energy_breakdown(self) -> Dict[str, float]:
        """Get energy breakdown by memory level."""
        total = self.total_energy_pj
        if total == 0:
            return {'RF': 0, 'L1': 0, 'L2': 0, 'DRAM': 0}

        return {
            'RF': self.rf_energy_pj / total * 100,
            'L1': self.l1_energy_pj / total * 100,
            'L2': self.l2_energy_pj / total * 100,
            'DRAM': self.dram_energy_pj / total * 100,
        }

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'rf_reads': self.rf_reads,
            'rf_writes': self.rf_writes,
            'l1_reads': self.l1_reads,
            'l1_writes': self.l1_writes,
            'l2_reads': self.l2_reads,
            'l2_writes': self.l2_writes,
            'dram_reads': self.dram_reads,
            'dram_writes': self.dram_writes,
            'input_reuse_factor': self.input_reuse_factor,
            'weight_reuse_factor': self.weight_reuse_factor,
            'output_reuse_factor': self.output_reuse_factor,
            'rf_energy_pj': self.rf_energy_pj,
            'l1_energy_pj': self.l1_energy_pj,
            'l2_energy_pj': self.l2_energy_pj,
            'dram_energy_pj': self.dram_energy_pj,
            'total_energy_pj': self.total_energy_pj,
            'total_energy_mj': self.total_energy_mj,
            'dram_bytes': self.dram_bytes,
        }


class DataMovementAnalyzer:
    """
    Analyze data movement through memory hierarchy.

    Computes bytes transferred and energy consumed at each level
    based on the tiling schedule and dataflow.
    """

    def __init__(
        self,
        energy_per_rf_access: float = DEFAULT_ENERGY_PJ['RF'],
        energy_per_l1_access: float = DEFAULT_ENERGY_PJ['L1'],
        energy_per_l2_access: float = DEFAULT_ENERGY_PJ['L2'],
        energy_per_dram_access: float = DEFAULT_ENERGY_PJ['DRAM'],
    ):
        """
        Initialize analyzer with energy model.

        Args:
            energy_per_rf_access: pJ per RF access
            energy_per_l1_access: pJ per L1 access
            energy_per_l2_access: pJ per L2 access
            energy_per_dram_access: pJ per DRAM access
        """
        self.energy_per_rf = energy_per_rf_access
        self.energy_per_l1 = energy_per_l1_access
        self.energy_per_l2 = energy_per_l2_access
        self.energy_per_dram = energy_per_dram_access

    def analyze(
        self,
        schedule: TileSchedule,
        batch_size: int = 1,
    ) -> DataMovementBreakdown:
        """
        Analyze data movement for a tiling schedule.

        Args:
            schedule: TileSchedule with tiling parameters
            batch_size: Batch size (affects weight reuse)

        Returns:
            DataMovementBreakdown with complete analysis
        """
        # Create loop nest for detailed analysis
        loop_nest = create_loop_nest_from_schedule(schedule)

        # Analyze based on dataflow type
        if schedule.dataflow == DataflowType.WEIGHT_STATIONARY:
            return self._analyze_weight_stationary(schedule, batch_size)
        elif schedule.dataflow == DataflowType.OUTPUT_STATIONARY:
            return self._analyze_output_stationary(schedule, batch_size)
        else:  # ROW_STATIONARY
            return self._analyze_row_stationary(schedule, batch_size)

    def _analyze_weight_stationary(
        self,
        schedule: TileSchedule,
        batch_size: int,
    ) -> DataMovementBreakdown:
        """
        Analyze weight-stationary dataflow.

        Weight-stationary: weights loaded once per (K_tile, N_tile),
        reused across all M_tiles.
        """
        M, K, N = schedule.M, schedule.K, schedule.N
        Tm, Tk, Tn = schedule.Tm, schedule.Tk, schedule.Tn
        elem_size = schedule.element_size

        # Number of tiles
        num_m_tiles = schedule.num_m_tiles
        num_k_tiles = schedule.num_k_tiles
        num_n_tiles = schedule.num_n_tiles

        # DRAM traffic:
        # - Inputs: Each input tile loaded once per M,K tile (reused across N)
        #   Total: M * K * elem_size
        input_dram_bytes = M * K * elem_size

        # - Weights: Each weight tile loaded once per K,N tile (reused across M)
        #   Total: K * N * elem_size (amortized by batch_size for inference)
        weight_dram_bytes = K * N * elem_size // batch_size

        # - Outputs: Each output tile written once per M,N tile
        #   Total: M * N * elem_size
        output_dram_bytes = M * N * elem_size

        dram_reads = input_dram_bytes + weight_dram_bytes
        dram_writes = output_dram_bytes

        # L1 traffic (on-chip buffer):
        # Weights loaded from L1 multiple times as we iterate through M tiles
        l1_reads = (input_dram_bytes +  # Inputs read once from L1
                   weight_dram_bytes * num_m_tiles)  # Weights reused
        l1_writes = output_dram_bytes  # Outputs written to L1 before DRAM

        # RF traffic (register file in systolic array):
        # Each element accessed multiple times for MAC operations
        # MACs = M * K * N, each MAC reads 2 values and writes 1 partial sum
        total_macs = M * K * N
        rf_reads = 2 * total_macs * elem_size
        rf_writes = total_macs * elem_size

        # Reuse factors
        input_reuse = num_n_tiles  # Input reused across N dimension
        weight_reuse = num_m_tiles * batch_size  # Weights reused across M and batches
        output_reuse = num_k_tiles  # Output accumulated across K tiles

        return DataMovementBreakdown(
            rf_reads=rf_reads,
            rf_writes=rf_writes,
            l1_reads=l1_reads,
            l1_writes=l1_writes,
            l2_reads=0,  # Assuming no L2 in simple model
            l2_writes=0,
            dram_reads=dram_reads,
            dram_writes=dram_writes,
            input_reuse_factor=input_reuse,
            weight_reuse_factor=weight_reuse,
            output_reuse_factor=output_reuse,
            energy_per_rf_access=self.energy_per_rf,
            energy_per_l1_access=self.energy_per_l1,
            energy_per_l2_access=self.energy_per_l2,
            energy_per_dram_access=self.energy_per_dram,
        )

    def _analyze_output_stationary(
        self,
        schedule: TileSchedule,
        batch_size: int,
    ) -> DataMovementBreakdown:
        """
        Analyze output-stationary dataflow.

        Output-stationary: outputs stay in accumulators while
        K dimension is reduced.
        """
        M, K, N = schedule.M, schedule.K, schedule.N
        Tm, Tk, Tn = schedule.Tm, schedule.Tk, schedule.Tn
        elem_size = schedule.element_size

        num_m_tiles = schedule.num_m_tiles
        num_k_tiles = schedule.num_k_tiles
        num_n_tiles = schedule.num_n_tiles

        # DRAM traffic:
        # - Inputs: loaded multiple times (once per K tile, not reused well across N)
        input_dram_bytes = M * K * elem_size

        # - Weights: loaded once per K,N combination
        weight_dram_bytes = K * N * elem_size // batch_size

        # - Outputs: written once (stay in accumulators during K reduction)
        output_dram_bytes = M * N * elem_size

        dram_reads = input_dram_bytes + weight_dram_bytes
        dram_writes = output_dram_bytes

        # L1 traffic: both inputs and weights loaded for each output tile
        l1_reads = (input_dram_bytes * num_n_tiles +  # Inputs for each N tile
                   weight_dram_bytes * num_m_tiles)   # Weights for each M tile
        l1_writes = output_dram_bytes

        # RF traffic
        total_macs = M * K * N
        rf_reads = 2 * total_macs * elem_size
        rf_writes = total_macs * elem_size

        # Reuse factors (output reuse is maximized)
        input_reuse = num_n_tiles
        weight_reuse = num_m_tiles * batch_size
        output_reuse = num_k_tiles  # Full K reduction before writeback

        return DataMovementBreakdown(
            rf_reads=rf_reads,
            rf_writes=rf_writes,
            l1_reads=l1_reads,
            l1_writes=l1_writes,
            l2_reads=0,
            l2_writes=0,
            dram_reads=dram_reads,
            dram_writes=dram_writes,
            input_reuse_factor=input_reuse,
            weight_reuse_factor=weight_reuse,
            output_reuse_factor=output_reuse,
            energy_per_rf_access=self.energy_per_rf,
            energy_per_l1_access=self.energy_per_l1,
            energy_per_l2_access=self.energy_per_l2,
            energy_per_dram_access=self.energy_per_dram,
        )

    def _analyze_row_stationary(
        self,
        schedule: TileSchedule,
        batch_size: int,
    ) -> DataMovementBreakdown:
        """
        Analyze row-stationary dataflow.

        Row-stationary: balanced reuse of all three operands.
        """
        M, K, N = schedule.M, schedule.K, schedule.N
        elem_size = schedule.element_size

        num_m_tiles = schedule.num_m_tiles
        num_k_tiles = schedule.num_k_tiles
        num_n_tiles = schedule.num_n_tiles

        # DRAM traffic (balanced approach)
        input_dram_bytes = M * K * elem_size
        weight_dram_bytes = K * N * elem_size // batch_size
        output_dram_bytes = M * N * elem_size

        dram_reads = input_dram_bytes + weight_dram_bytes
        dram_writes = output_dram_bytes

        # L1 traffic (moderate reuse from all operands)
        l1_reads = (input_dram_bytes +
                   weight_dram_bytes +
                   output_dram_bytes * (num_k_tiles - 1))  # Partial sums read back
        l1_writes = output_dram_bytes * num_k_tiles  # Partial sums written

        # RF traffic
        total_macs = M * K * N
        rf_reads = 2 * total_macs * elem_size
        rf_writes = total_macs * elem_size

        # Balanced reuse
        input_reuse = num_n_tiles
        weight_reuse = num_m_tiles * batch_size
        output_reuse = num_k_tiles

        return DataMovementBreakdown(
            rf_reads=rf_reads,
            rf_writes=rf_writes,
            l1_reads=l1_reads,
            l1_writes=l1_writes,
            l2_reads=0,
            l2_writes=0,
            dram_reads=dram_reads,
            dram_writes=dram_writes,
            input_reuse_factor=input_reuse,
            weight_reuse_factor=weight_reuse,
            output_reuse_factor=output_reuse,
            energy_per_rf_access=self.energy_per_rf,
            energy_per_l1_access=self.energy_per_l1,
            energy_per_l2_access=self.energy_per_l2,
            energy_per_dram_access=self.energy_per_dram,
        )

    def compare_dataflows(
        self,
        schedule_ws: TileSchedule,
        schedule_os: TileSchedule,
        schedule_rs: TileSchedule,
        batch_size: int = 1,
    ) -> Dict[str, DataMovementBreakdown]:
        """
        Compare data movement across dataflow strategies.

        Args:
            schedule_ws: Weight-stationary schedule
            schedule_os: Output-stationary schedule
            schedule_rs: Row-stationary schedule
            batch_size: Batch size

        Returns:
            Dictionary mapping dataflow name to breakdown
        """
        return {
            'weight_stationary': self.analyze(schedule_ws, batch_size),
            'output_stationary': self.analyze(schedule_os, batch_size),
            'row_stationary': self.analyze(schedule_rs, batch_size),
        }
