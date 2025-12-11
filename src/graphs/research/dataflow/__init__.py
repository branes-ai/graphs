"""
Dataflow Analysis Module

Tiling schedules, loop nests, and data movement analysis.

Classes:
    TileSchedule: Complete tiling schedule for a matrix operation
    TileScheduler: Generate tiling schedules for target hardware
    LoopNest: Explicit loop nest with trip counts and orderings
    LoopLevel: Single loop level in nest
    DataMovementBreakdown: Complete data movement analysis
    DataMovementAnalyzer: Analyze data movement through memory hierarchy

Dataflow Types:
    - Weight-stationary (TPU-style): Maximize weight reuse
    - Output-stationary: Maximize partial sum reuse
    - Row-stationary (Eyeriss-style): Balance all reuse
"""

from graphs.research.dataflow.tiling import (
    TileSchedule,
    TileScheduler,
    DataflowType,
)
from graphs.research.dataflow.loop_nests import (
    LoopNest,
    LoopLevel,
)
from graphs.research.dataflow.data_movement import (
    DataMovementBreakdown,
    DataMovementAnalyzer,
)
from graphs.research.dataflow.dataflows import (
    generate_weight_stationary_loop_nest,
    generate_output_stationary_loop_nest,
    generate_row_stationary_loop_nest,
)

__all__ = [
    'TileSchedule',
    'TileScheduler',
    'DataflowType',
    'LoopNest',
    'LoopLevel',
    'DataMovementBreakdown',
    'DataMovementAnalyzer',
    'generate_weight_stationary_loop_nest',
    'generate_output_stationary_loop_nest',
    'generate_row_stationary_loop_nest',
]
