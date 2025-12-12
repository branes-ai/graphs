"""
Research Facility for DNN Tensor Shape Analysis and Systolic Array Utilization

This package provides tools for:
1. Shape Collection: Extract tensor shapes from 140+ TorchVision and HuggingFace models
2. Shape Visualization: Publication-ready distribution plots and heatmaps
3. Systolic Utilization: Analyze array utilization across 13 different array sizes
4. Dataflow Analysis: Full tiling schedules with loop nests and data movement
5. Tile Reuse: Block algebra, double-buffering, distributed L3 orchestration

Modules:
    shape_collection: TensorShapeRecord, ShapeExtractor, DNNClassifier, ShapeDatabase
    visualization: Distribution plots, heatmaps, publication styling
    systolic: Utilization calculator, array size sweep, utilization visualization
    dataflow: Tiling schedules, loop nests, data movement analysis
    tiling: Block algebra, hierarchical blocking, tile rotation algorithms
"""

from graphs.research.shape_collection import (
    TensorShapeRecord,
    ShapeExtractor,
    DNNClassifier,
    ShapeDatabase,
)
from graphs.research.systolic import (
    SystolicArrayConfig,
    UtilizationResult,
    SystolicUtilizationCalculator,
    ArraySizeSweeper,
)
from graphs.research.dataflow import (
    TileSchedule,
    TileScheduler,
    LoopNest,
    LoopLevel,
    DataMovementBreakdown,
)

__all__ = [
    # Shape collection
    'TensorShapeRecord',
    'ShapeExtractor',
    'DNNClassifier',
    'ShapeDatabase',
    # Systolic
    'SystolicArrayConfig',
    'UtilizationResult',
    'SystolicUtilizationCalculator',
    'ArraySizeSweeper',
    # Dataflow
    'TileSchedule',
    'TileScheduler',
    'LoopNest',
    'LoopLevel',
    'DataMovementBreakdown',
]
