"""
Core Graph Data Structures

This package provides hardware-independent data structures for representing
computational graphs in the graphs framework.

Migrated from graphs.ir (deprecated).
"""

from .structures import (
    OperationType,
    BottleneckType,
    PartitionReason,
    TensorDescriptor,
    ParallelismDescriptor,
    SubgraphDescriptor,
    SubgraphConcurrency,
    ConcurrencyDescriptor,
    PartitionReport,
)

from .confidence import (
    ConfidenceLevel,
    EstimationConfidence,
    DEFAULT_CALIBRATED_SCORE,
    DEFAULT_INTERPOLATED_SCORE,
    DEFAULT_THEORETICAL_SCORE,
)

__all__ = [
    # Graph structures
    'OperationType',
    'BottleneckType',
    'PartitionReason',
    'TensorDescriptor',
    'ParallelismDescriptor',
    'SubgraphDescriptor',
    'SubgraphConcurrency',
    'ConcurrencyDescriptor',
    'PartitionReport',
    # Confidence
    'ConfidenceLevel',
    'EstimationConfidence',
    'DEFAULT_CALIBRATED_SCORE',
    'DEFAULT_INTERPOLATED_SCORE',
    'DEFAULT_THEORETICAL_SCORE',
]
