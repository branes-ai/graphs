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

__all__ = [
    'OperationType',
    'BottleneckType',
    'PartitionReason',
    'TensorDescriptor',
    'ParallelismDescriptor',
    'SubgraphDescriptor',
    'SubgraphConcurrency',
    'ConcurrencyDescriptor',
    'PartitionReport',
]
