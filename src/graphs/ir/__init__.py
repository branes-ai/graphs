"""
Graph Intermediate Representation

This package provides data structures and utilities for representing
computational graphs in a hardware-independent manner.
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
