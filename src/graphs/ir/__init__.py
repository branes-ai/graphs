"""
Graph Intermediate Representation

DEPRECATED: This module has been renamed to graphs.core.
Please update your imports:
    from graphs.ir import ...  ->  from graphs.core import ...

This shim will be removed in version 1.0.
"""

import warnings

warnings.warn(
    "graphs.ir is deprecated. Use graphs.core instead. "
    "This module will be removed in version 1.0.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export everything from the new location
from graphs.core import (
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
