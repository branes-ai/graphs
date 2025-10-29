"""
Graph Transformations

This package provides transformations that modify or reorganize computational graphs.
Includes partitioning, fusion, tiling, and other graph restructuring operations.
"""

from .decomposing_tracer import (
    DecomposingAttentionTracer,
    trace_with_decomposition,
)

__all__ = [
    'DecomposingAttentionTracer',
    'trace_with_decomposition',
]
