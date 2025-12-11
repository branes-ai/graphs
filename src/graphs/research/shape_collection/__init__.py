"""
Shape Collection Module

Extract and categorize tensor shapes from DNN models.

Classes:
    TensorShapeRecord: Single tensor shape record with full metadata
    ShapeExtractor: Extract tensor shapes from FX-traced models
    DNNClassifier: Classify models into DNN architecture classes
    ShapeDatabase: In-memory and file-backed shape database
"""

from graphs.research.shape_collection.extractor import (
    TensorShapeRecord,
    ShapeExtractor,
)
from graphs.research.shape_collection.categorizer import DNNClassifier
from graphs.research.shape_collection.database import ShapeDatabase

__all__ = [
    'TensorShapeRecord',
    'ShapeExtractor',
    'DNNClassifier',
    'ShapeDatabase',
]
