"""
Shape Database

In-memory and file-backed database for tensor shape records.
Supports Parquet and CSV I/O for efficient storage and querying.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
import ast

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import pyarrow.parquet as pq
    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False

from graphs.research.shape_collection.extractor import TensorShapeRecord


@dataclass
class ShapeDatabase:
    """
    In-memory and file-backed shape database.

    Stores TensorShapeRecord objects and provides efficient querying
    and filtering capabilities.
    """
    records: List[TensorShapeRecord] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self):
        return iter(self.records)

    def add(self, record: TensorShapeRecord) -> None:
        """Add a single record to the database."""
        self.records.append(record)

    def add_all(self, records: List[TensorShapeRecord]) -> None:
        """Add multiple records to the database."""
        self.records.extend(records)

    def clear(self) -> None:
        """Clear all records from the database."""
        self.records = []

    # =========================================================================
    # Filtering Methods
    # =========================================================================

    def filter_by_class(self, dnn_class: str) -> 'ShapeDatabase':
        """
        Filter records by DNN class.

        Args:
            dnn_class: One of 'CNN', 'Encoder', 'Decoder', 'FullTransformer'

        Returns:
            New ShapeDatabase with filtered records
        """
        filtered = [r for r in self.records if r.model_class == dnn_class]
        return ShapeDatabase(records=filtered)

    def filter_by_op_type(self, op_type: str) -> 'ShapeDatabase':
        """
        Filter records by operation type.

        Args:
            op_type: Operation type (e.g., 'conv2d', 'linear', 'matmul')

        Returns:
            New ShapeDatabase with filtered records
        """
        filtered = [r for r in self.records if r.op_type == op_type]
        return ShapeDatabase(records=filtered)

    def filter_by_op_types(self, op_types: List[str]) -> 'ShapeDatabase':
        """
        Filter records by multiple operation types.

        Args:
            op_types: List of operation types to include

        Returns:
            New ShapeDatabase with filtered records
        """
        op_set = set(op_types)
        filtered = [r for r in self.records if r.op_type in op_set]
        return ShapeDatabase(records=filtered)

    def filter_by_model(self, model_name: str) -> 'ShapeDatabase':
        """
        Filter records by model name.

        Args:
            model_name: Exact model name to match

        Returns:
            New ShapeDatabase with filtered records
        """
        filtered = [r for r in self.records if r.model_name == model_name]
        return ShapeDatabase(records=filtered)

    def filter_by_model_family(self, family: str) -> 'ShapeDatabase':
        """
        Filter records by model family (case-insensitive prefix match).

        Args:
            family: Model family prefix (e.g., 'resnet', 'bert')

        Returns:
            New ShapeDatabase with filtered records
        """
        family_lower = family.lower()
        filtered = [r for r in self.records
                   if r.model_name.lower().startswith(family_lower)]
        return ShapeDatabase(records=filtered)

    def filter_matmul_ops(self) -> 'ShapeDatabase':
        """
        Filter to only matmul-like operations (ops that map to systolic arrays).

        Returns:
            New ShapeDatabase with only conv2d, linear, matmul, bmm operations
        """
        matmul_ops = {'conv2d', 'conv2d_depthwise', 'conv2d_grouped',
                      'linear', 'matmul', 'bmm', 'multihead_attention'}
        filtered = [r for r in self.records if r.op_type in matmul_ops]
        return ShapeDatabase(records=filtered)

    def filter_by_min_flops(self, min_flops: int) -> 'ShapeDatabase':
        """Filter to operations with at least min_flops."""
        filtered = [r for r in self.records if r.flops >= min_flops]
        return ShapeDatabase(records=filtered)

    def filter_by_min_M(self, min_M: int) -> 'ShapeDatabase':
        """Filter to operations with M dimension >= min_M."""
        filtered = [r for r in self.records if r.M >= min_M]
        return ShapeDatabase(records=filtered)

    # =========================================================================
    # Analysis Methods
    # =========================================================================

    def get_matmul_dimensions(self) -> 'pd.DataFrame':
        """
        Get DataFrame with (model, layer, M, K, N) for systolic array analysis.

        Returns:
            pandas DataFrame with matmul dimensions
        """
        if not HAS_PANDAS:
            raise ImportError("pandas required for get_matmul_dimensions()")

        # Filter to matmul-like ops only
        matmul_db = self.filter_matmul_ops()

        data = []
        for r in matmul_db.records:
            if r.M > 0 and r.K > 0 and r.N > 0:
                data.append({
                    'model_name': r.model_name,
                    'model_class': r.model_class,
                    'layer_name': r.layer_name,
                    'layer_index': r.layer_index,
                    'op_type': r.op_type,
                    'M': r.M,
                    'K': r.K,
                    'N': r.N,
                    'flops': r.flops,
                    'macs': r.macs,
                })

        return pd.DataFrame(data)

    def get_unique_models(self) -> List[str]:
        """Get list of unique model names in database."""
        return sorted(set(r.model_name for r in self.records))

    def get_unique_classes(self) -> List[str]:
        """Get list of unique DNN classes in database."""
        return sorted(set(r.model_class for r in self.records))

    def get_unique_op_types(self) -> List[str]:
        """Get list of unique operation types in database."""
        return sorted(set(r.op_type for r in self.records))

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get summary statistics for the database.

        Returns:
            Dictionary with statistics
        """
        if not self.records:
            return {'total_records': 0}

        matmul_db = self.filter_matmul_ops()

        # Collect M, K, N values
        M_values = [r.M for r in matmul_db.records if r.M > 0]
        K_values = [r.K for r in matmul_db.records if r.K > 0]
        N_values = [r.N for r in matmul_db.records if r.N > 0]

        def safe_stats(values):
            if not values:
                return {'min': 0, 'max': 0, 'mean': 0, 'median': 0}
            sorted_v = sorted(values)
            return {
                'min': min(values),
                'max': max(values),
                'mean': sum(values) / len(values),
                'median': sorted_v[len(sorted_v) // 2],
            }

        # Class distribution
        class_counts = {}
        for r in self.records:
            class_counts[r.model_class] = class_counts.get(r.model_class, 0) + 1

        # Op type distribution
        op_counts = {}
        for r in self.records:
            op_counts[r.op_type] = op_counts.get(r.op_type, 0) + 1

        return {
            'total_records': len(self.records),
            'matmul_ops': len(matmul_db.records),
            'unique_models': len(self.get_unique_models()),
            'unique_classes': len(self.get_unique_classes()),
            'class_distribution': class_counts,
            'op_type_distribution': op_counts,
            'M_stats': safe_stats(M_values),
            'K_stats': safe_stats(K_values),
            'N_stats': safe_stats(N_values),
        }

    # =========================================================================
    # I/O Methods
    # =========================================================================

    def to_dataframe(self) -> 'pd.DataFrame':
        """
        Convert database to pandas DataFrame.

        Returns:
            DataFrame with all records
        """
        if not HAS_PANDAS:
            raise ImportError("pandas required for to_dataframe()")

        data = [r.to_dict() for r in self.records]
        return pd.DataFrame(data)

    def save_csv(self, path: Union[str, Path]) -> None:
        """
        Save database to CSV file.

        Args:
            path: Output CSV file path
        """
        if not HAS_PANDAS:
            raise ImportError("pandas required for save_csv()")

        df = self.to_dataframe()
        df.to_csv(path, index=False)

    def save_parquet(self, path: Union[str, Path]) -> None:
        """
        Save database to Parquet file (efficient columnar storage).

        Args:
            path: Output Parquet file path
        """
        if not HAS_PANDAS:
            raise ImportError("pandas required for save_parquet()")
        if not HAS_PYARROW:
            raise ImportError("pyarrow required for save_parquet()")

        df = self.to_dataframe()
        df.to_parquet(path, index=False)

    def save_json(self, path: Union[str, Path]) -> None:
        """
        Save database to JSON file.

        Args:
            path: Output JSON file path
        """
        data = [r.to_dict() for r in self.records]
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load_csv(cls, path: Union[str, Path]) -> 'ShapeDatabase':
        """
        Load database from CSV file.

        Args:
            path: Input CSV file path

        Returns:
            ShapeDatabase instance
        """
        if not HAS_PANDAS:
            raise ImportError("pandas required for load_csv()")

        df = pd.read_csv(path)
        return cls._from_dataframe(df)

    @classmethod
    def load_parquet(cls, path: Union[str, Path]) -> 'ShapeDatabase':
        """
        Load database from Parquet file.

        Args:
            path: Input Parquet file path

        Returns:
            ShapeDatabase instance
        """
        if not HAS_PANDAS:
            raise ImportError("pandas required for load_parquet()")

        df = pd.read_parquet(path)
        return cls._from_dataframe(df)

    @classmethod
    def load_json(cls, path: Union[str, Path]) -> 'ShapeDatabase':
        """
        Load database from JSON file.

        Args:
            path: Input JSON file path

        Returns:
            ShapeDatabase instance
        """
        with open(path, 'r') as f:
            data = json.load(f)

        records = []
        for d in data:
            records.append(cls._dict_to_record(d))

        return cls(records=records)

    @classmethod
    def load(cls, path: Union[str, Path]) -> 'ShapeDatabase':
        """
        Load database from file (auto-detect format from extension).

        Args:
            path: Input file path (.csv, .parquet, .json)

        Returns:
            ShapeDatabase instance
        """
        path = Path(path)
        suffix = path.suffix.lower()

        if suffix == '.csv':
            return cls.load_csv(path)
        elif suffix == '.parquet':
            return cls.load_parquet(path)
        elif suffix == '.json':
            return cls.load_json(path)
        else:
            raise ValueError(f"Unknown file format: {suffix}")

    def save(self, path: Union[str, Path]) -> None:
        """
        Save database to file (auto-detect format from extension).

        Args:
            path: Output file path (.csv, .parquet, .json)
        """
        path = Path(path)
        suffix = path.suffix.lower()

        # Create parent directories if needed
        path.parent.mkdir(parents=True, exist_ok=True)

        if suffix == '.csv':
            self.save_csv(path)
        elif suffix == '.parquet':
            if HAS_PYARROW:
                self.save_parquet(path)
            else:
                # Fall back to CSV if pyarrow not available
                csv_path = path.with_suffix('.csv')
                print(f"Note: pyarrow not installed, saving as CSV: {csv_path}")
                self.save_csv(csv_path)
        elif suffix == '.json':
            self.save_json(path)
        else:
            raise ValueError(f"Unknown file format: {suffix}")

    @classmethod
    def _from_dataframe(cls, df: 'pd.DataFrame') -> 'ShapeDatabase':
        """Create ShapeDatabase from pandas DataFrame."""
        records = []
        for _, row in df.iterrows():
            records.append(cls._dict_to_record(row.to_dict()))
        return cls(records=records)

    @classmethod
    def _dict_to_record(cls, d: Dict[str, Any]) -> TensorShapeRecord:
        """Convert dictionary to TensorShapeRecord."""

        def parse_shape(s):
            if not s or pd.isna(s) if HAS_PANDAS else not s:
                return None
            if isinstance(s, tuple):
                return s
            if isinstance(s, str):
                try:
                    return tuple(ast.literal_eval(s))
                except (ValueError, SyntaxError):
                    return None
            return None

        input_shape = parse_shape(d.get('input_shape'))
        output_shape = parse_shape(d.get('output_shape'))
        weight_shape = parse_shape(d.get('weight_shape'))

        return TensorShapeRecord(
            model_name=str(d.get('model_name', '')),
            model_class=str(d.get('model_class', '')),
            layer_name=str(d.get('layer_name', '')),
            layer_index=int(d.get('layer_index', 0)),
            op_type=str(d.get('op_type', '')),
            input_shape=input_shape or (),
            input_dtype=str(d.get('input_dtype', 'float32')),
            output_shape=output_shape or (),
            output_dtype=str(d.get('output_dtype', 'float32')),
            weight_shape=weight_shape,
            M=int(d.get('M', 0)),
            K=int(d.get('K', 0)),
            N=int(d.get('N', 0)),
            flops=int(d.get('flops', 0)),
            macs=int(d.get('macs', 0)),
            input_bytes=int(d.get('input_bytes', 0)),
            weight_bytes=int(d.get('weight_bytes', 0)),
            output_bytes=int(d.get('output_bytes', 0)),
            precision=str(d.get('precision', 'float32')),
        )

    # =========================================================================
    # Merge / Combine
    # =========================================================================

    def merge(self, other: 'ShapeDatabase') -> 'ShapeDatabase':
        """
        Merge two databases, removing duplicates.

        Args:
            other: Another ShapeDatabase

        Returns:
            New merged ShapeDatabase
        """
        # Use (model_name, layer_name, layer_index) as unique key
        seen = set()
        merged = []

        for r in self.records:
            key = (r.model_name, r.layer_name, r.layer_index)
            if key not in seen:
                seen.add(key)
                merged.append(r)

        for r in other.records:
            key = (r.model_name, r.layer_name, r.layer_index)
            if key not in seen:
                seen.add(key)
                merged.append(r)

        return ShapeDatabase(records=merged)

    def __add__(self, other: 'ShapeDatabase') -> 'ShapeDatabase':
        """Concatenate two databases (with duplicates)."""
        return ShapeDatabase(records=self.records + other.records)
