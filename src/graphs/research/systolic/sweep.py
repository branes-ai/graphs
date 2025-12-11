"""
Array Size Sweep Analysis

Sweep systolic array sizes and analyze utilization across DNN workloads.
"""

import math
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from pathlib import Path

try:
    import pandas as pd
    import numpy as np
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

from graphs.research.shape_collection.database import ShapeDatabase
from graphs.research.shape_collection.extractor import TensorShapeRecord
from graphs.research.systolic.utilization import (
    SystolicArrayConfig,
    SystolicUtilizationCalculator,
    UtilizationResult,
)


# Default array sizes to sweep (common in literature)
ARRAY_SIZES = [4, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128]

# Precisions to analyze
PRECISIONS = ['FP32', 'BF16', 'INT8']


@dataclass
class SweepResult:
    """
    Results from sweeping array sizes.

    Contains per-shape utilization and aggregate statistics.
    """
    array_size: int
    precision: str

    # Per-shape utilization (list of results)
    utilization_results: List[UtilizationResult] = field(default_factory=list)

    # Aggregate statistics
    mean_utilization: float = 0.0
    median_utilization: float = 0.0
    std_utilization: float = 0.0
    p10_utilization: float = 0.0  # 10th percentile (worst 10%)
    p25_utilization: float = 0.0  # 25th percentile
    p75_utilization: float = 0.0  # 75th percentile
    p90_utilization: float = 0.0  # 90th percentile (best 10%)

    # FLOPs-weighted statistics (more representative of actual performance)
    weighted_mean_utilization: float = 0.0

    # Distribution histogram
    utilization_histogram: List[int] = field(default_factory=list)
    histogram_bins: List[float] = field(default_factory=list)

    # Count statistics
    total_shapes: int = 0
    shapes_above_50pct: int = 0
    shapes_above_75pct: int = 0
    shapes_above_90pct: int = 0

    def to_dict(self) -> Dict:
        """Convert to summary dictionary."""
        return {
            'array_size': self.array_size,
            'precision': self.precision,
            'total_shapes': self.total_shapes,
            'mean_utilization': self.mean_utilization,
            'median_utilization': self.median_utilization,
            'std_utilization': self.std_utilization,
            'p10_utilization': self.p10_utilization,
            'p25_utilization': self.p25_utilization,
            'p75_utilization': self.p75_utilization,
            'p90_utilization': self.p90_utilization,
            'weighted_mean_utilization': self.weighted_mean_utilization,
            'pct_above_50': self.shapes_above_50pct / self.total_shapes * 100 if self.total_shapes > 0 else 0,
            'pct_above_75': self.shapes_above_75pct / self.total_shapes * 100 if self.total_shapes > 0 else 0,
            'pct_above_90': self.shapes_above_90pct / self.total_shapes * 100 if self.total_shapes > 0 else 0,
        }


class ArraySizeSweeper:
    """
    Sweep array sizes and analyze utilization.

    Computes utilization statistics for each array size across
    a database of tensor shapes.
    """

    def __init__(
        self,
        array_sizes: List[int] = ARRAY_SIZES,
        precisions: List[str] = PRECISIONS,
    ):
        """
        Initialize sweeper.

        Args:
            array_sizes: List of array sizes to sweep
            precisions: List of precisions to analyze
        """
        self.array_sizes = array_sizes
        self.precisions = precisions

    def sweep(
        self,
        db: ShapeDatabase,
        progress_callback=None,
    ) -> Dict[Tuple[int, str], SweepResult]:
        """
        Run sweep across all sizes and precisions.

        Args:
            db: ShapeDatabase with tensor shapes
            progress_callback: Optional callback(array_size, precision) for progress

        Returns:
            Dictionary mapping (array_size, precision) to SweepResult
        """
        # Filter to matmul-like operations
        matmul_db = db.filter_matmul_ops()
        records = [r for r in matmul_db.records if r.M > 0 and r.K > 0 and r.N > 0]

        results = {}

        for precision in self.precisions:
            for array_size in self.array_sizes:
                if progress_callback:
                    progress_callback(array_size, precision)

                result = self._sweep_single(records, array_size, precision)
                results[(array_size, precision)] = result

        return results

    def _sweep_single(
        self,
        records: List[TensorShapeRecord],
        array_size: int,
        precision: str,
    ) -> SweepResult:
        """
        Run sweep for a single array size and precision.

        Args:
            records: List of shape records
            array_size: Array dimension (rows = cols = array_size)
            precision: Precision string

        Returns:
            SweepResult with statistics
        """
        config = SystolicArrayConfig(rows=array_size, cols=array_size, precision=precision)
        calculator = SystolicUtilizationCalculator(config)

        # Calculate utilization for each shape
        utilization_results = calculator.calculate_batch_utilization(records)

        if not utilization_results:
            return SweepResult(array_size=array_size, precision=precision)

        # Extract utilization values
        utils = [r.effective_utilization for r in utilization_results]
        flops = [r.shape_record.flops if r.shape_record else r.M * r.K * r.N * 2
                for r in utilization_results]

        # Compute statistics
        utils_arr = np.array(utils) if HAS_PANDAS else utils
        flops_arr = np.array(flops) if HAS_PANDAS else flops

        mean_util = float(np.mean(utils_arr)) if HAS_PANDAS else sum(utils) / len(utils)
        median_util = float(np.median(utils_arr)) if HAS_PANDAS else sorted(utils)[len(utils) // 2]
        std_util = float(np.std(utils_arr)) if HAS_PANDAS else self._std(utils)

        # Percentiles
        if HAS_PANDAS:
            p10 = float(np.percentile(utils_arr, 10))
            p25 = float(np.percentile(utils_arr, 25))
            p75 = float(np.percentile(utils_arr, 75))
            p90 = float(np.percentile(utils_arr, 90))
        else:
            sorted_utils = sorted(utils)
            n = len(sorted_utils)
            p10 = sorted_utils[int(n * 0.1)]
            p25 = sorted_utils[int(n * 0.25)]
            p75 = sorted_utils[int(n * 0.75)]
            p90 = sorted_utils[int(n * 0.9)]

        # Weighted mean
        total_flops = sum(flops)
        if total_flops > 0:
            weighted_mean = sum(u * f for u, f in zip(utils, flops)) / total_flops
        else:
            weighted_mean = mean_util

        # Histogram (10 bins from 0 to 1)
        histogram_bins = [i * 0.1 for i in range(11)]
        histogram = [0] * 10
        for u in utils:
            bin_idx = min(9, int(u * 10))
            histogram[bin_idx] += 1

        # Count statistics
        above_50 = sum(1 for u in utils if u >= 0.5)
        above_75 = sum(1 for u in utils if u >= 0.75)
        above_90 = sum(1 for u in utils if u >= 0.9)

        return SweepResult(
            array_size=array_size,
            precision=precision,
            utilization_results=utilization_results,
            mean_utilization=mean_util,
            median_utilization=median_util,
            std_utilization=std_util,
            p10_utilization=p10,
            p25_utilization=p25,
            p75_utilization=p75,
            p90_utilization=p90,
            weighted_mean_utilization=weighted_mean,
            utilization_histogram=histogram,
            histogram_bins=histogram_bins,
            total_shapes=len(utils),
            shapes_above_50pct=above_50,
            shapes_above_75pct=above_75,
            shapes_above_90pct=above_90,
        )

    def _std(self, values: List[float]) -> float:
        """Calculate standard deviation without numpy."""
        if not values:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return math.sqrt(variance)

    def find_optimal_size(
        self,
        results: Dict[Tuple[int, str], SweepResult],
        metric: str = 'weighted_mean_utilization',
        precision: str = 'BF16',
    ) -> Tuple[int, float]:
        """
        Find optimal array size for a given metric.

        Args:
            results: Sweep results dictionary
            metric: Metric to optimize ('mean_utilization', 'weighted_mean_utilization', etc.)
            precision: Precision to filter by

        Returns:
            (optimal_size, metric_value) tuple
        """
        best_size = self.array_sizes[0]
        best_value = 0.0

        for size in self.array_sizes:
            key = (size, precision)
            if key in results:
                value = getattr(results[key], metric, 0.0)
                if value > best_value:
                    best_value = value
                    best_size = size

        return best_size, best_value

    def to_dataframe(
        self,
        results: Dict[Tuple[int, str], SweepResult],
    ) -> 'pd.DataFrame':
        """
        Convert sweep results to DataFrame.

        Args:
            results: Sweep results dictionary

        Returns:
            pandas DataFrame with summary statistics
        """
        if not HAS_PANDAS:
            raise ImportError("pandas required for to_dataframe()")

        data = [result.to_dict() for result in results.values()]
        return pd.DataFrame(data)

    def to_csv(
        self,
        results: Dict[Tuple[int, str], SweepResult],
        path: str,
    ) -> None:
        """
        Save sweep results to CSV.

        Args:
            results: Sweep results dictionary
            path: Output CSV path
        """
        df = self.to_dataframe(results)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)


def sweep_by_class(
    db: ShapeDatabase,
    array_sizes: List[int] = ARRAY_SIZES,
    precision: str = 'BF16',
) -> Dict[str, Dict[int, SweepResult]]:
    """
    Run sweep separately for each DNN class.

    Args:
        db: ShapeDatabase
        array_sizes: Array sizes to sweep
        precision: Precision

    Returns:
        Dictionary mapping class -> {array_size: SweepResult}
    """
    results = {}
    sweeper = ArraySizeSweeper(array_sizes=array_sizes, precisions=[precision])

    for dnn_class in db.get_unique_classes():
        class_db = db.filter_by_class(dnn_class)
        class_results = sweeper.sweep(class_db)

        # Extract just the results for this precision
        results[dnn_class] = {
            size: class_results[(size, precision)]
            for size in array_sizes
            if (size, precision) in class_results
        }

    return results


def analyze_size_sensitivity(
    db: ShapeDatabase,
    precision: str = 'BF16',
) -> Dict[str, List[float]]:
    """
    Analyze how sensitive utilization is to array size.

    Returns utilization values at each array size for plotting.

    Args:
        db: ShapeDatabase
        precision: Precision

    Returns:
        Dictionary with 'sizes' and 'utilization' lists
    """
    sweeper = ArraySizeSweeper(precisions=[precision])
    results = sweeper.sweep(db)

    sizes = []
    mean_utils = []
    weighted_utils = []
    median_utils = []

    for size in ARRAY_SIZES:
        key = (size, precision)
        if key in results:
            sizes.append(size)
            mean_utils.append(results[key].mean_utilization)
            weighted_utils.append(results[key].weighted_mean_utilization)
            median_utils.append(results[key].median_utilization)

    return {
        'sizes': sizes,
        'mean_utilization': mean_utils,
        'weighted_mean_utilization': weighted_utils,
        'median_utilization': median_utils,
    }
