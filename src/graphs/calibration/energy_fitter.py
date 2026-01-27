"""
Energy Coefficient Fitting from Power Measurements

Fits energy model coefficients from benchmark data with power measurements:
- pJ/op (picojoules per operation) for compute operations
- pJ/byte for memory operations
- Static/idle power (watts)

The energy model is:
    Power = (ops_per_sec * pj_per_op) + (bytes_per_sec * pj_per_byte) + static_power

Or equivalently:
    Energy = (ops * pj_per_op) + (bytes * pj_per_byte) + (static_power * time)

This module uses linear regression to fit these coefficients from
benchmark results that include power measurements.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import numpy as np

from graphs.benchmarks.schema import BenchmarkResult, GEMMSpec, MemoryBenchSpec


class EnergyFitQuality(Enum):
    """Quality level of the energy fit"""
    EXCELLENT = "excellent"  # R-squared >= 0.95
    GOOD = "good"            # R-squared >= 0.85
    FAIR = "fair"            # R-squared >= 0.70
    POOR = "poor"            # R-squared < 0.70
    INSUFFICIENT = "insufficient"  # Not enough data


@dataclass
class EnergyFitMetrics:
    """Metrics describing quality of energy coefficient fit"""
    num_data_points: int
    r_squared: float
    residual_std: float  # Standard deviation of residuals in watts
    quality: EnergyFitQuality
    condition_number: float = 0.0  # Numerical stability indicator

    def to_dict(self) -> Dict[str, Any]:
        return {
            'num_data_points': self.num_data_points,
            'r_squared': self.r_squared,
            'residual_std': self.residual_std,
            'quality': self.quality.value,
            'condition_number': self.condition_number,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EnergyFitMetrics':
        data = data.copy()
        data['quality'] = EnergyFitQuality(data['quality'])
        return cls(**data)


@dataclass
class EnergyCoefficients:
    """
    Calibrated energy coefficients for power modeling.

    Energy model:
        Power = compute_pj_per_op * GOPS + memory_pj_per_byte * GB/s * 1e9 + static_power_watts

    Or for total energy:
        Energy_J = compute_pj_per_op * ops * 1e-12 + memory_pj_per_byte * bytes * 1e-12 + static_power_watts * time
    """

    # Compute energy coefficient (picojoules per operation)
    compute_pj_per_op: float

    # Memory energy coefficient (picojoules per byte)
    memory_pj_per_byte: float

    # Static/idle power (watts)
    static_power_watts: float

    # Per-precision coefficients (optional)
    compute_pj_per_op_by_precision: Dict[str, float] = field(default_factory=dict)

    # Per-memory-level coefficients (optional)
    memory_pj_per_byte_by_level: Dict[str, float] = field(default_factory=dict)

    # Fit quality metrics
    fit_metrics: Optional[EnergyFitMetrics] = None

    # Metadata
    device_name: str = ""
    precision: str = "fp32"

    # Comparison to theoretical
    theoretical_compute_pj_per_op: Optional[float] = None
    theoretical_memory_pj_per_byte: Optional[float] = None

    def predict_power(
        self,
        gops: float,
        gbps: float,
        precision: str = "fp32",
    ) -> float:
        """
        Predict power consumption given compute and memory rates.

        Args:
            gops: Compute throughput in GOPS (giga-ops per second)
            gbps: Memory bandwidth in GB/s
            precision: Numerical precision (for per-precision coefficients)

        Returns:
            Predicted power in watts
        """
        # Get precision-specific coefficient if available
        pj_per_op = self.compute_pj_per_op_by_precision.get(
            precision, self.compute_pj_per_op
        )

        # Power = (GOPS * pJ/op * 1e9 * 1e-12) + (GB/s * pJ/byte * 1e9 * 1e-12) + static
        #       = GOPS * pJ/op * 1e-3 + GB/s * pJ/byte * 1e-3 + static
        compute_power = gops * pj_per_op * 1e-3  # GOPS to W
        memory_power = gbps * self.memory_pj_per_byte * 1e-3  # GB/s to W

        return compute_power + memory_power + self.static_power_watts

    def predict_energy(
        self,
        ops: int,
        bytes_transferred: int,
        duration_s: float,
        precision: str = "fp32",
    ) -> float:
        """
        Predict energy consumption for a workload.

        Args:
            ops: Total operations (FLOPs or integer ops)
            bytes_transferred: Total bytes transferred
            duration_s: Execution time in seconds
            precision: Numerical precision

        Returns:
            Predicted energy in joules
        """
        pj_per_op = self.compute_pj_per_op_by_precision.get(
            precision, self.compute_pj_per_op
        )

        compute_energy = ops * pj_per_op * 1e-12  # pJ to J
        memory_energy = bytes_transferred * self.memory_pj_per_byte * 1e-12
        static_energy = self.static_power_watts * duration_s

        return compute_energy + memory_energy + static_energy

    def to_dict(self) -> Dict[str, Any]:
        return {
            'compute_pj_per_op': self.compute_pj_per_op,
            'memory_pj_per_byte': self.memory_pj_per_byte,
            'static_power_watts': self.static_power_watts,
            'compute_pj_per_op_by_precision': self.compute_pj_per_op_by_precision,
            'memory_pj_per_byte_by_level': self.memory_pj_per_byte_by_level,
            'fit_metrics': self.fit_metrics.to_dict() if self.fit_metrics else None,
            'device_name': self.device_name,
            'precision': self.precision,
            'theoretical_compute_pj_per_op': self.theoretical_compute_pj_per_op,
            'theoretical_memory_pj_per_byte': self.theoretical_memory_pj_per_byte,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EnergyCoefficients':
        data = data.copy()
        if data.get('fit_metrics'):
            data['fit_metrics'] = EnergyFitMetrics.from_dict(data['fit_metrics'])
        return cls(**data)


@dataclass
class EnergyDataPoint:
    """Single data point for energy fitting"""
    power_watts: float  # Measured average power
    gops: float  # Compute rate (giga-ops/second)
    gbps: float  # Memory bandwidth (GB/s)
    precision: str = "fp32"
    source: str = ""  # Benchmark name


class EnergyFitter:
    """
    Fits energy coefficients from benchmark power measurements.

    The fitter uses linear regression on the power equation:
        Power = a * GOPS + b * GB/s + c

    Where:
        a = compute_pj_per_op * 1e-3 (converts pJ/op to W per GOPS)
        b = memory_pj_per_byte * 1e-3 (converts pJ/byte to W per GB/s)
        c = static_power_watts

    Usage:
        fitter = EnergyFitter()

        # Add compute benchmarks (high GOPS, low memory)
        for result in gemm_results:
            fitter.add_compute_result(result, gemm_spec)

        # Add memory benchmarks (low GOPS, high memory)
        for result in memory_results:
            fitter.add_memory_result(result)

        # Add idle measurement
        fitter.add_idle_measurement(idle_power_watts)

        # Fit coefficients
        coefficients = fitter.fit()
    """

    MIN_DATA_POINTS = 3

    def __init__(
        self,
        theoretical_compute_pj_per_op: Optional[float] = None,
        theoretical_memory_pj_per_byte: Optional[float] = None,
    ):
        """
        Initialize the energy fitter.

        Args:
            theoretical_compute_pj_per_op: Theoretical pJ/op from datasheets
            theoretical_memory_pj_per_byte: Theoretical pJ/byte from datasheets
        """
        self.theoretical_compute_pj_per_op = theoretical_compute_pj_per_op
        self.theoretical_memory_pj_per_byte = theoretical_memory_pj_per_byte

        self._data_points: List[EnergyDataPoint] = []
        self._idle_power: Optional[float] = None
        self._device_name: str = ""

    def add_compute_result(
        self,
        result: BenchmarkResult,
        spec: Optional[GEMMSpec] = None,
    ) -> None:
        """
        Add a compute benchmark result with power measurement.

        Args:
            result: BenchmarkResult with power data (avg_power_watts)
            spec: Optional GEMMSpec for calculating ops
        """
        if not result.success:
            return

        if result.avg_power_watts is None or result.avg_power_watts <= 0:
            return

        # Get compute rate (GOPS)
        gops = result.gflops if result.gflops > 0 else 0.0

        # Estimate memory bandwidth if not provided
        gbps = result.bandwidth_gbps if result.bandwidth_gbps > 0 else 0.0

        if gops <= 0:
            return

        self._data_points.append(EnergyDataPoint(
            power_watts=result.avg_power_watts,
            gops=gops,
            gbps=gbps,
            precision=result.precision,
            source=result.spec_name,
        ))

        if result.device_name and not self._device_name:
            self._device_name = result.device_name

    def add_memory_result(self, result: BenchmarkResult) -> None:
        """
        Add a memory benchmark result with power measurement.

        Args:
            result: BenchmarkResult with power data
        """
        if not result.success:
            return

        if result.avg_power_watts is None or result.avg_power_watts <= 0:
            return

        gbps = result.bandwidth_gbps if result.bandwidth_gbps > 0 else 0.0

        if gbps <= 0:
            return

        # Memory benchmarks have minimal compute
        self._data_points.append(EnergyDataPoint(
            power_watts=result.avg_power_watts,
            gops=0.0,
            gbps=gbps,
            precision=result.precision,
            source=result.spec_name,
        ))

        if result.device_name and not self._device_name:
            self._device_name = result.device_name

    def add_idle_measurement(self, idle_power_watts: float) -> None:
        """
        Add idle power measurement.

        Args:
            idle_power_watts: Power consumption when GPU is idle
        """
        self._idle_power = idle_power_watts

    def add_data_point(
        self,
        power_watts: float,
        gops: float,
        gbps: float,
        precision: str = "fp32",
        source: str = "",
    ) -> None:
        """
        Add a raw data point.

        Args:
            power_watts: Measured power
            gops: Compute rate in GOPS
            gbps: Memory bandwidth in GB/s
            precision: Numerical precision
            source: Source identifier
        """
        self._data_points.append(EnergyDataPoint(
            power_watts=power_watts,
            gops=gops,
            gbps=gbps,
            precision=precision,
            source=source,
        ))

    def can_fit(self) -> bool:
        """Check if sufficient data for fitting"""
        return len(self._data_points) >= self.MIN_DATA_POINTS

    def fit(self) -> EnergyCoefficients:
        """
        Fit energy coefficients from collected data.

        Uses ordinary least squares regression:
            power = a * gops + b * gbps + c

        Returns:
            EnergyCoefficients with fitted values

        Raises:
            ValueError: If insufficient data for fitting
        """
        n = len(self._data_points)
        if n < self.MIN_DATA_POINTS:
            raise ValueError(
                f"Insufficient data for fitting: {n} points, "
                f"need >= {self.MIN_DATA_POINTS}"
            )

        # Build design matrix [gops, gbps, 1]
        X = np.zeros((n, 3))
        y = np.zeros(n)

        for i, dp in enumerate(self._data_points):
            X[i, 0] = dp.gops
            X[i, 1] = dp.gbps
            X[i, 2] = 1.0  # Constant term for static power
            y[i] = dp.power_watts

        # Solve least squares: X @ coeffs = y
        try:
            # Use numpy's lstsq for numerical stability
            coeffs, residuals, rank, singular_values = np.linalg.lstsq(X, y, rcond=None)

            a, b, c = coeffs

            # Calculate condition number for stability check
            if len(singular_values) > 0 and singular_values[-1] > 0:
                condition_number = singular_values[0] / singular_values[-1]
            else:
                condition_number = float('inf')

        except np.linalg.LinAlgError as e:
            # Fallback to simpler fitting if full regression fails
            a, b, c, condition_number = self._fallback_fit(X, y)

        # Convert coefficients from regression units to pJ
        # a is in W per GOPS, so pJ/op = a * 1e3
        compute_pj_per_op = max(0.0, a * 1e3)

        # b is in W per GB/s, so pJ/byte = b * 1e3
        memory_pj_per_byte = max(0.0, b * 1e3)

        # c is static power in watts
        static_power = max(0.0, c)

        # Use measured idle power if available
        if self._idle_power is not None:
            static_power = self._idle_power

        # Calculate fit quality
        predictions = X @ np.array([a, b, c])
        residuals = y - predictions
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        residual_std = float(np.std(residuals))

        quality = self._assess_fit_quality(n, r_squared)

        fit_metrics = EnergyFitMetrics(
            num_data_points=n,
            r_squared=max(0.0, r_squared),
            residual_std=residual_std,
            quality=quality,
            condition_number=condition_number,
        )

        # Group by precision for per-precision coefficients
        compute_by_precision = self._fit_per_precision()

        return EnergyCoefficients(
            compute_pj_per_op=compute_pj_per_op,
            memory_pj_per_byte=memory_pj_per_byte,
            static_power_watts=static_power,
            compute_pj_per_op_by_precision=compute_by_precision,
            fit_metrics=fit_metrics,
            device_name=self._device_name,
            theoretical_compute_pj_per_op=self.theoretical_compute_pj_per_op,
            theoretical_memory_pj_per_byte=self.theoretical_memory_pj_per_byte,
        )

    def _fallback_fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[float, float, float, float]:
        """
        Fallback fitting when full regression fails.

        Uses simpler approach: fit memory-only and compute-only separately.
        """
        # Separate compute-heavy and memory-heavy points
        compute_mask = X[:, 0] > X[:, 1]  # gops > gbps
        memory_mask = ~compute_mask

        # Static power: use minimum observed or idle
        c = np.min(y) * 0.3 if self._idle_power is None else self._idle_power

        # Memory coefficient from memory-heavy points
        if np.sum(memory_mask) >= 1:
            mem_points = X[memory_mask]
            mem_power = y[memory_mask]
            b = np.mean((mem_power - c) / (mem_points[:, 1] + 0.1))
        else:
            b = 0.0

        # Compute coefficient from compute-heavy points
        if np.sum(compute_mask) >= 1:
            comp_points = X[compute_mask]
            comp_power = y[compute_mask]
            a = np.mean((comp_power - c - b * comp_points[:, 1]) / (comp_points[:, 0] + 0.1))
        else:
            a = 0.0

        return float(a), float(b), float(c), float('inf')

    def _fit_per_precision(self) -> Dict[str, float]:
        """Fit compute coefficients per precision"""
        precisions: Dict[str, List[EnergyDataPoint]] = {}

        for dp in self._data_points:
            if dp.gops > 0:  # Only compute-heavy points
                if dp.precision not in precisions:
                    precisions[dp.precision] = []
                precisions[dp.precision].append(dp)

        result = {}
        for precision, points in precisions.items():
            if len(points) >= 2:
                # Simple average: (power - estimated_static) / gops
                avg_power = np.mean([p.power_watts for p in points])
                avg_gops = np.mean([p.gops for p in points])
                static_est = self._idle_power if self._idle_power else avg_power * 0.2

                if avg_gops > 0:
                    # pJ/op = (W - static) / GOPS * 1e3
                    pj_per_op = (avg_power - static_est) / avg_gops * 1e3
                    result[precision] = max(0.0, pj_per_op)

        return result

    def _assess_fit_quality(self, n: int, r_squared: float) -> EnergyFitQuality:
        """Assess fit quality from number of points and R-squared"""
        if n < 3:
            return EnergyFitQuality.INSUFFICIENT

        if r_squared >= 0.95 and n >= 10:
            return EnergyFitQuality.EXCELLENT
        elif r_squared >= 0.85 and n >= 5:
            return EnergyFitQuality.GOOD
        elif r_squared >= 0.70 and n >= 3:
            return EnergyFitQuality.FAIR
        else:
            return EnergyFitQuality.POOR

    def reset(self) -> None:
        """Clear all stored data"""
        self._data_points.clear()
        self._idle_power = None
        self._device_name = ""


def fit_energy_model(
    results: List[BenchmarkResult],
    idle_power_watts: Optional[float] = None,
    theoretical_compute_pj_per_op: Optional[float] = None,
    theoretical_memory_pj_per_byte: Optional[float] = None,
) -> EnergyCoefficients:
    """
    Main fitting function - fits energy coefficients from benchmark results.

    This is the primary interface for energy coefficient fitting.

    Args:
        results: List of BenchmarkResult with power measurements
        idle_power_watts: Optional measured idle power
        theoretical_compute_pj_per_op: Theoretical pJ/op for comparison
        theoretical_memory_pj_per_byte: Theoretical pJ/byte for comparison

    Returns:
        EnergyCoefficients with fitted values

    Example:
        >>> results = load_benchmark_results("results/")
        >>> coeffs = fit_energy_model(results, idle_power_watts=30.0)
        >>> print(f"Compute: {coeffs.compute_pj_per_op:.1f} pJ/op")
        >>> print(f"Memory: {coeffs.memory_pj_per_byte:.1f} pJ/byte")
    """
    fitter = EnergyFitter(
        theoretical_compute_pj_per_op=theoretical_compute_pj_per_op,
        theoretical_memory_pj_per_byte=theoretical_memory_pj_per_byte,
    )

    if idle_power_watts is not None:
        fitter.add_idle_measurement(idle_power_watts)

    for result in results:
        if not result.success:
            continue

        # Determine if compute-heavy or memory-heavy based on result
        is_memory = (
            'memory' in result.spec_name.lower() or
            'stream' in result.spec_name.lower() or
            (result.gflops == 0 and result.bandwidth_gbps > 0)
        )

        if is_memory:
            fitter.add_memory_result(result)
        else:
            fitter.add_compute_result(result)

    return fitter.fit()
