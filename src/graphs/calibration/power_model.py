"""
Calibrated Power Model

Provides a three-component power model using calibrated coefficients:
1. Compute power: energy_per_op * operations_per_second
2. Memory power: energy_per_byte * bytes_per_second
3. Static power: idle power (leakage, always-on circuits)

This model can use either:
- Calibrated coefficients from EnergyFitter (accurate)
- Theoretical coefficients from datasheets (fallback)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum

from graphs.calibration.energy_fitter import EnergyCoefficients


class PowerSource(Enum):
    """Source of power model coefficients"""
    CALIBRATED = "calibrated"    # From actual measurements
    THEORETICAL = "theoretical"  # From datasheets/estimates
    INTERPOLATED = "interpolated"  # Interpolated from similar hardware
    UNKNOWN = "unknown"


@dataclass
class PowerBreakdown:
    """Breakdown of power consumption by component"""
    compute_watts: float
    memory_watts: float
    static_watts: float
    total_watts: float

    # Percentages
    compute_percent: float = 0.0
    memory_percent: float = 0.0
    static_percent: float = 0.0

    def __post_init__(self):
        if self.total_watts > 0:
            self.compute_percent = self.compute_watts / self.total_watts * 100
            self.memory_percent = self.memory_watts / self.total_watts * 100
            self.static_percent = self.static_watts / self.total_watts * 100

    def to_dict(self) -> Dict[str, Any]:
        return {
            'compute_watts': self.compute_watts,
            'memory_watts': self.memory_watts,
            'static_watts': self.static_watts,
            'total_watts': self.total_watts,
            'compute_percent': self.compute_percent,
            'memory_percent': self.memory_percent,
            'static_percent': self.static_percent,
        }


@dataclass
class EnergyBreakdown:
    """Breakdown of energy consumption by component"""
    compute_joules: float
    memory_joules: float
    static_joules: float
    total_joules: float
    duration_seconds: float

    # Percentages
    compute_percent: float = 0.0
    memory_percent: float = 0.0
    static_percent: float = 0.0

    # Convenience conversions
    total_millijoules: float = 0.0
    total_microjoules: float = 0.0

    def __post_init__(self):
        if self.total_joules > 0:
            self.compute_percent = self.compute_joules / self.total_joules * 100
            self.memory_percent = self.memory_joules / self.total_joules * 100
            self.static_percent = self.static_joules / self.total_joules * 100

        self.total_millijoules = self.total_joules * 1e3
        self.total_microjoules = self.total_joules * 1e6

    def to_dict(self) -> Dict[str, Any]:
        return {
            'compute_joules': self.compute_joules,
            'memory_joules': self.memory_joules,
            'static_joules': self.static_joules,
            'total_joules': self.total_joules,
            'duration_seconds': self.duration_seconds,
            'compute_percent': self.compute_percent,
            'memory_percent': self.memory_percent,
            'static_percent': self.static_percent,
            'total_millijoules': self.total_millijoules,
            'total_microjoules': self.total_microjoules,
        }


class CalibratedPowerModel:
    """
    Three-component power model with calibrated or theoretical coefficients.

    Power Model:
        P_total = P_compute + P_memory + P_static

    Where:
        P_compute = ops_per_sec * pJ_per_op * 1e-12  (pJ to W conversion)
        P_memory = bytes_per_sec * pJ_per_byte * 1e-12
        P_static = idle_power_watts (constant)

    Energy Model:
        E_total = E_compute + E_memory + E_static

    Where:
        E_compute = total_ops * pJ_per_op * 1e-12  (pJ to J)
        E_memory = total_bytes * pJ_per_byte * 1e-12
        E_static = idle_power * duration

    Usage:
        # From calibrated coefficients
        model = CalibratedPowerModel.from_coefficients(coefficients)

        # From theoretical values
        model = CalibratedPowerModel.from_theoretical(
            tdp_watts=300,
            peak_gflops=50000,
            peak_bandwidth_gbps=2000,
        )

        # Predict power
        power = model.predict_power(gops=40000, gbps=1500)

        # Predict energy
        energy = model.predict_energy(ops=1e12, bytes=1e9, duration=0.025)
    """

    def __init__(
        self,
        compute_pj_per_op: float,
        memory_pj_per_byte: float,
        static_power_watts: float,
        source: PowerSource = PowerSource.UNKNOWN,
        device_name: str = "",
        compute_pj_per_op_by_precision: Optional[Dict[str, float]] = None,
        memory_pj_per_byte_by_level: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize power model with coefficients.

        Args:
            compute_pj_per_op: Energy per operation in picojoules
            memory_pj_per_byte: Energy per byte transferred in picojoules
            static_power_watts: Idle/leakage power in watts
            source: Source of coefficients (calibrated, theoretical, etc.)
            device_name: Name of hardware device
            compute_pj_per_op_by_precision: Per-precision compute coefficients
            memory_pj_per_byte_by_level: Per-memory-level coefficients
        """
        self.compute_pj_per_op = compute_pj_per_op
        self.memory_pj_per_byte = memory_pj_per_byte
        self.static_power_watts = static_power_watts
        self.source = source
        self.device_name = device_name

        self.compute_pj_per_op_by_precision = compute_pj_per_op_by_precision or {}
        self.memory_pj_per_byte_by_level = memory_pj_per_byte_by_level or {}

    @classmethod
    def from_coefficients(cls, coefficients: EnergyCoefficients) -> 'CalibratedPowerModel':
        """
        Create power model from calibrated coefficients.

        Args:
            coefficients: EnergyCoefficients from EnergyFitter

        Returns:
            CalibratedPowerModel using calibrated values
        """
        return cls(
            compute_pj_per_op=coefficients.compute_pj_per_op,
            memory_pj_per_byte=coefficients.memory_pj_per_byte,
            static_power_watts=coefficients.static_power_watts,
            source=PowerSource.CALIBRATED,
            device_name=coefficients.device_name,
            compute_pj_per_op_by_precision=coefficients.compute_pj_per_op_by_precision,
            memory_pj_per_byte_by_level=coefficients.memory_pj_per_byte_by_level,
        )

    @classmethod
    def from_theoretical(
        cls,
        tdp_watts: float,
        peak_gflops: float,
        peak_bandwidth_gbps: float,
        idle_power_fraction: float = 0.3,
        compute_fraction: float = 0.5,
        memory_fraction: float = 0.2,
        device_name: str = "",
    ) -> 'CalibratedPowerModel':
        """
        Create power model from theoretical hardware specifications.

        Estimates coefficients based on TDP and peak performance:
        - Assumes power budget splits between compute, memory, and static
        - Calculates pJ/op and pJ/byte from these fractions

        Args:
            tdp_watts: Thermal design power
            peak_gflops: Peak GFLOPS (FP32)
            peak_bandwidth_gbps: Peak memory bandwidth in GB/s
            idle_power_fraction: Fraction of TDP that is idle/static (default 0.3)
            compute_fraction: Fraction of dynamic power for compute (default 0.5)
            memory_fraction: Fraction of dynamic power for memory (default 0.2)
            device_name: Name of hardware

        Returns:
            CalibratedPowerModel with theoretical coefficients
        """
        # Static power
        static_power = tdp_watts * idle_power_fraction

        # Dynamic power budget
        dynamic_power = tdp_watts - static_power

        # Compute power budget and pJ/op
        compute_power = dynamic_power * compute_fraction
        # Power = GFLOPS * pJ/op * 1e-3, so pJ/op = Power / GFLOPS * 1e3
        compute_pj_per_op = (compute_power / peak_gflops * 1e3) if peak_gflops > 0 else 0.0

        # Memory power budget and pJ/byte
        memory_power = dynamic_power * memory_fraction
        # Power = GB/s * pJ/byte * 1e-3, so pJ/byte = Power / GB/s * 1e3
        memory_pj_per_byte = (memory_power / peak_bandwidth_gbps * 1e3) if peak_bandwidth_gbps > 0 else 0.0

        return cls(
            compute_pj_per_op=compute_pj_per_op,
            memory_pj_per_byte=memory_pj_per_byte,
            static_power_watts=static_power,
            source=PowerSource.THEORETICAL,
            device_name=device_name,
        )

    def get_compute_pj_per_op(self, precision: str = "fp32") -> float:
        """Get compute coefficient for specific precision"""
        return self.compute_pj_per_op_by_precision.get(
            precision, self.compute_pj_per_op
        )

    def get_memory_pj_per_byte(self, memory_level: str = "dram") -> float:
        """Get memory coefficient for specific memory level"""
        return self.memory_pj_per_byte_by_level.get(
            memory_level, self.memory_pj_per_byte
        )

    def predict_power(
        self,
        gops: float,
        gbps: float,
        precision: str = "fp32",
        memory_level: str = "dram",
    ) -> float:
        """
        Predict total power consumption.

        Args:
            gops: Compute throughput in GOPS (giga-ops/second)
            gbps: Memory bandwidth in GB/s
            precision: Numerical precision
            memory_level: Memory hierarchy level

        Returns:
            Total power in watts
        """
        pj_per_op = self.get_compute_pj_per_op(precision)
        pj_per_byte = self.get_memory_pj_per_byte(memory_level)

        # Power = rate * pJ/unit * 1e-3 (converts pJ rate to W)
        compute_power = gops * pj_per_op * 1e-3
        memory_power = gbps * pj_per_byte * 1e-3

        return compute_power + memory_power + self.static_power_watts

    def predict_power_breakdown(
        self,
        gops: float,
        gbps: float,
        precision: str = "fp32",
        memory_level: str = "dram",
    ) -> PowerBreakdown:
        """
        Predict power with component breakdown.

        Args:
            gops: Compute throughput in GOPS
            gbps: Memory bandwidth in GB/s
            precision: Numerical precision
            memory_level: Memory hierarchy level

        Returns:
            PowerBreakdown with component details
        """
        pj_per_op = self.get_compute_pj_per_op(precision)
        pj_per_byte = self.get_memory_pj_per_byte(memory_level)

        compute_power = gops * pj_per_op * 1e-3
        memory_power = gbps * pj_per_byte * 1e-3
        total_power = compute_power + memory_power + self.static_power_watts

        return PowerBreakdown(
            compute_watts=compute_power,
            memory_watts=memory_power,
            static_watts=self.static_power_watts,
            total_watts=total_power,
        )

    def predict_energy(
        self,
        ops: int,
        bytes_transferred: int,
        duration_seconds: float,
        precision: str = "fp32",
        memory_level: str = "dram",
    ) -> float:
        """
        Predict total energy consumption.

        Args:
            ops: Total operations (FLOPs or integer ops)
            bytes_transferred: Total bytes transferred
            duration_seconds: Execution time in seconds
            precision: Numerical precision
            memory_level: Memory hierarchy level

        Returns:
            Total energy in joules
        """
        pj_per_op = self.get_compute_pj_per_op(precision)
        pj_per_byte = self.get_memory_pj_per_byte(memory_level)

        # Energy = count * pJ/unit * 1e-12 (converts pJ to J)
        compute_energy = ops * pj_per_op * 1e-12
        memory_energy = bytes_transferred * pj_per_byte * 1e-12
        static_energy = self.static_power_watts * duration_seconds

        return compute_energy + memory_energy + static_energy

    def predict_energy_breakdown(
        self,
        ops: int,
        bytes_transferred: int,
        duration_seconds: float,
        precision: str = "fp32",
        memory_level: str = "dram",
    ) -> EnergyBreakdown:
        """
        Predict energy with component breakdown.

        Args:
            ops: Total operations
            bytes_transferred: Total bytes transferred
            duration_seconds: Execution time
            precision: Numerical precision
            memory_level: Memory hierarchy level

        Returns:
            EnergyBreakdown with component details
        """
        pj_per_op = self.get_compute_pj_per_op(precision)
        pj_per_byte = self.get_memory_pj_per_byte(memory_level)

        compute_energy = ops * pj_per_op * 1e-12
        memory_energy = bytes_transferred * pj_per_byte * 1e-12
        static_energy = self.static_power_watts * duration_seconds
        total_energy = compute_energy + memory_energy + static_energy

        return EnergyBreakdown(
            compute_joules=compute_energy,
            memory_joules=memory_energy,
            static_joules=static_energy,
            total_joules=total_energy,
            duration_seconds=duration_seconds,
        )

    def estimate_latency_from_energy_budget(
        self,
        energy_budget_joules: float,
        ops: int,
        bytes_transferred: int,
        precision: str = "fp32",
    ) -> float:
        """
        Estimate maximum duration given an energy budget.

        Useful for energy-constrained scenarios.

        Args:
            energy_budget_joules: Maximum energy allowed
            ops: Total operations to perform
            bytes_transferred: Total bytes to transfer
            precision: Numerical precision

        Returns:
            Maximum duration in seconds that fits within budget
        """
        pj_per_op = self.get_compute_pj_per_op(precision)

        # Dynamic energy (fixed, independent of time)
        dynamic_energy = (
            ops * pj_per_op * 1e-12 +
            bytes_transferred * self.memory_pj_per_byte * 1e-12
        )

        # Remaining budget for static energy
        static_budget = energy_budget_joules - dynamic_energy

        if static_budget <= 0:
            return 0.0  # Not achievable

        if self.static_power_watts <= 0:
            return float('inf')  # No static power constraint

        # max_time = static_budget / static_power
        return static_budget / self.static_power_watts

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary"""
        return {
            'compute_pj_per_op': self.compute_pj_per_op,
            'memory_pj_per_byte': self.memory_pj_per_byte,
            'static_power_watts': self.static_power_watts,
            'source': self.source.value,
            'device_name': self.device_name,
            'compute_pj_per_op_by_precision': self.compute_pj_per_op_by_precision,
            'memory_pj_per_byte_by_level': self.memory_pj_per_byte_by_level,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CalibratedPowerModel':
        """Deserialize from dictionary"""
        return cls(
            compute_pj_per_op=data['compute_pj_per_op'],
            memory_pj_per_byte=data['memory_pj_per_byte'],
            static_power_watts=data['static_power_watts'],
            source=PowerSource(data.get('source', 'unknown')),
            device_name=data.get('device_name', ''),
            compute_pj_per_op_by_precision=data.get('compute_pj_per_op_by_precision'),
            memory_pj_per_byte_by_level=data.get('memory_pj_per_byte_by_level'),
        )

    def __repr__(self) -> str:
        return (
            f"CalibratedPowerModel("
            f"compute={self.compute_pj_per_op:.2f} pJ/op, "
            f"memory={self.memory_pj_per_byte:.2f} pJ/byte, "
            f"static={self.static_power_watts:.1f} W, "
            f"source={self.source.value})"
        )
