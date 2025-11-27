"""
Unified Hardware Profile

Combines hardware specification (theoretical) with calibration data (measured)
into a single coherent view of hardware capabilities.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from pathlib import Path
import json

from ..calibration.schema import (
    HardwareCalibration,
    CalibrationMetadata,
    CPUClockData,
    GPUClockData,
    PrecisionCapabilityMatrix,
)


@dataclass
class HardwareProfile:
    """
    Unified view of hardware specification and calibration.

    This combines:
    - Hardware specification (theoretical peaks, architecture info)
    - Calibration data (measured performance, if available)

    The profile provides a single source of truth for hardware capabilities,
    with calibration data taking precedence over theoretical specs when available.
    """

    # Identity
    id: str
    """Unique identifier (e.g., 'i7_12700k', 'h100_sxm5')."""

    # Basic info
    vendor: str
    """Hardware vendor (e.g., 'Intel', 'NVIDIA')."""

    model: str
    """Model name (e.g., 'Core i7-12700K', 'H100 SXM5')."""

    device_type: str
    """Device type: 'cpu', 'gpu', 'tpu', etc."""

    # Theoretical specifications
    theoretical_peaks: Dict[str, float] = field(default_factory=dict)
    """Theoretical peak GFLOPS/GIOPS by precision (e.g., {'fp32': 1000, 'fp16': 2000})."""

    peak_bandwidth_gbps: float = 0.0
    """Theoretical peak memory bandwidth in GB/s."""

    # Architecture details
    architecture: Optional[str] = None
    """Architecture name (e.g., 'Alder Lake', 'Hopper')."""

    compute_units: Optional[int] = None
    """Number of compute units (cores for CPU, SMs for GPU)."""

    memory_gb: Optional[float] = None
    """Memory capacity in GB."""

    tdp_watts: Optional[float] = None
    """Thermal design power in Watts."""

    # Clock frequencies (from spec, may be updated by calibration)
    base_clock_mhz: Optional[float] = None
    """Base clock frequency in MHz."""

    boost_clock_mhz: Optional[float] = None
    """Boost/turbo clock frequency in MHz."""

    # Calibration data (populated when calibration exists)
    calibration: Optional[HardwareCalibration] = None
    """Measured performance data from calibration run."""

    calibration_date: Optional[str] = None
    """When the calibration was performed (ISO format)."""

    # Additional metadata
    platform: Optional[str] = None
    """Platform identifier (e.g., 'x86_64', 'aarch64')."""

    tags: List[str] = field(default_factory=list)
    """Tags for categorization (e.g., ['datacenter', 'inference'])."""

    notes: Optional[str] = None
    """Additional notes about this hardware."""

    @property
    def is_calibrated(self) -> bool:
        """Whether this profile has calibration data."""
        return self.calibration is not None

    @property
    def effective_peak_gflops(self) -> float:
        """
        Get the effective peak GFLOPS (FP32).

        Returns calibrated peak if available, otherwise theoretical.
        """
        if self.calibration and self.calibration.best_measured_gflops > 0:
            return self.calibration.best_measured_gflops
        return self.theoretical_peaks.get('fp32', 0.0)

    @property
    def effective_bandwidth_gbps(self) -> float:
        """
        Get the effective memory bandwidth.

        Returns calibrated bandwidth if available, otherwise theoretical.
        """
        if self.calibration and self.calibration.measured_bandwidth_gbps > 0:
            return self.calibration.measured_bandwidth_gbps
        return self.peak_bandwidth_gbps

    @property
    def cpu_clock(self) -> Optional[CPUClockData]:
        """Get CPU clock data from calibration."""
        if self.calibration and self.calibration.metadata:
            return self.calibration.metadata.cpu_clock
        return None

    @property
    def gpu_clock(self) -> Optional[GPUClockData]:
        """Get GPU clock data from calibration."""
        if self.calibration and self.calibration.metadata:
            return self.calibration.metadata.gpu_clock
        return None

    @property
    def precision_support(self) -> Optional[PrecisionCapabilityMatrix]:
        """Get precision support matrix from calibration."""
        if self.calibration:
            return self.calibration.precision_matrix
        return None

    def get_peak(self, precision: str = 'fp32') -> float:
        """
        Get peak performance for a specific precision.

        Args:
            precision: Precision name (e.g., 'fp32', 'fp16', 'int8')

        Returns:
            Peak GFLOPS/GIOPS for the precision, or 0.0 if unknown
        """
        # Check calibrated peaks first
        if self.calibration and self.calibration.precision_matrix:
            measured = self.calibration.precision_matrix.peak_gflops_by_precision.get(precision)
            if measured:
                return measured

        # Fall back to theoretical
        return self.theoretical_peaks.get(precision, 0.0)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            'id': self.id,
            'vendor': self.vendor,
            'model': self.model,
            'device_type': self.device_type,
            'theoretical_peaks': self.theoretical_peaks,
            'peak_bandwidth_gbps': self.peak_bandwidth_gbps,
        }

        # Add optional fields if set
        if self.architecture:
            result['architecture'] = self.architecture
        if self.compute_units:
            result['compute_units'] = self.compute_units
        if self.memory_gb:
            result['memory_gb'] = self.memory_gb
        if self.tdp_watts:
            result['tdp_watts'] = self.tdp_watts
        if self.base_clock_mhz:
            result['base_clock_mhz'] = self.base_clock_mhz
        if self.boost_clock_mhz:
            result['boost_clock_mhz'] = self.boost_clock_mhz
        if self.platform:
            result['platform'] = self.platform
        if self.tags:
            result['tags'] = self.tags
        if self.notes:
            result['notes'] = self.notes

        # Calibration is stored separately, not inlined
        if self.calibration_date:
            result['calibration_date'] = self.calibration_date

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any], calibration: Optional[HardwareCalibration] = None) -> 'HardwareProfile':
        """
        Create from dictionary.

        Args:
            data: Specification data
            calibration: Optional calibration data to attach
        """
        return cls(
            id=data['id'],
            vendor=data.get('vendor', 'Unknown'),
            model=data.get('model', data['id']),
            device_type=data.get('device_type', 'cpu'),
            theoretical_peaks=data.get('theoretical_peaks', {}),
            peak_bandwidth_gbps=data.get('peak_bandwidth_gbps', 0.0),
            architecture=data.get('architecture'),
            compute_units=data.get('compute_units'),
            memory_gb=data.get('memory_gb'),
            tdp_watts=data.get('tdp_watts'),
            base_clock_mhz=data.get('base_clock_mhz'),
            boost_clock_mhz=data.get('boost_clock_mhz'),
            calibration=calibration,
            calibration_date=data.get('calibration_date'),
            platform=data.get('platform'),
            tags=data.get('tags', []),
            notes=data.get('notes'),
        )

    def save(self, directory: Path):
        """
        Save profile to a directory.

        Creates:
        - spec.json: Hardware specification
        - calibration.json: Calibration data (if available)

        Args:
            directory: Directory to save to (will be created if needed)
        """
        directory.mkdir(parents=True, exist_ok=True)

        # Save specification
        spec_path = directory / 'spec.json'
        with open(spec_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

        # Save calibration if available
        if self.calibration:
            cal_path = directory / 'calibration.json'
            self.calibration.save(cal_path)

    @classmethod
    def load(cls, directory: Path) -> 'HardwareProfile':
        """
        Load profile from a directory.

        Args:
            directory: Directory containing spec.json and optionally calibration.json

        Returns:
            HardwareProfile instance
        """
        # Load specification
        spec_path = directory / 'spec.json'
        if not spec_path.exists():
            raise FileNotFoundError(f"No spec.json found in {directory}")

        with open(spec_path) as f:
            spec_data = json.load(f)

        # Load calibration if available
        calibration = None
        cal_path = directory / 'calibration.json'
        if cal_path.exists():
            calibration = HardwareCalibration.load(cal_path)
            spec_data['calibration_date'] = calibration.metadata.calibration_date

        return cls.from_dict(spec_data, calibration)

    def __str__(self) -> str:
        """Human-readable string representation."""
        status = "calibrated" if self.is_calibrated else "uncalibrated"
        return f"{self.vendor} {self.model} ({self.device_type}, {status})"

    def print_summary(self):
        """Print a human-readable summary of the profile."""
        print(f"\n{'='*60}")
        print(f"Hardware Profile: {self.id}")
        print(f"{'='*60}")
        print(f"  Vendor:      {self.vendor}")
        print(f"  Model:       {self.model}")
        print(f"  Type:        {self.device_type.upper()}")

        if self.architecture:
            print(f"  Architecture: {self.architecture}")
        if self.compute_units:
            unit_name = "SMs" if self.device_type == 'gpu' else "cores"
            print(f"  Compute:     {self.compute_units} {unit_name}")
        if self.memory_gb:
            print(f"  Memory:      {self.memory_gb} GB")

        print()
        print("  Theoretical Peaks:")
        for prec, gops in sorted(self.theoretical_peaks.items()):
            unit = "GIOPS" if prec.startswith('int') else "GFLOPS"
            print(f"    {prec:8s}: {gops:,.0f} {unit}")
        print(f"    bandwidth: {self.peak_bandwidth_gbps:,.0f} GB/s")

        if self.is_calibrated:
            print()
            print(f"  Calibrated: {self.calibration_date}")
            print(f"    Measured FP32: {self.calibration.best_measured_gflops:,.1f} GFLOPS")
            print(f"    Measured BW:   {self.calibration.measured_bandwidth_gbps:,.1f} GB/s")

            if self.cpu_clock:
                print(f"    CPU Clock:     {self.cpu_clock.current_freq_mhz:.0f} MHz")
                if self.cpu_clock.governor:
                    print(f"    Governor:      {self.cpu_clock.governor}")

            if self.gpu_clock:
                print(f"    GPU Clock:     {self.gpu_clock.sm_clock_mhz} MHz")
                if self.gpu_clock.power_mode_name:
                    print(f"    Power Mode:    {self.gpu_clock.power_mode_name}")
        else:
            print()
            print("  Status: Not calibrated")
            print("  Run: cli/calibrate_hardware.py to calibrate")

        print()
