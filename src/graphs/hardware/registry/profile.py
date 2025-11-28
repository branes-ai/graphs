"""
Unified Hardware Profile

Combines hardware specification (theoretical) with calibration data (measured)
into a single coherent view of hardware capabilities.

Calibration files are stored in a calibrations/ subdirectory with naming:
    {power_mode}_{frequency_mhz}MHz_{framework}.json

Examples:
    calibrations/MAXN_625MHz_pytorch.json
    calibrations/7W_306MHz_pytorch.json
    calibrations/performance_4900MHz_numpy.json
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import json
import re

from ..calibration.schema import (
    HardwareCalibration,
    CalibrationMetadata,
    CPUClockData,
    GPUClockData,
    PrecisionCapabilityMatrix,
)


def _make_calibration_prefix(calibration: HardwareCalibration) -> str:
    """
    Generate calibration filename prefix from calibration metadata.

    Format: {power_mode}_{frequency_mhz}MHz_{framework}

    Examples:
        MAXN_625MHz_pytorch
        7W_306MHz_pytorch
        performance_4900MHz_numpy
    """
    metadata = calibration.metadata

    # Get power mode
    if metadata.gpu_clock and metadata.gpu_clock.power_mode_name:
        power_mode = metadata.gpu_clock.power_mode_name
    elif metadata.cpu_clock and metadata.cpu_clock.governor:
        power_mode = metadata.cpu_clock.governor
    else:
        power_mode = "unknown"

    # Get frequency (GPU SM clock or CPU frequency)
    if metadata.device_type == 'cuda' and metadata.gpu_clock and metadata.gpu_clock.sm_clock_mhz:
        freq_mhz = metadata.gpu_clock.sm_clock_mhz
    elif metadata.cpu_clock and metadata.cpu_clock.current_freq_mhz:
        freq_mhz = int(metadata.cpu_clock.current_freq_mhz)
    else:
        freq_mhz = 0

    # Get framework
    framework = metadata.framework or "unknown"

    # Sanitize power mode (replace spaces, special chars)
    power_mode = re.sub(r'[^a-zA-Z0-9]', '', power_mode)

    return f"{power_mode}_{freq_mhz}MHz_{framework}"


def _make_calibration_filename(calibration: HardwareCalibration) -> str:
    """
    Generate calibration filename from calibration metadata.

    Format: {power_mode}_{frequency_mhz}MHz_{framework}.json
    """
    return f"{_make_calibration_prefix(calibration)}.json"


def _parse_calibration_filename(filename: str) -> Optional[Dict[str, Any]]:
    """
    Parse calibration filename to extract metadata.

    Returns dict with power_mode, freq_mhz, framework or None if invalid.
    """
    match = re.match(r'^([^_]+)_(\d+)MHz_([^.]+)\.json$', filename)
    if match:
        return {
            'power_mode': match.group(1),
            'freq_mhz': int(match.group(2)),
            'framework': match.group(3),
        }
    return None


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

    calibration_log: Optional[str] = None
    """Log output from calibration run (saved alongside JSON)."""

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
        - calibrations/{power_mode}_{freq}MHz_{framework}.json: Calibration data (if available)
        - calibrations/{power_mode}_{freq}MHz_{framework}.log: Calibration log (if available)

        Args:
            directory: Directory to save to (will be created if needed)
        """
        directory.mkdir(parents=True, exist_ok=True)

        # Save specification
        spec_path = directory / 'spec.json'
        with open(spec_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

        # Save calibration to calibrations/ subdirectory if available
        if self.calibration:
            cal_dir = directory / 'calibrations'
            cal_dir.mkdir(parents=True, exist_ok=True)

            # Save calibration JSON
            cal_prefix = _make_calibration_prefix(self.calibration)
            cal_path = cal_dir / f"{cal_prefix}.json"
            self.calibration.save(cal_path)

            # Save calibration log if available
            if self.calibration_log:
                log_path = cal_dir / f"{cal_prefix}.log"
                with open(log_path, 'w') as f:
                    f.write(self.calibration_log)

    @classmethod
    def load(cls, directory: Path, calibration_filter: Optional[Dict[str, Any]] = None) -> 'HardwareProfile':
        """
        Load profile from a directory.

        Args:
            directory: Directory containing spec.json and optionally calibrations/
            calibration_filter: Optional filter to select specific calibration.
                               Keys: 'power_mode', 'freq_mhz', 'framework'
                               If None, loads the most recent calibration.

        Returns:
            HardwareProfile instance
        """
        # Load specification
        spec_path = directory / 'spec.json'
        if not spec_path.exists():
            raise FileNotFoundError(f"No spec.json found in {directory}")

        with open(spec_path) as f:
            spec_data = json.load(f)

        # Load calibration from calibrations/ subdirectory
        calibration = None
        cal_dir = directory / 'calibrations'
        if cal_dir.exists() and cal_dir.is_dir():
            calibration = cls._load_calibration(cal_dir, calibration_filter)
            if calibration:
                spec_data['calibration_date'] = calibration.metadata.calibration_date

        return cls.from_dict(spec_data, calibration)

    @classmethod
    def _load_calibration(
        cls,
        cal_dir: Path,
        calibration_filter: Optional[Dict[str, Any]] = None
    ) -> Optional[HardwareCalibration]:
        """
        Load a calibration from the calibrations/ directory.

        Args:
            cal_dir: Path to calibrations/ directory
            calibration_filter: Optional filter (power_mode, freq_mhz, framework)

        Returns:
            HardwareCalibration or None if no matching calibration found
        """
        # Find all calibration files
        cal_files = list(cal_dir.glob('*.json'))
        if not cal_files:
            return None

        # Parse all filenames
        candidates = []
        for cal_file in cal_files:
            parsed = _parse_calibration_filename(cal_file.name)
            if parsed:
                parsed['path'] = cal_file
                parsed['mtime'] = cal_file.stat().st_mtime
                candidates.append(parsed)

        if not candidates:
            return None

        # Apply filter if provided
        if calibration_filter:
            filtered = []
            for c in candidates:
                match = True
                if 'power_mode' in calibration_filter:
                    if c['power_mode'].lower() != calibration_filter['power_mode'].lower():
                        match = False
                if 'freq_mhz' in calibration_filter:
                    if c['freq_mhz'] != calibration_filter['freq_mhz']:
                        match = False
                if 'framework' in calibration_filter:
                    if c['framework'].lower() != calibration_filter['framework'].lower():
                        match = False
                if match:
                    filtered.append(c)
            candidates = filtered

        if not candidates:
            return None

        # Select most recent calibration
        candidates.sort(key=lambda c: c['mtime'], reverse=True)
        best = candidates[0]

        return HardwareCalibration.load(best['path'])

    @classmethod
    def list_calibrations(cls, directory: Path) -> List[Dict[str, Any]]:
        """
        List all available calibrations for a profile directory.

        Args:
            directory: Profile directory containing calibrations/ subdirectory

        Returns:
            List of dicts with calibration info (power_mode, freq_mhz, framework, path)
        """
        cal_dir = directory / 'calibrations'
        if not cal_dir.exists():
            return []

        results = []
        for cal_file in cal_dir.glob('*.json'):
            parsed = _parse_calibration_filename(cal_file.name)
            if parsed:
                parsed['path'] = str(cal_file)
                results.append(parsed)

        # Sort by power mode, then frequency
        results.sort(key=lambda x: (x['power_mode'], x['freq_mhz']))
        return results

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
