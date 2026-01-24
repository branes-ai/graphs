"""
Measurement Collectors for Benchmarks

Collectors gather additional metrics during benchmark execution:
- TimingCollector: High-precision timing (built into runner)
- PowerCollector: GPU power measurement via nvidia-smi/NVML
- MemoryCollector: Memory usage tracking

Usage:
    collector = PowerCollector()
    collector.start()
    # ... run benchmark ...
    collector.stop()
    measurement = collector.get_measurement()
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import threading
import time


@dataclass
class Measurement:
    """Base class for measurement results"""
    duration_ms: float
    collector_name: str = ""
    success: bool = True
    error_message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            'collector_name': self.collector_name,
            'duration_ms': self.duration_ms,
            'success': self.success,
            'error_message': self.error_message,
        }


@dataclass
class PowerMeasurement(Measurement):
    """Power measurement results"""
    samples: List[float] = field(default_factory=list)  # Power samples in watts
    avg_power_watts: float = 0.0
    peak_power_watts: float = 0.0
    min_power_watts: float = 0.0
    energy_joules: float = 0.0
    sampling_interval_ms: float = 100.0

    def __post_init__(self):
        self.collector_name = "power"
        if self.samples:
            self.avg_power_watts = sum(self.samples) / len(self.samples)
            self.peak_power_watts = max(self.samples)
            self.min_power_watts = min(self.samples)
            # Energy = average power * time
            self.energy_joules = self.avg_power_watts * (self.duration_ms / 1000)

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update({
            'avg_power_watts': self.avg_power_watts,
            'peak_power_watts': self.peak_power_watts,
            'min_power_watts': self.min_power_watts,
            'energy_joules': self.energy_joules,
            'num_samples': len(self.samples),
            'sampling_interval_ms': self.sampling_interval_ms,
        })
        return d


@dataclass
class MemoryMeasurement(Measurement):
    """Memory usage measurement results"""
    peak_allocated_bytes: int = 0
    peak_reserved_bytes: int = 0
    initial_allocated_bytes: int = 0
    final_allocated_bytes: int = 0

    def __post_init__(self):
        self.collector_name = "memory"

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update({
            'peak_allocated_bytes': self.peak_allocated_bytes,
            'peak_reserved_bytes': self.peak_reserved_bytes,
            'initial_allocated_bytes': self.initial_allocated_bytes,
            'final_allocated_bytes': self.final_allocated_bytes,
            'peak_allocated_mb': self.peak_allocated_bytes / (1024 * 1024),
        })
        return d


class MeasurementCollector(ABC):
    """
    Abstract base class for measurement collectors.

    Collectors run in the background during benchmark execution
    to gather additional metrics like power consumption or memory usage.
    """

    def __init__(self):
        self._running = False
        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None

    @abstractmethod
    def start(self) -> None:
        """Start collecting measurements"""
        pass

    @abstractmethod
    def stop(self) -> None:
        """Stop collecting measurements"""
        pass

    @abstractmethod
    def get_measurement(self) -> Measurement:
        """Get collected measurement results"""
        pass

    @property
    def duration_ms(self) -> float:
        """Duration of measurement in milliseconds"""
        if self._start_time is None or self._end_time is None:
            return 0.0
        return (self._end_time - self._start_time) * 1000

    def reset(self) -> None:
        """Reset collector state"""
        self._running = False
        self._start_time = None
        self._end_time = None


class PowerCollector(MeasurementCollector):
    """
    Collects GPU power measurements using nvidia-smi or NVML.

    Runs in a background thread to sample power at regular intervals.
    """

    def __init__(
        self,
        device_index: int = 0,
        sampling_interval_ms: float = 100.0,
        use_nvml: bool = True,
    ):
        super().__init__()
        self.device_index = device_index
        self.sampling_interval_ms = sampling_interval_ms
        self.use_nvml = use_nvml

        self._samples: List[float] = []
        self._thread: Optional[threading.Thread] = None
        self._nvml_handle = None
        self._nvml_available = False

        # Try to initialize NVML
        if use_nvml:
            self._init_nvml()

    def _init_nvml(self) -> None:
        """Initialize NVML library"""
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
            self._nvml_available = True
        except (ImportError, Exception):
            self._nvml_available = False

    def _get_power_nvml(self) -> Optional[float]:
        """Get power reading via NVML"""
        if not self._nvml_available:
            return None
        try:
            import pynvml
            power_mw = pynvml.nvmlDeviceGetPowerUsage(self._nvml_handle)
            return power_mw / 1000.0  # Convert to watts
        except Exception:
            return None

    def _get_power_smi(self) -> Optional[float]:
        """Get power reading via nvidia-smi"""
        try:
            import subprocess
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=power.draw', '--format=csv,noheader,nounits',
                 f'--id={self.device_index}'],
                capture_output=True,
                text=True,
                timeout=1.0,
            )
            if result.returncode == 0:
                return float(result.stdout.strip())
        except Exception:
            pass
        return None

    def _get_power(self) -> Optional[float]:
        """Get power reading from available source"""
        if self._nvml_available:
            return self._get_power_nvml()
        return self._get_power_smi()

    def _sampling_loop(self) -> None:
        """Background thread sampling loop"""
        interval_sec = self.sampling_interval_ms / 1000.0

        while self._running:
            power = self._get_power()
            if power is not None:
                self._samples.append(power)
            time.sleep(interval_sec)

    def start(self) -> None:
        """Start power measurement"""
        self._samples = []
        self._running = True
        self._start_time = time.perf_counter()

        self._thread = threading.Thread(target=self._sampling_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop power measurement"""
        self._running = False
        self._end_time = time.perf_counter()

        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def get_measurement(self) -> PowerMeasurement:
        """Get power measurement results"""
        if not self._samples:
            return PowerMeasurement(
                duration_ms=self.duration_ms,
                success=False,
                error_message="No power samples collected",
                sampling_interval_ms=self.sampling_interval_ms,
            )

        return PowerMeasurement(
            duration_ms=self.duration_ms,
            samples=self._samples.copy(),
            sampling_interval_ms=self.sampling_interval_ms,
            success=True,
        )

    def reset(self) -> None:
        """Reset collector state"""
        super().reset()
        self._samples = []


class MemoryCollector(MeasurementCollector):
    """
    Collects CUDA memory usage statistics.

    Uses torch.cuda memory tracking to measure peak allocation.
    """

    def __init__(self, device_index: int = 0):
        super().__init__()
        self.device_index = device_index
        self._initial_allocated = 0
        self._peak_allocated = 0
        self._peak_reserved = 0
        self._final_allocated = 0

    def start(self) -> None:
        """Start memory tracking"""
        try:
            import torch
            if torch.cuda.is_available():
                device = f"cuda:{self.device_index}"
                # Record initial state
                self._initial_allocated = torch.cuda.memory_allocated(device)
                # Reset peak stats
                torch.cuda.reset_peak_memory_stats(device)
                self._running = True
                self._start_time = time.perf_counter()
        except Exception:
            pass

    def stop(self) -> None:
        """Stop memory tracking and record final state"""
        self._end_time = time.perf_counter()
        self._running = False

        try:
            import torch
            if torch.cuda.is_available():
                device = f"cuda:{self.device_index}"
                self._peak_allocated = torch.cuda.max_memory_allocated(device)
                self._peak_reserved = torch.cuda.max_memory_reserved(device)
                self._final_allocated = torch.cuda.memory_allocated(device)
        except Exception:
            pass

    def get_measurement(self) -> MemoryMeasurement:
        """Get memory measurement results"""
        return MemoryMeasurement(
            duration_ms=self.duration_ms,
            peak_allocated_bytes=self._peak_allocated,
            peak_reserved_bytes=self._peak_reserved,
            initial_allocated_bytes=self._initial_allocated,
            final_allocated_bytes=self._final_allocated,
            success=True,
        )

    def reset(self) -> None:
        """Reset collector state"""
        super().reset()
        self._initial_allocated = 0
        self._peak_allocated = 0
        self._peak_reserved = 0
        self._final_allocated = 0


class CompositeCollector(MeasurementCollector):
    """
    Combines multiple collectors into one.

    Useful for collecting power, memory, and other metrics simultaneously.
    """

    def __init__(self, collectors: List[MeasurementCollector]):
        super().__init__()
        self.collectors = collectors

    def start(self) -> None:
        """Start all collectors"""
        self._start_time = time.perf_counter()
        for collector in self.collectors:
            collector.start()
        self._running = True

    def stop(self) -> None:
        """Stop all collectors"""
        self._running = False
        for collector in self.collectors:
            collector.stop()
        self._end_time = time.perf_counter()

    def get_measurement(self) -> Dict[str, Measurement]:
        """Get measurements from all collectors"""
        return {
            collector.__class__.__name__: collector.get_measurement()
            for collector in self.collectors
        }

    def reset(self) -> None:
        """Reset all collectors"""
        super().reset()
        for collector in self.collectors:
            collector.reset()
