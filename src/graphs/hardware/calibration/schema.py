"""
Calibration data schema and structures.

Defines the data model for hardware calibration results, including
per-operation performance profiles and aggregate statistics.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import json
from pathlib import Path


class OperationType(Enum):
    """Classification of operation types for calibration"""
    MATMUL = "matmul"
    CONV2D = "conv2d"
    CONV1D = "conv1d"
    DEPTHWISE_CONV = "depthwise_conv"
    TRANSPOSE_CONV = "transpose_conv"

    # Elementwise operations
    RELU = "relu"
    GELU = "gelu"
    SIGMOID = "sigmoid"
    ADD = "add"
    MUL = "mul"

    # Reduction operations
    SOFTMAX = "softmax"
    LAYERNORM = "layernorm"
    BATCHNORM = "batchnorm"

    # Pooling
    MAXPOOL = "maxpool"
    AVGPOOL = "avgpool"
    ADAPTIVE_POOL = "adaptive_pool"

    # Attention
    ATTENTION = "attention"
    FLASH_ATTENTION = "flash_attention"

    # Other
    EMBEDDING = "embedding"
    LINEAR = "linear"
    UNKNOWN = "unknown"


@dataclass
class PrecisionTestResult:
    """
    Result of testing a single precision on hardware.

    Captures whether the precision is supported and its performance characteristics.
    This enables reporting FAIL for unsupported precisions with clear failure reasons.
    """
    precision: str  # Precision enum value (e.g., "fp32", "int8", "fp8_e4m3")

    # Support status
    supported: bool  # True if hardware can execute this precision
    failure_reason: Optional[str] = None  # Why it failed (if supported=False)

    # Performance metrics (only populated if supported=True)
    measured_gflops: Optional[float] = None
    efficiency: Optional[float] = None  # Fraction of theoretical peak for this precision
    mean_latency_ms: Optional[float] = None
    std_latency_ms: Optional[float] = None
    min_latency_ms: Optional[float] = None
    max_latency_ms: Optional[float] = None

    # Comparison to FP32 baseline
    speedup_vs_fp32: Optional[float] = None  # e.g., FP16 = 2.0× faster than FP32

    # Test configuration
    test_size: int = 0  # Matrix size for matmul, kernel size for conv, etc.
    num_trials: int = 0

    # Additional metrics
    arithmetic_intensity: Optional[float] = None
    achieved_bandwidth_gbps: Optional[float] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'PrecisionTestResult':
        """Create from dictionary"""
        return cls(**data)


@dataclass
class OperationCalibration:
    """
    Measured performance for a specific operation type.

    This captures the real-world performance of an operation, including
    both compute and memory characteristics.
    """
    operation_type: str  # OperationType value

    # Performance metrics (legacy - use precision_results for multi-precision)
    measured_gflops: float
    efficiency: float  # Fraction of theoretical peak (0.0 to 1.0)
    achieved_bandwidth_gbps: float

    # Bottleneck analysis
    memory_bound: bool
    compute_bound: bool
    arithmetic_intensity: float  # FLOPs per byte

    # Test configuration
    batch_size: int
    input_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]

    # Timing details
    mean_latency_ms: float
    std_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    num_trials: int

    # Additional parameters (kernel size for conv, etc.)
    extra_params: Dict[str, Any] = field(default_factory=dict)

    # NEW: Multi-precision test results
    # Maps precision name (e.g., "fp32", "int8") -> PrecisionTestResult
    precision_results: Dict[str, PrecisionTestResult] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        d = asdict(self)
        # Convert PrecisionTestResult objects to dicts
        if 'precision_results' in d and d['precision_results']:
            d['precision_results'] = {
                k: v if isinstance(v, dict) else v.to_dict()
                for k, v in d['precision_results'].items()
            }
        return d

    @classmethod
    def from_dict(cls, data: Dict) -> 'OperationCalibration':
        """Create from dictionary"""
        # Convert tuple fields
        if isinstance(data.get('input_shape'), list):
            data['input_shape'] = tuple(data['input_shape'])
        if isinstance(data.get('output_shape'), list):
            data['output_shape'] = tuple(data['output_shape'])

        # Convert precision_results from dicts to PrecisionTestResult objects
        if 'precision_results' in data and data['precision_results']:
            data['precision_results'] = {
                k: PrecisionTestResult.from_dict(v) if isinstance(v, dict) else v
                for k, v in data['precision_results'].items()
            }

        return cls(**data)


@dataclass
class FusionCalibration:
    """
    Calibration for a fused kernel pattern.

    Measures the performance benefit of fusing multiple operations
    compared to running them separately.
    """
    # Fusion pattern identification
    fusion_pattern: str  # "Linear_Bias_ReLU", "Conv_BN_ReLU", etc.
    operators: List[str]  # ["linear", "bias", "relu"]
    num_operators: int    # 3

    # Unfused performance (baseline)
    unfused_latency_ms: float
    unfused_gflops: float
    unfused_memory_bytes: int

    # Fused performance (optimized)
    fused_latency_ms: float
    fused_gflops: float
    fused_memory_bytes: int

    # Fusion benefits
    speedup_factor: float         # unfused_latency / fused_latency
    memory_reduction: float       # 1 - (fused_bytes / unfused_bytes)
    gflops_improvement: float     # (fused_gflops - unfused_gflops) / unfused_gflops

    # Test configuration
    input_shape: Tuple[int, ...]
    extra_params: Dict[str, Any] = field(default_factory=dict)

    # Timing details
    num_trials: int = 100

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'FusionCalibration':
        """Create from dictionary"""
        if isinstance(data.get('input_shape'), list):
            data['input_shape'] = tuple(data['input_shape'])
        if isinstance(data.get('operators'), tuple):
            data['operators'] = list(data['operators'])
        return cls(**data)


@dataclass
class PrecisionCapabilityMatrix:
    """
    Hardware precision support matrix.

    Summarizes which precisions are supported across all operations tested.
    This provides a quick overview of hardware precision capabilities.
    """
    hardware_name: str

    # Precision support classification
    supported_precisions: List[str] = field(default_factory=list)  # ["fp32", "fp16", "int8"]
    unsupported_precisions: List[str] = field(default_factory=list)  # ["fp64", "fp8_e4m3"]

    # Per-precision peak performance (if supported)
    # Maps precision name -> best measured GFLOPS across all operations
    peak_gflops_by_precision: Dict[str, float] = field(default_factory=dict)

    # Speedup ratios relative to FP32 baseline
    # Maps precision name -> speedup factor (e.g., {"fp16": 2.0, "int8": 4.0})
    speedup_vs_fp32: Dict[str, float] = field(default_factory=dict)

    # Per-precision theoretical peaks (from hardware specs)
    # Populated from preset configuration
    theoretical_peaks: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'PrecisionCapabilityMatrix':
        """Create from dictionary"""
        return cls(**data)


@dataclass
class CalibrationMetadata:
    """Metadata about the calibration run"""
    hardware_name: str
    calibration_date: str
    calibration_tool_version: str

    # System information
    cpu_model: str
    cpu_count: int
    total_memory_gb: float

    # Software versions
    python_version: str
    pytorch_version: Optional[str] = None
    numpy_version: Optional[str] = None

    # Calibration settings
    num_warmup_runs: int = 3
    num_measurement_runs: int = 10

    # NEW: Device type for platform validation
    device_type: str = "cpu"  # "cpu" or "cuda"
    platform_architecture: str = "unknown"  # "x86_64", "aarch64", "arm64"

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'CalibrationMetadata':
        return cls(**data)


@dataclass
class HardwareCalibration:
    """
    Complete calibration profile for a hardware target.

    This represents the measured performance characteristics of hardware,
    obtained by running actual benchmarks rather than using theoretical specs.
    """
    metadata: CalibrationMetadata

    # Theoretical specifications (from datasheet)
    theoretical_peak_gflops: float
    theoretical_bandwidth_gbps: float

    # Measured aggregate performance
    best_measured_gflops: float  # Best case (e.g., large matmul)
    avg_measured_gflops: float   # Average across operations
    worst_measured_gflops: float  # Worst case (e.g., small elementwise)

    # Measured memory bandwidth
    measured_bandwidth_gbps: float
    bandwidth_efficiency: float  # measured / theoretical

    # Per-operation calibration profiles
    operation_profiles: Dict[str, OperationCalibration] = field(default_factory=dict)

    # Fusion pattern calibration profiles
    fusion_profiles: Dict[str, FusionCalibration] = field(default_factory=dict)

    # NEW: Precision capability summary
    precision_matrix: Optional[PrecisionCapabilityMatrix] = None

    # Summary statistics
    best_efficiency: float = 0.0     # Best case efficiency
    avg_efficiency: float = 0.0      # Average efficiency
    worst_efficiency: float = 0.0    # Worst case efficiency

    def add_operation(self, profile: OperationCalibration):
        """Add an operation calibration profile"""
        key = self._make_key(profile)
        self.operation_profiles[key] = profile
        self._update_statistics()

    def add_fusion_pattern(self, profile: FusionCalibration):
        """Add a fusion calibration profile"""
        key = profile.fusion_pattern
        self.fusion_profiles[key] = profile

    def get_fusion_pattern(self, pattern: str) -> Optional[FusionCalibration]:
        """Get calibration for a fusion pattern"""
        return self.fusion_profiles.get(pattern)

    def get_fusion_speedup(self, pattern: str, default: float = 1.0) -> float:
        """Get fusion speedup factor with fallback"""
        profile = self.get_fusion_pattern(pattern)
        return profile.speedup_factor if profile else default

    def get_operation(self, operation_type: str, **kwargs) -> Optional[OperationCalibration]:
        """
        Get calibration for a specific operation.

        Args:
            operation_type: Type of operation (e.g., "matmul", "conv2d")
            **kwargs: Additional parameters for matching (e.g., kernel_size=3)

        Returns:
            Matching OperationCalibration, or None if not found
        """
        # Try exact match first
        for key, profile in self.operation_profiles.items():
            if profile.operation_type == operation_type:
                # Check if extra params match
                match = True
                for k, v in kwargs.items():
                    if profile.extra_params.get(k) != v:
                        match = False
                        break
                if match:
                    return profile

        # Fallback: return any matching operation type
        for profile in self.operation_profiles.values():
            if profile.operation_type == operation_type:
                return profile

        return None

    def get_efficiency(self, operation_type: str, **kwargs) -> float:
        """
        Get efficiency for an operation type, with fallback to average.

        Returns:
            Efficiency (0.0 to 1.0), or avg_efficiency if operation not found
        """
        profile = self.get_operation(operation_type, **kwargs)
        if profile:
            return profile.efficiency
        return self.avg_efficiency

    def _make_key(self, profile: OperationCalibration) -> str:
        """Create unique key for operation profile"""
        key_parts = [profile.operation_type]

        # Add distinguishing parameters
        if profile.extra_params:
            for k, v in sorted(profile.extra_params.items()):
                key_parts.append(f"{k}={v}")

        # Add size category for operations where size matters
        if profile.operation_type in ["matmul", "conv2d"]:
            total_size = 1
            for dim in profile.input_shape:
                total_size *= dim

            # Categorize by size
            if total_size < 1024 * 1024:
                key_parts.append("small")
            elif total_size < 10 * 1024 * 1024:
                key_parts.append("medium")
            else:
                key_parts.append("large")

        return "_".join(key_parts)

    def _update_statistics(self):
        """
        Update aggregate statistics from operation profiles.

        Separates compute operations (matmul, conv) from memory operations (copy)
        to avoid mixing GFLOPS (0.0 for memory ops) with bandwidth efficiency.
        """
        if not self.operation_profiles:
            return

        # Separate compute operations (those that actually do FLOPs)
        # from memory operations (bandwidth-bound, 0 GFLOPS)
        compute_profiles = [p for p in self.operation_profiles.values()
                           if p.measured_gflops > 0]
        memory_profiles = [p for p in self.operation_profiles.values()
                          if p.memory_bound and p.measured_gflops == 0]

        # Compute statistics (only from compute operations)
        # This avoids "worst GFLOPS = 0.0" from memory operations
        if compute_profiles:
            compute_gflops = [p.measured_gflops for p in compute_profiles]
            compute_effs = [p.efficiency for p in compute_profiles]

            self.best_measured_gflops = max(compute_gflops)
            self.avg_measured_gflops = sum(compute_gflops) / len(compute_gflops)
            self.worst_measured_gflops = min(compute_gflops)

            self.best_efficiency = max(compute_effs)
            self.avg_efficiency = sum(compute_effs) / len(compute_effs)
            self.worst_efficiency = min(compute_effs)
        else:
            # Fallback if no compute operations (shouldn't happen)
            self.best_measured_gflops = 0.0
            self.avg_measured_gflops = 0.0
            self.worst_measured_gflops = 0.0
            self.best_efficiency = 0.0
            self.avg_efficiency = 0.0
            self.worst_efficiency = 0.0

        # Memory bandwidth statistics (already handled separately)
        # measured_bandwidth_gbps and bandwidth_efficiency are set directly
        # by the calibrator, not computed here

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'metadata': self.metadata.to_dict(),
            'theoretical_peak_gflops': self.theoretical_peak_gflops,
            'theoretical_bandwidth_gbps': self.theoretical_bandwidth_gbps,
            'best_measured_gflops': self.best_measured_gflops,
            'avg_measured_gflops': self.avg_measured_gflops,
            'worst_measured_gflops': self.worst_measured_gflops,
            'measured_bandwidth_gbps': self.measured_bandwidth_gbps,
            'bandwidth_efficiency': self.bandwidth_efficiency,
            'best_efficiency': self.best_efficiency,
            'avg_efficiency': self.avg_efficiency,
            'worst_efficiency': self.worst_efficiency,
            'operation_profiles': {
                k: v.to_dict() for k, v in self.operation_profiles.items()
            },
            'fusion_profiles': {
                k: v.to_dict() for k, v in self.fusion_profiles.items()
            },
            'precision_matrix': self.precision_matrix.to_dict() if self.precision_matrix else None
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'HardwareCalibration':
        """Create from dictionary"""
        metadata = CalibrationMetadata.from_dict(data['metadata'])
        operation_profiles = {
            k: OperationCalibration.from_dict(v)
            for k, v in data.get('operation_profiles', {}).items()
        }
        fusion_profiles = {
            k: FusionCalibration.from_dict(v)
            for k, v in data.get('fusion_profiles', {}).items()
        }

        precision_matrix = None
        if data.get('precision_matrix'):
            precision_matrix = PrecisionCapabilityMatrix.from_dict(data['precision_matrix'])

        return cls(
            metadata=metadata,
            theoretical_peak_gflops=data['theoretical_peak_gflops'],
            theoretical_bandwidth_gbps=data['theoretical_bandwidth_gbps'],
            best_measured_gflops=data.get('best_measured_gflops', 0),
            avg_measured_gflops=data.get('avg_measured_gflops', 0),
            worst_measured_gflops=data.get('worst_measured_gflops', 0),
            measured_bandwidth_gbps=data.get('measured_bandwidth_gbps', 0),
            bandwidth_efficiency=data.get('bandwidth_efficiency', 0),
            operation_profiles=operation_profiles,
            fusion_profiles=fusion_profiles,
            precision_matrix=precision_matrix,
            best_efficiency=data.get('best_efficiency', 0),
            avg_efficiency=data.get('avg_efficiency', 0),
            worst_efficiency=data.get('worst_efficiency', 0),
        )

    def save(self, filepath: Path):
        """Save calibration to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: Path) -> 'HardwareCalibration':
        """Load calibration from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)

    def print_summary(self):
        """Print human-readable summary"""
        print("=" * 80)
        print(f"Hardware Calibration: {self.metadata.hardware_name}")
        print(f"Date: {self.metadata.calibration_date}")
        print("=" * 80)
        print()

        print("Theoretical Specifications:")
        print(f"  Peak GFLOPS:    {self.theoretical_peak_gflops:.1f}")
        print(f"  Peak Bandwidth: {self.theoretical_bandwidth_gbps:.1f} GB/s")
        print()

        print("Measured Performance:")
        print(f"  Best GFLOPS:    {self.best_measured_gflops:.1f} ({self.best_efficiency*100:.1f}% efficiency)")
        print(f"  Avg GFLOPS:     {self.avg_measured_gflops:.1f} ({self.avg_efficiency*100:.1f}% efficiency)")
        print(f"  Worst GFLOPS:   {self.worst_measured_gflops:.1f} ({self.worst_efficiency*100:.1f}% efficiency)")
        print(f"  Bandwidth:      {self.measured_bandwidth_gbps:.1f} GB/s ({self.bandwidth_efficiency*100:.1f}% efficiency)")
        print()

        print(f"Operation Profiles ({len(self.operation_profiles)} total):")
        print(f"  {'Operation':<25} {'GFLOPS':>10} {'Efficiency':>12} {'Bound':>12}")
        print("  " + "-" * 65)

        for key, profile in sorted(self.operation_profiles.items()):
            bound = "Memory" if profile.memory_bound else "Compute"
            print(f"  {key:<25} {profile.measured_gflops:>10.1f} {profile.efficiency*100:>11.1f}% {bound:>12}")

        # Fusion patterns summary
        if self.fusion_profiles:
            print()
            print(f"Fusion Pattern Performance ({len(self.fusion_profiles)} total):")
            print("  " + "-" * 80)

            for pattern, profile in sorted(self.fusion_profiles.items()):
                print(f"\n  {profile.fusion_pattern}:")
                print(f"    Speedup:  {profile.speedup_factor:.2f}× faster")
                print(f"    Memory:   {profile.memory_reduction*100:.1f}% reduction")
                print(f"    GFLOPS:   {profile.gflops_improvement*100:+.1f}% change")

                # Verdict based on speedup
                if profile.speedup_factor >= 1.5:
                    verdict = "✓ Strong fusion benefit"
                elif profile.speedup_factor >= 1.1:
                    verdict = "✓ Moderate fusion benefit"
                elif profile.speedup_factor >= 0.95:
                    verdict = "⚠ Minimal benefit"
                else:
                    verdict = "✗ Fusion is slower!"

                print(f"    Verdict:  {verdict}")
