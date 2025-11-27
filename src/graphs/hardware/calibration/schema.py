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


# Canonical precision ordering: large to small, floating-point to integer
# This ordering is used for consistent display in reports and tables
CANONICAL_PRECISION_ORDER = [
    'fp64',   # IEEE Double Precision
    'fp32',   # IEEE Single Precision
    'tf32',   # NVIDIA TensorFloat-32 (19-bit, Tensor Cores only)
    'fp16',   # IEEE Half Precision
    'fp8',    # 8-bit floating point
    'fp4',    # 4-bit floating point
    'bf16',   # Brain Float 16
    'int64',  # 64-bit integer
    'int32',  # 32-bit integer
    'int16',  # 16-bit integer
    'int8',   # 8-bit integer
    'int4',   # 4-bit integer
]


class OperationType(Enum):
    """Classification of operation types for calibration"""
    # Legacy matmul (alias for BLAS3_GEMM)
    MATMUL = "matmul"

    # BLAS Level 1: Vector-Vector operations (O(n))
    BLAS1_DOT = "blas1_dot"          # dot(x, y) - inner product
    BLAS1_AXPY = "blas1_axpy"        # y = a*x + y
    BLAS1_SCAL = "blas1_scal"        # x = a*x

    # BLAS Level 2: Matrix-Vector operations (O(n²))
    BLAS2_GEMV = "blas2_gemv"        # y = alpha*A*x + beta*y
    BLAS2_GER = "blas2_ger"          # A = alpha*x*y' + A (outer product)

    # BLAS Level 3: Matrix-Matrix operations (O(n³))
    BLAS3_GEMM = "blas3_gemm"        # C = alpha*A*B + beta*C

    # Convolution operations
    CONV2D = "conv2d"
    CONV1D = "conv1d"
    DEPTHWISE_CONV = "depthwise_conv"
    TRANSPOSE_CONV = "transpose_conv"

    # STREAM memory bandwidth benchmarks
    STREAM_COPY = "stream_copy"      # a[i] = b[i]
    STREAM_SCALE = "stream_scale"    # a[i] = q * b[i]
    STREAM_ADD = "stream_add"        # a[i] = b[i] + c[i]
    STREAM_TRIAD = "stream_triad"    # a[i] = b[i] + q * c[i]

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

    Note: measured_gops is used generically:
    - For floating-point (FP64, FP32, FP16, BF16, FP8): GFLOPS (Giga Floating-Point Ops/Second)
    - For integer (INT32, INT16, INT8, INT4): GIOPS (Giga Integer Ops/Second)
    """
    precision: str  # Precision enum value (e.g., "fp32", "int8", "fp8_e4m3")

    # Support status
    supported: bool  # True if hardware can execute this precision
    failure_reason: Optional[str] = None  # Why it failed (if supported=False)

    # Performance metrics (only populated if supported=True)
    measured_gops: Optional[float] = None  # GFLOPS for float, GIOPS for int
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

    # Backward compatibility alias
    @property
    def measured_gflops(self) -> Optional[float]:
        """Deprecated: Use measured_gops instead"""
        return self.measured_gops

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
class GPUClockData:
    """GPU clock frequency data captured during calibration.

    Note: sm_clock_mhz is required - GPU calibration fails without it.
    Other fields are optional but recommended for full analysis.
    """

    # SM/Graphics clock frequency (MHz) - REQUIRED
    sm_clock_mhz: int
    """SM/Graphics clock frequency during calibration. Required."""

    # Query metadata - REQUIRED
    query_method: str
    """How clocks were queried: 'nvidia-smi', 'sysfs', 'tegrastats'. Required."""

    # Optional clock frequencies
    mem_clock_mhz: Optional[int] = None
    """Memory clock frequency during calibration."""

    max_sm_clock_mhz: Optional[int] = None
    """Maximum SM clock (for calculating % of max)."""

    max_mem_clock_mhz: Optional[int] = None
    """Maximum memory clock."""

    # Power state
    power_draw_watts: Optional[float] = None
    """Power draw during calibration."""

    power_limit_watts: Optional[float] = None
    """Configured power limit."""

    temperature_c: Optional[int] = None
    """GPU temperature during calibration."""

    # Jetson-specific
    nvpmodel_mode: Optional[int] = None
    """Jetson NVPModel mode ID."""

    power_mode_name: Optional[str] = None
    """Human-readable power mode (e.g., 'MAXN', '15W')."""

    def to_dict(self) -> Dict:
        """Convert to dictionary, excluding None values."""
        result = {}
        for key, value in asdict(self).items():
            if value is not None:
                result[key] = value
        return result

    @classmethod
    def from_dict(cls, data: Dict) -> 'GPUClockData':
        """Create from dictionary."""
        # Filter to only known fields
        known_fields = set(cls.__dataclass_fields__.keys())
        filtered = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered)


@dataclass
class CPUClockData:
    """CPU clock frequency data captured during calibration.

    Note: current_freq_mhz is required - calibration fails without it.
    Other fields are optional but recommended for full analysis.
    """

    # Current clock frequency (MHz) - REQUIRED
    current_freq_mhz: float
    """Current CPU frequency during calibration (average across cores). Required."""

    # Query metadata - REQUIRED
    query_method: str
    """How clocks were queried: 'sysfs', 'cpuinfo', 'sysctl', 'psutil'. Required."""

    # Optional frequency range
    min_freq_mhz: Optional[float] = None
    """Minimum CPU frequency (scaling_min_freq)."""

    max_freq_mhz: Optional[float] = None
    """Maximum CPU frequency (scaling_max_freq)."""

    base_freq_mhz: Optional[float] = None
    """Base/nominal CPU frequency from spec."""

    # Per-core frequencies (for heterogeneous CPUs)
    per_core_freq_mhz: List[float] = field(default_factory=list)
    """Per-core frequencies in MHz."""

    # Frequency scaling
    governor: Optional[str] = None
    """CPU frequency governor (e.g., 'performance', 'powersave')."""

    driver: Optional[str] = None
    """CPU frequency scaling driver (e.g., 'intel_pstate')."""

    # Turbo/boost state
    turbo_enabled: Optional[bool] = None
    """Whether turbo boost is enabled."""

    def to_dict(self) -> Dict:
        """Convert to dictionary, excluding None/empty values."""
        result = {}
        for key, value in asdict(self).items():
            if value is not None and value != []:
                result[key] = value
        return result

    @classmethod
    def from_dict(cls, data: Dict) -> 'CPUClockData':
        """Create from dictionary."""
        # Filter to only known fields
        known_fields = set(cls.__dataclass_fields__.keys())
        filtered = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered)


@dataclass
class PreflightCheckResult:
    """Result of a single pre-flight check (stored in calibration data)."""
    name: str
    status: str  # 'passed', 'warning', 'failed', 'skipped'
    message: str
    current_value: Optional[str] = None
    expected_value: Optional[str] = None

    def to_dict(self) -> Dict:
        result = {'name': self.name, 'status': self.status, 'message': self.message}
        if self.current_value:
            result['current_value'] = self.current_value
        if self.expected_value:
            result['expected_value'] = self.expected_value
        return result

    @classmethod
    def from_dict(cls, data: Dict) -> 'PreflightCheckResult':
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class PreflightData:
    """Pre-flight check results stored in calibration data.

    Records system state at calibration time to help interpret results.
    """
    timestamp: str
    """When pre-flight checks were run."""

    passed: bool
    """Whether all checks passed (no failures)."""

    forced: bool = False
    """Whether calibration was forced despite failures."""

    checks: List[PreflightCheckResult] = field(default_factory=list)
    """Individual check results."""

    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp,
            'passed': self.passed,
            'forced': self.forced,
            'checks': [c.to_dict() for c in self.checks]
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'PreflightData':
        checks = [PreflightCheckResult.from_dict(c) for c in data.get('checks', [])]
        return cls(
            timestamp=data['timestamp'],
            passed=data['passed'],
            forced=data.get('forced', False),
            checks=checks
        )


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

    # Device type for platform validation
    device_type: str = "cpu"  # "cpu" or "cuda"
    platform_architecture: str = "unknown"  # "x86_64", "aarch64", "arm64"
    framework: str = "numpy"  # "numpy" or "pytorch" - which framework ran the benchmarks

    # Clock data captured during calibration
    gpu_clock: Optional[GPUClockData] = None
    """GPU clock frequencies captured during calibration (CUDA only)."""

    cpu_clock: Optional[CPUClockData] = None
    """CPU clock frequencies captured during calibration."""

    # Pre-flight check results
    preflight: Optional[PreflightData] = None
    """Pre-flight check results (system state validation)."""

    def to_dict(self) -> Dict:
        result = asdict(self)
        # Handle nested GPUClockData
        if self.gpu_clock is not None:
            result['gpu_clock'] = self.gpu_clock.to_dict()
        elif 'gpu_clock' in result:
            del result['gpu_clock']  # Remove None gpu_clock
        # Handle nested CPUClockData
        if self.cpu_clock is not None:
            result['cpu_clock'] = self.cpu_clock.to_dict()
        elif 'cpu_clock' in result:
            del result['cpu_clock']  # Remove None cpu_clock
        # Handle nested PreflightData
        if self.preflight is not None:
            result['preflight'] = self.preflight.to_dict()
        elif 'preflight' in result:
            del result['preflight']  # Remove None preflight
        return result

    @classmethod
    def from_dict(cls, data: Dict) -> 'CalibrationMetadata':
        # Handle nested clock data
        data = data.copy()
        if 'gpu_clock' in data and data['gpu_clock'] is not None:
            data['gpu_clock'] = GPUClockData.from_dict(data['gpu_clock'])
        if 'cpu_clock' in data and data['cpu_clock'] is not None:
            data['cpu_clock'] = CPUClockData.from_dict(data['cpu_clock'])
        if 'preflight' in data and data['preflight'] is not None:
            data['preflight'] = PreflightData.from_dict(data['preflight'])
        known_fields = set(cls.__dataclass_fields__.keys())
        return cls(**{k: v for k, v in data.items() if k in known_fields})


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

        # Show framework and device info if available
        if hasattr(self.metadata, 'framework'):
            print(f"Framework: {self.metadata.framework.upper()}")
            if hasattr(self.metadata, 'device_type'):
                print(f"Device:    {self.metadata.device_type.upper()}")
            print()

        print("Theoretical Specifications:")
        print(f"  Peak GFLOPS (FP32): {self.theoretical_peak_gflops:.1f}")
        print(f"  Peak Bandwidth:     {self.theoretical_bandwidth_gbps:.1f} GB/s")
        print()

        # Separate operations by category: STREAM (memory), BLAS (compute), other
        stream_ops = {}
        blas_ops = {}
        other_memory_ops = {}
        compute_ops = {}

        for key, profile in self.operation_profiles.items():
            if 'stream' in profile.operation_type:
                stream_ops[key] = profile
            elif 'blas' in profile.operation_type:
                blas_ops[key] = profile
            elif profile.memory_bound:
                other_memory_ops[key] = profile
            else:
                compute_ops[key] = profile

        # STREAM Memory Bandwidth Benchmark Results
        if stream_ops:
            print("STREAM Memory Bandwidth Benchmark:")
            print(f"  {'Kernel':<15} {'Size (MB)':>10} {'Bandwidth':>12} {'Latency':>10} {'Efficiency':>12} {'Description'}")
            print("  " + "-" * 95)

            # Group by kernel
            stream_kernels = {}
            for key, profile in sorted(stream_ops.items()):
                kernel = profile.extra_params.get('kernel', 'unknown')
                if kernel not in stream_kernels:
                    stream_kernels[kernel] = []
                stream_kernels[kernel].append(profile)

            # Define kernel descriptions
            kernel_descriptions = {
                'copy': 'a[i] = b[i]',
                'scale': 'a[i] = q * b[i]',
                'add': 'a[i] = b[i] + c[i]',
                'triad': 'a[i] = b[i] + q * c[i]'
            }

            # Print results by kernel
            for kernel_name in ['copy', 'scale', 'add', 'triad']:
                if kernel_name in stream_kernels:
                    profiles = stream_kernels[kernel_name]
                    # Find best bandwidth for this kernel
                    best_profile = max(profiles, key=lambda p: p.achieved_bandwidth_gbps)
                    bandwidth = best_profile.achieved_bandwidth_gbps
                    latency = best_profile.mean_latency_ms
                    eff = bandwidth / self.theoretical_bandwidth_gbps
                    size = best_profile.extra_params.get('size_mb', '?')
                    desc = kernel_descriptions.get(kernel_name, '')

                    print(f"  {kernel_name.upper():<15} {size:>10} {bandwidth:>10.1f} GB/s {latency:>8.2f} ms {eff*100:>10.1f}%  {desc}")

            # Print STREAM score (minimum bandwidth)
            all_stream_bw = [p.achieved_bandwidth_gbps for p in stream_ops.values()]
            if all_stream_bw:
                stream_score = min(all_stream_bw)
                print()
                print(f"  STREAM Score (minimum): {stream_score:.1f} GB/s")
            print()

        # BLAS Compute Performance Benchmark Results
        if blas_ops:
            # Group by BLAS level and operation
            blas_by_level = {1: {}, 2: {}, 3: {}}
            for key, profile in sorted(blas_ops.items()):
                level = profile.extra_params.get('blas_level', 0)
                operation = profile.extra_params.get('operation', 'unknown')
                if level in blas_by_level:
                    if operation not in blas_by_level[level]:
                        blas_by_level[level][operation] = []
                    blas_by_level[level][operation].append(profile)

            # Check if any operations have precision results
            has_precision_results = False
            all_precisions = set()
            for profiles_by_op in blas_by_level.values():
                for profiles in profiles_by_op.values():
                    for profile in profiles:
                        if profile.precision_results:
                            has_precision_results = True
                            all_precisions.update(profile.precision_results.keys())

            # FORMAT 1: Compact Grid Summary (if multi-precision data exists)
            if has_precision_results:
                print("BLAS Performance Summary (Highest Throughput by Precision):")
                # Sort precisions using canonical order
                precisions_sorted = [p for p in CANONICAL_PRECISION_ORDER if p in all_precisions]

                # Build header
                header = f"  {'Operation':<12}"
                for prec in precisions_sorted:
                    header += f" {prec:>10}"
                header += f" {'Best Precision':>16}"
                print(header)
                print("  " + "-" * (14 + len(precisions_sorted) * 11 + 16))

                # Print each operation
                for level in [1, 2, 3]:
                    if blas_by_level[level]:
                        for operation in sorted(blas_by_level[level].keys()):
                            profiles = blas_by_level[level][operation]

                            # Find best across all sizes for each precision
                            best_by_precision = {}
                            for profile in profiles:
                                if profile.precision_results:
                                    for prec, result in profile.precision_results.items():
                                        if result.supported and result.measured_gops:
                                            if prec not in best_by_precision or result.measured_gops > best_by_precision[prec]:
                                                best_by_precision[prec] = result.measured_gops

                            # Determine overall best precision
                            if best_by_precision:
                                best_prec = max(best_by_precision.items(), key=lambda x: x[1])[0]

                                row = f"  {operation.upper():<12}"
                                for prec in precisions_sorted:
                                    if prec in best_by_precision:
                                        row += f" {best_by_precision[prec]:>10.1f}"
                                    else:
                                        row += f" {'N/A':>10}"
                                row += f" {best_prec:>16}"
                                print(row)

                print()

            # FORMAT 2: Hierarchical Precision Breakdown
            print("BLAS Compute Performance (by Operation and Precision):")
            print("=" * 120)

            for level in [1, 2, 3]:
                if blas_by_level[level]:
                    print(f"\nLevel {level}: {'Vector-Vector (O(n))' if level == 1 else 'Matrix-Vector (O(n²))' if level == 2 else 'Matrix-Matrix (O(n³))'}")
                    print("-" * 120)

                    for operation in sorted(blas_by_level[level].keys()):
                        profiles = blas_by_level[level][operation]

                        print(f"\n{operation.upper()}:")

                        if has_precision_results and any(p.precision_results for p in profiles):
                            # Multi-precision view
                            print(f"  {'Precision':<10} {'Best Size':>10} {'Highest Throughput':>18} {'Latency':>10} {'AI':>8} {'Efficiency':>12}")
                            print("  " + "-" * 80)

                            # Collect best result per precision
                            precision_best = {}
                            for profile in profiles:
                                if profile.precision_results:
                                    for prec, result in profile.precision_results.items():
                                        if result.supported and result.measured_gops:
                                            if prec not in precision_best or result.measured_gops > precision_best[prec][0]:
                                                precision_best[prec] = (result.measured_gops, result.mean_latency_ms,
                                                                       result.arithmetic_intensity, result.efficiency,
                                                                       profile.extra_params.get('size', '?'))

                            # Print sorted by canonical precision order
                            for prec in [p for p in CANONICAL_PRECISION_ORDER if p in precision_best]:
                                gflops, latency, ai, eff, size = precision_best[prec]

                                # Format size
                                if isinstance(size, int):
                                    if size >= 1000000:
                                        size_str = f"{size // 1000000}M"
                                    elif size >= 1000:
                                        size_str = f"{size // 1000}K"
                                    else:
                                        size_str = str(size)
                                else:
                                    size_str = str(size)

                                # Format latency
                                if latency >= 1000:
                                    lat_str = f"{latency/1000:.2f}s"
                                else:
                                    lat_str = f"{latency:.2f}ms"

                                # Determine unit based on precision type
                                unit = "GIOPS" if prec.startswith('int') else "GFLOPS"
                                throughput_str = f"{gflops:.1f} {unit}"

                                print(f"  {prec:<10} {size_str:>10} {throughput_str:>18} {lat_str:>10} {ai:>8.2f} {eff*100:>10.1f}%")
                        else:
                            # Single precision view (legacy)
                            print(f"  {'Best Size':>10} {'Highest Throughput':>18} {'Latency':>10} {'AI':>8} {'Efficiency':>12}")
                            print("  " + "-" * 70)

                            best_profile = max(profiles, key=lambda p: p.measured_gflops)
                            gflops = best_profile.measured_gflops
                            latency = best_profile.mean_latency_ms
                            ai = best_profile.arithmetic_intensity
                            eff = best_profile.efficiency
                            size = best_profile.extra_params.get('size', '?')

                            # Format size
                            if isinstance(size, int):
                                if size >= 1000000:
                                    size_str = f"{size // 1000000}M"
                                elif size >= 1000:
                                    size_str = f"{size // 1000}K"
                                else:
                                    size_str = str(size)
                            else:
                                size_str = str(size)

                            # Format latency
                            if latency >= 1000:
                                lat_str = f"{latency/1000:.2f}s"
                            else:
                                lat_str = f"{latency:.2f}ms"

                            # Legacy view: assume GFLOPS (typically FP32)
                            throughput_str = f"{gflops:.1f} GFLOPS"

                            print(f"  {size_str:>10} {throughput_str:>18} {lat_str:>10} {ai:>8.2f} {eff*100:>10.1f}%")

            print()

        # Other memory operations (if any)
        if other_memory_ops:
            print("Other Memory Operations:")
            print(f"  {'Operation':<40} {'Bandwidth':>12} {'Efficiency':>12}")
            print("  " + "-" * 70)

            for key, profile in sorted(other_memory_ops.items()):
                bandwidth = profile.achieved_bandwidth_gbps if profile.achieved_bandwidth_gbps > 0 else self.measured_bandwidth_gbps
                eff = self.bandwidth_efficiency if profile.achieved_bandwidth_gbps == 0 else (profile.achieved_bandwidth_gbps / self.theoretical_bandwidth_gbps)
                print(f"  {key:<40} {bandwidth:>10.1f} GB/s {eff*100:>10.1f}%")
            print()

        # Compute Operations Section - Group by precision (legacy matmul only)
        # Collect all matmul results by precision and size first
        # Structure: matmul_by_precision[precision][size] = [(gops, latency_ms), ...]
        matmul_by_precision = {}

        if compute_ops:
            for key, profile in sorted(compute_ops.items()):
                if profile.operation_type == 'matmul' and profile.precision_results:
                    # Extract matrix size from extra_params or key
                    size = profile.extra_params.get('matrix_size', 'unknown')

                    # Process each precision
                    for prec_name, prec_result in profile.precision_results.items():
                        if prec_result.supported:
                            if prec_name not in matmul_by_precision:
                                matmul_by_precision[prec_name] = {}

                            if size not in matmul_by_precision[prec_name]:
                                matmul_by_precision[prec_name][size] = []

                            matmul_by_precision[prec_name][size].append((prec_result.measured_gops, prec_result.mean_latency_ms))
                        else:
                            # Track failed/skipped precisions
                            if prec_name not in matmul_by_precision:
                                matmul_by_precision[prec_name] = {}

                            if size not in matmul_by_precision[prec_name]:
                                matmul_by_precision[prec_name][size] = []

                            matmul_by_precision[prec_name][size].append(None)  # Mark as skipped

        # Only print the section if there's actual matmul data
        if matmul_by_precision:
            print("Matrix Multiplication Performance (by precision):")
            print(f"  {'Precision':<10} {'Size':<12} {'Latency':>10} {'Min GOPS':>10} {'Avg GOPS':>10} {'Max GOPS':>10} {'Efficiency':>11}")
            print("  " + "-" * 90)

            # Print results grouped by precision
            for prec_name in sorted(matmul_by_precision.keys()):
                sizes_data = matmul_by_precision[prec_name]

                for size in sorted(sizes_data.keys()):
                    results = sizes_data[size]

                    # Format size as a single string with fixed width
                    size_str = f"{size}×{size}"

                    # Check if this precision was skipped
                    if results and results[0] is None:
                        # Use "-" for unmeasured values, put skip reason on the right
                        print(f"  {prec_name:<10} {size_str:<12} {'-':>10} {'-':>10} {'-':>10} {'-':>10} {'-':>10}  SKIPPED (< 50 GOPS)")
                    elif results:
                        # Extract gops and latency from tuples
                        gops_values = [r[0] for r in results]
                        latency_values = [r[1] for r in results]

                        min_gops = min(gops_values)
                        avg_gops = sum(gops_values) / len(gops_values)
                        max_gops = max(gops_values)
                        avg_latency = sum(latency_values) / len(latency_values)

                        # Format latency: show seconds if >= 1000ms, else milliseconds
                        if avg_latency >= 1000:
                            latency_str = f"{avg_latency/1000:.2f}s"
                        else:
                            latency_str = f"{avg_latency:.1f}ms"

                        # Get efficiency from precision matrix
                        theoretical_peak = self.precision_matrix.theoretical_peaks.get(prec_name, None) if self.precision_matrix else None
                        if theoretical_peak and theoretical_peak > 0:
                            efficiency = max_gops / theoretical_peak
                        else:
                            efficiency = 0.0

                        # Determine units
                        is_int = prec_name in ['int32', 'int16', 'int8', 'int4']
                        units = "GIOPS" if is_int else "GFLOPS"

                        # Format efficiency with warning if >110%
                        if efficiency > 1.10:
                            eff_str = f"{efficiency*100:>10.1f}%  ⚠ ABOVE THEORETICAL"
                        else:
                            eff_str = f"{efficiency*100:>10.1f}%"

                        print(f"  {prec_name:<10} {size_str:<12} {latency_str:>10} {min_gops:>10.1f} {avg_gops:>10.1f} {max_gops:>10.1f} {eff_str}")

            print()

        # Precision Support Summary
        if self.precision_matrix:
            print("Precision Support Summary:")
            print(f"  Supported:   {', '.join(self.precision_matrix.supported_precisions)}")
            if self.precision_matrix.unsupported_precisions:
                print(f"  Unsupported: {', '.join(self.precision_matrix.unsupported_precisions)}")
            print()

        # Add efficiency explanation if any results exceed theoretical
        has_high_efficiency = False
        if compute_ops:
            for key, profile in compute_ops.items():
                if profile.precision_results:
                    for prec_result in profile.precision_results.values():
                        if prec_result.supported and prec_result.efficiency and prec_result.efficiency > 1.10:
                            has_high_efficiency = True
                            break
                if has_high_efficiency:
                    break

        if has_high_efficiency:
            print("Note on >100% Efficiency:")
            print("  Efficiency above theoretical peak typically indicates:")
            print("    • Turbo Boost / GPU Boost clocks exceeding base frequency")
            print("    • Optimized BLAS libraries (MKL, cuBLAS) exceeding naive calculations")
            print("    • Conservative theoretical peaks (based on sustained, not peak clocks)")
            print("  This is normal and indicates good hardware utilization.")
            print()

        # Fusion patterns summary
        if self.fusion_profiles:
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
