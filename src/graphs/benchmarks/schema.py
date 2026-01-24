"""
Benchmark Specification Schema

Defines the data models for benchmark specifications and results.
Specifications describe WHAT to benchmark, while results capture
the measurements from running benchmarks.

This module uses Pydantic for validation and serialization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import json
import yaml


class BenchmarkCategory(Enum):
    """Classification of benchmark types"""
    MICROBENCHMARK = "microbenchmark"  # Single operator benchmarks
    WORKLOAD = "workload"              # Full model benchmarks
    MEMORY = "memory"                  # Memory bandwidth benchmarks
    CUSTOM = "custom"                  # User-defined benchmarks


class Precision(Enum):
    """Supported numerical precisions"""
    FP64 = "fp64"
    FP32 = "fp32"
    TF32 = "tf32"
    FP16 = "fp16"
    BF16 = "bf16"
    FP8 = "fp8"
    INT8 = "int8"
    INT4 = "int4"


class DeviceType(Enum):
    """Target device types"""
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"      # Apple Metal
    XPU = "xpu"      # Intel
    AUTO = "auto"    # Auto-detect best available


@dataclass
class ExecutionConfig:
    """Configuration for benchmark execution"""
    warmup_iterations: int = 10
    measurement_iterations: int = 100
    min_duration_ms: float = 100.0  # Minimum total measurement time
    sync_before_timing: bool = True  # GPU synchronization
    clear_cache: bool = True         # Clear caches between runs

    def to_dict(self) -> Dict[str, Any]:
        return {
            'warmup_iterations': self.warmup_iterations,
            'measurement_iterations': self.measurement_iterations,
            'min_duration_ms': self.min_duration_ms,
            'sync_before_timing': self.sync_before_timing,
            'clear_cache': self.clear_cache,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExecutionConfig':
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class BenchmarkSpec:
    """
    Base specification for all benchmarks.

    This defines the common fields that all benchmark types share.
    Specific benchmark types (microbenchmark, workload) extend this.
    """
    # Identity
    name: str                          # Unique identifier (e.g., "gemm_1024x1024_fp32")
    description: str = ""              # Human-readable description
    category: BenchmarkCategory = BenchmarkCategory.MICROBENCHMARK
    tags: List[str] = field(default_factory=list)

    # Execution settings
    precisions: List[Precision] = field(default_factory=lambda: [Precision.FP32])
    devices: List[DeviceType] = field(default_factory=lambda: [DeviceType.AUTO])
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)

    # Metadata
    version: str = "1.0"
    created_at: Optional[str] = None
    author: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'name': self.name,
            'description': self.description,
            'category': self.category.value,
            'tags': self.tags,
            'precisions': [p.value for p in self.precisions],
            'devices': [d.value for d in self.devices],
            'execution': self.execution.to_dict(),
            'version': self.version,
            'created_at': self.created_at,
            'author': self.author,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BenchmarkSpec':
        """Create from dictionary"""
        data = data.copy()
        if 'category' in data:
            data['category'] = BenchmarkCategory(data['category'])
        if 'precisions' in data:
            data['precisions'] = [Precision(p) for p in data['precisions']]
        if 'devices' in data:
            data['devices'] = [DeviceType(d) for d in data['devices']]
        if 'execution' in data:
            data['execution'] = ExecutionConfig.from_dict(data['execution'])
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class GEMMSpec(BenchmarkSpec):
    """
    Specification for GEMM (General Matrix Multiply) benchmarks.

    Computes: C = alpha * A @ B + beta * C
    Where A is (M x K), B is (K x N), C is (M x N)
    """
    # Matrix dimensions
    M: int = 1024
    N: int = 1024
    K: int = 1024

    # Optional batch dimension for batched GEMM
    batch_size: int = 1

    # GEMM parameters
    alpha: float = 1.0
    beta: float = 0.0
    transpose_a: bool = False
    transpose_b: bool = False

    def __post_init__(self):
        self.category = BenchmarkCategory.MICROBENCHMARK
        if not self.tags:
            self.tags = ['gemm', 'compute', 'blas3']

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update({
            'M': self.M,
            'N': self.N,
            'K': self.K,
            'batch_size': self.batch_size,
            'alpha': self.alpha,
            'beta': self.beta,
            'transpose_a': self.transpose_a,
            'transpose_b': self.transpose_b,
        })
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GEMMSpec':
        base = BenchmarkSpec.from_dict(data)
        return cls(
            name=base.name,
            description=base.description,
            category=base.category,
            tags=base.tags,
            precisions=base.precisions,
            devices=base.devices,
            execution=base.execution,
            version=base.version,
            created_at=base.created_at,
            author=base.author,
            M=data.get('M', 1024),
            N=data.get('N', 1024),
            K=data.get('K', 1024),
            batch_size=data.get('batch_size', 1),
            alpha=data.get('alpha', 1.0),
            beta=data.get('beta', 0.0),
            transpose_a=data.get('transpose_a', False),
            transpose_b=data.get('transpose_b', False),
        )

    @property
    def flops(self) -> int:
        """Total FLOPs for this GEMM operation"""
        # GEMM: 2*M*N*K (multiply-add counted as 2 ops)
        return 2 * self.batch_size * self.M * self.N * self.K

    @property
    def bytes_accessed(self) -> int:
        """Total bytes accessed (assuming fp32)"""
        # A: M*K, B: K*N, C: M*N (read and write)
        element_size = 4  # fp32
        a_bytes = self.batch_size * self.M * self.K * element_size
        b_bytes = self.batch_size * self.K * self.N * element_size
        c_bytes = self.batch_size * self.M * self.N * element_size * 2  # read + write
        return a_bytes + b_bytes + c_bytes

    @property
    def arithmetic_intensity(self) -> float:
        """FLOPs per byte"""
        return self.flops / self.bytes_accessed if self.bytes_accessed > 0 else 0.0


@dataclass
class Conv2dSpec(BenchmarkSpec):
    """
    Specification for Conv2d benchmarks.

    Standard 2D convolution with configurable kernel, stride, padding.
    """
    # Input dimensions (N, C, H, W)
    batch_size: int = 1
    in_channels: int = 64
    height: int = 56
    width: int = 56

    # Convolution parameters
    out_channels: int = 64
    kernel_size: int = 3
    stride: int = 1
    padding: int = 1
    dilation: int = 1
    groups: int = 1  # 1 = standard conv, in_channels = depthwise

    # Bias
    bias: bool = False

    def __post_init__(self):
        self.category = BenchmarkCategory.MICROBENCHMARK
        if not self.tags:
            if self.groups == self.in_channels:
                self.tags = ['conv2d', 'depthwise', 'compute']
            elif self.kernel_size == 1:
                self.tags = ['conv2d', 'pointwise', 'compute']
            else:
                self.tags = ['conv2d', 'compute']

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update({
            'batch_size': self.batch_size,
            'in_channels': self.in_channels,
            'height': self.height,
            'width': self.width,
            'out_channels': self.out_channels,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'padding': self.padding,
            'dilation': self.dilation,
            'groups': self.groups,
            'bias': self.bias,
        })
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Conv2dSpec':
        base = BenchmarkSpec.from_dict(data)
        return cls(
            name=base.name,
            description=base.description,
            category=base.category,
            tags=base.tags,
            precisions=base.precisions,
            devices=base.devices,
            execution=base.execution,
            version=base.version,
            created_at=base.created_at,
            author=base.author,
            batch_size=data.get('batch_size', 1),
            in_channels=data.get('in_channels', 64),
            height=data.get('height', 56),
            width=data.get('width', 56),
            out_channels=data.get('out_channels', 64),
            kernel_size=data.get('kernel_size', 3),
            stride=data.get('stride', 1),
            padding=data.get('padding', 1),
            dilation=data.get('dilation', 1),
            groups=data.get('groups', 1),
            bias=data.get('bias', False),
        )

    @property
    def output_height(self) -> int:
        return (self.height + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) // self.stride + 1

    @property
    def output_width(self) -> int:
        return (self.width + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) // self.stride + 1

    @property
    def flops(self) -> int:
        """Total FLOPs for this convolution"""
        # Per output element: 2 * kernel_size^2 * (in_channels / groups)
        flops_per_output = 2 * self.kernel_size * self.kernel_size * (self.in_channels // self.groups)
        total_outputs = self.batch_size * self.out_channels * self.output_height * self.output_width
        return flops_per_output * total_outputs


@dataclass
class MemoryBenchSpec(BenchmarkSpec):
    """
    Specification for memory bandwidth benchmarks.

    Based on STREAM benchmark patterns: copy, scale, add, triad.
    """
    # Array size in elements
    array_size: int = 10_000_000

    # Pattern type
    pattern: str = "triad"  # copy, scale, add, triad

    def __post_init__(self):
        self.category = BenchmarkCategory.MEMORY
        if not self.tags:
            self.tags = ['memory', 'bandwidth', self.pattern]

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update({
            'array_size': self.array_size,
            'pattern': self.pattern,
        })
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryBenchSpec':
        base = BenchmarkSpec.from_dict(data)
        return cls(
            name=base.name,
            description=base.description,
            category=base.category,
            tags=base.tags,
            precisions=base.precisions,
            devices=base.devices,
            execution=base.execution,
            version=base.version,
            created_at=base.created_at,
            author=base.author,
            array_size=data.get('array_size', 10_000_000),
            pattern=data.get('pattern', 'triad'),
        )


@dataclass
class WorkloadSpec(BenchmarkSpec):
    """
    Specification for full model workload benchmarks.

    Runs inference on a complete model to measure end-to-end performance.
    """
    # Model specification
    model_name: str = "resnet18"
    model_source: str = "torchvision"  # torchvision, timm, huggingface, local

    # Input specification
    batch_size: int = 1
    input_shape: Tuple[int, ...] = (3, 224, 224)

    # Inference mode
    inference_only: bool = True
    use_amp: bool = False  # Automatic mixed precision
    compile_model: bool = False  # torch.compile

    def __post_init__(self):
        self.category = BenchmarkCategory.WORKLOAD
        if not self.tags:
            self.tags = ['workload', 'inference', self.model_name]

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update({
            'model_name': self.model_name,
            'model_source': self.model_source,
            'batch_size': self.batch_size,
            'input_shape': list(self.input_shape),
            'inference_only': self.inference_only,
            'use_amp': self.use_amp,
            'compile_model': self.compile_model,
        })
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WorkloadSpec':
        base = BenchmarkSpec.from_dict(data)
        input_shape = data.get('input_shape', (3, 224, 224))
        if isinstance(input_shape, list):
            input_shape = tuple(input_shape)
        return cls(
            name=base.name,
            description=base.description,
            category=base.category,
            tags=base.tags,
            precisions=base.precisions,
            devices=base.devices,
            execution=base.execution,
            version=base.version,
            created_at=base.created_at,
            author=base.author,
            model_name=data.get('model_name', 'resnet18'),
            model_source=data.get('model_source', 'torchvision'),
            batch_size=data.get('batch_size', 1),
            input_shape=input_shape,
            inference_only=data.get('inference_only', True),
            use_amp=data.get('use_amp', False),
            compile_model=data.get('compile_model', False),
        )


# =============================================================================
# BENCHMARK RESULTS
# =============================================================================

@dataclass
class TimingStats:
    """Statistical summary of timing measurements"""
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    median_ms: float
    p95_ms: float
    p99_ms: float
    num_iterations: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            'mean_ms': self.mean_ms,
            'std_ms': self.std_ms,
            'min_ms': self.min_ms,
            'max_ms': self.max_ms,
            'median_ms': self.median_ms,
            'p95_ms': self.p95_ms,
            'p99_ms': self.p99_ms,
            'num_iterations': self.num_iterations,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TimingStats':
        return cls(**data)


@dataclass
class BenchmarkResult:
    """
    Result from running a benchmark.

    Captures all measurements from a single benchmark execution.
    """
    # Identity
    spec_name: str                     # Name of benchmark spec that was run
    timestamp: str                     # ISO format timestamp

    # Device info
    device: str                        # Device used (e.g., "cuda:0", "cpu")
    device_name: str = ""              # Human-readable name (e.g., "NVIDIA H100")

    # Precision used
    precision: str = "fp32"

    # Core timing results
    timing: Optional[TimingStats] = None

    # Derived metrics
    throughput_ops_per_sec: float = 0.0  # Operations per second
    throughput_samples_per_sec: float = 0.0  # Samples (batches) per second
    gflops: float = 0.0                 # GFLOPS achieved
    bandwidth_gbps: float = 0.0         # Memory bandwidth (GB/s)

    # Power measurements (optional)
    avg_power_watts: Optional[float] = None
    peak_power_watts: Optional[float] = None
    energy_joules: Optional[float] = None

    # Memory usage (optional)
    peak_memory_bytes: Optional[int] = None
    allocated_memory_bytes: Optional[int] = None

    # Status
    success: bool = True
    error_message: str = ""

    # Additional data
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'spec_name': self.spec_name,
            'timestamp': self.timestamp,
            'device': self.device,
            'device_name': self.device_name,
            'precision': self.precision,
            'timing': self.timing.to_dict() if self.timing else None,
            'throughput_ops_per_sec': self.throughput_ops_per_sec,
            'throughput_samples_per_sec': self.throughput_samples_per_sec,
            'gflops': self.gflops,
            'bandwidth_gbps': self.bandwidth_gbps,
            'avg_power_watts': self.avg_power_watts,
            'peak_power_watts': self.peak_power_watts,
            'energy_joules': self.energy_joules,
            'peak_memory_bytes': self.peak_memory_bytes,
            'allocated_memory_bytes': self.allocated_memory_bytes,
            'success': self.success,
            'error_message': self.error_message,
            'extra': self.extra,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BenchmarkResult':
        data = data.copy()
        if data.get('timing'):
            data['timing'] = TimingStats.from_dict(data['timing'])
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string"""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> 'BenchmarkResult':
        """Deserialize from JSON string"""
        return cls.from_dict(json.loads(json_str))


# =============================================================================
# SPEC LOADING UTILITIES
# =============================================================================

def load_spec_from_yaml(path: Union[str, Path]) -> BenchmarkSpec:
    """Load a benchmark spec from a YAML file"""
    path = Path(path)
    with open(path, 'r') as f:
        data = yaml.safe_load(f)

    # Determine spec type from category or explicit type field
    spec_type = data.get('type', data.get('category', 'microbenchmark'))

    if spec_type == 'gemm' or 'gemm' in data.get('tags', []):
        return GEMMSpec.from_dict(data)
    elif spec_type == 'conv2d' or 'conv2d' in data.get('tags', []):
        return Conv2dSpec.from_dict(data)
    elif spec_type == 'memory' or spec_type == 'bandwidth':
        return MemoryBenchSpec.from_dict(data)
    elif spec_type == 'workload':
        return WorkloadSpec.from_dict(data)
    else:
        return BenchmarkSpec.from_dict(data)


def save_spec_to_yaml(spec: BenchmarkSpec, path: Union[str, Path]) -> None:
    """Save a benchmark spec to a YAML file"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        yaml.dump(spec.to_dict(), f, default_flow_style=False, sort_keys=False)


def load_result_from_json(path: Union[str, Path]) -> BenchmarkResult:
    """Load a benchmark result from a JSON file"""
    path = Path(path)
    with open(path, 'r') as f:
        return BenchmarkResult.from_dict(json.load(f))


def save_result_to_json(result: BenchmarkResult, path: Union[str, Path]) -> None:
    """Save a benchmark result to a JSON file"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(result.to_dict(), f, indent=2)


# Type alias for spec registry
SpecRegistry = Dict[str, BenchmarkSpec]


def load_specs_from_directory(directory: Union[str, Path]) -> SpecRegistry:
    """Load all benchmark specs from a directory"""
    directory = Path(directory)
    specs = {}

    for yaml_file in directory.glob('**/*.yaml'):
        try:
            spec = load_spec_from_yaml(yaml_file)
            specs[spec.name] = spec
        except Exception as e:
            print(f"Warning: Failed to load {yaml_file}: {e}")

    for yml_file in directory.glob('**/*.yml'):
        try:
            spec = load_spec_from_yaml(yml_file)
            specs[spec.name] = spec
        except Exception as e:
            print(f"Warning: Failed to load {yml_file}: {e}")

    return specs
