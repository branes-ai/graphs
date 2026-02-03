"""
Benchmark Runner Interface and Implementations

Provides the infrastructure for executing benchmarks with proper timing,
warmup, and statistical aggregation.

Key classes:
- BenchmarkRunner: Abstract base class for all runners
- PyTorchRunner: Runner for PyTorch operations on CPU/CUDA
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import statistics
import time

import torch

from .schema import (
    BenchmarkSpec,
    GEMMSpec,
    Conv2dSpec,
    MemoryBenchSpec,
    WorkloadSpec,
    BenchmarkResult,
    TimingStats,
    ExecutionConfig,
    DeviceType,
    Precision,
)


# Precision to PyTorch dtype mapping
PRECISION_TO_DTYPE = {
    Precision.FP64: torch.float64,
    Precision.FP32: torch.float32,
    Precision.TF32: torch.float32,  # TF32 uses float32 tensors
    Precision.FP16: torch.float16,
    Precision.BF16: torch.bfloat16,
    Precision.INT8: torch.int8,
    Precision.INT4: torch.int8,  # INT4 not directly supported, use INT8
}


def get_dtype(precision: Precision) -> torch.dtype:
    """Convert Precision enum to PyTorch dtype"""
    return PRECISION_TO_DTYPE.get(precision, torch.float32)


def resolve_device(device_type: DeviceType) -> str:
    """Resolve device type to actual device string"""
    if device_type == DeviceType.AUTO:
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    return device_type.value


def get_device_name(device: str) -> str:
    """Get human-readable device name"""
    if device.startswith("cuda"):
        idx = 0 if ":" not in device else int(device.split(":")[1])
        if torch.cuda.is_available() and idx < torch.cuda.device_count():
            return torch.cuda.get_device_name(idx)
        return "CUDA Device"
    elif device == "mps":
        return "Apple Metal"
    elif device == "cpu":
        return "CPU"
    return device


@dataclass
class RunContext:
    """Context for a benchmark run"""
    device: str
    dtype: torch.dtype
    precision: Precision
    config: ExecutionConfig

    # Timing infrastructure
    use_cuda_events: bool = False
    start_event: Optional[Any] = None
    end_event: Optional[Any] = None

    def __post_init__(self):
        if self.device.startswith("cuda") and torch.cuda.is_available():
            self.use_cuda_events = True
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)


class BenchmarkRunner(ABC):
    """
    Abstract base class for benchmark runners.

    Runners handle:
    - Setting up the execution environment
    - Running warmup iterations
    - Timing measurement iterations
    - Statistical aggregation of results
    """

    def __init__(self, config: Optional[ExecutionConfig] = None):
        self.config = config or ExecutionConfig()

    @abstractmethod
    def run(
        self,
        spec: BenchmarkSpec,
        device: str = "auto",
        precision: Precision = Precision.FP32,
    ) -> BenchmarkResult:
        """
        Run a benchmark and return results.

        Args:
            spec: Benchmark specification
            device: Target device (cpu, cuda, cuda:0, etc.)
            precision: Numerical precision to use

        Returns:
            BenchmarkResult with timing and performance metrics
        """
        pass

    def run_all_precisions(
        self,
        spec: BenchmarkSpec,
        device: str = "auto",
    ) -> List[BenchmarkResult]:
        """Run benchmark for all precisions in spec"""
        results = []
        for precision in spec.precisions:
            try:
                result = self.run(spec, device, precision)
                results.append(result)
            except Exception as e:
                # Record failure
                results.append(BenchmarkResult(
                    spec_name=spec.name,
                    timestamp=datetime.now().isoformat(),
                    device=device,
                    precision=precision.value,
                    success=False,
                    error_message=str(e),
                ))
        return results

    def run_all_devices(
        self,
        spec: BenchmarkSpec,
        precision: Precision = Precision.FP32,
    ) -> List[BenchmarkResult]:
        """Run benchmark on all devices in spec"""
        results = []
        for device_type in spec.devices:
            device = resolve_device(device_type)
            try:
                result = self.run(spec, device, precision)
                results.append(result)
            except Exception as e:
                results.append(BenchmarkResult(
                    spec_name=spec.name,
                    timestamp=datetime.now().isoformat(),
                    device=device,
                    precision=precision.value,
                    success=False,
                    error_message=str(e),
                ))
        return results


class PyTorchRunner(BenchmarkRunner):
    """
    Benchmark runner for PyTorch operations.

    Supports CPU and CUDA devices with proper synchronization
    and timing using CUDA events for GPU accuracy.
    """

    def __init__(self, config: Optional[ExecutionConfig] = None):
        super().__init__(config)
        self._operation_cache: Dict[str, Callable] = {}

    def run(
        self,
        spec: BenchmarkSpec,
        device: str = "auto",
        precision: Precision = Precision.FP32,
    ) -> BenchmarkResult:
        """Run a benchmark specification"""

        # Resolve device
        if device == "auto":
            device = resolve_device(DeviceType.AUTO)

        # Get dtype
        dtype = get_dtype(precision)

        # Create run context
        ctx = RunContext(
            device=device,
            dtype=dtype,
            precision=precision,
            config=self.config,
        )

        # Dispatch to appropriate handler
        if isinstance(spec, GEMMSpec):
            return self._run_gemm(spec, ctx)
        elif isinstance(spec, Conv2dSpec):
            return self._run_conv2d(spec, ctx)
        elif isinstance(spec, MemoryBenchSpec):
            return self._run_memory(spec, ctx)
        elif isinstance(spec, WorkloadSpec):
            return self._run_workload(spec, ctx)
        else:
            raise ValueError(f"Unsupported spec type: {type(spec)}")

    def _run_gemm(self, spec: GEMMSpec, ctx: RunContext) -> BenchmarkResult:
        """Run GEMM benchmark"""

        # Create input tensors
        if spec.batch_size > 1:
            A = torch.randn(spec.batch_size, spec.M, spec.K,
                          dtype=ctx.dtype, device=ctx.device)
            B = torch.randn(spec.batch_size, spec.K, spec.N,
                          dtype=ctx.dtype, device=ctx.device)
        else:
            A = torch.randn(spec.M, spec.K, dtype=ctx.dtype, device=ctx.device)
            B = torch.randn(spec.K, spec.N, dtype=ctx.dtype, device=ctx.device)

        # Define operation
        def gemm_op():
            return torch.matmul(A, B)

        # Run benchmark
        timings = self._run_timed(gemm_op, ctx)

        # Calculate metrics
        timing_stats = self._compute_stats(timings)
        flops = spec.flops
        gflops = (flops / 1e9) / (timing_stats.mean_ms / 1000)

        return BenchmarkResult(
            spec_name=spec.name,
            timestamp=datetime.now().isoformat(),
            device=ctx.device,
            device_name=get_device_name(ctx.device),
            precision=ctx.precision.value,
            timing=timing_stats,
            throughput_ops_per_sec=flops / (timing_stats.mean_ms / 1000),
            throughput_samples_per_sec=spec.batch_size / (timing_stats.mean_ms / 1000),
            gflops=gflops,
            success=True,
            extra={
                'M': spec.M,
                'N': spec.N,
                'K': spec.K,
                'batch_size': spec.batch_size,
                'arithmetic_intensity': spec.arithmetic_intensity,
            }
        )

    def _run_conv2d(self, spec: Conv2dSpec, ctx: RunContext) -> BenchmarkResult:
        """Run Conv2d benchmark"""

        # INT8 requires quantized convolution
        if ctx.precision == Precision.INT8:
            return self._run_conv2d_int8(spec, ctx)

        # Create input tensor
        x = torch.randn(
            spec.batch_size, spec.in_channels, spec.height, spec.width,
            dtype=ctx.dtype, device=ctx.device
        )

        # Create conv layer
        conv = torch.nn.Conv2d(
            spec.in_channels,
            spec.out_channels,
            spec.kernel_size,
            stride=spec.stride,
            padding=spec.padding,
            dilation=spec.dilation,
            groups=spec.groups,
            bias=spec.bias,
        ).to(device=ctx.device, dtype=ctx.dtype)

        # Put in eval mode
        conv.eval()

        # Define operation
        def conv_op():
            with torch.no_grad():
                return conv(x)

        # Run benchmark
        timings = self._run_timed(conv_op, ctx)

        # Calculate metrics
        timing_stats = self._compute_stats(timings)
        flops = spec.flops
        gflops = (flops / 1e9) / (timing_stats.mean_ms / 1000)

        return BenchmarkResult(
            spec_name=spec.name,
            timestamp=datetime.now().isoformat(),
            device=ctx.device,
            device_name=get_device_name(ctx.device),
            precision=ctx.precision.value,
            timing=timing_stats,
            throughput_ops_per_sec=flops / (timing_stats.mean_ms / 1000),
            throughput_samples_per_sec=spec.batch_size / (timing_stats.mean_ms / 1000),
            gflops=gflops,
            success=True,
            extra={
                'in_channels': spec.in_channels,
                'out_channels': spec.out_channels,
                'kernel_size': spec.kernel_size,
                'output_height': spec.output_height,
                'output_width': spec.output_width,
            }
        )

    def _run_conv2d_int8(self, spec: Conv2dSpec, ctx: RunContext) -> BenchmarkResult:
        """Run INT8 quantized Conv2d benchmark"""

        # INT8 convolution only supported on CPU currently via PyTorch quantization
        # For CUDA, we use fake quantization to simulate INT8 performance characteristics
        if ctx.device.startswith("cuda"):
            # Use CUDA INT8 via cuDNN (requires qint8 tensors)
            # Fall back to simulated INT8 using int8 storage with float compute
            return self._run_conv2d_int8_cuda(spec, ctx)
        else:
            return self._run_conv2d_int8_cpu(spec, ctx)

    def _run_conv2d_int8_cpu(self, spec: Conv2dSpec, ctx: RunContext) -> BenchmarkResult:
        """Run INT8 Conv2d on CPU using PyTorch quantization"""
        import torch.ao.quantization as quant

        # Create FP32 model first
        conv = torch.nn.Conv2d(
            spec.in_channels,
            spec.out_channels,
            spec.kernel_size,
            stride=spec.stride,
            padding=spec.padding,
            dilation=spec.dilation,
            groups=spec.groups,
            bias=spec.bias,
        )
        conv.eval()

        # Quantize the model
        conv.qconfig = quant.get_default_qconfig('x86')
        conv_prepared = quant.prepare(conv, inplace=False)

        # Calibrate with sample data
        with torch.no_grad():
            sample = torch.randn(spec.batch_size, spec.in_channels, spec.height, spec.width)
            conv_prepared(sample)

        conv_quantized = quant.convert(conv_prepared, inplace=False)

        # Create quantized input
        x = torch.randn(spec.batch_size, spec.in_channels, spec.height, spec.width)
        scale = x.abs().max() / 127.0
        x_quant = torch.quantize_per_tensor(x, scale.item(), 0, torch.qint8)

        # Define operation
        def conv_op():
            with torch.no_grad():
                return conv_quantized(x_quant)

        # Run benchmark
        timings = self._run_timed(conv_op, ctx)

        # Calculate metrics
        timing_stats = self._compute_stats(timings)
        flops = spec.flops
        gflops = (flops / 1e9) / (timing_stats.mean_ms / 1000)

        return BenchmarkResult(
            spec_name=spec.name,
            timestamp=datetime.now().isoformat(),
            device=ctx.device,
            device_name=get_device_name(ctx.device),
            precision=ctx.precision.value,
            timing=timing_stats,
            throughput_ops_per_sec=flops / (timing_stats.mean_ms / 1000),
            throughput_samples_per_sec=spec.batch_size / (timing_stats.mean_ms / 1000),
            gflops=gflops,
            success=True,
            extra={
                'in_channels': spec.in_channels,
                'out_channels': spec.out_channels,
                'kernel_size': spec.kernel_size,
                'output_height': spec.output_height,
                'output_width': spec.output_width,
                'quantization': 'pytorch_native',
            }
        )

    def _run_conv2d_int8_cuda(self, spec: Conv2dSpec, ctx: RunContext) -> BenchmarkResult:
        """Run INT8 Conv2d on CUDA using cudnn int8 path"""

        # PyTorch 2.0+ supports cuda quantized operations
        # Try to use torch.backends.cudnn with int8
        try:
            # Create FP32 reference first
            conv_fp32 = torch.nn.Conv2d(
                spec.in_channels,
                spec.out_channels,
                spec.kernel_size,
                stride=spec.stride,
                padding=spec.padding,
                dilation=spec.dilation,
                groups=spec.groups,
                bias=spec.bias,
            ).cuda()
            conv_fp32.eval()

            # Use torch.ao.nn.quantized for CUDA quantized conv
            import torch.ao.nn.quantized as nnq

            # Create weight in int8
            weight_fp32 = conv_fp32.weight.data
            w_scale = weight_fp32.abs().max() / 127.0
            weight_int8 = torch.quantize_per_tensor(
                weight_fp32.cpu(), w_scale.item(), 0, torch.qint8
            )

            # Input
            x = torch.randn(
                spec.batch_size, spec.in_channels, spec.height, spec.width,
                device='cuda'
            )

            # For CUDA INT8, use cuDNN's implicit GEMM with int8
            # This requires going through cudnn directly or using TensorRT
            # PyTorch native doesn't fully support qint8 on CUDA yet

            # Fallback: simulate with fp16 (closest available precision)
            conv_fp16 = conv_fp32.half()
            x_fp16 = x.half()

            def conv_op():
                with torch.no_grad():
                    return conv_fp16(x_fp16)

            # Run benchmark
            timings = self._run_timed(conv_op, ctx)

            timing_stats = self._compute_stats(timings)
            flops = spec.flops
            gflops = (flops / 1e9) / (timing_stats.mean_ms / 1000)

            return BenchmarkResult(
                spec_name=spec.name,
                timestamp=datetime.now().isoformat(),
                device=ctx.device,
                device_name=get_device_name(ctx.device),
                precision='int8_simulated_fp16',
                timing=timing_stats,
                throughput_ops_per_sec=flops / (timing_stats.mean_ms / 1000),
                throughput_samples_per_sec=spec.batch_size / (timing_stats.mean_ms / 1000),
                gflops=gflops,
                success=True,
                extra={
                    'in_channels': spec.in_channels,
                    'out_channels': spec.out_channels,
                    'kernel_size': spec.kernel_size,
                    'output_height': spec.output_height,
                    'output_width': spec.output_width,
                    'quantization': 'simulated_fp16',
                    'note': 'PyTorch CUDA does not natively support qint8 conv. Using FP16 as proxy.',
                }
            )

        except Exception as e:
            return BenchmarkResult(
                spec_name=spec.name,
                timestamp=datetime.now().isoformat(),
                device=ctx.device,
                precision='int8',
                success=False,
                error_message=f"INT8 CUDA conv not supported: {e}",
            )

    def _run_memory(self, spec: MemoryBenchSpec, ctx: RunContext) -> BenchmarkResult:
        """Run memory bandwidth benchmark"""

        # Create arrays
        n = spec.array_size
        a = torch.zeros(n, dtype=ctx.dtype, device=ctx.device)
        b = torch.randn(n, dtype=ctx.dtype, device=ctx.device)
        c = torch.randn(n, dtype=ctx.dtype, device=ctx.device)
        q = 2.0

        # Select operation based on pattern
        if spec.pattern == "copy":
            def op():
                a.copy_(b)
            bytes_per_iter = 2 * n * a.element_size()  # read b, write a
        elif spec.pattern == "scale":
            def op():
                torch.mul(b, q, out=a)
            bytes_per_iter = 2 * n * a.element_size()
        elif spec.pattern == "add":
            def op():
                torch.add(b, c, out=a)
            bytes_per_iter = 3 * n * a.element_size()  # read b,c, write a
        elif spec.pattern == "triad":
            def op():
                torch.addcmul(b, c, torch.tensor(q, device=ctx.device, dtype=ctx.dtype), out=a)
            bytes_per_iter = 3 * n * a.element_size()
        else:
            raise ValueError(f"Unknown pattern: {spec.pattern}")

        # Run benchmark
        timings = self._run_timed(op, ctx)

        # Calculate metrics
        timing_stats = self._compute_stats(timings)
        bandwidth_gbps = (bytes_per_iter / 1e9) / (timing_stats.mean_ms / 1000)

        return BenchmarkResult(
            spec_name=spec.name,
            timestamp=datetime.now().isoformat(),
            device=ctx.device,
            device_name=get_device_name(ctx.device),
            precision=ctx.precision.value,
            timing=timing_stats,
            bandwidth_gbps=bandwidth_gbps,
            success=True,
            extra={
                'pattern': spec.pattern,
                'array_size': spec.array_size,
                'bytes_per_iteration': bytes_per_iter,
            }
        )

    def _run_workload(self, spec: WorkloadSpec, ctx: RunContext) -> BenchmarkResult:
        """Run full model workload benchmark"""

        # Load model
        model = self._load_model(spec, ctx)
        model.eval()

        # Create input
        input_shape = (spec.batch_size,) + spec.input_shape
        x = torch.randn(*input_shape, dtype=ctx.dtype, device=ctx.device)

        # Define operation
        def inference_op():
            with torch.no_grad():
                return model(x)

        # Run benchmark
        timings = self._run_timed(inference_op, ctx)

        # Calculate metrics
        timing_stats = self._compute_stats(timings)
        throughput = spec.batch_size / (timing_stats.mean_ms / 1000)

        return BenchmarkResult(
            spec_name=spec.name,
            timestamp=datetime.now().isoformat(),
            device=ctx.device,
            device_name=get_device_name(ctx.device),
            precision=ctx.precision.value,
            timing=timing_stats,
            throughput_samples_per_sec=throughput,
            success=True,
            extra={
                'model_name': spec.model_name,
                'model_source': spec.model_source,
                'batch_size': spec.batch_size,
                'input_shape': list(spec.input_shape),
            }
        )

    def _load_model(self, spec: WorkloadSpec, ctx: RunContext) -> torch.nn.Module:
        """Load a model from spec"""

        if spec.model_source == "torchvision":
            import torchvision.models as models

            model_fn = getattr(models, spec.model_name, None)
            if model_fn is None:
                raise ValueError(f"Unknown torchvision model: {spec.model_name}")

            model = model_fn(weights=None)

        elif spec.model_source == "timm":
            try:
                import timm
                model = timm.create_model(spec.model_name, pretrained=False)
            except ImportError:
                raise ImportError("timm not installed. Install with: pip install timm")

        else:
            raise ValueError(f"Unknown model source: {spec.model_source}")

        return model.to(device=ctx.device, dtype=ctx.dtype)

    def _run_timed(
        self,
        operation: Callable,
        ctx: RunContext,
    ) -> List[float]:
        """
        Run operation with timing.

        Returns list of timing measurements in milliseconds.
        """
        config = ctx.config
        timings = []

        # Warmup
        for _ in range(config.warmup_iterations):
            operation()

        # Synchronize before measurement
        if ctx.use_cuda_events:
            torch.cuda.synchronize()

        # Measurement iterations
        for _ in range(config.measurement_iterations):
            if ctx.use_cuda_events:
                # Use CUDA events for accurate GPU timing
                ctx.start_event.record()
                operation()
                ctx.end_event.record()
                torch.cuda.synchronize()
                elapsed_ms = ctx.start_event.elapsed_time(ctx.end_event)
            else:
                # CPU timing
                if config.sync_before_timing and ctx.device.startswith("cuda"):
                    torch.cuda.synchronize()

                start = time.perf_counter()
                operation()

                if config.sync_before_timing and ctx.device.startswith("cuda"):
                    torch.cuda.synchronize()

                elapsed_ms = (time.perf_counter() - start) * 1000

            timings.append(elapsed_ms)

        return timings

    def _compute_stats(self, timings: List[float]) -> TimingStats:
        """Compute statistical summary of timings"""

        sorted_timings = sorted(timings)
        n = len(timings)

        return TimingStats(
            mean_ms=statistics.mean(timings),
            std_ms=statistics.stdev(timings) if n > 1 else 0.0,
            min_ms=min(timings),
            max_ms=max(timings),
            median_ms=statistics.median(timings),
            p95_ms=sorted_timings[int(n * 0.95)] if n >= 20 else sorted_timings[-1],
            p99_ms=sorted_timings[int(n * 0.99)] if n >= 100 else sorted_timings[-1],
            num_iterations=n,
        )


# Convenience function
def run_benchmark(
    spec: BenchmarkSpec,
    device: str = "auto",
    precision: Precision = Precision.FP32,
    config: Optional[ExecutionConfig] = None,
) -> BenchmarkResult:
    """
    Convenience function to run a single benchmark.

    Args:
        spec: Benchmark specification
        device: Target device
        precision: Numerical precision
        config: Execution configuration

    Returns:
        BenchmarkResult
    """
    runner = PyTorchRunner(config)
    return runner.run(spec, device, precision)
