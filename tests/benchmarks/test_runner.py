"""
Tests for benchmark runner and collectors.

Tests cover:
- Runner creation and configuration
- GEMM benchmark execution
- Conv2d benchmark execution
- Memory benchmark execution
- Statistical aggregation
- Collector functionality
"""

import pytest
import torch

from graphs.benchmarks.schema import (
    GEMMSpec,
    Conv2dSpec,
    MemoryBenchSpec,
    WorkloadSpec,
    ExecutionConfig,
    Precision,
    DeviceType,
)
from graphs.benchmarks.runner import (
    PyTorchRunner,
    BenchmarkRunner,
    RunContext,
    run_benchmark,
    resolve_device,
    get_device_name,
    get_dtype,
)
from graphs.benchmarks.collectors import (
    PowerCollector,
    MemoryCollector,
    PowerMeasurement,
    MemoryMeasurement,
)


class TestDeviceResolution:
    """Tests for device resolution utilities"""

    def test_resolve_auto_device(self):
        device = resolve_device(DeviceType.AUTO)
        assert device in ["cpu", "cuda", "mps"]

    def test_resolve_cpu_device(self):
        device = resolve_device(DeviceType.CPU)
        assert device == "cpu"

    def test_get_cpu_name(self):
        name = get_device_name("cpu")
        assert name == "CPU"

    def test_get_dtype_fp32(self):
        dtype = get_dtype(Precision.FP32)
        assert dtype == torch.float32

    def test_get_dtype_fp16(self):
        dtype = get_dtype(Precision.FP16)
        assert dtype == torch.float16

    def test_get_dtype_bf16(self):
        dtype = get_dtype(Precision.BF16)
        assert dtype == torch.bfloat16


class TestRunContext:
    """Tests for RunContext"""

    def test_create_cpu_context(self):
        ctx = RunContext(
            device="cpu",
            dtype=torch.float32,
            precision=Precision.FP32,
            config=ExecutionConfig(),
        )
        assert ctx.device == "cpu"
        assert ctx.use_cuda_events is False

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_create_cuda_context(self):
        ctx = RunContext(
            device="cuda",
            dtype=torch.float32,
            precision=Precision.FP32,
            config=ExecutionConfig(),
        )
        assert ctx.device == "cuda"
        assert ctx.use_cuda_events is True
        assert ctx.start_event is not None
        assert ctx.end_event is not None


class TestPyTorchRunner:
    """Tests for PyTorchRunner"""

    def test_create_runner(self):
        runner = PyTorchRunner()
        assert runner.config is not None
        assert runner.config.warmup_iterations == 10

    def test_create_runner_custom_config(self):
        config = ExecutionConfig(warmup_iterations=5, measurement_iterations=20)
        runner = PyTorchRunner(config)
        assert runner.config.warmup_iterations == 5
        assert runner.config.measurement_iterations == 20

    def test_run_gemm_cpu(self):
        spec = GEMMSpec(
            name="test_gemm",
            M=64,
            N=64,
            K=64,
        )
        # Use minimal iterations for testing
        config = ExecutionConfig(warmup_iterations=2, measurement_iterations=5)
        runner = PyTorchRunner(config)

        result = runner.run(spec, device="cpu", precision=Precision.FP32)

        assert result.success
        assert result.spec_name == "test_gemm"
        assert result.device == "cpu"
        assert result.timing is not None
        assert result.timing.mean_ms > 0
        assert result.gflops > 0

    def test_run_gemm_batched(self):
        spec = GEMMSpec(
            name="test_gemm_batched",
            M=32,
            N=32,
            K=32,
            batch_size=4,
        )
        config = ExecutionConfig(warmup_iterations=2, measurement_iterations=5)
        runner = PyTorchRunner(config)

        result = runner.run(spec, device="cpu", precision=Precision.FP32)

        assert result.success
        assert result.extra['batch_size'] == 4

    def test_run_conv2d_cpu(self):
        spec = Conv2dSpec(
            name="test_conv",
            batch_size=1,
            in_channels=16,
            out_channels=16,
            height=32,
            width=32,
            kernel_size=3,
        )
        config = ExecutionConfig(warmup_iterations=2, measurement_iterations=5)
        runner = PyTorchRunner(config)

        result = runner.run(spec, device="cpu", precision=Precision.FP32)

        assert result.success
        assert result.spec_name == "test_conv"
        assert result.timing is not None
        assert result.gflops > 0

    def test_run_conv2d_depthwise(self):
        spec = Conv2dSpec(
            name="test_conv_dw",
            batch_size=1,
            in_channels=32,
            out_channels=32,
            height=28,
            width=28,
            kernel_size=3,
            groups=32,  # Depthwise
        )
        config = ExecutionConfig(warmup_iterations=2, measurement_iterations=5)
        runner = PyTorchRunner(config)

        result = runner.run(spec, device="cpu", precision=Precision.FP32)

        assert result.success

    def test_run_memory_benchmark(self):
        spec = MemoryBenchSpec(
            name="test_memory",
            array_size=100000,
            pattern="triad",
        )
        config = ExecutionConfig(warmup_iterations=2, measurement_iterations=5)
        runner = PyTorchRunner(config)

        result = runner.run(spec, device="cpu", precision=Precision.FP32)

        assert result.success
        assert result.bandwidth_gbps > 0
        assert result.extra['pattern'] == "triad"

    def test_run_memory_patterns(self):
        """Test all memory patterns"""
        config = ExecutionConfig(warmup_iterations=2, measurement_iterations=5)
        runner = PyTorchRunner(config)

        for pattern in ["copy", "scale", "add", "triad"]:
            spec = MemoryBenchSpec(
                name=f"test_{pattern}",
                array_size=50000,
                pattern=pattern,
            )
            result = runner.run(spec, device="cpu", precision=Precision.FP32)
            assert result.success, f"Pattern {pattern} failed"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_run_gemm_cuda(self):
        spec = GEMMSpec(
            name="test_gemm_cuda",
            M=128,
            N=128,
            K=128,
        )
        config = ExecutionConfig(warmup_iterations=3, measurement_iterations=10)
        runner = PyTorchRunner(config)

        result = runner.run(spec, device="cuda", precision=Precision.FP32)

        assert result.success
        assert result.device == "cuda"
        assert "NVIDIA" in result.device_name or "cuda" in result.device_name.lower()

    def test_run_all_precisions(self):
        spec = GEMMSpec(
            name="test_multi_precision",
            M=32,
            N=32,
            K=32,
            precisions=[Precision.FP32, Precision.FP16],
        )
        config = ExecutionConfig(warmup_iterations=2, measurement_iterations=5)
        runner = PyTorchRunner(config)

        results = runner.run_all_precisions(spec, device="cpu")

        assert len(results) == 2
        # FP32 should succeed
        assert results[0].success
        assert results[0].precision == "fp32"


class TestTimingStats:
    """Tests for timing statistics computation"""

    def test_stats_computation(self):
        spec = GEMMSpec(name="test", M=32, N=32, K=32)
        config = ExecutionConfig(warmup_iterations=2, measurement_iterations=20)
        runner = PyTorchRunner(config)

        result = runner.run(spec, device="cpu")

        stats = result.timing
        assert stats is not None
        assert stats.num_iterations == 20
        assert stats.mean_ms > 0
        assert stats.std_ms >= 0
        assert stats.min_ms <= stats.mean_ms
        assert stats.max_ms >= stats.mean_ms
        assert stats.min_ms <= stats.median_ms <= stats.max_ms


class TestConvenienceFunction:
    """Tests for run_benchmark convenience function"""

    def test_run_benchmark_simple(self):
        spec = GEMMSpec(name="test", M=32, N=32, K=32)
        config = ExecutionConfig(warmup_iterations=2, measurement_iterations=5)

        result = run_benchmark(spec, device="cpu", config=config)

        assert result.success
        assert result.timing is not None


class TestMemoryCollector:
    """Tests for MemoryCollector"""

    def test_create_collector(self):
        collector = MemoryCollector()
        assert collector.device_index == 0

    def test_get_measurement_without_cuda(self):
        collector = MemoryCollector()
        collector.start()
        collector.stop()
        measurement = collector.get_measurement()

        assert isinstance(measurement, MemoryMeasurement)
        assert measurement.success

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_memory_tracking_cuda(self):
        collector = MemoryCollector(device_index=0)

        collector.start()
        # Allocate some memory
        x = torch.randn(1000, 1000, device="cuda")
        y = torch.matmul(x, x)
        del x, y
        collector.stop()

        measurement = collector.get_measurement()
        assert measurement.success
        assert measurement.peak_allocated_bytes > 0


class TestPowerCollector:
    """Tests for PowerCollector"""

    def test_create_collector(self):
        collector = PowerCollector(device_index=0, use_nvml=False)
        assert collector.sampling_interval_ms == 100.0

    def test_get_measurement_no_gpu(self):
        # Test graceful handling when no GPU
        collector = PowerCollector(device_index=0, use_nvml=False)
        collector.start()
        import time
        time.sleep(0.05)  # Brief sleep
        collector.stop()

        measurement = collector.get_measurement()
        assert isinstance(measurement, PowerMeasurement)
        # May or may not have samples depending on nvidia-smi availability


class TestWorkloadBenchmark:
    """Tests for workload (full model) benchmarks"""

    def test_run_resnet18(self):
        spec = WorkloadSpec(
            name="test_resnet",
            model_name="resnet18",
            model_source="torchvision",
            batch_size=1,
            input_shape=(3, 224, 224),
        )
        config = ExecutionConfig(warmup_iterations=1, measurement_iterations=2)
        runner = PyTorchRunner(config)

        result = runner.run(spec, device="cpu", precision=Precision.FP32)

        assert result.success
        assert result.throughput_samples_per_sec > 0
        assert result.extra['model_name'] == "resnet18"

    def test_run_mobilenet(self):
        spec = WorkloadSpec(
            name="test_mobilenet",
            model_name="mobilenet_v2",
            model_source="torchvision",
            batch_size=1,
        )
        config = ExecutionConfig(warmup_iterations=1, measurement_iterations=2)
        runner = PyTorchRunner(config)

        result = runner.run(spec, device="cpu", precision=Precision.FP32)

        assert result.success


class TestErrorHandling:
    """Tests for error handling"""

    def test_invalid_spec_type(self):
        from graphs.benchmarks.schema import BenchmarkSpec

        spec = BenchmarkSpec(name="invalid")  # Base class, not runnable
        runner = PyTorchRunner()

        with pytest.raises(ValueError, match="Unsupported spec type"):
            runner.run(spec)

    def test_invalid_model_name(self):
        spec = WorkloadSpec(
            name="test_invalid",
            model_name="nonexistent_model_xyz",
            model_source="torchvision",
        )
        config = ExecutionConfig(warmup_iterations=1, measurement_iterations=1)
        runner = PyTorchRunner(config)

        with pytest.raises((ValueError, AttributeError)):
            runner.run(spec)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
