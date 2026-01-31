"""
TensorRT utility functions for DLA benchmarking.

Provides engine building, inference timing, and per-layer profiling
for DLA and GPU targets.
"""

import os
import time
import tempfile
import numpy as np
from typing import Optional, Dict, List, Tuple, Any
from pathlib import Path

# TensorRT is only available on Jetson / systems with TRT installed
try:
    import tensorrt as trt
    TRT_AVAILABLE = True
    TRT_VERSION = trt.__version__
except ImportError:
    TRT_AVAILABLE = False
    TRT_VERSION = None

try:
    import pycuda.driver as cuda
    import pycuda.autoinit
    PYCUDA_AVAILABLE = True
except ImportError:
    PYCUDA_AVAILABLE = False


class TRTLogger(trt.ILogger if TRT_AVAILABLE else object):
    """Custom TensorRT logger that captures warnings/errors."""

    def __init__(self, min_severity=None):
        if TRT_AVAILABLE:
            super().__init__()
            if min_severity is None:
                min_severity = trt.ILogger.Severity.ERROR
            self.min_severity = min_severity
        self.messages = []

    def log(self, severity, msg):
        if severity <= self.min_severity:
            self.messages.append((severity, msg))
            # Also print errors
            if TRT_AVAILABLE and severity <= trt.ILogger.Severity.ERROR:
                print(f"[TRT] {msg}")


class LayerProfiler:
    """TensorRT IProfiler implementation for per-layer timing."""

    def __init__(self):
        self.layer_times = {}  # layer_name -> list of times in ms

    def report_layer_time(self, layer_name, ms):
        if layer_name not in self.layer_times:
            self.layer_times[layer_name] = []
        self.layer_times[layer_name].append(ms)

    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Return summary statistics per layer."""
        summary = {}
        for name, times in self.layer_times.items():
            if times:
                summary[name] = {
                    'mean_ms': sum(times) / len(times),
                    'min_ms': min(times),
                    'max_ms': max(times),
                    'count': len(times),
                }
        return summary


def check_trt_available():
    """Check if TensorRT is available and raise if not."""
    if not TRT_AVAILABLE:
        raise RuntimeError(
            "TensorRT Python bindings not found. "
            "Install TensorRT via JetPack or NVIDIA SDK."
        )
    if not PYCUDA_AVAILABLE:
        raise RuntimeError(
            "PyCUDA not found. Install with: pip install pycuda"
        )


def get_dla_core_count() -> int:
    """Return the number of DLA cores available."""
    check_trt_available()
    logger = TRTLogger()
    runtime = trt.Runtime(logger)
    return runtime.num_DLA_cores


def build_engine_from_onnx(
    onnx_path: str,
    dla_core: int = -1,
    precision: str = "fp16",
    gpu_fallback: bool = True,
    workspace_mb: int = 256,
    logger: Optional[Any] = None,
) -> Any:
    """
    Build a TensorRT engine from an ONNX model.

    Args:
        onnx_path: Path to ONNX model file.
        dla_core: DLA core to target (-1 for GPU only, 0 or 1 for DLA).
        precision: "fp16" or "int8".
        gpu_fallback: Allow layers unsupported by DLA to run on GPU.
        workspace_mb: TensorRT workspace size in MB.
        logger: Optional TRT logger instance.

    Returns:
        TensorRT ICudaEngine.
    """
    check_trt_available()

    if logger is None:
        # Use INFO when targeting DLA so TRT prints layer placement
        if dla_core >= 0:
            logger = TRTLogger(min_severity=trt.ILogger.Severity.INFO)
        else:
            logger = TRTLogger()

    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)

    # Parse ONNX
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            errors = []
            for i in range(parser.num_errors):
                errors.append(str(parser.get_error(i)))
            raise RuntimeError(
                f"ONNX parse failed: {'; '.join(errors)}"
            )

    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, workspace_mb * (1 << 20)
    )

    # Precision flags
    if precision == "fp16":
        if not builder.platform_has_fast_fp16:
            raise RuntimeError("Platform does not support fast FP16")
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision == "int8":
        if not builder.platform_has_fast_int8:
            raise RuntimeError("Platform does not support fast INT8")
        config.set_flag(trt.BuilderFlag.INT8)
        config.set_flag(trt.BuilderFlag.FP16)  # INT8 needs FP16 fallback
    else:
        raise ValueError(f"Unsupported precision: {precision}. Use 'fp16' or 'int8'.")

    # DLA configuration
    if dla_core >= 0:
        runtime = trt.Runtime(logger)
        num_cores = runtime.num_DLA_cores
        if dla_core >= num_cores:
            raise RuntimeError(
                f"DLA core {dla_core} requested but only {num_cores} available"
            )
        config.default_device_type = trt.DeviceType.DLA
        config.DLA_core = dla_core
        if gpu_fallback:
            config.set_flag(trt.BuilderFlag.GPU_FALLBACK)

    # Enable detailed profiling so engine inspector returns layer metadata
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED

    # Build engine
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("Engine build failed")

    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized_engine)
    if engine is None:
        raise RuntimeError("Engine deserialization failed")

    return engine


def build_engine_from_network_def(
    build_fn,
    input_shape: Tuple[int, ...],
    dla_core: int = -1,
    precision: str = "fp16",
    gpu_fallback: bool = True,
    workspace_mb: int = 256,
    logger: Optional[Any] = None,
) -> Any:
    """
    Build a TensorRT engine from a network definition function.

    This avoids the ONNX export round-trip for simple synthetic layers.

    Args:
        build_fn: Callable(network, input_tensor) that adds layers to network.
        input_shape: Shape of input tensor (with batch dim).
        dla_core: DLA core (-1 for GPU).
        precision: "fp16" or "int8".
        gpu_fallback: Allow GPU fallback for unsupported DLA layers.
        workspace_mb: Workspace size in MB.
        logger: Optional TRT logger.

    Returns:
        TensorRT ICudaEngine.
    """
    check_trt_available()

    if logger is None:
        logger = TRTLogger()

    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )

    # Create input tensor
    input_tensor = network.add_input(
        name="input",
        dtype=trt.float32,
        shape=input_shape,
    )

    # Let build_fn add layers
    output_tensor = build_fn(network, input_tensor)
    if output_tensor is not None:
        output_tensor.name = "output"
        network.mark_output(output_tensor)

    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, workspace_mb * (1 << 20)
    )

    # Precision
    if precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision == "int8":
        config.set_flag(trt.BuilderFlag.INT8)
        config.set_flag(trt.BuilderFlag.FP16)

    # DLA
    if dla_core >= 0:
        config.default_device_type = trt.DeviceType.DLA
        config.DLA_core = dla_core
        if gpu_fallback:
            config.set_flag(trt.BuilderFlag.GPU_FALLBACK)

    # Enable detailed profiling for layer inspection
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED

    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("Engine build failed")

    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized_engine)
    return engine


def time_engine(
    engine,
    input_shapes: Optional[Dict[str, Tuple[int, ...]]] = None,
    warmup: int = 10,
    iterations: int = 100,
) -> Dict[str, float]:
    """
    Time inference on a TensorRT engine.

    Args:
        engine: TensorRT ICudaEngine.
        input_shapes: Optional dict of input_name -> shape overrides.
        warmup: Number of warmup iterations.
        iterations: Number of timed iterations.

    Returns:
        Dict with keys: median_ms, mean_ms, min_ms, max_ms, std_ms
    """
    check_trt_available()

    context = engine.create_execution_context()

    # Allocate buffers
    bindings = []
    device_buffers = []
    host_buffers = []

    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        shape = engine.get_tensor_shape(name)
        dtype = engine.get_tensor_dtype(name)

        # Override shape if provided
        if input_shapes and name in input_shapes:
            shape = input_shapes[name]
            context.set_input_shape(name, shape)

        # Convert TRT dtype to numpy
        if dtype == trt.float32:
            np_dtype = np.float32
        elif dtype == trt.float16:
            np_dtype = np.float16
        elif dtype == trt.int8:
            np_dtype = np.int8
        elif dtype == trt.int32:
            np_dtype = np.int32
        else:
            np_dtype = np.float32

        size = 1
        for d in shape:
            size *= d
        nbytes = size * np.dtype(np_dtype).itemsize

        host_buf = cuda.pagelocked_empty(size, np_dtype)
        device_buf = cuda.mem_alloc(nbytes)

        host_buffers.append(host_buf)
        device_buffers.append(device_buf)

        context.set_tensor_address(name, int(device_buf))

    stream = cuda.Stream()

    # Fill inputs with random data
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        mode = engine.get_tensor_mode(name)
        if mode == trt.TensorIOMode.INPUT:
            np.copyto(host_buffers[i], np.random.randn(host_buffers[i].size).astype(host_buffers[i].dtype))
            cuda.memcpy_htod_async(device_buffers[i], host_buffers[i], stream)

    stream.synchronize()

    # Warmup
    for _ in range(warmup):
        context.execute_async_v3(stream_handle=stream.handle)
    stream.synchronize()

    # Timed iterations
    latencies = []
    for _ in range(iterations):
        start = cuda.Event()
        end = cuda.Event()
        start.record(stream)
        context.execute_async_v3(stream_handle=stream.handle)
        end.record(stream)
        stream.synchronize()
        latencies.append(start.time_till(end))

    # Cleanup
    for buf in device_buffers:
        buf.free()

    latencies.sort()
    return {
        'median_ms': latencies[len(latencies) // 2],
        'mean_ms': sum(latencies) / len(latencies),
        'min_ms': latencies[0],
        'max_ms': latencies[-1],
        'std_ms': np.std(latencies),
        'num_iterations': iterations,
    }


def get_layer_info(engine, verbose: bool = False) -> List[Dict[str, Any]]:
    """
    Get per-layer device placement and type info from an engine.

    TensorRT 8.5 inspector returns either:
    - JSON objects (when ProfilingVerbosity.DETAILED is set)
    - Plain quoted strings (when LAYER_NAMES_ONLY, the default)

    DLA layers are identified by:
    - "ForeignNode" in the layer name (DLA ops are wrapped as foreign nodes)
    - "DLA" appearing anywhere in the inspector output
    - "device" field in JSON containing "DLA"

    "Reformatting CopyNode" layers are GPU-side format conversions
    between DLA and GPU memory layouts.

    Returns:
        List of dicts with keys: name, type, device (DLA or GPU), precision
    """
    check_trt_available()
    import json

    try:
        inspector = engine.create_engine_inspector()
    except AttributeError:
        return [{'name': 'unknown', 'type': 'unknown', 'device': 'unknown', 'precision': 'unknown'}]

    layers = []

    for i in range(engine.num_layers):
        try:
            raw = inspector.get_layer_information(i, trt.LayerInformationFormat.JSON)

            if verbose:
                print(f"  Layer {i} raw: {raw[:200]}")

            # TRT 8.5 may return a plain quoted string instead of JSON object
            try:
                info = json.loads(raw)
            except json.JSONDecodeError:
                info = raw

            # Handle plain string case (TRT 8.5 with LAYER_NAMES_ONLY)
            if isinstance(info, str):
                name = info.strip('"').strip()
                # Detect device from layer name convention
                if 'ForeignNode' in name:
                    device = 'DLA'
                    layer_type = 'DLA_fused'
                elif 'Reformatting CopyNode' in name:
                    device = 'GPU'
                    layer_type = 'reformat'
                else:
                    device = 'GPU'
                    layer_type = 'unknown'
                layers.append({
                    'name': name,
                    'type': layer_type,
                    'device': device,
                    'precision': 'unknown',
                })
                continue

            # JSON object case (DETAILED profiling)
            name = (info.get('Name') or info.get('name')
                    or info.get('LayerName') or f'layer_{i}')

            layer_type = (info.get('LayerType') or info.get('layerType')
                         or info.get('Type') or 'unknown')

            precision = (info.get('Precision') or info.get('precision')
                        or info.get('OutputPrecision') or 'unknown')

            # Detect device from multiple signals
            raw_upper = raw.upper()
            if 'FOREIGNNODE' in name.upper().replace(' ', ''):
                device = 'DLA'
            elif 'DLA' in raw_upper:
                device = 'DLA'
            elif 'Reformatting CopyNode' in name:
                device = 'GPU'
            else:
                device = 'GPU'

            layers.append({
                'name': name,
                'type': layer_type,
                'device': device,
                'precision': precision,
            })

        except (AttributeError, RuntimeError) as e:
            layers.append({
                'name': f'layer_{i}',
                'type': 'unknown',
                'device': 'unknown',
                'precision': 'unknown',
            })

    return layers


def export_pytorch_to_onnx(
    model,
    input_shape: Tuple[int, ...],
    output_path: str,
    opset_version: int = 13,
) -> str:
    """
    Export a PyTorch model to ONNX format.

    Args:
        model: PyTorch nn.Module.
        input_shape: Input tensor shape (with batch).
        output_path: Where to save the .onnx file.
        opset_version: ONNX opset version.

    Returns:
        Path to exported ONNX file.
    """
    try:
        import torch
    except ImportError:
        raise RuntimeError("PyTorch required for ONNX export")

    model.eval()
    dummy_input = torch.randn(*input_shape)

    # Suppress torch.onnx.export diagnostic banners
    import logging
    onnx_logger = logging.getLogger('torch.onnx')
    prev_level = onnx_logger.level
    onnx_logger.setLevel(logging.ERROR)
    try:
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            opset_version=opset_version,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes=None,
        )
    finally:
        onnx_logger.setLevel(prev_level)

    return output_path
