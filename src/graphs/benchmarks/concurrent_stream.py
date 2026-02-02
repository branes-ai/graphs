"""
Concurrent Multi-Engine STREAM Benchmark (Jetson-specific)

Runs STREAM Triad on all SoC compute engines simultaneously:
  - CPU (all cores via multiprocessing)
  - GPU (via PyTorch CUDA)
  - DLA0 (via TensorRT, memory-heavy conv workload)
  - DLA1 (via TensorRT, memory-heavy conv workload)

Measures per-engine bandwidth under contention vs isolated to quantify
shared EMC bandwidth saturation.

Dependencies: multiprocessing (stdlib), numpy, time
Optional: torch (GPU), tensorrt + pycuda (DLA)
"""

import os
import time
import multiprocessing
from typing import Dict, List, Optional, Any

import numpy as np

# Use 'spawn' start method to avoid CUDA/TensorRT fork issues.
# CUDA cannot reinitialize in forked subprocesses, and TensorRT pybind11
# objects fail in forked contexts.
_mp_ctx = multiprocessing.get_context('spawn')


# ---------------------------------------------------------------------------
# Manager-based barrier (Barrier is not picklable across spawn contexts)
# ---------------------------------------------------------------------------

def _wait_barrier(ready_counter, ready_event, total):
    """Manager-based barrier: increment counter, wait for all to arrive."""
    with ready_counter.get_lock():
        ready_counter.value += 1
        arrived = ready_counter.value
    if arrived >= total:
        ready_event.set()
    else:
        ready_event.wait()


# ---------------------------------------------------------------------------
# Worker functions (each runs in its own Process via spawn)
# ---------------------------------------------------------------------------

def _pin_to_cores(core_ids: List[int]):
    """Pin current process to a set of CPU cores (Linux only)."""
    try:
        os.sched_setaffinity(0, set(core_ids))
    except (AttributeError, OSError):
        pass


def _cpu_stream_worker(
    size_mb: int,
    num_trials: int,
    scalar: float,
    ready_counter,
    ready_event,
    total_engines: int,
    result_dict: dict,
    dict_lock,
):
    """CPU multi-core STREAM Triad worker.

    Spawns sub-workers on all available cores within this process.
    Reports aggregate CPU bandwidth.
    """
    from graphs.benchmarks.numpy_benchmarks.multicore_stream import benchmark_multicore_stream

    # Wait for all engines to be ready
    _wait_barrier(ready_counter, ready_event, total_engines)

    results = benchmark_multicore_stream(
        size_mb=size_mb, num_trials=num_trials, scalar=scalar,
    )

    with dict_lock:
        result_dict['cpu'] = {
            'engine': 'cpu',
            'bandwidth_gbps': results['aggregate_bw_gbps'],
            'num_cores': results['num_cores'],
            'scaling_efficiency': results['scaling_efficiency'],
            'single_core_bw_gbps': results['single_core_bw_gbps'],
        }


def _gpu_stream_worker(
    size_mb: int,
    num_trials: int,
    scalar: float,
    ready_counter,
    ready_event,
    total_engines: int,
    result_dict: dict,
    dict_lock,
):
    """GPU STREAM Triad worker via PyTorch CUDA."""
    try:
        import torch
        if not torch.cuda.is_available():
            with dict_lock:
                result_dict['gpu'] = {
                    'engine': 'gpu',
                    'bandwidth_gbps': 0.0,
                    'error': 'CUDA not available',
                }
            _wait_barrier(ready_counter, ready_event, total_engines)
            return
    except ImportError:
        with dict_lock:
            result_dict['gpu'] = {
                'engine': 'gpu',
                'bandwidth_gbps': 0.0,
                'error': 'PyTorch not installed',
            }
        _wait_barrier(ready_counter, ready_event, total_engines)
        return

    num_elements = (size_mb * 1024 * 1024) // 4  # float32
    device = torch.device('cuda')

    b = torch.randn(num_elements, dtype=torch.float32, device=device)
    c = torch.randn(num_elements, dtype=torch.float32, device=device)
    a = torch.zeros(num_elements, dtype=torch.float32, device=device)
    q = torch.tensor(scalar, dtype=torch.float32, device=device)

    # Warmup
    for _ in range(5):
        a.copy_(b + q * c)
    torch.cuda.synchronize()

    # Wait for all engines
    _wait_barrier(ready_counter, ready_event, total_engines)

    # Timed runs
    times = []
    for _ in range(num_trials):
        torch.cuda.synchronize()
        start = time.perf_counter()
        a.copy_(b + q * c)
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append(end - start)

    bytes_per_iter = 3 * num_elements * 4
    best_time = min(times)
    bandwidth = bytes_per_iter / best_time / 1e9

    with dict_lock:
        result_dict['gpu'] = {
            'engine': 'gpu',
            'bandwidth_gbps': bandwidth,
            'best_time_s': best_time,
        }


def _dla_stream_worker(
    dla_core: int,
    precision: str,
    iterations: int,
    ready_counter,
    ready_event,
    total_engines: int,
    result_dict: dict,
    dict_lock,
):
    """DLA inference worker using a memory-heavy Conv2D model.

    Uses a large Conv2D (3->64, k3, 224x224) to maximize memory traffic.
    Reports effective bandwidth from FLOPs/latency perspective.
    """
    engine_key = f'dla{dla_core}'

    try:
        from graphs.benchmarks.tensorrt_benchmarks.trt_utils import (
            check_trt_available, build_engine_from_onnx,
            time_engine, export_pytorch_to_onnx,
        )
        check_trt_available()
    except (ImportError, RuntimeError) as e:
        with dict_lock:
            result_dict[engine_key] = {
                'engine': engine_key,
                'bandwidth_gbps': 0.0,
                'error': str(e),
            }
        _wait_barrier(ready_counter, ready_event, total_engines)
        return

    import tempfile

    # Build a memory-heavy Conv2D model
    try:
        import torch
        import torch.nn as nn

        class HeavyConv(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
                self.bn = nn.BatchNorm2d(64)
                self.relu = nn.ReLU()

            def forward(self, x):
                return self.relu(self.bn(self.conv(x)))

        model = HeavyConv()
        input_shape = (1, 3, 224, 224)

        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
            onnx_path = f.name

        export_pytorch_to_onnx(model, input_shape, onnx_path)
        engine = build_engine_from_onnx(
            onnx_path, dla_core=dla_core, precision=precision,
            gpu_fallback=True, input_shape=input_shape,
        )

        # Warmup
        time_engine(engine, warmup=5, iterations=5)

        # Wait for all engines
        _wait_barrier(ready_counter, ready_event, total_engines)

        # Timed runs
        timing = time_engine(engine, warmup=0, iterations=iterations)

        # Estimate memory bandwidth from Conv2D data movement
        # Input: 1*3*224*224*2 bytes (fp16), Output: 1*64*224*224*2 bytes
        # Weights: 64*3*3*3*2 bytes
        input_bytes = 1 * 3 * 224 * 224 * 2
        output_bytes = 1 * 64 * 224 * 224 * 2
        weight_bytes = 64 * 3 * 3 * 3 * 2
        total_bytes = input_bytes + output_bytes + weight_bytes

        bandwidth = total_bytes / (timing['median_ms'] / 1000.0) / 1e9

        with dict_lock:
            result_dict[engine_key] = {
                'engine': engine_key,
                'bandwidth_gbps': bandwidth,
                'latency_ms': timing['median_ms'],
                'throughput_fps': 1000.0 / timing['median_ms'] if timing['median_ms'] > 0 else 0,
            }

    except Exception as e:
        # Still need to participate in barrier
        try:
            _wait_barrier(ready_counter, ready_event, total_engines)
        except Exception:
            pass
        with dict_lock:
            result_dict[engine_key] = {
                'engine': engine_key,
                'bandwidth_gbps': 0.0,
                'error': str(e),
            }
    finally:
        if 'onnx_path' in locals() and os.path.exists(onnx_path):
            os.unlink(onnx_path)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def _detect_available_engines() -> List[str]:
    """Detect which engines are available on this platform."""
    engines = ['cpu']  # CPU always available

    try:
        import torch
        if torch.cuda.is_available():
            engines.append('gpu')
    except ImportError:
        pass

    try:
        from .tensorrt_benchmarks.trt_utils import check_trt_available, get_dla_core_count
        check_trt_available()
        num_dla = get_dla_core_count()
        for i in range(num_dla):
            engines.append(f'dla{i}')
    except (ImportError, RuntimeError):
        pass

    return engines


def _run_engine_isolated(
    engine: str,
    size_mb: int,
    num_trials: int,
    scalar: float,
    dla_precision: str,
    dla_iterations: int,
) -> Dict[str, Any]:
    """Run a single engine in isolation and return its result."""
    manager = _mp_ctx.Manager()
    result_dict = manager.dict()
    dict_lock = manager.Lock()
    ready_counter = _mp_ctx.Value('i', 0)
    ready_event = _mp_ctx.Event()

    if engine == 'cpu':
        target = _cpu_stream_worker
        args = (size_mb, num_trials, scalar, ready_counter, ready_event, 1, result_dict, dict_lock)
    elif engine == 'gpu':
        target = _gpu_stream_worker
        args = (size_mb, num_trials, scalar, ready_counter, ready_event, 1, result_dict, dict_lock)
    elif engine.startswith('dla'):
        core = int(engine[3:])
        target = _dla_stream_worker
        args = (core, dla_precision, dla_iterations, ready_counter, ready_event, 1, result_dict, dict_lock)
    else:
        return {'engine': engine, 'bandwidth_gbps': 0.0, 'error': f'Unknown engine: {engine}'}

    p = _mp_ctx.Process(target=target, args=args)
    p.start()
    p.join(timeout=120)

    if p.is_alive():
        p.terminate()
        return {'engine': engine, 'bandwidth_gbps': 0.0, 'error': 'Timed out'}

    return dict(result_dict.get(engine, {'engine': engine, 'bandwidth_gbps': 0.0, 'error': 'No result'}))


def benchmark_concurrent_engines(
    size_mb: int = 256,
    num_trials: int = 10,
    scalar: float = 3.14159,
    engines: Optional[List[str]] = None,
    dla_precision: str = "fp16",
    dla_iterations: int = 100,
) -> Dict[str, Any]:
    """Run STREAM/inference on all engines simultaneously.

    Phase 1: Run each engine in isolation to get baseline bandwidth.
    Phase 2: Run all engines concurrently to measure contention.
    Phase 3: Compute contention ratios.

    Args:
        size_mb: Array size per engine in MB (CPU/GPU STREAM).
        num_trials: Number of timed iterations for STREAM.
        scalar: STREAM scalar value.
        engines: List of engines to test (default: auto-detect).
        dla_precision: DLA precision ("fp16" or "int8").
        dla_iterations: Number of DLA inference iterations.

    Returns:
        Dict with:
            engines: List of engine names
            isolated: Dict[engine] -> {bandwidth_gbps, ...}
            concurrent: Dict[engine] -> {bandwidth_gbps, ...}
            contention: Dict[engine] -> contention_ratio (concurrent/isolated)
            aggregate_concurrent_gbps: Sum of concurrent bandwidths
    """
    if engines is None:
        engines = _detect_available_engines()

    if not engines:
        return {'engines': [], 'error': 'No engines available'}

    print(f"Engines detected: {', '.join(engines)}")

    # Phase 1: Isolated runs
    print("\nPhase 1: Isolated engine benchmarks...")
    isolated = {}
    for eng in engines:
        print(f"  Running {eng} in isolation...")
        result = _run_engine_isolated(
            eng, size_mb, num_trials, scalar,
            dla_precision, dla_iterations,
        )
        isolated[eng] = result
        bw = result.get('bandwidth_gbps', 0)
        err = result.get('error')
        if err:
            print(f"    {eng}: FAILED ({err})")
        else:
            print(f"    {eng}: {bw:.1f} GB/s")

    # Phase 2: Concurrent run
    print("\nPhase 2: Concurrent engine benchmarks...")
    active_engines = [e for e in engines if isolated.get(e, {}).get('bandwidth_gbps', 0) > 0]

    if len(active_engines) < 2:
        print("  Need at least 2 working engines for concurrent test")
        return {
            'engines': engines,
            'isolated': isolated,
            'concurrent': isolated,
            'contention': {e: 1.0 for e in engines},
            'aggregate_concurrent_gbps': sum(
                r.get('bandwidth_gbps', 0) for r in isolated.values()
            ),
        }

    manager = _mp_ctx.Manager()
    result_dict = manager.dict()
    dict_lock = manager.Lock()
    ready_counter = _mp_ctx.Value('i', 0)
    ready_event = _mp_ctx.Event()
    total = len(active_engines)

    processes = []
    for eng in active_engines:
        if eng == 'cpu':
            target = _cpu_stream_worker
            args = (size_mb, num_trials, scalar, ready_counter, ready_event, total, result_dict, dict_lock)
        elif eng == 'gpu':
            target = _gpu_stream_worker
            args = (size_mb, num_trials, scalar, ready_counter, ready_event, total, result_dict, dict_lock)
        elif eng.startswith('dla'):
            core = int(eng[3:])
            target = _dla_stream_worker
            args = (core, dla_precision, dla_iterations, ready_counter, ready_event, total, result_dict, dict_lock)
        else:
            continue

        p = _mp_ctx.Process(target=target, args=args)
        processes.append((eng, p))

    for _, p in processes:
        p.start()
    for eng, p in processes:
        p.join(timeout=180)
        if p.is_alive():
            p.terminate()

    concurrent = {}
    for eng in active_engines:
        result = dict(result_dict.get(eng, {'engine': eng, 'bandwidth_gbps': 0.0, 'error': 'No result'}))
        concurrent[eng] = result
        bw = result.get('bandwidth_gbps', 0)
        err = result.get('error')
        if err:
            print(f"  {eng}: FAILED ({err})")
        else:
            print(f"  {eng}: {bw:.1f} GB/s")

    # Include engines that failed isolation in concurrent results
    for eng in engines:
        if eng not in concurrent:
            concurrent[eng] = isolated.get(eng, {'engine': eng, 'bandwidth_gbps': 0.0})

    # Phase 3: Compute contention ratios
    contention = {}
    for eng in engines:
        iso_bw = isolated.get(eng, {}).get('bandwidth_gbps', 0)
        con_bw = concurrent.get(eng, {}).get('bandwidth_gbps', 0)
        if iso_bw > 0:
            contention[eng] = con_bw / iso_bw
        else:
            contention[eng] = 0.0

    aggregate = sum(r.get('bandwidth_gbps', 0) for r in concurrent.values())

    return {
        'engines': engines,
        'isolated': isolated,
        'concurrent': concurrent,
        'contention': contention,
        'aggregate_concurrent_gbps': aggregate,
    }


def print_concurrent_results(results: Dict[str, Any], theoretical_bw_gbps: float = 0.0):
    """Print concurrent engine benchmark results."""
    engines = results.get('engines', [])
    isolated = results.get('isolated', {})
    concurrent = results.get('concurrent', {})
    contention = results.get('contention', {})
    aggregate = results.get('aggregate_concurrent_gbps', 0)

    print(f"\nConcurrent Engine Bandwidth:")
    print(f"  {'Engine':<12} {'Isolated':>12} {'Concurrent':>12} {'Contention':>12}")
    print("  " + "-" * 50)

    for eng in engines:
        iso_bw = isolated.get(eng, {}).get('bandwidth_gbps', 0)
        con_bw = concurrent.get(eng, {}).get('bandwidth_gbps', 0)
        cont = contention.get(eng, 0)

        iso_str = f"{iso_bw:.1f} GB/s" if iso_bw > 0 else "N/A"
        con_str = f"{con_bw:.1f} GB/s" if con_bw > 0 else "N/A"
        cont_str = f"{cont:.2f}x" if iso_bw > 0 else "N/A"

        print(f"  {eng:<12} {iso_str:>12} {con_str:>12} {cont_str:>12}")

    print("  " + "-" * 50)
    print(f"  {'Aggregate':<12} {'':>12} {aggregate:>10.1f} GB/s")

    if theoretical_bw_gbps > 0:
        util = aggregate / theoretical_bw_gbps * 100
        print(f"  Theoretical LPDDR5:  {theoretical_bw_gbps:.1f} GB/s")
        print(f"  EMC Utilization:     {util:.1f}%")


if __name__ == '__main__':
    print("Concurrent Multi-Engine STREAM Benchmark")
    print("=" * 60)
    results = benchmark_concurrent_engines(size_mb=256, num_trials=10)
    print_concurrent_results(results, theoretical_bw_gbps=204.8)
