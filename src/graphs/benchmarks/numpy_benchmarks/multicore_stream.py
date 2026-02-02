"""
Multi-Core CPU STREAM Benchmark

Runs STREAM Triad (a = b + q * c) on all CPU cores concurrently using
multiprocessing to measure aggregate memory bandwidth. Compares against
single-core to show scaling efficiency and EMC saturation.

Dependencies: multiprocessing (stdlib), numpy, time
"""

import os
import time
import multiprocessing
import numpy as np
from typing import Dict, List, Optional


def _pin_to_core(core_id: int):
    """Pin current process to a specific CPU core (Linux only)."""
    try:
        os.sched_setaffinity(0, {core_id})
    except (AttributeError, OSError):
        pass  # Not available on all platforms


def _stream_triad_worker(
    core_id: int,
    size_mb: int,
    num_trials: int,
    scalar: float,
    barrier: multiprocessing.Barrier,
    result_dict: dict,
    dict_lock: multiprocessing.Lock,
):
    """Single-core STREAM Triad worker.

    Pins to core, allocates private arrays, synchronizes with other workers
    via barrier, then runs timed Triad iterations.
    """
    _pin_to_core(core_id)

    num_elements = (size_mb * 1024 * 1024) // 4  # float32
    a = np.zeros(num_elements, dtype=np.float32)
    b = np.random.randn(num_elements).astype(np.float32)
    c = np.random.randn(num_elements).astype(np.float32)

    # Warmup
    for _ in range(3):
        a[:] = b + scalar * c

    # Synchronize all workers
    barrier.wait()

    # Timed runs
    times = []
    for _ in range(num_trials):
        start = time.perf_counter()
        a[:] = b + scalar * c
        end = time.perf_counter()
        times.append(end - start)

    # Bandwidth: 3 arrays * size_bytes / time
    bytes_per_iter = 3 * num_elements * 4  # 2 reads + 1 write
    best_time = min(times)
    mean_time = sum(times) / len(times)

    bandwidth_peak = bytes_per_iter / best_time / 1e9  # GB/s
    bandwidth_mean = bytes_per_iter / mean_time / 1e9

    with dict_lock:
        result_dict[core_id] = {
            'core_id': core_id,
            'bandwidth_peak_gbps': bandwidth_peak,
            'bandwidth_mean_gbps': bandwidth_mean,
            'best_time_s': best_time,
            'mean_time_s': mean_time,
        }


def benchmark_multicore_stream(
    size_mb: int = 256,
    num_trials: int = 20,
    num_cores: Optional[int] = None,
    scalar: float = 3.14159,
) -> Dict:
    """Run STREAM Triad on multiple CPU cores concurrently.

    Args:
        size_mb: Array size per core in MB.
        num_trials: Number of timed iterations per core.
        num_cores: Number of cores to use (default: all available).
        scalar: STREAM scalar value.

    Returns:
        Dict with:
            num_cores: Number of cores used
            per_core_bw: List of per-core bandwidths (GB/s)
            aggregate_bw_gbps: Sum of per-core bandwidths
            single_core_bw_gbps: Single-core baseline bandwidth
            scaling_efficiency: aggregate / (num_cores * single_core)
            size_mb: Array size per core
    """
    if num_cores is None:
        try:
            num_cores = len(os.sched_getaffinity(0))
        except AttributeError:
            num_cores = os.cpu_count() or 1

    # Phase 1: Single-core baseline
    manager = multiprocessing.Manager()
    result_dict = manager.dict()
    dict_lock = manager.Lock()
    barrier = multiprocessing.Barrier(1)

    p = multiprocessing.Process(
        target=_stream_triad_worker,
        args=(0, size_mb, num_trials, scalar, barrier, result_dict, dict_lock),
    )
    p.start()
    p.join()

    single_core_bw = result_dict.get(0, {}).get('bandwidth_peak_gbps', 0.0)

    # Phase 2: All cores concurrent
    result_dict_multi = manager.dict()
    barrier_multi = multiprocessing.Barrier(num_cores)

    processes = []
    for i in range(num_cores):
        p = multiprocessing.Process(
            target=_stream_triad_worker,
            args=(i, size_mb, num_trials, scalar, barrier_multi,
                  result_dict_multi, dict_lock),
        )
        processes.append(p)

    for p in processes:
        p.start()
    for p in processes:
        p.join()

    per_core_bw = []
    for i in range(num_cores):
        info = result_dict_multi.get(i, {})
        per_core_bw.append(info.get('bandwidth_peak_gbps', 0.0))

    aggregate_bw = sum(per_core_bw)
    ideal_aggregate = single_core_bw * num_cores
    scaling_efficiency = (aggregate_bw / ideal_aggregate) if ideal_aggregate > 0 else 0.0

    return {
        'num_cores': num_cores,
        'size_mb': size_mb,
        'single_core_bw_gbps': single_core_bw,
        'per_core_bw_gbps': per_core_bw,
        'aggregate_bw_gbps': aggregate_bw,
        'scaling_efficiency': scaling_efficiency,
        'num_trials': num_trials,
    }


def print_multicore_results(results: Dict):
    """Print multi-core STREAM results."""
    n = results['num_cores']
    print(f"\nMulti-Core STREAM Triad ({n} cores, {results['size_mb']} MB/core):")
    print(f"  Single-core baseline: {results['single_core_bw_gbps']:.1f} GB/s")
    print(f"  Aggregate ({n} cores):  {results['aggregate_bw_gbps']:.1f} GB/s")
    print(f"  Scaling efficiency:   {results['scaling_efficiency']*100:.1f}% "
          f"(ideal: {results['single_core_bw_gbps'] * n:.1f} GB/s)")
    print(f"  Per-core range:       "
          f"{min(results['per_core_bw_gbps']):.1f} - "
          f"{max(results['per_core_bw_gbps']):.1f} GB/s")


if __name__ == '__main__':
    print("Multi-Core CPU STREAM Benchmark")
    print("=" * 60)
    results = benchmark_multicore_stream(size_mb=256, num_trials=20)
    print_multicore_results(results)
