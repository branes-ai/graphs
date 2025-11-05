#!/usr/bin/env python3
"""
Benchmark NumPy matrix multiplication to compare with our C++ implementation.
NumPy typically uses optimized BLAS libraries (OpenBLAS, MKL, or BLIS).
"""

import numpy as np
import time
import sys

def benchmark_matmul(N, dtype=np.float32, warmup=True):
    """Benchmark N×N matrix multiplication"""
    print(f"Testing {N}×{N} (dtype={dtype.__name__})...", end=" ", flush=True)

    # Create random matrices
    A = np.random.rand(N, N).astype(dtype)
    B = np.random.rand(N, N).astype(dtype)

    # Warm-up run
    if warmup:
        _ = A @ B

    # Benchmark run
    start = time.time()
    C = A @ B
    elapsed = time.time() - start

    # Calculate performance
    flops = 2.0 * N * N * N  # Each element: N multiplies + N adds
    gflops = flops / elapsed / 1e9

    # Theoretical peak for i7-12700K
    # P-cores only: 8 × 2 FMA × 8 floats × 5.0 GHz = 640 GFLOPS
    # All cores: P-cores (640) + E-cores (4×2×8×3.8=243) + HT benefit ≈ 1000-1200 GFLOPS
    theoretical_peak_pcore = 640.0  # P-cores only (conservative baseline)
    theoretical_peak_all = 1000.0   # All cores + HT (realistic for BLAS)

    efficiency_pcore = (gflops / theoretical_peak_pcore) * 100.0
    efficiency_all = (gflops / theoretical_peak_all) * 100.0

    print(f"{elapsed*1000:.2f} ms, {gflops:.1f} GFLOPS "
          f"({efficiency_pcore:.1f}% P-core, {efficiency_all:.1f}% all-core)")

    return {
        'N': N,
        'time_ms': elapsed * 1000,
        'gflops': gflops,
        'efficiency_pcore': efficiency_pcore,
        'efficiency_all': efficiency_all
    }

def print_system_info():
    """Print NumPy/BLAS configuration"""
    print("=" * 80)
    print("NumPy Matrix Multiplication Benchmark")
    print("=" * 80)
    print(f"NumPy version: {np.__version__}")

    # Try to detect BLAS library
    try:
        config = np.__config__.show()
        print("\nBLAS Configuration:")
        print(config if config else "Unable to detect BLAS library")
    except:
        pass

    # Alternative method to detect BLAS
    try:
        from numpy.distutils.system_info import get_info
        blas_info = get_info('blas_opt')
        if blas_info:
            print("\nBLAS Library Info:")
            for key, value in blas_info.items():
                print(f"  {key}: {value}")
    except:
        pass

    print("\n" + "=" * 80)
    print()

def main():
    print_system_info()

    # Test various sizes
    sizes = [128, 256, 512, 1024, 2048, 4096, 8192]

    if len(sys.argv) > 1:
        # Custom size from command line
        try:
            custom_size = int(sys.argv[1])
            sizes = [custom_size]
            print(f"Running single benchmark: {custom_size}×{custom_size}\n")
        except ValueError:
            print(f"Invalid size: {sys.argv[1]}")
            print("Usage: python benchmark_numpy.py [size]")
            sys.exit(1)
    else:
        print("Running benchmark suite...\n")

    results = []
    for size in sizes:
        result = benchmark_matmul(size)
        results.append(result)

    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"{'Size':>10} | {'Time (ms)':>12} | {'GFLOPS':>12} | {'P-core %':>10} | {'All-core %':>11}")
    print("-" * 75)

    for r in results:
        print(f"{r['N']:>10} | {r['time_ms']:>12.2f} | {r['gflops']:>12.1f} | "
              f"{r['efficiency_pcore']:>9.1f}% | {r['efficiency_all']:>10.1f}%")

    # Best performance
    best = max(results, key=lambda x: x['gflops'])
    print()
    print(f"Peak Performance: {best['gflops']:.1f} GFLOPS ({best['N']}×{best['N']})")
    print(f"Peak Efficiency:  {best['efficiency_pcore']:.1f}% (P-cores), "
          f"{best['efficiency_all']:.1f}% (all cores)")

    if best['efficiency_all'] > 80:
        print("Status: ✓ EXCELLENT")
    elif best['efficiency_all'] > 60:
        print("Status: ✓ GOOD")
    elif best['efficiency_all'] > 40:
        print("Status: ⚠ FAIR")
    else:
        print("Status: ✗ POOR")

    print("\nComparison with C++ implementation:")
    print(f"  Target for our code: ~{best['gflops'] * 0.7:.0f}-{best['gflops'] * 0.9:.0f} GFLOPS")
    print(f"  (70-90% of NumPy/BLAS performance)")

if __name__ == "__main__":
    main()
