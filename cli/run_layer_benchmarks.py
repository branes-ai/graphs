#!/usr/bin/env python3
"""
Run Layer 1-2 empirical benchmarks and print results.

Usage:
    python cli/run_layer_benchmarks.py
    python cli/run_layer_benchmarks.py --layers 1
    python cli/run_layer_benchmarks.py --layers 1 2
    python cli/run_layer_benchmarks.py --device cpu --precision fp32
    python cli/run_layer_benchmarks.py --output results.json

RAPL power measurement requires read access to
/sys/class/powercap/intel-rapl:*/energy_uj. To enable:
    sudo chmod a+r /sys/class/powercap/intel-rapl:*/energy_uj

NOTE: These benchmarks measure PyTorch-level throughput, not raw ALU
rate. Per-call dispatch overhead (~5-50us) dominates for small vectors.
The measured GFLOPS reflects achievable throughput through the PyTorch
runtime, not the micro-architectural peak. For true ALU-level
characterization, C/C++ microbenchmarks with intrinsics are needed.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path


def run_layer1(device: str, precision: str, verbose: bool) -> list:
    """Run Layer 1 FMA rate benchmark."""
    from graphs.benchmarks.layer1_alu import run_precision_sweep

    print("=" * 70)
    print("Layer 1: ALU / FMA Throughput")
    print("=" * 70)

    results = run_precision_sweep(
        device=device,
        precisions=[precision],
        num_elements=4096,
        num_iterations=5000,
        warmup_iterations=200,
        num_trials=5,
        enable_power=True,
    )

    for r in results:
        pj = r.extra.get("pj_per_op")
        pj_str = f"{pj:.2f} pJ/op" if pj else "(no power measurement)"
        clamped = r.extra.get("clamped_trials", 0)
        print(f"  {r.precision:6s}  {r.gflops:8.1f} GFLOPS  {pj_str}")
        if clamped > 0:
            print(f"           WARNING: {clamped}/{r.extra['num_trials']} trials clamped")
        if verbose:
            print(f"           elements={r.extra['num_elements']}, "
                  f"iters={r.extra['num_iterations']}, "
                  f"overhead={r.extra['empty_loop_overhead_ms']:.3f} ms")

    print()
    return results


def run_layer2(device: str, precision: str, verbose: bool) -> list:
    """Run Layer 2 SIMD width + register pressure benchmarks."""
    from graphs.benchmarks.layer2_register_simd import (
        run_simd_width_sweep,
        run_register_pressure_benchmark,
    )

    print("=" * 70)
    print("Layer 2: SIMD Width Scaling")
    print("=" * 70)

    widths = [1, 4, 8, 16, 64, 256, 1024, 4096]
    width_results = run_simd_width_sweep(
        device=device,
        precision=precision,
        widths=widths,
        num_iterations=5000,
        warmup_iterations=200,
        num_trials=5,
    )

    for r in width_results:
        w = r.extra["vector_width"]
        clamped = r.extra.get("clamped_trials", 0)
        flag = " (clamped)" if clamped > 0 else ""
        print(f"  width={w:5d}  {r.gflops:8.1f} GFLOPS{flag}")

    # Derive SIMD efficiency
    if len(width_results) >= 2:
        narrow = width_results[0]
        wide = width_results[-1]
        if narrow.gflops > 0:
            speedup = wide.gflops / narrow.gflops
            width_ratio = widths[-1] / widths[0]
            efficiency = speedup / width_ratio
            print(f"\n  SIMD efficiency: {efficiency:.3f} "
                  f"(speedup={speedup:.1f}x over {width_ratio}x width)")
        else:
            print("\n  SIMD efficiency: N/A (narrow vector throughput too low)")

    print()
    print("=" * 70)
    print("Layer 2: Register Pressure (ILP)")
    print("=" * 70)

    rp = run_register_pressure_benchmark(
        device=device,
        precision=precision,
        num_elements=4096,
        num_iterations=2000,
        warmup_iterations=200,
        num_trials=5,
    )

    print(f"  independent: {rp.extra['independent_gflops']:8.1f} GFLOPS")
    print(f"  dependent:   {rp.extra['dependent_gflops']:8.1f} GFLOPS")
    print(f"  ILP ratio:   {rp.extra['ilp_ratio']:.2f}x")
    print()

    return width_results + [rp]


def run_fitter(layer1_results: list, layer2_results: list, device: str) -> None:
    """Run fitters on collected results and show provenance."""
    from graphs.calibration.fitters.layer1_alu_fitter import Layer1ALUFitter
    from graphs.calibration.fitters.layer2_register_fitter import Layer2RegisterFitter
    from graphs.hardware.resource_model import HardwareResourceModel, HardwareType

    print("=" * 70)
    print("Fitter Results")
    print("=" * 70)

    # Create a minimal model to receive provenance
    model = HardwareResourceModel(
        name=device,
        hardware_type=HardwareType.CPU,
        compute_units=10, threads_per_unit=1, warps_per_unit=1,
        peak_bandwidth=75e9, l1_cache_per_unit=32768,
        l2_cache_total=25*1024*1024, main_memory=64*1024**3,
        energy_per_flop_fp32=1.5e-12, energy_per_byte=25e-12,
    )

    if layer1_results:
        fitter1 = Layer1ALUFitter()
        fit1 = fitter1.fit(
            layer1_results,
            sustained_clock_hz=4.5e9,
            num_cores=10,
            hardware_name=device,
        )
        fitter1.apply_to_model(model, fit1)
        for prec, throughput in fit1.measured_throughput.items():
            ops_clk = fit1.ops_per_clock_per_core.get(prec, 0)
            pj = fit1.measured_pj_per_op.get(prec)
            pj_str = f"{pj:.2f} pJ/op" if pj else "N/A"
            print(f"  Layer 1 [{prec}]: {throughput/1e9:.1f} GOPS, "
                  f"{ops_clk:.2f} ops/clk/core, energy={pj_str}")

    if layer2_results:
        fitter2 = Layer2RegisterFitter()
        fit2 = fitter2.fit(layer2_results, hardware_name=device)
        fitter2.apply_to_model(model, fit2)
        if fit2.simd_efficiency is not None:
            print(f"  Layer 2 SIMD efficiency: {fit2.simd_efficiency:.3f}")
        if fit2.ilp_ratio is not None:
            print(f"  Layer 2 ILP ratio: {fit2.ilp_ratio:.2f}x")

    print(f"\n  Aggregate confidence: {model.aggregate_confidence()}")
    print()


def check_power_availability() -> None:
    """Report power measurement status."""
    from graphs.benchmarks.power_meter import (
        _rapl_available, _tegrastats_available, _nvml_available,
        auto_select_power_collector,
    )

    print("=" * 70)
    print("Power Measurement Status")
    print("=" * 70)
    print(f"  RAPL:       {'available' if _rapl_available() else 'unavailable (chmod a+r /sys/class/powercap/intel-rapl:*/energy_uj)'}")
    print(f"  tegrastats: {'available' if _tegrastats_available() else 'unavailable (Jetson only)'}")
    print(f"  NVML:       {'available' if _nvml_available() else 'unavailable (no discrete GPU or pynvml not installed)'}")
    collector = auto_select_power_collector("cpu")
    print(f"  Selected:   {type(collector).__name__}")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Layer 1-2 empirical benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--layers", nargs="+", type=int, default=[1, 2],
        help="Which layers to run (default: 1 2)",
    )
    parser.add_argument(
        "--device", default="cpu",
        help="Target device (default: cpu)",
    )
    parser.add_argument(
        "--precision", default="fp32",
        help="Precision (default: fp32)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Save results to JSON file",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Show extra detail per trial",
    )
    args = parser.parse_args()

    check_power_availability()

    all_results = []
    layer1_results = []
    layer2_results = []

    if 1 in args.layers:
        layer1_results = run_layer1(args.device, args.precision, args.verbose)
        all_results.extend(layer1_results)

    if 2 in args.layers:
        layer2_results = run_layer2(args.device, args.precision, args.verbose)
        all_results.extend(layer2_results)

    if layer1_results or layer2_results:
        run_fitter(layer1_results, layer2_results, args.device)

    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump([r.to_dict() for r in all_results], f, indent=2)
        print(f"Results saved to {path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
