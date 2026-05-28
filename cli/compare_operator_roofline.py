#!/usr/bin/env python3
"""Roofline speed-of-light + efficiency + energy across CPU / GPU / KPU.

For three micro-benchmarks -- a dot product and a matrix-vector product (both
memory-bound) and a square matmul (compute-bound) -- this evaluates each
operator on each architecture at its native precision and reports:

  - the architecture's *speed of light*: peak compute and peak memory bandwidth
  - per-operator latency, resource efficiency (attained / peak, separately for
    compute and bandwidth so the binding resource is obvious), and energy

All three operators are expressed as matmuls of different shape:
  dot    = (1 x K) . (K x 1)      -> arithmetic intensity ~1/bpe  (memory-bound)
  matvec = (M x K) . (K x 1)      -> arithmetic intensity ~2/bpe  (memory-bound)
  matmul = (M x K) . (K x N)      -> high arithmetic intensity     (compute-bound)

Defaults: CPU = Intel i7-12700K (FP32), GPU = Jetson Orin AGX (FP16),
KPU = Stillwater T64 (BF16). The chip's process node (from the unified
embodied-schemas ComputeProduct physical_spec) is printed per table.

Usage:
    python cli/compare_operator_roofline.py
    python cli/compare_operator_roofline.py --output operator_roofline.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from graphs.core.structures import OperationType, SubgraphDescriptor
from graphs.estimation.energy import EnergyAnalyzer
from graphs.estimation.roofline import RooflineAnalyzer
from graphs.hardware.mappers import get_mapper_by_name
from graphs.hardware.resource_model import Precision


# (label, mapper registry name, native precision)
DEFAULT_ARCHES: List[Tuple[str, str, Precision]] = [
    ("CPU", "Intel-i7-12700K", Precision.FP32),
    ("GPU", "Jetson-Orin-AGX-64GB", Precision.FP16),
    ("KPU", "Stillwater-KPU-T64", Precision.BF16),
]

# Operator shapes (M, K, N). dot/matvec are memory-bound, matmul compute-bound.
OPERATOR_SHAPES = {
    "dot":    (1, 1 << 23, 1),       # two ~8M-element vectors
    "matvec": (8192, 8192, 1),       # 8192x8192 matrix . 8192 vector
    "matmul": (2048, 2048, 2048),    # square GEMM
}


def _bpe(precision: Precision) -> int:
    return {
        Precision.FP64: 8, Precision.FP32: 4, Precision.TF32: 4,
        Precision.FP16: 2, Precision.BF16: 2, Precision.INT8: 1, Precision.FP8: 1,
    }.get(precision, 4)


def _matmul_sg(M: int, K: int, N: int, bpe: int) -> SubgraphDescriptor:
    """A (M x K) . (K x N) matmul. Both operands counted as streamed input
    bytes (no weight-stationary amortization), so all architectures see the
    same byte traffic for an apples-to-apples roofline."""
    return SubgraphDescriptor(
        subgraph_id=0,
        node_ids=["op"], node_names=["op"],
        operation_types=[OperationType.MATMUL],
        fusion_pattern="matmul",
        total_flops=2 * M * K * N,
        total_macs=M * K * N,
        total_input_bytes=(M * K + K * N) * bpe,
        total_output_bytes=M * N * bpe,
        total_weight_bytes=0,
    )


def _process_node(mapper) -> str:
    ps = getattr(mapper, "physical_spec", None)
    if ps is not None:
        name = getattr(ps, "process_node_name", None)
        if name:
            return name
        nm = getattr(ps, "process_node_nm", None)
        if nm:
            return f"{nm}nm"
    return "(unknown)"


def _fmt_time(s: float) -> str:
    if s <= 0:
        return "n/a"
    if s < 1e-6:
        return f"{s * 1e9:.1f} ns"
    if s < 1e-3:
        return f"{s * 1e6:.2f} us"
    if s < 1.0:
        return f"{s * 1e3:.3f} ms"
    return f"{s:.3f} s"


def _fmt_energy(j: float) -> str:
    if j <= 0:
        return "n/a"
    if j < 1e-6:
        return f"{j * 1e9:.1f} nJ"
    if j < 1e-3:
        return f"{j * 1e6:.2f} uJ"
    return f"{j * 1e3:.3f} mJ"


def evaluate_arch(label: str, mapper_name: str, precision: Precision) -> dict:
    """Speed-of-light + per-operator roofline/energy for one architecture."""
    mapper = get_mapper_by_name(mapper_name)
    rm = mapper.resource_model
    roof = RooflineAnalyzer(rm, precision=precision)
    energy = EnergyAnalyzer(rm, precision=precision)
    bpe = _bpe(precision)

    rows = []
    for op, (M, K, N) in OPERATOR_SHAPES.items():
        sg = _matmul_sg(M, K, N, bpe)
        lat = roof._analyze_subgraph(sg)
        report = energy.analyze(subgraphs=[sg], latencies=[lat.actual_latency])
        bn = lat.bottleneck.name.replace("_BOUND", "")
        rows.append({
            "op": op,
            "bottleneck": bn,
            "latency_s": lat.actual_latency,
            "compute_eff": lat.flops_utilization,
            "mem_eff": lat.bandwidth_utilization,
            "energy_j": report.total_energy_j,
        })
    return {
        "label": label,
        "chip": rm.name,
        "node": _process_node(mapper),
        "precision": precision.value,
        "peak_compute_tops": roof.peak_flops / 1e12,
        "peak_bw_gbps": roof.peak_bandwidth / 1e9,
        "rows": rows,
    }


def format_table(a: dict) -> str:
    unit = "TFLOPS" if a["precision"].startswith(("fp", "bf", "tf")) else "TOPS"
    lines = []
    lines.append("=" * 92)
    lines.append(
        f"{a['label']}: {a['chip']}  |  process node: {a['node']}  |  "
        f"precision: {a['precision']}"
    )
    lines.append(
        f"  speed of light:  peak compute = {a['peak_compute_tops']:.2f} {unit}"
        f"   peak memory = {a['peak_bw_gbps']:.1f} GB/s"
    )
    lines.append("-" * 92)
    hdr = (f"{'operator':<8} {'bottleneck':<10} {'latency':>11} "
           f"{'compute eff':>12} {'mem eff':>9} {'energy':>11}")
    lines.append(hdr)
    lines.append("-" * 92)
    for r in a["rows"]:
        lines.append(
            f"{r['op']:<8} {r['bottleneck']:<10} {_fmt_time(r['latency_s']):>11} "
            f"{r['compute_eff'] * 100:>11.1f}% {r['mem_eff'] * 100:>8.1f}% "
            f"{_fmt_energy(r['energy_j']):>11}"
        )
    lines.append("=" * 92)
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--output", "-o", help="Optional .md/.txt file for the tables.")
    args = p.parse_args(argv)

    results = [evaluate_arch(*spec) for spec in DEFAULT_ARCHES]
    text = "\n\n".join(format_table(a) for a in results)
    print(text)

    note = (
        "\nnote: compute eff = attained FLOP/s / nominal peak; mem eff = attained "
        "BW / peak. dot & matvec are memory-bound (high mem eff); matmul is "
        "compute-bound (high compute eff).\n"
        "      compute eff > 100% on large matmul is expected, not a bug: the "
        "nominal peak FLOPS is a conservative spec (base clock / no TF32-tensor / "
        "no MKL packing) and large GEMMs exceed it via the measurement-calibrated\n"
        "      efficiency curve. Treat the nominal peak as a calibration baseline, "
        "not a hard ceiling (see closed issue #68: i7 FP32 spec below MKL-achievable)."
    )
    print(note)

    if args.output:
        Path(args.output).write_text(text + note + "\n")
        print(f"\ninfo: wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
