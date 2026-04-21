#!/usr/bin/env python
"""
TDP feasibility check (first-principles ALU dynamic power vs. stated TDP).

Computes, for each SKU:
    P_alu_dynamic = num_PEs * sustained_clock_hz * mac_energy_per_precision

and compares against the stated TDP at the default thermal profile.
ALU budget is taken as a fraction of TDP (default 0.65), reflecting
typical memory + leakage + NoC + controller overhead in a well-
balanced SoC. Reports process node and includes a standard 1-bit
full-adder reference energy so the numbers can be sanity-checked
against circuit fundamentals.

Exit code: 0 if all SKUs pass, 1 if any SKU is infeasible.

Usage:
    ./cli/check_tdp_feasibility.py
    ./cli/check_tdp_feasibility.py --alu-fraction 0.50   # conservative
    ./cli/check_tdp_feasibility.py --hardware kpu_t128 kpu_t256
    ./cli/check_tdp_feasibility.py --precision fp16 --format json
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

_repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo_root))

from graphs.hardware.mappers import get_mapper_by_name  # noqa: E402
from graphs.hardware.resource_model import Precision  # noqa: E402


# Representative 1-bit CMOS full-adder dynamic energy per process node
# at nominal Vdd. Published ranges; actual values depend on cell library
# (HD / HPM / LP), voltage scaling, and layout. Used as a calibration
# reference only - any MAC energy above 2x the full-adder energy per
# bit of precision is generally realistic; energies below 0.5x are
# aggressive/implausible. See ITRS/IRDS and Horowitz "Computing's
# Energy Problem" (ISSCC 2014) for process-node energy trends.
FULL_ADDER_ENERGY_PJ_BY_PROCESS = {
    28: 0.025,
    22: 0.018,
    16: 0.010,
    14: 0.009,
    12: 0.007,
    10: 0.006,
    8:  0.005,
    7:  0.004,
    5:  0.003,
    4:  0.0025,
    3:  0.002,
}


def full_adder_energy_pj(process_node_nm: int) -> float:
    """Return the reference full-adder energy at the nearest process node."""
    if process_node_nm in FULL_ADDER_ENERGY_PJ_BY_PROCESS:
        return FULL_ADDER_ENERGY_PJ_BY_PROCESS[process_node_nm]
    # Nearest neighbor by numeric distance
    nearest = min(FULL_ADDER_ENERGY_PJ_BY_PROCESS.keys(),
                  key=lambda k: abs(k - process_node_nm))
    return FULL_ADDER_ENERGY_PJ_BY_PROCESS[nearest]


@dataclass
class FeasibilityRow:
    sku: str
    category: str
    process_node_nm: int
    full_adder_pj: float
    total_pes: int
    pe_array: str
    sustained_clock_ghz: float
    precision: str
    mac_energy_pj: float
    peak_tops: float
    alu_power_w: float
    tdp_w: float
    alu_budget_w: float
    alu_fraction_of_tdp: float
    feasible: bool
    overshoot: float  # alu_power_w / alu_budget_w
    notes: str = ""


def _mac_energy_pj(rm, precision: Precision) -> Optional[float]:
    """Return mac_energy (pJ/MAC) for the requested precision if modeled."""
    tem = getattr(rm, "tile_energy_model", None)
    if tem is not None:
        if precision == Precision.INT8 and hasattr(tem, "mac_energy_int8"):
            return tem.mac_energy_int8 * 1e12
        if precision == Precision.BF16 and hasattr(tem, "mac_energy_bf16"):
            return tem.mac_energy_bf16 * 1e12
        if precision == Precision.FP16 and hasattr(tem, "mac_energy_bf16"):
            return tem.mac_energy_bf16 * 1e12
        if precision == Precision.FP32 and hasattr(tem, "mac_energy_fp32"):
            return tem.mac_energy_fp32 * 1e12
    # Fall back to derived per-precision energy from ComputeFabric if present
    fabrics = getattr(rm, "compute_fabrics", []) or []
    for f in fabrics:
        if hasattr(f, "energy_per_flop_fp32"):
            base = f.energy_per_flop_fp32 * 1e12
            scale = f.energy_scaling.get(precision, 1.0) if hasattr(f, "energy_scaling") else 1.0
            return base * scale * 2.0  # * 2 for per-MAC (2 ops / MAC)
    return None


def check_sku(
    sku: str,
    precision: Precision = Precision.INT8,
    alu_fraction_of_tdp: float = 0.65,
) -> Optional[FeasibilityRow]:
    """Compute feasibility for a single SKU. Returns None if SKU unavailable."""
    mapper = get_mapper_by_name(sku)
    if mapper is None:
        return None
    rm = mapper.resource_model

    tp = rm.thermal_operating_points.get(rm.default_thermal_profile)
    if tp is None:
        return None
    tdp = tp.tdp_watts

    # Extract PE count + clock from the default-profile compute resource.
    perf = tp.performance_specs.get(precision)
    if perf is None and Precision.INT8 in tp.performance_specs:
        perf = tp.performance_specs[Precision.INT8]
    if perf is None:
        return None
    cr = perf.compute_resource

    # Path 1: KPU-style tile-based compute resource
    spec_list = getattr(cr, "tile_specializations", None)
    if spec_list:
        spec0 = spec_list[0]
        dims = spec0.array_dimensions
        # Guard against pre-M0.5 SKU configs that use positional
        # TileSpecialization constructors which mis-assign fields
        # (notably kpu_t768.py's inline thermal-profile tile specs).
        try:
            pe_rows, pe_cols = int(dims[0]), int(dims[1])
        except (TypeError, ValueError, KeyError, IndexError):
            return FeasibilityRow(
                sku=sku, category="unknown", process_node_nm=16,
                full_adder_pj=full_adder_energy_pj(16), total_pes=0,
                pe_array="malformed", sustained_clock_ghz=0.0,
                precision=precision.value, mac_energy_pj=0.0,
                peak_tops=0.0, alu_power_w=0.0,
                tdp_w=tdp, alu_budget_w=alu_fraction_of_tdp * tdp,
                alu_fraction_of_tdp=alu_fraction_of_tdp,
                feasible=True, overshoot=0.0,
                notes="SKU has malformed TileSpecialization (positional-arg misuse); skipped",
            )
        pes_per_tile = pe_rows * pe_cols
        total_pes = cr.total_tiles * pes_per_tile
        clock = spec0.clock_domain.sustained_clock_hz
        pe_array = f"{pe_rows}x{pe_cols}"
    else:
        # Path 2: ComputeFabric-based (CPU, GPU, DSP, TPU)
        fabrics = getattr(rm, "compute_fabrics", []) or []
        if not fabrics:
            return None
        # Pick the fabric with the highest peak at this precision
        def _fabric_peak(f):
            ops_per_clock = f.ops_per_unit_per_clock.get(precision, 0)
            return f.num_units * ops_per_clock * f.core_frequency_hz
        f = max(fabrics, key=_fabric_peak)
        ops_per_clock = f.ops_per_unit_per_clock.get(precision, 0)
        # "PE count" for a compute fabric = num_units * ops_per_clock / 2 (per-MAC)
        total_pes = int(f.num_units * max(ops_per_clock, 1) / 2)
        clock = f.core_frequency_hz
        pe_array = f"{f.fabric_type}"

    mac_energy_pj = _mac_energy_pj(rm, precision)
    if mac_energy_pj is None:
        return None

    peak_ops_per_sec = rm.precision_profiles.get(precision)
    peak_tops = (peak_ops_per_sec.peak_ops_per_sec / 1e12) if peak_ops_per_sec else 0.0

    # P = peak_macs_per_sec * mac_energy_j
    peak_macs = total_pes * clock
    alu_power_w = peak_macs * (mac_energy_pj * 1e-12)

    alu_budget_w = alu_fraction_of_tdp * tdp
    feasible = alu_power_w <= alu_budget_w
    overshoot = alu_power_w / alu_budget_w if alu_budget_w > 0 else float("inf")

    # Pick representative process node from the first compute fabric
    process_nm = 16
    fabrics = getattr(rm, "compute_fabrics", []) or []
    if fabrics:
        process_nm = fabrics[0].process_node_nm

    category = "unknown"
    try:
        from graphs.hardware.mappers import get_mapper_info
        info = get_mapper_info(sku)
        if info:
            category = info.get("category", "unknown")
    except Exception:
        pass

    return FeasibilityRow(
        sku=sku,
        category=category,
        process_node_nm=process_nm,
        full_adder_pj=full_adder_energy_pj(process_nm),
        total_pes=total_pes,
        pe_array=pe_array,
        sustained_clock_ghz=clock / 1e9,
        precision=precision.value,
        mac_energy_pj=mac_energy_pj,
        peak_tops=peak_tops,
        alu_power_w=alu_power_w,
        tdp_w=tdp,
        alu_budget_w=alu_budget_w,
        alu_fraction_of_tdp=alu_fraction_of_tdp,
        feasible=feasible,
        overshoot=overshoot,
    )


DEFAULT_SKUS = [
    "Stillwater-KPU-T64",
    "Stillwater-KPU-T128",
    "Stillwater-KPU-T256",
    "Stillwater-KPU-T768",
    "Google-Coral-Edge-TPU",
    "Jetson-Orin-AGX-64GB",
    "Hailo-8",
    "Hailo-10H",
]


def _lookup_sku(short_or_full: str) -> str:
    """Map friendly IDs (kpu_t128, jetson_orin_agx_64gb) to registry keys."""
    aliases = {
        "kpu_t64": "Stillwater-KPU-T64",
        "kpu_t128": "Stillwater-KPU-T128",
        "kpu_t256": "Stillwater-KPU-T256",
        "kpu_t768": "Stillwater-KPU-T768",
        "coral_edge_tpu": "Google-Coral-Edge-TPU",
        "jetson_orin_agx_64gb": "Jetson-Orin-AGX-64GB",
        "hailo8": "Hailo-8",
        "hailo10h": "Hailo-10H",
    }
    return aliases.get(short_or_full.lower(), short_or_full)


def format_text(rows: List[FeasibilityRow], alu_fraction: float) -> str:
    out = []
    out.append(
        f"TDP feasibility check - peak ALU dynamic power vs. stated TDP "
        f"(ALU budget = {alu_fraction*100:.0f}% of TDP)"
    )
    out.append("")
    out.append(
        f"{'SKU':<26s}{'proc':>5s}{'FA_ref':>8s}"
        f"{'PEs':>9s}{'Array':>10s}{'GHz':>6s}"
        f"{'pJ/MAC':>8s}{'peak':>8s}{'ALU W':>7s}"
        f"{'TDP W':>7s}{'Budget':>8s}{'over':>6s}{'STATUS':>10s}"
    )
    out.append("-" * 118)
    for r in rows:
        status = "PASS" if r.feasible else f"INFEASIBLE"
        fa = f"{r.full_adder_pj:.3f}"
        out.append(
            f"{r.sku:<26s}"
            f"{str(r.process_node_nm)+'nm':>5s}"
            f"{fa+'pJ':>8s}"
            f"{r.total_pes:>9d}"
            f"{r.pe_array:>10s}"
            f"{r.sustained_clock_ghz:>6.2f}"
            f"{r.mac_energy_pj:>8.3f}"
            f"{r.peak_tops:>7.1f}T"
            f"{r.alu_power_w:>7.2f}"
            f"{r.tdp_w:>7.1f}"
            f"{r.alu_budget_w:>8.2f}"
            f"{r.overshoot:>5.2f}x"
            f"{status:>10s}"
        )
    out.append("")
    out.append(
        "Reference: a 1-bit CMOS full adder consumes on the order of "
        "the 'FA_ref' energy per switching activation (at nominal Vdd). "
        "An INT8 MAC requires roughly 8 full-adder equivalents plus "
        "array / register-file overhead; values below ~0.5x * 8 * FA_ref "
        "are aggressive, values above ~4x * 8 * FA_ref are standard-cell "
        "baseline."
    )
    out.append(
        "Note: the ALU budget accounts only for MAC dynamic power; the "
        "remaining TDP headroom (TDP - alu_power) covers memory hierarchy, "
        "leakage, interconnect, and controllers."
    )
    return "\n".join(out)


def format_json(rows: List[FeasibilityRow]) -> str:
    return json.dumps([asdict(r) for r in rows], indent=2)


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(
        description="First-principles TDP feasibility check across SKUs.",
    )
    ap.add_argument("--hardware", nargs="+", default=None,
                    help="SKU IDs to check (friendly or registry form). "
                         "Default: a canonical set of SKUs.")
    ap.add_argument("--precision", default="int8",
                    choices=["fp32", "bf16", "fp16", "int8"],
                    help="Precision to evaluate (default: int8).")
    ap.add_argument("--alu-fraction", type=float, default=0.65,
                    help="Fraction of TDP budgeted for ALU dynamic power "
                         "(0.50 conservative, 0.65 typical, 0.80 aggressive).")
    ap.add_argument("--format", choices=["text", "json"], default="text",
                    help="Output format (default: text).")
    ap.add_argument("--fail-on-infeasible", action="store_true",
                    help="Exit code 1 if any SKU is infeasible (for CI).")
    args = ap.parse_args(argv)

    skus = [_lookup_sku(s) for s in (args.hardware or DEFAULT_SKUS)]
    precision = Precision(args.precision)

    rows: List[FeasibilityRow] = []
    for sku in skus:
        row = check_sku(sku, precision=precision,
                        alu_fraction_of_tdp=args.alu_fraction)
        if row is None:
            print(f"warning: {sku!r} not in registry or no energy model",
                  file=sys.stderr)
            continue
        rows.append(row)

    if args.format == "text":
        print(format_text(rows, args.alu_fraction))
    else:
        print(format_json(rows))

    any_infeasible = any(not r.feasible for r in rows)
    if args.fail_on_infeasible and any_infeasible:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
