#!/usr/bin/env python3
"""Diagnose the dip in KPU energy advantage ratio at tile boundaries."""

import numpy as np

# Constants (same as sweep script)
RF_ENERGY_PJ_PER_BYTE = 0.3 * (5 / 7.0)
SCRATCHPAD_ENERGY_PJ_PER_BYTE = RF_ENERGY_PJ_PER_BYTE * 1.3
CLOCK_GHZ = 1.5
CYCLE_NS = 1.0 / CLOCK_GHZ
LEAKAGE_PJ_PER_UNIT_PER_NS = 0.08

GPU_MACS = 512
GPU_DYNAMIC_PJ_PER_MAC = 1.262

KPU_COMPUTE_PJ = 0.15
KPU_PE_FORWARD_PJ = 0.05
KPU_CONTROL_PER_MAC_PJ = 0.015
KPU_WEIGHT_LOAD_PJ = SCRATCHPAD_ENERGY_PJ_PER_BYTE * 2
KPU_ACT_INJECT_PJ = SCRATCHPAD_ENERGY_PJ_PER_BYTE * 2


def analyze(N, A):
    """Return detailed energy breakdown for NxN matmul on AxA systolic.
    Uses per-tile execution model with actual tile dimensions."""
    total_macs = N ** 3
    total_pes = A * A

    e_compute = total_macs * KPU_COMPUTE_PJ
    e_forward = total_macs * KPU_PE_FORWARD_PJ
    e_control = total_macs * KPU_CONTROL_PER_MAC_PJ
    e_weight = N * N * KPU_WEIGHT_LOAD_PJ

    # Per-tile model
    n_full_k = N // A
    rem_k = N % A
    k_dims = [A] * n_full_k + ([rem_k] if rem_k else [])

    n_full_n = N // A
    rem_n = N % A
    n_dims = [A] * n_full_n + ([rem_n] if rem_n else [])

    total_cycles = 0
    total_act_inj = 0
    total_stream = 0
    total_filldrain = 0
    e_leakage = 0  # per-tile power-gated leakage

    for kd in k_dims:
        for nd in n_dims:
            stream = N
            fd = 2 * max(kd - 1, 0)
            tile_cycles = stream + fd
            total_cycles += tile_cycles
            total_stream += stream
            total_filldrain += fd
            total_act_inj += N * kd
            # Power-gated leakage: only kd*nd PEs active during this tile
            tile_time = tile_cycles * CYCLE_NS
            e_leakage += kd * nd * LEAKAGE_PJ_PER_UNIT_PER_NS * tile_time

    n_tiles_k = len(k_dims)
    n_tiles_n = len(n_dims)
    n_tile_pairs = n_tiles_k * n_tiles_n

    e_act = total_act_inj * KPU_ACT_INJECT_PJ
    e_result = N * N * KPU_ACT_INJECT_PJ
    e_dynamic = e_compute + e_forward + e_control + e_weight + e_act + e_result

    time_ns = total_cycles * CYCLE_NS

    # GPU reference
    gpu_dyn = total_macs * GPU_DYNAMIC_PJ_PER_MAC
    gpu_cycles = total_macs / GPU_MACS
    gpu_time = gpu_cycles * CYCLE_NS
    gpu_leak = GPU_MACS * LEAKAGE_PJ_PER_UNIT_PER_NS * gpu_time
    gpu_total = gpu_dyn + gpu_leak

    kpu_total = e_dynamic + e_leakage
    ratio = gpu_total / kpu_total

    return {
        "N": N, "A": A, "total_macs": total_macs,
        "n_tiles_k": n_tiles_k, "n_tiles_n": n_tiles_n,
        "n_tile_pairs": n_tile_pairs,
        "stream_cycles": total_stream,
        "filldrain_cycles": total_filldrain,
        "effective_cycles": total_cycles, "time_ns": time_ns,
        "e_compute": e_compute, "e_forward": e_forward,
        "e_control": e_control, "e_weight": e_weight,
        "e_act": e_act, "e_result": e_result,
        "e_dynamic": e_dynamic, "e_leakage": e_leakage,
        "kpu_total": kpu_total, "gpu_total": gpu_total,
        "ratio": ratio,
        # per-MAC values
        "dynamic_per_mac": e_dynamic / total_macs,
        "leakage_per_mac": e_leakage / total_macs,
        "stream_frac": total_stream / total_cycles,
    }


def print_comparison(points, label):
    print(f"\n{'='*90}")
    print(f"  {label}")
    print(f"{'='*90}")

    headers = ["N", "MACs", "tiles_k", "tiles_n", "pairs",
               "stream", "f/d", "stream%",
               "dyn/MAC", "leak/MAC", "total/MAC", "ratio"]
    fmt =     "{:>5} {:>10} {:>7} {:>7} {:>5} {:>8} {:>8} {:>7} {:>8} {:>9} {:>9} {:>7}"
    print(fmt.format(*headers))
    print("-" * 95)

    for p in points:
        print(fmt.format(
            p["N"],
            f"{p['total_macs']:.0f}",
            p["n_tiles_k"],
            p["n_tiles_n"],
            p["n_tile_pairs"],
            f"{p['stream_cycles']:.0f}",
            f"{p['filldrain_cycles']:.0f}",
            f"{p['stream_frac']:.0%}",
            f"{p['dynamic_per_mac']:.3f}",
            f"{p['leakage_per_mac']:.3f}",
            f"{p['dynamic_per_mac']+p['leakage_per_mac']:.3f}",
            f"{p['ratio']:.2f}x",
        ))

    # Highlight the changes
    if len(points) >= 2:
        print()
        p0, p1 = points[0], points[1]
        mac_growth = p1["total_macs"] / p0["total_macs"]
        print(f"  N: {p0['N']} -> {p1['N']}")
        print(f"  MACs growth:        {mac_growth:.2f}x")
        print(f"  tile_pairs:         {p0['n_tile_pairs']} -> {p1['n_tile_pairs']}  "
              f"({p1['n_tile_pairs']/p0['n_tile_pairs']:.1f}x)")
        print(f"  stream_cycles:      {p0['stream_cycles']:.0f} -> {p1['stream_cycles']:.0f}  "
              f"({p1['stream_cycles']/p0['stream_cycles']:.1f}x)")
        print(f"  filldrain_cycles:   {p0['filldrain_cycles']:.0f} -> {p1['filldrain_cycles']:.0f}  "
              f"({p1['filldrain_cycles']/max(1,p0['filldrain_cycles']):.1f}x)")
        print(f"  total_cycles:       {p0['effective_cycles']:.0f} -> {p1['effective_cycles']:.0f}  "
              f"({p1['effective_cycles']/p0['effective_cycles']:.1f}x)")
        print(f"  e_act (inject):     {p0['e_act']:.0f} -> {p1['e_act']:.0f}  "
              f"({p1['e_act']/p0['e_act']:.2f}x)")
        print(f"  leakage/MAC:        {p0['leakage_per_mac']:.3f} -> {p1['leakage_per_mac']:.3f}  "
              f"({p1['leakage_per_mac']/p0['leakage_per_mac']:.2f}x)")
        print(f"  dynamic/MAC:        {p0['dynamic_per_mac']:.3f} -> {p1['dynamic_per_mac']:.3f}  "
              f"({p1['dynamic_per_mac']/p0['dynamic_per_mac']:.2f}x)")
        print(f"  RATIO:              {p0['ratio']:.2f}x -> {p1['ratio']:.2f}x  "
              f"(change: {p1['ratio']-p0['ratio']:+.2f})")


def main():
    # Focus on 64x64 array around the dip
    A = 64
    dims = [48, 56, 64, 72, 80, 96, 112, 128, 192, 256]
    points = [analyze(n, A) for n in dims]

    print_comparison(points[:4], f"64x64 array: leading into and just past tile boundary (N=64)")
    print_comparison([points[2], points[3]], f"64x64 array: THE DIP (N=64 -> N=72)")
    print_comparison(points[2:], f"64x64 array: full recovery trajectory")

    # Also check 32x32
    A = 32
    dims32 = [24, 28, 32, 36, 40, 48, 56, 64]
    points32 = [analyze(n, A) for n in dims32]
    print_comparison([points32[2], points32[3]], f"32x32 array: THE DIP (N=32 -> N=36)")

    # Decompose what fraction of the dip comes from leakage vs activation injection
    print(f"\n{'='*90}")
    print(f"  ROOT CAUSE DECOMPOSITION (64x64 array, N=64 vs N=72)")
    print(f"{'='*90}")
    p_before = analyze(64, 64)
    p_after = analyze(72, 64)

    mac_ratio = p_after["total_macs"] / p_before["total_macs"]
    print(f"\n  MAC growth factor: {mac_ratio:.3f}x")

    # If all energy scaled with MACs, the ratio wouldn't change.
    # The dip comes from components that scale FASTER than MACs.
    components = [
        ("compute",   "e_compute"),
        ("forwarding","e_forward"),
        ("control",   "e_control"),
        ("weight_load","e_weight"),
        ("act_inject","e_act"),
        ("result_ext","e_result"),
        ("leakage",   "e_leakage"),
    ]
    print(f"\n  {'Component':<14} {'@N=64':>12} {'@N=72':>12} {'growth':>8} {'vs MACs':>10} {'excess pJ':>12}")
    print(f"  {'-'*70}")
    total_excess = 0
    for name, key in components:
        v0 = p_before[key]
        v1 = p_after[key]
        growth = v1 / v0
        # "excess" = energy beyond what linear MAC scaling would predict
        expected = v0 * mac_ratio
        excess = v1 - expected
        total_excess += excess
        flag = " <<<" if excess > 100 else ""
        print(f"  {name:<14} {v0:>12.0f} {v1:>12.0f} {growth:>7.2f}x "
              f"{'OVER' if growth > mac_ratio else 'ok':>10} {excess:>+12.0f}{flag}")

    print(f"  {'-'*70}")
    print(f"  {'TOTAL EXCESS':<14} {'':>12} {'':>12} {'':>8} {'':>10} {total_excess:>+12.0f}")
    print(f"\n  This excess energy is why the ratio drops from "
          f"{p_before['ratio']:.2f}x to {p_after['ratio']:.2f}x")


if __name__ == "__main__":
    main()
