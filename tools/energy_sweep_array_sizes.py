#!/usr/bin/env python3
"""
KPU Energy Advantage Ratio vs GPU for multiple systolic array sizes.

Sweeps 5 KPU array configurations (16x16, 32x32, 64x64, 128x128, 256x256)
against a fixed GPU baseline (512 TensorCore MACs, register-file sourced).

Shows how larger arrays amortize weight-load and leakage costs better
for large matmuls, but suffer worse underutilization for small ones.

Process: 5nm, FP16, weight-stationary dataflow.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ---------------------------------------------------------------------------
# Physical constants (5nm process, FP16)
# ---------------------------------------------------------------------------
PROCESS_NODE_NM = 5
PRECISION_BYTES = 2  # FP16

# Register file energy at 5nm (scaled from 7nm reference 0.3 pJ/byte)
RF_ENERGY_PJ_PER_BYTE = 0.3 * (PROCESS_NODE_NM / 7.0)  # 0.214 pJ/byte

# Scratchpad energy (~1.3x register file -- larger but no tags)
SCRATCHPAD_ENERGY_PJ_PER_BYTE = RF_ENERGY_PJ_PER_BYTE * 1.3  # 0.279 pJ/byte

CLOCK_GHZ = 1.5
CYCLE_NS = 1.0 / CLOCK_GHZ  # 0.667 ns

# Leakage per MAC unit per nanosecond (same 5nm process for both)
LEAKAGE_PJ_PER_UNIT_PER_NS = 0.08

# Power gating: inactive PEs retain this fraction of active leakage.
# Header switches typically achieve 90-99% leakage reduction;
# 5% residual is a conservative, widely-accepted estimate.
POWER_GATE_RESIDUAL = 0.05
LEAKAGE_INACTIVE_PJ = LEAKAGE_PJ_PER_UNIT_PER_NS * POWER_GATE_RESIDUAL

# ---------------------------------------------------------------------------
# GPU model constants (512 TensorCore MACs, register-file sourced)
# ---------------------------------------------------------------------------
GPU_MACS = 512
GPU_TENSOR_CORE_SIZE = 64  # 4x4x4 MACs per tensor core instruction

# Per-MAC energy components (pJ)
GPU_COMPUTE_PJ = 1.5 * 0.5 * 0.85      # 0.638 pJ  (5nm FP16 tensor core)
GPU_RF_BYTES_PER_MAC = 128.0 / 64       # 2.0 B/MAC
GPU_RF_ENERGY_PJ = GPU_RF_BYTES_PER_MAC * RF_ENERGY_PJ_PER_BYTE  # 0.429 pJ
GPU_INSTR_PER_MAC_PJ = 10.0 / GPU_TENSOR_CORE_SIZE               # 0.156 pJ
GPU_SCHED_PER_MAC_PJ = 0.04                                       # 0.040 pJ
GPU_DYNAMIC_PJ_PER_MAC = (GPU_COMPUTE_PJ + GPU_RF_ENERGY_PJ +
                           GPU_INSTR_PER_MAC_PJ + GPU_SCHED_PER_MAC_PJ)

# ---------------------------------------------------------------------------
# KPU model constants (per-MAC, invariant to array size)
# ---------------------------------------------------------------------------
KPU_COMPUTE_PJ = 0.15         # pJ  (hardwired systolic MAC, no instr decode)
KPU_PE_FORWARD_PJ = 0.05      # pJ  (PE-to-PE local wire, ~100um at 5nm)
KPU_WEIGHT_LOAD_PJ = SCRATCHPAD_ENERGY_PJ_PER_BYTE * PRECISION_BYTES  # 0.557 pJ/element
KPU_ACT_INJECT_PJ = SCRATCHPAD_ENERGY_PJ_PER_BYTE * PRECISION_BYTES   # 0.557 pJ/element
KPU_CONTROL_PER_MAC_PJ = 0.015  # pJ  (domain-flow distributed CAM)


def gpu_energy(N):
    """Total energy (pJ) for NxN matmul on GPU with 512 TensorCore MACs."""
    total_macs = N ** 3
    e_dynamic = total_macs * GPU_DYNAMIC_PJ_PER_MAC
    cycles = total_macs / GPU_MACS
    time_ns = cycles * CYCLE_NS
    e_leakage = GPU_MACS * LEAKAGE_PJ_PER_UNIT_PER_NS * time_ns
    return e_dynamic + e_leakage


def kpu_energy(N, array_dim):
    """
    Total energy (pJ) for NxN matmul on KPU with array_dim x array_dim
    systolic array, weight-stationary dataflow.

    Per-tile execution model with power-gated leakage:
      - Pipeline fill/drain uses actual tile dimensions (partial tiles cheap)
      - Activation injection uses actual per-tile k_dim
      - Leakage charged only for active PEs per tile (domain-flow
        architecture enables deterministic power gating of unused PEs)
    """
    total_macs = N ** 3
    A = array_dim

    # --- dynamic energy (scales purely with MACs) ---
    e_compute = total_macs * KPU_COMPUTE_PJ
    e_forward = total_macs * KPU_PE_FORWARD_PJ
    e_control = total_macs * KPU_CONTROL_PER_MAC_PJ

    # Weight load: N*N weight elements, each loaded once from scratchpad
    e_weight = N * N * KPU_WEIGHT_LOAD_PJ

    # --- per-tile execution time, activation injection, and leakage ---
    n_full_k = N // A
    rem_k = N % A
    k_dims = [A] * n_full_k + ([rem_k] if rem_k > 0 else [])

    n_full_n = N // A
    rem_n = N % A
    n_dims = [A] * n_full_n + ([rem_n] if rem_n > 0 else [])

    total_act_injections = 0
    e_leakage = 0.0

    for kd in k_dims:
        for nd in n_dims:
            # Stream all M=N activation rows through this tile
            stream_cycles = N
            # Pipeline fill + drain: depth is k_dim (reduction dimension)
            fill_drain = 2 * max(kd - 1, 0)
            tile_cycles = stream_cycles + fill_drain

            # Activation injection: N rows x kd elements per row
            total_act_injections += N * kd

            # Power-gated leakage:
            #   active PEs (kd*nd): full leakage
            #   inactive PEs (A^2 - kd*nd): residual leakage through power switches
            tile_time_ns = tile_cycles * CYCLE_NS
            active_pes = kd * nd
            inactive_pes = A * A - active_pes
            e_leakage += (active_pes * LEAKAGE_PJ_PER_UNIT_PER_NS +
                          inactive_pes * LEAKAGE_INACTIVE_PJ) * tile_time_ns

    e_act = total_act_injections * KPU_ACT_INJECT_PJ

    # Result extraction: N*N output elements read back
    e_result = N * N * KPU_ACT_INJECT_PJ

    e_dynamic = e_compute + e_forward + e_control + e_weight + e_act + e_result

    return e_dynamic + e_leakage


def main():
    # Sweep parameters
    dims = np.array([4, 8, 16, 32, 48, 64, 96, 128, 192, 256,
                     384, 512, 768, 1024, 1536, 2048, 3072, 4096])
    array_sizes = [16, 32, 64, 128, 256]

    # Compute GPU baseline
    gpu_energies = np.array([gpu_energy(n) for n in dims])

    # Compute KPU for each array size
    kpu_energies = {}
    ratios = {}
    for a in array_sizes:
        kpu_e = np.array([kpu_energy(n, a) for n in dims])
        kpu_energies[a] = kpu_e
        ratios[a] = gpu_energies / kpu_e

    # -----------------------------------------------------------------------
    # Plot
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 7))

    colors = {
        16:  "#E74C3C",   # red
        32:  "#E67E22",   # orange
        64:  "#27AE60",   # green
        128: "#2E86C1",   # blue
        256: "#8E44AD",   # purple
    }
    markers = {16: "v", 32: "^", 64: "D", 128: "s", 256: "o"}

    for a in array_sizes:
        label = f"{a}x{a}  ({a*a:,} PEs)"
        ax.semilogx(dims, ratios[a], marker=markers[a], color=colors[a],
                     linewidth=2.2, markersize=6, label=label,
                     markeredgecolor="white", markeredgewidth=0.5)

    # Parity line
    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    ax.text(dims[-1] * 1.05, 1.0, "parity", va="center", fontsize=8,
            color="gray")

    # Shade advantage / disadvantage zones
    ymin, ymax = ax.get_ylim()
    ax.fill_between(dims, 1.0, ymax, alpha=0.04, color="#27AE60")
    ax.fill_between(dims, ymin, 1.0, alpha=0.04, color="#E74C3C")
    ax.text(dims[0] * 0.85, ymax * 0.92, "KPU more energy-efficient",
            fontsize=10, color="#27AE60", fontweight="bold", ha="left",
            va="top")
    ax.text(dims[0] * 0.85, min(ymin * 1.1, 0.15),
            "GPU more energy-efficient",
            fontsize=10, color="#E74C3C", fontweight="bold", ha="left",
            va="bottom")

    # Mark where each array becomes fully utilized (N = array_dim)
    for a in array_sizes:
        if a in dims:
            idx = list(dims).index(a)
        else:
            idx = np.searchsorted(dims, a)
            if idx >= len(dims):
                continue
        r_at_full = ratios[a][idx]
        ax.plot(a, r_at_full, marker="*", markersize=14, color=colors[a],
                markeredgecolor="black", markeredgewidth=0.8, zorder=5)

    # Add annotation for the star markers
    ax.annotate("* = array fully utilized\n  (N = array dim)",
                xy=(256, ratios[256][list(dims).index(256)]),
                xytext=(400, ratios[256][list(dims).index(256)] + 1.2),
                fontsize=8, color="gray",
                arrowprops=dict(arrowstyle="->", color="gray", alpha=0.5))

    # Asymptotic ratio annotation
    asym = GPU_DYNAMIC_PJ_PER_MAC / (KPU_COMPUTE_PJ + KPU_PE_FORWARD_PJ +
                                      KPU_CONTROL_PER_MAC_PJ)
    ax.axhline(y=asym, color="gray", linestyle=":", linewidth=0.8, alpha=0.4)
    ax.text(dims[-1] * 1.05, asym, f"asymptotic\nlimit {asym:.1f}x",
            va="center", fontsize=7, color="gray")

    ax.set_xlabel("Matrix Dimension N  (NxN x NxN matmul)", fontsize=12)
    ax.set_ylabel("GPU Energy / KPU Energy  (higher = KPU wins)", fontsize=12)
    ax.set_title(
        "KPU Energy Advantage vs GPU (512 TensorCore MACs)\n"
        "Systolic array size sweep -- FP16, 5nm, weight-stationary dataflow",
        fontsize=13, fontweight="bold")
    ax.legend(title="KPU Array Size", fontsize=9, title_fontsize=10,
              loc="center right")
    ax.grid(True, which="both", alpha=0.25)
    ax.set_xlim(dims[0] * 0.8, dims[-1] * 1.3)

    # Force x-axis ticks at power-of-2 dimensions
    xticks = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(x) for x in xticks])
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())

    plt.tight_layout()
    plt.savefig("energy_advantage_array_sweep.png", dpi=150,
                bbox_inches="tight")
    plt.savefig("energy_advantage_array_sweep.svg", bbox_inches="tight")
    print("Saved: energy_advantage_array_sweep.png")
    print("Saved: energy_advantage_array_sweep.svg")

    # -----------------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------------
    header = f"{'N':>6}"
    for a in array_sizes:
        header += f"  {a}x{a:>3}{'':>3}"
    print("\n" + "=" * (8 + 11 * len(array_sizes)))
    print("KPU Energy Advantage Ratio (GPU_energy / KPU_energy)")
    print("=" * (8 + 11 * len(array_sizes)))
    print(header)
    print("-" * (8 + 11 * len(array_sizes)))
    for i, n in enumerate(dims):
        row = f"{n:>6}"
        for a in array_sizes:
            r = ratios[a][i]
            if r >= 1.0:
                row += f"  {r:>6.1f}x  "
            else:
                row += f"  {r:>6.2f}x  "
        print(row)
    print("=" * (8 + 11 * len(array_sizes)))

    # Print crossover points
    print("\nCrossover points (N where KPU becomes more efficient than GPU):")
    for a in array_sizes:
        r = ratios[a]
        crossover_idx = np.argmax(r >= 1.0)
        if r[crossover_idx] >= 1.0:
            print(f"  {a}x{a} array: N ~ {dims[crossover_idx]}")
        else:
            print(f"  {a}x{a} array: never crosses (max ratio = {r.max():.2f}x)")

    print(f"\nAsymptotic advantage (dynamic energy only): {asym:.1f}x")
    print(f"  GPU dynamic/MAC: {GPU_DYNAMIC_PJ_PER_MAC:.3f} pJ")
    print(f"  KPU dynamic/MAC: {KPU_COMPUTE_PJ + KPU_PE_FORWARD_PJ + KPU_CONTROL_PER_MAC_PJ:.3f} pJ")

    plt.show()


if __name__ == "__main__":
    main()
