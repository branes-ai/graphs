#!/usr/bin/env python3
"""
Energy comparison: GPU (512 TensorCore MACs) vs KPU (64x64 Systolic Array)
for matrix multiplication as a function of matmul MACs.

GPU model: 512 MAC units in TensorCores, operands fetched from register file.
           Each tensor core does 4x4x4 = 64 MACs per instruction.
           Every MAC requires reading operands from the register file.

KPU model: 64x64 = 4096 PE systolic array, weight-stationary dataflow.
           Weights loaded once into PEs, activations stream through.
           PE-to-PE forwarding via short local wires (no register file per MAC).
           No instruction fetch/decode per MAC operation.

Process: 5nm for both (fair comparison).
Precision: FP16 (2 bytes per element).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ---------------------------------------------------------------------------
# Physical constants (5nm process, FP16)
# ---------------------------------------------------------------------------
PROCESS_NODE_NM = 5
PRECISION_BYTES = 2  # FP16

# Base energy per FP32 op at 5nm (from resource_model.py)
BASE_ENERGY_PJ_FP32 = 1.5  # pJ
FP16_SCALE = 0.5

# Circuit-type multipliers (from technology_profile.py)
TENSOR_CORE_MULTIPLIER = 0.85
SYSTOLIC_MAC_MULTIPLIER = 0.80

# Register file energy at 5nm: scale from 7nm reference (0.3 pJ/byte)
RF_ENERGY_PJ_PER_BYTE_7NM = 0.3
RF_ENERGY_PJ_PER_BYTE = RF_ENERGY_PJ_PER_BYTE_7NM * (PROCESS_NODE_NM / 7.0)
# = 0.214 pJ/byte at 5nm

# Scratchpad energy (40-60% of equivalent cache, from architectural_energy.py)
SCRATCHPAD_ENERGY_PJ_PER_BYTE = RF_ENERGY_PJ_PER_BYTE * 1.3  # slightly higher than RF

# ---------------------------------------------------------------------------
# Architecture parameters
# ---------------------------------------------------------------------------
# GPU: 512 MAC units, organized as 8 tensor cores (4x4x4 each)
GPU_MACS = 512
GPU_TENSOR_CORE_SIZE = 64  # MACs per tensor core op (4x4x4)
GPU_NUM_TENSOR_CORES = GPU_MACS // GPU_TENSOR_CORE_SIZE  # 8

# KPU: 64x64 systolic array = 4096 PEs
KPU_ARRAY_DIM = 64
KPU_MACS = KPU_ARRAY_DIM * KPU_ARRAY_DIM  # 4096

CLOCK_GHZ = 1.5  # GHz (same for both)
CYCLE_NS = 1.0 / CLOCK_GHZ  # ~0.667 ns

# Leakage per MAC unit per nanosecond (same process => same leakage)
LEAKAGE_PJ_PER_UNIT_PER_NS = 0.08  # pJ per MAC-unit per ns

# Power gating: inactive PEs retain this fraction of active leakage.
# Header switches typically achieve 90-99% leakage reduction;
# 5% residual is a conservative, widely-accepted estimate.
POWER_GATE_RESIDUAL = 0.05
LEAKAGE_INACTIVE_PJ = LEAKAGE_PJ_PER_UNIT_PER_NS * POWER_GATE_RESIDUAL

# ---------------------------------------------------------------------------
# GPU energy model (per MAC, FP16)
# ---------------------------------------------------------------------------
# Compute energy per MAC
GPU_COMPUTE_PJ = BASE_ENERGY_PJ_FP32 * FP16_SCALE * TENSOR_CORE_MULTIPLIER
# = 1.5 * 0.5 * 0.85 = 0.6375 pJ

# Register file access per tensor core op (4x4x4):
#   Read A tile: 4*4 elements * 2B = 32 bytes
#   Read B tile: 4*4 elements * 2B = 32 bytes
#   Read C accum: 4*4 * 2B = 32 bytes (read-modify-write)
#   Write C accum: 4*4 * 2B = 32 bytes
#   Total RF bytes per TC op: 128 bytes
#   Per MAC: 128 / 64 = 2.0 bytes/MAC
GPU_RF_BYTES_PER_MAC = 128.0 / GPU_TENSOR_CORE_SIZE  # 2.0 B/MAC
GPU_RF_ENERGY_PJ = GPU_RF_BYTES_PER_MAC * RF_ENERGY_PJ_PER_BYTE
# = 2.0 * 0.214 = 0.429 pJ

# Instruction overhead (amortized over 64 MACs per TC instruction)
GPU_INSTR_FETCH_PJ = 5.0    # pJ per instruction fetch
GPU_INSTR_DECODE_PJ = 3.0   # pJ per instruction decode
GPU_INSTR_DISPATCH_PJ = 2.0 # pJ per instruction dispatch
GPU_INSTR_TOTAL_PJ = (GPU_INSTR_FETCH_PJ + GPU_INSTR_DECODE_PJ +
                       GPU_INSTR_DISPATCH_PJ)
GPU_INSTR_PER_MAC_PJ = GPU_INSTR_TOTAL_PJ / GPU_TENSOR_CORE_SIZE
# = 10.0 / 64 = 0.156 pJ/MAC

# Warp scheduling overhead (amortized)
GPU_SCHED_PER_MAC_PJ = 0.04  # pJ/MAC

# Total GPU dynamic energy per MAC
GPU_DYNAMIC_PJ_PER_MAC = (GPU_COMPUTE_PJ + GPU_RF_ENERGY_PJ +
                           GPU_INSTR_PER_MAC_PJ + GPU_SCHED_PER_MAC_PJ)

# ---------------------------------------------------------------------------
# KPU energy model (per MAC, FP16, weight-stationary)
# ---------------------------------------------------------------------------
# Compute energy per MAC (no instruction decode per MAC)
KPU_COMPUTE_PJ = BASE_ENERGY_PJ_FP32 * FP16_SCALE * SYSTOLIC_MAC_MULTIPLIER
# = 1.5 * 0.5 * 0.80 = 0.60 pJ
# But systolic has no instruction decode overhead baked in, so use lower value
# from architectural_energy.py systolic_mac reference
KPU_COMPUTE_PJ = 0.15  # pJ (pure MAC, hardwired datapath)

# PE-to-PE local wire forwarding energy
KPU_PE_FORWARD_PJ = 0.05  # pJ per PE transfer (~100um wire at 5nm)

# Weight preload energy: loading one FP16 weight from scratchpad into PE
KPU_WEIGHT_LOAD_PJ = SCRATCHPAD_ENERGY_PJ_PER_BYTE * PRECISION_BYTES
# = 0.278 * 2 = 0.557 pJ per weight element

# Activation injection energy: injecting one FP16 activation at array edge
KPU_ACT_INJECT_PJ = SCRATCHPAD_ENERGY_PJ_PER_BYTE * PRECISION_BYTES
# = 0.557 pJ per activation element

# Domain flow control (minimal per-PE overhead, from architectural_energy.py)
KPU_CONTROL_PER_MAC_PJ = 0.015  # pJ/MAC (distributed CAM tracking)


def gpu_energy_model(N):
    """
    Energy for square NxN matmul on GPU with 512 TensorCore MACs.

    C[N,N] = A[N,N] x B[N,N]
    Total MACs = N^3

    Returns dict with energy breakdown in picojoules.
    """
    total_macs = N ** 3

    # Dynamic energy components
    e_compute = total_macs * GPU_COMPUTE_PJ
    e_regfile = total_macs * GPU_RF_ENERGY_PJ
    e_instruction = total_macs * GPU_INSTR_PER_MAC_PJ
    e_scheduling = total_macs * GPU_SCHED_PER_MAC_PJ

    # Execution time
    cycles = total_macs / GPU_MACS
    time_ns = cycles * CYCLE_NS

    # Leakage energy (all 512 units leak for entire duration)
    e_leakage = GPU_MACS * LEAKAGE_PJ_PER_UNIT_PER_NS * time_ns

    return {
        "compute": e_compute,
        "register_file": e_regfile,
        "instruction": e_instruction,
        "scheduling": e_scheduling,
        "leakage": e_leakage,
        "total_dynamic": e_compute + e_regfile + e_instruction + e_scheduling,
        "total": (e_compute + e_regfile + e_instruction +
                  e_scheduling + e_leakage),
        "total_macs": total_macs,
        "time_ns": time_ns,
    }


def kpu_energy_model(N):
    """
    Energy for square NxN matmul on KPU with 64x64 systolic array.

    Per-tile execution model with power-gated leakage:
    - Pipeline fill/drain uses actual tile dimensions (partial tiles cheap)
    - Activation injection uses actual per-tile k_dim
    - Leakage charged only for active PEs per tile; inactive PEs retain
      5% residual leakage through power switches (domain-flow architecture
      enables deterministic power gating of unused PEs)

    Returns dict with energy breakdown in picojoules.
    """
    total_macs = N ** 3
    A = KPU_ARRAY_DIM

    # How many PEs are active at peak (for utilization reporting)
    active_dim = min(N, A)
    active_pes = active_dim * active_dim
    utilization = active_pes / KPU_MACS

    # --- Dynamic energy (scales with MACs) ---
    e_compute = total_macs * KPU_COMPUTE_PJ
    e_pe_forward = total_macs * KPU_PE_FORWARD_PJ
    e_control = total_macs * KPU_CONTROL_PER_MAC_PJ

    # Weight preload: N*N weight elements, each loaded once from scratchpad
    e_weight_load = N * N * KPU_WEIGHT_LOAD_PJ

    # --- Per-tile: execution time, activation injection, leakage ---
    n_full_k = N // A
    rem_k = N % A
    k_dims = [A] * n_full_k + ([rem_k] if rem_k > 0 else [])

    n_full_n = N // A
    rem_n = N % A
    n_dims = [A] * n_full_n + ([rem_n] if rem_n > 0 else [])

    total_act_injections = 0
    e_leakage = 0.0
    total_time_ns = 0.0

    for kd in k_dims:
        for nd in n_dims:
            # Stream all M=N activation rows through this tile
            stream_cycles = N
            # Pipeline fill + drain: depth is k_dim (reduction dimension)
            fill_drain = 2 * max(kd - 1, 0)
            tile_cycles = stream_cycles + fill_drain
            tile_time_ns = tile_cycles * CYCLE_NS
            total_time_ns += tile_time_ns

            # Activation injection: N rows x kd elements per row
            total_act_injections += N * kd

            # Power-gated leakage:
            #   active PEs (kd*nd): full leakage
            #   inactive PEs (A^2 - kd*nd): residual through power switches
            active = kd * nd
            inactive = A * A - active
            e_leakage += ((active * LEAKAGE_PJ_PER_UNIT_PER_NS +
                           inactive * LEAKAGE_INACTIVE_PJ) * tile_time_ns)

    e_act_inject = total_act_injections * KPU_ACT_INJECT_PJ

    # Result extraction: N*N output elements read back
    e_result_extract = N * N * KPU_ACT_INJECT_PJ

    total_dynamic = (e_compute + e_pe_forward + e_weight_load +
                     e_act_inject + e_result_extract + e_control)

    return {
        "compute": e_compute,
        "pe_forwarding": e_pe_forward,
        "weight_load": e_weight_load,
        "act_inject": e_act_inject,
        "result_extract": e_result_extract,
        "control": e_control,
        "leakage": e_leakage,
        "total_dynamic": total_dynamic,
        "total": total_dynamic + e_leakage,
        "total_macs": total_macs,
        "time_ns": total_time_ns,
        "utilization": utilization,
    }


def main():
    # Matrix dimensions to sweep (square NxN matmul)
    dims = np.array([4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096])

    # Collect results
    gpu_results = [gpu_energy_model(n) for n in dims]
    kpu_results = [kpu_energy_model(n) for n in dims]

    total_macs = np.array([r["total_macs"] for r in gpu_results])

    gpu_total = np.array([r["total"] for r in gpu_results])
    kpu_total = np.array([r["total"] for r in kpu_results])

    gpu_dynamic = np.array([r["total_dynamic"] for r in gpu_results])
    kpu_dynamic = np.array([r["total_dynamic"] for r in kpu_results])

    gpu_leakage = np.array([r["leakage"] for r in gpu_results])
    kpu_leakage = np.array([r["leakage"] for r in kpu_results])

    # Energy per MAC
    gpu_pj_per_mac = gpu_total / total_macs
    kpu_pj_per_mac = kpu_total / total_macs

    gpu_dynamic_pj = gpu_dynamic / total_macs
    kpu_dynamic_pj = kpu_dynamic / total_macs

    gpu_leakage_pj = gpu_leakage / total_macs
    kpu_leakage_pj = kpu_leakage / total_macs

    efficiency_ratio = gpu_total / kpu_total

    # -----------------------------------------------------------------------
    # Plotting
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Energy Comparison: GPU (512 TensorCore MACs) vs KPU (64x64 Systolic Array)\n"
        "FP16 matmul, 5nm process, weight-stationary dataflow",
        fontsize=13, fontweight="bold"
    )

    # Colors
    GPU_COLOR = "#E74C3C"       # red
    GPU_DYN_COLOR = "#E74C3C"
    GPU_LEAK_COLOR = "#F1948A"
    KPU_COLOR = "#2E86C1"       # blue
    KPU_DYN_COLOR = "#2E86C1"
    KPU_LEAK_COLOR = "#85C1E9"

    # --- Plot 1: Total energy vs MACs (log-log) ---
    ax1 = axes[0, 0]
    ax1.loglog(total_macs, gpu_total * 1e-12, "o-", color=GPU_COLOR,
               linewidth=2, markersize=5, label="GPU total")
    ax1.loglog(total_macs, kpu_total * 1e-12, "s-", color=KPU_COLOR,
               linewidth=2, markersize=5, label="KPU total")
    ax1.loglog(total_macs, gpu_dynamic * 1e-12, "--", color=GPU_DYN_COLOR,
               linewidth=1, alpha=0.6, label="GPU dynamic")
    ax1.loglog(total_macs, kpu_dynamic * 1e-12, "--", color=KPU_DYN_COLOR,
               linewidth=1, alpha=0.6, label="KPU dynamic")
    ax1.set_xlabel("Total MACs")
    ax1.set_ylabel("Total Energy (J)")
    ax1.set_title("Total Energy vs Matmul Size")
    ax1.legend(fontsize=8)
    ax1.grid(True, which="both", alpha=0.3)

    # --- Plot 2: Energy per MAC vs matrix dimension ---
    ax2 = axes[0, 1]
    ax2.semilogx(dims, gpu_pj_per_mac, "o-", color=GPU_COLOR,
                 linewidth=2, markersize=5, label="GPU total/MAC")
    ax2.semilogx(dims, kpu_pj_per_mac, "s-", color=KPU_COLOR,
                 linewidth=2, markersize=5, label="KPU total/MAC")
    ax2.semilogx(dims, gpu_dynamic_pj, "^--", color=GPU_DYN_COLOR,
                 linewidth=1, alpha=0.6, markersize=4, label="GPU dynamic/MAC")
    ax2.semilogx(dims, kpu_dynamic_pj, "v--", color=KPU_DYN_COLOR,
                 linewidth=1, alpha=0.6, markersize=4, label="KPU dynamic/MAC")

    # Mark the systolic array dimension
    ax2.axvline(x=64, color="gray", linestyle=":", alpha=0.5)
    ax2.annotate("64x64 array\nfully utilized",
                 xy=(64, ax2.get_ylim()[0] if ax2.get_ylim()[0] > 0 else 0.1),
                 xytext=(90, max(kpu_pj_per_mac) * 0.6),
                 fontsize=7, color="gray",
                 arrowprops=dict(arrowstyle="->", color="gray", alpha=0.5))

    ax2.set_xlabel("Matrix Dimension N (NxN matmul)")
    ax2.set_ylabel("Energy per MAC (pJ)")
    ax2.set_title("Energy Efficiency vs Matmul Dimension")
    ax2.legend(fontsize=8)
    ax2.grid(True, which="both", alpha=0.3)

    # --- Plot 3: Energy breakdown stacked bars at N=256 ---
    ax3 = axes[1, 0]
    ref_idx = list(dims).index(256)
    gpu_r = gpu_results[ref_idx]
    kpu_r = kpu_results[ref_idx]

    # GPU breakdown (pJ per MAC)
    gpu_macs = gpu_r["total_macs"]
    gpu_bars = [
        gpu_r["compute"] / gpu_macs,
        gpu_r["register_file"] / gpu_macs,
        gpu_r["instruction"] / gpu_macs,
        gpu_r["scheduling"] / gpu_macs,
        gpu_r["leakage"] / gpu_macs,
    ]
    gpu_labels = ["Compute", "Register File", "Instruction", "Scheduling",
                  "Leakage"]
    gpu_colors = ["#E74C3C", "#F39C12", "#E67E22", "#D35400", "#F1948A"]

    # KPU breakdown (pJ per MAC)
    kpu_macs = kpu_r["total_macs"]
    kpu_bars = [
        kpu_r["compute"] / kpu_macs,
        kpu_r["pe_forwarding"] / kpu_macs,
        kpu_r["weight_load"] / kpu_macs,
        (kpu_r["act_inject"] + kpu_r["result_extract"]) / kpu_macs,
        kpu_r["control"] / kpu_macs,
        kpu_r["leakage"] / kpu_macs,
    ]
    kpu_labels = ["Compute", "PE Forwarding", "Weight Load",
                  "Act Inject+Extract", "Control", "Leakage"]
    kpu_colors = ["#2E86C1", "#3498DB", "#5DADE2", "#85C1E9", "#AED6F1",
                  "#D4E6F1"]

    x_gpu = 0
    x_kpu = 1.2
    bar_width = 0.8

    # Stacked bars for GPU
    bottom = 0
    for val, label, color in zip(gpu_bars, gpu_labels, gpu_colors):
        ax3.bar(x_gpu, val, bar_width, bottom=bottom, color=color,
                label=f"GPU: {label}", edgecolor="white", linewidth=0.5)
        if val > 0.02:
            ax3.text(x_gpu, bottom + val / 2, f"{val:.3f}",
                     ha="center", va="center", fontsize=6, color="white",
                     fontweight="bold")
        bottom += val

    # Stacked bars for KPU
    bottom = 0
    for val, label, color in zip(kpu_bars, kpu_labels, kpu_colors):
        ax3.bar(x_kpu, val, bar_width, bottom=bottom, color=color,
                label=f"KPU: {label}", edgecolor="white", linewidth=0.5)
        if val > 0.005:
            ax3.text(x_kpu, bottom + val / 2, f"{val:.3f}",
                     ha="center", va="center", fontsize=6,
                     fontweight="bold")
        bottom += val

    ax3.set_xticks([x_gpu, x_kpu])
    ax3.set_xticklabels(["GPU\n512 TensorCore MACs", "KPU\n64x64 Systolic"])
    ax3.set_ylabel("Energy per MAC (pJ)")
    ax3.set_title(f"Energy Breakdown at N=256 ({256**3/1e6:.0f}M MACs)")
    ax3.legend(fontsize=6, loc="upper right", ncol=2)
    ax3.grid(True, axis="y", alpha=0.3)

    # --- Plot 4: Efficiency ratio ---
    ax4 = axes[1, 1]
    ax4.semilogx(dims, efficiency_ratio, "D-", color="#27AE60",
                 linewidth=2, markersize=6)
    ax4.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax4.fill_between(dims, 1.0, efficiency_ratio,
                     where=(efficiency_ratio > 1.0),
                     alpha=0.15, color="#27AE60")
    ax4.fill_between(dims, 1.0, efficiency_ratio,
                     where=(efficiency_ratio < 1.0),
                     alpha=0.15, color="#E74C3C")

    for i, (d, r) in enumerate(zip(dims, efficiency_ratio)):
        ax4.annotate(f"{r:.1f}x", xy=(d, r),
                     xytext=(0, 8), textcoords="offset points",
                     ha="center", fontsize=7, fontweight="bold",
                     color="#27AE60" if r > 1 else "#E74C3C")

    ax4.set_xlabel("Matrix Dimension N (NxN matmul)")
    ax4.set_ylabel("GPU Energy / KPU Energy")
    ax4.set_title("KPU Energy Advantage Ratio")
    ax4.grid(True, which="both", alpha=0.3)

    # Annotate crossover and sweet spot
    ax4.annotate("KPU more efficient",
                 xy=(dims[-1], max(efficiency_ratio) * 0.8),
                 fontsize=8, color="#27AE60", ha="right")
    if min(efficiency_ratio) < 1.0:
        ax4.annotate("GPU more efficient",
                     xy=(dims[0], min(efficiency_ratio) * 1.05),
                     fontsize=8, color="#E74C3C", ha="left")

    plt.tight_layout()
    plt.savefig("energy_comparison_gpu_vs_kpu.png", dpi=150,
                bbox_inches="tight")
    plt.savefig("energy_comparison_gpu_vs_kpu.svg", bbox_inches="tight")
    print("Saved: energy_comparison_gpu_vs_kpu.png")
    print("Saved: energy_comparison_gpu_vs_kpu.svg")

    # Print summary table
    print("\n" + "=" * 90)
    print(f"{'N':>6} {'MACs':>12} {'GPU (pJ)':>14} {'KPU (pJ)':>14} "
          f"{'Ratio':>8} {'KPU util':>10}")
    print("=" * 90)
    for i, n in enumerate(dims):
        print(f"{n:>6} {total_macs[i]:>12.0f} "
              f"{gpu_total[i]:>14.1f} {kpu_total[i]:>14.1f} "
              f"{efficiency_ratio[i]:>8.1f}x "
              f"{kpu_results[i]['utilization']:>9.1%}")
    print("=" * 90)

    print(f"\nGPU dynamic energy per MAC: {GPU_DYNAMIC_PJ_PER_MAC:.3f} pJ")
    print(f"  - Compute (TensorCore):   {GPU_COMPUTE_PJ:.3f} pJ")
    print(f"  - Register file access:   {GPU_RF_ENERGY_PJ:.3f} pJ")
    print(f"  - Instruction (amortized):{GPU_INSTR_PER_MAC_PJ:.3f} pJ")
    print(f"  - Scheduling:             {GPU_SCHED_PER_MAC_PJ:.3f} pJ")

    print(f"\nKPU dynamic energy per MAC (large N): ~{KPU_COMPUTE_PJ + KPU_PE_FORWARD_PJ + KPU_CONTROL_PER_MAC_PJ:.3f} pJ")
    print(f"  - Compute (systolic MAC): {KPU_COMPUTE_PJ:.3f} pJ")
    print(f"  - PE-to-PE forwarding:    {KPU_PE_FORWARD_PJ:.3f} pJ")
    print(f"  - Control (domain flow):  {KPU_CONTROL_PER_MAC_PJ:.3f} pJ")
    print(f"  - Weight load + act inject: amortized over N")

    plt.show()


if __name__ == "__main__":
    main()
