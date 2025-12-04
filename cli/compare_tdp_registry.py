#!/usr/bin/env python3
"""
Compare TDP estimates from physics model vs hardware registry specs.

This script loads hardware profiles from the registry and compares their
stated TDP against our physics-based TDP estimate (energy_per_op x ops/sec).

This helps identify:
1. GPUs that may be thermally constrained (spec TDP < estimated TDP)
2. GPUs with headroom (spec TDP > estimated TDP)
3. Validation of our energy model against real hardware
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from graphs.hardware.technology_profile import (
    PROCESS_NODE_BASE_ENERGY_PJ,
    CIRCUIT_TYPE_MULTIPLIER,
    get_process_base_energy_pj,
)

# Hardware specs extracted from the model files
#
# TENSOR CORE ARCHITECTURE:
# - 1 Tensor Core = 4x4 systolic array = 16 MAC units
# - Each MAC does 2 ops (multiply + accumulate) per clock in steady state
# - So 1 basic TC = 16 MACs/clock = 32 ops/clock at FP16
#
# NVIDIA's "ops_per_clock" specs are MUCH higher because:
# - They pack multiple 4x4 arrays per "tensor core"
# - At lower precision (FP8, INT8), they do 2x or 4x more ops
# - Marketing specs may include sparsity (2:4 structured sparsity = 2x)
#
# For PHYSICS-BASED TDP estimation, we need the ACTUAL MAC unit count:
#   tc_mac_units = tensor_cores * macs_per_tc
#
# NVIDIA Tensor Core generations:
# - V100 (1st gen): 4x4x4 = 64 FP16 FMAs per TC per clock (special case: processes 4x4x4 in one clock)
# - Turing (2nd gen): Similar to V100
# - A100 (3rd gen): 4x4x4 structure, but more ops via pipelining
# - H100 (4th gen): Larger arrays, ~256 FP16 FMAs per TC per clock
# - B100 (5th gen): Even larger, ~512 FP16 FMAs per TC per clock
#
# The key insight: each MAC unit consumes energy per operation regardless of
# how NVIDIA packages them into "tensor cores".

NVIDIA_GPUS = [
    # Datacenter GPUs
    {
        'name': 'B100-SXM6-192GB',
        'arch': 'Blackwell',
        'process_nm': 3,  # TSMC 4NP maps to 3nm energy
        'cuda_cores': 16896,  # 132 SMs x 128 cores
        'tensor_cores': 528,  # 132 SMs x 4 TCs
        # 5th gen TC: each TC has multiple 4x4 arrays
        # 512 FP16 FMAs/clock/TC = 512 MAC units per TC (or 32 4x4 arrays)
        'tc_macs_per_clock': 512,  # MACs per TC per clock (not ops!)
        'freq_ghz': 2.1,
        'spec_tdp_w': 1000,
    },
    {
        'name': 'H100-SXM5-80GB',
        'arch': 'Hopper',
        'process_nm': 4,  # TSMC 4N
        'cuda_cores': 16896,  # 132 SMs x 128 cores
        'tensor_cores': 528,  # 132 SMs x 4 TCs
        # 4th gen TC: 256 FP16 FMAs/clock/TC
        'tc_macs_per_clock': 256,
        'freq_ghz': 1.98,
        'spec_tdp_w': 700,
    },
    {
        'name': 'A100-SXM4-80GB',
        'arch': 'Ampere',
        'process_nm': 7,  # TSMC 7nm
        'cuda_cores': 13824,  # 108 SMs x 128 cores
        'tensor_cores': 432,  # 108 SMs x 4 TCs
        # 3rd gen TC: 256 FP16 FMAs/clock/TC
        'tc_macs_per_clock': 256,
        'freq_ghz': 1.41,
        'spec_tdp_w': 400,
    },
    {
        'name': 'V100-SXM3-32GB',
        'arch': 'Volta',
        'process_nm': 12,  # TSMC 12nm FFN
        'cuda_cores': 5120,  # 80 SMs x 64 cores
        'tensor_cores': 640,  # 80 SMs x 8 TCs
        # 1st gen TC: 4x4x4 matrix multiply = 64 FP16 FMAs/clock/TC
        'tc_macs_per_clock': 64,
        'freq_ghz': 1.53,
        'spec_tdp_w': 350,
    },
    {
        'name': 'T4-PCIe-16GB',
        'arch': 'Turing',
        'process_nm': 12,  # TSMC 12nm FFN
        'cuda_cores': 2560,  # 40 SMs x 64 cores
        'tensor_cores': 320,  # 40 SMs x 8 TCs
        # 2nd gen TC: same as V100 (64 FP16 FMAs/clock/TC)
        'tc_macs_per_clock': 64,
        'freq_ghz': 1.59,
        'spec_tdp_w': 70,
    },
]


def estimate_cuda_core_tdp(gpu: dict, precision: str = 'FP32') -> float:
    """
    Estimate TDP from CUDA cores only (FP32/FP64 workloads).

    TDP = energy_per_op x num_cores x ops_per_clock x frequency
    """
    base_energy_pj = get_process_base_energy_pj(gpu['process_nm'])
    circuit_mult = CIRCUIT_TYPE_MULTIPLIER['standard_cell']

    # Precision scaling
    precision_scale = {'FP64': 2.0, 'FP32': 1.0, 'FP16': 0.5}[precision]
    ops_per_clock = 2  # FMA = 2 ops/clock

    energy_per_op_pj = base_energy_pj * circuit_mult * precision_scale
    energy_per_op_j = energy_per_op_pj * 1e-12

    ops_per_sec = gpu['cuda_cores'] * ops_per_clock * gpu['freq_ghz'] * 1e9
    tdp_watts = energy_per_op_j * ops_per_sec

    return tdp_watts


def estimate_tensor_core_tdp(gpu: dict, precision: str = 'FP16') -> float:
    """
    Estimate TDP from Tensor Cores using physics-based MAC unit count.

    A Tensor Core is a systolic array of MAC units.
    Each MAC unit does 1 multiply-accumulate per clock.

    NVIDIA counts 1 MAC = 2 ops (1 multiply + 1 add) in their FLOPS specs.
    So if NVIDIA says "512 FP16 ops/clock/TC", that's 256 MACs/clock/TC.

    Our tc_macs_per_clock stores the MAC count (not ops count).

    Energy model:
    - Each MAC unit consumes energy for both multiply and accumulate
    - energy_per_mac = energy_for_multiply + energy_for_accumulate
    - For a fused MAC unit, this is roughly 2x the energy of a single op

    TDP = energy_per_mac x total_macs_per_second
    """
    base_energy_pj = get_process_base_energy_pj(gpu['process_nm'])
    circuit_mult = CIRCUIT_TYPE_MULTIPLIER['systolic_mac']  # Use systolic array efficiency

    # Precision scaling for energy per operation
    # Lower precision = less energy per op (smaller data paths, fewer transistors switching)
    precision_energy_scale = {
        'FP64': 2.0,    # 64-bit: 2x energy
        'FP32': 1.0,    # 32-bit: baseline
        'TF32': 0.6,    # 19-bit effective
        'BF16': 0.5,    # 16-bit
        'FP16': 0.5,    # 16-bit
        'FP8': 0.25,    # 8-bit
        'INT8': 0.25,   # 8-bit
        'INT4': 0.125,  # 4-bit
    }.get(precision, 0.5)

    # Throughput multiplier: at lower precision, same silicon can do more MACs
    # (NVIDIA packs more lanes for lower precision)
    # FP16 is our baseline (tc_macs_per_clock is defined at FP16)
    precision_throughput_mult = {
        'FP64': 0.25,   # FP64: 1/4 the MACs (needs 4x the datapath width)
        'FP32': 0.5,    # FP32: 1/2 the MACs
        'TF32': 0.5,    # TF32: same as FP32 throughput
        'BF16': 1.0,    # BF16: baseline
        'FP16': 1.0,    # FP16: baseline
        'FP8': 2.0,     # FP8: 2x MACs (half the datapath width)
        'INT8': 2.0,    # INT8: 2x MACs
        'INT4': 4.0,    # INT4: 4x MACs
    }.get(precision, 1.0)

    # Energy per MAC unit operation
    # A MAC = multiply + accumulate, so it's 2 arithmetic operations
    # base_energy_pj is energy per single op, so MAC energy = 2x
    #
    # Additionally, there's overhead for:
    # - Register file reads/writes (feeding operands, storing results)
    # - Data distribution network within the systolic array
    # - Control logic and clock distribution
    # This overhead is typically 30-50% of the raw compute energy
    datapath_overhead = 1.4  # 40% overhead for register files, control, etc.

    energy_per_op_pj = base_energy_pj * circuit_mult * precision_energy_scale
    energy_per_mac_pj = energy_per_op_pj * 2 * datapath_overhead  # MAC = 2 ops + overhead
    energy_per_mac_j = energy_per_mac_pj * 1e-12

    # Total MACs per clock across all tensor cores
    # tc_macs_per_clock is the number of MAC units per TC (at FP16 baseline)
    macs_per_clock_per_tc = gpu['tc_macs_per_clock'] * precision_throughput_mult
    total_macs_per_clock = gpu['tensor_cores'] * macs_per_clock_per_tc

    # MACs per second
    macs_per_sec = total_macs_per_clock * gpu['freq_ghz'] * 1e9

    # TDP from tensor cores
    tdp_watts = energy_per_mac_j * macs_per_sec

    return tdp_watts


def get_total_tc_mac_units(gpu: dict) -> int:
    """
    Get total MAC units across all tensor cores.

    This represents the actual physical ALU count in the tensor core fabric.
    """
    # tc_macs_per_clock represents how many MAC operations happen per clock
    # In a systolic array, this equals the number of MAC units (in steady state)
    return gpu['tensor_cores'] * gpu['tc_macs_per_clock']


def estimate_combined_tdp(gpu: dict, tc_utilization: float = 1.0) -> float:
    """
    Estimate TDP with both CUDA and Tensor cores active.

    In practice, workloads often use tensor cores primarily, but CUDA cores
    still consume some power for control/address calculation.

    Args:
        gpu: GPU spec dict
        tc_utilization: Fraction of tensor core utilization (0-1)

    Returns:
        Combined TDP estimate
    """
    cuda_tdp = estimate_cuda_core_tdp(gpu, 'FP32')
    tensor_tdp = estimate_tensor_core_tdp(gpu, 'FP16')

    # When tensor cores are active, CUDA cores typically do ~20% of their peak
    # (address calculation, control, etc.)
    cuda_background = 0.2

    combined = cuda_tdp * cuda_background + tensor_tdp * tc_utilization
    return combined


def estimate_full_chip_tdp(gpu: dict) -> dict:
    """
    Estimate full chip TDP including all power components.

    GPU power breakdown (typical):
    - Compute (ALUs): 30-40% of TDP
    - Memory subsystem (HBM/GDDR + controllers): 20-30%
    - Interconnect (NoC, NVLink): 10-15%
    - Static/leakage: 15-25%
    - I/O, voltage regulators, misc: 5-10%

    Returns dict with breakdown.
    """
    # Compute power (CUDA + Tensor cores)
    cuda_power = estimate_cuda_core_tdp(gpu, 'FP32')
    tc_power = estimate_tensor_core_tdp(gpu, 'FP16')

    # Memory subsystem power estimate
    # HBM: ~3-4 pJ/bit = 24-32 pJ/byte at full bandwidth
    # Memory power = bandwidth * energy_per_byte
    memory_bw_tb_s = {
        'Blackwell': 8.0,  # 8 TB/s HBM3e
        'Hopper': 3.35,    # 3.35 TB/s HBM3
        'Ampere': 2.0,     # 2 TB/s HBM2e
        'Volta': 0.9,      # 900 GB/s HBM2
        'Turing': 0.32,    # 320 GB/s GDDR6
    }.get(gpu['arch'], 1.0)

    # Energy per byte varies by memory type
    memory_pj_per_byte = {
        'Blackwell': 5.0,   # HBM3e
        'Hopper': 5.5,      # HBM3
        'Ampere': 6.5,      # HBM2e
        'Volta': 7.0,       # HBM2
        'Turing': 15.0,     # GDDR6
    }.get(gpu['arch'], 10.0)

    # Memory power at full bandwidth utilization
    memory_power = memory_bw_tb_s * 1e12 * memory_pj_per_byte * 1e-12  # W

    # Static/leakage power (scales with transistor count and process node)
    # Rough estimate: leakage is ~15-25% of TDP, higher for smaller nodes
    leakage_fraction = {
        3: 0.25,   # 3nm: high leakage
        4: 0.22,
        5: 0.20,
        7: 0.18,
        12: 0.15,
        14: 0.15,
    }.get(gpu['process_nm'], 0.18)

    # Interconnect power (NoC, L2, register files, etc.)
    # Roughly proportional to compute units
    interconnect_power_per_sm = 0.5  # ~0.5W per SM for interconnect
    num_sms = gpu['cuda_cores'] // (128 if gpu['arch'] in ['Ampere', 'Hopper', 'Blackwell'] else 64)
    interconnect_power = num_sms * interconnect_power_per_sm

    # I/O and misc (voltage regulators, PCIe/NVLink PHYs, etc.)
    io_misc_power = 15.0  # ~15W baseline

    # Total dynamic power
    dynamic_power = cuda_power + tc_power + memory_power + interconnect_power + io_misc_power

    # Add leakage (leakage_fraction of total, so total = dynamic / (1 - leakage_fraction))
    total_power = dynamic_power / (1 - leakage_fraction)
    leakage_power = total_power * leakage_fraction

    return {
        'cuda_power': cuda_power,
        'tc_power': tc_power,
        'memory_power': memory_power,
        'interconnect_power': interconnect_power,
        'io_misc_power': io_misc_power,
        'leakage_power': leakage_power,
        'total_power': total_power,
    }


def print_comparison_table():
    """Print comparison of spec TDP vs estimated TDP."""

    print("\n" + "="*155)
    print("HARDWARE REGISTRY TDP vs ESTIMATED TDP COMPARISON")
    print("="*155)
    print("\nPhysics Model: TDP = energy_per_mac x macs_per_second")
    print("               energy_per_mac = 2 x base_energy(process) x circuit_multiplier x precision_scale")
    print("               (MAC = multiply + accumulate = 2 ops)")
    print()

    # Header
    print(f"{'GPU':<22} {'Arch':<10} {'Node':>5} {'CUDA':>7} {'TC':>5} {'ALUs':>7} {'TC MACs':>9} "
          f"{'Spec TDP':>10} {'CUDA Est':>10} {'TC Est':>10} {'Combined':>10} {'Ratio':>8}")
    print(f"{'':22} {'':10} {'(nm)':>5} {'Cores':>7} {'':>5} {'per TC':>7} {'(total)':>9} "
          f"{'(W)':>10} {'FP32 (W)':>10} {'FP16 (W)':>10} {'(W)':>10} {'Est/Spec':>8}")
    print("-"*155)

    for gpu in NVIDIA_GPUS:
        cuda_est = estimate_cuda_core_tdp(gpu, 'FP32')
        tc_est = estimate_tensor_core_tdp(gpu, 'FP16')
        combined = estimate_combined_tdp(gpu)
        ratio = combined / gpu['spec_tdp_w']
        total_tc_macs = get_total_tc_mac_units(gpu)
        alus_per_tc = gpu['tc_macs_per_clock']  # Each MAC unit is an ALU

        # Flag if estimated significantly exceeds spec
        flag = " ***" if ratio > 1.2 else " *" if ratio > 1.0 else ""

        print(f"{gpu['name']:<22} {gpu['arch']:<10} {gpu['process_nm']:>5} {gpu['cuda_cores']:>7,} "
              f"{gpu['tensor_cores']:>5} {alus_per_tc:>7,} {total_tc_macs:>9,} {gpu['spec_tdp_w']:>10.0f} {cuda_est:>10.1f} "
              f"{tc_est:>10.1f} {combined:>10.1f} {ratio:>7.2f}x{flag}")

    print("-"*155)
    print("\nLegend:")
    print("  ALUs per TC = MAC units per Tensor Core (each MAC = 1 multiply + 1 accumulate ALU)")
    print("  TC MACs = Total MAC units in tensor core fabric (TC count x ALUs per TC)")
    print("  * = Estimated TDP exceeds Spec TDP (slight thermal constraint)")
    print("  *** = Estimated TDP significantly exceeds Spec TDP (>20% over)")
    print()

    # Detailed analysis
    print("\n" + "="*120)
    print("DETAILED ANALYSIS BY PRECISION")
    print("="*120)

    precisions = ['FP32', 'FP16', 'FP8', 'INT8']

    for gpu in NVIDIA_GPUS:
        print(f"\n{gpu['name']} ({gpu['arch']}, {gpu['process_nm']}nm)")
        print(f"  Spec TDP: {gpu['spec_tdp_w']}W")
        print()

        print(f"  {'Precision':<10} {'Fabric':<12} {'Est TDP (W)':>12} {'vs Spec':>10} {'Status':<20}")
        print(f"  {'-'*70}")

        # CUDA core estimates
        for prec in ['FP64', 'FP32']:
            est = estimate_cuda_core_tdp(gpu, prec)
            ratio = est / gpu['spec_tdp_w']
            status = "OVER BUDGET" if ratio > 1.0 else "OK"
            print(f"  {prec:<10} {'CUDA Cores':<12} {est:>12.1f} {ratio:>9.2f}x  {status:<20}")

        # Tensor core estimates (only for precisions the architecture supports)
        tc_precisions = ['FP16']
        if gpu['arch'] in ['Ampere', 'Hopper', 'Blackwell']:
            tc_precisions = ['BF16', 'FP16', 'FP8', 'INT8']

        for prec in tc_precisions:
            est = estimate_tensor_core_tdp(gpu, prec)
            ratio = est / gpu['spec_tdp_w']
            status = "OVER BUDGET" if ratio > 1.0 else "OK"
            print(f"  {prec:<10} {'Tensor Core':<12} {est:>12.1f} {ratio:>9.2f}x  {status:<20}")

        # Combined estimate
        combined = estimate_combined_tdp(gpu)
        ratio = combined / gpu['spec_tdp_w']
        status = "OVER BUDGET" if ratio > 1.0 else "OK"
        print(f"  {'Combined':<10} {'CUDA+TC':<12} {combined:>12.1f} {ratio:>9.2f}x  {status:<20}")

    print()
    print("="*120)
    print("OBSERVATIONS")
    print("="*120)
    print("""
1. CUDA Core Analysis:
   - CUDA core-only TDP estimates are 9-35% of spec TDP
   - CUDA cores alone consume a small fraction of the thermal budget
   - This leaves significant headroom for tensor cores and memory

2. Tensor Core Analysis:
   - Datacenter GPUs (B100, H100, A100, V100): TC power is 50-79% of spec TDP
   - This is well within thermal limits, leaving room for memory/interconnect/leakage
   - T4 (inference GPU): TC power EXCEEDS spec TDP (130%), indicating thermal constraint

3. Full Chip Analysis:
   - H100 and V100 full-chip estimates are within 10% of spec TDP (good model fit)
   - B100, A100, T4 estimates exceed spec TDP, suggesting:
     * These chips cannot sustain full utilization of all compute units
     * Power management throttles to stay within thermal limits
     * Peak specs are theoretical, not sustained performance

4. Model Validation:
   - Physics-based model produces reasonable estimates for most GPUs
   - 40% datapath overhead (register files, control) is necessary for accuracy
   - Inference GPUs (T4) show largest discrepancy due to aggressive thermal limits
""")


def print_process_node_energy():
    """Print process node base energies for reference."""
    print("\n" + "="*60)
    print("PROCESS NODE BASE ENERGY REFERENCE")
    print("="*60)
    print(f"\n{'Node (nm)':<12} {'Base Energy (pJ)':<18} {'Tensor Core (pJ)':<18}")
    print("-"*50)
    for node in sorted(PROCESS_NODE_BASE_ENERGY_PJ.keys()):
        base = PROCESS_NODE_BASE_ENERGY_PJ[node]
        tc = base * CIRCUIT_TYPE_MULTIPLIER['tensor_core']
        print(f"{node:<12} {base:<18.2f} {tc:<18.2f}")
    print()


def print_full_chip_breakdown():
    """Print full chip power breakdown."""
    print("\n" + "="*130)
    print("FULL CHIP TDP BREAKDOWN (Compute + Memory + Interconnect + Leakage)")
    print("="*130)
    print("\nThis model accounts for all major power consumers, not just ALUs.")
    print()

    # Header
    print(f"{'GPU':<22} {'Spec':>6} {'CUDA':>7} {'TC':>7} {'Mem':>7} {'Intcon':>7} {'I/O':>6} "
          f"{'Leak':>7} {'Total':>8} {'Ratio':>8}")
    print(f"{'':22} {'TDP':>6} {'(W)':>7} {'(W)':>7} {'(W)':>7} {'(W)':>7} {'(W)':>6} "
          f"{'(W)':>7} {'Est(W)':>8} {'Est/Spec':>8}")
    print("-"*130)

    for gpu in NVIDIA_GPUS:
        breakdown = estimate_full_chip_tdp(gpu)
        ratio = breakdown['total_power'] / gpu['spec_tdp_w']

        flag = " ***" if ratio > 1.2 else " *" if ratio > 1.0 else ""

        print(f"{gpu['name']:<22} {gpu['spec_tdp_w']:>6.0f} "
              f"{breakdown['cuda_power']:>7.1f} "
              f"{breakdown['tc_power']:>7.1f} "
              f"{breakdown['memory_power']:>7.1f} "
              f"{breakdown['interconnect_power']:>7.1f} "
              f"{breakdown['io_misc_power']:>6.1f} "
              f"{breakdown['leakage_power']:>7.1f} "
              f"{breakdown['total_power']:>8.1f} "
              f"{ratio:>7.2f}x{flag}")

    print("-"*130)
    print()

    # Percentage breakdown
    print("\nPERCENTAGE BREAKDOWN:")
    print()
    print(f"{'GPU':<22} {'CUDA':>8} {'TC':>8} {'Memory':>8} {'Intcon':>8} {'I/O':>8} {'Leakage':>8}")
    print("-"*80)

    for gpu in NVIDIA_GPUS:
        breakdown = estimate_full_chip_tdp(gpu)
        total = breakdown['total_power']

        print(f"{gpu['name']:<22} "
              f"{100*breakdown['cuda_power']/total:>7.1f}% "
              f"{100*breakdown['tc_power']/total:>7.1f}% "
              f"{100*breakdown['memory_power']/total:>7.1f}% "
              f"{100*breakdown['interconnect_power']/total:>7.1f}% "
              f"{100*breakdown['io_misc_power']/total:>7.1f}% "
              f"{100*breakdown['leakage_power']/total:>7.1f}%")

    print()
    print("="*130)
    print("ANALYSIS OF FULL-CHIP MODEL")
    print("="*130)
    print("""
With the full-chip model including memory, interconnect, and leakage:

1. Memory Subsystem Power:
   - HBM memory at full bandwidth consumes 40-50W (HBM3e at 8 TB/s = 40W)
   - This is a significant portion of total chip power
   - Memory-bound workloads will see high memory power, low compute power

2. Leakage Power:
   - Smaller process nodes (3nm, 4nm) have higher leakage fractions (22-25%)
   - This is "always on" power regardless of workload
   - B100 at 3nm has ~150-200W just from leakage

3. Implications for Our Model:
   - Pure compute TDP (ALUs only) is ~30-50% of total chip TDP
   - The remaining 50-70% goes to memory, interconnect, I/O, and leakage
   - This explains why ALU-only estimates were well below spec TDP

4. Accuracy Check:
   - Our full-chip estimates should be closer to spec TDP
   - If still significantly off, indicates our energy-per-op values may need calibration
""")


if __name__ == '__main__':
    print_process_node_energy()
    print_comparison_table()
    print_full_chip_breakdown()
