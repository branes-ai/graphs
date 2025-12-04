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


def get_cuda_energy_per_mac(gpu: dict, precision: str = 'FP32') -> float:
    """Get energy per MAC for CUDA cores in pJ."""
    base_energy_pj = get_process_base_energy_pj(gpu['process_nm'])
    circuit_mult = CIRCUIT_TYPE_MULTIPLIER['standard_cell']
    precision_scale = {'FP64': 2.0, 'FP32': 1.0, 'FP16': 0.5}.get(precision, 1.0)
    # CUDA core does FMA = 2 ops, so energy per MAC = 2 * energy per op
    return base_energy_pj * circuit_mult * precision_scale * 2


def get_tc_energy_per_mac(gpu: dict, precision: str = 'FP16') -> float:
    """Get energy per MAC for Tensor Cores in pJ (including datapath overhead)."""
    base_energy_pj = get_process_base_energy_pj(gpu['process_nm'])
    circuit_mult = CIRCUIT_TYPE_MULTIPLIER['systolic_mac']
    precision_scale = {
        'FP64': 2.0, 'FP32': 1.0, 'TF32': 0.6,
        'BF16': 0.5, 'FP16': 0.5, 'FP8': 0.25, 'INT8': 0.25
    }.get(precision, 0.5)
    datapath_overhead = 1.4  # 40% overhead for register files, control
    # MAC = 2 ops
    return base_energy_pj * circuit_mult * precision_scale * 2 * datapath_overhead


def print_comparison_table():
    """Print comparison of spec TDP vs estimated TDP."""

    print("\n" + "="*175)
    print("HARDWARE REGISTRY TDP vs ESTIMATED TDP COMPARISON")
    print("="*175)
    print("\nPhysics Model: TDP = energy_per_mac x macs_per_second")
    print("               energy_per_mac = 2 x base_energy(process) x circuit_multiplier x precision_scale x overhead")
    print("               (MAC = multiply + accumulate = 2 ops)")
    print()

    # Header
    print(f"{'GPU':<22} {'Arch':<10} {'Node':>5} {'CUDA':>7} {'TC':>5} {'ALUs':>7} {'TC MACs':>9} "
          f"{'CUDA':>7} {'TC':>7} {'Spec TDP':>10} {'CUDA Est':>10} {'TC Est':>10} {'Combined':>10} {'Ratio':>8}")
    print(f"{'':22} {'':10} {'(nm)':>5} {'Cores':>7} {'':>5} {'per TC':>7} {'(total)':>9} "
          f"{'pJ/MAC':>7} {'pJ/MAC':>7} {'(W)':>10} {'FP32 (W)':>10} {'FP16 (W)':>10} {'(W)':>10} {'Est/Spec':>8}")
    print("-"*175)

    for gpu in NVIDIA_GPUS:
        cuda_est = estimate_cuda_core_tdp(gpu, 'FP32')
        tc_est = estimate_tensor_core_tdp(gpu, 'FP16')
        combined = estimate_combined_tdp(gpu)
        ratio = combined / gpu['spec_tdp_w']
        total_tc_macs = get_total_tc_mac_units(gpu)
        alus_per_tc = gpu['tc_macs_per_clock']

        # Energy per MAC
        cuda_pj_per_mac = get_cuda_energy_per_mac(gpu, 'FP32')
        tc_pj_per_mac = get_tc_energy_per_mac(gpu, 'FP16')

        # Flag if estimated significantly exceeds spec
        flag = " ***" if ratio > 1.2 else " *" if ratio > 1.0 else ""

        print(f"{gpu['name']:<22} {gpu['arch']:<10} {gpu['process_nm']:>5} {gpu['cuda_cores']:>7,} "
              f"{gpu['tensor_cores']:>5} {alus_per_tc:>7,} {total_tc_macs:>9,} "
              f"{cuda_pj_per_mac:>7.2f} {tc_pj_per_mac:>7.2f} "
              f"{gpu['spec_tdp_w']:>10.0f} {cuda_est:>10.1f} "
              f"{tc_est:>10.1f} {combined:>10.1f} {ratio:>7.2f}x{flag}")

    print("-"*175)
    print("\nLegend:")
    print("  ALUs per TC = MAC units per Tensor Core (systolic array size)")
    print("  TC MACs = Total MAC units in tensor core fabric (TC count x ALUs per TC)")
    print("  CUDA pJ/MAC = Energy per FP32 MAC on CUDA cores")
    print("  TC pJ/MAC = Energy per FP16 MAC on Tensor Cores (includes 40% datapath overhead)")
    print("  * = Estimated TDP exceeds Spec TDP (slight thermal constraint)")
    print("  *** = Estimated TDP significantly exceeds Spec TDP (>20% over)")

    # Add the critical insight
    print("\n" + "="*175)
    print("ENERGY SENSITIVITY ANALYSIS")
    print("="*175)
    print("\nWhat if the actual energy per MAC is HIGHER than our model?")
    print("At what energy would compute alone exhaust the TDP budget?\n")

    print(f"{'GPU':<22} {'TC pJ/MAC':>10} {'TC TDP':>10} {'Spec TDP':>10} {'Headroom':>10} {'Max pJ/MAC':>12} {'Margin':>10}")
    print(f"{'':22} {'(model)':>10} {'(W)':>10} {'(W)':>10} {'(W)':>10} {'(to hit TDP)':>12} {'':>10}")
    print("-"*100)

    for gpu in NVIDIA_GPUS:
        tc_est = estimate_tensor_core_tdp(gpu, 'FP16')
        tc_pj_per_mac = get_tc_energy_per_mac(gpu, 'FP16')
        headroom = gpu['spec_tdp_w'] - tc_est
        # What pJ/MAC would make TC power = spec TDP?
        max_pj_per_mac = tc_pj_per_mac * (gpu['spec_tdp_w'] / tc_est)
        margin = max_pj_per_mac / tc_pj_per_mac

        status = "OVER" if headroom < 0 else f"{margin:.2f}x"

        print(f"{gpu['name']:<22} {tc_pj_per_mac:>10.2f} {tc_est:>10.1f} {gpu['spec_tdp_w']:>10.0f} "
              f"{headroom:>10.1f} {max_pj_per_mac:>12.2f} {status:>10}")

    print("-"*100)
    print("""
INTERPRETATION:
  'Max pJ/MAC' = The energy per MAC that would make TC power alone = Spec TDP
  'Margin' = How much higher the actual energy could be before exceeding TDP
             (with NO budget for memory, interconnect, leakage, or CUDA cores)

  If actual silicon has higher energy than our model:
    - T4 is ALREADY over budget (0.77x margin = needs 23% LOWER energy)
    - B100 has only 1.28x margin - very tight
    - H100 has 1.72x margin - more comfortable
    - V100 has 1.93x margin - most headroom

  This suggests NVIDIA may be designing to different efficiency targets per product:
    - Inference GPUs (T4): Aggressive specs, thermal throttling expected
    - Training GPUs (V100, H100): More conservative, sustainable throughput
""")
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
    print("ANALYSIS: COMPUTE-ONLY TDP vs SPEC TDP")
    print("="*120)
    print("""
This analysis compares PHYSICS-BASED compute power estimates against NVIDIA's spec TDP.

The compute estimate includes ONLY:
  - CUDA core power (FP32 ALUs)
  - Tensor core power (systolic MAC arrays)
  - 40% datapath overhead (register files, data distribution, control logic)

It does NOT include memory, interconnect, I/O, or leakage - we lack calibrated
physics models for these components.

KEY QUESTION: Can the advertised ALUs actually run at full throughput within the TDP?

If Compute TDP > Spec TDP:
  -> The chip CANNOT sustain peak throughput - it must throttle
  -> Peak TOPS/TFLOPS numbers are THEORETICAL, not achievable
  -> Marketing specs may be inflated beyond what the power delivery supports

If Compute TDP << Spec TDP:
  -> Either our energy model is too low
  -> Or there's significant headroom for memory/interconnect/leakage
  -> Or the chip is over-provisioned with power delivery

FINDINGS:
  - T4: Compute alone (95W) EXCEEDS spec TDP (70W) by 1.36x
    * This chip CANNOT run all tensor cores at full speed
    * Peak INT8 TOPS is not achievable sustained

  - H100: Compute (407W) is 58% of spec TDP (700W)
    * Leaves 293W for memory, interconnect, leakage
    * Seems plausible, but needs validation

  - B100: Compute (780W) is 78% of spec TDP (1000W)
    * Leaves only 220W for everything else
    * Tighter margins than H100

  - V100: Compute (183W) is only 52% of spec TDP (350W)
    * Either our 12nm energy model is too low
    * Or V100 has significant non-compute power draw

CONCLUSION:
  The T4 clearly shows that NVIDIA's peak specs exceed sustainable compute.
  For datacenter GPUs, more investigation is needed to determine if the
  remaining TDP headroom is sufficient for memory and other subsystems.
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


if __name__ == '__main__':
    print_process_node_energy()
    print_comparison_table()
