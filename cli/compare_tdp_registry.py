#!/usr/bin/env python3
"""
Compare TDP estimates from physics model vs hardware registry specs.

This script loads hardware profiles from the registry and compares their
stated TDP against our physics-based TDP estimate (energy_per_mac x macs/sec).

The energy model includes the MINIMUM energy required per MAC operation:
  1. ALU energy: The actual multiply-accumulate circuit
  2. Operand fetch: Register file reads, operand collector, routing to ALU
  3. Result writeback: Routing result back, register file write

This is the FLOOR for compute energy - it excludes:
  - Memory hierarchy (L2, HBM)
  - Interconnect
  - I/O
  - Leakage/static power

If our floor estimate exceeds spec TDP, the marketing claims are physically
impossible to sustain.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from graphs.hardware.technology_profile import (
    TechnologyProfile,
    MemoryType,
    PROCESS_NODE_BASE_ENERGY_PJ,
    CIRCUIT_TYPE_MULTIPLIER,
)
from graphs.hardware.operand_fetch import TensorCoreOperandFetchModel


# =============================================================================
# GPU Hardware Specifications
# =============================================================================
# These are the marketing specs from NVIDIA's datasheets.
# The key question: can the chip actually deliver this throughput within TDP?

NVIDIA_GPUS = [
    {
        'name': 'B100-SXM6-192GB',
        'arch': 'Blackwell',
        'process_nm': 3,          # TSMC 4NP maps to ~3nm energy
        'memory_type': MemoryType.HBM3E,
        'cuda_cores': 16896,      # 132 SMs x 128 cores
        'tensor_cores': 528,      # 132 SMs x 4 TCs
        'tc_macs_per_clock': 512, # 5th gen TC: 512 MACs/clock/TC
        'freq_ghz': 2.1,
        'spec_tdp_w': 1000,
    },
    {
        'name': 'H100-SXM5-80GB',
        'arch': 'Hopper',
        'process_nm': 4,          # TSMC 4N
        'memory_type': MemoryType.HBM3,
        'cuda_cores': 16896,
        'tensor_cores': 528,
        'tc_macs_per_clock': 256, # 4th gen TC: 256 MACs/clock/TC
        'freq_ghz': 1.98,
        'spec_tdp_w': 700,
    },
    {
        'name': 'A100-SXM4-80GB',
        'arch': 'Ampere',
        'process_nm': 7,          # TSMC 7nm
        'memory_type': MemoryType.HBM2E,
        'cuda_cores': 13824,
        'tensor_cores': 432,
        'tc_macs_per_clock': 256, # 3rd gen TC
        'freq_ghz': 1.41,
        'spec_tdp_w': 400,
    },
    {
        'name': 'V100-SXM3-32GB',
        'arch': 'Volta',
        'process_nm': 12,         # TSMC 12nm FFN
        'memory_type': MemoryType.HBM2,
        'cuda_cores': 5120,
        'tensor_cores': 640,
        'tc_macs_per_clock': 64,  # 1st gen TC: 4x4x4 = 64 MACs
        'freq_ghz': 1.53,
        'spec_tdp_w': 350,
    },
    {
        'name': 'T4-PCIe-16GB',
        'arch': 'Turing',
        'process_nm': 12,         # TSMC 12nm FFN
        'memory_type': MemoryType.GDDR6,
        'cuda_cores': 2560,
        'tensor_cores': 320,
        'tc_macs_per_clock': 64,  # 2nd gen TC
        'freq_ghz': 1.59,
        'spec_tdp_w': 70,
    },
]


def create_tech_profile(gpu: dict) -> TechnologyProfile:
    """Create a TechnologyProfile for a GPU based on its specs."""
    return TechnologyProfile.create(
        process_node_nm=gpu['process_nm'],
        memory_type=gpu['memory_type'],
        target_market='datacenter',
        name=gpu['name'],
        frequency_ghz=gpu['freq_ghz'],
        tdp_w=gpu['spec_tdp_w'],
    )


def get_total_tc_macs(gpu: dict) -> int:
    """Get total MAC units across all tensor cores."""
    return gpu['tensor_cores'] * gpu['tc_macs_per_clock']


def estimate_tensor_core_tdp(
    gpu: dict,
    tc_model: TensorCoreOperandFetchModel,
    precision: str = 'FP16'
) -> float:
    """
    Estimate TDP from Tensor Cores using the physics-based operand fetch model.

    This computes:
      TDP = energy_per_mac * macs_per_second

    Where energy_per_mac includes:
      - ALU energy (the actual MAC circuit)
      - Operand fetch energy (register reads, collector, routing)
      - Result writeback energy (routing, register write)
    """
    # Get energy per MAC from the model (includes ALU + operand fetch + writeback)
    energy_per_mac_pj = tc_model.get_energy_per_mac_pj(precision)
    energy_per_mac_j = energy_per_mac_pj * 1e-12

    # Precision affects throughput (lower precision = more MACs/clock)
    precision_throughput_mult = {
        'FP64': 0.25,   # 1/4 throughput
        'FP32': 0.5,    # 1/2 throughput
        'TF32': 0.5,
        'BF16': 1.0,    # baseline
        'FP16': 1.0,    # baseline
        'FP8': 2.0,     # 2x throughput
        'INT8': 2.0,
    }.get(precision, 1.0)

    # Total MACs per second
    macs_per_clock = gpu['tensor_cores'] * gpu['tc_macs_per_clock'] * precision_throughput_mult
    macs_per_sec = macs_per_clock * gpu['freq_ghz'] * 1e9

    # TDP in watts
    tdp_watts = energy_per_mac_j * macs_per_sec
    return tdp_watts


def estimate_cuda_core_tdp(gpu: dict, tech_profile: TechnologyProfile, precision: str = 'FP32') -> float:
    """
    Estimate TDP from CUDA cores (FP32/FP64 workloads).

    CUDA cores have higher overhead than TensorCores because:
    - Each thread independently fetches operands (no bulk matrix access)
    - More bank conflicts (unstructured access patterns)
    - Full scalar pipeline overhead
    """
    # CUDA core energy is higher than TensorCore per MAC
    # Use base ALU energy with standard cell multiplier
    base_alu_pj = tech_profile.base_alu_energy_pj

    # Precision scaling
    precision_scale = {'FP64': 2.0, 'FP32': 1.0, 'FP16': 0.5}.get(precision, 1.0)

    # CUDA has higher operand fetch overhead than TC
    # Approximate as 2x base ALU for total MAC energy
    # (This is conservative - actual CUDA overhead is higher)
    cuda_overhead_mult = 2.5  # ALU + register file + crossbar
    energy_per_mac_pj = base_alu_pj * precision_scale * cuda_overhead_mult
    energy_per_mac_j = energy_per_mac_pj * 1e-12

    # Each CUDA core does 1 FMA per clock = 1 MAC
    ops_per_clock = 1
    macs_per_sec = gpu['cuda_cores'] * ops_per_clock * gpu['freq_ghz'] * 1e9

    return energy_per_mac_j * macs_per_sec


def print_comparison_table():
    """Print comparison of spec TDP vs estimated TDP using physics model."""

    print("\n" + "=" * 180)
    print("HARDWARE REGISTRY TDP vs PHYSICS-BASED TDP ESTIMATE")
    print("=" * 180)
    print("""
Physics Model: TDP = energy_per_mac x macs_per_second

  energy_per_mac = ALU_energy + operand_fetch_energy + writeback_energy

  Where:
    ALU_energy      = base_energy(process_nm) x circuit_multiplier x precision_scale
    operand_fetch   = reg_addr_gen + operand_collector + bank_arbitration + routing + reg_reads
    writeback       = result_routing + reg_write

  This is the MINIMUM compute energy floor. Does NOT include memory, interconnect, or leakage.
""")

    # Header
    print(f"{'GPU':<22} {'Arch':<10} {'Node':>5} {'TC':>5} {'MACs/TC':>8} {'Total MACs':>11} "
          f"{'ALU':>6} {'Fetch':>6} {'WB':>5} {'Total':>6} "
          f"{'Spec TDP':>9} {'Est TDP':>9} {'Ratio':>7}")
    print(f"{'':22} {'':10} {'(nm)':>5} {'':>5} {'':>8} {'(millions)':>11} "
          f"{'(pJ)':>6} {'(pJ)':>6} {'(pJ)':>5} {'(pJ)':>6} "
          f"{'(W)':>9} {'(W)':>9} {'':>7}")
    print("-" * 180)

    for gpu in NVIDIA_GPUS:
        # Create technology profile and TensorCore model
        tech_profile = create_tech_profile(gpu)
        tc_model = TensorCoreOperandFetchModel(tech_profile=tech_profile)

        # Get energy breakdown
        breakdown = tc_model.get_energy_breakdown_per_mac_pj('FP16')
        total_pj = breakdown['total']

        # Estimate TDP
        tc_tdp = estimate_tensor_core_tdp(gpu, tc_model, 'FP16')
        ratio = tc_tdp / gpu['spec_tdp_w']

        # Total MAC units
        total_macs = get_total_tc_macs(gpu) / 1e6  # In millions

        # Flag if estimate exceeds spec
        flag = " ***" if ratio > 1.2 else " *" if ratio > 1.0 else ""

        print(f"{gpu['name']:<22} {gpu['arch']:<10} {gpu['process_nm']:>5} "
              f"{gpu['tensor_cores']:>5} {gpu['tc_macs_per_clock']:>8} {total_macs:>11.1f} "
              f"{breakdown['alu']:>6.2f} {breakdown['operand_fetch']:>6.3f} {breakdown['result_writeback']:>5.3f} {total_pj:>6.2f} "
              f"{gpu['spec_tdp_w']:>9.0f} {tc_tdp:>9.1f} {ratio:>6.2f}x{flag}")

    print("-" * 180)
    print("""
Legend:
  ALU (pJ)   = Pure MAC circuit energy per operation
  Fetch (pJ) = Operand fetch energy (reg addr gen + collector + arbitration + routing + reg reads)
  WB (pJ)    = Writeback energy (result routing + reg write)
  Total (pJ) = Complete energy per MAC = ALU + Fetch + WB
  * = Estimated exceeds spec (slight thermal constraint)
  *** = Estimated significantly exceeds spec (>20% over)
""")


def print_energy_breakdown_detail():
    """Print detailed energy breakdown for each GPU."""

    print("\n" + "=" * 120)
    print("DETAILED ENERGY BREAKDOWN PER MAC")
    print("=" * 120)
    print("""
This shows where the energy goes for each MAC operation on TensorCores.
All values in picojoules (pJ) for FP16 precision.
""")

    for gpu in NVIDIA_GPUS:
        tech_profile = create_tech_profile(gpu)
        tc_model = TensorCoreOperandFetchModel(tech_profile=tech_profile)

        print(f"\n{gpu['name']} ({gpu['arch']}, {gpu['process_nm']}nm)")
        print("-" * 60)

        # Get raw energy parameters
        print(f"  Per-MMA Instruction Overhead (64 MACs):")
        print(f"    Register address gen (3x):    {tc_model.reg_addr_gen_energy_pj * 3:>6.2f} pJ")
        print(f"    Operand collector:            {tc_model.operand_collector_energy_pj:>6.2f} pJ")
        print(f"    Bank arbitration:             {tc_model.bank_arbitration_energy_pj:>6.2f} pJ")
        print(f"    Operand routing:              {tc_model.operand_routing_energy_pj:>6.2f} pJ")
        print(f"    Register reads (A,B):         {tc_model.register_read_energy_pj * 2:>6.2f} pJ")
        print(f"    Result routing:               {tc_model.result_routing_energy_pj:>6.2f} pJ")
        print(f"    Register write (C):           {tc_model.register_write_energy_pj:>6.2f} pJ")

        total_overhead = (
            tc_model.reg_addr_gen_energy_pj * 3 +
            tc_model.operand_collector_energy_pj +
            tc_model.bank_arbitration_energy_pj +
            tc_model.operand_routing_energy_pj +
            tc_model.register_read_energy_pj * 2 +
            tc_model.result_routing_energy_pj +
            tc_model.register_write_energy_pj
        )
        print(f"    ----------------------------------------")
        print(f"    Total per MMA:                {total_overhead:>6.2f} pJ")
        print(f"    Per MAC (/ 64):               {total_overhead / 64:>6.4f} pJ")

        print(f"\n  Per-MAC Energy:")
        breakdown = tc_model.get_energy_breakdown_per_mac_pj('FP16')
        print(f"    ALU (TensorCore MAC):         {breakdown['alu']:>6.4f} pJ")
        print(f"    Operand fetch:                {breakdown['operand_fetch']:>6.4f} pJ")
        print(f"    Result writeback:             {breakdown['result_writeback']:>6.4f} pJ")
        print(f"    ----------------------------------------")
        print(f"    TOTAL per MAC:                {breakdown['total']:>6.4f} pJ")

        # Compute TDP
        tc_tdp = estimate_tensor_core_tdp(gpu, tc_model, 'FP16')
        print(f"\n  TDP Analysis:")
        print(f"    MACs/second at {gpu['freq_ghz']} GHz:      {get_total_tc_macs(gpu) * gpu['freq_ghz'] * 1e9 / 1e12:.2f} TMAC/s")
        print(f"    Estimated TC TDP:             {tc_tdp:>6.1f} W")
        print(f"    Spec TDP:                     {gpu['spec_tdp_w']:>6.0f} W")
        print(f"    Ratio:                        {tc_tdp / gpu['spec_tdp_w']:>6.2f}x")

        if tc_tdp > gpu['spec_tdp_w']:
            print(f"    STATUS: EXCEEDS SPEC - Cannot sustain peak throughput!")
        else:
            headroom = gpu['spec_tdp_w'] - tc_tdp
            print(f"    STATUS: OK - {headroom:.0f}W headroom for memory/interconnect/leakage")


def print_sensitivity_analysis():
    """Analyze what energy per MAC would hit TDP."""

    print("\n" + "=" * 120)
    print("ENERGY SENSITIVITY ANALYSIS")
    print("=" * 120)
    print("""
What if our energy model is wrong? At what energy per MAC would compute
alone exhaust the TDP budget (leaving nothing for memory, interconnect, etc.)?
""")

    print(f"{'GPU':<22} {'Model pJ/MAC':>12} {'Est TDP (W)':>12} {'Spec TDP (W)':>12} {'Max pJ/MAC':>12} {'Margin':>10}")
    print("-" * 90)

    for gpu in NVIDIA_GPUS:
        tech_profile = create_tech_profile(gpu)
        tc_model = TensorCoreOperandFetchModel(tech_profile=tech_profile)

        model_pj = tc_model.get_energy_per_mac_pj('FP16')
        est_tdp = estimate_tensor_core_tdp(gpu, tc_model, 'FP16')

        # What pJ/MAC would make compute = spec TDP?
        if est_tdp > 0:
            max_pj = model_pj * (gpu['spec_tdp_w'] / est_tdp)
            margin = max_pj / model_pj
        else:
            max_pj = float('inf')
            margin = float('inf')

        status = "OVER" if est_tdp > gpu['spec_tdp_w'] else f"{margin:.2f}x"

        print(f"{gpu['name']:<22} {model_pj:>12.4f} {est_tdp:>12.1f} {gpu['spec_tdp_w']:>12.0f} "
              f"{max_pj:>12.4f} {status:>10}")

    print("-" * 90)
    print("""
Interpretation:
  'Max pJ/MAC' = Energy per MAC that would make compute power = Spec TDP
  'Margin' = How much higher actual energy could be before exceeding TDP
             (with NO budget for memory, interconnect, leakage)

  If margin < 1.0: Our model already shows compute exceeds TDP
  If margin ~ 1.0-1.5: Very tight thermal budget
  If margin > 2.0: Either our model is too low, or there's significant
                   headroom for non-compute power
""")


def print_process_node_reference():
    """Print process node energy reference."""
    print("\n" + "=" * 80)
    print("PROCESS NODE ENERGY REFERENCE")
    print("=" * 80)
    print(f"\n{'Node (nm)':<12} {'Base ALU (pJ)':<15} {'TensorCore MAC (pJ)':<20}")
    print("-" * 50)

    for node in sorted(PROCESS_NODE_BASE_ENERGY_PJ.keys()):
        base = PROCESS_NODE_BASE_ENERGY_PJ[node]
        tc = base * CIRCUIT_TYPE_MULTIPLIER['tensor_core']
        print(f"{node:<12} {base:<15.2f} {tc:<20.3f}")

    print("""
Note: TensorCore MAC = base_alu x tensor_core_multiplier (0.85)
This is just the ALU circuit - operand fetch adds ~0.02-0.05 pJ/MAC on top.
""")


def print_architecture_comparison():
    """Compare TensorCore vs TPU systolic vs KPU domain flow efficiency."""

    print("\n" + "=" * 120)
    print("ARCHITECTURE EFFICIENCY COMPARISON")
    print("=" * 120)
    print("""
TensorCores are more efficient than CUDA cores, but less efficient than
true systolic arrays (TPU) or domain flow (KPU) because:

  TensorCore: Each MMA instruction fetches operands from register file
              -> Still has operand collector, bank arbitration, routing
              -> ~0.02-0.05 pJ/MAC operand overhead

  TPU Systolic: Weights stay in PE registers, inputs stream through array
                -> 128x spatial reuse, no per-MAC register fetch
                -> ~0.001-0.002 pJ/MAC operand overhead (boundary injection only)

  KPU Domain Flow: Programmable spatial reuse via domain flow control
                   -> 64x+ reuse with recirculation
                   -> ~0.003-0.005 pJ/MAC operand overhead

The fundamental difference: TensorCores are still stored-program machines
where each instruction fetches operands. Spatial architectures achieve
reuse through physical data flow.
""")

    # Create a comparison for H100-class technology
    tech_4nm = TechnologyProfile.create(
        process_node_nm=4,
        memory_type=MemoryType.HBM3,
        target_market='datacenter',
    )

    tc_model = TensorCoreOperandFetchModel(tech_profile=tech_4nm)
    tc_breakdown = tc_model.get_energy_breakdown_per_mac_pj('FP16')

    print(f"At 4nm process (H100-class):")
    print(f"  TensorCore: {tc_breakdown['total']:.4f} pJ/MAC "
          f"(ALU: {tc_breakdown['alu']:.4f}, Fetch+WB: {tc_breakdown['operand_fetch'] + tc_breakdown['result_writeback']:.4f})")
    print(f"  TPU Systolic: ~{tech_4nm.systolic_mac_energy_pj:.4f} pJ/MAC (ALU only, 128x reuse)")
    print(f"  KPU Domain: ~{tech_4nm.domain_flow_mac_energy_pj:.4f} pJ/MAC (ALU only, 64x reuse)")
    print(f"\n  Efficiency ratio (lower is better):")
    print(f"    TensorCore/TPU: {tc_breakdown['total'] / tech_4nm.systolic_mac_energy_pj:.1f}x")
    print(f"    TensorCore/KPU: {tc_breakdown['total'] / tech_4nm.domain_flow_mac_energy_pj:.1f}x")


if __name__ == '__main__':
    print_process_node_reference()
    print_comparison_table()
    print_energy_breakdown_detail()
    print_sensitivity_analysis()
    print_architecture_comparison()
