#!/usr/bin/env python3
"""
Energy Breakdown Utilities

Shared utilities for printing hierarchical energy breakdowns across different
architecture-specific CLI tools (analyze_cpu_energy.py, analyze_gpu_energy.py, etc.).

Extracted from compare_architectures_energy.py to avoid code duplication.
"""

from typing import Dict, Optional


def print_cpu_hierarchical_breakdown(
    arch_specific_events: Dict[str, float],
    total_energy_j: float,
    compute_energy_j: float = 0.0,
    memory_energy_j: float = 0.0,
    latency_s: Optional[float] = None,
    idle_power_w: float = 15.0
) -> None:
    """
    Print hierarchical energy breakdown for CPU (Stored-Program architecture).

    Args:
        arch_specific_events: Dict of architectural energy events (from StoredProgramEnergyModel)
        total_energy_j: Total energy consumption in Joules
        compute_energy_j: Baseline compute energy in Joules
        memory_energy_j: Baseline memory energy in Joules
        latency_s: Latency in seconds (for idle/leakage calculation)
        idle_power_w: Idle power consumption in Watts (default: 15W)
    """
    print(f"\n{'─'*80}")
    print(f"CPU (STORED-PROGRAM MULTICORE) ENERGY BREAKDOWN")
    print(f"{'─'*80}")

    # Component 1: Instruction Pipeline
    print(f"\n  1. INSTRUCTION PIPELINE (Fetch → Decode → Dispatch)")
    inst_fetch = arch_specific_events.get('instruction_fetch_energy', 0) * 1e6
    inst_decode = arch_specific_events.get('instruction_decode_energy', 0) * 1e6
    inst_dispatch = arch_specific_events.get('instruction_dispatch_energy', 0) * 1e6
    num_instructions = arch_specific_events.get('num_instructions', 0)
    pipeline_total = inst_fetch + inst_decode + inst_dispatch

    print(f"     • Instruction Fetch (I-cache):      {inst_fetch:8.3f} μJ  ({inst_fetch/pipeline_total*100 if pipeline_total > 0 else 0:5.1f}%)  [{num_instructions:,} instructions]")
    print(f"     • Instruction Decode:               {inst_decode:8.3f} μJ  ({inst_decode/pipeline_total*100 if pipeline_total > 0 else 0:5.1f}%)")
    print(f"     • Instruction Dispatch:             {inst_dispatch:8.3f} μJ  ({inst_dispatch/pipeline_total*100 if pipeline_total > 0 else 0:5.1f}%)")
    print(f"     └─ Subtotal:                        {pipeline_total:8.3f} μJ")
    print(f"        NOTE: Dispatch writes control signals; actual ALU execution tracked separately")

    # Component 2: Register File Operations
    print(f"\n  2. REGISTER FILE OPERATIONS (2 reads + 1 write per instruction)")
    reg_read = arch_specific_events.get('register_read_energy', 0) * 1e6
    reg_write = arch_specific_events.get('register_write_energy', 0) * 1e6
    num_reg_reads = arch_specific_events.get('num_register_reads', 0)
    num_reg_writes = arch_specific_events.get('num_register_writes', 0)
    regfile_total = reg_read + reg_write

    print(f"     • Register Reads:                   {reg_read:8.3f} μJ  ({reg_read/regfile_total*100 if regfile_total > 0 else 0:5.1f}%) [{num_reg_reads:,} reads]")
    print(f"     • Register Writes:                  {reg_write:8.3f} μJ  ({reg_write/regfile_total*100 if regfile_total > 0 else 0:5.1f}%) [{num_reg_writes:,} writes]")
    print(f"     └─ Subtotal:                        {regfile_total:8.3f} μJ")
    print(f"        NOTE: Register energy ≈ ALU energy (both ~0.6-0.8 pJ per op)")

    # Component 3: Memory Hierarchy (4-Stage)
    print(f"\n  3. MEMORY HIERARCHY (4-Stage: L1 → L2 → L3 → DRAM)")
    l1 = arch_specific_events.get('l1_cache_energy', 0) * 1e6
    l2 = arch_specific_events.get('l2_cache_energy', 0) * 1e6
    l3 = arch_specific_events.get('l3_cache_energy', 0) * 1e6
    dram = arch_specific_events.get('dram_energy', 0) * 1e6
    l1_accesses = arch_specific_events.get('l1_accesses', 0)
    l2_accesses = arch_specific_events.get('l2_accesses', 0)
    l3_accesses = arch_specific_events.get('l3_accesses', 0)
    dram_accesses = arch_specific_events.get('dram_accesses', 0)
    mem_total = l1 + l2 + l3 + dram

    print(f"     • L1 Cache (per-core, 32 KB):      {l1:8.3f} μJ  ({l1/mem_total*100 if mem_total > 0 else 0:5.1f}%)  [{l1_accesses:,} accesses]")
    print(f"     • L2 Cache (per-core, 256 KB):     {l2:8.3f} μJ  ({l2/mem_total*100 if mem_total > 0 else 0:5.1f}%)  [{l2_accesses:,} accesses]")
    print(f"     • L3 Cache (shared LLC, 8 MB):     {l3:8.3f} μJ  ({l3/mem_total*100 if mem_total > 0 else 0:5.1f}%)  [{l3_accesses:,} accesses]")
    print(f"     • DRAM (off-chip DDR4):            {dram:8.3f} μJ  ({dram/mem_total*100 if mem_total > 0 else 0:5.1f}%)  [{dram_accesses:,} accesses]")
    print(f"     └─ Subtotal:                        {mem_total:8.3f} μJ")

    # Component 4: ALU Operations
    print(f"\n  4. ALU OPERATIONS (Floating-point arithmetic)")
    alu = arch_specific_events.get('alu_energy', 0) * 1e6

    print(f"     • ALU Energy:                       {alu:8.3f} μJ  [{num_instructions:,} ops]")
    print(f"     └─ Subtotal:                        {alu:8.3f} μJ")

    # Component 5: Branch Prediction
    print(f"\n  5. BRANCH PREDICTION (Control flow)")
    branch = arch_specific_events.get('branch_energy', 0) * 1e6
    num_branches = arch_specific_events.get('num_branches', 0)
    num_mispredicted = arch_specific_events.get('num_mispredicted_branches', 0)
    prediction_rate = arch_specific_events.get('branch_prediction_success_rate', 0.95) * 100

    print(f"     • Branch Prediction:                {branch:8.3f} μJ  [{num_branches:,} branches, {num_mispredicted:,} mispredicted @ {prediction_rate:.0f}% success]")
    print(f"     └─ Subtotal:                        {branch:8.3f} μJ")

    # Total
    arch_overhead = pipeline_total + regfile_total + mem_total + alu + branch
    dynamic_energy_total = arch_overhead + compute_energy_j*1e6 + memory_energy_j*1e6

    # Calculate idle/leakage energy
    idle_leakage_energy = 0.0
    if latency_s is not None:
        idle_leakage_energy = total_energy_j*1e6 - dynamic_energy_total

    print(f"\n  TOTAL CPU ARCHITECTURAL OVERHEAD:     {arch_overhead:8.3f} μJ")
    print(f"  Base Compute Energy (from mapper):    {compute_energy_j*1e6:8.3f} μJ")
    print(f"  Base Memory Energy (from mapper):     {memory_energy_j*1e6:8.3f} μJ")
    print(f"  {'─'*80}")
    print(f"  SUBTOTAL DYNAMIC ENERGY:              {dynamic_energy_total:8.3f} μJ  ({dynamic_energy_total/total_energy_j/1e6*100:.1f}%)")

    if latency_s is not None:
        print(f"  Idle/Leakage Energy ({idle_power_w}W × latency):  {idle_leakage_energy:8.3f} μJ  ({idle_leakage_energy/total_energy_j/1e6*100:.1f}%)")

    print(f"  {'─'*80}")
    print(f"  TOTAL CPU ENERGY:                     {total_energy_j*1e6:8.3f} μJ")

    print(f"\n  CPU CHARACTERISTICS:")
    print(f"  • Instruction fetch overhead: {inst_fetch:.3f} μJ ({num_instructions:,} instructions)")
    print(f"  • Register file energy: {regfile_total:.3f} μJ (comparable to ALU: {alu:.3f} μJ)")
    if mem_total > 0:
        print(f"  • Memory hierarchy: {mem_total:.3f} μJ (L1: {l1/mem_total*100:.0f}%, DRAM: {dram/mem_total*100:.0f}%)")
    print(f"  • Lower than GPU (no massive coherence machinery)")
    print(f"  • Higher than KPU (dataflow eliminates instruction fetch)")


def aggregate_subgraph_events(
    subgraph_events_list: list[Dict[str, float]]
) -> Dict[str, float]:
    """
    Aggregate architectural events across multiple subgraphs.

    Args:
        subgraph_events_list: List of event dicts from each subgraph

    Returns:
        Dict with aggregated events (summed across all subgraphs)
    """
    aggregated = {}

    for events in subgraph_events_list:
        for key, value in events.items():
            if key in aggregated:
                aggregated[key] += value
            else:
                aggregated[key] = value

    return aggregated
