#!/usr/bin/env python
"""
Energy Breakdown Utilities

Shared utilities for printing hierarchical energy breakdowns across different
architecture-specific CLI tools (analyze_cpu_energy.py, analyze_gpu_energy.py, etc.).

Extracted from compare_architectures_energy.py to avoid code duplication.
"""

from typing import Dict, Optional, Any, Tuple
import json
import csv
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from graphs.hardware.operand_fetch import (
    OperandFetchBreakdown,
    CPUOperandFetchModel,
    GPUOperandFetchModel,
    TPUOperandFetchModel,
    KPUOperandFetchModel,
)
from graphs.hardware.technology_profile import ARCH_COMPARISON_8NM_X86


def print_operand_fetch_breakdown(
    arch_type: str,
    num_ops: int,
    verbose: bool = False
) -> Tuple[float, float, float, float]:
    """
    Print operand fetch energy breakdown for an architecture.

    This shows the critical "last mile" energy cost: delivering operands from
    registers to ALU inputs. This is the key differentiator between fetch-dominated
    architectures (CPU/GPU) and ALU-dominated architectures (TPU/KPU).

    Args:
        arch_type: Architecture type ('cpu', 'gpu', 'tpu', 'kpu')
        num_ops: Number of operations to analyze
        verbose: Print detailed breakdown

    Returns:
        Tuple of (alu_energy_pj, fetch_energy_pj, total_energy_pj, reuse_factor)
    """
    # Get architecture comparison set with all technology profiles
    arch_comparison = ARCH_COMPARISON_8NM_X86

    # Select appropriate technology profile and create operand fetch model
    arch_lower = arch_type.lower()
    if arch_lower == 'cpu':
        tech_profile = arch_comparison.cpu_profile
        model = CPUOperandFetchModel(tech_profile=tech_profile)
        desc = "CPU (x86 SIMD)"
        reuse_hint = 1.0
    elif arch_lower == 'gpu':
        tech_profile = arch_comparison.gpu_profile
        model = GPUOperandFetchModel(tech_profile=tech_profile)
        desc = "GPU (CUDA Core)"
        reuse_hint = 1.0
    elif arch_lower == 'tpu':
        tech_profile = arch_comparison.tpu_profile
        model = TPUOperandFetchModel(tech_profile=tech_profile)
        desc = "TPU (Systolic Array)"
        reuse_hint = 128.0  # 128x128 array
    elif arch_lower == 'kpu':
        tech_profile = arch_comparison.kpu_profile
        model = KPUOperandFetchModel(tech_profile=tech_profile)
        desc = "KPU (Domain-Flow)"
        reuse_hint = 64.0
    else:
        raise ValueError(f"Unknown architecture type: {arch_type}")

    # Calculate energy breakdown
    breakdown = model.compute_operand_fetch_energy(
        num_ops,
        operand_width_bytes=4,  # FP32
        spatial_reuse_factor=reuse_hint
    )

    # Get pure ALU energy from the technology profile
    # Use the appropriate MAC energy based on the architecture type
    if arch_lower == 'cpu':
        pure_alu_energy_pj = tech_profile.simd_mac_energy_pj
    elif arch_lower == 'gpu':
        pure_alu_energy_pj = tech_profile.tensor_core_mac_energy_pj
    elif arch_lower == 'tpu':
        pure_alu_energy_pj = tech_profile.systolic_mac_energy_pj
    elif arch_lower == 'kpu':
        pure_alu_energy_pj = tech_profile.domain_flow_mac_energy_pj
    else:
        pure_alu_energy_pj = tech_profile.base_alu_energy_pj

    # Calculate totals - convert from Joules to pJ
    fetch_energy_pj = breakdown.energy_per_operation * 1e12  # J to pJ
    total_energy_pj = pure_alu_energy_pj + fetch_energy_pj
    reuse_factor = breakdown.operand_reuse_factor

    # Determine if fetch-dominated or ALU-dominated
    alu_fetch_ratio = pure_alu_energy_pj / fetch_energy_pj if fetch_energy_pj > 0 else float('inf')
    is_fetch_dominated = alu_fetch_ratio < 1.0

    if verbose:
        print(f"\n{'-'*60}")
        print(f"OPERAND FETCH ENERGY BREAKDOWN: {desc}")
        print(f"{'-'*60}")
        print(f"  Operations:        {num_ops:,}")
        print(f"")
        print(f"  Pure ALU Energy:   {pure_alu_energy_pj:.3f} pJ/op")
        print(f"  Operand Fetch:     {fetch_energy_pj:.3f} pJ/op")
        print(f"  -----------------------------------")
        print(f"  Total:             {total_energy_pj:.3f} pJ/op")
        print(f"")
        print(f"  Reuse Factor:      {reuse_factor:.0f}x")
        print(f"  ALU/Fetch Ratio:   {alu_fetch_ratio:.3f}")
        print(f"  Bottleneck:        {'FETCH-DOMINATED' if is_fetch_dominated else 'ALU-DOMINATED'}")

        # Show component breakdown (convert J to pJ)
        print(f"\n  Fetch Components:")
        if breakdown.register_read_energy > 0:
            print(f"    Register Read:   {breakdown.register_read_energy / num_ops * 1e12:.3f} pJ/op")
        if breakdown.register_write_energy > 0:
            print(f"    Register Write:  {breakdown.register_write_energy / num_ops * 1e12:.3f} pJ/op")
        if breakdown.operand_collector_energy > 0:
            print(f"    Operand Collector: {breakdown.operand_collector_energy / num_ops * 1e12:.3f} pJ/op")
        if breakdown.crossbar_routing_energy > 0:
            print(f"    Crossbar/Route:  {breakdown.crossbar_routing_energy / num_ops * 1e12:.3f} pJ/op")
        if breakdown.bank_conflict_penalty > 0:
            print(f"    Bank Conflicts:  {breakdown.bank_conflict_penalty / num_ops * 1e12:.3f} pJ/op")
        if breakdown.pe_forwarding_energy > 0:
            print(f"    PE Forwarding:   {breakdown.pe_forwarding_energy / num_ops * 1e12:.3f} pJ/op")
        if breakdown.array_injection_energy > 0:
            print(f"    Array Injection: {breakdown.array_injection_energy / num_ops * 1e12:.3f} pJ/op")

    return pure_alu_energy_pj, fetch_energy_pj, total_energy_pj, reuse_factor


def print_operand_fetch_comparison(
    num_ops: int = 1000,
) -> None:
    """
    Print a comparison table of operand fetch energy across all architectures.

    This highlights the key insight: spatial architectures (TPU/KPU) achieve
    10-100x better TOPS/W because they have massive operand reuse, making them
    ALU-dominated rather than fetch-dominated like CPU/GPU.

    Args:
        num_ops: Number of operations for the comparison
    """
    print(f"\n{'='*80}")
    print(f"OPERAND FETCH ENERGY COMPARISON (8nm process)")
    print(f"{'='*80}")
    print(f"")
    print(f"The 'last mile' problem: delivering operands from registers to ALU inputs")
    print(f"This is the key architectural differentiator for energy efficiency.")
    print(f"")

    # Header
    print(f"{'Architecture':<20} {'ALU(pJ)':<10} {'Fetch(pJ)':<12} {'Total(pJ)':<12} {'Reuse':<10} {'ALU/Fetch':<12} {'Type':<15}")
    print(f"{'-'*20} {'-'*10} {'-'*12} {'-'*12} {'-'*10} {'-'*12} {'-'*15}")

    archs = ['cpu', 'gpu', 'tpu', 'kpu']
    results = []

    for arch in archs:
        alu_pj, fetch_pj, total_pj, reuse = print_operand_fetch_breakdown(
            arch, num_ops, verbose=False
        )
        ratio = alu_pj / fetch_pj if fetch_pj > 0 else float('inf')
        bottleneck = "Fetch-dominated" if ratio < 1.0 else "ALU-dominated"

        arch_name = {
            'cpu': 'CPU (x86 SIMD)',
            'gpu': 'GPU (CUDA Core)',
            'tpu': 'TPU (Systolic)',
            'kpu': 'KPU (Domain-Flow)'
        }[arch]

        print(f"{arch_name:<20} {alu_pj:<10.3f} {fetch_pj:<12.3f} {total_pj:<12.3f} {reuse:<10.0f}x {ratio:<12.3f} {bottleneck:<15}")
        results.append((arch, alu_pj, fetch_pj, total_pj, reuse, ratio))

    # Show relative efficiency
    print(f"\n{'-'*80}")
    print(f"RELATIVE EFFICIENCY (normalized to GPU):")
    print(f"{'-'*80}")

    gpu_total = results[1][3]  # GPU total energy
    for arch, alu_pj, fetch_pj, total_pj, reuse, ratio in results:
        efficiency = gpu_total / total_pj
        arch_name = {
            'cpu': 'CPU',
            'gpu': 'GPU',
            'tpu': 'TPU',
            'kpu': 'KPU'
        }[arch]
        print(f"  {arch_name}: {efficiency:.1f}x GPU efficiency ({total_pj:.3f} pJ/op)")

    print(f"\n  KEY INSIGHT: TPU/KPU are 3-5x more energy efficient than GPU")
    print(f"               because spatial reuse eliminates operand fetch overhead.")


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


def print_gpu_hierarchical_breakdown(
    arch_specific_events: Dict[str, float],
    total_energy_j: float,
    compute_energy_j: float = 0.0,
    memory_energy_j: float = 0.0,
    latency_s: Optional[float] = None,
    idle_power_w: float = 15.0
) -> None:
    """
    Print hierarchical energy breakdown for GPU (Data-Parallel/SIMT architecture).

    Args:
        arch_specific_events: Dict of architectural energy events (from SIMTEnergyModel)
        total_energy_j: Total energy consumption in Joules
        compute_energy_j: Baseline compute energy in Joules
        memory_energy_j: Baseline memory energy in Joules
        latency_s: Latency in seconds (for idle/leakage calculation)
        idle_power_w: Idle power consumption in Watts (default: 15W)
    """
    print(f"\n{'─'*80}")
    print(f"GPU (DATA-PARALLEL SIMT) ENERGY BREAKDOWN")
    print(f"{'─'*80}")

    # Category 1: Compute Units
    print(f"\n  1. COMPUTE UNITS (Tensor Cores vs CUDA Cores)")
    # Combine MAC and FLOP energies for CUDA cores
    tensor_core = arch_specific_events.get('tensor_core_mac_energy', 0) * 1e6
    tensor_core_ops = arch_specific_events.get('tensor_core_ops', 0)
    cuda_core_mac = arch_specific_events.get('cuda_core_mac_energy', 0) * 1e6
    cuda_core_flop = arch_specific_events.get('cuda_core_flop_energy', 0) * 1e6
    cuda_core = cuda_core_mac + cuda_core_flop
    cuda_core_macs = arch_specific_events.get('cuda_core_macs', 0)
    cuda_core_flops = arch_specific_events.get('cuda_core_flops', 0)
    register_file = arch_specific_events.get('register_file_energy', 0) * 1e6
    num_register_accesses = arch_specific_events.get('num_register_accesses', 0)
    compute_total = tensor_core + cuda_core + register_file

    print(f"     • Tensor Core Operations:        {tensor_core:8.3f} μJ  ({tensor_core/compute_total*100 if compute_total > 0 else 0:5.1f}%)  [{tensor_core_ops:,} ops]")
    print(f"     • CUDA Core Operations:          {cuda_core:8.3f} μJ  ({cuda_core/compute_total*100 if compute_total > 0 else 0:5.1f}%)  [{cuda_core_macs:,} MACs, {cuda_core_flops:,} FLOPs]")
    print(f"     • Register File Access:          {register_file:8.3f} μJ  ({register_file/compute_total*100 if compute_total > 0 else 0:5.1f}%)  [{num_register_accesses:,} accesses]")
    print(f"     └─ Subtotal:                     {compute_total:8.3f} μJ")

    # Category 2: Instruction Pipeline
    print(f"\n  2. INSTRUCTION PIPELINE (Fetch → Decode → Execute)")
    inst_fetch = arch_specific_events.get('instruction_fetch_energy', 0) * 1e6
    inst_decode = arch_specific_events.get('instruction_decode_energy', 0) * 1e6
    inst_execute = arch_specific_events.get('instruction_execute_energy', 0) * 1e6
    num_instructions = arch_specific_events.get('num_instructions', 0)
    pipeline_total = inst_fetch + inst_decode + inst_execute

    print(f"     • Instruction Fetch:             {inst_fetch:8.3f} μJ  ({inst_fetch/pipeline_total*100 if pipeline_total > 0 else 0:5.1f}%)  [{num_instructions:,} instructions]")
    print(f"     • Instruction Decode:            {inst_decode:8.3f} μJ  ({inst_decode/pipeline_total*100 if pipeline_total > 0 else 0:5.1f}%)")
    print(f"     • Instruction Execute:           {inst_execute:8.3f} μJ  ({inst_execute/pipeline_total*100 if pipeline_total > 0 else 0:5.1f}%)")
    print(f"     └─ Subtotal:                     {pipeline_total:8.3f} μJ")

    # Category 3: Memory Hierarchy (NVIDIA Ampere nomenclature)
    print(f"\n  3. MEMORY HIERARCHY (Register File → Shared Mem/L1 → L2 → DRAM)")
    shared_mem_l1 = arch_specific_events.get('shared_mem_l1_unified_energy', 0) * 1e6
    shared_mem_l1_accesses = arch_specific_events.get('shared_mem_l1_accesses', 0)
    l2_cache = arch_specific_events.get('l2_cache_energy', 0) * 1e6
    l2_accesses = arch_specific_events.get('l2_accesses', 0)
    dram = arch_specific_events.get('dram_energy', 0) * 1e6
    dram_accesses = arch_specific_events.get('dram_accesses', 0)
    memory_total = shared_mem_l1 + l2_cache + dram

    print(f"     • Shared Memory/L1 (unified):    {shared_mem_l1:8.3f} μJ  ({shared_mem_l1/memory_total*100 if memory_total > 0 else 0:5.1f}%)  [{shared_mem_l1_accesses:,} accesses]")
    print(f"     • L2 Cache (shared across SMs):  {l2_cache:8.3f} μJ  ({l2_cache/memory_total*100 if memory_total > 0 else 0:5.1f}%)  [{l2_accesses:,} accesses]")
    print(f"     • DRAM (HBM2e/LPDDR5):           {dram:8.3f} μJ  ({dram/memory_total*100 if memory_total > 0 else 0:5.1f}%)  [{dram_accesses:,} accesses]")
    print(f"     └─ Subtotal:                     {memory_total:8.3f} μJ")

    # Category 4: SIMT Control Overheads
    print(f"\n  4. SIMT CONTROL OVERHEADS (GPU-Specific)")
    coherence = arch_specific_events.get('coherence_energy', 0) * 1e6
    num_concurrent_warps = arch_specific_events.get('num_concurrent_warps', 0)
    num_memory_ops = arch_specific_events.get('num_memory_ops', 0)
    scheduling = arch_specific_events.get('scheduling_energy', 0) * 1e6
    concurrent_threads = arch_specific_events.get('concurrent_threads', 0)
    divergence = arch_specific_events.get('divergence_energy', 0) * 1e6
    num_divergent_ops = arch_specific_events.get('num_divergent_ops', 0)
    coalescing = arch_specific_events.get('coalescing_energy', 0) * 1e6
    num_uncoalesced = arch_specific_events.get('num_uncoalesced', 0)
    barriers = arch_specific_events.get('barrier_energy', 0) * 1e6
    num_barriers = arch_specific_events.get('num_barriers', 0)
    simt_total = coherence + scheduling + divergence + coalescing + barriers

    print(f"     • Coherence Machinery:           {coherence:8.3f} μJ  ({coherence/simt_total*100 if simt_total > 0 else 0:5.1f}%)  [{num_concurrent_warps:,} warps × {num_memory_ops:,} mem ops] ← DOMINANT!")
    print(f"     • Thread Scheduling:             {scheduling:8.3f} μJ  ({scheduling/simt_total*100 if simt_total > 0 else 0:5.1f}%)  [{concurrent_threads:,} threads]")
    print(f"     • Warp Divergence:               {divergence:8.3f} μJ  ({divergence/simt_total*100 if simt_total > 0 else 0:5.1f}%)  [{num_divergent_ops:,} divergent ops]")
    print(f"     • Memory Coalescing:             {coalescing:8.3f} μJ  ({coalescing/simt_total*100 if simt_total > 0 else 0:5.1f}%)  [{num_uncoalesced:,} uncoalesced]")
    print(f"     • Synchronization Barriers:      {barriers:8.3f} μJ  ({barriers/simt_total*100 if simt_total > 0 else 0:5.1f}%)  [{num_barriers:,} barriers]")
    print(f"     └─ Subtotal:                     {simt_total:8.3f} μJ")

    # Total architectural overhead
    arch_total = compute_total + pipeline_total + memory_total + simt_total
    dynamic_energy_total = arch_total + compute_energy_j*1e6 + memory_energy_j*1e6

    # Calculate idle/leakage energy
    idle_leakage_energy = 0.0
    if latency_s is not None:
        idle_leakage_energy = total_energy_j*1e6 - dynamic_energy_total

    print(f"\n  TOTAL GPU ARCHITECTURAL OVERHEAD:  {arch_total:8.3f} μJ")
    print(f"  Base Compute Energy:               {compute_energy_j*1e6:8.3f} μJ")
    print(f"  Base Memory Energy:                {memory_energy_j*1e6:8.3f} μJ")
    print(f"  {'─'*80}")
    print(f"  SUBTOTAL DYNAMIC ENERGY:           {dynamic_energy_total:8.3f} μJ  ({dynamic_energy_total/total_energy_j/1e6*100:.1f}%)")

    if latency_s is not None:
        print(f"  Idle/Leakage Energy ({idle_power_w}W × latency): {idle_leakage_energy:8.3f} μJ  ({idle_leakage_energy/total_energy_j/1e6*100:.1f}%)")

    print(f"  {'─'*80}")
    print(f"  TOTAL GPU ENERGY:                  {total_energy_j*1e6:8.3f} μJ")

    print(f"\n  GPU CHARACTERISTICS:")
    print(f"  • SIMT control overhead: {simt_total:.3f} μJ (coherence: {coherence:.3f} μJ is dominant)")
    print(f"  • Coherence scales with warps × memory ops: {num_concurrent_warps:,} × {num_memory_ops:,}")
    print(f"  • Much higher than CPU (massive parallelism requires coherence machinery)")
    print(f"  • Memory hierarchy: {memory_total:.3f} μJ (shared/L1, L2, HBM)")


def print_tpu_hierarchical_breakdown(
    arch_specific_events: Dict[str, float],
    total_energy_j: float,
    compute_energy_j: float = 0.0,
    memory_energy_j: float = 0.0,
    latency_s: Optional[float] = None,
    idle_power_w: float = 30.0
) -> None:
    """
    Print hierarchical energy breakdown for TPU (Systolic-Array architecture).

    Args:
        arch_specific_events: Dict of architectural energy events (from SystolicArrayEnergyModel)
        total_energy_j: Total energy consumption in Joules
        compute_energy_j: Baseline compute energy in Joules
        memory_energy_j: Baseline memory energy in Joules
        latency_s: Latency in seconds (for idle/leakage calculation)
        idle_power_w: Idle power consumption in Watts (default: 30W)
    """
    print(f"\n{'─'*80}")
    print(f"TPU (SYSTOLIC-ARRAY) ENERGY BREAKDOWN")
    print(f"{'─'*80}")

    # Check if we have detailed systolic array energy model data
    if 'instruction_decode' in arch_specific_events:
        # Component 1: Instruction Decode (per matrix operation, not per MAC!)
        print(f"\n  1. INSTRUCTION DECODE (Per matrix operation)")
        inst_decode = arch_specific_events.get('instruction_decode', 0) * 1e6
        num_matrix_ops = arch_specific_events.get('num_matrix_ops', 0)

        print(f"     • Matrix Operation Decode:          {inst_decode:8.3f} μJ  [{num_matrix_ops:,} matrix ops]")
        print(f"     └─ Subtotal:                        {inst_decode:8.3f} μJ")

        # Component 2: DMA Controller
        print(f"\n  2. DMA CONTROLLER (Off-chip data transfers)")
        dma_setup = arch_specific_events.get('dma_setup', 0) * 1e6
        dma_addr_gen = arch_specific_events.get('dma_address_gen', 0) * 1e6
        num_dma_transfers = arch_specific_events.get('num_dma_transfers', 0)
        num_cache_lines = arch_specific_events.get('num_cache_lines', 0)
        dma_total = dma_setup + dma_addr_gen

        print(f"     • Descriptor Setup:                 {dma_setup:8.3f} μJ  ({dma_setup/dma_total*100 if dma_total > 0 else 0:5.1f}%)  [{num_dma_transfers:,} transfers]")
        print(f"     • Address Generation:               {dma_addr_gen:8.3f} μJ  ({dma_addr_gen/dma_total*100 if dma_total > 0 else 0:5.1f}%)  [{num_cache_lines:,} cache lines]")
        print(f"     └─ Subtotal:                        {dma_total:8.3f} μJ")

        # Component 3: Weight Loading Sequencer
        print(f"\n  3. WEIGHT LOADING SEQUENCER (Shift into systolic array)")
        weight_shift = arch_specific_events.get('weight_shift_control', 0) * 1e6
        weight_column = arch_specific_events.get('weight_column_select', 0) * 1e6
        num_cycles = arch_specific_events.get('num_systolic_cycles', 0)
        num_weight_elements = arch_specific_events.get('num_weight_elements', 0)
        weight_total = weight_shift + weight_column

        print(f"     • Weight Shift Control:             {weight_shift:8.3f} μJ  ({weight_shift/weight_total*100 if weight_total > 0 else 0:5.1f}%)  [{num_weight_elements:,} elements]")
        print(f"     • Column Select:                    {weight_column:8.3f} μJ  ({weight_column/weight_total*100 if weight_total > 0 else 0:5.1f}%)  [{num_cycles:,} cycles]")
        print(f"     └─ Subtotal:                        {weight_total:8.3f} μJ")

        # Component 4: Unified Buffer Controller
        print(f"\n  4. UNIFIED BUFFER CONTROLLER (Activation scratchpad)")
        ub_addr_gen = arch_specific_events.get('ub_address_gen', 0) * 1e6
        ub_arbitration = arch_specific_events.get('ub_arbitration', 0) * 1e6
        num_ub_accesses = arch_specific_events.get('num_ub_accesses', 0)
        ub_total = ub_addr_gen + ub_arbitration

        print(f"     • Address Generation:               {ub_addr_gen:8.3f} μJ  ({ub_addr_gen/ub_total*100 if ub_total > 0 else 0:5.1f}%)  [{num_ub_accesses:,} accesses]")
        print(f"     • Arbitration:                      {ub_arbitration:8.3f} μJ  ({ub_arbitration/ub_total*100 if ub_total > 0 else 0:5.1f}%)  [{num_ub_accesses:,} requests]")
        print(f"     └─ Subtotal:                        {ub_total:8.3f} μJ")

        # Component 5: Accumulator Controller
        print(f"\n  5. ACCUMULATOR CONTROLLER (Partial sum staging)")
        acc_read = arch_specific_events.get('accumulator_read', 0) * 1e6
        acc_write = arch_specific_events.get('accumulator_write', 0) * 1e6
        acc_addr = arch_specific_events.get('accumulator_address', 0) * 1e6
        num_accumulator_ops = arch_specific_events.get('num_accumulator_ops', 0)
        acc_total = acc_read + acc_write + acc_addr

        print(f"     • Read Control:                     {acc_read:8.3f} μJ  ({acc_read/acc_total*100 if acc_total > 0 else 0:5.1f}%)  [{num_accumulator_ops:,} reads]")
        print(f"     • Write Control:                    {acc_write:8.3f} μJ  ({acc_write/acc_total*100 if acc_total > 0 else 0:5.1f}%)  [{num_accumulator_ops:,} writes]")
        print(f"     • Address Generation:               {acc_addr:8.3f} μJ  ({acc_addr/acc_total*100 if acc_total > 0 else 0:5.1f}%)  [{num_accumulator_ops:,} addresses]")
        print(f"     └─ Subtotal:                        {acc_total:8.3f} μJ")

        # Component 6: Tile Loop Control
        print(f"\n  6. TILE LOOP CONTROL (Tiled matrix operations)")
        tile_loop = arch_specific_events.get('tile_loop_control', 0) * 1e6
        num_tiles = arch_specific_events.get('num_tiles', 0)

        print(f"     • Tile Iteration:                   {tile_loop:8.3f} μJ  [{num_tiles:,} tiles]")
        print(f"     └─ Subtotal:                        {tile_loop:8.3f} μJ")

        # Component 7: Data Injection/Extraction
        print(f"\n  7. DATA INJECTION/EXTRACTION (Spatial array interface)")
        injection = arch_specific_events.get('injection_energy', 0) * 1e6
        extraction = arch_specific_events.get('extraction_energy', 0) * 1e6
        num_elements = arch_specific_events.get('num_elements', 0)
        data_move_total = injection + extraction

        print(f"     • Data Injection:                   {injection:8.3f} μJ  ({injection/data_move_total*100 if data_move_total > 0 else 0:5.1f}%)  [{num_elements:,} elements]")
        print(f"     • Data Extraction:                  {extraction:8.3f} μJ  ({extraction/data_move_total*100 if data_move_total > 0 else 0:5.1f}%)  [{num_elements:,} elements]")
        print(f"     └─ Subtotal:                        {data_move_total:8.3f} μJ")

        # Total Control Overhead
        total_control = inst_decode + dma_total + weight_total + ub_total + acc_total + tile_loop + data_move_total
        control_per_mac = arch_specific_events.get('control_overhead_per_mac_pj', 0)
        array_dim = arch_specific_events.get('array_dimension', 128)

        print(f"\n  TOTAL TPU CONTROL OVERHEAD:        {total_control:8.3f} μJ")
        print(f"  Control per MAC:                     {control_per_mac:.4f} pJ")
        print(f"  Systolic Array Dimension:            {array_dim} × {array_dim} MACs")
    else:
        # Fallback: simple systolic array model
        print(f"\n  (Simplified systolic array model - no detailed breakdown available)")
        systolic_mac = arch_specific_events.get('systolic_array_mac_energy', 0) * 1e6
        on_chip = arch_specific_events.get('on_chip_buffer_energy', 0) * 1e6
        dram = arch_specific_events.get('dram_energy', 0) * 1e6
        total_control = systolic_mac + on_chip + dram

        if systolic_mac > 0 or on_chip > 0 or dram > 0:
            print(f"  • Systolic Array Operations:        {systolic_mac:8.3f} μJ")
            print(f"  • On-Chip Buffer Access:            {on_chip:8.3f} μJ")
            print(f"  • Off-Chip DRAM Access:             {dram:8.3f} μJ")

    # Summary
    arch_total = total_control if 'instruction_decode' in arch_specific_events else (
        arch_specific_events.get('systolic_array_mac_energy', 0) * 1e6 +
        arch_specific_events.get('on_chip_buffer_energy', 0) * 1e6 +
        arch_specific_events.get('dram_energy', 0) * 1e6
    )
    dynamic_energy_total = arch_total + compute_energy_j*1e6 + memory_energy_j*1e6

    # Calculate idle/leakage energy
    idle_leakage_energy = 0.0
    if latency_s is not None:
        idle_leakage_energy = total_energy_j*1e6 - dynamic_energy_total

    print(f"\n  TOTAL TPU ARCHITECTURAL OVERHEAD:  {arch_total:8.3f} μJ")
    print(f"  Base Compute Energy:               {compute_energy_j*1e6:8.3f} μJ")
    print(f"  Base Memory Energy:                {memory_energy_j*1e6:8.3f} μJ")
    print(f"  {'─'*80}")
    print(f"  SUBTOTAL DYNAMIC ENERGY:           {dynamic_energy_total:8.3f} μJ  ({dynamic_energy_total/total_energy_j/1e6*100:.1f}%)")

    if latency_s is not None:
        print(f"  Idle/Leakage Energy ({idle_power_w}W × latency): {idle_leakage_energy:8.3f} μJ  ({idle_leakage_energy/total_energy_j/1e6*100:.1f}%)")

    print(f"  {'─'*80}")
    print(f"  TOTAL TPU ENERGY:                  {total_energy_j*1e6:8.3f} μJ")

    if 'instruction_decode' in arch_specific_events:
        print(f"\n  TPU CHARACTERISTICS:")
        print(f"  • Control overhead per MAC: {control_per_mac:.4f} pJ (vs CPU: ~1.5 pJ, GPU: ~0.5 pJ)")
        print(f"  • Systolic array: {array_dim}×{array_dim} = {array_dim*array_dim:,} MACs per cycle")
        print(f"  • Weight-stationary dataflow: weights stay in place, activations flow")
        print(f"  • Much lower control overhead than CPU/GPU (1 instruction controls many MACs)")


def print_kpu_hierarchical_breakdown(
    arch_specific_events: Dict[str, float],
    total_energy_j: float,
    compute_energy_j: float = 0.0,
    memory_energy_j: float = 0.0,
    latency_s: Optional[float] = None,
    idle_power_w: float = 15.0
) -> None:
    """
    Print hierarchical energy breakdown for KPU (Domain-Flow architecture).

    Args:
        arch_specific_events: Dict of architectural energy events (from KPUTileEnergyAdapter)
        total_energy_j: Total energy consumption in Joules
        compute_energy_j: Baseline compute energy in Joules
        memory_energy_j: Baseline memory energy in Joules
        latency_s: Latency in seconds (for idle/leakage calculation)
        idle_power_w: Idle power consumption in Watts (default: 15W)
    """
    print(f"\n{'─'*80}")
    print(f"KPU (DOMAIN-FLOW) ENERGY BREAKDOWN")
    print(f"{'─'*80}")

    # Check if we have detailed tile energy model data
    if 'dram_energy' in arch_specific_events:
        # Component 1: SURE Program Loading
        print(f"\n  1. SURE PROGRAM LOADING (Spatial dataflow configuration)")
        program_load = arch_specific_events.get('program_load_energy', 0) * 1e6
        cache_hit_rate = arch_specific_events.get('cache_hit_rate', 0.9)

        print(f"     • Program Load/Broadcast:           {program_load:8.3f} μJ")
        print(f"     • Cache Hit Rate:                   {cache_hit_rate*100:5.1f}%")
        print(f"     └─ Subtotal:                        {program_load:8.3f} μJ")

        # Component 2: 4-Stage Memory Hierarchy
        print(f"\n  2. MEMORY HIERARCHY (4-Stage: DRAM → L3 → L2 → L1)")
        dram = arch_specific_events.get('dram_energy', 0) * 1e6
        l3 = arch_specific_events.get('l3_energy', 0) * 1e6
        l2 = arch_specific_events.get('l2_energy', 0) * 1e6
        l1 = arch_specific_events.get('l1_energy', 0) * 1e6
        dram_accesses = arch_specific_events.get('dram_accesses', 0)
        l3_accesses = arch_specific_events.get('l3_accesses', 0)
        l2_accesses = arch_specific_events.get('l2_accesses', 0)
        l1_accesses = arch_specific_events.get('l1_accesses', 0)
        total_bytes = arch_specific_events.get('total_bytes', 0)
        mem_total = dram + l3 + l2 + l1

        print(f"     • DRAM (off-chip):                  {dram:8.3f} μJ  ({dram/mem_total*100 if mem_total > 0 else 0:5.1f}%)  [{dram_accesses:,} accesses]")
        print(f"     • L3 Cache (distributed scratchpad):{l3:8.3f} μJ  ({l3/mem_total*100 if mem_total > 0 else 0:5.1f}%)  [{l3_accesses:,} accesses]")
        print(f"     • L2 Cache (tile-local):            {l2:8.3f} μJ  ({l2/mem_total*100 if mem_total > 0 else 0:5.1f}%)  [{l2_accesses:,} accesses]")
        print(f"     • L1 Cache (PE-local):              {l1:8.3f} μJ  ({l1/mem_total*100 if mem_total > 0 else 0:5.1f}%)  [{l1_accesses:,} accesses]")
        print(f"     • Total Data Transferred:           [{total_bytes/1024:.1f} KB]")
        print(f"     └─ Subtotal:                        {mem_total:8.3f} μJ")

        # Component 3: Data Movement Engines
        print(f"\n  3. DATA MOVEMENT ENGINES (3 Specialized Engines)")
        dma = arch_specific_events.get('dma_energy', 0) * 1e6
        blockmover = arch_specific_events.get('blockmover_energy', 0) * 1e6
        streamer = arch_specific_events.get('streamer_energy', 0) * 1e6
        dma_bytes = arch_specific_events.get('dma_bytes', 0)
        blockmover_bytes = arch_specific_events.get('blockmover_bytes', 0)
        streamer_bytes = arch_specific_events.get('streamer_bytes', 0)
        dme_total = dma + blockmover + streamer

        print(f"     • DMA Engine (DRAM ↔ L3):           {dma:8.3f} μJ  ({dma/dme_total*100 if dme_total > 0 else 0:5.1f}%)  [{dma_bytes/1024:.1f} KB]")
        print(f"     • BlockMover (L3 ↔ L2):             {blockmover:8.3f} μJ  ({blockmover/dme_total*100 if dme_total > 0 else 0:5.1f}%)  [{blockmover_bytes/1024:.1f} KB]")
        print(f"     • Streamer (L2 ↔ L1):               {streamer:8.3f} μJ  ({streamer/dme_total*100 if dme_total > 0 else 0:5.1f}%)  [{streamer_bytes/1024:.1f} KB]")
        print(f"     └─ Subtotal:                        {dme_total:8.3f} μJ")

        # Component 4: Distributed L3 Scratchpad Routing
        print(f"\n  4. DISTRIBUTED L3 SCRATCHPAD (NoC routing)")
        l3_routing = arch_specific_events.get('l3_routing_energy', 0) * 1e6
        avg_hops = arch_specific_events.get('average_l3_hops', 0)
        l3_routing_accesses = arch_specific_events.get('l3_routing_accesses', 0)

        print(f"     • NoC Routing Energy:               {l3_routing:8.3f} μJ  [{l3_routing_accesses:,} accesses]")
        print(f"     • Average Hops:                     {avg_hops:5.1f}")
        print(f"     └─ Subtotal:                        {l3_routing:8.3f} μJ")

        # Component 5: Instruction Token Signature Matching & Dispatch
        print(f"\n  5. INSTRUCTION TOKEN MATCHING & DISPATCH (Dataflow execution)")
        token_match = arch_specific_events.get('token_matching_energy', 0) * 1e6
        signature = arch_specific_events.get('signature_matching_energy', 0) * 1e6
        dispatch = arch_specific_events.get('dispatch_energy', 0) * 1e6
        num_signature_matches = arch_specific_events.get('num_signature_matches', 0)
        num_tokens = arch_specific_events.get('num_tokens', 0)

        print(f"     • Signature Matching:               {signature:8.3f} μJ  [{num_signature_matches:,} matches]")
        print(f"     • Instruction Token Dispatch:       {dispatch:8.3f} μJ  [{num_tokens:,} tokens fired]")
        print(f"     └─ Subtotal:                        {token_match:8.3f} μJ")

        # Component 6: Operator Fusion
        print(f"\n  6. OPERATOR FUSION (Hardware fusion)")
        fusion_net = arch_specific_events.get('fusion_net_energy', 0) * 1e6

        print(f"     • Fusion Coordination:              {fusion_net:8.3f} μJ")
        print(f"     └─ Subtotal:                        {fusion_net:8.3f} μJ")

        # Component 7: Token Routing
        print(f"\n  7. TOKEN ROUTING (Mesh routing)")
        token_routing = arch_specific_events.get('token_routing_energy', 0) * 1e6
        routing_dist = arch_specific_events.get('average_routing_distance', 0)
        num_tokens = arch_specific_events.get('num_tokens', 0)

        print(f"     • Token Routing Hops:               {token_routing:8.3f} μJ  [{num_tokens:,} tokens]")
        print(f"     • Average Distance:                 {routing_dist:5.1f} hops")
        print(f"     └─ Subtotal:                        {token_routing:8.3f} μJ")

        # Component 8: Computation
        print(f"\n  8. COMPUTATION (Compute Fabric operators)")
        compute = arch_specific_events.get('compute_energy', 0) * 1e6
        total_ops = arch_specific_events.get('total_ops', 0)

        print(f"     • MAC Operations:                   {compute:8.3f} μJ  [{total_ops:,} ops]")
        print(f"     └─ Subtotal:                        {compute:8.3f} μJ")

        # Total
        arch_overhead = mem_total + dme_total + token_match + program_load + l3_routing + fusion_net + token_routing
        dynamic_energy_total = arch_overhead + compute

        # Calculate idle/leakage energy
        idle_leakage_energy = 0.0
        if latency_s is not None:
            idle_leakage_energy = total_energy_j*1e6 - dynamic_energy_total

        print(f"\n  TOTAL KPU ARCHITECTURAL OVERHEAD:     {arch_overhead:8.3f} μJ")
        print(f"  Base Compute Energy (from above):     {compute:8.3f} μJ")
        print(f"  {'─'*80}")
        print(f"  SUBTOTAL DYNAMIC ENERGY:              {dynamic_energy_total:8.3f} μJ  ({dynamic_energy_total/total_energy_j/1e6*100:.1f}%)")

        if latency_s is not None:
            print(f"  Idle/Leakage Energy ({idle_power_w}W × latency):  {idle_leakage_energy:8.3f} μJ  ({idle_leakage_energy/total_energy_j/1e6*100:.1f}%)")

        print(f"  {'─'*80}")
        print(f"  TOTAL KPU ENERGY:                     {total_energy_j*1e6:8.3f} μJ")

        # Metrics
        if 'energy_per_mac_pj' in arch_specific_events:
            print(f"\n  EFFICIENCY METRICS:")
            print(f"  • Energy per MAC:                     {arch_specific_events['energy_per_mac_pj']:.2f} pJ")
            print(f"  • Arithmetic Intensity:               {arch_specific_events.get('arithmetic_intensity', 0):.2f} ops/byte")
            print(f"  • Compute vs Memory:                  {arch_specific_events.get('compute_percentage', 0):.1f}% compute")

        print(f"\n  WHY SO EFFICIENT? Token-based distributed dataflow:")
        print(f"  • No instruction fetch/decode (dataflow, not stored-program)")
        print(f"  • No coherence machinery (explicit spatial token routing vs random cache coherence)")
        print(f"  • 4-stage memory hierarchy reduces DRAM traffic")
        print(f"  • 3 specialized data movement engines (DMA, BlockMover, Streamer) to implement system level execution schedules")
    else:
        print(f"\nNo detailed KPU energy breakdown data available.")


def print_dsp_hierarchical_breakdown(
    arch_specific_events: Dict[str, float],
    total_energy_j: float,
    compute_energy_j: float = 0.0,
    memory_energy_j: float = 0.0,
    latency_s: Optional[float] = None,
    idle_power_w: float = 10.0
) -> None:
    """
    Print hierarchical energy breakdown for DSP (VLIW heterogeneous architecture).

    Args:
        arch_specific_events: Dict of architectural energy events (from VLIWEnergyModel)
        total_energy_j: Total energy consumption in Joules
        compute_energy_j: Baseline compute energy in Joules
        memory_energy_j: Baseline memory energy in Joules
        latency_s: Latency in seconds (for idle/leakage calculation)
        idle_power_w: Idle power consumption in Watts (default: 10W)
    """
    print(f"\n{'─'*80}")
    print(f"DSP (VLIW HETEROGENEOUS) ENERGY BREAKDOWN")
    print(f"{'─'*80}")

    # Check if we have detailed VLIW energy model data
    if 'scalar_alu_energy' in arch_specific_events:
        # Component 1: Heterogeneous Execution Units
        print(f"\n  1. HETEROGENEOUS EXECUTION UNITS")
        scalar_alu = arch_specific_events.get('scalar_alu_energy', 0) * 1e6
        vector_unit = arch_specific_events.get('vector_unit_energy', 0) * 1e6
        tensor_unit = arch_specific_events.get('tensor_unit_energy', 0) * 1e6
        load_store = arch_specific_events.get('load_store_energy', 0) * 1e6
        scalar_ops = arch_specific_events.get('scalar_ops', 0)
        vector_ops = arch_specific_events.get('vector_ops', 0)
        tensor_ops = arch_specific_events.get('tensor_ops', 0)
        load_store_ops = arch_specific_events.get('load_store_ops', 0)
        exec_total = scalar_alu + vector_unit + tensor_unit + load_store

        print(f"     • Scalar ALU:                       {scalar_alu:8.3f} μJ  ({scalar_alu/exec_total*100 if exec_total > 0 else 0:5.1f}%)  [{scalar_ops:,} ops]")
        print(f"     • Vector Unit (HVX/SIMD):           {vector_unit:8.3f} μJ  ({vector_unit/exec_total*100 if exec_total > 0 else 0:5.1f}%)  [{vector_ops:,} ops]")
        print(f"     • Tensor Accelerator (HTA/MMA):     {tensor_unit:8.3f} μJ  ({tensor_unit/exec_total*100 if exec_total > 0 else 0:5.1f}%)  [{tensor_ops:,} ops]")
        print(f"     • Load/Store Unit:                  {load_store:8.3f} μJ  ({load_store/exec_total*100 if exec_total > 0 else 0:5.1f}%)  [{load_store_ops:,} ops]")
        print(f"     └─ Subtotal:                        {exec_total:8.3f} μJ")

        # Component 2: VLIW Instruction Bundling
        print(f"\n  2. VLIW INSTRUCTION BUNDLING")
        inst_fetch = arch_specific_events.get('instruction_fetch_energy', 0) * 1e6
        inst_decode = arch_specific_events.get('instruction_decode_energy', 0) * 1e6
        bundle_dispatch = arch_specific_events.get('bundle_dispatch_energy', 0) * 1e6
        num_bundles = arch_specific_events.get('num_instruction_bundles', 0)
        avg_bundle_width = arch_specific_events.get('average_bundle_width', 0)
        vliw_total = inst_fetch + inst_decode + bundle_dispatch

        print(f"     • Instruction Fetch (wide bundles): {inst_fetch:8.3f} μJ  ({inst_fetch/vliw_total*100 if vliw_total > 0 else 0:5.1f}%)  [{num_bundles:,} bundles]")
        print(f"     • Instruction Decode:               {inst_decode:8.3f} μJ  ({inst_decode/vliw_total*100 if vliw_total > 0 else 0:5.1f}%)")
        print(f"     • Bundle Dispatch:                  {bundle_dispatch:8.3f} μJ  ({bundle_dispatch/vliw_total*100 if vliw_total > 0 else 0:5.1f}%)")
        print(f"     • Average Bundle Width:             {avg_bundle_width:.1f} ops/bundle")
        print(f"     └─ Subtotal:                        {vliw_total:8.3f} μJ")

        # Component 3: Memory Hierarchy
        print(f"\n  3. MEMORY HIERARCHY")
        l1_cache = arch_specific_events.get('l1_cache_energy', 0) * 1e6
        l2_cache = arch_specific_events.get('l2_cache_energy', 0) * 1e6
        dram = arch_specific_events.get('dram_energy', 0) * 1e6
        l1_accesses = arch_specific_events.get('l1_accesses', 0)
        l2_accesses = arch_specific_events.get('l2_accesses', 0)
        dram_accesses = arch_specific_events.get('dram_accesses', 0)
        mem_total = l1_cache + l2_cache + dram

        print(f"     • L1 Cache (per-unit):              {l1_cache:8.3f} μJ  ({l1_cache/mem_total*100 if mem_total > 0 else 0:5.1f}%)  [{l1_accesses:,} accesses]")
        print(f"     • L2 Cache (shared):                {l2_cache:8.3f} μJ  ({l2_cache/mem_total*100 if mem_total > 0 else 0:5.1f}%)  [{l2_accesses:,} accesses]")
        print(f"     • DRAM (LPDDR):                     {dram:8.3f} μJ  ({dram/mem_total*100 if mem_total > 0 else 0:5.1f}%)  [{dram_accesses:,} accesses]")
        print(f"     └─ Subtotal:                        {mem_total:8.3f} μJ")

        # Component 4: Register File
        print(f"\n  4. REGISTER FILE (Large register banks)")
        reg_file = arch_specific_events.get('register_file_energy', 0) * 1e6
        num_reg_accesses = arch_specific_events.get('register_accesses', 0)

        print(f"     • Register File Access:             {reg_file:8.3f} μJ  [{num_reg_accesses:,} accesses]")
        print(f"     └─ Subtotal:                        {reg_file:8.3f} μJ")

        # Total
        arch_overhead = exec_total + vliw_total + mem_total + reg_file
        dynamic_energy_total = arch_overhead + compute_energy_j*1e6 + memory_energy_j*1e6

        # Calculate idle/leakage energy
        idle_leakage_energy = 0.0
        if latency_s is not None:
            idle_leakage_energy = total_energy_j*1e6 - dynamic_energy_total

        print(f"\n  TOTAL DSP ARCHITECTURAL OVERHEAD:     {arch_overhead:8.3f} μJ")
        print(f"  Base Compute Energy:                  {compute_energy_j*1e6:8.3f} μJ")
        print(f"  Base Memory Energy:                   {memory_energy_j*1e6:8.3f} μJ")
        print(f"  {'─'*80}")
        print(f"  SUBTOTAL DYNAMIC ENERGY:              {dynamic_energy_total:8.3f} μJ  ({dynamic_energy_total/total_energy_j/1e6*100:.1f}%)")

        if latency_s is not None:
            print(f"  Idle/Leakage Energy ({idle_power_w}W × latency):  {idle_leakage_energy:8.3f} μJ  ({idle_leakage_energy/total_energy_j/1e6*100:.1f}%)")

        print(f"  {'─'*80}")
        print(f"  TOTAL DSP ENERGY:                     {total_energy_j*1e6:8.3f} μJ")

        print(f"\n  DSP CHARACTERISTICS:")
        print(f"  • VLIW bundles: {avg_bundle_width:.1f} ops/bundle (high instruction-level parallelism)")
        print(f"  • Heterogeneous units: scalar + vector + tensor accelerators")
        print(f"  • Explicit parallelism (compiler-scheduled vs runtime-scheduled)")
        print(f"  • Energy efficiency through specialized units")
    else:
        print(f"\nNote: Detailed VLIW energy breakdown not yet available.")
        print(f"      (VLIWEnergyModel not configured for this DSP)")
        print(f"\nBasic energy stats:")
        print(f"  Total Energy: {total_energy_j * 1000:.3f} mJ")
        print(f"  Compute Energy: {compute_energy_j * 1000:.3f} mJ")
        print(f"  Memory Energy: {memory_energy_j * 1000:.3f} mJ")


def print_dpu_hierarchical_breakdown(
    arch_specific_events: Dict[str, float],
    total_energy_j: float,
    compute_energy_j: float = 0.0,
    memory_energy_j: float = 0.0,
    latency_s: Optional[float] = None,
    idle_power_w: float = 17.5
) -> None:
    """
    Print hierarchical energy breakdown for DPU (FPGA-based reconfigurable architecture).

    Args:
        arch_specific_events: Dict of architectural energy events (from DPUEnergyModel)
        total_energy_j: Total energy consumption in Joules
        compute_energy_j: Baseline compute energy in Joules
        memory_energy_j: Baseline memory energy in Joules
        latency_s: Latency in seconds (for idle/leakage calculation)
        idle_power_w: Idle power consumption in Watts (default: 17.5W for Vitis AI)
    """
    print(f"\n{'─'*80}")
    print(f"DPU (FPGA RECONFIGURABLE) ENERGY BREAKDOWN")
    print(f"{'─'*80}")

    # Check if we have detailed DPU energy model data
    if 'aie_tile_energy' in arch_specific_events:
        # Component 1: AIE (AI Engine) Tiles
        print(f"\n  1. AIE TILES (Reconfigurable compute)")
        aie_tile = arch_specific_events.get('aie_tile_energy', 0) * 1e6
        num_aie_tiles = arch_specific_events.get('num_aie_tiles', 0)
        aie_ops = arch_specific_events.get('aie_ops', 0)

        print(f"     • AIE Tile Computation:             {aie_tile:8.3f} μJ  [{num_aie_tiles} tiles, {aie_ops:,} ops]")
        print(f"     └─ Subtotal:                        {aie_tile:8.3f} μJ")

        # Component 2: Memory Hierarchy (AIE local + NoC + DDR)
        print(f"\n  2. MEMORY HIERARCHY (Tile-local → NoC → DDR)")
        tile_mem = arch_specific_events.get('tile_memory_energy', 0) * 1e6
        noc = arch_specific_events.get('noc_energy', 0) * 1e6
        ddr = arch_specific_events.get('ddr_energy', 0) * 1e6
        tile_mem_accesses = arch_specific_events.get('tile_memory_accesses', 0)
        noc_transfers = arch_specific_events.get('noc_transfers', 0)
        ddr_accesses = arch_specific_events.get('ddr_accesses', 0)
        mem_total = tile_mem + noc + ddr

        print(f"     • Tile Local Memory (64KB):         {tile_mem:8.3f} μJ  ({tile_mem/mem_total*100 if mem_total > 0 else 0:5.1f}%)  [{tile_mem_accesses:,} accesses]")
        print(f"     • NoC (Network-on-Chip):            {noc:8.3f} μJ  ({noc/mem_total*100 if mem_total > 0 else 0:5.1f}%)  [{noc_transfers:,} transfers]")
        print(f"     • DDR (off-chip):                   {ddr:8.3f} μJ  ({ddr/mem_total*100 if mem_total > 0 else 0:5.1f}%)  [{ddr_accesses:,} accesses]")
        print(f"     └─ Subtotal:                        {mem_total:8.3f} μJ")

        # Component 3: Reconfiguration Overhead
        print(f"\n  3. RECONFIGURATION OVERHEAD (FPGA bitstream loading)")
        reconfig = arch_specific_events.get('reconfiguration_energy', 0) * 1e6
        num_reconfigs = arch_specific_events.get('num_reconfigurations', 0)
        reconfig_cycles = arch_specific_events.get('reconfiguration_cycles', 0)

        print(f"     • Bitstream Loading:                {reconfig:8.3f} μJ  [{num_reconfigs} reconfigs, {reconfig_cycles:,} cycles]")
        print(f"     └─ Subtotal:                        {reconfig:8.3f} μJ")

        # Component 4: Control Logic
        print(f"\n  4. CONTROL LOGIC (PS ARM cores)")
        control = arch_specific_events.get('control_logic_energy', 0) * 1e6

        print(f"     • ARM Control Processors:           {control:8.3f} μJ")
        print(f"     └─ Subtotal:                        {control:8.3f} μJ")

        # Total
        arch_overhead = aie_tile + mem_total + reconfig + control
        dynamic_energy_total = arch_overhead + compute_energy_j*1e6 + memory_energy_j*1e6

        # Calculate idle/leakage energy
        idle_leakage_energy = 0.0
        if latency_s is not None:
            idle_leakage_energy = total_energy_j*1e6 - dynamic_energy_total

        print(f"\n  TOTAL DPU ARCHITECTURAL OVERHEAD:     {arch_overhead:8.3f} μJ")
        print(f"  Base Compute Energy:                  {compute_energy_j*1e6:8.3f} μJ")
        print(f"  Base Memory Energy:                   {memory_energy_j*1e6:8.3f} μJ")
        print(f"  {'─'*80}")
        print(f"  SUBTOTAL DYNAMIC ENERGY:              {dynamic_energy_total:8.3f} μJ  ({dynamic_energy_total/total_energy_j/1e6*100:.1f}%)")

        if latency_s is not None:
            print(f"  Idle/Leakage Energy ({idle_power_w}W × latency):  {idle_leakage_energy:8.3f} μJ  ({idle_leakage_energy/total_energy_j/1e6*100:.1f}%)")

        print(f"  {'─'*80}")
        print(f"  TOTAL DPU ENERGY:                     {total_energy_j*1e6:8.3f} μJ")

        print(f"\n  DPU CHARACTERISTICS:")
        print(f"  • Reconfigurable FPGA fabric with AI Engine tiles")
        print(f"  • Reconfiguration overhead: {reconfig:.3f} μJ ({num_reconfigs} reconfigs)")
        print(f"  • Flexibility vs efficiency tradeoff (more flexible than ASIC, less efficient)")
        print(f"  • Good for multi-model deployment and algorithm exploration")
    else:
        print(f"\nNote: Detailed DPU energy breakdown not yet available.")
        print(f"      (DPUEnergyModel not configured)")
        print(f"\nBasic energy stats:")
        print(f"  Total Energy: {total_energy_j * 1000:.3f} mJ")
        print(f"  Compute Energy: {compute_energy_j * 1000:.3f} mJ")
        print(f"  Memory Energy: {memory_energy_j * 1000:.3f} mJ")


def print_cgra_hierarchical_breakdown(
    arch_specific_events: Dict[str, float],
    total_energy_j: float,
    compute_energy_j: float = 0.0,
    memory_energy_j: float = 0.0,
    latency_s: Optional[float] = None,
    idle_power_w: float = 15.0
) -> None:
    """
    Print hierarchical energy breakdown for CGRA (Coarse-Grained Reconfigurable Array).

    Args:
        arch_specific_events: Dict of architectural energy events (from CGRAEnergyModel)
        total_energy_j: Total energy consumption in Joules
        compute_energy_j: Baseline compute energy in Joules
        memory_energy_j: Baseline memory energy in Joules
        latency_s: Latency in seconds (for idle/leakage calculation)
        idle_power_w: Idle power consumption in Watts (default: 15W)
    """
    print(f"\n{'─'*80}")
    print(f"CGRA (COARSE-GRAINED RECONFIGURABLE) ENERGY BREAKDOWN")
    print(f"{'─'*80}")

    # Check if we have detailed CGRA energy model data
    if 'pcu_energy' in arch_specific_events:
        # Component 1: PCUs (Pattern Compute Units)
        print(f"\n  1. PCUs (Pattern Compute Units)")
        pcu = arch_specific_events.get('pcu_energy', 0) * 1e6
        num_pcus = arch_specific_events.get('num_pcus', 0)
        pcu_ops = arch_specific_events.get('pcu_ops', 0)

        print(f"     • PCU Computation:                  {pcu:8.3f} μJ  [{num_pcus} PCUs, {pcu_ops:,} ops]")
        print(f"     └─ Subtotal:                        {pcu:8.3f} μJ")

        # Component 2: PMUs (Pattern Memory Units)
        print(f"\n  2. PMUs (Pattern Memory Units)")
        pmu = arch_specific_events.get('pmu_energy', 0) * 1e6
        num_pmus = arch_specific_events.get('num_pmus', 0)
        pmu_accesses = arch_specific_events.get('pmu_accesses', 0)

        print(f"     • PMU Scratchpads:                  {pmu:8.3f} μJ  [{num_pmus} PMUs, {pmu_accesses:,} accesses]")
        print(f"     └─ Subtotal:                        {pmu:8.3f} μJ")

        # Component 3: Interconnect Network
        print(f"\n  3. INTERCONNECT NETWORK (Switch boxes)")
        interconnect = arch_specific_events.get('interconnect_energy', 0) * 1e6
        num_hops = arch_specific_events.get('average_hops', 0)
        num_transfers = arch_specific_events.get('num_transfers', 0)

        print(f"     • Switch Box Network:               {interconnect:8.3f} μJ  [{num_transfers:,} transfers, {num_hops:.1f} avg hops]")
        print(f"     └─ Subtotal:                        {interconnect:8.3f} μJ")

        # Component 4: Reconfiguration Overhead
        print(f"\n  4. RECONFIGURATION OVERHEAD (Configuration loading)")
        reconfig = arch_specific_events.get('reconfiguration_energy', 0) * 1e6
        num_reconfigs = arch_specific_events.get('num_reconfigurations', 0)
        reconfig_cycles = arch_specific_events.get('reconfiguration_cycles', 0)

        print(f"     • Configuration Loading:            {reconfig:8.3f} μJ  [{num_reconfigs} reconfigs, {reconfig_cycles:,} cycles]")
        print(f"     └─ Subtotal:                        {reconfig:8.3f} μJ")

        # Component 5: Off-chip Memory
        print(f"\n  5. OFF-CHIP MEMORY (DRAM)")
        dram = arch_specific_events.get('dram_energy', 0) * 1e6
        dram_accesses = arch_specific_events.get('dram_accesses', 0)

        print(f"     • DRAM Access:                      {dram:8.3f} μJ  [{dram_accesses:,} accesses]")
        print(f"     └─ Subtotal:                        {dram:8.3f} μJ")

        # Total
        arch_overhead = pcu + pmu + interconnect + reconfig + dram
        dynamic_energy_total = arch_overhead + compute_energy_j*1e6 + memory_energy_j*1e6

        # Calculate idle/leakage energy
        idle_leakage_energy = 0.0
        if latency_s is not None:
            idle_leakage_energy = total_energy_j*1e6 - dynamic_energy_total

        print(f"\n  TOTAL CGRA ARCHITECTURAL OVERHEAD:    {arch_overhead:8.3f} μJ")
        print(f"  Base Compute Energy:                  {compute_energy_j*1e6:8.3f} μJ")
        print(f"  Base Memory Energy:                   {memory_energy_j*1e6:8.3f} μJ")
        print(f"  {'─'*80}")
        print(f"  SUBTOTAL DYNAMIC ENERGY:              {dynamic_energy_total:8.3f} μJ  ({dynamic_energy_total/total_energy_j/1e6*100:.1f}%)")

        if latency_s is not None:
            print(f"  Idle/Leakage Energy ({idle_power_w}W × latency):  {idle_leakage_energy:8.3f} μJ  ({idle_leakage_energy/total_energy_j/1e6*100:.1f}%)")

        print(f"  {'─'*80}")
        print(f"  TOTAL CGRA ENERGY:                    {total_energy_j*1e6:8.3f} μJ")

        print(f"\n  CGRA CHARACTERISTICS:")
        print(f"  • Spatial dataflow with explicit reconfiguration")
        print(f"  • Reconfiguration overhead: {reconfig:.3f} μJ ({num_reconfigs} reconfigs, {reconfig_cycles:,} cycles)")
        print(f"  • Coarser granularity than FPGA (PCU/PMU vs LUT/FF)")
        print(f"  • More efficient than FPGA, less flexible than software")
    else:
        print(f"\nNote: Detailed CGRA energy breakdown not yet available.")
        print(f"      (CGRAEnergyModel not configured)")
        print(f"\nBasic energy stats:")
        print(f"  Total Energy: {total_energy_j * 1000:.3f} mJ")
        print(f"  Compute Energy: {compute_energy_j * 1000:.3f} mJ")
        print(f"  Memory Energy: {memory_energy_j * 1000:.3f} mJ")


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


# ============================================================================
# Export Functions
# ============================================================================

def export_energy_results(
    output_path: str,
    architecture: str,
    hardware_name: str,
    model_name: str,
    batch_size: int,
    precision: str,
    total_energy_j: float,
    latency_s: float,
    throughput_inf_per_s: float,
    compute_energy_j: float = 0.0,
    memory_energy_j: float = 0.0,
    static_energy_j: float = 0.0,
    utilization: float = 0.0,
    num_subgraphs: int = 0,
    arch_specific_events: Optional[Dict[str, float]] = None,
    subgraph_allocations: Optional[list] = None
) -> None:
    """
    Export energy analysis results to JSON or CSV based on file extension.

    Args:
        output_path: Output file path (.json or .csv)
        architecture: Architecture type (CPU, GPU, TPU, KPU, DSP, DPU, CGRA)
        hardware_name: Hardware configuration name
        model_name: DNN model name
        batch_size: Batch size used
        precision: Numerical precision (FP32, FP16, INT8, etc.)
        total_energy_j: Total energy in Joules
        latency_s: Latency in seconds
        throughput_inf_per_s: Throughput in inferences/sec
        compute_energy_j: Baseline compute energy in Joules
        memory_energy_j: Baseline memory energy in Joules
        static_energy_j: Static/idle energy in Joules
        utilization: Hardware utilization (0.0-1.0)
        num_subgraphs: Number of fused subgraphs
        arch_specific_events: Architecture-specific energy breakdown events
        subgraph_allocations: Per-subgraph allocation details
    """
    path = Path(output_path)
    suffix = path.suffix.lower()

    if suffix == '.json':
        _export_json(
            output_path, architecture, hardware_name, model_name, batch_size,
            precision, total_energy_j, latency_s, throughput_inf_per_s,
            compute_energy_j, memory_energy_j, static_energy_j, utilization,
            num_subgraphs, arch_specific_events, subgraph_allocations
        )
    elif suffix == '.csv':
        _export_csv(
            output_path, architecture, hardware_name, model_name, batch_size,
            precision, total_energy_j, latency_s, throughput_inf_per_s,
            compute_energy_j, memory_energy_j, static_energy_j, utilization,
            num_subgraphs, arch_specific_events
        )
    else:
        raise ValueError(f"Unsupported output format: {suffix}. Use .json or .csv")

    print(f"\n✓ Results exported to: {output_path}")


def _export_json(
    output_path: str,
    architecture: str,
    hardware_name: str,
    model_name: str,
    batch_size: int,
    precision: str,
    total_energy_j: float,
    latency_s: float,
    throughput_inf_per_s: float,
    compute_energy_j: float,
    memory_energy_j: float,
    static_energy_j: float,
    utilization: float,
    num_subgraphs: int,
    arch_specific_events: Optional[Dict[str, float]],
    subgraph_allocations: Optional[list]
) -> None:
    """Export results to JSON format with full hierarchical structure."""

    # Build result dictionary
    result = {
        "metadata": {
            "architecture": architecture,
            "hardware": hardware_name,
            "model": model_name,
            "batch_size": batch_size,
            "precision": precision,
            "num_subgraphs": num_subgraphs
        },
        "performance": {
            "latency_ms": latency_s * 1000,
            "latency_s": latency_s,
            "throughput_inf_per_s": throughput_inf_per_s,
            "utilization": utilization
        },
        "energy": {
            "total_energy_mj": total_energy_j * 1000,
            "total_energy_j": total_energy_j,
            "energy_per_inference_mj": (total_energy_j / batch_size) * 1000,
            "energy_per_inference_j": total_energy_j / batch_size,
            "breakdown": {
                "compute_energy_mj": compute_energy_j * 1000,
                "compute_energy_j": compute_energy_j,
                "memory_energy_mj": memory_energy_j * 1000,
                "memory_energy_j": memory_energy_j,
                "static_idle_energy_mj": static_energy_j * 1000,
                "static_idle_energy_j": static_energy_j
            }
        }
    }

    # Add architecture-specific events if available
    if arch_specific_events:
        result["architectural_breakdown"] = {}
        for key, value in arch_specific_events.items():
            # Convert energy values to μJ for readability
            if 'energy' in key.lower():
                result["architectural_breakdown"][key] = {
                    "value_uj": value * 1e6,
                    "value_j": value
                }
            else:
                result["architectural_breakdown"][key] = value

    # Add per-subgraph details if available
    if subgraph_allocations:
        result["subgraphs"] = []
        for i, alloc in enumerate(subgraph_allocations):
            sg_data = {
                "subgraph_id": i,
                "compute_time_s": getattr(alloc, 'compute_time', 0),
                "compute_energy_mj": getattr(alloc, 'compute_energy', 0) * 1000,
                "memory_energy_mj": getattr(alloc, 'memory_energy', 0) * 1000,
                "total_energy_mj": (
                    getattr(alloc, 'compute_energy', 0) +
                    getattr(alloc, 'memory_energy', 0)
                ) * 1000
            }
            result["subgraphs"].append(sg_data)

    # Write to file
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)


def _export_csv(
    output_path: str,
    architecture: str,
    hardware_name: str,
    model_name: str,
    batch_size: int,
    precision: str,
    total_energy_j: float,
    latency_s: float,
    throughput_inf_per_s: float,
    compute_energy_j: float,
    memory_energy_j: float,
    static_energy_j: float,
    utilization: float,
    num_subgraphs: int,
    arch_specific_events: Optional[Dict[str, float]]
) -> None:
    """Export results to CSV format (flat structure, one row per analysis)."""

    # Build row dictionary with standard columns
    row = {
        'architecture': architecture,
        'hardware': hardware_name,
        'model': model_name,
        'batch_size': batch_size,
        'precision': precision,
        'num_subgraphs': num_subgraphs,
        'latency_ms': latency_s * 1000,
        'throughput_inf_per_s': throughput_inf_per_s,
        'utilization': utilization,
        'total_energy_mj': total_energy_j * 1000,
        'energy_per_inference_mj': (total_energy_j / batch_size) * 1000,
        'compute_energy_mj': compute_energy_j * 1000,
        'memory_energy_mj': memory_energy_j * 1000,
        'static_idle_energy_mj': static_energy_j * 1000
    }

    # Add architecture-specific events as columns
    if arch_specific_events:
        for key, value in arch_specific_events.items():
            # Convert energy values to μJ for readability
            if 'energy' in key.lower():
                row[f'{key}_uj'] = value * 1e6
            else:
                row[key] = value

    # Check if file exists to determine if we need to write header
    file_exists = Path(output_path).exists()

    # Write to CSV (append mode if file exists)
    with open(output_path, 'a' if file_exists else 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
