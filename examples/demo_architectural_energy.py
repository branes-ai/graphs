#!/usr/bin/env python3
"""
Demonstration: Architectural Energy Event Modeling

This example demonstrates how different hardware architectures have
fundamentally different energy characteristics based on their resource
contention management mechanisms.

Architecture progression (toward energy efficiency):
1. CPU (Stored Program) - sequential with modest parallelism
2. GPU (Data Parallel) - SIMT with massive parallelism + coherence overhead
3. TPU (Systolic Array) - fixed spatial schedule, eliminates instruction fetch
4. KPU (Domain Flow) - programmable spatial schedule with domain tracking

For the same computational workload, we show:
- Baseline energy (compute + memory)
- Architectural overhead/savings
- Energy event breakdown
- Total energy comparison

Goal: Quantify WHY systolic arrays are more energy efficient than GPUs,
and explain the trade-offs in the CPU → GPU → TPU → KPU progression.
"""

from graphs.hardware.architectural_energy import (
    StoredProgramEnergyModel,
    DataParallelEnergyModel,
    SystolicArrayEnergyModel,
    DomainFlowEnergyModel,
)


def demonstrate_architectural_energy_comparison():
    """
    Compare architectural energy for a typical ResNet-18 Conv layer.

    Workload:
    - 1.8 GFLOPS (900M MACs × 2)
    - 10 MB data transfer (input + weights + output)
    - Batch size: 1
    - Precision: FP32
    """

    print("=" * 80)
    print("ARCHITECTURAL ENERGY COMPARISON")
    print("=" * 80)
    print()
    print("Workload: ResNet-18 Conv2D Layer (batch=1, FP32)")
    print("  Operations: 1.8 GFLOPS (900M MACs)")
    print("  Data Transfer: 10 MB (input + weights + output)")
    print()

    # Workload parameters
    ops = int(1.8e9)  # 1.8 billion FLOPs
    bytes_transferred = 10 * 1024 * 1024  # 10 MB

    # Baseline energy (what all architectures would use without overhead)
    baseline_energy_per_flop = 1.0e-12  # 1 pJ per FLOP (typical)
    baseline_energy_per_byte = 10.0e-12  # 10 pJ per byte (typical DRAM)

    baseline_compute = ops * baseline_energy_per_flop
    baseline_memory = bytes_transferred * baseline_energy_per_byte
    baseline_total = baseline_compute + baseline_memory

    print("Baseline Energy (no architectural overhead):")
    print(f"  Compute: {baseline_compute*1e9:.2f} nJ ({ops/1e9:.1f} GFLOPS × {baseline_energy_per_flop*1e12:.1f} pJ)")
    print(f"  Memory:  {baseline_memory*1e9:.2f} nJ ({bytes_transferred/1e6:.1f} MB × {baseline_energy_per_byte*1e12:.1f} pJ)")
    print(f"  Total:   {baseline_total*1e9:.2f} nJ")
    print()

    # ========================================================================
    # 1. CPU (Intel Xeon) - Stored Program Architecture
    # ========================================================================
    print("-" * 80)
    print("1. CPU (Intel Xeon) - Stored Program Architecture")
    print("-" * 80)
    print()

    cpu_model = StoredProgramEnergyModel(
        instruction_fetch_energy=2.0e-12,
        operand_fetch_overhead=10.0e-12,
        branch_prediction_overhead=0.3e-12,
    )

    cpu_context = {
        'cache_line_size': 64,  # x86 cache line
    }

    cpu_breakdown = cpu_model.compute_architectural_energy(
        ops=ops,
        bytes_transferred=bytes_transferred,
        compute_energy_baseline=baseline_compute,
        memory_energy_baseline=baseline_memory,
        execution_context=cpu_context
    )

    cpu_total = baseline_total + cpu_breakdown.total_overhead

    print(cpu_breakdown.explanation)
    print()
    print(f"Total Energy: {cpu_total*1e9:.2f} nJ")
    print(f"  Baseline: {baseline_total*1e9:.2f} nJ")
    print(f"  Architectural Overhead: {cpu_breakdown.total_overhead*1e9:.2f} nJ")
    print()

    # ========================================================================
    # 2. GPU (H100) - Data Parallel Architecture
    # ========================================================================
    print("-" * 80)
    print("2. GPU (NVIDIA H100) - Data Parallel SIMT Architecture")
    print("-" * 80)
    print()

    gpu_model = DataParallelEnergyModel(
        instruction_fetch_energy=2.0e-12,
        operand_fetch_overhead=10.0e-12,
        coherence_energy_per_request=5.0e-12,  # GPU-specific!
        thread_scheduling_overhead=1.0e-12,
        warp_divergence_penalty=3.0e-12,
        memory_coalescing_overhead=2.0e-12,
    )

    gpu_context = {
        'concurrent_threads': 200_000,  # Typical for ResNet Conv layer
        'warp_size': 32,
        'cache_line_size': 128,  # H100 cache line
    }

    gpu_breakdown = gpu_model.compute_architectural_energy(
        ops=ops,
        bytes_transferred=bytes_transferred,
        compute_energy_baseline=baseline_compute,
        memory_energy_baseline=baseline_memory,
        execution_context=gpu_context
    )

    gpu_total = baseline_total + gpu_breakdown.total_overhead

    print(gpu_breakdown.explanation)
    print()
    print(f"Total Energy: {gpu_total*1e9:.2f} nJ")
    print(f"  Baseline: {baseline_total*1e9:.2f} nJ")
    print(f"  Architectural Overhead: {gpu_breakdown.total_overhead*1e9:.2f} nJ")
    print()

    # ========================================================================
    # 3. TPU (Google TPU v4) - Systolic Array
    # ========================================================================
    print("-" * 80)
    print("3. Google TPU v4 - Systolic Array Architecture")
    print("-" * 80)
    print()

    tpu_model = SystolicArrayEnergyModel(
        schedule_setup_energy=100.0e-12,
        data_injection_per_element=0.5e-12,
        data_extraction_per_element=0.5e-12,
        compute_efficiency=0.15,  # 85% reduction!
        memory_efficiency=0.20,   # 80% reduction!
    )

    tpu_context = {}

    tpu_breakdown = tpu_model.compute_architectural_energy(
        ops=ops,
        bytes_transferred=bytes_transferred,
        compute_energy_baseline=baseline_compute,
        memory_energy_baseline=baseline_memory,
        execution_context=tpu_context
    )

    tpu_total = baseline_total + tpu_breakdown.total_overhead

    print(tpu_breakdown.explanation)
    print()
    print(f"Total Energy: {tpu_total*1e9:.2f} nJ")
    print(f"  Baseline: {baseline_total*1e9:.2f} nJ")
    print(f"  Architectural Savings: {tpu_breakdown.total_overhead*1e9:.2f} nJ (negative = savings)")
    print()

    # ========================================================================
    # 4. KPU (Stillwater KPU-T256) - Domain Flow Architecture
    # ========================================================================
    print("-" * 80)
    print("4. Stillwater KPU-T256 - Domain Flow Architecture")
    print("-" * 80)
    print()

    kpu_model = DomainFlowEnergyModel(
        domain_tracking_per_op=1.0e-12,
        network_overlay_update=2.0e-12,
        wavefront_control=0.8e-12,
        schedule_adaptation_energy=50.0e-12,
        domain_data_injection=0.7e-12,
        domain_data_extraction=0.7e-12,
        compute_efficiency=0.30,  # 70% reduction
        memory_efficiency=0.35,   # 65% reduction
    )

    kpu_context = {
        'schedule_changes': 10,  # Adaptive scheduling
    }

    kpu_breakdown = kpu_model.compute_architectural_energy(
        ops=ops,
        bytes_transferred=bytes_transferred,
        compute_energy_baseline=baseline_compute,
        memory_energy_baseline=baseline_memory,
        execution_context=kpu_context
    )

    kpu_total = baseline_total + kpu_breakdown.total_overhead

    print(kpu_breakdown.explanation)
    print()
    print(f"Total Energy: {kpu_total*1e9:.2f} nJ")
    print(f"  Baseline: {baseline_total*1e9:.2f} nJ")
    print(f"  Architectural Savings: {kpu_breakdown.total_overhead*1e9:.2f} nJ")
    print()

    # ========================================================================
    # Summary Comparison
    # ========================================================================
    print("=" * 80)
    print("SUMMARY: Architecture Progression (Energy Comparison)")
    print("=" * 80)
    print()
    print(f"{'Architecture':<35} {'Energy (nJ)':<15} {'vs CPU':<12} {'vs GPU':<12} {'Progression'}")
    print("-" * 100)
    print(f"{'Baseline (no overhead)':<35} {baseline_total*1e9:>10.2f} nJ   {'—':<12} {'—':<12} {'Pure computation'}")
    print(f"{'1. CPU (Xeon) Sequential':<35} {cpu_total*1e9:>10.2f} nJ   {'1.00×':<12} {f'{cpu_total/gpu_total:.2f}×':<12} {'Instruction stream'}")
    print(f"{'2. GPU (H100) Data Parallel':<35} {gpu_total*1e9:>10.2f} nJ   {f'{gpu_total/cpu_total:.2f}×':<12} {'1.00×':<12} {'+ Coherence overhead'}")
    print(f"{'3. TPU (v4) Systolic Array':<35} {tpu_total*1e9:>10.2f} nJ   {f'{tpu_total/cpu_total:.2f}×':<12} {f'{tpu_total/gpu_total:.2f}×':<12} {'- Instruction fetch'}")
    print(f"{'4. KPU (T256) Domain Flow':<35} {kpu_total*1e9:>10.2f} nJ   {f'{kpu_total/cpu_total:.2f}×':<12} {f'{kpu_total/gpu_total:.2f}×':<12} {'+ Programmability'}")
    print()

    print("Architecture Progression Insights:")
    print()
    print(f"  CPU → GPU: {gpu_total/cpu_total:.2f}× WORSE")
    print(f"    GPU adds {gpu_breakdown.total_overhead*1e9:.2f} nJ coherence machinery overhead")
    print(f"    to manage {gpu_context['concurrent_threads']:,} concurrent threads.")
    print(f"    This is only worth it at large batch sizes!")
    print()
    print(f"  GPU → TPU: {gpu_total/tpu_total:.1f}× BETTER")
    print(f"    TPU SAVES {-tpu_breakdown.total_overhead*1e9:.2f} nJ by eliminating")
    print(f"    instruction fetch and using pre-designed spatial schedule.")
    print(f"    Fixed function limits flexibility.")
    print()
    print(f"  TPU → KPU: {kpu_total/tpu_total:.2f}× overhead")
    print(f"    KPU adds {(kpu_breakdown.total_overhead - tpu_breakdown.total_overhead)*1e9:.2f} nJ for domain tracking")
    print(f"    and programmable scheduling, but gains flexibility.")
    print(f"    Still {gpu_total/kpu_total:.1f}× more energy efficient than GPU!")
    print()
    print("CONCLUSION:")
    print("  The progression shows the trade-offs in resource contention management:")
    print()
    print("  CPU (Sequential)")
    print("    ↓ Add massive SIMT parallelism")
    print(f"  GPU (Data Parallel) - {gpu_total/cpu_total:.1f}× worse due to coherence machinery")
    print("    ↓ Eliminate instruction fetch, use spatial schedule")
    print(f"  TPU (Systolic Array) - {gpu_total/tpu_total:.0f}× better than GPU!")
    print("    ↓ Add programmability")
    print(f"  KPU (Domain Flow) - {gpu_total/kpu_total:.0f}× better than GPU, programmable")
    print()
    print(f"  Key: Systolic/domain flow architectures achieve {gpu_total/tpu_total:.0f}-{gpu_total/kpu_total:.0f}× better energy")
    print("  efficiency by eliminating the instruction fetch and memory contention")
    print("  overhead that dominates von Neumann (CPU/GPU) architectures.")
    print()


if __name__ == "__main__":
    demonstrate_architectural_energy_comparison()
