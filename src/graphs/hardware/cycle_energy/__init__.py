"""
Cycle-Level Energy Models for Architecture Comparison

This package provides detailed cycle-level energy models for different
processor architectures, enabling apples-to-apples energy comparisons.

Architecture Classes:
- Stored Program Machines: CPU (MIMD), GPU (SIMT), DSP (VLIW)
- Dataflow Architectures: TPU (Systolic), KPU (Spatial Dataflow)

Each model breaks down energy into:
- Instruction overhead (fetch, decode)
- Compute (ALU, MAC, tensor units)
- Register file access
- Memory hierarchy (L1, L2, L3, DRAM/HBM)
- Architecture-specific overhead (SIMT coherence, systolic data movement, etc.)
"""

from .base import (
    CyclePhase,
    OperatingMode,
    OperatorType,
    MemoryType,
    HitRatios,
    DEFAULT_HIT_RATIOS,
    # Cache sizes by architecture
    GPU_L2_CACHE_SIZES,
    CPU_L3_CACHE_SIZES,
    TPU_L2_SRAM_SIZES,
    KPU_L1_STREAMING_BUFFER_SIZES,
    KPU_L2_TILE_STAGING_SIZES,
    KPU_L3_GLOBAL_SCRATCHPAD_SIZES,
    DEFAULT_L2_CACHE_SIZES,  # backward compat
    # Energy event classes
    EnergyEvent,
    CycleEnergyBreakdown,
    get_mode_description,
    # Cache hit ratio computation
    compute_cache_hit_ratio,
    will_flush_cache,
    compute_l2_hit_ratio,  # backward compat
    will_flush_l2,  # backward compat
)

from .cpu import build_cpu_cycle_energy
from .gpu import build_gpu_cycle_energy
from .dsp import build_dsp_cycle_energy
from .tpu import build_tpu_cycle_energy
from .kpu import build_kpu_cycle_energy

from .comparison import (
    format_energy,
    format_phase_breakdown,
    format_comparison_table,
    format_key_insights,
    SweepResult,
    run_sweep,
    run_mode_sweep,
    format_sweep_table,
    format_mode_comparison_table,
    # Consistent-scale formatting
    ENERGY_SCALES,
    determine_common_scale,
    format_energy_with_scale,
    # 3-category energy breakdown
    format_energy_categories_table,
    format_energy_categories_per_op_table,
)

__all__ = [
    # Base classes
    'CyclePhase',
    'OperatingMode',
    'OperatorType',
    'MemoryType',
    'HitRatios',
    'DEFAULT_HIT_RATIOS',
    # Cache sizes by architecture
    'GPU_L2_CACHE_SIZES',
    'CPU_L3_CACHE_SIZES',
    'TPU_L2_SRAM_SIZES',
    'KPU_L1_STREAMING_BUFFER_SIZES',
    'KPU_L2_TILE_STAGING_SIZES',
    'KPU_L3_GLOBAL_SCRATCHPAD_SIZES',
    'DEFAULT_L2_CACHE_SIZES',
    # Energy event classes
    'EnergyEvent',
    'CycleEnergyBreakdown',
    'get_mode_description',
    # Cache hit ratio computation
    'compute_cache_hit_ratio',
    'will_flush_cache',
    'compute_l2_hit_ratio',
    'will_flush_l2',
    # Architecture models
    'build_cpu_cycle_energy',
    'build_gpu_cycle_energy',
    'build_dsp_cycle_energy',
    'build_tpu_cycle_energy',
    'build_kpu_cycle_energy',
    # Comparison utilities
    'format_energy',
    'format_phase_breakdown',
    'format_comparison_table',
    'format_key_insights',
    'SweepResult',
    'run_sweep',
    'run_mode_sweep',
    'format_sweep_table',
    'format_mode_comparison_table',
    # Consistent-scale formatting
    'ENERGY_SCALES',
    'determine_common_scale',
    'format_energy_with_scale',
    # 3-category energy breakdown
    'format_energy_categories_table',
    'format_energy_categories_per_op_table',
]
