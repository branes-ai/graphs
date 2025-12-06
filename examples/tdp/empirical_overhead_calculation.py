"""
CORRECTED First Principles Analysis

The issue with the previous calculation is the energy per op was too low.
Let's use actual Horowitz ISSCC 2014 data (scaled to modern nodes).

From Horowitz ISSCC 2014 (45nm baseline):
    - 32-bit FP add:  0.9 pJ
    - 32-bit FP mul:  3.7 pJ
    - 32-bit FP FMA: ~4.6 pJ (add + mul)

Scaling to smaller nodes (energy scales ~linearly with feature size):
    - 14nm: 0.31x of 45nm -> FP32 FMA: ~1.4 pJ
    - 10nm: 0.22x of 45nm -> FP32 FMA: ~1.0 pJ
    - 8nm:  0.18x of 45nm -> FP32 FMA: ~0.8 pJ
    - 7nm:  0.16x of 45nm -> FP32 FMA: ~0.7 pJ
    - 5nm:  0.11x of 45nm -> FP32 FMA: ~0.5 pJ

But wait - the Horowitz numbers are for JUST the ALU.
Modern chips also have:
    - Register file access: ~0.5 pJ
    - Local wiring: ~0.5 pJ per 1mm
    - Scheduler overhead: varies

Let me use the technology_profile.py values which are calibrated for this codebase.
"""

print("="*80)
print("CORRECTED FIRST PRINCIPLES: Using Calibrated Energy Values")
print("="*80)

# Import the actual energy values from the codebase
import sys
sys.path.insert(0, 'src')
from graphs.hardware.technology_profile import (
    get_process_base_energy_pj,
    CIRCUIT_TYPE_MULTIPLIER,
)

# =============================================================================
# Intel Core i7-12700K (Intel 7 / 10nm equivalent)
# =============================================================================
print("\n" + "="*80)
print("Intel Core i7-12700K (Intel 7 = 10nm-class)")
print("="*80)

# Intel 7 is marketed as 10nm-class but more like 12nm in actual metrics
process_nm_intel = 12  # Use 12nm as approximation for Intel 7
base_energy_intel = get_process_base_energy_pj(process_nm_intel)
circuit_mult_x86 = CIRCUIT_TYPE_MULTIPLIER['x86_performance']
energy_per_fp32_intel = base_energy_intel * circuit_mult_x86

print(f"\nEnergy model (from technology_profile.py):")
print(f"  Process node:    {process_nm_intel}nm (Intel 7)")
print(f"  Base energy:     {base_energy_intel:.2f} pJ")
print(f"  x86_performance: {circuit_mult_x86:.2f}x")
print(f"  FP32 op energy:  {energy_per_fp32_intel:.2f} pJ")

# Empirical data
measured_gflops_cpu = 772.4
tdp_watts_cpu = 125.0

# Compute power with corrected energy
ops_per_sec_cpu = measured_gflops_cpu * 1e9
compute_power_cpu = ops_per_sec_cpu * energy_per_fp32_intel * 1e-12

print(f"\nCompute Power Calculation:")
print(f"  Measured throughput: {measured_gflops_cpu:.1f} GFLOPS")
print(f"  Energy per op:       {energy_per_fp32_intel:.2f} pJ")
print(f"  Compute power:       {compute_power_cpu:.1f} W")

compute_fraction_cpu = compute_power_cpu / tdp_watts_cpu
print(f"\nPower Breakdown (TDP = {tdp_watts_cpu:.0f} W):")
print(f"  Pure Compute:   {compute_power_cpu:>6.1f} W  ({compute_fraction_cpu*100:>5.1f}%)")
print(f"  Infrastructure: {tdp_watts_cpu - compute_power_cpu:>6.1f} W  ({(1-compute_fraction_cpu)*100:>5.1f}%)")

# =============================================================================
# NVIDIA Jetson Orin Nano (Samsung 8nm)
# =============================================================================
print("\n" + "="*80)
print("NVIDIA Jetson Orin Nano 8GB (Samsung 8nm)")
print("="*80)

process_nm_gpu = 8
base_energy_gpu = get_process_base_energy_pj(process_nm_gpu)
# CUDA cores are closer to tensor_core efficiency (0.85x)
circuit_mult_cuda = CIRCUIT_TYPE_MULTIPLIER['cuda_core']
energy_per_fp32_gpu = base_energy_gpu * circuit_mult_cuda

print(f"\nEnergy model (from technology_profile.py):")
print(f"  Process node:    {process_nm_gpu}nm (Samsung)")
print(f"  Base energy:     {base_energy_gpu:.2f} pJ")
print(f"  cuda_core mult:  {circuit_mult_cuda:.2f}x")
print(f"  FP32 op energy:  {energy_per_fp32_gpu:.2f} pJ")

# Empirical data
measured_gflops_gpu = 800.7
tdp_watts_gpu = 15.0

# Compute power with corrected energy
ops_per_sec_gpu = measured_gflops_gpu * 1e9
compute_power_gpu = ops_per_sec_gpu * energy_per_fp32_gpu * 1e-12

print(f"\nCompute Power Calculation:")
print(f"  Measured throughput: {measured_gflops_gpu:.1f} GFLOPS")
print(f"  Energy per op:       {energy_per_fp32_gpu:.2f} pJ")
print(f"  Compute power:       {compute_power_gpu:.1f} W")

compute_fraction_gpu = compute_power_gpu / tdp_watts_gpu
print(f"\nPower Breakdown (TDP = {tdp_watts_gpu:.0f} W):")
print(f"  Pure Compute:   {compute_power_gpu:>6.1f} W  ({compute_fraction_gpu*100:>5.1f}%)")
print(f"  Infrastructure: {tdp_watts_gpu - compute_power_gpu:>6.1f} W  ({(1-compute_fraction_gpu)*100:>5.1f}%)")

# =============================================================================
# Comparison
# =============================================================================
print("\n" + "="*80)
print("EMPIRICALLY-DERIVED COMPUTE FRACTIONS")
print("="*80)
print(f"\n{'Hardware':<25} {'TDP':>8} {'Compute':>10} {'Infra':>10} {'Compute%':>10}")
print("-"*80)
print(f"{'Intel i7-12700K (CPU)':<25} {tdp_watts_cpu:>6.0f} W {compute_power_cpu:>8.1f} W {tdp_watts_cpu - compute_power_cpu:>8.1f} W{compute_fraction_cpu*100:>9.1f}%")
print(f"{'Jetson Orin Nano (GPU)':<25} {tdp_watts_gpu:>6.0f} W {compute_power_gpu:>8.1f} W {tdp_watts_gpu - compute_power_gpu:>8.1f} W{compute_fraction_gpu*100:>9.1f}%")

print("\n" + "="*80)
print("VALIDATION: Does GPU have higher compute fraction?")
print("="*80)
if compute_fraction_gpu > compute_fraction_cpu:
    print(f"\n  YES! GPU compute fraction ({compute_fraction_gpu*100:.1f}%) > CPU ({compute_fraction_cpu*100:.1f}%)")
    print(f"  Ratio: GPU is {compute_fraction_gpu/compute_fraction_cpu:.1f}x more compute-efficient per watt")
else:
    print(f"\n  NO! Something is wrong with the model.")
    print(f"  CPU: {compute_fraction_cpu*100:.1f}%, GPU: {compute_fraction_gpu*100:.1f}%")

# =============================================================================
# What this means for the soc_infrastructure model
# =============================================================================
print("\n" + "="*80)
print("IMPLICATIONS FOR SoC INFRASTRUCTURE MODEL")
print("="*80)
print(f"""
The current model shows:
    - x86_performance: ~21% compute fraction
    - tensor_core:     ~19% compute fraction

But empirical data shows:
    - CPU: {compute_fraction_cpu*100:.1f}% compute fraction (much lower!)
    - GPU: {compute_fraction_gpu*100:.1f}% compute fraction (higher than CPU!)

The model's compute power is OVER-ESTIMATED because:
    1. We use 100% ALU utilization assumption
    2. Real workloads have memory stalls, cache misses
    3. The measured throughput already accounts for overhead

To fix the model:
    - Use empirical efficiency factors from calibration data
    - Or scale compute power by measured efficiency vs peak
""")

# Show what the calibration says about efficiency
cpu_efficiency = measured_gflops_cpu / 1280.0  # vs JSON peak
gpu_efficiency = measured_gflops_gpu / 1280.0  # vs JSON peak

print(f"\nEmpirical Efficiency Factors:")
print(f"  CPU (vs 1280 GFLOPS spec): {cpu_efficiency*100:.1f}%")
print(f"  GPU (vs 1280 GFLOPS spec): {gpu_efficiency*100:.1f}%")
