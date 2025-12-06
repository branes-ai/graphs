"""
First Principles Analysis: Compute Power vs Infrastructure Overhead

Using empirical data from calibrated hardware profiles to back out
realistic compute fractions for CPU (i7-12700K) and GPU (Jetson Orin Nano).
"""

print("="*80)
print("FIRST PRINCIPLES: COMPUTE POWER FROM EMPIRICAL DATA")
print("="*80)

# =============================================================================
# Intel Core i7-12700K (SIMD CPU)
# =============================================================================
print("\n" + "="*80)
print("Intel Core i7-12700K (Alder Lake)")
print("="*80)

# Hardware specs from cpu.py
p_cores = 8
e_cores = 4
e_core_efficiency = 0.6
effective_cores = p_cores + int(e_cores * e_core_efficiency)  # 10
ops_per_core_per_cycle_fp32 = 16  # AVX2: 1 FMA unit = 8 lanes x 2 ops
sustained_clock_hz = 4.5e9  # 4.5 GHz all-core turbo

# Peak compute (theoretical)
peak_fp32_gflops = effective_cores * ops_per_core_per_cycle_fp32 * sustained_clock_hz / 1e9
print(f"\nTheoretical Peak FP32: {peak_fp32_gflops:.0f} GFLOPS")
print(f"  = {effective_cores} cores x {ops_per_core_per_cycle_fp32} ops/cycle x {sustained_clock_hz/1e9:.1f} GHz")

# Empirical data from calibration
measured_gflops = 772.4  # Best from 2048x2048 matmul (compute-bound)
efficiency = measured_gflops / 1280.0  # vs theoretical peak in JSON
print(f"\nEmpirical Best (matmul 2048): {measured_gflops:.1f} GFLOPS")
print(f"  Efficiency vs spec peak:    {efficiency*100:.1f}%")

# The 1280 GFLOPS in JSON is likely: 10 cores x 32 ops/cycle (both FMA units) x 4.0 GHz
# More realistic sustained: 720 GFLOPS = 10 x 16 x 4.5 GHz
theoretical_peak_720 = 720.0
compute_efficiency = measured_gflops / theoretical_peak_720
print(f"  Efficiency vs 720 GFLOPS:   {compute_efficiency*100:.1f}%")

# TDP and power
tdp_watts = 125.0  # PL1

# Energy per FP32 op (from first principles)
# At 14nm Intel: ~1.0 pJ per FP32 operation for the ALU only
# At 10nm/Intel 7: ~0.7 pJ per FP32 operation
# Let's use 0.8 pJ as a middle ground
energy_per_op_pj = 0.8  # pJ per FP32 op (ALU switching only)

# Compute power = ops/sec x energy/op
ops_per_sec = measured_gflops * 1e9
compute_power_w = ops_per_sec * energy_per_op_pj * 1e-12
print(f"\nCompute Power (from first principles):")
print(f"  Energy per FP32 op:  {energy_per_op_pj} pJ (ALU only, Intel 7 process)")
print(f"  Measured ops/sec:    {ops_per_sec/1e9:.1f} GFLOPS")
print(f"  Pure Compute Power:  {compute_power_w:.1f} W")

compute_fraction_cpu = compute_power_w / tdp_watts
overhead_fraction_cpu = 1 - compute_fraction_cpu
print(f"\nPower Breakdown (TDP = {tdp_watts:.0f} W):")
print(f"  Compute:        {compute_power_w:>6.1f} W  ({compute_fraction_cpu*100:>5.1f}%)")
print(f"  Infrastructure: {tdp_watts - compute_power_w:>6.1f} W  ({overhead_fraction_cpu*100:>5.1f}%)")

# =============================================================================
# NVIDIA Jetson Orin Nano 8GB (SIMT GPU)
# =============================================================================
print("\n" + "="*80)
print("NVIDIA Jetson Orin Nano 8GB (Ampere GPU)")
print("="*80)

# Hardware specs from jetson_orin_nano_8gb.py
num_sms = 8
cuda_cores_per_sm = 128
total_cuda_cores = num_sms * cuda_cores_per_sm
tensor_cores = 32  # 4 per SM
fp32_ops_per_sm_per_clock = 256  # 128 CUDA cores x 2 (FMA)
sustained_clock_hz_gpu = 650e6  # 650 MHz (15W mode)
boost_clock_hz_gpu = 918e6

# Peak compute (theoretical)
peak_fp32_cuda = num_sms * fp32_ops_per_sm_per_clock * sustained_clock_hz_gpu / 1e9
peak_fp32_boost = num_sms * fp32_ops_per_sm_per_clock * boost_clock_hz_gpu / 1e9
print(f"\nTheoretical Peak FP32:")
print(f"  At 650 MHz (sustained): {peak_fp32_cuda:.0f} GFLOPS")
print(f"  At 918 MHz (boost):     {peak_fp32_boost:.0f} GFLOPS")
print(f"  = {num_sms} SMs x {fp32_ops_per_sm_per_clock} ops/SM/clock x freq")

# Empirical data from calibration (15W mode)
measured_gflops_gpu = 800.7  # Best from 4096x4096 matmul
spec_peak = 1280.0  # From JSON
efficiency_gpu = measured_gflops_gpu / spec_peak
print(f"\nEmpirical Best (matmul 4096): {measured_gflops_gpu:.1f} GFLOPS")
print(f"  Efficiency vs 1280 spec:    {efficiency_gpu*100:.1f}%")

# The GPU achieved 800 GFLOPS, which is 4.0x the 650 MHz sustained peak!
# This means the GPU boosted to near 918 MHz during the benchmark
# 800 / (8 * 256 / 1e9) = 390 MHz? No wait, let me recalculate:
# 800e9 / (8 * 256) = 390 MHz effective? That can't be right.
# Let me check: 8 SMs x 256 ops/SM/clock = 2048 ops/clock
# 800e9 / 2048 = 390e6 Hz? That's way below 650 MHz...

# Actually, I think the ops count is different. Let me recalculate:
# 128 CUDA cores x 2 (FMA) = 256 FP32 ops/SM/clock
# But empirically, only 1 FMA per cycle is typical (not 2)
# So realistic: 128 CUDA cores x 1 = 128 ops/SM/clock for sustained
realistic_ops_per_sm = 128
peak_realistic = num_sms * realistic_ops_per_sm * boost_clock_hz_gpu / 1e9
print(f"\n  Realistic peak (128 ops/SM @ 918 MHz): {peak_realistic:.0f} GFLOPS")
actual_utilization = measured_gflops_gpu / peak_realistic
print(f"  Measured utilization: {actual_utilization*100:.1f}%")

# TDP for 15W mode
tdp_watts_gpu = 15.0

# Energy per FP32 op for Jetson (8nm Samsung)
# GPU transistors are smaller and more efficient than CPU
# At 8nm: ~0.6 pJ per FP32 op for CUDA core
energy_per_op_pj_gpu = 0.6  # pJ per FP32 op (CUDA core switching only)

# Compute power
ops_per_sec_gpu = measured_gflops_gpu * 1e9
compute_power_w_gpu = ops_per_sec_gpu * energy_per_op_pj_gpu * 1e-12
print(f"\nCompute Power (from first principles):")
print(f"  Energy per FP32 op:  {energy_per_op_pj_gpu} pJ (CUDA core, 8nm Samsung)")
print(f"  Measured ops/sec:    {ops_per_sec_gpu/1e9:.1f} GFLOPS")
print(f"  Pure Compute Power:  {compute_power_w_gpu:.2f} W")

compute_fraction_gpu = compute_power_w_gpu / tdp_watts_gpu
overhead_fraction_gpu = 1 - compute_fraction_gpu
print(f"\nPower Breakdown (TDP = {tdp_watts_gpu:.0f} W):")
print(f"  Compute:        {compute_power_w_gpu:>6.2f} W  ({compute_fraction_gpu*100:>5.1f}%)")
print(f"  Infrastructure: {tdp_watts_gpu - compute_power_w_gpu:>6.2f} W  ({overhead_fraction_gpu*100:>5.1f}%)")

# =============================================================================
# Comparison Summary
# =============================================================================
print("\n" + "="*80)
print("SUMMARY: EMPIRICALLY-DERIVED COMPUTE FRACTIONS")
print("="*80)
print(f"\n{'Hardware':<25} {'TDP':>8} {'Compute':>10} {'Infra':>10} {'Compute%':>10}")
print("-"*80)
print(f"{'Intel i7-12700K (CPU)':<25} {tdp_watts:>6.0f} W {compute_power_w:>8.1f} W {tdp_watts - compute_power_w:>8.1f} W{compute_fraction_cpu*100:>9.1f}%")
print(f"{'Jetson Orin Nano (GPU)':<25} {tdp_watts_gpu:>6.0f} W {compute_power_w_gpu:>8.2f} W {tdp_watts_gpu - compute_power_w_gpu:>8.2f} W{compute_fraction_gpu*100:>9.1f}%")

print("\n" + "="*80)
print("KEY INSIGHT: GPU has HIGHER compute fraction than CPU!")
print("="*80)
print(f"""
This matches the physical reality:
    - GPU: {compute_fraction_gpu*100:.0f}% compute, {overhead_fraction_gpu*100:.0f}% infrastructure
    - CPU: {compute_fraction_cpu*100:.0f}% compute, {overhead_fraction_cpu*100:.0f}% infrastructure

The GPU dedicates {compute_fraction_gpu/compute_fraction_cpu:.1f}x more of its power budget to
actual computation because:
    1. Simpler cores (no OoO, no branch prediction)
    2. Shared control logic (warp scheduler serves 32 threads)
    3. Higher ALU density per mm^2
    4. More efficient memory hierarchy for regular access patterns

This is WHY GPUs are used for AI/HPC - they convert more power into compute!
""")
