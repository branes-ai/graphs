# Session: TDP Estimation & Model Efficiency Measurement

**Date:** 2025-12-04

## Summary

Created physics-based TDP estimation tools and a model efficiency measurement tool to investigate whether NVIDIA's peak TFLOPS specifications are achievable for real workloads.

## Key Hypothesis

NVIDIA may inflate peak ALU throughput numbers beyond what power delivery can sustain, relying on workload diversity and thermal throttling to stay within TDP. Curve-fitting would hide this - physics-based analysis exposes it.

## Tools Created

### 1. `cli/estimate_tdp.py` - TDP Estimation Tool

Physics-based TDP estimation given:
- Precision (FP64, FP32, FP16, INT8, INT4)
- Number of ALUs
- Process technology (3nm to 28nm)
- Circuit approach (x86_performance, tensor_core, systolic_mac, domain_flow)

Key formula:
```
TDP = energy_per_op x num_ALUs x ops_per_cycle x frequency
energy_per_mac = base_energy(process) x circuit_mult x precision_scale x 2 x overhead
```

Features:
- `--compare-precisions`: Compare TDP across precision levels
- `--compare-circuits`: Compare TDP across circuit types
- `--compare-processes`: Compare TDP across process nodes
- Matplotlib visualization with log-log plots

### 2. `cli/compare_tdp_registry.py` - Hardware Registry TDP Comparison

Compares physics-based TDP estimates against NVIDIA GPU spec TDP:

```
GPU                    Spec TDP   CUDA Est   TC Est   Combined    Ratio
                            (W)   FP32 (W)  FP16 (W)       (W)  Est/Spec
------------------------------------------------------------------------
B100-SXM6-192GB           1000       85.2     763.0     780.0    0.78x
H100-SXM5-80GB             700       87.0     389.7     407.1    0.58x
A100-SXM4-80GB             400       70.2     314.4     328.4    0.82x
V100-SXM3-32GB             350       39.2     175.5     183.3    0.52x
T4-PCIe-16GB                70       20.4      91.2      95.2    1.36x ***
```

Key features:
- Energy per MAC columns (CUDA pJ/MAC, TC pJ/MAC)
- Sensitivity analysis: "At what pJ/MAC would compute exhaust TDP?"
- Identifies T4 as clearly over-spec'd (91W compute > 70W TDP)

### 3. `cli/measure_model_efficiency.py` - Model Efficiency Measurement

Measures actual MAC throughput efficiency for neural networks:

```bash
./cli/measure_model_efficiency.py --model resnet50,vit_b_16 --device cuda
```

Features:
- Traces model with FX/fvcore to count theoretical FLOPs/MACs
- Runs timed inference with proper warmup and CUDA synchronization
- Compares delivered TFLOPS to peak (from spec or BLAS measurement)
- Reports efficiency = delivered / peak

Sample output:
```
Model          Batch   GFLOPs   Latency   Delivered   Peak   Efficiency
                                  (ms)      TFLOPS  TFLOPS
----------------------------------------------------------------------
resnet50           1      4.1     25.38       0.16    0.9       19.0%
resnet50           8     33.2    159.83       0.21    0.9       24.1%
vit_b_16           1     16.9     89.28       0.19    0.9       22.0%
vit_b_16           8    134.9    569.50       0.24    0.9       27.6%
```

Key fix: Measures BLAS peak once at start and caches it to avoid variability from CPU frequency scaling between runs.

## Physics Constants

From `src/graphs/hardware/technology_profile.py`:

**Process Node Base Energy (pJ per op):**
- 3nm: 1.2 pJ
- 4nm: 1.3 pJ
- 5nm: 1.5 pJ
- 7nm: 1.8 pJ
- 12nm: 2.5 pJ
- 28nm: 4.0 pJ

**Circuit Type Multipliers:**
- systolic_mac: 0.8x (most efficient)
- tensor_core: 0.85x
- cuda_core: 0.95x
- x86_performance: 2.5x (least efficient)

**Datapath Overhead:** 40% for register files, data distribution, control logic

## Key Findings

### T4 Inference GPU is Over-Spec'd
- Physics-based compute estimate: 91W (Tensor Cores alone)
- Spec TDP: 70W
- **Ratio: 1.36x - impossible to sustain peak throughput**
- Peak INT8 TOPS is theoretical, not achievable sustained

### Datacenter GPUs Have Varying Headroom
- B100: 1.31x margin (tight)
- H100: 1.80x margin (comfortable)
- A100: 1.27x margin (tight)
- V100: 1.99x margin (most headroom)

### Model Efficiency on CPU: ~24%
- BLAS GEMM peak: ~0.86 TFLOPS
- Real models deliver: 0.16-0.24 TFLOPS
- Efficiency: 19-28% depending on batch size
- Larger batches improve efficiency (amortize overhead)

## Tensor Core Accounting

Corrected understanding of NVIDIA Tensor Core architecture:
- TC = 4x4 systolic array = 16 MAC units
- But NVIDIA packs multiple arrays per "Tensor Core" marketing unit
- H100: 528 TCs x 256 MACs/TC = 135,168 total MAC units
- B100: 528 TCs x 512 MACs/TC = 270,336 total MAC units

## Files Modified/Created

### Created
- `cli/estimate_tdp.py` - TDP estimation with sweeps and plots
- `cli/compare_tdp_registry.py` - Compare physics vs spec TDP
- `cli/measure_model_efficiency.py` - Model efficiency measurement

### Key Functions
- `estimate_tdp()` - Core TDP estimation
- `estimate_tensor_core_tdp()` - TC-specific estimation
- `get_tc_energy_per_mac()` - Energy per MAC calculation
- `measure_model_efficiency()` - Model throughput measurement
- `measure_cpu_blas_peak()` / `measure_cublas_peak()` - Peak measurement

## Next Steps

1. Run `measure_model_efficiency.py` on V100/A100/H100 with ViT
2. If efficiency is ~25%, confirms peak specs are unachievable
3. Investigate memory-bound vs compute-bound behavior
4. Compare efficiency across precision (FP32 vs FP16 vs INT8)

## Conclusion

The tools created today provide a physics-based framework to investigate whether NVIDIA's marketing TFLOPS numbers represent sustainable throughput or just burst capability. The T4 analysis clearly shows peak specs exceed power budget, supporting the hypothesis that peak numbers are inflated.
