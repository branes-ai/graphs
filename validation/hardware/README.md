# Hardware Mapper Validation

This directory validates hardware mapper implementations by comparing estimated performance against published benchmarks and known hardware characteristics.

## Test Files

### Comprehensive Comparisons
- **`test_all_hardware.py`** - Complete 10-way comparison (GPU/TPU/KPU/DPU/CGRA/Jetson/CPU)
  - Primary validation for Phase 2
  - Tests across FP32, BF16, INT8 precisions
  - Includes energy, latency, utilization analysis
  - Model: DeepLabV3-ResNet101

### Individual Mapper Tests
- **`test_hardware_mapping.py`** - ResNet-18 on H100 (3 precisions)
- **`test_cgra_mapper.py`** - Stanford Plasticine validation
- **`test_dpu_mapper.py`** - Xilinx Vitis AI DPU validation
- **`test_cpu_vs_gpu_mapping.py`** - CPU vs GPU comparison (AVX-512, AVX-2)
- **`test_gpu_cpu_kpu_comparison.py`** - 3-way comparison validation
- **`test_kpu_simple.py`** - Quick KPU sanity check

## Running Tests

### All Hardware (Recommended)
```bash
python validation/hardware/test_all_hardware.py

# Expected output:
# - Performance rankings (latency, throughput)
# - Quantization speedup analysis
# - Energy efficiency comparison
# - Utilization analysis
# - Head-to-head vs CPU baseline
# - Cost-benefit analysis
```

### Individual Mappers
```bash
# Test specific hardware
python validation/hardware/test_cgra_mapper.py
python validation/hardware/test_dpu_mapper.py
python validation/hardware/test_kpu_simple.py
```

## Validation Criteria

### Utilization
- **GPU (H100):** 20-40% at batch=1 (not 100% - realistic!)
- **TPU/KPU/CGRA:** 60-100% (efficient at batch=1)
- **CPU:** 90-100% (all cores used)

### Quantization Speedup
- **GPU:** 8-10× (FP32 → INT8)
- **KPU:** 4-5× (FP32 → INT8)
- **TPU:** 1-2× (BF16 already native)
- **CPU:** 1× (bandwidth-bound, no speedup)

### Energy Efficiency
- **KPU:** Best (0.001 J/inference at INT8)
- **TPU:** Good (0.001 J/inference)
- **GPU:** Moderate (0.001 J/inference, but 700W TDP)
- **CPU:** Poor (0.002 J/inference)

### Latency Rankings (INT8, Batch=1)
1. H100 GPU: ~0.02 ms (fastest)
2. TPU v4: ~0.04 ms
3. KPU-T100: ~0.05 ms
4. CGRA: ~5-10 ms
5. DPU: ~3-5 ms
6. CPU: ~600 ms (baseline)

## Common Issues

**All tests fail with import errors:**
- Check sys.path setup (should go up 2 levels: `'../..'`)
- Verify repo root has `src/` directory

**Performance doesn't match published benchmarks:**
- Check batch size (we test batch=1, benchmarks often use batch=64)
- Verify precision (FP32 vs BF16 vs INT8)
- Confirm model architecture matches

**Unrealistic utilization (>100% or <0%):**
- Bug in mapper - check allocation logic
- Verify thread count calculations

## Hardware Requirements

**Minimum:**
- PyTorch (CPU-only mode works)
- 8 GB RAM
- Python 3.8+

**For realistic validation:**
- Access to target hardware (H100, TPU, etc.)
- Ability to run actual benchmarks
- Measurement tools (nvidia-smi, perf, etc.)

## Key Insights from Validation

### Phase 2 Key Findings
1. **Utilization matters:** Assuming 100% utilization causes 1000× errors
2. **Quantization is hardware-specific:** GPU benefits massively, CPU doesn't
3. **Bandwidth dominates at batch=1:** All hardware >60% bandwidth-bound
4. **KPU wins for embodied AI:** Best perf/watt at 6-24W power envelope
5. **Spatial dataflow has overhead:** CGRA reconfiguration costs ~0.3% latency

### Embodied AI Rankings (100 Wh battery, INT8)
1. **Coral Edge TPU:** 50 hours runtime (ultra-low-power champion)
2. **KPU-T100 @ 6W:** 16.7 hours (best balance)
3. **Jetson-Orin @ 15W:** 6.7 hours (performance-focused)
4. **Jetson-Thor @ 30W:** 3.3 hours (automotive)
5. **KPU-T300 @ 50W:** 2.0 hours (automotive)

## Documentation

See also:
- `../estimators/README.md` - Estimator accuracy tests
- `../../docs/realistic_performance_modeling_plan.md` - Architecture
- `../../CHANGELOG.md` - Phase 2 implementation history
