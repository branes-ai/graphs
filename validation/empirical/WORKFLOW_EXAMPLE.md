# Empirical Benchmark Workflow: Complete Example

This guide shows the complete end-to-end workflow for running empirical benchmarks, analyzing results, and calibrating your performance models.

## Quick Start (5 minutes)

```bash
# 1. Run quick smoke test (5 configs, 1 minute)
python validation/empirical/sweep_mlp.py --quick --device cpu

# 2. Analyze results and get recommendations
python validation/empirical/calibration_analysis.py \
    --input validation/empirical/results/mlp_sweep_quick_cpu.csv

# 3. Read the calibration report
cat validation/empirical/results/mlp_sweep_quick_cpu_calibration.md
```

---

## Complete Workflow

### Step 1: Run Empirical Sweep

Choose your sweep scope:

```bash
# Option A: Quick smoke test (5 configs, ~1 minute)
python validation/empirical/sweep_mlp.py --quick --device cpu

# Option B: Memory scenario sweep (test cache spilling)
python validation/empirical/sweep_mlp.py --scenario L3_fit --device cpu

# Option C: Full parameter sweep (150+ configs, hours!)
python validation/empirical/sweep_mlp.py --full --device cpu
```

**What happens**:
1. Builds MLPs with varying architectures (1-4 layers, different sizes)
2. Runs empirical benchmarks:
   - Measures actual execution time (with warmup, 50 runs)
   - Tracks peak memory usage
   - Computes throughput (samples/sec)
3. Runs analytical estimates:
   - FX tracing + shape propagation
   - FXGraphWalker characterization
   - Hardware mapper (CPU/GPU/KPU)
4. Compares empirical vs analytical
5. Saves results to CSV

**Output**: `validation/empirical/results/mlp_sweep_quick_cpu.csv`

---

### Step 2: Analyze Results and Generate Calibration Report

```bash
python validation/empirical/calibration_analysis.py \
    --input validation/empirical/results/mlp_sweep_quick_cpu.csv \
    --output validation/empirical/results/calibration_report.md
```

**What happens**:
1. Loads sweep results from CSV
2. Computes error metrics:
   - MAPE (Mean Absolute Percentage Error)
   - R² (coefficient of determination)
   - Min/max/std errors
3. Separates by bottleneck type:
   - Compute-bound (AI > 10)
   - Memory-bound (AI ≤ 10)
4. Recommends `efficiency_factor` coefficient updates
5. Generates comprehensive Markdown report

**Output**: `validation/empirical/results/calibration_report.md`

**Example Report Excerpt**:
```markdown
## Calibration Recommendations

### Empirical Derate Coefficient

**Current (assumed)**: `efficiency_factor = 0.70`
**Recommended**: `efficiency_factor = 0.634`

**Change**: -9.4%

#### How to Apply

Update the hardware mapper for **cpu** at precision **fp32**:

```python
# File: src/graphs/characterize/cpu_mapper.py

Precision.FP32: PerformanceCharacteristics(
    precision=Precision.FP32,
    compute_resource=compute_resource,
    efficiency_factor=0.634,  # ← UPDATE THIS
    tile_utilization=0.95,
    native_acceleration=True,
),
```

**Expected improvement**: MAPE should decrease from 18.3% to ~12.8% (±30% reduction)
```

---

### Step 3: Apply Calibration Updates

Based on the calibration report, update the hardware mapper:

```python
# File: src/graphs/characterize/cpu_mapper.py

def create_intel_cpu_mapper():
    # ... (existing code)

    thermal_profile = ThermalOperatingPoint(
        name="default",
        tdp_watts=125.0,
        cooling_solution="tower-cooler",
        performance_specs={
            Precision.FP32: PerformanceCharacteristics(
                precision=Precision.FP32,
                compute_resource=avx512_compute,
                efficiency_factor=0.634,  # ← UPDATED from 0.70
                memory_bottleneck_factor=0.55,  # ← UPDATED from 0.60
                tile_utilization=0.95,
                native_acceleration=True,
            ),
            # ... other precisions
        }
    )
```

**Commit the changes**:
```bash
git add src/graphs/characterize/cpu_mapper.py
git commit -m "Calibrate CPU FP32 efficiency_factor to 0.634 based on MLP sweep"
```

---

### Step 4: Verify Calibration

Re-run validation tests to verify improved accuracy:

```bash
# Run estimator validation
python validation/estimators/test_resnet18.py
python validation/estimators/test_mobilenet.py

# Run hardware comparison
python validation/hardware/test_all_hardware.py

# Check FLOP accuracy (should be within ±6%)
python validation/estimators/test_resnet_family.py
```

**Expected result**: MAPE should improve by ~30% after calibration.

---

### Step 5: Iterate If Needed

If errors are still too high (MAPE > 15%), iterate:

1. **Run more comprehensive sweeps**:
   ```bash
   python validation/empirical/sweep_mlp.py --full --device cpu
   ```

2. **Test different model families**:
   ```bash
   python validation/empirical/sweep_conv.py --quick  # TODO: Create this
   python validation/empirical/sweep_resnet.py --quick  # TODO: Create this
   ```

3. **Analyze bottleneck-specific errors**:
   - If compute-bound errors are high: Adjust `efficiency_factor`
   - If memory-bound errors are high: Adjust `memory_bottleneck_factor`
   - If both are high: Check tiling strategy

---

## Understanding the Results

### CSV Output Format

```csv
model,input_dim,hidden_dims,output_dim,batch_size,device,precision,
  empirical_time_ms,empirical_time_std_ms,empirical_memory_mb,empirical_throughput,
  analytical_time_ms,analytical_memory_mb,analytical_flops,analytical_ai,analytical_bottleneck,
  time_error_pct,memory_error_pct

mlp_2layer,512,"[512, 512]",128,32,cpu,fp32,
  12.345,0.234,44.2,2593.4,
  10.567,44.6,2.15e9,12.3,compute,
  16.8,0.9
```

### Key Columns Explained

| Column | Description | Interpretation |
|--------|-------------|----------------|
| `empirical_time_ms` | Actual measured time | Ground truth |
| `analytical_time_ms` | Estimated time | What we're calibrating |
| `time_error_pct` | % difference | Lower is better |
| `analytical_ai` | Arithmetic Intensity (FLOPs/Byte) | > 10 = compute-bound |
| `analytical_bottleneck` | Predicted bottleneck | 'compute' or 'memory' |

### Error Thresholds

| MAPE Range | Assessment | Action |
|------------|------------|--------|
| < 10% | ✓ Excellent | Production ready |
| 10-15% | ⚠ Good | Acceptable for research |
| 15-25% | ⚠ Needs tuning | Calibrate coefficients |
| > 25% | ✗ Poor | Investigate systematic errors |

---

## Memory Scenario Sweeps

Test specific memory hierarchy levels:

### L3 Cache Fit (1-32 MB working set)

```bash
python validation/empirical/sweep_mlp.py --scenario L3_fit --device cpu
```

**Expected observations**:
- Slowdown vs L1_fit: ~3-4×
- Arithmetic Intensity: 5-20
- Bottleneck: Memory (but still reasonable performance)

### DRAM Spill (> 32 MB working set)

```bash
python validation/empirical/sweep_mlp.py --scenario DRAM_spill --device cpu
```

**Expected observations**:
- Slowdown vs L3_fit: ~3-5× additional
- Arithmetic Intensity: < 5
- Bottleneck: Severe memory bandwidth limitation

**Use case**: Validates that your tiling strategy correctly predicts when to partition the graph for memory constraints.

---

## GPU Benchmarking

If you have CUDA available:

```bash
# Quick GPU sweep
python validation/empirical/sweep_mlp.py --quick --device cuda

# Analyze GPU-specific calibration
python validation/empirical/calibration_analysis.py \
    --input validation/empirical/results/mlp_sweep_quick_cuda.csv
```

**Key differences GPU vs CPU**:
- Higher `efficiency_factor` (0.8-0.9 for FP32)
- Batch size matters more (launch overhead)
- FP16 Tensor Cores: 2-3× speedup
- INT8 Tensor Cores: 3-5× speedup

---

## Advanced: Custom Sweeps

### Create Your Own Sweep Parameters

Edit `validation/empirical/sweep_mlp.py`:

```python
CUSTOM_SWEEP = {
    'input_dim': [1024],
    'hidden_configs': [
        # Test specific model size
        ([2048, 2048, 2048], 1024),
    ],
    'batch_size': [1, 2, 4, 8, 16, 32, 64, 128],  # Focus on batch scaling
    'precision': ['fp32', 'fp16', 'int8'],  # All precisions
}
```

Then run:
```bash
python validation/empirical/sweep_mlp.py  # Will use CUSTOM_SWEEP if --quick not specified
```

### Adding Conv2D Sweeps

Create `validation/empirical/sweep_conv.py` following the same pattern:

```python
SWEEP_PARAMS = {
    'in_channels': [3, 16, 32, 64],
    'out_channels': [16, 32, 64, 128],
    'kernel_size': [3, 5, 7],
    'num_layers': [1, 2, 3, 5],
    'image_size': [64, 128, 224],
    'batch_size': [1, 16, 64],
}
```

---

## Troubleshooting

### Error: "CUDA out of memory"

**Solution**: Reduce batch size or model size in GPU sweeps

```bash
# Use smaller configs for GPU
python validation/empirical/sweep_mlp.py --quick --device cuda
```

### Error: "psutil not installed"

**Solution**: Install psutil for CPU memory tracking

```bash
pip install psutil
```

### High variance in empirical times

**Solution**: Increase warmup and measurement runs

Edit `sweep_mlp.py`:
```python
empirical = run_empirical_benchmark(
    model, input_tensor, device, precision,
    num_warmup=20,   # Increase from 10
    num_runs=100     # Increase from 50
)
```

### Analytical estimates way off (> 50% error)

**Possible causes**:
1. **FX tracing failed**: Check if model is FX-traceable
2. **Shape propagation failed**: Verify input tensor shape
3. **Hardware mapper misconfigured**: Check device name matches
4. **Precision mismatch**: Ensure precision string matches enum

**Debug**:
```python
# Add debug prints to sweep_mlp.py
traced = symbolic_trace(model)
print(f"FX graph nodes: {len(list(traced.graph.nodes))}")

ShapeProp(traced).propagate(input_tensor)
for node in traced.graph.nodes:
    print(f"{node.name}: {node.meta.get('tensor_meta', 'no meta')}")
```

---

## Integration with Existing Workloads

Your existing MLP workloads in `workloads/pytorch/mlp/` are already integrated!

The sweep uses:
- `multi_layer_perceptrons.py`: `OneLayerMLP`, `TwoLayerMLP`, `ThreeLayerMLP`, `FourLayerMLP`
- `MLP_CONFIGS`: Standard size configurations

**To add your custom models**:

```python
# In sweep_mlp.py, add to imports:
from oneLayerMLP_large import OneLayerMLPLarge

# Add to build_mlp():
def build_mlp(hidden_dims, input_dim, output_dim):
    if hidden_dims == [8192]:  # Your large config
        return OneLayerMLPLarge(input_dim, output_dim)
    # ... existing code
```

---

## Next Steps

1. ✓ Run quick smoke test
2. ✓ Generate calibration report
3. ✓ Update hardware mapper coefficients
4. ✓ Verify with validation tests
5. → Run full sweep on your target hardware
6. → Create Conv2D and ResNet sweeps
7. → Integrate with CI/CD for continuous calibration

**Goal**: Achieve < 10% MAPE across all model families for production deployment!
