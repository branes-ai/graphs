# Validation

This directory contains functional validation tests that verify the **accuracy** of estimators and mappers against known benchmarks and published results.

## Directory Structure

```
validation/
├── hardware/          # Hardware mapper validation
│   └── test_*.py     # Validate hardware performance estimates
├── estimators/        # Estimator accuracy tests
│   └── test_*.py     # Validate FLOP/memory calculations
└── README.md          # This file
```

## Purpose

**Validation tests answer:** "Are our estimates accurate?"

These tests compare our characterization results against:
- Published benchmarks (MLPerf, vendor specs)
- Theoretical calculations (FLOP counts)
- Real hardware measurements

## Validation vs Testing

| Aspect | Validation (`./validation/`) | Testing (`./tests/`) |
|--------|------------------------------|----------------------|
| **Purpose** | Accuracy of estimates | Correctness of code |
| **Question** | "Are results accurate?" | "Does code work?" |
| **Compares** | Against benchmarks | Against expected behavior |
| **Scope** | End-to-end estimation | Unit-level functionality |

## Running Validation

### Hardware Validation
```bash
# Complete 10-way hardware comparison
python validation/hardware/test_all_hardware.py

# Individual mapper tests
python validation/hardware/test_cgra_mapper.py
python validation/hardware/test_dpu_mapper.py
python validation/hardware/test_hardware_mapping.py
```

### Estimator Validation
```bash
# ResNet family
python validation/estimators/test_resnet18.py
python validation/estimators/test_resnet_family.py

# MobileNet and EfficientNet
python validation/estimators/test_mobilenet.py
python validation/estimators/test_efficientnet.py

# Basic operations
python validation/estimators/test_conv2d.py
```

## Expected Results

**Estimator Accuracy:**
- FLOP counts: Within ±6% of theoretical values
- Memory estimates: Within ±10% of actual usage
- Latency (relative): Within 2× of real hardware

**Hardware Mapper Validation:**
- Utilization: Realistic (20-100%, not always 100%)
- Speedups: Consistent with published benchmarks
- Energy: Within 2× of measured values

## Adding New Validations

### For Hardware Mappers
1. Create `test_<hardware>_mapper.py` in `validation/hardware/`
2. Test on standard model (ResNet-18)
3. Compare against published benchmarks
4. Add to `test_all_hardware.py` comparison

### For Estimators
1. Create `test_<model>.py` in `validation/estimators/`
2. Calculate theoretical FLOPs manually
3. Compare against our estimates
4. Ensure ±6% accuracy

## Common Issues

**Import errors:**
- All validation scripts include: `sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))`
- This adds repo root to path for imports

**PyTorch not installed:**
- Validation requires PyTorch: `pip install torch torchvision`

**Large variance in results:**
- Check if comparing same precision (FP32 vs INT8)
- Verify batch size matches
- Confirm input shapes are identical

## Success Criteria

Validation passes if:
- ✅ All scripts run without errors
- ✅ FLOP estimates within ±6% of theory
- ✅ Hardware speedups consistent with published data
- ✅ Energy estimates within 2× of measurements
- ✅ Utilization realistic (not always 100%)

## Documentation

See also:
- `../tests/README.md` - Unit tests
- `../examples/README.md` - Usage demonstrations
- `../docs/realistic_performance_modeling_plan.md` - Architecture
