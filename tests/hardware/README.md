# Hardware Tests

Comprehensive test suite for validating power, performance, and energy metrics across all hardware mappers.

## Test Structure

```
tests/hardware/
├── README.md                    # This file
├── test_power_modeling.py       # Idle power modeling tests
├── test_thermal_profiles.py     # Thermal operating point validation
└── __init__.py                  # Package marker
```

## Test Coverage

### Power Modeling Tests (`test_power_modeling.py`)

Tests for idle power modeling (50% TDP idle consumption) across all 6 mapper types:

**Test Classes:**
- `TestIdlePowerConstant` - Validates IDLE_POWER_FRACTION = 0.5 for all mappers
- `TestIdlePowerMethod` - Validates compute_energy_with_idle_power() method exists
- `TestIdlePowerCalculation` - Tests idle power calculations are correct
- `TestThermalProfileIntegration` - Tests thermal profile usage

**Mappers Tested:**
- GPU (H100, Jetson Thor)
- TPU (TPU v4, Coral Edge TPU)
- CPU (Intel Xeon, AMD EPYC)
- DSP (QRB5165, TI TDA4VM)
- DPU (Xilinx Vitis AI)
- KPU (KPU-T64, T256, T768)

**Key Test Cases:**
1. Idle power constant exists and equals 0.5
2. Idle power method exists and is callable
3. At negligible dynamic power, total power ≈ 50% TDP
4. Idle power dominates for low-utilization workloads
5. Dynamic power dominates for high-utilization workloads
6. Energy scales linearly with latency
7. Thermal profiles are correctly used

### Thermal Profile Tests (`test_thermal_profiles.py`)

Tests for thermal operating points across all 32 hardware models:

**Test Classes:**
- `TestThermalOperatingPoints` - All models have thermal_operating_points
- `TestTDPValues` - TDP values are in reasonable ranges
- `TestMultiPowerProfiles` - Multi-profile models tested correctly
- `TestThermalProfileStructure` - Required fields present

**Validation Categories:**
- Datacenter GPUs: 300-700W
- Edge GPUs: 5-150W
- Datacenter CPUs: 200-600W
- DSPs: 3-30W
- DPUs: 15-50W (edge FPGA)
- KPUs: 3-100W (depending on model)

**Multi-Profile Models:**
- Jetson Orin AGX (15W, 30W, 60W)
- Jetson Thor (30W, 60W, 100W)
- KPU-T64 (3W, 6W, 10W)
- KPU-T256 (15W, 30W, 50W)
- KPU-T768 (30W, 60W, 100W)

## Running Tests

### Run All Hardware Tests
```bash
pytest tests/hardware/ -v
```

### Run Specific Test File
```bash
pytest tests/hardware/test_power_modeling.py -v
pytest tests/hardware/test_thermal_profiles.py -v
```

### Run Specific Test Class
```bash
pytest tests/hardware/test_power_modeling.py::TestIdlePowerCalculation -v
```

### Run Specific Test
```bash
pytest tests/hardware/test_power_modeling.py::TestIdlePowerCalculation::test_idle_power_equals_half_tdp_datacenter -v
```

### Run with Coverage
```bash
pytest tests/hardware/ --cov=src/graphs/hardware/mappers --cov-report=html
```

## Expected Results

All tests should pass with the following validations:

### Power Modeling
- ✅ All 6 mapper types have IDLE_POWER_FRACTION = 0.5
- ✅ All mappers have compute_energy_with_idle_power() method
- ✅ Idle power calculations accurate to within 1%
- ✅ Idle power dominates (>90%) for low-utilization workloads
- ✅ Dynamic power dominates (>50%) for high-utilization workloads

### Thermal Profiles
- ✅ All 32 hardware models have thermal_operating_points
- ✅ All TDP values in expected ranges for hardware category
- ✅ Multi-profile models have 2+ profiles
- ✅ All thermal profiles have required fields (name, tdp_watts, cooling_solution)
- ✅ All TDP values are positive

## Implementation Notes

### Idle Power Model

The idle power model is based on nanoscale transistor leakage:

```python
P_total = P_idle + P_dynamic
P_idle = TDP × 0.5  # Constant, independent of frequency
```

**Physical Basis:**
- Modern SoCs (7nm, 5nm, 3nm) have significant leakage current
- ~50% of TDP consumed at idle due to always-on circuitry and leakage
- DVFS reduces dynamic power but not leakage
- Validated against industry practice for nanoscale processes

### Test Philosophy

1. **Unit Tests**: Each test validates a single specific behavior
2. **Comprehensive Coverage**: Tests all 6 mapper types and 32 models
3. **Realistic Scenarios**: Tests both low and high utilization cases
4. **Regression Prevention**: Ensures idle power modeling doesn't break

### Adding New Tests

When adding new hardware models:

1. Add thermal_operating_points to the model
2. Update test_thermal_profiles.py with new model
3. Add model to appropriate mapper test in test_power_modeling.py
4. Verify TDP range is appropriate for hardware category

### Debugging Failed Tests

If tests fail:

1. **Check thermal_operating_points**: Ensure model has thermal profiles
2. **Check TDP value**: Verify TDP is reasonable for hardware type
3. **Check IDLE_POWER_FRACTION**: Should be 0.5 for all mappers
4. **Check method exists**: compute_energy_with_idle_power() must be present
5. **Run in verbose mode**: `pytest -v -s` shows detailed output

## References

- Idle power modeling: docs/sessions/2025-10-25_leakage_power_modeling.md
- CHANGELOG.md: Detailed implementation notes for Phase 1 and Phase 2
- Thermal operating points: src/graphs/hardware/resource_model.py

## Future Enhancements

Potential additions to this test suite:

1. End-to-end tests with actual model graphs
2. Performance regression tests (latency, throughput)
3. Energy efficiency comparisons across hardware
4. Batch size scaling tests
5. Precision-specific power tests (INT8 vs FP32)
6. Thermal throttling behavior tests
