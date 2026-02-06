# Calibration Report: Auto-detected: 12th Gen Intel(R) Core(TM) i7-12700K

## Hardware Configuration

- **Hardware ID**: 12th_gen_intel_r_core_tm_i7_12700k
- **Device**: cpu
- **Precision**: FP16
- **Thermal Profile**: None
- **Calibration Date**: 2026-02-05T18:25:00.458515

## Models Calibrated

| Model | Status |
|-------|--------|
| resnet18 | OK |
| resnet50 | OK |
| mobilenet_v2 | OK |
| efficientnet_b0 | OK |
| vgg16 | OK |
| vit_b_16 | OK |

## Summary

- **Total Models**: 6
- **Successful**: 6
- **Failed**: 0

## Files Generated

- `measurements/` - Per-model measurement JSON files
- `efficiency_curves.json` - Aggregated efficiency curves
- `calibration_report.md` - This report

## Usage

To use these calibration curves:

```python
# Load calibration data
import json
with open('calibration_data/12th_gen_intel_r_core_tm_i7_12700k/fp16/efficiency_curves.json') as f:
    curves = json.load(f)

# Or copy to hardware_registry
cp calibration_data/12th_gen_intel_r_core_tm_i7_12700k/fp16/efficiency_curves.json \
   hardware_registry/cpu/12th_gen_intel_r_core_tm_i7_12700k/fp16/efficiency_curves.json
```
