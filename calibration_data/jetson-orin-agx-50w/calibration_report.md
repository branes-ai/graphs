# Calibration Report: NVIDIA Jetson AGX Orin 64GB (50W)

## Hardware Configuration

- **Hardware ID**: jetson-orin-agx-50w
- **Device**: cuda
- **Thermal Profile**: 50W
- **Calibration Date**: 2026-02-05T00:01:44.594991

## Models Calibrated

| Model | Status |
|-------|--------|

## Summary

- **Total Models**: 0
- **Successful**: 0
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
with open('calibration_data/jetson-orin-agx-50w/efficiency_curves.json') as f:
    curves = json.load(f)

# Or copy to hardware_registry
cp calibration_data/jetson-orin-agx-50w/efficiency_curves.json \
   hardware_registry/cuda/jetson-orin-agx-50w/efficiency_curves.json
```
