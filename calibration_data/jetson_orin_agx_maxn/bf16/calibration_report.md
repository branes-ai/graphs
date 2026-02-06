# Calibration Report: NVIDIA Jetson AGX Orin 64GB (MAXN)

## Hardware Configuration

- **Hardware ID**: jetson_orin_agx_maxn
- **Device**: cuda
- **Precision**: BF16
- **Thermal Profile**: MAXN
- **Calibration Date**: 2026-02-06T00:51:27.393756

## Models Calibrated

| Model | Status |
|-------|--------|
| resnet18 | OK |
| resnet34 | OK |
| resnet50 | OK |
| resnet101 | OK |
| mobilenet_v2 | OK |
| mobilenet_v3_small | OK |
| mobilenet_v3_large | OK |
| efficientnet_b0 | OK |
| efficientnet_b1 | OK |
| vgg11 | OK |
| vgg16 | OK |
| vit_b_16 | OK |
| vit_b_32 | OK |
| vit_l_16 | OK |
| maxvit_t | OK |

## Summary

- **Total Models**: 15
- **Successful**: 15
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
with open('calibration_data/jetson_orin_agx_maxn/bf16/efficiency_curves.json') as f:
    curves = json.load(f)

# Or copy to hardware_registry
cp calibration_data/jetson_orin_agx_maxn/bf16/efficiency_curves.json \
   hardware_registry/cuda/jetson_orin_agx_maxn/bf16/efficiency_curves.json
```
