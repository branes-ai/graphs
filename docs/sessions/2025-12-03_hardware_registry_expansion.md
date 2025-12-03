# Session: Hardware Registry Expansion & Energy Efficiency Analysis

**Date**: 2025-12-03

## Summary

Expanded the hardware registry with new accelerators and added energy efficiency analysis capabilities. Introduced `product_category` field for proper market segment classification, and added a new energy efficiency (TOPS/W) comparison table to the hardware summary tool.

## Key Accomplishments

### 1. Product Category Classification

Added `product_category` field to all 40+ spec.json files to properly classify hardware by market segment instead of relying on TDP-based heuristics.

**Categories defined:**
- `datacenter` - Server/cloud accelerators (TPUs, H100, A100, Cloud AI 100)
- `desktop` - Consumer desktop hardware (GTX 1070, i7-12700K)
- `embodied` - Autonomous vehicles, robots (Jetson AGX, KPU-T256, Snapdragon Ride)
- `edge` - Edge AI devices (Jetson Nano, Hailo, Coral, KPU-T64)
- `mobile` - Laptops, phones (Ryzen 8845HS, Mali GPU)

**Why this was needed:**
TDP-based classification was putting datacenter products in wrong categories. For example, TPU v1 at 40W was being classified as "Edge AI" when it's clearly a datacenter product.

### 2. New Hardware Added

**Qualcomm Accelerators:**
- `qualcomm_cloud_ai_100` - Datacenter AI inference (400 TOPS INT8 @ 75W, 5.3 TOPS/W)
- `qualcomm_qrb5165` - Robotics platform with Hexagon 698 DSP (15 TOPS @ 7W)
- `qualcomm_snapdragon_ride` - Automotive L4/L5 platform (700 TOPS @ 130W)

**Stillwater KPU Family:**
- `stillwater_kpu_t64` - Edge AI, 64 tiles (72 TOPS INT8 @ 10W, 7.2 TOPS/W)
- `stillwater_kpu_t256` - Embodied AI, 256 tiles (426 TOPS INT8 @ 50W, 8.5 TOPS/W)
- `stillwater_kpu_t768` - Datacenter AI, 768 tiles (1087 TOPS INT8 @ 100W, 10.9 TOPS/W)

### 3. Energy Efficiency Analysis Table

Added third summary table to `cli/show_tops_per_watt.py` showing:
- Hardware name and product category
- TDP (watts)
- INT8 TOPS (absolute performance)
- INT8 TOPS/W (energy efficiency)
- BF16 TOPS/W (important for AI workloads)

Sorted by INT8 TOPS/W descending to highlight most efficient accelerators.

## Files Modified

### New Files
```
hardware_registry/accelerator/qualcomm_cloud_ai_100/spec.json
hardware_registry/accelerator/stillwater_kpu_t64/spec.json
hardware_registry/accelerator/stillwater_kpu_t256/spec.json
hardware_registry/accelerator/stillwater_kpu_t768/spec.json
hardware_registry/dsp/qualcomm_qrb5165/spec.json
hardware_registry/dsp/qualcomm_snapdragon_ride/spec.json
```

### Modified Files
```
src/graphs/hardware/registry/profile.py
  - Added product_category field to HardwareProfile dataclass
  - Updated to_dict() and from_dict() methods

cli/show_tops_per_watt.py
  - Replaced get_power_category() with get_product_category()
  - Added PRODUCT_CATEGORY_DISPLAY mapping
  - Added energy efficiency table (TOPS/W comparison)
  - Added TDP entries for new hardware
  - Added power profiles for KPUs and Snapdragon Ride

hardware_registry/**/spec.json (40+ files)
  - Added product_category field to all existing specs
```

## Energy Efficiency Results

### INT8 TOPS/W Rankings

| Rank | Hardware | Category | TDP | INT8 TOPS/W |
|------|----------|----------|-----|-------------|
| 1 | Stillwater KPU-T768 | Datacenter | 100W | 10.9 |
| 2 | Hailo-8 | Edge AI | 2W | 10.4 |
| 3 | Stillwater KPU-T256 | Embodied AI | 50W | 8.5 |
| 4 | Hailo-10H | Edge AI | 2W | 8.0 |
| 5 | Stillwater KPU-T64 | Edge AI | 10W | 7.2 |
| 6 | NVIDIA H100 | Datacenter | 700W | 5.7 |
| 7 | Qualcomm Snapdragon Ride | Embodied AI | 130W | 5.4 |
| 8 | Qualcomm Cloud AI 100 | Datacenter | 75W | 5.3 |

### BF16 TOPS/W Rankings

| Rank | Hardware | BF16 TOPS/W |
|------|----------|-------------|
| 1 | Stillwater KPU-T768 | 2.13 |
| 2 | Stillwater KPU-T256 | 1.67 |
| 3 | Stillwater KPU-T64 | 1.58 |
| 4 | NVIDIA H100 | 1.41 |
| 5 | NVIDIA B100 | 0.90 |

## Key Insights

1. **KPU architecture leads in energy efficiency** - The tile-based spatial dataflow architecture achieves 10.9 INT8 TOPS/W, nearly 2x the H100's efficiency.

2. **Edge accelerators are extremely efficient** - Hailo-8 at 10.4 TOPS/W and Coral at 2.0 TOPS/W show dedicated inference accelerators excel at efficiency.

3. **BF16 efficiency matters for AI** - KPUs also lead in BF16 efficiency (2.13 TOPS/W), important for training and large model inference.

4. **CPUs are orders of magnitude less efficient** - General-purpose CPUs achieve 0.02-0.12 TOPS/W, 50-500x worse than dedicated accelerators.

5. **Product category != power class** - TPU v1 at 40W is a datacenter product, not edge. Qualcomm Cloud AI 100 at 75W competes with 400W GPUs.

## Usage

```bash
# View all three summary tables
PYTHONPATH=src:$PYTHONPATH python cli/show_tops_per_watt.py --summary

# Tables shown:
# 1. HARDWARE REGISTRY SUMMARY - by category and power
# 2. OPS PER CLOCK - by INT8 throughput (micro-architectural)
# 3. ENERGY EFFICIENCY - by INT8 TOPS/W (efficiency)
```

## Next Steps

- Add more edge accelerators (Rockchip NPU, MediaTek APU)
- Add Apple Neural Engine specs
- Consider adding cost efficiency metrics (TOPS/$)
- Add memory bandwidth efficiency metrics (TOPS per GB/s)
