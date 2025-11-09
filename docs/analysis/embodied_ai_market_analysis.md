# Embodied AI Hardware Market Analysis and Comparison Plan

**Date:** 2025-11-09
**Objective:** Deliver accurate latency, energy consumption, and BOM cost comparison between Stillwater KPU family and competitor Embodied AI accelerators

## Executive Summary

This document analyzes the Embodied AI hardware landscape and provides a comprehensive plan to position Stillwater KPU products (T64, T256, T768) against market competitors including NVIDIA Jetson family, Qualcomm products, Google Coral TPU, and Hailo AI accelerators.

**Market Reality**: Most marketed AI accelerator specs (TOPS) are based on sparse workloads, peak boost clocks, and ideal conditions. Real-world Embodied AI deployments see **2-10% of advertised peak performance** due to:
- Severe DVFS thermal throttling at deployable power budgets
- Memory bandwidth bottlenecks
- Dense (non-sparse) network requirements
- Sequential execution of small DNNs

---

## Market Segmentation Matrix

### Entry-Level Edge AI (1-10W, $50-$300)
**Target Applications**: IoT cameras, drones, small robots, battery-powered devices

| Product | TOPS (Advertised) | TOPS (Realistic) | Power | BOM Cost | Status |
|---------|-------------------|------------------|-------|----------|--------|
| **Stillwater KPU-T64** | **16 INT8** | **~10 INT8** (60%) | **3-10W** | **$125** | ✅ Modeled |
| Google Coral Edge TPU | 4 INT8 | ~3.4 INT8 (85%) | 0.5-2W | $25 | ✅ Modeled |
| Hailo-8 | 26 INT8 | ~22 INT8 (85%) | 2.5W | $40 | ✅ Modeled |
| Jetson Orin Nano | 67 INT8 (Super) | ~2-4 INT8 (3-6%) | 7-15W | $205 | ✅ Modeled |
| Qualcomm QCS6490 | 12 TOPS mixed | ~6-8 TOPS (50-70%) | 5-15W | $85 | ✅ Modeled |

**KPU-T64 Positioning**:
- **Competitors**: Qualcomm Snapdragon 8 Elite (mobile), Hexagon NPU V79, Hailo-8, Coral Edge TPU
- **Differentiation**: 60% efficiency (vs 3-6% for Jetson, 50% for Snapdragon) due to native dataflow architecture
- **Target BOM**: $75-120 (competitive with Snapdragon, higher than Coral/Hailo but 3× performance)

### Mid-Range Edge AI (15-60W, $300-$1000)
**Target Applications**: Autonomous mobile robots, delivery robots, inspection drones, AR/VR headsets

| Product | TOPS (Advertised) | TOPS (Realistic) | Power | BOM Cost | Status |
|---------|-------------------|------------------|-------|----------|--------|
| **Stillwater KPU-T256** | **64 INT8** | **~45 INT8** (70%) | **15-50W** | **$455** | ✅ Modeled |
| Jetson Orin AGX | 275 INT8 | ~5-17 INT8 (2-6%) | 15-60W | $670 | ✅ Modeled |
| Hailo-10H | 40 INT4, 20 INT8 | ~32 INT4, ~17 INT8 (80-85%) | 2.5W | $70 | ✅ Modeled |
| Qualcomm SA8775P | 32 TOPS mixed | ~16-20 TOPS (50-60%) | 20-45W | $350 | ✅ Modeled |
| Intel ARC A370M | 8 TFLOPS FP32 | ~4 TFLOPS (50%) | 35-50W | $200-300 | ❌ Not Modeled |

**KPU-T256 Positioning**:
- **Competitors**: Jetson Orin AGX, Qualcomm SA8775P (automotive), Snapdragon 8cx Gen 3
- **Differentiation**: 3-9× effective TOPS vs Jetson AGX at same power budget
- **Target BOM**: $400-700 (competitive with Jetson AGX, positioned as superior alternative)

### High-End Automotive/Robotics (60-130W, $1000-$3000)
**Target Applications**: Autonomous vehicles (L3-L4), humanoid robots, industrial AGVs

| Product | TOPS (Advertised) | TOPS (Realistic) | Power | BOM Cost | Status |
|---------|-------------------|------------------|-------|----------|--------|
| **Stillwater KPU-T768** | **192 INT8** | **~135 INT8** (70%) | **60-100W** | **$1,225** | ✅ Modeled |
| Jetson Thor | 2000 INT8 (sparse) | ~30-100 INT8 (3-10%) | 30-130W | $1,600 | ✅ Modeled |
| Jetson Orin AGX (60W) | 275 INT8 | ~12-17 INT8 (4-6%) | 60W | $670 | ✅ Modeled |
| Qualcomm Snapdragon Ride | 700 TOPS mixed | ~350-420 TOPS (50-60%) | 65-130W | $800 | ✅ Modeled |
| Tesla FSD Computer | 144 TOPS | ~100 TOPS (70%) | 72W | $1500-2500 | ❌ Not Modeled |

**KPU-T768 Positioning**:
- **Competitors**: Jetson Thor, Qualcomm Snapdragon Ride, Mobileye EyeQ6
- **Differentiation**: 1.3-4.5× effective TOPS vs Jetson Thor, better power efficiency
- **Target BOM**: $1200-2000 (40-60% of Thor BOM, competitive with Snapdragon Ride)

---

## Existing Hardware Models Status

### ✅ Already Modeled (with thermal profiles and DVFS)
1. **Stillwater KPU-T64** - `src/graphs/hardware/models/accelerators/kpu_t64.py`
2. **Stillwater KPU-T256** - `src/graphs/hardware/models/accelerators/kpu_t256.py`
3. **Stillwater KPU-T768** - `src/graphs/hardware/models/accelerators/kpu_t768.py`
4. **Jetson Orin Nano** - `src/graphs/hardware/models/edge/jetson_orin_nano.py`
5. **Jetson Orin AGX** - `src/graphs/hardware/models/edge/jetson_orin_agx.py`
6. **Jetson Thor** - `src/graphs/hardware/models/automotive/jetson_thor.py`
7. **Google Coral Edge TPU** - `src/graphs/hardware/models/edge/coral_edge_tpu.py`
8. **Hailo-8** - `src/graphs/hardware/models/edge/hailo8.py` (+ mapper in `accelerators/hailo.py`)
9. **Hailo-10H** - `src/graphs/hardware/models/edge/hailo10h.py` (+ mapper in `accelerators/hailo.py`)
10. **Qualcomm QCS6490** - `src/graphs/hardware/models/edge/qualcomm_qcs6490.py`
11. **Qualcomm SA8775P** - `src/graphs/hardware/models/automotive/qualcomm_sa8775p.py`
12. **Qualcomm Snapdragon Ride** - `src/graphs/hardware/models/automotive/qualcomm_snapdragon_ride.py`

### ⚠️ Future Additions (Medium/Low Priority)

#### Medium Priority (Market Context)
1. **Qualcomm Snapdragon 8 Elite** (Latest mobile)
   - Hexagon NPU: 80 TOPS
   - Target: Mobile devices, edge AI
   - Location: `src/graphs/hardware/models/mobile/snapdragon_8_elite.py`

2. **Apple M4** (Competitive reference)
   - 38 TOPS neural engine
   - Target: Edge AI on ARM devices
   - Location: `src/graphs/hardware/models/edge/apple_m4.py`

#### Low Priority (Future work)
3. Tesla FSD Computer
4. Mobileye EyeQ6
5. Intel Arc A370M

---

## Embodied AI Benchmark Workload Suite

### Computer Vision Pipeline (Sequential Execution)

#### 1. Object Detection (YOLO family)
- **YOLOv8n** (nano): 3.2M params, 8.7 GFLOPs @ 640×640
- **YOLOv8s** (small): 11.2M params, 28.6 GFLOPs
- **YOLOv8m** (medium): 25.9M params, 78.9 GFLOPs
- **YOLOv11m** (latest medium): Similar to YOLOv8m
- **Precision**: INT8 quantized (required for edge deployment)

#### 2. Semantic Segmentation
- **DeepLabV3+ MobileNetV2**: 5.8M params, 17.3 GFLOPs @ 512×512
- **SegFormer B0**: 3.7M params, 15.6 GFLOPs
- **Precision**: INT8/FP16 mixed

#### 3. Object Re-Identification (Re-ID)
- **OSNet-x0.25**: 0.7M params, 0.4 GFLOPs (lightweight person Re-ID)
- **FastReID ResNet50**: 25M params, 4.1 GFLOPs
- **Precision**: FP16 (requires feature similarity preservation)

#### 4. Kalman Filter (State Estimation)
- **Traditional Kalman**: Minimal compute (matrix ops)
- **Extended Kalman Filter (EKF)**: Non-linear dynamics
- **Precision**: FP32 (numerical stability required)

#### 5. Full Embodied AI Pipeline (Sequential)
```
Input Frame (1280×720)
  → YOLOv8m (object detection): 78.9 GFLOPs
  → DeepLabV3+ (segmentation): 17.3 GFLOPs
  → OSNet (re-identification): 0.4 GFLOPs
  → EKF (state update): ~0.01 GFLOPs
  ──────────────────────────────────────
  Total: ~96.6 GFLOPs/frame

Target: 30 FPS → 2.9 TFLOPs/sec sustained compute
```

**Reality Check**: This pipeline runs **sequentially** on most edge accelerators (cannot overlap detection + segmentation due to memory constraints). Total latency = sum of individual latencies.

### Benchmark Characteristics

| Model | GFLOPs | Params | Precision | Bottleneck | KPU Tile Allocation |
|-------|--------|--------|-----------|------------|---------------------|
| YOLOv8n | 8.7 | 3.2M | INT8 | Memory BW | 70% INT8 tiles |
| YOLOv8m | 78.9 | 25.9M | INT8 | Balanced | 70% INT8, 20% BF16 |
| DeepLabV3+ | 17.3 | 5.8M | INT8/FP16 | Compute | 60% INT8, 30% BF16 |
| OSNet | 0.4 | 0.7M | FP16 | Memory BW | 80% BF16 tiles |
| EKF | 0.01 | - | FP32 | Compute | 10% Matrix tiles |

---

## Required Work Items

### 1. Hardware Resource Models (⚠️ Priority 1)

**Create models for:**
- `src/graphs/hardware/models/edge/qualcomm_qcs6490.py`
- `src/graphs/hardware/models/automotive/qualcomm_sa8775p.py`
- `src/graphs/hardware/models/automotive/qualcomm_snapdragon_ride.py`
- `src/graphs/hardware/models/mobile/snapdragon_8_elite.py`

**Each model needs:**
1. Thermal operating points (3-4 power profiles)
2. Clock domains with DVFS modeling
3. Compute resources with ops/clock per precision
4. Performance characteristics with efficiency factors
5. Memory hierarchy (bandwidth, caches)
6. Energy coefficients (pJ/op, pJ/byte)

**Calibration approach:**
- Start with published specs (advertised TOPS, TDP)
- Apply realistic derates (30-70% efficiency based on architecture)
- Cross-validate against published benchmarks where available
- Document calibration status (ESTIMATED vs CALIBRATED)

### 2. BOM Cost Modeling (⚠️ Priority 2)

Add `BOMCostProfile` to each hardware resource model:

```python
@dataclass
class BOMCostProfile:
    """Bill of Materials cost breakdown"""
    silicon_die_cost: float          # Die cost (process node dependent)
    package_cost: float               # Package (flip-chip, BGA, etc.)
    memory_cost: float                # On-package/on-module DRAM
    pcb_assembly_cost: float          # Board assembly
    thermal_solution_cost: float      # Heatsinks, thermal pads
    total_bom_cost: float             # Total BOM
    margin_multiplier: float          # Retail price = BOM × margin
    retail_price: float               # Customer price
    volume_tier: str                  # "10K+", "100K+", "1M+"
```

**Example BOM estimates:**
- **KPU-T64 (16nm)**: $75 (silicon) + $15 (package) + $20 (memory) + $10 (misc) = **$120 BOM**
- **Jetson Orin Nano**: $120 (silicon) + $25 (package) + $40 (8GB) + $15 (misc) = **$200 BOM** → $299 retail
- **Hailo-8 (16nm)**: $25 (silicon) + $8 (package) + $0 (on-chip) + $7 (misc) = **$40 BOM** → $60 retail

### 3. Benchmark Workload Models (⚠️ Priority 3)

**Add synthetic models or import from TorchVision:**
- `src/graphs/models/yolo/yolov8_nano.py`
- `src/graphs/models/yolo/yolov8_small.py`
- `src/graphs/models/yolo/yolov8_medium.py`
- `src/graphs/models/segmentation/deeplabv3_mobilenet.py`
- `src/graphs/models/reid/osnet.py`

**Alternatively**: Use FX tracing on pre-trained models from `torchvision` or `ultralytics` packages.

### 4. Validation Tests (⚠️ Priority 4)

**Create comprehensive validation suite:**

#### Individual Hardware Mapper Tests
- `validation/hardware/test_jetson_orin_nano.py`
- `validation/hardware/test_jetson_orin_agx.py`
- `validation/hardware/test_jetson_thor.py`
- `validation/hardware/test_coral_edge_tpu.py`
- `validation/hardware/test_hailo8.py`
- `validation/hardware/test_hailo10h.py`
- `validation/hardware/test_qualcomm_qcs6490.py` (NEW)
- `validation/hardware/test_qualcomm_sa8775p.py` (NEW)
- `validation/hardware/test_snapdragon_ride.py` (NEW)

**Test structure:**
```python
def test_jetson_orin_nano_resnet18():
    """Validate Jetson Orin Nano estimates against published benchmarks"""
    # Run ResNet-18 INT8 @ batch=1
    # Expected: ~20 FPS @ 7W (from NVIDIA benchmarks)
    # Our estimate should be within 20% of published data
    ...

def test_kpu_t64_vs_hailo8():
    """Cross-correlation: KPU-T64 vs Hailo-8 on YOLOv8n"""
    # Both are INT8 accelerators with similar power budgets
    # Relative performance ratio should match architecture differences
    ...
```

#### Cross-Correlation Tests
- `validation/hardware/test_embodied_ai_comparison.py` - Full 10-way comparison
- `validation/hardware/test_entry_level_comparison.py` - Entry-level products only
- `validation/hardware/test_automotive_comparison.py` - Automotive products only

**Validation approach:**
1. Run same model (e.g., ResNet-18 INT8 @ batch=1) on all hardware
2. Compare latency rankings (should match published benchmarks)
3. Compare energy rankings (should match TDP and efficiency)
4. Validate BOM cost vs market pricing

### 5. Comprehensive Comparison CLI Tool (⚠️ Priority 5)

**Create:** `cli/compare_embodied_ai.py`

**Features:**
```bash
# Compare all entry-level products on YOLOv8n
./cli/compare_embodied_ai.py \
  --tier entry \
  --model yolov8n \
  --precision int8 \
  --batch 1 \
  --output embodied_ai_comparison.csv

# Compare specific products on full pipeline
./cli/compare_embodied_ai.py \
  --hardware KPU-T64 KPU-T256 Jetson-Orin-Nano Jetson-Orin-AGX Hailo-8 \
  --workload embodied-ai-pipeline \
  --output pipeline_comparison.md

# Generate market positioning report
./cli/compare_embodied_ai.py \
  --all-tiers \
  --workload yolov8n yolov8m deeplabv3 \
  --metrics latency energy cost \
  --output market_positioning_report.pdf
```

**Output formats:**
- CSV: Raw data for Excel analysis
- Markdown: Human-readable tables
- JSON: For post-processing and visualization
- PDF: Executive summary with charts (using matplotlib)

### 6. Market Positioning Report (⚠️ Priority 6)

**Generate:** `docs/market_positioning_report.md`

**Contents:**
1. **Executive Summary**
   - KPU family positioning vs competitors
   - Key differentiators (efficiency, BOM cost, performance)

2. **Performance Comparison**
   - Latency comparison on YOLO/segmentation workloads
   - FPS vs Power trade-off curves
   - Batch size scaling analysis

3. **Energy Efficiency Analysis**
   - Energy per inference (mJ/frame)
   - Performance per Watt (FPS/W)
   - Battery life implications for mobile robots

4. **BOM Cost Analysis**
   - BOM cost vs performance (TOPS/dollar)
   - Total Cost of Ownership (TCO) for robotics deployment
   - Volume pricing assumptions

5. **Market Recommendations**
   - Target customers per KPU SKU
   - Competitive positioning strategies
   - Pricing recommendations

---

## Implementation Timeline

### Phase 1: Hardware Models (Week 1)
- [ ] Create Qualcomm QCS6490 resource model
- [ ] Create Qualcomm SA8775P resource model
- [ ] Create Snapdragon Ride resource model
- [ ] Add BOM cost profiles to all models

### Phase 2: Benchmark Models (Week 2)
- [ ] Add YOLO models (YOLOv8n, YOLOv8m, YOLOv11m)
- [ ] Add DeepLabV3+ segmentation
- [ ] Add OSNet re-identification
- [ ] Add Kalman filter baseline

### Phase 3: Validation (Week 3)
- [ ] Individual hardware mapper tests
- [ ] Cross-correlation tests
- [ ] Calibrate efficiency factors against published benchmarks
- [ ] Document calibration status

### Phase 4: Comparison Tools (Week 4)
- [ ] Create compare_embodied_ai.py CLI tool
- [ ] Generate comparison CSV/Markdown/JSON
- [ ] Create visualization scripts (matplotlib)

### Phase 5: Market Report (Week 5)
- [ ] Generate market positioning report
- [ ] Create executive summary
- [ ] Validate competitive analysis with sales/marketing
- [ ] Deliver final deliverables

---

## Key Insights for KPU Positioning

### 1. The "TOPS Gap" Reality
**Marketing vs Reality:**
- **Jetson Orin AGX**: 275 TOPS advertised → 5-17 TOPS realistic (2-6%)
- **Jetson Thor**: 2000 TOPS advertised → 30-100 TOPS realistic (1.5-5%)
- **KPU-T256**: 64 TOPS advertised → ~45 TOPS realistic (70%)

**Implication**: KPU-T256 delivers **3-9× effective TOPS** vs Jetson AGX at same power budget.

### 2. Power Budget Matters More Than Peak TOPS
**Embodied AI deployment constraints:**
- Battery-powered robots: 7-15W budget
- Tethered robots: 15-30W budget
- Autonomous vehicles: 60-100W budget

**At 15W:**
- Jetson Orin AGX: ~5 TOPS effective (18% throttling!)
- KPU-T256: ~32 TOPS effective (no throttling)

### 3. Sequential Execution Dominates Edge AI
**Most accelerators run Embodied AI pipelines sequentially:**
- Cannot overlap detection + segmentation (memory constraints)
- Small DNNs don't saturate massive arrays (TPU, Jetson)
- Latency = sum of individual latencies, not max

**KPU advantage**: Optimized for small DNN sequential execution with minimal tiling overhead.

### 4. BOM Cost is Critical for Volume Deployment
**Robotics companies deploying 10K+ units care about:**
- BOM cost (not retail price)
- Power consumption (affects battery size/weight)
- Thermal solution cost (passive vs active cooling)

**KPU BOM positioning:**
- T64: $120 BOM vs $200 for Jetson Nano (40% savings)
- T256: $550 BOM vs $700 for Jetson AGX (21% savings)
- T768: $1600 BOM vs $2500 for Thor (36% savings)

---

## Next Steps

1. **Immediate**: Add Qualcomm resource models (QCS6490, SA8775P, Snapdragon Ride)
2. **Week 1**: Add BOM cost profiles to all existing models
3. **Week 2**: Add YOLO benchmark workloads
4. **Week 3**: Create validation test suite
5. **Week 4**: Build comparison CLI tool
6. **Week 5**: Generate market positioning report

**Deliverable**: Comprehensive latency/energy/cost comparison that demonstrates KPU competitive advantage for Embodied AI applications.
