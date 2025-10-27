# How to Use: Specialized Comparison Tools

## Overview

The graphs project includes specialized comparison tools for evaluating hardware in specific deployment scenarios:

1. **compare_automotive_adas.py** - Automotive ADAS (Level 2-3)
2. **compare_datacenter_cpus.py** - Datacenter server CPUs
3. **compare_edge_ai_platforms.py** - Edge AI accelerators
4. **compare_ip_cores.py** - Licensable IP cores for SoC integration
5. **compare_i7_12700k_mappers.py** - Intel i7-12700K CPU variants

These tools provide **pre-configured comparisons** with realistic use cases, power budgets, and latency requirements.

---

## 1. compare_automotive_adas.py

### Purpose

Compare AI accelerators for automotive Advanced Driver Assistance Systems (ADAS Level 2-3).

### Quick Start

```bash
python3 cli/compare_automotive_adas.py
```

---

### Features

**Two ADAS Categories:**
1. **Front Camera ADAS (10-15W)**: Lane Keep Assist, ACC, Traffic Sign Recognition
2. **Multi-Camera ADAS (15-25W)**: Surround View, Parking Assist

**Hardware Evaluated:**
- TI TDA4VM (6 TOPS, automotive-grade)
- Jetson Orin Nano (40 TOPS, 7-15W)
- Jetson AGX Orin (275 TOPS, 15-60W)
- Stillwater KPU-T256 (33.8 TOPS, 30W)

**Models:**
- ResNet-50 (object detection backbone)
- FCN (lane segmentation)
- YOLOv5 (automotive variant)

**Requirements:**
- 30 FPS minimum (real-time control)
- <100ms latency (safety-critical)
- ASIL-D certification preferred

---

### Output Example

```
================================================================================
AUTOMOTIVE ADAS COMPARISON
================================================================================

CATEGORY 1: Front Camera ADAS (10-15W Budget)
--------------------------------------------------------------------------------
Hardware              Power  Latency  FPS    30FPS?  <100ms?  Score   Rec?
--------------------------------------------------------------------------------
TI-TDA4VM-C7x         10W    110.76   9.0    ✗       ✗        45/100  ⚠
Jetson-Orin-Nano      15W    5.45     183.5  ✓       ✓        92/100  ✓

CATEGORY 2: Multi-Camera ADAS (15-25W Budget)
--------------------------------------------------------------------------------
Hardware              Power  Latency  FPS    30FPS?  <100ms?  Score   Rec?
--------------------------------------------------------------------------------
Jetson-Orin-Nano      15W    5.45     183.5  ✓       ✓        92/100  ✓
Jetson-Orin-AGX       25W    3.12     320.5  ✓       ✓        95/100  ✓✓

RECOMMENDATIONS:
  Front Camera (10-15W):  Jetson Orin Nano @ 15W
  Multi-Camera (15-25W):  Jetson AGX Orin @ 25W

INSIGHTS:
  - TI TDA4VM insufficient for ResNet-50 @ 30 FPS
  - Jetson Orin family dominates automotive AI
  - KPU-T256 @ 30W competitive but over budget
```

---

### Use Cases

**Planning Automotive AI:**
- L2 ADAS (adaptive cruise, lane keep)
- L3 ADAS (highway pilot, parking)
- Selecting SoC for next-gen ECU

**Key Metrics:**
- **FPS/W**: Battery/thermal efficiency
- **30 FPS**: Real-time control requirement
- **<100ms latency**: Safety-critical response
- **Score**: Multi-factor (50% perf, 20% eff, 20% latency, 10% safety)

---

## 2. compare_datacenter_cpus.py

### Purpose

Compare ARM and x86 datacenter server CPUs for AI inference workloads.

### Quick Start

```bash
python3 cli/compare_datacenter_cpus.py
```

---

### Features

**CPUs Compared:**
- **Ampere AmpereOne 192-core** (ARM, 350W)
- **Intel Xeon Platinum 8490H** (x86, 60-core, 350W, AMX)
- **AMD EPYC 9654** (x86, 96-core, 360W)

**Models Tested:**
- ResNet-50 (CNN baseline)
- DeepLabV3+ (Segmentation)
- ViT-Base (Transformer)

**Key Question:** Which CPU architecture is best for different AI workloads?

---

### Output Example

```
================================================================================
DATACENTER CPU COMPARISON RESULTS
================================================================================

ResNet-50 (CNN Workload)
--------------------------------------------------------------------------------
CPU                            Cores  TDP    Latency    FPS      FPS/W    Rank
--------------------------------------------------------------------------------
Intel Xeon Platinum 8490H      60     350    0.87       1143.6   3.27     1 ✓
AMD EPYC 9654                  96     360    4.61       216.8    0.60     3
Ampere AmpereOne 192-core      192    350    4.24       235.8    0.67     2

Winner: Intel Xeon (AMX acceleration) - 4.8× faster than AMD

DeepLabV3+ (Segmentation)
--------------------------------------------------------------------------------
Intel Xeon Platinum 8490H      60     350    8.47       118.1    0.34     1 ✓
AMD EPYC 9654                  96     360    85.36      11.7     0.03     3
Ampere AmpereOne 192-core      192    350    74.03      13.5     0.04     2

Winner: Intel Xeon (AMX) - 10× faster than AMD

ViT-Base (Transformer)
--------------------------------------------------------------------------------
AMD EPYC 9654                  96     360    1.14       878.3    2.44     1 ✓
Ampere AmpereOne 192-core      192    350    1.53       654.0    1.87     2
Intel Xeon Platinum 8490H      60     350    1.65       606.3    1.73     3

Winner: AMD EPYC (highest bandwidth) - 1.4× faster than Intel

================================================================================
KEY INSIGHTS
================================================================================

1. Intel AMX Dominates CNNs:
   - 4-10× faster on Conv2D workloads
   - INT8 matrix extensions critical
   - Best for: ResNet, EfficientNet, MobileNet, segmentation

2. AMD's High Bandwidth Excels at Transformers:
   - 460.8 GB/s (vs 307 GB/s Intel, 332.8 GB/s Ampere)
   - Memory-bound operations benefit
   - Best for: ViT, BERT, GPT, LLM inference

3. Ampere's 192 Cores Best for General Compute:
   - Not AI-optimized (no AMX equivalent)
   - Good for cloud-native microservices
   - Poor AI performance relative to x86

RECOMMENDATIONS:
  CNN Inference:         Intel Xeon (AMX)
  Transformer Inference: AMD EPYC (high BW)
  Mixed Workloads:       Intel Xeon (balanced)
  Cloud-Native:          Ampere AmpereOne
```

---

### Use Cases

**Datacenter Planning:**
- Selecting CPUs for AI inference clusters
- Cost/performance optimization
- Workload-specific deployment

**Key Insight:** CPU architecture matters significantly for AI workloads (4-10× performance difference).

---

## 3. compare_edge_ai_platforms.py

### Purpose

Compare edge AI accelerators for embodied AI and robotics platforms.

### Quick Start

```bash
python3 cli/compare_edge_ai_platforms.py
```

---

### Features

**Two Categories:**
1. **Computer Vision / Low Power (≤10W)**: Drones, robots, smart cameras
2. **Transformers / Higher Power (≤50W)**: Autonomous vehicles, edge servers

**Hardware Evaluated:**
- Hailo-8 / Hailo-10H (13-26 TOPS, 2.5-8W)
- Jetson Orin Nano / AGX (40-275 TOPS, 7-60W)
- Stillwater KPU-T64 / T256 (6.9-33.8 TOPS, 6-30W)
- Qualcomm QRB5165 (15 TOPS, 5W)
- TI TDA4VM (6 TOPS, 5W)

**Models:**
- ResNet-50 (vision backbone)
- DeepLabV3+ (segmentation)
- ViT-Base (transformer, for Cat 2)

---

### Output Example

```
================================================================================
EDGE AI PLATFORM COMPARISON
================================================================================

CATEGORY 1: Computer Vision / Low Power (≤10W)
--------------------------------------------------------------------------------
Hardware              Peak TOPS  Power  Latency  FPS    FPS/W   TOPS/W  Rec
--------------------------------------------------------------------------------
Hailo-8               26         2.5W   8.2ms    122    48.8    10.4    ✓✓
KPU-T64               6.9        6.0W   4.2ms    238    39.8    1.15    ✓
Jetson-Orin-Nano      40         7.0W   5.5ms    183    26.1    5.7     ✓
QRB5165-Hexagon698    15         5.0W   45.3ms   22     4.4     3.0     -
TI-TDA4VM-C7x         6          5.0W   128.1ms  7.8    1.6     1.2     -

Best for Battery:     Hailo-8 (48.8 FPS/W)
Best Performance:     KPU-T64 (238 FPS)
Most Efficient:       Hailo-8 (10.4 TOPS/W)

CATEGORY 2: Transformers / Higher Power (≤50W)
--------------------------------------------------------------------------------
Hardware              Peak TOPS  Power  Latency  FPS    FPS/W   TOPS/W  Rec
--------------------------------------------------------------------------------
Jetson-Orin-AGX       275        50W    12.3ms   81     1.62    5.5     ✓
Hailo-10H             26         8W     85.2ms   11.7   1.46    3.25    -
KPU-T256              33.8       30W    94.7ms   10.6   0.35    1.13    -

Best Performance:     Jetson AGX Orin (81 FPS)
Recommendation:       Jetson AGX Orin @ 50W

================================================================================
KEY INSIGHTS
================================================================================

1. Hailo Dominates Ultra-Low-Power:
   - 48.8 FPS/W (best in class)
   - 10.4 TOPS/W (most efficient)
   - Ideal for: Drones, battery-powered cameras

2. KPU Excels at CNN Performance:
   - 238 FPS (highest throughput)
   - 39.8 FPS/W (2nd most efficient)
   - Ideal for: Robots, smart cameras, edge servers

3. Jetson Best for Transformers:
   - Only viable option for ViT @ 50W
   - Poor for CNNs vs specialized accelerators
   - Ideal for: Edge servers, AVs with ViT/attention

4. QRB5165/TI TDA4VM Insufficient:
   - <30 FPS on ResNet-50
   - Suitable for smaller models only
```

---

### Use Cases

**Edge AI Deployment:**
- Autonomous robots (mobile, industrial)
- Drones and UAVs
- Smart cameras and surveillance
- Edge servers
- Autonomous vehicles (edge inference)

**Key Metrics:**
- **FPS/W**: Battery life
- **TOPS/W**: Efficiency
- **Latency**: Real-time requirements

---

## 4. compare_ip_cores.py

### Purpose

Compare licensable AI/compute IP cores for custom SoC integration.

### Quick Start

```bash
python3 cli/compare_ip_cores.py
```

---

### Features

**Two Architecture Categories:**

1. **Traditional (Stored-Program Extensions):**
   - CEVA NeuPro-M NPM11 (DSP + NPU)
   - Cadence Tensilica Vision Q8 (Vision DSP)
   - Synopsys ARC EV7x (CPU + VPU + DNN)
   - ARM Mali-G78 MP20 (GPU)

2. **Dataflow (AI-Native):**
   - Stillwater KPU-T64/T256 (Dataflow NPU)

**Models:**
- ResNet-50, DeepLabV3+, ViT-Base

**Key Question:** Stored-program extensions vs AI-native dataflow?

---

### Output Example

```
================================================================================
IP CORE COMPARISON FOR SoC INTEGRATION
================================================================================

TRADITIONAL ARCHITECTURES (Stored-Program Extensions)
--------------------------------------------------------------------------------
IP Core                Vendor      Type        Power  Latency  FPS   FPS/W  Util%
--------------------------------------------------------------------------------
CEVA NeuPro-M NPM11    CEVA        DSP+NPU     2.0W   150.57   6.6   3.32   29.3%
Cadence Vision Q8      Cadence     Vision DSP  1.0W   225.30   4.4   4.44   47.7%
Synopsys ARC EV7x      Synopsys    CPU+VPU     5.0W   364.06   2.7   0.55   14.7%
ARM Mali-G78 MP20      ARM         GPU IP      5.0W   1221.83  0.8   0.16   99.2%

DATAFLOW ARCHITECTURES (AI-Native)
--------------------------------------------------------------------------------
IP Core                Vendor      Type        Power  Latency  FPS   FPS/W  Util%
--------------------------------------------------------------------------------
KPU-T64                Stillwater  Dataflow    6.0W   4.19     238.8 39.79  98.8%
KPU-T256               Stillwater  Dataflow    30.0W  1.12     893.2 29.77  90.9%

================================================================================
KEY INSIGHTS
================================================================================

1. Dataflow Architecture Advantage:
   - KPU-T64: 36× faster than CEVA (238 FPS vs 6.6 FPS)
   - KPU-T64: 12× more efficient (39.8 vs 3.3 FPS/W)
   - Root Cause: AI-native design vs retrofitted ISA extensions

2. Traditional IP Limitations:
   - Low utilization (14-48% typical)
   - Stored-program overhead
   - Not optimized for AI dataflow

3. SoC Integration Trade-offs:
   - Traditional: Proven toolchains, mature ecosystems
   - Dataflow: Superior performance, but new tooling

RECOMMENDATIONS BY USE CASE:
  Mobile Flagship:      CEVA NeuPro-M (2W, proven)
  Automotive ADAS:      KPU-T64/T256 (performance-critical)
  Edge AI / Embodied:   KPU-T64/T256 (best FPS/W)
  IoT / Always-On:      Cadence Vision Q8 (1W, lowest power)
  General Compute:      ARM Mali-G78 (GPU programmability)
```

---

### Use Cases

**SoC Design:**
- Integrating AI acceleration into custom chips
- Comparing traditional vs dataflow architectures
- Power/performance/cost optimization

**Key Insight:** AI-native dataflow architectures (KPU) provide 10-40× better efficiency than stored-program extensions.

---

## 5. compare_i7_12700k_mappers.py

### Purpose

Compare Intel i7-12700K CPU mapper performance with different cache configurations.

### Quick Start

```bash
python3 cli/compare_i7_12700k_mappers.py
```

---

### Features

**CPU Variants:**
- Standard i7-12700K (25 MB L3 cache)
- Large cache variant (30 MB L3 cache)

**Models:**
- ResNet-50, DeepLabV3+, ViT-Base

**Key Question:** How much does L3 cache size impact AI inference?

---

### Output Example

```
================================================================================
i7-12700K CACHE COMPARISON
================================================================================

ResNet-50
--------------------------------------------------------------------------------
Variant              L3 Cache  Latency   FPS      Improvement
--------------------------------------------------------------------------------
Standard i7-12700K   25 MB     8.45ms    118.3    baseline
Large Cache          30 MB     7.98ms    125.3    +5.9%

ViT-Base (Memory-Sensitive)
--------------------------------------------------------------------------------
Variant              L3 Cache  Latency   FPS      Improvement
--------------------------------------------------------------------------------
Standard i7-12700K   25 MB     15.32ms   65.3     baseline
Large Cache          30 MB     13.87ms   72.1     +10.4%

KEY INSIGHTS:
  - L3 cache impact: 6-10% improvement
  - Higher impact on Transformers (memory-intensive)
  - Modest gains for CNNs (compute-bound)
```

---

### Use Cases

**Workstation/Desktop AI:**
- Selecting CPU for AI development
- Understanding cache sensitivity
- Cost/benefit of higher-end SKUs

---

## Common Patterns Across Tools

### Data-Driven Recommendations

All tools use **multi-factor scoring**:
- 50% Performance (latency, FPS)
- 20% Efficiency (FPS/W, TOPS/W)
- 20% Latency (<threshold requirements)
- 10% Deployment-specific (safety, certifications)

### Realistic Constraints

- **Power budgets** based on deployment scenarios
- **Latency requirements** for real-time systems
- **Thermal limits** for passively cooled systems

### Architecture Comparisons

- CPU vs GPU vs Accelerator
- Stored-program vs Dataflow
- ARM vs x86 vs custom

---

## When to Use Each Tool

| Tool | Use When |
|------|----------|
| **compare_automotive_adas.py** | Planning ADAS L2-L3 systems |
| **compare_datacenter_cpus.py** | Deploying AI inference on CPUs |
| **compare_edge_ai_platforms.py** | Selecting edge AI accelerators |
| **compare_ip_cores.py** | Integrating AI into custom SoCs |
| **compare_i7_12700k_mappers.py** | Optimizing desktop/workstation AI |

---

## General-Purpose Comparison Tools

For flexible comparisons, use:
- **analyze_graph_mapping.py --compare**: Custom hardware lists
- **compare_models.py**: One model across many hardware

---

## Further Reading

- **Hardware Discovery**: `cli/docs/list_hardware_mappers.md`
- **Graph Mapping**: `cli/docs/analyze_graph_mapping.md`
- **Architecture Guide**: `CLAUDE.md`

---

## Contact & Feedback

Report issues or request features at the project repository.
