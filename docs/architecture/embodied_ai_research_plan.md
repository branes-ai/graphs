# Embodied AI Accelerator Research Plan

**Research Question**: Which hardware architectures are best suited for real-time inference in robots and drones?

**Focus**: Edge accelerators for embodied AI (power & compute constrained)
**Timeline**: 2-3 weeks implementation + 1-2 weeks paper writing
**Date**: 2025-10-21

---

## 1. Research Motivation

### 1.1 The Embodied AI Challenge

**Embodied AI** (robots, drones, autonomous vehicles) has fundamentally different requirements than datacenter AI:

| Constraint | Datacenter (H100, TPU) | Embodied AI (Robot/Drone) |
|------------|------------------------|---------------------------|
| **Power Budget** | 300-700W | **5-20W** (battery limited) |
| **Latency** | 10-100ms acceptable | **<10ms** (real-time control) |
| **Batch Size** | 64-256 (throughput) | **1-4** (single sensor streams) |
| **Model Size** | 1-100B params | **1-100M params** (memory limited) |
| **Environment** | Controlled datacenter | **Harsh** (vibration, temperature) |
| **Cost** | $10K-30K per unit | **$100-1000** (mass production) |

**Key Insight**: Datacenter accelerators (H100, TPU v4) are **fundamentally unsuitable** for embodied AI despite being 10-100× faster. Power and cost constraints dominate.

### 1.2 Target Workloads

**Primary Models** (real embodied AI use cases):
1. **Vision Transformers (ViT)** - Visual understanding, mentioned by user
2. **YOLO (YOLOv5, YOLOv8)** - Object detection for navigation
3. **MobileNet-V2/V3** - Efficient CNN backbone
4. **Small language models (1-7B)** - Multimodal reasoning (optional)

**Key Operations** (must be supported):
- Conv2D (CNNs)
- Linear/MatMul (Transformers)
- Attention (Transformers)
- Pool, ReLU, Concat (universal)

### 1.3 Research Gap

**Existing work** compares datacenter hardware (GPU vs TPU) or studies single edge accelerators in isolation.

**Our contribution**: First comprehensive comparison of **edge accelerators** specifically for embodied AI:
- KPU (existing edge, 100 TOPS)
- **DPU (Xilinx Vitis AI, new)** ← Focus architecture 1
- **CGRA (Plasticine-like, new)** ← Focus architecture 2
- CPU (baseline)
- GPU/TPU (comparison only - not viable for edge)

**Novel insights**:
- Which architecture provides best performance/watt for ViT?
- Can DPU/CGRA support transformer attention efficiently?
- What's the performance/cost/flexibility trade-off?

---

## 2. Implementation Plan

### 2.1 Phase 1: DPU Mapper (Week 1)

**Target**: Xilinx Vitis AI DPU (Versal AI Engine based)

**Architecture Specifications** (to look up and confirm):
```
Xilinx Versal AI Engine (AIE) Array:
┌─────────────────────────────────────┐
│ AIE Tiles: 400 tiles (VC1902)       │
│ Each tile:                           │
│   - VLIW SIMD processor              │
│   - Vector unit: 8×INT8 MACs         │
│   - Scalar unit                      │
│   - 32KB data memory                 │
│   - 16KB program memory              │
│ Frequency: 1.0-1.3 GHz               │ ← NEED TO CONFIRM
│                                      │
│ DPU Configuration (typical):         │
│   - 4096 INT8 MACs (64×64 equiv)    │
│   - Or: 1280, 2560, 3136 MACs       │
└─────────────────────────────────────┘

Theoretical Peak (4096 MACs @ 1 GHz):
  = 4096 MACs × 2 ops/MAC × 1 GHz
  = 8.192 TOPS INT8

Realistic (75% efficiency, per user):
  = 8.192 × 0.75
  = 6.144 TOPS INT8
```

**Research Tasks** (Day 1):
- [ ] Confirm AIE clock frequency (Xilinx documentation)
- [ ] Confirm DPU configurations available (1280, 2560, 4096 MACs)
- [ ] Find power consumption specs (target: 5-15W)
- [ ] Document supported operations (Conv, Pool, ElementWise, MatMul?)

**Implementation** (Day 2-3):
```python
# File: src/graphs/characterize/dpu_mapper.py

class DPUMapper(HardwareMapper):
    """
    Xilinx Vitis AI DPU mapper.

    Models Versal AI Engine array running DPU workloads.
    Assumes 75% of theoretical peak performance.
    """

    def __init__(self, resource_model):
        # DPU-specific params
        self.aie_tiles = 400  # VC1902
        self.mac_units = 4096  # Typical config
        self.clock_freq = 1.0e9  # 1 GHz
        self.efficiency = 0.75  # Per user requirement
        self.scratchpad_per_tile = 32 * 1024  # 32KB

        # Theoretical peak
        self.peak_tops = (self.mac_units * 2 * self.clock_freq) / 1e12
        # = 8.192 TOPS

        # Realistic peak
        self.realistic_tops = self.peak_tops * self.efficiency
        # = 6.144 TOPS
```

**Key Modeling Decisions**:
1. **Array Utilization**: Calculate based on operation size vs 4096 MACs
2. **Tiling**: 32KB scratchpad per tile (similar to KPU but smaller)
3. **Operation Support**:
   - ✅ Conv2D, Pool, ElementWise (native DPU ops)
   - ⚠️ MatMul, Attention (need to check - critical for ViT!)
4. **INT8 Native**: No quantization overhead (unlike GPU/TPU FP32→INT8)

**Testing** (Day 4):
- Test on ResNet-18 (baseline CNN)
- Test on MobileNet-V2 (efficient CNN)
- Test on ViT-Tiny (transformer) ← **Critical for embodied AI**
- Compare vs KPU (similar edge accelerator)

**Expected Results**:
```
DPU (Vitis AI, 4096 MACs @ 1GHz, 75% eff):
  ResNet-18:    ~2-3 ms   (vs KPU: ~0.05 ms - KPU faster!)
  MobileNet-V2: ~1-2 ms   (vs KPU: ~0.03 ms)
  ViT-Tiny:     ~10-20 ms (vs KPU: unknown)

  Energy: ~0.015-0.030 J (15mJ @ 5-10W)
  Position: Between KPU and CPU
```

**Deliverable**: `dpu_mapper.py` (~300-400 lines) + validation on 3 models

---

### 2.2 Phase 2: CGRA Mapper (Week 2)

**Target**: Plasticine-like architecture (Stanford, ISCA 2017)

**Architecture Specifications**:
```
Plasticine CGRA:
┌─────────────────────────────────────┐
│ Pattern Compute Units (PCUs): 32    │
│ Pattern Memory Units (PMUs): 32     │
│                                      │
│ Each PCU:                            │
│   - 16 SIMD lanes                    │
│   - Reconfigurable pipeline          │
│   - Local register file (256 regs)  │
│   - FP32/INT32 support               │
│                                      │
│ Interconnect:                        │
│   - Hierarchical mesh                │
│   - 256-bit links                    │
│                                      │
│ Reconfiguration:                     │
│   - 16 contexts (configs)            │
│   - Switch time: ~10 cycles          │
│   - Load time: ~100 cycles           │
│                                      │
│ Frequency: 1 GHz                     │
│ Power: ~2-5W                         │
└─────────────────────────────────────┘

Performance:
  Peak: ~40 GFLOPS FP32
  Realistic (75%): ~30 GFLOPS FP32
```

**Implementation** (Day 1-3):
```python
# File: src/graphs/characterize/cgra_mapper.py

class CGRAMapper(HardwareMapper):
    """
    Plasticine-like CGRA mapper.

    Key differences from other mappers:
    1. Spatial mapping (not temporal)
    2. Reconfiguration overhead
    3. Greedy PnR (per user requirement)
    """

    def _greedy_place_and_route(self, subgraph):
        """
        Greedy PnR heuristic:
        1. Count operations and types
        2. Find critical path (longest dependency chain)
        3. Map critical path to PEs first
        4. Fill remaining PEs with parallel ops
        5. Check if fits (if not, partition)
        """
        # Count ops
        op_counts = count_operations(subgraph)
        # {conv2d: 5, relu: 10, pool: 2}

        # Find critical path
        critical_path = find_critical_path(subgraph)
        # [conv1 -> relu1 -> conv2 -> relu2 -> pool1]

        # Allocate PEs
        pe_allocation = allocate_to_amorphous_resources(
            critical_path=critical_path,
            available_pces=32,
            available_pmus=32
        )

        if pe_allocation.total_pes_needed > 64:
            # Doesn't fit - need to partition
            return partition_and_reconfigure(subgraph)

        return pe_allocation
```

**Key Modeling Decisions**:
1. **Greedy PnR**: Simple heuristic (not optimal, but fast)
2. **Critical Path First**: Minimize latency by optimizing longest path
3. **Amorphous Resources**: Don't model exact PE grid positions (too complex)
4. **Reconfiguration Cost**: Track context switches (10-100 cycles)
5. **Partitioning**: If subgraph too large, split and reconfigure

**Testing** (Day 4-5):
- Test on ResNet-18 (many reconfigurations expected)
- Test on MobileNet-V2 (depthwise separable convs)
- Test on ViT-Tiny (attention - does it map well?)
- Analyze reconfiguration overhead (critical for CGRA)

**Expected Results**:
```
CGRA (Plasticine, 32 PCUs @ 1GHz, 75% eff):
  ResNet-18:    ~15-30 ms  (slow - many reconfigs!)
  MobileNet-V2: ~10-20 ms  (better - fewer ops)
  ViT-Tiny:     ~20-40 ms  (attention may not map well)

  Energy: ~0.003-0.005 J (3-5mJ @ 2-5W)
  Position: Slowest but most energy efficient
```

**Challenge**: High reconfiguration overhead may hurt performance

**Deliverable**: `cgra_mapper.py` (~500-600 lines) + validation on 3 models

---

### 2.3 Phase 3: Embodied AI Model Testing (Week 3)

**Test Models** (beyond ResNet-18):

**1. Vision Transformer (ViT-Tiny)**:
```python
# ViT-Tiny for embodied AI
Parameters: ~5.7M
Input: 224×224×3
Patches: 16×16 (196 patches)
Embedding: 192
Heads: 3
Layers: 12

Operations:
  - Patch embedding: Conv2D (16×16, stride 16)
  - Self-attention: QKV MatMul + Attention + Output
  - MLP: 2 Linear layers
  - Total: ~1.3 GFLOPs
```

**Why ViT is critical**:
- Transformers dominate modern AI
- Self-attention is new operation (not just Conv)
- Tests if edge accelerators can handle transformers

**2. YOLOv5-Nano** (object detection):
```python
# YOLOv5-Nano for drone navigation
Parameters: ~1.9M
Input: 640×640×3
Backbone: CSPNet (Conv-based)
Head: Detection heads

Operations:
  - Conv2D (many small convs)
  - Concat (feature pyramid)
  - Total: ~4.5 GFLOPs
```

**3. MobileNet-V2** (already tested):
- Baseline efficient CNN
- Depthwise separable convolutions

**Testing Protocol**:
```python
# For each model, test on all 6 hardware types:
for model in [ResNet18, MobileNetV2, ViT_Tiny, YOLOv5_Nano]:
    for hardware in [GPU, TPU, KPU, DPU, CGRA, CPU]:
        for precision in [FP32, INT8]:
            results = test(model, hardware, precision, batch=1)

            metrics = {
                'latency': results.latency,
                'energy': results.energy,
                'throughput': 1000 / results.latency,
                'energy_per_inference': results.energy,
                'real_time': results.latency < 10ms,  # Can hit 100 FPS?
                'battery_life': estimate_battery_life(results.energy, battery_mAh=5000)
            }
```

**Key Metrics for Embodied AI**:
1. **Latency < 10ms?** (100 FPS for real-time control)
2. **Energy per inference** (battery life)
3. **Model coverage** (can it run ViT? YOLO?)
4. **Cost** (feasibility for mass production)

**Expected Results**:
```
Model: ViT-Tiny (1.3 GFLOPs, INT8, Batch=1)

GPU (H100):   0.01 ms, 0.001 J  ✓ Real-time  ✗ Too expensive/power
TPU (v4):     0.02 ms, 0.001 J  ✓ Real-time  ✗ Too expensive/power
KPU (T100):   0.10 ms, 0.001 J  ✓ Real-time  ✓ Energy OK  ✓ BEST FOR EMBODIED AI
DPU (Vitis):  5-10 ms, 0.015 J  ✗ Not real-time (but close)
CGRA (Plas):  20-40 ms, 0.003 J ✗ Not real-time  ✓ Best energy
CPU (Intel):  50-100 ms, 0.05 J ✗ Not real-time  ✗ High energy
```

**Key Finding** (predicted):
- **KPU wins for embodied AI** (real-time + energy efficient)
- DPU close second (slightly slower but good energy)
- CGRA most energy efficient but too slow for real-time
- GPU/TPU too power-hungry for edge (despite being fastest)

---

## 3. Research Paper Outline

**Title**: "Hardware Accelerator Evaluation for Embodied AI: A Comprehensive Study of Edge Architectures"

**Abstract** (~200 words):
```
Embodied AI systems (robots, drones, autonomous vehicles) require
real-time inference under severe power and compute constraints.
While datacenter accelerators (GPU, TPU) offer high performance,
their 300-700W power consumption makes them unsuitable for
battery-powered edge deployment. This paper presents the first
comprehensive evaluation of edge accelerators specifically for
embodied AI workloads, including Vision Transformers (ViT) which
are increasingly important for visual understanding.

We evaluate 6 hardware architectures: GPU (H100), TPU (v4),
KPU (edge tensor), DPU (Xilinx Vitis AI), CGRA (Plasticine-like),
and CPU (x86). Our analysis spans 4 representative models
(ResNet-18, MobileNet-V2, ViT-Tiny, YOLOv5-Nano) across multiple
precisions (FP32, INT8).

Key findings: (1) Edge accelerators (KPU, DPU) achieve 10-20×
better energy efficiency than datacenter accelerators while
maintaining real-time performance (<10ms latency). (2) DPU
provides excellent CNN performance but struggles with transformer
attention. (3) CGRA offers best energy efficiency (3mJ/inference)
but high reconfiguration overhead limits real-time use. (4) KPU
emerges as the optimal architecture for embodied AI, balancing
performance, energy, and model coverage.
```

**1. Introduction** (2 pages)
- Embodied AI challenge
- Why datacenter accelerators don't work for edge
- Research questions:
  1. Which edge accelerator is best for embodied AI?
  2. Can edge accelerators handle modern transformers (ViT)?
  3. What are the performance/energy/flexibility trade-offs?

**2. Background** (2 pages)
- 2.1 Embodied AI Workloads
  - Vision Transformers (ViT)
  - Object Detection (YOLO)
  - Efficient CNNs (MobileNet)
- 2.2 Hardware Architectures
  - Datacenter: GPU (SM-based), TPU (systolic array)
  - Edge: KPU (tile-based), DPU (2D ALU array), CGRA (spatial fabric)
  - Baseline: CPU (multi-core)

**3. Methodology** (3 pages)
- 3.1 Hardware Modeling Framework
  - Realistic utilization (not naive 100%)
  - Precision-aware performance
  - Energy modeling
- 3.2 Mapper Implementations
  - GPU: SM allocation
  - DPU: 2D array + tiling
  - CGRA: Greedy place-and-route
  - (Brief description of each)
- 3.3 Evaluation Metrics
  - Latency (real-time: <10ms)
  - Energy (battery life)
  - Model coverage (ViT support?)
  - Cost (deployment feasibility)

**4. Results** (4 pages)
- 4.1 CNN Performance (ResNet-18, MobileNet-V2)
  - Expected: KPU, DPU excel
  - Latency/energy trade-offs
- 4.2 Transformer Performance (ViT-Tiny)
  - **Critical section** - can edge accelerators handle transformers?
  - Attention operation analysis
- 4.3 Detection Performance (YOLOv5-Nano)
  - Real-world navigation task
- 4.4 Energy Analysis
  - Battery life projections
  - Energy per inference
- 4.5 Quantization Impact
  - INT8 vs FP32 across architectures

**5. Analysis** (2 pages)
- 5.1 Why Datacenter Accelerators Fail for Edge
  - Power consumption analysis
  - Cost analysis
- 5.2 Edge Accelerator Trade-offs
  - KPU: Best balance
  - DPU: Good CNN, weak transformer?
  - CGRA: Best energy, weak latency
- 5.3 Architecture Recommendations
  - For robots: KPU or DPU
  - For drones: KPU (strict power budget)
  - For autonomous vehicles: DPU or GPU (more power available)

**6. Related Work** (1 page)
- Datacenter accelerator comparisons (GPU vs TPU)
- Edge accelerator studies (individual)
- Embodied AI systems

**7. Conclusion** (0.5 pages)
- Key findings
- Future work: Multi-accelerator systems, dynamic voltage scaling

**Total**: ~15 pages + references

**Target Venues**:
- **ISCA** (International Symposium on Computer Architecture) - Top tier
- **MICRO** (IEEE/ACM International Symposium on Microarchitecture)
- **ASPLOS** (Architectural Support for Programming Languages and Operating Systems)
- **RAL** (IEEE Robotics and Automation Letters) - More application-focused

---

## 4. Timeline & Milestones

### Week 1: DPU Implementation
- **Day 1**: Research Xilinx Vitis AI specs (frequency, configs, power)
- **Day 2-3**: Implement `dpu_mapper.py` (~350 lines)
- **Day 4**: Test on ResNet-18, MobileNet-V2, ViT-Tiny
- **Day 5**: Document, add to comparison

**Milestone**: DPU mapper complete and validated

### Week 2: CGRA Implementation
- **Day 1-2**: Implement greedy PnR algorithm
- **Day 3-4**: Implement `cgra_mapper.py` (~550 lines)
- **Day 5**: Test on 3 models, analyze reconfiguration overhead

**Milestone**: CGRA mapper complete and validated

### Week 3: Embodied AI Analysis
- **Day 1-2**: Add YOLOv5-Nano to test suite
- **Day 3**: Run complete 6-way comparison across 4 models
- **Day 4**: Generate figures, tables, analysis
- **Day 5**: Document results, prepare summary

**Milestone**: Complete 6-way evaluation on embodied AI workloads

### Week 4: Paper Writing (Optional)
- **Day 1-2**: Write methodology, results sections
- **Day 3-4**: Write intro, background, analysis
- **Day 5**: Polish, generate figures

**Milestone**: Draft paper ready for review

---

## 5. Success Criteria

**Technical Success**:
- [x] Phase 2 complete (GPU, CPU, KPU, TPU) ✅
- [ ] DPU mapper implemented and validated
- [ ] CGRA mapper implemented and validated
- [ ] ViT-Tiny tested on all 6 hardware types
- [ ] Energy analysis complete

**Research Success**:
- [ ] Clear winner identified for embodied AI (predicted: KPU)
- [ ] Transformer (ViT) support analyzed across architectures
- [ ] Performance/energy/flexibility trade-offs quantified
- [ ] Paper draft complete

**Impact Success**:
- [ ] Framework can guide hardware selection for embodied AI
- [ ] Results inform future accelerator design
- [ ] Paper published in top-tier venue (ISCA/MICRO/ASPLOS)

---

## 6. Immediate Next Steps (This Week)

### Step 1: Research Xilinx Vitis AI Specifications (Today)
Need to confirm from Xilinx documentation:
- [ ] AIE clock frequency (target: 1.0-1.3 GHz)
- [ ] DPU configurations available (1280, 2560, 4096 MACs)
- [ ] Supported operations (Conv, Pool, MatMul, Attention?)
- [ ] Power consumption (target: 5-15W)
- [ ] Memory hierarchy (scratchpad sizes)

**Sources**:
- Xilinx Versal ACAP datasheets
- Vitis AI documentation
- Academic papers on Versal AI Engine

### Step 2: Implement DPU Resource Model (Tomorrow)
```python
# Add to hardware_mapper.py

def xilinx_vitis_ai_dpu_resource_model() -> HardwareResourceModel:
    """
    Xilinx Vitis AI DPU (Versal AI Engine based).

    Configuration: 4096 INT8 MACs @ 1 GHz
    Efficiency: 75% of theoretical peak (per user requirement)
    """
    return HardwareResourceModel(
        name="Xilinx-Vitis-AI-DPU",
        hardware_type=HardwareType.DPU,
        compute_units=400,  # AIE tiles

        # TODO: Confirm from Xilinx docs
        peak_flops=...,  # Based on 4096 MACs @ 1 GHz
        peak_bandwidth=...,

        precision_profiles={
            Precision.INT8: PrecisionProfile(
                precision=Precision.INT8,
                peak_ops_per_sec=6.144e12,  # 6.144 TOPS @ 75% eff
                # ...
            ),
        },
    )
```

### Step 3: Begin DPU Mapper Implementation (Day 3)
Following DPU_CGRA_ANALYSIS.md design

### Step 4: Test on ViT-Tiny (Day 4)
Critical test case for embodied AI

---

## 7. Research Questions to Answer

**Primary Questions**:
1. **Which edge accelerator is best for embodied AI?**
   - Hypothesis: KPU (balance of performance, energy, flexibility)

2. **Can edge accelerators handle Vision Transformers?**
   - DPU: Unknown (need to check MatMul support)
   - CGRA: Possible but may need many reconfigurations
   - KPU: Should work (matrix operations supported)

3. **What's the energy efficiency gap between edge and datacenter?**
   - Hypothesis: 10-100× better on edge accelerators
   - But: Datacenter faster in absolute terms

**Secondary Questions**:
4. At what model size do edge accelerators become infeasible?
5. Can multi-accelerator systems help? (KPU + DPU hybrid?)
6. How does batch size affect edge accelerators? (predicted: minimal gain)

---

## 8. Expected Contributions

**To Hardware Architecture Community**:
- First comprehensive edge accelerator comparison for embodied AI
- Methodology for realistic utilization modeling
- Insights on transformer support in edge accelerators

**To Robotics/Embodied AI Community**:
- Clear hardware selection guidance
- Performance/energy trade-off analysis
- Validation that edge accelerators can run modern models (ViT)

**To Open Source Community**:
- Complete modeling framework (~6,000 lines code)
- Reproducible results
- Extensible to new architectures

---

## Summary

**Goal**: Identify the best hardware architecture for embodied AI (robots, drones)

**Approach**:
1. Implement DPU mapper (Xilinx Vitis AI) - Week 1
2. Implement CGRA mapper (Plasticine) - Week 2
3. Test on embodied AI workloads (ViT, YOLO) - Week 3
4. Write research paper - Week 4

**Key Insight**: Reframe from "hardware comparison" to **"embodied AI accelerator analysis"**
- Datacenter (GPU, TPU) = comparison only (not viable for edge)
- Edge (KPU, DPU, CGRA) = primary focus
- CPU = baseline (what's used today)

**Expected Result**: KPU wins for embodied AI (real-time + energy efficient + model coverage)

**Next Immediate Action**: Research Xilinx Vitis AI specifications (AIE frequency, DPU configs, power)

---

**Ready to proceed with DPU implementation?** 🚀
