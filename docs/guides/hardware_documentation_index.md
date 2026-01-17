# Hardware Architecture Documentation

This directory contains comprehensive documentation about the hardware architectures supported by the graphs characterization framework.

## Overview Documents

### [Architecture Taxonomy](architecture_taxonomy.md) 🌟 **START HERE**
**Comprehensive guide to all hardware execution models and programming paradigms**

This is the definitive reference for understanding the fundamental differences between hardware architectures:
- CPU (Stored Program Machine / MIMD)
- GPU (SIMT - Single Instruction Multiple Thread)
- DSP (VLIW with heterogeneous units)
- TPU (Systolic Arrays / Weight-Stationary)
- KPU (Domain Flow / MIMD Spatial)
- DPU (Reconfigurable FPGA Tiles)
- CGRA (Coarse-Grained Reconfigurable Arrays)

**What's inside**:
- Flynn's Taxonomy classification for each architecture
- Execution paradigm explanations (temporal vs spatial)
- Bottleneck analysis and best use cases
- Mapper implementation strategies
- Memory hierarchy comparisons
- Quick reference selection guide

---

## Hardware-Specific Documentation

### [Jetson Specifications](jetson_specifications.md)
Official NVIDIA Jetson hardware specifications including:
- GPU configurations (SM counts, CUDA cores, Tensor Cores)
- Performance metrics (TOPS, TFLOPS)
- Memory specifications
- Covers: Jetson Thor (Blackwell), Jetson Orin (Ampere family)

### [KPU Architecture](../kpu_architecture.md)
Deep dive into Stillwater's Knowledge Processing Unit:
- Stream processing vs weight-stationary dataflow
- Checkerboard floorplan and tile architecture
- Heterogeneous tile strategy (70% INT8, 20% BF16, 10% FP32)
- Scaling strategies (T64/T256/T768)
- Comparison with TPU and GPU architectures

---

## Related Documentation

### Hardware Comparison Guides
- [`docs/edge_ai_categories.md`](../edge_ai_categories.md) - Edge AI deployment categories
- [`docs/datacenter_cpu_comparison.md`](../datacenter_cpu_comparison.md) - Datacenter CPU specifications
- [`docs/dsp_npu_mappers.md`](../dsp_npu_mappers.md) - DSP and NPU architecture details
- [`docs/ip_core_comparison.md`](../ip_core_comparison.md) - Licensable IP core comparison

### Hardware Analysis Sessions
- [`docs/sessions/2025-10-25_gpu_microarchitecture_modeling.md`](../sessions/2025-10-25_gpu_microarchitecture_modeling.md)
- [`docs/sessions/2025-10-24_dsp_mappers_automotive.md`](../sessions/2025-10-24_dsp_mappers_automotive.md)
- [`docs/sessions/2025-10-26_hardware_comparison_and_jetson_fix.md`](../sessions/2025-10-26_hardware_comparison_and_jetson_fix.md)

### Characterization Framework
- [`docs/hardware_characterization_2025-10.md`](../hardware_characterization_2025-10.md) - Overall characterization approach
- [`docs/characterization-architecture.md`](../characterization-architecture.md) - Characterization system architecture

---

## Quick Navigation

**Want to understand...**

| Question | Document |
|----------|----------|
| "What's the difference between CPU, GPU, and TPU execution?" | [Architecture Taxonomy](architecture_taxonomy.md) → Execution Model Comparison |
| "Why does KPU have better utilization at batch=1?" | [Architecture Taxonomy](architecture_taxonomy.md) → KPU section |
| "How do I choose hardware for my workload?" | [Architecture Taxonomy](architecture_taxonomy.md) → Architecture Selection Guide |
| "What are Jetson Orin specifications?" | [Jetson Specifications](jetson_specifications.md) |
| "How does KPU stream processing work?" | [KPU Architecture](../kpu_architecture.md) → Stream Processing |
| "Which DSPs are available for automotive?" | [DSP/NPU Mappers](../dsp_npu_mappers.md) |

---

## For Mapper Developers

If you're implementing a new hardware mapper:

1. **Start with**: [Architecture Taxonomy](architecture_taxonomy.md) → Mapper Implementation Strategies
2. **Review existing mappers**: `src/graphs/hardware/mappers/`
3. **Base class**: `HardwareMapper` in `src/graphs/hardware/resource_model.py`
4. **Common patterns**:
   - Thread-to-hardware mapping
   - Roofline performance model
   - Three-component energy model
   - Precision-aware performance scaling

**Key files**:
```
src/graphs/hardware/
├── resource_model.py          # Base HardwareMapper class
├── mappers/
│   ├── cpu.py                 # CPU mapper (MIMD)
│   ├── gpu.py                 # GPU mapper (SIMT)
│   ├── dsp.py                 # DSP mapper (VLIW)
│   └── accelerators/
│       ├── tpu.py             # TPU mapper (systolic arrays)
│       ├── kpu.py             # KPU mapper (domain flow)
│       ├── dpu.py             # DPU mapper (reconfigurable tiles)
│       └── cgra.py            # CGRA mapper (spatial dataflow)
```

---

## Contributing

When adding new hardware support:

1. Create resource model in `src/graphs/hardware/models/<category>/`
2. Create mapper in `src/graphs/hardware/mappers/` (or `accelerators/` subdirectory)
3. Add specifications document to this directory if warranted
4. Update [Architecture Taxonomy](architecture_taxonomy.md) with new architecture
5. Add tests to `validation/hardware/`

---

## Document Index

| Document | Type | Last Updated | Status |
|----------|------|--------------|--------|
| [Architecture Taxonomy](architecture_taxonomy.md) | Reference | 2025-11-01 | ✅ Current |
| [Jetson Specifications](jetson_specifications.md) | Specifications | 2024-10-26 | ✅ Current |
| [KPU Architecture](../kpu_architecture.md) | Architecture | 2025-10-22 | ✅ Current |

---

**Questions or suggestions?** File an issue or contact the architecture team.
