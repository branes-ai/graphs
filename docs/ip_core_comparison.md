# IP Core Comparison Tool

**Date**: 2025-10-24
**Task**: Create comprehensive IP core comparison tool for SoC integration
**Tool**: `cli/compare_ip_cores.py`

---

## Summary

Created a new CLI tool that compares licensable AI/compute IP cores for custom SoC integration, enabling hardware architects to evaluate IP licensing options before tape-out.

## New IP Core: ARM Mali-G78 MP20

Added ARM Mali-G78 MP20 GPU IP mapper:

**Specifications**:
- **Performance**: 1.94 TFLOPS FP32 @ 848 MHz (~97 GFLOPS per core)
- **Architecture**: 20 shader cores, 2nd gen Valhall, unified shader (graphics+compute)
- **Power**: 5W typical TDP
- **Precision**: FP32 (native), FP16 (2× FP32), INT8 (not optimized)
- **Use Cases**: Mobile gaming, computational photography, light AI inference
- **Real-world**: Used in Google Tensor (Pixel 6/6 Pro)

**Factory Function**: `create_arm_mali_g78_mp20_mapper()`
**Resource Model**: `arm_mali_g78_mp20_resource_model()`

---

## IP Cores Comparison Tool

### Purpose

Compare 6 licensable IP cores across 2 categories for SoC integration decisions:

**Category 1: Neural Processing IP (≤5W)**
- CEVA NeuPro-M NPM11: 20 TOPS INT8 @ 2W
- Cadence Tensilica Vision Q8: 3.8 TOPS INT8 @ 1W
- Synopsys ARC EV7x: 35 TOPS INT8 @ 5W
- ARM Mali-G78 MP20: 1.94 TFLOPS FP32 @ 5W

**Category 2: High-Performance IP (≤30W)**
- KPU-T64: 6.9 TOPS INT8 @ 6W
- KPU-T256: 33.8 TOPS INT8 @ 30W

### Key Features

1. **SoC Integration Focus**
   - Process node recommendations (5nm/7nm/16nm)
   - Typical use case mapping
   - Licensing considerations
   - Integration timeline estimates

2. **Performance Metrics**
   - Peak TOPS/TFLOPS
   - Latency and throughput (FPS)
   - Power efficiency (FPS/W, TOPS/W)
   - Energy per inference
   - Utilization analysis

3. **Test Models**
   - ResNet-50: Vision backbone
   - DeepLabV3+: Semantic segmentation
   - ViT-Base: Vision Transformer

---

## Benchmark Results (ResNet-50 @ INT8)

### Category 1: Neural Processing IP

| IP Core | Vendor | Power | Latency | FPS | FPS/W | Utilization |
|---------|--------|-------|---------|-----|-------|-------------|
| **CEVA NeuPro-M NPM11** | CEVA | 2W | 150.6 ms | 6.6 | 3.32 | 29.3% |
| **Cadence Vision Q8** | Cadence | 1W | 225.3 ms | 4.4 | **4.44** ⭐ | 47.7% |
| **Synopsys ARC EV7x** | Synopsys | 5W | 364.1 ms | 2.7 | 0.55 | 14.7% |
| **ARM Mali-G78 MP20** | ARM | 5W | 1221.8 ms | 0.8 | 0.16 | 99.2% |

**Key Insights**:
- **Best FPS/W**: Cadence Vision Q8 (4.44 FPS/W) - Lowest power, vision-optimized
- **Best Latency**: CEVA NeuPro-M (150.6 ms) - Highest TOPS/W (10 TOPS/W)
- **Highest Peak TOPS**: Synopsys ARC EV7x (35 TOPS) - Automotive-grade
- **Graphics GPU**: ARM Mali-G78 slower for AI workloads (graphics-optimized)

### Category 2: High-Performance IP

| IP Core | Vendor | Power | Latency | FPS | FPS/W | Utilization |
|---------|--------|-------|---------|-----|-------|-------------|
| **KPU-T64** | KPU | 6W | 4.2 ms | 238.8 | **39.79** ⭐ | 98.8% |
| **KPU-T256** | KPU | 30W | **1.1 ms** ⭐ | 893.2 | 29.77 | 90.9% |

**Key Insights**:
- **Best FPS/W**: KPU-T64 (39.79 FPS/W) - Best efficiency
- **Best Latency**: KPU-T256 (1.1 ms) - Highest throughput

---

## SoC Integration Recommendations

### Use Case Mapping

| Use Case | Recommended IP | Process Node | Rationale |
|----------|----------------|--------------|-----------|
| **Mobile Flagship** | CEVA NeuPro @ 7nm | 5nm/7nm | Highest TOPS/W, mobile-optimized |
| **Mobile Camera ISP** | Cadence Vision Q8 | 7nm | Vision pipeline integration, 1W |
| **Automotive ADAS** | Synopsys ARC EV7x | 16nm/7nm | Automotive-grade, 35 TOPS |
| **Mobile Gaming** | ARM Mali-G78 MP20 | 7nm/5nm | Graphics+compute hybrid |
| **Edge Server** | KPU-T256 | 16nm | Highest performance, 33.8 TOPS |

### Process Node Guidance

- **5nm**: Latest mobile SoCs (Apple A-series, Snapdragon 8 Gen)
  - Best: CEVA NeuPro, ARM Mali-G78
  - Power: Lowest, highest cost

- **7nm**: Mainstream mobile/automotive (Google Tensor, Exynos)
  - Best: CEVA NeuPro, Cadence Vision Q8, Synopsys ARC EV7x
  - Power: Good balance, moderate cost

- **16nm**: Cost-effective automotive/IoT
  - Best: Synopsys ARC EV7x, KPU-T64/T256
  - Power: Higher, lowest cost

### Licensing Considerations

1. **Upfront Licensing Fees**
   - One-time fee per IP core license
   - Varies by vendor ($100K - $10M+)
   - Volume discounts available

2. **Royalty Models**
   - Per-chip royalty (% of chip price)
   - Per-unit royalty (fixed per chip)
   - Royalty caps and thresholds

3. **Integration Timeline**
   - **6-12 months**: SoC integration and tape-out
   - **3-6 months**: Verification and validation
   - **Additional**: Automotive certification (if required)

4. **Support and Ecosystem**
   - Compiler and toolchain
   - Reference designs
   - Technical support
   - Training and documentation

---

## Files Modified/Created

### New Files

1. **`src/graphs/characterize/hardware_mapper.py`**
   - Added `arm_mali_g78_mp20_resource_model()` (lines 4497-4668)
   - +172 lines

2. **`src/graphs/characterize/gpu_mapper.py`**
   - Added `create_arm_mali_g78_mp20_mapper()` (lines 359-414)
   - +56 lines

3. **`cli/compare_ip_cores.py`** (new file)
   - Comprehensive IP core comparison tool
   - +570 lines
   - Categories: Neural Processing IP (≤5W), High-Performance IP (≤30W)
   - Executive summary with SoC integration guidance

4. **`docs/IP_CORE_COMPARISON_TOOL.md`** (this file)
   - Documentation of new tool and ARM Mali mapper

### Modified Files

1. **`cli/README.md`**
   - Added documentation for `compare_ip_cores.py`
   - Usage examples and output samples
   - SoC integration guidance summary

---

## Usage

### Basic Usage

```bash
# Run full IP core comparison
python cli/compare_ip_cores.py
```

### Output Structure

1. **Category 1 Results**: Neural Processing IP (≤5W)
   - ResNet-50, DeepLabV3+, ViT-Base benchmarks
   - Performance table with latency, FPS, FPS/W

2. **Category 2 Results**: High-Performance IP (≤30W)
   - Same models, higher performance IP cores
   - KPU-T64 and KPU-T256 comparison

3. **Executive Summary**
   - Best by metric analysis
   - Vendor comparison
   - SoC integration recommendations
   - Licensing considerations

---

## Classification: IP Cores vs. Platforms

### IP Cores (Licensable)
✅ **CEVA NeuPro-M NPM11**
✅ **Cadence Tensilica Vision Q8**
✅ **Synopsys ARC EV7x**
✅ **ARM Mali-G78 MP20**
✅ **KPU-T64** (can be licensed)
✅ **KPU-T256** (can be licensed)

**Characteristics**:
- Licensable for SoC integration
- Configurable (cores, frequency, cache)
- Requires customer tape-out
- Flexible process node
- Customer owns the silicon

### Complete Platforms (Not IP Cores)

❌ **TI TDA4VM** - Complete automotive SoC
❌ **Qualcomm QRB5165** - Complete robotics platform
❌ **NVIDIA Jetson** - Complete edge AI platform
❌ **Hailo-8/10H** - Complete AI accelerator chip

**Characteristics**:
- Fixed silicon product
- Vendor manufactures and sells
- Fixed configuration
- Pre-integrated ecosystem
- Customer buys the chip

---

## Performance Analysis

### Best Overall (ResNet-50 @ INT8)

**Power Efficiency**: Cadence Vision Q8
- 4.44 FPS/W
- Only 1W power
- Best for always-on applications

**Absolute Performance**: KPU-T256
- 1.1 ms latency
- 893 FPS throughput
- Best for edge servers

**Balanced**: CEVA NeuPro-M NPM11
- 10 TOPS/W efficiency
- 150 ms latency
- Best for mobile AI

**Automotive**: Synopsys ARC EV7x
- 35 TOPS peak
- Automotive-grade architecture
- Best for ADAS

**Graphics+AI**: ARM Mali-G78 MP20
- 1.94 TFLOPS FP32
- Mobile gaming optimized
- Use with dedicated NPU for AI

---

## Future Work

1. **Additional IP Cores**
   - ARM Ethos NPU series (Ethos-N78, Ethos-U55)
   - CEVA NeuPro-Nano (TinyML)
   - Cadence Tensilica Vision P6 (lower tier)
   - Synopsys ARC NPX6 (3,500 TOPS, 2024)

2. **More Process Nodes**
   - 3nm analysis for next-gen mobile
   - 28nm for cost-sensitive IoT
   - Power/performance scaling curves

3. **Multi-Core Configurations**
   - 2×, 4×, 8× IP core clusters
   - Heterogeneous configurations
   - Multi-die chiplet architectures

4. **Cost Analysis**
   - Licensing fee estimates
   - Per-chip cost at volume
   - NRE (non-recurring engineering) costs
   - Total cost of ownership (TCO)

---

## References

### ARM Mali-G78
- [ARM Mali-G78 Product Page](https://www.arm.com/products/silicon-ip-multimedia/gpu/mali-g78)
- ARM Mali-G78 Product Brief (2020)
- Google Tensor SoC specifications
- AnandTech: "Arm Announces The Mali-G78 GPU: Evolution to 24 Cores"

### IP Core Comparison
- CEVA NeuPro-M Product Brief
- Cadence Tensilica Vision Q8 announcement (2021)
- Synopsys ARC EV7x Product Brief (2019)
- Industry licensing models and integration timelines

---

## Conclusion

Successfully created a comprehensive IP core comparison tool that enables hardware architects to:

1. **Evaluate**: Compare 6 licensable IP cores across key metrics
2. **Decide**: Make informed SoC integration decisions
3. **Optimize**: Select best IP for specific use cases
4. **Budget**: Understand power, performance, and cost trade-offs

The tool provides actionable guidance for SoC design, from mobile flagship chips to automotive ADAS platforms, with realistic performance estimates and integration considerations.

**Next Steps**: Use this tool during the architecture phase of custom SoC design to select optimal IP licensing mix before committing to tape-out.
