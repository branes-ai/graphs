# New DSP/NPU Mapper Implementations

**Date**: 2025-10-24
**Author**: Claude Code
**Task**: Add CEVA NeuPro, Cadence Tensilica, and Synopsys ARC silicon IP mappers

---

## Summary

Added three new licensable IP core mappers to the DSP mapper module, expanding hardware support for edge AI and SoC integration scenarios.

## New Mappers

### 1. CEVA NeuPro-M NPM11 Neural Processing IP

**Specifications**:
- **Performance**: 20 TOPS INT8 @ 1.25 GHz (peak)
- **Power**: 2W typical TDP
- **Efficiency**: ~10 TOPS/W
- **Precision Support**: INT8 (native), INT16, FP16, INT4 (2× INT8)
- **Memory**: 2-4 MB local SRAM, 50 GB/s bandwidth
- **Use Cases**: Mobile AI, automotive ADAS, IoT, drones, wearables

**Factory Function**: `create_ceva_neupro_npm11_mapper()`

**Resource Model**: `ceva_neupro_npm11_resource_model()`

**Test Results** (ResNet-50 @ INT8):
- Latency: 150.6 ms (6.6 FPS)
- Energy: 4.3 mJ
- Utilization: 29.3% average, 100% peak
- Bottleneck: Mostly compute-bound

---

### 2. Cadence Tensilica Vision Q8 DSP IP (7th Generation)

**Specifications**:
- **Performance**: 3.8 TOPS INT8/INT16 @ 1.0 GHz, 129 GFLOPS FP32
- **Power**: 1W typical TDP
- **Efficiency**: ~3-7 TOPS/W
- **Precision Support**: INT8/INT16 (native, vision-optimized), FP32, FP16
- **Memory**: 512 KB - 2 MB local SRAM, 40 GB/s bandwidth
- **Use Cases**: Automotive ADAS cameras, mobile ISP+AI, surveillance, AR/VR

**Factory Function**: `create_cadence_vision_q8_mapper()`

**Resource Model**: `cadence_vision_q8_resource_model()`

**Test Results** (ResNet-50 @ INT8):
- Latency: 225.3 ms (4.4 FPS)
- Energy: 5.0 mJ
- Utilization: 47.7% average, 100% peak
- Bottleneck: Compute-bound with good utilization

---

### 3. Synopsys ARC EV7x Embedded Vision Processor IP

**Specifications**:
- **Performance**: 35 TOPS INT8 @ 1.0 GHz (4-core configuration)
- **Power**: 5W typical TDP
- **Efficiency**: ~7-10 TOPS/W
- **Precision Support**: INT8 (35 TOPS), INT16 (17.5 TOPS), INT32, FP32 (8.8 GFLOPS)
- **Architecture**: 1-4 VPUs (512-bit vector DSP) + DNN accelerator (880-14,080 MACs)
- **Memory**: 2-8 MB local, 60 GB/s bandwidth
- **Use Cases**: Automotive ADAS L2-3, surveillance, drones, AR/VR

**Factory Function**: `create_synopsys_arc_ev7x_mapper()`

**Resource Model**: `synopsys_arc_ev7x_resource_model()`

**Test Results** (ResNet-50 @ INT8):
- Latency: 364.1 ms (2.7 FPS)
- Energy: 5.2 mJ
- Utilization: 14.7% average, 50% peak
- Bottleneck: Compute-bound with lower utilization

---

## Implementation Details

### Files Modified

1. **`src/graphs/characterize/hardware_mapper.py`**
   - Added `ceva_neupro_npm11_resource_model()` (lines 3977-4152)
   - Added `cadence_vision_q8_resource_model()` (lines 4159-4318)
   - Added `synopsys_arc_ev7x_resource_model()` (lines 4325-4490)
   - Total additions: ~540 lines

2. **`src/graphs/characterize/dsp_mapper.py`**
   - Updated imports to include new resource models
   - Added `create_ceva_neupro_npm11_mapper()` (lines 588-637)
   - Added `create_cadence_vision_q8_mapper()` (lines 644-692)
   - Added `create_synopsys_arc_ev7x_mapper()` (lines 699-751)
   - Updated module docstring
   - Total additions: ~170 lines

3. **`validation/hardware/test_new_dsp_mappers.py`** (new file)
   - Comprehensive test script for all three mappers
   - Tests with ResNet-50 @ INT8 precision
   - Comparative analysis and summary

### Architecture Characteristics

All three mappers follow the DSP architecture pattern:

**Common Features**:
- Heterogeneous compute (vector + tensor/DNN accelerator + scalar)
- Native INT8/INT16 support
- Configurable thermal operating points
- Roofline-based performance modeling
- Energy-aware scheduling

**Key Differences**:
- **CEVA NeuPro**: Highest TOPS/W, optimized for mobile/edge AI
- **Cadence Vision Q8**: Vision-optimized with strong FP32 support
- **Synopsys ARC EV7x**: Highest peak TOPS, automotive-grade, multi-core

### Calibration Status

⚠️ **All three mappers are ESTIMATED** based on published vendor specifications:
- CEVA NeuPro-M: Based on CEVA Product Brief and press releases (2021-2024)
- Cadence Vision Q8: Based on Cadence announcement (2021)
- Synopsys ARC EV7x: Based on Synopsys Product Brief (2019) and EE Times coverage

**Future Work**:
- Empirical benchmarking on actual silicon implementations
- Calibration with real-world SoC integrations
- Refinement of efficiency factors based on measured data

---

## Usage Examples

### Basic Usage

```python
from src.graphs.characterize.dsp_mapper import (
    create_ceva_neupro_npm11_mapper,
    create_cadence_vision_q8_mapper,
    create_synopsys_arc_ev7x_mapper,
)
from src.graphs.characterize.hardware_mapper import Precision

# Create mapper
mapper = create_ceva_neupro_npm11_mapper()

# Map graph to hardware
hw_report = mapper.map_graph(
    fusion_report,
    execution_stages,
    batch_size=1,
    precision=Precision.INT8
)

print(f"Latency: {hw_report.total_latency * 1000:.3f} ms")
print(f"Energy: {hw_report.total_energy * 1000:.3f} mJ")
```

### Comparison Tool

```bash
# Run validation test
python validation/hardware/test_new_dsp_mappers.py
```

---

## Performance Comparison (ResNet-50 @ INT8)

| Hardware | Peak TOPS | Power | Latency (ms) | FPS | Energy (mJ) | Util % |
|----------|-----------|-------|--------------|-----|-------------|--------|
| CEVA NeuPro-M NPM11 | 20 | 2W | 150.6 | 6.6 | 4.3 | 29.3 |
| Cadence Vision Q8 | 3.8 | 1W | 225.3 | 4.4 | 5.0 | 47.7 |
| Synopsys ARC EV7x | 35 | 5W | 364.1 | 2.7 | 5.2 | 14.7 |

**Analysis**:
- **CEVA NeuPro** offers best balance of performance and efficiency
- **Cadence Vision Q8** has highest utilization despite lower peak TOPS
- **Synopsys ARC EV7x** has lower utilization on ResNet-50 due to high unit count (128 units vs 64/32)

---

## Integration with Existing Mappers

The new mappers integrate seamlessly with existing DSP mappers:

### Qualcomm DSPs
- QRB5165 (Hexagon 698): 15 TOPS INT8, robotics

### Texas Instruments C7x DSPs
- TDA4VM: 8 TOPS INT8, automotive ADAS
- TDA4VL: 4 TOPS INT8, entry-level ADAS
- TDA4AL: 8 TOPS INT8, mid-range ADAS
- TDA4VH: 32 TOPS INT8, L3-4 autonomy

### Licensable IP Cores (NEW)
- **CEVA NeuPro-M NPM11**: 20 TOPS INT8, edge AI NPU
- **Cadence Tensilica Vision Q8**: 3.8 TOPS, vision DSP
- **Synopsys ARC EV7x**: 35 TOPS INT8, embedded vision

---

## References

### CEVA NeuPro
- CEVA NeuPro-M Product Brief
- [CEVA Press Release (2021): NeuPro-M Architecture](https://www.prnewswire.com/news-releases/ceva-redefines-high-performance-aiml-processing-for-edge-ai-and-edge-compute-devices-with-its-neupro-m-heterogeneous-and-secure-processor-architecture-301455262.html)
- [Tom's Hardware: CEVA NeuPro Performance](https://www.tomshardware.com/news/ceva-neupro-embedded-ai-chips,36235.html)

### Cadence Tensilica Vision Q8
- [Cadence Product Brief (2021)](https://www.businesswire.com/news/home/20210422005315/en/Cadence-Extends-Popular-Tensilica-Vision-and-AI-DSP-IP-Product-Line-with-New-DSPs-Targeting-High-End-and-Always-On-Applications)
- [AnandTech: Tensilica Vision Q6 DSP](https://www.anandtech.com/show/12633/cadence-announces-tensilica-vision-q6-dsp)
- Cadence DesignWare IP Catalog

### Synopsys ARC EV7x
- [Synopsys Product Page: ARC EV Processors](https://www.synopsys.com/designware-ip/processor-solutions/ev-processors.html)
- [EE Times (2019): ARC EV7x 35 TOPS Performance](https://www.eetimes.com/synopsys-arc-embedded-vision-processors-deliver-35-tops/)
- [Synopsys Press Release (2019)](https://www.prnewswire.com/news-releases/synopsys-new-embedded-vision-processor-ip-delivers-industry-leading-35-tops-performance-for-artificial-intelligence-socs-300918135.html)
- Synopsys DesignWare ARC Processor IP Catalog

---

## Conclusion

Successfully implemented three new DSP/NPU mappers for licensable IP cores, expanding the hardware characterization framework to cover major silicon IP vendors. These mappers enable performance estimation for SoC designs using CEVA, Cadence, and Synopsys IP before silicon tape-out.

**Next Steps**:
- Empirical calibration with real silicon
- Add more IP variants (CEVA NeuPro-Nano, Tensilica Vision P6, ARC NPX6)
- Extend comparison tools to include new mappers
- Document SoC integration considerations
