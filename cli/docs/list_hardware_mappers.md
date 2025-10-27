# How to Use: list_hardware_mappers.py

## Overview

`list_hardware_mappers.py` is a hardware discovery tool that lists all available hardware mappers in the package with comprehensive specifications, categories, and comparisons.

**Key Capabilities:**
- Discover all hardware mappers (35+ models)
- View specifications (FLOPS, bandwidth, TDP, etc.)
- Filter by category (CPU, GPU, TPU, KPU, DSP, etc.)
- Export to JSON format
- Compare peak performance across hardware

**Target Users:**
- New users learning available hardware
- Engineers planning analyses
- System architects comparing options

---

## Installation

**Requirements:**
```bash
pip install torch  # (for package imports)
```

**Verify Installation:**
```bash
python3 cli/list_hardware_mappers.py --help
```

---

## Basic Usage

### List All Hardware

```bash
python3 cli/list_hardware_mappers.py
```

**Output:** Comprehensive table with all 35+ hardware models

---

### Filter by Category

```bash
python3 cli/list_hardware_mappers.py --category gpu
```

**Available Categories:**
- `cpu` - Intel Xeon, AMD EPYC, Ampere AmpereOne
- `gpu` - NVIDIA datacenter and edge GPUs
- `tpu` - Google TPU v4, Coral Edge TPU
- `kpu` - Stillwater KPU-T64/T256/T768
- `dsp` - Qualcomm, TI DSPs
- `dpu` - Xilinx Vitis AI
- `cgra` - CGRA accelerators
- `accelerators` - All fixed-function accelerators (TPU, KPU, DPU, CGRA)

---

### JSON Export

```bash
python3 cli/list_hardware_mappers.py --format json > hardware_specs.json
```

**Use Case:** Import into analysis scripts, dashboards, or databases

---

## Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--category` | str | (all) | Filter by hardware category |
| `--format` | str | table | Output format: table or json |
| `--sort-by` | str | name | Sort by: name, flops, bandwidth, power |
| `--deployment` | str | (all) | Filter by deployment: datacenter, edge, mobile, automotive |

---

## Output Format

### Table View (Default)

```
================================================================================
HARDWARE MAPPERS CATALOG
================================================================================

DATACENTER CPUs (8 models)
--------------------------------------------------------------------------------
Name                        Cores   FP32      INT8      BW        TDP    Use Cases
                                    GFLOPS    GOPS      (GB/s)    (W)
--------------------------------------------------------------------------------
Intel Xeon Platinum 8490H   60      3840      88700     307       350    Cloud inference
Intel Xeon Platinum 8592+   64      4096      90880     320       350    Emerald Rapids
Intel Granite Rapids        128     8192      163840    512       500    Next-gen 2024-2025
AMD EPYC 9654              96      7372      7372      460.8     360    Genoa Cloud
AMD EPYC 9754              128     8192      8192      460.8     360    Bergamo Cloud
AMD EPYC Turin             128     10240     10240     614.4     500    Next-gen 2024
Ampere AmpereOne 192       192     11059     22118     332.8     350    ARM Cloud
Ampere AmpereOne 128       128     7372      14745     332.8     250    ARM Edge Server

DATACENTER GPUs (6 models)
--------------------------------------------------------------------------------
Name                        SMs     FP32      INT8      BW        TDP    Use Cases
                                    TFLOPS    TOPS      (GB/s)    (W)
--------------------------------------------------------------------------------
NVIDIA H100 SXM5           132     67.0      1979.0    3350      700    ML Training/Inference
NVIDIA H100 PCIe           114     51.0      1513.0    2000      350    Datacenter Inference
NVIDIA A100 SXM4           108     19.5      624.0     2000      400    ML Training/Inference
NVIDIA A100 PCIe           80      19.5      624.0     1555      250    Inference
NVIDIA V100                80      14.0      112.0     900       300    Legacy Training
NVIDIA T4                  40      8.1       130.0     320       70     Inference

EDGE GPUs (2 models)
--------------------------------------------------------------------------------
Name                        SMs     FP32      INT8      BW        TDP    Use Cases
                                    GFLOPS    GOPS      (GB/s)    (W)
--------------------------------------------------------------------------------
Jetson AGX Orin            16      2662      10650     204.8     60     Autonomous Robots
Jetson Orin Nano           8       1331      5325      68        15     Drones, Cameras
Jetson Thor                20      3328      13312     256       60     Next-gen 2025

TPU ACCELERATORS (2 models)
--------------------------------------------------------------------------------
Name                        Tiles   FP32      INT8      BW        TDP    Use Cases
                                    TFLOPS    TOPS      (GB/s)    (W)
--------------------------------------------------------------------------------
Google TPU v4              2       44.0      176.0     1600      350    Datacenter
Google Coral Edge TPU      1       N/A       4.0       12.8      2      Edge Inference

KPU ACCELERATORS (3 models)
--------------------------------------------------------------------------------
Name                        Tiles   FP32      INT8      BW        TDP    Use Cases
                                    GFLOPS    GOPS      (GB/s)    (W)
--------------------------------------------------------------------------------
Stillwater KPU-T64         64      1749      6989      102.4     6      Edge AI
Stillwater KPU-T256        256     8597      45824     409.6     30     Edge Servers
Stillwater KPU-T768        768     25791     137472    1228.8    100    Datacenter

DSP PROCESSORS (2 models)
--------------------------------------------------------------------------------
Name                        VUs     FP32      INT8      BW        TDP    Use Cases
                                    GFLOPS    GOPS      (GB/s)    (W)
--------------------------------------------------------------------------------
Qualcomm QRB5165           4       107       858       34.1      15     Mobile Robotics
TI TDA4VM                  1       160       1280      17.1      10     Automotive ADAS

DPU/FPGA ACCELERATORS (1 model)
--------------------------------------------------------------------------------
Name                        PEs     FP32      INT8      BW        TDP    Use Cases
                                    GFLOPS    GOPS      (GB/s)    (W)
--------------------------------------------------------------------------------
Xilinx Vitis AI DPU        2048    N/A       4095      68.3      10     FPGA Inference

CGRA ACCELERATORS (1 model)
--------------------------------------------------------------------------------
Name                        PEs     FP32      INT8      BW        TDP    Use Cases
                                    GFLOPS    GOPS      (GB/s)    (W)
--------------------------------------------------------------------------------
Plasticine V2              8192    4096      16384     512       50     Research

================================================================================
SUMMARY
================================================================================
Total Hardware Models: 35
  - Datacenter CPUs: 8
  - Consumer CPUs: 2
  - Datacenter GPUs: 6
  - Edge GPUs: 3
  - TPU Accelerators: 2
  - KPU Accelerators: 3
  - DSP Processors: 2
  - DPU Accelerators: 1
  - CGRA Accelerators: 1

Peak Performance Range:
  - INT8: 4.0 TOPS (Coral TPU) to 1979 TOPS (H100)
  - FP32: 107 GFLOPS (QRB5165) to 67 TFLOPS (H100)
  - Memory BW: 12.8 GB/s (Coral) to 3350 GB/s (H100)

Power Range:
  - Min: 2W (Coral Edge TPU)
  - Max: 700W (H100 SXM5)
```

---

### JSON Format

```json
{
  "hardware_mappers": [
    {
      "name": "Intel Xeon Platinum 8490H",
      "category": "CPU",
      "deployment": "Datacenter",
      "manufacturer": "Intel",
      "compute_units": 60,
      "peak_flops_fp32": 3840.0,
      "peak_flops_int8": 88700.0,
      "memory_bandwidth": 307.0,
      "power_tdp": 350.0,
      "thermal_profiles": ["350W", "250W"],
      "use_cases": ["Cloud inference", "High-throughput servers"],
      "factory_function": "create_intel_xeon_platinum_8490h_mapper",
      "hardware_type": "programmable_isa"
    },
    ...
  ],
  "summary": {
    "total_models": 35,
    "categories": {
      "CPU": 10,
      "GPU": 9,
      "TPU": 2,
      "KPU": 3,
      "DSP": 2,
      "DPU": 1,
      "CGRA": 1
    }
  }
}
```

---

## Common Usage Examples

### Example 1: Discover Edge Hardware

```bash
python3 cli/list_hardware_mappers.py --deployment edge
```

**Output:** Jetson Orin Nano/AGX, KPU-T64, QRB5165, TI TDA4VM

**Use Case:** Planning edge AI deployment

---

### Example 2: Compare GPU Options

```bash
python3 cli/list_hardware_mappers.py --category gpu
```

**Output:** All NVIDIA datacenter and edge GPUs

**Use Case:** Select GPU for datacenter or edge deployment

---

### Example 3: Find High-Bandwidth Hardware

```bash
python3 cli/list_hardware_mappers.py --sort-by bandwidth
```

**Output:** Hardware sorted by memory bandwidth (H100 SXM5 tops at 3.35 TB/s)

**Use Case:** Memory-bound workload optimization

---

### Example 4: Export for Analysis

```bash
python3 cli/list_hardware_mappers.py --format json > specs.json
python3 my_analysis.py --specs specs.json
```

**Use Case:** Integrate hardware specs into custom analysis pipelines

---

### Example 5: Automotive ADAS Options

```bash
python3 cli/list_hardware_mappers.py --deployment automotive
```

**Output:** TI TDA4VM, Jetson Orin series, automotive-grade accelerators

**Use Case:** Planning automotive AI deployments

---

## Hardware Categories Explained

### Datacenter CPUs

**Characteristics:**
- High core count (60-192 cores)
- Large memory bandwidth (300-600 GB/s)
- High TDP (250-500W)

**Best For:**
- General-purpose inference
- Mixed workloads
- Legacy model deployment

**Top Options:**
- **Intel Xeon (AMX)**: Best for CNNs (4-10× speedup)
- **AMD EPYC**: Best for Transformers (highest bandwidth)
- **Ampere AmpereOne**: Best for cloud-native workloads

---

### Datacenter GPUs

**Characteristics:**
- Many SMs (40-132)
- Massive compute (8-67 TFLOPS FP32)
- High bandwidth (320-3350 GB/s)
- High power (70-700W)

**Best For:**
- High-throughput inference
- Real-time serving
- Batch processing

**Top Options:**
- **H100**: Fastest (67 TFLOPS FP32, 1979 TOPS INT8)
- **A100**: Balanced performance/cost
- **T4**: Most efficient (12.5 FPS/W)

---

### Edge GPUs

**Characteristics:**
- Fewer SMs (4-20)
- Lower power (7-60W)
- Integrated memory

**Best For:**
- Autonomous robots
- Drones
- Smart cameras
- Edge servers

**Top Options:**
- **Jetson AGX Orin**: High performance (60W max)
- **Jetson Orin Nano**: Balanced (7-15W)
- **Jetson Thor**: Next-gen 2025

---

### TPU Accelerators

**Characteristics:**
- Fixed systolic arrays
- Matrix operation focus
- Predictable performance

**Best For:**
- Vision Transformers
- Large matrix operations
- Batch inference

**Options:**
- **TPU v4**: Datacenter (350W, 176 TOPS INT8)
- **Coral Edge TPU**: Ultra-low-power (2W, 4 TOPS INT8)

---

### KPU Accelerators (Stillwater)

**Characteristics:**
- Heterogeneous tile architecture
- Exceptional energy efficiency (40+ FPS/W)
- Scales from 6W to 100W

**Best For:**
- CNN inference
- Edge AI with battery constraints
- High-parallelism workloads

**Options:**
- **KPU-T64**: Ultra-low-power edge (6W)
- **KPU-T256**: Edge servers (30W)
- **KPU-T768**: Datacenter (100W)

---

### DSP Processors

**Characteristics:**
- Integrated in mobile SoCs
- Vector processing units
- Low power (2-15W)

**Best For:**
- Mobile devices
- Automotive ADAS
- Signal processing + AI hybrid

**Options:**
- **Qualcomm QRB5165**: High-end mobile (15W)
- **TI TDA4VM**: Automotive-grade (10W)

---

## Interpreting Specifications

### Peak FLOPs

**FP32 GFLOPS:**
- Standard floating-point performance
- Baseline for comparison
- Used for most training

**INT8 GOPS (or TOPS):**
- Quantized integer performance
- 4-16× higher than FP32 (architecture dependent)
- Typical for inference

**Rule of Thumb:**
- INT8/FP32 ratio indicates accelerator efficiency
- GPUs: 16× (Tensor Cores)
- CPUs: 2-4× (AMX, AVX-VNNI)
- Accelerators: 2-4× (fixed-function)

---

### Memory Bandwidth

**Importance:**
- Critical for memory-bound operations (pooling, normalization)
- Transformers need high bandwidth
- CNNs less sensitive (more compute-bound)

**Ranges:**
- **Edge**: 12-200 GB/s
- **Midrange**: 200-1000 GB/s
- **High-end**: 1000-3350 GB/s (HBM3)

**Rule of Thumb:**
- Need ~10× model size in GB/s for real-time inference
- ResNet-50 (100 MB) → need 1 GB/s minimum

---

### TDP (Thermal Design Power)

**Interpretation:**
- Maximum sustained power consumption
- Does NOT include idle power
- Actual power varies with workload

**Deployment Constraints:**
- **Edge/Battery**: <15W
- **Edge Server**: 15-60W
- **Datacenter**: 70-700W

**Efficiency Metric:** TOPS/W (higher is better)
- **Best**: 40+ TOPS/W (KPU-T64, Coral TPU)
- **Good**: 10-40 TOPS/W (Jetson, T4)
- **Moderate**: 1-10 TOPS/W (Datacenter GPUs/CPUs)

---

## Advanced Usage

### Script Integration

Use in Python scripts:

```python
import json
import subprocess

# Get hardware specs
result = subprocess.run(
    ['python3', 'cli/list_hardware_mappers.py', '--format', 'json'],
    capture_output=True, text=True
)
specs = json.loads(result.stdout)

# Filter by criteria
edge_gpus = [
    hw for hw in specs['hardware_mappers']
    if hw['category'] == 'GPU' and hw['deployment'] == 'Edge'
]

for gpu in edge_gpus:
    print(f"{gpu['name']}: {gpu['power_tdp']}W, {gpu['peak_flops_int8']} GOPS")
```

---

### Custom Filtering

Combine filters:

```bash
# Edge GPUs only
python3 cli/list_hardware_mappers.py \
  --category gpu --deployment edge

# High-bandwidth accelerators
python3 cli/list_hardware_mappers.py \
  --category accelerators --sort-by bandwidth
```

---

## Factory Functions

Each hardware entry includes a `factory_function` that can be imported:

```python
from graphs.hardware.mappers.gpu import create_h100_mapper

# Create mapper instance
mapper = create_h100_mapper()

# Use mapper for analysis
latency, power = mapper.estimate_latency_and_power(subgraph, precision)
```

**Common Factory Functions:**

**CPUs:**
- `create_intel_xeon_platinum_8490h_mapper()`
- `create_amd_epyc_9654_mapper()`
- `create_ampere_ampereone_192_mapper()`

**GPUs:**
- `create_h100_mapper()`
- `create_a100_mapper()`
- `create_jetson_orin_agx_mapper(thermal_profile='30W')`
- `create_jetson_orin_nano_mapper(thermal_profile='7W')`

**Accelerators:**
- `create_tpu_v4_mapper()`
- `create_kpu_t64_mapper(thermal_profile='6W')`
- `create_kpu_t256_mapper(thermal_profile='30W')`

---

## Troubleshooting

### Import Errors

**Error:** `ModuleNotFoundError: No module named 'torch'`

**Solution:**
```bash
pip install torch
```

### Missing Hardware

**Question:** "Why isn't hardware X listed?"

**Answer:** Hardware must have:
1. Resource model in `src/graphs/hardware/models/`
2. Mapper in `src/graphs/hardware/mappers/`
3. Factory function in `list_hardware_mappers.py`

**To Add New Hardware:** See `docs/ADDING_NEW_HARDWARE.md` (TBD)

---

## Related Tools

| Tool | Purpose |
|------|---------|
| `analyze_graph_mapping.py` | Use discovered hardware for analysis |
| `compare_models.py` | Compare models across hardware |
| `discover_models.py` | Find available models |

---

## Further Reading

- **Hardware Specifications**: `docs/hardware/jetson_specifications.md`
- **Architecture Guide**: `CLAUDE.md`
- **Comparison Tools**: `cli/docs/analyze_graph_mapping.md`

---

## Contact & Feedback

Report issues or request features at the project repository.
