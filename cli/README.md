# Command-Line Tools

This directory contains command-line utilities for graph characterization, profiling, and model analysis.

---

## 📚 Detailed Documentation

Comprehensive how-to guides for each tool:


### Discovery Tools: Profiling & Partitioning
- **[discover_models.py](docs/discover_models.md)** - Find FX-traceable models (140+ models)
- **[profile_graph.py](docs/profile_graph.md)** - Hardware-independent graph profiling
- **[profile_graph_with_fvcore](docs/profile_graph.md)** - PyTorch Engineering fvcore-based graph profiling
- **[graph_explorer.py](docs/graph_explorer.md)** - Explore FX graphs interactively (discovery → summary → visualization)
- **[partition_analyzer.py](docs/partition_analyzer.md)** - Analyze and compare partitioning strategies
- **[list_hardware_mappers.py](docs/list_hardware_mappers.md)** - Discover available hardware (35+ models)

### Core Analysis Tools
- **[compare_models.py](docs/compare_models.md)** - Compare models across hardware targets
- **[analyze_graph_mapping.py](docs/analyze_graph_mapping.md)** - Complete guide to hardware mapping analysis

### Specialized Comparisons
- **[Comparison Tools](docs/comparison_tools.md)** - Automotive, Edge, IP Cores, Datacenter

*💡 Tip: Start with the detailed guides above for step-by-step instructions, examples, and troubleshooting.*

---

## Common Conventions

### Unified Range Selection

CLI tools that visualize graphs (`graph_explorer.py`, `partition_analyzer.py`) use unified node addressing:

**Node Numbering:**
- Node numbers are **1-based** (matching the display output)
- Ranges are **inclusive** on both ends
- Example: `--start 5 --end 10` shows nodes 5, 6, 7, 8, 9, and 10

**Range Selection Methods:**
1. **Explicit Range**: `--start N --end M` (show nodes N through M)
2. **Context View**: `--around N --context K` (show K nodes before/after N)
3. **Max Nodes**: `--max-nodes N` (show first N nodes from start)

**Examples:**
```bash
# Show nodes 5-10 (inclusive, 6 nodes total)
./cli/graph_explorer.py --model resnet18 --start 5 --end 10

# Show 10 nodes around node 35 (nodes 25-45)
./cli/graph_explorer.py --model resnet18 --around 35 --context 10

# Show first 20 nodes (nodes 1-20)
./cli/partition_analyzer.py --model resnet18 --strategy fusion --visualize --max-nodes 20
```

---

## Tools

### `partition_analyzer.py`
Analyze and compare different partitioning strategies to quantify fusion benefits.

**Usage:**
```bash
# Compare all strategies
./cli/partition_analyzer.py --model resnet18 --strategy all --compare

# Visualize with specific range
./cli/partition_analyzer.py --model resnet18 --strategy fusion --visualize --start 5 --end 20

# Investigate around specific node
./cli/partition_analyzer.py --model mobilenet_v2 --strategy fusion --visualize --around 15 --context 5
```

**Features:**
- Compare partitioning strategies (unfused vs fusion)
- Visualize partitioned graphs with range selection
- Unified range selection (--start/--end, --around/--context, --max-nodes)
- **Node addressing**: 1-based, inclusive ranges matching display output
- Quantify fusion benefits (subgraph reduction, memory savings)
- Analyze fusion patterns and bottlenecks

---

### `graph_explorer.py`
Explore FX computational graphs interactively with three progressive modes.

**Three Modes:**
```bash
# 1. Discover models (no arguments)
./cli/graph_explorer.py

# 2. Get model summary (model only)
./cli/graph_explorer.py --model resnet18

# 3. Visualize sections (model + range)
./cli/graph_explorer.py --model resnet18 --max-nodes 20
./cli/graph_explorer.py --model resnet18 --start 5 --end 20
./cli/graph_explorer.py --model resnet18 --around 35 --context 10
```

**Features:**
- Progressive disclosure: models → summary → visualization
- Prevents accidental output floods (large models have 300+ nodes)
- Comprehensive summary statistics (FLOPs, bottlenecks, operation distribution)
- Side-by-side visualization of FX graph and partitions
- Unified range selection (--start/--end, --around/--context, --max-nodes)
- **Node addressing**: 1-based, inclusive ranges matching display output
- Export to file (--output)
- Shows operation details, arithmetic intensity, partition reasoning

---

### `profile_graph.py`
Profile PyTorch models to understand computational characteristics.

**Usage:**
```bash
# Profile ResNet-18
./cli/profile_graph.py --model resnet18

# Profile with custom input shape
./cli/profile_graph.py --model efficientnet_b0 --input-shape 1,3,240,240

# Output profiling data
./cli/profile_graph.py --model mobilenet_v2 --output profile.json
```

**Outputs:**
- FLOPs per layer
- Memory per layer
- Arithmetic intensity
- Bottleneck analysis (compute vs memory bound)
- Critical path identification

---

### `profile_graph_with_fvcore.py`
Compare our FLOP estimates against fvcore library.

**Usage:**
```bash
# Compare ResNet-18
./cli/profile_graph_with_fvcore.py --model resnet18

# Compare multiple models
./cli/profile_graph_with_fvcore.py --models resnet18,mobilenet_v2,efficientnet_b0
```

**Outputs:**
- Side-by-side FLOP comparison
- Accuracy percentages
- Discrepancy analysis

---

### `discover_models.py`
Discover and list available models from torchvision and custom sources.

**Usage:**
```bash
# List all torchvision models
./cli/discover_models.py

# Filter by pattern
./cli/discover_models.py --filter resnet

# Show model details
./cli/discover_models.py --model resnet18 --details
```

**Outputs:**
- Model names
- Parameter counts
- Input shapes
- Model families (ResNet, MobileNet, etc.)

---

### `model_registry_tv2dot7.py`
Model registry for torchvision 2.7 compatibility.

**Usage:**
```python
from cli.model_registry_tv2dot7 import get_model

model = get_model('resnet18')
```

---

## Hardware Comparison Tools

### `compare_automotive_adas.py`
Compare AI accelerators for automotive Advanced Driver Assistance Systems (ADAS Level 2-3).

**Usage:**
```bash
# Run full automotive comparison
python cli/compare_automotive_adas.py
```

**Features:**
- **Category 1**: Front Camera ADAS (10-15W) - Lane Keep, ACC, TSR
- **Category 2**: Multi-Camera ADAS (15-25W) - Surround View, Parking
- **Hardware**: TI TDA4VM, Jetson Orin Nano/AGX, KPU-T256
- **Models**: ResNet-50, FCN lane segmentation, YOLOv5 automotive
- **Metrics**: 30 FPS requirement, <100ms latency, ASIL-D certification

**Output:**
```
CATEGORY 1 RESULTS: Front Camera ADAS (10-15W)
--------------------------------------------------
Hardware              Power  TDP   Latency  FPS    FPS/W   30FPS?  <100ms?  Util%
TI-TDA4VM-C7x         10W    10.0  110.76   9.0    0.90    ✗       ✗        47.7
Jetson-Orin-Nano      15W    15.0  5.45     183.5  12.23   ✓       ✓        97.9
```

---

### `compare_edge_ai_platforms.py`
Compare edge AI accelerators for embodied AI and robotics platforms.

**Usage:**
```bash
# Run edge AI comparison
python cli/compare_edge_ai_platforms.py
```

**Features:**
- **Category 1**: Computer Vision / Low Power (≤10W) - Drones, robots, cameras
- **Category 2**: Transformers / Higher Power (≤50W) - Autonomous vehicles, edge servers
- **Hardware**: Hailo-8/10H, Jetson Orin, KPU-T64/T256, QRB5165, TI TDA4VM
- **Models**: ResNet-50, DeepLabV3+, ViT-Base
- **Metrics**: Latency, throughput, power efficiency (FPS/W), TOPS/W

**Output:**
```
CATEGORY 1: Computer Vision / Low Power (≤10W)
--------------------------------------------------
Hardware              Peak TOPS  FPS/W   Best for
Hailo-8 @ 2.5W        26         10.4    Edge cameras
Jetson-Orin-Nano      40         12.2    Robots
QRB5165-Hexagon698    15         2.1     Mobile robots
```

---

### `compare_i7_12700k_mappers.py`
Compare CPU mapper performance for Intel i7-12700K (standard vs large L3 cache).

**Usage:**
```bash
# Run CPU mapper comparison
python cli/compare_i7_12700k_mappers.py
```

**Features:**
- Standard i7-12700K (25 MB L3)
- Large cache variant (30 MB L3)
- Models: ResNet-50, DeepLabV3+, ViT-Base
- Metrics: Latency, throughput, cache efficiency

---

### `compare_ip_cores.py`
Compare licensable AI/compute IP cores for custom SoC integration.

**Usage:**
```bash
# Run IP core comparison
python cli/compare_ip_cores.py
```

**Features:**
- **Traditional Architectures** (Stored-Program Extensions):
  * CEVA NeuPro-M NPM11: 20 TOPS INT8 @ 2W (DSP + NPU)
  * Cadence Tensilica Vision Q8: 3.8 TOPS INT8 @ 1W (Vision DSP)
  * Synopsys ARC EV7x: 35 TOPS INT8 @ 5W (CPU + VPU + DNN)
  * ARM Mali-G78 MP20: 1.94 TFLOPS FP32 @ 5W (GPU)

- **Dataflow Architectures** (AI-Native):
  * KPU-T64: 6.9 TOPS INT8 @ 6W (64-tile dataflow)
  * KPU-T256: 33.8 TOPS INT8 @ 30W (256-tile dataflow)

- **Models**: ResNet-50, DeepLabV3+, ViT-Base
- **Metrics**: Peak TOPS, latency, FPS/W, architecture comparison

**Output:**
```
ALL IP CORES - COMPREHENSIVE RESULTS
-------------------------------------------------------
IP Core                Vendor      Type              Power  Latency    FPS     FPS/W   Util%
CEVA NeuPro-M NPM11    CEVA        DSP+NPU IP        2.0    150.57     6.6     3.32    29.3
Cadence Vision Q8      Cadence     Vision DSP IP     1.0    225.30     4.4     4.44    47.7
Synopsys ARC EV7x      Synopsys    CPU+VPU+DNN IP    5.0    364.06     2.7     0.55    14.7
ARM Mali-G78 MP20      ARM         GPU IP            5.0    1221.83    0.8     0.16    99.2
KPU-T64                KPU         Dataflow NPU IP   6.0    4.19       238.8   39.79   98.8
KPU-T256               KPU         Dataflow NPU IP   30.0   1.12       893.2   29.77   90.9
```

**Key Insight:**
KPU dataflow architecture achieves superior efficiency through AI-native design,
not just higher power. Traditional IPs extend stored-program machines, while KPU
is purpose-built for AI workloads from the ground up.

**Typical Use Cases:**
- Mobile flagship: CEVA NeuPro, ARM Mali-G78
- Automotive ADAS: Synopsys ARC EV7x (traditional), KPU-T64/T256 (dataflow)
- Edge AI / Embodied AI: KPU-T64/T256
- Edge servers: KPU-T256
- Base station servers: KPU-T768 (larger variant)

---

### `compare_datacenter_cpus.py`
Compare ARM and x86 datacenter server processors for AI inference workloads.

**Usage:**
```bash
# Run datacenter CPU comparison
python cli/compare_datacenter_cpus.py
```

**Features:**
- **Ampere AmpereOne 192-core**: ARM v8.6+ (5nm TSMC)
  * 192 cores, 22.1 TOPS INT8, 332.8 GB/s memory
  * Best for cloud-native microservices

- **Intel Xeon Platinum 8490H**: x86 Sapphire Rapids (10nm Intel 7)
  * 60 cores, 88.7 TOPS INT8 (AMX), 307 GB/s memory
  * Best for CNN inference (4-10× faster with AMX)

- **AMD EPYC 9654**: x86 Genoa (5nm TSMC)
  * 96 cores, 7.4 TOPS INT8, 460.8 GB/s memory
  * Best for Transformer inference (highest bandwidth)

**Models Tested:**
- ResNet-50 (CNN): Intel Xeon wins (1144 FPS vs 236 FPS Ampere, 217 FPS AMD)
- DeepLabV3+ (Segmentation): Intel Xeon wins (118 FPS vs 13.5 FPS Ampere, 11.7 FPS AMD)
- ViT-Base (Transformer): AMD EPYC wins (878 FPS vs 654 FPS Ampere, 606 FPS Intel)

**Key Insights:**
- **Intel AMX** dominates CNN workloads (4-10× faster)
- **AMD's high bandwidth** (460 GB/s) excels at Transformers
- **Ampere's 192 cores** best for general-purpose compute, not AI

**Output:**
```
DATACENTER CPU COMPARISON RESULTS
============================================================================
ResNet-50
----------------------------------------------------------------------------
CPU                            Cores    TDP      Latency      FPS        FPS/W
Ampere AmpereOne 192-core      192      283      4.24         235.8      0.83
Intel Xeon Platinum 8490H      60       350      0.87         1143.6     3.27  ← Winner
AMD EPYC 9654                  96       360      4.61         216.8      0.60
```

**Documentation**: See `docs/DATACENTER_CPU_COMPARISON.md` for comprehensive analysis

---

## Common Usage Patterns

### Quick Model Analysis
```bash
# 1. Discover available models
./cli/discover_models.py --filter resnet

# 2. Profile the model
./cli/profile_graph.py --model resnet18

# 3. Partition into subgraphs
./cli/partition_analyzer.py --model resnet18 --output results.json

# 4. Compare against fvcore
./cli/show_fvcore_table.py --model resnet18
```

### Custom Model Workflow
```bash
# 1. Profile your model
./cli/profile_graph.py --model path/to/model.py --input-shape 1,3,224,224

# 2. Partition and analyze
./cli/partition_analyzer.py --model path/to/model.py --verbose

# 3. Export for further analysis
./cli/partition_analyzer.py --model path/to/model.py --output analysis.json
```

## Output Formats

### JSON Format
```json
{
  "model": "resnet18",
  "total_flops": 3.6e9,
  "total_memory": 44.6e6,
  "subgraphs": [
    {
      "id": 0,
      "operations": ["conv2d", "batchnorm2d", "relu"],
      "flops": 1.2e8,
      "memory": 2.1e6
    },
    ...
  ]
}
```

### CSV Format
```csv
subgraph_id,operations,flops,memory,bottleneck
0,"conv2d+bn+relu",1.2e8,2.1e6,compute
1,"conv2d+bn+relu",3.7e7,1.5e6,memory
...
```

### Text Format
```
Model: resnet18
Total FLOPs: 3.60 G
Total Memory: 44.6 MB
Subgraphs: 32

Subgraph 0: conv2d+bn+relu
  FLOPs: 120 M
  Memory: 2.1 MB
  Bottleneck: compute-bound (AI=57)
```

## Requirements

All tools require:
- Python 3.8+
- PyTorch
- torchvision

Optional:
- fvcore (for `show_fvcore_table.py`)

Install:
```bash
pip install torch torchvision fvcore
```

## Environment Setup

Tools use the repo root as the working directory:
```bash
# Run from repo root
./cli/partition_analyzer.py --model resnet18

# Or set PYTHONPATH
export PYTHONPATH=/path/to/graphs/repo
python cli/partition_analyzer.py --model resnet18
```

## Troubleshooting

**Import errors:**
- Run from repo root directory
- Check PYTHONPATH includes repo root
- Verify `src/graphs/` package structure

**Model not found:**
- Use `./cli/discover_models.py` to list available models
- Check torchvision version (some models require v0.13+)
- For custom models, provide absolute path

**FLOP mismatch with fvcore:**
- Different counting methodologies
- Our counts include operations fvcore may skip
- ±10% variance is expected and acceptable

## Adding New Tools

1. Create `tool_name.py` in `cli/`
2. Add shebang and make executable: `chmod +x cli/tool_name.py`
3. Include argparse for CLI arguments
4. Add tool to this README
5. Add usage examples

Template:
```python
#!/usr/bin/env python
"""Tool description"""

import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.graphs.characterize.<module> import <Component>


def main():
    parser = argparse.ArgumentParser(description="Tool description")
    parser.add_argument('--model', required=True, help="Model name")
    args = parser.parse_args()

    # Tool logic here
    ...


if __name__ == '__main__':
    main()
```

---

## Quick Reference

### Common Workflows

**1. Discover and Profile a Model**
```bash
# Find available models
python3 cli/discover_models.py

# Profile the model
python3 cli/profile_graph.py --model resnet50

# Analyze hardware mapping
python3 cli/analyze_graph_mapping.py --model resnet50 --hardware H100
```

**2. Compare Hardware Options**
```bash
# List available hardware
python3 cli/list_hardware_mappers.py

# Compare multiple hardware targets
python3 cli/analyze_graph_mapping.py --model resnet50 \
  --compare "H100,Jetson-Orin-AGX,KPU-T256"
```

**3. Evaluate Edge Deployment**
```bash
# Quick edge platform comparison
python3 cli/compare_edge_ai_platforms.py

# Detailed edge hardware analysis
python3 cli/analyze_graph_mapping.py --model mobilenet_v2 \
  --hardware Jetson-Orin-Nano --thermal-profile 7W
```

**4. Specialized Comparisons**
```bash
# Automotive ADAS platforms
python3 cli/compare_automotive_adas.py

# Datacenter CPUs
python3 cli/compare_datacenter_cpus.py

# IP cores for SoC integration
python3 cli/compare_ip_cores.py
```

### Tool Selection Guide

| Goal | Tool |
|------|------|
| Find available models | `discover_models.py` |
| Profile model (HW-independent) | `profile_graph.py` |
| Find available hardware | `list_hardware_mappers.py` |
| Analyze single HW target | `analyze_graph_mapping.py --hardware` |
| Compare multiple HW targets | `analyze_graph_mapping.py --compare` |
| Compare models on same HW | `compare_models.py` |
| Automotive deployment | `compare_automotive_adas.py` |
| Edge deployment | `compare_edge_ai_platforms.py` |
| Datacenter CPUs | `compare_datacenter_cpus.py` |

---

## Documentation

See also:
- `cli/docs/` - **Detailed how-to guides for each tool**
- `../examples/README.md` - Usage demonstrations
- `../validation/README.md` - Validation tests
- `../docs/` - Architecture documentation
