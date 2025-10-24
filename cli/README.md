# Command-Line Tools

This directory contains command-line utilities for graph characterization, profiling, and model analysis.

## Tools

### `partitioner.py`
Full-featured graph partitioning tool with multiple output formats.

**Usage:**
```bash
# Partition a torchvision model
./cli/partitioner.py --model resnet18 --input-shape 1,3,224,224

# Custom model with detailed output
./cli/partitioner.py --model path/to/model.py --verbose

# Export results to JSON
./cli/partitioner.py --model mobilenet_v2 --output results.json
```

**Features:**
- Supports torchvision models
- Custom model loading
- Multiple output formats (JSON, CSV, text)
- Verbose logging
- Fusion-based partitioning
- Subgraph analysis

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

### `show_fvcore_table.py`
Compare our FLOP estimates against fvcore library.

**Usage:**
```bash
# Compare ResNet-18
./cli/show_fvcore_table.py --model resnet18

# Compare multiple models
./cli/show_fvcore_table.py --models resnet18,mobilenet_v2,efficientnet_b0
```

**Outputs:**
- Side-by-side FLOP comparison
- Accuracy percentages
- Discrepancy analysis

---

### `model_registry_tv2.3.py`
Model registry for torchvision 2.3 compatibility.

**Usage:**
```python
from cli.model_registry_tv2.3 import get_model

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

## Common Usage Patterns

### Quick Model Analysis
```bash
# 1. Discover available models
./cli/discover_models.py --filter resnet

# 2. Profile the model
./cli/profile_graph.py --model resnet18

# 3. Partition into subgraphs
./cli/partitioner.py --model resnet18 --output results.json

# 4. Compare against fvcore
./cli/show_fvcore_table.py --model resnet18
```

### Custom Model Workflow
```bash
# 1. Profile your model
./cli/profile_graph.py --model path/to/model.py --input-shape 1,3,224,224

# 2. Partition and analyze
./cli/partitioner.py --model path/to/model.py --verbose

# 3. Export for further analysis
./cli/partitioner.py --model path/to/model.py --output analysis.json
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
./cli/partitioner.py --model resnet18

# Or set PYTHONPATH
export PYTHONPATH=/path/to/graphs/repo
python cli/partitioner.py --model resnet18
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

## Documentation

See also:
- `../examples/README.md` - Usage demonstrations
- `../validation/README.md` - Validation tests
- `../docs/` - Architecture documentation
