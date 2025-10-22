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
