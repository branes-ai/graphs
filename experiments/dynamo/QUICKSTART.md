# Dynamo Graph Extraction - Quick Start Guide

This is a 5-minute quick start to get you tracing complex models with Dynamo.

## Why Dynamo?

**TL;DR**: FX fails on complex models (YOLO, DETR). Dynamo succeeds.

```python
# ❌ FX fails on control flow
from torch.fx import symbolic_trace
traced = symbolic_trace(yolo_model)  # ERROR!

# ✅ Dynamo handles it gracefully
import torch._dynamo as dynamo
compiled = torch.compile(yolo_model, backend=custom_backend)
```

## Installation

```bash
# PyTorch 2.0+ required
pip install torch>=2.0.0

# For HuggingFace examples
pip install transformers pillow

# For YOLO examples
pip install ultralytics
```

## Basic Usage (30 seconds)

```python
import torch
import torch._dynamo as dynamo

# 1. Define your custom backend (graph extractor)
class GraphExtractor:
    def __init__(self):
        self.graphs = []

    def __call__(self, gm: torch.fx.GraphModule, example_inputs):
        self.graphs.append(gm.graph)
        gm.graph.print_tabular()  # Print graph structure
        return gm.forward  # Return original (no optimization)

# 2. Create model and input
model = torch.nn.Sequential(
    torch.nn.Conv2d(3, 16, 3),
    torch.nn.ReLU(),
)
example_input = torch.randn(1, 3, 224, 224)

# 3. Extract graph
extractor = GraphExtractor()
compiled_model = torch.compile(model, backend=extractor)

with torch.no_grad():
    _ = compiled_model(example_input)  # Triggers tracing

# 4. Use the extracted graph
print(f"Captured {len(extractor.graphs)} graph(s)")
```

## Examples Provided

### 1. Basic Patterns (`basic_dynamo_tracing.py`)
Learn the fundamentals with simple examples.

```bash
python experiments/dynamo/basic_dynamo_tracing.py simple     # Basic CNN
python experiments/dynamo/basic_dynamo_tracing.py control    # Control flow (FX fails)
python experiments/dynamo/basic_dynamo_tracing.py loops      # Dynamic loops
python experiments/dynamo/basic_dynamo_tracing.py compare    # FX vs Dynamo
```

**What you'll learn:**
- Custom backend creation
- Graph node inspection
- Operation counting
- Metadata extraction

### 2. HuggingFace Models (`huggingface_complex_models.py`)
Real-world complex models from HuggingFace.

```bash
# Trace DETR (Detection Transformer)
python experiments/dynamo/huggingface_complex_models.py --model detr

# Trace BERT (NLP Transformer)
python experiments/dynamo/huggingface_complex_models.py --model bert

# Trace ViT (Vision Transformer)
python experiments/dynamo/huggingface_complex_models.py --model vit

# Save extracted graph code
python experiments/dynamo/huggingface_complex_models.py --model detr --save-graph output/detr.py
```

**What you'll learn:**
- Complex model tracing
- Memory analysis
- FLOP estimation
- Graph code export

### 3. YOLO Models (`trace_yolo.py`)
Object detection models (YOLOv8, YOLOv11).

```bash
# Trace YOLOv8 nano
python experiments/dynamo/trace_yolo.py --model yolov8n.pt

# Trace YOLOv8 small
python experiments/dynamo/trace_yolo.py --model yolov8s.pt

# Architecture analysis
python experiments/dynamo/trace_yolo.py --model yolov8n.pt --analyze

# Compare variants
python experiments/dynamo/trace_yolo.py --compare
```

**What you'll learn:**
- YOLO-specific patterns (FPN/PAN, detection heads)
- Architecture stage detection (backbone/neck/head)
- Multi-variant comparison

### 4. Integration with graphs Package (`integrate_with_graphs.py`)
Bridge Dynamo → graphs characterization pipeline.

```bash
# Simple integration demo
python experiments/dynamo/integrate_with_graphs.py --model simple

# Complex model demo
python experiments/dynamo/integrate_with_graphs.py --model complex
```

**What you'll learn:**
- Converting Dynamo graphs to graphs.ir structures
- Running hardware mapping analysis
- Multi-hardware comparison (H100, CPU, TPU)

## Key Concepts

### 1. Graph Partitions (Graph Breaks)
Complex models may be split into multiple partitions:

```python
extractor = GraphExtractor()
compiled_model = torch.compile(model, backend=extractor, fullgraph=False)
_ = compiled_model(input)

print(f"Got {len(extractor.graphs)} partitions")  # May be > 1
```

**This is normal!** Dynamo creates graph breaks for:
- Unsupported operations
- Dynamic control flow
- Python-native operations

### 2. Metadata Propagation
Dynamo automatically propagates tensor shapes/dtypes:

```python
for node in graph.nodes:
    if 'tensor_meta' in node.meta:
        tm = node.meta['tensor_meta']
        print(f"{node.name}: shape={tm.shape}, dtype={tm.dtype}")
```

No need for manual `ShapeProp` like FX!

### 3. Operation Types

| node.op | Meaning | Example |
|---------|---------|---------|
| `placeholder` | Input/parameter | Model inputs, weights |
| `call_function` | PyTorch function | `torch.relu`, `torch.matmul` |
| `call_module` | Module call | `nn.Conv2d`, `nn.Linear` |
| `get_attr` | Get parameter/buffer | Weight access |
| `output` | Return value | Model output |

### 4. Exporting Graphs
Save as executable Python code:

```python
with open('graph.py', 'w') as f:
    f.write(graph_module.code)
```

The generated code can be:
- Inspected manually
- Run standalone
- Compared across versions

## Common Patterns

### Pattern 1: Count Operations
```python
op_counts = {}
for node in graph.nodes:
    if node.op == 'call_function':
        name = node.target.__name__
        op_counts[name] = op_counts.get(name, 0) + 1

print(op_counts)  # {'conv2d': 10, 'relu': 10, ...}
```

### Pattern 2: Estimate Memory
```python
total_memory = 0
for node in graph.nodes:
    if 'tensor_meta' in node.meta:
        tm = node.meta['tensor_meta']
        numel = 1
        for dim in tm.shape:
            numel *= dim
        total_memory += numel * 4  # Assume float32

print(f"Total memory: {total_memory / (1024**2):.2f} MB")
```

### Pattern 3: Identify Bottlenecks
```python
for node in graph.nodes:
    if node.op == 'call_function':
        # Estimate arithmetic intensity
        flops = estimate_flops(node)
        memory = estimate_memory_traffic(node)

        if memory > 0:
            ai = flops / memory
            if ai < 1.0:
                print(f"{node.name} is memory-bound (AI={ai:.2f})")
            else:
                print(f"{node.name} is compute-bound (AI={ai:.2f})")
```

## Next Steps

1. **Start with basic examples**: Run `basic_dynamo_tracing.py` to understand the fundamentals

2. **Try your own model**: Replace the example model with your DNN

3. **Integrate with graphs package**: Use `integrate_with_graphs.py` as a template

4. **Read the full README**: See `experiments/dynamo/README.md` for detailed documentation

## Comparison: FX vs Dynamo vs ONNX

| Feature | FX | Dynamo | ONNX |
|---------|-----|---------|------|
| Control flow | ❌ | ✅ | ⚠️ |
| Dynamic shapes | ❌ | ✅ | ⚠️ |
| Complex models | ❌ | ✅ | ⚠️ |
| PyTorch native | ✅ | ✅ | ❌ |
| Ease of use | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| Best for | Simple models | Analysis | Deployment |

**Recommendation**: Use Dynamo for graph extraction and analysis.

## Troubleshooting

### "Module 'torch' has no attribute 'compile'"
- PyTorch version too old
- Fix: `pip install --upgrade torch>=2.0.0`

### "Graph breaks detected"
- Normal for complex models
- Each partition is still useful
- Use `fullgraph=True` to force single graph (may fail)

### "transformers not installed"
- For HuggingFace examples
- Fix: `pip install transformers pillow`

### "ultralytics not installed"
- For YOLO examples
- Fix: `pip install ultralytics`

## Resources

- **Files in this directory**:
  - `README.md` - Full documentation
  - `basic_dynamo_tracing.py` - Basic patterns
  - `huggingface_complex_models.py` - DETR, BERT, ViT
  - `trace_yolo.py` - YOLO models
  - `integrate_with_graphs.py` - Integration example

- **External resources**:
  - [PyTorch Dynamo Docs](https://pytorch.org/docs/stable/dynamo/)
  - [torch.compile Tutorial](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)
  - [TorchDynamo Paper](https://arxiv.org/abs/2301.10169)

## Questions?

See the full documentation in `README.md` or check the inline comments in the example files.
