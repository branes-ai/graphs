# Dynamo Graph Extraction Experiments

This directory contains experiments with PyTorch Dynamo for extracting computational graphs from complex neural network models.

## Overview

**Dynamo** (the backend of `torch.compile`) provides a more robust graph extraction mechanism than FX symbolic tracing, particularly for real-world models with:
- Dynamic control flow (if/else, loops)
- Data-dependent operations
- Complex attention mechanisms
- Models that fail with FX tracing (YOLO, DETR, etc.)

## Why Dynamo?

### Comparison: FX vs Dynamo vs ONNX vs StableHLO

| Feature | FX symbolic_trace | Dynamo | ONNX | StableHLO |
|---------|------------------|---------|------|-----------|
| Control flow | ❌ Limited | ✅ Full support | ⚠️ Limited | ⚠️ Limited |
| Dynamic shapes | ❌ No | ✅ Yes | ⚠️ Limited | ✅ Yes |
| PyTorch native | ✅ Yes | ✅ Yes | ❌ No (export) | ❌ No (export) |
| Complex models | ❌ Often fails | ✅ Robust | ⚠️ Lossy | ⚠️ Conversion overhead |
| Graph breaks | ❌ Fails | ✅ Handles gracefully | N/A | N/A |
| Ease of use | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| Best for | Simple models | Analysis & optimization | Deployment | JAX ecosystem |

**Recommendation**: Use **Dynamo** for graph-based analysis and characterization of complex DNNs.

## Files

### 1. `basic_dynamo_tracing.py`
Basic patterns for Dynamo graph extraction with educational examples.

**Examples included:**
- Simple CNN (baseline)
- Model with control flow (where FX fails)
- Model with loops (dynamic iteration)
- FX vs Dynamo comparison

**Usage:**
```bash
# Run all examples
python experiments/dynamo/basic_dynamo_tracing.py

# Run specific example
python experiments/dynamo/basic_dynamo_tracing.py simple
python experiments/dynamo/basic_dynamo_tracing.py control
python experiments/dynamo/basic_dynamo_tracing.py loops
python experiments/dynamo/basic_dynamo_tracing.py compare
```

**Key Components:**
- `DynamoGraphExtractor`: Custom backend that captures graph information
- `trace_model_with_dynamo()`: Main tracing function
- Node-level analysis with metadata inspection
- Operation statistics and counting

### 2. `huggingface_complex_models.py`
Advanced examples using real-world HuggingFace models.

**Models covered:**
- **DETR** (Detection Transformer): Object detection with transformers
- **BERT**: Transformer encoder for NLP
- **ViT** (Vision Transformer): Image classification

**Usage:**
```bash
# Trace DETR
python experiments/dynamo/huggingface_complex_models.py --model detr

# Trace BERT
python experiments/dynamo/huggingface_complex_models.py --model bert

# Trace ViT
python experiments/dynamo/huggingface_complex_models.py --model vit

# Trace all models
python experiments/dynamo/huggingface_complex_models.py --model all

# Save extracted graphs to file
python experiments/dynamo/huggingface_complex_models.py --model detr --save-graph output/detr_graph.py
```

**Requirements:**
```bash
pip install transformers pillow requests
```

**Features:**
- Enhanced `DynamoGraphExtractor` with memory analysis
- FLOP counting and operation profiling
- Graph code export for inspection
- Handles HuggingFace model input formats (dicts)

## Key Concepts

### 1. Graph Partitioning
Dynamo may split complex models into multiple graph partitions ("graph breaks") when it encounters operations it cannot fully trace. This is **normal and expected** for complex models.

Example output:
```
Graph Partition 1
==================
... attention operations ...

Graph Partition 2
==================
... FFN operations ...
```

### 2. Custom Backend
The `DynamoGraphExtractor` is a custom backend for `torch.compile` that intercepts the graph without applying optimizations:

```python
extractor = DynamoGraphExtractor()
compiled_model = torch.compile(model, backend=extractor)
output = compiled_model(input)  # Triggers tracing
```

### 3. Metadata Extraction
Dynamo propagates tensor metadata (shape, dtype) through the graph, which is essential for:
- Memory estimation
- FLOP counting
- Hardware mapping

Access metadata via:
```python
for node in graph.nodes:
    if 'tensor_meta' in node.meta:
        shape = node.meta['tensor_meta'].shape
        dtype = node.meta['tensor_meta'].dtype
```

### 4. Operation Analysis
The extractor counts and categorizes operations:
- **call_function**: PyTorch functional operations (e.g., `torch.relu`, `torch.matmul`)
- **call_module**: Module calls (e.g., `nn.Conv2d`, `nn.Linear`)
- **placeholder**: Input arguments
- **output**: Return values
- **get_attr**: Parameter/buffer access

## Integration with graphs Package

### Adapting for characterization Pipeline

To integrate Dynamo graphs with the existing characterization pipeline:

```python
from graphs.ir.structures import TensorDescriptor, SubgraphDescriptor
from graphs.transform.partitioning import FusionPartitioner

# 1. Extract graph with Dynamo
extractor = trace_model_with_dynamo(model, example_input)
graph_module = extractor.graph_modules[0]

# 2. Convert to IR structures
subgraphs = []
for node in graph_module.graph.nodes:
    if node.op == 'call_function':
        # Extract tensor metadata
        if 'tensor_meta' in node.meta:
            tm = node.meta['tensor_meta']
            tensor_desc = TensorDescriptor(
                shape=tuple(tm.shape),
                dtype=tm.dtype,
                name=node.name
            )

        # Create subgraph descriptor
        subgraph = SubgraphDescriptor(
            operations=[node.target.__name__],
            input_tensors=[...],
            output_tensors=[...],
            # ... etc
        )
        subgraphs.append(subgraph)

# 3. Run existing analysis pipeline
from graphs.analysis.unified_analyzer import UnifiedAnalyzer
analyzer = UnifiedAnalyzer()
# ... continue with existing workflow
```

### Key Differences from FX Workflow

| Aspect | FX Workflow | Dynamo Workflow |
|--------|-------------|-----------------|
| Tracing | `symbolic_trace(model)` | `torch.compile(model, backend=extractor)` |
| Execution | Trace only | Must execute to trace |
| Graph breaks | Fatal error | Handled gracefully |
| Complex models | Limited | Full support |
| Metadata | Manual ShapeProp | Automatic |

## Advanced Topics

### 1. Handling Graph Breaks
If a model has graph breaks, you'll get multiple `GraphModule` objects. To analyze:

```python
extractor = trace_model_with_dynamo(model, input)

for i, gm in enumerate(extractor.graph_modules):
    print(f"Partition {i+1}:")
    # Analyze each partition separately
    analyze_partition(gm)
```

### 2. Exporting Graphs
Save extracted graphs as executable Python code:

```python
save_graph_to_file(extractor, "output/model_graph.py")
```

This generates a standalone `.py` file with the graph implementation that can be:
- Inspected manually
- Used for debugging
- Compared across model versions

### 3. Memory Timeline Analysis
Track tensor lifetime across graph execution:

```python
def analyze_memory_timeline(graph: torch.fx.Graph):
    live_tensors = {}
    timeline = []

    for i, node in enumerate(graph.nodes):
        # Tensor becomes live
        if 'tensor_meta' in node.meta:
            live_tensors[node.name] = node.meta['tensor_meta']

        # Check when tensors are last used
        for arg in node.args:
            if isinstance(arg, torch.fx.Node):
                if is_last_use(arg, graph, node):
                    del live_tensors[arg.name]

        # Record memory at this point
        total_memory = sum(get_tensor_size(t) for t in live_tensors.values())
        timeline.append((i, node.name, total_memory))

    return timeline
```

## Troubleshooting

### Common Issues

1. **"transformers not installed"**
   ```bash
   pip install transformers pillow requests
   ```

2. **"Graph breaks detected"**
   - This is normal for complex models
   - Each partition is still useful for analysis
   - Use `fullgraph=True` to force single graph (may fail)

3. **"Out of memory"**
   - Use smaller models or batch size
   - Run on GPU if available
   - Disable memory analysis: `analyze_memory=False`

4. **"Module has no attribute 'compile'"**
   - PyTorch version too old
   - Upgrade: `pip install --upgrade torch>=2.0.0`

## Future Directions

1. **Integration with graphs.ir**
   - Automatic conversion from Dynamo graphs to `SubgraphDescriptor`
   - Support for multi-partition models

2. **YOLO Support**
   - Add YOLO-specific tracing examples
   - Handle YOLOv8/YOLOv11 architecture

3. **Comparative Analysis**
   - Side-by-side FX vs Dynamo analysis
   - ONNX export and comparison
   - StableHLO export experiments

4. **Optimization Patterns**
   - Fusion opportunities from Dynamo graphs
   - Automatic partitioning for hardware mapping

## References

- [PyTorch Dynamo Documentation](https://pytorch.org/docs/stable/dynamo/index.html)
- [torch.compile Tutorial](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [TorchDynamo Paper](https://arxiv.org/abs/2301.10169)
