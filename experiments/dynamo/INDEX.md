# Dynamo Graph Extraction - File Index

This directory contains a complete framework for extracting computational graphs from complex PyTorch models using Dynamo.

## Start Here

1. **New to Dynamo?** → Read [`QUICKSTART.md`](QUICKSTART.md) (5 minutes)
2. **Need detailed comparison?** → Read [`COMPARISON.md`](COMPARISON.md)
3. **Want full documentation?** → Read [`README.md`](README.md)

## Files Overview

### Documentation

| File | Purpose | Read Time |
|------|---------|-----------|
| [`QUICKSTART.md`](QUICKSTART.md) | 5-minute quick start guide | 5 min |
| [`README.md`](README.md) | Complete documentation | 20 min |
| [`COMPARISON.md`](COMPARISON.md) | FX vs Dynamo vs ONNX vs StableHLO | 15 min |
| [`TORCH_COMPILE_EXPLAINED.md`](TORCH_COMPILE_EXPLAINED.md) | torch.compile ↔ Dynamo relationship | 10 min |
| [`INDUCTOR_VALIDATION.md`](INDUCTOR_VALIDATION.md) | Using inductor for validation baseline | 15 min |
| `INDEX.md` (this file) | Navigation guide | 2 min |

### Code Examples

| File | Purpose | Complexity | Key Learning |
|------|---------|------------|--------------|
| [`basic_dynamo_tracing.py`](basic_dynamo_tracing.py) | Basic patterns & examples | ⭐⭐ Beginner | Custom backends, graph inspection |
| [`huggingface_complex_models.py`](huggingface_complex_models.py) | Real-world HF models (DETR, BERT, ViT) | ⭐⭐⭐ Intermediate | Complex model tracing, memory analysis |
| [`trace_yolo.py`](trace_yolo.py) | YOLO object detection models | ⭐⭐⭐ Intermediate | YOLO-specific patterns, architecture analysis |
| [`integrate_with_graphs.py`](integrate_with_graphs.py) | Integration with graphs package | ⭐⭐⭐⭐ Advanced | Dynamo → graphs.ir conversion |
| [`torch_compile_backends.py`](torch_compile_backends.py) | Backend comparison (educational) | ⭐ Reference | torch.compile vs Dynamo relationship |
| [`inductor_validation.py`](inductor_validation.py) | Inductor validation baseline | ⭐⭐⭐ Intermediate | Validation, performance benchmarking |

## Learning Path

### Path 1: Quick Evaluation (30 minutes)
1. Read [`QUICKSTART.md`](QUICKSTART.md)
2. Run `python basic_dynamo_tracing.py simple`
3. Run `python basic_dynamo_tracing.py compare` (FX vs Dynamo)
4. **Decision point**: Is Dynamo right for your use case?

### Path 2: Deep Dive (2 hours)
1. Read [`COMPARISON.md`](COMPARISON.md) - Understand tradeoffs
2. Read [`README.md`](README.md) - Full documentation
3. Run all `basic_dynamo_tracing.py` examples
4. Try `huggingface_complex_models.py` with a model
5. Study `integrate_with_graphs.py` for integration patterns

### Path 3: Integration (1 day)
1. Complete Path 2
2. Analyze `integrate_with_graphs.py` thoroughly
3. Implement conversion layer for your codebase
4. Test with complex models (YOLO, DETR)
5. Validate FLOP counts against fvcore

## Quick Command Reference

```bash
# Basic examples
python experiments/dynamo/basic_dynamo_tracing.py simple
python experiments/dynamo/basic_dynamo_tracing.py control
python experiments/dynamo/basic_dynamo_tracing.py compare

# HuggingFace models
python experiments/dynamo/huggingface_complex_models.py --model detr
python experiments/dynamo/huggingface_complex_models.py --model bert
python experiments/dynamo/huggingface_complex_models.py --model vit

# YOLO models
python experiments/dynamo/trace_yolo.py --model yolov8n.pt
python experiments/dynamo/trace_yolo.py --model yolov8n.pt --analyze
python experiments/dynamo/trace_yolo.py --compare

# Integration demo
python experiments/dynamo/integrate_with_graphs.py --model simple
python experiments/dynamo/integrate_with_graphs.py --model complex
```

## Key Concepts by File

### `basic_dynamo_tracing.py`
- Custom backend creation (`DynamoGraphExtractor`)
- Graph node inspection
- Operation counting
- FX vs Dynamo comparison
- **Use when**: Learning Dynamo basics

### `huggingface_complex_models.py`
- Enhanced extractor with memory analysis
- FLOP counting
- Graph code export
- Complex model handling (transformers)
- **Use when**: Working with production models

### `trace_yolo.py`
- YOLO-specific operation patterns
- Architecture stage detection (backbone/neck/head)
- Multi-variant comparison
- FPN/PAN feature fusion identification
- **Use when**: Analyzing object detection models

### `integrate_with_graphs.py`
- Dynamo graph → graphs.ir conversion
- Hardware mapping integration
- Multi-hardware comparison (H100, CPU, TPU)
- Roofline analysis integration
- **Use when**: Integrating with characterization pipeline

## Decision Trees

### "Which file should I start with?"

```
Are you new to Dynamo?
├─ Yes → Read QUICKSTART.md, then run basic_dynamo_tracing.py
└─ No
   └─ What's your goal?
      ├─ Learn advanced patterns → huggingface_complex_models.py
      ├─ Trace YOLO models → trace_yolo.py
      ├─ Integrate with graphs package → integrate_with_graphs.py
      └─ Understand tradeoffs → COMPARISON.md
```

### "Which graph extraction technology should I use?"

```
What model are you analyzing?
├─ Simple feedforward CNN (e.g., basic ResNet)
│  └─ FX or Dynamo (both work, FX is simpler)
│
├─ Complex model (YOLO, DETR, attention-based)
│  └─ Dynamo (only viable option)
│
├─ Cross-framework (PyTorch + TensorFlow)
│  └─ ONNX (for exchange) + Dynamo (for PyTorch analysis)
│
└─ JAX-native model
   └─ StableHLO (JAX ecosystem)
```

## Integration Checklist

Planning to integrate Dynamo with the graphs package? Use this checklist:

- [ ] Read `COMPARISON.md` to understand why Dynamo
- [ ] Run all examples to understand Dynamo capabilities
- [ ] Study `integrate_with_graphs.py` conversion patterns
- [ ] Identify integration points in existing codebase
- [ ] Implement `DynamoGraphExtractor` custom backend
- [ ] Implement conversion: Dynamo FX graph → `graphs.ir.structures`
- [ ] Test with simple models (MLPs, basic CNNs)
- [ ] Test with complex models (YOLO, DETR)
- [ ] Validate FLOP counts against fvcore
- [ ] Update tests to use Dynamo
- [ ] Update documentation
- [ ] Deploy to production

## Troubleshooting

| Issue | Solution | Where to Look |
|-------|----------|---------------|
| "torch has no attribute 'compile'" | Upgrade PyTorch | `README.md` → Troubleshooting |
| "transformers not installed" | `pip install transformers` | `huggingface_complex_models.py` docstring |
| "Graph breaks detected" | Normal for complex models | `README.md` → Key Concepts |
| "How do I save graphs?" | Use `save_graph_to_file()` | `huggingface_complex_models.py:180` |
| "Integration pattern unclear" | Study examples | `integrate_with_graphs.py` |
| "FX vs Dynamo?" | Read comparison | `COMPARISON.md` |

## Common Patterns

### Pattern 1: Extract and Save
```python
from huggingface_complex_models import trace_model_with_dynamo, save_graph_to_file

extractor = trace_model_with_dynamo(model, example_input)
save_graph_to_file(extractor, "output/graph.py")
```

### Pattern 2: Operation Analysis
```python
from basic_dynamo_tracing import DynamoGraphExtractor

extractor = DynamoGraphExtractor()
compiled = torch.compile(model, backend=extractor)
_ = compiled(input)

# Analyze operations
for node in extractor.graphs[0].nodes:
    if node.op == 'call_function':
        print(f"Op: {node.target.__name__}")
```

### Pattern 3: Memory Estimation
```python
# See huggingface_complex_models.py:_analyze_memory_usage()
total_memory = 0
for node in graph.nodes:
    if 'tensor_meta' in node.meta:
        tm = node.meta['tensor_meta']
        total_memory += calculate_tensor_size(tm)
```

### Pattern 4: Integration with graphs.ir
```python
# See integrate_with_graphs.py:convert_dynamo_graph_to_ir()
from graphs.ir.structures import TensorDescriptor, SubgraphDescriptor

subgraphs = []
for node in graph.nodes:
    # Extract tensor metadata
    tensor_desc = TensorDescriptor(...)
    # Create subgraph
    subgraph = SubgraphDescriptor(...)
    subgraphs.append(subgraph)
```

##  Key Insight

```
  torch.compile = Camera (housing)
      │
      ├─ Dynamo = Sensor (captures image/graph)
      │
      └─ Backend = Processing (what to do with image/graph)
           │
           ├─ "inductor" → Enhance image (optimize graph) ✅ Performance baseline
           ├─ "aot_eager" → Preview only (debug) ✅ Another baseline
           └─ Custom → Save RAW (extract graph) ✅ For analysis!
```

For workload characterization you want the RAW capture (unoptimized graph)!

##  What Each Backend Does

  | Backend              | What It Does                                 | For Analysis?                  |
  |----------------------|----------------------------------------------|--------------------------------|
  | "inductor" (default) | Optimizes graph (fuses ops, generates code)  | ❌ No - changes graph           |
  | "aot_eager"          | Captures graph, executes eagerly (debugging) | ❌ No - just for debugging      |
  | "cudagraphs"         | CUDA-specific optimization                   | ❌ No - GPU-specific            |
  | Custom (yours)       | Extracts graph, analyzes it                  | ✅ YES - this is what you need! |


## FAQ

**Q: Should I use FX or Dynamo?**
A: Use Dynamo for all new development. FX is limited to simple models.

**Q: What about ONNX?**
A: ONNX is good for cross-framework work and deployment, but Dynamo is better for PyTorch analysis.

**Q: Why do I get multiple graph partitions?**
A: Normal for complex models. Dynamo creates "graph breaks" for unsupported operations. Each partition is still useful.

**Q: How do I integrate with the graphs package?**
A: See `integrate_with_graphs.py` for complete example. Key steps: extract graph → convert to IR → run analysis.

**Q: Can Dynamo handle YOLO/DETR?**
A: Yes! That's the main reason to use it. See `trace_yolo.py` and `huggingface_complex_models.py`.

**Q: What's the performance overhead?**
A: Tracing is slower than FX (~500ms vs ~100ms for ResNet18) but still acceptable for offline analysis.

## Next Steps

After working through these examples, you should:

1. **For analysis work**: Start using Dynamo for all complex model tracing
2. **For integration**: Implement the conversion layer shown in `integrate_with_graphs.py`
3. **For production**: Make Dynamo the primary graph extraction mechanism
4. **For learning**: Contribute new examples or improve existing ones

## Contributing

If you add new Dynamo patterns or examples:
1. Add them to the appropriate file (or create a new one)
2. Update this INDEX.md
3. Update README.md if adding new features
4. Add docstrings and comments

## Resources

- [PyTorch Dynamo Documentation](https://pytorch.org/docs/stable/dynamo/)
- [torch.compile Tutorial](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)
- [TorchDynamo Paper](https://arxiv.org/abs/2301.10169)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [Ultralytics YOLO](https://docs.ultralytics.com/)

---

**Last Updated**: 2025-11-07
**Status**: ✅ Complete and ready for use
**Maintainer**: graphs package team
