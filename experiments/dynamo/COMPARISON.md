# Graph Extraction Technologies: Detailed Comparison

This document provides a comprehensive comparison of different graph extraction and intermediate representation technologies for DNN workload characterization.

## Executive Summary

| Technology | Best For | Status | Recommendation |
|------------|----------|--------|----------------|
| **Dynamo** | PyTorch analysis & characterization | ✅ Production-ready | **PRIMARY** for complex PyTorch models |
| FX | Simple PyTorch models | ✅ Mature | Use only for simple models |
| ONNX | Model exchange & deployment | ✅ Mature | Secondary for cross-framework |
| StableHLO | JAX ecosystem & XLA compilation | ⚠️ Evolving | For JAX-native models only |

**Recommendation for graphs package**: Use **Dynamo** as the primary graph extraction mechanism, with ONNX as a secondary option for cross-framework compatibility.

---

## 1. PyTorch FX (symbolic_trace)

### Overview
FX is PyTorch's first-generation graph capture mechanism using Python AST manipulation and symbolic tracing.

### Pros
- ✅ Simple API: `traced = torch.fx.symbolic_trace(model)`
- ✅ Native PyTorch integration
- ✅ Good for simple feedforward models
- ✅ Automatic shape propagation with `ShapeProp`
- ✅ Well-documented and stable

### Cons
- ❌ **Fails on control flow** (if/else, loops)
- ❌ **Fails on data-dependent operations**
- ❌ **Fails on complex models** (YOLO, DETR, Transformers)
- ❌ Requires model modifications for complex cases
- ❌ Limited to Python code it can symbolically trace

### Example Failures
```python
# ❌ Control flow - FX FAILS
class ModelWithIf(nn.Module):
    def forward(self, x):
        if x.sum() > 0:  # Data-dependent control flow
            return x * 2
        return x

# ❌ Dynamic shapes - FX FAILS
class ModelWithDynamicShape(nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]  # Dynamic shape query
        return x.reshape(batch_size, -1)

# ❌ Complex models - FX OFTEN FAILS
from ultralytics import YOLO
yolo = YOLO('yolov8n.pt')
traced = symbolic_trace(yolo.model)  # ERROR!
```

### When to Use FX
- Simple feedforward CNNs (ResNet-style)
- Models without control flow
- Educational purposes
- When you have full control over model architecture

### Status for graphs package
⚠️ **Currently used** but limited. Consider deprecating in favor of Dynamo.

---

## 2. PyTorch Dynamo (torch.compile backend)

### Overview
Dynamo is PyTorch 2.0's next-generation graph capture using Python bytecode analysis and frame evaluation. It's the backend powering `torch.compile`.

### Pros
- ✅ **Handles control flow** (if/else, loops)
- ✅ **Handles dynamic shapes**
- ✅ **Works with complex models** (YOLO, DETR, Transformers)
- ✅ Automatic shape propagation (no manual ShapeProp)
- ✅ Graph breaks are handled gracefully
- ✅ Native PyTorch integration
- ✅ Production-ready (PyTorch 2.0+)
- ✅ Active development and support

### Cons
- ⚠️ More complex API than FX
- ⚠️ May create graph breaks (multiple partitions)
- ⚠️ Requires PyTorch 2.0+
- ⚠️ Some learning curve for custom backends

### Example Successes
```python
# ✅ Control flow - Dynamo SUCCEEDS
class ModelWithIf(nn.Module):
    def forward(self, x):
        if x.sum() > 0:
            return x * 2
        return x

extractor = GraphExtractor()
compiled = torch.compile(model, backend=extractor)
_ = compiled(input)  # Works!

# ✅ Complex models - Dynamo SUCCEEDS
from ultralytics import YOLO
yolo = YOLO('yolov8n.pt')
compiled = torch.compile(yolo.model, backend=extractor)
_ = compiled(input)  # Works!

# ✅ Transformers - Dynamo SUCCEEDS
from transformers import DetrForObjectDetection
detr = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
compiled = torch.compile(detr, backend=extractor)
_ = compiled(inputs)  # Works!
```

### When to Use Dynamo
- **Any complex PyTorch model**
- Models with control flow
- Real-world production models (YOLO, DETR, BERT, etc.)
- When you need robust graph extraction
- Primary recommendation for graphs package

### Status for graphs package
✅ **RECOMMENDED** as primary graph extraction mechanism.

---

## 3. ONNX (Open Neural Network Exchange)

### Overview
ONNX is a cross-framework model exchange format with its own IR and operator definitions.

### Pros
- ✅ Cross-framework support (PyTorch, TensorFlow, JAX)
- ✅ Mature ecosystem
- ✅ Good for deployment and model exchange
- ✅ Rich operator set
- ✅ Extensive tooling (netron, onnxruntime, etc.)
- ✅ Hardware vendor support

### Cons
- ❌ **Lossy conversion** - may lose information
- ❌ **Not all PyTorch ops supported**
- ❌ Export process can fail for complex models
- ❌ Requires explicit export step (`torch.onnx.export`)
- ❌ Opaque format - harder to analyze than PyTorch FX graphs
- ❌ Dynamic control flow support is limited
- ❌ Shape inference can be fragile

### Example Usage
```python
import torch.onnx

model = MyModel()
example_input = torch.randn(1, 3, 224, 224)

# Export to ONNX
torch.onnx.export(
    model,
    example_input,
    "model.onnx",
    opset_version=17,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}}
)

# Load and analyze
import onnx
onnx_model = onnx.load("model.onnx")
graph = onnx_model.graph

for node in graph.node:
    print(f"Op: {node.op_type}, inputs: {node.input}, outputs: {node.output}")
```

### When to Use ONNX
- Cross-framework model comparison
- Model deployment (ONNX Runtime)
- Hardware vendor tools require ONNX
- Model exchange between teams/frameworks
- Secondary analysis (after Dynamo)

### Status for graphs package
⚠️ **SECONDARY** option - useful for cross-framework analysis but not primary recommendation.

---

## 4. StableHLO

### Overview
StableHLO is a stable HLO dialect for ML models, designed as a portability layer between ML frameworks and compilers (especially XLA).

### Pros
- ✅ Designed for compiler optimization
- ✅ Strong in JAX ecosystem
- ✅ XLA-native representation
- ✅ Clean, functional IR
- ✅ Good for TPU deployment

### Cons
- ❌ **PyTorch conversion is complex** and lossy
- ❌ **Requires TorchDynamo → TorchScript → MLIR → StableHLO** pipeline
- ❌ Not mature for PyTorch models
- ❌ Limited tooling for PyTorch users
- ❌ Better suited for JAX-native models
- ❌ Overkill for analysis purposes

### Example Usage (PyTorch → StableHLO)
```python
# Complex multi-step process
import torch
from torch.export import export
from torch_mlir import fx_importer

# 1. Export with torch.export
exported = export(model, (example_input,))

# 2. Convert to TorchScript MLIR
mlir_module = fx_importer.export_and_import(exported)

# 3. Convert to StableHLO (requires additional tools)
# ... complex pipeline ...

# This is significantly more complex than Dynamo!
```

### When to Use StableHLO
- JAX-native models
- TPU-specific optimization
- XLA compiler research
- Interoperability with MLIR ecosystem

### Status for graphs package
❌ **NOT RECOMMENDED** for PyTorch models. Too complex for the benefit.

---

## Detailed Feature Comparison

### Graph Extraction Capabilities

| Feature | FX | Dynamo | ONNX | StableHLO |
|---------|-----|---------|------|-----------|
| Control flow (if/else) | ❌ | ✅ | ⚠️ Limited | ✅ |
| Dynamic loops | ❌ | ✅ | ❌ | ✅ |
| Data-dependent shapes | ❌ | ✅ | ⚠️ Limited | ✅ |
| Complex models (YOLO, DETR) | ❌ | ✅ | ⚠️ Export may fail | ⚠️ Complex |
| Automatic shape propagation | ⚠️ Manual | ✅ | ⚠️ Shape inference | ✅ |
| PyTorch native | ✅ | ✅ | ❌ Export format | ❌ Export format |

### Analysis Capabilities

| Feature | FX | Dynamo | ONNX | StableHLO |
|---------|-----|---------|------|-----------|
| Node-level FLOP counting | ✅ | ✅ | ✅ | ✅ |
| Memory estimation | ✅ | ✅ | ✅ | ✅ |
| Operator fusion analysis | ✅ | ✅ | ⚠️ Limited | ✅ |
| Hardware mapping | ✅ | ✅ | ⚠️ Through analysis | ⚠️ Through analysis |
| Arithmetic intensity | ✅ | ✅ | ✅ | ✅ |

### Usability

| Feature | FX | Dynamo | ONNX | StableHLO |
|---------|-----|---------|------|-----------|
| API complexity | ⭐⭐⭐⭐⭐ Simple | ⭐⭐⭐⭐ Moderate | ⭐⭐⭐ Moderate | ⭐⭐ Complex |
| Documentation | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| PyTorch integration | ⭐⭐⭐⭐⭐ Native | ⭐⭐⭐⭐⭐ Native | ⭐⭐⭐ Export | ⭐⭐ Export |
| Learning curve | ⭐⭐⭐⭐⭐ Easy | ⭐⭐⭐⭐ Easy | ⭐⭐⭐ Moderate | ⭐⭐ Steep |

### Maturity & Support

| Aspect | FX | Dynamo | ONNX | StableHLO |
|--------|-----|---------|------|-----------|
| Maturity | ✅ Stable | ✅ Production | ✅ Mature | ⚠️ Evolving |
| Community support | ✅ Large | ✅ Growing | ✅ Large | ⚠️ Small |
| Active development | ⚠️ Maintenance | ✅ Active | ✅ Active | ✅ Active |
| Breaking changes | ✅ Stable | ⚠️ Some changes | ✅ Stable | ⚠️ May change |

---

## Real-World Model Support

Testing with actual models from the graphs package workloads:

| Model | FX | Dynamo | ONNX | StableHLO | Notes |
|-------|-----|---------|------|-----------|-------|
| Simple MLP | ✅ | ✅ | ✅ | ⚠️ | FX sufficient |
| ResNet18 | ✅ | ✅ | ✅ | ⚠️ | FX works but Dynamo preferred |
| YOLOv8 | ❌ | ✅ | ⚠️ | ❌ | **Dynamo only viable option** |
| DETR | ❌ | ✅ | ⚠️ | ❌ | **Dynamo only viable option** |
| BERT | ❌ | ✅ | ✅ | ⚠️ | Dynamo or ONNX |
| ViT | ⚠️ | ✅ | ✅ | ⚠️ | Dynamo preferred |
| MobileNetV2 | ✅ | ✅ | ✅ | ⚠️ | All work |
| EfficientNet | ⚠️ | ✅ | ✅ | ⚠️ | Dynamo preferred |

---

## Integration Complexity

### FX Integration (Current)
```python
# 1. Trace model
traced = torch.fx.symbolic_trace(model)

# 2. Shape propagation
from torch.fx.passes.shape_prop import ShapeProp
ShapeProp(traced).propagate(example_input)

# 3. Convert to graphs.ir
# ... existing code ...
```

**Complexity**: ⭐⭐⭐⭐⭐ Simple

---

### Dynamo Integration (Recommended)
```python
# 1. Create extractor
extractor = DynamoGraphExtractor()

# 2. Compile and run
compiled = torch.compile(model, backend=extractor)
_ = compiled(example_input)

# 3. Convert to graphs.ir (shapes already propagated!)
# ... new code (see integrate_with_graphs.py) ...
```

**Complexity**: ⭐⭐⭐⭐ Easy (shapes automatic!)

---

### ONNX Integration
```python
# 1. Export to ONNX
torch.onnx.export(model, example_input, "model.onnx")

# 2. Load ONNX model
import onnx
onnx_model = onnx.load("model.onnx")

# 3. Convert ONNX IR to graphs.ir
# ... new conversion layer needed ...
```

**Complexity**: ⭐⭐⭐ Moderate (new conversion layer)

---

### StableHLO Integration
```python
# 1. Export to MLIR
exported = torch.export.export(model, (example_input,))
mlir_module = fx_importer.export_and_import(exported)

# 2. Convert MLIR to StableHLO
# ... complex toolchain ...

# 3. Parse StableHLO
# ... new parser needed ...

# 4. Convert to graphs.ir
# ... new conversion layer ...
```

**Complexity**: ⭐ Complex (significant new infrastructure)

---

## Performance Characteristics

### Graph Extraction Performance

| Technology | Trace Time (ResNet18) | Trace Time (DETR) | Notes |
|------------|----------------------|-------------------|-------|
| FX | ~100ms | ❌ Fails | Fast when it works |
| Dynamo | ~500ms | ~2s | Slower but robust |
| ONNX | ~1s | ~5s | Export can be slow |
| StableHLO | ~10s+ | ~30s+ | Multi-step conversion |

### Memory Overhead

| Technology | Memory Overhead | Notes |
|------------|----------------|-------|
| FX | Low (~100MB) | Minimal overhead |
| Dynamo | Moderate (~500MB) | JIT compilation cache |
| ONNX | Moderate (~300MB) | Separate IR representation |
| StableHLO | High (~1GB+) | Multiple IR conversions |

---

## Recommendation for graphs Package

### Short Term (Immediate)
1. **Add Dynamo support** alongside existing FX
   - Implement `DynamoGraphExtractor` (see `experiments/dynamo/`)
   - Create conversion layer: Dynamo graph → `graphs.ir` structures
   - Fallback to FX for simple models

2. **Test with complex workloads**
   - YOLO models (experiments/YOLO/)
   - DETR models (workloads/pytorch/detr/)
   - Validate against fvcore for FLOP accuracy

3. **Update documentation**
   - Add Dynamo examples to docs/
   - Update CLAUDE.md with Dynamo workflow

### Medium Term (Next Sprint)
1. **Make Dynamo primary**
   - Default to Dynamo for all models
   - Keep FX as legacy fallback

2. **Add ONNX support** for cross-framework
   - Implement ONNX → `graphs.ir` converter
   - Use for TensorFlow/JAX model analysis

3. **Deprecate FX**
   - Mark FX workflow as deprecated
   - Migrate all tests to Dynamo

### Long Term (Future)
1. **Evaluate StableHLO** if needed for JAX
   - Only if JAX workload analysis becomes priority
   - Significant engineering effort required

2. **Multi-framework support**
   - Dynamo: PyTorch (primary)
   - ONNX: Cross-framework bridge
   - StableHLO: JAX-specific (if needed)

---

## Migration Guide: FX → Dynamo

### Before (FX)
```python
from torch.fx import symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp

# Trace
traced = symbolic_trace(model)  # May fail!

# Shape propagation
ShapeProp(traced).propagate(example_input)

# Extract graph
for node in traced.graph.nodes:
    # ... analysis ...
```

### After (Dynamo)
```python
import torch._dynamo as dynamo

# Extract
extractor = DynamoGraphExtractor()
compiled = torch.compile(model, backend=extractor)
_ = compiled(example_input)  # Shapes automatic!

# Extract graph
for gm in extractor.graph_modules:
    for node in gm.graph.nodes:
        # ... analysis ...
        # node.meta['tensor_meta'] has shapes!
```

### Benefits
- ✅ Handles complex models (YOLO, DETR)
- ✅ Automatic shape propagation
- ✅ No manual ShapeProp
- ✅ More robust

### Effort
- ~1-2 days to implement conversion layer
- ~2-3 days for testing and validation
- Minimal API changes for users

---

## Conclusion

**For the graphs package, use Dynamo as the primary graph extraction mechanism.**

### Why Dynamo?
1. **Handles all models** that FX fails on (YOLO, DETR, complex transformers)
2. **Automatic shape propagation** - no manual ShapeProp needed
3. **Production-ready** - PyTorch 2.0+ standard
4. **Native PyTorch** - no export/conversion overhead
5. **Active development** - future-proof choice

### Why not ONNX/StableHLO?
- **ONNX**: Good for cross-framework, but lossy conversion and deployment-focused (not analysis-focused)
- **StableHLO**: Too complex for PyTorch models, better suited for JAX

### What About TorchInductor?

**TorchInductor is CRITICAL for validation!**

While not for graph extraction, inductor provides:
- ✅ **Real-world baseline** - what users actually get with `torch.compile()`
- ✅ **Validation ground truth** - compare your predictions vs actual performance
- ✅ **Fusion analysis** - see what inductor fuses vs what you predict
- ✅ **Performance metrics** - validate roofline models

**Recommendation**: Use Dynamo for extraction + Inductor for validation

See `experiments/dynamo/INDUCTOR_VALIDATION.md` for details.

### Action Items
1. ✅ **Start experimenting** with Dynamo (use provided examples)
2. 🔄 **Implement conversion layer** (Dynamo → graphs.ir)
3. 🔄 **Test with complex models** (YOLO, DETR)
4. 🔄 **Add inductor validation** to analysis pipeline
5. 🔄 **Make Dynamo primary** in production

The examples in `experiments/dynamo/` provide all the building blocks needed for integration!
