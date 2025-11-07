# torch.compile and Dynamo Relationship Explained

## TL;DR

**You DON'T need separate torch.compile experiments!**

- `torch.compile` = User-facing API
- Dynamo = Graph capture engine (inside `torch.compile`)
- Your experiments **already use** `torch.compile` with a custom backend

## Visual Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          USER CODE                                  │
│                                                                     │
│   model = MyModel()                                                 │
│   compiled = torch.compile(model, backend="...")                   │
│                              │                                      │
└──────────────────────────────┼──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    torch.compile (API Layer)                        │
│  - Entry point for compilation                                      │
│  - Handles backend selection                                        │
│  - Manages caching                                                  │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                 TorchDynamo (Graph Capture Engine)                  │
│  - Analyzes Python bytecode                                         │
│  - Captures computational graph                                     │
│  - Handles control flow                                             │
│  - Manages graph breaks                                             │
│  - Propagates tensor metadata                                       │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
                        Captured FX Graph
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Backend (User Selectable)                        │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ "inductor" (default)                                        │   │
│  │  → TorchInductor optimization                               │   │
│  │  → Fuses kernels, generates code                            │   │
│  │  → Purpose: Make inference faster                           │   │
│  │  → graphs package: ❌ Don't need                            │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ "aot_eager"                                                 │   │
│  │  → Ahead-of-time graph capture, eager execution             │   │
│  │  → Purpose: Debugging                                       │   │
│  │  → graphs package: ❌ Don't need                            │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ CUSTOM BACKEND (Our GraphExtractor)                         │   │
│  │  → Extract graph structure                                  │   │
│  │  → Count operations                                         │   │
│  │  → Analyze memory                                           │   │
│  │  → NO optimization                                          │   │
│  │  → Purpose: Workload characterization                       │   │
│  │  → graphs package: ✅ YES! This is what we use             │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Code Comparison

### What torch.compile Usually Does (Production)

```python
# For production inference speedup
model = MyModel()

# Use default "inductor" backend
compiled = torch.compile(model)  # backend="inductor" is default

# Now model is optimized (fused ops, better memory layout, etc.)
output = compiled(input)  # Faster!
```

### What We Do (Analysis)

```python
# For workload characterization
model = MyModel()

# Create custom backend
class GraphExtractor:
    def __call__(self, gm, example_inputs):
        # gm = FX GraphModule from Dynamo
        # Analyze it, don't optimize it
        analyze_graph(gm.graph)
        return gm.forward  # Return original

extractor = GraphExtractor()

# Use torch.compile with OUR backend
compiled = torch.compile(model, backend=extractor)

# This triggers graph capture via Dynamo
# But instead of optimizing, we extract!
output = compiled(input)
```

## Key Points

### 1. torch.compile is Just the Entry Point

```python
torch.compile(model, backend=backend)
        │              └─ This determines what happens
        │
        └─ Always uses Dynamo for graph capture
```

### 2. Dynamo is Always Involved

```python
# All these use Dynamo internally:
torch.compile(model, backend="inductor")    # Dynamo → Inductor
torch.compile(model, backend="aot_eager")   # Dynamo → AOT Eager
torch.compile(model, backend=custom)        # Dynamo → Custom (us!)

# Dynamo = graph capture
# Backend = what to do with the graph
```

### 3. We Already Use torch.compile!

Look at our existing code:

```python
# From basic_dynamo_tracing.py:74
compiled_model = torch.compile(
    model,
    backend=extractor,  # ← Our custom backend
    fullgraph=False,
)
```

**This IS torch.compile!** We're not avoiding it - we're using it with a custom backend.

## Why You Don't Need Separate Experiments

### Reason 1: Already Using It

Every Dynamo example you have **already uses** `torch.compile`:

| File | Line | Usage |
|------|------|-------|
| `basic_dynamo_tracing.py` | 74 | `torch.compile(model, backend=extractor)` |
| `huggingface_complex_models.py` | 105 | `torch.compile(model, backend=extractor)` |
| `trace_yolo.py` | 170 | `torch.compile(pytorch_model, backend=extractor)` |
| `integrate_with_graphs.py` | 50 | `torch.compile(model, backend=extractor)` |

### Reason 2: Other Backends Not Useful

| Backend | Purpose | Useful for Characterization? |
|---------|---------|------------------------------|
| `"inductor"` | Optimization | ❌ No - changes the graph |
| `"aot_eager"` | Debugging | ❌ No - just for torch.compile debugging |
| `"cudagraphs"` | GPU optimization | ❌ No - CUDA-specific optimization |
| `"ipex"` | Intel optimization | ❌ No - Intel-specific optimization |
| Custom backend | Graph analysis | ✅ YES - this is what we need! |

### Reason 3: Separation of Concerns

```
torch.compile responsibilities:
├─ API entry point
├─ Backend management
└─ Caching

Dynamo responsibilities:
├─ Graph capture          ← We care about this
├─ Bytecode analysis      ← We care about this
└─ Metadata propagation   ← We care about this

Backend responsibilities:
├─ Graph transformation   ← We DON'T optimize
└─ Code generation        ← We DON'T compile

Our custom backend:
├─ Graph extraction       ← We DO extract
└─ Analysis               ← We DO analyze
```

## Analogy

Think of it like a camera system:

```
torch.compile = Camera body (housing, controls)
Dynamo       = Sensor (captures the image)
Backend      = Processing pipeline (what to do with the image)

┌────────────────────────────────────────────┐
│  Camera Body (torch.compile)               │
│                                            │
│  ┌──────────────────────────────────────┐ │
│  │ Sensor (Dynamo)                      │ │
│  │  Captures graph                      │ │
│  └──────────────┬───────────────────────┘ │
│                 │                          │
│                 ▼                          │
│  ┌──────────────────────────────────────┐ │
│  │ Processing (Backend)                 │ │
│  │                                      │ │
│  │ ├─ "inductor" → Enhance (sharpen)   │ │
│  │ ├─ "aot_eager" → Preview only       │ │
│  │ └─ Custom → Just save RAW (us!)     │ │
│  └──────────────────────────────────────┘ │
└────────────────────────────────────────────┘
```

**For analysis, we want the RAW capture (unoptimized graph), not the enhanced version!**

## Common Misconceptions

### ❌ Misconception 1: "torch.compile and Dynamo are different things"

**Reality**: Dynamo is **inside** torch.compile. You can't use torch.compile without Dynamo (in PyTorch 2.0+).

### ❌ Misconception 2: "I need to use torch.compile for optimization"

**Reality**: torch.compile **can** optimize (with inductor backend), but it can also **extract** (with custom backend). We use it for extraction.

### ❌ Misconception 3: "I should import torch._dynamo directly"

**Reality**: You **can**, but `torch.compile` is the cleaner API:

```python
# Less clean (but works)
import torch._dynamo as dynamo
dynamo.optimize(backend=custom)(model)

# Cleaner (recommended)
torch.compile(model, backend=custom)
```

### ❌ Misconception 4: "I need separate experiments for torch.compile"

**Reality**: Your Dynamo experiments **are** torch.compile experiments! They use torch.compile with a custom backend.

## What About torch._dynamo Import?

You might see this in code:

```python
import torch._dynamo as dynamo
dynamo.reset()
```

**Why?** This is just for:
- Resetting cached graphs
- Accessing advanced debugging features
- Configuration

**The actual compilation still uses torch.compile:**

```python
import torch._dynamo as dynamo  # For reset(), config, etc.

# Reset cache
dynamo.reset()

# But compilation still uses torch.compile
compiled = torch.compile(model, backend=custom)
```

## Summary: What You Actually Have

Your current setup:

```
experiments/dynamo/
├─ basic_dynamo_tracing.py          ← Uses torch.compile + custom backend ✓
├─ huggingface_complex_models.py    ← Uses torch.compile + custom backend ✓
├─ trace_yolo.py                    ← Uses torch.compile + custom backend ✓
├─ integrate_with_graphs.py         ← Uses torch.compile + custom backend ✓
└─ torch_compile_backends.py        ← Explains relationship (educational) ✓
```

**This is complete!** You have:
- ✅ torch.compile usage (with custom backend)
- ✅ Dynamo graph capture
- ✅ Graph extraction for analysis
- ✅ Integration with graphs package

**You DON'T need:**
- ❌ Separate torch.compile experiments
- ❌ Inductor backend examples (not for analysis)
- ❌ AOT eager backend examples (not for analysis)

## When You Might Care About Other Backends

You might want to experiment with other backends **only if**:

1. **Comparing optimized vs unoptimized performance**
   - Benchmark: `torch.compile(model, backend="inductor")` vs eager
   - But this is inference optimization, not characterization

2. **Debugging graph breaks in complex models**
   - Use: `torch.compile(model, backend="aot_eager")`
   - Helps find why graphs break without compilation overhead

3. **Hardware-specific optimization research**
   - Study how inductor generates code
   - Not relevant for workload characterization

**For DNN workload characterization: You have everything you need!**

## Final Recommendation

**✅ DO:**
- Continue using your current Dynamo examples
- Use `torch.compile(model, backend=custom_extractor)` pattern
- Focus on graph extraction and analysis
- Integrate with graphs.ir structures

**❌ DON'T:**
- Create separate torch.compile experiments
- Experiment with inductor/aot_eager/etc. for analysis work
- Worry about optimization backends

**📚 READ:**
- If confused: `torch_compile_backends.py` (the file I just created)
- For usage: Your existing examples (they're correct!)

---

**Bottom line**: Your experiments already use torch.compile correctly. The custom backend approach is exactly right for workload characterization. No additional torch.compile experiments needed!
