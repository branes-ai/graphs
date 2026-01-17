# GitHub Issue Template: YOLOv5/YOLOv8 FX Tracing Support

**Use this template when posting to:**
- https://github.com/ultralytics/yolov5/issues
- https://github.com/ultralytics/ultralytics/issues
- PyTorch Forums: https://discuss.pytorch.org/

---

## Title

`[Question] PyTorch FX symbolic_trace support for YOLOv5/YOLOv8?`

or

`[Feature Request] FX-traceable YOLO models for graph analysis`

---

## Issue Body

### Description

I'm working on a hardware performance estimation tool for automotive ADAS systems that uses `torch.fx.symbolic_trace` to analyze model computation graphs. While YOLO models work great with TorchScript export, they fail when attempting FX symbolic tracing.

### Environment

```
- PyTorch: 2.7.1
- ultralytics: 8.3.159 (for YOLOv8)
- Python: 3.11.14
- OS: Linux
```

### Minimal Reproducible Example

**YOLOv5:**
```python
import torch
from torch.fx import symbolic_trace

model = torch.hub.load('ultralytics/yolov5', 'yolov5s',
                        pretrained=False, trust_repo=True)
model = model.model  # Remove autoshape wrapper
model.eval()

input_tensor = torch.randn(1, 3, 640, 640)
traced = symbolic_trace(model)  # TypeError: cat() received invalid args
```

**YOLOv8:**
```python
import torch
from torch.fx import symbolic_trace
from ultralytics import YOLO

model = YOLO('yolov8n.pt').model
model.eval()

input_tensor = torch.randn(1, 3, 640, 640)
traced = symbolic_trace(model)  # TraceError: Proxy object cannot be iterated
```

### Error Messages

**YOLOv5:**
```
TypeError: cat() received an invalid combination of arguments - got (Proxy, int)
```

**YOLOv8:**
```
torch.fx.proxy.TraceError: Proxy object cannot be iterated. This can be attempted
when the Proxy is used in a loop or as a *args or **kwargs function argument.
```

### Use Case

I need FX tracing (not TorchScript) for:
1. Graph-level operation fusion analysis (Conv+BN+ReLU patterns)
2. Subgraph partitioning for heterogeneous hardware (CPU+GPU+DSP)
3. Hardware resource mapping for automotive SoCs (TI TDA4VM, Qualcomm Hexagon)
4. Performance estimation for ADAS workloads (30 FPS, <100ms latency requirements)

### Questions

1. **Is there an FX-traceable version of YOLO models?**
   - Some sources mention YOLOv7/v8 "FX compatibility" but testing shows they fail
   - Is this referring to FX-based quantization rather than full graph tracing?

2. **Are there plans to support `torch.fx.symbolic_trace`?**
   - Would require avoiding Proxy iteration in forward passes
   - Common issues: `torch.cat()` usage, tensor unpacking, dynamic *args

3. **Alternative approaches?**
   - Custom FX-traceable YOLO variant?
   - Specific modules that need modification (Concat, Detect)?
   - Recommended workarounds?

### Additional Context

**Why TorchScript isn't sufficient:**
- TorchScript provides a compiled model but doesn't expose the computation graph structure
- FX provides graph IR needed for hardware mapper analysis
- Need to analyze operation-level characteristics for performance modeling

**Current workaround:**
- Using ResNet-50 / FCN-ResNet50 as proxies for YOLO workloads
- Works for hardware estimation but would prefer actual YOLO models for accuracy

**Detailed error documentation:**
- Full error traces, stack traces, and analysis available
- Can provide complete test scripts and reproduction steps
- Happy to test proposed solutions or contribute fixes

### Related Issues/Discussions

- YOLOv5 TorchScript tracing: #9341
- PyTorch FX Proxy iteration: https://pytorch.org/docs/stable/fx.html#torch.fx.Proxy

---

### Additional Files to Attach

If posting to GitHub, consider attaching:
1. Complete error traces (from `YOLO_FX_TRACING_ISSUES.md`)
2. Test scripts showing both YOLOv5 and YOLOv8 failures
3. Environment details (`pip list | grep -E "torch|ultralytics"`)

---

### Expected Response

I'm hoping to learn:
- [ ] Whether FX tracing is fundamentally incompatible with YOLO architecture
- [ ] If there's a specific model variant or flag for FX compatibility
- [ ] Any community workarounds or custom implementations
- [ ] Whether this is a potential feature request for future versions

---

### Optional: Offer to Help

> I'm happy to:
> - Test proposed solutions on automotive hardware targets
> - Provide performance comparison data (proxy vs real models)
> - Contribute to making YOLO models FX-traceable if there's interest
> - Share our hardware estimation pipeline if it helps with context

---

## Tags to Use

**GitHub:**
- `question`
- `enhancement` (if requesting feature)
- `pytorch` / `torch.fx`
- `graph-mode`

**PyTorch Forums:**
- Category: "FX (Functional Transformations)"
- Tags: `yolo`, `symbolic-trace`, `proxy-iteration`
