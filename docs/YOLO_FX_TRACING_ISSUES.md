# PyTorch FX Tracing Issues with YOLO Models

**Date**: 2025-10-24
**Purpose**: Document FX tracing errors for Ultralytics community feedback
**Context**: Hardware performance estimation pipeline using `torch.fx.symbolic_trace`

---

## Summary

We are building a hardware performance estimation tool that uses PyTorch FX (`torch.fx.symbolic_trace`) to analyze model graphs for automotive ADAS applications. While YOLO models (YOLOv5, YOLOv8) can be exported to TorchScript, they fail when attempting **FX symbolic tracing** due to Proxy iteration issues.

## Environment

```
Python: 3.11.14
PyTorch: 2.7.1+cu126
ultralytics: 8.3.159
ultralytics-thop: 2.0.14
seaborn: 0.13.2
gitpython: 3.1.45
```

## Use Case

We need to trace YOLO models with `torch.fx.symbolic_trace` (not `torch.jit.trace`) for:
1. Graph partitioning and fusion analysis
2. Hardware resource mapping (CPU, GPU, DSP, NPU)
3. Performance estimation for automotive ADAS systems
4. Target: TI TDA4VM, Qualcomm Hexagon, NVIDIA Jetson platforms

---

## Error 1: YOLOv5 from torch.hub

### Code to Reproduce

```python
import torch
from torch.fx import symbolic_trace

# Load YOLOv5s
model = torch.hub.load('ultralytics/yolov5', 'yolov5s',
                        pretrained=False, verbose=False, trust_repo=True)
model.eval()

# Extract base model (remove autoshape wrapper)
if hasattr(model, 'model'):
    model = model.model

# Attempt FX tracing
input_tensor = torch.randn(1, 3, 640, 640)
traced = symbolic_trace(model)  # FAILS HERE
```

### Error Output

```
TypeError: cat() received an invalid combination of arguments - got (Proxy, int), but expected one of:
 * (tuple of Tensors tensors, int dim = 0, *, Tensor out = None)
 * (tuple of Tensors tensors, name dim, *, Tensor out = None)
      didn't match because some of the arguments have invalid types: (Proxy, int)
```

### Analysis

The error occurs in YOLOv5's `Concat` module (line 12, 16, 19, 22 in the model):
```
12           [-1, 6]  1         0  models.common.Concat                    [1]
```

The `Concat` module uses `torch.cat()` in a way that creates Proxy objects which FX cannot handle during tracing.

---

## Error 2: YOLOv8 from ultralytics

### Code to Reproduce

```python
import torch
from torch.fx import symbolic_trace
from ultralytics import YOLO

# Load YOLOv8n (nano - smallest version)
model = YOLO('yolov8n.pt')
torch_model = model.model
torch_model.eval()

# Attempt FX tracing
input_tensor = torch.randn(1, 3, 640, 640)
traced = symbolic_trace(torch_model)  # FAILS HERE
```

### Error Output

```
torch.fx.proxy.TraceError: Proxy object cannot be iterated. This can be attempted when
the Proxy is used in a loop or as a *args or **kwargs function argument. See the torch.fx
docs on pytorch.org for a more detailed explanation of what types of control flow can be
traced, and check out the Proxy docstring for help troubleshooting Proxy iteration errors
```

### Stack Trace

```
File "/home/stillwater/venv/p311/lib/python3.11/site-packages/torch/fx/_symbolic_trace.py", line 837, in trace
    (self.create_arg(fn(*args)),),
                     ^^^^^^^^^
  File "/home/stillwater/venv/p311/lib/python3.11/site-packages/ultralytics/nn/tasks.py", line -1, in forward
  File "/home/stillwater/venv/p311/lib/python3.11/site-packages/torch/fx/proxy.py", line 520, in __iter__
    return self.tracer.iter(self)
           ^^^^^^^^^^^^^^^^^^^^^^
  File "/home/stillwater/venv/p311/lib/python3.11/site-packages/torch/fx/proxy.py", line 377, in iter
    raise TraceError(...)
```

### Analysis

YOLOv8's `DetectionModel.forward()` uses operations that attempt to iterate over Proxy objects, which is not supported by FX tracing. This commonly occurs with:
- Tensor unpacking: `x, y, z = tensor`
- Dynamic arguments: `func(*args)` where args contains Proxies
- Loop iteration over Proxy objects

---

## What We've Tried

1. **Removing autoshape wrapper**: YOLOv5 has an autoshape wrapper that adds preprocessing - removing it didn't help
2. **Using base model directly**: Extracted `model.model` to get the core Sequential model - still fails
3. **Testing multiple versions**: YOLOv5s (7.2M params), YOLOv8n (3.2M params) - both fail
4. **Different input sizes**: 640x640, 416x416 - no difference

---

## Questions for Ultralytics Community

1. **Is there an FX-traceable version of YOLO?**
   - Some sources mention "FX compatibility" for YOLOv7/YOLOv8, but testing shows standard models fail
   - Is there a specific branch, flag, or model variant designed for `torch.fx.symbolic_trace`?

2. **FXModel wrapper class?**
   - Is there a custom `FXModel` class or wrapper that makes YOLO models FX-traceable?
   - We found references to "FX mode" but these seem to refer to quantization, not full tracing

3. **Alternative approaches?**
   - Is there a way to modify the model architecture to avoid Proxy iteration issues?
   - Are there specific modules (Concat, Detect) that need custom FX-compatible implementations?

4. **TorchScript vs FX tracing?**
   - We know TorchScript export (`torch.jit.trace`) works
   - Is FX tracing (`torch.fx.symbolic_trace`) fundamentally incompatible with YOLO architecture?

---

## Temporary Workaround

For our hardware estimation pipeline, we've implemented a **proxy model approach**:

```python
# Instead of real YOLO, use FX-traceable proxies with similar computational characteristics
def create_yolov5s_automotive(batch_size=1):
    """Proxy: ResNet-50 represents similar detection backbone complexity"""
    model = models.resnet50(weights=None)
    model.eval()
    input_tensor = torch.randn(batch_size, 3, 640, 640)
    return model, input_tensor, "YOLOv5s-Automotive"

def create_unet_lane_segmentation(batch_size=1):
    """Proxy: FCN-ResNet50 for lane segmentation workload"""
    model = models.segmentation.fcn_resnet50(weights=None)
    model.eval()
    input_tensor = torch.randn(batch_size, 3, 640, 360)
    return model, input_tensor, "UNet-LaneSegmentation"
```

**This works for our use case** because we're modeling hardware execution characteristics (FLOPs, memory bandwidth, latency) rather than exact model behavior. However, we'd prefer to use actual YOLO models if FX tracing is possible.

---

## Why We Need FX Tracing (Not TorchScript)

Our pipeline uses PyTorch FX for:

1. **Graph Fusion Analysis**: Identify fusable operations (Conv+BN+ReLU)
2. **Subgraph Partitioning**: Split graph for heterogeneous hardware (CPU+GPU+DSP)
3. **Shape Propagation**: Track tensor shapes through the graph
4. **Hardware Mapping**: Map operations to target accelerators (NPU, DSP, GPU cores)

TorchScript provides a compiled model but doesn't expose the computation graph structure we need for these analyses.

---

## Expected Behavior

Ideally, we would be able to:

```python
import torch
from torch.fx import symbolic_trace
from ultralytics import YOLO

model = YOLO('yolov8s.pt')
traced = symbolic_trace(model.model)  # Should succeed
traced.graph.print_tabular()  # Inspect computation graph
```

This would allow us to:
- Analyze YOLO's actual computational structure
- Provide accurate hardware performance estimates for automotive ADAS
- Compare real YOLO vs proxy models for validation

---

## References

- **PyTorch FX Documentation**: https://pytorch.org/docs/stable/fx.html
- **Proxy Iteration Issues**: https://pytorch.org/docs/stable/fx.html#torch.fx.Proxy
- **Our Project**: Hardware performance estimation for automotive ADAS (TI TDA4VM, Qualcomm Hexagon 698, NVIDIA Jetson)
- **Target Use Cases**: Lane detection, object detection, traffic sign recognition (30 FPS, <100ms latency, ASIL-D)

---

## Additional Information

If helpful, we can provide:
- Complete reproducible test scripts
- Our FX-based graph analysis pipeline code
- Specific hardware mapper requirements
- Performance comparison data (proxy vs real models if FX tracing becomes available)

---

## Contact

Please feel free to ask for clarification or additional details. We're happy to test proposed solutions or contribute to making YOLO models FX-traceable if there's community interest.

**Project Context**: Deep learning hardware performance estimation for automotive embedded systems
**Repository**: https://github.com/[your-repo] (if public)
