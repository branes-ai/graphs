import torch
import torch._dynamo
from ultralytics import YOLO

# 1. Load your YOLO model and define inputs
model = YOLO('yolov8n.pt').model.eval()
dummy_input = torch.randn(1, 3, 640, 640)

# --- Start of new pattern ---

# 2. Use torch._dynamo.export to get the FX GraphModule
# The function returns the graph_module, the guards, and the module back (for state).
# The default backend for export is 'fx', which returns an FX GraphModule.
traced_module, guards = torch._dynamo.export(
    model, 
    dummy_input,
    # Export returns an FX GraphModule by default.
    # Optionally, specify a backend like 'aot_eager' if you prefer.
)

# 3. 'traced_module' is now a standard torch.fx.GraphModule!
print("FX GraphModule successfully extracted.")

# 4. You can now enumerate the nodes
print("\n--- Enumerating Graph Nodes ---")
for node in traced_module.graph.nodes:
    print(node)

# --- End of new pattern ---

"""
You've hit another very common and frustrating obstacle when trying 
to trace complex models like Ultralytics YOLO with advanced tools!

The error AssertionError: Mutating module attribute anchors during export. 
means the YOLO model's forward or a helper method it calls is writing to 
an attribute of the module itself (self.anchors and self.strides in 
ultralytics/nn/modules/head.py).

TorchDynamo's export mode (and FX tracing in general) strictly requires 
that the model is immutable during tracing. The model must not change its 
internal state (module attributes) during the forward pass. This is a 
fundamental constraint for creating a static computational graph.

Here is the explanation and the most reliable solution to get an FX 
graph of a YOLO model.

🛑 Why export Failed (The Mutability Problem)
When you look at the traceback, the error points directly to the 
Ultralytics code:

Python

File ".../ultralytics/nn/modules/head.py", line 164, in _inference
    self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
This line is calculating and storing the anchor points and strides 
onto the detection head module (self) during the first forward pass. 
This is necessary for the YOLO model to be versatile, but it 
constitutes in-place module mutation during tracing, which breaks the 
FX/Dynamo graph generation process.

✅ The Solution: Pre-run the Mutating Logic (Warming Up)
The fix is to ensure the anchors and strides are calculated and 
stored before calling torch._dynamo.export. This "warms up" the model, 
completing the state mutation so the subsequent tracing attempt sees a 
static, immutable model.

See yolo8n_tracer.py for an example of how to do this.
"""