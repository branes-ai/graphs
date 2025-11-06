import torch
import torch._dynamo
from ultralytics import YOLO

"""
The cleanest way to use torch._dynamo.export is to wrap the module call 
inside a simple Python function (often a lambda or a defined function). 
This aligns with the preferred, function-based export API.

Here is the updated, cleaner code, incorporating the crucial 
warm-up step and the function wrapper.

🐍 Clean Dynamo Export with a Function Wrapper
The function wrapper isolates the model's forward pass, 
making the call to torch._dynamo.export cleaner and 
adhering to PyTorch's evolving export conventions.
"""

# 1. Setup Model and Input
# Use .model to get the core torch.nn.Module, and set to evaluation mode
yolo_model = YOLO('yolov8n.pt').model.eval()
dummy_input = torch.randn(1, 3, 640, 640)

# 2. 🔥 CRITICAL: WARM-UP THE MODEL 🔥
# This executes the internal YOLO logic that mutates (sets) self.anchors/self.strides,
# preventing the "Mutating module attribute" error during export.
print("Warming up the YOLO model...")
with torch.no_grad():
    _ = yolo_model(dummy_input)

# 3. Define the Function Wrapper
# We use a lambda function here for the most concise syntax.
# The `export` function traces THIS function call.
def forward_fn(x):
    # Pass the input tensor (x) through the warmed-up YOLO model
    return yolo_model(x)

# 4. Perform the Export
# The modern syntax calls `torch._dynamo.export` with the function and then the inputs.
print("Starting Dynamo Export with function wrapper...")
traced_module, guards = torch._dynamo.export(
    forward_fn, 
    dummy_input
)

# 4. 'traced_module' is the standard torch.fx.GraphModule.
print("FX GraphModule successfully extracted using torch._dynamo.export.")

# 5. Enumerate the Graph

print("\n--- Enumerating Graph Nodes ---")
for i, node in enumerate(traced_module.graph.nodes):
    print(f"[{i+1}] {node}")

print(f"\nTotal nodes in the traced graph: {len(list(traced_module.graph.nodes))}")

"""
Key Improvements:
Clean Syntax: Using torch._dynamo.export(forward_fn, dummy_input) 
is the canonical way to export arbitrary functions, including the 
forward pass of a module.

Immutability: The warm-up step ensures the model is static when 
export starts tracing, avoiding the AssertionError regarding 
attribute mutation.

Now you have a fully traced torch.fx.GraphModule that represents 
the computational flow of the YOLO model!
"""