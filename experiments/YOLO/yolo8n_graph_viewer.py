import torch
import torch._dynamo
from torchview import draw_graph
from ultralytics import YOLO

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

# Assume 'traced_module' is the torch.fx.GraphModule you exported
# traced_module, guards = torch._dynamo.export(...)

# 1. Define input shape (must match the tracing input)
INPUT_SIZE = (1, 3, 640, 640) 

# 2. Draw the graph
# Use the .model attribute if the traced module is wrapped, 
# but since you used export, 'traced_module' is the GraphModule itself.
model_graph = draw_graph(
    traced_module, 
    input_size=INPUT_SIZE, 
    # Use the device of your inputs, or 'meta' if supported
    device='cpu', 
    graph_name='YOLO_FX_Graph', 
    # This is useful for large models, it collapses submodules (like C2f blocks)
    roll=True, 
    expand_nested=True, # Will expand the graph to show all FX nodes
    # Visualization args that might break tracing are removed here:
    #save_graph=True, 
    #filename='yolo_fx_graph', # The output file will be yolo_fx_graph.png (or .pdf)
    #direction='TB' # Top to Bottom layout
)

print("\nGraph visualization saved to 'yolo_fx_graph.png' (or .pdf if you prefer).")
# If you are in a Jupyter environment, model_graph.visual_graph will display it automatically.

"""
The error TypeError: forward() got an unexpected keyword argument 
'direction' indicates that Torchview is internally calling your 
traced_module's forward method, and is passing a keyword argument, 
direction, that your model's forward method does not accept.

This happens because:

When you call draw_graph, you pass visualization arguments like 
direction='TB'.

Torchview's internal logic attempts to clone and run the model's 
forward method (your traced_module) to collect tensor shape and 
flow information.

The way it passes arguments through the wrapper is sometimes too 
aggressive, including the visualization parameters like direction 
in the arguments for your model's forward function.

Since your traced_module is a simple FX graph module, its forward 
method signature is typically very clean, like forward(self, x), 
and it rejects the unexpected direction keyword argument, causing 
the crash.

🌟 The Fix: Isolate Visualization Arguments
The best way to fix this is to remove the positional/optional 
arguments that are specifically for visualization, such as 
direction, save_graph, and filename, and use only the core 
arguments that guide the model structure inference (input_size, 
expand_nested).
"""