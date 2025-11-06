import torch
import torch._dynamo
from ultralytics import YOLO

# 1. Load your YOLO model and set up the input
yolo_model = YOLO('yolov8n.pt').model.eval()
dummy_input = torch.randn(1, 3, 640, 640)

## 2. ⭐ CRITICAL STEP: WARM-UP THE MODEL ⭐
# Run a single forward pass *before* calling torch._dynamo.export
# This executes the internal logic that mutates (sets) `self.anchors` 
# and `self.strides`.
# Since these attributes are now set, the subsequent export call will 
# skip the mutation, allowing the graph to be generated.
print("Warming up the YOLO model to trigger internal state initialization...")
with torch.no_grad():
    _ = yolo_model(dummy_input)

print("Model warmed up successfully. Starting export...")

# 3. Use torch._dynamo.export to get the FX GraphModule
# Note: The warning about the function call style is fine for now, 
# but you can also define a wrapper function if you want to follow 
# the suggested syntax.
traced_module, guards = torch._dynamo.export(yolo_model)( 
    dummy_input,
)

# 4. 'traced_module' is the standard torch.fx.GraphModule!
print("FX GraphModule successfully extracted using torch._dynamo.export.")

# 5. You can now enumerate the nodes
print("\n--- Enumerating Graph Nodes ---")
for i, node in enumerate(traced_module.graph.nodes):
    print(f"[{i+1}] {node}")

print(f"\nTotal nodes in the traced graph: {len(list(traced_module.graph.nodes))}")