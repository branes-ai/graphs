import torch
from torchview import draw_graph
from graphviz import Source # Import the Source class directly
from ultralytics import YOLO

# ... (Setup and Warm-up code, unchanged) ...
yolo_model = YOLO('yolov8n.pt').model.eval()
dummy_input = torch.randn(1, 3, 640, 640)
with torch.no_grad():
    _ = yolo_model(dummy_input)

def forward_fn(x):
    return yolo_model(x)

# Perform the Export
traced_module, guards = torch._dynamo.export(forward_fn, dummy_input)

# 1. Generate the Graphviz object using draw_graph (without saving)
INPUT_SIZE = (1, 3, 640, 640) 

model_graph = draw_graph(
    traced_module, 
    input_size=INPUT_SIZE, 
    device='cpu', 
    graph_name='YOLO_FX_Graph', 
    roll=True, 
    expand_nested=True,
)

# 2. Extract the raw DOT source string
dot_source_code = model_graph.visual_graph.source

# 3. Use the native graphviz package to render and save the file
# We specify the directory ('.') and the final filename prefix
print("\nAttempting to save using native graphviz.Source().render()...")
graph = Source(dot_source_code, filename='yolo_fx_graph', format='png', directory='.')
graph.render(view=False)

print("Graph saving executed. Check for 'yolo_fx_graph.png' and 'yolo_fx_graph'.")