import torch
import torch.fx
import operator
from typing import Dict, Any, List
from ultralytics import YOLO

# --- Assuming 'traced_module' is the torch.fx.GraphModule from your export ---

class ShapeProp(torch.fx.Interpreter):
    """
    A simple FX Interpreter to run a forward pass and record the output shape
    and dtype for every node in the graph.
    """
    def run_node(self, node: torch.fx.Node) -> Any:
        # Run the node operation (e.g., call_function, call_module)
        result = super().run_node(node)
        
        # Store the shape/dtype information on the node itself
        if isinstance(result, torch.Tensor):
            node.output_shape = tuple(result.shape)
            node.output_dtype = result.dtype
        elif isinstance(result, (tuple, list)):
            # Handle list/tuple returns (e.g., for multi-output ops)
            shapes = []
            dtypes = []
            for item in result:
                if isinstance(item, torch.Tensor):
                    shapes.append(tuple(item.shape))
                    dtypes.append(item.dtype)
                else:
                    shapes.append(type(item).__name__)
                    dtypes.append("-")
            node.output_shape = shapes
            node.output_dtype = dtypes
        
        return result

def print_graph_with_shapes(traced_module: torch.fx.GraphModule, sample_input: torch.Tensor):
    # 1. Run the shape propagation pass
    ShapeProp(traced_module).run(sample_input)

    print("\n--- 📊 Dynamo FX Graph Node Details (with Shapes) 📊 ---")
    print(f"{'Node Name':<20} | {'Op Type':<15} | {'Target Operator':<30} | {'Input Nodes':<20} | {'Output Shape/Dtype':<40}")
    print("-" * 140)

    for node in traced_module.graph.nodes:
        # Get Operator and Target
        op_type = node.op
        target = str(node.target)
        
        # Get Input Source Nodes (Arguments)
        input_nodes = ", ".join(str(arg) for arg in node.args if isinstance(arg, torch.fx.Node))
        if not input_nodes:
            input_nodes = "-"
            
        # Get Output Shape and Dtype
        shape_info = getattr(node, 'output_shape', 'N/A')
        dtype_info = getattr(node, 'output_dtype', 'N/A')
        
        # Formatting for the table
        output_info = f"Shape: {shape_info}, Dtype: {dtype_info}".replace("torch.float32", "f32")
        
        print(f"{node.name:<20} | {op_type:<15} | {target:<30} | {input_nodes:<20} | {output_info:<40}")



# ... (Setup and Warm-up code, unchanged) ...
yolo_model = YOLO('yolov8n.pt').model.eval()
dummy_input = torch.randn(1, 3, 640, 640)
with torch.no_grad():
    _ = yolo_model(dummy_input)

def forward_fn(x):
    return yolo_model(x)

# Perform the Export
traced_module, guards = torch._dynamo.export(forward_fn)(dummy_input)

# Example Call
print_graph_with_shapes(traced_module, dummy_input)

