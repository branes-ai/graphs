import torch
import torch.nn as nn
import torch.nn.functional as F

from multi_layer_perceptrons import ThreeLayerMLP 

# Define a large 3-layer MLP model
input_dim = 1024
hidden_dim1 = 8*1024
hidden_dim2 = 8*1024
output_dim = 128
model = ThreeLayerMLP(input_dim, hidden_dim1, hidden_dim2, output_dim)

# Example input
batch_size = 10
x = torch.randn(batch_size, input_dim)
y = model(x)
print("Output of 3-layer MLP:", y)

print("Symbolic tracing the model...")
graph = torch.fx.symbolic_trace(model, concrete_args=(x,))
print(graph)

print("Exporting the model...")
exported = torch.export.export(model, (x,))
graph = exported.graph
print(graph)

# export to ONNX format
torch.onnx.export(model,
                  x,
                  "threeLayerMLP.onnx",
                  export_params=True,
                  opset_version=17,
                  input_names=['input'],
                  output_names=['output'])
