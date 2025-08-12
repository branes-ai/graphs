import torch
import torch.nn as nn
import torch.nn.functional as F

# Define a 3-layer MLP
class ThreeLayerMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        super(ThreeLayerMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        h1 = torch.tanh(self.fc1(x))
        h2 = torch.tanh(self.fc2(h1))
        y = self.fc3(h2)
        return self.softmax(y)

# Example usage
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
