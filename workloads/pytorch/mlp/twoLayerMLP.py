import torch

from multi_layer_perceptrons import TwoLayerMLP 

# Define a small 2-layer MLP model
input_dim = 5
hidden_dim = 4
output_dim = 3
model = TwoLayerMLP(input_dim, hidden_dim, output_dim)

# Example input
batch_size = 10
x = torch.randn(batch_size, input_dim)
y = model(x)
print("Output of 2-layer MLP:", y)
