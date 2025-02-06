import torch
import torch.nn as nn
import torch.nn.functional as F

# Define a 2-layer MLP
class TwoLayerMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TwoLayerMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h = torch.tanh(self.fc1(x))
        return self.fc2(h)

# Example usage
input_dim = 5
hidden_dim = 4
output_dim = 3
model = TwoLayerMLP(input_dim, hidden_dim, output_dim)

# Example input
x = torch.randn(1, input_dim)
y = model(x)
print("Output of 2-layer MLP:", y)
