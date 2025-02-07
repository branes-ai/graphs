import torch
import torch.nn as nn
import torch.nn.functional as F

# Define a 2-layer MLP
class TwoLayerMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TwoLayerMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        h = torch.tanh(self.fc1(x))
        y = self.fc2(h)
        return self.softmax(y)

# Example usage
input_dim = 5
hidden_dim = 4
output_dim = 3
model = TwoLayerMLP(input_dim, hidden_dim, output_dim)

# Example input
batch_size = 10
x = torch.randn(batch_size, input_dim)
y = model(x)
print("Output of 2-layer MLP:", y)
