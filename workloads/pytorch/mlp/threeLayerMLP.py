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
input_dim = 5
hidden_dim1 = 4
hidden_dim2 = 4
output_dim = 3
model = ThreeLayerMLP(input_dim, hidden_dim1, hidden_dim2, output_dim)

# Example input
batch_size = 10
x = torch.randn(batch_size, input_dim)
y = model(x)
print("Output of 3-layer MLP:", y)
