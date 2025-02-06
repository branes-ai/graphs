import torch
import torch.nn as nn

# Define a 1-layer MLP
class OneLayerMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(OneLayerMLP, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x)

# Example usage
input_dim = 5
output_dim = 3
model = OneLayerMLP(input_dim, output_dim)

# Example input
x = torch.randn(1, input_dim)
y = model(x)
print("Output of 1-layer MLP:", y)
