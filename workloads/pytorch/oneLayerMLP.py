import torch
import torch.nn as nn

# Define a 1-layer MLP
class OneLayerMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(OneLayerMLP, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        y = self.fc(x)
        return self.softmax(y)
        
# Example usage
input_dim = 5
output_dim = 3
model = OneLayerMLP(input_dim, output_dim)

# Example input
batch_size = 10
x = torch.randn(batch_size, input_dim)
y = model(x)
print("Output of 1-layer MLP:", y)
