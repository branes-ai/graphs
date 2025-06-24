import torch
import torch.nn as nn
import torch.nn.functional as F
import os

# Create output directory
os.makedirs('test_models', exist_ok=True)

# 1. Simple Linear Model
class SimpleLinear(nn.Module):
    def __init__(self):
        super(SimpleLinear, self).__init__()
        self.linear = nn.Linear(10, 1)
    
    def forward(self, x):
        return self.linear(x)

# Create and save simple linear model
simple_model = SimpleLinear()
# Add some test data to make it more realistic
test_input = torch.randn(1, 10)
simple_model.eval()

with torch.no_grad():
    output = simple_model(test_input)

print(f"PyTorch version: {torch.__version__}")
# Convert to TorchScript and save to a pt file so it can be read in C++ with torch::jit::load()
traced_model = torch.jit.trace(simple_model, test_input)
traced_model.save('test_models/simple_linear.pt')
print("Created: simple_linear.pt (TorchScript format)")

# Print model info for reference
print("\nModel architectures:")
print("\nSimple Linear:")
print(simple_model)
print(f"Parameters: {sum(p.numel() for p in simple_model.parameters())}")

