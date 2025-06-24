import torch
import torch.nn as nn
import torch.nn.functional as F
import os

# Create output directory
os.makedirs('test_models', exist_ok=True)

# 2. Multi-layer Neural Network
class MLPModel(nn.Module):
    def __init__(self):
        super(MLPModel, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

# Create and save MLP model
mlp_model = MLPModel()
test_input = torch.randn(1, 784)
mlp_model.eval()
with torch.no_grad():
    _ = mlp_model(test_input)

torch.save(mlp_model, 'test_models/mlp_classifier.pt')
print("Created: mlp_classifier.pt")



# Print model info for reference
print("\nMLP Classifier:")
print(mlp_model) 
print(f"Parameters: {sum(p.numel() for p in mlp_model.parameters())}")

