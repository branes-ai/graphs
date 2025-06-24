import torch
import torch.nn as nn
import torch.nn.functional as F
import os

# Create output directory
os.makedirs('test_models', exist_ok=True)

# 3. Convolutional Neural Network
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)
        self.batch_norm1 = nn.BatchNorm2d(16)
        self.batch_norm2 = nn.BatchNorm2d(32)
    
    def forward(self, x):
        x = self.pool(F.relu(self.batch_norm1(self.conv1(x))))
        x = self.pool(F.relu(self.batch_norm2(self.conv2(x))))
        x = x.view(-1, 32 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Create and save CNN model
cnn_model = SimpleCNN()
test_input = torch.randn(1, 3, 32, 32)
cnn_model.eval()
with torch.no_grad():
    _ = cnn_model(test_input)

torch.save(cnn_model, 'test_models/simple_cnn.pt')
print("Created: simple_cnn.pt")

# Print model info for reference
print("\nModel architectures:")

print("\nSimple CNN:")
print(cnn_model)
print(f"Parameters: {sum(p.numel() for p in cnn_model.parameters())}")
