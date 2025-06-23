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
    _ = simple_model(test_input)

torch.save(simple_model, 'test_models/simple_linear.pt')
print("Created: simple_linear.pt")

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

# 4. Model with different parameter types and custom metadata
class ComplexModel(nn.Module):
    def __init__(self):
        super(ComplexModel, self).__init__()
        self.embedding = nn.Embedding(1000, 64)
        self.lstm = nn.LSTM(64, 128, batch_first=True, bidirectional=True)
        self.attention = nn.MultiheadAttention(256, 8, batch_first=True)
        self.layer_norm = nn.LayerNorm(256)
        self.classifier = nn.Linear(256, 5)
        
        # Add some custom attributes
        self.model_version = "1.0"
        self.training_steps = 1000
    
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        normed = self.layer_norm(attn_out)
        pooled = torch.mean(normed, dim=1)
        return self.classifier(pooled)

# Create and save complex model
complex_model = ComplexModel()
test_input = torch.randint(0, 1000, (1, 50))  # sequence of token IDs
complex_model.eval()
with torch.no_grad():
    _ = complex_model(test_input)

torch.save(complex_model, 'test_models/complex_model.pt')
print("Created: complex_model.pt")

# 5. State dict only (no model class)
simple_state_dict = {
    'weight': torch.randn(5, 10),
    'bias': torch.randn(5),
    'running_mean': torch.zeros(5),
    'running_var': torch.ones(5),
    'num_batches_tracked': torch.tensor(100)
}

torch.save(simple_state_dict, 'test_models/state_dict_only.pt')
print("Created: state_dict_only.pt")

# 6. Model with optimizer state (checkpoint format)
checkpoint_model = SimpleLinear()
optimizer = torch.optim.Adam(checkpoint_model.parameters(), lr=0.001)

checkpoint = {
    'model_state_dict': checkpoint_model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': 42,
    'loss': 0.123,
    'accuracy': 0.95,
    'hyperparameters': {
        'learning_rate': 0.001,
        'batch_size': 32,
        'num_epochs': 100
    }
}

torch.save(checkpoint, 'test_models/checkpoint.pt')
print("Created: checkpoint.pt")

# Print summary
print("\nTest models created in 'test_models/' directory:")
print("1. simple_linear.pt - Basic linear layer")
print("2. mlp_classifier.pt - Multi-layer perceptron with dropout")
print("3. simple_cnn.pt - CNN with conv layers, batch norm, pooling")
print("4. complex_model.pt - LSTM + attention model with embeddings")
print("5. state_dict_only.pt - Just parameter tensors")
print("6. checkpoint.pt - Full training checkpoint with optimizer state")

# Print model info for reference
print("\nModel architectures:")
print("\nSimple Linear:")
print(simple_model)
print(f"Parameters: {sum(p.numel() for p in simple_model.parameters())}")

print("\nMLP Classifier:")
print(mlp_model) 
print(f"Parameters: {sum(p.numel() for p in mlp_model.parameters())}")

print("\nSimple CNN:")
print(cnn_model)
print(f"Parameters: {sum(p.numel() for p in cnn_model.parameters())}")

print("\nComplex Model:")
print(complex_model)
print(f"Parameters: {sum(p.numel() for p in complex_model.parameters())}")
