import torch
import torch.nn as nn
import torch.nn.functional as F
import os

# Create output directory
os.makedirs('test_models', exist_ok=True)

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

# Print model info for reference
print("\nModel architectures:")

print("\nComplex Model:")
print(complex_model)
print(f"Parameters: {sum(p.numel() for p in complex_model.parameters())}")
