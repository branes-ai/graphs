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
#traced = torch.jit.trace(checkpoint, example_input)
#traced.save("checkpoint.pt")
print("Created: checkpoint.pt")

