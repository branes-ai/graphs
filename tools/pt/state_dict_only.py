import torch
import torch.nn as nn
import torch.nn.functional as F
import os

# Create output directory
os.makedirs('test_models', exist_ok=True)

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

