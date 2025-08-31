import torch.nn as nn

class ParamMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)

def make_mlp(in_dim=128, hidden_dim=256, out_dim=64):
    return ParamMLP(in_dim, hidden_dim, out_dim)
