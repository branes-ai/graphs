import torch
import torch.nn as nn
import torch.nn.functional as F

# Define a 1-layer MLP
class OneLayerMLP(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(OneLayerMLP, self).__init__()
        self.fc = nn.Linear(in_features, out_features, bias)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        y = self.fc(x)
        return self.softmax(y)
    
# Define a 2-layer MLP
class TwoLayerMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TwoLayerMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        h = torch.tanh(self.fc1(x))
        y = self.fc2(h)
        return self.softmax(y)

# Define a 3-layer MLP
class ThreeLayerMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        super(ThreeLayerMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        h1 = torch.tanh(self.fc1(x))
        h2 = torch.tanh(self.fc2(h1))
        y = self.fc3(h2)
        return self.softmax(y)
    
# Define a 4-layer MLP
class FourLayerMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, hidden_dim3, output_dim):
        super(FourLayerMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, hidden_dim3)
        self.fc4 = nn.Linear(hidden_dim3, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        h1 = torch.tanh(self.fc1(x))
        h2 = torch.tanh(self.fc2(h1))
        h3 = torch.tanh(self.fc3(h2))
        y = self.fc4(h3)
        return self.softmax(y)