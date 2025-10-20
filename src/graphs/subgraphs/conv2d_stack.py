import torch.nn as nn

class ParamConv2DStack(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, kernel_size=3, stride=1):
        super().__init__()
        layers = []
        for i in range(num_layers):
            in_ch = in_channels if i == 0 else out_channels
            layers.append(nn.Conv2d(in_ch, out_channels, kernel_size, stride, padding=1))
            layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def make_conv2d(in_channels=3, out_channels=16, num_layers=3, kernel_size=3, stride=1):
    return ParamConv2DStack(in_channels, out_channels, num_layers, kernel_size, stride)
