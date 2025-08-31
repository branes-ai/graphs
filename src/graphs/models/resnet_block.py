import torch.nn as nn

class ParamResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.skip = nn.Identity()
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        identity = self.skip(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)

def make_resnet_block(in_channels=64, out_channels=128, kernel_size=3, stride=1):
    return ParamResNetBlock(in_channels, out_channels, kernel_size, stride)
