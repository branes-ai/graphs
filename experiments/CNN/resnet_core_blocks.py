import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# =============================================================================
# ResNet Building Blocks
# =============================================================================

class BasicBlock(nn.Module):
    """ResNet Basic Block (used in ResNet-18, ResNet-34)
    
    Core operators: Conv2d, BatchNorm2d, ReLU, Add (residual connection)
    """
    expansion = 1
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, 
                 downsample: Optional[nn.Module] = None):
        super(BasicBlock, self).__init__()
        
        # First convolution
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # Second convolution
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        
        # First conv-bn-relu
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        # Second conv-bn
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Downsample residual if needed
        if self.downsample is not None:
            residual = self.downsample(x)
        
        # Add residual connection
        out += residual
        out = self.relu(out)
        
        return out


class Bottleneck(nn.Module):
    """ResNet Bottleneck Block (used in ResNet-50, ResNet-101, ResNet-152)
    
    Core operators: Conv2d (1x1, 3x3, 1x1), BatchNorm2d, ReLU, Add (residual)
    """
    expansion = 4
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1,
                 downsample: Optional[nn.Module] = None):
        super(Bottleneck, self).__init__()
        
        # 1x1 convolution (dimension reduction)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # 3x3 convolution
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 1x1 convolution (dimension expansion)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, 
                              kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        
        # 1x1 conv-bn-relu (reduce)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        # 3x3 conv-bn-relu
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        # 1x1 conv-bn (expand)
        out = self.conv3(out)
        out = self.bn3(out)
        
        # Downsample residual if needed
        if self.downsample is not None:
            residual = self.downsample(x)
        
        # Add residual connection
        out += residual
        out = self.relu(out)
        
        return out


# =============================================================================
# Example Usage for MLIR Analysis
# =============================================================================

if __name__ == "__main__":
    # Test tensor
    batch_size, channels, height, width = 1, 64, 32, 32
    x = torch.randn(batch_size, channels, height, width)
    
    print("=== Core Building Blocks for MLIR Dialect Analysis ===\n")
    
    # ResNet blocks
    print("1. ResNet BasicBlock:")
    basic_block = BasicBlock(64, 64, stride=1)
    out = basic_block(x)
    print(f"   Input: {x.shape} -> Output: {out.shape}")
    
    print("\n2. ResNet Bottleneck:")
    # The Bottleneck block has an expansion factor of 4, 
    # so when we pass out_channels=64, the final output actually has 64 * 4 = 256 channels, 
    # but the residual connection still has the original 64 channels.

    # In ResNet, when there's a channel dimension mismatch, a downsample module (typically '
    # 'a 1x1 convolution) is used to match the dimensions of the residual connection. '
    # We need to add the proper downsample module to handle this case.

    # Need downsample when input channels != output channels * expansion
    downsample = nn.Sequential(
        nn.Conv2d(64, 64 * Bottleneck.expansion, kernel_size=1, stride=1, bias=False),
        nn.BatchNorm2d(64 * Bottleneck.expansion)
    )
    bottleneck = Bottleneck(64, 64, stride=1, downsample=downsample)
    out = bottleneck(x)
    print(f"   Input: {x.shape} -> Output: {out.shape}")
    
    print("\nResNet: Conv2d, BatchNorm2d, ReLU, Add (residual)")



