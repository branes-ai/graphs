import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# =============================================================================
# MobileNet Building Blocks
# =============================================================================

class DepthwiseConv2d(nn.Module):
    """Depthwise Convolution - key operator in MobileNet"""
    
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1, 
                 padding: int = 0):
        super(DepthwiseConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                  stride=stride, padding=padding, groups=in_channels, 
                                  bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.depthwise(x)


class DepthwiseSeparableConv(nn.Module):
    """Depthwise Separable Convolution Block - core MobileNet building block
    
    Core operators: Depthwise Conv2d, Pointwise Conv2d (1x1), BatchNorm2d, ReLU
    """
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1):
        super(DepthwiseSeparableConv, self).__init__()
        
        # Depthwise convolution
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                  stride=stride, padding=padding, groups=in_channels,
                                  bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        
        # Pointwise convolution (1x1)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Depthwise conv-bn-relu
        out = self.depthwise(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        # Pointwise conv-bn-relu
        out = self.pointwise(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        return out


class InvertedResidualBlock(nn.Module):
    """MobileNetV2 Inverted Residual Block
    
    Core operators: Conv2d (1x1 expand), Depthwise Conv2d, Conv2d (1x1 project), 
                    BatchNorm2d, ReLU6, Add (skip connection)
    """
    
    def __init__(self, in_channels: int, out_channels: int, stride: int, 
                 expand_ratio: int):
        super(InvertedResidualBlock, self).__init__()
        self.stride = stride
        self.use_residual = stride == 1 and in_channels == out_channels
        
        hidden_dim = in_channels * expand_ratio
        
        layers = []
        
        # Expand phase (1x1 conv) - only if expand_ratio != 1
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True)
            ])
        
        # Depthwise conv phase
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, 
                     padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            
            # Project phase (1x1 conv, no activation)
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])
        
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        
        # Add skip connection if applicable
        if self.use_residual:
            out = out + x
        
        return out

# =============================================================================
# Example Usage for MLIR Analysis
# =============================================================================

if __name__ == "__main__":
    # Test tensor
    batch_size, channels, height, width = 1, 64, 32, 32
    x = torch.randn(batch_size, channels, height, width)
    
    print("=== Core Building Blocks for MLIR Dialect Analysis ===\n")
    
    # MobileNet blocks
    print("\n1. MobileNet Depthwise Separable Conv:")
    mb_block = DepthwiseSeparableConv(64, 128, stride=2)
    out = mb_block(x)
    print(f"   Input: {x.shape} -> Output: {out.shape}")
    
    print("\n2. MobileNetV2 Inverted Residual:")
    inv_res = InvertedResidualBlock(64, 64, stride=1, expand_ratio=6)
    out = inv_res(x)
    print(f"   Input: {x.shape} -> Output: {out.shape}")
    
    print("\nMobileNet: Depthwise Conv2d, Pointwise Conv2d (1x1), BatchNorm2d, ReLU/ReLU6")
