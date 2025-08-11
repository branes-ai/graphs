import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# =============================================================================
# EfficientNet Building Blocks
# =============================================================================

class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block - key component in EfficientNet
    
    Core operators: AdaptiveAvgPool2d, Conv2d (1x1), ReLU, Sigmoid, Mul (channel attention)
    """
    
    def __init__(self, in_channels: int, reduction: int = 4):
        super(SEBlock, self).__init__()
        
        reduced_channels = max(1, in_channels // reduction)
        
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Squeeze: Global average pooling
        scale = self.squeeze(x)
        
        # Excitation: FC layers with ReLU and Sigmoid
        scale = self.excitation(scale)
        
        # Scale the input
        return x * scale


class MBConvBlock(nn.Module):
    """Mobile Inverted Bottleneck Convolution Block - EfficientNet building block
    
    Core operators: Conv2d (1x1 expand), Depthwise Conv2d, SEBlock, Conv2d (1x1 project),
                    BatchNorm2d, Swish, Add (skip connection), Dropout
    """
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int, expand_ratio: int, se_ratio: float = 0.25,
                 drop_rate: float = 0.0):
        super(MBConvBlock, self).__init__()
        
        self.use_residual = stride == 1 and in_channels == out_channels
        self.drop_rate = drop_rate
        
        # Expansion phase
        expanded_channels = in_channels * expand_ratio
        
        layers = []
        
        # Expand (1x1 conv) if expand_ratio != 1
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, expanded_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(expanded_channels),
                nn.SiLU(inplace=True)  # Swish activation
            ])
        
        # Depthwise convolution
        layers.extend([
            nn.Conv2d(expanded_channels, expanded_channels, kernel_size=kernel_size,
                     stride=stride, padding=kernel_size//2, groups=expanded_channels, 
                     bias=False),
            nn.BatchNorm2d(expanded_channels),
            nn.SiLU(inplace=True)
        ])
        
        self.conv = nn.Sequential(*layers)
        
        # Squeeze-and-Excitation
        if se_ratio > 0:
            se_channels = max(1, int(in_channels * se_ratio))
            self.se = SEBlock(expanded_channels, reduction=expanded_channels//se_channels)
        else:
            self.se = nn.Identity()
        
        # Project phase (1x1 conv, no activation)
        self.project = nn.Sequential(
            nn.Conv2d(expanded_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        # Dropout for skip connection
        if drop_rate > 0:
            self.dropout = nn.Dropout2d(drop_rate)
        else:
            self.dropout = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        out = self.se(out)
        out = self.project(out)
        
        # Add skip connection if applicable
        if self.use_residual:
            out = self.dropout(out) + x
        
        return out


# =============================================================================
# Example Usage for MLIR Analysis
# =============================================================================

if __name__ == "__main__":
    # Test tensor
    batch_size, channels, height, width = 1, 64, 32, 32
    x = torch.randn(batch_size, channels, height, width)
    
    print("=== Core Building Blocks for MLIR Dialect Analysis ===\n")
    
    # EfficientNet blocks
    print("\n1. EfficientNet MBConv Block:")
    mbconv = MBConvBlock(64, 64, kernel_size=3, stride=1, expand_ratio=4, se_ratio=0.25)
    out = mbconv(x)
    print(f"   Input: {x.shape} -> Output: {out.shape}")
    
    print("\n2. Squeeze-and-Excitation Block:")
    se_block = SEBlock(64, reduction=4)
    out = se_block(x)
    print(f"   Input: {x.shape} -> Output: {out.shape}")
    
    print("\nEfficientNet: Depthwise Conv2d, SEBlock, Conv2d, BatchNorm2d, Swish/SiLU, Dropout2d")
