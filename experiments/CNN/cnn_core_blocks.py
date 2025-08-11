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
    
    # ResNet blocks
    print("1. ResNet BasicBlock:")
    basic_block = BasicBlock(64, 64, stride=1)
    out = basic_block(x)
    print(f"   Input: {x.shape} -> Output: {out.shape}")
    
    print("\n2. ResNet Bottleneck:")
    # bottleneck = Bottleneck(64, 64, stride=1)
    # out = bottleneck(x)
    # print(f"   Input: {x.shape} -> Output: {out.shape}")
        # Need downsample when input channels != output channels * expansion
    downsample = nn.Sequential(
        nn.Conv2d(64, 64 * Bottleneck.expansion, kernel_size=1, stride=1, bias=False),
        nn.BatchNorm2d(64 * Bottleneck.expansion)
    )
    bottleneck = Bottleneck(64, 64, stride=1, downsample=downsample)
    out = bottleneck(x)
    print(f"   Input: {x.shape} -> Output: {out.shape}")
    
    # MobileNet blocks
    print("\n3. MobileNet Depthwise Separable Conv:")
    mb_block = DepthwiseSeparableConv(64, 128, stride=2)
    out = mb_block(x)
    print(f"   Input: {x.shape} -> Output: {out.shape}")
    
    print("\n4. MobileNetV2 Inverted Residual:")
    inv_res = InvertedResidualBlock(64, 64, stride=1, expand_ratio=6)
    out = inv_res(x)
    print(f"   Input: {x.shape} -> Output: {out.shape}")
    
    # EfficientNet blocks
    print("\n5. EfficientNet MBConv Block:")
    mbconv = MBConvBlock(64, 64, kernel_size=3, stride=1, expand_ratio=4, se_ratio=0.25)
    out = mbconv(x)
    print(f"   Input: {x.shape} -> Output: {out.shape}")
    
    print("\n6. Squeeze-and-Excitation Block:")
    se_block = SEBlock(64, reduction=4)
    out = se_block(x)
    print(f"   Input: {x.shape} -> Output: {out.shape}")
    
    print("\n=== Key Operators by Architecture ===")
    print("ResNet: Conv2d, BatchNorm2d, ReLU, Add (residual)")
    print("MobileNet: Depthwise Conv2d, Pointwise Conv2d (1x1), BatchNorm2d, ReLU/ReLU6")
    print("EfficientNet: Depthwise Conv2d, SEBlock, Conv2d, BatchNorm2d, Swish/SiLU, Dropout2d")