\# CNN Building Blocks



The core building blocks for MobileNet, EfficientNet, and ResNet in PyTorch. These are the fundamental layers that get repeated to form the complete architectures. 



\*\*ResNet Core Operators:\*\*

\- `BasicBlock`: Conv2d → BatchNorm2d → ReLU → Conv2d → BatchNorm2d → Add (residual) → ReLU

\- `Bottleneck`: Three Conv2d layers (1x1→3x3→1x1) with BatchNorm2d, ReLU, and residual connection



\*\*MobileNet Core Operators:\*\*

\- `DepthwiseSeparableConv`: Depthwise Conv2d → Pointwise Conv2d (1x1) with BatchNorm2d and ReLU

\- `InvertedResidualBlock` (MobileNetV2): Expand (1x1) → Depthwise → Project (1x1) with skip connections



\*\*EfficientNet Core Operators:\*\*

\- `MBConvBlock`: Mobile inverted bottleneck with Squeeze-and-Excitation

\- `SEBlock`: Global average pooling → FC layers → sigmoid gating for channel attention

\- Uses Swish (SiLU) activation and Dropout2d



These blocks represent the fundamental computational patterns that get repeated to form complete networks (e.g., ResNet50 uses multiple Bottleneck blocks, EfficientNet-B0 uses multiple MBConv blocks with different expansion ratios).



For MLIR analysis, we can see how each architecture emphasizes different operator compositions:

\- ResNet focuses on standard convolutions with residual connections

\- MobileNet introduces depthwise separable convolutions for efficiency

\- EfficientNet adds squeeze-and-excitation attention and more sophisticated activation functions



