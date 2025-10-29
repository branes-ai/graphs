"""
Subgraph building blocks for DNN models.

This package provides modular components for building neural networks:
- MLPs: Multi-layer perceptrons
- Conv2D stacks: Convolutional layers
- ResNet blocks: Residual blocks
- Attention: Multi-head attention mechanisms (decomposed for better fusion)
"""

from .mlp import ParamMLP, make_mlp
from .conv2d_stack import ParamConv2DStack, make_conv2d
from .resnet_block import ParamResNetBlock, make_resnet_block
from .attention import (
    DecomposedMultiheadAttention,
    SimpleAttentionBlock,
    make_attention_block,
)

__all__ = [
    # MLP
    "ParamMLP",
    "make_mlp",
    # Conv2D
    "ParamConv2DStack",
    "make_conv2d",
    # ResNet
    "ParamResNetBlock",
    "make_resnet_block",
    # Attention
    "DecomposedMultiheadAttention",
    "SimpleAttentionBlock",
    "make_attention_block",
]
