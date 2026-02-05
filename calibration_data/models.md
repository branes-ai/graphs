● The models were selected to maximize coverage of different operation types and FLOP sizes with minimal redundancy. Here's the reasoning:

  Operation Type Coverage
  ┌──────────────┬─────────────────────────────────────────┬──────────────────────────────────────────────────┐
  │ Model Family │         Primary Operation Types         │                    Why Needed                    │
  ├──────────────┼─────────────────────────────────────────┼──────────────────────────────────────────────────┤
  │ ResNet       │ Conv2D + BatchNorm + ReLU               │ Standard convolutions, the backbone of most CNNs │
  ├──────────────┼─────────────────────────────────────────┼──────────────────────────────────────────────────┤
  │ VGG          │ Large Conv2D (no residuals)             │ Very large convolutions (100M-500M FLOPs each)   │
  ├──────────────┼─────────────────────────────────────────┼──────────────────────────────────────────────────┤
  │ MobileNet    │ Depthwise separable, inverted residuals │ Memory-bound ops with very low efficiency (~3%)  │
  ├──────────────┼─────────────────────────────────────────┼──────────────────────────────────────────────────┤
  │ EfficientNet │ MBConv blocks, squeeze-excite           │ Fused depthwise+pointwise patterns               │
  ├──────────────┼─────────────────────────────────────────┼──────────────────────────────────────────────────┤
  │ ViT          │ MatMul/Linear, attention                │ Transformer ops, different from CNN patterns     │
  ├──────────────┼─────────────────────────────────────────┼──────────────────────────────────────────────────┤
  │ MaxViT       │ Hybrid conv + attention                 │ Mixed architecture                               │
  └──────────────┴─────────────────────────────────────────┴──────────────────────────────────────────────────┘
  FLOP Size Coverage

  The efficiency curves need data points across the full range:
  ┌────────────┬──────────────────────────────────┐
  │ Size Range │      Models Providing Data       │
  ├────────────┼──────────────────────────────────┤
  │ <1M FLOPs  │ MobileNet tiny ops, EfficientNet │
  ├────────────┼──────────────────────────────────┤
  │ 1-10M      │ MobileNet, EfficientNet blocks   │
  ├────────────┼──────────────────────────────────┤
  │ 10-50M     │ ResNet early layers, MobileNet   │
  ├────────────┼──────────────────────────────────┤
  │ 50-200M    │ ResNet main layers               │
  ├────────────┼──────────────────────────────────┤
  │ 200M-1G    │ ResNet large layers, VGG, ViT    │
  ├────────────┼──────────────────────────────────┤
  │ >1G        │ VGG large convs, ViT-L attention │
  └────────────┴──────────────────────────────────┘
  Why Multiple Variants?

  - ResNet-18/34/50/101: Same architecture but different depths = different layer sizes
  - MobileNet-V2/V3-small/V3-large: Different efficiency optimizations
  - ViT-B/L: Different attention matrix sizes (B=768 hidden, L=1024 hidden)
  - VGG-11/16: Different number of large conv layers

  What's NOT Included (and why)

  - ResNet-152: Very similar to ResNet-101, adds runtime but little new data
  - VGG-19: Nearly identical to VGG-16
  - EfficientNet-B2+: Similar patterns to B0/B1
  - ViT-H/14: Very large, long runtime, similar patterns to ViT-L
  - YOLO/detection models: Different input sizes, could add later
  - Segmentation models: Same ops as classification, just larger activations

  The "Quick" Set Rationale

  The 6 quick models were chosen to hit each major category with one representative:
  - resnet18 - standard conv
  - resnet50 - bottleneck conv
  - mobilenet_v2 - depthwise
  - efficientnet_b0 - MBConv
  - vgg16 - large conv
  - vit_b_16 - attention/matmul

  This gives reasonable curve coverage in ~5 minutes vs ~30 minutes for the full set.

