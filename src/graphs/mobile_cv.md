# Summary

Added MobileNet and EfficientNet characterization to the pipeline.

  1. Enhanced Fusion Estimators (src/graphs/characterize/fused_ops.py)

  Added support for:
  - Depthwise separable convolutions (critical for MobileNet/EfficientNet)
  - Conv2d + ReLU6 (MobileNet activation)
  - Conv2d + Hardswish (MobileNetV3 activation)
  - Standalone Conv2d (for layers without activation)
  - Grouped convolutions (general case handling)

  The estimators now correctly calculate FLOPs for depthwise convolutions: 2 × K × K MACs per
  output instead of 2 × C_in × K × K.

  2. MobileNet Characterization (src/graphs/validation/test_mobilenet.py)

  Characterized 3 models across 6 hardware architectures:
  - MobileNet-V2: 1.87 GFLOPs, 3.50M params
  - MobileNet-V3-Small: 0.29 GFLOPs, 2.54M params (most efficient!)
  - MobileNet-V3-Large: 1.17 GFLOPs, 5.48M params

  Results: results/validation/mobilenet_results.csv

  3. EfficientNet Characterization (src/graphs/validation/test_efficientnet.py)

  Characterized 5 models across 6 hardware architectures:
  - EfficientNet-B0: 2.35 GFLOPs, 5.29M params
  - EfficientNet-B1: 3.41 GFLOPs, 7.79M params
  - EfficientNet-B2: 3.96 GFLOPs, 9.11M params
  - EfficientNet-V2-S: 17.52 GFLOPs, 21.46M params
  - EfficientNet-V2-M: 33.11 GFLOPs, 54.14M params

  Results: results/validation/efficientnet_results.csv

  4. Comprehensive Comparison Report (docs/validation/mobilenet_efficientnet_comparison.md)

  Created a 450-line analysis covering:
  - Model complexity comparison vs ResNet-18
  - Performance analysis across all 6 architectures (H100, TPU, KPU-T2/T100, CPUs)
  - Energy efficiency deep dive (MobileNet-V3-Small: 10× battery life!)
  - Edge deployment analysis (11.5K FPS on KPU-T2)
  - Use case recommendations for different scenarios
  - Memory footprint analysis
  - Architecture design patterns explained

  5. Key Findings

  Efficiency Champions:
  - MobileNet-V3-Small: 92% fewer FLOPs than ResNet-18, 10× battery life on edge
  - EfficientNet-B0: Best accuracy/efficiency balance (62% of ResNet-18 FLOPs)
  - Depthwise separable convolutions: 8-10× FLOP reduction

  Performance Highlights:
  - H100-PCIe: MobileNet-V3-Small achieves 4.3M FPS (11× faster than ResNet-18)
  - KPU-T2 (Edge IoT): MobileNet-V3-Small @ 11.5K FPS with 0.029J per inference
  - KPU-T100 (Edge Server): All models provide 100-19K× real-time margin @ 30 FPS

  6. Updated Documentation

  - Updated src/graphs/validation/README.md with new tests
  - Marked MobileNet and EfficientNet as completed in future work section
  - Added expected FLOP ranges for efficient architectures

