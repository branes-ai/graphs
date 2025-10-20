# Model Comparisons

Summary

  1. Comparison Tool 

  cli/compare_models.py - Compare multiple models side-by-side

  Key Metrics Provided:
  - Model Size: Parameters, trainable params, model size (MB), layers, max width
  - Computational Cost: MACs, FLOPs, FLOPs/param, params/MAC
  - Memory Traffic: Total, input, output, weights, bytes/FLOP
  - Arithmetic Intensity: Avg AI, compute-bound %, memory-bound %, classification
  - Efficiency Rankings: Best/worst in each category

  2. Documentation 

  docs/MODEL_COMPARISON_METRICS.md - Comprehensive guide explaining:
  - What each metric means
  - How to interpret values
  - When to use which metric
  - Trade-offs and recommendations
  - Real-world examples

  3. Usage Examples

  # Compare architectures
  python cli/compare_models.py resnet18 mobilenet_v2 efficientnet_b0

  # Compare family scaling
  python cli/compare_models.py resnet18 resnet34 resnet50 resnet101

  # Compare CNN vs Transformers
  python cli/compare_models.py resnet50 vit_b_16 swin_t

  # Sort by efficiency
  python cli/compare_models.py resnet18 mobilenet_v2 --sort-by efficiency

  4. Key Insights from Examples

  MobileNet vs ResNet-18:
  - MobileNet: 3.3× smaller, 6× less MACs, but low AI (4.68 - memory-bound)
  - ResNet: Higher AI (24.76 - balanced), better GPU utilization

  ResNet vs ViT vs Swin:
  - ResNet50: Most compute/param efficient (322 FLOPs/param)
  - Swin-T: Least FLOPs (5.93G), best memory efficiency
  - ViT-B16: Largest (86M params), most FLOPs (22G), highest AI (28.92)

  Recommended Metrics by Use Case

  Mobile/Edge Deployment:
  - Minimize: Parameters, MACs, model size
  - Metric: --sort-by params or --sort-by macs

  GPU/Server Deployment:
  - Maximize: AI (arithmetic intensity), FLOPs/param
  - Metric: --sort-by ai or --sort-by efficiency

  Memory-Constrained:
  - Minimize: Total memory, bytes/FLOP
  - Metric: --sort-by memory

  Latency-Critical:
  - Minimize: Layers (depth), MACs
  - Look at: Layer count, sequential dependencies

