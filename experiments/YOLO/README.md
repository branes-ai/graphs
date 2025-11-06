# YOLO Experiments

This directory contains experiments with YOLO models (YOLOv5, YOLOv8, YOLO11) for FX tracing, graph analysis, and profiling.

## Tools

### YOLO Profiler (`yolo_profiler.py`)

A specialized profiler for YOLO models that generates hierarchical tables with compute and memory characteristics.

**Features:**
- Hierarchical view of YOLO architecture (backbone, neck, head)
- Per-layer compute metrics (MACs, FLOPs)
- Memory breakdown (inputs, outputs, weights)
- Parameter counts and tensor shapes
- Bottleneck analysis (compute-bound vs memory-bound)
- **Intelligent name abbreviation** - Converts unwieldy FX node names like `getattr_getattr_L__yolo_model___model_23_cv3_0___0_____0___conv` into readable names like `model.22.cv3.0.0.conv` (see IMPROVEMENTS.md for details)

**Usage:**

```bash
# Basic profiling
python experiments/YOLO/yolo_profiler.py --model yolov8n.pt

# Show parameter and tensor shapes
python experiments/YOLO/yolo_profiler.py --model yolov8n.pt --showshape

# Custom input size (e.g., for different detection resolutions)
python experiments/YOLO/yolo_profiler.py --model yolov8s.pt --input-shape 1 3 320 320

# Profile larger models
python experiments/YOLO/yolo_profiler.py --model yolov8m.pt
python experiments/YOLO/yolo_profiler.py --model yolo11m.pt
```

**Output:**

The profiler generates three sections:

1. **Hierarchical Graph Profile**: Table showing each layer with:
   - Module name (abbreviated for readability)
   - Parameter count
   - Tensor shape (output dimensions, if `--showshape` is used)
   - MACs (multiply-accumulate operations)
   - FLOPs (floating-point operations)
   - Memory usage (input + output + weight tensors)

2. **Model Summary**: Overall statistics including:
   - Total parameters
   - Total FLOPs and MACs
   - Memory breakdown by tensor type

3. **Bottleneck Analysis**: Classification of operations as compute-bound or memory-bound

**Example Output:**

```
====================================================================================================
HIERARCHICAL GRAPH PROFILE
====================================================================================================

| Module                              | #Parameters          | MACs         | FLOPs        | Memory       |
|:------------------------------------|:---------------------|:-------------|:-------------|:-------------|
| model                               | 3.157M               | 4.372G       | 8.815G       | 398.86MB     |
|  L__yolo_model___model_0_conv       | 432                  | 44.237M      | 88.474M      | 11.47MB      |
|  L__yolo_model___model_0_bn         | 32                   |              | 8.192M       | 13.11MB      |
|  L__yolo_model___model_1_conv       | 4.608K               | 117.965M     | 235.930M     | 9.85MB       |
...

====================================================================================================
MODEL SUMMARY
====================================================================================================

Model: yolov8n.pt
Input shape: (1, 3, 640, 640)

Total parameters: 3.16M (3,157,200)
Total FLOPs: 8.815 GFLOPs (8,814,603,200)
Total MACs: 4.372 GMACs (4,371,993,600)

Memory breakdown:
  Input tensors:  204.33 MB
  Output tensors: 181.86 MB
  Weights:        12.67 MB
  Total:          398.86 MB

Graph structure:
  Subgraphs (fused ops): 194
  Average arithmetic intensity: 35.64 FLOPs/byte

Bottleneck analysis:
  Compute-bound ops: 60 (30.9%)
  Memory-bound ops:  134 (69.1%)
  (Threshold: AI > 10 FLOPs/byte)
```

## Other Experiments

### Tracing Scripts

- **`yolo8n_tracer.py`**: Basic Dynamo export with shape propagation
- **`yolo8n_tracer_v1.py`, `yolo8n_tracer_v2.py`**: Earlier tracing experiments
- **`dynamo_tracer.py`**: Standalone Dynamo tracing example
- **`module_tracer.py`**: Module-level tracing experiments

### Visualization Scripts

- **`yolo8n_viewer.py`**: Generate visual graph with torchview
- **`yolo8n_graph_viewer.py`**: Alternative graph visualization

## Requirements

```bash
pip install ultralytics torch torchvision
```

## Supported Models

The profiler works with any YOLO model supported by ultralytics:
- **YOLOv8**: n, s, m, l, x variants
- **YOLOv5**: n, s, m, l, x variants
- **YOLO11**: n, s, m, l, x variants
- **YOLOv9**: Various sizes
- And other YOLO architectures

Models are automatically downloaded on first use if not found locally.

## Notes

- **Warm-up Required**: The profiler includes a critical warm-up step that initializes YOLO's internal anchors and strides, preventing "Mutating module attribute" errors during Dynamo export.

- **Name Abbreviation**: Long FX node names (like `L__yolo_model___model_22_cv3_2___1___bn`) are automatically handled by the table formatter to show only the relevant parts.

- **Memory Estimates**: Memory values represent peak activation memory for single-image inference. Batch sizes > 1 will scale memory proportionally.

- **Arithmetic Intensity**: The ratio of compute (FLOPs) to memory traffic (bytes). Higher values indicate compute-bound operations that benefit from faster processors; lower values indicate memory-bound operations bottlenecked by bandwidth.
