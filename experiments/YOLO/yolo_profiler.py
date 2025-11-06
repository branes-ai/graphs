#!/usr/bin/env python
"""
YOLO Graph Profiler - Hierarchical Table for YOLO Models
=========================================================

Profiles YOLO models (YOLOv8, YOLOv5, etc.) by characterizing each operator
in the computational graph with readable hierarchical tables:
- Execution order and hierarchical structure
- Computational requirements (MACs, FLOPs)
- Memory bandwidth demands
- Parameter counts and tensor shapes
- Operation-level resource analysis

Usage:
    python experiments/YOLO/yolo_profiler.py --model yolov8n.pt
    python experiments/YOLO/yolo_profiler.py --model yolov8n.pt --showshape
    python experiments/YOLO/yolo_profiler.py --model yolov8s.pt --input-shape 1 3 640 640
"""

import torch
import torch._dynamo
import argparse
import sys
from pathlib import Path

# Add src to path for imports
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root / "src"))

from ultralytics import YOLO
from torch.fx.passes.shape_prop import ShapeProp
from graphs.transform.partitioning import GraphPartitioner
from graphs.hardware.table_formatter import format_partition_table


def load_and_trace_yolo(model_path: str, input_shape: tuple) -> tuple:
    """
    Load YOLO model and trace it with PyTorch Dynamo

    Args:
        model_path: Path to YOLO .pt file (e.g., 'yolov8n.pt')
        input_shape: Input tensor shape (batch, channels, height, width)

    Returns:
        (traced_module, dummy_input) tuple
    """
    print(f"\n[1/5] Loading YOLO model from {model_path}...")

    # Load YOLO model and extract the core torch.nn.Module
    yolo_model = YOLO(model_path).model.eval()
    dummy_input = torch.randn(*input_shape)

    # CRITICAL: Warm-up to prevent "Mutating module attribute" errors
    # This initializes internal YOLO attributes (anchors, strides)
    print("[2/5] Warming up model (initializing anchors/strides)...")
    with torch.no_grad():
        _ = yolo_model(dummy_input)

    # Define wrapper function for export
    def forward_fn(x):
        return yolo_model(x)

    # Trace with Dynamo
    print("[3/5] Tracing with PyTorch Dynamo (torch._dynamo.export)...")
    try:
        traced_module, guards = torch._dynamo.export(forward_fn, dummy_input)
    except Exception as e:
        print(f"Error during Dynamo export: {e}")
        print("\nTip: Some YOLO models may have compatibility issues with Dynamo.")
        print("     Try using a different YOLO version or model size.")
        raise

    # Propagate shapes through the graph
    print("[4/5] Propagating tensor shapes through FX graph...")
    ShapeProp(traced_module).propagate(dummy_input)

    return traced_module, dummy_input


def profile_yolo(model_path: str, show_shapes: bool = False, input_shape: tuple = (1, 3, 640, 640)):
    """
    Profile YOLO model and display hierarchical table

    Args:
        model_path: Path to YOLO .pt file
        show_shapes: If True, show parameter and tensor shapes
        input_shape: Input tensor shape
    """
    print("=" * 100)
    print(f"YOLO MODEL PROFILER: {model_path}")
    print("=" * 100)

    # Load and trace
    try:
        traced_module, dummy_input = load_and_trace_yolo(model_path, input_shape)
    except Exception as e:
        print(f"\nFailed to trace model: {e}")
        return

    # Partition the graph
    print("[5/5] Running graph partitioner...")
    partitioner = GraphPartitioner()
    report = partitioner.partition(traced_module)

    # Format hierarchical table
    print("\n" + "=" * 100)
    print("HIERARCHICAL GRAPH PROFILE")
    print("=" * 100)
    print()

    table = format_partition_table(traced_module, report, show_shapes=show_shapes)
    print(table)

    # Summary statistics
    print("\n" + "=" * 100)
    print("MODEL SUMMARY")
    print("=" * 100)

    # Count parameters
    total_params = sum(p.numel() for p in traced_module.parameters())
    print(f"\nModel: {model_path}")
    print(f"Input shape: {input_shape}")
    print(f"\nTotal parameters: {total_params / 1e6:.2f}M ({total_params:,})")
    print(f"Total FLOPs: {report.total_flops / 1e9:.3f} GFLOPs ({report.total_flops:,})")
    print(f"Total MACs: {report.total_macs / 1e9:.3f} GMACs ({report.total_macs:,})")

    # Memory breakdown
    total_input = sum(sg.total_input_bytes for sg in report.subgraphs)
    total_output = sum(sg.total_output_bytes for sg in report.subgraphs)
    total_weight = sum(sg.total_weight_bytes for sg in report.subgraphs)
    total_memory = total_input + total_output + total_weight

    print(f"\nMemory breakdown:")
    print(f"  Input tensors:  {total_input / 1e6:.2f} MB")
    print(f"  Output tensors: {total_output / 1e6:.2f} MB")
    print(f"  Weights:        {total_weight / 1e6:.2f} MB")
    print(f"  Total:          {total_memory / 1e6:.2f} MB")

    print(f"\nGraph structure:")
    print(f"  Subgraphs (fused ops): {len(report.subgraphs)}")
    print(f"  Average arithmetic intensity: {report.average_arithmetic_intensity:.2f} FLOPs/byte")

    # Bottleneck analysis
    compute_bound = sum(1 for sg in report.subgraphs if sg.arithmetic_intensity > 10)
    memory_bound = len(report.subgraphs) - compute_bound
    print(f"\nBottleneck analysis:")
    print(f"  Compute-bound ops: {compute_bound} ({compute_bound/len(report.subgraphs)*100:.1f}%)")
    print(f"  Memory-bound ops:  {memory_bound} ({memory_bound/len(report.subgraphs)*100:.1f}%)")
    print(f"  (Threshold: AI > 10 FLOPs/byte)")


def main():
    parser = argparse.ArgumentParser(
        description='Profile YOLO models with hierarchical tables showing compute and memory characteristics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Profile YOLOv8n
  python experiments/YOLO/yolo_profiler.py --model yolov8n.pt

  # Show tensor shapes and parameter details
  python experiments/YOLO/yolo_profiler.py --model yolov8n.pt --showshape

  # Profile YOLOv8s with custom input size
  python experiments/YOLO/yolo_profiler.py --model yolov8s.pt --input-shape 1 3 320 320

  # Profile YOLOv8m
  python experiments/YOLO/yolo_profiler.py --model yolov8m.pt

Note:
  - You need ultralytics installed: pip install ultralytics
  - Download models first or they'll be auto-downloaded on first run
  - Supported models: yolov8n, yolov8s, yolov8m, yolov8l, yolov8x (and similar for v5, v9, etc.)
        """
    )

    parser.add_argument('--model', type=str, required=True,
                       help='Path to YOLO .pt file (e.g., yolov8n.pt, yolov8s.pt)')

    parser.add_argument('--showshape', action='store_true',
                       help='Show parameter shapes (weight/bias dimensions) and tensor shapes')

    parser.add_argument('--input-shape', type=int, nargs=4, default=[1, 3, 640, 640],
                       metavar=('B', 'C', 'H', 'W'),
                       help='Input tensor shape (default: 1 3 640 640)')

    args = parser.parse_args()

    # Check if model file exists
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Warning: Model file '{args.model}' not found.")
        print(f"The ultralytics library will attempt to download it automatically.")
        print()

    profile_yolo(args.model, show_shapes=args.showshape, input_shape=tuple(args.input_shape))


if __name__ == "__main__":
    main()
