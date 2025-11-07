"""
Dynamo Graph Extraction for YOLO Models

This module demonstrates how to trace YOLO object detection models using Dynamo.
YOLO models are notoriously difficult to trace with FX due to:
- Dynamic tensor operations
- Complex post-processing
- Data-dependent control flow
- Various output formats

Supported YOLO versions:
- YOLOv8 (Ultralytics)
- YOLOv11 (latest from Ultralytics)

Requirements:
    pip install ultralytics torch torchvision

Usage:
    python experiments/dynamo/trace_yolo.py --model yolov8n
    python experiments/dynamo/trace_yolo.py --model yolov8s --save-graph output/yolo_graph.py
"""

import torch
import torch._dynamo as dynamo
from typing import Optional, List, Dict, Any
import argparse
from pathlib import Path
import sys


class YOLOGraphExtractor:
    """
    Specialized extractor for YOLO models with detection-specific analysis.
    """

    def __init__(self, verbose: bool = True):
        self.graphs: List[torch.fx.Graph] = []
        self.graph_modules: List[torch.fx.GraphModule] = []
        self.subgraph_count = 0
        self.verbose = verbose

        # YOLO-specific tracking
        self.conv_ops = 0
        self.detection_heads = 0
        self.concat_ops = 0

    def __call__(self, gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
        """Custom backend for YOLO graph extraction."""
        self.subgraph_count += 1
        self.graphs.append(gm.graph)
        self.graph_modules.append(gm)

        if self.verbose:
            print(f"\n{'='*80}")
            print(f"Graph Partition {self.subgraph_count}")
            print(f"{'='*80}")
            print(f"Total nodes: {len(list(gm.graph.nodes))}")

            # Analyze YOLO-specific patterns
            self._analyze_yolo_patterns(gm.graph)

        return gm.forward

    def _analyze_yolo_patterns(self, graph: torch.fx.Graph):
        """Analyze YOLO-specific operation patterns."""
        conv_count = 0
        concat_count = 0
        detect_patterns = 0
        max_pool_count = 0
        upsample_count = 0

        for node in graph.nodes:
            if node.op == 'call_function':
                target_name = self._get_target_name(node.target)

                # Count convolutions (including depthwise)
                if 'conv' in target_name.lower():
                    conv_count += 1

                # Count concatenations (FPN/PAN feature fusion)
                elif target_name in ['cat', 'concat']:
                    concat_count += 1

                # Upsampling (for FPN)
                elif 'upsample' in target_name.lower() or 'interpolate' in target_name.lower():
                    upsample_count += 1

                # MaxPool
                elif 'max_pool' in target_name.lower():
                    max_pool_count += 1

            elif node.op == 'call_module':
                # Detection head modules
                if 'detect' in str(node.target).lower():
                    detect_patterns += 1

        self.conv_ops += conv_count
        self.concat_ops += concat_count

        if self.verbose:
            print("\nYOLO Operation Analysis:")
            print(f"  Convolutions: {conv_count}")
            print(f"  Concatenations (feature fusion): {concat_count}")
            print(f"  Upsampling ops: {upsample_count}")
            print(f"  MaxPool ops: {max_pool_count}")
            print(f"  Detection patterns: {detect_patterns}")

            # Estimate architecture stage
            if concat_count > 0 or upsample_count > 0:
                print("\n  → Likely FPN/PAN neck operations")
            if detect_patterns > 0:
                print("  → Detection head found")

    def _get_target_name(self, target) -> str:
        """Extract readable name from target."""
        if isinstance(target, str):
            return target
        elif hasattr(target, '__name__'):
            return target.__name__
        else:
            target_str = str(target)
            if "'" in target_str:
                return target_str.split("'")[1].split('.')[-1]
            return target_str

    def print_summary(self):
        """Print overall YOLO model summary."""
        print("\n" + "="*80)
        print("YOLO MODEL SUMMARY")
        print("="*80)
        print(f"Graph partitions: {self.subgraph_count}")
        print(f"Total convolutions: {self.conv_ops}")
        print(f"Total concatenations: {self.concat_ops}")

        total_nodes = sum(len(list(g.nodes)) for g in self.graphs)
        print(f"Total graph nodes: {total_nodes}")


def trace_yolo_model(
    model_name: str = "yolov8n.pt",
    img_size: int = 640,
    verbose: bool = True
) -> Optional[YOLOGraphExtractor]:
    """
    Trace a YOLO model using Dynamo.

    Args:
        model_name: YOLO model name (e.g., 'yolov8n', 'yolov8s', 'yolov11n')
        img_size: Input image size
        verbose: Print detailed information

    Returns:
        YOLOGraphExtractor with captured graphs
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Error: ultralytics not installed")
        print("Install with: pip install ultralytics")
        return None

    print(f"\nLoading YOLO model: {model_name}")
    print("="*80)

    # Load model
    model = YOLO(model_name)
    print(f"Model loaded: {model_name}")

    # Get the PyTorch model (not the wrapper)
    pytorch_model = model.model
    pytorch_model.eval()

    # Count parameters
    total_params = sum(p.numel() for p in pytorch_model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Create example input
    example_input = torch.randn(1, 3, img_size, img_size)
    print(f"Input shape: {example_input.shape}")

    # Create extractor
    extractor = YOLOGraphExtractor(verbose=verbose)

    # Reset dynamo
    dynamo.reset()

    print("\nStarting Dynamo tracing...")
    print("Note: YOLO models may have multiple graph breaks due to complexity")
    print("="*80)

    # Compile with custom backend
    try:
        compiled_model = torch.compile(
            pytorch_model,
            backend=extractor,
            fullgraph=False,  # Allow graph breaks
        )

        # Run forward pass to trigger tracing
        with torch.no_grad():
            _ = compiled_model(example_input)

        if verbose:
            extractor.print_summary()

        return extractor

    except Exception as e:
        print(f"\nError during tracing: {e}")
        import traceback
        traceback.print_exc()
        return None


def analyze_yolo_architecture(extractor: YOLOGraphExtractor):
    """
    Analyze YOLO architecture from extracted graphs.

    Identifies:
    - Backbone (feature extraction)
    - Neck (FPN/PAN feature fusion)
    - Head (detection layers)
    """
    print("\n" + "="*80)
    print("YOLO ARCHITECTURE ANALYSIS")
    print("="*80)

    if not extractor or len(extractor.graphs) == 0:
        print("No graphs to analyze")
        return

    # Analyze each partition
    for i, graph in enumerate(extractor.graphs):
        print(f"\nPartition {i+1}:")

        conv_count = 0
        concat_count = 0
        pool_count = 0

        for node in graph.nodes:
            if node.op == 'call_function':
                name = str(node.target).lower()
                if 'conv' in name:
                    conv_count += 1
                elif 'cat' in name or 'concat' in name:
                    concat_count += 1
                elif 'pool' in name:
                    pool_count += 1

        # Heuristic classification
        stage = "Unknown"
        if concat_count > 2:
            stage = "Neck (FPN/PAN)"
        elif pool_count > 0 and conv_count > 5:
            stage = "Backbone (feature extraction)"
        elif i == len(extractor.graphs) - 1:
            stage = "Head (detection)"

        print(f"  Estimated stage: {stage}")
        print(f"  Convolutions: {conv_count}")
        print(f"  Concatenations: {concat_count}")
        print(f"  Pooling ops: {pool_count}")


def compare_yolo_variants():
    """
    Compare different YOLO model variants.
    """
    print("\n" + "="*80)
    print("COMPARING YOLO VARIANTS")
    print("="*80)

    variants = ['yolov8n.pt', 'yolov8s.pt']
    results = {}

    for variant in variants:
        print(f"\n\nTracing {variant}...")
        extractor = trace_yolo_model(variant, verbose=False)

        if extractor:
            results[variant] = {
                'partitions': extractor.subgraph_count,
                'conv_ops': extractor.conv_ops,
                'concat_ops': extractor.concat_ops,
                'total_nodes': sum(len(list(g.nodes)) for g in extractor.graphs)
            }

    # Print comparison table
    print("\n" + "="*80)
    print("COMPARISON TABLE")
    print("="*80)
    print(f"{'Variant':<15} {'Partitions':<12} {'Convs':<10} {'Concats':<10} {'Total Nodes':<12}")
    print("-"*80)

    for variant, stats in results.items():
        print(f"{variant:<15} {stats['partitions']:<12} {stats['conv_ops']:<10} "
              f"{stats['concat_ops']:<10} {stats['total_nodes']:<12}")


def save_graph_code(extractor: YOLOGraphExtractor, output_path: str):
    """
    Save extracted YOLO graphs to a Python file.

    Args:
        extractor: YOLOGraphExtractor with captured graphs
        output_path: Path to save the graph code
    """
    if not extractor or len(extractor.graph_modules) == 0:
        print("No graphs to save")
        return

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        f.write("# Auto-generated YOLO graph code from Dynamo extraction\n")
        f.write(f"# Total partitions: {extractor.subgraph_count}\n")
        f.write(f"# Total convolutions: {extractor.conv_ops}\n")
        f.write(f"# Total concatenations: {extractor.concat_ops}\n\n")
        f.write("import torch\n")
        f.write("import torch.fx as fx\n\n")

        for i, gm in enumerate(extractor.graph_modules):
            f.write(f"\n{'='*80}\n")
            f.write(f"# Partition {i+1}\n")
            f.write(f"{'='*80}\n\n")
            f.write(gm.code)
            f.write("\n\n")

    print(f"\nYOLO graph code saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Trace YOLO models with Dynamo"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="YOLO model to trace (e.g., yolov8n.pt, yolov8s.pt, yolov11n.pt)"
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=640,
        help="Input image size"
    )
    parser.add_argument(
        "--save-graph",
        type=str,
        default=None,
        help="Save extracted graph code to file"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare multiple YOLO variants"
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Perform detailed architecture analysis"
    )

    args = parser.parse_args()

    if args.compare:
        compare_yolo_variants()
    else:
        # Trace single model
        extractor = trace_yolo_model(
            model_name=args.model,
            img_size=args.img_size,
            verbose=True
        )

        if extractor:
            # Architecture analysis
            if args.analyze:
                analyze_yolo_architecture(extractor)

            # Save graph code
            if args.save_graph:
                save_graph_code(extractor, args.save_graph)


if __name__ == "__main__":
    main()
