#!/usr/bin/env python
"""
Graph Explorer CLI
==================

Command-line tool to explore FX computational graphs interactively.
Provides three modes: model discovery, graph summary, and detailed visualization.

Supports:
- TorchVision models (ResNet, MobileNet, EfficientNet, ViT, etc.)
- HuggingFace transformers (BERT, GPT-2, etc.)
- YOLO models (.pt files)
- DETR models

Usage:
    # Discover available models
    python cli/graph_explorer.py

    # Get model summary statistics (vision models)
    python cli/graph_explorer.py --model resnet18

    # Get model summary statistics (transformers)
    python cli/graph_explorer.py --model bert-base-uncased --seq-len 128

    # Get model summary statistics (YOLO)
    python cli/graph_explorer.py --model yolov8n.pt

    # Visualize first 50 nodes
    python cli/graph_explorer.py --model resnet18 --max-nodes 50

    # Visualize specific range (nodes 20-50)
    python cli/graph_explorer.py --model resnet18 --start 20 --end 50

    # Investigate around node 35 (±10 nodes context)
    python cli/graph_explorer.py --model resnet18 --around 35 --context 10

Command-line Options:
    --model:       Model name (resnet18, bert-base-uncased, yolov8n.pt, etc.)
    --start:       Start node index (0-based)
    --end:         End node index (exclusive)
    --around:      Center node for context view
    --context:     Number of nodes before/after center (default: 10)
    --max-nodes:   Maximum nodes to display (from start)
    --input-shape: Input tensor shape for vision models (default: 1,3,224,224)
    --seq-len:     Sequence length for transformer models (default: 128)
    --output:      Save visualization to file

"""

import torch
import torch._dynamo
import torchvision.models as models
from torch.fx import symbolic_trace, GraphModule
from torch.fx.passes.shape_prop import ShapeProp
import argparse
import sys
import warnings
from pathlib import Path
from typing import Optional, Tuple, Union

from graphs.transform.partitioning.graph_partitioner import GraphPartitioner


def load_yolo_model(model_path: str) -> torch.nn.Module:
    """Load a YOLO model from .pt file"""
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError(
            "YOLO models require ultralytics. Install with: pip install ultralytics"
        )

    yolo = YOLO(model_path)
    model = yolo.model.eval()
    return model


def load_transformer_model(model_name: str) -> Tuple[torch.nn.Module, str]:
    """
    Load a transformer model from Hugging Face

    Returns:
        (model, input_type) where input_type is 'tokens' or 'detr'
    """
    try:
        from transformers import AutoModel
    except ImportError:
        raise ImportError(
            "Transformer models require transformers. Install with: pip install transformers"
        )

    print(f"Loading transformer model '{model_name}' from Hugging Face...")

    # DETR and similar vision-transformer models need special handling
    if 'detr' in model_name.lower():
        from transformers import DetrForObjectDetection
        model = DetrForObjectDetection.from_pretrained(model_name)
        model.eval()
        return model, 'detr'

    # Default: text transformer
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    return model, 'tokens'


def trace_model_hybrid(
    model: torch.nn.Module,
    example_inputs: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
    model_name: str = "model"
) -> Tuple[GraphModule, str]:
    """
    Trace a model using hybrid strategy:
    1. Try standard FX symbolic_trace (fast)
    2. Fall back to Dynamo export (robust)

    Args:
        model: Model to trace
        example_inputs: Example input(s) - single tensor or tuple of tensors
        model_name: Name for error messages

    Returns:
        (traced_graph_module, method_used)
    """

    # Normalize inputs to tuple
    if not isinstance(example_inputs, tuple):
        example_inputs = (example_inputs,)

    # Always warm-up first (safe for all models, critical for some)
    print("[1/3] Warming up model (initializing any lazy modules)...")
    with torch.no_grad():
        try:
            _ = model(*example_inputs)
        except Exception as e:
            print(f"  Note: Warm-up forward pass failed: {e}")
            print("  Continuing with tracing anyway...")

    # Try standard FX symbolic_trace first (preferred)
    print("[2/3] Attempting standard FX symbolic_trace...")
    try:
        traced = symbolic_trace(model)
        print("  ✓ Standard FX trace successful")
        return traced, "symbolic_trace"
    except Exception as e:
        print(f"  ✗ Standard FX trace failed: {type(e).__name__}")
        print("  Falling back to Dynamo export...")

    # Fall back to Dynamo export
    print("[2/3] Using PyTorch Dynamo export (more robust)...")
    try:
        # Create wrapper with correct signature
        if len(example_inputs) == 1:
            def forward_fn(x):
                return model(x)
        elif len(example_inputs) == 2:
            def forward_fn(x1, x2):
                return model(x1, x2)
        else:
            def forward_fn(*args):
                return model(*args)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            traced, guards = torch._dynamo.export(forward_fn, *example_inputs)

        print("  ✓ Dynamo export successful")
        return traced, "dynamo_export"
    except Exception as e:
        print(f"  ✗ Dynamo export failed: {e}")
        raise RuntimeError(
            f"Failed to trace model '{model_name}' with both FX and Dynamo. "
            "The model may not be compatible with graph tracing."
        )


class GraphExplorerCLI:
    """Command-line interface for exploring FX computational graphs"""

    SUPPORTED_MODELS = {
        'resnet18': models.resnet18,
        'resnet34': models.resnet34,
        'resnet50': models.resnet50,
        'resnet101': models.resnet101,
        'resnet152': models.resnet152,
        'mobilenet_v2': models.mobilenet_v2,
        'mobilenet_v3_small': models.mobilenet_v3_small,
        'mobilenet_v3_large': models.mobilenet_v3_large,
        'efficientnet_b0': models.efficientnet_b0,
        'efficientnet_b1': models.efficientnet_b1,
        'efficientnet_b2': models.efficientnet_b2,
        'efficientnet_b3': models.efficientnet_b3,
        'efficientnet_b4': models.efficientnet_b4,
        'vit_b_16': models.vit_b_16,
        'vit_b_32': models.vit_b_32,
        'vit_l_16': models.vit_l_16,
        'swin_t': models.swin_t,
        'swin_s': models.swin_s,
        'swin_b': models.swin_b,
        'convnext_tiny': models.convnext_tiny,
        'convnext_small': models.convnext_small,
        'convnext_base': models.convnext_base,
    }

    def __init__(self):
        self.fx_graph = None
        self.model_name = None
        self.partitioner = GraphPartitioner()
        self.report = None

    @classmethod
    def show_model_list(cls):
        """Display organized list of supported models"""
        print("=" * 80)
        print("ERROR: Please specify a model with --model")
        print("=" * 80)
        print()
        print("Usage: ./cli/graph_explorer.py --model MODEL_NAME [OPTIONS]")
        print()
        print("=" * 80)
        print("SUPPORTED MODEL TYPES")
        print("=" * 80)
        print()

        print("1. TorchVision Models (built-in)")
        print("=" * 80)
        # Organize by family
        families = {
            "ResNet": [k for k in cls.SUPPORTED_MODELS.keys() if k.startswith('resnet')],
            "MobileNet": [k for k in cls.SUPPORTED_MODELS.keys() if k.startswith('mobilenet')],
            "EfficientNet": [k for k in cls.SUPPORTED_MODELS.keys() if k.startswith('efficientnet')],
            "Vision Transformer (ViT)": [k for k in cls.SUPPORTED_MODELS.keys() if k.startswith('vit')],
            "Swin Transformer": [k for k in cls.SUPPORTED_MODELS.keys() if k.startswith('swin')],
            "ConvNeXt": [k for k in cls.SUPPORTED_MODELS.keys() if k.startswith('convnext')],
        }

        for family, models in families.items():
            if models:
                print(f"\n{family}:")
                for model in sorted(models):
                    print(f"  - {model}")

        print("\n")
        print("2. HuggingFace Transformers")
        print("=" * 80)
        print("\nSupports any HuggingFace transformer model:")
        print("  - BERT family: bert-base-uncased, roberta-base, albert-base-v2")
        print("  - GPT family: gpt2, distilgpt2, EleutherAI/gpt-neo-125m")
        print("  - DETR: facebook/detr-resnet-50")
        print("\nDiscovery tool:")
        print("  python cli/discover_transformers.py")

        print("\n")
        print("3. YOLO Models")
        print("=" * 80)
        print("\nSupports .pt file paths:")
        print("  - yolov8n.pt, yolov8s.pt, yolov8m.pt, etc.")
        print("\nDiscovery tool:")
        print("  python cli/discover_models.py")

        print("\n")
        print("=" * 80)
        print("EXAMPLES")
        print("=" * 80)
        print()
        print("# TorchVision model summary")
        print("./cli/graph_explorer.py --model resnet18")
        print()
        print("# Transformer model summary")
        print("./cli/graph_explorer.py --model bert-base-uncased --seq-len 128")
        print()
        print("# YOLO model summary")
        print("./cli/graph_explorer.py --model yolov8n.pt")
        print()
        print("# Visualize specific range")
        print("./cli/graph_explorer.py --model resnet18 --max-nodes 20")
        print()
        print("# Investigate around specific node")
        print("./cli/graph_explorer.py --model resnet18 --around 35 --context 10")
        print()

    def show_summary(self):
        """Display model summary without full visualization"""
        print()
        print("=" * 80)
        print(f"MODEL SUMMARY: {self.model_name}")
        print("=" * 80)
        print()

        # Basic statistics
        total_nodes = len(list(self.fx_graph.graph.nodes))
        print(f"Total FX Nodes:        {total_nodes}")
        print(f"Partitioned Subgraphs: {self.report.total_subgraphs}")
        print(f"Nodes Not Partitioned: {total_nodes - self.report.total_subgraphs}")
        print()

        # Computation statistics
        print(f"Total FLOPs:           {self.report.total_flops / 1e9:.2f} GFLOPs")
        print(f"Total MACs:            {self.report.total_macs / 1e6:.2f} M")
        print(f"Total Memory Traffic:  {self.report.total_memory_traffic / 1e6:.2f} MB")
        print()

        # Arithmetic intensity
        print(f"Arithmetic Intensity:")
        print(f"  Average: {self.report.average_arithmetic_intensity:.1f} FLOPs/byte")
        print(f"  Range:   {self.report.min_arithmetic_intensity:.1f} - {self.report.max_arithmetic_intensity:.1f} FLOPs/byte")
        print()

        # Bottleneck distribution
        if self.report.bottleneck_distribution:
            print("Bottleneck Distribution:")
            total_bottlenecks = sum(self.report.bottleneck_distribution.values())
            for bottleneck, count in sorted(self.report.bottleneck_distribution.items()):
                percentage = (count / total_bottlenecks) * 100
                print(f"  {bottleneck:20s}: {count:3d} ({percentage:5.1f}%)")
            print()

        # Operation type distribution
        if self.report.operation_type_counts:
            print("Operation Type Distribution:")
            total_ops = sum(self.report.operation_type_counts.values())
            # Show top 10 most common
            sorted_ops = sorted(self.report.operation_type_counts.items(),
                              key=lambda x: x[1], reverse=True)[:10]
            for op_type, count in sorted_ops:
                percentage = (count / total_ops) * 100
                print(f"  {op_type:20s}: {count:3d} ({percentage:5.1f}%)")
            if len(self.report.operation_type_counts) > 10:
                print(f"  ... and {len(self.report.operation_type_counts) - 10} more")
            print()

        # Partition reason distribution
        if self.report.partition_reason_distribution:
            print("Partition Reason Distribution:")
            total_reasons = sum(self.report.partition_reason_distribution.values())
            for reason, count in sorted(self.report.partition_reason_distribution.items()):
                percentage = (count / total_reasons) * 100
                print(f"  {reason:30s}: {count:3d} ({percentage:5.1f}%)")
            print()

        # Guidance
        print("=" * 80)
        print("NEXT STEPS: Visualize Specific Sections")
        print("=" * 80)
        print()
        print("The graph has too many nodes to display all at once.")
        print("Use one of these options to visualize specific sections:")
        print()
        print("# Show first 20 nodes")
        print(f"./cli/graph_explorer.py --model {self.model_name} --max-nodes 20")
        print()
        print("# Show nodes 20-50")
        print(f"./cli/graph_explorer.py --model {self.model_name} --start 20 --end 50")
        print()
        print("# Investigate around node 35 (±10 context)")
        print(f"./cli/graph_explorer.py --model {self.model_name} --around 35 --context 10")
        print()
        print("# Save full visualization to file")
        print(f"./cli/graph_explorer.py --model {self.model_name} --output full_viz.txt")
        print()

    def load_and_trace_model(self, model_name: str, input_shape=(1, 3, 224, 224), seq_len: int = 128):
        """Load model and trace with hybrid FX/Dynamo approach"""
        print("=" * 80)
        print(f"LOADING MODEL: {model_name}")
        print("=" * 80)
        print()

        # Determine model type and load
        input_type = 'image'  # Default to image tensors

        # Try registry first (TorchVision models)
        if model_name in self.SUPPORTED_MODELS:
            print(f"Loading '{model_name}' from TorchVision registry...")
            model_obj = self.SUPPORTED_MODELS[model_name](weights=None)
            model_obj.eval()
            input_type = 'image'
        else:
            # Check if it's a file path (YOLO)
            path = Path(model_name)
            if path.exists():
                print(f"Loading model from '{model_name}'...")
                model_obj = load_yolo_model(model_name)
                input_type = 'image'
            else:
                # Assume it's a HuggingFace model
                model_obj, input_type = load_transformer_model(model_name)

        self.model_name = model_name

        # Create example inputs based on model type
        if input_type == 'tokens':
            # Transformer models need token IDs
            batch_size = input_shape[0] if input_shape else 1

            # Get actual vocabulary size from model to avoid index errors
            # Different models have different vocab sizes:
            # - bert-base-uncased: 30,522
            # - bert-base-cased: 28,996
            # - gpt2: 50,257
            # - roberta-base: 50,265
            vocab_size = model_obj.config.vocab_size
            print(f"Model vocabulary size: {vocab_size}")

            # Generate random token IDs within valid range [0, vocab_size)
            input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

            # Check if model is BERT-style or GPT-style
            is_gpt_style = 'gpt' in model_name.lower()

            if is_gpt_style:
                example_inputs = input_ids
                print(f"Input type: Tokens (batch_size={batch_size}, seq_len={seq_len}) [GPT-style]")
            else:
                attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
                example_inputs = (input_ids, attention_mask)
                print(f"Input type: Tokens (batch_size={batch_size}, seq_len={seq_len}, with attention_mask) [BERT-style]")
        elif input_type == 'detr':
            # DETR models need image tensors with pixel_values keyword
            example_inputs = torch.randn(*input_shape)
            print(f"Input type: Image tensor {input_shape} [DETR - vision transformer]")

            # Wrap DETR to convert positional to keyword argument
            original_model = model_obj
            class DetrWrapper(torch.nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.model = model

                def forward(self, x):
                    return self.model(pixel_values=x)

            model_obj = DetrWrapper(original_model)
        else:
            # Standard vision models need image tensors
            example_inputs = torch.randn(*input_shape)
            print(f"Input type: Image tensor {input_shape}")

        # Trace using hybrid strategy
        try:
            self.fx_graph, method = trace_model_hybrid(model_obj, example_inputs, model_name)
            print(f"Tracing method: {method}")
        except Exception as e:
            print(f"\nFailed to trace model: {e}")
            sys.exit(1)

        # Propagate shapes
        print("[3/3] Propagating tensor shapes through graph...")
        if isinstance(example_inputs, tuple):
            ShapeProp(self.fx_graph).propagate(*example_inputs)
        else:
            ShapeProp(self.fx_graph).propagate(example_inputs)

        print("Partitioning graph...")
        self.report = self.partitioner.partition(self.fx_graph)

        total_nodes = len(list(self.fx_graph.graph.nodes))
        print(f"Created {self.report.total_subgraphs} subgraphs from {total_nodes} FX nodes")
        print()

        return self.fx_graph, self.report

    def determine_range(self, args) -> Tuple[Optional[int], Optional[int]]:
        """Determine start/end range based on arguments

        Note: User-provided node numbers are 1-based (display numbering).
        This method converts them to 0-based array indices for slicing.
        """
        total_nodes = len(list(self.fx_graph.graph.nodes))

        # Priority 1: --around with --context
        if args.around is not None:
            context = args.context if args.context is not None else 10
            # Convert 1-based display node number to 0-based index
            center_idx = args.around - 1
            start = max(0, center_idx - context)
            end = min(total_nodes, center_idx + context + 1)
            print(f"Showing nodes around #{args.around} (context: ±{context} nodes)")
            print(f"Range: nodes {start+1} to {end} (total: {end-start} nodes)")
            return start, end

        # Priority 2: --start and/or --end
        if args.start is not None or args.end is not None:
            # Convert 1-based display numbers to 0-based indices
            # start: subtract 1 (node 5 -> index 4)
            # end: keep as-is (node 10 -> slice index 10, since slicing is exclusive on end)
            start = (args.start - 1) if args.start is not None else 0
            end = args.end if args.end is not None else total_nodes
            start = max(0, start)
            end = min(total_nodes, end)
            print(f"Showing nodes {start+1} to {end} (total: {end-start} nodes)")
            return start, end

        # Priority 3: --max-nodes
        if args.max_nodes is not None:
            start = 0
            end = min(args.max_nodes, total_nodes)
            print(f"Showing first {end} nodes")
            return start, end

        # Default: show all nodes
        print(f"Showing all {total_nodes} nodes")
        return None, None

    def visualize(self, args):
        """Generate and display visualization"""
        print()
        print("=" * 80)
        print("GENERATING VISUALIZATION")
        print("=" * 80)
        print()

        # Determine range
        start, end = self.determine_range(args)

        # Generate visualization with start/end range
        visualization = self.partitioner.visualize_partitioning(
            self.fx_graph,
            start=start,
            end=end
        )

        # Display or save
        if args.output:
            with open(args.output, 'w') as f:
                f.write(visualization)
            print(f"Visualization saved to: {args.output}")
            print(f"View with: cat {args.output}")
        else:
            print(visualization)

    def show_tips(self, args):
        """Display usage tips"""
        print()
        print("=" * 80)
        print("TIPS")
        print("=" * 80)
        print()
        print("Range Selection:")
        print(f"  # Show nodes 20-50")
        print(f"  python cli/graph_explorer.py --model {self.model_name} --start 20 --end 50")
        print()
        print(f"  # Investigate around node 35 (±10 context)")
        print(f"  python cli/graph_explorer.py --model {self.model_name} --around 35 --context 10")
        print()
        print(f"  # Show first 100 nodes")
        print(f"  python cli/graph_explorer.py --model {self.model_name} --max-nodes 100")
        print()
        print("Output to File:")
        print(f"  python cli/graph_explorer.py --model {self.model_name} --output viz.txt")
        print()
        print("Different Models:")
        print(f"  python cli/graph_explorer.py --model mobilenet_v2")
        print(f"  python cli/graph_explorer.py --model vit_b_16")
        print()


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description='Visualize FX graph partitioning side-by-side',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize all nodes
  %(prog)s --model resnet18

  # Visualize first 50 nodes
  %(prog)s --model resnet18 --max-nodes 50

  # Visualize specific range
  %(prog)s --model resnet18 --start 20 --end 50

  # Investigate around node 35
  %(prog)s --model resnet18 --around 35 --context 10

  # Save to file
  %(prog)s --model mobilenet_v2 --output viz.txt
        """
    )

    # Model selection
    parser.add_argument('--model', type=str, default=None,
                        help='Model name (required)')
    parser.add_argument('--input-shape', type=str, default='1,3,224,224',
                        help='Input tensor shape as comma-separated values for vision models (default: 1,3,224,224)')
    parser.add_argument('--seq-len', type=int, default=128,
                        help='Sequence length for transformer models (default: 128)')

    # Range selection (mutually exclusive groups)
    range_group = parser.add_argument_group('range selection',
                                            'Choose one method to select node range')

    # Method 1: Explicit start/end
    range_group.add_argument('--start', type=int, default=None,
                            help='Start node index (0-based, inclusive)')
    range_group.add_argument('--end', type=int, default=None,
                            help='End node index (exclusive)')

    # Method 2: Context around a node
    range_group.add_argument('--around', type=int, default=None,
                            help='Center node for context view')
    range_group.add_argument('--context', type=int, default=10,
                            help='Number of nodes before/after center (default: 10)')

    # Method 3: Simple max-nodes (backward compatible)
    range_group.add_argument('--max-nodes', '-n', type=int, default=None,
                            help='Maximum nodes to display from start')

    # Output options
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Save visualization to file')

    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_args()

    # Level 1: No model specified → show model list
    if args.model is None:
        GraphExplorerCLI.show_model_list()
        sys.exit(1)

    # Parse input shape
    try:
        input_shape = tuple(map(int, args.input_shape.split(',')))
    except ValueError:
        print(f"Error: Invalid input shape '{args.input_shape}'")
        print("Expected format: 1,3,224,224")
        sys.exit(1)

    # Validate mutually exclusive range options
    range_methods = sum([
        args.around is not None,
        args.start is not None or args.end is not None,
        args.max_nodes is not None
    ])
    if range_methods > 1:
        print("Error: Cannot use multiple range selection methods simultaneously")
        print("Choose one: --start/--end, --around/--context, or --max-nodes")
        sys.exit(1)

    # Create CLI and load model
    cli = GraphExplorerCLI()
    cli.load_and_trace_model(args.model, input_shape, seq_len=args.seq_len)

    # Level 2: Model only (no range) → show summary
    # Level 3: Model + range → show visualization
    if range_methods == 0 and args.output is None:
        # Summary mode (no visualization range specified, not saving to file)
        cli.show_summary()
    else:
        # Visualization mode
        cli.visualize(args)
        cli.show_tips(args)


if __name__ == "__main__":
    main()
