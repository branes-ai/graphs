#!/usr/bin/env python
"""
Unified Graph Profiler - Works with any PyTorch model
======================================================

Profiles PyTorch models by characterizing each operator in the computational graph:
- Execution order and hierarchical structure
- Computational requirements (MACs, FLOPs)
- Memory bandwidth demands
- Parameter counts and tensor shapes
- Operation-level resource analysis

This unified profiler uses a hybrid tracing strategy:
1. Try standard FX symbolic_trace (fast, works for most models)
2. Fall back to Dynamo export (slower, more robust for complex models)
3. Always includes warm-up for models with lazy initialization

Supports:
- TorchVision models (ResNet, MobileNet, EfficientNet, ViT, etc.)
- YOLO models (YOLOv5, YOLOv8, YOLO11)
- Custom models from file paths
- Any FX-traceable PyTorch model
"""

import torch
import torch._dynamo
import torchvision.models as models
from torch.fx import symbolic_trace, GraphModule
from torch.fx.passes.shape_prop import ShapeProp
import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple, Union

# Add src to path for imports
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "src"))

from graphs.transform.partitioning import GraphPartitioner
from graphs.hardware.table_formatter import format_partition_table


# Model registry for torchvision models
MODEL_REGISTRY = {
    # ResNet family
    'resnet18': models.resnet18,
    'resnet34': models.resnet34,
    'resnet50': models.resnet50,
    'resnet101': models.resnet101,
    'resnet152': models.resnet152,
    # MobileNet family
    'mobilenet_v2': models.mobilenet_v2,
    'mobilenet_v3_large': models.mobilenet_v3_large,
    'mobilenet_v3_small': models.mobilenet_v3_small,
    # EfficientNet family
    'efficientnet_b0': models.efficientnet_b0,
    'efficientnet_b1': models.efficientnet_b1,
    'efficientnet_b4': models.efficientnet_b4,
    'efficientnet_b7': models.efficientnet_b7,
    # DenseNet family
    'densenet121': models.densenet121,
    'densenet161': models.densenet161,
    'densenet201': models.densenet201,
    # VGG family
    'vgg16': models.vgg16,
    'vgg19': models.vgg19,
    # Classic architectures
    'alexnet': models.alexnet,
    'squeezenet1_0': models.squeezenet1_0,
    # Modern CNNs
    'convnext_tiny': models.convnext_tiny,
    'convnext_small': models.convnext_small,
    'convnext_base': models.convnext_base,
    # Vision Transformers
    'vit_b_16': models.vit_b_16,
    'vit_b_32': models.vit_b_32,
    'vit_l_16': models.vit_l_16,
    # Swin Transformers
    'swin_t': models.swin_t,
    'swin_s': models.swin_s,
    'swin_b': models.swin_b,
    # RegNets
    'regnet_y_400mf': models.regnet_y_400mf,
    'regnet_y_800mf': models.regnet_y_800mf,
    'regnet_y_1_6gf': models.regnet_y_1_6gf,
    'regnet_y_3_2gf': models.regnet_y_3_2gf,
    'regnet_y_8gf': models.regnet_y_8gf,
}


def load_model_from_registry(model_name: str) -> torch.nn.Module:
    """Load a model from the torchvision registry"""
    if model_name not in MODEL_REGISTRY:
        available = ', '.join(sorted(MODEL_REGISTRY.keys()))
        raise ValueError(f"Unknown model '{model_name}'. Available: {available}")

    model = MODEL_REGISTRY[model_name](weights=None)
    model.eval()
    return model


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
        (model, input_type) where input_type is 'tokens', 'image', or 'detr'
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


def load_model_from_path(model_path: str) -> torch.nn.Module:
    """Load a model from a file path (assumes it's a YOLO model or similar)"""
    path = Path(model_path)

    # Check if it's a YOLO model
    if path.suffix == '.pt' and path.exists():
        return load_yolo_model(model_path)

    raise ValueError(f"Unsupported model file: {model_path}")


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
    print("[1/4] Warming up model (initializing any lazy modules)...")
    with torch.no_grad():
        try:
            _ = model(*example_inputs)
        except Exception as e:
            print(f"  Note: Warm-up forward pass failed: {e}")
            print("  Continuing with tracing anyway...")

    # Try standard FX symbolic_trace first (preferred)
    print("[2/4] Attempting standard FX symbolic_trace...")
    try:
        traced = symbolic_trace(model)
        print("  ✓ Standard FX trace successful")
        return traced, "symbolic_trace"
    except Exception as e:
        print(f"  ✗ Standard FX trace failed: {type(e).__name__}")
        print("  Falling back to Dynamo export...")

    # Fall back to Dynamo export
    print("[2/4] Using PyTorch Dynamo export (more robust)...")
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

        traced, guards = torch._dynamo.export(forward_fn)(*example_inputs)
        print("  ✓ Dynamo export successful")
        return traced, "dynamo_export"
    except Exception as e:
        print(f"  ✗ Dynamo export failed: {e}")
        raise RuntimeError(
            f"Failed to trace model '{model_name}' with both FX and Dynamo. "
            "The model may not be compatible with graph tracing."
        )


def profile_model(
    model: Union[str, torch.nn.Module],
    input_shape: Optional[Tuple[int, ...]] = None,
    seq_len: int = 128,
    show_shapes: bool = False,
    model_name: Optional[str] = None,
    visualize: Optional[str] = None
) -> None:
    """
    Profile a PyTorch model with hierarchical table

    Args:
        model: Model name (from registry/HuggingFace), file path (.pt), or torch.nn.Module
        input_shape: Input tensor shape for vision models (default: (1, 3, 224, 224))
        seq_len: Sequence length for transformer models (default: 128)
        show_shapes: If True, show parameter and tensor shapes
        model_name: Display name (inferred if not provided)
        visualize: Optional output path for visualization (PNG/DOT/PDF)
    """

    # Default input shape for vision models
    if input_shape is None:
        input_shape = (1, 3, 224, 224)

    # Determine model name, load model, and detect input type
    input_type = 'image'  # Default to image tensors
    if isinstance(model, str):
        model_name = model_name or model

        # Try registry first
        if model in MODEL_REGISTRY:
            print(f"Loading '{model}' from torchvision registry...")
            model_obj = load_model_from_registry(model)
            input_type = 'image'
        else:
            # Check if it's a file path
            path = Path(model)
            if path.exists():
                print(f"Loading model from '{model}'...")
                model_obj = load_model_from_path(model)
                input_type = 'image'
            else:
                # Assume it's a HuggingFace model
                model_obj, input_type = load_transformer_model(model)
    else:
        model_obj = model
        model_name = model_name or model.__class__.__name__

    print("=" * 100)
    print(f"UNIFIED GRAPH PROFILER: {model_name}")
    print("=" * 100)
    print()

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

        # Check if model is BERT-style (needs attention_mask) or GPT-style (doesn't)
        # GPT models have 'gpt' in their name and work better without attention_mask for tracing
        is_gpt_style = 'gpt' in model_name.lower()

        if is_gpt_style:
            # GPT-style models: just input_ids
            example_inputs = input_ids
            print(f"Input type: Tokens (batch_size={batch_size}, seq_len={seq_len}) [GPT-style]")
        else:
            # BERT-style models: input_ids + attention_mask
            attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
            example_inputs = (input_ids, attention_mask)
            print(f"Input type: Tokens (batch_size={batch_size}, seq_len={seq_len}, with attention_mask) [BERT-style]")
    elif input_type == 'detr':
        # DETR models need image tensors with pixel_values keyword
        example_inputs = torch.randn(*input_shape)
        print(f"Input type: Image tensor {input_shape} [DETR - vision transformer]")
    else:
        # Standard vision models need image tensors
        example_inputs = torch.randn(*input_shape)
        print(f"Input type: Image tensor {input_shape}")

    # Trace with hybrid strategy
    # DETR needs special handling (uses pixel_values keyword)
    if input_type == 'detr':
        # Create a wrapper that converts positional arg to keyword arg
        original_model = model_obj
        class DetrWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, x):
                return self.model(pixel_values=x)

        model_obj = DetrWrapper(original_model)

    try:
        traced_graph, method = trace_model_hybrid(model_obj, example_inputs, model_name)
    except Exception as e:
        print(f"\nFailed to trace model: {e}")
        return

    # Propagate shapes
    print("[3/4] Propagating tensor shapes through graph...")
    if isinstance(example_inputs, tuple):
        ShapeProp(traced_graph).propagate(*example_inputs)
    else:
        ShapeProp(traced_graph).propagate(example_inputs)

    # Partition
    print("[4/4] Running graph partitioner...")
    partitioner = GraphPartitioner()
    report = partitioner.partition(traced_graph)

    # Display table
    print("\n" + "=" * 100)
    print("HIERARCHICAL GRAPH PROFILE")
    print("=" * 100)
    print()

    table = format_partition_table(traced_graph, report, show_shapes=show_shapes)
    print(table)

    # Summary
    print("\n" + "=" * 100)
    print("MODEL SUMMARY")
    print("=" * 100)

    total_params = sum(p.numel() for p in traced_graph.parameters())
    print(f"\nModel: {model_name}")
    if input_type == 'tokens':
        print(f"Input: Tokens (batch_size={input_shape[0] if input_shape else 1}, seq_len={seq_len})")
    else:
        print(f"Input: Image tensor {input_shape}")
    print(f"Tracing method: {method}")
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

    # Generate visualization if requested
    if visualize:
        print("\n" + "=" * 100)
        print("GENERATING VISUALIZATION")
        print("=" * 100)
        generate_visualization(traced_graph, example_inputs, visualize, model_name, method)


def generate_visualization(
    traced_graph: GraphModule,
    example_inputs: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
    output_path: str,
    model_name: str,
    trace_method: str = "symbolic_trace"
):
    """
    Generate graph visualization using torchview

    Args:
        traced_graph: FX traced graph module
        example_inputs: Example input tensor(s)
        output_path: Output file path (PNG/DOT/PDF/SVG)
        model_name: Model name for graph title
        trace_method: Tracing method used ('symbolic_trace' or 'dynamo_export')
    """
    try:
        from torchview import draw_graph
        from graphviz import Source
    except ImportError:
        print("\n❌ Visualization requires additional dependencies:")
        print("   pip install torchview graphviz")
        print("\nNote: You also need graphviz system package:")
        print("   Ubuntu/Debian: sudo apt install graphviz")
        print("   macOS: brew install graphviz")
        print("   Windows: choco install graphviz")
        return

    # Determine output format from extension
    path = Path(output_path)
    format_ext = path.suffix.lstrip('.')
    if not format_ext:
        format_ext = 'png'  # Default to PNG
        output_path = str(path) + '.png'

    supported_formats = ['png', 'pdf', 'svg', 'dot']
    if format_ext not in supported_formats:
        print(f"\n⚠️  Unknown format '{format_ext}', using PNG")
        format_ext = 'png'
        output_path = str(path.with_suffix('.png'))

    print(f"\nGenerating {format_ext.upper()} visualization...")
    print(f"Output: {output_path}")

    # Normalize inputs to tuple
    if not isinstance(example_inputs, tuple):
        example_inputs_tuple = (example_inputs,)
    else:
        example_inputs_tuple = example_inputs

    # Get input size from first tensor
    input_size = tuple(example_inputs_tuple[0].shape)

    try:
        # Generate graph visualization
        model_graph = draw_graph(
            traced_graph,
            input_size=input_size,
            device='cpu',
            graph_name=f'{model_name}_FX_Graph',
            roll=True,
            expand_nested=True,
        )

        # Extract DOT source
        dot_source = model_graph.visual_graph.source

        if format_ext == 'dot':
            # Save DOT source directly
            with open(output_path, 'w') as f:
                f.write(dot_source)
            print(f"✓ DOT source saved to {output_path}")
        else:
            # Render to image format
            graph = Source(dot_source, filename=str(path.stem), format=format_ext, directory=str(path.parent or '.'))
            output_file = graph.render(view=False, cleanup=True)
            print(f"✓ Visualization saved to {output_file}")

        # Show graph statistics
        num_nodes = dot_source.count('->') + dot_source.count('[label=')
        print(f"\nGraph statistics:")
        print(f"  Nodes: ~{num_nodes}")
        print(f"  Format: {format_ext.upper()}")

    except Exception as e:
        print(f"\n❌ Visualization failed: {e}")
        print("\nTips:")
        print("  - Ensure graphviz is installed: sudo apt install graphviz")

        if trace_method == "dynamo_export":
            print("\n⚠️  Note: This model was traced with Dynamo, which has known")
            print("   compatibility issues with torchview for complex models.")
            print("\n   Recommended alternatives:")
            print()
            print("   1. Use graph_explorer.py for interactive text visualization:")
            print(f"      python cli/graph_explorer.py --model {model_name}")
            print(f"      python cli/graph_explorer.py --model {model_name} --max-nodes 20")
            print(f"      python cli/graph_explorer.py --model {model_name} --output graph_viz.txt")
            print()
            print("   2. Generate DOT file and render manually:")
            print(f"      python cli/profile_graph.py --model {model_name} --visualize {Path(output_path).stem}.dot")
            print(f"      dot -Tpng {Path(output_path).stem}.dot -o {output_path}")
            print()
            print("   3. Use torch.fx.graph.print_readable() for simple text view:")
            print(f"      python -c \"import torch; from cli.profile_graph import *; traced, _ = trace_model_hybrid(...); print(traced.graph)\"")
        else:
            print("  - For large models, try reducing model complexity")
            print("  - DOT format is always supported: --visualize graph.dot")


def main():
    parser = argparse.ArgumentParser(
        description='Unified graph profiler: works with TorchVision models, YOLO, and custom models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Profile TorchVision models
  python cli/profile_graph.py --model resnet18
  python cli/profile_graph.py --model mobilenet_v2 --showshape
  python cli/profile_graph.py --model vit_b_16

  # Profile YOLO models (requires: pip install ultralytics)
  python cli/profile_graph.py --model yolov8n.pt
  python cli/profile_graph.py --model yolov8s.pt --showshape
  python cli/profile_graph.py --model yolo11m.pt --input-shape 1 3 640 640

  # Profile Transformer models (requires: pip install transformers)
  python cli/profile_graph.py --model bert-base-uncased
  python cli/profile_graph.py --model gpt2 --seq-len 256
  python cli/profile_graph.py --model distilbert-base-uncased --showshape

  # List available torchvision models
  python cli/profile_graph.py --list

Tracing Strategy:
  The profiler uses a hybrid approach for maximum compatibility:
  1. Warm-up: Run model once to initialize lazy modules (safe for all models)
  2. Try standard FX symbolic_trace (fast, works for most models)
  3. Fall back to Dynamo export if needed (robust for complex models like YOLO)

  This makes the profiler work with:
  - Standard CNNs (ResNet, VGG, DenseNet)
  - Mobile networks (MobileNet, EfficientNet)
  - Transformers (ViT, Swin)
  - Detection models (YOLO - requires warm-up + Dynamo)
  - Custom models with lazy initialization
        """
    )

    parser.add_argument('--model', type=str,
                       help='Model name (torchvision/HuggingFace) or path (.pt file for YOLO)')

    parser.add_argument('--showshape', action='store_true',
                       help='Show parameter shapes (weight/bias dimensions) and tensor shapes')

    parser.add_argument('--input-shape', type=int, nargs=4, default=[1, 3, 224, 224],
                       metavar=('B', 'C', 'H', 'W'),
                       help='Input tensor shape for vision models (default: 1 3 224 224)')

    parser.add_argument('--seq-len', type=int, default=128,
                       help='Sequence length for transformer models (default: 128)')

    parser.add_argument('--list', action='store_true',
                       help='List available torchvision models and exit')

    parser.add_argument('--visualize', '-v', type=str, metavar='OUTPUT',
                       help='Generate graph visualization (PNG/PDF/SVG/DOT). Example: --visualize graph.png')

    args = parser.parse_args()

    if args.list:
        print("=" * 100)
        print("AVAILABLE MODELS")
        print("=" * 100)

        print("\n📦 TorchVision Models (Vision CNNs & Transformers)")
        print("-" * 100)
        for name in sorted(MODEL_REGISTRY.keys()):
            print(f"  {name}")
        print(f"\nTotal: {len(MODEL_REGISTRY)} TorchVision models")

        print("\n" + "=" * 100)
        print("\n🤗 HuggingFace Transformer Models (Text & Vision)")
        print("-" * 100)
        print("  Discover available models:")
        print("    python cli/discover_transformers.py")
        print("    python cli/discover_transformers.py --generate-examples")
        print("\n  Common examples:")
        print("    bert-base-uncased, distilbert-base-uncased, roberta-base")
        print("    gpt2, gpt2-medium, distilgpt2")
        print("    EleutherAI/gpt-neo-125m, facebook/opt-125m")
        print("    facebook/detr-resnet-50  (DETR object detection)")

        print("\n" + "=" * 100)
        print("\n🎯 YOLO Models (Object Detection)")
        print("-" * 100)
        print("  Provide path to .pt file:")
        print("    yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt")
        print("    yolo11n.pt, yolo11s.pt, yolo11m.pt, yolo11l.pt")
        print("\n  Download from: https://github.com/ultralytics/ultralytics")

        print("\n" + "=" * 100)
        print("\n📚 Documentation")
        print("-" * 100)
        print("  Complete model names guide: docs/MODEL_NAMES_GUIDE.md")
        print("  Transformer discovery: python cli/discover_transformers.py")
        print("  Vision model discovery: python cli/discover_models.py")
        print("\n" + "=" * 100)
        return

    if not args.model:
        parser.print_help()
        print("\nError: --model is required (or use --list to see available models)")
        sys.exit(1)

    profile_model(
        args.model,
        input_shape=tuple(args.input_shape),
        seq_len=args.seq_len,
        show_shapes=args.showshape,
        visualize=args.visualize
    )


if __name__ == "__main__":
    main()
