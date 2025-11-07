"""
Dynamo Graph Extraction for Complex HuggingFace Models

This module demonstrates how to trace complex real-world models from HuggingFace
that cannot be traced with FX symbolic_trace.

Models covered:
1. DETR (Detection Transformer) - Object detection with transformers
2. YOLO (if available) - Real-time object detection
3. BERT - Transformer encoder
4. ViT (Vision Transformer) - Image classification with transformers

These models have:
- Dynamic control flow
- Complex attention mechanisms
- Data-dependent operations
- Nested submodules

Requirements:
    pip install transformers torch torchvision

Usage:
    python experiments/dynamo/huggingface_complex_models.py --model detr
    python experiments/dynamo/huggingface_complex_models.py --model bert
    python experiments/dynamo/huggingface_complex_models.py --model vit
"""

import torch
import torch._dynamo as dynamo
from typing import Dict, Any, Optional, List, Tuple
import argparse
import sys
from pathlib import Path


# Import the basic extractor from our skeleton
class DynamoGraphExtractor:
    """
    Enhanced extractor with FLOP counting and memory analysis.
    """

    def __init__(self, analyze_memory: bool = True):
        self.graphs: List[torch.fx.Graph] = []
        self.graph_modules: List[torch.fx.GraphModule] = []
        self.subgraph_count = 0
        self.analyze_memory = analyze_memory
        self.total_params = 0
        self.operation_counts = {}

    def __call__(self, gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
        """Custom backend that captures graph information."""
        self.subgraph_count += 1
        self.graphs.append(gm.graph)
        self.graph_modules.append(gm)

        print(f"\n{'='*80}")
        print(f"Graph Partition {self.subgraph_count}")
        print(f"{'='*80}")

        # Print graph structure
        print("\nGraph Structure (first 50 nodes):")
        self._print_graph_summary(gm.graph)

        # Analyze operations
        self._analyze_operations(gm.graph)

        # Memory analysis
        if self.analyze_memory:
            self._analyze_memory_usage(gm.graph)

        return gm.forward

    def _print_graph_summary(self, graph: torch.fx.Graph, max_nodes: int = 50):
        """Print a summary of the graph structure."""
        nodes = list(graph.nodes)
        print(f"Total nodes: {len(nodes)}")

        # Print first N nodes in tabular format
        print(f"\nFirst {min(max_nodes, len(nodes))} nodes:")
        temp_graph = torch.fx.Graph()
        for node in nodes[:max_nodes]:
            temp_graph.node_copy(node, lambda x: x)
        temp_graph.print_tabular()

        if len(nodes) > max_nodes:
            print(f"\n... and {len(nodes) - max_nodes} more nodes")

    def _analyze_operations(self, graph: torch.fx.Graph):
        """Analyze and count different operation types."""
        op_types = {}
        function_calls = {}

        for node in graph.nodes:
            # Count by operation type
            op_types[node.op] = op_types.get(node.op, 0) + 1

            # Count function calls
            if node.op == 'call_function':
                target_name = self._get_target_name(node.target)
                function_calls[target_name] = function_calls.get(target_name, 0) + 1

        print("\nOperation Type Distribution:")
        for op, count in sorted(op_types.items()):
            print(f"  {op}: {count}")

        if function_calls:
            print("\nTop Function Calls:")
            sorted_calls = sorted(function_calls.items(), key=lambda x: x[1], reverse=True)
            for func, count in sorted_calls[:20]:  # Top 20
                print(f"  {func}: {count}")

        self.operation_counts[self.subgraph_count] = {
            'op_types': op_types,
            'function_calls': function_calls
        }

    def _analyze_memory_usage(self, graph: torch.fx.Graph):
        """Estimate memory usage from tensor metadata."""
        total_memory_bytes = 0
        max_tensor_bytes = 0
        tensor_count = 0

        for node in graph.nodes:
            if hasattr(node, 'meta') and 'tensor_meta' in node.meta:
                tm = node.meta['tensor_meta']
                if hasattr(tm, 'shape') and hasattr(tm, 'dtype'):
                    # Estimate size
                    numel = 1
                    for dim in tm.shape:
                        numel *= dim

                    # Bytes per element (rough estimate)
                    dtype_size = {
                        torch.float32: 4,
                        torch.float16: 2,
                        torch.bfloat16: 2,
                        torch.int64: 8,
                        torch.int32: 4,
                    }.get(tm.dtype, 4)

                    tensor_bytes = numel * dtype_size
                    total_memory_bytes += tensor_bytes
                    max_tensor_bytes = max(max_tensor_bytes, tensor_bytes)
                    tensor_count += 1

        if tensor_count > 0:
            print(f"\nMemory Analysis:")
            print(f"  Tensors: {tensor_count}")
            print(f"  Total memory: {total_memory_bytes / (1024**2):.2f} MB")
            print(f"  Largest tensor: {max_tensor_bytes / (1024**2):.2f} MB")
            print(f"  Average tensor: {total_memory_bytes / tensor_count / 1024:.2f} KB")

    def _get_target_name(self, target) -> str:
        """Extract readable name from target."""
        if isinstance(target, str):
            return target
        elif hasattr(target, '__name__'):
            return target.__name__
        else:
            return str(target).split("'")[1] if "'" in str(target) else str(target)

    def print_summary(self):
        """Print overall summary of all captured graphs."""
        print("\n" + "="*80)
        print("EXTRACTION SUMMARY")
        print("="*80)
        print(f"Total graph partitions: {self.subgraph_count}")

        total_nodes = sum(len(g.nodes) for g in self.graphs)
        print(f"Total nodes across all partitions: {total_nodes}")

        # Aggregate operation counts
        all_ops = {}
        all_functions = {}
        for partition_id, counts in self.operation_counts.items():
            for op, count in counts['op_types'].items():
                all_ops[op] = all_ops.get(op, 0) + count
            for func, count in counts['function_calls'].items():
                all_functions[func] = all_functions.get(func, 0) + count

        print("\nAggregate Operation Counts:")
        for op, count in sorted(all_ops.items()):
            print(f"  {op}: {count}")

        if all_functions:
            print("\nTop 20 Functions:")
            sorted_funcs = sorted(all_functions.items(), key=lambda x: x[1], reverse=True)
            for func, count in sorted_funcs[:20]:
                print(f"  {func}: {count}")


def trace_model_with_dynamo(
    model: torch.nn.Module,
    example_input: Any,
    verbose: bool = True,
    analyze_memory: bool = True
) -> DynamoGraphExtractor:
    """
    Trace a model using Dynamo with enhanced analysis.

    Args:
        model: PyTorch model to trace
        example_input: Example input (tensor or dict for HuggingFace models)
        verbose: Print detailed information
        analyze_memory: Include memory analysis

    Returns:
        DynamoGraphExtractor with captured graphs
    """
    extractor = DynamoGraphExtractor(analyze_memory=analyze_memory)
    dynamo.reset()

    # Compile with custom backend
    compiled_model = torch.compile(
        model,
        backend=extractor,
        fullgraph=False,
    )

    if verbose:
        print("\n" + "="*80)
        print("Starting Dynamo Tracing...")
        print("="*80)

    # Run model to trigger tracing
    with torch.no_grad():
        if isinstance(example_input, dict):
            _ = compiled_model(**example_input)
        else:
            _ = compiled_model(example_input)

    if verbose:
        extractor.print_summary()

    return extractor


# ============================================================================
# Example 1: DETR (Detection Transformer)
# ============================================================================

def trace_detr():
    """
    Trace Facebook DETR model for object detection.

    DETR is a complex model with:
    - Transformer encoder-decoder architecture
    - CNN backbone (ResNet)
    - Positional encodings
    - Bipartite matching loss
    """
    print("\n" + "="*80)
    print("EXAMPLE: DETR (Detection Transformer)")
    print("="*80)

    try:
        from transformers import DetrForObjectDetection, DetrImageProcessor
        import requests
        from PIL import Image

        print("\nLoading DETR model from HuggingFace...")
        model = DetrForObjectDetection.from_pretrained(
            "facebook/detr-resnet-50",
            torch_dtype=torch.float32
        )
        model.eval()

        # Create example input
        print("Creating example input...")
        processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

        # Use a simple dummy image
        dummy_image = Image.new('RGB', (800, 600), color='red')
        inputs = processor(images=dummy_image, return_tensors="pt")

        print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Input shape: {inputs['pixel_values'].shape}")

        # Trace with Dynamo
        extractor = trace_model_with_dynamo(
            model,
            example_input=inputs,
            verbose=True
        )

        return extractor

    except ImportError as e:
        print(f"\nError: {e}")
        print("Please install: pip install transformers pillow requests")
        return None


# ============================================================================
# Example 2: BERT (Transformer Encoder)
# ============================================================================

def trace_bert():
    """
    Trace BERT model for NLP tasks.

    BERT features:
    - Multi-head self-attention
    - Layer normalization
    - Feed-forward networks
    - Position embeddings
    """
    print("\n" + "="*80)
    print("EXAMPLE: BERT (Transformer Encoder)")
    print("="*80)

    try:
        from transformers import BertModel, BertTokenizer

        print("\nLoading BERT model from HuggingFace...")
        model = BertModel.from_pretrained("bert-base-uncased")
        model.eval()

        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        # Create example input
        text = "Hello, this is a test sentence for BERT model tracing."
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

        print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Input IDs shape: {inputs['input_ids'].shape}")

        # Trace with Dynamo
        extractor = trace_model_with_dynamo(
            model,
            example_input=inputs,
            verbose=True
        )

        return extractor

    except ImportError as e:
        print(f"\nError: {e}")
        print("Please install: pip install transformers")
        return None


# ============================================================================
# Example 3: Vision Transformer (ViT)
# ============================================================================

def trace_vit():
    """
    Trace Vision Transformer for image classification.

    ViT features:
    - Patch embedding
    - Transformer encoder
    - Position embeddings
    - Classification head
    """
    print("\n" + "="*80)
    print("EXAMPLE: Vision Transformer (ViT)")
    print("="*80)

    try:
        from transformers import ViTForImageClassification, ViTImageProcessor
        from PIL import Image

        print("\nLoading ViT model from HuggingFace...")
        model = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224"
        )
        model.eval()

        processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")

        # Create example input
        dummy_image = Image.new('RGB', (224, 224), color='blue')
        inputs = processor(images=dummy_image, return_tensors="pt")

        print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Input shape: {inputs['pixel_values'].shape}")

        # Trace with Dynamo
        extractor = trace_model_with_dynamo(
            model,
            example_input=inputs,
            verbose=True
        )

        return extractor

    except ImportError as e:
        print(f"\nError: {e}")
        print("Please install: pip install transformers pillow")
        return None


# ============================================================================
# Example 4: Save extracted graph to file
# ============================================================================

def save_graph_to_file(extractor: DynamoGraphExtractor, output_path: str):
    """
    Save extracted graphs to a Python file for inspection.

    Args:
        extractor: DynamoGraphExtractor with captured graphs
        output_path: Path to save the graph code
    """
    if not extractor or len(extractor.graph_modules) == 0:
        print("No graphs to save")
        return

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        f.write("# Auto-generated graph code from Dynamo extraction\n")
        f.write("# This code represents the computational graph structure\n\n")
        f.write("import torch\n")
        f.write("import torch.fx as fx\n\n")

        for i, gm in enumerate(extractor.graph_modules):
            f.write(f"\n{'='*80}\n")
            f.write(f"# Graph Partition {i+1}\n")
            f.write(f"{'='*80}\n\n")

            # Write the graph code
            f.write(gm.code)
            f.write("\n\n")

    print(f"\nGraph code saved to: {output_path}")


# ============================================================================
# Main CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Trace complex HuggingFace models with Dynamo"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["detr", "bert", "vit", "all"],
        default="detr",
        help="Model to trace"
    )
    parser.add_argument(
        "--save-graph",
        type=str,
        default=None,
        help="Save extracted graph to file"
    )

    args = parser.parse_args()

    # Run specified example
    extractors = {}

    if args.model == "all":
        extractors['detr'] = trace_detr()
        extractors['bert'] = trace_bert()
        extractors['vit'] = trace_vit()
    elif args.model == "detr":
        extractors['detr'] = trace_detr()
    elif args.model == "bert":
        extractors['bert'] = trace_bert()
    elif args.model == "vit":
        extractors['vit'] = trace_vit()

    # Save graphs if requested
    if args.save_graph:
        for model_name, extractor in extractors.items():
            if extractor:
                output_path = args.save_graph.replace('.py', f'_{model_name}.py')
                save_graph_to_file(extractor, output_path)


if __name__ == "__main__":
    main()
