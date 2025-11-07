"""
Integration Example: Dynamo Graph Extraction → graphs Package Pipeline

This module demonstrates how to integrate Dynamo-extracted graphs with the
existing graphs package characterization pipeline.

The workflow:
1. Extract graph using Dynamo (handles complex models)
2. Convert Dynamo graph to graphs.ir structures
3. Run existing analysis pipeline (roofline, energy, memory)
4. Generate hardware mapping reports

This bridges the gap between:
- Dynamo (robust graph extraction)
- graphs package (comprehensive DNN characterization)

Usage:
    python experiments/dynamo/integrate_with_graphs.py --model resnet18
"""

import torch
import torch._dynamo as dynamo
from typing import List, Dict, Any, Optional, Tuple
import argparse
from dataclasses import dataclass


# ============================================================================
# Dynamo Graph Extraction (reuse from basic_dynamo_tracing.py)
# ============================================================================

class GraphExtractor:
    """Simple extractor for integration demo."""

    def __init__(self):
        self.graph_modules: List[torch.fx.GraphModule] = []

    def __call__(self, gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
        self.graph_modules.append(gm)
        return gm.forward


def extract_graph_with_dynamo(
    model: torch.nn.Module,
    example_input: torch.Tensor
) -> Optional[torch.fx.GraphModule]:
    """Extract graph using Dynamo."""
    extractor = GraphExtractor()
    dynamo.reset()

    compiled_model = torch.compile(model, backend=extractor, fullgraph=False)

    with torch.no_grad():
        _ = compiled_model(example_input)

    if len(extractor.graph_modules) > 0:
        return extractor.graph_modules[0]
    return None


# ============================================================================
# Conversion to graphs.ir Structures
# ============================================================================

@dataclass
class SimplifiedTensorDescriptor:
    """Simplified tensor descriptor for demo (mirrors graphs.ir.TensorDescriptor)."""
    name: str
    shape: Tuple[int, ...]
    dtype: torch.dtype
    memory_bytes: int


@dataclass
class SimplifiedSubgraphDescriptor:
    """Simplified subgraph descriptor for demo (mirrors graphs.ir.SubgraphDescriptor)."""
    operations: List[str]
    input_tensors: List[SimplifiedTensorDescriptor]
    output_tensors: List[SimplifiedTensorDescriptor]
    compute_flops: int
    memory_bytes: int


def convert_dynamo_graph_to_ir(
    graph_module: torch.fx.GraphModule
) -> List[SimplifiedSubgraphDescriptor]:
    """
    Convert Dynamo FX graph to graphs.ir-compatible structures.

    This is a simplified demonstration. In production, you would:
    1. Use actual graphs.ir.TensorDescriptor
    2. Implement proper FLOP counting
    3. Handle all operation types
    4. Integrate with FusionPartitioner

    Args:
        graph_module: Dynamo-extracted GraphModule

    Returns:
        List of subgraph descriptors
    """
    subgraphs = []
    current_ops = []
    current_inputs = []
    current_outputs = []

    print("\nConverting Dynamo graph to IR structures...")
    print("="*80)

    for node in graph_module.graph.nodes:
        if node.op == 'placeholder':
            # Input tensor
            tensor_desc = _extract_tensor_descriptor(node)
            if tensor_desc:
                current_inputs.append(tensor_desc)
                print(f"  Input: {tensor_desc.name} {tensor_desc.shape}")

        elif node.op == 'call_function':
            # Operation
            op_name = _get_operation_name(node)
            current_ops.append(op_name)
            print(f"  Op: {op_name}")

            # Estimate FLOPs for this operation
            flops = _estimate_flops(node)

            # If this is a significant operation, create a subgraph
            if flops > 0 or len(current_ops) > 5:
                output_desc = _extract_tensor_descriptor(node)
                if output_desc:
                    current_outputs.append(output_desc)

                    subgraph = SimplifiedSubgraphDescriptor(
                        operations=current_ops.copy(),
                        input_tensors=current_inputs.copy(),
                        output_tensors=current_outputs.copy(),
                        compute_flops=flops,
                        memory_bytes=sum(t.memory_bytes for t in current_inputs + current_outputs)
                    )
                    subgraphs.append(subgraph)

                    # Reset for next subgraph
                    current_ops = []
                    current_inputs = [output_desc]  # Output becomes next input
                    current_outputs = []

        elif node.op == 'output':
            # Final output
            if current_ops:
                subgraph = SimplifiedSubgraphDescriptor(
                    operations=current_ops,
                    input_tensors=current_inputs,
                    output_tensors=current_outputs,
                    compute_flops=0,
                    memory_bytes=sum(t.memory_bytes for t in current_inputs + current_outputs)
                )
                subgraphs.append(subgraph)

    print(f"\nCreated {len(subgraphs)} subgraphs")
    return subgraphs


def _extract_tensor_descriptor(node: torch.fx.Node) -> Optional[SimplifiedTensorDescriptor]:
    """Extract tensor metadata from a node."""
    if hasattr(node, 'meta') and 'tensor_meta' in node.meta:
        tm = node.meta['tensor_meta']
        if hasattr(tm, 'shape') and hasattr(tm, 'dtype'):
            shape = tuple(tm.shape)
            dtype = tm.dtype

            # Calculate memory
            numel = 1
            for dim in shape:
                numel *= dim

            dtype_size = {
                torch.float32: 4,
                torch.float16: 2,
                torch.bfloat16: 2,
                torch.int64: 8,
                torch.int32: 4,
            }.get(dtype, 4)

            memory_bytes = numel * dtype_size

            return SimplifiedTensorDescriptor(
                name=node.name,
                shape=shape,
                dtype=dtype,
                memory_bytes=memory_bytes
            )
    return None


def _get_operation_name(node: torch.fx.Node) -> str:
    """Get human-readable operation name."""
    if isinstance(node.target, str):
        return node.target
    elif hasattr(node.target, '__name__'):
        return node.target.__name__
    else:
        target_str = str(node.target)
        if "'" in target_str:
            return target_str.split("'")[1].split('.')[-1]
        return target_str


def _estimate_flops(node: torch.fx.Node) -> int:
    """
    Estimate FLOPs for an operation (simplified).

    In production, use fvcore.nn.FlopCountAnalysis or implement
    detailed FLOP counting per operation type.
    """
    op_name = _get_operation_name(node).lower()

    # Get output shape if available
    if hasattr(node, 'meta') and 'tensor_meta' in node.meta:
        tm = node.meta['tensor_meta']
        if hasattr(tm, 'shape'):
            numel = 1
            for dim in tm.shape:
                numel *= dim

            # Rough FLOP estimates
            if 'conv' in op_name:
                # Conv: 2 * C_in * K * K * C_out * H * W (simplified)
                # Use output numel as proxy
                return 2 * numel
            elif 'matmul' in op_name or 'linear' in op_name:
                # MatMul: 2 * M * N * K (simplified)
                return 2 * numel
            elif op_name in ['relu', 'sigmoid', 'tanh', 'gelu']:
                # Activation: 1 FLOP per element
                return numel
            elif op_name in ['add', 'sub', 'mul', 'div']:
                # Elementwise: 1 FLOP per element
                return numel

    return 0


# ============================================================================
# Integration with graphs Package Analysis
# ============================================================================

def analyze_with_graphs_pipeline(
    subgraphs: List[SimplifiedSubgraphDescriptor],
    hardware_target: str = "H100"
):
    """
    Demonstrate integration with graphs package analysis pipeline.

    In production, this would:
    1. Use graphs.analysis.unified_analyzer.UnifiedAnalyzer
    2. Use graphs.hardware.mappers for hardware-specific mapping
    3. Generate comprehensive reports

    Args:
        subgraphs: IR subgraph descriptors
        hardware_target: Target hardware (e.g., 'H100', 'CPU', 'TPU')
    """
    print("\n" + "="*80)
    print(f"ANALYSIS WITH GRAPHS PIPELINE (Target: {hardware_target})")
    print("="*80)

    total_flops = sum(sg.compute_flops for sg in subgraphs)
    total_memory = sum(sg.memory_bytes for sg in subgraphs)

    print(f"\nTotal subgraphs: {len(subgraphs)}")
    print(f"Total FLOPs: {total_flops:,}")
    print(f"Total memory: {total_memory / (1024**2):.2f} MB")

    print("\nPer-subgraph analysis:")
    for i, sg in enumerate(subgraphs):
        print(f"\n  Subgraph {i+1}:")
        print(f"    Operations: {', '.join(sg.operations[:5])}" +
              (f" ... +{len(sg.operations)-5} more" if len(sg.operations) > 5 else ""))
        print(f"    FLOPs: {sg.compute_flops:,}")
        print(f"    Memory: {sg.memory_bytes / 1024:.2f} KB")
        print(f"    Inputs: {len(sg.input_tensors)}, Outputs: {len(sg.output_tensors)}")

        # Compute arithmetic intensity (FLOPs / bytes)
        if sg.memory_bytes > 0:
            arithmetic_intensity = sg.compute_flops / sg.memory_bytes
            print(f"    Arithmetic Intensity: {arithmetic_intensity:.2f} FLOPs/byte")

            # Simple roofline classification
            if arithmetic_intensity < 1.0:
                bottleneck = "Memory-bound"
            else:
                bottleneck = "Compute-bound"
            print(f"    Bottleneck: {bottleneck}")

    # Demonstrate hardware mapping (mock)
    print(f"\n{'-'*80}")
    print(f"Hardware Mapping: {hardware_target}")
    print(f"{'-'*80}")

    # Mock hardware specs
    hardware_specs = {
        'H100': {'peak_flops': 1979e12, 'memory_bw': 3350e9, 'name': 'NVIDIA H100'},
        'CPU': {'peak_flops': 2.5e12, 'memory_bw': 85e9, 'name': 'Intel Xeon'},
        'TPU': {'peak_flops': 450e12, 'memory_bw': 900e9, 'name': 'Google TPU v4'},
    }

    if hardware_target in hardware_specs:
        hw = hardware_specs[hardware_target]
        print(f"  Device: {hw['name']}")
        print(f"  Peak FLOPs: {hw['peak_flops'] / 1e12:.1f} TFLOPS")
        print(f"  Memory BW: {hw['memory_bw'] / 1e9:.1f} GB/s")

        # Estimate latency
        compute_time = total_flops / hw['peak_flops']
        memory_time = total_memory / hw['memory_bw']
        latency = max(compute_time, memory_time)

        print(f"\n  Estimated latency: {latency * 1000:.2f} ms")
        print(f"    Compute time: {compute_time * 1000:.2f} ms")
        print(f"    Memory time: {memory_time * 1000:.2f} ms")

        if memory_time > compute_time:
            print(f"  → Memory-bound workload")
        else:
            print(f"  → Compute-bound workload")


# ============================================================================
# End-to-End Example
# ============================================================================

def example_integration_workflow():
    """
    Complete workflow demonstrating Dynamo → graphs package integration.
    """
    print("\n" + "="*80)
    print("INTEGRATION WORKFLOW: Dynamo → graphs Package")
    print("="*80)

    # 1. Define a model (use simple model for demo)
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(3, 16, 3, padding=1)
            self.conv2 = torch.nn.Conv2d(16, 32, 3, padding=1)
            self.fc = torch.nn.Linear(32 * 56 * 56, 10)

        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = torch.nn.functional.max_pool2d(x, 2)
            x = torch.relu(self.conv2(x))
            x = torch.nn.functional.max_pool2d(x, 2)
            x = torch.flatten(x, 1)
            x = self.fc(x)
            return x

    model = SimpleModel()
    model.eval()

    example_input = torch.randn(1, 3, 224, 224)

    print("\n1. Extract graph with Dynamo...")
    print("-"*80)
    graph_module = extract_graph_with_dynamo(model, example_input)

    if graph_module is None:
        print("Failed to extract graph")
        return

    print(f"✓ Extracted graph with {len(list(graph_module.graph.nodes))} nodes")

    # 2. Convert to IR structures
    print("\n2. Convert to graphs.ir structures...")
    print("-"*80)
    subgraphs = convert_dynamo_graph_to_ir(graph_module)

    if len(subgraphs) == 0:
        print("No subgraphs created")
        return

    print(f"✓ Created {len(subgraphs)} subgraphs")

    # 3. Run analysis pipeline
    print("\n3. Run graphs package analysis...")
    print("-"*80)
    analyze_with_graphs_pipeline(subgraphs, hardware_target='H100')

    # 4. Compare multiple hardware targets
    print("\n4. Compare hardware targets...")
    print("-"*80)
    for hw in ['H100', 'CPU', 'TPU']:
        print(f"\n{'='*40}")
        analyze_with_graphs_pipeline(subgraphs, hardware_target=hw)


def example_complex_model_integration():
    """
    Example with a more complex model (ResNet-like).
    """
    print("\n" + "="*80)
    print("COMPLEX MODEL INTEGRATION")
    print("="*80)

    # Use a small ResNet block
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 64, 7, stride=2, padding=3),
        torch.nn.BatchNorm2d(64),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(3, stride=2, padding=1),
        torch.nn.Conv2d(64, 128, 3, padding=1),
        torch.nn.BatchNorm2d(128),
        torch.nn.ReLU(),
    )
    model.eval()

    example_input = torch.randn(1, 3, 224, 224)

    # Extract and analyze
    graph_module = extract_graph_with_dynamo(model, example_input)

    if graph_module:
        subgraphs = convert_dynamo_graph_to_ir(graph_module)
        analyze_with_graphs_pipeline(subgraphs, hardware_target='H100')


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Integration example: Dynamo → graphs package"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["simple", "complex"],
        default="simple",
        help="Model to use for demo"
    )

    args = parser.parse_args()

    if args.model == "simple":
        example_integration_workflow()
    elif args.model == "complex":
        example_complex_model_integration()


if __name__ == "__main__":
    main()
