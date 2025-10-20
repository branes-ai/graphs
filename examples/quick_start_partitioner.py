#!/usr/bin/env python
"""
Quick Start: Graph Partitioner
==============================

This script demonstrates the basic usage of the graph partitioner.
Run this first to get familiar with the output.

Usage:
    python examples/quick_start_partitioner.py
"""

import torch
import torchvision.models as models
from torch.fx import symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp
import sys
sys.path.insert(0, 'src')

from graphs.characterize.graph_partitioner import GraphPartitioner
from graphs.characterize.concurrency_analyzer import ConcurrencyAnalyzer
from collections import Counter


def analyze_fx_graph_nodes(fx_graph, shape_prop=None):
    """
    Analyze FX graph nodes and generate statistics

    Returns summary of node types, their counts, and operations
    """

    # Count node types
    node_op_counts = Counter()
    node_target_counts = Counter()
    call_function_counts = Counter()
    call_method_counts = Counter()
    call_module_counts = Counter()

    # Track all nodes for detailed analysis
    all_nodes = []

    for node in fx_graph.graph.nodes:
        node_op_counts[node.op] += 1

        if node.op == 'call_function':
            # For call_function, target is the actual function
            func_name = node.target.__name__ if hasattr(node.target, '__name__') else str(node.target)
            call_function_counts[func_name] += 1
            node_target_counts[f"call_function:{func_name}"] += 1

        elif node.op == 'call_method':
            # For call_method, target is the method name (string)
            call_method_counts[node.target] += 1
            node_target_counts[f"call_method:{node.target}"] += 1

        elif node.op == 'call_module':
            # For call_module, target is the module path (string)
            # Get the actual module to see its type
            module = fx_graph.get_submodule(node.target)
            module_type = type(module).__name__
            call_module_counts[module_type] += 1
            node_target_counts[f"call_module:{module_type}"] += 1

        else:
            # placeholder, get_attr, output
            node_target_counts[f"{node.op}"] += 1

        all_nodes.append({
            'name': node.name,
            'op': node.op,
            'target': node.target,
            'node': node
        })

    return {
        'total_nodes': len(all_nodes),
        'node_op_counts': dict(node_op_counts),
        'node_target_counts': dict(node_target_counts),
        'call_function_counts': dict(call_function_counts),
        'call_method_counts': dict(call_method_counts),
        'call_module_counts': dict(call_module_counts),
        'all_nodes': all_nodes
    }


def print_fx_graph_summary(fx_graph, analysis):
    """Print a formatted summary of FX graph nodes"""

    print("\n" + "=" * 80)
    print("FX GRAPH NODE STATISTICS")
    print("=" * 80)

    print(f"\nTotal nodes: {analysis['total_nodes']}")

    # Node operation types (placeholder, call_function, call_module, etc.)
    print("\nNode Operation Types:")
    print("-" * 80)
    for op, count in sorted(analysis['node_op_counts'].items(), key=lambda x: x[1], reverse=True):
        pct = count / analysis['total_nodes'] * 100
        print(f"  {op:<20} {count:>5} ({pct:>5.1f}%)")

    # Call Module types (Conv2d, BatchNorm2d, ReLU, etc.)
    if analysis['call_module_counts']:
        print("\nCall Module Types (Layers):")
        print("-" * 80)
        total_modules = sum(analysis['call_module_counts'].values())
        for module_type, count in sorted(analysis['call_module_counts'].items(),
                                         key=lambda x: x[1], reverse=True):
            pct = count / total_modules * 100
            print(f"  {module_type:<30} {count:>5} ({pct:>5.1f}%)")

    # Call Function types (torch.add, torch.flatten, etc.)
    if analysis['call_function_counts']:
        print("\nCall Function Types:")
        print("-" * 80)
        total_functions = sum(analysis['call_function_counts'].values())
        for func_name, count in sorted(analysis['call_function_counts'].items(),
                                       key=lambda x: x[1], reverse=True):
            pct = count / total_functions * 100
            print(f"  {func_name:<30} {count:>5} ({pct:>5.1f}%)")

    # Call Method types (.view, .flatten, etc.)
    if analysis['call_method_counts']:
        print("\nCall Method Types:")
        print("-" * 80)
        total_methods = sum(analysis['call_method_counts'].values())
        for method_name, count in sorted(analysis['call_method_counts'].items(),
                                         key=lambda x: x[1], reverse=True):
            pct = count / total_methods * 100
            print(f"  {method_name:<30} {count:>5} ({pct:>5.1f}%)")


def main():
    print("=" * 80)
    print("Graph Partitioner Quick Start")
    print("=" * 80)

    # Step 1: Load a model (start with ResNet-18)
    print("\n[1/6] Loading ResNet-18...")
    model = models.resnet18(weights=None)
    model.eval()

    # Step 2: Trace the model with PyTorch FX
    print("[2/6] Tracing model with PyTorch FX...")
    input_tensor = torch.randn(1, 3, 224, 224)

    try:
        fx_graph = symbolic_trace(model)
        print("    FX tracing successful")
    except Exception as e:
        print(f"    FX tracing failed: {e}")
        return

    # Step 3: Analyze FX graph nodes
    print("[3/6] Analyzing FX graph nodes...")
    fx_analysis = analyze_fx_graph_nodes(fx_graph)
    print(f"    Found {fx_analysis['total_nodes']} nodes in FX graph")
    print_fx_graph_summary(fx_graph, fx_analysis)

    # Step 4: Propagate shapes through the graph
    print("\n[4/6] Propagating shapes...")
    shape_prop = ShapeProp(fx_graph)
    shape_prop.propagate(input_tensor)
    print("    Shape propagation complete")

    # Step 5: Partition the graph
    print("\n[5/6] Partitioning graph into subgraphs...")
    partitioner = GraphPartitioner()
    report = partitioner.partition(fx_graph)
    print(f"    Created {report.total_subgraphs} subgraphs")

    # Show mapping from FX nodes to subgraphs
    print("\n" + "=" * 80)
    print("FX NODE -> SUBGRAPH MAPPING")
    print("=" * 80)
    print(f"\nFX Graph has {fx_analysis['call_module_counts'].get('Conv2d', 0) + fx_analysis['call_module_counts'].get('Linear', 0) + fx_analysis['call_module_counts'].get('BatchNorm2d', 0) + fx_analysis['call_module_counts'].get('ReLU', 0)} call_module nodes")
    print(f"Created {report.total_subgraphs} subgraphs from call_module nodes")
    print(f"\nNote: GraphPartitioner currently only processes 'call_module' nodes")
    print(f"      (Conv2d, Linear, BatchNorm, ReLU, etc.)")
    print(f"      Other node types (call_function, call_method) are not yet partitioned.")

    if report.total_subgraphs > 0:
        print("\nSubgraph breakdown by operation type:")
        print("-" * 80)
        for op_type, count in sorted(report.operation_type_counts.items(),
                                     key=lambda x: x[1], reverse=True):
            pct = count / report.total_subgraphs * 100
            print(f"  {op_type:<30} {count:>5} ({pct:>5.1f}%)")

        # Show what was NOT partitioned
        print("\nNodes NOT converted to subgraphs:")
        print("-" * 80)
        if fx_analysis['call_function_counts']:
            print(f"  call_function nodes: {sum(fx_analysis['call_function_counts'].values())} nodes")
            for func_name, count in fx_analysis['call_function_counts'].items():
                print(f"    - {func_name}: {count}")
        if fx_analysis['call_method_counts']:
            print(f"  call_method nodes: {sum(fx_analysis['call_method_counts'].values())} nodes")
            for method_name, count in fx_analysis['call_method_counts'].items():
                print(f"    - {method_name}: {count}")

        total_not_partitioned = (sum(fx_analysis['call_function_counts'].values()) +
                                sum(fx_analysis['call_method_counts'].values()))
        if total_not_partitioned > 0:
            print(f"\n  Total: {total_not_partitioned} nodes not partitioned")
            print(f"  -> These operations are not included in FLOPs/memory analysis")
    else:
        print("\nWARNING: No subgraphs created!")
        print("   This might indicate:")
        print("   1. The model has no call_module nodes (unusual)")
        print("   2. FX tracing didn't capture the module structure")
        print("   3. GraphPartitioner needs to be enhanced for this model type")

    # Step 6: Visualize partitioning (first 15 nodes)
    print("[6/7] Generating partition visualization...")
    print("\n")
    visualization = partitioner.visualize_partitioning(fx_graph, max_nodes=15)
    print(visualization)

    # Step 7: Analyze concurrency
    print("\n[7/7] Analyzing concurrency...")
    analyzer = ConcurrencyAnalyzer()
    concurrency = analyzer.analyze(report)
    print(f"    Found {concurrency.num_stages} execution stages")

    # Display results
    print("\n" + "=" * 80)
    print("PARTITION SUMMARY")
    print("=" * 80)
    print(report.summary_stats())

    print("\n" + "=" * 80)
    print("CONCURRENCY ANALYSIS")
    print("=" * 80)
    print(concurrency.explanation)

    # Show top 5 most compute-intensive operations
    print("\n" + "=" * 80)
    print("TOP 5 OPERATIONS BY FLOPs")
    print("=" * 80)

    sorted_subgraphs = sorted(report.subgraphs, key=lambda sg: sg.flops, reverse=True)[:5]

    for i, sg in enumerate(sorted_subgraphs, 1):
        flop_pct = sg.flops / report.total_flops * 100
        parallelism_str = ""
        if sg.parallelism:
            parallelism_str = f"{sg.parallelism.total_threads:,} threads"

        print(f"\n{i}. {sg.node_name}")
        print(f"   Type: {sg.operation_type.value}")
        print(f"   FLOPs: {sg.flops / 1e9:.3f} G ({flop_pct:.1f}% of total)")
        print(f"   Arithmetic Intensity: {sg.arithmetic_intensity:.2f} FLOPs/byte")
        print(f"   Bottleneck: {sg.recommended_bottleneck.value}")
        print(f"   Parallelism: {parallelism_str}")
        print(f"   Partition Reason: {sg.partition_reason.value}")
        if sg.fusion_candidates:
            print(f"   Fusion Candidates: {len(sg.fusion_candidates)} operations")

    # Show partition reasoning examples
    print("\n" + "=" * 80)
    print("PARTITION REASONING EXAMPLES")
    print("=" * 80)
    print("\nShowing why operations were partitioned separately:\n")

    # Show a few diverse examples
    examples_shown = 0
    seen_reasons = set()
    for sg in report.subgraphs:
        if sg.partition_reason.value not in seen_reasons and examples_shown < 3:
            print(f"\n{sg.node_name} ({sg.operation_type.value}):")
            print(sg.partition_reasoning_summary())
            seen_reasons.add(sg.partition_reason.value)
            examples_shown += 1
        if examples_shown >= 3:
            break

    # Interactive exploration prompts
    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("\nNow that you've seen the basic output, try exploring:")
    print("\n1. See the full partition visualization:")
    print("   - python examples/visualize_partitioning.py 999")
    print("   - This shows all FX nodes and their subgraphs side-by-side")
    print("\n2. Look at different models:")
    print("   - Change 'resnet18' to 'mobilenet_v2' or 'efficientnet_b0'")
    print("   - Compare their concurrency and arithmetic intensity")
    print("\n3. Explore specific properties:")
    print("   - Print all subgraph names: for sg in report.subgraphs: print(sg.node_name)")
    print("   - Find depthwise convs: [sg for sg in report.subgraphs if sg.parallelism and sg.parallelism.is_depthwise]")
    print("   - Count bottleneck types: from collections import Counter; Counter(sg.recommended_bottleneck.value for sg in report.subgraphs)")
    print("\n4. Modify the input:")
    print("   - Try different batch sizes: torch.randn(8, 3, 224, 224)")
    print("   - See how parallelism changes")
    print("\n5. Check out the full tutorial:")
    print("   - docs/graph_partitioner_tutorial.md")
    print("\n6. Run validation tests:")
    print("   - python tests/test_graph_partitioner_general.py resnet18")
    print("   - python tests/test_graph_partitioner_general.py mobilenet_v2 efficientnet_b0")


if __name__ == "__main__":
    main()
