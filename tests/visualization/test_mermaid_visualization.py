#!/usr/bin/env python3
"""
Test script for Mermaid visualization functionality.

This script validates Phases 1, 2, and 3 of the Mermaid visualization system by:
1. Loading ResNet18 model
2. Running FX tracing and partitioning
3. Generating various Mermaid diagrams
4. Saving results to markdown files in a temp directory
"""

import sys
import os
import tempfile

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import pytest
import torch
import torch.fx as fx
from torchvision.models import resnet18

from graphs.transform.partitioning.fusion_partitioner import FusionBasedPartitioner
from graphs.visualization.mermaid_generator import MermaidGenerator, ColorScheme


def trace_resnet18():
    """Trace ResNet18 model with FX."""
    print("📦 Loading ResNet18...")
    model = resnet18(weights=None)
    model.eval()

    print("🔍 Tracing with PyTorch FX...")
    example_input = torch.randn(1, 3, 224, 224)

    # Symbolic trace
    traced = fx.symbolic_trace(model)

    # Run shape propagation
    from torch.fx.passes.shape_prop import ShapeProp
    ShapeProp(traced).propagate(example_input)

    print(f"✅ Traced successfully: {len(list(traced.graph.nodes))} nodes")
    return traced, example_input


def run_partitioning(traced_model, example_input):
    """Run fusion partitioning."""
    print("\n🔧 Running fusion partitioning...")

    partitioner = FusionBasedPartitioner()
    partition_report = partitioner.partition(traced_model)

    print(f"✅ Partitioned into {len(partition_report.fused_subgraphs)} subgraphs")

    # Add some mock metrics for visualization testing
    for idx, sg in enumerate(partition_report.fused_subgraphs):
        # Mock bottleneck percentages
        if idx % 3 == 0:
            sg.compute_bound_pct = 85.0
            sg.memory_bound_pct = 15.0
        elif idx % 3 == 1:
            sg.compute_bound_pct = 20.0
            sg.memory_bound_pct = 75.0
        else:
            sg.compute_bound_pct = 50.0
            sg.memory_bound_pct = 50.0

        # Mock latency (for bottleneck analysis)
        sg.latency_ms = 0.001 * (idx + 1) * (100 if idx < 5 else 10)

    return partition_report


# Pytest fixtures
@pytest.fixture
def traced_model():
    """Fixture to provide traced model."""
    model, _ = trace_resnet18()
    return model


@pytest.fixture
def partition_report(traced_model):
    """Fixture to provide partition report."""
    example_input = torch.randn(1, 3, 224, 224)
    return run_partitioning(traced_model, example_input)


def test_fx_graph_visualization(traced_model, tmp_path):
    """Test Phase 1: FX graph visualization."""
    print("\n" + "="*80)
    print("TEST 1: FX Graph Visualization (Phase 1)")
    print("="*80)

    generator = MermaidGenerator(style='colorful')

    # Generate basic FX graph
    print("Generating FX graph diagram...")
    diagram = generator.generate_fx_graph(
        traced_model,
        direction='TD',
        max_nodes=20,  # Limit for readability
        show_shapes=True,
        show_types=True
    )

    # Save to temp file
    output_file = tmp_path / 'test_fx_graph.md'
    with open(output_file, 'w') as f:
        f.write("# ResNet18: FX Graph Visualization\n\n")
        f.write("This diagram shows the first 20 nodes of the ResNet18 FX graph.\n\n")
        f.write("```mermaid\n")
        f.write(diagram)
        f.write("\n```\n")
        f.write(generator.generate_legend(ColorScheme.OP_TYPE))

    print(f"Saved to {output_file}")
    print(f"   Lines: {len(diagram.split(chr(10)))}")

    # Verify file was created and has content
    assert output_file.exists()
    assert output_file.stat().st_size > 0


def test_partitioned_graph_visualization(partition_report, tmp_path):
    """Test Phase 1 & 2: Partitioned graph with color schemes."""
    print("\n" + "="*80)
    print("TEST 2: Partitioned Graph Visualization (Phases 1 & 2)")
    print("="*80)

    generator = MermaidGenerator(style='default')

    # Test 1: Bottleneck color scheme
    print("Generating partitioned graph with bottleneck colors...")
    diagram = generator.generate_partitioned_graph(
        partition_report,
        direction='TD',
        color_by='bottleneck',
        show_metrics=True,
        max_subgraphs=15
    )

    output_file = tmp_path / 'test_partitioned_bottleneck.md'
    with open(output_file, 'w') as f:
        f.write("# ResNet18: Partitioned Graph (Bottleneck Analysis)\n\n")
        f.write("This diagram shows fused subgraphs colored by bottleneck type.\n\n")
        f.write("```mermaid\n")
        f.write(diagram)
        f.write("\n```\n\n")
        f.write(generator.generate_legend(ColorScheme.BOTTLENECK))

    print(f"Saved to {output_file}")
    assert output_file.exists()

    # Test 2: Operation type color scheme
    print("Generating partitioned graph with operation type colors...")
    generator2 = MermaidGenerator(style='colorful')
    diagram2 = generator2.generate_partitioned_graph(
        partition_report,
        direction='TD',
        color_by='op_type',
        show_metrics=True,
        max_subgraphs=15
    )

    output_file2 = tmp_path / 'test_partitioned_optype.md'
    with open(output_file2, 'w') as f:
        f.write("# ResNet18: Partitioned Graph (Operation Types)\n\n")
        f.write("This diagram shows fused subgraphs colored by operation type.\n\n")
        f.write("```mermaid\n")
        f.write(diagram2)
        f.write("\n```\n\n")
        f.write(generator2.generate_legend(ColorScheme.OP_TYPE))

    print(f"Saved to {output_file2}")
    assert output_file2.exists()


def test_hardware_mapping_visualization(partition_report, tmp_path):
    """Test Phase 3: Hardware mapping visualization."""
    print("\n" + "="*80)
    print("TEST 3: Hardware Mapping Visualization (Phase 3)")
    print("="*80)

    generator = MermaidGenerator(style='default')

    # Test H100 GPU mapping
    print("Generating H100 GPU hardware mapping...")
    diagram = generator.generate_hardware_mapping(
        partition_report,
        hardware_name="H100 GPU",
        peak_compute_units=132,  # 132 SMs
        direction='TD',
        show_allocation=True,
        show_utilization=True,
        max_subgraphs=15
    )

    output_file = tmp_path / 'test_hardware_mapping_h100.md'
    with open(output_file, 'w') as f:
        f.write("# ResNet18: H100 GPU Hardware Mapping\n\n")
        f.write("This diagram shows how ResNet18 subgraphs map to H100 GPU resources.\n\n")
        f.write("```mermaid\n")
        f.write(diagram)
        f.write("\n```\n\n")
        f.write(generator.generate_legend(ColorScheme.UTILIZATION))
        f.write("\n\n**Insights**:\n")
        f.write("- Green: High utilization (>60%) - efficient use of SMs\n")
        f.write("- Yellow: Medium utilization (40-60%) - moderate efficiency\n")
        f.write("- Pink/Red: Low utilization (<40%) - underutilized hardware\n")
        f.write("- Red 'IDLE' box shows unused compute resources\n")

    print(f"Saved to {output_file}")
    assert output_file.exists()

    # Test TPU mapping
    print("Generating TPU-v4 hardware mapping...")
    diagram2 = generator.generate_hardware_mapping(
        partition_report,
        hardware_name="TPU-v4",
        peak_compute_units=2,  # 2 MXUs
        direction='TD',
        show_allocation=True,
        show_utilization=True,
        max_subgraphs=15
    )

    output_file2 = tmp_path / 'test_hardware_mapping_tpu.md'
    with open(output_file2, 'w') as f:
        f.write("# ResNet18: TPU-v4 Hardware Mapping\n\n")
        f.write("This diagram shows how ResNet18 subgraphs map to TPU-v4 resources.\n\n")
        f.write("```mermaid\n")
        f.write(diagram2)
        f.write("\n```\n\n")
        f.write(generator.generate_legend(ColorScheme.UTILIZATION))
        f.write("\n\n**Insights**:\n")
        f.write("- TPU has only 2 MXUs (Matrix Multiplier Units)\n")
        f.write("- Small model shows severe underutilization\n")
        f.write("- Most operations can't saturate even 1 MXU\n")

    print(f"Saved to {output_file2}")
    assert output_file2.exists()


def test_architecture_comparison(partition_report, tmp_path):
    """Test Phase 3: Multi-architecture comparison."""
    print("\n" + "="*80)
    print("TEST 4: Architecture Comparison (Phase 3)")
    print("="*80)

    generator = MermaidGenerator(style='default')

    # Create comparison with CPU, GPU, TPU
    print("Generating 3-way architecture comparison...")

    # We'll use the same partition report but show it would look with different resources
    partition_reports = [
        ("CPU (60 cores)", partition_report),
        ("H100 GPU (132 SMs)", partition_report),
        ("TPU-v4 (2 MXUs)", partition_report)
    ]

    peak_units = [60, 132, 2]

    diagram = generator.generate_architecture_comparison(
        partition_reports,
        peak_units,
        layout='side_by_side',
        max_subgraphs=8
    )

    output_file = tmp_path / 'test_architecture_comparison.md'
    with open(output_file, 'w') as f:
        f.write("# ResNet18: CPU vs GPU vs TPU Comparison\n\n")
        f.write("This diagram shows how the same ResNet18 graph executes on 3 different architectures.\n\n")
        f.write("```mermaid\n")
        f.write(diagram)
        f.write("\n```\n\n")
        f.write(generator.generate_legend(ColorScheme.UTILIZATION))
        f.write("\n\n**Key Observations**:\n")
        f.write("- **CPU**: Moderate utilization (40-60%), well-balanced\n")
        f.write("- **GPU**: Lower utilization due to massive parallelism not fully used\n")
        f.write("- **TPU**: Severe underutilization with only 2 large MXUs\n")

    print(f"Saved to {output_file}")
    assert output_file.exists()


def test_bottleneck_analysis(partition_report, tmp_path):
    """Test Phase 2: Bottleneck analysis visualization."""
    print("\n" + "="*80)
    print("TEST 5: Bottleneck Analysis (Phase 2)")
    print("="*80)

    generator = MermaidGenerator(style='default')

    print("Generating bottleneck analysis diagram...")
    diagram = generator.generate_bottleneck_analysis(
        partition_report,
        threshold=0.15,  # Highlight ops using >15% of time
        direction='TD',
        max_subgraphs=20
    )

    output_file = tmp_path / 'test_bottleneck_analysis.md'
    with open(output_file, 'w') as f:
        f.write("# ResNet18: Bottleneck Analysis\n\n")
        f.write("This diagram highlights operations that dominate execution time.\n\n")
        f.write("```mermaid\n")
        f.write(diagram)
        f.write("\n```\n\n")
        f.write("**Legend**:\n")
        f.write("- Red (thick border): Critical bottleneck (>20% of total time)\n")
        f.write("- Pink: Significant contributor (15-20% of time)\n")
        f.write("- Yellow: Moderate contributor (10-15% of time)\n")
        f.write("- Gray: Minor contributor (<10% of time)\n")
        f.write("\n\n**Optimization Priority**:\n")
        f.write("Focus optimization efforts on the critical bottleneck operations.\n")

    print(f"Saved to {output_file}")
    assert output_file.exists()


def create_comprehensive_report(traced_model, partition_report, output_dir):
    """Create a comprehensive markdown report with all visualizations.

    Args:
        traced_model: Traced FX model
        partition_report: Partition report from FusionBasedPartitioner
        output_dir: Directory to write output files (pathlib.Path or str)
    """
    print("\n" + "="*80)
    print("CREATING COMPREHENSIVE REPORT")
    print("="*80)

    generator = MermaidGenerator(style='default')

    output_file = os.path.join(output_dir, 'mermaid_visualization_demo.md')

    with open(output_file, 'w') as f:
        f.write("# Mermaid Visualization Demo: ResNet18\n\n")
        f.write("This document demonstrates the Mermaid visualization capabilities (Phases 1-3).\n\n")

        f.write("---\n\n")

        # Section 1: FX Graph
        f.write("## 1. FX Graph Structure (Phase 1)\n\n")
        f.write("Shows the raw PyTorch FX graph structure with operation types.\n\n")
        diagram = generator.generate_fx_graph(traced_model, max_nodes=15, show_shapes=False)
        f.write("```mermaid\n")
        f.write(diagram)
        f.write("\n```\n\n")

        # Section 2: Partitioned Graph
        f.write("---\n\n")
        f.write("## 2. Partitioned Graph (Phases 1 & 2)\n\n")
        f.write("Shows fused subgraphs with bottleneck analysis.\n\n")
        diagram = generator.generate_partitioned_graph(
            partition_report,
            color_by='bottleneck',
            show_metrics=True,
            max_subgraphs=12
        )
        f.write("```mermaid\n")
        f.write(diagram)
        f.write("\n```\n\n")
        f.write(generator.generate_legend(ColorScheme.BOTTLENECK))

        # Section 3: Hardware Mapping
        f.write("\n---\n\n")
        f.write("## 3. Hardware Mapping: H100 GPU (Phase 3)\n\n")
        f.write("Shows how subgraphs map to H100 GPU streaming multiprocessors.\n\n")
        diagram = generator.generate_hardware_mapping(
            partition_report,
            "H100 GPU",
            132,
            max_subgraphs=12
        )
        f.write("```mermaid\n")
        f.write(diagram)
        f.write("\n```\n\n")
        f.write(generator.generate_legend(ColorScheme.UTILIZATION))

        # Section 4: Bottleneck Analysis
        f.write("\n---\n\n")
        f.write("## 4. Bottleneck Analysis (Phase 2)\n\n")
        f.write("Identifies operations that dominate execution time.\n\n")
        diagram = generator.generate_bottleneck_analysis(
            partition_report,
            threshold=0.15,
            max_subgraphs=15
        )
        f.write("```mermaid\n")
        f.write(diagram)
        f.write("\n```\n\n")

        # Section 5: Summary
        f.write("\n---\n\n")
        f.write("## Summary\n\n")
        f.write("This demo shows all visualization types implemented in Phases 1-3:\n\n")
        f.write("- **Phase 1**: FX graph and partitioned graph visualization\n")
        f.write("- **Phase 2**: Color schemes (bottleneck, utilization, op_type) and legends\n")
        f.write("- **Phase 3**: Hardware mapping with resource allocation\n\n")
        f.write("**Next Steps**: Test these visualizations with real analysis data from the unified analyzer.\n")

    print(f"Created comprehensive report: {output_file}")


def main():
    """Run all tests standalone (outputs to temp directory)."""
    from pathlib import Path

    print("Testing Mermaid Visualization (Phases 1-3)\n")

    # Create temp directory for outputs
    output_dir = Path(tempfile.mkdtemp(prefix='mermaid_viz_'))
    print(f"Output directory: {output_dir}\n")

    # Trace model
    traced_model, example_input = trace_resnet18()

    # Run partitioning
    partition_report = run_partitioning(traced_model, example_input)

    # Run tests (pass output_dir as tmp_path equivalent)
    test_fx_graph_visualization(traced_model, output_dir)
    test_partitioned_graph_visualization(partition_report, output_dir)
    test_hardware_mapping_visualization(partition_report, output_dir)
    test_architecture_comparison(partition_report, output_dir)
    test_bottleneck_analysis(partition_report, output_dir)

    # Create comprehensive report
    create_comprehensive_report(traced_model, partition_report, output_dir)

    print("\n" + "="*80)
    print("ALL TESTS PASSED!")
    print("="*80)
    print(f"\nGenerated files in: {output_dir}")
    for f in sorted(output_dir.iterdir()):
        print(f"  - {f.name}")
    print("\nYou can view these files in any markdown viewer that supports Mermaid.")


if __name__ == '__main__':
    main()
