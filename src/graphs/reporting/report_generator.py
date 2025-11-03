"""
Report Generator

Flexible reporting engine that transforms UnifiedAnalysisResult into various output formats.

Supports:
- Text: Human-readable console output
- JSON: Machine-readable, preserves all data
- CSV: Spreadsheet-friendly, flattened data
- Markdown: Documentation-friendly with tables
- HTML: Rich formatted reports (basic implementation)

Usage:
    from graphs.reporting import ReportGenerator
    from graphs.analysis.unified_analyzer import UnifiedAnalyzer

    # Analyze model
    analyzer = UnifiedAnalyzer()
    result = analyzer.analyze_model('resnet18', 'H100')

    # Generate reports
    generator = ReportGenerator()

    # Text report
    print(generator.generate_text_report(result))

    # JSON export
    generator.save_report(result, 'report.json')

    # Comparison report
    results = [result1, result2, result3]
    print(generator.generate_comparison_report(results))
"""

import json
import csv
from io import StringIO
from typing import List, Dict, Any, Optional, Union
from pathlib import Path

from graphs.analysis.unified_analyzer import UnifiedAnalysisResult
from graphs.visualization.mermaid_generator import MermaidGenerator, ColorScheme


class ReportGenerator:
    """
    Flexible report generation from unified analysis results.

    Supports multiple output formats:
    - Text: Human-readable console output
    - JSON: Machine-readable, preserves all data
    - CSV: Spreadsheet-friendly, flattened data
    - Markdown: Documentation-friendly with tables
    - HTML: Rich formatted reports with styling

    Usage:
        generator = ReportGenerator()

        # Text report
        text = generator.generate_text_report(result)
        print(text)

        # JSON export
        json_data = generator.generate_json_report(result)
        with open('report.json', 'w') as f:
            f.write(json_data)

        # Comparison report
        comparison = generator.generate_comparison_report([result1, result2, result3])
        print(comparison)
    """

    def __init__(self, style: str = 'default'):
        """
        Initialize report generator.

        Args:
            style: Report style ('default', 'compact', 'detailed')
        """
        self.style = style
        self.mermaid_generator = MermaidGenerator()

    # =========================================================================
    # Single Model Reports
    # =========================================================================

    def generate_text_report(
        self,
        result: UnifiedAnalysisResult,
        include_sections: Optional[List[str]] = None,
        show_executive_summary: bool = True
    ) -> str:
        """
        Generate human-readable text report.

        Args:
            result: Analysis result
            include_sections: List of sections to include (all if None)
                             ['executive', 'performance', 'energy', 'memory', 'recommendations']
            show_executive_summary: Include executive summary at top

        Returns:
            Formatted text report
        """
        lines = []

        # Header
        lines.append("=" * 79)
        lines.append("                 COMPREHENSIVE ANALYSIS REPORT")
        lines.append("=" * 79)
        lines.append("")

        # Executive Summary
        if show_executive_summary and (include_sections is None or 'executive' in include_sections):
            lines.append("EXECUTIVE SUMMARY")
            lines.append("-" * 79)
            lines.append(f"Model:                   {result.display_name}")
            lines.append(f"Hardware:                {result.hardware_display_name}")
            lines.append(f"Precision:               {result.precision.name}")
            lines.append(f"Batch Size:              {result.batch_size}")
            lines.append("")
            lines.append(f"Performance:             {result.total_latency_ms:.2f} ms latency, {result.throughput_fps:.1f} fps")

            if result.energy_report:
                lines.append(f"Energy:                  {result.total_energy_mj:.1f} mJ total ({result.energy_per_inference_mj:.1f} mJ/inference)")
                static_pct = (result.energy_report.static_energy_j /
                             (result.energy_report.compute_energy_j + result.energy_report.memory_energy_j + result.energy_report.static_energy_j)) * 100
                lines.append(f"Energy per Inference:    {result.energy_per_inference_mj:.1f} mJ ({static_pct:.0f}% static overhead)")

            lines.append(f"Efficiency:              {result.average_utilization_pct:.1f}% hardware utilization")
            lines.append("")

            if result.memory_report:
                lines.append(f"Memory:                  Peak {result.peak_memory_mb:.1f} MB")
                lines.append(f"                         (activations: {result.memory_report.activation_memory_bytes/1e6:.1f} MB, "
                           f"weights: {result.memory_report.weight_memory_bytes/1e6:.1f} MB)")

                if result.memory_report.fits_in_l2_cache:
                    lines.append(f"                         ✓ Fits in L2 cache ({result.memory_report.l2_cache_size_bytes/1e6:.1f} MB)")
                else:
                    lines.append(f"                         ✗ Does not fit in L2 cache ({result.memory_report.l2_cache_size_bytes/1e6:.1f} MB)")

            lines.append("")

        # Performance Analysis
        if include_sections is None or 'performance' in include_sections:
            if result.roofline_report:
                lines.append("PERFORMANCE ANALYSIS")
                lines.append("-" * 79)
                lines.append(f"Total Latency:           {result.total_latency_ms:.2f} ms")
                lines.append(f"Throughput:              {result.throughput_fps:.1f} fps")
                lines.append(f"Hardware Utilization:    {result.average_utilization_pct:.1f}%")
                lines.append(f"Total FLOPs:             {result.partition_report.total_flops / 1e9:.2f} GFLOPs")
                lines.append(f"Subgraphs:               {len(result.partition_report.subgraphs)}")

                # Bottleneck analysis
                memory_bound = sum(1 for lat in result.roofline_report.latencies if lat.bottleneck == 'memory')
                compute_bound = len(result.roofline_report.latencies) - memory_bound
                lines.append(f"Bottlenecks:             {compute_bound} compute-bound, {memory_bound} memory-bound")
                lines.append("")

        # Energy Analysis
        if include_sections is None or 'energy' in include_sections:
            if result.energy_report:
                lines.append("ENERGY ANALYSIS")
                lines.append("-" * 79)
                lines.append(f"Total Energy:            {result.total_energy_mj:.1f} mJ")
                lines.append(f"  Compute Energy:        {result.energy_report.compute_energy_j * 1000:.1f} mJ")
                lines.append(f"  Memory Energy:         {result.energy_report.memory_energy_j * 1000:.1f} mJ")
                lines.append(f"  Static Energy:         {result.energy_report.static_energy_j * 1000:.1f} mJ")
                lines.append("")
                lines.append(f"Energy per Inference:    {result.energy_per_inference_mj:.1f} mJ")
                lines.append(f"Average Power:           {result.energy_report.average_power_w:.1f} W")
                lines.append(f"Peak Power:              {result.energy_report.peak_power_w:.1f} W")
                lines.append(f"Energy Efficiency:       {result.energy_report.average_efficiency * 100:.1f}%")

                # Power Management section (if hardware mapping was used)
                if result.energy_report.average_allocated_units > 0:
                    lines.append("")
                    lines.append("Power Management:")
                    lines.append(f"  Average Units Allocated: {result.energy_report.average_allocated_units:.1f}")
                    lines.append(f"  Allocated Units Idle:    {result.energy_report.total_allocated_units_energy_j * 1000:.1f} mJ")
                    lines.append(f"  Unallocated Units Idle:  {result.energy_report.total_unallocated_units_energy_j * 1000:.1f} mJ")
                    if result.energy_report.power_gating_enabled:
                        savings_pct = (result.energy_report.total_power_gating_savings_j /
                                      (result.energy_report.total_power_gating_savings_j + result.energy_report.static_energy_j) * 100)
                        lines.append(f"  Power Gating:            ENABLED")
                        lines.append(f"  Power Gating Savings:    {result.energy_report.total_power_gating_savings_j * 1000:.1f} mJ ({savings_pct:.1f}%)")
                    else:
                        lines.append(f"  Power Gating:            DISABLED (conservative estimate)")

                lines.append("")

        # Memory Analysis
        if include_sections is None or 'memory' in include_sections:
            if result.memory_report:
                lines.append("MEMORY ANALYSIS")
                lines.append("-" * 79)
                lines.append(f"Peak Memory:             {result.peak_memory_mb:.1f} MB")
                lines.append(f"  Activations:           {result.memory_report.activation_memory_bytes/1e6:.1f} MB")
                lines.append(f"  Weights:               {result.memory_report.weight_memory_bytes/1e6:.1f} MB")
                lines.append(f"  Workspace:             {result.memory_report.workspace_memory_bytes/1e6:.1f} MB")
                lines.append("")
                lines.append(f"Average Memory:          {result.memory_report.average_memory_bytes/1e6:.1f} MB")
                lines.append(f"Memory Utilization:      {result.memory_report.memory_utilization * 100:.1f}%")
                lines.append("")

                # Hardware fit
                lines.append("Hardware Fit:")
                lines.append(f"  L2 Cache ({result.memory_report.l2_cache_size_bytes/1e6:.1f} MB):       {'✓ Fits' if result.memory_report.fits_in_l2_cache else '✗ Does not fit'}")
                lines.append(f"  Device Memory ({result.memory_report.device_memory_bytes/1e9:.1f} GB):  {'✓ Fits' if result.memory_report.fits_on_device else '✗ Does not fit'}")
                lines.append("")

        # Recommendations
        if include_sections is None or 'recommendations' in include_sections:
            recommendations = result._generate_recommendations()
            if recommendations:
                lines.append("RECOMMENDATIONS")
                lines.append("-" * 79)
                for i, rec in enumerate(recommendations, 1):
                    lines.append(f"  {i}. {rec}")
                lines.append("")

        # Validation Warnings
        if result.validation_warnings:
            lines.append("VALIDATION WARNINGS")
            lines.append("-" * 79)
            for warning in result.validation_warnings:
                lines.append(f"  ⚠ {warning}")
            lines.append("")

        return "\n".join(lines)

    def generate_json_report(
        self,
        result: UnifiedAnalysisResult,
        include_raw_reports: bool = True,
        pretty_print: bool = True
    ) -> str:
        """
        Generate JSON report.

        Args:
            result: Analysis result
            include_raw_reports: Include full Phase 3 reports
            pretty_print: Return formatted string vs compact

        Returns:
            JSON string
        """
        data = {
            'metadata': {
                'model': result.display_name,
                'model_internal_name': result.model_name,
                'hardware': result.hardware_display_name,
                'batch_size': result.batch_size,
                'precision': result.precision.name,
                'timestamp': result.analysis_timestamp,
            },
            'executive_summary': result.get_executive_summary(),
            'derived_metrics': {
                'latency_ms': result.total_latency_ms,
                'throughput_fps': result.throughput_fps,
                'total_energy_mj': result.total_energy_mj,
                'energy_per_inference_mj': result.energy_per_inference_mj,
                'peak_memory_mb': result.peak_memory_mb,
                'utilization_pct': result.average_utilization_pct,
            },
        }

        # Include raw reports if requested
        if include_raw_reports:
            if result.roofline_report:
                data['roofline_analysis'] = {
                    'total_latency_s': sum(lat.actual_latency for lat in result.roofline_report.latencies),
                    'average_utilization': result.roofline_report.average_flops_utilization,
                    'memory_bound_count': sum(1 for lat in result.roofline_report.latencies if lat.bottleneck == 'memory'),
                    'compute_bound_count': sum(1 for lat in result.roofline_report.latencies if lat.bottleneck == 'compute'),
                }

            if result.energy_report:
                data['energy_analysis'] = {
                    'total_energy_j': result.energy_report.total_energy_j,
                    'compute_energy_j': result.energy_report.compute_energy_j,
                    'memory_energy_j': result.energy_report.memory_energy_j,
                    'static_energy_j': result.energy_report.static_energy_j,
                    'average_power_w': result.energy_report.average_power_w,
                    'peak_power_w': result.energy_report.peak_power_w,
                    'average_efficiency': result.energy_report.average_efficiency,
                }

                # Add power management fields if hardware mapping was used
                if result.energy_report.average_allocated_units > 0:
                    data['energy_analysis']['power_management'] = {
                        'average_allocated_units': result.energy_report.average_allocated_units,
                        'total_allocated_units_energy_j': result.energy_report.total_allocated_units_energy_j,
                        'total_unallocated_units_energy_j': result.energy_report.total_unallocated_units_energy_j,
                        'power_gating_enabled': result.energy_report.power_gating_enabled,
                        'total_power_gating_savings_j': result.energy_report.total_power_gating_savings_j,
                    }

            if result.memory_report:
                data['memory_analysis'] = {
                    'peak_memory_bytes': result.memory_report.peak_memory_bytes,
                    'activation_memory_bytes': result.memory_report.activation_memory_bytes,
                    'weight_memory_bytes': result.memory_report.weight_memory_bytes,
                    'workspace_memory_bytes': result.memory_report.workspace_memory_bytes,
                    'average_memory_bytes': result.memory_report.average_memory_bytes,
                    'memory_utilization': result.memory_report.memory_utilization,
                    'fits_in_l2_cache': result.memory_report.fits_in_l2_cache,
                    'fits_on_device': result.memory_report.fits_on_device,
                }

        # Add recommendations
        data['recommendations'] = result._generate_recommendations()

        # Add validation warnings
        if result.validation_warnings:
            data['validation_warnings'] = result.validation_warnings

        if pretty_print:
            return json.dumps(data, indent=2)
        else:
            return json.dumps(data)

    def generate_csv_report(
        self,
        result: UnifiedAnalysisResult,
        include_subgraph_details: bool = False
    ) -> str:
        """
        Generate CSV report (flattened data).

        Args:
            result: Analysis result
            include_subgraph_details: Include per-subgraph rows

        Returns:
            CSV string
        """
        output = StringIO()

        if not include_subgraph_details:
            # Single row summary
            writer = csv.DictWriter(output, fieldnames=[
                'model', 'hardware', 'batch_size', 'precision',
                'latency_ms', 'throughput_fps', 'energy_mj', 'energy_per_inf_mj',
                'peak_mem_mb', 'utilization_pct',
                'compute_energy_mj', 'memory_energy_mj', 'static_energy_mj',
                'activation_mem_mb', 'weight_mem_mb'
            ])
            writer.writeheader()

            row = {
                'model': result.display_name,
                'hardware': result.hardware_display_name,
                'batch_size': result.batch_size,
                'precision': result.precision.name,
                'latency_ms': round(result.total_latency_ms, 2),
                'throughput_fps': round(result.throughput_fps, 1),
                'energy_mj': round(result.total_energy_mj, 2),
                'energy_per_inf_mj': round(result.energy_per_inference_mj, 2),
                'peak_mem_mb': round(result.peak_memory_mb, 1),
                'utilization_pct': round(result.average_utilization_pct, 1),
            }

            if result.energy_report:
                row['compute_energy_mj'] = round(result.energy_report.compute_energy_j * 1000, 2)
                row['memory_energy_mj'] = round(result.energy_report.memory_energy_j * 1000, 2)
                row['static_energy_mj'] = round(result.energy_report.static_energy_j * 1000, 2)

            if result.memory_report:
                row['activation_mem_mb'] = round(result.memory_report.activation_memory_bytes / 1e6, 1)
                row['weight_mem_mb'] = round(result.memory_report.weight_memory_bytes / 1e6, 1)

            writer.writerow(row)

        else:
            # Per-subgraph details
            writer = csv.DictWriter(output, fieldnames=[
                'model', 'hardware', 'batch_size', 'precision', 'subgraph_id',
                'flops', 'memory_bytes', 'bottleneck', 'latency_ms',
                'energy_mj', 'compute_energy_mj', 'memory_energy_mj', 'static_energy_mj'
            ])
            writer.writeheader()

            for i, sg in enumerate(result.partition_report.subgraphs):
                row = {
                    'model': result.display_name,
                    'hardware': result.hardware_display_name,
                    'batch_size': result.batch_size,
                    'precision': result.precision.name,
                    'subgraph_id': i,
                    'flops': sg.flops,
                    'memory_bytes': sg.total_input_bytes + sg.total_output_bytes + sg.total_weight_bytes,
                }

                if result.roofline_report and i < len(result.roofline_report.latencies):
                    lat = result.roofline_report.latencies[i]
                    row['bottleneck'] = lat.bottleneck
                    row['latency_ms'] = round(lat.actual_latency * 1000, 3)

                if result.energy_report and i < len(result.energy_report.energy_descriptors):
                    desc = result.energy_report.energy_descriptors[i]
                    row['energy_mj'] = round(desc.total_energy_j * 1000, 3)
                    row['compute_energy_mj'] = round(desc.compute_energy_j * 1000, 3)
                    row['memory_energy_mj'] = round(desc.memory_energy_j * 1000, 3)
                    row['static_energy_mj'] = round(desc.static_energy_j * 1000, 3)

                writer.writerow(row)

        return output.getvalue()

    def generate_markdown_report(
        self,
        result: UnifiedAnalysisResult,
        include_tables: bool = True,
        include_charts: bool = False,
        include_diagrams: bool = False,
        diagram_types: Optional[List[str]] = None
    ) -> str:
        """
        Generate Markdown report.

        Args:
            result: Analysis result
            include_tables: Include formatted tables
            include_charts: Include ASCII charts (experimental)
            include_diagrams: Include Mermaid diagrams
            diagram_types: List of diagram types to include:
                          ['partitioned', 'bottleneck', 'hardware_mapping']
                          If None, includes all available diagrams

        Returns:
            Markdown string
        """
        lines = []

        # Header
        lines.append(f"# Analysis Report: {result.display_name}")
        lines.append("")
        lines.append(f"**Hardware:** {result.hardware_display_name}  ")
        lines.append(f"**Precision:** {result.precision.name}  ")
        lines.append(f"**Batch Size:** {result.batch_size}  ")
        lines.append(f"**Timestamp:** {result.analysis_timestamp}  ")
        lines.append("")

        # Executive Summary
        lines.append("## Executive Summary")
        lines.append("")
        summary = result.get_executive_summary()

        if include_tables:
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")
            lines.append(f"| Latency | {result.total_latency_ms:.2f} ms |")
            lines.append(f"| Throughput | {result.throughput_fps:.1f} fps |")
            lines.append(f"| Total Energy | {result.total_energy_mj:.1f} mJ |")
            lines.append(f"| Energy/Inference | {result.energy_per_inference_mj:.1f} mJ |")
            lines.append(f"| Peak Memory | {result.peak_memory_mb:.1f} MB |")
            lines.append(f"| Utilization | {result.average_utilization_pct:.1f}% |")
        else:
            lines.append(f"- **Latency:** {result.total_latency_ms:.2f} ms")
            lines.append(f"- **Throughput:** {result.throughput_fps:.1f} fps")
            lines.append(f"- **Total Energy:** {result.total_energy_mj:.1f} mJ")
            lines.append(f"- **Energy/Inference:** {result.energy_per_inference_mj:.1f} mJ")
            lines.append(f"- **Peak Memory:** {result.peak_memory_mb:.1f} MB")
            lines.append(f"- **Utilization:** {result.average_utilization_pct:.1f}%")

        lines.append("")

        # Performance Details
        if result.roofline_report:
            lines.append("## Performance Analysis")
            lines.append("")
            lines.append(f"- **Total FLOPs:** {result.partition_report.total_flops / 1e9:.2f} GFLOPs")
            lines.append(f"- **Subgraphs:** {len(result.partition_report.subgraphs)}")

            memory_bound = sum(1 for lat in result.roofline_report.latencies if lat.bottleneck == 'memory')
            compute_bound = len(result.roofline_report.latencies) - memory_bound
            lines.append(f"- **Compute-bound operations:** {compute_bound}")
            lines.append(f"- **Memory-bound operations:** {memory_bound}")
            lines.append("")

            # Add diagrams if requested
            if include_diagrams:
                if diagram_types is None:
                    diagram_types = ['partitioned', 'bottleneck']

                if 'partitioned' in diagram_types and result.partition_report:
                    lines.append("### Graph Partitioning Visualization")
                    lines.append("")
                    diagram = self.mermaid_generator.generate_partitioned_graph(
                        result.partition_report,
                        direction='TD',
                        color_by='bottleneck',
                        show_metrics=True,
                        max_subgraphs=15
                    )
                    lines.append("```mermaid")
                    lines.append(diagram)
                    lines.append("```")
                    lines.append("")
                    lines.append(self.mermaid_generator.generate_legend(ColorScheme.BOTTLENECK))
                    lines.append("")

                if 'bottleneck' in diagram_types and result.partition_report:
                    lines.append("### Bottleneck Analysis")
                    lines.append("")
                    lines.append("Critical path analysis showing operations by execution time:")
                    lines.append("")
                    diagram = self.mermaid_generator.generate_bottleneck_analysis(
                        result.partition_report,
                        threshold=0.15
                    )
                    lines.append("```mermaid")
                    lines.append(diagram)
                    lines.append("```")
                    lines.append("")

        # Energy Details
        if result.energy_report:
            lines.append("## Energy Analysis")
            lines.append("")

            if include_tables:
                lines.append("| Component | Energy (mJ) | Percentage |")
                lines.append("|-----------|-------------|------------|")
                total = result.total_energy_mj
                compute_pct = (result.energy_report.compute_energy_j * 1000 / total * 100) if total > 0 else 0
                memory_pct = (result.energy_report.memory_energy_j * 1000 / total * 100) if total > 0 else 0
                static_pct = (result.energy_report.static_energy_j * 1000 / total * 100) if total > 0 else 0

                lines.append(f"| Compute | {result.energy_report.compute_energy_j * 1000:.1f} | {compute_pct:.1f}% |")
                lines.append(f"| Memory | {result.energy_report.memory_energy_j * 1000:.1f} | {memory_pct:.1f}% |")
                lines.append(f"| Static/Leakage | {result.energy_report.static_energy_j * 1000:.1f} | {static_pct:.1f}% |")
                lines.append(f"| **Total** | **{total:.1f}** | **100.0%** |")
            else:
                lines.append(f"- **Compute Energy:** {result.energy_report.compute_energy_j * 1000:.1f} mJ")
                lines.append(f"- **Memory Energy:** {result.energy_report.memory_energy_j * 1000:.1f} mJ")
                lines.append(f"- **Static Energy:** {result.energy_report.static_energy_j * 1000:.1f} mJ")

            lines.append("")
            lines.append(f"- **Average Power:** {result.energy_report.average_power_w:.1f} W")
            lines.append(f"- **Peak Power:** {result.energy_report.peak_power_w:.1f} W")
            lines.append("")

        # Memory Details
        if result.memory_report:
            lines.append("## Memory Analysis")
            lines.append("")

            if include_tables:
                lines.append("| Component | Memory (MB) |")
                lines.append("|-----------|-------------|")
                lines.append(f"| Activations | {result.memory_report.activation_memory_bytes/1e6:.1f} |")
                lines.append(f"| Weights | {result.memory_report.weight_memory_bytes/1e6:.1f} |")
                lines.append(f"| Workspace | {result.memory_report.workspace_memory_bytes/1e6:.1f} |")
                lines.append(f"| **Peak Total** | **{result.peak_memory_mb:.1f}** |")
            else:
                lines.append(f"- **Activations:** {result.memory_report.activation_memory_bytes/1e6:.1f} MB")
                lines.append(f"- **Weights:** {result.memory_report.weight_memory_bytes/1e6:.1f} MB")
                lines.append(f"- **Workspace:** {result.memory_report.workspace_memory_bytes/1e6:.1f} MB")
                lines.append(f"- **Peak Total:** {result.peak_memory_mb:.1f} MB")

            lines.append("")

        # Recommendations
        recommendations = result._generate_recommendations()
        if recommendations:
            lines.append("## Recommendations")
            lines.append("")
            for i, rec in enumerate(recommendations, 1):
                lines.append(f"{i}. {rec}")
            lines.append("")

        return "\n".join(lines)

    def generate_html_report(
        self,
        result: UnifiedAnalysisResult,
        include_diagrams: bool = True,
        diagram_types: Optional[List[str]] = None,
        title: Optional[str] = None,
        description: Optional[str] = None
    ) -> str:
        """
        Generate interactive HTML report with Mermaid diagrams.

        Args:
            result: Analysis result
            include_diagrams: Include interactive Mermaid diagrams
            diagram_types: List of diagram types to include:
                          ['partitioned', 'bottleneck', 'hardware_mapping']
                          If None, includes partitioned and bottleneck diagrams
            title: Custom page title (auto-generated if None)
            description: Custom description (auto-generated if None)

        Returns:
            Complete HTML document as string
        """
        # Generate title and description
        if title is None:
            title = f"{result.display_name} Analysis - {result.hardware_display_name}"

        if description is None:
            description = (
                f"Comprehensive performance analysis of {result.display_name} on {result.hardware_display_name} "
                f"(Batch: {result.batch_size}, Precision: {result.precision.name})"
            )

        # Determine which diagrams to include
        if diagram_types is None:
            diagram_types = ['partitioned', 'bottleneck']

        # Generate Mermaid diagrams
        diagrams_html = ""
        if include_diagrams and result.partition_report:
            for diagram_type in diagram_types:
                if diagram_type == 'partitioned':
                    diagram = self.mermaid_generator.generate_partitioned_graph(
                        result.partition_report,
                        direction='TD',
                        color_by='bottleneck',
                        show_metrics=True,
                        max_subgraphs=20
                    )
                    diagrams_html += f"""
                    <div class="diagram-section">
                        <h2>Graph Partitioning Visualization</h2>
                        <p>Fused subgraphs colored by bottleneck type (green=compute-bound, red=memory-bound)</p>
                        <div class="mermaid-container">
                            <div class="mermaid">
{diagram}
                            </div>
                        </div>
                    </div>
                    """

                elif diagram_type == 'bottleneck':
                    diagram = self.mermaid_generator.generate_bottleneck_analysis(
                        result.partition_report,
                        threshold=0.15,
                        max_subgraphs=25
                    )
                    diagrams_html += f"""
                    <div class="diagram-section">
                        <h2>Bottleneck Analysis</h2>
                        <p>Critical path analysis highlighting operations consuming >15% of total execution time</p>
                        <div class="mermaid-container">
                            <div class="mermaid">
{diagram}
                            </div>
                        </div>
                    </div>
                    """

                elif diagram_type == 'hardware_mapping' and result.partition_report:
                    # Estimate peak compute units from hardware name
                    peak_units = 132  # Default (H100 SMs)
                    if 'A100' in result.hardware_display_name:
                        peak_units = 108
                    elif 'V100' in result.hardware_display_name:
                        peak_units = 80
                    elif 'Jetson' in result.hardware_display_name:
                        peak_units = 16
                    elif 'TPU' in result.hardware_display_name:
                        peak_units = 128
                    elif 'KPU' in result.hardware_display_name:
                        peak_units = 256

                    diagram = self.mermaid_generator.generate_hardware_mapping(
                        result.partition_report,
                        result.hardware_display_name,
                        peak_units,
                        direction='TD',
                        show_allocation=True,
                        show_utilization=True,
                        max_subgraphs=15
                    )
                    diagrams_html += f"""
                    <div class="diagram-section">
                        <h2>Hardware Resource Mapping</h2>
                        <p>Subgraph allocation to {result.hardware_display_name} compute units</p>
                        <div class="mermaid-container">
                            <div class="mermaid">
{diagram}
                            </div>
                        </div>
                    </div>
                    """

        # Build HTML document
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen', 'Ubuntu', sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #e0e0e0;
            min-height: 100vh;
            padding: 20px;
            line-height: 1.6;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}

        .header {{
            text-align: center;
            margin-bottom: 30px;
            padding: 30px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 12px;
            backdrop-filter: blur(10px);
        }}

        .header h1 {{
            font-size: 2.5rem;
            background: linear-gradient(135deg, #b794f6 0%, #7c3aed 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 10px;
        }}

        .description {{
            color: #cbd5e0;
            font-size: 1.1rem;
        }}

        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}

        .metric-card {{
            background: rgba(255, 255, 255, 0.05);
            padding: 20px;
            border-radius: 12px;
            border: 1px solid rgba(183, 148, 246, 0.3);
            transition: all 0.3s ease;
        }}

        .metric-card:hover {{
            transform: translateY(-2px);
            border-color: #b794f6;
            box-shadow: 0 4px 12px rgba(183, 148, 246, 0.2);
        }}

        .metric-label {{
            font-size: 0.9rem;
            color: #cbd5e0;
            margin-bottom: 8px;
        }}

        .metric-value {{
            font-size: 1.8rem;
            font-weight: 700;
            color: #b794f6;
        }}

        .metric-unit {{
            font-size: 1rem;
            color: #cbd5e0;
            margin-left: 5px;
        }}

        .section {{
            background: rgba(255, 255, 255, 0.03);
            padding: 25px;
            border-radius: 12px;
            margin-bottom: 25px;
            border: 1px solid rgba(183, 148, 246, 0.2);
        }}

        .section h2 {{
            color: #b794f6;
            margin-bottom: 15px;
            font-size: 1.5rem;
        }}

        .section table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }}

        .section th, .section td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid rgba(183, 148, 246, 0.2);
        }}

        .section th {{
            color: #b794f6;
            font-weight: 600;
        }}

        .diagram-section {{
            margin-bottom: 40px;
        }}

        .diagram-section h2 {{
            color: #b794f6;
            margin-bottom: 10px;
            font-size: 1.4rem;
        }}

        .diagram-section p {{
            color: #cbd5e0;
            margin-bottom: 20px;
        }}

        .mermaid-container {{
            background: rgba(255, 255, 255, 0.03);
            border-radius: 12px;
            padding: 20px;
            overflow-x: auto;
            border: 1px solid rgba(183, 148, 246, 0.2);
        }}

        .recommendations {{
            background: rgba(183, 148, 246, 0.1);
            padding: 20px;
            border-radius: 12px;
            border-left: 4px solid #b794f6;
        }}

        .recommendations h2 {{
            color: #b794f6;
            margin-bottom: 15px;
        }}

        .recommendations ul {{
            list-style-position: inside;
            padding-left: 20px;
        }}

        .recommendations li {{
            margin-bottom: 10px;
            color: #cbd5e0;
        }}

        @media (max-width: 768px) {{
            .header h1 {{
                font-size: 1.8rem;
            }}
            .metrics-grid {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{title}</h1>
            <p class="description">{description}</p>
        </div>

        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">Latency</div>
                <div class="metric-value">{result.total_latency_ms:.2f}<span class="metric-unit">ms</span></div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Throughput</div>
                <div class="metric-value">{result.throughput_fps:.1f}<span class="metric-unit">fps</span></div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Energy/Inference</div>
                <div class="metric-value">{result.energy_per_inference_mj:.1f}<span class="metric-unit">mJ</span></div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Peak Memory</div>
                <div class="metric-value">{result.peak_memory_mb:.1f}<span class="metric-unit">MB</span></div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Utilization</div>
                <div class="metric-value">{result.average_utilization_pct:.1f}<span class="metric-unit">%</span></div>
            </div>
        </div>

        <div class="section">
            <h2>Performance Breakdown</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>Total FLOPs</td>
                    <td>{result.partition_report.total_flops / 1e9:.2f} GFLOPs</td>
                </tr>
                <tr>
                    <td>Subgraphs</td>
                    <td>{len(result.partition_report.subgraphs)}</td>
                </tr>
"""

        # Add bottleneck stats if available
        if result.roofline_report:
            memory_bound = sum(1 for lat in result.roofline_report.latencies if lat.bottleneck == 'memory')
            compute_bound = len(result.roofline_report.latencies) - memory_bound
            html += f"""
                <tr>
                    <td>Compute-bound operations</td>
                    <td>{compute_bound}</td>
                </tr>
                <tr>
                    <td>Memory-bound operations</td>
                    <td>{memory_bound}</td>
                </tr>
"""

        html += """
            </table>
        </div>

        <div class="section">
            <h2>Energy Breakdown</h2>
            <table>
                <tr>
                    <th>Component</th>
                    <th>Energy (mJ)</th>
                    <th>Percentage</th>
                </tr>
"""

        if result.energy_report:
            total = result.total_energy_mj
            compute_pct = (result.energy_report.compute_energy_j * 1000 / total * 100) if total > 0 else 0
            memory_pct = (result.energy_report.memory_energy_j * 1000 / total * 100) if total > 0 else 0
            static_pct = (result.energy_report.static_energy_j * 1000 / total * 100) if total > 0 else 0

            html += f"""
                <tr>
                    <td>Compute</td>
                    <td>{result.energy_report.compute_energy_j * 1000:.1f}</td>
                    <td>{compute_pct:.1f}%</td>
                </tr>
                <tr>
                    <td>Memory</td>
                    <td>{result.energy_report.memory_energy_j * 1000:.1f}</td>
                    <td>{memory_pct:.1f}%</td>
                </tr>
                <tr>
                    <td>Static/Leakage</td>
                    <td>{result.energy_report.static_energy_j * 1000:.1f}</td>
                    <td>{static_pct:.1f}%</td>
                </tr>
                <tr style="font-weight: bold;">
                    <td>Total</td>
                    <td>{total:.1f}</td>
                    <td>100.0%</td>
                </tr>
"""

        html += """
            </table>
        </div>

        <div class="section">
            <h2>Memory Breakdown</h2>
            <table>
                <tr>
                    <th>Component</th>
                    <th>Memory (MB)</th>
                </tr>
"""

        if result.memory_report:
            html += f"""
                <tr>
                    <td>Activations</td>
                    <td>{result.memory_report.activation_memory_bytes/1e6:.1f}</td>
                </tr>
                <tr>
                    <td>Weights</td>
                    <td>{result.memory_report.weight_memory_bytes/1e6:.1f}</td>
                </tr>
                <tr>
                    <td>Workspace</td>
                    <td>{result.memory_report.workspace_memory_bytes/1e6:.1f}</td>
                </tr>
                <tr style="font-weight: bold;">
                    <td>Peak Total</td>
                    <td>{result.peak_memory_mb:.1f}</td>
                </tr>
"""

        html += """
            </table>
        </div>
"""

        # Add diagrams
        html += diagrams_html

        # Add recommendations
        recommendations = result._generate_recommendations()
        if recommendations:
            html += """
        <div class="recommendations">
            <h2>Recommendations</h2>
            <ul>
"""
            for rec in recommendations:
                html += f"                <li>{rec}</li>\n"
            html += """
            </ul>
        </div>
"""

        html += """
    </div>

    <script>
        mermaid.initialize({
            startOnLoad: true,
            theme: 'dark',
            themeVariables: {
                primaryColor: '#b794f6',
                primaryTextColor: '#fff',
                primaryBorderColor: '#7c3aed',
                lineColor: '#cbd5e0',
                secondaryColor: '#4a5568',
                tertiaryColor: '#2d3748'
            },
            flowchart: {
                useMaxWidth: true,
                htmlLabels: true,
                curve: 'basis'
            }
        });
    </script>
</body>
</html>"""

        return html

    # =========================================================================
    # Comparison Reports
    # =========================================================================

    def generate_comparison_report(
        self,
        results: List[UnifiedAnalysisResult],
        comparison_dimension: str = 'auto',
        format: str = 'text',
        sort_by: str = 'latency'
    ) -> str:
        """
        Generate comparison report across multiple configurations.

        Args:
            results: List of analysis results to compare
            comparison_dimension: What varies ('model', 'hardware', 'batch_size', or 'auto')
            format: Output format ('text', 'csv', 'markdown', 'html')
            sort_by: Metric to sort by ('latency', 'energy', 'throughput', 'efficiency')

        Returns:
            Formatted comparison report
        """
        if not results:
            return "No results to compare"

        # Auto-detect comparison dimension
        if comparison_dimension == 'auto':
            comparison_dimension = self._detect_comparison_dimension(results)

        # Sort results
        results_sorted = self._sort_results(results, sort_by)

        if format == 'csv':
            return self._comparison_csv(results_sorted, comparison_dimension)
        elif format == 'markdown':
            return self._comparison_markdown(results_sorted, comparison_dimension)
        elif format == 'html':
            return self._comparison_html(results_sorted, comparison_dimension)
        else:
            return self._comparison_text(results_sorted, comparison_dimension)

    def _detect_comparison_dimension(self, results: List[UnifiedAnalysisResult]) -> str:
        """Auto-detect what dimension is being compared"""
        models = set(r.model_name for r in results)
        hardwares = set(r.hardware_name for r in results)
        batch_sizes = set(r.batch_size for r in results)

        if len(models) > 1:
            return 'model'
        elif len(hardwares) > 1:
            return 'hardware'
        elif len(batch_sizes) > 1:
            return 'batch_size'
        else:
            return 'mixed'

    def _sort_results(self, results: List[UnifiedAnalysisResult], sort_by: str) -> List[UnifiedAnalysisResult]:
        """Sort results by specified metric"""
        if sort_by == 'latency':
            return sorted(results, key=lambda r: r.total_latency_ms)
        elif sort_by == 'energy':
            return sorted(results, key=lambda r: r.total_energy_mj)
        elif sort_by == 'throughput':
            return sorted(results, key=lambda r: r.throughput_fps, reverse=True)
        elif sort_by == 'efficiency':
            return sorted(results, key=lambda r: r.average_utilization_pct, reverse=True)
        else:
            return results

    def _comparison_text(self, results: List[UnifiedAnalysisResult], dimension: str) -> str:
        """Generate text comparison report"""
        lines = []
        lines.append("=" * 79)
        lines.append(f"                 COMPARISON REPORT ({dimension.upper()})")
        lines.append("=" * 79)
        lines.append("")

        # Table header
        lines.append(f"{'Name':<30} {'Latency':<12} {'Throughput':<12} {'Energy':<12} {'Memory':<10} {'Util%':<8}")
        lines.append("-" * 79)

        for result in results:
            if dimension == 'model':
                name = result.display_name
            elif dimension == 'hardware':
                name = result.hardware_display_name
            elif dimension == 'batch_size':
                name = f"Batch {result.batch_size}"
            else:
                name = f"{result.display_name}@{result.hardware_display_name}"

            lines.append(f"{name:<30} "
                        f"{result.total_latency_ms:>10.2f}ms "
                        f"{result.throughput_fps:>10.1f}fps "
                        f"{result.total_energy_mj:>10.1f}mJ "
                        f"{result.peak_memory_mb:>8.1f}MB "
                        f"{result.average_utilization_pct:>6.1f}%")

        lines.append("")
        return "\n".join(lines)

    def _comparison_csv(self, results: List[UnifiedAnalysisResult], dimension: str) -> str:
        """Generate CSV comparison report"""
        output = StringIO()
        writer = csv.DictWriter(output, fieldnames=[
            'name', 'model', 'hardware', 'batch_size', 'precision',
            'latency_ms', 'throughput_fps', 'energy_mj', 'energy_per_inf_mj',
            'peak_mem_mb', 'utilization_pct'
        ])
        writer.writeheader()

        for result in results:
            if dimension == 'model':
                name = result.display_name
            elif dimension == 'hardware':
                name = result.hardware_display_name
            elif dimension == 'batch_size':
                name = f"Batch {result.batch_size}"
            else:
                name = f"{result.display_name}@{result.hardware_display_name}"

            writer.writerow({
                'name': name,
                'model': result.display_name,
                'hardware': result.hardware_display_name,
                'batch_size': result.batch_size,
                'precision': result.precision.name,
                'latency_ms': round(result.total_latency_ms, 2),
                'throughput_fps': round(result.throughput_fps, 1),
                'energy_mj': round(result.total_energy_mj, 2),
                'energy_per_inf_mj': round(result.energy_per_inference_mj, 2),
                'peak_mem_mb': round(result.peak_memory_mb, 1),
                'utilization_pct': round(result.average_utilization_pct, 1),
            })

        return output.getvalue()

    def _comparison_markdown(self, results: List[UnifiedAnalysisResult], dimension: str) -> str:
        """Generate Markdown comparison report"""
        lines = []
        lines.append(f"# Comparison Report: {dimension.title()}")
        lines.append("")
        lines.append("| Name | Latency (ms) | Throughput (fps) | Energy (mJ) | Memory (MB) | Utilization (%) |")
        lines.append("|------|--------------|------------------|-------------|-------------|-----------------|")

        for result in results:
            if dimension == 'model':
                name = result.display_name
            elif dimension == 'hardware':
                name = result.hardware_display_name
            elif dimension == 'batch_size':
                name = f"Batch {result.batch_size}"
            else:
                name = f"{result.display_name}@{result.hardware_display_name}"

            lines.append(f"| {name} | "
                        f"{result.total_latency_ms:.2f} | "
                        f"{result.throughput_fps:.1f} | "
                        f"{result.total_energy_mj:.1f} | "
                        f"{result.peak_memory_mb:.1f} | "
                        f"{result.average_utilization_pct:.1f} |")

        lines.append("")
        return "\n".join(lines)

    def _comparison_html(self, results: List[UnifiedAnalysisResult], dimension: str) -> str:
        """
        Generate interactive HTML comparison report with rich Mermaid visualization.

        For hardware comparisons, uses the multi-track rich visualization.
        For other comparisons, uses a table-based comparison.
        """
        if dimension == 'hardware' and len(results) >= 2:
            return self._hardware_comparison_html_rich(results)
        else:
            return self._comparison_html_table(results, dimension)

    def _hardware_comparison_html_rich(self, results: List[UnifiedAnalysisResult]) -> str:
        """
        Generate rich HTML hardware comparison using MermaidGenerator.generate_hardware_comparison_rich().

        This creates the beautiful multi-track visualization with performance metrics.
        """
        # Prepare hardware analysis data for Mermaid generator
        model_name = results[0].display_name if results else "Model"
        hardware_analyses = []

        for result in results:
            # Build layer data from partition report
            layers = []
            if result.partition_report and result.partition_report.subgraphs:
                # Group subgraphs into logical layers (first 8 or so)
                subgraphs = result.partition_report.subgraphs[:8]

                for idx, sg in enumerate(subgraphs):
                    layer_name = f"Layer{idx+1}"
                    if hasattr(sg, 'ops') and sg.ops:
                        # Get first operation name
                        layer_name = sg.ops[0].split('.')[-1] if sg.ops else f"SG{idx}"

                    layer = {'name': layer_name}

                    # Add latency if available
                    if hasattr(sg, 'latency_ms') and sg.latency_ms > 0:
                        layer['latency_ms'] = sg.latency_ms

                    # Estimate energy from FLOPs if energy report available
                    if result.energy_report and hasattr(sg, 'total_flops'):
                        # Rough estimate: proportional to FLOPs
                        energy_fraction = sg.total_flops / result.partition_report.total_flops if result.partition_report.total_flops > 0 else 0
                        layer['energy_uj'] = result.total_energy_mj * 1000 * energy_fraction

                    # Add bandwidth (use hardware's memory bandwidth as reference)
                    if hasattr(sg, 'memory_bytes') and sg.latency_ms > 0:
                        bandwidth_gbps = (sg.memory_bytes / 1e9) / (sg.latency_ms / 1000)
                        layer['bandwidth_gbps'] = min(bandwidth_gbps, 1000)  # Cap at reasonable max

                    layers.append(layer)

            # Determine architecture type
            hw_name = result.hardware_display_name.lower()
            if 'h100' in hw_name or 'a100' in hw_name or 'v100' in hw_name or 'jetson' in hw_name:
                arch_type = 'gpu'
            elif 'tpu' in hw_name or 'coral' in hw_name:
                arch_type = 'tpu'
            elif 'kpu' in hw_name:
                arch_type = 'kpu'
            else:
                arch_type = 'cpu'

            # Build summary
            summary = {
                'total_latency_ms': result.total_latency_ms,
                'total_energy_uj': result.total_energy_mj * 1000,  # Convert mJ to μJ
                'power_w': result.energy_report.average_power_w if result.energy_report else 0,
                'throughput': result.throughput_fps,
            }

            hardware_analyses.append({
                'name': result.hardware_display_name,
                'arch_type': arch_type,
                'layers': layers,
                'summary': summary,
            })

        # Generate rich Mermaid diagram
        diagram = self.mermaid_generator.generate_hardware_comparison_rich(
            model_name=model_name,
            hardware_analyses=hardware_analyses,
            include_summary=True
        )

        # Wrap in HTML
        title = f"{model_name}: Multi-Hardware Performance Comparison"
        description = f"Comparing {len(results)} hardware architectures on {model_name} (Batch: {results[0].batch_size}, Precision: {results[0].precision.name})"

        html = self.mermaid_generator.generate_html_wrapper(
            mermaid_diagram=diagram,
            title=title,
            description=description,
            include_controls=True,
            include_legend=True
        )

        return html

    def _comparison_html_table(self, results: List[UnifiedAnalysisResult], dimension: str) -> str:
        """Generate table-based HTML comparison report"""
        title = f"Comparison Report: {dimension.title()}"
        description = f"Performance comparison across {len(results)} configurations"

        # Build comparison table rows
        table_rows = ""
        for result in results:
            if dimension == 'model':
                name = result.display_name
            elif dimension == 'hardware':
                name = result.hardware_display_name
            elif dimension == 'batch_size':
                name = f"Batch {result.batch_size}"
            else:
                name = f"{result.display_name}@{result.hardware_display_name}"

            table_rows += f"""
                <tr>
                    <td>{name}</td>
                    <td>{result.total_latency_ms:.2f}</td>
                    <td>{result.throughput_fps:.1f}</td>
                    <td>{result.total_energy_mj:.1f}</td>
                    <td>{result.peak_memory_mb:.1f}</td>
                    <td>{result.average_utilization_pct:.1f}%</td>
                </tr>
"""

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen', 'Ubuntu', sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #e0e0e0;
            min-height: 100vh;
            padding: 20px;
            line-height: 1.6;
        }}

        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}

        .header {{
            text-align: center;
            margin-bottom: 30px;
            padding: 30px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 12px;
            backdrop-filter: blur(10px);
        }}

        .header h1 {{
            font-size: 2.5rem;
            background: linear-gradient(135deg, #b794f6 0%, #7c3aed 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 10px;
        }}

        .description {{
            color: #cbd5e0;
            font-size: 1.1rem;
        }}

        .comparison-table {{
            width: 100%;
            background: rgba(255, 255, 255, 0.03);
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
        }}

        th, td {{
            padding: 15px;
            text-align: left;
            border-bottom: 1px solid rgba(183, 148, 246, 0.2);
        }}

        th {{
            background: rgba(183, 148, 246, 0.2);
            color: #b794f6;
            font-weight: 600;
        }}

        tr:hover {{
            background: rgba(183, 148, 246, 0.1);
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{title}</h1>
            <p class="description">{description}</p>
        </div>

        <div class="comparison-table">
            <table>
                <thead>
                    <tr>
                        <th>Name</th>
                        <th>Latency (ms)</th>
                        <th>Throughput (fps)</th>
                        <th>Energy (mJ)</th>
                        <th>Memory (MB)</th>
                        <th>Utilization</th>
                    </tr>
                </thead>
                <tbody>
{table_rows}
                </tbody>
            </table>
        </div>
    </div>
</body>
</html>"""

        return html

    # =========================================================================
    # File Output
    # =========================================================================

    def save_report(
        self,
        result: UnifiedAnalysisResult,
        output_path: str,
        format: Optional[str] = None
    ) -> None:
        """
        Save report to file.

        Args:
            result: Analysis result
            output_path: Output file path
            format: Format override (auto-detect from extension if None)
        """
        # Auto-detect format from extension
        if format is None:
            ext = Path(output_path).suffix.lower()
            format_map = {
                '.json': 'json',
                '.csv': 'csv',
                '.md': 'markdown',
                '.txt': 'text',
                '.html': 'html',
            }
            format = format_map.get(ext, 'text')

        # Generate report
        if format == 'json':
            content = self.generate_json_report(result)
        elif format == 'csv':
            content = self.generate_csv_report(result)
        elif format == 'markdown':
            content = self.generate_markdown_report(result)
        elif format == 'html':
            content = self.generate_html_report(result)
        else:
            content = self.generate_text_report(result)

        # Write to file
        with open(output_path, 'w') as f:
            f.write(content)

    def save_comparison_report(
        self,
        results: List[UnifiedAnalysisResult],
        output_path: str,
        **kwargs
    ) -> None:
        """Save comparison report to file"""
        # Auto-detect format from extension
        ext = Path(output_path).suffix.lower()
        format_map = {
            '.csv': 'csv',
            '.md': 'markdown',
            '.txt': 'text',
            '.html': 'html',
        }
        format = format_map.get(ext, 'text')

        # Generate report
        content = self.generate_comparison_report(results, format=format, **kwargs)

        # Write to file
        with open(output_path, 'w') as f:
            f.write(content)
