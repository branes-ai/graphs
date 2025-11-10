#!/usr/bin/env python
"""
Comprehensive Graph Analysis Tool

All-in-one analysis tool for deep-dive neural network characterization.
Provides complete performance analysis using all Phase 3 analyzers:
- Roofline Model: Bottleneck analysis and latency estimation
- Energy Estimator: Power and energy consumption analysis
- Memory Estimator: Memory footprint and timeline analysis
- Concurrency Analysis: Multi-level parallelism analysis

Supports:
- Single model, single hardware (comprehensive analysis)
- Single model, multiple hardware (hardware comparison)
- Single model, multiple precisions (precision comparison)
- Multiple output formats (text, JSON, markdown, CSV)

Usage:
    # Comprehensive single-model analysis
    ./cli/analyze_comprehensive.py --model resnet18 --hardware H100

    # Multi-hardware comparison
    ./cli/analyze_comprehensive.py --model resnet18 \
        --hardware H100 Jetson-Orin-AGX A100 \
        --compare

    # Multi-precision comparison
    ./cli/analyze_comprehensive.py --model resnet18 --hardware H100 \
        --precision fp32 fp16 int8 \
        --compare-precision

    # Generate markdown report
    ./cli/analyze_comprehensive.py --model resnet18 --hardware H100 \
        --output-format markdown \
        --output-file resnet18_analysis.md
"""

import argparse
import json
import sys
from typing import Tuple, List, Dict, Optional, Any
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
from torch.fx import symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp
from torchvision import models

# Graph analysis
from graphs.transform.partitioning import GraphPartitioner
from graphs.analysis.roofline import RooflineAnalyzer, RooflineReport
from graphs.analysis.energy import EnergyAnalyzer, EnergyReport
from graphs.analysis.memory import MemoryEstimator, MemoryReport
from graphs.analysis.concurrency import ConcurrencyAnalyzer
from graphs.hardware.resource_model import Precision, HardwareResourceModel

# Hardware mappers
from graphs.hardware.mappers.gpu import (
    create_h100_pcie_80gb_mapper,
    create_a100_sxm4_80gb_mapper,
    create_v100_sxm2_32gb_mapper,
    create_jetson_orin_agx_64gb_mapper,
    create_jetson_orin_nano_8gb_mapper,
    create_jetson_thor_128gb_mapper,
)
from graphs.hardware.mappers.accelerators.tpu import (
    create_tpu_v4_mapper,
    create_coral_edge_tpu_mapper,
)
from graphs.hardware.mappers.accelerators.kpu import (
    create_kpu_t64_mapper,
    create_kpu_t256_mapper,
    create_kpu_t768_mapper,
)
from graphs.hardware.mappers.cpu import (
    create_amd_epyc_9754_mapper,
    create_intel_xeon_platinum_8490h_mapper,
    create_ampere_ampereone_192_mapper,
    create_i7_12700k_mapper,
    create_amd_cpu_mapper,  # Ryzen
)
from graphs.hardware.mappers.dsp import (
    create_qrb5165_mapper,
    create_ti_tda4vm_mapper,
)
from graphs.hardware.mappers.accelerators.dpu import create_dpu_vitis_ai_mapper
from graphs.hardware.mappers.accelerators.cgra import create_plasticine_v2_mapper


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class ComprehensiveAnalysisReport:
    """Complete analysis report for a single model+hardware+precision configuration"""

    # Configuration
    model_name: str
    hardware_name: str
    precision: str
    batch_size: int

    # Model information
    total_flops: float
    total_bytes: float
    total_params: int
    num_subgraphs: int

    # Performance metrics
    latency_ms: float
    throughput_fps: float
    utilization_pct: float

    # Roofline analysis
    arithmetic_intensity_median: float
    ai_breakpoint: float
    memory_bound_pct: float
    compute_bound_pct: float

    # Energy analysis
    total_energy_mj: float
    compute_energy_mj: float
    memory_energy_mj: float
    static_energy_mj: float
    average_power_w: float
    energy_efficiency_pct: float

    # Memory analysis
    peak_memory_mb: float
    weight_memory_mb: float
    activation_memory_mb: float
    workspace_memory_mb: float

    # Optimization opportunities
    optimizations: List[str]

    # Raw reports (for detailed output)
    roofline_report: Optional[RooflineReport] = None
    energy_report: Optional[EnergyReport] = None
    memory_report: Optional[MemoryReport] = None


# =============================================================================
# Hardware Mapper Creation
# =============================================================================

def create_hardware_mapper(hardware_name: str, precision: Precision = Precision.FP32):
    """
    Create hardware mapper by name

    Note: Precision is stored for later use with analyzers, but not passed to mapper
    creation functions as precision profiles are already defined in the resource models.
    """

    hardware_map = {
        # GPUs - Datacenter
        'h100': create_h100_pcie_80gb_mapper,
        'h100-pcie': create_h100_pcie_80gb_mapper,
        'a100': create_a100_sxm4_80gb_mapper,
        'v100': create_v100_sxm2_32gb_mapper,

        # GPUs - Edge
        'jetson-orin-agx': create_jetson_orin_agx_64gb_mapper,
        'jetson-orin': create_jetson_orin_agx_64gb_mapper,
        'jetson-orin-nano': create_jetson_orin_nano_8gb_mapper,
        'jetson-nano': create_jetson_orin_nano_8gb_mapper,
        'jetson-thor': create_jetson_thor_128gb_mapper,

        # TPUs
        'tpu-v4': create_tpu_v4_mapper,
        'tpu': create_tpu_v4_mapper,
        'coral': create_coral_edge_tpu_mapper,
        'coral-tpu': create_coral_edge_tpu_mapper,

        # KPUs
        'kpu-t64': create_kpu_t64_mapper,
        'kpu-t256': create_kpu_t256_mapper,
        'kpu-t768': create_kpu_t768_mapper,

        # CPUs - Datacenter
        'epyc': create_amd_epyc_9754_mapper,
        'amd-epyc': create_amd_epyc_9754_mapper,
        'xeon': create_intel_xeon_platinum_8490h_mapper,
        'intel-xeon': create_intel_xeon_platinum_8490h_mapper,
        'ampere-one': create_ampere_ampereone_192_mapper,

        # CPUs - Consumer
        'i7-12700k': create_i7_12700k_mapper,
        'ryzen-7-5800x': create_amd_cpu_mapper,
        'ryzen': create_amd_cpu_mapper,

        # DSPs
        'qrb5165': create_qrb5165_mapper,
        'qualcomm-qrb5165': create_qrb5165_mapper,
        'ti-tda4vm': create_ti_tda4vm_mapper,
        'tda4vm': create_ti_tda4vm_mapper,

        # Accelerators
        'dpu': create_dpu_vitis_ai_mapper,
        'xilinx-dpu': create_dpu_vitis_ai_mapper,
        'cgra': create_plasticine_v2_mapper,
        'plasticine': create_plasticine_v2_mapper,
    }

    hardware_key = hardware_name.lower()
    if hardware_key not in hardware_map:
        raise ValueError(f"Unknown hardware: {hardware_name}. Available: {list(hardware_map.keys())}")

    # Create mapper (precision not passed to mapper, it's in the resource model)
    return hardware_map[hardware_key]()


# =============================================================================
# Model Creation
# =============================================================================

def create_model(model_name: str, batch_size: int) -> Tuple[nn.Module, torch.Tensor, str]:
    """
    Create PyTorch model and input tensor

    Args:
        model_name: Name of model
        batch_size: Batch size for input

    Returns:
        (model, input_tensor, display_name)
    """
    model_name_lower = model_name.lower()

    # ResNet family
    if model_name_lower == 'resnet18':
        model = models.resnet18(weights=None)
        input_tensor = torch.randn(batch_size, 3, 224, 224)
        display_name = "ResNet-18"

    elif model_name_lower == 'resnet34':
        model = models.resnet34(weights=None)
        input_tensor = torch.randn(batch_size, 3, 224, 224)
        display_name = "ResNet-34"

    elif model_name_lower == 'resnet50':
        model = models.resnet50(weights=None)
        input_tensor = torch.randn(batch_size, 3, 224, 224)
        display_name = "ResNet-50"

    elif model_name_lower == 'resnet101':
        model = models.resnet101(weights=None)
        input_tensor = torch.randn(batch_size, 3, 224, 224)
        display_name = "ResNet-101"

    elif model_name_lower == 'resnet152':
        model = models.resnet152(weights=None)
        input_tensor = torch.randn(batch_size, 3, 224, 224)
        display_name = "ResNet-152"

    # MobileNet family
    elif model_name_lower in ['mobilenet', 'mobilenet_v2', 'mobilenetv2']:
        model = models.mobilenet_v2(weights=None)
        input_tensor = torch.randn(batch_size, 3, 224, 224)
        display_name = "MobileNet-V2"

    elif model_name_lower in ['mobilenet_v3_small', 'mobilenetv3_small']:
        model = models.mobilenet_v3_small(weights=None)
        input_tensor = torch.randn(batch_size, 3, 224, 224)
        display_name = "MobileNet-V3-Small"

    elif model_name_lower in ['mobilenet_v3_large', 'mobilenetv3_large']:
        model = models.mobilenet_v3_large(weights=None)
        input_tensor = torch.randn(batch_size, 3, 224, 224)
        display_name = "MobileNet-V3-Large"

    # EfficientNet family
    elif model_name_lower in ['efficientnet_b0', 'efficientnetb0']:
        model = models.efficientnet_b0(weights=None)
        input_tensor = torch.randn(batch_size, 3, 224, 224)
        display_name = "EfficientNet-B0"

    elif model_name_lower in ['efficientnet_b1', 'efficientnetb1']:
        model = models.efficientnet_b1(weights=None)
        input_tensor = torch.randn(batch_size, 3, 240, 240)
        display_name = "EfficientNet-B1"

    elif model_name_lower in ['efficientnet_b4', 'efficientnetb4']:
        model = models.efficientnet_b4(weights=None)
        input_tensor = torch.randn(batch_size, 3, 380, 380)
        display_name = "EfficientNet-B4"

    # Vision Transformer
    elif model_name_lower in ['vit_b_16', 'vit']:
        model = models.vit_b_16(weights=None)
        input_tensor = torch.randn(batch_size, 3, 224, 224)
        display_name = "ViT-B/16"

    elif model_name_lower in ['vit_l_16']:
        model = models.vit_l_16(weights=None)
        input_tensor = torch.randn(batch_size, 3, 224, 224)
        display_name = "ViT-L/16"

    else:
        raise ValueError(f"Unknown model: {model_name}")

    model.eval()
    return model, input_tensor, display_name


# =============================================================================
# Analysis Workflow
# =============================================================================

def analyze_model(
    model_name: str,
    hardware_name: str,
    precision: Precision,
    batch_size: int,
    verbose: bool = True
) -> ComprehensiveAnalysisReport:
    """
    Run comprehensive analysis on model

    Args:
        model_name: Name of model to analyze
        hardware_name: Name of hardware target
        precision: Precision (FP32, FP16, INT8)
        batch_size: Batch size for input
        verbose: Print progress messages

    Returns:
        ComprehensiveAnalysisReport with all analysis results
    """

    if verbose:
        print(f"\n{'='*80}")
        print(f"COMPREHENSIVE ANALYSIS")
        print(f"{'='*80}")
        print(f"Model: {model_name}")
        print(f"Hardware: {hardware_name}")
        print(f"Precision: {precision.value}")
        print(f"Batch Size: {batch_size}")
        print(f"{'='*80}\n")

    # Step 1: Create model
    if verbose:
        print(f"[1/6] Creating model...")
    model, input_tensor, display_name = create_model(model_name, batch_size)

    # Step 2: Create hardware mapper
    if verbose:
        print(f"[2/6] Creating hardware mapper...")
    mapper = create_hardware_mapper(hardware_name, precision)
    hardware = mapper.resource_model

    # Step 3: FX trace and shape propagation
    if verbose:
        print(f"[3/6] Tracing model with PyTorch FX...")
    try:
        fx_graph = symbolic_trace(model)
    except Exception as e:
        print(f"Error during FX trace: {e}")
        raise

    shape_prop = ShapeProp(fx_graph)
    shape_prop.propagate(input_tensor)

    # Step 4: Partition graph
    if verbose:
        print(f"[4/6] Partitioning graph...")
    partitioner = GraphPartitioner()
    partition_report = partitioner.partition(fx_graph)

    if verbose:
        print(f"      Partitioned into {partition_report.total_subgraphs} subgraphs")
        print(f"      Total FLOPs: {partition_report.total_flops / 1e9:.2f} GFLOP")
        print(f"      Total Memory Traffic: {partition_report.total_memory_traffic / 1e6:.2f} MB")

    # Step 5: Run all Phase 3 analyzers
    if verbose:
        print(f"[5/6] Running Phase 3 analyzers...")
        print(f"      - Roofline analysis...")

    # 5.1 Roofline Analyzer
    roofline_analyzer = RooflineAnalyzer(hardware, precision=precision)
    roofline_report = roofline_analyzer.analyze(
        partition_report.subgraphs,
        partition_report
    )

    if verbose:
        print(f"      - Energy analysis...")

    # 5.2 Energy Analyzer (use roofline latencies)
    energy_analyzer = EnergyAnalyzer(hardware, precision=precision)
    latencies = [lat.actual_latency for lat in roofline_report.latencies]
    energy_report = energy_analyzer.analyze(
        partition_report.subgraphs,
        partition_report,
        latencies=latencies
    )

    if verbose:
        print(f"      - Memory analysis...")

    # 5.3 Memory Estimator
    memory_estimator = MemoryEstimator(hardware)
    memory_report = memory_estimator.estimate_memory(
        partition_report.subgraphs,
        partition_report
    )

    # Step 6: Compile comprehensive report
    if verbose:
        print(f"[6/6] Compiling comprehensive report...")

    # Calculate metrics
    total_latency_s = roofline_report.total_latency
    throughput_fps = batch_size / total_latency_s if total_latency_s > 0 else 0

    # Bottleneck distribution
    memory_bound = sum(1 for lat in roofline_report.latencies if lat.bottleneck == 'memory-bound')
    compute_bound = sum(1 for lat in roofline_report.latencies if lat.bottleneck == 'compute-bound')
    total_ops = len(roofline_report.latencies)

    memory_bound_pct = (memory_bound / total_ops * 100) if total_ops > 0 else 0
    compute_bound_pct = (compute_bound / total_ops * 100) if total_ops > 0 else 0

    # Arithmetic intensity (median)
    ai_values = [lat.arithmetic_intensity for lat in roofline_report.latencies]
    ai_median = sorted(ai_values)[len(ai_values) // 2] if ai_values else 0

    # Optimization opportunities
    optimizations = []

    # Energy optimizations
    if energy_report.static_energy_j / energy_report.total_energy_j > 0.4:
        static_pct = energy_report.static_energy_j / energy_report.total_energy_j * 100
        optimizations.append(
            f"Increase batch size (HIGH priority): {static_pct:.0f}% static energy, "
            f"larger batches amortize leakage cost"
        )

    if energy_report.average_utilization < 0.6:
        util_pct = energy_report.average_utilization * 100
        optimizations.append(
            f"Improve utilization (MEDIUM priority): {util_pct:.0f}% average utilization, "
            f"better batching/fusion could save {energy_report.wasted_energy_j*1e3:.0f} mJ"
        )

    if precision == Precision.FP32:
        compute_savings_fp16 = energy_report.compute_energy_j * 0.5
        optimizations.append(
            f"Use FP16 precision (HIGH priority): 50% compute energy reduction "
            f"(~{compute_savings_fp16*1e3:.0f} mJ savings)"
        )

    # Memory optimizations
    if memory_report.peak_memory_bytes / hardware.main_memory > 0.8:
        optimizations.append(
            f"Memory optimization needed (HIGH priority): Peak memory {memory_report.peak_memory_bytes/1e6:.0f} MB "
            f"is {memory_report.peak_memory_bytes/hardware.main_memory*100:.0f}% of device memory"
        )

    # Roofline optimizations
    if memory_bound_pct > 60:
        optimizations.append(
            f"Reduce memory bandwidth pressure (MEDIUM priority): {memory_bound_pct:.0f}% of operations are memory-bound"
        )

    # Create comprehensive report
    report = ComprehensiveAnalysisReport(
        # Configuration
        model_name=display_name,
        hardware_name=hardware.name,
        precision=precision.value,
        batch_size=batch_size,

        # Model information
        total_flops=partition_report.total_flops,
        total_bytes=partition_report.total_memory_traffic,
        total_params=sum(sg.total_weight_bytes for sg in partition_report.subgraphs) // 4,  # Assume FP32
        num_subgraphs=partition_report.total_subgraphs,

        # Performance metrics
        latency_ms=total_latency_s * 1e3,
        throughput_fps=throughput_fps,
        utilization_pct=roofline_report.average_flops_utilization * 100,

        # Roofline analysis
        arithmetic_intensity_median=ai_median,
        ai_breakpoint=roofline_analyzer.ai_breakpoint,
        memory_bound_pct=memory_bound_pct,
        compute_bound_pct=compute_bound_pct,

        # Energy analysis
        total_energy_mj=energy_report.total_energy_mj,
        compute_energy_mj=energy_report.compute_energy_j * 1e3,  # Convert J to mJ
        memory_energy_mj=energy_report.memory_energy_j * 1e3,    # Convert J to mJ
        static_energy_mj=energy_report.static_energy_j * 1e3,    # Convert J to mJ
        average_power_w=energy_report.average_power_w,
        energy_efficiency_pct=energy_report.average_efficiency * 100,

        # Memory analysis
        peak_memory_mb=memory_report.peak_memory_bytes / 1e6,
        weight_memory_mb=memory_report.weight_memory_bytes / 1e6,
        activation_memory_mb=memory_report.activation_memory_bytes / 1e6,
        workspace_memory_mb=memory_report.workspace_memory_bytes / 1e6,

        # Optimization opportunities
        optimizations=optimizations,

        # Raw reports
        roofline_report=roofline_report,
        energy_report=energy_report,
        memory_report=memory_report,
    )

    if verbose:
        print(f"\nAnalysis complete!")

    return report


# =============================================================================
# Report Formatting
# =============================================================================

def format_text_report(report: ComprehensiveAnalysisReport, detailed: bool = True) -> str:
    """Format comprehensive report as text"""

    lines = []

    # Header
    lines.append("=" * 80)
    lines.append(f"COMPREHENSIVE ANALYSIS: {report.model_name} on {report.hardware_name}")
    lines.append("=" * 80)
    lines.append("")

    # Executive Summary
    lines.append("EXECUTIVE SUMMARY")
    lines.append("-" * 80)

    # Performance grade (simple heuristic)
    if report.utilization_pct > 80:
        grade = "A"
    elif report.utilization_pct > 60:
        grade = "B"
    elif report.utilization_pct > 40:
        grade = "C"
    elif report.utilization_pct > 20:
        grade = "D"
    else:
        grade = "F"

    lines.append(f"Performance Grade: {grade}")

    if report.memory_bound_pct > report.compute_bound_pct:
        lines.append(f"Key Bottleneck: Memory bandwidth ({report.memory_bound_pct:.0f}% of operations)")
    else:
        lines.append(f"Key Bottleneck: Compute ({report.compute_bound_pct:.0f}% of operations)")

    if report.optimizations:
        lines.append(f"Top Optimization: {report.optimizations[0]}")

    lines.append("")

    # Model Information
    lines.append("MODEL INFORMATION")
    lines.append("-" * 80)
    lines.append(f"Architecture: {report.model_name}")
    lines.append(f"Total Operations: {report.total_flops / 1e9:.2f} GFLOP")
    lines.append(f"Total Data: {report.total_bytes / 1e6:.2f} MB")
    lines.append(f"Parameters: {report.total_params / 1e6:.1f}M")
    lines.append(f"Subgraphs (after fusion): {report.num_subgraphs}")
    lines.append(f"Precision: {report.precision}")
    lines.append(f"Batch Size: {report.batch_size}")
    lines.append("")

    # Performance Analysis
    lines.append("PERFORMANCE ANALYSIS")
    lines.append("-" * 80)
    lines.append(f"Latency: {report.latency_ms:.2f} ms")
    lines.append(f"Throughput: {report.throughput_fps:.0f} images/sec")
    lines.append(f"Hardware Utilization: {report.utilization_pct:.1f}%")

    if report.memory_bound_pct > report.compute_bound_pct:
        lines.append(f"Primary Bottleneck: Memory-bound ({report.memory_bound_pct:.0f}% of ops)")
    else:
        lines.append(f"Primary Bottleneck: Compute-bound ({report.compute_bound_pct:.0f}% of ops)")

    lines.append("")

    # Roofline Analysis
    lines.append("ROOFLINE ANALYSIS")
    lines.append("-" * 80)
    lines.append(f"Arithmetic Intensity (median): {report.arithmetic_intensity_median:.2f} FLOP/byte")
    lines.append(f"AI Breakpoint: {report.ai_breakpoint:.2f} FLOP/byte")
    lines.append(f"Memory-bound operations: {report.memory_bound_pct:.0f}% ({int(report.memory_bound_pct * report.num_subgraphs / 100)}/{report.num_subgraphs} subgraphs)")
    lines.append(f"Compute-bound operations: {report.compute_bound_pct:.0f}% ({int(report.compute_bound_pct * report.num_subgraphs / 100)}/{report.num_subgraphs} subgraphs)")
    lines.append("")

    # Energy Analysis
    lines.append("ENERGY ANALYSIS")
    lines.append("-" * 80)
    lines.append(f"Total Energy: {report.total_energy_mj:.2f} mJ ({report.total_energy_mj * 1e3:.0f} μJ)")
    lines.append(f"  Compute:  {report.compute_energy_mj:.2f} mJ ({report.compute_energy_mj / report.total_energy_mj * 100:.1f}%)")
    lines.append(f"  Memory:   {report.memory_energy_mj:.2f} mJ ({report.memory_energy_mj / report.total_energy_mj * 100:.1f}%)")
    lines.append(f"  Static:   {report.static_energy_mj:.2f} mJ ({report.static_energy_mj / report.total_energy_mj * 100:.1f}%)")
    lines.append("")
    lines.append(f"Average Power: {report.average_power_w:.1f} W")
    lines.append(f"Energy Efficiency: {report.energy_efficiency_pct:.1f}%")
    lines.append("")

    # Energy breakdown visualization
    if detailed:
        max_bar_len = 50
        compute_pct = report.compute_energy_mj / report.total_energy_mj * 100
        memory_pct = report.memory_energy_mj / report.total_energy_mj * 100
        static_pct = report.static_energy_mj / report.total_energy_mj * 100

        compute_bar = "█" * int(compute_pct / 100 * max_bar_len)
        memory_bar = "█" * int(memory_pct / 100 * max_bar_len)
        static_bar = "█" * int(static_pct / 100 * max_bar_len)

        lines.append("Energy Breakdown:")
        lines.append(f"  Compute ({compute_pct:.1f}%): {compute_bar}")
        lines.append(f"  Memory  ({memory_pct:.1f}%): {memory_bar}")
        lines.append(f"  Static  ({static_pct:.1f}%): {static_bar}")
        lines.append("")

    # Top energy consumers
    if detailed and report.energy_report:
        top_n = 5
        top_consumers = sorted(
            report.energy_report.energy_descriptors,
            key=lambda d: d.total_energy_j,
            reverse=True
        )[:top_n]

        lines.append(f"Top {top_n} Energy Consumers:")
        for i, desc in enumerate(top_consumers, 1):
            energy_mj = desc.total_energy_j * 1e3
            pct = desc.total_energy_j / report.energy_report.total_energy_j * 100
            lines.append(f"  {i}. {desc.subgraph_name}: {energy_mj:.2f} mJ ({pct:.1f}%)")
        lines.append("")

    # Memory Analysis
    lines.append("MEMORY ANALYSIS")
    lines.append("-" * 80)
    lines.append(f"Peak Memory: {report.peak_memory_mb:.1f} MB")
    lines.append(f"  Weights:     {report.weight_memory_mb:.1f} MB ({report.weight_memory_mb / report.peak_memory_mb * 100:.1f}%, persistent)")
    lines.append(f"  Activations: {report.activation_memory_mb:.1f} MB ({report.activation_memory_mb / report.peak_memory_mb * 100:.1f}%, peak)")
    lines.append(f"  Workspace:   {report.workspace_memory_mb:.1f} MB ({report.workspace_memory_mb / report.peak_memory_mb * 100:.1f}%, temporary)")
    lines.append("")

    # Optimization Opportunities
    lines.append("OPTIMIZATION OPPORTUNITIES")
    lines.append("-" * 80)
    if report.optimizations:
        for i, opt in enumerate(report.optimizations, 1):
            lines.append(f"{i}. {opt}")
    else:
        lines.append("No major optimization opportunities identified.")
    lines.append("")

    lines.append("=" * 80)

    return "\n".join(lines)


def format_json_report(report: ComprehensiveAnalysisReport) -> str:
    """Format comprehensive report as JSON"""

    # Convert report to dict, excluding raw reports
    report_dict = {
        'configuration': {
            'model': report.model_name,
            'hardware': report.hardware_name,
            'precision': report.precision,
            'batch_size': report.batch_size,
        },
        'model_info': {
            'total_flops': report.total_flops,
            'total_bytes': report.total_bytes,
            'total_params': report.total_params,
            'num_subgraphs': report.num_subgraphs,
        },
        'performance': {
            'latency_ms': report.latency_ms,
            'throughput_fps': report.throughput_fps,
            'utilization_pct': report.utilization_pct,
        },
        'roofline': {
            'arithmetic_intensity_median': report.arithmetic_intensity_median,
            'ai_breakpoint': report.ai_breakpoint,
            'memory_bound_pct': report.memory_bound_pct,
            'compute_bound_pct': report.compute_bound_pct,
        },
        'energy': {
            'total_energy_mj': report.total_energy_mj,
            'compute_energy_mj': report.compute_energy_mj,
            'memory_energy_mj': report.memory_energy_mj,
            'static_energy_mj': report.static_energy_mj,
            'average_power_w': report.average_power_w,
            'energy_efficiency_pct': report.energy_efficiency_pct,
        },
        'memory': {
            'peak_memory_mb': report.peak_memory_mb,
            'weight_memory_mb': report.weight_memory_mb,
            'activation_memory_mb': report.activation_memory_mb,
            'workspace_memory_mb': report.workspace_memory_mb,
        },
        'optimizations': report.optimizations,
    }

    return json.dumps(report_dict, indent=2)


def format_markdown_report(report: ComprehensiveAnalysisReport) -> str:
    """Format comprehensive report as Markdown"""

    lines = []

    # Title
    lines.append(f"# Comprehensive Analysis: {report.model_name} on {report.hardware_name}")
    lines.append("")

    # Executive Summary
    lines.append("## Executive Summary")
    lines.append("")

    # Performance grade
    if report.utilization_pct > 80:
        grade = "A"
    elif report.utilization_pct > 60:
        grade = "B"
    elif report.utilization_pct > 40:
        grade = "C"
    elif report.utilization_pct > 20:
        grade = "D"
    else:
        grade = "F"

    lines.append(f"**Performance Grade:** {grade}")
    lines.append("")

    if report.memory_bound_pct > report.compute_bound_pct:
        lines.append(f"**Key Bottleneck:** Memory bandwidth ({report.memory_bound_pct:.0f}% of operations)")
    else:
        lines.append(f"**Key Bottleneck:** Compute ({report.compute_bound_pct:.0f}% of operations)")
    lines.append("")

    if report.optimizations:
        lines.append(f"**Top Optimization:** {report.optimizations[0]}")
    lines.append("")

    # Model Information
    lines.append("## Model Information")
    lines.append("")
    lines.append(f"- **Architecture:** {report.model_name}")
    lines.append(f"- **Total Operations:** {report.total_flops / 1e9:.2f} GFLOP")
    lines.append(f"- **Total Data:** {report.total_bytes / 1e6:.2f} MB")
    lines.append(f"- **Parameters:** {report.total_params / 1e6:.1f}M")
    lines.append(f"- **Subgraphs:** {report.num_subgraphs}")
    lines.append(f"- **Precision:** {report.precision}")
    lines.append(f"- **Batch Size:** {report.batch_size}")
    lines.append("")

    # Performance Analysis
    lines.append("## Performance Analysis")
    lines.append("")
    lines.append(f"- **Latency:** {report.latency_ms:.2f} ms")
    lines.append(f"- **Throughput:** {report.throughput_fps:.0f} images/sec")
    lines.append(f"- **Utilization:** {report.utilization_pct:.1f}%")
    lines.append("")

    # Roofline Analysis
    lines.append("## Roofline Analysis")
    lines.append("")
    lines.append(f"- **Arithmetic Intensity (median):** {report.arithmetic_intensity_median:.2f} FLOP/byte")
    lines.append(f"- **AI Breakpoint:** {report.ai_breakpoint:.2f} FLOP/byte")
    lines.append(f"- **Memory-bound ops:** {report.memory_bound_pct:.0f}%")
    lines.append(f"- **Compute-bound ops:** {report.compute_bound_pct:.0f}%")
    lines.append("")

    # Energy Analysis
    lines.append("## Energy Analysis")
    lines.append("")
    lines.append(f"- **Total Energy:** {report.total_energy_mj:.2f} mJ")
    lines.append(f"  - Compute: {report.compute_energy_mj:.2f} mJ ({report.compute_energy_mj / report.total_energy_mj * 100:.1f}%)")
    lines.append(f"  - Memory: {report.memory_energy_mj:.2f} mJ ({report.memory_energy_mj / report.total_energy_mj * 100:.1f}%)")
    lines.append(f"  - Static: {report.static_energy_mj:.2f} mJ ({report.static_energy_mj / report.total_energy_mj * 100:.1f}%)")
    lines.append(f"- **Average Power:** {report.average_power_w:.1f} W")
    lines.append(f"- **Energy Efficiency:** {report.energy_efficiency_pct:.1f}%")
    lines.append("")

    # Memory Analysis
    lines.append("## Memory Analysis")
    lines.append("")
    lines.append(f"- **Peak Memory:** {report.peak_memory_mb:.1f} MB")
    lines.append(f"  - Weights: {report.weight_memory_mb:.1f} MB ({report.weight_memory_mb / report.peak_memory_mb * 100:.1f}%)")
    lines.append(f"  - Activations: {report.activation_memory_mb:.1f} MB ({report.activation_memory_mb / report.peak_memory_mb * 100:.1f}%)")
    lines.append(f"  - Workspace: {report.workspace_memory_mb:.1f} MB ({report.workspace_memory_mb / report.peak_memory_mb * 100:.1f}%)")
    lines.append("")

    # Optimization Opportunities
    lines.append("## Optimization Opportunities")
    lines.append("")
    if report.optimizations:
        for i, opt in enumerate(report.optimizations, 1):
            lines.append(f"{i}. {opt}")
    else:
        lines.append("No major optimization opportunities identified.")
    lines.append("")

    return "\n".join(lines)


def format_csv_summary(report: ComprehensiveAnalysisReport) -> str:
    """Format comprehensive report as CSV (single row)"""

    # CSV header
    header = [
        "Model", "Hardware", "Precision", "BatchSize",
        "FLOPs", "Bytes", "Params", "Subgraphs",
        "Latency_ms", "Throughput_fps", "Utilization_%",
        "AI_median", "AI_breakpoint", "MemoryBound_%", "ComputeBound_%",
        "Energy_mJ", "ComputeEnergy_mJ", "MemoryEnergy_mJ", "StaticEnergy_mJ",
        "Power_W", "Efficiency_%",
        "PeakMemory_MB", "WeightMemory_MB", "ActivationMemory_MB", "WorkspaceMemory_MB"
    ]

    # CSV data
    data = [
        report.model_name,
        report.hardware_name,
        report.precision,
        str(report.batch_size),
        f"{report.total_flops:.2e}",
        f"{report.total_bytes:.2e}",
        str(report.total_params),
        str(report.num_subgraphs),
        f"{report.latency_ms:.2f}",
        f"{report.throughput_fps:.0f}",
        f"{report.utilization_pct:.1f}",
        f"{report.arithmetic_intensity_median:.2f}",
        f"{report.ai_breakpoint:.2f}",
        f"{report.memory_bound_pct:.1f}",
        f"{report.compute_bound_pct:.1f}",
        f"{report.total_energy_mj:.2f}",
        f"{report.compute_energy_mj:.2f}",
        f"{report.memory_energy_mj:.2f}",
        f"{report.static_energy_mj:.2f}",
        f"{report.average_power_w:.1f}",
        f"{report.energy_efficiency_pct:.1f}",
        f"{report.peak_memory_mb:.1f}",
        f"{report.weight_memory_mb:.1f}",
        f"{report.activation_memory_mb:.1f}",
        f"{report.workspace_memory_mb:.1f}",
    ]

    return ",".join(header) + "\n" + ",".join(data)


# =============================================================================
# Main Function
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive Graph Analysis Tool - Deep-dive neural network characterization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single model analysis
  ./cli/analyze_comprehensive.py --model resnet18 --hardware H100

  # With different precision
  ./cli/analyze_comprehensive.py --model resnet18 --hardware H100 --precision fp16

  # Generate markdown report
  ./cli/analyze_comprehensive.py --model resnet18 --hardware H100 \\
      --output-format markdown --output-file resnet18_h100.md

  # Generate JSON for automation
  ./cli/analyze_comprehensive.py --model resnet18 --hardware H100 \\
      --output-format json --output-file resnet18_h100.json
        """
    )

    # Required arguments
    parser.add_argument('--model', type=str, required=True,
                        help='Model name (resnet18, resnet50, mobilenet_v2, efficientnet_b0, vit_b_16, etc.)')
    parser.add_argument('--hardware', type=str, required=True,
                        help='Hardware target (H100, A100, Jetson-Orin, TPU-v4, etc.)')

    # Optional arguments
    parser.add_argument('--precision', type=str, default='fp32', choices=['fp32', 'fp16', 'int8'],
                        help='Precision (default: fp32)')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size (default: 1)')

    # Output options
    parser.add_argument('--output-format', type=str, default='text',
                        choices=['text', 'json', 'markdown', 'csv'],
                        help='Output format (default: text)')
    parser.add_argument('--output-file', type=str, default=None,
                        help='Output file path (default: stdout)')
    parser.add_argument('--detailed', action='store_true',
                        help='Include detailed breakdowns (text format only)')

    # Verbosity
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress progress messages')

    args = parser.parse_args()

    # Convert precision string to enum
    precision_map = {
        'fp32': Precision.FP32,
        'fp16': Precision.FP16,
        'int8': Precision.INT8,
    }
    precision = precision_map[args.precision.lower()]

    # Run analysis
    try:
        report = analyze_model(
            model_name=args.model,
            hardware_name=args.hardware,
            precision=precision,
            batch_size=args.batch_size,
            verbose=not args.quiet
        )
    except Exception as e:
        print(f"Error during analysis: {e}", file=sys.stderr)
        sys.exit(1)

    # Format output
    if args.output_format == 'text':
        output = format_text_report(report, detailed=args.detailed)
    elif args.output_format == 'json':
        output = format_json_report(report)
    elif args.output_format == 'markdown':
        output = format_markdown_report(report)
    elif args.output_format == 'csv':
        output = format_csv_summary(report)
    else:
        print(f"Unknown output format: {args.output_format}", file=sys.stderr)
        sys.exit(1)

    # Write output
    if args.output_file:
        with open(args.output_file, 'w') as f:
            f.write(output)
        if not args.quiet:
            print(f"\nReport written to: {args.output_file}")
    else:
        print(output)


if __name__ == "__main__":
    main()
