#!/usr/bin/env python
"""
Graph Mapping Analysis Tool

Analyzes how computational graphs are partitioned and mapped onto hardware resources.
Provides detailed insight into:
- Graph partitioning into subgraphs
- Memory and compute requirements per subgraph
- Hardware resource allocation per subgraph
- Power and latency estimates per subgraph
- Sequential execution modeling
- Total power and latency for complete execution

This tool helps compiler and hardware designers understand:
- How computational graphs use hardware
- Where performance is lost (low utilization, bottlenecks)
- Optimization opportunities (fusion, data layout, etc.)

Usage:
    ./cli/analyze_graph_mapping.py --model resnet18 --hardware H100
    ./cli/analyze_graph_mapping.py --model mobilenet_v2 --hardware Jetson-Orin --batch-size 4
    ./cli/analyze_graph_mapping.py --model resnet50 --hardware TPU-v4 --precision int8
"""

import argparse
import torch
import torch.nn as nn
from torch.fx import symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp
from torchvision import models
import sys
from pathlib import Path
from typing import Tuple, List, Dict

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.graphs.transform.partitioning import FusionBasedPartitioner
from src.graphs.analysis.allocation import (
    SubgraphAllocation,
    ExecutionPlan,
    HardwareAllocation
)
from src.graphs.hardware.resource_model import Precision
from src.graphs.ir.structures import BottleneckType

# Import hardware mappers
from src.graphs.hardware.mappers.gpu import (
    create_h100_mapper,
    create_jetson_orin_agx_mapper,
    create_jetson_orin_nano_mapper,
)
from src.graphs.hardware.mappers.accelerators.tpu import (
    create_tpu_v4_mapper,
    create_coral_edge_tpu_mapper,
)
from src.graphs.hardware.mappers.accelerators.kpu import (
    create_kpu_t64_mapper,
    create_kpu_t256_mapper,
    create_kpu_t768_mapper,
)
from src.graphs.hardware.mappers.cpu import (
    create_amd_epyc_9754_mapper,
    create_intel_xeon_platinum_8490h_mapper,
    create_i7_12700k_mapper,
    create_amd_cpu_mapper,
)
from src.graphs.hardware.mappers.dsp import (
    create_qrb5165_mapper,
    create_ti_tda4vm_mapper,
)


# =============================================================================
# Model Creation
# =============================================================================

def create_model(model_name: str, batch_size: int) -> Tuple[nn.Module, torch.Tensor, str]:
    """
    Create PyTorch model and input tensor

    Args:
        model_name: Name of model ('resnet18', 'resnet50', 'mobilenet_v2', etc.)
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

    # MobileNet family
    elif model_name_lower == 'mobilenet_v2':
        model = models.mobilenet_v2(weights=None)
        input_tensor = torch.randn(batch_size, 3, 224, 224)
        display_name = "MobileNet-V2"

    elif model_name_lower == 'mobilenet_v3_small':
        model = models.mobilenet_v3_small(weights=None)
        input_tensor = torch.randn(batch_size, 3, 224, 224)
        display_name = "MobileNet-V3-Small"

    elif model_name_lower == 'mobilenet_v3_large':
        model = models.mobilenet_v3_large(weights=None)
        input_tensor = torch.randn(batch_size, 3, 224, 224)
        display_name = "MobileNet-V3-Large"

    # EfficientNet family
    elif model_name_lower in ['efficientnet_b0', 'efficientnet-b0']:
        model = models.efficientnet_b0(weights=None)
        input_tensor = torch.randn(batch_size, 3, 224, 224)
        display_name = "EfficientNet-B0"

    elif model_name_lower in ['efficientnet_b1', 'efficientnet-b1']:
        model = models.efficientnet_b1(weights=None)
        input_tensor = torch.randn(batch_size, 3, 240, 240)
        display_name = "EfficientNet-B1"

    # VGG family
    elif model_name_lower == 'vgg16':
        model = models.vgg16(weights=None)
        input_tensor = torch.randn(batch_size, 3, 224, 224)
        display_name = "VGG-16"

    else:
        raise ValueError(f"Unknown model: {model_name}")

    model.eval()
    return model, input_tensor, display_name


# =============================================================================
# Hardware Mapper Creation
# =============================================================================

# Supported hardware configurations (exact names only)
SUPPORTED_HARDWARE = {
    # GPUs
    'H100': (create_h100_mapper, "GPU", "NVIDIA H100 PCIe"),
    'Jetson-Orin-AGX': (create_jetson_orin_agx_mapper, "GPU", "NVIDIA Jetson Orin AGX"),
    'Jetson-Orin-Nano': (create_jetson_orin_nano_mapper, "GPU", "NVIDIA Jetson Orin Nano"),

    # TPUs
    'TPU-v4': (create_tpu_v4_mapper, "TPU", "Google TPU v4"),
    'Coral-Edge-TPU': (create_coral_edge_tpu_mapper, "TPU", "Google Coral Edge TPU"),

    # KPUs
    'KPU-T64': (create_kpu_t64_mapper, "KPU", "Stillwater KPU-T64"),
    'KPU-T256': (create_kpu_t256_mapper, "KPU", "Stillwater KPU-T256"),
    'KPU-T768': (create_kpu_t768_mapper, "KPU", "Stillwater KPU-T768"),

    # CPUs - Datacenter
    'Xeon-8490H': (create_intel_xeon_platinum_8490h_mapper, "CPU", "Intel Xeon Platinum 8490H"),
    'EPYC-9754': (create_amd_epyc_9754_mapper, "CPU", "AMD EPYC 9754"),

    # CPUs - Consumer
    'i7-12700K': (create_i7_12700k_mapper, "CPU", "Intel Core i7-12700K"),
    'Ryzen-7-5800X': (create_amd_cpu_mapper, "CPU", "AMD Ryzen 7 5800X"),

    # DSPs
    'QRB5165': (create_qrb5165_mapper, "DSP", "Qualcomm QRB5165"),
    'TDA4VM': (create_ti_tda4vm_mapper, "DSP", "TI TDA4VM"),
}


def create_hardware_mapper(hardware_name: str, thermal_profile: str = None):
    """
    Create hardware mapper for exact hardware name

    Args:
        hardware_name: Exact hardware name (see SUPPORTED_HARDWARE)
        thermal_profile: Thermal/power profile ('10W', '350W', etc.)

    Returns:
        (mapper, hardware_type, display_name)

    Raises:
        ValueError: If hardware_name is not in SUPPORTED_HARDWARE
    """
    if hardware_name not in SUPPORTED_HARDWARE:
        # Generate helpful error message
        error_msg = f"Unsupported hardware: '{hardware_name}'\n\n"
        error_msg += "Supported hardware options:\n"
        error_msg += "  GPUs:\n"
        for name, (_, hw_type, display) in SUPPORTED_HARDWARE.items():
            if hw_type == "GPU":
                error_msg += f"    - {name:<20} ({display})\n"
        error_msg += "  TPUs:\n"
        for name, (_, hw_type, display) in SUPPORTED_HARDWARE.items():
            if hw_type == "TPU":
                error_msg += f"    - {name:<20} ({display})\n"
        error_msg += "  KPUs:\n"
        for name, (_, hw_type, display) in SUPPORTED_HARDWARE.items():
            if hw_type == "KPU":
                error_msg += f"    - {name:<20} ({display})\n"
        error_msg += "  CPUs:\n"
        for name, (_, hw_type, display) in SUPPORTED_HARDWARE.items():
            if hw_type == "CPU":
                error_msg += f"    - {name:<20} ({display})\n"
        error_msg += "  DSPs:\n"
        for name, (_, hw_type, display) in SUPPORTED_HARDWARE.items():
            if hw_type == "DSP":
                error_msg += f"    - {name:<20} ({display})\n"

        raise ValueError(error_msg)

    # Get the mapper creation function
    mapper_fn, hw_type, display_name = SUPPORTED_HARDWARE[hardware_name]

    # Create mapper (some take thermal_profile, some don't)
    try:
        mapper = mapper_fn(thermal_profile)
    except TypeError:
        # Function doesn't take thermal_profile argument
        mapper = mapper_fn()

    return mapper, hw_type, display_name


# =============================================================================
# Allocation Stub (Phase 2 will implement this)
# =============================================================================

def allocate_hardware_resources(subgraph, hw_model, hw_type: str) -> HardwareAllocation:
    """
    Allocate hardware resources for a subgraph

    Dispatches to hardware-specific allocation functions

    Args:
        subgraph: FusedSubgraph
        hw_model: HardwareResourceModel
        hw_type: Hardware type ('GPU', 'TPU', 'KPU', 'CPU', 'DSP')

    Returns:
        HardwareAllocation
    """
    if hw_type == "GPU":
        return allocate_gpu_resources(subgraph, hw_model)
    elif hw_type == "TPU":
        return allocate_tpu_resources(subgraph, hw_model)
    elif hw_type == "KPU":
        return allocate_kpu_resources(subgraph, hw_model)
    elif hw_type == "CPU":
        return allocate_cpu_resources(subgraph, hw_model)
    elif hw_type == "DSP":
        return allocate_dsp_resources(subgraph, hw_model)
    else:
        # Fallback for unknown hardware types
        return HardwareAllocation(
            hardware_type=hw_type,
            allocated_units=1,
            total_available_units=hw_model.compute_units,
            utilization=0.1,
            allocation_details={'note': 'Unknown hardware type'}
        )


def allocate_gpu_resources(subgraph, hw_model) -> HardwareAllocation:
    """
    Allocate GPU resources (SMs) based on subgraph parallelism

    GPU hierarchy: Thread → Warp → SM → GPU
    - Threads are grouped into warps (32 threads/warp on NVIDIA)
    - Warps execute on SMs (up to ~64 warps/SM depending on occupancy)
    - SMs are allocated in waves (quantization: typically 4 SMs/wave)

    Args:
        subgraph: FusedSubgraph with parallelism info
        hw_model: GPU HardwareResourceModel

    Returns:
        HardwareAllocation with GPU-specific allocation details
    """
    import math

    # Extract parallelism
    if subgraph.parallelism is None:
        # No parallelism info - allocate minimum
        threads_required = 1
    else:
        threads_required = subgraph.parallelism.total_threads

    # GPU configuration
    warp_size = hw_model.warp_size  # 32 for NVIDIA
    max_warps_per_sm = hw_model.warps_per_unit  # e.g., 64 for H100
    threads_per_sm = warp_size * max_warps_per_sm  # e.g., 2048 for H100
    total_sms = hw_model.compute_units  # e.g., 132 for H100
    wave_quantization = hw_model.wave_quantization  # e.g., 4

    # Calculate warps needed
    warps_required = math.ceil(threads_required / warp_size)

    # Calculate SMs needed (minimum)
    sms_needed_ideal = math.ceil(warps_required / max_warps_per_sm)

    # Apply wave quantization (SMs allocated in groups)
    if sms_needed_ideal > 0:
        waves = math.ceil(sms_needed_ideal / wave_quantization)
        sms_allocated = waves * wave_quantization
    else:
        sms_allocated = wave_quantization  # Minimum 1 wave

    # Cap at available SMs
    sms_allocated = min(sms_allocated, total_sms)

    # Calculate actual utilization
    total_threads_capacity = sms_allocated * threads_per_sm
    utilization = threads_required / total_threads_capacity if total_threads_capacity > 0 else 0.0
    utilization = min(1.0, utilization)  # Cap at 100%

    # Build allocation details
    allocation_details = {
        'threads_required': threads_required,
        'warps_required': warps_required,
        'sms_needed_ideal': sms_needed_ideal,
        'sms_allocated': sms_allocated,
        'waves': sms_allocated // wave_quantization,
        'threads_per_sm': threads_per_sm,
        'warp_size': warp_size,
        'utilization_per_sm': utilization,
    }

    return HardwareAllocation(
        hardware_type="GPU",
        allocated_units=sms_allocated,
        total_available_units=total_sms,
        utilization=utilization,
        allocation_details=allocation_details
    )


def allocate_tpu_resources(subgraph, hw_model) -> HardwareAllocation:
    """
    Allocate TPU resources (systolic array tiles)

    TPU uses 2D systolic arrays for matrix multiplication:
    - Each tile is a 2D grid of MAC units (e.g., 128×128 for TPU v4)
    - Operations are tiled to fit the array
    - Utilization depends on how well matrix dims map to tile size

    Args:
        subgraph: FusedSubgraph
        hw_model: TPU HardwareResourceModel

    Returns:
        HardwareAllocation with TPU-specific allocation details
    """
    import math

    # For TPUs, compute_units typically represents number of cores/tiles
    total_tiles = hw_model.compute_units

    # Estimate tile usage based on operation size
    # For simplicity, assume we need 1 tile minimum, more for larger ops
    if subgraph.parallelism is None:
        tiles_needed = 1
    else:
        # Rough heuristic: larger parallelism = more tiles
        parallelism = subgraph.parallelism.total_threads
        tiles_needed = min(max(1, parallelism // 10000), total_tiles)

    tiles_allocated = tiles_needed

    # Utilization: how well the operation fits the systolic array
    # Simplified: assume 60-80% for matrix ops, 30% for non-matrix
    is_matrix_op = any(op in ['conv2d', 'linear', 'matmul']
                      for op in [str(ot.value) for ot in subgraph.operation_types])
    utilization = 0.7 if is_matrix_op else 0.3

    allocation_details = {
        'tiles_allocated': tiles_allocated,
        'array_size': '128x128',  # TPU v4 typical
        'is_matrix_op': is_matrix_op,
        'tile_utilization': utilization,
    }

    return HardwareAllocation(
        hardware_type="TPU",
        allocated_units=tiles_allocated,
        total_available_units=total_tiles,
        utilization=utilization,
        allocation_details=allocation_details
    )


def allocate_kpu_resources(subgraph, hw_model) -> HardwareAllocation:
    """
    Allocate KPU resources (compute tiles + L3 cache tiles)

    KPU architecture:
    - Compute tiles for processing
    - L3 cache tiles for data
    - Spatial dataflow execution

    Args:
        subgraph: FusedSubgraph
        hw_model: KPU HardwareResourceModel

    Returns:
        HardwareAllocation with KPU-specific allocation details
    """
    import math

    total_tiles = hw_model.compute_units  # e.g., 64, 256, 768

    # Estimate tiles based on parallelism
    if subgraph.parallelism is None:
        tiles_needed = 1
    else:
        # KPU tiles process spatial regions
        spatial = subgraph.parallelism.spatial
        channels = subgraph.parallelism.channels
        tiles_needed = min(max(1, spatial // 64), total_tiles)

    tiles_allocated = tiles_needed

    # Utilization depends on how well spatial dimensions tile
    if subgraph.parallelism:
        # Better utilization for larger spatial dimensions
        spatial = subgraph.parallelism.spatial
        utilization = min(0.9, 0.4 + (spatial / 10000))
    else:
        utilization = 0.3

    allocation_details = {
        'compute_tiles': tiles_allocated,
        'l3_tiles': tiles_allocated // 4,  # Rough ratio
        'schedule': 'spatial',
    }

    return HardwareAllocation(
        hardware_type="KPU",
        allocated_units=tiles_allocated,
        total_available_units=total_tiles,
        utilization=utilization,
        allocation_details=allocation_details
    )


def allocate_cpu_resources(subgraph, hw_model) -> HardwareAllocation:
    """
    Allocate CPU resources (cores + SIMD lanes)

    CPU execution:
    - Allocate cores for thread-level parallelism
    - SIMD vectorization within each core
    - Consider hyperthreading (2× threads/core)

    Args:
        subgraph: FusedSubgraph
        hw_model: CPU HardwareResourceModel

    Returns:
        HardwareAllocation with CPU-specific allocation details
    """
    import math

    total_cores = hw_model.compute_units
    threads_per_core = hw_model.threads_per_unit  # e.g., 2 for hyperthreading

    # Estimate cores needed
    if subgraph.parallelism is None:
        cores_needed = 1
    else:
        threads_required = subgraph.parallelism.total_threads
        # Each core can handle threads_per_core threads
        cores_needed = min(math.ceil(threads_required / threads_per_core), total_cores)

    cores_allocated = max(1, cores_needed)

    # Utilization: depends on thread efficiency and SIMD usage
    if subgraph.parallelism:
        threads_required = subgraph.parallelism.total_threads
        total_thread_capacity = cores_allocated * threads_per_core
        thread_utilization = min(1.0, threads_required / total_thread_capacity)
        # SIMD adds another dimension (assume 50% SIMD efficiency)
        utilization = thread_utilization * 0.7  # Account for SIMD efficiency
    else:
        utilization = 0.3

    allocation_details = {
        'cores_allocated': cores_allocated,
        'threads_per_core': threads_per_core,
        'simd_lanes': 8,  # e.g., AVX2
        'vector_utilization': 0.5,
    }

    return HardwareAllocation(
        hardware_type="CPU",
        allocated_units=cores_allocated,
        total_available_units=total_cores,
        utilization=utilization,
        allocation_details=allocation_details
    )


def allocate_dsp_resources(subgraph, hw_model) -> HardwareAllocation:
    """
    Allocate DSP resources (vector units + HVX threads)

    DSP architecture (e.g., Qualcomm Hexagon, TI C7x):
    - Vector units for SIMD processing
    - Tensor accelerators for matrix ops
    - Specialized for signal processing and vision

    Args:
        subgraph: FusedSubgraph
        hw_model: DSP HardwareResourceModel

    Returns:
        HardwareAllocation with DSP-specific allocation details
    """
    import math

    # DSPs typically have fewer units than GPUs
    total_vector_units = hw_model.compute_units  # e.g., 4 for Hexagon
    threads_per_unit = hw_model.threads_per_unit

    # Estimate units needed
    if subgraph.parallelism is None:
        units_needed = 1
    else:
        threads_required = subgraph.parallelism.total_threads
        units_needed = min(math.ceil(threads_required / threads_per_unit), total_vector_units)

    units_allocated = max(1, units_needed)

    # Utilization: DSPs excel at certain operations (conv, vision)
    is_dsp_friendly = any(op in ['conv2d', 'pool', 'activation']
                         for op in [str(ot.value) for ot in subgraph.operation_types])
    base_utilization = 0.7 if is_dsp_friendly else 0.4

    if subgraph.parallelism:
        threads_required = subgraph.parallelism.total_threads
        capacity = units_allocated * threads_per_unit
        thread_util = min(1.0, threads_required / capacity)
        utilization = base_utilization * thread_util
    else:
        utilization = base_utilization * 0.5

    allocation_details = {
        'vector_units': units_allocated,
        'hvx_threads': threads_per_unit,
        'tensor_accelerator': is_dsp_friendly,
        'lane_width': 128,  # Typical for Hexagon HVX
    }

    return HardwareAllocation(
        hardware_type="DSP",
        allocated_units=units_allocated,
        total_available_units=total_vector_units,
        utilization=utilization,
        allocation_details=allocation_details
    )


def estimate_subgraph_performance(subgraph, allocation, hw_model, thermal_profile, tdp) -> Tuple[float, float, float, float, float]:
    """
    Estimate power and latency for subgraph on allocated resources using roofline model

    Roofline Model:
    - Compute time = FLOPs / (effective_flops)
    - Memory time = bytes / memory_bandwidth
    - Actual latency = max(compute_time, memory_time)

    Power Model (with idle power):
    - Idle power = TDP × 0.5 (nanoscale leakage)
    - Dynamic power = (TDP × 0.5) × (allocated_units / total_units) × utilization
    - Total power = idle_power + dynamic_power

    Args:
        subgraph: FusedSubgraph with FLOPs and memory info
        allocation: HardwareAllocation with utilization
        hw_model: HardwareResourceModel
        thermal_profile: Selected thermal profile name
        tdp: TDP in watts

    Returns:
        (compute_time_ms, memory_time_ms, idle_power_w, dynamic_power_w, total_power_w)
    """
    # Get performance characteristics for the thermal profile and precision
    # Use FP16 as default precision for modern accelerators
    precision = Precision.FP16

    # Check if hardware has thermal operating points
    if hw_model.thermal_operating_points and thermal_profile:
        thermal_point = hw_model.thermal_operating_points[thermal_profile]
        perf_spec = thermal_point.performance_specs.get(precision)

        if perf_spec is None:
            # Fallback to getting any available precision
            if thermal_point.performance_specs:
                perf_spec = list(thermal_point.performance_specs.values())[0]
                effective_ops_per_sec = perf_spec.effective_ops_per_sec
            else:
                # Last resort: use peak ops from precision profiles
                peak_ops_per_sec = hw_model.get_peak_ops(precision)
                # Create a simple effective ops estimate
                effective_ops_per_sec = peak_ops_per_sec * 0.6  # Assume 60% efficiency
        else:
            effective_ops_per_sec = perf_spec.effective_ops_per_sec
    else:
        # Hardware without thermal profiles - use legacy precision profiles
        peak_ops_per_sec = hw_model.get_peak_ops(precision)
        # Apply typical efficiency factor for consumer hardware
        effective_ops_per_sec = peak_ops_per_sec * 0.6  # Assume 60% efficiency

    # Calculate effective FLOPS based on allocation
    # Only the allocated portion of hardware is working
    allocation_fraction = allocation.allocated_units / allocation.total_available_units
    effective_flops = effective_ops_per_sec * allocation_fraction * allocation.utilization

    # Prevent division by zero
    if effective_flops == 0:
        effective_flops = effective_ops_per_sec * 0.01  # Use 1% as minimum

    # Compute time (roofline model)
    compute_time_sec = subgraph.total_flops / effective_flops
    compute_time_ms = compute_time_sec * 1000.0

    # Memory time (roofline model)
    total_memory_bytes = (subgraph.total_input_bytes +
                         subgraph.total_output_bytes +
                         subgraph.total_weight_bytes)

    # Memory bandwidth might be affected by concurrent access
    # For sequential execution, we get full bandwidth
    memory_bandwidth_bytes_per_sec = hw_model.peak_bandwidth

    if total_memory_bytes == 0 or memory_bandwidth_bytes_per_sec == 0:
        memory_time_sec = 0.0
    else:
        memory_time_sec = total_memory_bytes / memory_bandwidth_bytes_per_sec

    memory_time_ms = memory_time_sec * 1000.0

    # Power modeling with idle + dynamic
    # Modern hardware consumes ~50% TDP at idle due to nanoscale leakage
    idle_power_w = tdp * 0.5

    # Dynamic power scales with:
    # 1. What fraction of chip is allocated (allocation_fraction)
    # 2. How well that allocation is utilized (utilization)
    dynamic_power_w = (tdp * 0.5) * allocation_fraction * allocation.utilization

    total_power_w = idle_power_w + dynamic_power_w

    return compute_time_ms, memory_time_ms, idle_power_w, dynamic_power_w, total_power_w


# =============================================================================
# Graph Mapping Analysis
# =============================================================================

def analyze_graph_mapping(
    model_name: str,
    hardware_name: str,
    batch_size: int = 1,
    precision: str = 'fp16',
    thermal_profile: str = 'default'
) -> ExecutionPlan:
    """
    Analyze how a model is partitioned and mapped to hardware

    Args:
        model_name: Name of model to analyze
        hardware_name: Target hardware
        batch_size: Batch size
        precision: Precision ('fp32', 'fp16', 'int8')
        thermal_profile: Thermal/power profile

    Returns:
        ExecutionPlan with complete allocation and execution info
    """
    print(f"\n{'='*100}")
    print(f"Graph Mapping Analysis: {model_name} → {hardware_name}")
    print(f"{'='*100}\n")

    # 1. Create model
    print(f"[1/5] Creating model: {model_name} (batch_size={batch_size})...")
    model, input_tensor, display_name = create_model(model_name, batch_size)

    # 2. Trace and partition graph
    print(f"[2/5] Tracing and partitioning graph...")
    traced = symbolic_trace(model)
    ShapeProp(traced).propagate(input_tensor)

    partitioner = FusionBasedPartitioner()
    fusion_report = partitioner.partition(traced)

    print(f"      ✓ Graph partitioned into {fusion_report.total_subgraphs} subgraphs")
    print(f"      ✓ Total FLOPs: {fusion_report.total_flops / 1e9:.2f} GFLOPs")
    print(f"      ✓ Total memory: {fusion_report.total_memory_traffic_fused / 1e6:.2f} MB")

    # 3. Create hardware mapper
    print(f"[3/5] Creating hardware mapper: {hardware_name}...")
    try:
        mapper, hw_type, hw_display_name = create_hardware_mapper(hardware_name, thermal_profile)
        hw_model = mapper.resource_model
    except ValueError as e:
        if "Thermal profile" in str(e) and "not found" in str(e):
            print(f"\n❌ Error: {e}")
            print(f"\nTip: Omit --thermal-profile to use the default, or choose from the available profiles listed above.")
            raise SystemExit(1)
        else:
            raise

    print(f"      ✓ Hardware: {hw_display_name}")

    # Get peak FLOPS for default precision (FP16)
    peak_ops = hw_model.get_peak_ops(Precision.FP16)
    print(f"      ✓ Peak FLOPS: {peak_ops / 1e12:.2f} TFLOPS (FP16)")
    print(f"      ✓ Memory BW: {hw_model.peak_bandwidth / 1e9:.1f} GB/s")

    # Get TDP from selected thermal profile or fallback
    if hw_model.thermal_operating_points and mapper.thermal_profile:
        tdp = hw_model.thermal_operating_points[mapper.thermal_profile].tdp_watts
        print(f"      ✓ TDP: {tdp}W (thermal profile: {mapper.thermal_profile})")
    else:
        # Fallback for hardware without thermal profiles
        # Use reasonable defaults based on hardware type
        tdp_defaults = {
            "CPU": 105.0,   # Typical consumer/datacenter CPU TDP
            "GPU": 300.0,   # Typical discrete GPU TDP
            "DSP": 15.0,    # Typical DSP TDP
            "KPU": 30.0,    # Typical accelerator TDP
            "TPU": 200.0,   # Typical TPU TDP
        }
        tdp = tdp_defaults.get(hw_type, 100.0)  # Generic fallback
        print(f"      ✓ TDP: {tdp}W (estimated)")

    # 4. Allocate resources and estimate performance
    print(f"[4/5] Allocating resources and estimating performance...")

    subgraph_allocations = []

    for fused_sg in fusion_report.fused_subgraphs:
        # Create subgraph descriptor from fused subgraph
        # (We need to convert FusedSubgraph to SubgraphDescriptor format)
        from graphs.ir.structures import SubgraphDescriptor, OperationType

        # Allocate hardware resources (STUB - Phase 2 will implement)
        hw_allocation = allocate_hardware_resources(fused_sg, hw_model, hw_type)

        # Estimate performance (STUB - Phase 3 will implement)
        compute_ms, memory_ms, idle_w, dynamic_w, total_w = estimate_subgraph_performance(
            fused_sg, hw_allocation, hw_model, mapper.thermal_profile, tdp
        )

        # Create allocation record
        alloc = SubgraphAllocation(
            subgraph_id=fused_sg.subgraph_id,
            subgraph_descriptor=None,  # Will populate in Phase 2
            operation_types=[op.value for op in fused_sg.operation_types],
            fusion_pattern=fused_sg.fusion_pattern,
            flops=fused_sg.total_flops,
            memory_bytes=fused_sg.total_input_bytes + fused_sg.total_output_bytes + fused_sg.total_weight_bytes,
            arithmetic_intensity=fused_sg.arithmetic_intensity,
            total_threads=fused_sg.parallelism.total_threads if fused_sg.parallelism else 1,
            hardware_allocation=hw_allocation,
            compute_time_ms=compute_ms,
            memory_time_ms=memory_ms,
            actual_latency_ms=max(compute_ms, memory_ms),
            idle_power_watts=idle_w,
            dynamic_power_watts=dynamic_w,
            total_power_watts=total_w,
            bottleneck_type=fused_sg.recommended_bottleneck,
            bottleneck_explanation="",
            depends_on=[],  # Will populate from dependency analysis
            dependency_type="sequential"
        )

        subgraph_allocations.append(alloc)

    print(f"      ✓ Allocated resources for {len(subgraph_allocations)} subgraphs")

    # 5. Create execution plan
    print(f"[5/5] Creating execution plan...")

    execution_plan = ExecutionPlan(
        model_name=display_name,
        total_operations=len(fusion_report.fused_subgraphs),
        total_flops=fusion_report.total_flops,
        total_memory_traffic=fusion_report.total_memory_traffic_fused,
        hardware_name=hw_display_name,
        hardware_type=hw_type,
        peak_flops=peak_ops,
        memory_bandwidth=hw_model.peak_bandwidth / 1e9,  # Convert to GB/s
        tdp_watts=tdp,
        subgraph_allocations=subgraph_allocations
    )

    # Compute aggregates
    execution_plan.compute_aggregates()
    execution_plan.generate_warnings_and_suggestions()

    print(f"      ✓ Execution plan created")
    print(f"      ✓ Total latency: {execution_plan.total_latency_ms:.2f} ms")
    print(f"      ✓ Average power: {execution_plan.average_power_watts:.1f} W")
    print(f"      ✓ Average utilization: {execution_plan.average_utilization*100:.1f}%")

    return execution_plan


# =============================================================================
# Report Generation (Phase 4 will enhance this)
# =============================================================================

def print_execution_plan_report(plan: ExecutionPlan):
    """
    Print comprehensive execution plan report with detailed subgraph breakdown
    """
    print(f"\n{'='*120}")
    print(f"GRAPH MAPPING ANALYSIS REPORT")
    print(f"{'='*120}\n")

    # =========================================================================
    # Section 1: Model Summary
    # =========================================================================
    print(f"{'─'*120}")
    print(f"MODEL SUMMARY")
    print(f"{'─'*120}")
    print(f"Model:              {plan.model_name}")
    print(f"Total Operations:   {plan.total_operations} subgraphs")
    print(f"Total FLOPs:        {plan.total_flops / 1e9:.2f} GFLOPs ({plan.total_flops:,} ops)")
    print(f"Total Memory:       {plan.total_memory_traffic / 1e6:.2f} MB ({plan.total_memory_traffic:,} bytes)")
    avg_ai = plan.total_flops / plan.total_memory_traffic if plan.total_memory_traffic > 0 else 0
    print(f"Arithmetic Intensity: {avg_ai:.2f} FLOPs/byte")
    print()

    # =========================================================================
    # Section 2: Hardware Summary
    # =========================================================================
    print(f"{'─'*120}")
    print(f"HARDWARE TARGET")
    print(f"{'─'*120}")
    print(f"Hardware:           {plan.hardware_name} ({plan.hardware_type})")
    print(f"Peak FLOPS:         {plan.peak_flops / 1e12:.2f} TFLOPS ({plan.peak_flops:,.0f} ops/sec)")
    print(f"Memory Bandwidth:   {plan.memory_bandwidth:.1f} GB/s ({plan.memory_bandwidth * 1e9:,.0f} bytes/sec)")
    print(f"TDP:                {plan.tdp_watts:.0f} W")
    print()

    # =========================================================================
    # Section 3: Execution Summary
    # =========================================================================
    print(f"{'─'*120}")
    print(f"EXECUTION SUMMARY")
    print(f"{'─'*120}")
    print(f"Total Latency:      {plan.total_latency_ms:.3f} ms ({plan.throughput_fps():.1f} FPS)")
    print(f"Average Power:      {plan.average_power_watts:.1f} W")
    print(f"Peak Power:         {plan.peak_power_watts:.1f} W")
    print(f"Total Energy:       {plan.energy_per_inference_mj():.2f} mJ/inference")
    print(f"Average Utilization: {plan.average_utilization*100:.1f}% (min: {plan.min_utilization*100:.1f}%, peak: {plan.peak_utilization*100:.1f}%)")
    print(f"Hardware Efficiency: {plan.hardware_efficiency*100:.1f}% of peak FLOPS")
    print(f"Memory Efficiency:  {plan.memory_efficiency*100:.1f}% of peak bandwidth")
    print()

    # =========================================================================
    # Section 4: Detailed Subgraph Breakdown
    # =========================================================================
    print(f"{'─'*120}")
    print(f"DETAILED SUBGRAPH BREAKDOWN")
    print(f"{'─'*120}")

    # Table header
    print(f"{'ID':<4} {'Operations':<25} {'FLOPs':<12} {'Mem':<10} {'Alloc':<12} {'Util%':<7} {'Bottleneck':<12} {'Latency':<10} {'Power':<8}")
    print(f"{'─'*120}")

    # Track cumulative metrics for progress visualization
    cumulative_latency = 0.0

    for i, alloc in enumerate(plan.subgraph_allocations):
        # Format operation list
        ops_str = alloc.fusion_pattern if len(alloc.fusion_pattern) <= 23 else alloc.fusion_pattern[:20] + "..."

        # Format FLOPs
        if alloc.flops >= 1e9:
            flops_str = f"{alloc.flops / 1e9:.2f}G"
        elif alloc.flops >= 1e6:
            flops_str = f"{alloc.flops / 1e6:.2f}M"
        elif alloc.flops >= 1e3:
            flops_str = f"{alloc.flops / 1e3:.2f}K"
        else:
            flops_str = f"{alloc.flops}"

        # Format memory
        if alloc.memory_bytes >= 1e6:
            mem_str = f"{alloc.memory_bytes / 1e6:.2f}MB"
        elif alloc.memory_bytes >= 1e3:
            mem_str = f"{alloc.memory_bytes / 1e3:.2f}KB"
        else:
            mem_str = f"{alloc.memory_bytes}B"

        # Format allocation based on hardware type
        hw_alloc = alloc.hardware_allocation
        if hw_alloc.hardware_type == "GPU":
            alloc_str = f"{hw_alloc.allocated_units} SMs"
        elif hw_alloc.hardware_type == "TPU":
            alloc_str = f"{hw_alloc.allocated_units} tiles"
        elif hw_alloc.hardware_type == "KPU":
            alloc_str = f"{hw_alloc.allocated_units} tiles"
        elif hw_alloc.hardware_type == "CPU":
            alloc_str = f"{hw_alloc.allocated_units} cores"
        elif hw_alloc.hardware_type == "DSP":
            alloc_str = f"{hw_alloc.allocated_units} units"
        else:
            alloc_str = f"{hw_alloc.allocated_units} units"

        # Utilization percentage
        util_pct = hw_alloc.utilization * 100

        # Bottleneck type
        bottleneck = alloc.bottleneck_type.value.replace('_', '-')[:11]

        # Latency
        latency_str = f"{alloc.actual_latency_ms:.3f}ms"

        # Power
        power_str = f"{alloc.total_power_watts:.1f}W"

        # Print row
        print(f"{i:<4} {ops_str:<25} {flops_str:<12} {mem_str:<10} {alloc_str:<12} {util_pct:>6.1f} {bottleneck:<12} {latency_str:<10} {power_str:<8}")

        cumulative_latency += alloc.actual_latency_ms

    print(f"{'─'*120}")
    print(f"{'TOTAL':<4} {'':<25} {plan.total_flops/1e9:>10.2f}G {plan.total_memory_traffic/1e6:>8.2f}MB {'':<12} {plan.average_utilization*100:>6.1f} {'':<12} {plan.total_latency_ms:>8.3f}ms {plan.average_power_watts:>6.1f}W")
    print()

    # =========================================================================
    # Section 5: Bottleneck Analysis
    # =========================================================================
    print(f"{'─'*120}")
    print(f"BOTTLENECK ANALYSIS")
    print(f"{'─'*120}")

    total_subs = len(plan.subgraph_allocations)
    compute_pct = (plan.compute_bound_subgraphs / total_subs * 100) if total_subs > 0 else 0
    memory_pct = (plan.memory_bound_subgraphs / total_subs * 100) if total_subs > 0 else 0

    print(f"Compute-bound:      {plan.compute_bound_subgraphs:>3} subgraphs ({compute_pct:>5.1f}%)")
    print(f"Memory-bound:       {plan.memory_bound_subgraphs:>3} subgraphs ({memory_pct:>5.1f}%)")

    # Identify most expensive subgraphs
    sorted_by_latency = sorted(enumerate(plan.subgraph_allocations),
                               key=lambda x: x[1].actual_latency_ms, reverse=True)

    print(f"\nTop 5 Most Expensive Subgraphs (by latency):")
    for rank, (idx, alloc) in enumerate(sorted_by_latency[:5], 1):
        pct_of_total = (alloc.actual_latency_ms / plan.total_latency_ms * 100) if plan.total_latency_ms > 0 else 0
        print(f"  {rank}. Subgraph {idx:>2}: {alloc.fusion_pattern:<30} {alloc.actual_latency_ms:>8.3f}ms ({pct_of_total:>5.1f}% of total)")

    # Identify low utilization subgraphs
    low_util = [(i, alloc) for i, alloc in enumerate(plan.subgraph_allocations)
                if alloc.hardware_allocation.utilization < 0.3]

    if low_util:
        print(f"\nLow Utilization Subgraphs (<30%):")
        for idx, alloc in low_util[:5]:  # Show top 5
            util_pct = alloc.hardware_allocation.utilization * 100
            print(f"  Subgraph {idx:>2}: {alloc.fusion_pattern:<30} {util_pct:>5.1f}% utilization")
    print()

    # =========================================================================
    # Section 6: Warnings and Optimization Suggestions
    # =========================================================================
    if plan.bottleneck_warnings or plan.optimization_suggestions:
        print(f"{'─'*120}")
        print(f"WARNINGS & OPTIMIZATION SUGGESTIONS")
        print(f"{'─'*120}")

    if plan.bottleneck_warnings:
        print(f"⚠️  Warnings:")
        for warning in plan.bottleneck_warnings:
            print(f"    {warning}")
        print()

    if plan.optimization_suggestions:
        print(f"💡 Optimization Suggestions:")
        for suggestion in plan.optimization_suggestions:
            print(f"    {suggestion}")
        print()

    print(f"{'='*120}\n")


# =============================================================================
# Main CLI
# =============================================================================

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Analyze graph partitioning and hardware mapping",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Model name (resnet18, resnet50, mobilenet_v2, etc.)'
    )

    parser.add_argument(
        '--hardware',
        type=str,
        required=True,
        help='Exact hardware name (run with invalid name to see full list)'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help='Batch size (default: 1)'
    )

    parser.add_argument(
        '--precision',
        type=str,
        default='fp16',
        choices=['fp32', 'fp16', 'int8', 'int4'],
        help='Precision (default: fp16)'
    )

    parser.add_argument(
        '--thermal-profile',
        type=str,
        default=None,
        help='Thermal/power profile (10W, 350W, etc.) - uses hardware default if not specified'
    )

    args = parser.parse_args()

    # Run analysis
    execution_plan = analyze_graph_mapping(
        model_name=args.model,
        hardware_name=args.hardware,
        batch_size=args.batch_size,
        precision=args.precision,
        thermal_profile=args.thermal_profile
    )

    # Print report
    print_execution_plan_report(execution_plan)


if __name__ == '__main__':
    main()
