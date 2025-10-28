#!/usr/bin/env python3
"""
Hardware Mapper Discovery and Comparison Tool

This CLI tool discovers all available hardware mappers in the package and generates
a comprehensive report with categories, specifications, and comparisons.

Usage:
    python cli/list_hardware_mappers.py
    python cli/list_hardware_mappers.py --category cpu
    python cli/list_hardware_mappers.py --category accelerators
    python cli/list_hardware_mappers.py --format json
"""

import json
import argparse
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

from graphs.hardware.resource_model import HardwareType, Precision


@dataclass
class HardwareMapperInfo:
    """Information about a hardware mapper"""
    name: str
    category: str  # CPU, GPU, DSP, TPU, KPU, DPU, CGRA, NPU
    deployment: str  # Datacenter, Edge, Mobile, Automotive
    manufacturer: str
    compute_units: int
    peak_flops_fp32: float  # GFLOPS
    peak_flops_int8: float  # GOPS
    memory_bandwidth: float  # GB/s
    power_tdp: float  # Watts
    thermal_profiles: List[str]
    use_cases: List[str]
    factory_function: str
    hardware_type: str  # programmable_isa or accelerator


def discover_cpu_mappers() -> List[HardwareMapperInfo]:
    """Discover all CPU mappers"""
    from graphs.hardware.mappers.cpu import (
        create_intel_xeon_platinum_8490h_mapper,
        create_amd_epyc_9754_mapper,
        create_amd_epyc_9654_mapper,
        create_ampere_ampereone_192_mapper,
        create_ampere_ampereone_128_mapper,
        create_intel_xeon_platinum_8592plus_mapper,
        create_intel_granite_rapids_mapper,
        create_amd_epyc_turin_mapper,
        create_i7_12700k_mapper,
    )

    mappers = []

    # Intel Xeon Platinum 8490H
    mapper = create_intel_xeon_platinum_8490h_mapper()
    mappers.append(HardwareMapperInfo(
        name="Intel Xeon Platinum 8490H",
        category="CPU",
        deployment="Datacenter",
        manufacturer="Intel",
        compute_units=mapper.resource_model.compute_units,
        peak_flops_fp32=mapper.resource_model.get_peak_ops(Precision.FP32) / 1e9,
        peak_flops_int8=mapper.resource_model.get_peak_ops(Precision.INT8) / 1e9,
        memory_bandwidth=mapper.resource_model.peak_bandwidth / 1e9,
        power_tdp=350.0,
        thermal_profiles=[],
        use_cases=["Cloud inference", "High-throughput servers"],
        factory_function="create_intel_xeon_platinum_8490h_mapper",
        hardware_type="programmable_isa"
    ))

    # Intel Xeon Platinum 8592+
    mapper = create_intel_xeon_platinum_8592plus_mapper()
    mappers.append(HardwareMapperInfo(
        name="Intel Xeon Platinum 8592+",
        category="CPU",
        deployment="Datacenter",
        manufacturer="Intel",
        compute_units=mapper.resource_model.compute_units,
        peak_flops_fp32=mapper.resource_model.get_peak_ops(Precision.FP32) / 1e9,
        peak_flops_int8=mapper.resource_model.get_peak_ops(Precision.INT8) / 1e9,
        memory_bandwidth=mapper.resource_model.peak_bandwidth / 1e9,
        power_tdp=350.0,
        thermal_profiles=[],
        use_cases=["Emerald Rapids", "Next-gen datacenter"],
        factory_function="create_intel_xeon_platinum_8592plus_mapper",
        hardware_type="programmable_isa"
    ))

    # Intel Granite Rapids
    mapper = create_intel_granite_rapids_mapper()
    mappers.append(HardwareMapperInfo(
        name="Intel Granite Rapids",
        category="CPU",
        deployment="Datacenter",
        manufacturer="Intel",
        compute_units=mapper.resource_model.compute_units,
        peak_flops_fp32=mapper.resource_model.get_peak_ops(Precision.FP32) / 1e9,
        peak_flops_int8=mapper.resource_model.get_peak_ops(Precision.INT8) / 1e9,
        memory_bandwidth=mapper.resource_model.peak_bandwidth / 1e9,
        power_tdp=500.0,
        thermal_profiles=[],
        use_cases=["Next-gen Intel", "2024-2025"],
        factory_function="create_intel_granite_rapids_mapper",
        hardware_type="programmable_isa"
    ))

    # AMD EPYC 9654
    mapper = create_amd_epyc_9654_mapper()
    mappers.append(HardwareMapperInfo(
        name="AMD EPYC 9654",
        category="CPU",
        deployment="Datacenter",
        manufacturer="AMD",
        compute_units=mapper.resource_model.compute_units,
        peak_flops_fp32=mapper.resource_model.get_peak_ops(Precision.FP32) / 1e9,
        peak_flops_int8=mapper.resource_model.get_peak_ops(Precision.INT8) / 1e9,
        memory_bandwidth=mapper.resource_model.peak_bandwidth / 1e9,
        power_tdp=360.0,
        thermal_profiles=[],
        use_cases=["Genoa", "Cloud"],
        factory_function="create_amd_epyc_9654_mapper",
        hardware_type="programmable_isa"
    ))

    # AMD EPYC 9754
    mapper = create_amd_epyc_9754_mapper()
    mappers.append(HardwareMapperInfo(
        name="AMD EPYC 9754",
        category="CPU",
        deployment="Datacenter",
        manufacturer="AMD",
        compute_units=mapper.resource_model.compute_units,
        peak_flops_fp32=mapper.resource_model.get_peak_ops(Precision.FP32) / 1e9,
        peak_flops_int8=mapper.resource_model.get_peak_ops(Precision.INT8) / 1e9,
        memory_bandwidth=mapper.resource_model.peak_bandwidth / 1e9,
        power_tdp=360.0,
        thermal_profiles=[],
        use_cases=["Bergamo", "Cloud-native"],
        factory_function="create_amd_epyc_9754_mapper",
        hardware_type="programmable_isa"
    ))

    # AMD EPYC Turin
    mapper = create_amd_epyc_turin_mapper()
    mappers.append(HardwareMapperInfo(
        name="AMD EPYC Turin",
        category="CPU",
        deployment="Datacenter",
        manufacturer="AMD",
        compute_units=mapper.resource_model.compute_units,
        peak_flops_fp32=mapper.resource_model.get_peak_ops(Precision.FP32) / 1e9,
        peak_flops_int8=mapper.resource_model.get_peak_ops(Precision.INT8) / 1e9,
        memory_bandwidth=mapper.resource_model.peak_bandwidth / 1e9,
        power_tdp=500.0,
        thermal_profiles=[],
        use_cases=["Next-gen AMD", "2024-2025"],
        factory_function="create_amd_epyc_turin_mapper",
        hardware_type="programmable_isa"
    ))

    # Ampere AmpereOne 192
    mapper = create_ampere_ampereone_192_mapper()
    mappers.append(HardwareMapperInfo(
        name="Ampere AmpereOne 192",
        category="CPU",
        deployment="Datacenter",
        manufacturer="Ampere",
        compute_units=mapper.resource_model.compute_units,
        peak_flops_fp32=mapper.resource_model.get_peak_ops(Precision.FP32) / 1e9,
        peak_flops_int8=mapper.resource_model.get_peak_ops(Precision.INT8) / 1e9,
        memory_bandwidth=mapper.resource_model.peak_bandwidth / 1e9,
        power_tdp=350.0,
        thermal_profiles=[],
        use_cases=["ARM cloud servers", "Energy-efficient"],
        factory_function="create_ampere_ampereone_192_mapper",
        hardware_type="programmable_isa"
    ))

    # Ampere AmpereOne 128
    mapper = create_ampere_ampereone_128_mapper()
    mappers.append(HardwareMapperInfo(
        name="Ampere AmpereOne 128",
        category="CPU",
        deployment="Datacenter",
        manufacturer="Ampere",
        compute_units=mapper.resource_model.compute_units,
        peak_flops_fp32=mapper.resource_model.get_peak_ops(Precision.FP32) / 1e9,
        peak_flops_int8=mapper.resource_model.get_peak_ops(Precision.INT8) / 1e9,
        memory_bandwidth=mapper.resource_model.peak_bandwidth / 1e9,
        power_tdp=250.0,
        thermal_profiles=[],
        use_cases=["ARM cloud", "Medium workloads"],
        factory_function="create_ampere_ampereone_128_mapper",
        hardware_type="programmable_isa"
    ))

    # Intel i7-12700K
    mapper = create_i7_12700k_mapper()
    mappers.append(HardwareMapperInfo(
        name="Intel Core i7-12700K",
        category="CPU",
        deployment="Desktop",
        manufacturer="Intel",
        compute_units=mapper.resource_model.compute_units,
        peak_flops_fp32=mapper.resource_model.get_peak_ops(Precision.FP32) / 1e9,
        peak_flops_int8=mapper.resource_model.get_peak_ops(Precision.INT8) / 1e9,
        memory_bandwidth=mapper.resource_model.peak_bandwidth / 1e9,
        power_tdp=125.0,
        thermal_profiles=[],
        use_cases=["Desktop AI", "Gaming + inference"],
        factory_function="create_i7_12700k_mapper",
        hardware_type="programmable_isa"
    ))

    return mappers


def discover_gpu_mappers() -> List[HardwareMapperInfo]:
    """Discover all GPU mappers"""
    from graphs.hardware.mappers.gpu import (
        create_h100_mapper,
        create_jetson_orin_agx_mapper,
        create_jetson_orin_nano_mapper,
        create_jetson_thor_mapper,
        create_arm_mali_g78_mp20_mapper,
    )

    mappers = []

    # NVIDIA H100
    mapper = create_h100_mapper()
    mappers.append(HardwareMapperInfo(
        name="NVIDIA H100 PCIe",
        category="GPU",
        deployment="Datacenter",
        manufacturer="NVIDIA",
        compute_units=mapper.resource_model.compute_units,
        peak_flops_fp32=mapper.resource_model.get_peak_ops(Precision.FP32) / 1e9,
        peak_flops_int8=mapper.resource_model.get_peak_ops(Precision.INT8) / 1e9,
        memory_bandwidth=mapper.resource_model.peak_bandwidth / 1e9,
        power_tdp=350.0,
        thermal_profiles=[],
        use_cases=["LLM inference", "Cloud AI"],
        factory_function="create_h100_mapper",
        hardware_type="programmable_isa"
    ))

    # Jetson Orin AGX
    mapper = create_jetson_orin_agx_mapper()
    mappers.append(HardwareMapperInfo(
        name="NVIDIA Jetson Orin AGX",
        category="GPU",
        deployment="Edge",
        manufacturer="NVIDIA",
        compute_units=mapper.resource_model.compute_units,
        peak_flops_fp32=mapper.resource_model.get_peak_ops(Precision.FP32) / 1e9,
        peak_flops_int8=mapper.resource_model.get_peak_ops(Precision.INT8) / 1e9,
        memory_bandwidth=mapper.resource_model.peak_bandwidth / 1e9,
        power_tdp=60.0,
        thermal_profiles=["15W", "30W", "60W"],
        use_cases=["Autonomous robots", "Edge AI"],
        factory_function="create_jetson_orin_agx_mapper",
        hardware_type="programmable_isa"
    ))

    # Jetson Orin Nano
    mapper = create_jetson_orin_nano_mapper()
    mappers.append(HardwareMapperInfo(
        name="NVIDIA Jetson Orin Nano",
        category="GPU",
        deployment="Edge",
        manufacturer="NVIDIA",
        compute_units=mapper.resource_model.compute_units,
        peak_flops_fp32=mapper.resource_model.get_peak_ops(Precision.FP32) / 1e9,
        peak_flops_int8=mapper.resource_model.get_peak_ops(Precision.INT8) / 1e9,
        memory_bandwidth=mapper.resource_model.peak_bandwidth / 1e9,
        power_tdp=15.0,
        thermal_profiles=["7W", "15W"],
        use_cases=["Edge inference", "IoT"],
        factory_function="create_jetson_orin_nano_mapper",
        hardware_type="programmable_isa"
    ))

    # Jetson Thor
    mapper = create_jetson_thor_mapper()
    mappers.append(HardwareMapperInfo(
        name="NVIDIA Jetson Thor",
        category="GPU",
        deployment="Automotive",
        manufacturer="NVIDIA",
        compute_units=mapper.resource_model.compute_units,
        peak_flops_fp32=mapper.resource_model.get_peak_ops(Precision.FP32) / 1e9,
        peak_flops_int8=mapper.resource_model.get_peak_ops(Precision.INT8) / 1e9,
        memory_bandwidth=mapper.resource_model.peak_bandwidth / 1e9,
        power_tdp=100.0,
        thermal_profiles=["30W", "60W", "100W"],
        use_cases=["Autonomous vehicles", "Next-gen ADAS"],
        factory_function="create_jetson_thor_mapper",
        hardware_type="programmable_isa"
    ))

    # ARM Mali-G78 MP20
    mapper = create_arm_mali_g78_mp20_mapper()
    mappers.append(HardwareMapperInfo(
        name="ARM Mali-G78 MP20",
        category="GPU",
        deployment="Mobile",
        manufacturer="ARM",
        compute_units=mapper.resource_model.compute_units,
        peak_flops_fp32=mapper.resource_model.get_peak_ops(Precision.FP32) / 1e9,
        peak_flops_int8=mapper.resource_model.get_peak_ops(Precision.INT8) / 1e9,
        memory_bandwidth=mapper.resource_model.peak_bandwidth / 1e9,
        power_tdp=5.0,
        thermal_profiles=["5W"],
        use_cases=["Mobile gaming", "On-device AI"],
        factory_function="create_arm_mali_g78_mp20_mapper",
        hardware_type="programmable_isa"
    ))

    return mappers


def discover_dsp_mappers() -> List[HardwareMapperInfo]:
    """Discover all DSP mappers"""
    from graphs.hardware.mappers.dsp import (
        create_qrb5165_mapper,
        create_ti_tda4vm_mapper,
    )

    mappers = []

    # Qualcomm Hexagon 698 (QRB5165)
    mapper = create_qrb5165_mapper()
    mappers.append(HardwareMapperInfo(
        name="Qualcomm Hexagon 698 (QRB5165)",
        category="DSP",
        deployment="Edge",
        manufacturer="Qualcomm",
        compute_units=mapper.resource_model.compute_units,
        peak_flops_fp32=mapper.resource_model.get_peak_ops(Precision.FP32) / 1e9,
        peak_flops_int8=mapper.resource_model.get_peak_ops(Precision.INT8) / 1e9,
        memory_bandwidth=mapper.resource_model.peak_bandwidth / 1e9,
        power_tdp=7.0,
        thermal_profiles=["7W"],
        use_cases=["Drones", "Robotics"],
        factory_function="create_qrb5165_mapper",
        hardware_type="programmable_isa"
    ))

    # TI TDA4VM C7x
    mapper = create_ti_tda4vm_mapper()
    mappers.append(HardwareMapperInfo(
        name="TI TDA4VM C7x DSP",
        category="DSP",
        deployment="Automotive",
        manufacturer="Texas Instruments",
        compute_units=mapper.resource_model.compute_units,
        peak_flops_fp32=mapper.resource_model.get_peak_ops(Precision.FP32) / 1e9,
        peak_flops_int8=mapper.resource_model.get_peak_ops(Precision.INT8) / 1e9,
        memory_bandwidth=mapper.resource_model.peak_bandwidth / 1e9,
        power_tdp=10.0,
        thermal_profiles=["10W"],
        use_cases=["ADAS", "Automotive vision"],
        factory_function="create_ti_tda4vm_mapper",
        hardware_type="programmable_isa"
    ))

    return mappers


def discover_accelerator_mappers() -> List[HardwareMapperInfo]:
    """Discover all accelerator mappers (TPU, KPU, DPU, CGRA, NPU)"""
    from graphs.hardware.mappers.accelerators.tpu import (
        create_tpu_v4_mapper,
        create_coral_edge_tpu_mapper,
    )
    from graphs.hardware.mappers.accelerators.kpu import (
        create_kpu_t64_mapper,
        create_kpu_t256_mapper,
        create_kpu_t768_mapper,
    )
    from graphs.hardware.mappers.accelerators.dpu import create_dpu_vitis_ai_mapper
    from graphs.hardware.mappers.accelerators.cgra import create_plasticine_v2_mapper
    from graphs.hardware.mappers.accelerators.hailo import (
        create_hailo8_mapper,
        create_hailo10h_mapper,
    )

    mappers = []

    # TPU v4
    mapper = create_tpu_v4_mapper()
    mappers.append(HardwareMapperInfo(
        name="Google TPU v4",
        category="TPU",
        deployment="Datacenter",
        manufacturer="Google",
        compute_units=mapper.resource_model.compute_units,
        peak_flops_fp32=mapper.resource_model.get_peak_ops(Precision.FP32) / 1e9,
        peak_flops_int8=mapper.resource_model.get_peak_ops(Precision.INT8) / 1e9,
        memory_bandwidth=mapper.resource_model.peak_bandwidth / 1e9,
        power_tdp=200.0,
        thermal_profiles=[],
        use_cases=["LLM training", "Large-batch inference"],
        factory_function="create_tpu_v4_mapper",
        hardware_type="accelerator"
    ))

    # Coral Edge TPU
    mapper = create_coral_edge_tpu_mapper()
    mappers.append(HardwareMapperInfo(
        name="Google Coral Edge TPU",
        category="TPU",
        deployment="Edge",
        manufacturer="Google",
        compute_units=mapper.resource_model.compute_units,
        peak_flops_fp32=0.0,  # INT8 only
        peak_flops_int8=mapper.resource_model.get_peak_ops(Precision.INT8) / 1e9,
        memory_bandwidth=mapper.resource_model.peak_bandwidth / 1e9,
        power_tdp=2.0,
        thermal_profiles=[],
        use_cases=["Ultra-low-power edge", "IoT"],
        factory_function="create_coral_edge_tpu_mapper",
        hardware_type="accelerator"
    ))

    # Stillwater KPU T64
    mapper = create_kpu_t64_mapper()
    mappers.append(HardwareMapperInfo(
        name="Stillwater KPU-T64",
        category="KPU",
        deployment="Embodied AI",
        manufacturer="Stillwater",
        compute_units=mapper.resource_model.compute_units,
        peak_flops_fp32=0.0,  # INT8 primary
        peak_flops_int8=mapper.resource_model.get_peak_ops(Precision.INT8) / 1e9,
        memory_bandwidth=mapper.resource_model.peak_bandwidth / 1e9,
        power_tdp=6.0,
        thermal_profiles=["3W", "6W", "10W"],
        use_cases=["Embodied AI", "Robotics", "Drones", "Edge devices"],
        factory_function="create_kpu_t64_mapper",
        hardware_type="accelerator"
    ))

    # Stillwater KPU T256
    mapper = create_kpu_t256_mapper()
    mappers.append(HardwareMapperInfo(
        name="Stillwater KPU-T256",
        category="KPU",
        deployment="Embodied AI",
        manufacturer="Stillwater",
        compute_units=mapper.resource_model.compute_units,
        peak_flops_fp32=0.0,  # INT8 primary
        peak_flops_int8=mapper.resource_model.get_peak_ops(Precision.INT8) / 1e9,
        memory_bandwidth=mapper.resource_model.peak_bandwidth / 1e9,
        power_tdp=30.0,
        thermal_profiles=["15W", "30W", "50W"],
        use_cases=["Embodied AI", "High-performance edge", "Autonomous vehicles"],
        factory_function="create_kpu_t256_mapper",
        hardware_type="accelerator"
    ))

    # Stillwater KPU T768
    mapper = create_kpu_t768_mapper()
    mappers.append(HardwareMapperInfo(
        name="Stillwater KPU-T768",
        category="KPU",
        deployment="Embodied AI",
        manufacturer="Stillwater",
        compute_units=mapper.resource_model.compute_units,
        peak_flops_fp32=0.0,  # INT8 primary
        peak_flops_int8=mapper.resource_model.get_peak_ops(Precision.INT8) / 1e9,
        memory_bandwidth=mapper.resource_model.peak_bandwidth / 1e9,
        power_tdp=60.0,
        thermal_profiles=["30W", "60W", "100W"],
        use_cases=["Embodied AI", "Datacenter inference", "LLM serving"],
        factory_function="create_kpu_t768_mapper",
        hardware_type="accelerator"
    ))

    # Xilinx Vitis AI DPU
    mapper = create_dpu_vitis_ai_mapper()
    mappers.append(HardwareMapperInfo(
        name="Xilinx Vitis AI DPU",
        category="DPU",
        deployment="Edge/Datacenter",
        manufacturer="AMD/Xilinx",
        compute_units=mapper.resource_model.compute_units,
        peak_flops_fp32=0.0,  # INT8 only
        peak_flops_int8=mapper.resource_model.get_peak_ops(Precision.INT8) / 1e9,
        memory_bandwidth=mapper.resource_model.peak_bandwidth / 1e9,
        power_tdp=50.0,
        thermal_profiles=[],
        use_cases=["FPGA inference", "Reconfigurable AI"],
        factory_function="create_dpu_vitis_ai_mapper",
        hardware_type="accelerator"
    ))

    # Plasticine v2 CGRA
    mapper = create_plasticine_v2_mapper()
    mappers.append(HardwareMapperInfo(
        name="Plasticine v2 CGRA",
        category="CGRA",
        deployment="Research",
        manufacturer="Stanford",
        compute_units=mapper.resource_model.compute_units,
        peak_flops_fp32=mapper.resource_model.get_peak_ops(Precision.FP32) / 1e9,
        peak_flops_int8=mapper.resource_model.get_peak_ops(Precision.INT8) / 1e9,
        memory_bandwidth=mapper.resource_model.peak_bandwidth / 1e9,
        power_tdp=100.0,
        thermal_profiles=[],
        use_cases=["Spatial dataflow research", "Novel architectures"],
        factory_function="create_plasticine_v2_mapper",
        hardware_type="accelerator"
    ))

    # Hailo-8
    mapper = create_hailo8_mapper()
    mappers.append(HardwareMapperInfo(
        name="Hailo-8",
        category="NPU",
        deployment="Edge",
        manufacturer="Hailo",
        compute_units=mapper.resource_model.compute_units,
        peak_flops_fp32=0.0,  # INT8 only
        peak_flops_int8=mapper.resource_model.get_peak_ops(Precision.INT8) / 1e9,
        memory_bandwidth=mapper.resource_model.peak_bandwidth / 1e9,
        power_tdp=2.5,
        thermal_profiles=[],
        use_cases=["Edge vision", "Smart cameras"],
        factory_function="create_hailo8_mapper",
        hardware_type="accelerator"
    ))

    # Hailo-10H
    mapper = create_hailo10h_mapper()
    mappers.append(HardwareMapperInfo(
        name="Hailo-10H",
        category="NPU",
        deployment="Automotive",
        manufacturer="Hailo",
        compute_units=mapper.resource_model.compute_units,
        peak_flops_fp32=0.0,  # INT8 only
        peak_flops_int8=mapper.resource_model.get_peak_ops(Precision.INT8) / 1e9,
        memory_bandwidth=mapper.resource_model.peak_bandwidth / 1e9,
        power_tdp=25.0,
        thermal_profiles=[],
        use_cases=["Automotive ADAS", "In-cabin monitoring"],
        factory_function="create_hailo10h_mapper",
        hardware_type="accelerator"
    ))

    return mappers


def generate_text_report(all_mappers: List[HardwareMapperInfo], category_filter: Optional[str] = None):
    """Generate text-based report"""

    if category_filter:
        all_mappers = [m for m in all_mappers if m.category.lower() == category_filter.lower()]

    print("=" * 100)
    print("HARDWARE MAPPER DISCOVERY REPORT")
    print("=" * 100)
    print()
    print(f"Total Mappers Found: {len(all_mappers)}")
    print()

    # Group by category
    categories = {}
    for mapper in all_mappers:
        if mapper.category not in categories:
            categories[mapper.category] = []
        categories[mapper.category].append(mapper)

    # Print by category
    for category, mappers in sorted(categories.items()):
        print("=" * 100)
        print(f"CATEGORY: {category} ({len(mappers)} mappers)")
        print("=" * 100)
        print()

        # Group by deployment
        deployments = {}
        for mapper in mappers:
            if mapper.deployment not in deployments:
                deployments[mapper.deployment] = []
            deployments[mapper.deployment].append(mapper)

        for deployment, dep_mappers in sorted(deployments.items()):
            print(f"--- {deployment} ---")
            print()

            for mapper in sorted(dep_mappers, key=lambda x: x.peak_flops_int8, reverse=True):
                print(f"  • {mapper.name}")
                print(f"    Manufacturer: {mapper.manufacturer}")
                print(f"    Compute Units: {mapper.compute_units}")
                print(f"    Peak FP32: {mapper.peak_flops_fp32:.1f} GFLOPS" if mapper.peak_flops_fp32 > 0 else "    Peak FP32: N/A")
                print(f"    Peak INT8: {mapper.peak_flops_int8:.1f} GOPS")
                print(f"    Memory BW: {mapper.memory_bandwidth:.1f} GB/s")
                print(f"    Power (TDP): {mapper.power_tdp:.1f} W")
                if mapper.thermal_profiles:
                    print(f"    Thermal Profiles: {', '.join(mapper.thermal_profiles)}")
                print(f"    Use Cases: {', '.join(mapper.use_cases)}")
                print(f"    Factory: {mapper.factory_function}()")
                print()

        # Category summary
        total_int8_ops = sum(m.peak_flops_int8 for m in mappers)
        avg_power = sum(m.power_tdp for m in mappers) / len(mappers)
        print(f"  Category Summary:")
        print(f"    Total INT8 Ops: {total_int8_ops:.1f} GOPS")
        print(f"    Average Power: {avg_power:.1f} W")
        print()

    # Overall comparison table
    print("=" * 100)
    print("PERFORMANCE COMPARISON (sorted by INT8 TOPS)")
    print("=" * 100)
    print()
    print(f"{'Hardware':<35} {'Category':<8} {'Deployment':<12} {'INT8 TOPS':<12} {'Power (W)':<10} {'Efficiency (TOPS/W)':<15}")
    print("-" * 100)

    for mapper in sorted(all_mappers, key=lambda x: x.peak_flops_int8, reverse=True):
        int8_tops = mapper.peak_flops_int8 / 1000
        efficiency = int8_tops / mapper.power_tdp if mapper.power_tdp > 0 else 0
        print(f"{mapper.name:<35} {mapper.category:<8} {mapper.deployment:<12} {int8_tops:<12.1f} {mapper.power_tdp:<10.1f} {efficiency:<15.2f}")

    print()
    print("=" * 100)
    print(f"Total: {len(all_mappers)} hardware mappers available")
    print("=" * 100)


def generate_json_report(all_mappers: List[HardwareMapperInfo], category_filter: Optional[str] = None):
    """Generate JSON report"""

    if category_filter:
        all_mappers = [m for m in all_mappers if m.category.lower() == category_filter.lower()]

    report = {
        "total_mappers": len(all_mappers),
        "categories": {},
        "mappers": [asdict(m) for m in all_mappers]
    }

    # Group by category
    for mapper in all_mappers:
        if mapper.category not in report["categories"]:
            report["categories"][mapper.category] = {
                "count": 0,
                "total_int8_ops": 0,
                "total_power": 0,
                "mappers": []
            }

        report["categories"][mapper.category]["count"] += 1
        report["categories"][mapper.category]["total_int8_ops"] += mapper.peak_flops_int8
        report["categories"][mapper.category]["total_power"] += mapper.power_tdp
        report["categories"][mapper.category]["mappers"].append(mapper.name)

    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(
        description="Discover and report on all available hardware mappers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all mappers
  python cli/list_hardware_mappers.py

  # List only CPUs
  python cli/list_hardware_mappers.py --category cpu

  # List only accelerators
  python cli/list_hardware_mappers.py --category tpu

  # Generate JSON report
  python cli/list_hardware_mappers.py --format json
        """
    )

    parser.add_argument(
        "--category",
        choices=["cpu", "gpu", "dsp", "tpu", "kpu", "dpu", "cgra", "npu"],
        help="Filter by hardware category"
    )

    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)"
    )

    args = parser.parse_args()

    # Discover all mappers
    all_mappers = []
    all_mappers.extend(discover_cpu_mappers())
    all_mappers.extend(discover_gpu_mappers())
    all_mappers.extend(discover_dsp_mappers())
    all_mappers.extend(discover_accelerator_mappers())

    # Generate report
    if args.format == "json":
        generate_json_report(all_mappers, args.category)
    else:
        generate_text_report(all_mappers, args.category)


if __name__ == "__main__":
    main()
