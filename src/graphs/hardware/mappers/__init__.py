"""
Hardware-Specific Mappers

Maps computational graphs to specific hardware architectures (CPUs, GPUs, DSPs, accelerators).

This module provides a registry of all available hardware mappers and functions
to discover and instantiate them.
"""

from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass

# Lazy import to avoid circular dependencies
_MAPPER_REGISTRY: Dict[str, Dict[str, Any]] = {}
_REGISTRY_INITIALIZED = False


@dataclass
class MapperInfo:
    """Information about a hardware mapper."""
    name: str
    factory_func: Callable
    category: str  # gpu, cpu, tpu, dsp, kpu, dpu, cgra
    vendor: str
    description: str
    default_tdp_w: float  # Default TDP in watts
    memory_gb: float  # Memory in GB


def _init_registry():
    """Initialize the mapper registry lazily."""
    global _MAPPER_REGISTRY, _REGISTRY_INITIALIZED
    if _REGISTRY_INITIALIZED:
        return

    # GPU Mappers
    from .gpu import (
        create_h100_sxm5_80gb_mapper,
        create_h100_pcie_80gb_mapper,
        create_b100_sxm6_192gb_mapper,
        create_a100_sxm4_80gb_mapper,
        create_v100_sxm3_32gb_mapper,
        create_t4_pcie_16gb_mapper,
        create_jetson_orin_agx_64gb_mapper,
        create_jetson_orin_nano_8gb_mapper,
        create_jetson_orin_nx_16gb_mapper,
        create_jetson_thor_128gb_mapper,
        create_arm_mali_g78_mp20_mapper,
    )

    # CPU Mappers
    from .cpu import (
        create_intel_xeon_platinum_8490h_mapper,
        create_intel_xeon_platinum_8592plus_mapper,
        create_intel_granite_rapids_mapper,
        create_amd_epyc_9654_mapper,
        create_amd_epyc_9754_mapper,
        create_amd_epyc_turin_mapper,
        create_ampere_ampereone_192_mapper,
        create_ampere_ampereone_128_mapper,
        create_i7_12700k_mapper,
        create_jetson_orin_agx_cpu_mapper,
    )

    # DSP Mappers
    from .dsp import (
        create_qrb5165_mapper,
        create_ti_tda4vm_mapper,
        create_ti_tda4vl_mapper,
        create_ti_tda4al_mapper,
        create_ti_tda4vh_mapper,
        create_ceva_neupro_npm11_mapper,
        create_cadence_vision_q8_mapper,
        create_synopsys_arc_ev7x_mapper,
        create_qualcomm_sa8775p_mapper,
        create_qualcomm_snapdragon_ride_mapper,
    )

    # Accelerator Mappers
    from .accelerators.tpu import (
        create_tpu_v1_mapper,
        create_tpu_v3_mapper,
        create_tpu_v4_mapper,
        create_tpu_v5p_mapper,
        create_coral_edge_tpu_mapper,
        create_tpu_edge_pro_mapper,
    )
    from .accelerators.kpu import (
        create_kpu_t64_mapper,
        create_kpu_t256_mapper,
        create_kpu_t768_mapper,
    )
    from .accelerators.hailo import (
        create_hailo8_mapper,
        create_hailo10h_mapper,
    )
    from .accelerators.dpu import create_dpu_vitis_ai_mapper
    from .accelerators.cgra import create_plasticine_v2_mapper

    # Research Mappers
    from .research.dfm import create_dfm_128_mapper

    _MAPPER_REGISTRY = {
        # =====================================================================
        # Datacenter GPUs (100-700W)
        # =====================================================================
        "B100-SXM6-192GB": {
            "factory": create_b100_sxm6_192gb_mapper,
            "category": "gpu",
            "vendor": "NVIDIA",
            "description": "NVIDIA B100 Blackwell datacenter GPU",
            "default_tdp_w": 700.0,
            "memory_gb": 192.0,
        },
        "H100-SXM5-80GB": {
            "factory": create_h100_sxm5_80gb_mapper,
            "category": "gpu",
            "vendor": "NVIDIA",
            "description": "NVIDIA H100 Hopper datacenter GPU (SXM5)",
            "default_tdp_w": 700.0,
            "memory_gb": 80.0,
        },
        "H100-PCIe-80GB": {
            "factory": create_h100_pcie_80gb_mapper,
            "category": "gpu",
            "vendor": "NVIDIA",
            "description": "NVIDIA H100 Hopper datacenter GPU (PCIe)",
            "default_tdp_w": 350.0,
            "memory_gb": 80.0,
        },
        "A100-SXM4-80GB": {
            "factory": create_a100_sxm4_80gb_mapper,
            "category": "gpu",
            "vendor": "NVIDIA",
            "description": "NVIDIA A100 Ampere datacenter GPU",
            "default_tdp_w": 400.0,
            "memory_gb": 80.0,
        },
        "V100-SXM3-32GB": {
            "factory": create_v100_sxm3_32gb_mapper,
            "category": "gpu",
            "vendor": "NVIDIA",
            "description": "NVIDIA V100 Volta datacenter GPU",
            "default_tdp_w": 300.0,
            "memory_gb": 32.0,
        },
        "T4-PCIe-16GB": {
            "factory": create_t4_pcie_16gb_mapper,
            "category": "gpu",
            "vendor": "NVIDIA",
            "description": "NVIDIA T4 inference GPU",
            "default_tdp_w": 70.0,
            "memory_gb": 16.0,
        },

        # =====================================================================
        # Edge GPUs (7-60W)
        # =====================================================================
        "Jetson-Orin-AGX-64GB": {
            "factory": create_jetson_orin_agx_64gb_mapper,
            "category": "gpu",
            "vendor": "NVIDIA",
            "description": "NVIDIA Jetson Orin AGX 64GB edge AI module",
            "default_tdp_w": 60.0,
            "memory_gb": 64.0,
        },
        "Jetson-Orin-NX-16GB": {
            "factory": create_jetson_orin_nx_16gb_mapper,
            "category": "gpu",
            "vendor": "NVIDIA",
            "description": "NVIDIA Jetson Orin NX 16GB edge AI module",
            "default_tdp_w": 25.0,
            "memory_gb": 16.0,
        },
        "Jetson-Orin-Nano-8GB": {
            "factory": create_jetson_orin_nano_8gb_mapper,
            "category": "gpu",
            "vendor": "NVIDIA",
            "description": "NVIDIA Jetson Orin Nano 8GB edge AI module",
            "default_tdp_w": 15.0,
            "memory_gb": 8.0,
        },
        "Jetson-Thor-128GB": {
            "factory": create_jetson_thor_128gb_mapper,
            "category": "gpu",
            "vendor": "NVIDIA",
            "description": "NVIDIA Jetson Thor automotive AI computer",
            "default_tdp_w": 100.0,
            "memory_gb": 128.0,
        },

        # =====================================================================
        # Mobile GPUs (2-10W)
        # =====================================================================
        "ARM-Mali-G78-MP20": {
            "factory": create_arm_mali_g78_mp20_mapper,
            "category": "gpu",
            "vendor": "ARM",
            "description": "ARM Mali-G78 mobile GPU (20 cores)",
            "default_tdp_w": 5.0,
            "memory_gb": 8.0,  # Shared system memory
        },

        # =====================================================================
        # Datacenter CPUs (200-400W)
        # =====================================================================
        "Intel-Xeon-Platinum-8490H": {
            "factory": create_intel_xeon_platinum_8490h_mapper,
            "category": "cpu",
            "vendor": "Intel",
            "description": "Intel Xeon Platinum 8490H (60 cores)",
            "default_tdp_w": 350.0,
            "memory_gb": 256.0,
        },
        "Intel-Xeon-Platinum-8592+": {
            "factory": create_intel_xeon_platinum_8592plus_mapper,
            "category": "cpu",
            "vendor": "Intel",
            "description": "Intel Xeon Platinum 8592+ (64 cores)",
            "default_tdp_w": 350.0,
            "memory_gb": 256.0,
        },
        "Intel-Granite-Rapids": {
            "factory": create_intel_granite_rapids_mapper,
            "category": "cpu",
            "vendor": "Intel",
            "description": "Intel Xeon Granite Rapids (next-gen)",
            "default_tdp_w": 350.0,
            "memory_gb": 512.0,
        },
        "AMD-EPYC-9654": {
            "factory": create_amd_epyc_9654_mapper,
            "category": "cpu",
            "vendor": "AMD",
            "description": "AMD EPYC 9654 Genoa (96 cores)",
            "default_tdp_w": 360.0,
            "memory_gb": 384.0,
        },
        "AMD-EPYC-9754": {
            "factory": create_amd_epyc_9754_mapper,
            "category": "cpu",
            "vendor": "AMD",
            "description": "AMD EPYC 9754 Bergamo (128 cores)",
            "default_tdp_w": 360.0,
            "memory_gb": 384.0,
        },
        "AMD-EPYC-Turin": {
            "factory": create_amd_epyc_turin_mapper,
            "category": "cpu",
            "vendor": "AMD",
            "description": "AMD EPYC Turin (next-gen)",
            "default_tdp_w": 400.0,
            "memory_gb": 512.0,
        },
        "Ampere-AmpereOne-192": {
            "factory": create_ampere_ampereone_192_mapper,
            "category": "cpu",
            "vendor": "Ampere",
            "description": "Ampere AmpereOne (192 cores)",
            "default_tdp_w": 350.0,
            "memory_gb": 512.0,
        },
        "Ampere-AmpereOne-128": {
            "factory": create_ampere_ampereone_128_mapper,
            "category": "cpu",
            "vendor": "Ampere",
            "description": "Ampere AmpereOne (128 cores)",
            "default_tdp_w": 250.0,
            "memory_gb": 512.0,
        },

        # =====================================================================
        # Desktop/Edge CPUs (15-125W)
        # =====================================================================
        "Intel-i7-12700K": {
            "factory": create_i7_12700k_mapper,
            "category": "cpu",
            "vendor": "Intel",
            "description": "Intel Core i7-12700K (12 cores)",
            "default_tdp_w": 125.0,
            "memory_gb": 128.0,
        },
        "Jetson-Orin-AGX-CPU": {
            "factory": create_jetson_orin_agx_cpu_mapper,
            "category": "cpu",
            "vendor": "NVIDIA",
            "description": "Jetson Orin AGX ARM CPU (12 cores)",
            "default_tdp_w": 15.0,
            "memory_gb": 64.0,
        },

        # =====================================================================
        # DSP/Vision Processors (5-100W)
        # =====================================================================
        "Qualcomm-QRB5165": {
            "factory": create_qrb5165_mapper,
            "category": "dsp",
            "vendor": "Qualcomm",
            "description": "Qualcomm QRB5165 Robotics Platform",
            "default_tdp_w": 15.0,
            "memory_gb": 8.0,
        },
        "TI-TDA4VM": {
            "factory": create_ti_tda4vm_mapper,
            "category": "dsp",
            "vendor": "Texas Instruments",
            "description": "TI TDA4VM automotive vision processor",
            "default_tdp_w": 10.0,
            "memory_gb": 8.0,
        },
        "TI-TDA4VL": {
            "factory": create_ti_tda4vl_mapper,
            "category": "dsp",
            "vendor": "Texas Instruments",
            "description": "TI TDA4VL low-power vision processor",
            "default_tdp_w": 7.0,
            "memory_gb": 4.0,
        },
        "TI-TDA4AL": {
            "factory": create_ti_tda4al_mapper,
            "category": "dsp",
            "vendor": "Texas Instruments",
            "description": "TI TDA4AL automotive-lite processor",
            "default_tdp_w": 10.0,
            "memory_gb": 4.0,
        },
        "TI-TDA4VH": {
            "factory": create_ti_tda4vh_mapper,
            "category": "dsp",
            "vendor": "Texas Instruments",
            "description": "TI TDA4VH high-performance vision processor",
            "default_tdp_w": 20.0,
            "memory_gb": 8.0,
        },
        "CEVA-NeuPro-NPM11": {
            "factory": create_ceva_neupro_npm11_mapper,
            "category": "dsp",
            "vendor": "CEVA",
            "description": "CEVA NeuPro NPM11 AI processor",
            "default_tdp_w": 2.0,
            "memory_gb": 1.0,
        },
        "Cadence-Vision-Q8": {
            "factory": create_cadence_vision_q8_mapper,
            "category": "dsp",
            "vendor": "Cadence",
            "description": "Cadence Tensilica Vision Q8 DSP",
            "default_tdp_w": 1.0,
            "memory_gb": 0.5,
        },
        "Synopsys-ARC-EV7x": {
            "factory": create_synopsys_arc_ev7x_mapper,
            "category": "dsp",
            "vendor": "Synopsys",
            "description": "Synopsys ARC EV7x embedded vision processor",
            "default_tdp_w": 0.5,
            "memory_gb": 0.25,
        },
        "Qualcomm-SA8775P": {
            "factory": create_qualcomm_sa8775p_mapper,
            "category": "dsp",
            "vendor": "Qualcomm",
            "description": "Qualcomm SA8775P automotive platform",
            "default_tdp_w": 30.0,
            "memory_gb": 16.0,
        },
        "Qualcomm-Snapdragon-Ride": {
            "factory": create_qualcomm_snapdragon_ride_mapper,
            "category": "dsp",
            "vendor": "Qualcomm",
            "description": "Qualcomm Snapdragon Ride autonomous driving",
            "default_tdp_w": 100.0,
            "memory_gb": 32.0,
        },

        # =====================================================================
        # TPU Accelerators (2-500W)
        # =====================================================================
        "Google-TPU-v1": {
            "factory": create_tpu_v1_mapper,
            "category": "tpu",
            "vendor": "Google",
            "description": "Google TPU v1 (original inference)",
            "default_tdp_w": 75.0,
            "memory_gb": 8.0,
        },
        "Google-TPU-v3": {
            "factory": create_tpu_v3_mapper,
            "category": "tpu",
            "vendor": "Google",
            "description": "Google TPU v3 (training/inference)",
            "default_tdp_w": 200.0,
            "memory_gb": 16.0,
        },
        "Google-TPU-v4": {
            "factory": create_tpu_v4_mapper,
            "category": "tpu",
            "vendor": "Google",
            "description": "Google TPU v4 (datacenter)",
            "default_tdp_w": 300.0,
            "memory_gb": 32.0,
        },
        "Google-TPU-v5p": {
            "factory": create_tpu_v5p_mapper,
            "category": "tpu",
            "vendor": "Google",
            "description": "Google TPU v5p (latest datacenter)",
            "default_tdp_w": 500.0,
            "memory_gb": 96.0,
        },
        "Google-Coral-Edge-TPU": {
            "factory": create_coral_edge_tpu_mapper,
            "category": "tpu",
            "vendor": "Google",
            "description": "Google Coral Edge TPU (USB/M.2)",
            "default_tdp_w": 2.0,
            "memory_gb": 0.008,  # 8 MB SRAM
        },
        "Google-TPU-Edge-Pro": {
            "factory": create_tpu_edge_pro_mapper,
            "category": "tpu",
            "vendor": "Google",
            "description": "Google TPU Edge Pro (enhanced edge)",
            "default_tdp_w": 4.0,
            "memory_gb": 0.016,  # 16 MB SRAM
        },

        # =====================================================================
        # KPU Accelerators (1-20W)
        # =====================================================================
        "Stillwater-KPU-T64": {
            "factory": create_kpu_t64_mapper,
            "category": "kpu",
            "vendor": "Stillwater",
            "description": "Stillwater KPU T64 (64 tiles)",
            "default_tdp_w": 5.0,
            "memory_gb": 4.0,
        },
        "Stillwater-KPU-T256": {
            "factory": create_kpu_t256_mapper,
            "category": "kpu",
            "vendor": "Stillwater",
            "description": "Stillwater KPU T256 (256 tiles)",
            "default_tdp_w": 10.0,
            "memory_gb": 8.0,
        },
        "Stillwater-KPU-T768": {
            "factory": create_kpu_t768_mapper,
            "category": "kpu",
            "vendor": "Stillwater",
            "description": "Stillwater KPU T768 (768 tiles)",
            "default_tdp_w": 20.0,
            "memory_gb": 16.0,
        },

        # =====================================================================
        # Hailo Accelerators (2-15W)
        # =====================================================================
        "Hailo-8": {
            "factory": create_hailo8_mapper,
            "category": "accelerator",
            "vendor": "Hailo",
            "description": "Hailo-8 edge AI accelerator (26 TOPS)",
            "default_tdp_w": 2.5,
            "memory_gb": 0.004,  # 4 MB SRAM
        },
        "Hailo-10H": {
            "factory": create_hailo10h_mapper,
            "category": "accelerator",
            "vendor": "Hailo",
            "description": "Hailo-10H enhanced edge accelerator (40 TOPS)",
            "default_tdp_w": 4.0,
            "memory_gb": 0.008,  # 8 MB SRAM
        },

        # =====================================================================
        # FPGA/DPU Accelerators (10-75W)
        # =====================================================================
        "Xilinx-Vitis-AI-DPU": {
            "factory": create_dpu_vitis_ai_mapper,
            "category": "dpu",
            "vendor": "AMD/Xilinx",
            "description": "Xilinx Vitis AI DPU (FPGA-based)",
            "default_tdp_w": 75.0,
            "memory_gb": 8.0,
        },

        # =====================================================================
        # CGRA Accelerators (10-50W)
        # =====================================================================
        "Stanford-Plasticine-v2": {
            "factory": create_plasticine_v2_mapper,
            "category": "cgra",
            "vendor": "Stanford",
            "description": "Stanford Plasticine v2 CGRA (research)",
            "default_tdp_w": 30.0,
            "memory_gb": 4.0,
        },

        # =====================================================================
        # Research/Experimental (5-20W)
        # =====================================================================
        "Stillwater-DFM-128": {
            "factory": create_dfm_128_mapper,
            "category": "dfm",
            "vendor": "Stillwater",
            "description": "Stillwater DFM-128 dataflow machine",
            "default_tdp_w": 15.0,
            "memory_gb": 8.0,
        },
    }

    _REGISTRY_INITIALIZED = True


def list_all_mappers() -> List[str]:
    """Return list of all available mapper names."""
    _init_registry()
    return list(_MAPPER_REGISTRY.keys())


def get_mapper_by_name(name: str, thermal_profile: str = None):
    """
    Get a hardware mapper by name.

    Args:
        name: Mapper name (e.g., "Jetson-Orin-Nano-8GB")
        thermal_profile: Optional thermal profile override

    Returns:
        HardwareMapper instance or None if not found
    """
    _init_registry()
    info = _MAPPER_REGISTRY.get(name)
    if info is None:
        return None

    factory = info["factory"]
    try:
        if thermal_profile:
            return factory(thermal_profile=thermal_profile)
        return factory()
    except TypeError:
        # Factory doesn't accept thermal_profile
        return factory()


def get_mapper_info(name: str) -> Optional[Dict[str, Any]]:
    """
    Get metadata about a mapper without instantiating it.

    Args:
        name: Mapper name

    Returns:
        Dict with category, vendor, description, default_tdp_w, memory_gb
    """
    _init_registry()
    info = _MAPPER_REGISTRY.get(name)
    if info is None:
        return None

    return {
        "name": name,
        "category": info["category"],
        "vendor": info["vendor"],
        "description": info["description"],
        "default_tdp_w": info["default_tdp_w"],
        "memory_gb": info["memory_gb"],
    }


def list_mappers_by_category(category: str) -> List[str]:
    """List all mappers of a given category (gpu, cpu, tpu, dsp, kpu, etc.)."""
    _init_registry()
    return [
        name for name, info in _MAPPER_REGISTRY.items()
        if info["category"] == category
    ]


def list_mappers_by_vendor(vendor: str) -> List[str]:
    """List all mappers from a given vendor."""
    _init_registry()
    vendor_lower = vendor.lower()
    return [
        name for name, info in _MAPPER_REGISTRY.items()
        if vendor_lower in info["vendor"].lower()
    ]


def list_mappers_by_tdp_range(min_tdp_w: float, max_tdp_w: float) -> List[str]:
    """List all mappers within a TDP range."""
    _init_registry()
    return [
        name for name, info in _MAPPER_REGISTRY.items()
        if min_tdp_w <= info["default_tdp_w"] <= max_tdp_w
    ]


__all__ = [
    "list_all_mappers",
    "get_mapper_by_name",
    "get_mapper_info",
    "list_mappers_by_category",
    "list_mappers_by_vendor",
    "list_mappers_by_tdp_range",
]
