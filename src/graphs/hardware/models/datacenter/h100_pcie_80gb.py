"""
NVIDIA H100 PCIe 80GB Resource Model

FORM FACTOR: PCIe (add-in card)
MEMORY: 80 GB HBM2e

Extracted from resource_model.py during refactoring.
"""

from ...resource_model import (
    HardwareResourceModel,
    HardwareType,
    Precision,
    PrecisionProfile,
    ComputeFabric,
    get_base_alu_energy,
    ClockDomain,
    ComputeResource,
    TileSpecialization,
    KPUComputeResource,
    PerformanceCharacteristics,
    ThermalOperatingPoint,
)


def h100_pcie_80gb_resource_model() -> HardwareResourceModel:
    """
    NVIDIA H100 PCIe 80GB resource model.

    FORM FACTOR: PCIe (add-in card)
    MEMORY: 80 GB HBM2e

    Key characteristics:
    - 114 SMs with 4th gen Tensor Cores (PCIe version, slightly cut down)
    - Massive speedup for low-precision (BF16: 12.5×, FP8: 25× vs FP32)
    - 2 TB/s HBM2e bandwidth
    """
    # ========================================================================
    # CUDA Core Fabric (Standard Cell FP32 ALUs)
    # ========================================================================
    cuda_fabric = ComputeFabric(
        fabric_type="cuda_core",
        circuit_type="standard_cell",
        num_units=114 * 128,           # 114 SMs × 128 CUDA cores/SM = 14,592 cores
        ops_per_unit_per_clock={
            Precision.FP64: 2,         # FMA: 2 ops/clock
            Precision.FP32: 2,         # FMA: 2 ops/clock
        },
        core_frequency_hz=1.755e9,     # 1.755 GHz boost (PCIe version)
        process_node_nm=4,             # TSMC 4N (4nm custom for NVIDIA)
        energy_per_flop_fp32=get_base_alu_energy(4, 'standard_cell'),  # 1.3 pJ
        energy_scaling={
            Precision.FP64: 2.0,
            Precision.FP32: 1.0,
            Precision.FP16: 0.5,
            Precision.INT8: 0.125,
        }
    )

    # ========================================================================
    # Tensor Core Fabric (15% more efficient, 4th generation)
    # ========================================================================
    tensor_fabric = ComputeFabric(
        fabric_type="tensor_core",
        circuit_type="tensor_core",
        num_units=114 * 4,             # 456 Tensor Cores (114 SMs × 4 TCs/SM)
        ops_per_unit_per_clock={
            Precision.BF16: 512,       # 512 BF16 ops/clock/TC (4th gen)
            Precision.FP16: 512,       # 512 FP16 ops/clock/TC
            Precision.FP8_E4M3: 1024,  # 1024 FP8 ops/clock/TC (2× FP16, NEW in Hopper)
            Precision.FP8_E5M2: 1024,  # 1024 FP8 ops/clock/TC
            Precision.INT8: 1024,      # 1024 INT8 ops/clock/TC (2× FP16)
        },
        core_frequency_hz=1.755e9,
        process_node_nm=4,             # TSMC 4N (4nm custom for NVIDIA)
        energy_per_flop_fp32=get_base_alu_energy(4, 'tensor_core'),  # 1.11 pJ (15% better)
        energy_scaling={
            Precision.BF16: 0.5,
            Precision.FP16: 0.5,
            Precision.FP8_E4M3: 0.25,
            Precision.FP8_E5M2: 0.25,
            Precision.INT8: 0.125,
        }
    )

    # Calculate peak ops from fabrics
    cuda_fp64_peak = cuda_fabric.get_peak_ops_per_sec(Precision.FP64)
    cuda_fp32_peak = cuda_fabric.get_peak_ops_per_sec(Precision.FP32)
    tensor_bf16_peak = tensor_fabric.get_peak_ops_per_sec(Precision.BF16)
    tensor_fp16_peak = tensor_fabric.get_peak_ops_per_sec(Precision.FP16)
    tensor_fp8_peak = tensor_fabric.get_peak_ops_per_sec(Precision.FP8_E4M3)
    tensor_int8_peak = tensor_fabric.get_peak_ops_per_sec(Precision.INT8)

    # Thermal operating point (datacenter PCIe card)
    thermal_default = ThermalOperatingPoint(
        name="default",
        tdp_watts=350.0,  # H100 PCIe TDP
        cooling_solution="active-air",
        performance_specs={}  # Uses precision_profiles for performance
    )

    return HardwareResourceModel(
        name="H100-PCIe-80GB",
        hardware_type=HardwareType.GPU,

        # NEW: Compute fabrics
        compute_fabrics=[cuda_fabric, tensor_fabric],

        # Legacy fields
        compute_units=114,  # SMs (PCIe version has 114 SMs, not 132)
        threads_per_unit=2048,  # Max threads per SM
        warps_per_unit=64,  # Max warps per SM (2048 / 32)
        warp_size=32,

        # Legacy precision profiles (calculated from fabrics)
        precision_profiles={
            Precision.FP64: PrecisionProfile(
                precision=Precision.FP64,
                peak_ops_per_sec=cuda_fp64_peak,  # ~51 TFLOPS (PCIe)
                tensor_core_supported=False,
                relative_speedup=1.0,
                bytes_per_element=8,
            ),
            Precision.FP32: PrecisionProfile(
                precision=Precision.FP32,
                peak_ops_per_sec=cuda_fp32_peak,  # ~51 TFLOPS (PCIe, without Tensor Cores)
                tensor_core_supported=False,
                relative_speedup=1.0,
                bytes_per_element=4,
            ),
            Precision.BF16: PrecisionProfile(
                precision=Precision.BF16,
                peak_ops_per_sec=tensor_bf16_peak,  # ~650 TFLOPS (PCIe, with Tensor Cores)
                tensor_core_supported=True,
                relative_speedup=12.5,
                bytes_per_element=2,
                accumulator_precision=Precision.FP32,
            ),
            Precision.FP16: PrecisionProfile(
                precision=Precision.FP16,
                peak_ops_per_sec=tensor_fp16_peak,  # ~650 TFLOPS (PCIe, with Tensor Cores)
                tensor_core_supported=True,
                relative_speedup=12.5,
                bytes_per_element=2,
                accumulator_precision=Precision.FP32,
            ),
            Precision.FP8_E4M3: PrecisionProfile(
                precision=Precision.FP8_E4M3,
                peak_ops_per_sec=tensor_fp8_peak,  # ~1.3 PFLOPS (PCIe, with Tensor Cores)
                tensor_core_supported=True,
                relative_speedup=25.0,
                bytes_per_element=1,
                accumulator_precision=Precision.FP32,
            ),
            Precision.FP8_E5M2: PrecisionProfile(
                precision=Precision.FP8_E5M2,
                peak_ops_per_sec=tensor_fp8_peak,  # ~1.3 PFLOPS (PCIe, with Tensor Cores)
                tensor_core_supported=True,
                relative_speedup=25.0,
                bytes_per_element=1,
                accumulator_precision=Precision.FP32,
            ),
            Precision.INT8: PrecisionProfile(
                precision=Precision.INT8,
                peak_ops_per_sec=tensor_int8_peak,  # ~1.3 POPS (PCIe)
                tensor_core_supported=True,
                relative_speedup=25.0,
                bytes_per_element=1,
                accumulator_precision=Precision.INT32,
            ),
        },
        default_precision=Precision.FP32,

        peak_bandwidth=2e12,  # 2 TB/s HBM2e
        l1_cache_per_unit=256 * 1024,  # 256 KB per SM
        l2_cache_total=50 * 1024 * 1024,  # 50 MB
        main_memory=80 * 1024**3,  # 80 GB HBM2e

        # Legacy energy (use CUDA fabric as baseline)
        energy_per_flop_fp32=cuda_fabric.energy_per_flop_fp32,  # 1.5 pJ
        energy_per_byte=15e-12,  # ~15 pJ/byte (HBM2e)
        energy_scaling={
            Precision.FP64: 2.0,
            Precision.FP32: 1.0,
            Precision.BF16: 0.5,
            Precision.FP16: 0.5,
            Precision.FP8_E4M3: 0.25,
            Precision.FP8_E5M2: 0.25,
            Precision.INT8: 0.125,
        },
        min_occupancy=0.25,
        max_concurrent_kernels=128,  # Can run many kernels concurrently
        wave_quantization=4,  # Launch in waves of 4 SMs

        # Hopper microarchitecture (for compute modeling)
        cuda_cores_per_sm=128,          # Doubled from Turing (64→128)
        ops_per_clock_per_core=2.0,     # FMA: 2 ops/clock
        sm_boost_clock_hz=1755e6,       # 1755 MHz boost (PCIe version)
        sm_sustained_clock_hz=1620e6,   # 1620 MHz sustained (~92% of boost)

        # Tensor Core details
        tensor_cores_per_sm=4,          # 4th gen Tensor Cores
        tensor_core_ops_per_clock=512,  # 512 FP16 FMAs per clock per TC

        # Thermal profile
        thermal_operating_points={
            "default": thermal_default,
        },
        default_thermal_profile="default",
    )


