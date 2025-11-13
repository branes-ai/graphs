"""
NVIDIA V100 SXM3 32GB Resource Model

FORM FACTOR: SXM3 (Server Module)
MEMORY: 32 GB HBM2

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


def v100_sxm3_32gb_resource_model() -> HardwareResourceModel:
    """
    NVIDIA V100 SXM3 32GB resource model - Volta generation (2017).

    FORM FACTOR: SXM3 (Server Module)
    MEMORY: 32 GB HBM2

    ARCHITECTURE:
    - First generation with Tensor Cores
    - 80 SMs with 64 CUDA cores each (5,120 CUDA cores total)
    - 8 Tensor Cores per SM (640 total)
    - Volta microarchitecture

    PERFORMANCE:
    - FP32: 15.7 TFLOPS (CUDA cores)
    - FP16: 31.4 TFLOPS (CUDA cores, 2x FP32)
    - FP16 (Tensor Cores): 125 TFLOPS (8x CUDA cores)
    - Boost clock: 1530 MHz

    MEMORY:
    - 32 GB HBM2
    - 900 GB/s bandwidth

    POWER:
    - 300W TDP

    USE CASE:
    - Training pioneer (first with Tensor Cores)
    - DGX-1 V100, Cloud instances (AWS P3, GCP)
    """
    # ========================================================================
    # CUDA Core Fabric (Standard Cell FP32 ALUs)
    # ========================================================================
    cuda_fabric = ComputeFabric(
        fabric_type="cuda_core",
        circuit_type="standard_cell",
        num_units=80 * 64,            # 80 SMs × 64 CUDA cores/SM = 5,120 cores
        ops_per_unit_per_clock={
            Precision.FP64: 2,         # FMA: 2 ops/clock
            Precision.FP32: 2,         # FMA: 2 ops/clock
        },
        core_frequency_hz=1.53e9,     # 1.53 GHz boost
        process_node_nm=12,            # TSMC 12nm FFN
        energy_per_flop_fp32=get_base_alu_energy(12, 'standard_cell'),  # 2.5 pJ
        energy_scaling={
            Precision.FP64: 2.0,
            Precision.FP32: 1.0,
            Precision.FP16: 0.5,
            Precision.INT8: 0.125,
        }
    )

    # ========================================================================
    # Tensor Core Fabric (15% more efficient, 1st generation)
    # ========================================================================
    tensor_fabric = ComputeFabric(
        fabric_type="tensor_core",
        circuit_type="tensor_core",
        num_units=80 * 8,             # 640 Tensor Cores (80 SMs × 8 TCs/SM)
        ops_per_unit_per_clock={
            Precision.FP16: 256,       # 256 FP16 ops/clock/TC (1st gen)
        },
        core_frequency_hz=1.53e9,
        process_node_nm=12,
        energy_per_flop_fp32=get_base_alu_energy(12, 'tensor_core'),  # 2.125 pJ (15% better)
        energy_scaling={
            Precision.FP16: 0.5,
        }
    )

    # Calculate peak ops from fabrics
    cuda_fp64_peak = cuda_fabric.get_peak_ops_per_sec(Precision.FP64)
    cuda_fp32_peak = cuda_fabric.get_peak_ops_per_sec(Precision.FP32)
    tensor_fp16_peak = tensor_fabric.get_peak_ops_per_sec(Precision.FP16)

    return HardwareResourceModel(
        name="V100-SXM3-32GB",
        hardware_type=HardwareType.GPU,

        # NEW: Compute fabrics
        compute_fabrics=[cuda_fabric, tensor_fabric],

        # Legacy fields
        compute_units=80,  # SMs
        threads_per_unit=2048,
        warps_per_unit=64,
        warp_size=32,

        # Legacy precision profiles (calculated from fabrics)
        precision_profiles={
            Precision.FP64: PrecisionProfile(
                precision=Precision.FP64,
                peak_ops_per_sec=cuda_fp64_peak,  # ~15.6 TFLOPS
                tensor_core_supported=False,
                relative_speedup=0.5,
                bytes_per_element=8,
            ),
            Precision.FP32: PrecisionProfile(
                precision=Precision.FP32,
                peak_ops_per_sec=cuda_fp32_peak,  # ~15.6 TFLOPS (CUDA cores)
                tensor_core_supported=False,
                relative_speedup=1.0,
                bytes_per_element=4,
            ),
            Precision.FP16: PrecisionProfile(
                precision=Precision.FP16,
                peak_ops_per_sec=tensor_fp16_peak,  # ~125 TFLOPS (Tensor Cores)
                tensor_core_supported=True,
                relative_speedup=8.0,
                bytes_per_element=2,
                accumulator_precision=Precision.FP32,
            ),
            Precision.INT8: PrecisionProfile(
                precision=Precision.INT8,
                peak_ops_per_sec=tensor_fp16_peak,  # Same as FP16 (1st gen TCs)
                tensor_core_supported=True,
                relative_speedup=8.0,
                bytes_per_element=1,
                accumulator_precision=Precision.INT32,
            ),
        },
        default_precision=Precision.FP32,

        peak_bandwidth=900e9,  # 900 GB/s HBM2
        l1_cache_per_unit=128 * 1024,  # 128 KB per SM
        l2_cache_total=6 * 1024 * 1024,  # 6 MB
        main_memory=32 * 1024**3,  # 32 GB HBM2

        # Legacy energy (use CUDA fabric as baseline)
        energy_per_flop_fp32=cuda_fabric.energy_per_flop_fp32,  # 2.5 pJ
        energy_per_byte=12e-12,  # ~12 pJ/byte (HBM2)
        energy_scaling={
            Precision.FP64: 2.0,
            Precision.FP32: 1.0,
            Precision.FP16: 0.5,
            Precision.INT8: 0.125,
        },
        min_occupancy=0.25,
        max_concurrent_kernels=32,
        wave_quantization=4,

        # Volta microarchitecture
        cuda_cores_per_sm=64,           # First with 64 cores/SM
        ops_per_clock_per_core=2.0,     # FMA
        sm_boost_clock_hz=1530e6,       # 1530 MHz boost
        sm_sustained_clock_hz=1400e6,   # 1400 MHz sustained (~91% of boost)

        # Tensor Core details (1st generation)
        tensor_cores_per_sm=8,          # 8 TCs per SM (first generation)
        tensor_core_ops_per_clock=256,  # 256 FP16 FMAs per clock per TC
    )


