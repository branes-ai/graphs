"""
A100 Sxm4 80Gb Resource Model hardware resource model.

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


def a100_sxm4_80gb_resource_model() -> HardwareResourceModel:
    """
    NVIDIA A100 SXM4 (80GB) resource model - Ampere generation (2020).

    ARCHITECTURE:
    - Ampere microarchitecture (GA100 die)
    - 108 SMs with 128 CUDA cores each (6,912 CUDA cores total)
    - 4 Tensor Cores per SM (3rd gen, added TF32, BF16, FP64 TC support)
    - Doubled CUDA cores per SM (64→128)

    PERFORMANCE:
    - FP32: 19.5 TFLOPS (CUDA cores)
    - TF32 (Tensor Cores): 156 TFLOPS (new in Ampere)
    - BF16 (Tensor Cores): 312 TFLOPS
    - FP16 (Tensor Cores): 312 TFLOPS
    - INT8 (Tensor Cores): 624 TOPS
    - Boost clock: 1410 MHz

    MEMORY:
    - 80 GB HBM2e
    - 2 TB/s bandwidth (same as H100)

    POWER:
    - 400W TDP

    USE CASE:
    - Training standard (DGX A100, cloud)
    - First with TF32 and BF16 support
    - Strong balance of training and inference
    """
    # ========================================================================
    # CUDA Core Fabric (Standard Cell FP32 ALUs)
    # ========================================================================
    cuda_fabric = ComputeFabric(
        fabric_type="cuda_core",
        circuit_type="standard_cell",
        num_units=108 * 128,          # 108 SMs × 128 CUDA cores/SM = 13,824 cores
        ops_per_unit_per_clock={
            Precision.FP64: 2,         # FMA: 2 ops/clock
            Precision.FP32: 2,         # FMA: 2 ops/clock
        },
        core_frequency_hz=1.41e9,     # 1.41 GHz boost
        process_node_nm=7,             # TSMC 7nm
        energy_per_flop_fp32=get_base_alu_energy(7, 'standard_cell'),  # 1.8 pJ
        energy_scaling={
            Precision.FP64: 2.0,       # Double precision = 2× energy
            Precision.FP32: 1.0,       # Baseline
            Precision.FP16: 0.5,       # Half precision (emulated on CUDA cores)
            Precision.INT8: 0.125,     # INT8 (emulated on CUDA cores)
        }
    )

    # ========================================================================
    # Tensor Core Fabric (15% more efficient for fused MAC+accumulate)
    # ========================================================================
    # 3rd generation Tensor Cores with TF32/BF16/FP64 support
    tensor_fabric = ComputeFabric(
        fabric_type="tensor_core",
        circuit_type="tensor_core",
        num_units=108 * 4,            # 432 Tensor Cores (108 SMs × 4 TCs/SM)
        ops_per_unit_per_clock={
            Precision.FP64: 256,       # 256 FP64 ops/clock/TC (new in Ampere)
            Precision.BF16: 512,       # 512 BF16 ops/clock/TC (3rd gen)
            Precision.FP16: 512,       # 512 FP16 ops/clock/TC
            Precision.INT8: 1024,      # 1024 INT8 ops/clock/TC
        },
        core_frequency_hz=1.41e9,
        process_node_nm=7,
        energy_per_flop_fp32=get_base_alu_energy(7, 'tensor_core'),  # 1.53 pJ (15% better)
        energy_scaling={
            Precision.FP64: 2.0,       # Double precision
            Precision.BF16: 0.5,       # Half precision
            Precision.FP16: 0.5,
            Precision.INT8: 0.125,     # INT8
        }
    )

    # ========================================================================
    # Legacy Precision Profiles (for backward compatibility)
    # ========================================================================
    cuda_fp64_peak = cuda_fabric.get_peak_ops_per_sec(Precision.FP64)
    cuda_fp32_peak = cuda_fabric.get_peak_ops_per_sec(Precision.FP32)
    tensor_fp64_peak = tensor_fabric.get_peak_ops_per_sec(Precision.FP64)
    tensor_bf16_peak = tensor_fabric.get_peak_ops_per_sec(Precision.BF16)
    tensor_fp16_peak = tensor_fabric.get_peak_ops_per_sec(Precision.FP16)
    tensor_int8_peak = tensor_fabric.get_peak_ops_per_sec(Precision.INT8)

    return HardwareResourceModel(
        name="A100-SXM4-80GB",
        hardware_type=HardwareType.GPU,

        # NEW: Compute fabrics
        compute_fabrics=[cuda_fabric, tensor_fabric],

        # Legacy fields (for backward compatibility)
        compute_units=108,  # SMs
        threads_per_unit=2048,
        warps_per_unit=64,
        warp_size=32,

        # Legacy precision profiles (calculated from fabrics)
        precision_profiles={
            Precision.FP64: PrecisionProfile(
                precision=Precision.FP64,
                peak_ops_per_sec=tensor_fp64_peak,  # ~156 TFLOPS (Tensor Cores, new in Ampere!)
                tensor_core_supported=True,  # New in Ampere!
                relative_speedup=8.0,
                bytes_per_element=8,
            ),
            Precision.FP32: PrecisionProfile(
                precision=Precision.FP32,
                peak_ops_per_sec=cuda_fp32_peak,  # ~39 TFLOPS (CUDA cores)
                tensor_core_supported=False,
                relative_speedup=1.0,
                bytes_per_element=4,
            ),
            # TF32: New in Ampere (FP32 range, FP16 precision)
            Precision.BF16: PrecisionProfile(
                precision=Precision.BF16,
                peak_ops_per_sec=tensor_bf16_peak,  # ~312 TFLOPS (Tensor Cores)
                tensor_core_supported=True,
                relative_speedup=8.0,
                bytes_per_element=2,
                accumulator_precision=Precision.FP32,
            ),
            Precision.FP16: PrecisionProfile(
                precision=Precision.FP16,
                peak_ops_per_sec=tensor_fp16_peak,  # ~312 TFLOPS (Tensor Cores)
                tensor_core_supported=True,
                relative_speedup=8.0,
                bytes_per_element=2,
                accumulator_precision=Precision.FP32,
            ),
            Precision.INT8: PrecisionProfile(
                precision=Precision.INT8,
                peak_ops_per_sec=tensor_int8_peak,  # ~624 TOPS (Tensor Cores)
                tensor_core_supported=True,
                relative_speedup=16.0,
                bytes_per_element=1,
                accumulator_precision=Precision.INT32,
            ),
        },
        default_precision=Precision.FP32,

        peak_bandwidth=2e12,  # 2 TB/s HBM2e (same as H100)
        l1_cache_per_unit=192 * 1024,  # 192 KB per SM
        l2_cache_total=40 * 1024 * 1024,  # 40 MB
        main_memory=80 * 1024**3,  # 80 GB HBM2e

        # Legacy energy (use CUDA fabric as baseline)
        energy_per_flop_fp32=cuda_fabric.energy_per_flop_fp32,  # 1.8 pJ
        energy_per_byte=14e-12,  # ~14 pJ/byte (HBM2e)
        energy_scaling={
            Precision.FP64: 2.0,
            Precision.FP32: 1.0,
            Precision.FP16: 0.5,
            Precision.BF16: 0.5,
            Precision.INT8: 0.125,
        },
        min_occupancy=0.25,
        max_concurrent_kernels=128,
        wave_quantization=4,

        # Ampere microarchitecture (doubled CUDA cores!)
        cuda_cores_per_sm=128,          # Doubled from Volta/Turing (64→128)
        ops_per_clock_per_core=2.0,     # FMA
        sm_boost_clock_hz=1410e6,       # 1410 MHz boost
        sm_sustained_clock_hz=1300e6,   # 1300 MHz sustained (~92% of boost)

        # Tensor Core details (3rd generation, added TF32/BF16/FP64)
        tensor_cores_per_sm=4,          # Reduced from 8 (but much more capable)
        tensor_core_ops_per_clock=512,  # 512 ops/clock (doubled throughput)
    )


