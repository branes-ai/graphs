"""
A100 Sxm4 80Gb Resource Model hardware resource model.

Extracted from resource_model.py during refactoring.
"""

from ...resource_model import (
    HardwareResourceModel,
    HardwareType,
    Precision,
    PrecisionProfile,
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
    return HardwareResourceModel(
        name="A100-SXM4-80GB",
        hardware_type=HardwareType.GPU,
        compute_units=108,  # SMs
        threads_per_unit=2048,
        warps_per_unit=64,
        warp_size=32,

        # Precision profiles
        precision_profiles={
            Precision.FP64: PrecisionProfile(
                precision=Precision.FP64,
                peak_ops_per_sec=9.7e12,  # 9.7 TFLOPS
                tensor_core_supported=True,  # New in Ampere!
                relative_speedup=0.5,
                bytes_per_element=8,
            ),
            Precision.FP32: PrecisionProfile(
                precision=Precision.FP32,
                peak_ops_per_sec=19.5e12,  # 19.5 TFLOPS (CUDA cores)
                tensor_core_supported=False,
                relative_speedup=1.0,
                bytes_per_element=4,
            ),
            # TF32: New in Ampere (FP32 range, FP16 precision)
            Precision.BF16: PrecisionProfile(
                precision=Precision.BF16,
                peak_ops_per_sec=312e12,  # 312 TFLOPS (Tensor Cores)
                tensor_core_supported=True,
                relative_speedup=16.0,
                bytes_per_element=2,
                accumulator_precision=Precision.FP32,
            ),
            Precision.FP16: PrecisionProfile(
                precision=Precision.FP16,
                peak_ops_per_sec=312e12,  # 312 TFLOPS (Tensor Cores)
                tensor_core_supported=True,
                relative_speedup=16.0,
                bytes_per_element=2,
                accumulator_precision=Precision.FP32,
            ),
            Precision.INT8: PrecisionProfile(
                precision=Precision.INT8,
                peak_ops_per_sec=624e12,  # 624 TOPS (2× FP16)
                tensor_core_supported=True,
                relative_speedup=32.0,
                bytes_per_element=1,
                accumulator_precision=Precision.INT32,
            ),
        },
        default_precision=Precision.FP32,

        peak_bandwidth=2e12,  # 2 TB/s HBM2e (same as H100)
        l1_cache_per_unit=192 * 1024,  # 192 KB per SM
        l2_cache_total=40 * 1024 * 1024,  # 40 MB
        main_memory=80 * 1024**3,  # 80 GB HBM2e
        energy_per_flop_fp32=0.69e-12,  # ~0.69 pJ/FLOP (400W / 19.5 TFLOPS / efficiency)
        energy_per_byte=14e-12,  # ~14 pJ/byte
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


