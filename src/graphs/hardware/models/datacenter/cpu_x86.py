"""
Cpu X86 Resource Model hardware resource model.

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


def cpu_x86_resource_model() -> HardwareResourceModel:
    """
    Generic high-end x86 CPU resource model (Intel/AMD).

    Key characteristics:
    - Decent FP32/FP64 with AVX-512
    - Limited INT8 acceleration (VNNI on newer CPUs)
    - Much slower than GPUs but flexible
    """
    return HardwareResourceModel(
        name="CPU-x86-16core",
        hardware_type=HardwareType.CPU,
        compute_units=16,  # 16 cores
        threads_per_unit=2,  # SMT/HyperThreading
        warps_per_unit=1,  # No warp concept
        warp_size=1,

        precision_profiles={
            Precision.FP64: PrecisionProfile(
                precision=Precision.FP64,
                peak_ops_per_sec=750e9,  # 750 GFLOPS (half of FP32)
                tensor_core_supported=False,
                relative_speedup=0.5,
                bytes_per_element=8,
            ),
            Precision.FP32: PrecisionProfile(
                precision=Precision.FP32,
                peak_ops_per_sec=1.5e12,  # 1.5 TFLOPS
                tensor_core_supported=False,
                relative_speedup=1.0,
                bytes_per_element=4,
            ),
            Precision.BF16: PrecisionProfile(
                precision=Precision.BF16,
                peak_ops_per_sec=3e12,  # 3 TFLOPS (with AMX on newer CPUs)
                tensor_core_supported=True,  # AMX
                relative_speedup=2.0,
                bytes_per_element=2,
            ),
            Precision.INT8: PrecisionProfile(
                precision=Precision.INT8,
                peak_ops_per_sec=6e12,  # 6 TOPS (with VNNI/AMX)
                tensor_core_supported=True,  # VNNI/AMX
                relative_speedup=4.0,
                bytes_per_element=1,
                accumulator_precision=Precision.INT32,
            ),
        },
        default_precision=Precision.FP32,

        peak_bandwidth=80e9,  # 80 GB/s DDR5
        l1_cache_per_unit=32 * 1024,  # 32 KB per core
        l2_cache_total=16 * 1024 * 1024,  # 16 MB total L2
        main_memory=64 * 1024**3,  # 64 GB DDR5
        energy_per_flop_fp32=1.24e-10,  # ~123.7 pJ/FLOP (10nm Intel, 3.5 GHz, x86 with AVX-512)
        energy_per_byte=20e-12,
        min_occupancy=0.5,
        max_concurrent_kernels=16,  # One per core
        wave_quantization=1,
    )


