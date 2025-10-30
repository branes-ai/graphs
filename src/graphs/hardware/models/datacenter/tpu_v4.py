"""
Tpu V4 Resource Model hardware resource model.

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


def tpu_v4_resource_model() -> HardwareResourceModel:
    """
    Google TPU v4 resource model.

    Architecture:
    - 2 Matrix Multiplier Units (MXUs)
    - Each MXU: 128×128 systolic array (16,384 MACs)
    - Per MXU: 137.5 TFLOPS BF16, 275 TOPS INT8

    Key characteristics:
    - Optimized for BF16 and INT8
    - 2× INT8 performance vs BF16
    - Very energy efficient
    """
    # Thermal operating point (datacenter TPU pod)
    thermal_default = ThermalOperatingPoint(
        name="default",
        tdp_watts=350.0,  # TPU v4 TDP
        cooling_solution="active-liquid",
        performance_specs={}  # Uses precision_profiles for performance
    )

    return HardwareResourceModel(
        name="TPU-v4",
        hardware_type=HardwareType.TPU,
        compute_units=2,  # 2 MXUs (Matrix Multiplier Units)
        threads_per_unit=128 * 128,  # 128×128 systolic array per MXU
        warps_per_unit=128,  # rows in systolic array
        warp_size=128,  # columns in systolic array

        precision_profiles={
            Precision.FP32: PrecisionProfile(
                precision=Precision.FP32,
                peak_ops_per_sec=137.5e12,  # Half of BF16 (not native)
                tensor_core_supported=True,
                relative_speedup=0.5,
                bytes_per_element=4,
            ),
            Precision.BF16: PrecisionProfile(
                precision=Precision.BF16,
                peak_ops_per_sec=275e12,  # 275 TFLOPS
                tensor_core_supported=True,
                relative_speedup=1.0,
                bytes_per_element=2,
            ),
            Precision.INT8: PrecisionProfile(
                precision=Precision.INT8,
                peak_ops_per_sec=550e12,  # 550 TOPS (2× BF16)
                tensor_core_supported=True,
                relative_speedup=2.0,
                bytes_per_element=1,
                accumulator_precision=Precision.INT32,
            ),
        },
        default_precision=Precision.BF16,

        peak_bandwidth=1.2e12,  # 1.2 TB/s HBM2e
        l1_cache_per_unit=16 * 1024 * 1024,  # 16 MB per MXU
        l2_cache_total=32 * 1024 * 1024,  # 32 MB shared
        main_memory=32 * 1024**3,  # 32 GB HBM2e
        energy_per_flop_fp32=0.4e-12,  # Very efficient (assuming FP32 equiv)
        energy_per_byte=10e-12,
        min_occupancy=0.5,  # Systolic arrays need high utilization
        max_concurrent_kernels=1,  # Typically runs one large batch
        wave_quantization=1,

        # Thermal profile
        thermal_operating_points={
            "default": thermal_default,
        },
        default_thermal_profile="default",
    )


