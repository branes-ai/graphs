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
    - 132 SMs with 4th gen Tensor Cores
    - Massive speedup for low-precision (BF16: 12.5×, FP8: 25× vs FP32)
    - 2 TB/s HBM2e bandwidth
    """
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
        compute_units=132,  # SMs (Streaming Multiprocessors)
        threads_per_unit=2048,  # Max threads per SM
        warps_per_unit=64,  # Max warps per SM (2048 / 32)
        warp_size=32,

        # Precision profiles (NVIDIA H100 specifications)
        precision_profiles={
            Precision.FP64: PrecisionProfile(
                precision=Precision.FP64,
                peak_ops_per_sec=60e12,  # 60 TFLOPS
                tensor_core_supported=False,
                relative_speedup=1.0,
                bytes_per_element=8,
            ),
            Precision.FP32: PrecisionProfile(
                precision=Precision.FP32,
                peak_ops_per_sec=60e12,  # 60 TFLOPS (without Tensor Cores)
                tensor_core_supported=False,
                relative_speedup=1.0,
                bytes_per_element=4,
            ),
            Precision.BF16: PrecisionProfile(
                precision=Precision.BF16,
                peak_ops_per_sec=750e12,  # 750 TFLOPS (with Tensor Cores)
                tensor_core_supported=True,
                relative_speedup=12.5,
                bytes_per_element=2,
                accumulator_precision=Precision.FP32,
            ),
            Precision.FP16: PrecisionProfile(
                precision=Precision.FP16,
                peak_ops_per_sec=750e12,  # 750 TFLOPS (with Tensor Cores)
                tensor_core_supported=True,
                relative_speedup=12.5,
                bytes_per_element=2,
                accumulator_precision=Precision.FP32,
            ),
            Precision.FP8_E4M3: PrecisionProfile(
                precision=Precision.FP8_E4M3,
                peak_ops_per_sec=1500e12,  # 1.5 PFLOPS (with Tensor Cores)
                tensor_core_supported=True,
                relative_speedup=25.0,
                bytes_per_element=1,
                accumulator_precision=Precision.FP32,
            ),
            Precision.FP8_E5M2: PrecisionProfile(
                precision=Precision.FP8_E5M2,
                peak_ops_per_sec=1500e12,  # 1.5 PFLOPS (with Tensor Cores)
                tensor_core_supported=True,
                relative_speedup=25.0,
                bytes_per_element=1,
                accumulator_precision=Precision.FP32,
            ),
            Precision.INT8: PrecisionProfile(
                precision=Precision.INT8,
                peak_ops_per_sec=1500e12,  # 1.5 POPS
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
        energy_per_flop_fp32=0.501e-12,  # ~0.5 pJ/FLOP at FP32
        energy_per_byte=15e-12,  # ~15 pJ/byte
        min_occupancy=0.25,
        max_concurrent_kernels=128,  # Can run many kernels concurrently
        wave_quantization=4,  # Launch in waves of 4 SMs

        # Hopper microarchitecture (for compute modeling)
        cuda_cores_per_sm=128,          # Doubled from Turing (64→128)
        ops_per_clock_per_core=2.0,     # FMA: 2 ops/clock
        sm_boost_clock_hz=1980e6,       # 1980 MHz boost
        sm_sustained_clock_hz=1830e6,   # 1830 MHz sustained (92% of boost)

        # Tensor Core details
        tensor_cores_per_sm=4,          # 4th gen Tensor Cores
        tensor_core_ops_per_clock=512,  # 512 FP16 FMAs per clock per TC

        # Thermal profile
        thermal_operating_points={
            "default": thermal_default,
        },
        default_thermal_profile="default",
    )


