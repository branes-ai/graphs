"""
NVIDIA T4 PCIe 16GB Resource Model

FORM FACTOR: PCIe (add-in card, inference-optimized)
MEMORY: 16 GB GDDR6

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


def t4_pcie_16gb_resource_model() -> HardwareResourceModel:
    """
    NVIDIA T4 PCIe 16GB resource model - Turing generation (2018).

    FORM FACTOR: PCIe (add-in card, inference-optimized)
    MEMORY: 16 GB GDDR6

    ARCHITECTURE:
    - Inference-optimized GPU (low power, high INT8 throughput)
    - 40 SMs with 64 CUDA cores each (2,560 CUDA cores total)
    - 8 Tensor Cores per SM (2nd gen, improved INT8)
    - Turing microarchitecture

    PERFORMANCE:
    - FP32: 8.1 TFLOPS (CUDA cores)
    - FP16 (Tensor Cores): 65 TFLOPS
    - INT8 (Tensor Cores): 130 TOPS (2× FP16)
    - INT4: 260 TOPS
    - Boost clock: 1590 MHz

    MEMORY:
    - 16 GB GDDR6
    - 320 GB/s bandwidth

    POWER:
    - 70W TDP (inference-optimized!)

    USE CASE:
    - Inference-optimized (low latency, high throughput)
    - Cloud inference (AWS G4, GCP T4)
    - Edge servers
    """
    return HardwareResourceModel(
        name="T4-PCIe-16GB",
        hardware_type=HardwareType.GPU,
        compute_units=40,  # SMs
        threads_per_unit=1024,  # Reduced from V100
        warps_per_unit=32,
        warp_size=32,

        # Precision profiles (inference-optimized)
        precision_profiles={
            Precision.FP32: PrecisionProfile(
                precision=Precision.FP32,
                peak_ops_per_sec=8.1e12,  # 8.1 TFLOPS (CUDA cores)
                tensor_core_supported=False,
                relative_speedup=1.0,
                bytes_per_element=4,
            ),
            Precision.FP16: PrecisionProfile(
                precision=Precision.FP16,
                peak_ops_per_sec=65e12,  # 65 TFLOPS (Tensor Cores)
                tensor_core_supported=True,
                relative_speedup=8.0,
                bytes_per_element=2,
                accumulator_precision=Precision.FP32,
            ),
            Precision.INT8: PrecisionProfile(
                precision=Precision.INT8,
                peak_ops_per_sec=130e12,  # 130 TOPS (2× FP16)
                tensor_core_supported=True,
                relative_speedup=16.0,
                bytes_per_element=1,
                accumulator_precision=Precision.INT32,
            ),
            Precision.INT4: PrecisionProfile(
                precision=Precision.INT4,
                peak_ops_per_sec=260e12,  # 260 TOPS (2× INT8)
                tensor_core_supported=True,
                relative_speedup=32.0,
                bytes_per_element=0.5,
                accumulator_precision=Precision.INT32,
            ),
        },
        default_precision=Precision.FP32,

        peak_bandwidth=320e9,  # 320 GB/s GDDR6
        l1_cache_per_unit=96 * 1024,  # 96 KB per SM
        l2_cache_total=4 * 1024 * 1024,  # 4 MB
        main_memory=16 * 1024**3,  # 16 GB GDDR6
        energy_per_flop_fp32=0.29e-12,  # ~0.29 pJ/FLOP (70W / 8.1 TFLOPS / efficiency)
        energy_per_byte=8e-12,  # ~8 pJ/byte
        min_occupancy=0.25,
        max_concurrent_kernels=32,
        wave_quantization=4,

        # Turing microarchitecture (same core count as Volta)
        cuda_cores_per_sm=64,           # Same as Volta
        ops_per_clock_per_core=2.0,     # FMA
        sm_boost_clock_hz=1590e6,       # 1590 MHz boost
        sm_sustained_clock_hz=1470e6,   # 1470 MHz sustained (~92% of boost)

        # Tensor Core details (2nd generation, improved INT8)
        tensor_cores_per_sm=8,          # 8 TCs per SM
        tensor_core_ops_per_clock=256,  # Similar to Volta
    )


