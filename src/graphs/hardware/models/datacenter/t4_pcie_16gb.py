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
    ComputeFabric,
    get_base_alu_energy,
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
    # ========================================================================
    # CUDA Core Fabric (Standard Cell FP32 ALUs)
    # ========================================================================
    cuda_fabric = ComputeFabric(
        fabric_type="cuda_core",
        circuit_type="standard_cell",
        num_units=40 * 64,            # 40 SMs × 64 CUDA cores/SM = 2,560 cores
        ops_per_unit_per_clock={
            Precision.FP32: 2,         # FMA: 2 ops/clock
        },
        core_frequency_hz=1.59e9,     # 1.59 GHz boost
        process_node_nm=12,            # TSMC 12nm FFN
        energy_per_flop_fp32=get_base_alu_energy(12, 'standard_cell'),  # 2.5 pJ
        energy_scaling={
            Precision.FP32: 1.0,
            Precision.FP16: 0.5,
            Precision.INT8: 0.125,
            Precision.INT4: 0.0625,
        }
    )

    # ========================================================================
    # Tensor Core Fabric (15% more efficient, 2nd generation)
    # ========================================================================
    tensor_fabric = ComputeFabric(
        fabric_type="tensor_core",
        circuit_type="tensor_core",
        num_units=40 * 8,             # 320 Tensor Cores (40 SMs × 8 TCs/SM)
        ops_per_unit_per_clock={
            Precision.FP16: 256,       # 256 FP16 ops/clock/TC (2nd gen)
            Precision.INT8: 512,       # 512 INT8 ops/clock/TC (2× FP16, 2nd gen improvement)
            Precision.INT4: 1024,      # 1024 INT4 ops/clock/TC (2× INT8)
        },
        core_frequency_hz=1.59e9,
        process_node_nm=12,
        energy_per_flop_fp32=get_base_alu_energy(12, 'tensor_core'),  # 2.125 pJ (15% better)
        energy_scaling={
            Precision.FP16: 0.5,
            Precision.INT8: 0.125,
            Precision.INT4: 0.0625,
        }
    )

    # Calculate peak ops from fabrics
    cuda_fp32_peak = cuda_fabric.get_peak_ops_per_sec(Precision.FP32)
    tensor_fp16_peak = tensor_fabric.get_peak_ops_per_sec(Precision.FP16)
    tensor_int8_peak = tensor_fabric.get_peak_ops_per_sec(Precision.INT8)
    tensor_int4_peak = tensor_fabric.get_peak_ops_per_sec(Precision.INT4)

    return HardwareResourceModel(
        name="T4-PCIe-16GB",
        hardware_type=HardwareType.GPU,

        # NEW: Compute fabrics
        compute_fabrics=[cuda_fabric, tensor_fabric],

        # Legacy fields
        compute_units=40,  # SMs
        threads_per_unit=1024,  # Reduced from V100
        warps_per_unit=32,
        warp_size=32,

        # Legacy precision profiles (calculated from fabrics)
        precision_profiles={
            Precision.FP32: PrecisionProfile(
                precision=Precision.FP32,
                peak_ops_per_sec=cuda_fp32_peak,  # ~8.1 TFLOPS (CUDA cores)
                tensor_core_supported=False,
                relative_speedup=1.0,
                bytes_per_element=4,
            ),
            Precision.FP16: PrecisionProfile(
                precision=Precision.FP16,
                peak_ops_per_sec=tensor_fp16_peak,  # ~65 TFLOPS (Tensor Cores)
                tensor_core_supported=True,
                relative_speedup=8.0,
                bytes_per_element=2,
                accumulator_precision=Precision.FP32,
            ),
            Precision.INT8: PrecisionProfile(
                precision=Precision.INT8,
                peak_ops_per_sec=tensor_int8_peak,  # ~130 TOPS (2× FP16)
                tensor_core_supported=True,
                relative_speedup=16.0,
                bytes_per_element=1,
                accumulator_precision=Precision.INT32,
            ),
            Precision.INT4: PrecisionProfile(
                precision=Precision.INT4,
                peak_ops_per_sec=tensor_int4_peak,  # ~260 TOPS (2× INT8)
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

        # Legacy energy (use CUDA fabric as baseline)
        energy_per_flop_fp32=cuda_fabric.energy_per_flop_fp32,  # 2.5 pJ
        energy_per_byte=8e-12,  # ~8 pJ/byte (GDDR6)
        energy_scaling={
            Precision.FP32: 1.0,
            Precision.FP16: 0.5,
            Precision.INT8: 0.125,
            Precision.INT4: 0.0625,
        },
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


