"""
NVIDIA H100 SXM5 80GB Resource Model

FORM FACTOR: SXM (Server Module with high-bandwidth interconnect)
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
    ThermalOperatingPoint,
)


def h100_sxm5_80gb_resource_model() -> HardwareResourceModel:
    """
    NVIDIA H100 SXM5 80GB resource model with dual compute fabrics.

    FORM FACTOR: SXM (Server Module with high-bandwidth interconnect)
    MEMORY: 80 GB HBM2e

    Key characteristics:
    - 132 SMs with 4th gen Tensor Cores
    - Dual compute fabrics:
      1. CUDA Cores: Standard cell FP32/FP64 ALUs
      2. Tensor Cores: 15% more efficient for matrix ops
    - 2 TB/s HBM2e bandwidth

    Hopper Architecture (5nm TSMC 4N):
    - 132 SMs × 128 CUDA cores = 16,896 CUDA cores
    - 132 SMs × 4 Tensor Cores = 528 Tensor Cores
    - 1.98 GHz boost, 1.83 GHz sustained
    """

    # ========================================================================
    # CUDA Core Fabric (Standard Cell FP32 ALUs)
    # ========================================================================
    cuda_fabric = ComputeFabric(
        fabric_type="cuda_core",
        circuit_type="standard_cell",
        num_units=132 * 128,          # 132 SMs × 128 CUDA cores/SM = 16,896 cores
        ops_per_unit_per_clock={
            Precision.FP64: 2,         # FMA: 2 ops/clock
            Precision.FP32: 2,         # FMA: 2 ops/clock
        },
        core_frequency_hz=1.98e9,     # 1.98 GHz boost (use boost for peak)
        process_node_nm=4,             # TSMC 4N (4nm custom for NVIDIA)
        energy_per_flop_fp32=get_base_alu_energy(4, 'standard_cell'),  # 1.3 pJ
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
    tensor_fabric = ComputeFabric(
        fabric_type="tensor_core",
        circuit_type="tensor_core",
        num_units=132 * 4,            # 132 SMs × 4 Tensor Cores/SM = 528 TCs
        ops_per_unit_per_clock={
            Precision.BF16: 512,       # 512 BF16 ops/clock/TC (4th gen)
            Precision.FP16: 512,       # 512 FP16 ops/clock/TC
            Precision.FP8_E4M3: 1024,  # 1024 FP8 ops/clock/TC
            Precision.FP8_E5M2: 1024,  # 1024 FP8 ops/clock/TC
            Precision.INT8: 1024,      # 1024 INT8 ops/clock/TC
        },
        core_frequency_hz=1.98e9,
        process_node_nm=4,             # TSMC 4N (4nm custom for NVIDIA)
        energy_per_flop_fp32=get_base_alu_energy(4, 'tensor_core'),  # 1.11 pJ (15% better)
        energy_scaling={
            Precision.BF16: 0.5,       # Half precision
            Precision.FP16: 0.5,
            Precision.FP8_E4M3: 0.25,  # Quarter precision
            Precision.FP8_E5M2: 0.25,
            Precision.INT8: 0.125,     # INT8
        }
    )

    # ========================================================================
    # Legacy Precision Profiles (for backward compatibility)
    # ========================================================================

    # Calculate peak ops using fabrics
    cuda_fp64_peak = cuda_fabric.get_peak_ops_per_sec(Precision.FP64)
    cuda_fp32_peak = cuda_fabric.get_peak_ops_per_sec(Precision.FP32)
    tensor_bf16_peak = tensor_fabric.get_peak_ops_per_sec(Precision.BF16)
    tensor_fp16_peak = tensor_fabric.get_peak_ops_per_sec(Precision.FP16)
    tensor_fp8_peak = tensor_fabric.get_peak_ops_per_sec(Precision.FP8_E4M3)
    tensor_int8_peak = tensor_fabric.get_peak_ops_per_sec(Precision.INT8)

    # Thermal operating point (datacenter SXM)
    thermal_default = ThermalOperatingPoint(
        name="default",
        tdp_watts=700.0,  # H100 SXM5 TDP
        cooling_solution="active-air",
        performance_specs={}  # Uses precision_profiles for performance
    )

    return HardwareResourceModel(
        name="H100-SXM5-80GB",
        hardware_type=HardwareType.GPU,

        # NEW: Compute fabrics
        compute_fabrics=[cuda_fabric, tensor_fabric],

        # Legacy fields (for backward compatibility)
        compute_units=132,            # SMs (Streaming Multiprocessors)
        threads_per_unit=2048,        # Max threads per SM
        warps_per_unit=64,            # Max warps per SM (2048 / 32)
        warp_size=32,

        # Legacy precision profiles (calculated from fabrics)
        precision_profiles={
            Precision.FP64: PrecisionProfile(
                precision=Precision.FP64,
                peak_ops_per_sec=cuda_fp64_peak,  # ~67 TFLOPS
                tensor_core_supported=False,
                relative_speedup=1.0,
                bytes_per_element=8,
            ),
            Precision.FP32: PrecisionProfile(
                precision=Precision.FP32,
                peak_ops_per_sec=cuda_fp32_peak,  # ~67 TFLOPS (CUDA cores)
                tensor_core_supported=False,
                relative_speedup=1.0,
                bytes_per_element=4,
            ),
            Precision.BF16: PrecisionProfile(
                precision=Precision.BF16,
                peak_ops_per_sec=tensor_bf16_peak,  # ~537 TFLOPS (Tensor Cores)
                tensor_core_supported=True,
                relative_speedup=8.0,
                bytes_per_element=2,
                accumulator_precision=Precision.FP32,
            ),
            Precision.FP16: PrecisionProfile(
                precision=Precision.FP16,
                peak_ops_per_sec=tensor_fp16_peak,  # ~537 TFLOPS (Tensor Cores)
                tensor_core_supported=True,
                relative_speedup=8.0,
                bytes_per_element=2,
                accumulator_precision=Precision.FP32,
            ),
            Precision.FP8_E4M3: PrecisionProfile(
                precision=Precision.FP8_E4M3,
                peak_ops_per_sec=tensor_fp8_peak,  # ~1074 TOPS (Tensor Cores)
                tensor_core_supported=True,
                relative_speedup=16.0,
                bytes_per_element=1,
                accumulator_precision=Precision.FP32,
            ),
            Precision.FP8_E5M2: PrecisionProfile(
                precision=Precision.FP8_E5M2,
                peak_ops_per_sec=tensor_fp8_peak,  # ~1074 TOPS (Tensor Cores)
                tensor_core_supported=True,
                relative_speedup=16.0,
                bytes_per_element=1,
                accumulator_precision=Precision.FP32,
            ),
            Precision.INT8: PrecisionProfile(
                precision=Precision.INT8,
                peak_ops_per_sec=tensor_int8_peak,  # ~1074 TOPS (Tensor Cores)
                tensor_core_supported=True,
                relative_speedup=16.0,
                bytes_per_element=1,
                accumulator_precision=Precision.INT32,
            ),
        },
        default_precision=Precision.FP32,

        # Memory hierarchy
        peak_bandwidth=2e12,          # 2 TB/s HBM2e
        l1_cache_per_unit=256 * 1024,  # 256 KB per SM
        l2_cache_total=50 * 1024 * 1024,  # 50 MB
        main_memory=80 * 1024**3,     # 80 GB HBM2e

        # Legacy energy (use CUDA fabric as baseline)
        energy_per_flop_fp32=cuda_fabric.energy_per_flop_fp32,  # 1.5 pJ
        energy_per_byte=15e-12,       # 15 pJ/byte (HBM2e)
        energy_scaling={
            Precision.FP64: 2.0,
            Precision.FP32: 1.0,
            Precision.FP16: 0.5,
            Precision.BF16: 0.5,
            Precision.FP8_E4M3: 0.25,
            Precision.FP8_E5M2: 0.25,
            Precision.INT8: 0.125,
        },

        # Scheduling
        min_occupancy=0.25,
        max_concurrent_kernels=128,   # Can run many kernels concurrently
        wave_quantization=4,          # Launch in waves of 4 SMs

        # Hopper microarchitecture (for legacy compute modeling)
        cuda_cores_per_sm=128,        # Doubled from Turing (64→128)
        ops_per_clock_per_core=2.0,   # FMA: 2 ops/clock
        sm_boost_clock_hz=1.98e9,     # 1.98 GHz boost
        sm_sustained_clock_hz=1.83e9,  # 1.83 GHz sustained (92% of boost)

        # Tensor Core details
        tensor_cores_per_sm=4,        # 4th gen Tensor Cores
        tensor_core_ops_per_clock=512,  # 512 FP16 FMAs per clock per TC

        # Thermal profile
        thermal_operating_points={
            "default": thermal_default,
        },
        default_thermal_profile="default",
    )
