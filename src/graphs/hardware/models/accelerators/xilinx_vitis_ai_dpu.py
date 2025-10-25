"""
Xilinx Vitis Ai Dpu Resource Model hardware resource model.

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


def xilinx_vitis_ai_dpu_resource_model() -> HardwareResourceModel:
    """
    Xilinx Vitis AI DPU (Deep Processing Unit) resource model.

    Configuration: B4096 (4096 MACs) on Versal VE2302 (embodied AI target)
    Architecture: AIE-ML v1 with 2D array of INT8 ALUs

    Key characteristics:
    - FPGA-based, reconfigurable
    - Native INT8 support (best performance)
    - Power efficient: 15-20W (embodied AI sweet spot)
    - Scratchpad-based memory hierarchy (similar to KPU)
    - 75% realistic efficiency (per specification)

    References:
    - Versal VE2302: 15-20W, edge-optimized
    - AIE-ML v1: 512 INT8 ops/clock @ 1.25 GHz
    - B4096: 4096 MACs
    - Realistic peak: 10.24 TOPS × 0.75 = 7.68 TOPS INT8
    """
    # Calculate theoretical and realistic peak performance
    mac_units = 4096
    clock_freq = 1.25e9  # 1.25 GHz (confirmed from Versal docs)
    ops_per_mac = 2  # Multiply + Accumulate
    efficiency = 0.75  # User-specified efficiency

    theoretical_tops = mac_units * ops_per_mac * clock_freq  # 10.24 TOPS
    realistic_tops = theoretical_tops * efficiency  # 7.68 TOPS

    # Power profile (VE2302: 15-20W)
    power_avg = 17.5  # Watts
    idle_power = 3.0  # Estimated idle
    dynamic_power = power_avg - idle_power  # 14.5W

    # Energy per operation
    energy_per_int8_op = dynamic_power / realistic_tops  # 1.89e-12 J/op
    # Convert to FP32 equivalent (INT8 is ~4× more efficient)
    energy_per_flop_fp32 = energy_per_int8_op * 4  # 7.56e-12 J/FLOP

    return HardwareResourceModel(
        name="DPU-Vitis-AI-B4096",
        hardware_type=HardwareType.DPU,
        compute_units=64,  # Tiles (estimate for B4096)
        threads_per_unit=64,  # Operations per tile
        warps_per_unit=8,  # Vector lanes per tile
        warp_size=8,

        precision_profiles={
            Precision.FP32: PrecisionProfile(
                precision=Precision.FP32,
                peak_ops_per_sec=realistic_tops / 8,  # 0.96 TFLOPS (not native)
                tensor_core_supported=False,
                relative_speedup=0.125,
                bytes_per_element=4,
            ),
            Precision.FP16: PrecisionProfile(
                precision=Precision.FP16,
                peak_ops_per_sec=realistic_tops / 4,  # 1.92 TFLOPS
                tensor_core_supported=True,  # AIE support
                relative_speedup=0.25,
                bytes_per_element=2,
            ),
            Precision.INT8: PrecisionProfile(
                precision=Precision.INT8,
                peak_ops_per_sec=realistic_tops,  # 7.68 TOPS (native, best)
                tensor_core_supported=True,  # Native INT8 MACs
                relative_speedup=1.0,
                bytes_per_element=1,
                accumulator_precision=Precision.INT32,
            ),
        },
        default_precision=Precision.INT8,

        peak_bandwidth=50e9,  # 50 GB/s DDR4 (edge device)
        l1_cache_per_unit=64 * 1024,  # 64 KB scratchpad per tile
        l2_cache_total=4 * 1024 * 1024,  # 4 MB (estimate)
        main_memory=8 * 1024**3,  # 8 GB DDR4 (edge deployment)
        energy_per_flop_fp32=energy_per_flop_fp32,  # ~7.56 pJ/FLOP
        energy_per_byte=15e-12,  # Similar to GPU (FPGA I/O)
        min_occupancy=0.3,
        max_concurrent_kernels=4,
        wave_quantization=2,  # Tiles allocated in pairs
    )


