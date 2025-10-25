"""
Stanford Plasticine Cgra Resource Model hardware resource model.

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


def stanford_plasticine_cgra_resource_model() -> HardwareResourceModel:
    """
    Stanford Plasticine CGRA (Coarse-Grained Reconfigurable Architecture) resource model.

    Configuration: Hypothetical Plasticine-v2 (newer generation)
    Architecture: Spatial dataflow with medium-granularity PCUs

    Key characteristics:
    - Spatial dataflow execution (fundamentally different from temporal)
    - Medium-grained PCUs (balanced coverage vs overhead)
    - Reconfigurable fabric (like DPU but different execution model)
    - Conservative reconfiguration overhead (1000 cycles - Achilles heel)
    - Power budget: 15W (embodied AI range: 10-25W)

    Design Trade-offs:
    - PCU granularity: Medium (not too fine, not too coarse)
      - Too fine: Reconfigurable fabric overhead kills cost/power
      - Too coarse: Becomes multi-core CPU, loses flexibility
    - 32 PCUs covers most DNN operations (Conv, MatMul, etc.)
    - Each PCU: ~8 MACs, ~16 GOPS INT8

    Performance:
    - Peak: 10 TOPS INT8 theoretical
    - Realistic @ 60% efficiency: 6 TOPS INT8
    - Similar to DPU (7.68 TOPS) but different trade-offs

    References:
    - Stanford Plasticine architecture (Prabhakar et al.)
    - Hypothetical v2: 32 PCUs, medium granularity
    - Power: 15W (embodied AI target)
    - Reconfiguration: 1000 cycles (conservative modeling)
    """
    # Calculate theoretical and realistic peak performance
    num_pcus = 32  # Pattern Compute Units
    macs_per_pcu = 8  # Medium granularity
    clock_freq = 1.0e9  # 1 GHz typical for CGRAs
    ops_per_mac = 2  # Multiply + Accumulate
    efficiency = 0.60  # 60% efficiency (fabric overhead)

    # Theoretical peak
    theoretical_tops = num_pcus * macs_per_pcu * ops_per_mac * clock_freq  # 10.24 TOPS
    realistic_tops = theoretical_tops * efficiency  # 6.14 TOPS

    # Power profile (embodied AI range: 10-25W, use 15W midpoint)
    power_avg = 15.0  # Watts
    idle_power = 2.0  # Estimated idle (lower than DPU due to simpler fabric)
    dynamic_power = power_avg - idle_power  # 13W

    # Energy per operation
    energy_per_int8_op = dynamic_power / realistic_tops  # 2.12e-12 J/op
    # Convert to FP32 equivalent (INT8 is ~4× more efficient)
    energy_per_flop_fp32 = energy_per_int8_op * 4  # 8.48e-12 J/FLOP

    return HardwareResourceModel(
        name="CGRA-Plasticine-v2",
        hardware_type=HardwareType.CGRA,
        compute_units=32,  # PCUs (Pattern Compute Units)
        threads_per_unit=8,  # Operations per PCU
        warps_per_unit=1,  # Spatial execution (no warp concept)
        warp_size=1,

        precision_profiles={
            Precision.FP32: PrecisionProfile(
                precision=Precision.FP32,
                peak_ops_per_sec=realistic_tops / 8,  # 0.77 TFLOPS (not native)
                tensor_core_supported=False,
                relative_speedup=0.125,
                bytes_per_element=4,
            ),
            Precision.FP16: PrecisionProfile(
                precision=Precision.FP16,
                peak_ops_per_sec=realistic_tops / 4,  # 1.54 TFLOPS
                tensor_core_supported=False,
                relative_speedup=0.25,
                bytes_per_element=2,
            ),
            Precision.INT8: PrecisionProfile(
                precision=Precision.INT8,
                peak_ops_per_sec=realistic_tops,  # 6.14 TOPS (best)
                tensor_core_supported=False,  # Spatial execution, not tensor cores
                relative_speedup=1.0,
                bytes_per_element=1,
                accumulator_precision=Precision.INT32,
            ),
        },
        default_precision=Precision.INT8,

        peak_bandwidth=40e9,  # 40 GB/s on-chip interconnect
        l1_cache_per_unit=64 * 1024,  # 64 KB scratchpad per PCU
        l2_cache_total=2 * 1024 * 1024,  # 2 MB shared
        main_memory=4 * 1024**3,  # 4 GB DDR4 (edge device)
        energy_per_flop_fp32=energy_per_flop_fp32,  # ~8.48 pJ/FLOP
        energy_per_byte=12e-12,  # Similar to KPU (on-chip network)
        min_occupancy=0.3,
        max_concurrent_kernels=1,  # Spatial execution (entire graph mapped)
        wave_quantization=1,
    )


