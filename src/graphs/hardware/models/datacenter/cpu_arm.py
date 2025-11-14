"""
ARM CPU Resource Model (Datacenter/Edge - Generic ARM Cortex-A78AE style)

Standard ARM RISC CPU with:
  - Scalar ALUs (standard cell)
  - NEON SIMD units (10% more efficient packed ops)

This model represents embedded/edge ARM CPUs like:
  - Jetson Orin AGX CPU (12x Cortex-A78AE @ 2.2 GHz, 8nm)
  - AWS Graviton3 (64x Neoverse V1 @ 2.6 GHz, 5nm)
  - Ampere Altra (80x Neoverse N1 @ 3.0 GHz, 7nm)

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


def cpu_arm_resource_model(
    num_cores: int = 12,
    process_node_nm: int = 8,
    scalar_freq_ghz: float = 2.2,
    name_suffix: str = "Cortex-A78AE",
    tdp_watts: float = None
) -> HardwareResourceModel:
    """
    Generic ARM CPU resource model with scalar + NEON fabrics.

    Args:
        num_cores: Number of CPU cores
        process_node_nm: Process node (5, 7, 8, 16 nm)
        scalar_freq_ghz: CPU frequency (GHz)
        name_suffix: CPU core type name
        tdp_watts: Thermal Design Power in Watts (optional)

    Returns:
        HardwareResourceModel with dual compute fabrics

    Example:
        # Jetson Orin AGX CPU
        >>> cpu_arm_resource_model(12, 8, 2.2, "Cortex-A78AE")

        # AWS Graviton3
        >>> cpu_arm_resource_model(64, 5, 2.6, "Neoverse-V1")
    """

    # ========================================================================
    # Scalar ALU Fabric (Standard Cell)
    # ========================================================================
    scalar_fabric = ComputeFabric(
        fabric_type=f"scalar_alu_{name_suffix.lower()}",
        circuit_type="standard_cell",
        num_units=num_cores * 2,      # 2 ALUs per core (typical)
        ops_per_unit_per_clock={
            Precision.FP64: 2,          # FMA: 2 ops/clock
            Precision.FP32: 2,
            Precision.INT32: 2,
        },
        core_frequency_hz=scalar_freq_ghz * 1e9,
        process_node_nm=process_node_nm,
        energy_per_flop_fp32=get_base_alu_energy(process_node_nm, 'standard_cell'),
        energy_scaling={
            Precision.FP64: 2.0,        # Double precision = 2× energy
            Precision.FP32: 1.0,        # Baseline
            Precision.INT32: 0.5,
        }
    )

    # ========================================================================
    # NEON SIMD Fabric (10% more efficient due to packed ops)
    # ========================================================================
    neon_fabric = ComputeFabric(
        fabric_type="neon",
        circuit_type="simd_packed",
        num_units=num_cores,            # 1 NEON unit per core
        ops_per_unit_per_clock={
            Precision.FP32: 4,          # 128-bit NEON: 4× FP32/clock
            Precision.FP16: 8,          # 8× FP16/clock
            Precision.INT32: 4,
            Precision.INT16: 8,
            Precision.INT8: 16,         # 16× INT8/clock
        },
        core_frequency_hz=scalar_freq_ghz * 1e9,
        process_node_nm=process_node_nm,
        energy_per_flop_fp32=get_base_alu_energy(process_node_nm, 'simd_packed'),
        energy_scaling={
            Precision.FP32: 1.0,
            Precision.FP16: 0.5,
            Precision.INT32: 0.5,
            Precision.INT16: 0.25,
            Precision.INT8: 0.125,
        }
    )

    # ========================================================================
    # Hardware Resource Model
    # ========================================================================

    # Calculate legacy energy_per_flop_fp32 (for backward compatibility)
    # Use scalar fabric as baseline
    legacy_energy = scalar_fabric.energy_per_flop_fp32

    # Calculate peak FP32 performance (scalar + NEON)
    scalar_peak = scalar_fabric.get_peak_ops_per_sec(Precision.FP32)
    neon_peak = neon_fabric.get_peak_ops_per_sec(Precision.FP32)
    total_peak_fp32 = scalar_peak + neon_peak

    # Thermal operating point (if TDP is provided)
    thermal_default = None
    if tdp_watts is not None:
        thermal_default = ThermalOperatingPoint(
            name="default",
            tdp_watts=tdp_watts,
            cooling_solution="active-air",
            performance_specs={}
        )

    return HardwareResourceModel(
        name=f"CPU-ARM-{num_cores}core-{name_suffix}",
        hardware_type=HardwareType.CPU,

        # Compute fabrics (NEW)
        compute_fabrics=[scalar_fabric, neon_fabric],

        # Legacy fields (for backward compatibility)
        compute_units=num_cores,
        threads_per_unit=2,  # SMT (if supported)
        warps_per_unit=1,    # No warp concept
        warp_size=1,

        # Legacy precision profiles (use scalar + NEON combined)
        precision_profiles={
            Precision.FP64: PrecisionProfile(
                precision=Precision.FP64,
                peak_ops_per_sec=scalar_fabric.get_peak_ops_per_sec(Precision.FP64),
                tensor_core_supported=False,
                relative_speedup=0.5,
                bytes_per_element=8,
            ),
            Precision.FP32: PrecisionProfile(
                precision=Precision.FP32,
                peak_ops_per_sec=total_peak_fp32,
                tensor_core_supported=False,
                relative_speedup=1.0,
                bytes_per_element=4,
            ),
            Precision.FP16: PrecisionProfile(
                precision=Precision.FP16,
                peak_ops_per_sec=neon_fabric.get_peak_ops_per_sec(Precision.FP16),
                tensor_core_supported=False,
                relative_speedup=2.0,
                bytes_per_element=2,
            ),
            Precision.INT8: PrecisionProfile(
                precision=Precision.INT8,
                peak_ops_per_sec=neon_fabric.get_peak_ops_per_sec(Precision.INT8),
                tensor_core_supported=False,
                relative_speedup=4.0,
                bytes_per_element=1,
                accumulator_precision=Precision.INT32,
            ),
        },
        default_precision=Precision.FP32,

        # Memory hierarchy
        peak_bandwidth=80e9,  # 80 GB/s DDR5 (typical for edge/embedded)
        l1_cache_per_unit=64 * 1024,  # 64 KB per core (32 KB I + 32 KB D)
        l2_cache_total=num_cores * 512 * 1024,  # 512 KB per core
        main_memory=64 * 1024**3,  # 64 GB DDR5

        # Energy (legacy field - use scalar baseline)
        energy_per_flop_fp32=legacy_energy,
        energy_per_byte=20e-12,  # 20 pJ/byte (DRAM access)
        energy_scaling={
            Precision.FP64: 2.0,
            Precision.FP32: 1.0,
            Precision.FP16: 0.5,
            Precision.INT32: 0.5,
            Precision.INT16: 0.25,
            Precision.INT8: 0.125,
        },

        # Scheduling
        min_occupancy=0.5,
        max_concurrent_kernels=num_cores,  # One per core
        wave_quantization=1,

        # Thermal profile (if provided)
        thermal_operating_points={
            "default": thermal_default,
        } if thermal_default is not None else {},
        default_thermal_profile="default" if thermal_default is not None else None,
    )


# Convenience functions for common ARM CPU configurations

def jetson_orin_agx_cpu_arm() -> HardwareResourceModel:
    """Jetson Orin AGX CPU: 12x Cortex-A78AE @ 2.2 GHz, 8nm"""
    return cpu_arm_resource_model(
        num_cores=12,
        process_node_nm=8,
        scalar_freq_ghz=2.2,
        name_suffix="Cortex-A78AE"
    )


def graviton3_arm() -> HardwareResourceModel:
    """AWS Graviton3: 64x Neoverse-V1 @ 2.6 GHz, 5nm"""
    return cpu_arm_resource_model(
        num_cores=64,
        process_node_nm=5,
        scalar_freq_ghz=2.6,
        name_suffix="Neoverse-V1"
    )


def ampere_altra_arm() -> HardwareResourceModel:
    """Ampere Altra: 80x Neoverse-N1 @ 3.0 GHz, 7nm"""
    return cpu_arm_resource_model(
        num_cores=80,
        process_node_nm=7,
        scalar_freq_ghz=3.0,
        name_suffix="Neoverse-N1"
    )
