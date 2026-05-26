"""ARM Neoverse N2 resource model -- a real single-core ARM reference.

Neoverse N2 ("Perseus", ARMv9.0-A, 2021) is ARM's published reference IP for
efficiency-optimized server / edge cores. Unlike the synthetic
``ampere_ampereone_1core_reference`` (a num_cores=1 slice of the generic
``cpu_arm_resource_model``), this is a faithful model of N2's *published*
per-core datapath, so it can serve both as a standalone reference and as a
cross-check anchor for the generic ARM helper (issue #176).

Key microarchitecture (the differentiator vs the generic NEON model):
  - 2x 128-bit SVE2 vector pipelines per core (also execute NEON/AdvSIMD).
    -> 8 FP32, 16 FP16/BF16 lane-ops per clock per core (vs 4/8 for one NEON).
  - SVE2/NEON INT8 dot product (SDOT/UDOT): ~32 INT8 MAC/clock/core.
  - BF16 (BFDOT/BFMMLA) -- ARMv9 / SVE2.
  - L1: 64 KiB I + 64 KiB D per core (vs 32+32 on A78-class).
  - L2: configurable private 512 KiB or 1 MiB per core (reference: 1 MiB).

Single-core defaults match the issue's "real single-core ARM reference" ask;
``num_cores`` scales the fabrics for multi-core N2 SKUs (NXP S32, Marvell
OCTEON, Microsoft Cobalt 100, ...).

Sources (THEORETICAL confidence -- ARM does not publish per-core power, and
vector throughput is the architectural lane count, not a measured figure):
  - ARM Neoverse N2 Product Brief / Technical Reference Manual (2021)
  - ARM "Neoverse N2 platform" ISC/HotChips disclosures
  - Microsoft Cobalt 100 (128x Neoverse N2 @ ~3.4 GHz) as a shipping anchor
"""

from ...resource_model import (
    ComputeFabric,
    HardwareResourceModel,
    HardwareType,
    Precision,
    PrecisionProfile,
    ThermalOperatingPoint,
    get_base_alu_energy,
)


def arm_neoverse_n2_resource_model(
    num_cores: int = 1,
    process_node_nm: int = 5,
    freq_ghz: float = 3.2,
    l2_per_core_kib: int = 1024,
    tdp_watts: float = 4.0,
    name_suffix: str = "Neoverse-N2",
) -> HardwareResourceModel:
    """ARM Neoverse N2 reference model with 2x 128-bit SVE2 per core.

    Args:
        num_cores: Number of N2 cores (default 1 -- the reference design).
        process_node_nm: Process node (N2 ships on 5nm; 4nm also used).
        freq_ghz: Core clock (3.2 GHz reference; Cobalt 100 runs ~3.4 GHz).
        l2_per_core_kib: Private L2 per core (512 or 1024; reference 1 MiB).
        tdp_watts: Package TDP. Single-core ~4 W (N-series is efficiency
            -optimized; lower than the AmpereOne custom-core 5 W estimate).
            ARM does not separately spec per-core power -- this is an estimate.
        name_suffix: Core IP name (for the model name).
    """
    freq_hz = freq_ghz * 1e9

    # Scalar/FP issue -- 2 FMA-capable pipes per core (FP64 path).
    scalar_fabric = ComputeFabric(
        fabric_type=f"scalar_alu_{name_suffix.lower()}",
        circuit_type="standard_cell",
        num_units=num_cores * 2,
        ops_per_unit_per_clock={
            Precision.FP64: 2,
            Precision.FP32: 2,
            Precision.INT32: 2,
        },
        core_frequency_hz=freq_hz,
        process_node_nm=process_node_nm,
        energy_per_flop_fp32=get_base_alu_energy(process_node_nm, "standard_cell"),
        energy_scaling={
            Precision.FP64: 2.0,
            Precision.FP32: 1.0,
            Precision.INT32: 0.5,
        },
    )

    # 2x 128-bit SVE2 vector pipelines per core -- N2's defining feature.
    # ops/clock are lane counts for one 128-bit pipe x 2 pipes:
    #   FP32: 4 lanes x 2 = 8 ; FP16/BF16: 8 x 2 = 16 ; INT8 (SDOT): 16 x 2 = 32.
    sve2_fabric = ComputeFabric(
        fabric_type="sve2_128bx2",
        circuit_type="simd_packed",
        num_units=num_cores * 2,        # two 128-bit SVE2 pipes per core
        ops_per_unit_per_clock={
            Precision.FP32: 4,
            Precision.FP16: 8,
            Precision.BF16: 8,
            Precision.INT32: 4,
            Precision.INT16: 8,
            Precision.INT8: 16,          # SDOT/UDOT dot product
        },
        core_frequency_hz=freq_hz,
        process_node_nm=process_node_nm,
        energy_per_flop_fp32=get_base_alu_energy(process_node_nm, "simd_packed"),
        energy_scaling={
            Precision.FP32: 1.0,
            Precision.FP16: 0.5,
            Precision.BF16: 0.5,
            Precision.INT32: 0.5,
            Precision.INT16: 0.25,
            Precision.INT8: 0.125,
        },
    )

    scalar_peak_fp32 = scalar_fabric.get_peak_ops_per_sec(Precision.FP32)
    sve2_peak_fp32 = sve2_fabric.get_peak_ops_per_sec(Precision.FP32)
    total_peak_fp32 = scalar_peak_fp32 + sve2_peak_fp32

    thermal_default = ThermalOperatingPoint(
        name="default",
        tdp_watts=tdp_watts,
        cooling_solution="active-air",
        performance_specs={},
    )

    return HardwareResourceModel(
        name=f"CPU-ARM-{num_cores}core-{name_suffix}",
        hardware_type=HardwareType.CPU,
        compute_fabrics=[scalar_fabric, sve2_fabric],
        compute_units=num_cores,
        threads_per_unit=1,   # N2 is single-threaded per core (no SMT)
        warps_per_unit=1,
        warp_size=1,
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
                peak_ops_per_sec=sve2_fabric.get_peak_ops_per_sec(Precision.FP16),
                tensor_core_supported=False,
                relative_speedup=2.0,
                bytes_per_element=2,
            ),
            Precision.BF16: PrecisionProfile(
                precision=Precision.BF16,
                peak_ops_per_sec=sve2_fabric.get_peak_ops_per_sec(Precision.BF16),
                tensor_core_supported=False,
                relative_speedup=2.0,
                bytes_per_element=2,
            ),
            Precision.INT8: PrecisionProfile(
                precision=Precision.INT8,
                peak_ops_per_sec=sve2_fabric.get_peak_ops_per_sec(Precision.INT8),
                tensor_core_supported=False,
                relative_speedup=4.0,
                bytes_per_element=1,
                accumulator_precision=Precision.INT32,
            ),
        },
        default_precision=Precision.FP32,
        # Memory hierarchy (per-core L1/L2 from the N2 TRM).
        peak_bandwidth=80e9,   # single-core-saturable estimate; multi-core SKUs override
        l1_cache_per_unit=128 * 1024,           # 64 KiB I + 64 KiB D
        l2_cache_total=num_cores * l2_per_core_kib * 1024,
        main_memory=64 * 1024**3,
        energy_per_flop_fp32=scalar_fabric.energy_per_flop_fp32,
        energy_per_byte=20e-12,
        energy_scaling={
            Precision.FP64: 2.0,
            Precision.FP32: 1.0,
            Precision.FP16: 0.5,
            Precision.BF16: 0.5,
            Precision.INT32: 0.5,
            Precision.INT16: 0.25,
            Precision.INT8: 0.125,
        },
        min_occupancy=0.5,
        max_concurrent_kernels=num_cores,
        wave_quantization=1,
        thermal_operating_points={"default": thermal_default},
        default_thermal_profile="default",
        cpu_l1_spill_haircut=True,  # uncalibrated reference; model L1-spill (#178)
    )
