"""
AMD Ryzen 9 8945HS Resource Model (Phoenix, Zen 4, TSMC N4).

Mobile / laptop SoC: 8 Zen 4 cores, AVX-512 enabled (256-bit physical
double-pumped), Radeon 780M iGPU not modeled here. The compute model
focuses on the CPU cores -- the iGPU lives in a separate mapper if
ever modeled.

This factory exposes ``compute_fabrics`` populated with per-precision
``ops_per_unit_per_clock`` for FP64, FP32, FP16, BF16, INT8 (VNNI),
and INT4 (packed). Per-precision provenance is tagged THEORETICAL;
the Layer 1 ALU fitter upgrades these to INTERPOLATED / CALIBRATED
when measurements arrive.
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
from graphs.core.confidence import EstimationConfidence


# --------------------------------------------------------------------
# ISA capability summary (Zen 4)
# --------------------------------------------------------------------
# AVX-512:        Native (256-bit physical, 2-cycle issue for full
#                 512-bit ops). Available on retail Zen 4.
# AVX-512_BF16:   Supported (VDPBF16PS).
# AVX-512_VNNI:   Supported (VPDPBUSD).
# AVX-512_FP16:   NOT supported on Zen 4 (Intel SPR has it).
# AMX:            NOT supported (Intel server-only).
# FP8 / TF32:     NOT supported (no IP block).
# --------------------------------------------------------------------


def _ryzen_9_8945hs_zen4_fabric(freq_hz: float) -> ComputeFabric:
    """
    Zen 4 AVX-512 fabric.

    Per-core peak ops/clock (1 Zen 4 core, AVX-512 double-pumped to
    256-bit datapath -- counts as AVX2-effective throughput per cycle):
      FP32:  32 (8 lanes * 2 FMA pipes * 2 ops/FMA)
      FP64:  16 (half-rate)
      BF16:  64 (VDPBF16PS: 16 BF16 ops/256-bit op * 2 FMA)
      FP16:  4  (no native AVX-512_FP16; emulated)
      INT8:  64 (VPDPBUSD: 4 INT8 ops/lane * 8 lanes * 2 FMA)
      INT4:  128 (packed via INT8 path)

    ``ops_per_unit_per_clock`` is per core; num_units is core count.
    """
    return ComputeFabric(
        fabric_type="zen4_avx512",
        circuit_type="simd_packed",
        num_units=8,  # 8 Zen 4 cores
        ops_per_unit_per_clock={
            Precision.FP64: 16,
            Precision.FP32: 32,
            Precision.FP16: 4,    # emulated
            Precision.BF16: 64,   # native AVX-512_BF16
            Precision.INT8: 64,   # native AVX-512_VNNI
            Precision.INT4: 128,
        },
        core_frequency_hz=freq_hz,
        process_node_nm=4,  # TSMC N4 (Phoenix)
        energy_per_flop_fp32=get_base_alu_energy(4, 'simd_packed'),
        energy_scaling={
            Precision.FP64: 2.0,
            Precision.FP32: 1.0,
            Precision.FP16: 1.0,
            Precision.BF16: 0.50,  # native BF16 saves over FP32
            Precision.INT8: 0.25,
            Precision.INT4: 0.15,
        },
    )


def ryzen_9_8945hs_resource_model() -> HardwareResourceModel:
    """
    Build the resource model for AMD Ryzen 9 8945HS.

    Targets Layer 1 (ALU) population for the M1 micro-arch report:
    every fabric has ``ops_per_unit_per_clock`` populated for FP64,
    FP32, FP16, BF16, INT8, and INT4. Field provenance starts at
    THEORETICAL.
    """
    sustained_freq_hz = 4.4e9   # all-core sustained Zen 4 P-core clock

    z4_fabric = _ryzen_9_8945hs_zen4_fabric(sustained_freq_hz)

    def _peak(prec: Precision) -> float:
        return z4_fabric.get_peak_ops_per_sec(prec)

    thermal_45w = ThermalOperatingPoint(
        name="45W-mobile",
        tdp_watts=45.0,
        cooling_solution="laptop-vapor-chamber",
        performance_specs={},
    )

    model = HardwareResourceModel(
        name="AMD-Ryzen-9-8945HS",
        hardware_type=HardwareType.CPU,
        compute_fabrics=[z4_fabric],

        # Legacy compatibility
        compute_units=8,
        threads_per_unit=2,         # SMT
        warps_per_unit=1,
        warp_size=16,               # 512-bit AVX -> 16 FP32 lanes effective

        precision_profiles={
            Precision.FP64: PrecisionProfile(
                precision=Precision.FP64,
                peak_ops_per_sec=_peak(Precision.FP64),
                tensor_core_supported=False,
                relative_speedup=0.5,
                bytes_per_element=8,
            ),
            Precision.FP32: PrecisionProfile(
                precision=Precision.FP32,
                peak_ops_per_sec=_peak(Precision.FP32),
                tensor_core_supported=False,
                relative_speedup=1.0,
                bytes_per_element=4,
            ),
            Precision.FP16: PrecisionProfile(
                precision=Precision.FP16,
                peak_ops_per_sec=_peak(Precision.FP16),
                tensor_core_supported=False,
                relative_speedup=0.125,  # emulated
                bytes_per_element=2,
            ),
            Precision.BF16: PrecisionProfile(
                precision=Precision.BF16,
                peak_ops_per_sec=_peak(Precision.BF16),
                tensor_core_supported=True,   # native VDPBF16PS
                relative_speedup=2.0,
                bytes_per_element=2,
                accumulator_precision=Precision.FP32,
            ),
            Precision.INT8: PrecisionProfile(
                precision=Precision.INT8,
                peak_ops_per_sec=_peak(Precision.INT8),
                tensor_core_supported=True,
                relative_speedup=2.0,
                bytes_per_element=1,
                accumulator_precision=Precision.INT32,
            ),
            Precision.INT4: PrecisionProfile(
                precision=Precision.INT4,
                peak_ops_per_sec=_peak(Precision.INT4),
                tensor_core_supported=False,
                relative_speedup=4.0,
                bytes_per_element=0.5,
                accumulator_precision=Precision.INT16,
            ),
        },
        default_precision=Precision.FP32,

        peak_bandwidth=89.6e9,  # LPDDR5x-7500 dual-channel
        l1_cache_per_unit=32 * 1024,        # 32 KB L1D per Zen 4 core
        # Schema convention: ``l2_cache_total`` is the Last-Level
        # Cache (LLC), not the physical L2. For Phoenix this is the
        # 16 MB L3. See the i7-12700K mapper in
        # ``graphs.hardware.mappers.cpu`` for the full rationale.
        l2_cache_total=16 * 1024 * 1024,    # 16 MB L3 (LLC)
        main_memory=32 * 1024**3,

        energy_per_flop_fp32=z4_fabric.energy_per_flop_fp32,
        energy_per_byte=18e-12,
        energy_scaling={
            Precision.FP64: 2.0,
            Precision.FP32: 1.0,
            Precision.FP16: 1.0,
            Precision.BF16: 0.50,
            Precision.INT8: 0.25,
            Precision.INT4: 0.15,
        },

        min_occupancy=0.4,
        max_concurrent_kernels=16,
        wave_quantization=1,

        thermal_operating_points={"45W-mobile": thermal_45w},
        default_thermal_profile="45W-mobile",
    )

    for prec in (Precision.FP64, Precision.FP32, Precision.FP16,
                 Precision.BF16, Precision.INT8, Precision.INT4):
        model.set_provenance(
            f"compute_fabric.ops_per_clock.{prec.value}",
            EstimationConfidence.theoretical(
                score=0.60,
                source=("AMD Ryzen 9 8945HS datasheet (Phoenix, Zen 4, "
                        "TSMC N4), AVX-512 / AVX-512_BF16 / AVX-512_VNNI peak"),
            ),
        )
        model.set_provenance(
            f"energy_per_op.{prec.value}",
            EstimationConfidence.theoretical(
                score=0.45,
                source="Horowitz / process-node-energy table at 4nm",
            ),
        )

    return model
