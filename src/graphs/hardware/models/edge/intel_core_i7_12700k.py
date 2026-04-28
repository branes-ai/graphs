"""
Intel Core i7-12700K Resource Model (12th Gen Alder Lake).

Hybrid 8 P-core (Golden Cove) + 4 E-core (Gracemont) consumer desktop
CPU. Process: Intel 7 (10nm Enhanced SuperFin). AVX2 only on retail
Alder Lake -- AVX-512 was fused off for hybrid scheduling. AMX is
exclusive to server Sapphire Rapids; not available here.

This factory exposes ``compute_fabrics`` populated with per-precision
``ops_per_unit_per_clock`` for FP64, FP32, FP16 (emulated), BF16
(emulated), INT8 (VNNI), and INT4 (packed). Per-precision provenance
is tagged THEORETICAL; the Layer 1 ALU fitter upgrades these to
INTERPOLATED / CALIBRATED when measurements arrive.
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
# ISA capability summary (Alder Lake retail SKUs)
# --------------------------------------------------------------------
# AVX-512:   FUSED OFF on retail Alder Lake (was enabled in early
#            steppings; later disabled by microcode update).
# AVX2:      256-bit FMA, 2 ports (P0 + P1) on Golden Cove.
# AVX-VNNI:  Supported (subset of AVX-512_VNNI re-encoded for AVX2 /
#            128-bit and 256-bit). Provides 4-way INT8 dot product.
# BF16:      No native AVX-512_BF16; software emulation only.
# FP16:      No AVX-512_FP16; software emulation only.
# AMX:       NOT supported (Sapphire Rapids server-only).
# FP8/TF32:  NOT supported (no IP block).
# --------------------------------------------------------------------


def _i7_12700k_p_core_fabric(freq_hz: float) -> ComputeFabric:
    """
    Golden Cove (P-core) AVX2 fabric.

    Per-core peak ops/clock (1 P-core, AVX2):
      FP32:  16 (8 lanes * 2 FMA pipes * 2 ops/FMA / 2 = 16, 1 issue/cyc)
      FP64:  8  (half-rate vs FP32)
      FP16:  4  (no native; scalar emulation only)
      BF16:  4  (no native; scalar emulation only)
      INT8:  64 (AVX-VNNI VPDPBUSD: 4 INT8 ops per 32-bit lane *
                 8 lanes * 2 FMA pipes)
      INT4:  128 (packed-INT4 emulation; double-pump VPDPBUSD)

    Note: ``ops_per_unit_per_clock`` here is *per core* (the core is the
    "unit"). num_units is the P-core count.
    """
    return ComputeFabric(
        fabric_type="alder_lake_p_core_avx2",
        circuit_type="simd_packed",
        num_units=8,  # 8 P-cores
        ops_per_unit_per_clock={
            Precision.FP64: 8,
            Precision.FP32: 16,
            Precision.FP16: 4,    # emulated (no native AVX-512_FP16)
            Precision.BF16: 4,    # emulated (no native AVX-512_BF16)
            Precision.INT8: 64,   # AVX-VNNI
            Precision.INT4: 128,  # packed via INT8 (datasheet upper bound)
        },
        core_frequency_hz=freq_hz,
        process_node_nm=10,  # Intel 7 (10nm ESF)
        energy_per_flop_fp32=get_base_alu_energy(10, 'simd_packed'),
        energy_scaling={
            Precision.FP64: 2.0,
            Precision.FP32: 1.0,
            Precision.FP16: 1.0,   # emulated -> same as FP32
            Precision.BF16: 1.0,
            Precision.INT8: 0.30,  # VNNI saves some, but x86 CPU overhead caps it
            Precision.INT4: 0.18,
        },
    )


def _i7_12700k_e_core_fabric(freq_hz: float) -> ComputeFabric:
    """
    Gracemont (E-core) AVX2 fabric.

    Per-core peak ops/clock (1 E-core, AVX2):
      FP32:  8  (8-wide AVX2, 1 FMA pipe -- half of P-core)
      FP64:  4
      INT8:  32 (AVX-VNNI single-pipe)
      Other precisions: emulated, low rate.
    """
    return ComputeFabric(
        fabric_type="gracemont_e_core_avx2",
        circuit_type="simd_packed",
        num_units=4,  # 4 E-cores
        ops_per_unit_per_clock={
            Precision.FP64: 4,
            Precision.FP32: 8,
            Precision.FP16: 2,
            Precision.BF16: 2,
            Precision.INT8: 32,
            Precision.INT4: 64,
        },
        core_frequency_hz=freq_hz,
        process_node_nm=10,
        energy_per_flop_fp32=get_base_alu_energy(10, 'simd_packed'),
        energy_scaling={
            Precision.FP64: 2.0,
            Precision.FP32: 1.0,
            Precision.FP16: 1.0,
            Precision.BF16: 1.0,
            Precision.INT8: 0.30,
            Precision.INT4: 0.18,
        },
    )


def intel_core_i7_12700k_resource_model() -> HardwareResourceModel:
    """
    Build the resource model for Intel Core i7-12700K.

    Targets Layer 1 (ALU) population for the M1 micro-arch report:
    every fabric has ``ops_per_unit_per_clock`` populated for FP64,
    FP32, FP16, BF16, INT8, and INT4. Field provenance starts at
    THEORETICAL; downstream Layer 1 fitters (FMA-rate sweep) upgrade
    these as measurements come in.
    """
    p_core_freq_hz = 4.7e9   # all-core sustained P-core clock
    e_core_freq_hz = 3.6e9   # all-core sustained E-core clock

    p_fabric = _i7_12700k_p_core_fabric(p_core_freq_hz)
    e_fabric = _i7_12700k_e_core_fabric(e_core_freq_hz)

    # Hybrid-core thread budget: P-cores SMT (16 threads), E-cores do not
    # (4 threads). The legacy compute_units / threads_per_unit pair is
    # used by some consumers as compute_units * threads_per_unit, so
    # express the effective core count and weighted threads_per_unit
    # the same way the existing mapper in graphs.hardware.mappers.cpu
    # does: 8 P + 0.6 * 4 E = 10 effective cores, threads_per_unit=2
    # (P-core-dominant), giving 20 runnable threads.
    p_cores = 8
    e_cores = 4
    e_core_efficiency = 0.6
    effective_cores = p_cores + int(e_cores * e_core_efficiency)  # 10

    # Aggregate peak ops across both fabrics, per precision
    def _peak(prec: Precision) -> float:
        p = p_fabric.get_peak_ops_per_sec(prec)
        e = e_fabric.get_peak_ops_per_sec(prec)
        return p + e

    thermal_125w = ThermalOperatingPoint(
        name="125W-PL1",
        tdp_watts=125.0,
        cooling_solution="tower-cooler",
        performance_specs={},
    )

    model = HardwareResourceModel(
        name="Intel-Core-i7-12700K",
        hardware_type=HardwareType.CPU,
        compute_fabrics=[p_fabric, e_fabric],

        # Legacy compatibility -- effective_cores * threads_per_unit
        # gives the realistic 20-thread budget (8 P-cores SMT'd + 4
        # non-SMT E-cores, with E-core derate applied).
        compute_units=effective_cores,   # 10 effective cores
        threads_per_unit=2,              # P-core-dominant (E-cores no SMT)
        warps_per_unit=1,
        warp_size=8,                     # 256-bit AVX2 -> 8 FP32 lanes

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
                relative_speedup=0.25,  # emulated
                bytes_per_element=2,
            ),
            Precision.BF16: PrecisionProfile(
                precision=Precision.BF16,
                peak_ops_per_sec=_peak(Precision.BF16),
                tensor_core_supported=False,
                relative_speedup=0.25,
                bytes_per_element=2,
            ),
            Precision.INT8: PrecisionProfile(
                precision=Precision.INT8,
                peak_ops_per_sec=_peak(Precision.INT8),
                tensor_core_supported=True,    # AVX-VNNI counts as a fused MAC
                relative_speedup=4.0,
                bytes_per_element=1,
                accumulator_precision=Precision.INT32,
            ),
            Precision.INT4: PrecisionProfile(
                precision=Precision.INT4,
                peak_ops_per_sec=_peak(Precision.INT4),
                tensor_core_supported=False,   # packed emulation
                relative_speedup=8.0,
                bytes_per_element=0.5,
                accumulator_precision=Precision.INT16,
            ),
        },
        default_precision=Precision.FP32,

        peak_bandwidth=76.8e9,  # DDR5-4800 dual-channel ~76.8 GB/s
        l1_cache_per_unit=48 * 1024,        # 48 KB L1D per P-core
        # Schema convention: ``l2_cache_total`` is the Last-Level Cache
        # (the cache that determines DRAM-spill behavior), not the
        # physical L2. For Alder Lake this is the 25 MB L3. See the
        # i7-12700K mapper in ``graphs.hardware.mappers.cpu`` for the
        # full rationale.
        l2_cache_total=25 * 1024 * 1024,    # 25 MB L3 (LLC)
        main_memory=64 * 1024**3,

        energy_per_flop_fp32=p_fabric.energy_per_flop_fp32,
        energy_per_byte=25e-12,
        energy_scaling={
            Precision.FP64: 2.0,
            Precision.FP32: 1.0,
            Precision.FP16: 1.0,
            Precision.BF16: 1.0,
            Precision.INT8: 0.30,
            Precision.INT4: 0.18,
        },

        min_occupancy=0.4,
        max_concurrent_kernels=20,
        wave_quantization=1,

        thermal_operating_points={"125W-PL1": thermal_125w},
        default_thermal_profile="125W-PL1",

        # M2 Layer 2: AVX2-only retail Alder Lake. Hybrid scheduling
        # (P-core SMT + non-SMT E-core) and the lack of AVX-512 keep
        # vector efficiency below Zen 4. Reference values from
        # CPUMapper._analyze_vectorization (0.95 / 0.80 / 0.70).
        simd_efficiency={
            "elementwise": 0.95,
            "matrix":      0.80,
            "default":     0.70,
        },

        # M3 Layer 3: hardware-managed L1 (split D + I) per Golden Cove
        # / Gracemont core.
        l1_storage_kind="cache",

        # M4 Layer 4: physical L2 per core. Golden Cove P-cores carry
        # 1.25 MB private L2; Gracemont E-cores share 2 MB per E-core
        # cluster (4 cores). Reported as the P-core value (P-cores
        # dominate compute throughput); legacy l2_cache_total still
        # holds the LLC (= L3 = 25 MB).
        l2_cache_per_unit=int(1.25 * 1024 * 1024),  # 1.25 MB per P-core
        l2_topology="per-unit",
    )

    # ------------------------------------------------------------------
    # Provenance: every per-precision ALU rate starts THEORETICAL.
    # The Layer 1 FMA-rate fitter upgrades these to CALIBRATED.
    # ------------------------------------------------------------------
    for prec in (Precision.FP64, Precision.FP32, Precision.FP16,
                 Precision.BF16, Precision.INT8, Precision.INT4):
        model.set_provenance(
            f"compute_fabric.ops_per_clock.{prec.value}",
            EstimationConfidence.theoretical(
                score=0.55,
                source=("Intel Core i7-12700K datasheet (Alder Lake, "
                        "Intel 7), AVX2 / AVX-VNNI peak"),
            ),
        )
        model.set_provenance(
            f"energy_per_op.{prec.value}",
            EstimationConfidence.theoretical(
                score=0.45,
                source="Horowitz / process-node-energy table at 10nm",
            ),
        )

    # M2 Layer 2 provenance for SIMD-efficiency coefficients
    for op_kind in ("elementwise", "matrix", "default"):
        model.set_provenance(
            f"simd_efficiency.{op_kind}",
            EstimationConfidence.theoretical(
                score=0.50,
                source=("CPUMapper analytical default for AVX2 hybrid "
                        "Alder Lake (no AVX-512, hybrid scheduling)"),
            ),
        )

    # M3 Layer 3 provenance for L1 cache fields
    model.set_provenance(
        "l1_cache_per_unit",
        EstimationConfidence.theoretical(
            score=0.85,
            source=("Intel Core i7-12700K datasheet: 48 KB L1D per "
                    "Golden Cove P-core; model reports the P-core "
                    "value as the per-unit L1 (Gracemont E-core L1D "
                    "is 32 KB but a smaller fraction of compute "
                    "throughput)"),
        ),
    )
    model.set_provenance(
        "l1_storage_kind",
        EstimationConfidence.theoretical(
            score=0.95,
            source="x86 architectural fact: hardware-managed coherent L1",
        ),
    )

    # M4 Layer 4 provenance for L2 cache fields
    model.set_provenance(
        "l2_cache_per_unit",
        EstimationConfidence.theoretical(
            score=0.85,
            source=("Intel Core i7-12700K datasheet: 1.25 MB private L2 "
                    "per Golden Cove P-core (Gracemont E-cores share "
                    "2 MB per cluster; P-core value reported as primary)"),
        ),
    )
    model.set_provenance(
        "l2_topology",
        EstimationConfidence.theoretical(
            score=0.95,
            source="Alder Lake architectural fact: private per-core L2",
        ),
    )

    return model
