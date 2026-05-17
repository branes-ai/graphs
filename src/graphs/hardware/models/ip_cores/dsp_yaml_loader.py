"""DSP resource model loader from embodied-schemas ComputeProduct YAMLs.

PR 4 of the DSP mini-sprint scoped at issue #211. Mirrors
``src/graphs/hardware/models/datacenter/tpu_yaml_loader.py`` (TPU
sprint) and ``accelerators/dpu_yaml_loader.py`` (DPU sprint) but reads
a DSPBlock-bearing ``ComputeProduct`` and builds a
``HardwareResourceModel`` with the DSP-shaped fields populated.

Sibling to the hand-coded factory at
``src/graphs/hardware/models/ip_cores/cadence_vision_q8.py`` during the
parallel-migration phase. Adding the factory swap is PR 5; this PR
ships the loader machinery and a parity test that proves the YAML-
loaded model matches the hand-coded one's key fields.

Scope of DSPBlock -> HardwareResourceModel mapping:

  DSPBlock.compute_fabrics[*]        -> compute_fabrics[] (one per
                                        DSPComputeFabric; num_units =
                                        DSPComputeFabric.num_units)
  DSPBlock.compute_fabrics[0].num_units
                                     -> compute_units (the canonical
                                        "how many units" surface; uses
                                        the first fabric's count since
                                        most DSPs are single-fabric)
  DSPBlock.memory.l1_size_bytes_per_unit
                                     -> l1_cache_per_unit
  DSPBlock.memory.l2_size_bytes_total -> l2_cache_total
  DSPBlock.memory.external_dram_*    -> main_memory, peak_bandwidth,
                                        memory_technology,
                                        memory_*_energy_per_byte_pj
  DSPBlock.{min_occupancy,           -> matching HardwareResourceModel
    max_concurrent_kernels,             fields
    wave_quantization}
  ComputeProduct.power.thermal_profiles[]
                                     -> thermal_operating_points

DSP-specific loader decisions:

1. **peak_bandwidth = external_dram_bandwidth_gbps** when
   has_external_dram=True. For IP cores (Cadence/CEVA/Synopsys) this
   is "typical SoC integration" -- the YAML's
   ``external_dram_bandwidth_kind: 'typical'`` discriminator marks
   this. For SoC-integrated DSPs (TDA4VM, SA8775P) this is measured
   datasheet bandwidth. The loader emits the same value either way;
   downstream cost models that care about the distinction read
   ``deployment_kind`` + ``external_dram_bandwidth_kind`` directly
   from the ComputeProduct.

2. **No tile energy decomposition.** Unlike TPUs (with weight FIFOs
   + accumulators + unified buffers), DSPs don't have a centralized
   tile energy story. Per-fabric ``energy_per_op_fp32_pj`` + per-
   precision ``energy_scaling`` is sufficient. No ``tile_energy_model``
   attached.

3. **No SoCFabricModel.** DSPBlock carries ``noc_confidence`` but no
   full NoC sub-type (DSP vendors rarely publish NoC details).
   Downstream consumers that need NoC modeling for the surrounding
   SoC fabric read directly from the parent ComputeProduct.

4. **HardwareType.DSP** set directly -- DSP already in graphs enum.

5. **default_precision honors the YAML** (vs TPU which forces BF16
   and NPU/DPU/CGRA which force INT8). DSPs span signal processing
   (FP32) through ML inference (INT8); each SKU declares its
   canonical default.

6. **Multi-fabric is the rule.** DSPBlock.compute_fabrics has 1
   entry for IP cores (Cadence single-fabric) and 2 for SoC DSPs
   (HVX+HMX, C7x+MMA). The loader produces one ComputeFabric per
   DSPComputeFabric. compute_units uses the FIRST fabric's num_units
   (the canonical "main" fabric) -- consumers needing per-fabric
   detail iterate compute_fabrics directly.

7. **First loader to use compute_block_common from day 1** -- no
   per-architecture *TheoreticalPerformance imports; pulls
   ``TheoreticalPerformance`` directly through DSPBlock's field.

Out of scope for v9 (deferred to v10+):

  - BOMCostProfile (YAML doesn't carry; v10 Market.bom)
  - Per-field provenance copy-through with rich source citations
  - VLIW issue width surfacing on HardwareResourceModel (v10
    reconciliation -- informational only)
  - external_dram_bandwidth_kind surfacing on HardwareResourceModel
    (v10 reconciliation -- mappers/cost models that care read it
    from the ComputeProduct directly)
"""

from __future__ import annotations

from typing import Optional

from embodied_schemas.compute_product import ComputeProduct
from embodied_schemas.dsp_block import (
    DSPBlock,
    DSPDeploymentKind,
    DSPFabricKind,
)
from embodied_schemas.loaders import load_compute_products
from embodied_schemas.process_node import ProcessNodeEntry

from graphs.core.confidence import EstimationConfidence
from graphs.hardware.resource_model import (
    ComputeFabric,
    HardwareResourceModel,
    HardwareType,
    Precision,
    PrecisionProfile,
    ThermalOperatingPoint,
)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class DSPYamlLoaderError(Exception):
    """Raised when a ComputeProduct YAML can't be turned into a DSP
    HardwareResourceModel (missing block, unsupported topology,
    unresolved references, etc.)."""


# ---------------------------------------------------------------------------
# Mapping tables
# ---------------------------------------------------------------------------

_PRECISION_BY_NAME: dict[str, Precision] = {
    "fp64": Precision.FP64,
    "fp32": Precision.FP32,
    "tf32": Precision.TF32,
    "fp16": Precision.FP16,
    "bf16": Precision.BF16,
    "fp8_e4m3": Precision.FP8_E4M3,
    "fp8_e5m2": Precision.FP8_E5M2,
    "int8": Precision.INT8,
    "int16": Precision.INT16,
    "int4": Precision.INT4,
}

_BYTES_PER_PRECISION: dict[Precision, float] = {
    Precision.FP64: 8, Precision.FP32: 4, Precision.TF32: 4,
    Precision.FP16: 2, Precision.BF16: 2,
    Precision.FP8_E4M3: 1, Precision.FP8_E5M2: 1,
    Precision.INT8: 1, Precision.INT16: 2, Precision.INT4: 0.5,
}

# DSPFabricKind -> ComputeFabric.fabric_type. Matches the hand-coded
# factory's fabric_type strings so parity tests can pin equality.
# Cadence Vision Q8's hand-coded factory uses "vision_q8_simd"; SoC
# DSPs will use "hvx_vector" / "c7x_dsp" / etc. The mapping here is
# generic ("vector_simd" / "tensor_matrix") -- factory cleanups (PR 5)
# can pass name_override or fabric_type_override to preserve legacy
# strings on a per-SKU basis.
_FABRIC_TYPE_BY_KIND: dict[DSPFabricKind, str] = {
    DSPFabricKind.VECTOR_SIMD:   "vector_simd",
    DSPFabricKind.TENSOR_MATRIX: "tensor_matrix",
    DSPFabricKind.VLIW_SCALAR:   "vliw_scalar",
    DSPFabricKind.HYBRID:        "hybrid",
}

_CIRCUIT_TYPE_DEFAULT = "simd_packed"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dsp_block(cp: ComputeProduct) -> DSPBlock:
    """Pick the (single) DSPBlock from a ComputeProduct's first die."""
    for block in cp.dies[0].blocks:
        if isinstance(block, DSPBlock):
            return block
    raise DSPYamlLoaderError(
        f"ComputeProduct {cp.id!r} has no DSPBlock in dies[0].blocks "
        f"(found: {[type(b).__name__ for b in cp.dies[0].blocks]})"
    )


def _precisions_from_ops_dict(
    ops: dict[str, int],
    *,
    source: str = "ops_per_unit_per_clock",
) -> dict[Precision, int]:
    """Convert YAML's str-keyed ops/clock dict to graphs' Precision-keyed
    dict. Fails fast on unknown precision names."""
    out: dict[Precision, int] = {}
    unknown: list[str] = []
    for name, count in ops.items():
        prec = _PRECISION_BY_NAME.get(name.lower())
        if prec is None:
            unknown.append(name)
            continue
        out[prec] = count
    if unknown:
        raise DSPYamlLoaderError(
            f"unknown precision name(s) in {source}: {sorted(unknown)}. "
            f"Known: {sorted(_PRECISION_BY_NAME)}"
        )
    return out


def _energy_scaling_from_yaml(
    scaling: dict[str, float],
    *,
    source: str = "energy_scaling",
) -> dict[Precision, float]:
    out: dict[Precision, float] = {}
    unknown: list[str] = []
    for name, factor in scaling.items():
        prec = _PRECISION_BY_NAME.get(name.lower())
        if prec is None:
            unknown.append(name)
            continue
        out[prec] = factor
    if unknown:
        raise DSPYamlLoaderError(
            f"unknown precision name(s) in {source}: {sorted(unknown)}. "
            f"Known: {sorted(_PRECISION_BY_NAME)}"
        )
    return out


def _resolve_default_precision(
    cp_default: str,
    available: dict[Precision, PrecisionProfile],
) -> Precision:
    """Honor the YAML's ``default_precision`` when present in the
    profile set; fall back to INT8 then to first available."""
    prec = _PRECISION_BY_NAME.get(cp_default.lower())
    if prec is not None and prec in available:
        return prec
    if Precision.INT8 in available:
        return Precision.INT8
    return next(iter(available))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_dsp_resource_model_from_yaml(
    base_id: str,
    *,
    products: Optional[dict[str, ComputeProduct]] = None,
    process_nodes: Optional[dict[str, ProcessNodeEntry]] = None,
    name_override: Optional[str] = None,
    fabric_type_overrides: Optional[dict[DSPFabricKind, str]] = None,
) -> HardwareResourceModel:
    """Build a ``HardwareResourceModel`` for ``base_id`` from a DSP
    ComputeProduct YAML.

    Args:
        base_id: ComputeProduct id, e.g., "cadence_tensilica_vision_q8".
        products / process_nodes: optional pre-loaded catalogs.
        name_override: optional override for the resource model's
            ``name`` field. Hand-coded factories use specific legacy
            strings (e.g., "Cadence-Tensilica-Vision-Q8"); pass that
            to preserve parity.
        fabric_type_overrides: per-fabric-kind override for
            ``ComputeFabric.fabric_type``. The generic loader uses
            short kind names ("vector_simd"); SKU-specific factory
            wrappers can pass ``{VECTOR_SIMD: "vision_q8_simd"}`` to
            keep legacy fabric_type strings intact during the
            parallel-migration phase.

    Raises:
        DSPYamlLoaderError: when ``base_id`` is not in the catalog,
            doesn't carry a DSPBlock, or has unresolvable references.
    """
    if products is None:
        products = load_compute_products()
    if process_nodes is None:
        from embodied_schemas.loaders import load_process_nodes
        process_nodes = load_process_nodes()

    cp = products.get(base_id)
    if cp is None:
        raise DSPYamlLoaderError(
            f"no ComputeProduct with id={base_id!r}. Available: "
            f"{', '.join(sorted(products))}"
        )
    block = _dsp_block(cp)

    node = process_nodes.get(cp.dies[0].process_node_id)
    if node is None:
        raise DSPYamlLoaderError(
            f"SKU {base_id!r} references process_node_id="
            f"{cp.dies[0].process_node_id!r} which does not resolve"
        )
    process_node_nm = int(node.node_nm)

    # ------------------------------------------------------------------
    # Default thermal profile -> clock used as fabric core_frequency_hz
    # ------------------------------------------------------------------
    default_profile = next(
        (p for p in cp.power.thermal_profiles
         if p.name == cp.power.default_thermal_profile),
        None,
    )
    if default_profile is None:
        raise DSPYamlLoaderError(
            f"SKU {base_id!r}: default_thermal_profile "
            f"{cp.power.default_thermal_profile!r} is not in thermal_profiles"
        )
    default_clock_hz = default_profile.clock_mhz * 1e6

    # ------------------------------------------------------------------
    # ComputeFabric per DSP fabric. DSPs are typically 1 or 2 fabrics.
    # ------------------------------------------------------------------
    overrides = fabric_type_overrides or {}
    compute_fabrics: list[ComputeFabric] = []
    for dsp_fabric in block.compute_fabrics:
        ops_dict = _precisions_from_ops_dict(
            dsp_fabric.ops_per_unit_per_clock,
            source=f"fabric.{dsp_fabric.fabric_kind.value}",
        )
        if not ops_dict:
            continue
        energy_per_flop_fp32_j = dsp_fabric.energy_per_op_fp32_pj * 1e-12
        scaling = _energy_scaling_from_yaml(dsp_fabric.energy_scaling)
        fabric_type = overrides.get(dsp_fabric.fabric_kind) or \
            _FABRIC_TYPE_BY_KIND[dsp_fabric.fabric_kind]
        compute_fabrics.append(ComputeFabric(
            fabric_type=fabric_type,
            circuit_type=_CIRCUIT_TYPE_DEFAULT,
            num_units=dsp_fabric.num_units,
            ops_per_unit_per_clock=ops_dict,
            core_frequency_hz=default_clock_hz,
            process_node_nm=process_node_nm,
            energy_per_flop_fp32=energy_per_flop_fp32_j,
            energy_scaling=scaling,
        ))

    if not compute_fabrics:
        raise DSPYamlLoaderError(
            f"SKU {base_id!r}: no DSPComputeFabric produced a valid "
            f"ComputeFabric (check ops_per_unit_per_clock entries)."
        )

    # ------------------------------------------------------------------
    # PrecisionProfile dict -- chip-wide peak ops/sec per precision.
    # Honors the YAML's theoretical_performance roll-up when present
    # (rather than re-deriving from fabrics) so multi-fabric SKUs can
    # express precision peaks that span fabrics correctly.
    # ------------------------------------------------------------------
    yaml_peaks = block.theoretical_performance.peak_ops_per_sec_by_precision
    precision_profiles: dict[Precision, PrecisionProfile] = {}

    # First pass: emit profiles from the YAML roll-up
    for name, peak in yaml_peaks.items():
        prec = _PRECISION_BY_NAME.get(name.lower())
        if prec is None or peak <= 0:
            continue
        # Relative speedup vs default precision (INT8 for most DSPs,
        # FP32 for signal-processing legacy DSPs).
        baseline_name = (block.default_precision or "int8").lower()
        baseline_peak = yaml_peaks.get(baseline_name, peak)
        relative_speedup = peak / baseline_peak if baseline_peak > 0 else 1.0
        precision_profiles[prec] = PrecisionProfile(
            precision=prec,
            peak_ops_per_sec=peak,
            tensor_core_supported=False,   # DSPs are vector/SIMD, not tensor
            relative_speedup=relative_speedup,
            bytes_per_element=_BYTES_PER_PRECISION.get(prec, 4),
        )

    # Same guard as DPU/TPU loader fix (graphs#202): empty profiles
    # would crash on default_precision selection; raise a clear error.
    if not precision_profiles:
        raise DSPYamlLoaderError(
            f"SKU {base_id!r}: no precision_profiles produced. The YAML's "
            f"theoretical_performance.peak_ops_per_sec_by_precision is "
            f"empty or contains only unknown precision names."
        )

    # ------------------------------------------------------------------
    # ThermalOperatingPoint per chip-level thermal_profile
    # ------------------------------------------------------------------
    thermal_operating_points: dict[str, ThermalOperatingPoint] = {}
    for profile in cp.power.thermal_profiles:
        thermal_operating_points[profile.name] = ThermalOperatingPoint(
            name=profile.name,
            tdp_watts=profile.tdp_watts,
            cooling_solution=profile.cooling_solution_id,
            performance_specs={},
        )

    # ------------------------------------------------------------------
    # Memory + cache
    # ------------------------------------------------------------------
    mem = block.memory
    # peak_bandwidth: external DRAM if present (the canonical bandwidth
    # surface for DSP roofline analysis). For IP cores this is the
    # typical-integration estimate; for SoCs it's measured.
    if mem.has_external_dram and mem.external_dram_bandwidth_gbps is not None:
        peak_bandwidth_bps = mem.external_dram_bandwidth_gbps * 1e9
    elif mem.l2_bandwidth_gbps is not None and mem.l2_bandwidth_gbps > 0:
        peak_bandwidth_bps = mem.l2_bandwidth_gbps * 1e9
    else:
        # Last-resort fallback: 0 bandwidth (loader has nothing to use)
        peak_bandwidth_bps = 0.0

    # External DRAM -> main_memory.
    if mem.has_external_dram and mem.external_dram_size_gb is not None:
        main_memory_bytes = int(mem.external_dram_size_gb * 1024**3)
    else:
        main_memory_bytes = 0

    # L1 / L2 from the DSP memory subsystem.
    l1_per_unit_bytes = mem.l1_size_bytes_per_unit
    l2_total_bytes = mem.l2_size_bytes_total or 0

    # Memory technology label and per-byte access energy.
    if mem.has_external_dram and mem.external_dram_type is not None:
        memory_technology = mem.external_dram_type.value.upper()
        read_pj = mem.external_dram_access_energy_pj_per_byte
        write_pj = read_pj * 1.2   # standard write-vs-read ratio
    else:
        memory_technology = "on-chip SRAM (no external DRAM)"
        # Use a conservative on-chip energy default if no DRAM
        read_pj = 1.0
        write_pj = 1.0

    # ------------------------------------------------------------------
    # Assemble HardwareResourceModel
    # ------------------------------------------------------------------
    default_precision = _resolve_default_precision(
        block.default_precision, precision_profiles,
    )

    # Energy fields: use the first fabric.
    first_fabric = compute_fabrics[0]
    energy_per_flop_fp32 = first_fabric.energy_per_flop_fp32

    # Chip-level energy_scaling: merge across fabrics (last write wins
    # for overlapping precisions -- fine in practice since fabrics
    # don't overlap on the same precision in canonical DSP designs).
    energy_scaling: dict[Precision, float] = {}
    for fabric in compute_fabrics:
        energy_scaling.update(fabric.energy_scaling)

    # compute_units uses the FIRST fabric's num_units (the canonical
    # "main" fabric). Multi-fabric SKUs iterate compute_fabrics directly
    # for per-fabric detail.
    compute_units = first_fabric.num_units

    # threads_per_unit / warps_per_unit / warp_size: DSPs don't have a
    # canonical SIMT structure. Match the hand-coded factory's
    # convention (threads_per_unit=4, warps_per_unit=1, warp_size matches
    # num_units for the "wavefront-as-fabric" model).
    threads_per_unit = block.wave_quantization
    warps_per_unit = 1
    warp_size = compute_units

    model = HardwareResourceModel(
        name=name_override or cp.name,
        hardware_type=HardwareType.DSP,

        compute_fabrics=compute_fabrics,

        compute_units=compute_units,
        threads_per_unit=threads_per_unit,
        warps_per_unit=warps_per_unit,
        warp_size=warp_size,

        thermal_operating_points=thermal_operating_points,
        default_thermal_profile=cp.power.default_thermal_profile,

        precision_profiles=precision_profiles,
        default_precision=default_precision,

        peak_bandwidth=peak_bandwidth_bps,
        l1_cache_per_unit=l1_per_unit_bytes,
        l2_cache_total=l2_total_bytes,
        main_memory=main_memory_bytes,

        energy_per_flop_fp32=energy_per_flop_fp32,
        energy_per_byte=read_pj * 1e-12,
        energy_scaling=energy_scaling,

        min_occupancy=block.min_occupancy,
        max_concurrent_kernels=block.max_concurrent_kernels,
        wave_quantization=block.wave_quantization,

        # Memory-subsystem M-layer fields
        coherence_protocol=mem.coherence_protocol,
        memory_technology=memory_technology,
        memory_read_energy_per_byte_pj=read_pj,
        memory_write_energy_per_byte_pj=write_pj,
    )

    # Generic provenance for the YAML-loaded fields
    yaml_provenance = EstimationConfidence.theoretical(
        score=0.75,
        source=f"Loaded from compute_products YAML ({base_id})",
    )
    for key in (
        "l1_cache_per_unit", "l2_cache_total", "main_memory",
        "peak_bandwidth", "coherence_protocol",
        "memory_technology",
        "memory_read_energy_per_byte_pj",
        "memory_write_energy_per_byte_pj",
    ):
        model.set_provenance(key, yaml_provenance)

    fabric_provenance = EstimationConfidence.theoretical(
        score=0.55,
        source=(
            f"Loaded from compute_products YAML ({base_id}): per-fabric "
            f"ops_per_unit_per_clock. DSP deployment_kind="
            f"{block.deployment_kind.value}"
            + (
                f", external_dram_bandwidth_kind="
                f"{mem.external_dram_bandwidth_kind!r}"
                if mem.has_external_dram else ""
            )
        ),
    )
    for prec in precision_profiles:
        model.set_provenance(
            f"compute_fabric.ops_per_clock.{prec.value}",
            fabric_provenance,
        )

    return model


# ---------------------------------------------------------------------------
# Convenience helpers for known SKUs (re-exported by the per-SKU
# factory wrappers after PR 5 cleanup)
# ---------------------------------------------------------------------------

def is_dsp_deployment_kind(
    cp: ComputeProduct, kind: DSPDeploymentKind,
) -> bool:
    """Predicate: does the ComputeProduct carry a DSPBlock with the
    given deployment_kind? Useful for catalog filters."""
    try:
        block = _dsp_block(cp)
    except DSPYamlLoaderError:
        return False
    return block.deployment_kind == kind
