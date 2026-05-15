"""GPU resource model loader from embodied-schemas ComputeProduct YAMLs.

PR 4 of the GPU sprint scoped at issue #171. Mirrors the KPU loader at
``src/graphs/hardware/models/accelerators/kpu_yaml_loader.py`` but
reads a GPUBlock-bearing ``ComputeProduct`` and builds a
``HardwareResourceModel`` with the GPU-shaped fields populated.

Sibling to the hand-coded factories in ``src/graphs/hardware/models/edge/``
(``jetson_orin_agx_64gb.py``, etc.) during the parallel-migration
phase. Adding ``create_jetson_orin_agx_64gb_from_yaml_mapper()`` is
PR 5; this PR ships the loader machinery and a parity test that
proves the YAML-loaded model matches the hand-coded one's key fields.

Scope of GPUBlock -> HardwareResourceModel mapping (one-to-one
unless noted):

  GPUBlock.num_sms                  -> compute_units
  GPUBlock.cuda_cores_per_sm        -> cuda_cores_per_sm
  GPUBlock.tensor_cores_per_sm      -> tensor_cores_per_sm
  GPUBlock.threads_per_sm           -> threads_per_unit
  GPUBlock.warps_per_sm             -> warps_per_unit
  GPUBlock.warp_size                -> warp_size
  GPUBlock.compute_fabrics[]        -> compute_fabrics[]
  GPUBlock.memory.{lpddr5, l1, l2,  -> peak_bandwidth, l1_cache_per_unit,
    l3, coherence, energy/byte}        l2_cache_total, l3_present, etc.
  GPUBlock.noc                      -> soc_fabric (SoCFabricModel)
  GPUBlock.{occupancy, kernels,     -> min_occupancy,
            wave_quantization}         max_concurrent_kernels,
                                       wave_quantization
  ComputeProduct.power.thermal_     -> thermal_operating_points
    profiles[]
  Roll-up performance               -> precision_profiles

Out of scope for v2 (deferred to v3):
  - BOMCostProfile (the YAML doesn't carry BOM cost yet; v3 adds
    Market.bom optional field)
  - Per-field provenance copy-through (the hand-coded factory uses
    set_provenance() with rich source strings; the loader could attach
    a generic "loaded from compute_products YAML" provenance but that's
    cosmetic for v2)
  - GPU-specific thermal profile structures (per-precision DVFS lives
    on GPUBlock.compute_fabrics; the chip-level thermal_profiles in
    Power use KPUThermalProfile shape and carry only the scalar
    clock_mhz + per-precision efficiency_factor)
"""

from __future__ import annotations

from typing import Optional

from embodied_schemas.compute_product import ComputeProduct
from embodied_schemas.gpu_block import (
    GPUBlock,
    GPUFabricKind,
    GPUL1Kind,
    GPUL2Topology,
    GPUNoCTopology,
)
from embodied_schemas.loaders import load_compute_products
from embodied_schemas.process_node import ProcessNodeEntry

from graphs.core.confidence import EstimationConfidence
from graphs.hardware.fabric_model import SoCFabricModel, Topology
from graphs.hardware.resource_model import (
    ClockDomain,
    ComputeFabric,
    ComputeResource,
    HardwareResourceModel,
    HardwareType,
    PerformanceCharacteristics,
    Precision,
    PrecisionProfile,
    ThermalOperatingPoint,
)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class GPUYamlLoaderError(Exception):
    """Raised when a ComputeProduct YAML can't be turned into a
    GPU HardwareResourceModel (missing block, unsupported topology,
    unresolved references, etc.)."""


# ---------------------------------------------------------------------------
# Mapping tables (YAML enums / strings -> graphs internal types)
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
    "int4": Precision.INT4,
}

_BYTES_PER_PRECISION: dict[Precision, float] = {
    Precision.FP64: 8, Precision.FP32: 4, Precision.TF32: 4,
    Precision.FP16: 2, Precision.BF16: 2,
    Precision.FP8_E4M3: 1, Precision.FP8_E5M2: 1,
    Precision.INT8: 1, Precision.INT4: 0.5,
}

# GPUFabricKind -> ComputeFabric.fabric_type / circuit_type
_FABRIC_TYPE_BY_KIND: dict[GPUFabricKind, str] = {
    GPUFabricKind.CUDA_CORE: "cuda_core",
    GPUFabricKind.TENSOR_CORE: "tensor_core",
    GPUFabricKind.RT_CORE: "rt_core",
}

_CIRCUIT_TYPE_BY_KIND: dict[GPUFabricKind, str] = {
    GPUFabricKind.CUDA_CORE: "standard_cell",
    GPUFabricKind.TENSOR_CORE: "tensor_core",
    GPUFabricKind.RT_CORE: "tensor_core",  # RT cores share the dense-MAC family
}

# GPUNoCTopology -> graphs SoCFabricModel Topology enum. graphs has no
# HIERARCHICAL value (it's a fabric *family* rather than a primitive
# topology), so map it to CLOS as the closest 2-level approximation.
# When a future SKU actually needs HIERARCHICAL the graphs Topology
# enum can grow that value.
_TOPOLOGY_BY_KIND: dict[GPUNoCTopology, Topology] = {
    GPUNoCTopology.CROSSBAR: Topology.CROSSBAR,
    GPUNoCTopology.MESH_2D: Topology.MESH_2D,
    GPUNoCTopology.RING: Topology.RING,
    GPUNoCTopology.HIERARCHICAL: Topology.CLOS,
}

# Memory technology -> dram per-byte read / write energy (pJ).
# Values mirror the KPU loader's _MEM_PJ_PER_BYTE table; write energy
# typically ~1.2x read for LPDDR5 / DDR5.
_DRAM_READ_PJ_PER_BYTE: dict[str, float] = {
    "lpddr5": 15.0, "lpddr5x": 13.0, "lpddr4": 18.0, "lpddr4x": 16.0,
    "ddr5": 13.0, "hbm2": 7.0, "hbm2e": 6.5,
    "hbm3": 6.0, "hbm3e": 5.5, "gddr6": 8.0, "gddr6x": 7.5,
}
_DRAM_WRITE_RATIO = 1.2  # write energy slightly above read for LPDDR/DDR


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gpu_block(cp: ComputeProduct) -> GPUBlock:
    """Pick the (single) GPUBlock from a ComputeProduct's first die."""
    for block in cp.dies[0].blocks:
        if isinstance(block, GPUBlock):
            return block
    raise GPUYamlLoaderError(
        f"ComputeProduct {cp.id!r} has no GPUBlock in dies[0].blocks "
        f"(found: {[type(b).__name__ for b in cp.dies[0].blocks]})"
    )


def _precisions_from_ops_dict(
    ops: dict[str, int],
    *,
    source: str = "ops_per_unit_per_clock",
) -> dict[Precision, int]:
    """Convert YAML's str-keyed ops/clock dict to graphs' Precision-keyed dict.

    Fails fast on unknown precision names rather than silently dropping
    them -- a typo in the YAML (e.g. ``"fp_16"`` instead of ``"fp16"``)
    would otherwise undercount the fabric's true ops/clock and produce
    a model that mysteriously misses a precision."""
    out: dict[Precision, int] = {}
    unknown: list[str] = []
    for name, count in ops.items():
        prec = _PRECISION_BY_NAME.get(name.lower())
        if prec is None:
            unknown.append(name)
            continue
        out[prec] = count
    if unknown:
        raise GPUYamlLoaderError(
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
        raise GPUYamlLoaderError(
            f"unknown precision name(s) in {source}: {sorted(unknown)}. "
            f"Known: {sorted(_PRECISION_BY_NAME)}"
        )
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_gpu_resource_model_from_yaml(
    base_id: str,
    *,
    products: Optional[dict[str, ComputeProduct]] = None,
    process_nodes: Optional[dict[str, ProcessNodeEntry]] = None,
    name_override: Optional[str] = None,
) -> HardwareResourceModel:
    """Build a ``HardwareResourceModel`` for ``base_id`` from a GPU
    ComputeProduct YAML.

    Args:
        base_id: ComputeProduct id, e.g., "nvidia_jetson_agx_orin_64gb".
        products / process_nodes: optional pre-loaded catalogs (tests
            pass in-memory dicts to avoid disk I/O). Defaults load from
            the installed ``embodied-schemas`` package.
        name_override: optional override for the resource model's
            ``name`` field. The hand-coded Jetson factories use
            "Jetson-Orin-AGX-64GB" (no spaces); the YAML uses "NVIDIA
            Jetson AGX Orin 64GB". Pass the legacy name here when
            doing parity-testing against a hand-coded model.

    Raises:
        GPUYamlLoaderError: when ``base_id`` is not in the catalog,
            doesn't carry a GPUBlock, or has unresolvable references.
    """
    if products is None:
        products = load_compute_products()
    if process_nodes is None:
        from embodied_schemas.loaders import load_process_nodes
        process_nodes = load_process_nodes()

    cp = products.get(base_id)
    if cp is None:
        raise GPUYamlLoaderError(
            f"no ComputeProduct with id={base_id!r}. Available: "
            f"{', '.join(sorted(products))}"
        )
    block = _gpu_block(cp)

    node = process_nodes.get(cp.dies[0].process_node_id)
    if node is None:
        raise GPUYamlLoaderError(
            f"SKU {base_id!r} references process_node_id="
            f"{cp.dies[0].process_node_id!r} which does not resolve"
        )
    process_node_nm = int(node.node_nm)

    # ------------------------------------------------------------------
    # Default profile -> sustained clock used as fabric core_frequency_hz
    # ------------------------------------------------------------------
    default_profile = next(
        (p for p in cp.power.thermal_profiles
         if p.name == cp.power.default_thermal_profile),
        None,
    )
    if default_profile is None:
        raise GPUYamlLoaderError(
            f"SKU {base_id!r}: default_thermal_profile "
            f"{cp.power.default_thermal_profile!r} is not in thermal_profiles"
        )
    default_clock_hz = default_profile.clock_mhz * 1e6
    boost_clock_hz = cp.dies[0].clocks.boost_clock_mhz * 1e6
    base_clock_hz = cp.dies[0].clocks.base_clock_mhz * 1e6

    # ------------------------------------------------------------------
    # ComputeFabric per GPUComputeFabric (CUDA core + Tensor core)
    # ------------------------------------------------------------------
    compute_fabrics: list[ComputeFabric] = []
    for gpu_fabric in block.compute_fabrics:
        ops_dict = _precisions_from_ops_dict(gpu_fabric.ops_per_unit_per_clock)
        if not ops_dict:
            continue
        compute_fabrics.append(ComputeFabric(
            fabric_type=_FABRIC_TYPE_BY_KIND[gpu_fabric.fabric_kind],
            circuit_type=_CIRCUIT_TYPE_BY_KIND[gpu_fabric.fabric_kind],
            num_units=block.num_sms * gpu_fabric.units_per_sm,
            ops_per_unit_per_clock=ops_dict,
            core_frequency_hz=default_clock_hz,
            process_node_nm=process_node_nm,
            energy_per_flop_fp32=gpu_fabric.energy_per_flop_fp32_pj * 1e-12,
            energy_scaling=_energy_scaling_from_yaml(gpu_fabric.energy_scaling),
        ))

    if not compute_fabrics:
        raise GPUYamlLoaderError(
            f"SKU {base_id!r}: no GPUComputeFabric produced a valid "
            f"ComputeFabric (check ops_per_unit_per_clock entries)."
        )

    # ------------------------------------------------------------------
    # PrecisionProfile dict -- chip-wide peak ops/sec per precision
    # ------------------------------------------------------------------
    supported_precisions: set[Precision] = set()
    for fabric in compute_fabrics:
        supported_precisions.update(fabric.ops_per_unit_per_clock.keys())

    # Derive tensor-core-supported precisions from the actual tensor_core
    # fabric(s) instead of a hardcoded set. A SKU that doesn't ship Tensor
    # cores at all (e.g. a hypothetical CUDA-only entry-level GPU) gets
    # tensor_core_supported=False uniformly; a future RT_CORE-bearing SKU
    # would surface its precisions automatically.
    tensor_fabric_precisions: set[Precision] = set()
    for fabric in compute_fabrics:
        if fabric.circuit_type == "tensor_core":
            tensor_fabric_precisions.update(fabric.ops_per_unit_per_clock.keys())

    precision_profiles: dict[Precision, PrecisionProfile] = {}
    for precision in supported_precisions:
        peak = sum(f.get_peak_ops_per_sec(precision) for f in compute_fabrics)
        if peak <= 0:
            continue
        tensor_supported = precision in tensor_fabric_precisions
        # Relative speedup vs FP32 baseline (used by some downstream
        # consumers for quick comparisons).
        fp32_peak = sum(
            f.get_peak_ops_per_sec(Precision.FP32) for f in compute_fabrics
        )
        relative_speedup = peak / fp32_peak if fp32_peak > 0 else 1.0
        precision_profiles[precision] = PrecisionProfile(
            precision=precision,
            peak_ops_per_sec=peak,
            tensor_core_supported=tensor_supported,
            relative_speedup=relative_speedup,
            bytes_per_element=_BYTES_PER_PRECISION.get(precision, 4),
            accumulator_precision=(
                Precision.INT32 if precision == Precision.INT8 else
                Precision.FP32 if precision in {Precision.FP16, Precision.BF16} else
                None
            ),
        )

    # ------------------------------------------------------------------
    # ThermalOperatingPoint per chip-level thermal_profile
    # ------------------------------------------------------------------
    thermal_operating_points: dict[str, ThermalOperatingPoint] = {}
    for profile in cp.power.thermal_profiles:
        profile_clock_hz = profile.clock_mhz * 1e6
        clock_domain = ClockDomain(
            base_clock_hz=base_clock_hz,
            max_boost_clock_hz=boost_clock_hz,
            sustained_clock_hz=profile_clock_hz,
            dvfs_enabled=True,
        )
        # Per-profile ComputeResource -- single resource across all SMs,
        # carries the profile-specific clock for accurate calc_*_ops().
        ops_per_sm_per_clock: dict[Precision, int] = {}
        for fabric in block.compute_fabrics:
            fabric_ops = _precisions_from_ops_dict(
                fabric.ops_per_unit_per_clock
            )
            for prec, n in fabric_ops.items():
                ops_per_sm_per_clock[prec] = (
                    ops_per_sm_per_clock.get(prec, 0)
                    + fabric.units_per_sm * n
                )
        compute_resource = ComputeResource(
            resource_type="GPU-SM",
            num_units=block.num_sms,
            ops_per_unit_per_clock=ops_per_sm_per_clock,
            clock_domain=clock_domain,
        )

        eff_by_prec = profile.efficiency_factor_by_precision or {}
        performance_specs: dict[Precision, PerformanceCharacteristics] = {}
        for precision in supported_precisions:
            performance_specs[precision] = PerformanceCharacteristics(
                precision=precision,
                compute_resource=compute_resource,
                efficiency_factor=eff_by_prec.get(precision.value, 0.50),
                native_acceleration=True,
            )

        thermal_operating_points[profile.name] = ThermalOperatingPoint(
            name=profile.name,
            tdp_watts=profile.tdp_watts,
            cooling_solution=profile.cooling_solution_id,
            performance_specs=performance_specs,
        )

    # ------------------------------------------------------------------
    # Memory + cache
    # ------------------------------------------------------------------
    mem = block.memory
    peak_bandwidth_bps = mem.memory_bandwidth_gbps * 1e9
    main_memory_bytes = int(mem.memory_size_gb * 1024**3)

    # Resource model expects bytes; YAML carries KiB.
    l1_per_sm_bytes = mem.l1_kib_per_sm * 1024
    l2_total_bytes = mem.l2_total_kib * 1024
    l3_total_bytes = mem.l3_total_kib * 1024 if mem.l3_present else 0

    # L1 storage kind: GPU YAML uses an enum; resource model expects a
    # short string.
    l1_storage_kind = {
        GPUL1Kind.CACHE: "cache",
        GPUL1Kind.SCRATCHPAD: "scratchpad",
        GPUL1Kind.UNIFIED: "cache",  # unified L1 is dominantly cache-mode
    }[mem.l1_kind]

    l2_topology = {
        GPUL2Topology.SHARED_LLC: "shared-llc",
        GPUL2Topology.BANKED: "banked",
        GPUL2Topology.DISTRIBUTED: "distributed",
    }[mem.l2_topology]

    # Per-SM L2 share: total L2 / num_sms (matches the hand-coded model).
    l2_per_sm_bytes = l2_total_bytes // block.num_sms

    # Memory energy: prefer YAML-declared, fall back to table.
    mem_tech = mem.memory_type.value.lower()
    read_pj = (
        mem.read_energy_pj_per_byte
        if mem.read_energy_pj_per_byte is not None
        else _DRAM_READ_PJ_PER_BYTE.get(mem_tech, 12.0)
    )
    write_pj = (
        mem.write_energy_pj_per_byte
        if mem.write_energy_pj_per_byte is not None
        else read_pj * _DRAM_WRITE_RATIO
    )

    # ------------------------------------------------------------------
    # SoC fabric
    # ------------------------------------------------------------------
    soc_fabric = SoCFabricModel(
        topology=_TOPOLOGY_BY_KIND[block.noc.topology],
        hop_latency_ns=block.noc.hop_latency_ns,
        pj_per_flit_per_hop=block.noc.pj_per_flit_per_hop,
        bisection_bandwidth_gbps=block.noc.bisection_bandwidth_gbps,
        controller_count=block.noc.controller_count,
        flit_size_bytes=block.noc.flit_size_bytes,
        routing_distance_factor=block.noc.routing_distance_factor,
        provenance=(
            f"Loaded from compute_products YAML "
            f"({base_id}.dies[0].blocks[0].noc)"
        ),
    )

    # ------------------------------------------------------------------
    # Assemble HardwareResourceModel
    # ------------------------------------------------------------------
    # Default precision: prefer INT8 if supported, else FP16, else FP32.
    if Precision.INT8 in precision_profiles:
        default_precision = Precision.INT8
    elif Precision.FP16 in precision_profiles:
        default_precision = Precision.FP16
    else:
        default_precision = Precision.FP32

    # Energy fields: legacy scalar fields used by older code paths.
    # The CUDA-core fabric is the FP32 baseline; if it's not present
    # (some future GPU might ship tensor-only) fall back to the first
    # fabric.
    cuda_fabric = next(
        (f for f in compute_fabrics if f.circuit_type == "standard_cell"),
        compute_fabrics[0],
    )
    energy_per_flop_fp32 = cuda_fabric.energy_per_flop_fp32

    # Model-level energy_scaling rolls up the YAML's per-fabric scaling.
    # If multiple fabrics report scaling for the same precision (e.g. CUDA
    # cores AND tensor cores both scale FP16), prefer the tensor-core
    # entry since it's the lower-energy / higher-throughput path that
    # downstream estimators use for that precision. CUDA-only precisions
    # (FP64, FP32) come from the CUDA fabric.
    energy_scaling: dict[Precision, float] = {}
    # First populate from CUDA / standard fabrics, then overwrite with
    # tensor-core values where present (tensor-core-friendly precisions
    # win the tie).
    for fabric in compute_fabrics:
        if fabric.circuit_type != "tensor_core":
            energy_scaling.update(fabric.energy_scaling)
    for fabric in compute_fabrics:
        if fabric.circuit_type == "tensor_core":
            energy_scaling.update(fabric.energy_scaling)

    model = HardwareResourceModel(
        name=name_override or cp.name,
        hardware_type=HardwareType.GPU,

        compute_fabrics=compute_fabrics,

        compute_units=block.num_sms,
        threads_per_unit=block.threads_per_sm,
        warps_per_unit=block.warps_per_sm,
        warp_size=block.warp_size,

        cuda_cores_per_sm=block.cuda_cores_per_sm,
        tensor_cores_per_sm=block.tensor_cores_per_sm,
        ops_per_clock_per_core=2.0,  # FMA = 2 ops/clock (FP32 baseline)
        sm_boost_clock_hz=boost_clock_hz,
        sm_sustained_clock_hz=default_clock_hz,

        thermal_operating_points=thermal_operating_points,
        default_thermal_profile=cp.power.default_thermal_profile,

        precision_profiles=precision_profiles,
        default_precision=default_precision,

        peak_bandwidth=peak_bandwidth_bps,
        l1_cache_per_unit=l1_per_sm_bytes,
        l2_cache_total=l2_total_bytes,
        main_memory=main_memory_bytes,

        energy_per_flop_fp32=energy_per_flop_fp32,
        energy_per_byte=read_pj * 1e-12,
        energy_scaling=energy_scaling,

        min_occupancy=block.min_occupancy,
        max_concurrent_kernels=block.max_concurrent_kernels,
        wave_quantization=block.wave_quantization,

        l1_storage_kind=l1_storage_kind,
        l2_cache_per_unit=l2_per_sm_bytes,
        l2_topology=l2_topology,
        l3_present=mem.l3_present,
        l3_cache_total=l3_total_bytes,
        coherence_protocol=mem.coherence_protocol,
        memory_technology=mem.memory_type.value.upper(),
        memory_read_energy_per_byte_pj=read_pj,
        memory_write_energy_per_byte_pj=write_pj,
        soc_fabric=soc_fabric,
    )

    # Generic provenance for the YAML-loaded fields. The hand-coded
    # factories attach richer per-field provenance with whitepaper
    # citations; that level of detail will move into the YAML in a
    # future schema PR (per-field provenance is orthogonal to this
    # sprint, see graphs#179 design doc section "Out of scope").
    yaml_provenance = EstimationConfidence.theoretical(
        score=0.80,
        source=f"Loaded from compute_products YAML ({base_id})",
    )
    for key in (
        "l1_cache_per_unit", "l1_storage_kind",
        "l2_cache_per_unit", "l2_topology",
        "l3_present", "l3_cache_total", "coherence_protocol",
        "memory_technology",
        "memory_read_energy_per_byte_pj",
        "memory_write_energy_per_byte_pj",
        "soc_fabric",
    ):
        model.set_provenance(key, yaml_provenance)

    return model
