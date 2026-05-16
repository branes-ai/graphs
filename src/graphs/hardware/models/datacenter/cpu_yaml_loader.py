"""CPU resource model loader from embodied-schemas ComputeProduct YAMLs.

PR 4 of the CPU sprint scoped at issue #182. Mirrors
``src/graphs/hardware/models/edge/gpu_yaml_loader.py`` but reads a
CPUBlock-bearing ``ComputeProduct`` and builds a
``HardwareResourceModel`` with the CPU-shaped fields populated.

Sibling to the hand-coded factories in
``src/graphs/hardware/models/{edge,datacenter}/`` during the parallel-
migration phase. Adding ``create_i7_12700k_from_yaml_mapper()`` is
PR 5; this PR ships the loader machinery and a parity test that
proves the YAML-loaded model matches the hand-coded one's key fields.

Scope of CPUBlock -> HardwareResourceModel mapping (one-to-one
unless noted):

  CPUBlock.core_clusters[*].compute_fabrics[*]
                                  -> compute_fabrics[] (flattened
                                     across all clusters; num_units
                                     = cluster.num_cores)
  CPUBlock.total_effective_cores  -> compute_units
  CPUBlock.simd_width_lanes       -> warp_size
  CPUBlock.simd_efficiency_by_op_kind
                                  -> simd_efficiency
  CPUBlock.memory.{ddr, l3, coherence, energy/byte}
                                  -> peak_bandwidth, l2_cache_total (=LLC),
                                     l3_present, l3_cache_total,
                                     coherence_protocol, memory_technology,
                                     read/write energy
  CPUBlock.core_clusters[*].{l1_kib_per_core, l2_kib_per_core/_shared}
                                  -> l1_cache_per_unit (P-cluster value),
                                     l2_cache_per_unit (physical L2 total /
                                     effective_cores)
  CPUBlock.noc                    -> soc_fabric (SoCFabricModel)
  CPUBlock.{min_occupancy, max_concurrent_threads, wave_quantization}
                                  -> matching HardwareResourceModel fields
  ComputeProduct.power.thermal_profiles[]
                                  -> thermal_operating_points
  Roll-up performance             -> precision_profiles

Out of scope for v3 (deferred to v4):
  - BOMCostProfile (YAML doesn't carry; v4 Market.bom)
  - Per-field provenance copy-through with whitepaper citations
  - Per-cluster ClockDomain in thermal profiles (chip-level KPUThermalProfile
    carries only the scalar clock_mhz today; the v3 CPUBlock already
    has CPUThermalProfile.per_cluster_clock_domain but it lives at the
    block level, not the chip Power.thermal_profiles level)
"""

from __future__ import annotations

from typing import Optional

from embodied_schemas.compute_product import ComputeProduct
from embodied_schemas.cpu_block import (
    CoreClusterKind,
    CPUBlock,
    CPUISAExtension,
    CPUNoCTopology,
    L2Layout,
)
from embodied_schemas.loaders import load_compute_products
from embodied_schemas.process_node import ProcessNodeEntry

from graphs.core.confidence import EstimationConfidence
from graphs.hardware.fabric_model import SoCFabricModel, Topology
from graphs.hardware.resource_model import (
    ComputeFabric,
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

class CPUYamlLoaderError(Exception):
    """Raised when a ComputeProduct YAML can't be turned into a CPU
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
    "int4": Precision.INT4,
}

_BYTES_PER_PRECISION: dict[Precision, float] = {
    Precision.FP64: 8, Precision.FP32: 4, Precision.TF32: 4,
    Precision.FP16: 2, Precision.BF16: 2,
    Precision.FP8_E4M3: 1, Precision.FP8_E5M2: 1,
    Precision.INT8: 1, Precision.INT4: 0.5,
}

# CPUISAExtension -> ComputeFabric.fabric_type / circuit_type.
# CPUs identify fabrics by ISA extension; map each to a short string
# the downstream graphs mappers consume. The simd_packed circuit_type
# is the right choice for AVX2/AVX-512/AVX-VNNI/NEON/SVE (vector
# datapath); AMX is a matrix engine and gets tensor_core.
_FABRIC_TYPE_BY_ISA: dict[CPUISAExtension, str] = {
    CPUISAExtension.SCALAR_X86:  "scalar_x86",
    CPUISAExtension.SSE2:        "sse2",
    CPUISAExtension.AVX2:        "avx2",
    CPUISAExtension.AVX_VNNI:    "avx_vnni",
    CPUISAExtension.AVX512:      "avx512",
    CPUISAExtension.AVX512_VNNI: "avx512_vnni",
    CPUISAExtension.AVX512_BF16: "avx512_bf16",
    CPUISAExtension.AVX512_FP16: "avx512_fp16",
    CPUISAExtension.AMX_TILE:    "amx_tile",
    CPUISAExtension.AMX_BF16:    "amx_bf16",
    CPUISAExtension.AMX_INT8:    "amx_int8",
    CPUISAExtension.SCALAR_ARM:  "scalar_arm",
    CPUISAExtension.NEON:        "neon",
    CPUISAExtension.SVE:         "sve",
    CPUISAExtension.SVE2:        "sve2",
    CPUISAExtension.SVE_BF16:    "sve_bf16",
    CPUISAExtension.SVE_INT8:    "sve_int8",
    CPUISAExtension.SME:         "sme",
}

_CIRCUIT_TYPE_BY_ISA: dict[CPUISAExtension, str] = {
    # AMX is a matrix engine -> tensor_core family
    CPUISAExtension.AMX_TILE:    "tensor_core",
    CPUISAExtension.AMX_BF16:    "tensor_core",
    CPUISAExtension.AMX_INT8:    "tensor_core",
    CPUISAExtension.SME:         "tensor_core",  # ARM SME is matrix-shaped
}
# Default for everything else (vector ISAs)
_DEFAULT_CIRCUIT_TYPE = "simd_packed"

# CPUNoCTopology -> graphs SoCFabricModel Topology enum.
# graphs has no DOUBLE_RING or IO_DIE_PLUS_CCD or INFINITY_FABRIC
# in its Topology enum; map them to the closest available value
# with a comment so the v4 Topology extension can absorb them.
_TOPOLOGY_BY_KIND: dict[CPUNoCTopology, Topology] = {
    CPUNoCTopology.RING:             Topology.RING,
    CPUNoCTopology.DOUBLE_RING:      Topology.RING,   # graphs has no DOUBLE_RING
    CPUNoCTopology.MESH_2D:          Topology.MESH_2D,
    CPUNoCTopology.IO_DIE_PLUS_CCD:  Topology.CLOS,   # closest 2-level approx
    CPUNoCTopology.INFINITY_FABRIC:  Topology.CLOS,   # closest 2-level approx
}

# Memory technology -> dram per-byte energy (pJ). Wider table than GPU
# loader's because CPUs ship a broader range of DRAM types.
_DRAM_READ_PJ_PER_BYTE: dict[str, float] = {
    "lpddr5": 15.0, "lpddr5x": 13.0, "lpddr4": 18.0, "lpddr4x": 16.0,
    "ddr5": 25.0, "ddr4": 28.0,
    "hbm2": 7.0, "hbm2e": 6.5, "hbm3": 6.0, "hbm3e": 5.5,
    "gddr6": 8.0, "gddr6x": 7.5,
}
_DRAM_WRITE_RATIO = 1.2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cpu_block(cp: ComputeProduct) -> CPUBlock:
    """Pick the (single) CPUBlock from a ComputeProduct's first die."""
    for block in cp.dies[0].blocks:
        if isinstance(block, CPUBlock):
            return block
    raise CPUYamlLoaderError(
        f"ComputeProduct {cp.id!r} has no CPUBlock in dies[0].blocks "
        f"(found: {[type(b).__name__ for b in cp.dies[0].blocks]})"
    )


def _precisions_from_ops_dict(
    ops: dict[str, int],
    *,
    source: str = "ops_per_core_per_clock",
) -> dict[Precision, int]:
    """Convert YAML's str-keyed ops/clock dict to graphs' Precision-keyed dict.

    Fails fast on unknown precision names rather than silently dropping
    them (same defensive pattern as gpu_yaml_loader's PR #180 fix)."""
    out: dict[Precision, int] = {}
    unknown: list[str] = []
    for name, count in ops.items():
        prec = _PRECISION_BY_NAME.get(name.lower())
        if prec is None:
            unknown.append(name)
            continue
        out[prec] = count
    if unknown:
        raise CPUYamlLoaderError(
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
        raise CPUYamlLoaderError(
            f"unknown precision name(s) in {source}: {sorted(unknown)}. "
            f"Known: {sorted(_PRECISION_BY_NAME)}"
        )
    return out


def _physical_l2_total_kib(block: CPUBlock) -> int:
    """Sum physical L2 capacity across all clusters.

    PRIVATE_PER_CORE clusters contribute ``num_cores * l2_kib_per_core``;
    SHARED_PER_CLUSTER clusters contribute ``l2_kib_shared`` (regardless
    of how many cores share it); SHARED_GLOBAL clusters contribute 0
    (the L2 is reported on CPUMemorySubsystem and counted separately)."""
    total = 0
    for cluster in block.core_clusters:
        if cluster.l2_layout == L2Layout.PRIVATE_PER_CORE:
            total += cluster.num_cores * cluster.l2_kib_per_core
        elif cluster.l2_layout == L2Layout.SHARED_PER_CLUSTER:
            total += cluster.l2_kib_shared
        # SHARED_GLOBAL: nothing here
    return total


def _pick_p_cluster(block: CPUBlock):
    """Pick the cluster that represents the 'P-core' for legacy field
    reporting. Preference order: PERFORMANCE > BIG > HOMOGENEOUS >
    EFFICIENT > LITTLE > first."""
    pref_order = [
        CoreClusterKind.PERFORMANCE,
        CoreClusterKind.BIG,
        CoreClusterKind.HOMOGENEOUS,
        CoreClusterKind.EFFICIENT,
        CoreClusterKind.LITTLE,
    ]
    by_kind = {c.cluster_kind: c for c in block.core_clusters}
    for kind in pref_order:
        if kind in by_kind:
            return by_kind[kind]
    return block.core_clusters[0]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_cpu_resource_model_from_yaml(
    base_id: str,
    *,
    products: Optional[dict[str, ComputeProduct]] = None,
    process_nodes: Optional[dict[str, ProcessNodeEntry]] = None,
    name_override: Optional[str] = None,
) -> HardwareResourceModel:
    """Build a ``HardwareResourceModel`` for ``base_id`` from a CPU
    ComputeProduct YAML.

    Args:
        base_id: ComputeProduct id, e.g., "intel_core_i7_12700k".
        products / process_nodes: optional pre-loaded catalogs.
        name_override: optional override for the resource model's
            ``name`` field. The hand-coded i7 factory uses
            "Intel-Core-i7-12700K"; the YAML uses "Intel Core
            i7-12700K". Pass the legacy name here for parity testing.

    Raises:
        CPUYamlLoaderError: when ``base_id`` is not in the catalog,
            doesn't carry a CPUBlock, or has unresolvable references.
    """
    if products is None:
        products = load_compute_products()
    if process_nodes is None:
        from embodied_schemas.loaders import load_process_nodes
        process_nodes = load_process_nodes()

    cp = products.get(base_id)
    if cp is None:
        raise CPUYamlLoaderError(
            f"no ComputeProduct with id={base_id!r}. Available: "
            f"{', '.join(sorted(products))}"
        )
    block = _cpu_block(cp)

    node = process_nodes.get(cp.dies[0].process_node_id)
    if node is None:
        raise CPUYamlLoaderError(
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
        raise CPUYamlLoaderError(
            f"SKU {base_id!r}: default_thermal_profile "
            f"{cp.power.default_thermal_profile!r} is not in thermal_profiles"
        )
    default_clock_hz = default_profile.clock_mhz * 1e6
    # Note: CPUs run different P-core / E-core clocks; here we use the
    # chip-level default profile's scalar clock_mhz for both clusters.
    # The v3 CPUBlock supports per-cluster ClockDomain via
    # CPUThermalProfile but that lives inside the block, not on the
    # chip-level Power.thermal_profiles. Per-cluster fabric frequency
    # is a v4 concern.

    # ------------------------------------------------------------------
    # ComputeFabric per (cluster, fabric) pair
    # ------------------------------------------------------------------
    compute_fabrics: list[ComputeFabric] = []
    for cluster in block.core_clusters:
        for cpu_fabric in cluster.compute_fabrics:
            ops_dict = _precisions_from_ops_dict(
                cpu_fabric.ops_per_core_per_clock,
                source=f"cluster.{cluster.cluster_kind.value}.compute_fabrics."
                       f"{cpu_fabric.isa_extension.value}",
            )
            if not ops_dict:
                continue
            compute_fabrics.append(ComputeFabric(
                fabric_type=_FABRIC_TYPE_BY_ISA[cpu_fabric.isa_extension],
                circuit_type=_CIRCUIT_TYPE_BY_ISA.get(
                    cpu_fabric.isa_extension, _DEFAULT_CIRCUIT_TYPE,
                ),
                num_units=cluster.num_cores,
                ops_per_unit_per_clock=ops_dict,
                core_frequency_hz=default_clock_hz,
                process_node_nm=process_node_nm,
                energy_per_flop_fp32=cpu_fabric.energy_per_flop_fp32_pj * 1e-12,
                energy_scaling=_energy_scaling_from_yaml(
                    cpu_fabric.energy_scaling,
                ),
            ))

    if not compute_fabrics:
        raise CPUYamlLoaderError(
            f"SKU {base_id!r}: no CPUComputeFabric produced a valid "
            f"ComputeFabric (check ops_per_core_per_clock entries)."
        )

    # ------------------------------------------------------------------
    # PrecisionProfile dict -- chip-wide peak ops/sec per precision
    # ------------------------------------------------------------------
    supported_precisions: set[Precision] = set()
    for fabric in compute_fabrics:
        supported_precisions.update(fabric.ops_per_unit_per_clock.keys())

    precision_profiles: dict[Precision, PrecisionProfile] = {}
    for precision in supported_precisions:
        peak = sum(f.get_peak_ops_per_sec(precision) for f in compute_fabrics)
        if peak <= 0:
            continue
        # Tensor-core support comes from any AMX/SME fabric exposing
        # this precision (matches the GPU loader's pattern for tensor
        # cores). AVX-VNNI is a fused MAC but not a "tensor core" in
        # the GPU sense; mark it False to align with the hand-coded
        # i7 model's convention (i7 sets tensor_core_supported=True
        # for INT8 because AVX-VNNI is fused -- match that for parity).
        tensor_supported = (
            precision == Precision.INT8 and any(
                f.fabric_type in {"avx_vnni", "avx512_vnni", "amx_int8",
                                   "amx_tile", "sve_int8"}
                for f in compute_fabrics
            )
        )
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
                Precision.INT16 if precision == Precision.INT4 else
                None
            ),
        )

    # ------------------------------------------------------------------
    # ThermalOperatingPoint per chip-level thermal_profile
    # ------------------------------------------------------------------
    thermal_operating_points: dict[str, ThermalOperatingPoint] = {}
    for profile in cp.power.thermal_profiles:
        # CPU thermal profiles use empty performance_specs to match the
        # hand-coded i7 factory's convention (it sets performance_specs={}
        # because the precision-level Performance lives in
        # precision_profiles instead).
        thermal_operating_points[profile.name] = ThermalOperatingPoint(
            name=profile.name,
            tdp_watts=profile.tdp_watts,
            cooling_solution=profile.cooling_solution_id,
            performance_specs={},
        )

    # ------------------------------------------------------------------
    # Memory + cache (chip-shared from CPUMemorySubsystem;
    # per-cluster L1/L2 from CoreClusterSpec)
    # ------------------------------------------------------------------
    mem = block.memory
    peak_bandwidth_bps = mem.memory_bandwidth_gbps * 1e9
    main_memory_bytes = int(mem.memory_size_gb * 1024**3)

    # Per the existing convention: ``l2_cache_total`` IS the LLC = L3
    # on CPUs with an L3 (or = chip-shared L2 on cache-light SKUs).
    if mem.l3_present:
        llc_bytes = mem.l3_total_kib * 1024
    else:
        # No L3: LLC = the largest chip-shared cache. Conservative
        # fallback to 0 for SKUs we don't yet model.
        llc_bytes = 0

    l3_total_bytes = mem.l3_total_kib * 1024 if mem.l3_present else 0

    # P-cluster (preferred) for legacy l1_cache_per_unit reporting
    p_cluster = _pick_p_cluster(block)
    l1_per_unit_bytes = p_cluster.l1_kib_per_core * 1024

    # l2_cache_per_unit = physical_l2_total / effective_cores. Matches
    # the hand-coded i7 model's convention so consumers that compute
    # ``l2_cache_per_unit * compute_units`` land on the physical L2 total.
    physical_l2_kib = _physical_l2_total_kib(block)
    l2_per_unit_bytes = (
        (physical_l2_kib * 1024) // block.total_effective_cores
        if block.total_effective_cores > 0 else 0
    )

    # Memory energy: prefer YAML-declared, fall back to table
    mem_tech = mem.memory_type.value.lower()
    read_pj = (
        mem.read_energy_pj_per_byte
        if mem.read_energy_pj_per_byte is not None
        else _DRAM_READ_PJ_PER_BYTE.get(mem_tech, 25.0)
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
        controller_count=block.noc.stop_count,
        flit_size_bytes=block.noc.flit_size_bytes,
        routing_distance_factor=block.noc.routing_distance_factor,
        provenance=(
            f"Loaded from compute_products YAML "
            f"({base_id}.dies[0].blocks[0].noc); CPU NoC topology "
            f"{block.noc.topology.value} mapped to graphs SoCFabric "
            f"{_TOPOLOGY_BY_KIND[block.noc.topology].name}"
        ),
    )

    # ------------------------------------------------------------------
    # Assemble HardwareResourceModel
    # ------------------------------------------------------------------
    # Default precision: prefer INT8 if supported, else FP16, else FP32
    if Precision.INT8 in precision_profiles:
        default_precision = Precision.INT8
    elif Precision.FP16 in precision_profiles:
        default_precision = Precision.FP16
    else:
        default_precision = Precision.FP32

    # Pick the dominant fabric for legacy energy_per_flop_fp32 reporting.
    # Use the first fabric from the P-cluster, which matches the
    # hand-coded i7 model's "p_fabric.energy_per_flop_fp32" choice.
    p_first_fabric = next(
        (f for f in compute_fabrics if f.num_units == p_cluster.num_cores),
        compute_fabrics[0],
    )
    energy_per_flop_fp32 = p_first_fabric.energy_per_flop_fp32

    # Model-level energy_scaling: roll up from fabrics, preferring the
    # P-cluster (dominant) values where they overlap.
    energy_scaling: dict[Precision, float] = {}
    for fabric in compute_fabrics:
        if fabric.num_units != p_cluster.num_cores:
            energy_scaling.update(fabric.energy_scaling)
    for fabric in compute_fabrics:
        if fabric.num_units == p_cluster.num_cores:
            energy_scaling.update(fabric.energy_scaling)

    model = HardwareResourceModel(
        name=name_override or cp.name,
        hardware_type=HardwareType.CPU,

        compute_fabrics=compute_fabrics,

        # Legacy core fields
        compute_units=block.total_effective_cores,
        threads_per_unit=p_cluster.smt_threads,
        warps_per_unit=1,    # CPUs don't have warps; legacy compat field
        warp_size=block.simd_width_lanes,

        thermal_operating_points=thermal_operating_points,
        default_thermal_profile=cp.power.default_thermal_profile,

        precision_profiles=precision_profiles,
        default_precision=default_precision,

        peak_bandwidth=peak_bandwidth_bps,
        l1_cache_per_unit=l1_per_unit_bytes,
        # Convention: l2_cache_total IS the LLC (= L3 on i7-12700K).
        # The "physical L2" total goes via l2_cache_per_unit * compute_units.
        l2_cache_total=llc_bytes,
        main_memory=main_memory_bytes,

        energy_per_flop_fp32=energy_per_flop_fp32,
        energy_per_byte=read_pj * 1e-12,
        energy_scaling=energy_scaling,

        min_occupancy=block.min_occupancy,
        max_concurrent_kernels=block.max_concurrent_threads,
        wave_quantization=block.wave_quantization,

        # SIMD efficiency is a CPU-specific concept (graphs CPUMapper
        # reads it for vectorization friendliness).
        simd_efficiency=block.simd_efficiency_by_op_kind,

        # M3 Layer 3
        l1_storage_kind="cache",  # x86 architectural fact

        # M4 Layer 4: physical L2 (per-cluster shape collapsed to a
        # per-unit value)
        l2_cache_per_unit=l2_per_unit_bytes,
        l2_topology="per-unit",   # CPU L2 is per-core/per-cluster, not a chip-wide LLC

        # M5 Layer 5: L3 from CPUMemorySubsystem
        l3_present=mem.l3_present,
        l3_cache_total=l3_total_bytes,
        coherence_protocol=mem.coherence_protocol,

        # M7 Layer 7
        memory_technology=mem.memory_type.value.upper(),
        memory_read_energy_per_byte_pj=read_pj,
        memory_write_energy_per_byte_pj=write_pj,

        # M6 Layer 6
        soc_fabric=soc_fabric,
    )

    # Generic provenance for the YAML-loaded fields (rich per-field
    # whitepaper citations are graphs-side today; v4 schema can add
    # per-field provenance if/when wanted).
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
