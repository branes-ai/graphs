"""CGRA resource model loader from embodied-schemas ComputeProduct YAMLs.

PR 4 of the CGRA mini-sprint scoped at issue #196. Mirrors
``src/graphs/hardware/models/edge/npu_yaml_loader.py`` (NPU sprint
#189) but reads a CGRABlock-bearing ``ComputeProduct`` and builds a
``HardwareResourceModel`` with the CGRA-shaped fields populated.

Sibling to the hand-coded factory at
``src/graphs/hardware/models/accelerators/stanford_plasticine_cgra.py``
during the parallel-migration phase. Adding the factory swap is PR 5;
this PR ships the loader machinery and a parity test that proves the
YAML-loaded model matches the hand-coded one's key fields.

Scope of CGRABlock -> HardwareResourceModel mapping (one-to-one
unless noted):

  CGRABlock.num_pcus              -> compute_units
  CGRABlock.macs_per_pcu          -> threads_per_unit
  CGRABlock.compute_fabrics[*]    -> compute_fabrics[] (num_units =
                                     num_pcus per fabric)
  CGRABlock.memory.{pmu, shared,  -> peak_bandwidth, l1/l2_cache_*,
    host_dram}                       main_memory, memory_technology,
                                     memory_*_energy_per_byte_pj
  CGRABlock.noc                   -> soc_fabric (SoCFabricModel)
  CGRABlock.{min_occupancy,       -> matching HardwareResourceModel fields
    max_concurrent_models,
    wave_quantization}
  ComputeProduct.power.thermal_profiles[]
                                  -> thermal_operating_points
  Roll-up performance             -> precision_profiles

CGRA-specific design decisions:

1. **peak_bandwidth picks on_chip_bandwidth_gbps** (PCU mesh
   bisection), NOT host_dram_bandwidth_gbps. CGRA workloads are
   compiled to fit in on-chip SRAM (PMU + shared L2); host DRAM is
   only used for bitstream load + spills. Plasticine's hand-coded
   factory uses 40 GB/s (the PCU mesh), not the host DDR4 bandwidth.
   This contrasts with NPU's "DRAM tier when present" convention --
   CGRA's reconfig + dataflow pattern means on-chip is the
   steady-state bottleneck.

2. **energy_per_flop_fp32 uses fabric's FP32 scaling** when present.
   Unlike NPUs (INT-only; FP32 synthesized as INT8 * 8), CGRAs ship
   honest FP energy scaling in ``CGRAComputeFabric.energy_scaling``.
   The loader uses the fabric's INT8-baseline * the FP32 scaling
   multiplier when available, giving more accurate FP energy.

3. **HardwareType.CGRA already exists** on the graphs side (unlike
   NPU which needed #191). The loader sets it directly with no
   transitional period.

Out of scope for v5 (deferred to v6):

  - BOMCostProfile (YAML doesn't carry; v6 Market.bom)
  - Per-field provenance copy-through with rich source citations
  - ``reconfig_overhead_cycles`` surfacing on HardwareResourceModel --
    the v5 schema carries it on CGRABlock but the legacy graphs model
    has no equivalent field. PR 5 cleanup or v6 will add it. Until
    then, downstream estimators read it from the ComputeProduct
    directly when they need it.
  - ``supports_partial_reconfig`` -- same story.
"""

from __future__ import annotations

from typing import Optional

from embodied_schemas.compute_product import ComputeProduct
from embodied_schemas.cgra_block import (
    CGRABlock,
    CGRAFabricKind,
    CGRANoCTopology,
)
from embodied_schemas.loaders import load_compute_products
from embodied_schemas.process_node import ProcessNodeEntry

from graphs.core.confidence import EstimationConfidence
from graphs.hardware.fabric_model import SoCFabricModel, Topology
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

class CGRAYamlLoaderError(Exception):
    """Raised when a ComputeProduct YAML can't be turned into a CGRA
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

# CGRAFabricKind -> ComputeFabric.fabric_type. The choice of name
# matches the existing hand-coded factory's fabric_type string
# ("pcu_spatial_dataflow") so parity tests can pin equality.
_FABRIC_TYPE_BY_KIND: dict[CGRAFabricKind, str] = {
    CGRAFabricKind.PCU_SPATIAL_DATAFLOW: "pcu_spatial_dataflow",
    CGRAFabricKind.SYSTOLIC_PCU:         "systolic_pcu",
    CGRAFabricKind.HETEROGENEOUS_PCU:    "heterogeneous_pcu",
}

# CGRAs use custom standard-cell designs across all fabric kinds.
_CIRCUIT_TYPE_DEFAULT = "standard_cell"

# CGRANoCTopology -> graphs SoCFabricModel Topology enum.
_TOPOLOGY_BY_KIND: dict[CGRANoCTopology, Topology] = {
    CGRANoCTopology.MESH_2D:  Topology.MESH_2D,
    CGRANoCTopology.TORUS_2D: Topology.MESH_2D,   # torus = degenerate mesh w/ wrap-around
    CGRANoCTopology.CROSSBAR: Topology.CROSSBAR,
}

# Memory technology -> per-byte energy table (DRAM only; on-chip SRAM
# energy comes from the YAML's pmu_access_energy_pj_per_byte).
_DRAM_READ_PJ_PER_BYTE: dict[str, float] = {
    "ddr4": 28.0, "ddr5": 25.0,
    "lpddr4": 18.0, "lpddr4x": 16.0,
    "lpddr5": 15.0, "lpddr5x": 13.0,
    "hbm2": 7.0, "hbm3": 6.0,
}
_DRAM_WRITE_RATIO = 1.2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cgra_block(cp: ComputeProduct) -> CGRABlock:
    """Pick the (single) CGRABlock from a ComputeProduct's first die."""
    for block in cp.dies[0].blocks:
        if isinstance(block, CGRABlock):
            return block
    raise CGRAYamlLoaderError(
        f"ComputeProduct {cp.id!r} has no CGRABlock in dies[0].blocks "
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
        raise CGRAYamlLoaderError(
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
        raise CGRAYamlLoaderError(
            f"unknown precision name(s) in {source}: {sorted(unknown)}. "
            f"Known: {sorted(_PRECISION_BY_NAME)}"
        )
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_cgra_resource_model_from_yaml(
    base_id: str,
    *,
    products: Optional[dict[str, ComputeProduct]] = None,
    process_nodes: Optional[dict[str, ProcessNodeEntry]] = None,
    name_override: Optional[str] = None,
) -> HardwareResourceModel:
    """Build a ``HardwareResourceModel`` for ``base_id`` from a CGRA
    ComputeProduct YAML.

    Args:
        base_id: ComputeProduct id, e.g., "stanford_plasticine_v2".
        products / process_nodes: optional pre-loaded catalogs.
        name_override: optional override for the resource model's
            ``name`` field. Plasticine v2 YAML name is
            "Stanford Plasticine v2"; the hand-coded factory uses
            "CGRA-Plasticine-v2" -- pass that to preserve compat.

    Raises:
        CGRAYamlLoaderError: when ``base_id`` is not in the catalog,
            doesn't carry a CGRABlock, or has unresolvable references.
    """
    if products is None:
        products = load_compute_products()
    if process_nodes is None:
        from embodied_schemas.loaders import load_process_nodes
        process_nodes = load_process_nodes()

    cp = products.get(base_id)
    if cp is None:
        raise CGRAYamlLoaderError(
            f"no ComputeProduct with id={base_id!r}. Available: "
            f"{', '.join(sorted(products))}"
        )
    block = _cgra_block(cp)

    node = process_nodes.get(cp.dies[0].process_node_id)
    if node is None:
        raise CGRAYamlLoaderError(
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
        raise CGRAYamlLoaderError(
            f"SKU {base_id!r}: default_thermal_profile "
            f"{cp.power.default_thermal_profile!r} is not in thermal_profiles"
        )
    default_clock_hz = default_profile.clock_mhz * 1e6

    # ------------------------------------------------------------------
    # ComputeFabric per CGRA fabric. CGRAs usually ship a single fabric
    # (Plasticine: PCU spatial dataflow). num_units = num_pcus because
    # the fabric "lives" on all PCUs.
    # ------------------------------------------------------------------
    compute_fabrics: list[ComputeFabric] = []
    for cgra_fabric in block.compute_fabrics:
        ops_dict = _precisions_from_ops_dict(
            cgra_fabric.ops_per_unit_per_clock,
            source=f"fabric.{cgra_fabric.fabric_kind.value}",
        )
        if not ops_dict:
            continue
        # CGRAs ship honest FP energy scaling (unlike NPUs which are
        # INT-only). Use the fabric's FP32 scaling when present;
        # fall back to INT8 * 8 (the NPU loader's convention).
        energy_per_op_int8_j = cgra_fabric.energy_per_op_int8_pj * 1e-12
        cgra_scaling = _energy_scaling_from_yaml(cgra_fabric.energy_scaling)
        if Precision.FP32 in cgra_scaling:
            energy_per_flop_fp32_j = energy_per_op_int8_j * cgra_scaling[Precision.FP32]
        else:
            energy_per_flop_fp32_j = energy_per_op_int8_j * 8.0
        # ComputeFabric.energy_scaling expects FP32-baseline scaling
        # (legacy convention). Translate: scaling_fp32 = scaling_int8 / fp32_scaling.
        # INT8 itself gets 1.0 / fp32_scaling.
        fp32_scaling: dict[Precision, float] = {}
        fp32_baseline = cgra_scaling.get(Precision.FP32, 8.0)
        fp32_scaling[Precision.INT8] = 1.0 / fp32_baseline
        for prec, factor in cgra_scaling.items():
            if prec == Precision.FP32:
                continue   # FP32 is the baseline (1.0) after rescaling
            fp32_scaling[prec] = factor / fp32_baseline
        compute_fabrics.append(ComputeFabric(
            fabric_type=_FABRIC_TYPE_BY_KIND[cgra_fabric.fabric_kind],
            circuit_type=_CIRCUIT_TYPE_DEFAULT,
            num_units=block.num_pcus,
            ops_per_unit_per_clock=ops_dict,
            core_frequency_hz=default_clock_hz,
            process_node_nm=process_node_nm,
            energy_per_flop_fp32=energy_per_flop_fp32_j,
            energy_scaling=fp32_scaling,
        ))

    if not compute_fabrics:
        raise CGRAYamlLoaderError(
            f"SKU {base_id!r}: no CGRAComputeFabric produced a valid "
            f"ComputeFabric (check ops_per_unit_per_clock entries)."
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
        # CGRA tensor-core analogue: PCUs are reconfigurable spatial
        # fabric, not fixed tensor cores. tensor_core_supported=False.
        tensor_supported = False
        # Relative speedup vs INT8 (the CGRA baseline). FP16 = 1/4,
        # FP32 = 1/8 for Plasticine-style emulation.
        int8_peak = sum(
            f.get_peak_ops_per_sec(Precision.INT8) for f in compute_fabrics
        )
        relative_speedup = peak / int8_peak if int8_peak > 0 else 1.0
        precision_profiles[precision] = PrecisionProfile(
            precision=precision,
            peak_ops_per_sec=peak,
            tensor_core_supported=tensor_supported,
            relative_speedup=relative_speedup,
            bytes_per_element=_BYTES_PER_PRECISION.get(precision, 4),
            accumulator_precision=(
                Precision.INT32 if precision == Precision.INT8 else
                Precision.FP32 if precision == Precision.FP16 else
                None
            ),
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
            performance_specs={},   # per-precision details live in precision_profiles
        )

    # ------------------------------------------------------------------
    # Memory + cache (PMU + shared L2 + host DRAM)
    # ------------------------------------------------------------------
    mem = block.memory
    # peak_bandwidth: PCU mesh bisection (on_chip), NOT host DRAM.
    # CGRAs compile to fit on-chip; host DRAM is only for bitstream load
    # + spills. Matches Plasticine's hand-coded 40 GB/s. See module
    # docstring for the design decision.
    peak_bandwidth_bps = mem.on_chip_bandwidth_gbps * 1e9

    # Host DRAM -> main_memory (legacy field). When has_host_dram=False
    # the chip uses no main memory (Cerebras-style).
    if mem.has_host_dram and mem.host_dram_size_gb is not None:
        main_memory_bytes = int(mem.host_dram_size_gb * 1024**3)
    else:
        main_memory_bytes = 0

    # Per-PCU PMU is the "L1" of CGRA-land (software-managed scratchpad).
    l1_per_unit_bytes = mem.pmu_kib_per_pcu * 1024

    # Shared SRAM is the "L2/LLC" of CGRA-land.
    l2_total_bytes = mem.shared_sram_kib * 1024
    l2_per_unit_bytes = (
        l2_total_bytes // block.num_pcus
        if block.num_pcus > 0 else 0
    )

    # Memory technology label. When has_host_dram=False, name the
    # primary memory (SRAM) so downstream consumers don't claim DRAM.
    if mem.has_host_dram and mem.host_dram_type is not None:
        memory_technology = mem.host_dram_type.value.upper()
        dram_tech = mem.host_dram_type.value.lower()
        read_pj = _DRAM_READ_PJ_PER_BYTE.get(dram_tech, 25.0)
        write_pj = read_pj * _DRAM_WRITE_RATIO
        # Honor the YAML's host_dram_access_energy_pj_per_byte when
        # populated (more accurate than the lookup table)
        if mem.host_dram_access_energy_pj_per_byte > 0:
            read_pj = mem.host_dram_access_energy_pj_per_byte
            write_pj = read_pj * _DRAM_WRITE_RATIO
    else:
        memory_technology = "on-chip SRAM (no host DRAM)"
        read_pj = mem.pmu_access_energy_pj_per_byte
        write_pj = read_pj

    # ------------------------------------------------------------------
    # SoC fabric
    # ------------------------------------------------------------------
    soc_fabric_kwargs = dict(
        topology=_TOPOLOGY_BY_KIND[block.noc.topology],
        hop_latency_ns=block.noc.hop_latency_ns,
        pj_per_flit_per_hop=block.noc.pj_per_flit_per_hop,
        bisection_bandwidth_gbps=block.noc.bisection_bandwidth_gbps,
        controller_count=block.noc.unit_count,
        flit_size_bytes=block.noc.flit_size_bytes,
        routing_distance_factor=block.noc.routing_distance_factor,
        provenance=(
            f"Loaded from compute_products YAML "
            f"({base_id}.dies[0].blocks[0].noc); CGRA NoC topology "
            f"{block.noc.topology.value} mapped to graphs SoCFabric "
            f"{_TOPOLOGY_BY_KIND[block.noc.topology].name}"
        ),
    )
    # Mesh dimensions: pass through when MESH_2D or TORUS_2D
    if (block.noc.topology in (CGRANoCTopology.MESH_2D, CGRANoCTopology.TORUS_2D)
            and block.noc.mesh_rows is not None
            and block.noc.mesh_cols is not None):
        soc_fabric_kwargs["mesh_dimensions"] = (
            block.noc.mesh_rows, block.noc.mesh_cols
        )
    # Confidence flag
    from embodied_schemas.process_node import DataConfidence
    if block.noc.confidence == DataConfidence.THEORETICAL:
        soc_fabric_kwargs["low_confidence"] = True
    soc_fabric = SoCFabricModel(**soc_fabric_kwargs)

    # ------------------------------------------------------------------
    # Assemble HardwareResourceModel
    # ------------------------------------------------------------------
    # Default precision: INT8 (the CGRA default; FP is emulated)
    if Precision.INT8 in precision_profiles:
        default_precision = Precision.INT8
    elif Precision.INT4 in precision_profiles:
        default_precision = Precision.INT4
    else:
        default_precision = next(iter(precision_profiles))

    # Energy fields: use the first fabric (CGRAs typically have one)
    first_fabric = compute_fabrics[0]
    energy_per_flop_fp32 = first_fabric.energy_per_flop_fp32
    # Model-level energy_scaling: roll up from all fabrics
    energy_scaling: dict[Precision, float] = {}
    for fabric in compute_fabrics:
        energy_scaling.update(fabric.energy_scaling)

    model = HardwareResourceModel(
        name=name_override or cp.name,
        hardware_type=HardwareType.CGRA,

        compute_fabrics=compute_fabrics,

        compute_units=block.num_pcus,
        threads_per_unit=block.macs_per_pcu,
        warps_per_unit=1,    # CGRAs don't have warps; legacy compat
        warp_size=1,         # spatial dataflow

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
        max_concurrent_kernels=block.max_concurrent_models,
        wave_quantization=block.wave_quantization,

        # M3 Layer 3: software-managed PMU scratchpad (compiler-routed).
        l1_storage_kind="scratchpad",

        # M4 Layer 4: shared SRAM acts as LLC above per-PCU partitions
        l2_cache_per_unit=l2_per_unit_bytes,
        l2_topology="shared-llc",

        # M5 Layer 5: CGRAs don't have L3 between PCUs.
        l3_present=False,
        l3_cache_total=0,

        # Compiler-routed spatial dataflow has no coherence
        coherence_protocol=mem.coherence_protocol,

        # M7 Layer 7
        memory_technology=memory_technology,
        memory_read_energy_per_byte_pj=read_pj,
        memory_write_energy_per_byte_pj=write_pj,

        # M6 Layer 6
        soc_fabric=soc_fabric,
    )

    # Generic provenance for the YAML-loaded fields
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

    # Per-precision provenance on compute_fabric ops/clock
    fabric_provenance = EstimationConfidence.theoretical(
        score=0.55,
        source=(
            f"Loaded from compute_products YAML ({base_id}): per-fabric "
            f"ops_per_unit_per_clock at the dominant PCU fabric"
        ),
    )
    for precision in supported_precisions:
        model.set_provenance(
            f"compute_fabric.ops_per_clock.{precision.value}",
            fabric_provenance,
        )

    return model
