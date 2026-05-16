"""NPU resource model loader from embodied-schemas ComputeProduct YAMLs.

PR 4 of the NPU sprint scoped at issue #187. Mirrors
``src/graphs/hardware/models/edge/gpu_yaml_loader.py`` (GPU sprint #180)
and ``src/graphs/hardware/models/datacenter/cpu_yaml_loader.py``
(CPU sprint #185) but reads an NPUBlock-bearing ``ComputeProduct``
and builds a ``HardwareResourceModel`` with the NPU-shaped fields
populated.

Sibling to the hand-coded factories in
``src/graphs/hardware/models/edge/`` during the parallel-migration
phase. Adding the factory swap is PR 5; this PR ships the loader
machinery and a parity test that proves the YAML-loaded model
matches the hand-coded one's key fields.

Scope of NPUBlock -> HardwareResourceModel mapping (one-to-one
unless noted):

  NPUBlock.num_dataflow_units     -> compute_units
  NPUBlock.lanes_per_unit         -> threads_per_unit
  NPUBlock.compute_fabrics[*]     -> compute_fabrics[] (num_units =
                                     num_dataflow_units per fabric)
  NPUBlock.memory.{sram, dram}    -> peak_bandwidth, l1/l2_cache_*,
                                     main_memory, memory_technology,
                                     memory_*_energy_per_byte_pj
  NPUBlock.noc                    -> soc_fabric (SoCFabricModel)
  NPUBlock.{min_occupancy, max_concurrent_models, wave_quantization}
                                  -> matching HardwareResourceModel fields
  ComputeProduct.power.thermal_profiles[]
                                  -> thermal_operating_points
  Roll-up performance             -> precision_profiles

One known shape quirk (documented; no parity gap):

  NPUs don't ship FP32; the ``energy_per_flop_fp32`` legacy field
  is synthesized from ``energy_per_op_int8_pj * 8`` (FP32 is roughly
  8x the energy of INT8 in standard cells -- the rule of thumb the
  existing models use). Downstream consumers that read
  ``energy_per_flop_fp32`` get a sensible proxy.

(Previously this section also listed ``HardwareType.NPU`` as a quirk
because the graphs enum had no NPU value -- issue #191 added it; the
loader now sets ``hardware_type=HardwareType.NPU`` directly.)

Out of scope for v4 (deferred to v5):
  - BOMCostProfile (YAML doesn't carry; v5 Market.bom)
  - Per-field provenance copy-through with rich source citations
  - KVCacheSpec for transformer-capable NPUs (Hailo-10H follow-up)
  - HardwareType.NPU enum value on the graphs side
"""

from __future__ import annotations

from typing import Optional

from embodied_schemas.compute_product import ComputeProduct
from embodied_schemas.npu_block import (
    NPUBlock,
    NPUDataflowKind,
    NPUNoCTopology,
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

class NPUYamlLoaderError(Exception):
    """Raised when a ComputeProduct YAML can't be turned into an NPU
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

# NPUDataflowKind -> ComputeFabric.fabric_type. Each dataflow kind
# maps to a short string the downstream graphs consumers can use to
# distinguish architectures across SKUs. The choice of name matches
# the existing hand-coded factories' fabric_type strings (e.g.
# Hailo-8 uses "dataflow_architecture").
_FABRIC_TYPE_BY_DATAFLOW: dict[NPUDataflowKind, str] = {
    NPUDataflowKind.STRUCTURE_DRIVEN:    "dataflow_architecture",
    NPUDataflowKind.SYSTOLIC:            "systolic_array",
    NPUDataflowKind.SPATIAL:             "spatial_dataflow",
    NPUDataflowKind.WEIGHT_STATIONARY:   "weight_stationary",
    NPUDataflowKind.OUTPUT_STATIONARY:   "output_stationary",
    NPUDataflowKind.INPUT_STATIONARY:    "input_stationary",
}

# NPUs use custom ASIC standard-cell designs across all dataflow
# kinds. The "standard_cell" classification matches the hand-coded
# Hailo-8 factory's circuit_type choice.
_CIRCUIT_TYPE_DEFAULT = "standard_cell"

# NPUNoCTopology -> graphs SoCFabricModel Topology enum.
_TOPOLOGY_BY_KIND: dict[NPUNoCTopology, Topology] = {
    NPUNoCTopology.MESH_2D:        Topology.MESH_2D,
    NPUNoCTopology.CROSSBAR:       Topology.CROSSBAR,
    NPUNoCTopology.SYSTOLIC:       Topology.MESH_2D,  # systolic = degenerate mesh
    NPUNoCTopology.DATAFLOW_RING:  Topology.RING,
}

# Memory technology -> per-byte energy table. NPUs typically run on
# on-chip SRAM (cheap, ~2 pJ/B); the DRAM table applies only when
# has_external_dram=True.
_DRAM_READ_PJ_PER_BYTE: dict[str, float] = {
    "lpddr4": 18.0, "lpddr4x": 16.0,
    "lpddr5": 15.0, "lpddr5x": 13.0,
    "ddr5": 25.0, "ddr4": 28.0,
    "hbm2": 7.0, "hbm3": 6.0,
}
_DRAM_WRITE_RATIO = 1.2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _npu_block(cp: ComputeProduct) -> NPUBlock:
    """Pick the (single) NPUBlock from a ComputeProduct's first die."""
    for block in cp.dies[0].blocks:
        if isinstance(block, NPUBlock):
            return block
    raise NPUYamlLoaderError(
        f"ComputeProduct {cp.id!r} has no NPUBlock in dies[0].blocks "
        f"(found: {[type(b).__name__ for b in cp.dies[0].blocks]})"
    )


def _precisions_from_ops_dict(
    ops: dict[str, int],
    *,
    source: str = "ops_per_unit_per_clock",
) -> dict[Precision, int]:
    """Convert YAML's str-keyed ops/clock dict to graphs' Precision-keyed dict.

    Fails fast on unknown precision names (same defensive pattern as
    gpu/cpu_yaml_loader)."""
    out: dict[Precision, int] = {}
    unknown: list[str] = []
    for name, count in ops.items():
        prec = _PRECISION_BY_NAME.get(name.lower())
        if prec is None:
            unknown.append(name)
            continue
        out[prec] = count
    if unknown:
        raise NPUYamlLoaderError(
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
        raise NPUYamlLoaderError(
            f"unknown precision name(s) in {source}: {sorted(unknown)}. "
            f"Known: {sorted(_PRECISION_BY_NAME)}"
        )
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_npu_resource_model_from_yaml(
    base_id: str,
    *,
    products: Optional[dict[str, ComputeProduct]] = None,
    process_nodes: Optional[dict[str, ProcessNodeEntry]] = None,
    name_override: Optional[str] = None,
) -> HardwareResourceModel:
    """Build a ``HardwareResourceModel`` for ``base_id`` from an NPU
    ComputeProduct YAML.

    Args:
        base_id: ComputeProduct id, e.g., "hailo_hailo_8".
        products / process_nodes: optional pre-loaded catalogs.
        name_override: optional override for the resource model's
            ``name`` field. The hand-coded Hailo-8 factory uses
            "Hailo-8"; the YAML uses "Hailo-8" too -- override is
            for symmetry with the GPU/CPU loaders.

    Raises:
        NPUYamlLoaderError: when ``base_id`` is not in the catalog,
            doesn't carry an NPUBlock, or has unresolvable references.
    """
    if products is None:
        products = load_compute_products()
    if process_nodes is None:
        from embodied_schemas.loaders import load_process_nodes
        process_nodes = load_process_nodes()

    cp = products.get(base_id)
    if cp is None:
        raise NPUYamlLoaderError(
            f"no ComputeProduct with id={base_id!r}. Available: "
            f"{', '.join(sorted(products))}"
        )
    block = _npu_block(cp)

    node = process_nodes.get(cp.dies[0].process_node_id)
    if node is None:
        raise NPUYamlLoaderError(
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
        raise NPUYamlLoaderError(
            f"SKU {base_id!r}: default_thermal_profile "
            f"{cp.power.default_thermal_profile!r} is not in thermal_profiles"
        )
    default_clock_hz = default_profile.clock_mhz * 1e6

    # ------------------------------------------------------------------
    # ComputeFabric per NPU fabric. NPUs usually ship a single fabric
    # (Hailo-8: structure-driven dataflow). num_units = num_dataflow_units
    # because the fabric "lives" on all units.
    # ------------------------------------------------------------------
    compute_fabrics: list[ComputeFabric] = []
    for npu_fabric in block.compute_fabrics:
        ops_dict = _precisions_from_ops_dict(
            npu_fabric.ops_per_unit_per_clock,
            source=f"fabric.{npu_fabric.dataflow_kind.value}",
        )
        if not ops_dict:
            continue
        # Synthesize FP32 energy from INT8 (NPUs don't ship FP32; the
        # legacy field still has consumers). Standard cells: FP32 ~= 8x
        # INT8 energy.
        energy_per_op_int8_j = npu_fabric.energy_per_op_int8_pj * 1e-12
        energy_per_flop_fp32_j = energy_per_op_int8_j * 8.0
        # energy_scaling: dict normalizes against the FP32 baseline,
        # but NPUs scale against INT8. Convert: scaling_fp32 = (scaling_int8 / 8).
        # Then INT8 itself is 0.125 of FP32 in this scheme.
        npu_scaling = _energy_scaling_from_yaml(npu_fabric.energy_scaling)
        fp32_scaling: dict[Precision, float] = {
            Precision.INT8: 0.125,  # INT8 is 1/8 of FP32 by construction
        }
        for prec, factor in npu_scaling.items():
            # YAML's energy_scaling is relative to INT8; convert to FP32-baseline
            fp32_scaling[prec] = factor / 8.0
        compute_fabrics.append(ComputeFabric(
            fabric_type=_FABRIC_TYPE_BY_DATAFLOW[npu_fabric.dataflow_kind],
            circuit_type=_CIRCUIT_TYPE_DEFAULT,
            num_units=block.num_dataflow_units,
            ops_per_unit_per_clock=ops_dict,
            core_frequency_hz=default_clock_hz,
            process_node_nm=process_node_nm,
            energy_per_flop_fp32=energy_per_flop_fp32_j,
            energy_scaling=fp32_scaling,
        ))

    if not compute_fabrics:
        raise NPUYamlLoaderError(
            f"SKU {base_id!r}: no NPUComputeFabric produced a valid "
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
        # On an NPU, "tensor_core_supported" is True for the dominant
        # INT8 / INT4 paths (dataflow IS the tensor engine).
        tensor_supported = precision in {Precision.INT8, Precision.INT4}
        # Relative speedup vs INT8 (the NPU baseline). INT4 = 2x.
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
                Precision.INT16 if precision == Precision.INT4 else
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
    # Memory + cache (SRAM-dominant)
    # ------------------------------------------------------------------
    mem = block.memory
    # peak_bandwidth = on-chip SRAM bandwidth (the dominant tier).
    # When has_external_dram=True, downstream consumers can compare
    # external_dram_bandwidth via the DRAM-specific fields, but the
    # legacy peak_bandwidth field stays on the on-chip side for
    # parity with the hand-coded factory.
    peak_bandwidth_bps = mem.on_chip_bandwidth_gbps * 1e9

    # External DRAM. Most NPUs have none (Hailo-8 / Coral); Hailo-10H
    # has LPDDR4X. Set main_memory = 0 when has_external_dram=False.
    if mem.has_external_dram and mem.external_dram_size_gb is not None:
        main_memory_bytes = int(mem.external_dram_size_gb * 1024**3)
    else:
        main_memory_bytes = 0

    # Per-unit SRAM is the "L1" of NPU-land (software-managed scratchpad).
    l1_per_unit_bytes = mem.sram_kib_per_unit * 1024

    # Shared SRAM is the "L2/LLC" of NPU-land.
    l2_total_bytes = mem.shared_sram_kib * 1024
    l2_per_unit_bytes = (
        l2_total_bytes // block.num_dataflow_units
        if block.num_dataflow_units > 0 else 0
    )

    # Memory technology label. When has_external_dram=False, name the
    # primary memory (SRAM) so downstream consumers don't claim DRAM.
    if mem.has_external_dram and mem.external_dram_type is not None:
        memory_technology = mem.external_dram_type.value.upper()
        # When DRAM is present, use its energy as the "memory" energy
        dram_tech = mem.external_dram_type.value.lower()
        read_pj = _DRAM_READ_PJ_PER_BYTE.get(dram_tech, 18.0)
        write_pj = read_pj * _DRAM_WRITE_RATIO
    else:
        memory_technology = f"on-chip SRAM (no external DRAM)"
        # On-chip SRAM energy: use the YAML's sram_access_energy_pj_per_byte
        read_pj = mem.sram_access_energy_pj_per_byte
        write_pj = read_pj   # SRAM read/write energy is symmetric

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
            f"({base_id}.dies[0].blocks[0].noc); NPU NoC topology "
            f"{block.noc.topology.value} mapped to graphs SoCFabric "
            f"{_TOPOLOGY_BY_KIND[block.noc.topology].name}"
        ),
    )
    # Mesh dimensions: pass through when MESH_2D
    if (block.noc.topology == NPUNoCTopology.MESH_2D
            and block.noc.mesh_rows is not None
            and block.noc.mesh_cols is not None):
        soc_fabric_kwargs["mesh_dimensions"] = (
            block.noc.mesh_rows, block.noc.mesh_cols
        )
    # Confidence flag for SKUs where the NoC details aren't published
    from embodied_schemas.process_node import DataConfidence
    if block.noc.confidence == DataConfidence.THEORETICAL:
        soc_fabric_kwargs["low_confidence"] = True
    soc_fabric = SoCFabricModel(**soc_fabric_kwargs)

    # ------------------------------------------------------------------
    # Assemble HardwareResourceModel
    # ------------------------------------------------------------------
    # Default precision: INT8 (the NPU default; FP precisions are rare)
    if Precision.INT8 in precision_profiles:
        default_precision = Precision.INT8
    elif Precision.INT4 in precision_profiles:
        default_precision = Precision.INT4
    else:
        default_precision = next(iter(precision_profiles))

    # Energy fields: use the first fabric (NPUs typically have one)
    first_fabric = compute_fabrics[0]
    energy_per_flop_fp32 = first_fabric.energy_per_flop_fp32
    # Model-level energy_scaling: roll up from all fabrics
    energy_scaling: dict[Precision, float] = {}
    for fabric in compute_fabrics:
        energy_scaling.update(fabric.energy_scaling)

    model = HardwareResourceModel(
        name=name_override or cp.name,
        hardware_type=HardwareType.NPU,

        compute_fabrics=compute_fabrics,

        compute_units=block.num_dataflow_units,
        threads_per_unit=block.lanes_per_unit,
        warps_per_unit=1,    # NPUs don't have warps; legacy compat
        warp_size=1,         # scalar dataflow

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

        # M3 Layer 3: software-managed scratchpad (NPU compiler routes
        # explicitly; no hardware cache).
        l1_storage_kind="scratchpad",

        # M4 Layer 4: shared SRAM acts as LLC above per-unit partitions
        l2_cache_per_unit=l2_per_unit_bytes,
        l2_topology="shared-llc",

        # M5 Layer 5: NPUs don't have inter-cluster cache. The
        # shared SRAM IS the top of the on-chip hierarchy.
        l3_present=False,
        l3_cache_total=0,

        # Compiler-routed dataflow has no coherence
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
            f"ops_per_unit_per_clock at the dominant dataflow kind"
        ),
    )
    for precision in supported_precisions:
        model.set_provenance(
            f"compute_fabric.ops_per_clock.{precision.value}",
            fabric_provenance,
        )

    return model
