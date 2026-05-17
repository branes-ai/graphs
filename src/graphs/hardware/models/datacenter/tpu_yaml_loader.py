"""TPU resource model loader from embodied-schemas ComputeProduct YAMLs.

PR 4 of the TPU mini-sprint scoped at issue #204. Mirrors
``src/graphs/hardware/models/accelerators/dpu_yaml_loader.py`` (DPU
sprint #202) but reads a TPUBlock-bearing ``ComputeProduct`` and
builds a ``HardwareResourceModel`` with the TPU-shaped fields populated.

Sibling to the hand-coded factory at
``src/graphs/hardware/models/datacenter/tpu_v4.py`` during the
parallel-migration phase. Adding the factory swap is PR 5; this PR
ships the loader machinery and a parity test that proves the YAML-
loaded model matches the hand-coded one's key fields.

Scope of TPUBlock -> HardwareResourceModel mapping:

  TPUBlock.num_mxus                  -> compute_units
  TPUBlock.{mxu_dim_rows,            -> derived: threads_per_unit =
    mxu_dim_cols}                       rows * cols (MACs per MXU)
  TPUBlock.compute_fabrics[*]        -> compute_fabrics[] (num_units =
                                        num_mxus * mxu_dim_rows *
                                        mxu_dim_cols = total MACs)
  TPUBlock.memory.{unified_buffer,   -> l1_cache_per_unit,
    external_dram}                      l2_cache_total, main_memory,
                                        peak_bandwidth, memory_technology,
                                        memory_*_energy_per_byte_pj
  TPUBlock.tile_energy_coefficients  -> ``model.tile_energy_model`` =
                                        TPUTileEnergyModel(...)
  TPUBlock.noc                       -> soc_fabric (SoCFabricModel)
  TPUBlock.{min_occupancy,           -> matching HardwareResourceModel fields
    max_concurrent_models,
    wave_quantization}
  ComputeProduct.power.thermal_profiles[]
                                     -> thermal_operating_points

TPU-specific loader decisions:

1. **peak_bandwidth = external_dram_bandwidth_gbps** when
   has_external_dram=True (TPU v4: 1.2 TB/s HBM2e). Mirrors NPU/DPU's
   "DRAM tier when present" convention -- TPUs are HBM-bound for
   training workloads. Falls back to UB bandwidth for SRAM-only SKUs
   (none today; future hypothetical edge TPUs might be SRAM-only).

2. **Unified Buffer is split across l1_cache_per_unit + l2_cache_total**
   for legacy compat. The UB is a single shared tier in the v7 schema
   (no L1/L2 distinction); the legacy HardwareResourceModel expects
   both fields, so the loader sets both to the same UB size divided
   into the per-MXU view (l1_cache_per_unit = unified_buffer_size /
   num_mxus) and the chip total (l2_cache_total = unified_buffer_size).
   The legacy hand-coded ``tpu_v4.py`` does the same split.

3. **Tile energy model reconstructed from TPUTileEnergyCoefficients**.
   The schema's ``TPUTileEnergyCoefficients`` (9 canonical fields)
   maps directly to ``graphs.hardware.architectural_energy.TPUTileEnergyModel``,
   combined with ``mxu_dim_rows``, ``mxu_dim_cols``, ``num_mxus``,
   ``weight_tile_size_kib``, ``weight_fifo_depth``, ``pipeline_fill_cycles``,
   ``accumulator_size_kib_per_mxu``, and ``unified_buffer_size_kib``
   from ``TPUBlock`` itself.

4. **HardwareType.TPU** set directly -- TPU already in graphs enum.

5. **ICI fields NOT surfaced on HardwareResourceModel**. The v7 schema
   carries `ici_port_count`, `ici_bandwidth_per_port_gbps`, and
   `ici_topology_hint` on TPUBlock but the legacy model has no
   equivalent fields. v8 reconciliation. Downstream consumers that
   need multi-chip surface read directly from the ComputeProduct.

6. **default_precision = BF16** (TPU training-default), not INT8 like
   NPU/DPU/CGRA. Inference-focused TPU SKUs (tpu_edge_pro) could
   override; the schema doesn't carry a per-SKU default_precision hint
   yet (v8 reconciliation).

Out of scope for v7 (deferred to v8):

  - BOMCostProfile (YAML doesn't carry; v8 Market.bom)
  - Per-field provenance copy-through with rich source citations
  - ``ici_*`` / ``ici_topology_hint`` surfacing on
    HardwareResourceModel (v8 reconciliation)
  - SparseCore as sibling block (v8 multi-block-per-die)
  - Per-SKU default_precision hint (v8 reconciliation)
"""

from __future__ import annotations

from typing import Optional

from embodied_schemas.compute_product import ComputeProduct
from embodied_schemas.tpu_block import (
    TPUBlock,
    TPUFabricKind,
    TPUNoCTopology,
)
from embodied_schemas.loaders import load_compute_products
from embodied_schemas.process_node import ProcessNodeEntry

from graphs.core.confidence import EstimationConfidence
from graphs.hardware.architectural_energy import TPUTileEnergyModel
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

class TPUYamlLoaderError(Exception):
    """Raised when a ComputeProduct YAML can't be turned into a TPU
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

# TPUFabricKind -> ComputeFabric.fabric_type. Matches the hand-coded
# factory's fabric_type string ("systolic_array") so parity tests can
# pin equality.
_FABRIC_TYPE_BY_KIND: dict[TPUFabricKind, str] = {
    TPUFabricKind.TPU_V1_STYLE: "systolic_array",
    TPUFabricKind.TPU_V2_PLUS:  "systolic_array",
    TPUFabricKind.TPU_HD:       "systolic_array",
}

_CIRCUIT_TYPE_DEFAULT = "standard_cell"

# TPUNoCTopology -> graphs SoCFabricModel Topology enum.
_TOPOLOGY_BY_KIND: dict[TPUNoCTopology, Topology] = {
    TPUNoCTopology.CROSSBAR:       Topology.CROSSBAR,
    TPUNoCTopology.MULTI_CROSSBAR: Topology.CROSSBAR,   # multi-crossbar = crossbar w/ multiple endpoints
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tpu_block(cp: ComputeProduct) -> TPUBlock:
    """Pick the (single) TPUBlock from a ComputeProduct's first die."""
    for block in cp.dies[0].blocks:
        if isinstance(block, TPUBlock):
            return block
    raise TPUYamlLoaderError(
        f"ComputeProduct {cp.id!r} has no TPUBlock in dies[0].blocks "
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
        raise TPUYamlLoaderError(
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
        raise TPUYamlLoaderError(
            f"unknown precision name(s) in {source}: {sorted(unknown)}. "
            f"Known: {sorted(_PRECISION_BY_NAME)}"
        )
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_tpu_resource_model_from_yaml(
    base_id: str,
    *,
    products: Optional[dict[str, ComputeProduct]] = None,
    process_nodes: Optional[dict[str, ProcessNodeEntry]] = None,
    name_override: Optional[str] = None,
) -> HardwareResourceModel:
    """Build a ``HardwareResourceModel`` for ``base_id`` from a TPU
    ComputeProduct YAML.

    Args:
        base_id: ComputeProduct id, e.g., "google_tpu_v4".
        products / process_nodes: optional pre-loaded catalogs.
        name_override: optional override for the resource model's
            ``name`` field. TPU v4 YAML name is "Google TPU v4"; the
            hand-coded factory uses "TPU-v4" -- pass that to preserve
            compat.

    Raises:
        TPUYamlLoaderError: when ``base_id`` is not in the catalog,
            doesn't carry a TPUBlock, or has unresolvable references.
    """
    if products is None:
        products = load_compute_products()
    if process_nodes is None:
        from embodied_schemas.loaders import load_process_nodes
        process_nodes = load_process_nodes()

    cp = products.get(base_id)
    if cp is None:
        raise TPUYamlLoaderError(
            f"no ComputeProduct with id={base_id!r}. Available: "
            f"{', '.join(sorted(products))}"
        )
    block = _tpu_block(cp)

    node = process_nodes.get(cp.dies[0].process_node_id)
    if node is None:
        raise TPUYamlLoaderError(
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
        raise TPUYamlLoaderError(
            f"SKU {base_id!r}: default_thermal_profile "
            f"{cp.power.default_thermal_profile!r} is not in thermal_profiles"
        )
    default_clock_hz = default_profile.clock_mhz * 1e6

    # ------------------------------------------------------------------
    # ComputeFabric per TPU fabric. TPU v4 ships a single systolic
    # fabric. num_units = total MACs across all MXUs.
    # ------------------------------------------------------------------
    total_macs = block.num_mxus * block.mxu_dim_rows * block.mxu_dim_cols
    compute_fabrics: list[ComputeFabric] = []
    for tpu_fabric in block.compute_fabrics:
        ops_dict = _precisions_from_ops_dict(
            tpu_fabric.ops_per_unit_per_clock,
            source=f"fabric.{tpu_fabric.fabric_kind.value}",
        )
        if not ops_dict:
            continue
        # TPU energy_per_op_bf16_pj is the per-BF16-op cost. Compute
        # per-FP32-op for the legacy field as bf16 * fp32_scaling.
        energy_per_op_bf16_j = tpu_fabric.energy_per_op_bf16_pj * 1e-12
        tpu_scaling = _energy_scaling_from_yaml(tpu_fabric.energy_scaling)
        if Precision.FP32 in tpu_scaling:
            energy_per_flop_fp32_j = energy_per_op_bf16_j * tpu_scaling[Precision.FP32]
        else:
            # Default: FP32 = 2x BF16 (TPU emulation cost; matches legacy)
            energy_per_flop_fp32_j = energy_per_op_bf16_j * 2.0
        # ComputeFabric.energy_scaling expects FP32-baseline scaling
        # (legacy convention). Translate from BF16-baseline:
        # scaling_fp32[prec] = scaling_bf16[prec] / fp32_scaling
        fp32_scaling: dict[Precision, float] = {}
        fp32_baseline = tpu_scaling.get(Precision.FP32, 2.0)
        # BF16 is the schema's 1.0 baseline; in FP32-baseline terms
        # it's 1/fp32_scaling
        fp32_scaling[Precision.BF16] = 1.0 / fp32_baseline
        for prec, factor in tpu_scaling.items():
            if prec == Precision.FP32:
                continue
            fp32_scaling[prec] = factor / fp32_baseline
        compute_fabrics.append(ComputeFabric(
            fabric_type=_FABRIC_TYPE_BY_KIND[tpu_fabric.fabric_kind],
            circuit_type=_CIRCUIT_TYPE_DEFAULT,
            num_units=total_macs,
            ops_per_unit_per_clock=ops_dict,
            core_frequency_hz=default_clock_hz,
            process_node_nm=process_node_nm,
            energy_per_flop_fp32=energy_per_flop_fp32_j,
            energy_scaling=fp32_scaling,
        ))

    if not compute_fabrics:
        raise TPUYamlLoaderError(
            f"SKU {base_id!r}: no TPUComputeFabric produced a valid "
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
        # TPUs ARE tensor cores -- the systolic array IS the tensor engine
        tensor_supported = True
        # Relative speedup vs BF16 (the TPU training baseline).
        bf16_peak = sum(
            f.get_peak_ops_per_sec(Precision.BF16) for f in compute_fabrics
        )
        relative_speedup = peak / bf16_peak if bf16_peak > 0 else 1.0
        precision_profiles[precision] = PrecisionProfile(
            precision=precision,
            peak_ops_per_sec=peak,
            tensor_core_supported=tensor_supported,
            relative_speedup=relative_speedup,
            bytes_per_element=_BYTES_PER_PRECISION.get(precision, 4),
            accumulator_precision=(
                Precision.INT32 if precision == Precision.INT8 else
                Precision.FP32 if precision == Precision.BF16 else
                None
            ),
        )

    # Same guard as DPU loader fix (graphs#202): empty profiles would
    # crash on default_precision selection; raise a clear error instead.
    if not precision_profiles:
        raise TPUYamlLoaderError(
            f"SKU {base_id!r}: no precision_profiles produced. All "
            f"compute_fabrics returned zero peak ops/sec for every "
            f"supported precision; check ops_per_unit_per_clock + "
            f"thermal profile clock values."
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
    # Memory + cache (Unified Buffer + HBM)
    # ------------------------------------------------------------------
    mem = block.memory
    # peak_bandwidth: TPUs are HBM-bound for training. Pick external
    # DRAM when present (TPU v4: HBM2e at 1.2 TB/s).
    if mem.has_external_dram and mem.external_dram_bandwidth_gbps is not None:
        peak_bandwidth_bps = mem.external_dram_bandwidth_gbps * 1e9
    else:
        peak_bandwidth_bps = mem.on_chip_bandwidth_gbps * 1e9

    # External DRAM -> main_memory.
    if mem.has_external_dram and mem.external_dram_size_gb is not None:
        main_memory_bytes = int(mem.external_dram_size_gb * 1024**3)
    else:
        main_memory_bytes = 0

    # Unified Buffer split across legacy l1/l2 fields. The UB is a
    # single tier in the v7 schema; the legacy expects both. Same
    # split as the hand-coded factory.
    ub_total_bytes = mem.unified_buffer_size_kib * 1024
    l1_per_unit_bytes = (
        ub_total_bytes // block.num_mxus
        if block.num_mxus > 0 else 0
    )
    l2_total_bytes = ub_total_bytes

    # Memory technology label and per-byte access energy.
    if mem.has_external_dram and mem.external_dram_type is not None:
        memory_technology = mem.external_dram_type.value.upper()
        if mem.external_dram_access_energy_pj_per_byte > 0:
            read_pj = mem.external_dram_access_energy_pj_per_byte
        else:
            read_pj = 10.0   # HBM2e default
        write_pj = read_pj * 1.2   # standard write-vs-read ratio
    else:
        memory_technology = "on-chip SRAM (no external DRAM)"
        read_pj = mem.unified_buffer_access_energy_pj_per_byte
        write_pj = read_pj

    # ------------------------------------------------------------------
    # SoC fabric (UB-to-MXU crossbar)
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
            f"({base_id}.dies[0].blocks[0].noc); TPU NoC topology "
            f"{block.noc.topology.value} mapped to graphs SoCFabric "
            f"{_TOPOLOGY_BY_KIND[block.noc.topology].name}"
        ),
    )
    from embodied_schemas.process_node import DataConfidence
    if block.noc.confidence == DataConfidence.THEORETICAL:
        soc_fabric_kwargs["low_confidence"] = True
    soc_fabric = SoCFabricModel(**soc_fabric_kwargs)

    # ------------------------------------------------------------------
    # Tile energy model reconstructed from TPUTileEnergyCoefficients
    # ------------------------------------------------------------------
    tile_coeffs = block.tile_energy_coefficients
    tile_energy_model = TPUTileEnergyModel(
        array_width=block.mxu_dim_cols,
        array_height=block.mxu_dim_rows,
        num_arrays=block.num_mxus,
        weight_tile_size=block.weight_tile_size_kib * 1024,
        weight_fifo_depth=block.weight_fifo_depth,
        pipeline_fill_cycles=block.pipeline_fill_cycles,
        clock_frequency_hz=default_clock_hz,
        accumulator_size=block.accumulator_size_kib_per_mxu * 1024,
        accumulator_width=block.mxu_dim_cols,
        unified_buffer_size=ub_total_bytes,
        weight_memory_energy_per_byte=tile_coeffs.weight_memory_energy_pj_per_byte * 1e-12,
        weight_fifo_energy_per_byte=tile_coeffs.weight_fifo_energy_pj_per_byte * 1e-12,
        unified_buffer_read_energy_per_byte=tile_coeffs.unified_buffer_read_energy_pj_per_byte * 1e-12,
        unified_buffer_write_energy_per_byte=tile_coeffs.unified_buffer_write_energy_pj_per_byte * 1e-12,
        accumulator_write_energy_per_element=tile_coeffs.accumulator_write_energy_pj_per_element * 1e-12,
        accumulator_read_energy_per_element=tile_coeffs.accumulator_read_energy_pj_per_element * 1e-12,
        weight_shift_in_energy_per_element=tile_coeffs.weight_shift_in_energy_pj_per_element * 1e-12,
        activation_stream_energy_per_element=tile_coeffs.activation_stream_energy_pj_per_element * 1e-12,
        mac_energy=tile_coeffs.mac_energy_pj * 1e-12,
    )

    # ------------------------------------------------------------------
    # Assemble HardwareResourceModel
    # ------------------------------------------------------------------
    # Default precision: BF16 (TPU training default; vs INT8 for
    # NPU/DPU/CGRA inference-default).
    if Precision.BF16 in precision_profiles:
        default_precision = Precision.BF16
    elif Precision.INT8 in precision_profiles:
        default_precision = Precision.INT8
    else:
        default_precision = next(iter(precision_profiles))

    # Energy fields: use the first fabric.
    first_fabric = compute_fabrics[0]
    energy_per_flop_fp32 = first_fabric.energy_per_flop_fp32
    energy_scaling: dict[Precision, float] = {}
    for fabric in compute_fabrics:
        energy_scaling.update(fabric.energy_scaling)

    model = HardwareResourceModel(
        name=name_override or cp.name,
        hardware_type=HardwareType.TPU,

        compute_fabrics=compute_fabrics,

        compute_units=block.num_mxus,
        threads_per_unit=block.mxu_dim_rows * block.mxu_dim_cols,
        warps_per_unit=block.mxu_dim_rows,    # rows in systolic array
        warp_size=block.mxu_dim_cols,         # columns in systolic array

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

        # M3 Layer 3: software-managed Unified Buffer.
        l1_storage_kind="scratchpad",

        # M4 Layer 4: UB IS the LLC (collapses L1+L2). No separate L2.
        l2_cache_per_unit=l1_per_unit_bytes,
        l2_topology="shared-llc",

        # M5 Layer 5: TPUs don't have L3.
        l3_present=False,
        l3_cache_total=0,

        # XLA-routed systolic dataflow has no coherence
        coherence_protocol=mem.coherence_protocol,

        # M7 Layer 7
        memory_technology=memory_technology,
        memory_read_energy_per_byte_pj=read_pj,
        memory_write_energy_per_byte_pj=write_pj,

        # M6 Layer 6
        soc_fabric=soc_fabric,
    )

    # Attach the TPU tile energy model (TPU-specific architectural
    # field; lives on the legacy HardwareResourceModel as an attribute)
    model.tile_energy_model = tile_energy_model

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

    fabric_provenance = EstimationConfidence.theoretical(
        score=0.55,
        source=(
            f"Loaded from compute_products YAML ({base_id}): per-fabric "
            f"ops_per_unit_per_clock at the systolic MXU fabric"
        ),
    )
    for precision in supported_precisions:
        model.set_provenance(
            f"compute_fabric.ops_per_clock.{precision.value}",
            fabric_provenance,
        )

    return model
