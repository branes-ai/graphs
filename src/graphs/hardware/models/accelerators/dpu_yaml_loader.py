"""DPU resource model loader from embodied-schemas ComputeProduct YAMLs.

PR 4 of the DPU mini-sprint scoped at issue #200. Mirrors
``src/graphs/hardware/models/accelerators/cgra_yaml_loader.py`` (CGRA
sprint #198) but reads a DPUBlock-bearing ``ComputeProduct`` and
builds a ``HardwareResourceModel`` with the DPU-shaped fields populated.

Sibling to the hand-coded factory at
``src/graphs/hardware/models/accelerators/xilinx_vitis_ai_dpu.py``
during the parallel-migration phase. Adding the factory swap is PR 5;
this PR ships the loader machinery and a parity test that proves the
YAML-loaded model matches the hand-coded one's key fields.

Scope of DPUBlock -> HardwareResourceModel mapping (one-to-one unless
noted):

  DPUBlock.num_aie_tiles          -> compute_units
  DPUBlock.macs_per_tile          -> threads_per_unit
  DPUBlock.simd_lanes_per_tile    -> warps_per_unit (legacy compat)
  DPUBlock.compute_fabrics[*]     -> compute_fabrics[] (num_units =
                                     num_aie_tiles per fabric)
  DPUBlock.memory.{scratchpad,    -> peak_bandwidth, l1/l2_cache_*,
    shared, external_dram}           main_memory, memory_technology,
                                     memory_*_energy_per_byte_pj
  DPUBlock.noc                    -> soc_fabric (SoCFabricModel)
  DPUBlock.{min_occupancy,        -> matching HardwareResourceModel fields
    max_concurrent_models,
    wave_quantization}
  ComputeProduct.power.thermal_profiles[]
                                  -> thermal_operating_points
  Roll-up performance             -> precision_profiles

DPU-specific design decisions:

1. **peak_bandwidth picks external_dram_bandwidth_gbps** when
   has_external_dram=True (Versal VE2302 has chip-attached DDR4 at
   50 GB/s). Mirrors NPU's "DRAM tier when present" convention --
   different from CGRA, where on-chip mesh is the bottleneck because
   workloads stay on-chip steady-state. DPUs use DDR4 as the model
   weight tier; bandwidth-bound workloads hit the DDR4 ceiling.

2. **energy_per_flop_fp32 uses fabric's FP32 scaling x
   fpga_fabric_overhead_factor**. DPUs ship honest FP energy scaling
   (like CGRAs) but also pay the FPGA-vs-ASIC penalty (25% for
   Vitis AI). The loader multiplies through both.

3. **HardwareType.DPU already exists** on the graphs side (unlike
   NPU which needed #191). The loader sets it directly with no
   transitional period.

4. **is_statically_reconfigurable + bitstream_load_time_ms NOT
   surfaced on HardwareResourceModel**. The v6 schema carries these
   on DPUBlock but the legacy graphs model has no equivalent fields.
   v7 reconciliation will add them. Downstream consumers that need
   them read directly from the ComputeProduct.

Out of scope for v6 (deferred to v7):

  - BOMCostProfile (YAML doesn't carry; v7 Market.bom)
  - Per-field provenance copy-through with rich source citations
  - ``is_statically_reconfigurable`` / ``bitstream_load_time_ms`` /
    ``fpga_fabric_overhead_factor`` surfacing on HardwareResourceModel
    (v7 reconciliation)
  - Multi-block-per-die (modeling the Versal ARM Cortex-A72 control
    complex as a sibling CPUBlock on the same die)
"""

from __future__ import annotations

from typing import Optional

from embodied_schemas.compute_product import ComputeProduct
from embodied_schemas.dpu_block import (
    DPUBlock,
    DPUFabricKind,
    DPUNoCTopology,
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

class DPUYamlLoaderError(Exception):
    """Raised when a ComputeProduct YAML can't be turned into a DPU
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

# DPUFabricKind -> ComputeFabric.fabric_type. The choice of name
# matches the existing hand-coded factory's fabric_type string
# ("aie_ml_tile") so parity tests can pin equality.
_FABRIC_TYPE_BY_KIND: dict[DPUFabricKind, str] = {
    DPUFabricKind.AIE_ML_V1:    "aie_ml_tile",
    DPUFabricKind.AIE_ML_V2:    "aie_ml_v2_tile",
    DPUFabricKind.AIE_HD:       "aie_hd_tile",
    DPUFabricKind.SOFT_LUT_DPU: "soft_lut_dpu",
}

# DPUs use standard-cell designs across all fabric kinds (AIE tiles
# are hardened standard cells; FPGA glue logic is LUT-based but
# wrapped under the same circuit_type for legacy compat).
_CIRCUIT_TYPE_DEFAULT = "standard_cell"

# DPUNoCTopology -> graphs SoCFabricModel Topology enum.
_TOPOLOGY_BY_KIND: dict[DPUNoCTopology, Topology] = {
    DPUNoCTopology.AIE_MESH: Topology.MESH_2D,  # AIE streaming mesh degenerates to mesh
    DPUNoCTopology.CROSSBAR: Topology.CROSSBAR,
}

# DRAM read-energy table (per-byte). External DRAM only.
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

def _dpu_block(cp: ComputeProduct) -> DPUBlock:
    """Pick the (single) DPUBlock from a ComputeProduct's first die."""
    for block in cp.dies[0].blocks:
        if isinstance(block, DPUBlock):
            return block
    raise DPUYamlLoaderError(
        f"ComputeProduct {cp.id!r} has no DPUBlock in dies[0].blocks "
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
        raise DPUYamlLoaderError(
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
        raise DPUYamlLoaderError(
            f"unknown precision name(s) in {source}: {sorted(unknown)}. "
            f"Known: {sorted(_PRECISION_BY_NAME)}"
        )
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_dpu_resource_model_from_yaml(
    base_id: str,
    *,
    products: Optional[dict[str, ComputeProduct]] = None,
    process_nodes: Optional[dict[str, ProcessNodeEntry]] = None,
    name_override: Optional[str] = None,
) -> HardwareResourceModel:
    """Build a ``HardwareResourceModel`` for ``base_id`` from a DPU
    ComputeProduct YAML.

    Args:
        base_id: ComputeProduct id, e.g., "xilinx_vitis_ai_b4096".
        products / process_nodes: optional pre-loaded catalogs.
        name_override: optional override for the resource model's
            ``name`` field. Vitis AI YAML name is
            "Xilinx Vitis AI B4096 (Versal VE2302)"; the hand-coded
            factory uses "DPU-Vitis-AI-B4096" -- pass that to
            preserve compat.

    Raises:
        DPUYamlLoaderError: when ``base_id`` is not in the catalog,
            doesn't carry a DPUBlock, or has unresolvable references.
    """
    if products is None:
        products = load_compute_products()
    if process_nodes is None:
        from embodied_schemas.loaders import load_process_nodes
        process_nodes = load_process_nodes()

    cp = products.get(base_id)
    if cp is None:
        raise DPUYamlLoaderError(
            f"no ComputeProduct with id={base_id!r}. Available: "
            f"{', '.join(sorted(products))}"
        )
    block = _dpu_block(cp)

    node = process_nodes.get(cp.dies[0].process_node_id)
    if node is None:
        raise DPUYamlLoaderError(
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
        raise DPUYamlLoaderError(
            f"SKU {base_id!r}: default_thermal_profile "
            f"{cp.power.default_thermal_profile!r} is not in thermal_profiles"
        )
    default_clock_hz = default_profile.clock_mhz * 1e6

    # ------------------------------------------------------------------
    # ComputeFabric per DPU fabric. DPUs usually ship a single fabric
    # (Vitis AI: AIE-ML v1 tile array). num_units = num_aie_tiles
    # because the fabric "lives" on all tiles.
    # ------------------------------------------------------------------
    compute_fabrics: list[ComputeFabric] = []
    for dpu_fabric in block.compute_fabrics:
        ops_dict = _precisions_from_ops_dict(
            dpu_fabric.ops_per_unit_per_clock,
            source=f"fabric.{dpu_fabric.fabric_kind.value}",
        )
        if not ops_dict:
            continue
        # DPUs ship honest FP energy scaling AND pay the FPGA-vs-ASIC
        # overhead penalty. Compute energy_per_flop_fp32 from the
        # fabric's INT8 baseline * FP32 scaling * fpga overhead.
        energy_per_op_int8_j = (
            dpu_fabric.energy_per_op_int8_pj * 1e-12
            * dpu_fabric.fpga_fabric_overhead_factor
        )
        dpu_scaling = _energy_scaling_from_yaml(dpu_fabric.energy_scaling)
        if Precision.FP32 in dpu_scaling:
            energy_per_flop_fp32_j = energy_per_op_int8_j * dpu_scaling[Precision.FP32]
        else:
            energy_per_flop_fp32_j = energy_per_op_int8_j * 8.0
        # ComputeFabric.energy_scaling expects FP32-baseline scaling
        # (legacy convention). Translate: scaling_fp32 = scaling_int8 / fp32_scaling.
        fp32_scaling: dict[Precision, float] = {}
        fp32_baseline = dpu_scaling.get(Precision.FP32, 8.0)
        fp32_scaling[Precision.INT8] = 1.0 / fp32_baseline
        for prec, factor in dpu_scaling.items():
            if prec == Precision.FP32:
                continue
            fp32_scaling[prec] = factor / fp32_baseline
        compute_fabrics.append(ComputeFabric(
            fabric_type=_FABRIC_TYPE_BY_KIND[dpu_fabric.fabric_kind],
            circuit_type=_CIRCUIT_TYPE_DEFAULT,
            num_units=block.num_aie_tiles,
            ops_per_unit_per_clock=ops_dict,
            core_frequency_hz=default_clock_hz,
            process_node_nm=process_node_nm,
            energy_per_flop_fp32=energy_per_flop_fp32_j,
            energy_scaling=fp32_scaling,
        ))

    if not compute_fabrics:
        raise DPUYamlLoaderError(
            f"SKU {base_id!r}: no DPUComputeFabric produced a valid "
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
        # AIE tiles act like tensor cores for INT8 / FP16 (native
        # hardware support).
        tensor_supported = precision in {Precision.INT8, Precision.FP16, Precision.INT4}
        # Relative speedup vs INT8 (the DPU baseline).
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
            performance_specs={},
        )

    # ------------------------------------------------------------------
    # Memory + cache (scratchpad + shared L2 + chip-attached DDR)
    # ------------------------------------------------------------------
    mem = block.memory
    # peak_bandwidth picks the dominant *externally-visible* memory tier.
    # DPUs use DDR4 as the model-weight tier; bandwidth-bound workloads
    # hit the DDR4 ceiling. Mirrors NPU loader convention ("DRAM tier
    # when present") -- different from CGRA where on-chip mesh dominates
    # because workloads stay on-chip steady-state.
    if mem.has_external_dram and mem.external_dram_bandwidth_gbps is not None:
        peak_bandwidth_bps = mem.external_dram_bandwidth_gbps * 1e9
    else:
        peak_bandwidth_bps = mem.on_chip_bandwidth_gbps * 1e9

    # External DRAM -> main_memory.
    if mem.has_external_dram and mem.external_dram_size_gb is not None:
        main_memory_bytes = int(mem.external_dram_size_gb * 1024**3)
    else:
        main_memory_bytes = 0

    # Per-AIE-tile scratchpad is the "L1" of DPU-land.
    l1_per_unit_bytes = mem.scratchpad_kib_per_tile * 1024

    # Shared L2 SRAM.
    l2_total_bytes = mem.shared_sram_kib * 1024
    l2_per_unit_bytes = (
        l2_total_bytes // block.num_aie_tiles
        if block.num_aie_tiles > 0 else 0
    )

    # Memory technology label and per-byte access energy.
    if mem.has_external_dram and mem.external_dram_type is not None:
        memory_technology = mem.external_dram_type.value.upper()
        dram_tech = mem.external_dram_type.value.lower()
        read_pj = _DRAM_READ_PJ_PER_BYTE.get(dram_tech, 25.0)
        write_pj = read_pj * _DRAM_WRITE_RATIO
        # Honor the YAML's external_dram_access_energy_pj_per_byte
        # when populated (more accurate than the lookup table).
        if mem.external_dram_access_energy_pj_per_byte > 0:
            read_pj = mem.external_dram_access_energy_pj_per_byte
            write_pj = read_pj * _DRAM_WRITE_RATIO
    else:
        memory_technology = "on-chip SRAM (no external DRAM)"
        read_pj = mem.scratchpad_access_energy_pj_per_byte
        write_pj = read_pj

    # ------------------------------------------------------------------
    # SoC fabric (AIE streaming mesh)
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
            f"({base_id}.dies[0].blocks[0].noc); DPU NoC topology "
            f"{block.noc.topology.value} mapped to graphs SoCFabric "
            f"{_TOPOLOGY_BY_KIND[block.noc.topology].name}"
        ),
    )
    # Mesh dimensions: pass through when AIE_MESH
    if (block.noc.topology == DPUNoCTopology.AIE_MESH
            and block.noc.mesh_rows is not None
            and block.noc.mesh_cols is not None):
        soc_fabric_kwargs["mesh_dimensions"] = (
            block.noc.mesh_rows, block.noc.mesh_cols
        )
    from embodied_schemas.process_node import DataConfidence
    if block.noc.confidence == DataConfidence.THEORETICAL:
        soc_fabric_kwargs["low_confidence"] = True
    soc_fabric = SoCFabricModel(**soc_fabric_kwargs)

    # ------------------------------------------------------------------
    # Assemble HardwareResourceModel
    # ------------------------------------------------------------------
    # Default precision: INT8 (the DPU default).
    if Precision.INT8 in precision_profiles:
        default_precision = Precision.INT8
    elif Precision.INT4 in precision_profiles:
        default_precision = Precision.INT4
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
        hardware_type=HardwareType.DPU,

        compute_fabrics=compute_fabrics,

        compute_units=block.num_aie_tiles,
        threads_per_unit=block.macs_per_tile,
        warps_per_unit=block.simd_lanes_per_tile,
        warp_size=block.simd_lanes_per_tile,

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

        # M3 Layer 3: software-managed AIE tile scratchpad.
        l1_storage_kind="scratchpad",

        # M4 Layer 4: shared L2 acts as LLC above per-tile scratchpads.
        l2_cache_per_unit=l2_per_unit_bytes,
        l2_topology="shared-llc",

        # M5 Layer 5: DPUs don't have L3 between AIE tiles.
        l3_present=False,
        l3_cache_total=0,

        # AIE streaming dataflow has no coherence
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
            f"ops_per_unit_per_clock at the dominant AIE-ML fabric"
        ),
    )
    for precision in supported_precisions:
        model.set_provenance(
            f"compute_fabric.ops_per_clock.{precision.value}",
            fabric_provenance,
        )

    return model
