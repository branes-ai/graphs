"""Phase C5a of the KPU heterogeneous-tile refactor (graphs#268): the
resource-model loader, per-class energy models, fixed-function units, and
the reporting ``tile_specializations[0]`` fixes.

Uniform legacy SKUs load unchanged (the KPU golden snapshot pins the full
resource model); the synthetic heterogeneous fixture now loads.
"""

from __future__ import annotations

import pytest
from embodied_schemas import ComputeProduct, load_compute_products, load_process_nodes

from graphs.hardware.kpu_access import has_kpu_block
from graphs.hardware.kpu_hetero_fixture import build_heterogeneous_kpu
from graphs.hardware.kpu_power_model import fixed_function_pj_per_unit
from graphs.hardware.kpu_sku_generator import generate_kpu_sku, input_spec_from_compute_product
from graphs.hardware.models.accelerators.kpu_yaml_loader import load_kpu_resource_model_from_yaml
from graphs.hardware.resource_model import (
    FixedFunctionUnit,
    HardwareResourceModel,
    KPUComputeResource,
    Precision,
    TileSpecialization,
)

NODES = load_process_nodes()
N16 = NODES["tsmc_n16"]
CATALOG = {k: v for k, v in load_compute_products().items() if has_kpu_block(v)}
HETERO = generate_kpu_sku(input_spec_from_compute_product(build_heterogeneous_kpu()),
                          process_nodes=NODES)


def _load(cp: ComputeProduct):
    return load_kpu_resource_model_from_yaml(cp.id, kpus={cp.id: cp}, process_nodes=NODES)


def _compute(rm, profile=None, precision=Precision.INT8):
    tp = rm.thermal_operating_points[profile or rm.default_thermal_profile]
    return tp.performance_specs[precision].compute_resource


HETERO_RM = _load(HETERO)


# ---------------------------------------------------------------------------
# Legacy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sku", sorted(CATALOG))
def test_legacy_skus_load_with_the_new_attachments(sku):
    cp = CATALOG[sku]
    rm = load_kpu_resource_model_from_yaml(sku)
    block = cp.dies[0].blocks[0]
    assert [f.fabric_type for f in rm.compute_fabrics] == [
        f"kpu_{t.tile_class_id}" for t in block.tiles
    ]
    assert rm.fixed_function_units == ()
    assert set(rm.tile_energy_models) == {t.tile_class_id for t in block.tiles}
    cr = _compute(rm)
    # The dominant class is the first one in every catalog SKU, so the
    # reports see the same representative as before for INT8.
    assert cr.representative_specialization(Precision.INT8) is cr.tile_specializations[0]


def test_new_attachments_are_not_dataclass_fields():
    """The golden snapshot serializes dataclass fields; the new attachments
    must not change it for other hardware or uniform KPUs."""
    import dataclasses

    names = {f.name for f in dataclasses.fields(HardwareResourceModel)}
    assert "fixed_function_units" not in names and "tile_energy_models" not in names
    assert HardwareResourceModel.fixed_function_units == ()
    assert HardwareResourceModel.tile_energy_models is None


# ---------------------------------------------------------------------------
# The heterogeneous fixture
# ---------------------------------------------------------------------------


def test_fixture_fabrics_and_specializations():
    rm = HETERO_RM
    # Programmable classes with a Precision-typed op only: the LNS and
    # min-plus classes have no enum precision; fixed-function tiles have no ops.
    assert [f.fabric_type for f in rm.compute_fabrics] == [
        "kpu_pe_int8_mac_i32", "kpu_systolic_int8_ws",
    ]
    specs = {s.tile_type: s for s in _compute(rm).tile_specializations}
    assert set(specs) == {"INT8-MAC", "Systolic-INT8-WS"}
    systolic = specs["Systolic-INT8-WS"]
    assert systolic.array_dimensions == (64, 64)
    assert systolic.schedule_class.value == "weight_stationary"
    assert systolic.pipeline_fill_cycles == 64
    assert rm.threads_per_unit == 64 * 64  # the largest programmable array
    clock = next(p.clock_mhz for p in HETERO.power.thermal_profiles
                 if p.name == HETERO.power.default_thermal_profile) * 1e6
    int8 = (24 * 2048 + 4 * 8192) * clock
    assert rm.precision_profiles[Precision.INT8].peak_ops_per_sec == pytest.approx(
        round(int8 / 1e12, 1) * 1e12
    )


def test_fixture_fixed_function_units():
    units = {u.tile_class_id: u for u in HETERO_RM.fixed_function_units}
    assert set(units) == {"ff_isp_raw2yuv", "ff_stereo_sgm", "ff_vio_stereo_inertial"}
    clock = next(p.clock_mhz for p in HETERO.power.thermal_profiles
                 if p.name == HETERO.power.default_thermal_profile) * 1e6
    tiles = {t.tile_class_id: t for t in HETERO.dies[0].blocks[0].tiles}
    for cid, unit in units.items():
        core = tiles[cid].core
        assert isinstance(unit, FixedFunctionUnit)
        assert unit.function_id == core.function_id and unit.work_unit == core.throughput.unit.value
        assert unit.units_per_second == pytest.approx(core.units_per_clock * clock)
        assert unit.energy_per_unit_j == pytest.approx(
            fixed_function_pj_per_unit(core, N16, NODES) * 1e-12
        )
    assert units["ff_isp_raw2yuv"].output_bytes_per_unit == 1.5


def test_per_class_energy_models():
    models = HETERO_RM.tile_energy_models
    # Every programmable class has a model, including the ones with no
    # Precision-enum ops and so no ComputeFabric (CodeRabbit on #283).
    assert set(models) == {
        "pe_int8_mac_i32", "pe_lns16_mac", "pe_minplus_i16", "systolic_int8_ws",
    }
    assert models["pe_minplus_i16"].pes_per_tile == 32 * 32
    anchor = N16.energy_per_op_pj["balanced_logic:int8"] * 1e-12
    assert models["pe_int8_mac_i32"].mac_energy_int8 == pytest.approx(anchor)  # ratio 1.0
    assert models["systolic_int8_ws"].mac_energy_int8 == pytest.approx(0.65 * anchor)
    assert models["systolic_int8_ws"].pes_per_tile == 64 * 64
    # The dominant-class model keeps the legacy construction.
    assert HETERO_RM.tile_energy_model.pes_per_tile == 32 * 32
    assert HETERO_RM.tile_energy_model.mac_energy_int8 == pytest.approx(anchor)


def _with_domain(op) -> ComputeProduct:
    data = HETERO.model_dump(mode="json")
    block = data["dies"][0]["blocks"][0]
    block["power_domains"] = [{"domain_id": "int8_pe", "kind": "tile_class",
                               "members": ["pe_int8_mac_i32"], "gateable": True}]
    for p in data["power"]["thermal_profiles"]:
        p["domain_operating_points"] = {"int8_pe": op}
    return ComputeProduct.model_validate(data)


def test_power_domain_clock_reaches_the_class():
    rm = _load(_with_domain({"clock_mhz": 200}))
    fabrics = {f.fabric_type: f for f in rm.compute_fabrics}
    default_hz = next(p.clock_mhz for p in HETERO.power.thermal_profiles
                      if p.name == HETERO.power.default_thermal_profile) * 1e6
    assert fabrics["kpu_pe_int8_mac_i32"].core_frequency_hz == 200e6
    assert fabrics["kpu_systolic_int8_ws"].core_frequency_hz == default_hz
    for name in rm.thermal_operating_points:
        specs = {s.tile_type: s for s in _compute(rm, name).tile_specializations}
        assert specs["INT8-MAC"].clock_domain.sustained_clock_hz == 200e6
        assert specs["Systolic-INT8-WS"].clock_domain.sustained_clock_hz != 200e6


def test_gated_domain_leaves_the_profile_compute():
    rm = _load(_with_domain({"gated": True}))
    for name in rm.thermal_operating_points:
        types = [s.tile_type for s in _compute(rm, name).tile_specializations]
        assert types == ["Systolic-INT8-WS"]


# ---------------------------------------------------------------------------
# representative_specialization
# ---------------------------------------------------------------------------


def _spec(tile_type, num_tiles, precisions):
    return TileSpecialization(
        tile_type=tile_type, num_tiles=num_tiles,
        ops_per_tile_per_clock={p: 1 for p in precisions},
        optimization_level={p: 1.0 for p in precisions},
        clock_domain=None, array_dimensions=(1, 1), pe_configuration=tile_type,
    )


def test_representative_specialization():
    a = _spec("A", 8, [Precision.INT8])
    b = _spec("B", 20, [Precision.INT8, Precision.BF16])
    c = _spec("C", 20, [Precision.BF16])
    cr = KPUComputeResource(total_tiles=48, tile_specializations=[a, b, c])
    assert cr.representative_specialization(Precision.INT8) is b  # most tiles running it
    assert cr.representative_specialization(Precision.BF16) is b  # tie: declaration order
    assert cr.representative_specialization(Precision.FP32) is a  # nobody runs it: first
    assert cr.representative_specialization() is b
    assert KPUComputeResource(total_tiles=0, tile_specializations=[]).representative_specialization() is None


def test_legacy_fp32_representative_is_a_class_that_runs_fp32():
    rm = load_kpu_resource_model_from_yaml("kpu_t64_32x32_lp5x4_16nm_tsmc_ffp")
    rep = _compute(rm, precision=Precision.FP32).representative_specialization(Precision.FP32)
    assert rep.tile_type == "BF16-primary"  # the only T64 class with fp32 ops


def test_mac_energy_override_reaches_every_class_model():
    """A SKU factory's measured MAC energies apply to the chip-level model
    and to each class model, so a report that selects a class model sees
    them too (CodeRabbit on #283)."""
    from graphs.hardware.models.accelerators.kpu_t64 import kpu_t64_resource_model

    rm = kpu_t64_resource_model()
    assert rm.tile_energy_model.mac_energy_fp32 == 0.30e-12
    assert rm.tile_energy_models
    for tem in rm.tile_energy_models.values():
        assert (tem.mac_energy_int8, tem.mac_energy_bf16, tem.mac_energy_fp32) == (
            0.10e-12, 0.16e-12, 0.30e-12
        )
    # The FP32 representative class is BF16-primary, and its model carries
    # the override.
    cr = _compute(rm, precision=Precision.FP32)
    spec = cr.representative_specialization(Precision.FP32)
    assert spec.tile_type == "BF16-primary"
    assert rm.energy_model_for_tile_type(spec.tile_type).mac_energy_fp32 == 0.30e-12


def test_native_op_energy_uses_the_selected_class_model():
    from graphs.reporting.native_op_energy import build_kpu_native_op

    op = build_kpu_native_op("Stillwater-KPU-T64", Precision.FP32)
    alu = next(layer for layer in op.layers if layer.name.startswith("ALU"))
    # ALU energy is the selected class's FP32 MAC energy (the T64 override),
    # and the source names the class it came from.
    assert alu.energy_pj_per_mac == pytest.approx(0.30)
    assert alu.source.startswith("tile_energy_model[BF16-primary].mac_energy_fp32")


def test_gated_profile_drops_its_precisions():
    """A profile whose only fp16 / int4 class is gated must not advertise
    those precisions (CodeRabbit on #283)."""
    rm = _load(_with_domain({"gated": True}))
    for name in rm.thermal_operating_points:
        tp = rm.thermal_operating_points[name]
        precisions = set(tp.performance_specs)
        assert precisions == {Precision.INT8}  # only the systolic class remains
        for precision, perf in tp.performance_specs.items():
            assert perf.compute_resource.calc_peak_ops(precision) > 0
