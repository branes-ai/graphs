"""
Internal-consistency tests for the M0.5 KPU domain-flow-tile abstraction.

Asserts the comparative behavior that the refined model is supposed to
encode. Not silicon validation - these are model self-check tests that
catch sign errors, unit mistakes, and regressions in the product
narrative.
"""
from __future__ import annotations

from graphs.hardware.mappers import get_mapper_by_name, list_all_mappers
from graphs.hardware.models.accelerators.kpu_t64 import kpu_t64_resource_model
from graphs.hardware.models.accelerators.kpu_t128 import kpu_t128_resource_model
from graphs.hardware.models.accelerators.kpu_t256 import kpu_t256_resource_model
from graphs.hardware.resource_model import (
    Precision,
    TileScheduleClass,
    TileSpecialization,
)


class TestPE_ArraySizes:
    """T64 / T128 / T256 use inverse-scaled per-tile PE array sizes."""

    def test_t64_uses_24x24_pe_arrays(self):
        """T64 uses 24x24 (revised from 32x32 after TDP feasibility check).
        At 6W TDP with 0.10 pJ/MAC @ 16nm, 32x32 x 64 tiles exceeded
        the ALU power budget. 24x24 fits comfortably."""
        model = kpu_t64_resource_model()
        tp = model.thermal_operating_points[model.default_thermal_profile]
        cr = tp.performance_specs[Precision.INT8].compute_resource
        for spec in cr.tile_specializations:
            assert spec.array_dimensions == (24, 24), (
                f"T64 tile {spec.tile_type} has {spec.array_dimensions}, "
                f"expected (24, 24)"
            )

    def test_t128_uses_24x24_pe_arrays(self):
        model = kpu_t128_resource_model()
        tp = model.thermal_operating_points[model.default_thermal_profile]
        cr = tp.performance_specs[Precision.INT8].compute_resource
        for spec in cr.tile_specializations:
            assert spec.array_dimensions == (24, 24)

    def test_t256_uses_20x20_pe_arrays(self):
        """T256 uses 20x20 (revised from 16x16 after commercial review).
        At 16x16 x 256 tiles, T256 had fewer total PEs (65,536) than T128
        (73,728) - making it commercially unjustifiable at 2.5x TDP.
        20x20 x 256 = 102,400 PEs gives T256 a ~1.56x peak advantage."""
        model = kpu_t256_resource_model()
        tp = model.thermal_operating_points[model.default_thermal_profile]
        cr = tp.performance_specs[Precision.INT8].compute_resource
        for spec in cr.tile_specializations:
            assert spec.array_dimensions == (20, 20)


class TestScheduleClass:
    """All three KPU SKUs mark tiles as OUTPUT_STATIONARY."""

    def test_t64_output_stationary(self):
        model = kpu_t64_resource_model()
        tp = model.thermal_operating_points[model.default_thermal_profile]
        cr = tp.performance_specs[Precision.INT8].compute_resource
        for spec in cr.tile_specializations:
            assert spec.schedule_class is TileScheduleClass.OUTPUT_STATIONARY

    def test_t128_output_stationary(self):
        model = kpu_t128_resource_model()
        tp = model.thermal_operating_points[model.default_thermal_profile]
        cr = tp.performance_specs[Precision.INT8].compute_resource
        for spec in cr.tile_specializations:
            assert spec.schedule_class is TileScheduleClass.OUTPUT_STATIONARY

    def test_t256_output_stationary(self):
        model = kpu_t256_resource_model()
        tp = model.thermal_operating_points[model.default_thermal_profile]
        cr = tp.performance_specs[Precision.INT8].compute_resource
        for spec in cr.tile_specializations:
            assert spec.schedule_class is TileScheduleClass.OUTPUT_STATIONARY


class TestPipelineFields:
    """Fill/drain cycles are populated on every tile."""

    def test_t64_fill_drain_positive(self):
        model = kpu_t64_resource_model()
        tp = model.thermal_operating_points[model.default_thermal_profile]
        cr = tp.performance_specs[Precision.INT8].compute_resource
        for spec in cr.tile_specializations:
            assert spec.pipeline_fill_cycles > 0
            assert spec.pipeline_drain_cycles > 0

    def test_t256_fill_drain_positive(self):
        model = kpu_t256_resource_model()
        tp = model.thermal_operating_points[model.default_thermal_profile]
        cr = tp.performance_specs[Precision.INT8].compute_resource
        for spec in cr.tile_specializations:
            assert spec.pipeline_fill_cycles > 0
            assert spec.pipeline_drain_cycles > 0


class TestEffectivePipelineUtilization:
    """
    Output-stationary utilization saturates -> 1.0 at many tiles;
    weight-stationary is a flat floor; unspecified returns 1.0.
    """

    def _os_tile(self) -> TileSpecialization:
        """Build a representative KPU output-stationary tile (not tied to a SKU)."""
        from graphs.hardware.resource_model import ClockDomain
        clock = ClockDomain(base_clock_hz=1e9, max_boost_clock_hz=1e9,
                            sustained_clock_hz=1e9, dvfs_enabled=False)
        return TileSpecialization(
            tile_type="OS", num_tiles=1, array_dimensions=(24, 24),
            pe_configuration="test",
            ops_per_tile_per_clock={Precision.INT8: 1152},
            optimization_level={Precision.INT8: 1.0},
            clock_domain=clock,
            schedule_class=TileScheduleClass.OUTPUT_STATIONARY,
            pipeline_fill_cycles=24, pipeline_drain_cycles=24,
        )

    def _ws_tile(self) -> TileSpecialization:
        from graphs.hardware.resource_model import ClockDomain
        clock = ClockDomain(base_clock_hz=1e9, max_boost_clock_hz=1e9,
                            sustained_clock_hz=1e9, dvfs_enabled=False)
        return TileSpecialization(
            tile_type="WS", num_tiles=1, array_dimensions=(64, 64),
            pe_configuration="test",
            ops_per_tile_per_clock={Precision.INT8: 8192},
            optimization_level={Precision.INT8: 1.0},
            clock_domain=clock,
            schedule_class=TileScheduleClass.WEIGHT_STATIONARY,
            pipeline_fill_cycles=64, pipeline_drain_cycles=64,
        )

    def test_os_saturates_at_many_tiles(self):
        tile = self._os_tile()
        util_1 = tile.effective_pipeline_utilization(1, steady_cycles_per_tile=128)
        util_64 = tile.effective_pipeline_utilization(64, steady_cycles_per_tile=128)
        util_256 = tile.effective_pipeline_utilization(256, steady_cycles_per_tile=128)
        assert util_1 < util_64 < util_256
        assert util_256 > 0.98

    def test_os_reaches_saturation_by_12_tiles(self):
        """KPU's key product claim: ~1.0 utilization at 12+ tiles."""
        tile = self._os_tile()
        util_12 = tile.effective_pipeline_utilization(12, steady_cycles_per_tile=128)
        assert util_12 > 0.96, (
            f"Output-stationary at 12 tiles should saturate above 0.96; "
            f"got {util_12:.3f}"
        )

    def test_ws_is_flat(self):
        """Weight-stationary utilization does not improve with tile count."""
        tile = self._ws_tile()
        util_1 = tile.effective_pipeline_utilization(1, steady_cycles_per_tile=128)
        util_256 = tile.effective_pipeline_utilization(256, steady_cycles_per_tile=128)
        assert abs(util_1 - util_256) < 1e-6, (
            f"Weight-stationary should be flat; got {util_1:.3f} vs {util_256:.3f}"
        )

    def test_ws_capped_below_one(self):
        tile = self._ws_tile()
        util = tile.effective_pipeline_utilization(100, steady_cycles_per_tile=128)
        assert util < 0.9, (
            f"Weight-stationary must be capped below 0.9 (fill+drain=128, "
            f"steady=128); got {util:.3f}"
        )

    def test_os_beats_ws_at_large_tile_count(self):
        """The scheduling-class story: OS beats WS when N is large."""
        os_tile = self._os_tile()
        ws_tile = self._ws_tile()
        os_util = os_tile.effective_pipeline_utilization(64, steady_cycles_per_tile=128)
        ws_util = ws_tile.effective_pipeline_utilization(64, steady_cycles_per_tile=128)
        assert os_util > ws_util + 0.3, (
            f"Output-stationary should substantially beat weight-stationary "
            f"at N=64; got OS={os_util:.3f} WS={ws_util:.3f}"
        )

    def test_unspecified_returns_one(self):
        from graphs.hardware.resource_model import ClockDomain
        clock = ClockDomain(base_clock_hz=1e9, max_boost_clock_hz=1e9,
                            sustained_clock_hz=1e9, dvfs_enabled=False)
        tile = TileSpecialization(
            tile_type="U", num_tiles=1, array_dimensions=(8, 8),
            pe_configuration="test",
            ops_per_tile_per_clock={Precision.INT8: 64},
            optimization_level={Precision.INT8: 1.0},
            clock_domain=clock,
            schedule_class=TileScheduleClass.UNSPECIFIED,
        )
        assert tile.effective_pipeline_utilization(100) == 1.0


class TestMapperRegistry:
    """All three KPU SKUs are registered and resolvable."""

    def test_kpu_t128_in_registry(self):
        assert "Stillwater-KPU-T128" in list_all_mappers()

    def test_all_three_kpu_skus_resolve(self):
        for sku in ("Stillwater-KPU-T64", "Stillwater-KPU-T128",
                    "Stillwater-KPU-T256"):
            mapper = get_mapper_by_name(sku)
            assert mapper is not None, f"{sku} did not resolve"

    def test_t128_total_tiles_is_128(self):
        model = kpu_t128_resource_model()
        assert model.compute_units == 128
        tp = model.thermal_operating_points[model.default_thermal_profile]
        cr = tp.performance_specs[Precision.INT8].compute_resource
        assert cr.total_tiles == 128
        total = sum(spec.num_tiles for spec in cr.tile_specializations)
        assert total == 128


class TestEnergyAdvantage:
    """
    The narrative assertion: KPU per-PE steady-state MAC energy is
    below NVIDIA Tensor Core per-op energy at matched precision.
    """

    def test_kpu_int8_energy_below_tensor_core(self):
        model = kpu_t128_resource_model()
        tem = model.tile_energy_model
        # KPU INT8: ~0.27 pJ/MAC = ~0.135 pJ/op
        kpu_pj_per_op = tem.mac_energy_int8 / 2.0 * 1e12
        # Tensor Core FP16: ~1.62 pJ/op on Ampere; INT8 is similar order
        tensor_core_pj_per_op_int8 = 0.81  # 1.62/2
        assert kpu_pj_per_op < tensor_core_pj_per_op_int8, (
            f"KPU {kpu_pj_per_op:.3f} pJ/op must be below "
            f"Tensor Core {tensor_core_pj_per_op_int8:.3f} pJ/op"
        )

    def test_kpu_array_size_sanity(self):
        """M0.5 final SKU lineup (post-TDP + commercial review):
        T64  = 24x24 (576 PEs/tile) - shrunk from 32x32 for 6W TDP
        T128 = 24x24 (576 PEs/tile) - sweet spot
        T256 = 20x20 (400 PEs/tile) - commercially defensible peak
        Inverse-scaling story: 24 >= 24 > 20 as engine grows. Larger
        engines still get smaller per-tile arrays, just less aggressively
        than the original 24->16 intent allowed under TDP constraints."""
        t64 = kpu_t64_resource_model()
        t128 = kpu_t128_resource_model()
        t256 = kpu_t256_resource_model()
        def pe_count(m):
            tp = m.thermal_operating_points[m.default_thermal_profile]
            cr = tp.performance_specs[Precision.INT8].compute_resource
            return cr.tile_specializations[0].pe_count
        assert pe_count(t64) == 576
        assert pe_count(t128) == 576
        assert pe_count(t256) == 400

    def test_t256_has_more_pes_than_t128(self):
        """Commercial sanity: T256 at 2.5x TDP must have more PEs than
        T128, not fewer. (The original 16x16 T256 had 65,536 PEs vs
        T128's 73,728 - economically indefensible.)"""
        t128 = kpu_t128_resource_model()
        t256 = kpu_t256_resource_model()
        def total_pes(m):
            tp = m.thermal_operating_points[m.default_thermal_profile]
            cr = tp.performance_specs[Precision.INT8].compute_resource
            return cr.total_tiles * cr.tile_specializations[0].pe_count
        assert total_pes(t256) > total_pes(t128), (
            f"T256 ({total_pes(t256)} PEs) must have more PEs than "
            f"T128 ({total_pes(t128)} PEs) to justify 2.5x the TDP."
        )
