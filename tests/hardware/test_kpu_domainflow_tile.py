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

    def test_t64_uses_32x32_pe_arrays(self):
        """T64 uses 32x32 - the canonical KPU tile size. Requires a TDP
        envelope above 6 W because 32x32 x 64 tiles at 0.10 pJ/MAC @ 16 nm
        exceeds the original 6 W ALU budget; run
        cli/check_tdp_feasibility.py to pick a valid envelope."""
        model = kpu_t64_resource_model()
        tp = model.thermal_operating_points[model.default_thermal_profile]
        cr = tp.performance_specs[Precision.INT8].compute_resource
        for spec in cr.tile_specializations:
            assert spec.array_dimensions == (32, 32), (
                f"T64 tile {spec.tile_type} has {spec.array_dimensions}, "
                f"expected (32, 32)"
            )

    def test_t128_uses_32x32_pe_arrays(self):
        model = kpu_t128_resource_model()
        tp = model.thermal_operating_points[model.default_thermal_profile]
        cr = tp.performance_specs[Precision.INT8].compute_resource
        for spec in cr.tile_specializations:
            assert spec.array_dimensions == (32, 32)

    def test_t256_uses_32x32_pe_arrays(self):
        """T256 uses 32x32 (uniform with T64/T128 since the post-PR#152
        family-consistency rebuild). The earlier 20x20 design was a
        hand-tuned compromise to keep T256 in a 30W envelope at 1.4 GHz;
        once TDP became derived from (clock, Vdd) per PR #153, the
        20x20 inverse-scaling design produced a discontinuity in the
        cost/perf curve and was retired in favor of the uniform 32x32.
        T256 now has 256 tiles x 1024 PEs = 262,144 PEs total."""
        model = kpu_t256_resource_model()
        tp = model.thermal_operating_points[model.default_thermal_profile]
        cr = tp.performance_specs[Precision.INT8].compute_resource
        for spec in cr.tile_specializations:
            assert spec.array_dimensions == (32, 32)


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
            tile_type="OS", num_tiles=1, array_dimensions=(32, 32),
            pe_configuration="test",
            ops_per_tile_per_clock={Precision.INT8: 2048},
            optimization_level={Precision.INT8: 1.0},
            clock_domain=clock,
            schedule_class=TileScheduleClass.OUTPUT_STATIONARY,
            pipeline_fill_cycles=32, pipeline_drain_cycles=32,
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
        """KPU's key product claim: ~1.0 utilization at 12+ tiles.
        At 32x32 canonical tile (fill/drain = 32 cycles each,
        steady = 128), util at 12 tiles is 1536/1600 = 0.96 exactly."""
        tile = self._os_tile()
        util_12 = tile.effective_pipeline_utilization(12, steady_cycles_per_tile=128)
        assert util_12 >= 0.96, (
            f"Output-stationary at 12 tiles should saturate at >= 0.96; "
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
        """All edge KPUs use the canonical 32x32 PE tile. T256 was
        retrofitted from 20x20 to 32x32 in the post-PR#152 family-
        consistency rebuild so PE count scales linearly with tile count
        across the family, making roadmap sweeps clean:
        T64  = 32x32 (1024 PEs/tile) -> 65,536 PEs total
        T128 = 32x32 (1024 PEs/tile) -> 131,072 PEs total
        T256 = 32x32 (1024 PEs/tile) -> 262,144 PEs total
        TDP scaling is handled per-profile via (clock, Vdd) operating
        points (PR #153); the 32x32 design no longer creates an
        envelope problem because the architect tunes Vdd per profile."""
        t64 = kpu_t64_resource_model()
        t128 = kpu_t128_resource_model()
        t256 = kpu_t256_resource_model()
        def pe_count(m):
            tp = m.thermal_operating_points[m.default_thermal_profile]
            cr = tp.performance_specs[Precision.INT8].compute_resource
            return cr.tile_specializations[0].pe_count
        assert pe_count(t64) == 1024
        assert pe_count(t128) == 1024
        assert pe_count(t256) == 1024

    def test_t256_peak_throughput_exceeds_t128(self):
        """Commercial sanity: T256 at the bigger TDP must deliver more
        peak INT8 throughput than T128.

        With the 32x32 canonical tile applied to T64/T128 and T256
        keeping its smaller 20x20 tiles, T128 now has MORE total PEs
        (128 x 1024 = 131,072) than T256 (256 x 400 = 102,400). T256
        still out-performs T128 on peak throughput because of its
        higher clock (1.4 GHz vs 1.0 GHz), but by a smaller margin
        than before. This is a design decision that may need
        revisiting if the tile-size inversion is commercially
        unacceptable."""
        t128 = kpu_t128_resource_model()
        t256 = kpu_t256_resource_model()
        def peak_mac_per_sec(m):
            tp = m.thermal_operating_points[m.default_thermal_profile]
            cr = tp.performance_specs[Precision.INT8].compute_resource
            spec = cr.tile_specializations[0]
            ops_int8 = spec.ops_per_tile_per_clock[Precision.INT8]
            clock_hz = spec.clock_domain.sustained_clock_hz
            return cr.total_tiles * ops_int8 * clock_hz
        assert peak_mac_per_sec(t256) > peak_mac_per_sec(t128), (
            f"T256 peak ({peak_mac_per_sec(t256)/1e12:.1f} TOPS) must "
            f"exceed T128 ({peak_mac_per_sec(t128)/1e12:.1f} TOPS)."
        )
