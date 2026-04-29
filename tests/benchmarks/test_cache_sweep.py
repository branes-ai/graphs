"""Tests for the cache_sweep analysis (Path B PR-1).

Hardware-touching code paths (`run_sweep`, RAPL probe) are not
exercised in CI -- they're nondeterministic and depend on hardware
energy counters. Instead we test the pure-analysis logic against
synthetic curves with known plateau structure.
"""
from __future__ import annotations

from typing import Optional

from graphs.benchmarks.cache_sweep import (
    CacheLevel,
    SweepConfig,
    WorkingSetPoint,
    detect_levels,
    estimate_per_level_energy,
)


def _point(bytes_resident: int, bandwidth: float,
           energy_pj: Optional[float] = None) -> WorkingSetPoint:
    return WorkingSetPoint(
        bytes_resident=bytes_resident,
        iterations=1,
        elapsed_seconds=0.1,
        bandwidth_gbps=bandwidth,
        energy_per_byte_pj=energy_pj,
    )


# A canonical synthetic curve that mimics a CPU with clean
# transitions. L1 ~32 KiB, L2 ~1 MiB, L3 ~16 MiB.
_CANONICAL_I7 = [
    _point(8 * 1024,    200.0, 0.20),
    _point(32 * 1024,   200.0, 0.20),
    _point(256 * 1024,   80.0, 0.50),  # L2 transition (200 -> 80 = 2.5x)
    _point(1024 * 1024,  80.0, 0.50),
    _point(4 * 1024**2,  35.0, 1.20),  # L3 transition (80 -> 35 = 2.3x)
    _point(16 * 1024**2, 35.0, 1.20),
    _point(64 * 1024**2,  8.0, 15.0),  # DRAM transition (35 -> 8 = 4.4x)
]


# --------------------------------------------------------------------
# SweepConfig sanity
# --------------------------------------------------------------------

class TestSweepConfig:
    def test_default_size_range_log_spaced(self):
        cfg = SweepConfig()
        sizes = cfg.working_set_sizes()
        assert len(sizes) == cfg.num_points
        assert sizes[0] == cfg.min_bytes
        assert sizes[-1] == cfg.max_bytes
        # Log-spaced -> consecutive ratios approximately equal
        ratios = [sizes[i+1] / sizes[i] for i in range(len(sizes) - 1)]
        avg = sum(ratios) / len(ratios)
        for r in ratios:
            assert abs(r / avg - 1.0) < 0.1, (
                "Sizes should be log-uniform"
            )

    def test_custom_range(self):
        cfg = SweepConfig(min_bytes=1024, max_bytes=1024 * 1024,
                          num_points=11)
        sizes = cfg.working_set_sizes()
        assert sizes[0] == 1024
        assert sizes[-1] == 1024 * 1024
        assert len(sizes) == 11

    def test_single_point_returns_min(self):
        cfg = SweepConfig(num_points=1, min_bytes=4096)
        assert cfg.working_set_sizes() == [4096]


# --------------------------------------------------------------------
# Plateau detection
# --------------------------------------------------------------------

class TestPlateauDetection:
    def test_canonical_curve_finds_four_plateaus(self):
        levels = detect_levels(_CANONICAL_I7)
        # Every level should be detected
        assert levels.levels[CacheLevel.L1] is not None
        assert levels.levels[CacheLevel.L2] is not None
        assert levels.levels[CacheLevel.L3] is not None
        assert levels.levels[CacheLevel.DRAM] is not None

    def test_plateaus_are_byte_ranges_in_order(self):
        levels = detect_levels(_CANONICAL_I7)
        l1_lo, l1_hi = levels.levels[CacheLevel.L1]
        l2_lo, l2_hi = levels.levels[CacheLevel.L2]
        l3_lo, l3_hi = levels.levels[CacheLevel.L3]
        dram_lo, _ = levels.levels[CacheLevel.DRAM]
        assert l1_hi <= l2_lo
        assert l2_hi <= l3_lo
        assert l3_hi <= dram_lo

    def test_flat_curve_yields_single_plateau(self):
        """A curve with no transitions should produce one plateau
        (assigned to L1) and leave L2/L3/DRAM as None."""
        flat = [
            _point(8 * 1024,    100.0),
            _point(64 * 1024,   100.0),
            _point(1024 * 1024, 100.0),
        ]
        levels = detect_levels(flat)
        assert levels.levels[CacheLevel.L1] is not None
        assert levels.levels[CacheLevel.L2] is None
        assert levels.levels[CacheLevel.L3] is None
        assert levels.levels[CacheLevel.DRAM] is None

    def test_only_two_plateaus_assigns_l1_l2(self):
        two_step = [
            _point(8 * 1024,    100.0),
            _point(32 * 1024,   100.0),
            _point(1024 * 1024, 50.0),    # only one transition
            _point(16 * 1024**2, 50.0),
        ]
        levels = detect_levels(two_step)
        assert levels.levels[CacheLevel.L1] is not None
        assert levels.levels[CacheLevel.L2] is not None
        assert levels.levels[CacheLevel.L3] is None
        assert levels.levels[CacheLevel.DRAM] is None

    def test_empty_input(self):
        levels = detect_levels([])
        assert levels.levels == {} or all(
            v is None for v in levels.levels.values()
        )

    def test_single_point_input(self):
        levels = detect_levels([_point(8192, 100.0)])
        assert levels.levels == {} or all(
            v is None for v in levels.levels.values()
        )

    def test_transition_threshold_resists_noise(self):
        """A 5-10% bandwidth wobble within a plateau must not register
        as a transition (the threshold is 1.2x)."""
        wobbly = [
            _point(8 * 1024,   100.0),
            _point(32 * 1024,   95.0),  # 5% drop
            _point(64 * 1024,  102.0),  # 7% rise
            _point(256 * 1024,  98.0),  # 4% drop
        ]
        levels = detect_levels(wobbly)
        # All four points should be in the L1 plateau
        l1_lo, l1_hi = levels.levels[CacheLevel.L1]
        assert l1_lo == 8 * 1024 and l1_hi == 256 * 1024
        assert levels.levels[CacheLevel.L2] is None


# --------------------------------------------------------------------
# Per-level energy estimation
# --------------------------------------------------------------------

class TestPerLevelEnergy:
    def test_canonical_curve_recovers_energy(self):
        ee = estimate_per_level_energy(_CANONICAL_I7)
        assert ee.energy_per_byte_pj[CacheLevel.L1] == 0.20
        assert ee.energy_per_byte_pj[CacheLevel.L2] == 0.50
        assert ee.energy_per_byte_pj[CacheLevel.L3] == 1.20
        assert ee.energy_per_byte_pj[CacheLevel.DRAM] == 15.0

    def test_canonical_curve_recovers_bandwidth(self):
        ee = estimate_per_level_energy(_CANONICAL_I7)
        assert ee.bandwidth_gbps[CacheLevel.L1] == 200.0
        assert ee.bandwidth_gbps[CacheLevel.L2] == 80.0
        assert ee.bandwidth_gbps[CacheLevel.L3] == 35.0
        assert ee.bandwidth_gbps[CacheLevel.DRAM] == 8.0

    def test_no_energy_capture_yields_none(self):
        """When every point has energy_per_byte_pj=None (RAPL
        unavailable), per-level energy is None but bandwidth is
        still populated."""
        no_energy = [
            _point(8 * 1024,   200.0, None),
            _point(64 * 1024,  200.0, None),
            _point(1024 * 1024, 80.0, None),
        ]
        ee = estimate_per_level_energy(no_energy)
        assert ee.energy_per_byte_pj[CacheLevel.L1] is None
        assert ee.energy_per_byte_pj[CacheLevel.L2] is None
        assert ee.bandwidth_gbps[CacheLevel.L1] == 200.0
        assert ee.bandwidth_gbps[CacheLevel.L2] == 80.0

    def test_has_calibrated_energy_predicate(self):
        ee = estimate_per_level_energy(_CANONICAL_I7)
        assert ee.has_calibrated_energy(CacheLevel.L1)
        assert ee.has_calibrated_energy(CacheLevel.DRAM)

        no_energy = [_point(8 * 1024, 100.0, None)]
        ee2 = estimate_per_level_energy(no_energy)
        assert not ee2.has_calibrated_energy(CacheLevel.L1)


# --------------------------------------------------------------------
# Mixed-energy edge cases
# --------------------------------------------------------------------

class TestMixedEnergy:
    def test_partial_energy_capture(self):
        """Some points have energy, some don't (counter wrap mid-sweep)."""
        mixed = [
            _point(8 * 1024,   200.0, 0.20),
            _point(32 * 1024,  200.0, None),  # missing
            _point(256 * 1024,  80.0, 0.50),
            _point(1024 * 1024, 80.0, None),  # missing
        ]
        ee = estimate_per_level_energy(mixed)
        # L1 plateau: only one point with energy -> use it
        assert ee.energy_per_byte_pj[CacheLevel.L1] == 0.20
        assert ee.energy_per_byte_pj[CacheLevel.L2] == 0.50
