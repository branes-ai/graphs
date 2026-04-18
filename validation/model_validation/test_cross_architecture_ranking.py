"""
Cross-Architecture Ranking Tests

Validates that the modeling infrastructure produces physically
plausible rankings when comparing architectures on the same workload.
These tests catch gross modeling errors like a CPU being modeled as
more energy-efficient than a TPU for large matmuls, or an edge device
producing lower latency than a datacenter GPU.

Three categories:
1. Energy ranking: architecture-class energy ordering for matmuls
2. Latency ranking: datacenter > edge throughput
3. Known-physics invariants: precision, bandwidth, compute scaling

All tests exercise the model code only -- no hardware required.
"""

from __future__ import annotations

import pytest

from graphs.hardware.resource_model import Precision


def _get_mapper(name: str):
    """Get a mapper by name; skip test if not found."""
    from graphs.hardware.mappers import get_mapper_by_name
    try:
        m = get_mapper_by_name(name)
    except (KeyError, ValueError, TypeError, AttributeError):
        m = None
    if m is None:
        pytest.skip(f"Mapper {name} not found")
    return m


def _try_get_mapper(name: str):
    """Get a mapper by name; return None if not found (for loop tests)."""
    from graphs.hardware.mappers import get_mapper_by_name
    try:
        return get_mapper_by_name(name)
    except (KeyError, ValueError, TypeError, AttributeError):
        return None


def _compute_energy(mapper, ops: int, bytes_t: int, precision: Precision) -> float:
    c, m = mapper._calculate_energy(ops, bytes_t, precision)
    return c + m


def _compute_latency(mapper, ops: int, bytes_t: int, precision: Precision) -> float:
    """Compute roofline latency from peak specs.

    Uses peak_ops directly rather than going through
    _calculate_latency, which can hit a 0.01x penalty path when
    thermal operating points exist but have empty performance_specs
    (a known data gap for datacenter GPUs). This means these tests
    do NOT exercise the production _calculate_latency path for
    datacenter GPUs -- that path needs its thermal specs populated
    first (tracked as a separate model data issue).
    """
    peak_ops = mapper.resource_model.get_peak_ops(precision)
    peak_bw = mapper.resource_model.peak_bandwidth
    ct = ops / peak_ops if peak_ops > 0 else float('inf')
    mt = bytes_t / peak_bw if peak_bw > 0 else float('inf')
    return max(ct, mt)


# Standard workload parameters
LARGE_MATMUL_OPS = 2 * 4096 * 4096 * 4096       # ~137G FLOPs
LARGE_MATMUL_BYTES = 3 * 4096 * 4096 * 4         # ~192 MB (in+out+weight)
MEDIUM_MATMUL_OPS = 2 * 1024 * 1024 * 1024       # ~2.1G FLOPs
MEDIUM_MATMUL_BYTES = 3 * 1024 * 1024 * 4         # ~12 MB


class TestEnergyRanking:
    """Architecture-class energy ordering for compute-bound workloads."""

    def test_gpu_lower_energy_per_op_than_cpu(self):
        """H100 should have lower energy_per_flop than Xeon (process + circuit advantage)."""
        h100 = _get_mapper("H100-SXM5-80GB")
        xeon = _get_mapper("Intel-Xeon-Platinum-8490H")
        assert (h100.resource_model.energy_per_flop_fp32
                < xeon.resource_model.energy_per_flop_fp32)

    def test_tpu_lower_energy_per_byte_than_cpu(self):
        """TPU v4 HBM should have lower energy_per_byte than CPU DDR5."""
        tpu = _get_mapper("Google-TPU-v4")
        xeon = _get_mapper("Intel-Xeon-Platinum-8490H")
        assert (tpu.resource_model.energy_per_byte
                < xeon.resource_model.energy_per_byte)

    def test_int8_always_cheaper_than_fp32_across_architectures(self):
        """INT8 compute energy < FP32 for all architecture classes."""
        representatives = [
            "Intel-Xeon-Platinum-8490H",
            "H100-SXM5-80GB",
            "Google-TPU-v4",
            "Stillwater-KPU-T256",
            "Qualcomm-QRB5165",
        ]
        for name in representatives:
            mapper = _try_get_mapper(name)
            if mapper is None:
                continue
            e_fp32 = _compute_energy(mapper, LARGE_MATMUL_OPS, LARGE_MATMUL_BYTES, Precision.FP32)
            e_int8 = _compute_energy(mapper, LARGE_MATMUL_OPS, LARGE_MATMUL_BYTES, Precision.INT8)
            assert e_int8 < e_fp32, f"{name}: INT8 total energy >= FP32"

    def test_large_matmul_energy_plausible_range(self):
        """Energy for a large matmul should be in [0.001 mJ, 10000 mJ] per validation rules."""
        representatives = [
            "Intel-i7-12700K",
            "H100-SXM5-80GB",
            "Google-TPU-v4",
            "Stillwater-KPU-T64",
        ]
        for name in representatives:
            mapper = _try_get_mapper(name)
            if mapper is None:
                continue
            e = _compute_energy(mapper, LARGE_MATMUL_OPS, LARGE_MATMUL_BYTES, Precision.FP32)
            e_mj = e * 1000.0
            assert 0.001 < e_mj < 10000, (
                f"{name}: energy {e_mj:.3f} mJ outside plausible range"
            )


class TestLatencyRanking:
    """Datacenter parts should be faster than edge for same workload."""

    def test_h100_faster_than_orin_agx_gpu(self):
        h100 = _get_mapper("H100-SXM5-80GB")
        orin = _get_mapper("Jetson-Orin-AGX-64GB")
        lat_h100 = _compute_latency(h100, LARGE_MATMUL_OPS, LARGE_MATMUL_BYTES, Precision.FP32)
        lat_orin = _compute_latency(orin, LARGE_MATMUL_OPS, LARGE_MATMUL_BYTES, Precision.FP32)
        assert lat_h100 < lat_orin, "H100 should be faster than Orin AGX"

    def test_tpu_v4_faster_than_coral_edge(self):
        tpu_v4 = _get_mapper("Google-TPU-v4")
        coral = _get_mapper("Google-Coral-Edge-TPU")
        lat_v4 = _compute_latency(tpu_v4, MEDIUM_MATMUL_OPS, MEDIUM_MATMUL_BYTES, Precision.INT8)
        lat_coral = _compute_latency(coral, MEDIUM_MATMUL_OPS, MEDIUM_MATMUL_BYTES, Precision.INT8)
        assert lat_v4 < lat_coral, "TPU v4 should be faster than Coral Edge"

    def test_kpu_t768_faster_than_t64(self):
        t768 = _get_mapper("Stillwater-KPU-T768")
        t64 = _get_mapper("Stillwater-KPU-T64")
        lat_768 = _compute_latency(t768, LARGE_MATMUL_OPS, LARGE_MATMUL_BYTES, Precision.INT8)
        lat_64 = _compute_latency(t64, LARGE_MATMUL_OPS, LARGE_MATMUL_BYTES, Precision.INT8)
        assert lat_768 < lat_64, "KPU T768 should be faster than T64"

    def test_large_matmul_latency_plausible_range(self):
        """Latency for a large matmul should be in [0.001 ms, 10000 ms] per validation rules."""
        representatives = [
            "Intel-i7-12700K",
            "H100-SXM5-80GB",
            "Google-TPU-v4",
            "Stillwater-KPU-T256",
        ]
        for name in representatives:
            mapper = _try_get_mapper(name)
            if mapper is None:
                continue
            lat = _compute_latency(mapper, LARGE_MATMUL_OPS, LARGE_MATMUL_BYTES, Precision.FP32)
            lat_ms = lat * 1000.0
            assert 0.001 < lat_ms < 10000, (
                f"{name}: latency {lat_ms:.3f} ms outside plausible range"
            )


class TestPhysicsInvariants:
    """Known physical relationships that must hold in every model."""

    def test_higher_bandwidth_means_lower_memory_latency(self):
        """HBM (H100) vs DDR (CPU): higher bandwidth = lower memory time."""
        h100 = _get_mapper("H100-SXM5-80GB")
        cpu = _get_mapper("Intel-i7-12700K")

        assert h100.resource_model.peak_bandwidth > cpu.resource_model.peak_bandwidth

        bytes_t = 100_000_000
        h100_mt = bytes_t / h100.resource_model.peak_bandwidth
        cpu_mt = bytes_t / cpu.resource_model.peak_bandwidth
        assert h100_mt < cpu_mt

    def test_more_compute_units_means_higher_peak(self):
        """More SMs/tiles/cores = higher peak FLOPS at same precision."""
        pairs = [
            ("H100-SXM5-80GB", "Jetson-Orin-Nano-8GB"),       # 132 vs 8 SMs
            ("AMD-EPYC-9754", "Intel-i7-12700K"),              # 128 vs 10 cores
            ("Stillwater-KPU-T768", "Stillwater-KPU-T64"),     # 768 vs 64 tiles
        ]
        for big_name, small_name in pairs:
            big = _try_get_mapper(big_name)
            small = _try_get_mapper(small_name)
            if big is None or small is None:
                continue
            assert big.resource_model.compute_units > small.resource_model.compute_units, (
                f"{big_name} should have more compute units than {small_name}"
            )
            big_peak = big.resource_model.get_peak_ops(Precision.FP32)
            small_peak = small.resource_model.get_peak_ops(Precision.FP32)
            assert big_peak > small_peak, (
                f"{big_name} ({big_peak/1e12:.1f}T) should have higher peak than "
                f"{small_name} ({small_peak/1e12:.1f}T)"
            )

    def test_energy_scaling_monotonic_with_precision_width(self):
        """Wider precision always costs more energy: FP64 > FP32 > FP16 > INT8 > INT4."""
        representatives = [
            "Intel-i7-12700K",
            "H100-SXM5-80GB",
            "Stillwater-KPU-T256",
        ]
        ordered_precisions = [
            (Precision.FP64, 8),
            (Precision.FP32, 4),
            (Precision.FP16, 2),
            (Precision.INT8, 1),
            (Precision.INT4, 0.5),
        ]
        for name in representatives:
            mapper = _try_get_mapper(name)
            if mapper is None:
                continue
            scaling = mapper.resource_model.energy_scaling
            for i in range(len(ordered_precisions) - 1):
                wider_prec, _ = ordered_precisions[i]
                narrower_prec, _ = ordered_precisions[i + 1]
                wider_scale = scaling.get(wider_prec)
                narrower_scale = scaling.get(narrower_prec)
                if wider_scale is not None and narrower_scale is not None:
                    assert wider_scale >= narrower_scale, (
                        f"{name}: {wider_prec.value} scale ({wider_scale}) "
                        f"< {narrower_prec.value} scale ({narrower_scale})"
                    )

    def test_no_mapper_has_zero_energy_coefficients(self):
        """Every mapper must have non-zero energy_per_flop and energy_per_byte."""
        from graphs.hardware.mappers import list_all_mappers
        for name in sorted(list_all_mappers()):
            mapper = _try_get_mapper(name)
            if mapper is None:
                continue
            assert mapper.resource_model.energy_per_flop_fp32 > 0, (
                f"{name}: zero energy_per_flop_fp32"
            )
            assert mapper.resource_model.energy_per_byte > 0, (
                f"{name}: zero energy_per_byte"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
