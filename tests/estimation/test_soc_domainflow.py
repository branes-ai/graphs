"""Domain-flow ceilings: what a KPU fabric could do at best (graphs#269 6.4)."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest
from embodied_schemas import load_compute_products

from graphs.benchmarks.soc_kernels import kernel_suite
from graphs.estimation.soc import KernelClass
from graphs.estimation.soc.domainflow import (
    BYTES,
    NO_SCHEDULE,
    SHAPES,
    Fabric,
    Shape,
    against_requirement,
    ceilings,
    fabric_ceilings,
    fabrics_of,
)
from graphs.hardware.kpu_sku_generator import input_spec_from_compute_product
from graphs.hardware.soc.kpu_cores import KPU_CORES

REPO = Path(__file__).resolve().parents[2]
T128 = "kpu_t128_32x32_lp5x8_7nm_tsmc_hpc"


@pytest.fixture(scope="module")
def spec():
    return input_spec_from_compute_product(load_compute_products()[T128])


@pytest.fixture(scope="module")
def fp16(spec):
    return next(f for f in fabrics_of(spec) if f.precision == "fp16")


# ---------------------------------------------------------------------------
# The shapes are the benchmark's
# ---------------------------------------------------------------------------


def test_every_shape_matches_the_benchmark_suite():
    """The model and a measurement must describe the same work, so each
    shape's ops must equal the suite's for the same kernel."""
    suite = {(k.kernel_class, k.name): k for k in kernel_suite()}
    for shape in SHAPES:
        key = (shape.kernel_class.value, shape.name)
        assert key in suite, key
        assert shape.ops_per_call == pytest.approx(suite[key].ops_per_call), key


def test_a_gemm_shapes_macs_are_its_ops_over_two():
    gemm = next(s for s in SHAPES if s.name == "gemm")
    assert gemm.macs == pytest.approx(gemm.ops_per_call / 2)
    assert gemm.gemms == ((2048, 2048, 2048),)


def test_the_classes_with_no_schedule_are_named_not_guessed():
    covered = {s.kernel_class for s in SHAPES}
    assert covered.isdisjoint(NO_SCHEDULE)
    assert covered | set(NO_SCHEDULE) == set(KernelClass)
    assert all(reason for reason in NO_SCHEDULE.values())


# ---------------------------------------------------------------------------
# The bounds
# ---------------------------------------------------------------------------


def _fabric(**kwargs) -> Fabric:
    base = dict(sku="test", precision="fp16", rows=32, cols=32, tiles=1, fill_cycles=32,
                drain_cycles=32, clock_hz=1e9, ops_per_clock=2048.0,
                dram_bytes_per_s=1e15)  # unlimited DRAM unless a test says otherwise
    return Fabric(**{**base, **kwargs})


def test_the_wavefront_bound_is_the_passes_fill_and_drain():
    """One 64x64x64 GEMM on a 32x32 array: 4 passes of 64 + 64 cycles, so
    the array issues 4 x 128 x 1024 MAC slots for 64^3 MACs."""
    shape = Shape(KernelClass.DENSE_CONV_GEMM, "t", ((64, 64, 64),), 0, 2.0 * 64 ** 3, "test")
    got = ceilings(_fabric(), (shape,))[0]
    assert got.bounds["wavefront"] == pytest.approx(64 ** 3 / (4 * (64 + 64) * 1024))
    assert got.binding == "wavefront"


def test_a_perfectly_tiled_gemm_loses_only_fill_and_drain():
    shape = Shape(KernelClass.DENSE_CONV_GEMM, "t", ((32, 32, 4096),), 0, 2.0 * 32 * 32 * 4096,
                  "test")
    got = ceilings(_fabric(), (shape,))[0]
    assert got.bounds["wavefront"] == pytest.approx(4096 / (4096 + 64))


def test_the_dram_bound_is_compute_time_over_traffic_time():
    """A GEMV moves its weight matrix once; at 100 GB/s that dominates."""
    shape = Shape(KernelClass.WEIGHT_STREAM_DECODE, "t", ((1, 4096, 4096),),
                  4096 * 4096, 2.0 * 4096 * 4096, "test")
    fabric = _fabric(dram_bytes_per_s=100e9, tiles=128, ops_per_clock=128 * 2048.0)
    got = ceilings(fabric, (shape,))[0]
    t_dram = 4096 * 4096 * BYTES["fp16"] / 100e9
    t_compute = shape.ops_per_call / fabric.peak_ops_per_s
    assert got.bounds["dram_compulsory"] == pytest.approx(t_compute / t_dram)
    assert got.binding == "dram compulsory" and got.value < 0.01


def test_a_bound_never_exceeds_one():
    shape = Shape(KernelClass.DENSE_CONV_GEMM, "t", ((2048, 2048, 2048),), 3 * 2048 ** 2,
                  2.0 * 2048 ** 3, "test")
    for bandwidth in (1e9, 1e12, 1e18):
        got = ceilings(_fabric(dram_bytes_per_s=bandwidth), (shape,))[0]
        assert 0 < got.value <= 1.0
        assert all(v is None or v <= 1.0 for v in got.bounds.values())


def test_operand_delivery_binds_only_where_the_tile_states_it():
    shape = Shape(KernelClass.DENSE_CONV_GEMM, "t", ((32, 32, 4096),), 0, 2.0 * 32 * 32 * 4096,
                  "test")
    silent = ceilings(_fabric(), (shape,))[0]
    assert silent.bounds["operand_delivery"] is None
    assert silent.gaps and "no fabric interconnect" in silent.gaps[0]
    # 32 bits per clock is one FP32 operand, or two FP16.
    narrow = ceilings(_fabric(precision="fp32", operand_bits_per_clock=16), (shape,))[0]
    assert narrow.bounds["operand_delivery"] == pytest.approx(0.5)
    assert narrow.binding == "operand delivery"
    wide = ceilings(_fabric(operand_bits_per_clock=32), (shape,))[0]
    assert wide.bounds["operand_delivery"] == pytest.approx(1.0)


def test_more_tiles_never_lower_a_ceiling(spec):
    shape = next(s for s in SHAPES if s.name == "gemm")
    values = [ceilings(_fabric(tiles=t, ops_per_clock=t * 2048.0), (shape,))[0].value
              for t in (1, 8, 64, 128)]
    assert values == sorted(values) or math.isclose(values[0], values[-1], rel_tol=0.2)


# ---------------------------------------------------------------------------
# On the catalog's SKUs
# ---------------------------------------------------------------------------


def test_the_fabric_is_read_from_the_sku(spec, fp16):
    assert (fp16.rows, fp16.cols, fp16.tiles) == (32, 32, 128)
    assert (fp16.fill_cycles, fp16.drain_cycles) == (32, 32)
    assert fp16.clock_hz == pytest.approx(spec.clocks.boost_clock_mhz * 1e6)
    assert fp16.dram_bytes_per_s == pytest.approx(
        spec.kpu_architecture.memory.memory_bandwidth_gbps * 1e9)
    # Only the BF16-primary tiles have FP32, so that fabric is smaller.
    fp32 = next(f for f in fabrics_of(spec) if f.precision == "fp32")
    assert fp32.tiles < fp16.tiles


def test_dense_work_is_near_the_top_and_streaming_work_is_not(spec):
    fc = fabric_ceilings(spec)
    gemm = fc.best(KernelClass.DENSE_CONV_GEMM, "int8")
    decode = fc.best(KernelClass.WEIGHT_STREAM_DECODE, "int8")
    norm = fc.best(KernelClass.ELEMENTWISE_NORM, "int8")
    assert gemm.value > 0.9 and gemm.binding in ("wavefront", "dram compulsory")
    assert decode.value < 0.01 and decode.binding == "dram compulsory"
    assert norm.value < 0.01 and norm.binding == "dram compulsory"


def test_attention_pays_for_a_short_inner_dimension(spec):
    """The context GEMM's K is 64 against 64 cycles of fill and drain, so
    half the array's time is pipeline, not arithmetic."""
    fc = fabric_ceilings(spec)
    sdpa = fc.best(KernelClass.ATTENTION_PREFILL, "fp16")
    assert 0.6 < sdpa.value < 0.7 and sdpa.binding == "wavefront"


def test_every_ceiling_carries_its_source_and_confidence(spec):
    fc = fabric_ceilings(spec)
    assert fc.estimation_confidence.level.value == "theoretical"
    assert "shapes from graphs.benchmarks.soc_kernels" in fc.estimation_confidence.source
    for entry in fc.entries:
        assert "PEs x" in entry.source and "DRAM" in entry.source
        assert entry.value is not None and 0 < entry.value <= 1


def test_a_ceiling_answers_a_requirement_only_when_both_exist(spec):
    fc = fabric_ceilings(spec)
    gemm = fc.best(KernelClass.DENSE_CONV_GEMM, "int8")
    assert against_requirement(gemm, 0.5) is True
    assert against_requirement(gemm, 0.999) is False
    assert against_requirement(gemm, None) is None
    assert against_requirement(None, 0.5) is None
    assert fc.best(KernelClass.RAYCAST, "fp32") is None


# ---------------------------------------------------------------------------
# The artifact and the CLI
# ---------------------------------------------------------------------------


def test_the_written_ceilings_match_the_model():
    result = subprocess.run([sys.executable, "tools/generate_kpu_ceilings.py", "--check"],
                            capture_output=True, text=True, cwd=REPO, timeout=300)
    assert result.returncode == 0, result.stdout + result.stderr
    assert len(list((REPO / "soc_designs" / "ceilings").glob("*.yaml"))) == len(KPU_CORES)


def _cli(*args, expect=0):
    result = subprocess.run([sys.executable, "cli/analyze_kpu_ceiling.py", *args],
                            capture_output=True, text=True, cwd=REPO, timeout=300)
    assert result.returncode == expect, result.stderr
    return result


def test_cli_decides_far_flight_against_the_ceiling():
    """vlm streams weights, which is DRAM-bound at well under 1% of dense
    peak, so far flight is over its period even at the ceiling."""
    out = _cli("--design", "kpu_uniform_t64", "--regime", "far flight",
               "--node", "tsmc_n7", "-v").stdout
    assert "DECIDED: over 1" in out
    assert "vlm" in out.split("stages that alone exceed their period at the ceiling:")[1]


def test_cli_leaves_air_superiority_open():
    out = _cli("--design", "kpu_uniform_t128", "--regime", "air superiority",
               "--node", "tsmc_n7").stdout
    assert "-- open" in out and "LOWER BOUND" in out


def test_cli_ceilings_only_needs_no_workload(tmp_path):
    path = tmp_path / "c.json"
    _cli("--sku", T128, "--ceilings-only", "--output", str(path))
    payload = json.loads(path.read_text())
    assert payload[0]["sku"] == T128
    assert {c["kernel_class"] for c in payload[0]["ceilings"]} == {
        s.kernel_class.value for s in SHAPES}


def test_cli_rejects_a_design_without_a_kpu_core():
    assert "no generated KPU core" in _cli("--design", "orin_class_reference",
                                           "--regime", "far flight", expect=2).stderr
