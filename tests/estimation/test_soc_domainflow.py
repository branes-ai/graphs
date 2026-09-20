"""Domain-flow ceilings: what a KPU fabric could do at best (graphs#269 6.4)."""

from __future__ import annotations

import json
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
                ops_per_tile_per_clock=2048.0,  # one FMA per PE per clock
                dram_bytes_per_s=1e15)  # unlimited DRAM unless a test says otherwise
    return Fabric(**{**base, **kwargs})


def test_the_wavefront_bound_is_the_passes_fill_and_drain():
    """One 64x64x64 GEMM on a 32x32 array issuing every clock: 4 passes of
    64 + 64 cycles, against the 64^3 / 1024 cycles a dense peak would take."""
    shape = Shape(KernelClass.DENSE_CONV_GEMM, "t", ((64, 64, 64),), 0, 2.0 * 64 ** 3, "test")
    got = ceilings(_fabric(), (shape,))[0]
    assert got.bounds["wavefront"] == pytest.approx(64 ** 3 / 1024 / (4 * (64 + 64)))
    assert got.binding == "wavefront"


def test_the_schedule_pays_the_skus_issue_interval():
    """A tile that states half its PE count in MACs per clock issues every
    other clock, so K steps take 2K cycles and fill and drain weigh less."""
    shape = Shape(KernelClass.DENSE_CONV_GEMM, "t", ((32, 32, 1024),), 0,
                  2.0 * 32 * 32 * 1024, "test")
    every_clock = _fabric()
    every_other = _fabric(ops_per_clock=1024.0, ops_per_tile_per_clock=1024.0)
    assert every_clock.issue_interval == 1 and every_other.issue_interval == 2
    assert ceilings(every_clock, (shape,))[0].bounds["wavefront"] == pytest.approx(
        1024 / (1024 + 64))
    assert ceilings(every_other, (shape,))[0].bounds["wavefront"] == pytest.approx(
        1024 / (2 * 1024 + 64) * 2)


def test_a_perfectly_tiled_gemm_loses_only_fill_and_drain():
    shape = Shape(KernelClass.DENSE_CONV_GEMM, "t", ((32, 32, 4096),), 0, 2.0 * 32 * 32 * 4096,
                  "test")
    got = ceilings(_fabric(), (shape,))[0]
    assert got.bounds["wavefront"] == pytest.approx(4096 / (4096 + 64))


def test_an_elementwise_pass_uses_the_adder_only():
    """No multiply, so a PE contributes one op per clock where a dense peak
    counts two: the wavefront bound is 0.5 whatever the size."""
    shape = Shape(KernelClass.ELEMENTWISE_NORM, "t", (), 0, 5.0 * 4096 * 4096, "test",
                  elementwise=(4096 * 4096, 5.0))
    got = ceilings(_fabric(tiles=128, ops_per_clock=128 * 2048.0), (shape,))[0]
    assert got.bounds["wavefront"] == pytest.approx(0.5)


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


def test_operand_delivery_needs_both_a_row_and_a_column_path():
    shape = Shape(KernelClass.DENSE_CONV_GEMM, "t", ((32, 32, 4096),), 0, 2.0 * 32 * 32 * 4096,
                  "test")
    silent = ceilings(_fabric(), (shape,))[0]
    assert silent.bounds["operand_delivery"] is None
    assert any("operand path" in g for g in silent.gaps)
    # One direction alone does not bound it: the other is still unstated.
    half = ceilings(_fabric(operand_row_bits=32), (shape,))[0]
    assert half.bounds["operand_delivery"] is None
    # 16 bits per clock is half an FP32 operand: every cycle stretches 2x.
    narrow = ceilings(_fabric(precision="fp32", operand_row_bits=16, operand_col_bits=16),
                      (shape,))[0]
    wide = ceilings(_fabric(precision="fp32", operand_row_bits=32, operand_col_bits=32),
                    (shape,))[0]
    assert narrow.bounds["operand_delivery"] == pytest.approx(
        wide.bounds["operand_delivery"] / 2)
    assert narrow.binding == "operand delivery"
    # The narrower of the two directions is the one that binds.
    lopsided = ceilings(_fabric(precision="fp32", operand_row_bits=32, operand_col_bits=16),
                        (shape,))[0]
    assert lopsided.bounds["operand_delivery"] == pytest.approx(
        narrow.bounds["operand_delivery"])


def test_an_unstated_dram_bandwidth_is_a_gap_not_a_free_pass():
    shape = Shape(KernelClass.DENSE_CONV_GEMM, "t", ((32, 32, 4096),), 3 * 32 * 4096,
                  2.0 * 32 * 32 * 4096, "test")
    got = ceilings(_fabric(dram_bytes_per_s=0.0), (shape,))[0]
    assert got.bounds["dram_compulsory"] is None
    assert any("no DRAM bandwidth" in g for g in got.gaps)
    assert got.binding == "wavefront"


def test_more_tiles_never_lower_a_ceiling(spec):
    shape = next(s for s in SHAPES if s.name == "gemm")
    values = [ceilings(_fabric(tiles=t, ops_per_clock=t * 2048.0), (shape,))[0].value
              for t in (1, 8, 64, 128)]
    assert values == sorted(values), values


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
    assert 0.6 < sdpa.value < 0.85
    assert sdpa.bounds["wavefront"] < 0.9  # fill and drain against a 64-deep K


def test_every_ceiling_carries_its_source_and_confidence(spec):
    fc = fabric_ceilings(spec)
    assert fc.estimation_confidence.level.value == "theoretical"
    assert "shapes from graphs.benchmarks.soc_kernels" in fc.estimation_confidence.source
    for entry in fc.entries:
        assert "PEs x" in entry.source and "DRAM" in entry.source
        assert entry.value is not None and 0 < entry.value <= 1
        assert entry.estimation_confidence.level.value == "theoretical"
        assert entry.to_dict()["confidence"] == "theoretical"


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


# ---------------------------------------------------------------------------
# What the #324 review changed
# ---------------------------------------------------------------------------


def test_a_stage_is_charged_its_own_format_per_class(tmp_path):
    """det runs Class A in INT8 and Class B in FP16, and the two have
    different ceilings. The time at the ceiling is the sum of each class
    over its own, not the whole stage over the lower one."""
    from embodied_schemas import load_process_nodes

    from graphs.core.pipeline_workload import load_autonomy_workload
    from graphs.estimation.soc import load_kernel_classes, required_efficiency
    from graphs.hardware.soc import compose_soc, load_designs, load_ip_library

    workload = load_autonomy_workload()
    soc = compose_soc(load_designs()["kpu_uniform_t128"], load_ip_library(),
                      load_process_nodes(), "tsmc_n7")
    kernels = load_kernel_classes()
    air = next(p for p in workload.regimes() if p.regime == "air superiority")
    result = required_efficiency(workload, air, soc, kernels=kernels)
    fc = _ceilings()
    det = next(s for s in result.stages if s.stage == "det")
    assert set(det.class_seconds) == {"A", "B"} and det.formats["A"] != det.formats["B"]
    expected = sum(seconds * det.rate_hz / fc.best(kernels.of("det"), det.formats[cls]).value
                   for cls, seconds in det.class_seconds.items())
    lowest = min(fc.best(kernels.of("det"), f).value for f in det.formats.values())
    assert expected < det.occupancy_at_peak / lowest  # the old, overstated figure

    path = tmp_path / "c.json"
    _cli("--design", "kpu_uniform_t128", "--regime", "air superiority", "--node", "tsmc_n7",
         "--output", str(path))
    row = next(r for r in json.loads(path.read_text())[0]["stages"] if r["stage"] == "det")
    assert row["at_ceiling"] == pytest.approx(expected)
    assert isinstance(row["at_ceiling"], float)  # numeric, not a rounded display string


def _ceilings():
    return fabric_ceilings(input_spec_from_compute_product(load_compute_products()[T128]))


def test_cli_matches_a_profile_whatever_its_case():
    mixed = _cli("--design", "kpu_uniform_t128", "--profile", "AIR superiority",
                 "--node", "tsmc_n7").stdout
    assert "drone_interceptor_terminal_engagement" in mixed


def test_cli_writes_markdown_when_asked(tmp_path):
    workload_md = tmp_path / "w.md"
    _cli("--design", "kpu_uniform_t128", "--regime", "air superiority", "--node", "tsmc_n7",
         "-v", "--output", str(workload_md))
    text = workload_md.read_text()
    assert text.startswith("## kpu_uniform_t128") and "| stage |" in text
    ceilings_md = tmp_path / "c.md"
    _cli("--sku", T128, "--ceilings-only", "--output", str(ceilings_md))
    assert ceilings_md.read_text().startswith(f"## {T128}")
    assert "| kernel_class |" in ceilings_md.read_text()


def test_cli_reads_an_explicit_mapping_file(tmp_path):
    """A mapping file names the engine; the stages it leaves out fall back
    to the capability rule, so a one-stage file still analyses the rest."""
    path = tmp_path / "m.yaml"
    path.write_text("design: kpu_uniform_t128\nworkload: branes_7tier_v1\n"
                    "stages:\n  det:\n    engine: kpu\n"
                    "    reason: the test names one stage; the rest fall back to capability\n")
    out = _cli("--design", "kpu_uniform_t128", "--regime", "air superiority", "--node",
               "tsmc_n7", "--mapping", str(path), "-v").stdout
    assert "det" in out and "dense_conv_gemm" in out
