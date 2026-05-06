"""KPU consistency-only invariant checks (V4-5).

The KPU is a research/in-development spatial-dataflow accelerator with
no commercially-available silicon to measure against. Per the v4 plan
(Principle 2 + the "Phasing" recommendation), KPU validation is
**consistency-only**: the analyzer's predictions must satisfy
internal-math and physical-plausibility invariants, but we do NOT
assert any per-shape latency or energy band against ground truth.

Invariant categories:

1. Roofline self-consistency
   - actual_latency = max(compute_time, memory_time) + overhead
   - compute_time and memory_time are non-negative
2. Monotonicity
   - latency non-decreasing in working-set size at fixed dim ratios
   - latency non-increasing across the KPU family at fixed shape
   - memory_time scales monotonically with bytes_transferred
3. Physical bounds
   - achieved compute throughput <= theoretical peak FLOPS
   - achieved memory bandwidth <= theoretical peak DRAM bandwidth
4. Power below TDP (currently *known* to be violated; surfaced as a
   diagnostic-only check pending a calibrated KPU energy model).

Each function returns a list of human-readable violation messages
(empty list = pass). The thin pytest wrapper aggregates them into
assertion failures with a clear pointer at the violating shape.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Tuple

from graphs.core.structures import OperationType, SubgraphDescriptor
from graphs.estimation.energy import EnergyAnalyzer
from graphs.estimation.roofline import RooflineAnalyzer
from graphs.hardware.resource_model import HardwareResourceModel, Precision


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _MatmulShape:
    M: int
    K: int
    N: int

    @property
    def flops(self) -> int:
        return 2 * self.M * self.K * self.N

    def working_set_bytes(self, bytes_per_element: int) -> int:
        return (self.M * self.K + self.K * self.N + self.M * self.N) * bytes_per_element

    def to_subgraph(self, bytes_per_element: int) -> SubgraphDescriptor:
        return SubgraphDescriptor(
            subgraph_id=0,
            node_ids=["mm"], node_names=["mm"],
            operation_types=[OperationType.MATMUL],
            fusion_pattern="matmul",
            total_flops=self.flops,
            total_macs=self.M * self.K * self.N,
            total_input_bytes=(self.M * self.K + self.K * self.N) * bytes_per_element,
            total_output_bytes=self.M * self.N * bytes_per_element,
            total_weight_bytes=0,
        )


def _bpe(precision: Precision) -> int:
    """Bytes per element for the precision argument."""
    if precision in (Precision.FP64,):
        return 8
    if precision in (Precision.FP32, Precision.TF32):
        return 4
    if precision in (Precision.FP16, Precision.BF16):
        return 2
    if precision in (Precision.INT8, Precision.FP8, Precision.FP8_E4M3, Precision.FP8_E5M2):
        return 1
    if precision in (Precision.INT4, Precision.FP4):
        return 1   # round up; 0.5 doesn't fit in int. Tests use larger precisions.
    return 4


# Default shape grid for monotonicity / scaling sweeps. Square shapes
# at fixed dim ratios so working_set ~= 3 * N^2 * bpe.
DEFAULT_SHAPE_GRID: Tuple[_MatmulShape, ...] = (
    _MatmulShape(64, 64, 64),
    _MatmulShape(128, 128, 128),
    _MatmulShape(256, 256, 256),
    _MatmulShape(512, 512, 512),
    _MatmulShape(1024, 1024, 1024),
    _MatmulShape(2048, 2048, 2048),
)


# ---------------------------------------------------------------------------
# 1. Roofline self-consistency
# ---------------------------------------------------------------------------


def check_roofline_self_consistency(
    hw: HardwareResourceModel,
    precision: Precision,
    shapes: Tuple[_MatmulShape, ...] = DEFAULT_SHAPE_GRID,
) -> List[str]:
    """For every shape, ``actual_latency`` must equal ``max(compute_time,
    memory_time) + overhead`` exactly. Also: both component times must
    be non-negative.

    This is the bedrock invariant -- if it ever fails, the
    ``_analyze_subgraph`` math has drifted from its own definition."""
    failures: List[str] = []
    analyzer = RooflineAnalyzer(hw, precision=precision)
    bpe = _bpe(precision)
    for s in shapes:
        sg = s.to_subgraph(bpe)
        lat = analyzer._analyze_subgraph(sg)
        if lat.compute_time < 0:
            failures.append(
                f"shape=({s.M},{s.K},{s.N}): compute_time={lat.compute_time:.3e} < 0")
        if lat.memory_time < 0:
            failures.append(
                f"shape=({s.M},{s.K},{s.N}): memory_time={lat.memory_time:.3e} < 0")
        expected = max(lat.compute_time, lat.memory_time) + lat.overhead
        # Tiny floating-point slack is fine; anything > 1ns is suspicious.
        if abs(lat.actual_latency - expected) > 1e-9:
            failures.append(
                f"shape=({s.M},{s.K},{s.N}): actual_latency={lat.actual_latency:.3e} "
                f"!= max(compute={lat.compute_time:.3e}, memory={lat.memory_time:.3e}) "
                f"+ overhead={lat.overhead:.3e} (expected {expected:.3e})")
    return failures


# ---------------------------------------------------------------------------
# 2. Monotonicity
# ---------------------------------------------------------------------------


def check_latency_non_decreasing_in_size(
    hw: HardwareResourceModel,
    precision: Precision,
    shapes: Tuple[_MatmulShape, ...] = DEFAULT_SHAPE_GRID,
) -> List[str]:
    """Sweeping square matmul size N in ascending order, predicted
    actual_latency must be non-decreasing. A monotonically larger
    workload cannot run faster on the same hardware."""
    failures: List[str] = []
    analyzer = RooflineAnalyzer(hw, precision=precision)
    bpe = _bpe(precision)
    sorted_shapes = sorted(shapes, key=lambda s: s.flops)
    prev_lat: float = 0.0
    prev_label = "<none>"
    for s in sorted_shapes:
        lat = analyzer._analyze_subgraph(s.to_subgraph(bpe))
        if lat.actual_latency < prev_lat - 1e-9:
            failures.append(
                f"non-monotonic: shape=({s.M},{s.K},{s.N}) "
                f"latency={lat.actual_latency:.3e}s < prior {prev_label} "
                f"latency={prev_lat:.3e}s (work grew but latency dropped)")
        prev_lat = lat.actual_latency
        prev_label = f"({s.M},{s.K},{s.N})"
    return failures


def check_memory_time_scales_with_bytes(
    hw: HardwareResourceModel,
    precision: Precision,
    shapes: Tuple[_MatmulShape, ...] = DEFAULT_SHAPE_GRID,
) -> List[str]:
    """Memory time must be non-decreasing in working_set_bytes at fixed
    bandwidth efficiency. (Linear scaling is too strict because
    bw_efficiency_scale itself depends on working-set size; we only
    require monotonicity.)"""
    failures: List[str] = []
    analyzer = RooflineAnalyzer(hw, precision=precision)
    bpe = _bpe(precision)
    rows = []
    for s in shapes:
        lat = analyzer._analyze_subgraph(s.to_subgraph(bpe))
        rows.append((s.working_set_bytes(bpe), lat.memory_time, s))
    rows.sort(key=lambda r: r[0])
    for (b1, m1, s1), (b2, m2, s2) in zip(rows, rows[1:]):
        if m2 < m1 - 1e-9:
            failures.append(
                f"memory_time non-monotonic: ws={b1} -> mem_t={m1:.3e}s vs "
                f"ws={b2} -> mem_t={m2:.3e}s (later shape has more bytes "
                f"but lower memory_time)")
    return failures


def check_family_latency_non_increasing(
    family: List[Tuple[str, HardwareResourceModel]],
    precision: Precision,
    shape: _MatmulShape,
) -> List[str]:
    """Across a family of accelerators ordered by *increasing* compute
    capability (e.g., T64 -> T128 -> T256 -> T768), predicted latency
    for the same shape must be *non-increasing*. More tiles + more BW
    cannot make a fixed workload slower."""
    failures: List[str] = []
    bpe = _bpe(precision)
    sg = shape.to_subgraph(bpe)
    measured: List[Tuple[str, float]] = []
    for name, hw in family:
        analyzer = RooflineAnalyzer(hw, precision=precision)
        lat = analyzer._analyze_subgraph(sg)
        measured.append((name, lat.actual_latency))
    for (n1, l1), (n2, l2) in zip(measured, measured[1:]):
        if l2 > l1 + 1e-9:
            failures.append(
                f"family-monotonicity: {n1}->{l1:.3e}s, {n2}->{l2:.3e}s "
                f"(more capable {n2} predicts SLOWER than {n1} for "
                f"shape ({shape.M},{shape.K},{shape.N}))")
    return failures


# ---------------------------------------------------------------------------
# 3. Physical bounds (achieved <= peak)
# ---------------------------------------------------------------------------


# 5% slack absorbs floating-point and overhead-amortization noise; any
# real violation will exceed this comfortably.
_PEAK_SLACK = 1.05


def check_achieved_compute_below_peak(
    hw: HardwareResourceModel,
    precision: Precision,
    shapes: Tuple[_MatmulShape, ...] = DEFAULT_SHAPE_GRID,
) -> List[str]:
    """Achieved FLOPS = flops / compute_time must not exceed the
    theoretical peak * PEAK_SLACK."""
    failures: List[str] = []
    analyzer = RooflineAnalyzer(hw, precision=precision)
    bpe = _bpe(precision)
    peak = hw.get_peak_ops(precision)
    for s in shapes:
        lat = analyzer._analyze_subgraph(s.to_subgraph(bpe))
        if lat.compute_time <= 0:
            continue
        achieved = s.flops / lat.compute_time
        if achieved > peak * _PEAK_SLACK:
            failures.append(
                f"shape=({s.M},{s.K},{s.N}): achieved compute "
                f"{achieved/1e9:.1f} GFLOPS exceeds peak "
                f"{peak/1e9:.1f} GFLOPS (slack {_PEAK_SLACK:.0%})")
    return failures


def check_achieved_bw_below_peak(
    hw: HardwareResourceModel,
    precision: Precision,
    shapes: Tuple[_MatmulShape, ...] = DEFAULT_SHAPE_GRID,
) -> List[str]:
    """Achieved BW = bytes / memory_time must not exceed peak
    bandwidth * PEAK_SLACK."""
    failures: List[str] = []
    analyzer = RooflineAnalyzer(hw, precision=precision)
    bpe = _bpe(precision)
    peak = hw.peak_bandwidth
    for s in shapes:
        lat = analyzer._analyze_subgraph(s.to_subgraph(bpe))
        if lat.memory_time <= 0:
            continue
        bytes_total = s.working_set_bytes(bpe)
        achieved = bytes_total / lat.memory_time
        if achieved > peak * _PEAK_SLACK:
            failures.append(
                f"shape=({s.M},{s.K},{s.N}): achieved BW "
                f"{achieved/1e9:.1f} GB/s exceeds peak "
                f"{peak/1e9:.1f} GB/s (slack {_PEAK_SLACK:.0%})")
    return failures


# ---------------------------------------------------------------------------
# 4. Power-below-TDP (diagnostic; currently violated, see comment)
# ---------------------------------------------------------------------------


def check_avg_power_below_tdp(
    hw: HardwareResourceModel,
    precision: Precision,
    shapes: Tuple[_MatmulShape, ...] = DEFAULT_SHAPE_GRID,
    tdp_slack: float = 1.10,
) -> List[str]:
    """Predicted total energy / latency must not exceed TDP * slack.

    NOTE: this invariant is currently *known to fail* on every KPU
    mapper because the energy model uses the GPU IDLE_POWER_FRACTION
    (0.3) plus dynamic energy_per_flop coefficients that were not
    calibrated against KPU silicon. Tracked in #81; the pytest wrapper
    marks this test as xfail. Once #81 lands the xfail can be removed
    and this becomes a hard test (same pattern as the #71 CPU fix).

    Returning the violation list (rather than just a bool) lets the
    diagnostic surface which shapes are most over-prediction-prone."""
    failures: List[str] = []
    roof = RooflineAnalyzer(hw, precision=precision)
    energy = EnergyAnalyzer(hw, precision=precision)
    bpe = _bpe(precision)
    tdp = energy.tdp_watts
    for s in shapes:
        sg = s.to_subgraph(bpe)
        lat = roof._analyze_subgraph(sg)
        if lat.actual_latency <= 0:
            continue
        report = energy.analyze(subgraphs=[sg], latencies=[lat.actual_latency])
        avg_power = report.total_energy_j / lat.actual_latency
        if avg_power > tdp * tdp_slack:
            failures.append(
                f"shape=({s.M},{s.K},{s.N}): avg power {avg_power:.1f}W "
                f"exceeds TDP {tdp:.1f}W (slack {tdp_slack:.0%}); "
                f"energy={report.total_energy_j*1e3:.2f}mJ over "
                f"latency={lat.actual_latency*1e6:.1f}us")
    return failures


# ---------------------------------------------------------------------------
# Convenience: run the full battery on one mapper
# ---------------------------------------------------------------------------


@dataclass
class InvariantReport:
    hardware_name: str
    precision: Precision
    failures_per_check: dict[str, List[str]]

    @property
    def total_failures(self) -> int:
        return sum(len(v) for v in self.failures_per_check.values())

    def hard_failures(self) -> dict[str, List[str]]:
        """Return only the failures for *hard* invariants (excluding the
        known-violated power-below-TDP diagnostic)."""
        return {k: v for k, v in self.failures_per_check.items()
                if k != "avg_power_below_tdp" and v}


def run_kpu_invariants(
    hw: HardwareResourceModel,
    precision: Precision,
    hardware_name: str = "kpu",
    shapes: Tuple[_MatmulShape, ...] = DEFAULT_SHAPE_GRID,
) -> InvariantReport:
    """Run every single-mapper invariant on ``hw``. Family-monotonicity
    is excluded because it spans multiple mappers (run separately by
    the pytest wrapper)."""
    checks: List[Tuple[str, Callable[[], List[str]]]] = [
        ("roofline_self_consistency",
         lambda: check_roofline_self_consistency(hw, precision, shapes)),
        ("latency_non_decreasing_in_size",
         lambda: check_latency_non_decreasing_in_size(hw, precision, shapes)),
        ("memory_time_scales_with_bytes",
         lambda: check_memory_time_scales_with_bytes(hw, precision, shapes)),
        ("achieved_compute_below_peak",
         lambda: check_achieved_compute_below_peak(hw, precision, shapes)),
        ("achieved_bw_below_peak",
         lambda: check_achieved_bw_below_peak(hw, precision, shapes)),
        ("avg_power_below_tdp",
         lambda: check_avg_power_below_tdp(hw, precision, shapes)),
    ]
    return InvariantReport(
        hardware_name=hardware_name,
        precision=precision,
        failures_per_check={name: fn() for name, fn in checks},
    )
