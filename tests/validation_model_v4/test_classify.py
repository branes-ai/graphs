"""Hand-curated regime classification cases for the v4 harness.

Each case below is computed by hand from publicly-known hardware capacities
and the working-set / OI formulas in classify.py. Locking these in means
the regime math can't drift unnoticed (Principle 1 of the v4 plan: every
sweep shape lands in exactly one known regime).

Conventions:
- Working set = (M*K + K*N + M*N) * bytes_per_element
- AI breakpoint = peak_FLOPS / peak_DRAM_BW
- Ambiguous band = +/-20% of any boundary
- Launch-bound: ideal_compute_time < 5 * launch_overhead (default 5us)
"""

import pytest

from graphs.hardware.mappers.cpu import create_i7_12700k_mapper
from graphs.hardware.mappers.gpu import create_h100_sxm5_80gb_mapper
from validation.model_v4.sweeps.classify import (
    AMBIGUOUS_BAND,
    Regime,
    bytes_per_element,
    classify_regime,
    hardware_capacities,
    op_footprint,
)
from graphs.hardware.resource_model import Precision


# ---------------------------------------------------------------------------
# Foundational math: bytes_per_element + op_footprint
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype,expected", [
    ("fp64", 8), ("fp32", 4), ("tf32", 4),
    ("fp16", 2), ("bf16", 2),
    ("int8", 1), ("fp8", 1), ("fp8_e4m3", 1),
    ("int4", 0.5), ("fp4", 0.5),
])
def test_bytes_per_element(dtype, expected):
    assert bytes_per_element(dtype) == expected


def test_bytes_per_element_rejects_unknown():
    with pytest.raises(ValueError, match="Unknown dtype"):
        bytes_per_element("complex32")


def test_matmul_footprint_math():
    """1024-cube matmul at fp16: WS = 3*1024^2*2 = 6MB, FLOPs = 2*1024^3."""
    fp = op_footprint("matmul", (1024, 1024, 1024), "fp16")
    assert fp.working_set_bytes == 3 * 1024 * 1024 * 2
    assert fp.flops == 2 * 1024 ** 3
    assert fp.operational_intensity == fp.flops / fp.working_set_bytes


def test_linear_footprint_math():
    """Linear B=8 IN=512 OUT=256 at fp32:
    WS = (8*512 + 512*256 + 8*256) * 4 = (4096+131072+2048)*4 = 549,664"""
    fp = op_footprint("linear", (8, 512, 256), "fp32")
    assert fp.working_set_bytes == (8 * 512 + 512 * 256 + 8 * 256) * 4
    assert fp.flops == 2 * 8 * 512 * 256


def test_op_footprint_rejects_unknown_op():
    with pytest.raises(ValueError, match="Unsupported op"):
        op_footprint("conv2d", (1, 1, 1), "fp32")


# ---------------------------------------------------------------------------
# hardware_capacities: derived constants are correct
# ---------------------------------------------------------------------------


def test_i7_12700k_capacities_fp32():
    """i7-12700K: 16 cores * 32KB L1 = 512KB L1 total. L2_cache_total
    holds the LLC value (25MB Intel L3 by M1 schema convention).

    cap.peak_flops is the EFFECTIVE peak (after thermal / efficiency
    derate), not the raw theoretical spec from precision_profiles. See
    the docstring of hardware_capacities for the rationale (#68)."""
    from graphs.estimation.roofline import RooflineAnalyzer
    hw = create_i7_12700k_mapper().resource_model
    cap = hardware_capacities(hw, Precision.FP32)
    assert cap.l1_total_bytes == hw.compute_units * hw.l1_cache_per_unit
    assert cap.l2_total_bytes == hw.l2_cache_total
    assert cap.on_chip_total_bytes == cap.l1_total_bytes + cap.l2_total_bytes
    assert cap.peak_dram_bandwidth_bps == hw.peak_bandwidth
    # Effective peak (used for AI breakpoint), NOT theoretical spec
    expected_peak = RooflineAnalyzer._get_effective_peak_ops(hw, Precision.FP32)
    assert cap.peak_flops == expected_peak
    assert cap.ai_breakpoint == cap.peak_flops / cap.peak_dram_bandwidth_bps
    # Sanity: effective should be derated from theoretical
    assert cap.peak_flops < hw.get_peak_ops(Precision.FP32)


def test_h100_capacities_fp16():
    """H100: 132 SMs * 256KB L1 = 33MB; 50MB L2; 989 TFLOPS fp16 / 3.35 TB/s
    DRAM BW gives an AI breakpoint around 295 (varies with mapper data)."""
    hw = create_h100_sxm5_80gb_mapper().resource_model
    cap = hardware_capacities(hw, Precision.FP16)
    # 132 SMs * 256 KB = 33 MB
    assert cap.l1_total_bytes == 132 * 256 * 1024
    assert cap.l2_total_bytes == 50 * 1024 * 1024
    # AI breakpoint should be in the right ballpark for Hopper fp16
    assert 200 < cap.ai_breakpoint < 700


# ---------------------------------------------------------------------------
# classify_regime: hand-curated shape -> bucket on i7-12700K (fp32)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def i7():
    return create_i7_12700k_mapper().resource_model


@pytest.fixture(scope="module")
def h100():
    return create_h100_sxm5_80gb_mapper().resource_model


# i7-12700K fp32: L1=512KB, LLC=25MB, peak=720 GFLOPS, BW=75 GB/s,
# AI_break = 720/75 = 9.6 FLOPS/byte. Launch overhead = 5us; threshold = 25us.
@pytest.mark.parametrize("shape,expected", [
    # Tiny: compute_time = 2*64^3 / 720 GFLOPS = 0.7us << 25us -> launch_bound
    ((64, 64, 64), Regime.LAUNCH_BOUND),
    # WS=768KB > L1 (512KB), in L2 (25MB), comfortably away from boundaries
    ((256, 256, 256), Regime.L2_BOUND),
    # WS=12MB squarely in (L1, L2]
    ((1024, 1024, 1024), Regime.L2_BOUND),
    # WS=48MB > L2 (25MB), away from boundary
    ((2048, 2048, 2048), Regime.DRAM_BOUND),
    # WS=192MB clearly DRAM
    ((4096, 4096, 4096), Regime.DRAM_BOUND),
])
def test_i7_matmul_fp32_classification(i7, shape, expected):
    assert classify_regime("matmul", shape, "fp32", i7) == expected


# H100 fp16: L1=33MB, L2=50MB, peak=989 TFLOPS, BW=3.35 TB/s, AI_break~295.
# Launch overhead = 5us; threshold = 25us.
@pytest.mark.parametrize("shape,expected", [
    # 64^3: compute = 2*64^3 / 989e12 = 0.5ns -> launch_bound
    ((64, 64, 64), Regime.LAUNCH_BOUND),
    # 1024^3 fp16: WS=6MB <<L1, OI=341 > AI_break(~295) but within band of breakpoint;
    # however compute = 2*1024^3 / 989 TFLOPS = 2.2us << 25us -> launch_bound
    ((1024, 1024, 1024), Regime.LAUNCH_BOUND),
    # 2048^3 fp16: WS=24MB < L1(33MB), OI=683 >> AI_break -> alu_bound
    ((2048, 2048, 2048), Regime.ALU_BOUND),
    # 4096^3 fp16: WS=96MB > L2(50MB) -> dram_bound (away from L2 boundary)
    ((4096, 4096, 4096), Regime.DRAM_BOUND),
])
def test_h100_matmul_fp16_classification(h100, shape, expected):
    assert classify_regime("matmul", shape, "fp16", h100) == expected


# ---------------------------------------------------------------------------
# Boundary / ambiguous band
# ---------------------------------------------------------------------------


def test_shape_at_l2_boundary_is_ambiguous(i7):
    """A shape whose working set lands within +/-20% of the L2 boundary
    is rejected as ambiguous so failures attribute to one layer."""
    cap = hardware_capacities(i7, Precision.FP32)
    target_ws = cap.l2_total_bytes  # land exactly on the L2 boundary
    # For matmul (M, K, N) with M=K=N: WS = 3*N^2*4 -> N = sqrt(target_ws/12)
    import math
    N = int(math.sqrt(target_ws / 12))
    fp = op_footprint("matmul", (N, N, N), "fp32")
    # confirm we landed near the boundary
    assert abs(fp.working_set_bytes - target_ws) / target_ws < AMBIGUOUS_BAND
    assert classify_regime("matmul", (N, N, N), "fp32", i7) == Regime.AMBIGUOUS


# ---------------------------------------------------------------------------
# Unsupported (hardware, dtype) combos
# ---------------------------------------------------------------------------


def test_i7_fp16_is_unsupported(i7):
    """Alder Lake P-cores have no native fp16. Post-#57 get_peak_ops raises;
    the classifier must surface this as UNSUPPORTED, not crash."""
    assert classify_regime("matmul", (1024, 1024, 1024), "fp16", i7) == Regime.UNSUPPORTED


def test_h100_fp64_is_supported(h100):
    """Negative control: H100 does support fp64, so this should NOT be UNSUPPORTED."""
    r = classify_regime("matmul", (4096, 4096, 4096), "fp64", h100)
    assert r != Regime.UNSUPPORTED


# ---------------------------------------------------------------------------
# Linear classification: same math, different shape vector
# ---------------------------------------------------------------------------


def test_i7_linear_dram_bound(i7):
    """Linear (B=64, IN=8192, OUT=8192) fp32:
    WS = (64*8192 + 8192*8192 + 64*8192)*4 = ~268 MB -> well over LLC."""
    assert classify_regime("linear", (64, 8192, 8192), "fp32", i7) == Regime.DRAM_BOUND


def test_h100_linear_alu_bound(h100):
    """Linear (B=4096, IN=4096, OUT=4096) fp16:
    WS = (4096^2 + 4096^2 + 4096^2)*2 = 96MB > L1(33MB) -- but actually
    Linear fits in L2 (50MB?) — let me recompute: 96 MB > L2 (50MB) -> dram.
    Use a smaller shape that fits L1 and is OI-rich for a clean ALU test."""
    # B=2048, IN=2048, OUT=2048 fp16 -> WS = 3*2048^2 * 2 = 24 MB < L1(33MB)
    # OI = 2*B*IN*OUT / WS = 2*2048^3 / (24MB) = 716 >> AI_break(~295) -> alu
    assert classify_regime("linear", (2048, 2048, 2048), "fp16", h100) == Regime.ALU_BOUND
