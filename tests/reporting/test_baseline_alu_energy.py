"""Tests for the baseline ALU energy model.

Validates that:
- All (op, precision) combinations produce non-negative finite energies.
- FADD < FMUL < FMA at fp32 (Horowitz ratios respected).
- Narrow precisions scale by the documented multiplier
  (fp16 ~= 0.5 x fp32 per-op, int8 ~= 0.20 x fp32 per-op).
- FMA total is always FF_read x 3 + MUL + ADD + FF_write.
- Per-FLOP normalization: FADD/FMUL = total/1, FMA = total/2.
- Confidence is THEORETICAL with the source string mentioning the
  technology profile and Horowitz ratios.
"""
from __future__ import annotations

import pytest

from graphs.core.confidence import ConfidenceLevel
from graphs.hardware.technology_profile import (
    EDGE_8NM_LPDDR5,
    DATACENTER_4NM_HBM3,
)
from graphs.reporting.baseline_alu_energy import (
    OpKind,
    Precision,
    baseline_alu_energy,
)


_ALL_OPS = list(OpKind)
_ALL_PRECS = list(Precision)


class TestStructure:
    @pytest.mark.parametrize("op", _ALL_OPS)
    @pytest.mark.parametrize("prec", _ALL_PRECS)
    def test_every_combo_returns_positive_finite(self, op, prec):
        b = baseline_alu_energy(EDGE_8NM_LPDDR5, op, prec)
        assert b.total_pj > 0
        assert b.pj_per_flop > 0
        assert b.components.ff_read_each_pj > 0
        assert b.components.ff_write_each_pj > 0

    def test_sources_match_op_kind(self):
        assert baseline_alu_energy(
            EDGE_8NM_LPDDR5, OpKind.FADD, Precision.FP32
        ).components.sources == 2
        assert baseline_alu_energy(
            EDGE_8NM_LPDDR5, OpKind.FMUL, Precision.FP32
        ).components.sources == 2
        assert baseline_alu_energy(
            EDGE_8NM_LPDDR5, OpKind.FMA, Precision.FP32
        ).components.sources == 3

    def test_flops_per_op(self):
        assert baseline_alu_energy(
            EDGE_8NM_LPDDR5, OpKind.FADD, Precision.FP32
        ).components.flops_per_op == 1
        assert baseline_alu_energy(
            EDGE_8NM_LPDDR5, OpKind.FMUL, Precision.FP32
        ).components.flops_per_op == 1
        assert baseline_alu_energy(
            EDGE_8NM_LPDDR5, OpKind.FMA, Precision.FP32
        ).components.flops_per_op == 2

    def test_fadd_has_no_mul_component(self):
        b = baseline_alu_energy(EDGE_8NM_LPDDR5, OpKind.FADD, Precision.FP32)
        assert b.components.mul_pj is None
        assert b.components.add_pj is not None and b.components.add_pj > 0

    def test_fmul_has_no_add_component(self):
        b = baseline_alu_energy(EDGE_8NM_LPDDR5, OpKind.FMUL, Precision.FP32)
        assert b.components.add_pj is None
        assert b.components.mul_pj is not None and b.components.mul_pj > 0

    def test_fma_has_both_components(self):
        b = baseline_alu_energy(EDGE_8NM_LPDDR5, OpKind.FMA, Precision.FP32)
        assert b.components.mul_pj is not None and b.components.mul_pj > 0
        assert b.components.add_pj is not None and b.components.add_pj > 0

    def test_total_is_sum_of_components(self):
        for op in _ALL_OPS:
            for prec in _ALL_PRECS:
                b = baseline_alu_energy(EDGE_8NM_LPDDR5, op, prec)
                expected = (
                    b.components.ff_total_read_pj
                    + (b.components.mul_pj or 0)
                    + (b.components.add_pj or 0)
                    + b.components.ff_write_each_pj
                )
                assert b.total_pj == pytest.approx(expected)


class TestRatios:
    def test_fadd_lt_fmul_lt_fma_at_fp32(self):
        fadd = baseline_alu_energy(EDGE_8NM_LPDDR5, OpKind.FADD, Precision.FP32).total_pj
        fmul = baseline_alu_energy(EDGE_8NM_LPDDR5, OpKind.FMUL, Precision.FP32).total_pj
        fma  = baseline_alu_energy(EDGE_8NM_LPDDR5, OpKind.FMA,  Precision.FP32).total_pj
        assert fadd < fmul < fma

    def test_fmul_to_fma_alu_ratio(self):
        """FMA's MUL component should match FMUL's MUL component
        (same multiplier, same precision). FMA adds a small ADD on
        top, so FMA total > FMUL total but compute ratio is in a
        narrow band."""
        fmul = baseline_alu_energy(EDGE_8NM_LPDDR5, OpKind.FMUL, Precision.FP32)
        fma  = baseline_alu_energy(EDGE_8NM_LPDDR5, OpKind.FMA,  Precision.FP32)
        # FMA's MUL == FMUL's MUL
        assert fma.components.mul_pj == pytest.approx(fmul.components.mul_pj)
        # FMA's ADD is ~12% of FMA, FMUL has no ADD
        assert fma.components.add_pj > 0

    def test_precision_scaling_fp16_vs_fp32(self):
        """fp16 ALU energy ~= 0.5 x fp32 (the documented scale)."""
        for op in _ALL_OPS:
            fp32 = baseline_alu_energy(EDGE_8NM_LPDDR5, op, Precision.FP32)
            fp16 = baseline_alu_energy(EDGE_8NM_LPDDR5, op, Precision.FP16)
            # Compare ALU components only (not flops, which don't scale).
            alu_fp32 = (fp32.components.mul_pj or 0) + (fp32.components.add_pj or 0)
            alu_fp16 = (fp16.components.mul_pj or 0) + (fp16.components.add_pj or 0)
            assert alu_fp16 == pytest.approx(alu_fp32 * 0.5)

    def test_precision_scaling_int8_vs_fp32(self):
        for op in _ALL_OPS:
            fp32 = baseline_alu_energy(EDGE_8NM_LPDDR5, op, Precision.FP32)
            int8 = baseline_alu_energy(EDGE_8NM_LPDDR5, op, Precision.INT8)
            alu_fp32 = (fp32.components.mul_pj or 0) + (fp32.components.add_pj or 0)
            alu_int8 = (int8.components.mul_pj or 0) + (int8.components.add_pj or 0)
            assert alu_int8 == pytest.approx(alu_fp32 * 0.20)

    def test_per_flop_normalization(self):
        fmul_fp32 = baseline_alu_energy(EDGE_8NM_LPDDR5, OpKind.FMUL, Precision.FP32)
        fma_fp32  = baseline_alu_energy(EDGE_8NM_LPDDR5, OpKind.FMA,  Precision.FP32)
        # FMUL: 1 FLOP per op
        assert fmul_fp32.pj_per_flop == pytest.approx(fmul_fp32.total_pj / 1)
        # FMA: 2 FLOPS per op
        assert fma_fp32.pj_per_flop == pytest.approx(fma_fp32.total_pj / 2)


class TestPlausibility:
    def test_fp32_fma_in_range(self):
        """FMA fp32 baseline should be in [0.5, 5] pJ for any
        modern node (Samsung 8nm: ~2 pJ; TSMC 4nm: ~1.5 pJ)."""
        for profile in (EDGE_8NM_LPDDR5, DATACENTER_4NM_HBM3):
            b = baseline_alu_energy(profile, OpKind.FMA, Precision.FP32)
            assert 0.5 <= b.total_pj <= 5.0, (
                f"{profile.name}: {b.total_pj} pJ outside expected range"
            )

    def test_smaller_node_lower_energy(self):
        """At fixed (op, precision), a smaller node uses less energy."""
        b8  = baseline_alu_energy(EDGE_8NM_LPDDR5, OpKind.FMA, Precision.FP32).total_pj
        b4  = baseline_alu_energy(DATACENTER_4NM_HBM3, OpKind.FMA, Precision.FP32).total_pj
        assert b4 < b8


class TestConfidence:
    def test_level_is_theoretical(self):
        b = baseline_alu_energy(EDGE_8NM_LPDDR5, OpKind.FMA, Precision.FP32)
        assert b.confidence.level is ConfidenceLevel.THEORETICAL

    def test_source_mentions_profile_and_horowitz(self):
        b = baseline_alu_energy(EDGE_8NM_LPDDR5, OpKind.FMA, Precision.FP32)
        assert "TechnologyProfile" in b.confidence.source
        assert EDGE_8NM_LPDDR5.name in b.confidence.source
        assert "horowitz" in b.confidence.source.lower()


class TestRow:
    def test_as_row_keys(self):
        b = baseline_alu_energy(EDGE_8NM_LPDDR5, OpKind.FMA, Precision.FP32)
        row = b.as_row()
        assert set(row.keys()) == {"FF_read", "MUL", "ADD", "FF_write", "Total"}
        # Total should equal the sum of the other cells.
        assert row["Total"] == pytest.approx(
            row["FF_read"] + row["MUL"] + row["ADD"] + row["FF_write"]
        )
