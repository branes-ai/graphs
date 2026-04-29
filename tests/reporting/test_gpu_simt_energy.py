"""Tests for the GPU SIMT instruction-energy model.

Validates:
- Pipeline structure: 9 stages with correct activity counts.
- Stages 1-4 fire ``sm_subpartitions`` times (NOT once).
- RF read activity = sources_per_op x lanes (256 for FADD/FMUL,
  384 for FMA at 128 lanes).
- Per-op normalization: pj_per_op = total / (lanes x packing).
- fp16-packed / fp32 per-op ratio in [0.45, 0.55] for every op.
- int8-packed / fp32 per-op ratio in [0.20, 0.30] for every op.
- FADD overhead ratio > FMUL > FMA (cheap compute -> bigger
  overhead share).
- Stage total + row total summations match grand total.
"""
from __future__ import annotations

import pytest

from graphs.core.confidence import ConfidenceLevel
from graphs.hardware.technology_profile import EDGE_8NM_LPDDR5
from graphs.reporting.baseline_alu_energy import (
    OpKind,
    Precision,
    baseline_alu_energy,
)
from graphs.reporting.gpu_simt_energy import (
    DEFAULT_LANES_PER_SUBPART,
    DEFAULT_SUBPARTITIONS,
    STAGE_LABELS,
    simt_instruction_energy,
)


_ALL_OPS = list(OpKind)
_ALL_PRECS = list(Precision)


class TestPipelineStructure:
    def test_nine_stages_in_order(self):
        s = simt_instruction_energy(EDGE_8NM_LPDDR5, OpKind.FMA, Precision.FP32)
        assert len(s.stages) == 9
        labels = [stage.label for stage in s.stages]
        assert labels == STAGE_LABELS

    def test_subpartition_stages_fire_four_times(self):
        s = simt_instruction_energy(EDGE_8NM_LPDDR5, OpKind.FMA, Precision.FP32)
        # Stages 1-4 (Fch / Dec / Sch / Dsp) should have activity_count
        # equal to sm_subpartitions, NOT 1. This was the bug the user
        # caught at the planning stage.
        for stage in s.stages[:4]:
            assert stage.activity_count == DEFAULT_SUBPARTITIONS, (
                f"{stage.label} fired {stage.activity_count} times; "
                f"expected {DEFAULT_SUBPARTITIONS} (one per subpartition)"
            )

    def test_rf_read_count_matches_sources(self):
        # FMUL: 2 sources x 128 lanes = 256
        s_fmul = simt_instruction_energy(
            EDGE_8NM_LPDDR5, OpKind.FMUL, Precision.FP32
        )
        rd = next(st for st in s_fmul.stages if st.label == "Rd")
        assert rd.activity_count == 256

        # FMA: 3 sources x 128 lanes = 384
        s_fma = simt_instruction_energy(
            EDGE_8NM_LPDDR5, OpKind.FMA, Precision.FP32
        )
        rd = next(st for st in s_fma.stages if st.label == "Rd")
        assert rd.activity_count == 384

    def test_writeback_count_is_lanes(self):
        s = simt_instruction_energy(EDGE_8NM_LPDDR5, OpKind.FMA, Precision.FP32)
        wb = next(st for st in s.stages if st.label == "WB")
        assert wb.activity_count == 128

    def test_compute_count_is_lanes(self):
        s = simt_instruction_energy(EDGE_8NM_LPDDR5, OpKind.FMA, Precision.FP32)
        exe = next(st for st in s.stages if st.label == "Exe")
        assert exe.activity_count == 128

    def test_lanes_property(self):
        s = simt_instruction_energy(EDGE_8NM_LPDDR5, OpKind.FMA, Precision.FP32)
        assert s.lanes == DEFAULT_SUBPARTITIONS * DEFAULT_LANES_PER_SUBPART
        assert s.lanes == 128

    def test_custom_topology(self):
        # GA100 datacenter SM is also 4x32, but verify that overrides
        # propagate through.
        s = simt_instruction_energy(
            EDGE_8NM_LPDDR5, OpKind.FMA, Precision.FP32,
            sm_subpartitions=2, lanes_per_subpartition=16,
        )
        assert s.lanes == 32
        for stage in s.stages[:4]:
            assert stage.activity_count == 2


class TestNormalization:
    def test_ops_executed_matches_packing(self):
        for op in _ALL_OPS:
            for prec, expected in [
                (Precision.FP32, 128),
                (Precision.FP16, 256),
                (Precision.INT8, 512),
            ]:
                s = simt_instruction_energy(EDGE_8NM_LPDDR5, op, prec)
                assert s.ops_executed == expected, (
                    f"{op}/{prec}: ops_executed={s.ops_executed}, "
                    f"expected {expected}"
                )

    def test_flops_executed_for_fma(self):
        # FMA = 2 FLOPS each
        assert simt_instruction_energy(
            EDGE_8NM_LPDDR5, OpKind.FMA, Precision.FP32
        ).flops_executed == 256
        assert simt_instruction_energy(
            EDGE_8NM_LPDDR5, OpKind.FMA, Precision.FP16
        ).flops_executed == 512
        assert simt_instruction_energy(
            EDGE_8NM_LPDDR5, OpKind.FMA, Precision.INT8
        ).flops_executed == 1024

    def test_per_op_equals_total_over_ops(self):
        for op in _ALL_OPS:
            for prec in _ALL_PRECS:
                s = simt_instruction_energy(EDGE_8NM_LPDDR5, op, prec)
                assert s.pj_per_op == pytest.approx(
                    s.total_pj / s.ops_executed
                )

    def test_total_equals_sum_of_stages(self):
        for op in _ALL_OPS:
            for prec in _ALL_PRECS:
                s = simt_instruction_energy(EDGE_8NM_LPDDR5, op, prec)
                assert s.total_pj == pytest.approx(
                    sum(stg.total_pj for stg in s.stages)
                )


class TestPrecisionRatios:
    @pytest.mark.parametrize("op", _ALL_OPS)
    def test_fp16_packed_halves_per_op_energy(self, op):
        """Per-instruction energy is approximately constant across
        precisions (same 32-bit datapath, packed); per-op halves
        because useful work doubles."""
        fp32 = simt_instruction_energy(EDGE_8NM_LPDDR5, op, Precision.FP32)
        fp16 = simt_instruction_energy(EDGE_8NM_LPDDR5, op, Precision.FP16)
        ratio = fp16.pj_per_op / fp32.pj_per_op
        assert 0.45 <= ratio <= 0.55, (
            f"{op}: fp16/fp32 per-op ratio = {ratio:.3f}, "
            f"expected in [0.45, 0.55]"
        )

    @pytest.mark.parametrize("op", _ALL_OPS)
    def test_int8_packed_quarters_per_op_energy(self, op):
        fp32 = simt_instruction_energy(EDGE_8NM_LPDDR5, op, Precision.FP32)
        int8 = simt_instruction_energy(EDGE_8NM_LPDDR5, op, Precision.INT8)
        ratio = int8.pj_per_op / fp32.pj_per_op
        assert 0.20 <= ratio <= 0.30, (
            f"{op}: int8/fp32 per-op ratio = {ratio:.3f}, "
            f"expected in [0.20, 0.30]"
        )


class TestOverheadOrder:
    @pytest.mark.parametrize("prec", _ALL_PRECS)
    def test_fadd_overhead_ratio_exceeds_fmul_and_fma(self, prec):
        """FADD has the cheapest compute, so SIMT/baseline ratio is
        largest (control + RF energy is the same across ops, but the
        denominator -- baseline ALU cost -- shrinks for cheap ops)."""
        ratios = {}
        for op in _ALL_OPS:
            b = baseline_alu_energy(EDGE_8NM_LPDDR5, op, prec)
            s = simt_instruction_energy(EDGE_8NM_LPDDR5, op, prec)
            ratios[op] = s.pj_per_op / b.total_pj
        assert ratios[OpKind.FADD] > ratios[OpKind.FMUL]
        assert ratios[OpKind.FADD] > ratios[OpKind.FMA]


class TestRowMatrix:
    def test_fma_has_three_source_rows(self):
        s = simt_instruction_energy(EDGE_8NM_LPDDR5, OpKind.FMA, Precision.FP32)
        keys = list(s.rows.keys())
        assert "Src operand A" in keys
        assert "Src operand B" in keys
        assert "Src operand C" in keys

    def test_fmul_has_two_source_rows(self):
        s = simt_instruction_energy(EDGE_8NM_LPDDR5, OpKind.FMUL, Precision.FP32)
        keys = list(s.rows.keys())
        assert "Src operand A" in keys
        assert "Src operand B" in keys
        assert "Src operand C" not in keys

    def test_compute_row_only_in_exe_stage(self):
        s = simt_instruction_energy(EDGE_8NM_LPDDR5, OpKind.FMA, Precision.FP32)
        compute_row = s.rows["ALU compute"]
        assert "Exe" in compute_row
        # The compute row should have no entries for fetch/decode/etc.
        for label in ("Fch", "Dec", "Sch", "Dsp", "Rd", "OC", "Disp", "WB"):
            assert label not in compute_row

    def test_writeback_row_only_in_wb_stage(self):
        s = simt_instruction_energy(EDGE_8NM_LPDDR5, OpKind.FMA, Precision.FP32)
        wb_row = s.rows["Dest writeback"]
        assert "WB" in wb_row
        for label in STAGE_LABELS[:-1]:
            assert label not in wb_row


class TestConfidence:
    def test_level_is_theoretical(self):
        s = simt_instruction_energy(EDGE_8NM_LPDDR5, OpKind.FMA, Precision.FP32)
        assert s.confidence.level is ConfidenceLevel.THEORETICAL

    def test_source_mentions_topology(self):
        s = simt_instruction_energy(EDGE_8NM_LPDDR5, OpKind.FMA, Precision.FP32)
        assert "TechnologyProfile" in s.confidence.source
        assert str(DEFAULT_SUBPARTITIONS) in s.confidence.source
        assert str(DEFAULT_LANES_PER_SUBPART) in s.confidence.source


class TestPlausibility:
    def test_simt_total_in_range(self):
        """SIMT FMA fp32 total per SM-cycle should land in
        [200, 3000] pJ for the Orin profile. Outside that = bug."""
        s = simt_instruction_energy(EDGE_8NM_LPDDR5, OpKind.FMA, Precision.FP32)
        assert 200 <= s.total_pj <= 3000, (
            f"SIMT FMA fp32 total = {s.total_pj} pJ, outside [200, 3000]"
        )

    def test_simt_exceeds_baseline(self):
        """SIMT must always cost more per op than baseline (overhead
        is non-negative)."""
        for op in _ALL_OPS:
            for prec in _ALL_PRECS:
                b = baseline_alu_energy(EDGE_8NM_LPDDR5, op, prec)
                s = simt_instruction_energy(EDGE_8NM_LPDDR5, op, prec)
                assert s.pj_per_op > b.total_pj
