"""
Accelerator Model Consistency Tests (DSP, DPU, CGRA, Hailo)

5-category consistency checks for the remaining architecture classes:
- DSP (Qualcomm Hexagon, TI C7x, CEVA, Cadence, Synopsys) -- 10 mappers
- DPU (Xilinx Vitis AI) -- 1 mapper
- CGRA (Stanford Plasticine) -- 1 mapper
- Hailo (spatial partition, registered under KPU hardware type) -- 2 mappers
"""

from __future__ import annotations

import pytest
from graphs.hardware.resource_model import Precision


# =========================================================================
# DSP Tests
# =========================================================================

class TestDSPLayerDecomposition:

    def test_baseline_energy_is_positive(self, dsp_mappers, medium_matmul):
        for name, mapper in dsp_mappers:
            ops = medium_matmul.total_flops
            bytes_t = (medium_matmul.total_input_bytes +
                       medium_matmul.total_output_bytes +
                       medium_matmul.total_weight_bytes)
            compute_e, memory_e = mapper._calculate_energy(ops, bytes_t, Precision.FP32)
            assert compute_e > 0, f"{name}: compute energy <= 0"
            assert memory_e > 0, f"{name}: memory energy <= 0"


class TestDSPMonotonicity:

    def test_more_ops_means_more_energy(self, dsp_mappers):
        for name, mapper in dsp_mappers:
            e_small = sum(mapper._calculate_energy(1_000_000, 4_000_000, Precision.FP32))
            e_large = sum(mapper._calculate_energy(100_000_000, 4_000_000, Precision.FP32))
            assert e_large > e_small, f"{name}: more ops did not increase energy"


class TestDSPPrecisionScaling:

    def test_int8_cheaper_than_fp32(self, dsp_mappers):
        for name, mapper in dsp_mappers:
            e_fp32 = mapper._calculate_energy(10_000_000, 4_000_000, Precision.FP32)
            e_int8 = mapper._calculate_energy(10_000_000, 4_000_000, Precision.INT8)
            assert e_int8[0] < e_fp32[0], f"{name}: INT8 compute >= FP32"

    def test_energy_scaling_in_range(self, dsp_mappers):
        for name, mapper in dsp_mappers:
            for prec, scale in mapper.resource_model.energy_scaling.items():
                assert 0 < scale <= 4.0, f"{name}: energy_scaling[{prec}] = {scale}"


class TestDSPCoefficientSensitivity:

    def test_energy_per_flop_perturbation(self, dsp_mappers, medium_matmul):
        for name, mapper in dsp_mappers:
            ops = medium_matmul.total_flops
            bytes_t = (medium_matmul.total_input_bytes +
                       medium_matmul.total_output_bytes +
                       medium_matmul.total_weight_bytes)
            base_c, _ = mapper._calculate_energy(ops, bytes_t, Precision.FP32)
            original = mapper.resource_model.energy_per_flop_fp32
            mapper.resource_model.energy_per_flop_fp32 = original * 1.10
            pert_c, _ = mapper._calculate_energy(ops, bytes_t, Precision.FP32)
            mapper.resource_model.energy_per_flop_fp32 = original
            if base_c > 0:
                delta = (pert_c - base_c) / base_c
                assert 0.05 < delta < 0.20, f"{name}: {delta*100:.1f}% (expected ~10%)"


class TestDSPCrossMapper:

    def test_no_dsp_has_negative_latency(self, dsp_mappers, medium_matmul):
        for name, mapper in dsp_mappers:
            ops = medium_matmul.total_flops
            bytes_t = (medium_matmul.total_input_bytes +
                       medium_matmul.total_output_bytes +
                       medium_matmul.total_weight_bytes)
            ct, mt, _ = mapper._calculate_latency(
                ops, bytes_t,
                allocated_units=mapper.resource_model.compute_units,
                occupancy=1.0, precision=Precision.FP32,
            )
            assert ct >= 0, f"{name}: negative compute_time"
            assert mt >= 0, f"{name}: negative memory_time"

    def test_no_dsp_has_zero_bandwidth(self, dsp_mappers):
        for name, mapper in dsp_mappers:
            assert mapper.resource_model.peak_bandwidth > 0, f"{name}: zero bandwidth"


# =========================================================================
# DPU / CGRA Tests
# =========================================================================

class TestDPUCGRADecomposition:

    def test_baseline_energy_is_positive(self, dpu_cgra_mappers, medium_matmul):
        for name, mapper in dpu_cgra_mappers:
            ops = medium_matmul.total_flops
            bytes_t = (medium_matmul.total_input_bytes +
                       medium_matmul.total_output_bytes +
                       medium_matmul.total_weight_bytes)
            compute_e, memory_e = mapper._calculate_energy(ops, bytes_t, Precision.FP32)
            assert compute_e > 0, f"{name}: compute energy <= 0"
            assert memory_e > 0, f"{name}: memory energy <= 0"


class TestDPUCGRAMonotonicity:

    def test_more_ops_means_more_energy(self, dpu_cgra_mappers):
        for name, mapper in dpu_cgra_mappers:
            e_small = sum(mapper._calculate_energy(1_000_000, 4_000_000, Precision.FP32))
            e_large = sum(mapper._calculate_energy(100_000_000, 4_000_000, Precision.FP32))
            assert e_large > e_small, f"{name}: more ops did not increase energy"


class TestDPUCGRAPrecision:

    def test_int8_cheaper_than_fp32(self, dpu_cgra_mappers):
        for name, mapper in dpu_cgra_mappers:
            e_fp32 = mapper._calculate_energy(10_000_000, 4_000_000, Precision.FP32)
            e_int8 = mapper._calculate_energy(10_000_000, 4_000_000, Precision.INT8)
            assert e_int8[0] < e_fp32[0], f"{name}: INT8 compute >= FP32"


class TestDPUCGRACrossMapper:

    def test_no_negative_latency(self, dpu_cgra_mappers, medium_matmul):
        for name, mapper in dpu_cgra_mappers:
            ops = medium_matmul.total_flops
            bytes_t = (medium_matmul.total_input_bytes +
                       medium_matmul.total_output_bytes +
                       medium_matmul.total_weight_bytes)
            ct, mt, _ = mapper._calculate_latency(
                ops, bytes_t,
                allocated_units=mapper.resource_model.compute_units,
                occupancy=1.0, precision=Precision.FP32,
            )
            assert ct >= 0, f"{name}: negative compute_time"
            assert mt >= 0, f"{name}: negative memory_time"


# =========================================================================
# Hailo Tests (spatial partition, registered under KPU hw type)
# =========================================================================

class TestHailoDecomposition:

    def test_baseline_energy_is_positive(self, hailo_mappers, medium_matmul):
        for name, mapper in hailo_mappers:
            ops = medium_matmul.total_flops
            bytes_t = (medium_matmul.total_input_bytes +
                       medium_matmul.total_output_bytes +
                       medium_matmul.total_weight_bytes)
            compute_e, memory_e = mapper._calculate_energy(ops, bytes_t, Precision.INT8)
            assert compute_e > 0, f"{name}: compute energy <= 0"
            assert memory_e > 0, f"{name}: memory energy <= 0"


class TestHailoMonotonicity:

    def test_more_ops_means_more_energy(self, hailo_mappers):
        for name, mapper in hailo_mappers:
            e_small = sum(mapper._calculate_energy(1_000_000, 4_000_000, Precision.INT8))
            e_large = sum(mapper._calculate_energy(100_000_000, 4_000_000, Precision.INT8))
            assert e_large > e_small, f"{name}: more ops did not increase energy"


class TestHailoPrecision:

    def test_int8_cheaper_than_fp32(self, hailo_mappers):
        for name, mapper in hailo_mappers:
            e_fp32 = mapper._calculate_energy(10_000_000, 4_000_000, Precision.FP32)
            e_int8 = mapper._calculate_energy(10_000_000, 4_000_000, Precision.INT8)
            assert e_int8[0] < e_fp32[0], f"{name}: INT8 compute >= FP32"


class TestHailoCrossMapper:

    def test_all_hailo_have_positive_peak(self, hailo_mappers):
        for name, mapper in hailo_mappers:
            peak = mapper.resource_model.get_peak_ops(Precision.INT8)
            assert peak > 0, f"{name}: zero or negative INT8 peak"

    def test_no_negative_latency(self, hailo_mappers, medium_matmul):
        for name, mapper in hailo_mappers:
            ops = medium_matmul.total_flops
            bytes_t = (medium_matmul.total_input_bytes +
                       medium_matmul.total_output_bytes +
                       medium_matmul.total_weight_bytes)
            ct, mt, _ = mapper._calculate_latency(
                ops, bytes_t,
                allocated_units=mapper.resource_model.compute_units,
                occupancy=1.0, precision=Precision.INT8,
            )
            assert ct >= 0, f"{name}: negative compute_time"
            assert mt >= 0, f"{name}: negative memory_time"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
