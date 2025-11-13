#!/usr/bin/env python3
"""
Phase 7 Research Energy Model Validation

Validates that the Phase 7 research model has correct:
1. Process node matching actual fabrication process
2. Physics-based energy values (process_node × circuit_type_multiplier)
3. Compute fabric configuration

This test ensures energy comparisons for research architectures are trustworthy.
"""
import sys
import pytest

# Add src to path for imports
sys.path.insert(0, 'src')

from graphs.hardware.models.research.dfm_128 import dfm_128_resource_model

# Model specifications: (name, factory_fn, expected_process_nm, expected_energy_pj, fabric_count, architecture_type)
RESEARCH_MODELS = [
    ('DFM-128', dfm_128_resource_model, 7, 1.62, 1, 'DataFlow'),
]


@pytest.mark.parametrize("name,model_fn,expected_process_nm,expected_energy_pj,fabric_count,arch_type", RESEARCH_MODELS)
def test_research_energy_values(name, model_fn, expected_process_nm, expected_energy_pj, fabric_count, arch_type):
    """Test research model has correct process node and energy value."""
    model = model_fn()

    # Check model has compute fabrics
    assert model.compute_fabrics is not None, f"{name}: Missing compute_fabrics"
    assert len(model.compute_fabrics) == fabric_count, \
        f"{name}: Expected {fabric_count} fabric(s), got {len(model.compute_fabrics)}"

    # Check fabric has correct process node
    fabric = model.compute_fabrics[0]
    assert fabric.process_node_nm == expected_process_nm, \
        f"{name}: Fabric {fabric.fabric_type} has process_node_nm={fabric.process_node_nm}, expected {expected_process_nm}"

    # Check energy value
    energy_pj = model.energy_per_flop_fp32 * 1e12
    assert abs(energy_pj - expected_energy_pj) < 0.01, \
        f"{name}: energy_per_flop_fp32={energy_pj:.2f} pJ, expected {expected_energy_pj:.2f} pJ"


def test_phase7_completeness():
    """Test that Phase 7 research model is accounted for."""
    assert len(RESEARCH_MODELS) == 1, "Expected 1 Phase 7 research model"


def test_dfm_128_processing_element_fabric():
    """Test that DFM-128 has dataflow processing element fabric."""
    model = dfm_128_resource_model()
    assert len(model.compute_fabrics) == 1, "Expected 1 fabric (processing elements)"

    fabric = model.compute_fabrics[0]
    assert fabric.fabric_type == "dfm_processing_element", "Expected dfm_processing_element fabric"
    assert fabric.circuit_type == "simd_packed", "DFM PEs should use simd_packed (VLIW-like datapath)"
    assert fabric.num_units == 8, "DFM-128 should have 8 processing elements"


def test_dfm_128_precision_support():
    """Test that DFM-128 supports FP32/FP16/BF16/INT8/INT4 with correct ops per clock."""
    model = dfm_128_resource_model()
    fabric = model.compute_fabrics[0]

    from graphs.hardware.resource_model import Precision

    # Check ops per clock for each precision
    assert Precision.FP32 in fabric.ops_per_unit_per_clock, "Missing FP32 support"
    assert Precision.FP16 in fabric.ops_per_unit_per_clock, "Missing FP16 support"
    assert Precision.BF16 in fabric.ops_per_unit_per_clock, "Missing BF16 support"
    assert Precision.INT8 in fabric.ops_per_unit_per_clock, "Missing INT8 support"
    assert Precision.INT4 in fabric.ops_per_unit_per_clock, "Missing INT4 support"

    # Check FP16 is 2× FP32
    fp32_ops = fabric.ops_per_unit_per_clock[Precision.FP32]
    fp16_ops = fabric.ops_per_unit_per_clock[Precision.FP16]
    assert fp16_ops == fp32_ops * 2, f"FP16 should be 2× FP32, got {fp16_ops} vs {fp32_ops}"

    # Check BF16 matches FP16
    bf16_ops = fabric.ops_per_unit_per_clock[Precision.BF16]
    assert bf16_ops == fp16_ops, f"BF16 should match FP16, got {bf16_ops} vs {fp16_ops}"


def test_dfm_128_peak_performance_fp32():
    """Test that DFM-128 achieves 64 GFLOPS FP32."""
    model = dfm_128_resource_model()
    fabric = model.compute_fabrics[0]

    from graphs.hardware.resource_model import Precision

    # Calculate peak FP32: 8 PEs × 4 ops/cycle × 2.0 GHz = 64 GFLOPS
    num_pes = fabric.num_units
    ops_per_pe = fabric.ops_per_unit_per_clock[Precision.FP32]
    clock_hz = fabric.core_frequency_hz

    peak_fp32_gflops = (num_pes * ops_per_pe * clock_hz) / 1e9
    expected_fp32_gflops = 64.0

    assert abs(peak_fp32_gflops - expected_fp32_gflops) < 0.01, \
        f"FP32 peak should be ~64 GFLOPS, got {peak_fp32_gflops:.2f} GFLOPS"


def test_dfm_128_peak_performance_int8():
    """Test that DFM-128 achieves 32 GOPS INT8."""
    model = dfm_128_resource_model()
    fabric = model.compute_fabrics[0]

    from graphs.hardware.resource_model import Precision

    # Calculate peak INT8: 8 PEs × 2 ops/cycle × 2.0 GHz = 32 GOPS
    num_pes = fabric.num_units
    ops_per_pe = fabric.ops_per_unit_per_clock[Precision.INT8]
    clock_hz = fabric.core_frequency_hz

    peak_int8_gops = (num_pes * ops_per_pe * clock_hz) / 1e9
    expected_int8_gops = 32.0

    assert abs(peak_int8_gops - expected_int8_gops) < 0.01, \
        f"INT8 peak should be ~32 GOPS, got {peak_int8_gops:.2f} GOPS"


def test_process_node_7nm():
    """Test that 7nm process node has correct energy."""
    model = dfm_128_resource_model()
    energy_pj = model.energy_per_flop_fp32 * 1e12

    # 7nm simd_packed should be 1.62 pJ
    assert abs(energy_pj - 1.62) < 0.01, f"7nm simd_packed should be ~1.62 pJ, got {energy_pj:.2f} pJ"


def test_dfm_128_clock_frequency():
    """Test that DFM-128 runs at 2.0 GHz."""
    model = dfm_128_resource_model()
    fabric = model.compute_fabrics[0]

    clock_ghz = fabric.core_frequency_hz / 1e9
    expected_ghz = 2.0

    assert abs(clock_ghz - expected_ghz) < 0.01, \
        f"Clock frequency should be ~2.0 GHz, got {clock_ghz:.2f} GHz"


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
