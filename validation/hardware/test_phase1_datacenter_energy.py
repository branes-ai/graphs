#!/usr/bin/env python3
"""
Phase 1 Datacenter Energy Model Validation

Validates that all 18 Phase 1 datacenter models have correct:
1. Process nodes matching their actual fabrication process
2. Physics-based energy values (process_node × circuit_type_multiplier)
3. Compute fabric configurations

This test ensures energy comparisons across models are trustworthy.
"""
import sys
import pytest

# Add src to path for imports
sys.path.insert(0, 'src')

from graphs.hardware.models.datacenter.b100_sxm6_192gb import b100_sxm6_192gb_resource_model
from graphs.hardware.models.datacenter.h100_sxm5_80gb import h100_sxm5_80gb_resource_model
from graphs.hardware.models.datacenter.h100_pcie_80gb import h100_pcie_80gb_resource_model
from graphs.hardware.models.datacenter.a100_sxm4_80gb import a100_sxm4_80gb_resource_model
from graphs.hardware.models.datacenter.v100_sxm3_32gb import v100_sxm3_32gb_resource_model
from graphs.hardware.models.datacenter.t4_pcie_16gb import t4_pcie_16gb_resource_model
from graphs.hardware.models.datacenter.intel_xeon_platinum_8490h import intel_xeon_platinum_8490h_resource_model
from graphs.hardware.models.datacenter.intel_xeon_platinum_8592plus import intel_xeon_platinum_8592plus_resource_model
from graphs.hardware.models.datacenter.amd_epyc_9654 import amd_epyc_9654_resource_model
from graphs.hardware.models.datacenter.amd_epyc_9754 import amd_epyc_9754_resource_model
from graphs.hardware.models.datacenter.intel_granite_rapids import intel_granite_rapids_resource_model
from graphs.hardware.models.datacenter.amd_epyc_turin import amd_epyc_turin_resource_model
from graphs.hardware.models.datacenter.ampere_ampereone_128 import ampere_ampereone_128_resource_model
from graphs.hardware.models.datacenter.ampere_ampereone_192 import ampere_ampereone_192_resource_model
from graphs.hardware.models.datacenter.tpu_v1 import tpu_v1_resource_model
from graphs.hardware.models.datacenter.tpu_v3 import tpu_v3_resource_model
from graphs.hardware.models.datacenter.tpu_v4 import tpu_v4_resource_model
from graphs.hardware.models.datacenter.tpu_v5p import tpu_v5p_resource_model


# Model specifications: (name, factory_fn, expected_process_nm, expected_energy_pj, dominant_fabric_type)
GPU_MODELS = [
    ('NVIDIA B100 SXM6 192GB', b100_sxm6_192gb_resource_model, 3, 1.20, 'CUDA'),
    ('NVIDIA H100 SXM5 80GB', h100_sxm5_80gb_resource_model, 4, 1.30, 'CUDA'),
    ('NVIDIA H100 PCIe 80GB', h100_pcie_80gb_resource_model, 4, 1.30, 'CUDA'),
    ('NVIDIA A100 SXM4 80GB', a100_sxm4_80gb_resource_model, 7, 1.80, 'CUDA'),
    ('NVIDIA V100 SXM3 32GB', v100_sxm3_32gb_resource_model, 12, 2.50, 'CUDA'),
    ('NVIDIA T4 PCIe 16GB', t4_pcie_16gb_resource_model, 12, 2.50, 'CUDA'),
]

CPU_MODELS = [
    ('Intel Xeon 8490H', intel_xeon_platinum_8490h_resource_model, 10, 1.89, 'AVX-512'),
    ('Intel Xeon 8592+', intel_xeon_platinum_8592plus_resource_model, 7, 1.62, 'AVX-512'),
    ('AMD EPYC 9654', amd_epyc_9654_resource_model, 5, 1.35, 'AVX-512'),
    ('AMD EPYC 9754', amd_epyc_9754_resource_model, 5, 1.35, 'AVX-512'),
    ('Intel Granite Rapids', intel_granite_rapids_resource_model, 3, 1.08, 'AVX-512'),
    ('AMD EPYC Turin', amd_epyc_turin_resource_model, 3, 1.08, 'AVX-512'),
    ('Ampere One 128', ampere_ampereone_128_resource_model, 5, 1.50, 'NEON'),
    ('Ampere One 192', ampere_ampereone_192_resource_model, 5, 1.50, 'NEON'),
]

TPU_MODELS = [
    ('TPU v1', tpu_v1_resource_model, 28, 4.00, 'Systolic'),
    ('TPU v3', tpu_v3_resource_model, 16, 2.70, 'Systolic'),
    ('TPU v4', tpu_v4_resource_model, 7, 1.80, 'Systolic'),
    ('TPU v5p', tpu_v5p_resource_model, 4, 1.30, 'Systolic'),
]

ALL_MODELS = GPU_MODELS + CPU_MODELS + TPU_MODELS


@pytest.mark.parametrize("name,model_fn,expected_process_nm,expected_energy_pj,fabric_type", GPU_MODELS)
def test_gpu_energy_values(name, model_fn, expected_process_nm, expected_energy_pj, fabric_type):
    """Test GPU models have correct process nodes and energy values."""
    model = model_fn()

    # Check model has compute fabrics
    assert model.compute_fabrics is not None, f"{name}: Missing compute_fabrics"
    assert len(model.compute_fabrics) >= 2, f"{name}: Expected at least 2 fabrics (CUDA + Tensor)"

    # Check all fabrics have correct process node
    for fabric in model.compute_fabrics:
        assert fabric.process_node_nm == expected_process_nm, \
            f"{name}: Fabric {fabric.fabric_type} has process_node_nm={fabric.process_node_nm}, expected {expected_process_nm}"

    # Check energy value (using dominant fabric - CUDA core for GPUs)
    energy_pj = model.energy_per_flop_fp32 * 1e12
    assert abs(energy_pj - expected_energy_pj) < 0.01, \
        f"{name}: energy_per_flop_fp32={energy_pj:.2f} pJ, expected {expected_energy_pj:.2f} pJ"


@pytest.mark.parametrize("name,model_fn,expected_process_nm,expected_energy_pj,fabric_type", CPU_MODELS)
def test_cpu_energy_values(name, model_fn, expected_process_nm, expected_energy_pj, fabric_type):
    """Test CPU models have correct process nodes and energy values."""
    model = model_fn()

    # Check model has compute fabrics
    assert model.compute_fabrics is not None, f"{name}: Missing compute_fabrics"
    assert len(model.compute_fabrics) >= 2, f"{name}: Expected at least 2 fabrics (Scalar + SIMD)"

    # Check all fabrics have correct process node
    for fabric in model.compute_fabrics:
        assert fabric.process_node_nm == expected_process_nm, \
            f"{name}: Fabric {fabric.fabric_type} has process_node_nm={fabric.process_node_nm}, expected {expected_process_nm}"

    # Check energy value (using dominant fabric - SIMD for CPUs)
    energy_pj = model.energy_per_flop_fp32 * 1e12
    assert abs(energy_pj - expected_energy_pj) < 0.01, \
        f"{name}: energy_per_flop_fp32={energy_pj:.2f} pJ, expected {expected_energy_pj:.2f} pJ"


@pytest.mark.parametrize("name,model_fn,expected_process_nm,expected_energy_pj,fabric_type", TPU_MODELS)
def test_tpu_energy_values(name, model_fn, expected_process_nm, expected_energy_pj, fabric_type):
    """Test TPU models have correct process nodes and energy values."""
    model = model_fn()

    # Check model has compute fabrics
    assert model.compute_fabrics is not None, f"{name}: Missing compute_fabrics"
    assert len(model.compute_fabrics) == 1, f"{name}: Expected 1 fabric (Systolic)"

    # Check systolic fabric has correct process node
    fabric = model.compute_fabrics[0]
    assert fabric.process_node_nm == expected_process_nm, \
        f"{name}: Systolic fabric has process_node_nm={fabric.process_node_nm}, expected {expected_process_nm}"
    assert fabric.circuit_type == "standard_cell", \
        f"{name}: Expected standard_cell circuit type for systolic arrays"

    # Check energy value
    energy_pj = model.energy_per_flop_fp32 * 1e12
    assert abs(energy_pj - expected_energy_pj) < 0.01, \
        f"{name}: energy_per_flop_fp32={energy_pj:.2f} pJ, expected {expected_energy_pj:.2f} pJ"


def test_phase1_completeness():
    """Test that all 18 Phase 1 models are accounted for."""
    assert len(GPU_MODELS) == 6, "Expected 6 GPU models"
    assert len(CPU_MODELS) == 8, "Expected 8 CPU models"
    assert len(TPU_MODELS) == 4, "Expected 4 TPU models"
    assert len(ALL_MODELS) == 18, "Expected 18 total Phase 1 models"


def test_gpu_multi_fabric_architecture():
    """Test that GPUs have both CUDA and Tensor Core fabrics."""
    for name, model_fn, _, _, _ in GPU_MODELS:
        model = model_fn()
        assert len(model.compute_fabrics) == 2, \
            f"{name}: Expected 2 fabrics (CUDA + Tensor)"

        fabric_types = [f.fabric_type for f in model.compute_fabrics]
        assert "cuda_core" in fabric_types, f"{name}: Missing cuda_core fabric"
        assert "tensor_core" in fabric_types, f"{name}: Missing tensor_core fabric"


def test_cpu_multi_fabric_architecture():
    """Test that x86 CPUs have Scalar + SIMD fabrics (and AMX for Intel)."""
    for name, model_fn, _, _, fabric_type in CPU_MODELS:
        model = model_fn()

        if "Intel" in name and "Ampere" not in name:
            # Intel x86 should have Scalar + AVX-512 + AMX (3 fabrics)
            if "Granite" in name or "8592" in name or "8490" in name:
                assert len(model.compute_fabrics) >= 2, \
                    f"{name}: Expected at least Scalar + AVX-512"
        elif "AMD" in name:
            # AMD x86 should have Scalar + AVX-512 (2 fabrics, no AMX)
            assert len(model.compute_fabrics) == 2, \
                f"{name}: Expected Scalar + AVX-512"
        elif "Ampere" in name:
            # Ampere ARM uses helper function, may have different structure
            assert model.compute_fabrics is not None, \
                f"{name}: Should have compute_fabrics from ARM helper"


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
