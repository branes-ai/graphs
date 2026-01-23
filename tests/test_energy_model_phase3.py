"""
Unit tests for Phase 3: Energy Model with MAC/FLOP/IntOp Separation

Tests the updated DataParallelEnergyModel with WorkloadCharacterization.
"""

import pytest
from graphs.estimation.workload_characterization import (
    WorkloadCharacterization,
    create_simple_workload
)
from graphs.hardware.architectural_energy import DataParallelEnergyModel
from graphs.hardware.technology_profile import DEFAULT_PROFILE


class TestEnergyModelPhase3:
    """Test energy model with MAC/FLOP/IntOp separation"""

    def test_backward_compatibility(self):
        """Test that legacy interface still works"""
        model = DataParallelEnergyModel(tech_profile=DEFAULT_PROFILE)

        # Old way: just pass ops
        breakdown = model.compute_architectural_energy(
            ops=66048,
            bytes_transferred=264192
        )

        # Should work without errors
        assert breakdown.mac_ops_executed == 66048
        assert breakdown.flop_ops_executed == 0
        assert breakdown.intop_ops_executed == 0

    def test_workload_characterization_path(self):
        """Test new path with WorkloadCharacterization"""
        model = DataParallelEnergyModel(tech_profile=DEFAULT_PROFILE)

        # MLP 256×256 with bias + ReLU
        workload = WorkloadCharacterization(
            macs=65536,
            flops=512,
            intops=0,
            bytes_transferred=264192
        )

        breakdown = model.compute_architectural_energy(
            workload=workload,
            bytes_transferred=264192
        )

        # Verify operation counts
        assert breakdown.mac_ops_executed == 65536
        assert breakdown.flop_ops_executed == 512
        assert breakdown.intop_ops_executed == 0
        assert breakdown.total_ops_executed == 66048

        # Verify energy components are populated
        assert breakdown.mac_energy > 0
        assert breakdown.flop_energy > 0
        assert breakdown.intop_energy == 0

        print(f"\nEnergy Breakdown:")
        print(f"  MAC Energy:  {breakdown.mac_energy * 1e12:.2f} pJ")
        print(f"  FLOP Energy: {breakdown.flop_energy * 1e12:.2f} pJ")
        print(f"  IntOp Energy: {breakdown.intop_energy * 1e12:.2f} pJ")
        print(f"  Total Compute: {breakdown.total_compute_energy * 1e12:.2f} pJ")

    def test_tensor_core_counting_fixed(self):
        """Test that Tensor Core operation counting is fixed"""
        model = DataParallelEnergyModel(tech_profile=DEFAULT_PROFILE)

        # 65,536 MACs
        workload = WorkloadCharacterization(
            macs=65536,
            flops=0,
            intops=0,
            bytes_transferred=0
        )

        breakdown = model.compute_architectural_energy(
            workload=workload,
            bytes_transferred=0
        )

        # Extract detailed breakdown
        tensor_core_macs = breakdown.extra_details['tensor_core_macs']
        tensor_core_ops = breakdown.extra_details['tensor_core_ops']

        # 80% utilization → 52,428 MACs on Tensor Cores
        assert tensor_core_macs == int(65536 * 0.80)

        # Each Tensor Core op does 64 MACs (4×4×4)
        # 52,428 / 64 = 819 operations
        expected_tensor_core_ops = tensor_core_macs // 64
        assert tensor_core_ops == expected_tensor_core_ops

        print(f"\nTensor Core Counting:")
        print(f"  Total MACs: {65536:,}")
        print(f"  Tensor Core MACs: {tensor_core_macs:,} (80%)")
        print(f"  Tensor Core Ops: {tensor_core_ops:,} (each op = 64 MACs)")
        print(f"  ✅ FIXED: Was incorrectly reporting {tensor_core_macs:,} ops")

    def test_mac_vs_flop_separation(self):
        """Test that MACs and FLOPs are handled separately"""
        model = DataParallelEnergyModel(tech_profile=DEFAULT_PROFILE)

        # Pure MACs (no bias)
        workload_macs = WorkloadCharacterization(
            macs=65536,
            flops=0,
            intops=0,
            bytes_transferred=0
        )

        # Pure FLOPs (no MACs)
        workload_flops = WorkloadCharacterization(
            macs=0,
            flops=65536,
            intops=0,
            bytes_transferred=0
        )

        breakdown_macs = model.compute_architectural_energy(
            workload=workload_macs,
            bytes_transferred=0
        )

        breakdown_flops = model.compute_architectural_energy(
            workload=workload_flops,
            bytes_transferred=0
        )

        # MAC energy should go through Tensor Cores (more efficient)
        # FLOP energy should go through CUDA cores
        assert breakdown_macs.mac_energy > 0
        assert breakdown_macs.flop_energy == 0

        assert breakdown_flops.mac_energy == 0
        assert breakdown_flops.flop_energy > 0

        print(f"\nMAC vs FLOP Separation:")
        print(f"  65,536 MACs:  MAC energy = {breakdown_macs.mac_energy * 1e12:.2f} pJ")
        print(f"  65,536 FLOPs: FLOP energy = {breakdown_flops.flop_energy * 1e12:.2f} pJ")

    def test_mlp_example(self):
        """Test complete MLP 256×256 + bias + ReLU example"""
        model = DataParallelEnergyModel(tech_profile=DEFAULT_PROFILE)

        # MLP 256×256 with bias + ReLU
        # MACs: 65,536 (matmul)
        # FLOPs: 512 (256 bias + 256 ReLU)
        workload = create_simple_workload(
            macs=65536,
            flops=512,
            intops=0,
            bytes_transferred=264192,
            model_name="mlp_256x256_relu"
        )

        breakdown = model.compute_architectural_energy(
            workload=workload,
            bytes_transferred=264192,
            execution_context={
                'concurrent_threads': 200_000,
                'warp_size': 32,
                'cache_line_size': 128
            }
        )

        # Verify counts
        assert breakdown.mac_ops_executed == 65536
        assert breakdown.flop_ops_executed == 512
        assert breakdown.total_ops_executed == 66048

        # Verify energy is calculated
        assert breakdown.mac_energy > 0
        assert breakdown.flop_energy > 0
        assert breakdown.total_compute_energy > 0

        # Verify Tensor Core operations
        tensor_core_ops = breakdown.extra_details['tensor_core_ops']
        tensor_core_macs = breakdown.extra_details['tensor_core_macs']
        assert tensor_core_macs == int(65536 * 0.8)
        assert tensor_core_ops == tensor_core_macs // 64

        print(f"\nMLP 256×256 + Bias + ReLU:")
        print(f"  MACs: {breakdown.mac_ops_executed:,}")
        print(f"  FLOPs: {breakdown.flop_ops_executed:,}")
        print(f"  IntOps: {breakdown.intop_ops_executed:,}")
        print(f"\n  Hardware Allocation:")
        print(f"    Tensor Cores: {tensor_core_ops:,} TC ops ({tensor_core_macs:,} MACs)")
        print(f"    CUDA Cores (MACs): {breakdown.extra_details['cuda_core_macs']:,} MACs")
        print(f"    CUDA Cores (FLOPs): {breakdown.extra_details['cuda_core_flops']:,} FLOPs")
        print(f"\n  Energy:")
        print(f"    MAC Energy: {breakdown.mac_energy * 1e12:.2f} pJ")
        print(f"    FLOP Energy: {breakdown.flop_energy * 1e12:.2f} pJ")
        print(f"    Total Compute: {breakdown.total_compute_energy * 1e12:.2f} pJ")

    def test_explanation_text(self):
        """Test that explanation text shows correct breakdown"""
        model = DataParallelEnergyModel(tech_profile=DEFAULT_PROFILE)

        workload = WorkloadCharacterization(
            macs=65536,
            flops=512,
            intops=100,
            bytes_transferred=264192
        )

        breakdown = model.compute_architectural_energy(
            workload=workload,
            bytes_transferred=264192
        )

        # Verify explanation contains all operation types
        assert "Tensor Cores" in breakdown.explanation
        assert "CUDA Cores (MACs)" in breakdown.explanation
        assert "CUDA Cores (FLOPs)" in breakdown.explanation
        assert "Integer ALUs" in breakdown.explanation

        # Verify counts are shown
        assert "65,536" in breakdown.explanation or "52,428" in breakdown.explanation  # MACs
        assert "512" in breakdown.explanation  # FLOPs
        assert "100" in breakdown.explanation  # IntOps

        print(f"\n{breakdown.explanation}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
