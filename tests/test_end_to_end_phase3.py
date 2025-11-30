"""
End-to-end integration test for Phase 3.

Tests the complete pipeline:
1. Dynamo-based workload characterization
2. Energy model with MAC/FLOP/IntOp separation
"""

import pytest
import torch
import torch.nn as nn

from graphs.analysis.dynamo_characterizer import characterize_with_dynamo
from graphs.hardware.architectural_energy import DataParallelEnergyModel
from graphs.hardware.technology_profile import DEFAULT_PROFILE


class TestEndToEndPhase3:
    """End-to-end integration tests"""

    def test_mlp_complete_pipeline(self):
        """Test complete pipeline: Dynamo → Energy Model"""

        # Step 1: Define model
        class MLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(256, 256, bias=True)

            def forward(self, x):
                x = self.linear(x)
                x = torch.relu(x)
                return x

        model = MLP()
        input_data = torch.randn(1, 256)

        # Step 2: Characterize with Dynamo
        print("\n" + "=" * 80)
        print("STEP 1: Dynamo Workload Characterization")
        print("=" * 80)
        workload = characterize_with_dynamo(model, input_data, verbose=True)
        print(f"\n{workload}")

        # Step 3: Calculate energy
        print("\n" + "=" * 80)
        print("STEP 2: Energy Calculation")
        print("=" * 80)
        energy_model = DataParallelEnergyModel(tech_profile=DEFAULT_PROFILE)
        breakdown = energy_model.compute_architectural_energy(
            workload=workload,
            bytes_transferred=workload.bytes_transferred,
            execution_context={
                'concurrent_threads': 200_000,
                'warp_size': 32,
                'cache_line_size': 128
            }
        )

        # Verify workload characterization
        assert workload.macs == 65536, "MLP 256×256 should have 65,536 MACs"
        assert workload.flops == 512, "Bias (256) + ReLU (256) = 512 FLOPs"
        assert workload.intops == 0, "No integer operations"

        # Verify energy breakdown
        assert breakdown.mac_ops_executed == 65536
        assert breakdown.flop_ops_executed == 512
        assert breakdown.intop_ops_executed == 0

        # Verify Tensor Core counting
        tensor_core_ops = breakdown.extra_details['tensor_core_ops']
        assert tensor_core_ops == 819, f"Expected 819 TC ops, got {tensor_core_ops}"

        # Print summary
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"\nWorkload:")
        print(f"  MACs:  {workload.macs:,}")
        print(f"  FLOPs: {workload.flops:,}")
        print(f"  IntOps: {workload.intops:,}")
        print(f"\nHardware Mapping:")
        print(f"  Tensor Cores: {tensor_core_ops:,} ops ({breakdown.extra_details['tensor_core_macs']:,} MACs)")
        print(f"  CUDA Cores (MACs): {breakdown.extra_details['cuda_core_macs']:,} MACs")
        print(f"  CUDA Cores (FLOPs): {breakdown.extra_details['cuda_core_flops']:,} FLOPs")
        print(f"\nEnergy:")
        print(f"  MAC Energy:  {breakdown.mac_energy * 1e12:.2f} pJ")
        print(f"  FLOP Energy: {breakdown.flop_energy * 1e12:.2f} pJ")
        print(f"  Total Compute: {breakdown.total_compute_energy * 1e12:.2f} pJ")
        print(f"  Total (with overhead): {(breakdown.total_overhead + breakdown.total_compute_energy) * 1e12:.2f} pJ")

    def test_linear_no_bias(self):
        """Test linear layer without bias (pure MACs)"""
        model = nn.Linear(256, 256, bias=False)
        input_data = torch.randn(1, 256)

        # Characterize
        workload = characterize_with_dynamo(model, input_data, verbose=False)

        # Should have only MACs, no FLOPs
        assert workload.macs == 65536
        assert workload.flops == 0
        assert workload.intops == 0

        # Calculate energy
        energy_model = DataParallelEnergyModel(tech_profile=DEFAULT_PROFILE)
        breakdown = energy_model.compute_architectural_energy(
            workload=workload,
            bytes_transferred=workload.bytes_transferred
        )

        # Should have MAC energy but no FLOP energy
        assert breakdown.mac_energy > 0
        assert breakdown.flop_energy == 0
        assert breakdown.intop_energy == 0

        print(f"\nLinear (no bias):")
        print(f"  MACs: {workload.macs:,}, Energy: {breakdown.mac_energy * 1e12:.2f} pJ")
        print(f"  FLOPs: {workload.flops:,}, Energy: {breakdown.flop_energy * 1e12:.2f} pJ")

    def test_batched_workload(self):
        """Test with batched input"""
        model = nn.Linear(256, 256, bias=True)
        input_data = torch.randn(8, 256)  # Batch size 8

        # Characterize
        workload = characterize_with_dynamo(model, input_data, verbose=False)

        # Batch of 8 should multiply operations
        # 8 × 256 × 256 = 524,288 MACs
        # 8 × 256 = 2,048 bias FLOPs
        assert workload.macs == 524288
        assert workload.flops == 2048

        # Calculate energy
        energy_model = DataParallelEnergyModel(tech_profile=DEFAULT_PROFILE)
        breakdown = energy_model.compute_architectural_energy(
            workload=workload,
            bytes_transferred=workload.bytes_transferred
        )

        # Verify scaling
        tensor_core_macs = breakdown.extra_details['tensor_core_macs']
        tensor_core_ops = breakdown.extra_details['tensor_core_ops']

        print(f"\nBatched (batch=8):")
        print(f"  MACs: {workload.macs:,}")
        print(f"  FLOPs: {workload.flops:,}")
        print(f"  Tensor Core Ops: {tensor_core_ops:,} (from {tensor_core_macs:,} MACs)")

        # 80% of 524,288 MACs = 419,430 MACs on Tensor Cores
        # 419,430 / 64 = 6,553 Tensor Core operations
        assert tensor_core_macs == int(524288 * 0.8)
        assert tensor_core_ops == tensor_core_macs // 64


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
