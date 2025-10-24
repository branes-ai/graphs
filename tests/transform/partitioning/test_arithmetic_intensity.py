#!/usr/bin/env python
"""
Arithmetic Intensity Validation Script
========================================

Validates the accuracy of FLOP counting and Arithmetic Intensity calculations
by comparing against:
1. Manual calculations for simple operations
2. Known values for standard models
3. Sanity checks for operation types
4. Fusion behavior validation

Usage:
    python tests/validate_arithmetic_intensity.py
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torch.fx import symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp
import sys
sys.path.insert(0, 'src')

from graphs.transform.partitioning import GraphPartitioner
from graphs.transform.partitioning import FusionBasedPartitioner


class ArithmeticIntensityValidator:
    """Comprehensive validation of Arithmetic Intensity calculations"""

    def __init__(self):
        self.tests_passed = 0
        self.tests_failed = 0
        self.tolerance_flops = 0.01  # 1% tolerance
        self.tolerance_ai = 0.1  # Absolute tolerance for AI

    def run_all_tests(self):
        """Run all validation tests"""
        print("=" * 80)
        print("ARITHMETIC INTENSITY VALIDATION")
        print("=" * 80)
        print()

        self.test_simple_conv2d()
        self.test_simple_linear()
        self.test_batchnorm()
        self.test_relu()
        self.test_resnet18_total()
        self.test_ai_ranges()
        self.test_fusion_improves_ai()

        # Summary
        print("\n" + "=" * 80)
        print("VALIDATION SUMMARY")
        print("=" * 80)
        total = self.tests_passed + self.tests_failed
        print(f"\nTests passed: {self.tests_passed}/{total}")
        print(f"Tests failed: {self.tests_failed}/{total}")

        if self.tests_failed == 0:
            print("\nAll tests PASSED!")
            return 0
        else:
            print(f"\n{self.tests_failed} tests FAILED - review calculations")
            return 1

    def test_simple_conv2d(self):
        """Test Conv2d FLOP counting and AI calculation"""
        print("\n" + "-" * 80)
        print("TEST 1: Simple Conv2d Layer")
        print("-" * 80)

        # Create simple Conv2d wrapped in a module (FX requires this)
        class SimpleConv(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
            def forward(self, x):
                return self.conv(x)

        model = SimpleConv()
        model.eval()
        input_tensor = torch.randn(1, 64, 56, 56)

        # Manual calculation
        B, C_in, H, W = 1, 64, 56, 56
        C_out, K = 128, 3
        H_out = W_out = 56  # padding=1, stride=1 preserves size

        # FLOPs = 2 * MACs (each MAC is multiply + add)
        # MACs = B * C_out * H_out * W_out * C_in * K * K
        macs = B * C_out * H_out * W_out * C_in * K * K
        expected_flops = 2 * macs
        expected_input_bytes = B * C_in * H * W * 4
        expected_output_bytes = B * C_out * H_out * W_out * 4
        expected_weight_bytes = C_out * C_in * K * K * 4
        expected_total_bytes = expected_input_bytes + expected_output_bytes + expected_weight_bytes
        expected_ai = expected_flops / expected_total_bytes

        # Run our partitioner
        fx_graph = symbolic_trace(model)
        ShapeProp(fx_graph).propagate(input_tensor)
        partitioner = GraphPartitioner()
        report = partitioner.partition(fx_graph)

        actual_flops = report.subgraphs[0].flops
        actual_total_bytes = (report.subgraphs[0].total_input_bytes +
                             report.subgraphs[0].total_output_bytes +
                             report.subgraphs[0].total_weight_bytes)
        actual_ai = report.subgraphs[0].arithmetic_intensity

        # Validate
        print(f"\nExpected FLOPs: {expected_flops:,}")
        print(f"Actual FLOPs:   {actual_flops:,}")
        print(f"Difference:     {abs(expected_flops - actual_flops):,}")

        print(f"\nExpected Total Bytes: {expected_total_bytes:,}")
        print(f"Actual Total Bytes:   {actual_total_bytes:,}")

        print(f"\nExpected AI: {expected_ai:.2f} FLOPs/byte")
        print(f"Actual AI:   {actual_ai:.2f} FLOPs/byte")
        print(f"Difference:  {abs(expected_ai - actual_ai):.2f}")

        flops_match = abs(expected_flops - actual_flops) / expected_flops < self.tolerance_flops
        ai_match = abs(expected_ai - actual_ai) < self.tolerance_ai

        if flops_match and ai_match:
            print("\nRESULT: PASS")
            self.tests_passed += 1
        else:
            print("\nRESULT: FAIL")
            self.tests_failed += 1

    def test_simple_linear(self):
        """Test Linear layer FLOP counting and AI calculation"""
        print("\n" + "-" * 80)
        print("TEST 2: Simple Linear Layer")
        print("-" * 80)

        # Create simple Linear layer wrapped in a module
        class SimpleLinear(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(512, 1000)
            def forward(self, x):
                return self.fc(x)

        model = SimpleLinear()
        model.eval()
        input_tensor = torch.randn(1, 512)

        # Manual calculation
        B, D_in, D_out = 1, 512, 1000

        # FLOPs = 2 * MACs
        # MACs = B * D_in * D_out
        macs = B * D_in * D_out
        expected_flops = 2 * macs
        expected_input_bytes = B * D_in * 4
        expected_output_bytes = B * D_out * 4
        expected_weight_bytes = D_in * D_out * 4 + D_out * 4  # weights + bias
        expected_total_bytes = expected_input_bytes + expected_output_bytes + expected_weight_bytes
        expected_ai = expected_flops / expected_total_bytes

        # Run our partitioner
        fx_graph = symbolic_trace(model)
        ShapeProp(fx_graph).propagate(input_tensor)
        partitioner = GraphPartitioner()
        report = partitioner.partition(fx_graph)

        actual_flops = report.subgraphs[0].flops
        actual_ai = report.subgraphs[0].arithmetic_intensity

        print(f"\nExpected FLOPs: {expected_flops:,}")
        print(f"Actual FLOPs:   {actual_flops:,}")
        print(f"\nExpected AI: {expected_ai:.2f} FLOPs/byte")
        print(f"Actual AI:   {actual_ai:.2f} FLOPs/byte")

        flops_match = abs(expected_flops - actual_flops) / expected_flops < self.tolerance_flops
        ai_match = abs(expected_ai - actual_ai) < self.tolerance_ai

        if flops_match and ai_match:
            print("\nRESULT: PASS")
            self.tests_passed += 1
        else:
            print("\nRESULT: FAIL")
            self.tests_failed += 1

    def test_batchnorm(self):
        """Test BatchNorm AI is low (memory-bound)"""
        print("\n" + "-" * 80)
        print("TEST 3: BatchNorm Arithmetic Intensity (should be low)")
        print("-" * 80)

        class SimpleBN(nn.Module):
            def __init__(self):
                super().__init__()
                self.bn = nn.BatchNorm2d(64)
            def forward(self, x):
                return self.bn(x)

        model = SimpleBN()
        model.eval()
        input_tensor = torch.randn(1, 64, 56, 56)

        fx_graph = symbolic_trace(model)
        ShapeProp(fx_graph).propagate(input_tensor)
        partitioner = GraphPartitioner()
        report = partitioner.partition(fx_graph)

        actual_ai = report.subgraphs[0].arithmetic_intensity

        print(f"BatchNorm AI: {actual_ai:.2f} FLOPs/byte")
        print(f"Expected range: 0.5 - 2.0 (memory-bound)")

        # BatchNorm should have low AI (< 2.0)
        if 0.3 < actual_ai < 2.0:
            print("\nRESULT: PASS (AI is appropriately low)")
            self.tests_passed += 1
        else:
            print("\nRESULT: FAIL (AI out of expected range)")
            self.tests_failed += 1

    def test_relu(self):
        """Test ReLU AI is very low (bandwidth-bound)"""
        print("\n" + "-" * 80)
        print("TEST 4: ReLU Arithmetic Intensity (should be very low)")
        print("-" * 80)

        class SimpleReLU(nn.Module):
            def __init__(self):
                super().__init__()
                self.relu = nn.ReLU()
            def forward(self, x):
                return self.relu(x)

        model = SimpleReLU()
        model.eval()
        input_tensor = torch.randn(1, 64, 56, 56)

        fx_graph = symbolic_trace(model)
        ShapeProp(fx_graph).propagate(input_tensor)
        partitioner = GraphPartitioner()
        report = partitioner.partition(fx_graph)

        actual_ai = report.subgraphs[0].arithmetic_intensity

        print(f"ReLU AI: {actual_ai:.2f} FLOPs/byte")
        print(f"Expected range: 0.05 - 0.2 (bandwidth-bound)")

        # ReLU should have very low AI (< 0.5)
        if 0.05 < actual_ai < 0.5:
            print("\nRESULT: PASS (AI is appropriately very low)")
            self.tests_passed += 1
        else:
            print("\nRESULT: FAIL (AI out of expected range)")
            self.tests_failed += 1

    def test_resnet18_total(self):
        """Test ResNet-18 total FLOPs against known value"""
        print("\n" + "-" * 80)
        print("TEST 5: ResNet-18 Total FLOPs (vs known value)")
        print("-" * 80)

        model = models.resnet18(weights=None)
        model.eval()
        input_tensor = torch.randn(1, 3, 224, 224)

        fx_graph = symbolic_trace(model)
        ShapeProp(fx_graph).propagate(input_tensor)
        partitioner = GraphPartitioner()
        report = partitioner.partition(fx_graph)

        actual_flops = report.total_flops / 1e9  # Convert to GFLOPs

        # Published values for ResNet-18 (224x224) are around 3.6-4.5 GFLOPs
        # depending on whether you count bias adds, etc.
        expected_min = 3.5
        expected_max = 4.6

        print(f"\nPublished range: {expected_min:.1f} - {expected_max:.1f} GFLOPs")
        print(f"Actual FLOPs:    {actual_flops:.2f} GFLOPs")

        if expected_min <= actual_flops <= expected_max:
            print("\nRESULT: PASS (within published range)")
            self.tests_passed += 1
        else:
            print("\nRESULT: FAIL (outside published range)")
            self.tests_failed += 1

    def test_ai_ranges(self):
        """Test that different operation types have expected AI ranges"""
        print("\n" + "-" * 80)
        print("TEST 6: AI Ranges by Operation Type")
        print("-" * 80)

        model = models.resnet18(weights=None)
        model.eval()
        input_tensor = torch.randn(1, 3, 224, 224)

        fx_graph = symbolic_trace(model)
        ShapeProp(fx_graph).propagate(input_tensor)
        partitioner = GraphPartitioner()
        report = partitioner.partition(fx_graph)

        # Collect AI by operation type
        ai_by_type = {}
        for sg in report.subgraphs:
            op_type = sg.operation_type.value
            if op_type not in ai_by_type:
                ai_by_type[op_type] = []
            ai_by_type[op_type].append(sg.arithmetic_intensity)

        # Expected ranges
        expected_ranges = {
            'conv2d': (50, 500),          # Compute-bound
            'conv2d_pointwise': (5, 50),  # Balanced to memory-bound
            'batchnorm': (0.3, 2.0),      # Memory-bound
            'relu': (0.05, 0.5),          # Bandwidth-bound
            'linear': (0.5, 200),         # Depends on size (small ones can be very low AI)
        }

        print("\nOperation Type AI Ranges:")
        print(f"{'Type':<20} {'Min AI':<10} {'Max AI':<10} {'Avg AI':<10} {'Expected':<20} {'Status':<10}")
        print("-" * 90)

        all_pass = True
        for op_type, ai_values in sorted(ai_by_type.items()):
            min_ai = min(ai_values)
            max_ai = max(ai_values)
            avg_ai = sum(ai_values) / len(ai_values)

            if op_type in expected_ranges:
                exp_min, exp_max = expected_ranges[op_type]
                expected_str = f"{exp_min:.1f} - {exp_max:.1f}"

                # Check if average is in expected range (allow some outliers)
                in_range = exp_min * 0.5 <= avg_ai <= exp_max * 2.0
                status = "PASS" if in_range else "FAIL"
                if not in_range:
                    all_pass = False
            else:
                expected_str = "N/A"
                status = "SKIP"

            print(f"{op_type:<20} {min_ai:<10.2f} {max_ai:<10.2f} {avg_ai:<10.2f} {expected_str:<20} {status:<10}")

        if all_pass:
            print("\nRESULT: PASS (all operation types in expected ranges)")
            self.tests_passed += 1
        else:
            print("\nRESULT: FAIL (some operation types out of range)")
            self.tests_failed += 1

    def test_fusion_improves_ai(self):
        """Test that fusion improves arithmetic intensity"""
        print("\n" + "-" * 80)
        print("TEST 7: Fusion Improves Arithmetic Intensity")
        print("-" * 80)

        model = models.resnet18(weights=None)
        model.eval()
        input_tensor = torch.randn(1, 3, 224, 224)

        fx_graph = symbolic_trace(model)
        ShapeProp(fx_graph).propagate(input_tensor)

        # Unfused
        unfused_partitioner = GraphPartitioner()
        unfused_report = unfused_partitioner.partition(fx_graph)
        unfused_ai = unfused_report.average_arithmetic_intensity

        # Fused
        fused_partitioner = FusionBasedPartitioner()
        fused_report = fused_partitioner.partition(fx_graph)

        # Calculate average AI for fused
        if fused_report.fused_subgraphs:
            fused_ai = sum(sg.arithmetic_intensity for sg in fused_report.fused_subgraphs) / len(fused_report.fused_subgraphs)
        else:
            fused_ai = 0

        print(f"\nUnfused AI: {unfused_ai:.2f} FLOPs/byte")
        print(f"Fused AI:   {fused_ai:.2f} FLOPs/byte")
        print(f"Improvement: {(fused_ai / unfused_ai - 1) * 100:.1f}%")

        # Fusion should improve AI by reducing memory traffic
        if fused_ai > unfused_ai * 1.1:  # At least 10% improvement
            print("\nRESULT: PASS (fusion improves AI)")
            self.tests_passed += 1
        else:
            print("\nRESULT: FAIL (fusion does not improve AI enough)")
            self.tests_failed += 1


def main():
    validator = ArithmeticIntensityValidator()
    return validator.run_all_tests()


if __name__ == "__main__":
    sys.exit(main())
