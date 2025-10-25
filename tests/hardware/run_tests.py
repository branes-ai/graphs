#!/usr/bin/env python3
"""
Simple test runner for hardware tests (doesn't require pytest).

Run all hardware tests and report results.
"""

import sys
import traceback
sys.path.insert(0, 'src')

# Import test modules
from graphs.hardware.mappers.gpu import create_h100_mapper, create_jetson_thor_mapper
from graphs.hardware.mappers.cpu import create_intel_xeon_platinum_8490h_mapper, create_amd_epyc_9654_mapper
from graphs.hardware.mappers.accelerators.tpu import create_tpu_v4_mapper, create_coral_edge_tpu_mapper
from graphs.hardware.mappers.dsp import create_qrb5165_mapper, create_ti_tda4vm_mapper
from graphs.hardware.mappers.accelerators.dpu import create_dpu_vitis_ai_mapper
from graphs.hardware.mappers.accelerators.kpu import create_kpu_t64_mapper, create_kpu_t256_mapper


class TestRunner:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def run_test(self, test_name, test_func):
        """Run a single test and track results"""
        try:
            test_func()
            self.passed += 1
            print(f"  ✓ {test_name}")
            return True
        except AssertionError as e:
            self.failed += 1
            self.errors.append((test_name, str(e)))
            print(f"  ✗ {test_name}: {e}")
            return False
        except Exception as e:
            self.failed += 1
            self.errors.append((test_name, f"Error: {e}"))
            print(f"  ✗ {test_name}: ERROR - {e}")
            traceback.print_exc()
            return False

    def print_summary(self):
        """Print test summary"""
        total = self.passed + self.failed
        print("\n" + "="*80)
        print(f"TEST SUMMARY: {self.passed}/{total} passed")
        print("="*80)

        if self.failed > 0:
            print(f"\n{self.failed} FAILED TESTS:")
            for test_name, error in self.errors:
                print(f"  - {test_name}")
                print(f"    {error}")
        else:
            print("\n✓ ALL TESTS PASSED!")


def test_idle_power_constants(runner):
    """Test IDLE_POWER_FRACTION constant across all mappers"""
    print("\n[1/5] Testing IDLE_POWER_FRACTION constants...")

    mappers = [
        ("GPU", create_h100_mapper()),
        ("TPU", create_tpu_v4_mapper()),
        ("CPU", create_intel_xeon_platinum_8490h_mapper()),
        ("DSP", create_qrb5165_mapper()),
        ("DPU", create_dpu_vitis_ai_mapper()),
        ("KPU", create_kpu_t64_mapper()),
    ]

    for name, mapper in mappers:
        runner.run_test(
            f"{name} has IDLE_POWER_FRACTION",
            lambda m=mapper, n=name: (
                assert_true(hasattr(m, 'IDLE_POWER_FRACTION'), f"{n} missing constant"),
                assert_equal(m.IDLE_POWER_FRACTION, 0.5, f"{n} wrong value")
            )
        )


def test_idle_power_methods(runner):
    """Test compute_energy_with_idle_power() method exists"""
    print("\n[2/5] Testing compute_energy_with_idle_power() methods...")

    mappers = [
        ("GPU", create_h100_mapper()),
        ("TPU", create_tpu_v4_mapper()),
        ("CPU", create_intel_xeon_platinum_8490h_mapper()),
        ("DSP", create_qrb5165_mapper()),
        ("DPU", create_dpu_vitis_ai_mapper()),
        ("KPU", create_kpu_t64_mapper()),
    ]

    for name, mapper in mappers:
        runner.run_test(
            f"{name} has compute_energy_with_idle_power()",
            lambda m=mapper, n=name: (
                assert_true(hasattr(m, 'compute_energy_with_idle_power'), f"{n} missing method"),
                assert_true(callable(m.compute_energy_with_idle_power), f"{n} not callable")
            )
        )


def test_idle_power_calculations(runner):
    """Test idle power calculations are correct"""
    print("\n[3/5] Testing idle power calculations...")

    # Test datacenter (TPU v4 @ 350W)
    def test_datacenter_idle():
        mapper = create_tpu_v4_mapper()
        latency = 1.0
        dynamic_energy = 0.001
        total_energy, avg_power = mapper.compute_energy_with_idle_power(latency, dynamic_energy)
        expected = 350.0 * 0.5 * latency + dynamic_energy
        assert abs(total_energy - expected) < 0.01, f"Expected {expected}J, got {total_energy}J"
        assert abs(avg_power - 175.0) < 1.0, f"Expected ~175W, got {avg_power}W"

    runner.run_test("Datacenter idle power (TPU v4)", test_datacenter_idle)

    # Test edge (KPU-T64 @ 6W)
    def test_edge_idle():
        mapper = create_kpu_t64_mapper()
        latency = 0.01
        dynamic_energy = 0.0001
        total_energy, avg_power = mapper.compute_energy_with_idle_power(latency, dynamic_energy)
        expected = 6.0 * 0.5 * latency + dynamic_energy
        assert abs(total_energy - expected) < 0.001, f"Expected {expected}J, got {total_energy}J"
        assert abs(avg_power - 3.0) < 0.1, f"Expected ~3W, got {avg_power}W"

    runner.run_test("Edge idle power (KPU-T64)", test_edge_idle)

    # Test idle dominates at low utilization
    def test_idle_dominates():
        mapper = create_intel_xeon_platinum_8490h_mapper()
        latency = 0.010
        dynamic_energy = 0.1
        total_energy, _ = mapper.compute_energy_with_idle_power(latency, dynamic_energy)
        idle_energy = 350.0 * 0.5 * latency
        idle_fraction = idle_energy / total_energy
        assert idle_fraction > 0.9, f"Idle should dominate (>90%), got {idle_fraction*100:.1f}%"

    runner.run_test("Idle dominates low utilization", test_idle_dominates)


def test_thermal_profiles(runner):
    """Test thermal operating points exist"""
    print("\n[4/5] Testing thermal operating points...")

    mappers = [
        ("H100", create_h100_mapper()),
        ("Jetson Thor", create_jetson_thor_mapper()),
        ("TPU v4", create_tpu_v4_mapper()),
        ("Intel Xeon", create_intel_xeon_platinum_8490h_mapper()),
        ("QRB5165", create_qrb5165_mapper()),
        ("DPU", create_dpu_vitis_ai_mapper()),
        ("KPU-T64", create_kpu_t64_mapper()),
    ]

    for name, mapper in mappers:
        def test_has_thermal(m=mapper, n=name):
            assert m.resource_model.thermal_operating_points is not None, f"{n} missing thermal_operating_points"
            assert len(m.resource_model.thermal_operating_points) > 0, f"{n} has empty thermal_operating_points"

        runner.run_test(f"{name} has thermal profiles", test_has_thermal)


def test_tdp_ranges(runner):
    """Test TDP values are in reasonable ranges"""
    print("\n[5/5] Testing TDP value ranges...")

    tests = [
        ("Datacenter GPU (H100)", create_h100_mapper(), 300, 700),
        ("Edge GPU (Jetson Thor)", create_jetson_thor_mapper(), 5, 150),
        ("Datacenter TPU", create_tpu_v4_mapper(), 200, 400),
        ("Datacenter CPU (Intel)", create_intel_xeon_platinum_8490h_mapper(), 200, 600),
        ("DSP (QRB5165)", create_qrb5165_mapper(), 3, 30),
        ("DPU", create_dpu_vitis_ai_mapper(), 15, 50),
        ("KPU-T64", create_kpu_t64_mapper(), 3, 10),
    ]

    for name, mapper, min_tdp, max_tdp in tests:
        def test_tdp_range(m=mapper, n=name, mn=min_tdp, mx=max_tdp):
            thermal_points = m.resource_model.thermal_operating_points
            profile = thermal_points.get("default") or next(iter(thermal_points.values()))
            tdp = profile.tdp_watts
            assert mn <= tdp <= mx, f"{n} TDP should be {mn}-{mx}W, got {tdp}W"

        runner.run_test(f"{name} TDP in range", test_tdp_range)


# Helper assertion functions
def assert_true(condition, message="Assertion failed"):
    if not condition:
        raise AssertionError(message)

def assert_equal(a, b, message=None):
    if a != b:
        msg = message or f"Expected {b}, got {a}"
        raise AssertionError(msg)


def main():
    print("="*80)
    print("HARDWARE TEST SUITE")
    print("="*80)
    print("\nTesting power, performance, and energy metrics across all mappers...")

    runner = TestRunner()

    # Run test suites
    test_idle_power_constants(runner)
    test_idle_power_methods(runner)
    test_idle_power_calculations(runner)
    test_thermal_profiles(runner)
    test_tdp_ranges(runner)

    # Print summary
    runner.print_summary()

    # Exit with appropriate code
    sys.exit(0 if runner.failed == 0 else 1)


if __name__ == "__main__":
    main()
