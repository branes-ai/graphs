#!/usr/bin/env python
"""Validate the ARM Neoverse N2 reference mapper (issue #176).

Three jobs:

1. Confirm the mapper loads with sane per-core specs from the N2 TRM
   (2x 128-bit SVE2, 64 KiB L1D, 1 MiB private L2, BF16 + INT8).

2. Run ResNet-50 batch=1 and check the estimate is physically plausible
   (matches the Ampere validation pattern in test_ampere_ampereone.py).

3. Cross-check against the generic ``cpu_arm_resource_model`` helper: N2's
   2x SVE2 datapath should deliver ~2x the per-core FP32 of the generic
   single-NEON model at the same clock. This is the "calibration anchor"
   the issue calls for -- a real published-IP model to sanity-check the
   parameterized helper that backs Graviton3 / Altra / the AmpereOne slice.
"""

import sys
from pathlib import Path

import torch
from torchvision import models

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.graphs.estimation.unified_analyzer import UnifiedAnalyzer
from src.graphs.hardware.mappers.cpu import create_arm_neoverse_n2_mapper
from src.graphs.hardware.models.datacenter.cpu_arm import cpu_arm_resource_model
from src.graphs.hardware.resource_model import Precision


def test_resource_model_shape() -> bool:
    """Sanity-check the N2 reference resource model against published specs."""
    print("=" * 78)
    print("Test 1: Neoverse N2 resource model shape")
    print("=" * 78)

    rm = create_arm_neoverse_n2_mapper().resource_model
    int8 = rm.precision_profiles[Precision.INT8].peak_ops_per_sec
    checks = [
        ("compute_units == 1 (single-core reference)", rm.compute_units == 1),
        ("name carries 'Neoverse-N2'", "neoverse-n2" in rm.name.lower()),
        ("L1 == 128 KiB/core (64 I + 64 D)", rm.l1_cache_per_unit == 128 * 1024),
        ("L2 == 1 MiB private", rm.l2_cache_total == 1024 * 1024),
        ("no SMT (threads_per_unit == 1)", rm.threads_per_unit == 1),
        ("BF16 supported (ARMv9 / SVE2)", Precision.BF16 in rm.precision_profiles),
        ("INT8 supported (SDOT/UDOT)", Precision.INT8 in rm.precision_profiles),
        ("INT8 peak in (0, 1 TOPS) for one core", 0 < int8 < 1e12),
        ("TDP == 4 W estimate", rm.thermal_operating_points["default"].tdp_watts == 4.0),
    ]
    return _report(checks)


def test_resnet50_inference() -> bool:
    """ResNet-50 batch=1 on a single N2 core -- estimate must be plausible."""
    print("=" * 78)
    print("Test 2: ResNet-50 (batch=1, INT8) on single-core Neoverse N2")
    print("=" * 78)

    model = models.resnet50(weights=None).eval()
    inp = torch.randn(1, 3, 224, 224)
    a = UnifiedAnalyzer()
    res = a.analyze_model_with_custom_hardware(
        model=model, input_tensor=inp, model_name="resnet50",
        hardware_mapper=create_arm_neoverse_n2_mapper(),
        precision=Precision.INT8,
    )
    lat_ms = res.total_latency_ms
    tp = 1000.0 / lat_ms if lat_ms > 0 else 0.0
    print(f"  predicted latency:   {lat_ms:>10.2f} ms")
    print(f"  predicted throughput:{tp:>10.2f} inf/s")
    print()
    # One efficiency-class ARM core on ResNet-50 INT8: expect well under
    # real-time-video rates and above a floor. Wide plausibility band.
    checks = [
        ("latency in (1 ms, 100 s)", 1.0 < lat_ms < 100_000.0),
        ("throughput positive and < 1000 inf/s (one 4 W core)",
         0 < tp < 1000.0),
    ]
    return _report(checks)


def test_cross_check_vs_generic_helper() -> bool:
    """N2's 2x SVE2 should ~double the generic single-NEON FP32 per core."""
    print("=" * 78)
    print("Test 3: N2 (2x SVE2) vs generic cpu_arm_resource_model (1x NEON)")
    print("=" * 78)

    n2 = create_arm_neoverse_n2_mapper().resource_model
    # Generic helper at the same core count / clock / node.
    generic = cpu_arm_resource_model(
        num_cores=1, process_node_nm=5, scalar_freq_ghz=3.2,
        name_suffix="Neoverse-N2-generic",
    )
    n2_fp32 = n2.get_peak_ops(Precision.FP32)
    gen_fp32 = generic.get_peak_ops(Precision.FP32)
    ratio = n2_fp32 / gen_fp32 if gen_fp32 else 0.0
    print(f"  N2 FP32 peak:        {n2_fp32/1e9:>10.2f} GFLOP/s")
    print(f"  generic FP32 peak:   {gen_fp32/1e9:>10.2f} GFLOP/s")
    print(f"  ratio (N2 / generic):{ratio:>10.2f}x")
    print()
    # Scalar FP32 is identical (2 ALUs); the vector half doubles (2 SVE2 pipes
    # vs 1 NEON), so the combined ratio sits between 1x and 2x, ~1.5-1.7x.
    checks = [
        ("N2 FP32 strictly exceeds generic (extra SVE2 pipe)",
         n2_fp32 > gen_fp32),
        ("ratio in (1.3, 2.0) -- vector half doubled, scalar unchanged",
         1.3 < ratio < 2.0),
    ]
    return _report(checks)


def _report(checks) -> bool:
    all_passed = True
    for name, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
        all_passed = all_passed and passed
    print()
    return all_passed


def main() -> int:
    print()
    results = [
        test_resource_model_shape(),
        test_resnet50_inference(),
        test_cross_check_vs_generic_helper(),
    ]
    print("=" * 78)
    if all(results):
        print("All checks PASSED")
        return 0
    print(f"FAILED: {sum(1 for r in results if not r)} of {len(results)} checks")
    return 1


if __name__ == "__main__":
    sys.exit(main())
