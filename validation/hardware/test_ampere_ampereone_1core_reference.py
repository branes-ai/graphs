#!/usr/bin/env python
"""Validate the synthetic AmpereOne 1-core reference mapper.

This is a *reference* design, not a shipping product (Ampere only sells
full-die parts). The validation has two jobs:

1. Confirm the mapper loads with sane core / cache / bandwidth /
   precision values that match a one-core slice of an AmpereOne 192.

2. Pin the comparison ratio between the 1-core reference and the
   192-core SKU on a single batch=1 matvec, which is the workload that
   surfaced issue #175 (CPU mapper batch=1 fanout overcount). At the
   time of writing the 192-core reports ~12x the 1-core throughput on
   this workload -- almost entirely a memory-hierarchy effect (LLC
   capacity scales with cores), not a compute parallelism effect. If
   the ratio swings dramatically (e.g. back toward 192x as compute
   fanout, or down to ~1x if a per-operator cap is added), this test
   becomes the canary.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.graphs.estimation.unified_analyzer import UnifiedAnalyzer
from src.graphs.hardware.mappers.cpu import (
    create_ampere_ampereone_192_mapper,
    create_ampere_ampereone_1core_reference_mapper,
)
from src.graphs.hardware.resource_model import Precision


class Atan(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.atan(x)


def test_resource_model_shape() -> bool:
    """Sanity-check the 1-core reference's resource model."""
    print("=" * 78)
    print("Test 1: 1-core reference resource model shape")
    print("=" * 78)

    mapper = create_ampere_ampereone_1core_reference_mapper()
    rm = mapper.resource_model

    checks = [
        ("compute_units == 1", rm.compute_units == 1),
        ("name carries '1core' marker", "1core" in rm.name.lower()),
        ("L2 == 512 KB (one core's slice)", rm.l2_cache_total == 512 * 1024),
        ("peak_bandwidth == 80 GB/s (helper default)",
         rm.peak_bandwidth == 80e9),
        ("INT8 supported", Precision.INT8 in rm.precision_profiles),
        ("INT8 peak ops/sec > 0",
         rm.precision_profiles[Precision.INT8].peak_ops_per_sec > 0),
        ("INT8 peak ops/sec < 1 TOPS (one core, not the full die)",
         rm.precision_profiles[Precision.INT8].peak_ops_per_sec < 1e12),
    ]

    all_passed = True
    for name, passed in checks:
        marker = "PASS" if passed else "FAIL"
        print(f"  [{marker}] {name}")
        if not passed:
            all_passed = False
    print()
    return all_passed


def test_192_to_1_core_ratio_on_batch1_linear() -> bool:
    """Pin the 192-core / 1-core throughput ratio on batch=1 matvec.

    This is the regression that locks in issue #175's diagnosis. If the
    mapper acquires a per-operator concurrency cap, this ratio will
    drop toward 1.0; if compute fanout is removed entirely, it will
    converge on the LLC-residency ratio. Either is a real change; the
    test prints the ratio and gates only on a wide sanity band so it
    doesn't break under reasonable refinement.
    """
    print("=" * 78)
    print("Test 2: 192-core vs 1-core ratio on batch=1 Linear(2048,2048)+atan")
    print("=" * 78)

    model = nn.Sequential(nn.Linear(2048, 2048), Atan()).eval()
    inp = torch.randn(1, 2048)

    a = UnifiedAnalyzer()

    one_core = a.analyze_model_with_custom_hardware(
        model=model, input_tensor=inp, model_name="lin2k+atan",
        hardware_mapper=create_ampere_ampereone_1core_reference_mapper(),
        precision=Precision.INT8,
    )
    full_die = a.analyze_model_with_custom_hardware(
        model=model, input_tensor=inp, model_name="lin2k+atan",
        hardware_mapper=create_ampere_ampereone_192_mapper(),
        precision=Precision.INT8,
    )

    one_core_tp = 1000.0 / one_core.total_latency_ms
    full_die_tp = 1000.0 / full_die.total_latency_ms
    ratio = full_die_tp / one_core_tp

    print(f"  1-core ref throughput:   {one_core_tp:>10.0f} inf/s")
    print(f"  192-core throughput:     {full_die_tp:>10.0f} inf/s")
    print(f"  ratio (192c / 1c):       {ratio:>10.2f}x")
    print()

    # Wide sanity band -- the test exists to flag *direction* changes.
    # Today the ratio is ~12x. A per-operator cap (issue #175) should
    # pull it toward 1-3x. Naive linear compute fanout would push it
    # toward 192x. Anything in [1.0, 50] is a defensible regime; outside
    # that band, we want to know.
    checks = [
        ("1-core throughput finite and positive",
         one_core_tp > 0 and one_core_tp < 1e9),
        ("192-core throughput finite and positive",
         full_die_tp > 0 and full_die_tp < 1e9),
        ("Ratio in [1.0, 50] sanity band (today: ~12x)",
         1.0 <= ratio <= 50.0),
    ]

    all_passed = True
    for name, passed in checks:
        marker = "PASS" if passed else "FAIL"
        print(f"  [{marker}] {name}")
        if not passed:
            all_passed = False
    print()
    return all_passed


def main() -> int:
    print()
    results = [
        test_resource_model_shape(),
        test_192_to_1_core_ratio_on_batch1_linear(),
    ]
    print("=" * 78)
    if all(results):
        print("All checks PASSED")
        return 0
    print(f"FAILED: {sum(1 for r in results if not r)} of {len(results)} checks")
    return 1


if __name__ == "__main__":
    sys.exit(main())
