#!/usr/bin/env python3
"""
Demonstration of New DVFS-Aware Performance Model with Heterogeneous Tiles

This script demonstrates the new performance modeling capabilities:
1. DVFS thermal throttling (Jetson Orin @ 15W: 3% of peak)
2. Heterogeneous tile allocation (KPU T100: 70/20/10 split)
3. Per-precision derates (different for INT8/BF16/FP32)
4. Realistic comparison: KPU beats Jetson at 40% of the power!

Key Results:
- Jetson Orin @ 15W: 170 TOPS peak → 3.1 TOPS effective (1.8% derate)
- KPU T100 @ 6W: 100 TOPS peak → 60 TOPS effective (60% derate)
- KPU delivers 19× better performance at 40% of the power!
"""

from graphs.hardware.resource_model import (
    jetson_orin_agx_resource_model,
    jetson_thor_resource_model,
    kpu_t100_resource_model,
    Precision,
)


def print_separator(title: str):
    """Print a fancy separator"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def demo_dvfs_throttling():
    """Demonstrate DVFS thermal throttling on Jetson Orin"""
    print_separator("DEMO 1: DVFS Thermal Throttling (Jetson Orin AGX)")

    orin = jetson_orin_agx_resource_model()

    print("NVIDIA Jetson Orin AGX - The Reality Behind the Marketing")
    print("-" * 60)
    print("Marketing claim: 275 TOPS INT8 (sparse), 170 TOPS (dense)")
    print("Customer reality: 2-4% of peak at deployable power budgets\n")

    # Show all three power profiles
    for profile_name in ["15W", "30W", "60W"]:
        thermal_point = orin.thermal_operating_points[profile_name]
        perf_spec = thermal_point.performance_specs[Precision.INT8]

        peak_tops = perf_spec.peak_ops_per_sec / 1e12
        sustained_tops = perf_spec.sustained_ops_per_sec / 1e12
        effective_tops = perf_spec.effective_ops_per_sec / 1e12

        clock = perf_spec.compute_resource.clock_domain
        throttle_pct = clock.thermal_throttle_factor * 100
        derate_vs_peak = (effective_tops / 170) * 100  # vs 170 TOPS dense peak

        print(f"{profile_name} Mode ({thermal_point.cooling_solution}):")
        print(f"  Base clock:      {clock.base_clock_hz/1e6:.0f} MHz")
        print(f"  Boost clock:     {clock.max_boost_clock_hz/1e6:.0f} MHz")
        print(f"  Sustained clock: {clock.sustained_clock_hz/1e6:.0f} MHz ({throttle_pct:.0f}% of boost)")
        print(f"  Peak INT8:       {peak_tops:.1f} TOPS (datasheet)")
        print(f"  Sustained INT8:  {sustained_tops:.1f} TOPS (DVFS throttled)")
        print(f"  Effective INT8:  {effective_tops:.1f} TOPS (empirical derate)")
        print(f"  → {derate_vs_peak:.1f}% of 170 TOPS peak")
        print()

    print("Key Insight:")
    print("  At 15W (realistic deployment): Only 1.8% of datasheet peak!")
    print("  Root cause: Severe DVFS throttling (39% of boost clock)")


def demo_heterogeneous_tiles():
    """Demonstrate heterogeneous tile allocation on KPU T100"""
    print_separator("DEMO 2: Heterogeneous Tile Allocation (KPU T100)")

    kpu = kpu_t100_resource_model()
    thermal_point = kpu.thermal_operating_points["6W"]

    print("KPU T100 - Workload-Driven Silicon Allocation")
    print("-" * 60)
    print("Design Philosophy:")
    print("  1. Characterize embodied AI workloads")
    print("  2. Discover: 70% INT8, 20% BF16, 10% large matmuls")
    print("  3. Allocate silicon to match workload distribution")
    print("  4. Result: All precisions native, no emulation!\n")

    # Get the KPU compute resource
    int8_perf = thermal_point.performance_specs[Precision.INT8]
    kpu_compute = int8_perf.compute_resource

    print("Silicon Allocation (100 tiles total):")
    print("-" * 60)

    for tile_spec in kpu_compute.tile_specializations:
        pct = (tile_spec.num_tiles / kpu_compute.total_tiles) * 100
        print(f"\n{tile_spec.tile_type}:")
        print(f"  Tiles: {tile_spec.num_tiles} ({pct:.0f}% of silicon)")
        print(f"  Array: {tile_spec.array_dimensions[0]}×{tile_spec.array_dimensions[1]}")
        print(f"  PE config: {tile_spec.pe_configuration}")
        print(f"  Ops/tile/clock:")

        for precision, ops_per_clock in tile_spec.ops_per_tile_per_clock.items():
            opt_level = tile_spec.optimization_level[precision]
            print(f"    {precision.value:6s}: {ops_per_clock:4d} ops/clock (opt: {opt_level:.0%})")

    print("\n\nPerformance Calculation:")
    print("-" * 60)

    clock_domain = kpu_compute.tile_specializations[0].clock_domain
    print(f"Clock: {clock_domain.sustained_clock_hz/1e6:.0f} MHz sustained")
    print(f"       ({clock_domain.thermal_throttle_factor*100:.0f}% of {clock_domain.max_boost_clock_hz/1e6:.0f} MHz boost)")

    for precision in [Precision.INT8, Precision.BF16, Precision.INT4]:
        if precision not in thermal_point.performance_specs:
            continue

        perf_spec = thermal_point.performance_specs[precision]
        peak = perf_spec.peak_ops_per_sec / 1e12
        sustained = perf_spec.sustained_ops_per_sec / 1e12
        effective = perf_spec.effective_ops_per_sec / 1e12
        derate = perf_spec.empirical_derate

        print(f"\n{precision.value}:")
        print(f"  Peak:      {peak:5.1f} TOPS @ boost clock")
        print(f"  Sustained: {sustained:5.1f} TOPS @ sustained clock")
        print(f"  Effective: {effective:5.1f} TOPS (derate: {derate:.0%})")

    print("\n\nKey Insight:")
    print("  KPU achieves 60-65% derate (vs Jetson's 2-4%)")
    print("  Reason: No DVFS throttling + well-optimized tile allocation")


def demo_head_to_head():
    """Head-to-head comparison: Jetson vs KPU"""
    print_separator("DEMO 3: Head-to-Head Comparison (Embodied AI Deployment)")

    orin = jetson_orin_agx_resource_model()
    thor = jetson_thor_resource_model()
    kpu = kpu_t100_resource_model()

    print("Realistic Embodied AI Deployment Scenario")
    print("-" * 60)
    print("Target: Autonomous robot with battery power constraint")
    print("Workload: DeepLabV3-ResNet101 @ 1024×1024 (semantic segmentation)")
    print("Requirement: 10-30 FPS (33-100ms per frame)\n")

    configs = [
        ("Jetson Orin AGX @ 15W", orin, "15W", 15.0),
        ("Jetson Orin AGX @ 30W", orin, "30W", 30.0),
        ("Jetson Thor @ 30W", thor, "30W", 30.0),
        ("KPU T100 @ 6W", kpu, "6W", 6.0),
    ]

    print(f"{'Hardware':<25} {'Power':>8} {'Peak':>10} {'Effective':>12} {'Derate':>10} {'$/W-TOPS':>12}")
    print("-" * 90)

    baseline_cost = {"Jetson Orin AGX": 2000, "Jetson Thor": 3000, "KPU T100": 500}

    for name, model, profile, power in configs:
        thermal_point = model.thermal_operating_points[profile]
        perf_spec = thermal_point.performance_specs[Precision.INT8]

        peak = perf_spec.peak_ops_per_sec / 1e12
        effective = perf_spec.effective_ops_per_sec / 1e12
        derate_pct = (effective / peak) * 100

        # Extract base hardware name
        base_name = name.split('@')[0].strip()
        cost = baseline_cost.get(base_name, 1000)
        cost_efficiency = cost / (power * effective) if effective > 0 else 999

        print(f"{name:<25} {power:7.1f}W {peak:9.1f}T {effective:11.1f}T {derate_pct:9.1f}% ${cost_efficiency:11.2f}")

    print("\n\nPerformance vs Power Analysis:")
    print("-" * 60)

    orin_15w_perf = orin.thermal_operating_points["15W"].performance_specs[Precision.INT8]
    kpu_perf = kpu.thermal_operating_points["6W"].performance_specs[Precision.INT8]

    orin_effective = orin_15w_perf.effective_ops_per_sec / 1e12
    kpu_effective = kpu_perf.effective_ops_per_sec / 1e12

    perf_ratio = kpu_effective / orin_effective
    power_ratio = 6.0 / 15.0
    efficiency_ratio = perf_ratio / power_ratio

    print(f"KPU vs Jetson Orin @ 15W:")
    print(f"  Performance: {kpu_effective:.1f} vs {orin_effective:.1f} TOPS → {perf_ratio:.1f}× faster")
    print(f"  Power: 6W vs 15W → {1/power_ratio:.1f}× less power")
    print(f"  Efficiency: {efficiency_ratio:.1f}× better performance per watt")
    print(f"\n  Battery Life (100 Wh battery, continuous inference):")

    # Estimate latency for 1929 GFLOP workload
    orin_latency = (1929e9 / orin_effective / 1e12)  # seconds
    kpu_latency = (1929e9 / kpu_effective / 1e12)

    orin_inferences_per_sec = 1 / orin_latency if orin_latency > 0 else 0
    kpu_inferences_per_sec = 1 / kpu_latency if kpu_latency > 0 else 0

    orin_battery_hours = (100 / 15.0)  # Wh / W
    kpu_battery_hours = (100 / 6.0)

    orin_total_inferences = orin_battery_hours * 3600 * orin_inferences_per_sec
    kpu_total_inferences = kpu_battery_hours * 3600 * kpu_inferences_per_sec

    print(f"    Orin: {orin_battery_hours:.1f} hours → {orin_total_inferences/1000:.0f}K inferences")
    print(f"    KPU:  {kpu_battery_hours:.1f} hours → {kpu_total_inferences/1000:.0f}K inferences")
    print(f"    → KPU enables {kpu_total_inferences/orin_total_inferences:.1f}× more inferences on same battery")

    print("\n\nKey Takeaway:")
    print("  ✓ KPU delivers superior performance at lower power")
    print("  ✓ Reason: Better thermal design (no DVFS throttling)")
    print("  ✓ Reason: Silicon optimized for workload (heterogeneous tiles)")
    print("  ✓ Result: Ideal for battery-powered embodied AI")


def main():
    """Run all demonstrations"""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "  NEW PERFORMANCE MODEL DEMONSTRATION".center(78) + "║")
    print("║" + "  DVFS Throttling + Heterogeneous Tiles".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "=" * 78 + "╝")

    demo_dvfs_throttling()
    demo_heterogeneous_tiles()
    demo_head_to_head()

    print("\n")
    print("=" * 80)
    print("  DEMONSTRATION COMPLETE")
    print("=" * 80)
    print("\nNext Steps:")
    print("  1. Update mappers to use thermal_operating_points.get_effective_ops()")
    print("  2. Update test_all_hardware.py to compare realistic power profiles")
    print("  3. Run full validation with DeepLabV3-ResNet101")
    print()


if __name__ == "__main__":
    main()
