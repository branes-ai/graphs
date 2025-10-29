#!/usr/bin/env python3
"""
Demonstration of New DVFS-Aware Performance Model with Heterogeneous Tiles

This script demonstrates the new performance modeling capabilities:
1. DVFS thermal throttling (Jetson Orin @ 15W: 3% of peak)
2. Heterogeneous tile allocation (KPU T256: 70/20/10 split)
3. Per-precision efficiency factors (different for INT8/BF16/FP32)
4. Realistic comparison: KPU beats Jetson at 40% of the power!

Key Results:
- Jetson Orin @ 15W: 170 TOPS peak → 3.1 TOPS effective (1.8% efficiency)
- KPU T64 @ 6W: 100 TOPS peak → 60 TOPS effective (60% efficiency)
- KPU delivers 19× better performance at 40% of the power!
"""
from graphs.hardware.resource_model import Precision
from graphs.hardware.models import (
    jetson_orin_agx_resource_model,   # Jetson Orin AGX model: 16 SMs, 128 CUDA cores each, 1.5 GHz base clock = 16 * 192GOPS peak = 3.072 TOPS
    jetson_thor_resource_model,
    kpu_t256_resource_model,          # KPU T256 model: 256 tiles, 16x16 = 256 cores each, 512 ops/clock at 1.5 GHz = 256 * 768GOPS peak = 196 TOPS
    kpu_t64_resource_model,           # KPU T64 model: 64 tiles, low-power edge deployment
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
        efficiency_vs_peak = (effective_tops / 170) * 100  # vs 170 TOPS dense peak

        print(f"{profile_name} Mode ({thermal_point.cooling_solution}):")
        print(f"  Base clock:      {clock.base_clock_hz/1e6:.0f} MHz")
        print(f"  Boost clock:     {clock.max_boost_clock_hz/1e6:.0f} MHz")
        print(f"  Sustained clock: {clock.sustained_clock_hz/1e6:.0f} MHz ({throttle_pct:.0f}% of boost)")
        print(f"  Peak INT8:       {peak_tops:.1f} TOPS (datasheet)")
        print(f"  Sustained INT8:  {sustained_tops:.1f} TOPS (DVFS throttled)")
        print(f"  Effective INT8:  {effective_tops:.1f} TOPS (with efficiency factors)")
        print(f"  → {efficiency_vs_peak:.1f}% of 170 TOPS peak")
        print()

    print("Key Insight:")
    print("  At 15W (realistic deployment): Only 1.8% of datasheet peak!")
    print("  Root cause: Severe DVFS throttling (39% of boost clock)")


def demo_heterogeneous_tiles():
    """Demonstrate heterogeneous tile allocation on KPU T256"""
    print_separator("DEMO 2: Heterogeneous Tile Allocation (KPU T256)")

    kpu = kpu_t256_resource_model()
    thermal_point = kpu.thermal_operating_points["30W"]   # mid-range power profile

    print("KPU T256 - Workload-Driven Silicon Allocation")
    print("-" * 60)
    print("Design Philosophy:")
    print("  1. Characterize embodied AI workloads")
    print("  2. Discover: 70% INT8, 20% BF16, 10% matmuls")
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
        efficiency = perf_spec.efficiency_factor

        print(f"\n{precision.value}:")
        print(f"  Peak:      {peak:5.1f} TOPS @ boost clock")
        print(f"  Sustained: {sustained:5.1f} TOPS @ sustained clock")
        print(f"  Effective: {effective:5.1f} TOPS (efficiency: {efficiency:.0%})")

    print("\n\nKey Insight:")
    print("  KPU achieves 60-65% efficiency (vs Jetson's 2-4%)")
    print("  Reason: No DVFS throttling + well-optimized tile allocation")


def demo_head_to_head():
    """Head-to-head comparison: 2 KPUs vs 2 Nvidia Jetsons"""
    print_separator("DEMO 3: 2×2 Comparison - KPU T64/T256 vs Jetson Orin/Thor")

    orin = jetson_orin_agx_resource_model()
    thor = jetson_thor_resource_model()
    kpu64 = kpu_t64_resource_model()
    kpu256 = kpu_t256_resource_model()

    print("Realistic Embodied AI Deployment Scenarios")
    print("-" * 90)
    print("Workload: DeepLabV3-ResNet101 @ 1024×1024 (semantic segmentation)")
    print("Requirement: 10-30 FPS (33-100ms per frame)")
    print("\nComparison organized by power tier:\n")

    # Organize configs by power tier for better comparison
    configs = [
        # Low power tier (edge/battery)
        ("KPU T64 @ 6W", kpu64, "6W", 6.0, "Edge/Battery"),
        ("KPU T64 @ 10W", kpu64, "10W", 10.0, "Edge/Battery"),
        ("Jetson Orin AGX @ 15W", orin, "15W", 15.0, "Edge/Battery"),

        # Medium power tier (balanced)
        ("KPU T256 @ 15W", kpu256, "15W", 15.0, "Balanced"),
        ("KPU T256 @ 30W", kpu256, "30W", 30.0, "Balanced"),
        ("Jetson Orin AGX @ 30W", orin, "30W", 30.0, "Balanced"),
        ("Jetson Thor @ 30W", thor, "30W", 30.0, "Balanced"),

        # High power tier (performance)
        ("KPU T256 @ 50W", kpu256, "50W", 50.0, "Performance"),
        ("Jetson Orin AGX @ 60W", orin, "60W", 60.0, "Performance"),
        ("Jetson Thor @ 60W", thor, "60W", 60.0, "Performance"),
    ]

    print(f"{'Hardware':<25} {'Power':>8} {'Peak':>10} {'Effective':>12} {'Efficiency':>12} {'$/W-TOPS':>12} {'Tier'}")
    print("-" * 110)

    baseline_cost = {"Jetson Orin AGX": 2000, "Jetson Thor": 3000, "KPU T64": 500, "KPU T256": 1200}

    current_tier = None
    for name, model, profile, power, tier in configs:
        # Print tier separator
        if tier != current_tier:
            if current_tier is not None:
                print()  # Blank line between tiers
            current_tier = tier

        thermal_point = model.thermal_operating_points[profile]
        perf_spec = thermal_point.performance_specs[Precision.INT8]

        peak = perf_spec.peak_ops_per_sec / 1e12
        effective = perf_spec.effective_ops_per_sec / 1e12
        efficiency_pct = (effective / peak) * 100

        # Extract base hardware name
        base_name = name.split('@')[0].strip()
        cost = baseline_cost.get(base_name, 1000)
        cost_efficiency = cost / (power * effective) if effective > 0 else 999

        print(f"{name:<25} {power:7.1f}W {peak:9.1f}T {effective:11.1f}T {efficiency_pct:11.1f}% ${cost_efficiency:11.2f}  {tier}")

    print("\n\n" + "="*90)
    print("DETAILED COMPARISON ANALYSIS")
    print("="*90)

    # Comparison 1: Low Power (Battery-Powered Edge)
    print("\n[1] Low Power Tier - Battery-Powered Edge Devices:")
    print("-" * 60)
    kpu64_6w = kpu64.thermal_operating_points["6W"].performance_specs[Precision.INT8]
    orin_15w = orin.thermal_operating_points["15W"].performance_specs[Precision.INT8]

    kpu64_eff = kpu64_6w.effective_ops_per_sec / 1e12
    orin_eff = orin_15w.effective_ops_per_sec / 1e12

    perf_ratio = kpu64_eff / orin_eff
    power_ratio = 6.0 / 15.0
    perf_per_watt = perf_ratio / power_ratio

    print(f"KPU T64 @ 6W vs Jetson Orin AGX @ 15W:")
    print(f"  Performance:     {kpu64_eff:.1f} vs {orin_eff:.1f} TOPS → {perf_ratio:.1f}× faster")
    print(f"  Power:           6W vs 15W → {1/power_ratio:.1f}× less power")
    print(f"  Perf/Watt:       {perf_per_watt:.1f}× better (KPU advantage)")
    print(f"  Cost:            ${baseline_cost['KPU T64']} vs ${baseline_cost['Jetson Orin AGX']} → {baseline_cost['Jetson Orin AGX']/baseline_cost['KPU T64']:.1f}× cheaper")

    # Battery life calculation
    battery_wh = 100
    orin_hours = battery_wh / 15.0
    kpu64_hours = battery_wh / 6.0
    workload_gflops = 1929e9

    orin_infs_per_sec = orin_eff * 1e12 / workload_gflops
    kpu64_infs_per_sec = kpu64_eff * 1e12 / workload_gflops

    orin_total = orin_hours * 3600 * orin_infs_per_sec
    kpu64_total = kpu64_hours * 3600 * kpu64_infs_per_sec

    print(f"\n  Battery Life ({battery_wh}Wh, {workload_gflops/1e9:.0f} GFLOPS workload):")
    print(f"    Orin: {orin_hours:.1f}h runtime → {orin_total/1000:.0f}K inferences")
    print(f"    KPU:  {kpu64_hours:.1f}h runtime → {kpu64_total/1000:.0f}K inferences")
    print(f"    → {kpu64_total/orin_total:.1f}× more inferences on same battery!")

    # Comparison 2: Medium Power (Balanced Performance)
    print("\n[2] Medium Power Tier - Balanced Deployment:")
    print("-" * 60)
    kpu256_30w = kpu256.thermal_operating_points["30W"].performance_specs[Precision.INT8]
    orin_30w = orin.thermal_operating_points["30W"].performance_specs[Precision.INT8]
    thor_30w = thor.thermal_operating_points["30W"].performance_specs[Precision.INT8]

    kpu256_eff = kpu256_30w.effective_ops_per_sec / 1e12
    orin30_eff = orin_30w.effective_ops_per_sec / 1e12
    thor30_eff = thor_30w.effective_ops_per_sec / 1e12

    print(f"KPU T256 @ 30W: {kpu256_eff:.1f} TOPS")
    print(f"Jetson Orin AGX @ 30W: {orin30_eff:.1f} TOPS → KPU is {kpu256_eff/orin30_eff:.1f}× faster")
    print(f"Jetson Thor @ 30W: {thor30_eff:.1f} TOPS → KPU is {kpu256_eff/thor30_eff:.1f}× faster")
    print(f"\nAll at same 30W power budget:")
    print(f"  KPU T256 delivers {kpu256_eff/orin30_eff:.1f}-{kpu256_eff/thor30_eff:.1f}× more throughput")

    # Comparison 3: High Power (Max Performance)
    print("\n[3] High Power Tier - Maximum Performance:")
    print("-" * 60)
    kpu256_50w = kpu256.thermal_operating_points["50W"].performance_specs[Precision.INT8]
    orin_60w = orin.thermal_operating_points["60W"].performance_specs[Precision.INT8]
    thor_60w = thor.thermal_operating_points["60W"].performance_specs[Precision.INT8]

    kpu256_50_eff = kpu256_50w.effective_ops_per_sec / 1e12
    orin60_eff = orin_60w.effective_ops_per_sec / 1e12
    thor60_eff = thor_60w.effective_ops_per_sec / 1e12

    print(f"KPU T256 @ 50W: {kpu256_50_eff:.1f} TOPS effective")
    print(f"Jetson Orin AGX @ 60W: {orin60_eff:.1f} TOPS effective")
    print(f"Jetson Thor @ 60W: {thor60_eff:.1f} TOPS effective")
    print(f"\nKPU T256 @ 50W vs Jetson Thor @ 60W:")
    print(f"  Performance:   {kpu256_50_eff:.1f} vs {thor60_eff:.1f} TOPS → {kpu256_50_eff/thor60_eff:.1f}× faster")
    print(f"  Power:         50W vs 60W → {60/50:.1f}× less power")
    print(f"  Perf/Watt:     {(kpu256_50_eff/50)/(thor60_eff/60):.1f}× better")

    print("\n" + "="*90)
    print("KEY TAKEAWAYS")
    print("="*90)
    print("\n✓ KPU T64 dominates low-power edge (6-10W):")
    print(f"    - {perf_per_watt:.1f}× better performance/watt than Jetson Orin")
    print(f"    - {baseline_cost['Jetson Orin AGX']/baseline_cost['KPU T64']:.1f}× lower cost")
    print("    - Ideal for battery-powered drones, robots, IoT devices")

    print("\n✓ KPU T256 dominates all power tiers (15W-50W):")
    print(f"    - {kpu256_eff/orin30_eff:.1f}× faster than Jetson Orin at 30W")
    print(f"    - {kpu256_eff/thor30_eff:.1f}× faster than Jetson Thor at 30W")
    print(f"    - {kpu256_50_eff/thor60_eff:.1f}× faster than Jetson Thor at 50W vs 60W")
    print("    - Ideal for autonomous vehicles, edge servers, high-throughput inference")

    print("\n✓ Root Causes of KPU Advantage:")
    print("    1. No DVFS throttling → predictable sustained performance")
    print("    2. Heterogeneous tiles → silicon optimized for workload distribution")
    print("    3. Higher efficiency_factor → 60-70% vs Nvidia's 2-20%")
    print("    4. Better thermal design → stable clocks under load")


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
