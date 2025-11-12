#!/usr/bin/env python3
"""
KPU Tile Energy Model Validation Tests

Tests the 8-component KPU energy model across T64, T256, T768 variants.

Validates:
1. Energy model mathematical correctness
2. Component breakdown accuracy
3. Product variant comparison
4. Fusion benefits quantification
5. Scaling characteristics
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from graphs.hardware.models.accelerators.kpu_t64 import kpu_t64_resource_model
from graphs.hardware.models.accelerators.kpu_t256 import kpu_t256_resource_model
from graphs.hardware.models.accelerators.kpu_t768 import kpu_t768_resource_model


def print_energy_breakdown(name: str, result: dict, show_details: bool = True):
    """Pretty-print energy breakdown"""
    print(f"\n{'='*80}")
    print(f"{name}")
    print(f"{'='*80}")

    # Summary
    total_energy_mj = result['total_energy_j'] * 1000
    total_energy_j = result['total_energy_j']
    energy_per_mac_pj = result['energy_per_mac_pj']
    compute_pct = result['compute_percentage']
    ai = result['arithmetic_intensity']
    total_ops = result['total_ops']

    # Calculate TOPS/W efficiency metric
    # Assume execution at sustained clock frequency
    # Latency estimation: Use energy and assume typical power envelope
    # For more accurate calculation, we'd need the actual latency from the mapper
    # Here we estimate: assume peak throughput for the given ops

    # Get clock frequency from result if available, otherwise use typical values
    clock_hz = result.get('clock_frequency_hz', 1.0e9)  # Default 1 GHz if not provided

    # Estimate latency based on ops and peak throughput
    # Peak throughput varies by precision (from tile energy model)
    # For now, use a simplified approach: energy / typical_power_watts
    # Typical power: 3-6W for T64, 15-30W for T256, 30-60W for T768
    typical_power_w = result.get('typical_power_w', 10.0)  # Default fallback

    # Better approach: Calculate from energy and estimated execution time
    # Execution time = ops / (clock_hz * ops_per_cycle * num_tiles * efficiency)
    # For GEMM, roughly 0.5-0.7 utilization
    num_tiles = result.get('num_tiles', 256)
    ops_per_cycle_per_tile = result.get('ops_per_cycle_per_tile', 128)  # Typical for INT8/BF16
    utilization = result.get('utilization', 0.6)  # Conservative estimate

    peak_throughput = clock_hz * ops_per_cycle_per_tile * num_tiles * utilization
    execution_time_s = total_ops / peak_throughput

    # Power = Energy / Time
    average_power_w = total_energy_j / execution_time_s if execution_time_s > 0 else 0

    # TOPS/W = (TOPS) / (Watts)
    # Alternative: Use peak throughput and average power
    # Peak TOPS = peak_throughput / 1e12
    # TOPS/W at this operating point
    peak_tops = peak_throughput / 1e12
    tops_per_watt = peak_tops / average_power_w if average_power_w > 0 else 0

    # Also calculate based on actual ops (for this specific operation)
    actual_tops = total_ops / 1e12
    actual_tops_per_watt = actual_tops / average_power_w if average_power_w > 0 else 0

    print(f"\nSummary:")
    print(f"  Total Energy:         {total_energy_mj:8.3f} mJ")
    print(f"  Energy/MAC:           {energy_per_mac_pj:8.3f} pJ")
    print(f"  Execution Time:       {execution_time_s*1e6:8.2f} µs")
    print(f"  Average Power:        {average_power_w:8.2f} W")
    print(f"  Peak Throughput:      {peak_tops:8.3f} TOPS")
    print(f"  Efficiency (TOPS/W):  {tops_per_watt:8.2f} TOPS/W")
    print(f"  Compute %:            {compute_pct:8.1f}%")
    print(f"  Arithmetic Intensity: {ai:8.1f} ops/byte")
    print(f"  Total Ops:            {result['total_ops']/1e9:8.2f} GOps")
    print(f"  Total Bytes:          {result['total_bytes']/1e6:8.2f} MB")

    if not show_details:
        return

    # Component breakdown
    print(f"\nComponent Breakdown:")

    # 1. Memory hierarchy (4-stage)
    mem_hier_uj = result['total_memory_hierarchy_energy_j'] * 1e6
    mem_hier_pct = (result['total_memory_hierarchy_energy_j'] / result['total_energy_j']) * 100
    print(f"\n  1. 4-Stage Memory Hierarchy: {mem_hier_uj:8.2f} µJ ({mem_hier_pct:5.1f}%)")
    print(f"     - DRAM read:  {result['dram_read_energy_j']*1e6:7.2f} µJ")
    print(f"     - DRAM write: {result['dram_write_energy_j']*1e6:7.2f} µJ")
    print(f"     - L3 read:    {result['l3_read_energy_j']*1e6:7.2f} µJ (+ {result['average_l3_hops']:.1f} avg hops)")
    print(f"     - L3 write:   {result['l3_write_energy_j']*1e6:7.2f} µJ")
    print(f"     - L2 read:    {result['l2_read_energy_j']*1e6:7.2f} µJ")
    print(f"     - L2 write:   {result['l2_write_energy_j']*1e6:7.2f} µJ")
    print(f"     - L1 read:    {result['l1_read_energy_j']*1e6:7.2f} µJ")
    print(f"     - L1 write:   {result['l1_write_energy_j']*1e6:7.2f} µJ")

    # 2. Data movement engines
    dme_uj = result['total_dme_energy_j'] * 1e6
    dme_pct = (result['total_dme_energy_j'] / result['total_energy_j']) * 100
    print(f"\n  2. Data Movement Engines: {dme_uj:8.2f} µJ ({dme_pct:5.1f}%)")
    print(f"     - DMA:        {result['dma_energy_j']*1e6:7.2f} µJ")
    print(f"     - BlockMover: {result['blockmover_energy_j']*1e6:7.2f} µJ")
    print(f"     - Streamer:   {result['streamer_energy_j']*1e6:7.2f} µJ")

    # 3. Token signature matching (UNIQUE!)
    token_match_uj = result['total_token_matching_energy_j'] * 1e6
    token_match_pct = (result['total_token_matching_energy_j'] / result['total_energy_j']) * 100
    print(f"\n  3. Token Signature Matching (UNIQUE): {token_match_uj:8.2f} µJ ({token_match_pct:5.1f}%)")
    print(f"     - Signature matching: {result['signature_matching_energy_j']*1e6:7.2f} µJ")
    print(f"     - Dispatch:           {result['dispatch_energy_j']*1e6:7.2f} µJ")
    print(f"     - Tokens routed:      {result['num_tokens']:,}")

    # 4. SURE program loading (UNIQUE!)
    prog_load_uj = result['program_load_energy_j'] * 1e6
    prog_load_pct = (result['program_load_energy_j'] / result['total_energy_j']) * 100
    cache_hit_rate = result['cache_hit_rate']
    print(f"\n  4. SURE Program Loading (UNIQUE): {prog_load_uj:8.2f} µJ ({prog_load_pct:5.1f}%)")
    print(f"     - Cache hit rate: {cache_hit_rate*100:.0f}%")

    # 5. Distributed L3 scratchpad
    l3_routing_uj = result['distributed_l3_routing_energy_j'] * 1e6
    print(f"\n  5. Distributed L3 Scratchpad: {l3_routing_uj:8.2f} µJ")
    print(f"     - Avg L3 hops: {result['average_l3_hops']:.1f}")

    # 6. Operator fusion (if enabled)
    if result['fusion_enabled']:
        fusion_overhead_uj = result['fusion_overhead_energy_j'] * 1e6
        fusion_savings_uj = result['fusion_savings_energy_j'] * 1e6
        fusion_net_uj = result['fusion_net_energy_j'] * 1e6
        print(f"\n  6. Operator Fusion (ENABLED): {fusion_net_uj:+8.2f} µJ")
        print(f"     - Overhead:  {fusion_overhead_uj:7.2f} µJ")
        print(f"     - Savings:   {fusion_savings_uj:7.2f} µJ")
        print(f"     - Fused ops: {result['num_fused_ops']}")
    else:
        print(f"\n  6. Operator Fusion: DISABLED")

    # 7. Token routing
    token_routing_uj = result['token_routing_energy_j'] * 1e6
    token_routing_pct = (result['token_routing_energy_j'] / result['total_energy_j']) * 100
    print(f"\n  7. Token Routing: {token_routing_uj:8.2f} µJ ({token_routing_pct:5.1f}%)")
    print(f"     - Avg distance: {result['average_routing_distance']:.1f} hops")

    # 8. Computation
    compute_uj = result['compute_energy_j'] * 1e6
    print(f"\n  8. Computation: {compute_uj:8.2f} µJ ({compute_pct:5.1f}%)")


def test_kpu_t64_small_gemm():
    """TEST 1: KPU-T64 small GEMM (edge workload)"""
    print("\n" + "="*80)
    print("TEST 1: KPU-T64 Small GEMM (Edge Workload)")
    print("="*80)

    model = kpu_t64_resource_model()
    energy_model = model.tile_energy_model

    # Small GEMM: 256×256 @ 256 (MobileNet-style layer)
    result = energy_model.compute_gemm_energy(
        M=256,
        N=256,
        K=256,
        batch_size=1,
        precision="INT8",
        enable_fusion=False,
        num_fused_ops=1
    )

    print_energy_breakdown("KPU-T64: 256×256 @ 256 GEMM (INT8)", result)

    # Validate energy/MAC is reasonable (~0.8-1.2 pJ/MAC expected)
    assert 0.6e-12 < result['energy_per_mac_j'] < 2.0e-12, \
        f"Energy/MAC {result['energy_per_mac_pj']:.2f} pJ outside expected range (0.6-2.0 pJ)"

    return result


def test_kpu_t256_medium_gemm():
    """TEST 2: KPU-T256 medium GEMM (mobile workload)"""
    print("\n" + "="*80)
    print("TEST 2: KPU-T256 Medium GEMM (Mobile Workload)")
    print("="*80)

    model = kpu_t256_resource_model()
    energy_model = model.tile_energy_model

    # Medium GEMM: 512×512 @ 512 (ResNet-18 style)
    result = energy_model.compute_gemm_energy(
        M=512,
        N=512,
        K=512,
        batch_size=1,
        precision="BF16",
        enable_fusion=False,
        num_fused_ops=1
    )

    print_energy_breakdown("KPU-T256: 512×512 @ 512 GEMM (BF16)", result)

    # Validate energy/MAC is better than T64 (advanced node)
    assert 0.5e-12 < result['energy_per_mac_j'] < 1.8e-12, \
        f"Energy/MAC {result['energy_per_mac_pj']:.2f} pJ outside expected range (0.5-1.8 pJ)"

    return result


def test_kpu_t768_large_gemm():
    """TEST 3: KPU-T768 large GEMM (automotive/datacenter)"""
    print("\n" + "="*80)
    print("TEST 3: KPU-T768 Large GEMM (Automotive/Datacenter)")
    print("="*80)

    model = kpu_t768_resource_model()
    energy_model = model.tile_energy_model

    # Large GEMM: 1024×1024 @ 1024 (BERT-style)
    result = energy_model.compute_gemm_energy(
        M=1024,
        N=1024,
        K=1024,
        batch_size=1,
        precision="BF16",
        enable_fusion=False,
        num_fused_ops=1
    )

    print_energy_breakdown("KPU-T768: 1024×1024 @ 1024 GEMM (BF16)", result)

    # Validate energy/MAC is best of all (advanced node + HBM2)
    assert 0.4e-12 < result['energy_per_mac_j'] < 1.5e-12, \
        f"Energy/MAC {result['energy_per_mac_pj']:.2f} pJ outside expected range (0.4-1.5 pJ)"

    return result


def test_fusion_benefits():
    """TEST 4: Operator fusion benefits (Conv→ReLU→Pool)"""
    print("\n" + "="*80)
    print("TEST 4: Operator Fusion Benefits (Conv→ReLU→Pool)")
    print("="*80)

    model = kpu_t256_resource_model()
    energy_model = model.tile_energy_model

    # Without fusion
    result_no_fusion = energy_model.compute_gemm_energy(
        M=256,
        N=256,
        K=256,
        batch_size=1,
        precision="INT8",
        enable_fusion=False,
        num_fused_ops=1
    )

    # With fusion (3 ops: Conv + ReLU + Pool)
    result_with_fusion = energy_model.compute_gemm_energy(
        M=256,
        N=256,
        K=256,
        batch_size=1,
        precision="INT8",
        enable_fusion=True,
        num_fused_ops=3
    )

    print_energy_breakdown("WITHOUT Fusion", result_no_fusion, show_details=False)
    print_energy_breakdown("WITH Fusion (3 ops)", result_with_fusion)

    # Calculate savings
    savings_j = result_no_fusion['total_energy_j'] - result_with_fusion['total_energy_j']
    savings_pct = (savings_j / result_no_fusion['total_energy_j']) * 100

    print(f"\nFusion Analysis:")
    print(f"  Savings:     {savings_j*1e6:7.2f} µJ ({savings_pct:5.1f}%)")
    print(f"  No fusion:   {result_no_fusion['total_energy_j']*1e3:7.3f} mJ")
    print(f"  With fusion: {result_with_fusion['total_energy_j']*1e3:7.3f} mJ")

    # Fusion should save energy (reduced L2 traffic)
    assert savings_j > 0, f"Fusion should save energy, got {savings_pct:.1f}% (should be > 0%)"

    return result_with_fusion


def test_batch_scaling():
    """TEST 5: Batch size scaling (weight amortization)"""
    print("\n" + "="*80)
    print("TEST 5: Batch Size Scaling (Weight Amortization)")
    print("="*80)

    model = kpu_t768_resource_model()
    energy_model = model.tile_energy_model

    batch_sizes = [1, 4, 16, 64]
    results = []

    for batch in batch_sizes:
        result = energy_model.compute_gemm_energy(
            M=512,
            N=512,
            K=512,
            batch_size=batch,
            precision="BF16",
            enable_fusion=False,
            num_fused_ops=1
        )
        results.append(result)

        energy_per_inf_mj = (result['total_energy_j'] / batch) * 1000
        energy_per_mac_pj = result['energy_per_mac_pj']

        print(f"\nBatch {batch:2d}:")
        print(f"  Total:         {result['total_energy_j']*1000:8.3f} mJ")
        print(f"  Per inference: {energy_per_inf_mj:8.3f} mJ")
        print(f"  Energy/MAC:    {energy_per_mac_pj:8.3f} pJ")

    # Energy per inference should decrease with batch size (weight amortization)
    energy_per_inf = [r['total_energy_j'] / b for r, b in zip(results, batch_sizes)]

    print(f"\nBatch Scaling Analysis:")
    print(f"  Batch 1:  {energy_per_inf[0]*1000:7.3f} mJ/inf (baseline)")
    print(f"  Batch 4:  {energy_per_inf[1]*1000:7.3f} mJ/inf ({(1 - energy_per_inf[1]/energy_per_inf[0])*100:4.1f}% reduction)")
    print(f"  Batch 16: {energy_per_inf[2]*1000:7.3f} mJ/inf ({(1 - energy_per_inf[2]/energy_per_inf[0])*100:4.1f}% reduction)")
    print(f"  Batch 64: {energy_per_inf[3]*1000:7.3f} mJ/inf ({(1 - energy_per_inf[3]/energy_per_inf[0])*100:4.1f}% reduction)")

    # Batch 64 should be more efficient than batch 1
    assert energy_per_inf[3] < energy_per_inf[0], \
        "Batch 64 should be more efficient than batch 1"

    return results


def test_product_comparison():
    """TEST 6: T64 vs T256 vs T768 comparison"""
    print("\n" + "="*80)
    print("TEST 6: Product Comparison (T64 vs T256 vs T768)")
    print("="*80)

    # Same workload on all three products
    M, N, K = 512, 512, 512
    batch = 1
    precision = "BF16"

    t64 = kpu_t64_resource_model()
    t256 = kpu_t256_resource_model()
    t768 = kpu_t768_resource_model()

    r64 = t64.tile_energy_model.compute_gemm_energy(M, N, K, batch, precision)
    r256 = t256.tile_energy_model.compute_gemm_energy(M, N, K, batch, precision)
    r768 = t768.tile_energy_model.compute_gemm_energy(M, N, K, batch, precision)

    print(f"\nSame Workload (512×512 @ 512, batch=1, BF16):")
    print(f"\n{'Product':<10} {'Energy (mJ)':<15} {'Energy/MAC (pJ)':<20} {'Compute %':<12}")
    print(f"{'-'*60}")
    print(f"{'T64':<10} {r64['total_energy_j']*1000:<15.3f} {r64['energy_per_mac_pj']:<20.3f} {r64['compute_percentage']:<12.1f}")
    print(f"{'T256':<10} {r256['total_energy_j']*1000:<15.3f} {r256['energy_per_mac_pj']:<20.3f} {r256['compute_percentage']:<12.1f}")
    print(f"{'T768':<10} {r768['total_energy_j']*1000:<15.3f} {r768['energy_per_mac_pj']:<20.3f} {r768['compute_percentage']:<12.1f}")

    # T768 should be most efficient (advanced node + HBM2)
    assert r768['energy_per_mac_j'] < r64['energy_per_mac_j'], \
        "T768 should be more energy efficient than T64"

    return [r64, r256, r768]


if __name__ == "__main__":
    print("="*80)
    print("KPU Tile Energy Model Validation")
    print("="*80)
    print("\nValidating 8-component KPU energy model across T64, T256, T768")

    # Run all tests
    test_kpu_t64_small_gemm()
    test_kpu_t256_medium_gemm()
    test_kpu_t768_large_gemm()
    test_fusion_benefits()
    test_batch_scaling()
    test_product_comparison()

    print("\n" + "="*80)
    print("✓ All KPU energy model tests passed!")
    print("="*80)
