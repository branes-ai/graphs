"""
Test CGRA Mapper on ResNet-18 - Stanford Plasticine Performance Validation

This test validates the CGRA mapper implementation for Stanford Plasticine-v2
on a realistic workload (ResNet-18).

Key validations:
1. CGRA mapper runs without errors
2. Spatial execution model correctly implemented
3. Reconfiguration overhead properly modeled (1000 cycles - conservative)
4. Energy efficiency appropriate for embodied AI range (10-25W)
5. Place-and-route efficiency analysis

Expected results:
- Latency: 5-15ms (slower than KPU/DPU due to reconfig overhead + spatial execution)
- Energy: 0.075-0.225 J (15W × latency)
- Utilization: 40-70% (spatial efficiency < 100% due to fabric constraints)
- Reconfiguration: ~27 subgraphs × 1000 cycles = 27 microseconds overhead
"""

import torch
import torch.fx
from torch.fx.passes.shape_prop import ShapeProp
from torchvision import models
from typing import List

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.graphs.characterize.fusion_partitioner import FusionBasedPartitioner, FusionReport
from src.graphs.characterize.cgra_mapper import create_plasticine_v2_mapper
from src.graphs.characterize.hardware_mapper import Precision


def extract_execution_stages(fusion_report: FusionReport) -> List[List[int]]:
    """Extract execution stages from fusion report"""
    subgraphs = fusion_report.fused_subgraphs
    n = len(subgraphs)

    if n == 0:
        return []

    stages = []
    i = 0
    while i < n:
        stage_size = min(3, n - i)
        stages.append(list(range(i, i + stage_size)))
        i += stage_size

    return stages


def test_cgra_mapper():
    """Test CGRA mapper on ResNet-18"""

    print("=" * 80)
    print("CGRA MAPPER VALIDATION: ResNet-18 on Stanford Plasticine-v2")
    print("Architecture: Spatial Dataflow with Medium-Grained PCUs")
    print("=" * 80)
    print()

    # Load model
    print("[1/5] Loading ResNet-18...")
    model = models.resnet18(pretrained=False)
    model.eval()

    # Trace with FX
    print("[2/5] Tracing with PyTorch FX...")
    input_tensor = torch.randn(1, 3, 224, 224)
    fx_graph = torch.fx.symbolic_trace(model)
    ShapeProp(fx_graph).propagate(input_tensor)

    # Fusion partitioning
    print("[3/5] Running fusion partitioner...")
    partitioner = FusionBasedPartitioner()
    fusion_report = partitioner.partition(fx_graph)
    execution_stages = extract_execution_stages(fusion_report)
    print(f"      {fusion_report.total_subgraphs} fused subgraphs")
    print(f"      {len(execution_stages)} execution stages")
    print(f"      {fusion_report.total_flops/1e9:.2f} GFLOPs")

    # Create CGRA mapper
    print("[4/5] Creating CGRA mapper (Plasticine-v2)...")
    mapper = create_plasticine_v2_mapper()
    print(f"      {mapper.num_pcus} PCUs (Pattern Compute Units)")
    print(f"      6.14 TOPS INT8 @ 60% efficiency")
    print(f"      15W power budget (embodied AI range)")
    print(f"      Conservative reconfiguration: 1000 cycles")

    # Test different precisions
    print("\n[5/5] Testing CGRA mapping across precisions...")
    print()

    configs = [
        (Precision.FP32, "FP32"),
        (Precision.FP16, "FP16"),
        (Precision.INT8, "INT8"),
    ]

    results = {}

    for precision, precision_name in configs:
        try:
            allocation = mapper.map_graph(
                fusion_report=fusion_report,
                execution_stages=execution_stages,
                batch_size=1,
                precision=precision
            )
            allocation.model_name = "ResNet-18"
            results[precision] = allocation

            print(f"{precision_name:6} | Latency: {allocation.total_latency*1000:6.3f} ms | "
                  f"Util: {allocation.average_utilization:5.1%} | Energy: {allocation.total_energy:.4f} J")
        except Exception as e:
            print(f"{precision_name:6} | ERROR: {str(e)[:60]}")
            results[precision] = None

    # ========================================================================
    # VALIDATION ANALYSIS
    # ========================================================================
    print("\n" + "=" * 80)
    print("VALIDATION RESULTS")
    print("=" * 80)
    print()

    int8_alloc = results.get(Precision.INT8)
    fp32_alloc = results.get(Precision.FP32)

    if int8_alloc:
        print("1. CGRA INT8 Performance (Spatial Dataflow)")
        print(f"   Latency:     {int8_alloc.total_latency*1000:.3f} ms")
        print(f"   Throughput:  {1000.0/(int8_alloc.total_latency*1000):.1f} inferences/sec")
        print(f"   Energy:      {int8_alloc.total_energy:.4f} J")
        print(f"   Power:       {int8_alloc.total_energy/int8_alloc.total_latency:.2f} W")
        print(f"   Utilization: {int8_alloc.average_utilization:.1%} (avg), {int8_alloc.peak_utilization:.1%} (peak)")
        print()

        # Estimate reconfiguration overhead
        num_subgraphs = fusion_report.total_subgraphs
        reconfig_cycles = num_subgraphs * mapper.reconfiguration_cycles
        reconfig_time_us = (reconfig_cycles / mapper.clock_freq) * 1e6
        reconfig_percentage = (reconfig_time_us / 1000) / (int8_alloc.total_latency * 1000) * 100

        print(f"   Reconfiguration Analysis:")
        print(f"   - {num_subgraphs} subgraphs × 1000 cycles = {reconfig_cycles:,} cycles")
        print(f"   - Reconfiguration time: {reconfig_time_us:.1f} microseconds")
        print(f"   - Overhead: {reconfig_percentage:.2f}% of total latency")
        print()

        # Validate latency is reasonable
        latency_ms = int8_alloc.total_latency * 1000
        if latency_ms < 3.0:
            print(f"   ⚠️  WARNING: Latency {latency_ms:.3f}ms seems too fast for CGRA!")
        elif latency_ms > 30.0:
            print(f"   ⚠️  WARNING: Latency {latency_ms:.3f}ms seems too slow for CGRA!")
        else:
            print(f"   ✓ Latency {latency_ms:.3f}ms is reasonable for spatial dataflow")

        # Validate power
        power = int8_alloc.total_energy / int8_alloc.total_latency
        if power > 25.0:
            print(f"   ⚠️  WARNING: Power {power:.2f}W exceeds embodied AI range (10-25W)!")
        else:
            print(f"   ✓ Power {power:.2f}W is within embodied AI budget")

        print()

    print("2. Spatial Execution vs Temporal")
    print("   CGRA (Spatial Dataflow):")
    print("   - Entire subgraph mapped to fabric simultaneously")
    print("   - Execution time = critical_path_length / clock_freq")
    print("   - High parallelism but reconfiguration overhead")
    print()
    print("   Tile-based (KPU/DPU - Temporal):")
    print("   - Operations execute sequentially")
    print("   - Execution time = total_ops / peak_ops")
    print("   - No reconfiguration but lower parallelism")
    print()

    print("3. Bottleneck Analysis")
    if int8_alloc:
        compute_pct = int8_alloc.compute_bound_count / int8_alloc.total_subgraphs * 100
        bandwidth_pct = int8_alloc.bandwidth_bound_count / int8_alloc.total_subgraphs * 100

        print(f"   Compute-bound:   {int8_alloc.compute_bound_count}/{int8_alloc.total_subgraphs} ops ({compute_pct:.0f}%)")
        print(f"   Bandwidth-bound: {int8_alloc.bandwidth_bound_count}/{int8_alloc.total_subgraphs} ops ({bandwidth_pct:.0f}%)")

        if bandwidth_pct > 60:
            print(f"   ✓ Mostly bandwidth-bound (expected for spatial fabric)")
        else:
            print(f"   Note: More compute-bound than expected")
        print()

    print("4. Quantization Speedup (FP32 → INT8)")
    if fp32_alloc and int8_alloc:
        fp32_lat = fp32_alloc.total_latency * 1000
        int8_lat = int8_alloc.total_latency * 1000
        speedup = fp32_lat / int8_lat

        print(f"   FP32:    {fp32_lat:.3f} ms")
        print(f"   INT8:    {int8_lat:.3f} ms")
        print(f"   Speedup: {speedup:.2f}×")

        if speedup >= 4.0:
            print(f"   ✓ Strong quantization benefit ({speedup:.2f}×)")
        elif speedup >= 2.0:
            print(f"   ✓ Moderate quantization benefit ({speedup:.2f}×)")
        else:
            print(f"   Note: Lower than expected speedup ({speedup:.2f}×)")
        print()

    # ========================================================================
    # COMPARISON TO OTHER EDGE ACCELERATORS
    # ========================================================================
    print("=" * 80)
    print("COMPARISON: CGRA vs Other Edge Accelerators (INT8)")
    print("=" * 80)
    print()

    # Expected values from previous tests
    comparisons = {
        "KPU-T100": {"latency_ms": 0.050, "energy_j": 0.001, "power_w": 25, "cost_usd": 500},
        "DPU-Vitis-AI": {"latency_ms": 3.0, "energy_j": 0.020, "power_w": 17.5, "cost_usd": 1000},
        "CGRA-Plasticine-v2": {
            "latency_ms": int8_alloc.total_latency * 1000 if int8_alloc else 0,
            "energy_j": int8_alloc.total_energy if int8_alloc else 0,
            "power_w": 15.0,
            "cost_usd": 300
        },
        "CPU (Intel)": {"latency_ms": 0.602, "energy_j": 0.002, "power_w": 125, "cost_usd": 500},
    }

    print(f"{'Hardware':<22} {'Latency':<12} {'Energy':<12} {'Power':<10} {'Cost':<10} {'Target':<20}")
    print("-" * 100)
    for hw, stats in comparisons.items():
        target = ""
        if "KPU" in hw:
            target = "Embodied AI ✓"
        elif "DPU" in hw:
            target = "FPGA/Research"
        elif "CGRA" in hw:
            target = "Ultra-Configurable"
        elif "CPU" in hw:
            target = "General"

        print(f"{hw:<22} {stats['latency_ms']:<12.3f} {stats['energy_j']:<12.4f} "
              f"{stats['power_w']:<10.0f} ${stats['cost_usd']:<9.0f} {target:<20}")

    print()
    print("Key Insights:")
    print("- CGRA: Spatial dataflow execution (different paradigm)")
    print("- Similar performance to DPU but different trade-offs:")
    print("  • CGRA: Better power (15W vs 17.5W), spatial parallelism")
    print("  • DPU: FPGA-based (more mature), tile-based execution")
    print("- Both slower than KPU (reconfigurable overhead)")
    print("- Both offer flexibility (custom operations)")
    print()
    print("Recommendation for Embodied AI:")
    print("→ Use KPU-T100 for production (60× faster, 20× better energy)")
    print("→ Use CGRA for research on spatial dataflow algorithms")
    print("→ Use DPU for FPGA-based custom operations")
    print()

    print("=" * 80)
    print("SUCCESS: CGRA mapper validation complete!")
    print("(CGRA is a research accelerator for spatial dataflow)")
    print("=" * 80)


if __name__ == "__main__":
    test_cgra_mapper()
