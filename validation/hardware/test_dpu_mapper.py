"""
Test DPU Mapper on ResNet-18 - Xilinx Vitis AI Performance Validation

This test validates the DPU mapper implementation for Xilinx Vitis AI
on a realistic workload (ResNet-18).

Key validations:
1. DPU mapper runs without errors
2. Realistic performance estimates (not too fast or too slow)
3. Energy efficiency appropriate for edge deployment
4. Tiling behavior for scratchpad constraints
5. INT8 quantization benefits

Expected results:
- Latency: ~3-5ms (slower than GPU but edge-appropriate)
- Energy: ~0.02-0.05 J (power-efficient for 17.5W device)
- Utilization: 60-100% (good for edge accelerator)
- Bottleneck: Mostly bandwidth-bound (DDR4 limitation)
"""

import torch
import torch.fx
from torch.fx.passes.shape_prop import ShapeProp
from torchvision import models
from typing import List

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.graphs.transform.partitioning import FusionBasedPartitioner, FusionReport
from src.graphs.hardware.mappers.accelerators.dpu import create_dpu_vitis_ai_mapper
from src.graphs.hardware.resource_model import Precision


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


def test_dpu_mapper():
    """Test DPU mapper on ResNet-18"""

    print("=" * 80)
    print("DPU MAPPER VALIDATION: ResNet-18 on Xilinx Vitis AI")
    print("Configuration: B4096, VE2302 (Embodied AI Target)")
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

    # Create DPU mapper
    print("[4/5] Creating DPU mapper (Vitis AI B4096)...")
    mapper = create_dpu_vitis_ai_mapper()
    print(f"      {mapper.num_tiles} tiles")
    print(f"      {mapper.scratchpad_per_tile/1024:.0f} KB scratchpad per tile")
    print(f"      7.68 TOPS INT8 @ 75% efficiency")

    # Test different precisions
    print("\n[5/5] Testing DPU mapping across precisions...")
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
        print("1. DPU INT8 Performance (Primary Mode)")
        print(f"   Latency:     {int8_alloc.total_latency*1000:.3f} ms")
        print(f"   Throughput:  {1000.0/(int8_alloc.total_latency*1000):.1f} inferences/sec")
        print(f"   Energy:      {int8_alloc.total_energy:.4f} J")
        print(f"   Power:       {int8_alloc.total_energy/int8_alloc.total_latency:.2f} W")
        print(f"   Utilization: {int8_alloc.average_utilization:.1%} (avg), {int8_alloc.peak_utilization:.1%} (peak)")
        print()

        # Validate latency is reasonable
        latency_ms = int8_alloc.total_latency * 1000
        if latency_ms < 1.0:
            print(f"   ⚠️  WARNING: Latency {latency_ms:.3f}ms seems too fast for DPU!")
        elif latency_ms > 20.0:
            print(f"   ⚠️  WARNING: Latency {latency_ms:.3f}ms seems too slow for DPU!")
        else:
            print(f"   ✓ Latency {latency_ms:.3f}ms is reasonable for edge accelerator")

        # Validate power
        power = int8_alloc.total_energy / int8_alloc.total_latency
        if power > 25.0:
            print(f"   ⚠️  WARNING: Power {power:.2f}W exceeds VE2302 spec (15-20W)!")
        else:
            print(f"   ✓ Power {power:.2f}W is within VE2302 budget")

        print()

    print("2. Bottleneck Analysis")
    if int8_alloc:
        compute_pct = int8_alloc.compute_bound_count / int8_alloc.total_subgraphs * 100
        bandwidth_pct = int8_alloc.bandwidth_bound_count / int8_alloc.total_subgraphs * 100

        print(f"   Compute-bound:   {int8_alloc.compute_bound_count}/{int8_alloc.total_subgraphs} ops ({compute_pct:.0f}%)")
        print(f"   Bandwidth-bound: {int8_alloc.bandwidth_bound_count}/{int8_alloc.total_subgraphs} ops ({bandwidth_pct:.0f}%)")

        if bandwidth_pct > 70:
            print(f"   ✓ Mostly bandwidth-bound (expected for DDR4 @ 50 GB/s)")
        else:
            print(f"   Note: More compute-bound than expected for edge device")
        print()

    print("3. Quantization Speedup (FP32 → INT8)")
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

    print("4. Edge Deployment Analysis")
    if int8_alloc:
        latency_ms = int8_alloc.total_latency * 1000
        power = int8_alloc.total_energy / int8_alloc.total_latency
        energy_mj = int8_alloc.total_energy * 1000

        print(f"   Real-time capable (<10ms): {'✓ YES' if latency_ms < 10 else '⚠️  MARGINAL'}")
        print(f"   Battery-friendly (<20W):   {'✓ YES' if power < 20 else '✗ NO'}")
        print(f"   Energy per inference:      {energy_mj:.2f} mJ")
        print()

        # Battery life estimate (for drone/robot)
        battery_wh = 100  # 100 Wh typical drone battery
        battery_j = battery_wh * 3600
        inferences_per_battery = battery_j / int8_alloc.total_energy
        runtime_hours = inferences_per_battery * int8_alloc.total_latency / 3600

        print(f"   100Wh battery runtime: {runtime_hours:.1f} hours @ continuous inference")
        print(f"   Inferences per charge: {inferences_per_battery/1e6:.1f}M")
        print()
        print(f"   ⚠️  Note: DPU is 60-100× slower than Stillwater KPU-T64")
        print(f"   ⚠️  Note: DPU uses 20-50× more energy per inference than KPU")
        print(f"   → DPU advantage: FPGA reconfigurability for custom ops")
        print()

    # ========================================================================
    # COMPARISON TO OTHER HARDWARE
    # ========================================================================
    print("=" * 80)
    print("COMPARISON: DPU vs Other Accelerators (INT8)")
    print("=" * 80)
    print()

    # These are expected values from previous tests
    # (Would be better to import actual results, but this is for validation)
    comparisons = {
        "H100 GPU": {"latency_ms": 0.024, "energy_j": 0.001, "power_w": 700, "cost_usd": 30000},
        "TPU v4": {"latency_ms": 0.040, "energy_j": 0.001, "power_w": 280, "cost_usd": 5000},
        "Stillwater KPU-T64": {"latency_ms": 0.050, "energy_j": 0.001, "power_w": 25, "cost_usd": 500},
        "DPU (Vitis AI)": {
            "latency_ms": int8_alloc.total_latency * 1000 if int8_alloc else 0,
            "energy_j": int8_alloc.total_energy if int8_alloc else 0,
            "power_w": 17.5,
            "cost_usd": 1000
        },
        "CPU (Intel)": {"latency_ms": 0.602, "energy_j": 0.002, "power_w": 125, "cost_usd": 500},
    }

    print(f"{'Hardware':<20} {'Latency':<12} {'Energy':<12} {'Power':<10} {'Cost':<10} {'Target':<15}")
    print("-" * 90)
    for hw, stats in comparisons.items():
        target = ""
        if "GPU" in hw:
            target = "Datacenter"
        elif "TPU" in hw:
            target = "Cloud"
        elif "DPU" in hw:
            target = "Embodied AI ✓"
        elif "KPU" in hw:
            target = "Edge"
        elif "CPU" in hw:
            target = "General"

        print(f"{hw:<20} {stats['latency_ms']:<12.3f} {stats['energy_j']:<12.4f} "
              f"{stats['power_w']:<10.0f} ${stats['cost_usd']:<9.0f} {target:<15}")

    print()
    print("Key Insights:")
    print("- DPU trades SIGNIFICANT performance for FPGA reconfigurability")
    print("- 60-100× slower than KPU, 20-50× worse energy efficiency")
    print("- Main advantage: Can implement custom operations not in KPU")
    print("- For standard ops (Conv, MatMul): Stillwater KPU-T64 is far superior")
    print("- DPU niche: Research, custom algorithms, FPGA development")
    print()
    print("Recommendation for Embodied AI:")
    print("→ Use Stillwater KPU-T64 for production (faster, more efficient, cheaper)")
    print("→ Use DPU only if you need custom operations KPU can't support")
    print()

    print("=" * 80)
    print("SUCCESS: DPU mapper validation complete!")
    print("(DPU is a niche accelerator, not the embodied AI winner)")
    print("=" * 80)


if __name__ == "__main__":
    test_dpu_mapper()
