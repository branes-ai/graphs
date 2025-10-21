"""
Test CPU vs GPU Hardware Mapping - Phase 2 Validation

This script compares CPU and GPU hardware mapping to understand:
1. How much faster is GPU vs CPU for deep learning?
2. How do different SIMD widths (AVX-2 vs AVX-512) affect CPU performance?
3. How does quantization help on CPU (AMX, VNNI)?
4. When does memory bandwidth become the bottleneck?

Expected results for ResNet-18:
- GPU (H100) ~100-500× faster than CPU at FP32
- CPU with AVX-512 ~1.5-2× faster than AVX-2
- INT8 on CPU with AMX/VNNI: ~3-4× faster than FP32
- CPU is memory-bound for most operations (80 GB/s vs 2 TB/s)
"""

import torch
import torch.fx
from torch.fx.passes.shape_prop import ShapeProp
from torchvision import models
from typing import List

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.graphs.characterize.fusion_partitioner import FusionBasedPartitioner, FusionReport
from src.graphs.characterize.gpu_mapper import create_h100_mapper
from src.graphs.characterize.cpu_mapper import create_intel_cpu_mapper, create_amd_cpu_mapper
from src.graphs.characterize.hardware_mapper import Precision


def extract_execution_stages(fusion_report: FusionReport) -> List[List[int]]:
    """Extract execution stages (same as test_hardware_mapping.py)"""
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


def test_cpu_vs_gpu():
    """Compare CPU and GPU hardware mapping"""

    print("=" * 80)
    print("CPU vs GPU Hardware Mapping: ResNet-18")
    print("=" * 80)
    print()

    # Load model
    print("[1/4] Loading ResNet-18...")
    model = models.resnet18(pretrained=False)
    model.eval()

    # Trace with FX
    print("[2/4] Tracing with PyTorch FX...")
    input_tensor = torch.randn(1, 3, 224, 224)
    fx_graph = torch.fx.symbolic_trace(model)
    ShapeProp(fx_graph).propagate(input_tensor)

    # Fusion partitioning
    print("[3/4] Running fusion partitioner...")
    partitioner = FusionBasedPartitioner()
    fusion_report = partitioner.partition(fx_graph)
    execution_stages = extract_execution_stages(fusion_report)
    print(f"      {fusion_report.total_subgraphs} fused subgraphs")
    print(f"      {len(execution_stages)} execution stages")

    # Hardware mappers
    print("[4/4] Creating hardware mappers...")
    mappers = {
        "H100 GPU": create_h100_mapper(),
        "Intel CPU (AVX-512)": create_intel_cpu_mapper("avx512"),
        "Intel CPU (AVX-2)": create_intel_cpu_mapper("avx2"),
        "AMD CPU (AVX-2)": create_amd_cpu_mapper(),
    }

    print()
    print("=" * 80)
    print("HARDWARE COMPARISON")
    print("=" * 80)
    print()

    # Test configurations
    configs = [
        (Precision.FP32, "FP32 (Baseline)"),
        (Precision.BF16, "BF16 (Tensor Cores / AMX)"),
        (Precision.INT8, "INT8 (Quantized / VNNI)"),
    ]

    results = {}

    for precision, precision_name in configs:
        print(f"\n{'='*80}")
        print(f"Precision: {precision_name}")
        print(f"{'='*80}\n")

        for hw_name, mapper in mappers.items():
            allocation = mapper.map_graph(
                fusion_report=fusion_report,
                execution_stages=execution_stages,
                batch_size=1,
                precision=precision
            )
            allocation.model_name = "ResNet-18"

            results[(hw_name, precision)] = allocation

            print(f"\n{hw_name}:")
            print(f"  Latency: {allocation.total_latency*1000:.3f} ms")
            print(f"  Utilization: {allocation.average_utilization:.1%} (avg)")
            print(f"  Energy: {allocation.total_energy:.3f} J")
            print(f"  Compute-bound: {allocation.compute_bound_count}/{allocation.total_subgraphs}")
            print(f"  Bandwidth-bound: {allocation.bandwidth_bound_count}/{allocation.total_subgraphs}")

    # Speedup analysis
    print("\n" + "=" * 80)
    print("SPEEDUP ANALYSIS (Batch=1)")
    print("=" * 80)
    print()

    # Baseline: AMD CPU FP32
    baseline_hw = "AMD CPU (AVX-2)"
    baseline_precision = Precision.FP32
    baseline_latency = results[(baseline_hw, baseline_precision)].total_latency

    print(f"Baseline: {baseline_hw} @ {baseline_precision.value}")
    print(f"Baseline Latency: {baseline_latency*1000:.3f} ms\n")

    print(f"{'Hardware':<25} {'Precision':<25} {'Latency (ms)':<15} {'Speedup':<12} {'Energy (J)':<12}")
    print("-" * 95)

    for precision, precision_name in configs:
        for hw_name in mappers.keys():
            alloc = results[(hw_name, precision)]
            latency_ms = alloc.total_latency * 1000
            speedup = baseline_latency / alloc.total_latency
            energy = alloc.total_energy

            speedup_str = f"{speedup:.1f}×"
            if speedup < 1:
                speedup_str = f"1/{1/speedup:.1f}×"

            print(f"{hw_name:<25} {precision_name:<25} {latency_ms:<15.3f} {speedup_str:<12} {energy:<12.3f}")

    # CPU SIMD comparison
    print("\n" + "=" * 80)
    print("CPU SIMD COMPARISON (AVX-512 vs AVX-2)")
    print("=" * 80)
    print()

    print(f"{'Precision':<25} {'AVX-2 (ms)':<15} {'AVX-512 (ms)':<15} {'Speedup':<12}")
    print("-" * 70)

    for precision, precision_name in configs:
        avx2_latency = results[("Intel CPU (AVX-2)", precision)].total_latency * 1000
        avx512_latency = results[("Intel CPU (AVX-512)", precision)].total_latency * 1000
        speedup = avx2_latency / avx512_latency

        print(f"{precision_name:<25} {avx2_latency:<15.3f} {avx512_latency:<15.3f} {speedup:.2f}×")

    # GPU vs CPU comparison
    print("\n" + "=" * 80)
    print("GPU vs CPU COMPARISON")
    print("=" * 80)
    print()

    print(f"{'Precision':<25} {'CPU (ms)':<15} {'GPU (ms)':<15} {'GPU Speedup':<15}")
    print("-" * 75)

    cpu_hw = "Intel CPU (AVX-512)"
    gpu_hw = "H100 GPU"

    for precision, precision_name in configs:
        cpu_latency = results[(cpu_hw, precision)].total_latency * 1000
        gpu_latency = results[(gpu_hw, precision)].total_latency * 1000
        speedup = cpu_latency / gpu_latency

        print(f"{precision_name:<25} {cpu_latency:<15.3f} {gpu_latency:<15.3f} {speedup:.1f}×")

    # Quantization benefits
    print("\n" + "=" * 80)
    print("QUANTIZATION BENEFITS (FP32 → INT8)")
    print("=" * 80)
    print()

    print(f"{'Hardware':<25} {'FP32 (ms)':<15} {'INT8 (ms)':<15} {'Speedup':<12}")
    print("-" * 70)

    for hw_name in mappers.keys():
        fp32_latency = results[(hw_name, Precision.FP32)].total_latency * 1000
        int8_latency = results[(hw_name, Precision.INT8)].total_latency * 1000
        speedup = fp32_latency / int8_latency

        print(f"{hw_name:<25} {fp32_latency:<15.3f} {int8_latency:<15.3f} {speedup:.2f}×")

    # Energy efficiency
    print("\n" + "=" * 80)
    print("ENERGY EFFICIENCY (Joules per inference)")
    print("=" * 80)
    print()

    print(f"{'Hardware':<25} {'FP32 (J)':<15} {'BF16 (J)':<15} {'INT8 (J)':<15}")
    print("-" * 75)

    for hw_name in mappers.keys():
        fp32_energy = results[(hw_name, Precision.FP32)].total_energy
        bf16_energy = results[(hw_name, Precision.BF16)].total_energy
        int8_energy = results[(hw_name, Precision.INT8)].total_energy

        print(f"{hw_name:<25} {fp32_energy:<15.3f} {bf16_energy:<15.3f} {int8_energy:<15.3f}")

    # Key insights
    print("\n" + "=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)
    print()

    gpu_alloc = results[("H100 GPU", Precision.FP32)]
    cpu_alloc = results[("Intel CPU (AVX-512)", Precision.FP32)]
    cpu_speedup = cpu_alloc.total_latency / gpu_alloc.total_latency

    print(f"1. **GPU vs CPU Performance**:")
    print(f"   - GPU (H100) is {cpu_speedup:.0f}× faster than CPU (Intel AVX-512) at FP32")
    print(f"   - GPU utilization: {gpu_alloc.average_utilization:.1%} (limited by parallelism)")
    print(f"   - CPU utilization: {cpu_alloc.average_utilization:.1%} (limited by core count)")
    print()

    avx2_alloc = results[("Intel CPU (AVX-2)", Precision.FP32)]
    avx512_alloc = results[("Intel CPU (AVX-512)", Precision.FP32)]
    simd_speedup = avx2_alloc.total_latency / avx512_alloc.total_latency

    print(f"2. **SIMD Impact on CPU**:")
    print(f"   - AVX-512 (16-wide) is {simd_speedup:.2f}× faster than AVX-2 (8-wide)")
    print(f"   - Vectorization is crucial for CPU performance")
    print()

    cpu_fp32 = results[("Intel CPU (AVX-512)", Precision.FP32)]
    cpu_int8 = results[("Intel CPU (AVX-512)", Precision.INT8)]
    cpu_quant_speedup = cpu_fp32.total_latency / cpu_int8.total_latency

    gpu_fp32 = results[("H100 GPU", Precision.FP32)]
    gpu_int8 = results[("H100 GPU", Precision.INT8)]
    gpu_quant_speedup = gpu_fp32.total_latency / gpu_int8.total_latency

    print(f"3. **Quantization Benefits**:")
    print(f"   - CPU INT8 (with VNNI): {cpu_quant_speedup:.2f}× faster than FP32")
    print(f"   - GPU INT8: {gpu_quant_speedup:.2f}× faster than FP32")
    print(f"   - Quantization helps both, but GPU benefits more (Tensor Cores)")
    print()

    print(f"4. **Bottleneck Analysis**:")
    print(f"   - CPU: {cpu_alloc.bandwidth_bound_count}/{cpu_alloc.total_subgraphs} ops are bandwidth-bound")
    print(f"   - GPU: {gpu_alloc.bandwidth_bound_count}/{gpu_alloc.total_subgraphs} ops are bandwidth-bound")
    print(f"   - CPU's 80 GB/s DDR5 vs GPU's 2 TB/s HBM2e (25× difference!)")
    print()

    cpu_fp32_energy = results[("Intel CPU (AVX-512)", Precision.FP32)].total_energy
    gpu_fp32_energy = results[("H100 GPU", Precision.FP32)].total_energy

    print(f"5. **Energy Efficiency**:")
    print(f"   - CPU FP32: {cpu_fp32_energy:.3f} J/inference")
    print(f"   - GPU FP32: {gpu_fp32_energy:.3f} J/inference")

    if cpu_fp32_energy > gpu_fp32_energy:
        energy_ratio = cpu_fp32_energy / gpu_fp32_energy
        print(f"   - CPU uses {energy_ratio:.1f}× MORE energy than GPU (despite being slower!)")
    else:
        energy_ratio = gpu_fp32_energy / cpu_fp32_energy
        print(f"   - GPU uses {energy_ratio:.1f}× more energy than CPU")

    print()

    print(f"6. **Practical Recommendations**:")
    print(f"   - For inference: Use GPU if available ({cpu_speedup:.0f}× faster)")
    print(f"   - For edge/embedded: Use INT8 on CPU (VNNI) or dedicated accelerator")
    print(f"   - For batch processing: GPU's massive parallelism wins")
    print(f"   - For single-sample latency: CPU may be sufficient for small models")
    print()

    print("=" * 80)
    print("SUCCESS: CPU vs GPU comparison complete!")
    print("=" * 80)


if __name__ == "__main__":
    test_cpu_vs_gpu()
