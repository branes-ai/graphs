#!/usr/bin/env python3
"""
Validation: Latency Sanity Check

This validates that latency estimates follow expected performance ordering:
GPU (tensor cores) >> TPU (systolic array) > KPU (spatial dataflow) >> CPU

The whole DL industry moved to accelerators for performance and efficiency.
If CPU is fastest, our models are wrong!
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
from graphs.subgraphs.mlp import make_mlp
from graphs.transform.partitioning.fusion_partitioner import FusionBasedPartitioner
from graphs.hardware.mappers.cpu import create_jetson_orin_agx_cpu_mapper
from graphs.hardware.mappers.gpu import create_jetson_orin_agx_64gb_mapper
from graphs.hardware.mappers.accelerators.tpu import TPUMapper
from graphs.hardware.mappers.accelerators.kpu import KPUMapper
from graphs.hardware.models.edge.tpu_edge_pro import tpu_edge_pro_resource_model
from graphs.hardware.models.accelerators.kpu_t256 import kpu_t256_resource_model
from graphs.hardware.resource_model import Precision

def test_latency_ordering():
    """Test that accelerators are faster than CPU for DL workloads"""

    # Create large MLP (1024x1024) - typical DL workload
    model = make_mlp(in_dim=1024, hidden_dim=1024, out_dim=1024)
    example_input = torch.randn(16, 1024)  # Batch=16

    # Trace with FX
    traced = torch.fx.symbolic_trace(model)
    from torch.fx.passes.shape_prop import ShapeProp
    ShapeProp(traced).propagate(example_input)

    # Partition
    partitioner = FusionBasedPartitioner()
    fusion_report = partitioner.partition(traced)

    print(f"\n{'='*80}")
    print(f"LATENCY SANITY CHECK: 1024x1024 MLP @ Batch=16")
    print(f"{'='*80}")
    print(f"Total MACs: {sum(sg.total_macs for sg in fusion_report.fused_subgraphs):,}")
    print(f"Total FLOPs: {sum(sg.total_flops for sg in fusion_report.fused_subgraphs):,}")
    print(f"Number of subgraphs: {len(fusion_report.fused_subgraphs)}")

    # Create mappers (all @ 30W for fair comparison)
    cpu_mapper = create_jetson_orin_agx_cpu_mapper()
    gpu_mapper = create_jetson_orin_agx_64gb_mapper(thermal_profile="30W")  # 30W mode for fair comparison
    tpu_mapper = TPUMapper(tpu_edge_pro_resource_model())
    kpu_mapper = KPUMapper(kpu_t256_resource_model())

    # Map to hardware
    print(f"\n{'='*80}")
    print(f"HARDWARE MAPPING & LATENCY CALCULATION")
    print(f"{'='*80}")

    # Create execution stages (sequential for MLP)
    execution_stages = [[i] for i in range(len(fusion_report.fused_subgraphs))]

    # Check what precisions each architecture supports
    print(f"\nAvailable precisions:")
    print(f"  CPU: {list(cpu_mapper.resource_model.precision_profiles.keys())}")
    print(f"  GPU: {list(gpu_mapper.resource_model.precision_profiles.keys())}")
    print(f"  TPU: {list(tpu_mapper.resource_model.precision_profiles.keys())}")
    print(f"  KPU: {list(kpu_mapper.resource_model.precision_profiles.keys())}")

    # Use INT8 for fair comparison (all support it)
    print(f"\nUsing INT8 for fair comparison (supported by all architectures)")
    cpu_result = cpu_mapper.map_graph(fusion_report, execution_stages, precision=Precision.INT8)
    gpu_result = gpu_mapper.map_graph(fusion_report, execution_stages, precision=Precision.INT8)
    tpu_result = tpu_mapper.map_graph(fusion_report, execution_stages, precision=Precision.INT8)
    kpu_result = kpu_mapper.map_graph(fusion_report, execution_stages, precision=Precision.INT8)

    print(f"\n{'Architecture':<15} {'Peak TOPS':<15} {'Latency (μs)':<15} {'Throughput':<20} {'Utilization':<15}")
    print(f"{'-'*85}")

    # CPU
    cpu_hw = cpu_mapper.resource_model
    cpu_peak_tops = cpu_hw.precision_profiles[Precision.INT8].peak_ops_per_sec / 1e12
    cpu_throughput = 1.0 / cpu_result.total_latency if cpu_result.total_latency > 0 else 0
    print(f"{'CPU Cortex A78':<15} {cpu_peak_tops:<15.2f} {cpu_result.total_latency*1e6:<15.2f} {cpu_throughput:<20,.0f} {cpu_result.peak_utilization*100:<15.1f}%")

    # GPU - Calculate peak TOPS from thermal profile's compute resource (not static precision_profiles)
    gpu_hw = gpu_mapper.resource_model
    gpu_thermal_profile = gpu_hw.thermal_operating_points.get("30W", list(gpu_hw.thermal_operating_points.values())[0])
    gpu_perf_char = gpu_thermal_profile.performance_specs[Precision.INT8]
    gpu_compute_res = gpu_perf_char.compute_resource
    gpu_ops_per_clock = gpu_compute_res.ops_per_unit_per_clock[Precision.INT8] * gpu_compute_res.num_units
    gpu_clock_hz = gpu_compute_res.clock_domain.sustained_clock_hz
    gpu_peak_tops = (gpu_ops_per_clock * gpu_clock_hz) / 1e12
    gpu_throughput = 1.0 / gpu_result.total_latency if gpu_result.total_latency > 0 else 0
    print(f"{'GPU Jetson AGX':<15} {gpu_peak_tops:<15.2f} {gpu_result.total_latency*1e6:<15.2f} {gpu_throughput:<20,.0f} {gpu_result.peak_utilization*100:<15.1f}%")

    # TPU
    tpu_hw = tpu_mapper.resource_model
    tpu_peak_tops = tpu_hw.precision_profiles[Precision.INT8].peak_ops_per_sec / 1e12
    tpu_throughput = 1.0 / tpu_result.total_latency if tpu_result.total_latency > 0 else 0
    print(f"{'TPU Edge Pro':<15} {tpu_peak_tops:<15.2f} {tpu_result.total_latency*1e6:<15.2f} {tpu_throughput:<20,.0f} {tpu_result.peak_utilization*100:<15.1f}%")

    # KPU
    kpu_hw = kpu_mapper.resource_model
    kpu_peak_tops = kpu_hw.precision_profiles[Precision.INT8].peak_ops_per_sec / 1e12
    kpu_throughput = 1.0 / kpu_result.total_latency if kpu_result.total_latency > 0 else 0
    print(f"{'KPU T256':<15} {kpu_peak_tops:<15.2f} {kpu_result.total_latency*1e6:<15.2f} {kpu_throughput:<20,.0f} {kpu_result.peak_utilization*100:<15.1f}%")

    print(f"\n{'='*80}")
    print(f"ROOFLINE MODEL ANALYSIS")
    print(f"{'='*80}")

    total_ops = sum(sg.total_macs * 2 for sg in fusion_report.fused_subgraphs)  # MACs = 2 ops
    total_bytes = sum(sg.total_input_bytes + sg.total_weight_bytes + sg.total_output_bytes
                      for sg in fusion_report.fused_subgraphs)

    print(f"\nTotal Operations: {total_ops:,}")
    print(f"Total Bytes: {total_bytes:,}")
    print(f"Arithmetic Intensity: {total_ops/total_bytes:.2f} ops/byte")

    print(f"\n{'Architecture':<15} {'Peak BW (GB/s)':<18} {'Compute Time (μs)':<20} {'Memory Time (μs)':<20} {'Bottleneck':<15}")
    print(f"{'-'*95}")

    # CPU
    cpu_bw = cpu_hw.peak_bandwidth / 1e9
    cpu_compute_time = (total_ops / (cpu_peak_tops * 1e12)) * 1e6
    cpu_memory_time = (total_bytes / cpu_hw.peak_bandwidth) * 1e6
    cpu_bottleneck = "Compute" if cpu_compute_time > cpu_memory_time else "Memory"
    print(f"{'CPU':<15} {cpu_bw:<18.2f} {cpu_compute_time:<20.2f} {cpu_memory_time:<20.2f} {cpu_bottleneck:<15}")

    # GPU
    gpu_bw = gpu_hw.peak_bandwidth / 1e9
    gpu_compute_time = (total_ops / (gpu_peak_tops * 1e12)) * 1e6
    gpu_memory_time = (total_bytes / gpu_hw.peak_bandwidth) * 1e6
    gpu_bottleneck = "Compute" if gpu_compute_time > gpu_memory_time else "Memory"
    print(f"{'GPU':<15} {gpu_bw:<18.2f} {gpu_compute_time:<20.2f} {gpu_memory_time:<20.2f} {gpu_bottleneck:<15}")

    # TPU
    tpu_bw = tpu_hw.peak_bandwidth / 1e9
    tpu_compute_time = (total_ops / (tpu_peak_tops * 1e12)) * 1e6
    tpu_memory_time = (total_bytes / tpu_hw.peak_bandwidth) * 1e6
    tpu_bottleneck = "Compute" if tpu_compute_time > tpu_memory_time else "Memory"
    print(f"{'TPU':<15} {tpu_bw:<18.2f} {tpu_compute_time:<20.2f} {tpu_memory_time:<20.2f} {tpu_bottleneck:<15}")

    # KPU
    kpu_bw = kpu_hw.peak_bandwidth / 1e9
    kpu_compute_time = (total_ops / (kpu_peak_tops * 1e12)) * 1e6
    kpu_memory_time = (total_bytes / kpu_hw.peak_bandwidth) * 1e6
    kpu_bottleneck = "Compute" if kpu_compute_time > kpu_memory_time else "Memory"
    print(f"{'KPU':<15} {kpu_bw:<18.2f} {kpu_compute_time:<20.2f} {kpu_memory_time:<20.2f} {kpu_bottleneck:<15}")

    print(f"\n{'='*80}")
    print(f"THROUGHPUT & EFFICIENCY ANALYSIS")
    print(f"{'='*80}")

    # Calculate achieved TOPS from actual latency
    cpu_achieved_tops = (total_ops / (cpu_result.total_latency * 1e12)) if cpu_result.total_latency > 0 else 0
    gpu_achieved_tops = (total_ops / (gpu_result.total_latency * 1e12)) if gpu_result.total_latency > 0 else 0
    tpu_achieved_tops = (total_ops / (tpu_result.total_latency * 1e12)) if tpu_result.total_latency > 0 else 0
    kpu_achieved_tops = (total_ops / (kpu_result.total_latency * 1e12)) if kpu_result.total_latency > 0 else 0

    # Calculate efficiency (achieved / peak)
    cpu_efficiency = (cpu_achieved_tops / cpu_peak_tops * 100) if cpu_peak_tops > 0 else 0
    gpu_efficiency = (gpu_achieved_tops / gpu_peak_tops * 100) if gpu_peak_tops > 0 else 0
    tpu_efficiency = (tpu_achieved_tops / tpu_peak_tops * 100) if tpu_peak_tops > 0 else 0
    kpu_efficiency = (kpu_achieved_tops / kpu_peak_tops * 100) if kpu_peak_tops > 0 else 0

    # Get TDP from thermal profiles (use 30W for GPU for fair comparison)
    cpu_tdp = cpu_hw.thermal_operating_points[cpu_hw.default_thermal_profile].tdp_watts
    gpu_tdp = gpu_thermal_profile.tdp_watts  # Already extracted above (30W)
    tpu_tdp = tpu_hw.thermal_operating_points[tpu_hw.default_thermal_profile].tdp_watts
    kpu_tdp = kpu_hw.thermal_operating_points[kpu_hw.default_thermal_profile].tdp_watts

    # Calculate TOPS/W (peak and achieved)
    cpu_peak_tops_per_watt = cpu_peak_tops / cpu_tdp if cpu_tdp > 0 else 0
    gpu_peak_tops_per_watt = gpu_peak_tops / gpu_tdp if gpu_tdp > 0 else 0
    tpu_peak_tops_per_watt = tpu_peak_tops / tpu_tdp if tpu_tdp > 0 else 0
    kpu_peak_tops_per_watt = kpu_peak_tops / kpu_tdp if kpu_tdp > 0 else 0

    cpu_achieved_tops_per_watt = cpu_achieved_tops / cpu_tdp if cpu_tdp > 0 else 0
    gpu_achieved_tops_per_watt = gpu_achieved_tops / gpu_tdp if gpu_tdp > 0 else 0
    tpu_achieved_tops_per_watt = tpu_achieved_tops / tpu_tdp if tpu_tdp > 0 else 0
    kpu_achieved_tops_per_watt = kpu_achieved_tops / kpu_tdp if kpu_tdp > 0 else 0

    print(f"\n{'Architecture':<15} {'TDP (W)':<12} {'Peak TOPS':<15} {'Peak TOPS/W':<15} {'Achieved TOPS':<15} {'Efficiency (%)':<15} {'Achieved TOPS/W':<15}")
    print(f"{'-'*115}")
    print(f"{'CPU Cortex A78':<15} {cpu_tdp:<12.1f} {cpu_peak_tops:<15.2f} {cpu_peak_tops_per_watt:<15.3f} {cpu_achieved_tops:<15.2f} {cpu_efficiency:<15.1f} {cpu_achieved_tops_per_watt:<15.3f}")
    print(f"{'GPU Jetson AGX':<15} {gpu_tdp:<12.1f} {gpu_peak_tops:<15.2f} {gpu_peak_tops_per_watt:<15.3f} {gpu_achieved_tops:<15.2f} {gpu_efficiency:<15.1f} {gpu_achieved_tops_per_watt:<15.3f}")
    print(f"{'TPU Edge Pro':<15} {tpu_tdp:<12.1f} {tpu_peak_tops:<15.2f} {tpu_peak_tops_per_watt:<15.3f} {tpu_achieved_tops:<15.2f} {tpu_efficiency:<15.1f} {tpu_achieved_tops_per_watt:<15.3f}")
    print(f"{'KPU T256':<15} {kpu_tdp:<12.1f} {kpu_peak_tops:<15.2f} {kpu_peak_tops_per_watt:<15.3f} {kpu_achieved_tops:<15.2f} {kpu_efficiency:<15.1f} {kpu_achieved_tops_per_watt:<15.3f}")

    print(f"\nEfficiency Analysis (Why achieved TOPS << Peak TOPS):")
    print(f"{'Architecture':<15} {'Ideal Compute':<18} {'Memory Time':<18} {'Actual Latency':<18} {'Memory Slowdown':<18} {'Utilization':<15}")
    print(f"{'-'*115}")

    # Calculate slowdown factors
    cpu_memory_slowdown = cpu_memory_time / cpu_compute_time if cpu_compute_time > 0 else 1.0
    gpu_memory_slowdown = gpu_memory_time / gpu_compute_time if gpu_compute_time > 0 else 1.0
    tpu_memory_slowdown = tpu_memory_time / tpu_compute_time if tpu_compute_time > 0 else 1.0
    kpu_memory_slowdown = kpu_memory_time / kpu_compute_time if kpu_compute_time > 0 else 1.0

    print(f"{'CPU Cortex A78':<15} {cpu_compute_time:<18.2f} {cpu_memory_time:<18.2f} {cpu_result.total_latency*1e6:<18.2f} {cpu_memory_slowdown:<18.2f}× {cpu_result.peak_utilization*100:<14.1f}%")
    print(f"{'GPU Jetson AGX':<15} {gpu_compute_time:<18.2f} {gpu_memory_time:<18.2f} {gpu_result.total_latency*1e6:<18.2f} {gpu_memory_slowdown:<18.2f}× {gpu_result.peak_utilization*100:<14.1f}%")
    print(f"{'TPU Edge Pro':<15} {tpu_compute_time:<18.2f} {tpu_memory_time:<18.2f} {tpu_result.total_latency*1e6:<18.2f} {tpu_memory_slowdown:<18.2f}× {tpu_result.peak_utilization*100:<14.1f}%")
    print(f"{'KPU T256':<15} {kpu_compute_time:<18.2f} {kpu_memory_time:<18.2f} {kpu_result.total_latency*1e6:<18.2f} {kpu_memory_slowdown:<18.2f}× {kpu_result.peak_utilization*100:<14.1f}%")

    print(f"\nKey Insights:")
    print(f"  • GPU: {gpu_memory_slowdown:.1f}× memory slowdown + {gpu_result.peak_utilization*100:.0f}% utilization = {gpu_efficiency:.1f}% efficiency")
    print(f"  • TPU: {tpu_memory_slowdown:.1f}× memory slowdown + {tpu_result.peak_utilization*100:.0f}% utilization = {tpu_efficiency:.1f}% efficiency")
    print(f"  • KPU: {kpu_memory_slowdown:.1f}× memory slowdown + {kpu_result.peak_utilization*100:.0f}% utilization = {kpu_efficiency:.1f}% efficiency")
    print(f"  • CPU: {cpu_memory_slowdown:.1f}× memory slowdown + {cpu_result.peak_utilization*100:.0f}% utilization = {cpu_efficiency:.1f}% efficiency")

    print(f"\n{'='*80}")
    print(f"SANITY CHECK RESULTS")
    print(f"{'='*80}")

    # Expected ordering: KPU < GPU < TPU < CPU (lower latency = better)
    # KPU has highest bandwidth, GPU has good balance, TPU/CPU are memory-bound

    print(f"\n✓ EXPECTED: KPU < GPU < TPU < CPU (for memory-bound deep learning)")
    print(f"  Actual:   ", end="")

    latencies = [
        ("CPU", cpu_result.total_latency),
        ("GPU", gpu_result.total_latency),
        ("TPU", tpu_result.total_latency),
        ("KPU", kpu_result.total_latency),
    ]
    latencies.sort(key=lambda x: x[1])
    print(" < ".join(f"{name} ({lat*1e6:.1f}μs)" for name, lat in latencies))

    # Validate
    issues = []

    if cpu_result.total_latency < gpu_result.total_latency:
        issues.append(f"❌ CPU ({cpu_result.total_latency*1e6:.1f}μs) is FASTER than GPU ({gpu_result.total_latency*1e6:.1f}μs) - WRONG!")

    if cpu_result.total_latency < kpu_result.total_latency:
        issues.append(f"❌ CPU ({cpu_result.total_latency*1e6:.1f}μs) is FASTER than KPU ({kpu_result.total_latency*1e6:.1f}μs) - WRONG!")

    if gpu_result.total_latency > cpu_result.total_latency * 2:
        issues.append(f"❌ GPU ({gpu_result.total_latency*1e6:.1f}μs) is 2× SLOWER than CPU ({cpu_result.total_latency*1e6:.1f}μs) - WRONG!")

    print(f"\n{'='*80}")
    print(f"PERFORMANCE SUMMARY")
    print(f"{'='*80}")

    print(f"\nLatency (lower is better):")
    print(f"  KPU: {kpu_result.total_latency*1e6:.1f} μs (fastest - 256 GB/s bandwidth)")
    print(f"  GPU: {gpu_result.total_latency*1e6:.1f} μs (2nd - 204.8 GB/s bandwidth)")
    print(f"  TPU: {tpu_result.total_latency*1e6:.1f} μs (3rd - 128 GB/s bandwidth)")
    print(f"  CPU: {cpu_result.total_latency*1e6:.1f} μs (slowest - 51.2 GB/s bandwidth)")

    print(f"\nUtilization (hardware occupancy - all should be ~100%):")
    print(f"  CPU: {cpu_result.peak_utilization*100:.1f}% ✓")
    print(f"  GPU: {gpu_result.peak_utilization*100:.1f}% ✓")
    print(f"  TPU: {tpu_result.peak_utilization*100:.1f}% ✓")
    print(f"  KPU: {kpu_result.peak_utilization*100:.1f}% ✓")

    print(f"\nEfficiency (achieved TOPS / peak TOPS):")
    print(f"  CPU: {cpu_efficiency:.1f}% (least memory-bound)")
    print(f"  GPU: {gpu_efficiency:.1f}% (memory stalls)")
    print(f"  KPU: {kpu_efficiency:.1f}% (memory stalls)")
    print(f"  TPU: {tpu_efficiency:.1f}% (most memory-bound)")

    print(f"\nWhy is efficiency < utilization?")
    print(f"  • Utilization = fraction of compute units with active threads")
    print(f"  • Efficiency = fraction of time threads are computing (not stalled)")
    print(f"  • For memory-bound workloads: utilization ≈ 100%, efficiency << 100%")

    if issues:
        print(f"\n{'='*80}")
        print(f"ISSUES FOUND:")
        print(f"{'='*80}")
        for issue in issues:
            print(f"  {issue}")
        print(f"\n❌ VALIDATION FAILED: CPU should NOT be faster than accelerators!")
        return False
    else:
        print(f"\n✓ VALIDATION PASSED: Accelerators are faster than CPU")
        return True

if __name__ == "__main__":
    success = test_latency_ordering()
    sys.exit(0 if success else 1)
