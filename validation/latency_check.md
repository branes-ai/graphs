
================================================================================
LATENCY SANITY CHECK: 1024x1024 MLP @ Batch=16
================================================================================
Total MACs: 33,554,432
Total FLOPs: 67,108,864
Number of subgraphs: 2

================================================================================
HARDWARE MAPPING & LATENCY CALCULATION
================================================================================

Available precisions:
  CPU: [<Precision.INT8: 'int8'>, <Precision.FP16: 'fp16'>, <Precision.FP32: 'fp32'>]
  GPU: [<Precision.INT8: 'int8'>]
  TPU: [<Precision.FP32: 'fp32'>, <Precision.BF16: 'bf16'>, <Precision.INT8: 'int8'>]
  KPU: [<Precision.INT8: 'int8'>, <Precision.BF16: 'bf16'>, <Precision.INT4: 'int4'>]

Using INT8 for fair comparison (supported by all architectures)

Architecture    Peak TOPS       Latency (μs)    Throughput           Utilization    
-------------------------------------------------------------------------------------
CPU Cortex A78  0.84            169.12          5,913                100.0          %
GPU Jetson AGX  2.66            62.28           16,057               100.0          %
TPU Edge Pro    27.85           67.78           14,754               100.0          %
KPU T256        33.80           33.82           29,565               100.0          %

================================================================================
ROOFLINE MODEL ANALYSIS
================================================================================

Total Operations: 67,108,864
Total Bytes: 8,658,944
Arithmetic Intensity: 7.75 ops/byte

Architecture    Peak BW (GB/s)     Compute Time (μs)    Memory Time (μs)     Bottleneck     
-----------------------------------------------------------------------------------------------
CPU             51.20              79.44                169.12               Memory         
GPU             204.80             25.21                42.28                Memory         
TPU             128.00             2.41                 67.65                Memory         
KPU             256.00             1.99                 33.82                Memory         

================================================================================
THROUGHPUT & EFFICIENCY ANALYSIS
================================================================================

Architecture    TDP (W)      Peak TOPS       Peak TOPS/W     Achieved TOPS   Efficiency (%)  Achieved TOPS/W
-------------------------------------------------------------------------------------------------------------------
CPU Cortex A78  30.0         0.84            0.028           0.40            47.0            0.013          
GPU Jetson AGX  30.0         2.66            0.089           1.08            40.5            0.036          
TPU Edge Pro    30.0         27.85           0.928           0.99            3.6             0.033          
KPU T256        30.0         33.80           1.127           1.98            5.9             0.066          

Efficiency Analysis (Why achieved TOPS << Peak TOPS):
Architecture    Ideal Compute      Memory Time        Actual Latency     Memory Slowdown    Utilization    
-------------------------------------------------------------------------------------------------------------------
CPU Cortex A78  79.44              169.12             169.12             2.13              × 100.0         %
GPU Jetson AGX  25.21              42.28              62.28              1.68              × 100.0         %
TPU Edge Pro    2.41               67.65              67.78              28.08             × 100.0         %
KPU T256        1.99               33.82              33.82              17.04             × 100.0         %

Key Insights:
  • GPU: 1.7× memory slowdown + 100% utilization = 40.5% efficiency
  • TPU: 28.1× memory slowdown + 100% utilization = 3.6% efficiency
  • KPU: 17.0× memory slowdown + 100% utilization = 5.9% efficiency
  • CPU: 2.1× memory slowdown + 100% utilization = 47.0% efficiency

================================================================================
SANITY CHECK RESULTS
================================================================================

✓ EXPECTED: KPU < GPU < TPU < CPU (for memory-bound deep learning)
  Actual:   KPU (33.8μs) < GPU (62.3μs) < TPU (67.8μs) < CPU (169.1μs)

================================================================================
PERFORMANCE SUMMARY
================================================================================

Latency (lower is better):
  KPU: 33.8 μs (fastest - 256 GB/s bandwidth)
  GPU: 62.3 μs (2nd - 204.8 GB/s bandwidth)
  TPU: 67.8 μs (3rd - 128 GB/s bandwidth)
  CPU: 169.1 μs (slowest - 51.2 GB/s bandwidth)

Utilization (hardware occupancy - all should be ~100%):
  CPU: 100.0% ✓
  GPU: 100.0% ✓
  TPU: 100.0% ✓
  KPU: 100.0% ✓

Efficiency (achieved TOPS / peak TOPS):
  CPU: 47.0% (least memory-bound)
  GPU: 40.5% (memory stalls)
  KPU: 5.9% (memory stalls)
  TPU: 3.6% (most memory-bound)

Why is efficiency < utilization?
  • Utilization = fraction of compute units with active threads
  • Efficiency = fraction of time threads are computing (not stalled)
  • For memory-bound workloads: utilization ≈ 100%, efficiency << 100%
