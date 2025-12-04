# Measuring Hardware efficiency on a DNN

(p311) stillwater@sw-21:~/dev/branes/clones/graphs$ cli/measure_model_efficiency.py --model resnet50 --device cpu --precision fp32 --warmup 3 --runs 5

================================================================================
MODEL EFFICIENCY MEASUREMENT
================================================================================

Device: CPU
Precision: fp32
Warmup runs: 3
Timed runs: 5

Measuring resnet50 (batch=1)...
  -> 0.17 TFLOPS, efficiency: 29.5%

========================================================================================================================
MODEL EFFICIENCY COMPARISON
========================================================================================================================

Model                 Batch   Prec     GFLOPs    Latency  Delivered       Peak Efficiency
                       Size                         (ms)     TFLOPS     TFLOPS
------------------------------------------------------------------------------------------------------------------------
resnet50                  1   fp32        4.1      24.57       0.17        0.6      29.5%
------------------------------------------------------------------------------------------------------------------------

Average Efficiency: 29.5%

FINDING: Low efficiency (<30%) suggests peak TFLOPS specs are not achievable
         for real workloads. This supports the hypothesis that marketing
         numbers exceed sustainable compute capacity.
(p311) stillwater@sw-21:~/dev/branes/clones/graphs$ cli/measure_model_efficiency.py --model resnet50,vit_b_16 --device cpu --precision fp32 --warmup 3 --runs 5

================================================================================
MODEL EFFICIENCY MEASUREMENT
================================================================================

Device: CPU
Precision: fp32
Warmup runs: 3
Timed runs: 5

Measuring resnet50 (batch=1)...
  -> 0.17 TFLOPS, efficiency: 18.6%
Measuring vit_b_16 (batch=1)...
  -> 0.21 TFLOPS, efficiency: 22.4%

========================================================================================================================
MODEL EFFICIENCY COMPARISON
========================================================================================================================

Model                 Batch   Prec     GFLOPs    Latency  Delivered       Peak Efficiency
                       Size                         (ms)     TFLOPS     TFLOPS
------------------------------------------------------------------------------------------------------------------------
resnet50                  1   fp32        4.1      24.40       0.17        0.9      18.6%
vit_b_16                  1   fp32       16.9      82.10       0.21        0.9      22.4%
------------------------------------------------------------------------------------------------------------------------

Average Efficiency: 20.5%

FINDING: Low efficiency (<30%) suggests peak TFLOPS specs are not achievable
         for real workloads. This supports the hypothesis that marketing
         numbers exceed sustainable compute capacity.
(p311) stillwater@sw-21:~/dev/branes/clones/graphs$ cli/measure_model_efficiency.py --model resnet50,vit_b_16 --device cpu --precision fp32 --warmup 3 --runs 5 --batch 1,4,8

================================================================================
MODEL EFFICIENCY MEASUREMENT
================================================================================

Device: CPU
Precision: fp32
Warmup runs: 3
Timed runs: 5

Measuring resnet50 (batch=1)...
  -> 0.17 TFLOPS, efficiency: 30.2%
Measuring resnet50 (batch=4)...
  -> 0.20 TFLOPS, efficiency: 36.2%
Measuring resnet50 (batch=8)...
  -> 0.21 TFLOPS, efficiency: 22.5%
Measuring vit_b_16 (batch=1)...
  -> 0.20 TFLOPS, efficiency: 22.3%
Measuring vit_b_16 (batch=4)...
  -> 0.23 TFLOPS, efficiency: 41.4%
Measuring vit_b_16 (batch=8)...
  -> 0.24 TFLOPS, efficiency: 25.6%

========================================================================================================================
MODEL EFFICIENCY COMPARISON
========================================================================================================================

Model                 Batch   Prec     GFLOPs    Latency  Delivered       Peak Efficiency
                       Size                         (ms)     TFLOPS     TFLOPS
------------------------------------------------------------------------------------------------------------------------
resnet50                  1   fp32        4.1      25.04       0.17        0.5      30.2%
resnet50                  4   fp32       16.6      81.19       0.20        0.6      36.2%
resnet50                  8   fp32       33.2     160.81       0.21        0.9      22.5%
vit_b_16                  1   fp32       16.9      82.47       0.20        0.9      22.3%
vit_b_16                  4   fp32       67.5     288.88       0.23        0.6      41.4%
vit_b_16                  8   fp32      134.9     574.05       0.24        0.9      25.6%
------------------------------------------------------------------------------------------------------------------------

Average Efficiency: 29.7%

FINDING: Low efficiency (<30%) suggests peak TFLOPS specs are not achievable
         for real workloads. This supports the hypothesis that marketing
         numbers exceed sustainable compute capacity.
(p311) stillwater@sw-21:~/dev/branes/clones/graphs$ cli/measure_model_efficiency.py --model resnet50,vit_b_16 --device cpu --precision fp32 --warmup 3 --runs 5 --batch 1,4,8

================================================================================
MODEL EFFICIENCY MEASUREMENT
================================================================================

Device: CPU
Precision: fp32
Warmup runs: 3
Timed runs: 5

Measuring peak performance (BLAS GEMM)...
Peak: 0.86 TFLOPS (cpu_blas (N=2048))

Measuring resnet50 (batch=1)...
  -> 0.13 TFLOPS, efficiency: 15.3%
Measuring resnet50 (batch=4)...
  -> 0.20 TFLOPS, efficiency: 23.7%
Measuring resnet50 (batch=8)...
  -> 0.21 TFLOPS, efficiency: 24.0%
Measuring vit_b_16 (batch=1)...
  -> 0.18 TFLOPS, efficiency: 20.9%
Measuring vit_b_16 (batch=4)...
  -> 0.23 TFLOPS, efficiency: 26.6%
Measuring vit_b_16 (batch=8)...
  -> 0.24 TFLOPS, efficiency: 27.4%

========================================================================================================================
MODEL EFFICIENCY COMPARISON
========================================================================================================================

Model                 Batch   Prec     GFLOPs    Latency  Delivered       Peak Efficiency
                       Size                         (ms)     TFLOPS     TFLOPS
------------------------------------------------------------------------------------------------------------------------
resnet50                  1   fp32        4.1      31.47       0.13        0.9      15.3%
resnet50                  4   fp32       16.6      81.32       0.20        0.9      23.7%
resnet50                  8   fp32       33.2     160.68       0.21        0.9      24.0%
vit_b_16                  1   fp32       16.9      93.61       0.18        0.9      20.9%
vit_b_16                  4   fp32       67.5     295.04       0.23        0.9      26.6%
vit_b_16                  8   fp32      134.9     572.26       0.24        0.9      27.4%
------------------------------------------------------------------------------------------------------------------------

Average Efficiency: 23.0%

FINDING: Low efficiency (<30%) suggests peak TFLOPS specs are not achievable
         for real workloads. This supports the hypothesis that marketing
         numbers exceed sustainable compute capacity.

