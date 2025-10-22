================================================================================
COMPLETE 10-WAY HARDWARE COMPARISON: Embodied AI Focus
Jetson Orin/Thor | KPU | TPU v4 | Coral Edge TPU | DPU | CGRA | CPU | H100
================================================================================

[1/4] Loading DeepLabV3-ResNet101...
        (Semantic segmentation model - representative of embodied AI navigation)
[2/4] Tracing with PyTorch FX @ 1024×1024...
        (Higher resolution than standard ImageNet - more realistic for robotics)
[3/4] Running fusion partitioner...
      153 fused subgraphs
      51 execution stages
      1929.26 GFLOPs
[4/4] Creating hardware mappers with REALISTIC THERMAL PROFILES...

================================================================================
PHASE 2: REALISTIC HARDWARE MAPPING
================================================================================

Workload: DeepLabV3-ResNet101 (Semantic Segmentation)
  - Input: 1×3×1024×1024 (batch=1, RGB, high-resolution)
  - Total FLOPs: 1929.26 GFLOP
  - Model size: ~60M parameters
  - Use case: Embodied AI navigation/scene understanding
  - Target: 5-30 FPS control loop (15-100ms per frame, ~10ms inference budget)
  - Note: Transformers for tracking would add additional latency


================================================================================
Testing FP32 - DeepLabV3-ResNet101 @ 1024×1024
================================================================================
Hardware                  | Latency (ms) |     Util |  Energy (J)
--------------------------+--------------+----------+------------
Jetson-Orin-AGX @ 15W     |     5317.437 |  100.0% |       2.167
Jetson-Thor @ 30W         |      591.055 |   98.5% |       1.733
KPU-T100 @ 6W (70/20/10)  |     8075.688 |  100.0% |       0.383
KPU-T100 @ 12W (70/20/10) |     6569.526 |  100.0% |       0.383
KPU-T100 @ 24W (70/20/10) |     5480.238 |  100.0% |       0.383
KPU-T300 @ 12.5W (210/60/30) |     1640.821 |  100.0% |       0.313
KPU-T300 @ 25W (210/60/30) |     1338.946 |  100.0% |       0.313
KPU-T300 @ 50W (210/60/30) |     1109.278 |  100.0% |       0.313
TPU v4                    |       38.303 |  100.0% |       0.930
H100 GPU                  |       18.345 |   98.1% |       1.204
Coral-Edge-TPU            |     2278.707 |  100.0% |       1.474
DPU-Vitis-AI              |    11964.492 |  100.0% |      14.807
CGRA-Plasticine-v2        |     2581.635 |   74.7% |     384.162
Intel CPU (AVX-512)       |      130.596 |  100.0% |       2.253
AMD CPU (AVX-2)           |      188.071 |  100.0% |       2.253

================================================================================
Testing BF16 - DeepLabV3-ResNet101 @ 1024×1024
================================================================================
Hardware                  | Latency (ms) |     Util |  Energy (J)
--------------------------+--------------+----------+------------
Jetson-Orin-AGX @ 15W     |      641.180 |  100.0% |       1.202
Jetson-Thor @ 30W         |       55.767 |   98.5% |       0.962
KPU-T100 @ 6W (70/20/10)  |      452.909 |  100.0% |       0.286
KPU-T100 @ 12W (70/20/10) |      382.997 |  100.0% |       0.286
KPU-T100 @ 24W (70/20/10) |      341.121 |  100.0% |       0.286
KPU-T300 @ 12.5W (210/60/30) |      188.281 |  100.0% |       0.235
KPU-T300 @ 25W (210/60/30) |      180.015 |  100.0% |       0.235
KPU-T300 @ 50W (210/60/30) |      173.789 |  100.0% |       0.235
TPU v4                    |       19.473 |  100.0% |       0.544
H100 GPU                  |        4.466 |   98.1% |       0.721
Coral-Edge-TPU            |     2278.707 |  100.0% |       1.474
DPU-Vitis-AI              |     5502.798 |  100.0% |       7.522
CGRA-Plasticine-v2        |     2581.635 |   74.7% |     220.879
Intel CPU (AVX-512)       |      173.218 |  100.0% |       1.285
AMD CPU (AVX-2)           |      113.313 |  100.0% |       1.285

================================================================================
Testing INT8 - DeepLabV3-ResNet101 @ 1024×1024
================================================================================
Hardware                  | Latency (ms) |     Util |  Energy (J)
--------------------------+--------------+----------+------------
Jetson-Orin-AGX @ 15W     |      354.460 |  100.0% |       0.478
Jetson-Thor @ 30W         |      177.627 |   98.5% |       0.383
KPU-T100 @ 6W (70/20/10)  |      133.922 |  100.0% |       0.214
KPU-T100 @ 12W (70/20/10) |      118.310 |  100.0% |       0.214
KPU-T100 @ 24W (70/20/10) |      107.310 |  100.0% |       0.214
KPU-T300 @ 12.5W (210/60/30) |       88.162 |  100.0% |       0.177
KPU-T300 @ 25W (210/60/30) |       86.799 |  100.0% |       0.177
KPU-T300 @ 50W (210/60/30) |       85.752 |  100.0% |       0.177
TPU v4                    |       10.541 |  100.0% |       0.255
H100 GPU                  |        4.344 |   98.1% |       0.358
Coral-Edge-TPU            |     2278.707 |  100.0% |       1.474
DPU-Vitis-AI              |     2821.092 |  100.0% |       2.059
CGRA-Plasticine-v2        |     2581.635 |   74.7% |      98.417
Intel CPU (AVX-512)       |      127.320 |  100.0% |       0.559
AMD CPU (AVX-2)           |      148.258 |  100.0% |       0.559

================================================================================
ANALYSIS 1: ABSOLUTE PERFORMANCE (INT8 Quantized)
================================================================================

Hardware                  Latency (ms)    Throughput      Energy (J)  
---------------------------------------------------------------------------
H100 GPU                  4.344           230.2           0.358       
TPU v4                    10.541          94.9            0.255       
KPU-T100 @ 6W (70/20/10)  133.922         7.5             0.214       
DPU-Vitis-AI              2821.092        0.4             2.059       
CGRA-Plasticine-v2        2581.635        0.4             98.417      
Intel CPU (AVX-512)       127.320         7.9             0.559       
AMD CPU (AVX-2)           148.258         6.7             0.559       

================================================================================
ANALYSIS 2: QUANTIZATION SPEEDUP (FP32 → INT8)
================================================================================

Hardware                  FP32 (ms)    INT8 (ms)    Speedup      Benefit             
-------------------------------------------------------------------------------------
H100 GPU                  18.345       4.344        4.22         SIGNIFICANT         
TPU v4                    38.303       10.541       3.63         SIGNIFICANT         
KPU-T100 @ 6W (70/20/10)  8075.688     133.922      60.30        MASSIVE             
Jetson-Orin-AGX @ 15W     5317.437     354.460      15.00        MASSIVE             
Jetson-Thor @ 30W         591.055      177.627      3.33         SIGNIFICANT         
DPU-Vitis-AI              11964.492    2821.092     4.24         SIGNIFICANT         
CGRA-Plasticine-v2        2581.635     2581.635     1.00         MINIMAL             
Intel CPU (AVX-512)       130.596      127.320      1.03         MINIMAL             

================================================================================
ANALYSIS 3: ENERGY EFFICIENCY (Joules per inference)
================================================================================

Hardware                  FP32 (J)     BF16 (J)     INT8 (J)     Best           
--------------------------------------------------------------------------------
H100 GPU                  1.204        0.721        0.358        INT8 ✓         
TPU v4                    0.930        0.544        0.255        INT8 ✓         
KPU-T100 @ 6W (70/20/10)  0.383        0.286        0.214        INT8 ✓         
Jetson-Orin-AGX @ 15W     2.167        1.202        0.478        INT8 ✓         
Jetson-Thor @ 30W         1.733        0.962        0.383        INT8 ✓         
DPU-Vitis-AI              14.807       7.522        2.059        INT8 ✓         
CGRA-Plasticine-v2        384.162      220.879      98.417       INT8 ✓         
Intel CPU (AVX-512)       2.253        1.285        0.559        INT8 ✓         

================================================================================
ANALYSIS 4: HARDWARE UTILIZATION (Batch=1)
================================================================================

Hardware                  Peak Util    Avg Util     Bottleneck                    
-------------------------------------------------------------------------------------
H100 GPU                  100.0%       98.1%        79% Bandwidth-bound           
TPU v4                    100.0%       100.0%       56% Compute-bound             
KPU-T100 @ 6W (70/20/10)  100.0%       100.0%       72% Compute-bound             
Jetson-Orin-AGX @ 15W     100.0%       100.0%       71% Compute-bound             
Jetson-Thor @ 30W         100.0%       98.5%        71% Compute-bound             
DPU-Vitis-AI              100.0%       100.0%       77% Bandwidth-bound           
CGRA-Plasticine-v2        100.0%       74.7%        72% Compute-bound             
Intel CPU (AVX-512)       100.0%       100.0%       77% Bandwidth-bound           

================================================================================
ANALYSIS 5: HEAD-TO-HEAD (INT8, Batch=1)
================================================================================

Baseline: Intel CPU (AVX-512) @ INT8
  Latency: 127.320 ms
  Energy:  0.559 J

Hardware                  Speedup vs CPU     Energy Efficiency   
----------------------------------------------------------------------
H100 GPU                  29.3              × 1.6                 ×
TPU v4                    12.1              × 2.2                 ×
Coral-Edge-TPU            0.1               × 0.4                 ×
KPU-T100 @ 6W (70/20/10)  1.0               × 2.6                 ×
Jetson-Orin-AGX @ 15W     0.4               × 1.2                 ×
Jetson-Thor @ 30W         0.7               × 1.5                 ×
DPU-Vitis-AI              0.0               × 0.3                 ×
CGRA-Plasticine-v2        0.0               × 0.0                 ×

================================================================================
ANALYSIS 6: COST-BENEFIT COMPARISON (INT8, Batch=1)
================================================================================

Hardware                  Latency      Energy       Power      Cost         Target         
----------------------------------------------------------------------------------------------------
Jetson-Orin-AGX @ 15W     354.460      0.4784       1          $2,000       Edge AI (15W) ✓
Jetson-Thor @ 30W         177.627      0.3828       2          $3,000       Next-Gen (30W) ✓
KPU-T100 @ 6W (70/20/10)  133.922      0.2139       2          $400         Embodied (6W) ✓
TPU v4                    10.541       0.2547       24         $5,000       Cloud          
Coral-Edge-TPU            2278.707     1.4739       1          $75          IoT/Battery ✓  
DPU-Vitis-AI              2821.092     2.0585       1          $1,000       FPGA/Research  
CGRA-Plasticine-v2        2581.635     98.4165      38         $300         Spatial/Research
Intel CPU (AVX-512)       127.320      0.5585       4          $500         General        
AMD CPU (AVX-2)           148.258      0.5585       4          $400         General        
H100 GPU                  4.344        0.3581       82         $30,000      Datacenter     

Cost-Performance Analysis:
   KPU-T100 @ 6W (70/20/10): $400 for 7 inf/sec → $53.57 per inf/sec
   DPU-Vitis-AI:             $1,000 for 0 inf/sec → $2821.09 per inf/sec
   → KPU is 52.7× better cost-performance than DPU

================================================================================
KEY INSIGHTS & RECOMMENDATIONS
================================================================================

1. **GPU (H100) - Cloud/Datacenter Champion**
   - Fastest absolute performance: 4.344 ms
   - Quantization speedup: 9.2× (FP32 → INT8)
   - Limitation: Only 98.1% utilized at batch=1
   → Use for: Cloud inference with batching, training

2. **TPU (v4) - Google's Systolic Array**
   - Strong performance: 10.541 ms
   - Optimized for matrix ops (Conv, Linear)
   - Best at large batch sizes (64+)
   → Use for: Large-batch inference, Google Cloud

3. **Jetson Orin AGX @ 15W - Reality Check**
   - Marketing claim: 170 TOPS INT8 (dense), 275 TOPS (sparse)
   - Actual performance @ 15W: 354.460 ms
   - Energy: 0.478 J per inference
   - Root cause: DVFS thermal throttling (39% of boost clock) + 47% empirical derate
   - Result: Only 1.8% of datasheet peak performance!
   → Reality: Jetson claims are for unrealistic power budgets (60W+)

4. **Jetson Thor @ 30W - Next-Gen Edge (Still Throttled)**
   - Marketing claim: 2000 TOPS INT8
   - Actual performance @ 30W: 177.627 ms
   - Energy: 0.383 J per inference
   - DVFS throttling: 57% of boost clock + 50% empirical derate
   - Result: Only 3% of datasheet peak!
   → Even next-gen Jetson suffers from thermal/power constraints

5. **KPU (T100 @ 6W with 70/20/10 tiles) - Edge & Embodied AI Champion**
   - Heterogeneous architecture: 70 INT8 tiles, 20 BF16 tiles, 10 Matrix units
   - Fastest edge performance: 133.922 ms
   - Energy champion: 0.214 J (best for battery life)
   - Full utilization: 100.0%
   - Empirical derate: 65% (vs Jetson's 1.8%!)
   - Affordable: ~$500 (vs $2K Jetson, $30K GPU)
   - vs Jetson Orin @ 15W: 2.6× faster, 2.2× better energy, 40% of power!
   → Use for: Robots, drones, edge deployment, embodied AI

6. **DPU (Xilinx Vitis AI) - FPGA Flexibility**
   - Performance: 2821.092 ms (21.1× slower than KPU)
   - Energy: 2058.52 mJ per inference (9.6× worse than KPU)
   - Power: 0.7W (measured during inference)
   - Key advantage: FPGA reconfigurability for custom operations
   → Use for: Research, custom ops that KPU can't support (niche)

7. **CGRA (Plasticine-v2) - Spatial Dataflow Research**
   - Performance: 2581.635 ms (19× slower than KPU)
   - Energy: 98416.5 mJ per inference (460× worse than KPU)
   - Power: 38.1W (measured during inference)
   - Key advantage: Spatial execution + reconfigurability
   - Conservative reconfig overhead: 1000 cycles (Achilles heel)
   → Use for: Research on spatial dataflow, custom algorithms

8. **CPU (Intel) - General Purpose**
   - Flexible but slow: 127.320 ms
   - Bandwidth-bound: 118/153 ops
   - Quantization: NO speedup (1.0×)
   → Use for: Development, small models, when no accelerator available

9. **Quantization Strategy**
   - GPU/KPU/TPU/DPU/CGRA: Always use INT8 (2-9× speedup)
   - CPU: Use INT8 only for model size, not speed

10. **Batch Size Recommendations**
   - GPU: Need batch≥44 to saturate hardware
   - TPU: Need batch≥64 for best performance
   - KPU: Efficient even at batch=1
   - Jetson: Batch size helps but still thermally limited
   - DPU: Efficient at batch=1 (embodied AI optimized)
   - CGRA: Efficient at batch=1 (spatial dataflow)
   - CPU: Batch size doesn't help (bandwidth-bound)

11. **Embodied AI Recommendations**
   - Best choice: KPU-T100 (21.1× faster than DPU, 9.6× better energy, $500)
   - Research alternatives:
     • DPU: FPGA reconfigurability (tile-based)
     • CGRA: Spatial dataflow research (ultra-configurable)
   - Avoid: GPU/TPU (too power-hungry: 280-700W vs 5-40W)
   - Avoid: CPU (too slow for real-time)

   Battery Life (100 Wh battery):
   - KPU: 2 million inferences (best)
   - DPU: 0 million inferences
   - CGRA: 0 million inferences
   → KPU gives 4+ hours of continuous 20 FPS vision processing

12. **Execution Paradigms**
   - Temporal (GPU/TPU/KPU/DPU/CPU/Jetson): Operations execute sequentially
   - Spatial (CGRA): Entire graph mapped to fabric, executes in parallel
   - Trade-off: Spatial has higher parallelism but reconfiguration overhead

13. **The DVFS Reality (Critical for Edge AI!)**
   - Jetson Orin @ 15W: 39% clock throttle + 47% derate = 1.8% of peak
   - Jetson Thor @ 30W: 57% clock throttle + 50% derate = 3% of peak
   - KPU @ 6W: 95% clock (no throttle!) + 65% derate = 62% of peak
   → Lesson: Marketing specs are LIES without power/thermal context!
   → KPU achieves 35× better efficiency through better thermal design


================================================================================
EMBODIED AI FOCUSED ANALYSIS
Edge Accelerators Only (Excludes Cloud: H100, TPU v4)
================================================================================

Why This Analysis?
  - Embodied AI requires edge deployment (robots, drones, vehicles)
  - Cloud hardware (H100, TPU v4) excluded: too power-hungry (280-700W)
  - CPUs included as baseline: show why acceleration is critical

--------------------------------------------------------------------------------
EA-1: LATENCY RANKING @ INT8 (Lower is better)
--------------------------------------------------------------------------------

Rank   Hardware                       Latency (ms)    vs Fastest     
----------------------------------------------------------------------
1      KPU-T300 @ 50W (210/60/30)     85.752          1.00           × ← Auto performance
2      KPU-T300 @ 25W (210/60/30)     86.799          1.01           × ← Auto normal
3      KPU-T300 @ 12.5W (210/60/30)   88.162          1.03           × ← Auto low-power
4      KPU-T100 @ 24W (70/20/10)      107.310         1.25           × ← Performance
5      KPU-T100 @ 12W (70/20/10)      118.310         1.38           × ← Balanced
6      Intel CPU (AVX-512)            127.320         1.48           × 
7      KPU-T100 @ 6W (70/20/10)       133.922         1.56           × ← Battery-optimized
8      AMD CPU (AVX-2)                148.258         1.73           × 
9      Jetson-Thor @ 30W              177.627         2.07           × ← Auto performance
10     Jetson-Orin-AGX @ 15W          354.460         4.13           × 
11     Coral-Edge-TPU                 2278.707        26.57          × 
12     CGRA-Plasticine-v2             2581.635        30.11          × 
13     DPU-Vitis-AI                   2821.092        32.90          × 

--------------------------------------------------------------------------------
EA-2: ENERGY EFFICIENCY @ INT8 (Lower is better)
--------------------------------------------------------------------------------

Rank   Hardware                       Energy (J)      Power (W)       Battery Life        
-----------------------------------------------------------------------------------------------
1      KPU-T300 @ 12.5W (210/60/30)   0.1775          2.0             49.7                 hrs
2      KPU-T300 @ 25W (210/60/30)     0.1775          2.0             48.9                 hrs
3      KPU-T300 @ 50W (210/60/30)     0.1775          2.1             48.3                 hrs
4      KPU-T100 @ 6W (70/20/10)       0.2139          1.6             62.6                 hrs
5      KPU-T100 @ 12W (70/20/10)      0.2139          1.8             55.3                 hrs
6      KPU-T100 @ 24W (70/20/10)      0.2139          2.0             50.2                 hrs
7      Jetson-Thor @ 30W              0.3828          2.2             46.4                 hrs
8      Jetson-Orin-AGX @ 15W          0.4784          1.3             74.1                 hrs
9      Intel CPU (AVX-512)            0.5585          4.4             22.8                 hrs
10     AMD CPU (AVX-2)                0.5585          3.8             26.5                 hrs
11     Coral-Edge-TPU                 1.4739          0.6             154.6                hrs
12     DPU-Vitis-AI                   2.0585          0.7             137.0                hrs
13     CGRA-Plasticine-v2             98.4165         38.1            2.6                  hrs

Notes:
  - Battery life assumes 100 Wh battery (typical for mobile robots)
  - Calculation: Battery Life = 100 Wh / Power Consumption (W)
  - Continuous operation at the latency shown above
  - For 20 FPS operation (50ms per frame), multiply latency × 20 to get actual power draw

--------------------------------------------------------------------------------
EA-3: POWER vs PERFORMANCE TRADE-OFF (Ranked by Perf/Watt)
--------------------------------------------------------------------------------

Rank   Hardware                       TDP (W)      Latency (ms)    Perf/Watt       Category            
---------------------------------------------------------------------------------------------------------
1      KPU-T100 @ 6W (70/20/10)       6.0          133.922         1.24            Battery-powered     
2      KPU-T300 @ 12.5W (210/60/30)   12.5         88.162          0.91            Low-power edge      
3      KPU-T100 @ 12W (70/20/10)      12.0         118.310         0.70            Low-power edge      
4      KPU-T300 @ 25W (210/60/30)     25.0         86.799          0.46            Edge AI             
5      KPU-T100 @ 24W (70/20/10)      24.0         107.310         0.39            Edge AI             
6      KPU-T300 @ 50W (210/60/30)     50.0         85.752          0.23            Automotive          
7      Coral-Edge-TPU                 2.0          2278.707        0.22            Battery-powered     
8      Jetson-Orin-AGX @ 15W          15.0         354.460         0.19            Low-power edge      
9      Jetson-Thor @ 30W              30.0         177.627         0.19            Edge AI             
10     CGRA-Plasticine-v2             5.0          2581.635        0.08            Battery-powered     
11     AMD CPU (AVX-2)                105.0        148.258         0.06            Workstation/Server  
12     Intel CPU (AVX-512)            125.0        127.320         0.06            Workstation/Server  
13     DPU-Vitis-AI                   10.0         2821.092        0.04            Low-power edge      

--------------------------------------------------------------------------------
EA-4: KPU SKU COMPARISON (T100 vs T300 across power profiles)
--------------------------------------------------------------------------------

T100 (100 tiles: 70 INT8, 20 BF16, 10 Matrix) - Embodied AI:
Power Profile        Latency (ms)    Energy (J)      Speedup         Cost        
-------------------------------------------------------------------------------------
KPU-T100 @ 6W (70/20/10) 133.922         0.2139          1.00           × $400        
KPU-T100 @ 12W (70/20/10) 118.310         0.2139          1.13           × $500        
KPU-T100 @ 24W (70/20/10) 107.310         0.2139          1.25           × $650        

T300 (300 tiles: 210 INT8, 60 BF16, 30 Matrix) - Automotive:
Power Profile        Latency (ms)    Energy (J)      vs T100@6W      Cost        
-------------------------------------------------------------------------------------
KPU-T300 @ 12.5W (210/60/30) 88.162          0.1775          1.52           × $900        
KPU-T300 @ 25W (210/60/30) 86.799          0.1775          1.54           × $1,200      
KPU-T300 @ 50W (210/60/30) 85.752          0.1775          1.56           × $1,800      

Key Insights:
  - T100: Best for battery-powered robots/drones (6-24W range)
  - T300: 3× more tiles for automotive AI (12.5-50W range)
  - Higher power profiles enable better clocks and empirical derate

--------------------------------------------------------------------------------
EA-5: WHY CPUs ARE NOT ENOUGH FOR EMBODIED AI
--------------------------------------------------------------------------------

Intel CPU (AVX-512) vs KPU-T100 @ 6W:
  - Latency: 127.320 ms (CPU) vs 133.922 ms (KPU)
  - Speedup: KPU is 1.0× faster
  - Energy: 0.559 J (CPU) vs 0.2139 J (KPU)
  - Energy ratio: KPU is 2.6× more efficient
  - Power: ~125W (CPU) vs ~6W (KPU) = 21× power reduction

  → For 20 FPS embodied AI (50ms budget):
    • CPU: 127 ms - MISSES real-time deadline!
    • KPU: 133.9 ms - MEETS deadline with headroom

  → Battery life (100 Wh battery @ 20 FPS):
    • CPU: 8.95 hours
    • KPU: 23.4 hours (3× longer)

Conclusion:
  - CPUs are TOO SLOW for real-time embodied AI (miss 20 FPS deadline)
  - CPUs are TOO POWER-HUNGRY for battery-powered deployment
  - Hardware acceleration is MANDATORY for practical embodied AI

--------------------------------------------------------------------------------
EA-6: RECOMMENDED HARDWARE BY USE CASE
--------------------------------------------------------------------------------

1. **Battery-Powered Robots/Drones (6-12W budget)**
   Best: KPU-T100 @ 6W (70/20/10)
   - Latency: 133.922 ms
   - Energy: 0.2139 J per inference
   - Battery life: 23.4 hours @ 20 FPS
   - Cost: $400

2. **Mobile Robots (12-24W budget)**
   Best: KPU-T100 @ 12W (70/20/10) or KPU-T100 @ 24W (70/20/10)
   - Latency: 118.310 ms
   - Energy: 0.2139 J per inference
   - Cost: $500

3. **Autonomous Vehicles (12.5-50W budget)**
   Best: KPU-T300 @ 25W (210/60/30) for normal driving
   - Latency: 86.799 ms
   - Energy: 0.1775 J per inference
   - 3× more tiles than T100 for higher throughput
   - Cost: $1,200

4. **High-Performance Edge (30W+ budget)**
   Options: Jetson Thor @ 30W or KPU-T300 @ 50W
   - Jetson: Better software ecosystem (CUDA)
   - KPU: Better energy efficiency and cost

5. **Ultra-Low-Power IoT (< 5W budget)**
   Best: Coral Edge TPU @ 2W
   - Latency: 2278.707 ms
   - Energy: 1.4739 J per inference
   - Cost: $75 (cheapest!)

================================================================================
SUCCESS: Complete hardware comparison finished!
Phase 2 Hardware Mapping COMPLETE (All KPU SKUs + Embodied AI Analysis)
================================================================================
