# Architectural Comparison

## Basic FMA/MAC Energy by Architecture

From technology_profile.py, the base ALU energy scales with process node:

  | Process Node | Base FP32 ALU Energy |
  |--------------|----------------------|
  | 4nm          | 1.3 pJ               |
  | 5nm          | 1.5 pJ               |
  | 7nm          | 1.8 pJ               |
  | 8nm          | 1.9 pJ               |
  | 16nm         | 2.7 pJ               |

Then circuit type multipliers are applied to model circuit approaches used for different architecture, ranging from low latency to low complexity:

  | Circuit Type       | Multiplier | Reason                                                 |
  |--------------------|------------|--------------------------------------------------------|
  | x86_performance    | 2.50x      | High voltage (5GHz), deep OoO pipelines, full IEEE-754 |
  | x86_efficiency     | 1.50x      | Lower clocks, simpler pipeline                         |
  | arm_performance    | 1.80x      | High-perf ARM cores                                    |
  | arm_efficiency     | 1.00x      | Baseline efficiency cores                              |
  | cuda_core          | 0.95x      | Simpler than CPU                                       |
  | tensor_core        | 0.85x      | Fused 4x4x4 MAC, minimal control                       |
  | systolic_mac (TPU) | 0.80x      | No instruction fetch, weight-stationary                |
  | domain_flow (KPU)  | 0.75x      | Streaming spatial, no fetch at all                     |

At 4nm (NVIDIA H100 Datacenter AI accelerator)

  | Architecture    | MAC Energy            |
  |-----------------|-----------------------|
  | CPU (x86 perf)  | 1.3 * 2.50 = 3.25 pJ  |
  | GPU Tensor Core | 1.3 * 0.85 = 1.105 pJ |
  | TPU Systolic    | 1.3 * 0.80 = 1.04 pJ  |
  | KPU Domain Flow | 1.3 * 0.75 = 0.975 pJ |

At 8nm (NVIDIA Jetson Orin Nano/AGX Ampere-class GPU SoC)

  | Architecture    | MAC Energy            |
  |-----------------|-----------------------|
  | CPU (x86 perf)  | 1.9 * 2.50 = 4.75 pJ  |
  | GPU Tensor Core | 1.9 * 0.85 = 1.615 pJ |
  | TPU Systolic    | 1.9 * 0.80 = 1.52 pJ  |
  | KPU Domain Flow | 1.9 * 0.75 = 1.475 pJ |

## 2. Resource Contention Management

KEY insight here is that the difference between computer architecture is the control overhead to manage the resource contention management per operation:

### 2.1 Stored Program Machine

#### 2.1.1 Multi-core Central Processing Unit
  CPU Basic Cycle (cpu.py lines 7-12):
  INSTRUCTION FETCH -> DECODE -> OPERAND FETCH -> EXECUTE -> WRITEBACK
                                                      |
                                                      v
                                              MEMORY ACCESS (L1->L2->L3->DRAM)
  - With AVX-512 (simd_width=16): instruction overhead amortized 16x
  - Without SIMD: 2 instructions per FMA

#### 2.1.2 Many-core Graphics Processing Unit

GPU SM Architecture (4 Partitions):
```
  +------------------+------------------+------------------+------------------+
  |   Partition 0    |   Partition 1    |   Partition 2    |   Partition 3    |
  +------------------+------------------+------------------+------------------+
  | Warp Scheduler   | Warp Scheduler   | Warp Scheduler   | Warp Scheduler   |
  | Scoreboard       | Scoreboard       | Scoreboard       | Scoreboard       |
  | Operand Collector| Operand Collector| Operand Collector| Operand Collector|
  +------------------+------------------+------------------+------------------+
  | 32 CUDA Cores    | 32 CUDA Cores    | 32 CUDA Cores    | 32 CUDA Cores    |
  | 1 TensorCore     | 1 TensorCore     | 1 TensorCore     | 1 TensorCore     |
  | Register Bank    | Register Bank    | Register Bank    | Register Bank    |
  +------------------+------------------+------------------+------------------+
                              |
                    Shared L1/Shared Memory (192KB)
                              |
                    L2 Cache (50MB shared across SMs)
                              |
                    HBM (80GB)
```

GPU Request/Reply Execution Cycle (Stored Program Machine):
  
  For EACH warp instruction, the GPU must:

    1. Scoreboard lookup - Check RAW/WAW/WAR hazards for this warp
    2. Generate register addresses - Decode source/destination register IDs
    3. Operand collector - Gather operands from banked register file
    4. Bank arbitration - Resolve conflicts when multiple threads access same bank
    5. Send operands to ALU - Route data through operand network
    6. ALU execution - Perform the actual computation
    7. Write result back - Route result back to register file

This request/reply overhead is FUNDAMENTAL to stored program machines.
Each instruction must explicitly specify WHERE its operands come from
and WHERE its result goes.

GPU Request/Reply Cycle Energy Breakdown (per warp instruction at 4nm):
  | Component           | Energy  | Description                                    |
  |---------------------|---------|------------------------------------------------|
  | Scoreboard lookup   | 0.3 pJ  | CAM-like structure for dependency tracking     |
  | Register addr gen   | 0.6 pJ  | Decode src1, src2, dst (3 x 0.2 pJ)           |
  | Operand collector   | 0.8 pJ  | Gather operands, buffer until ready           |
  | Bank arbitration    | 0.3 pJ  | Resolve bank conflicts (~10% conflict rate)   |
  | Operand routing     | 0.4 pJ  | Crossbar to route operands to execution units |
  | Result routing      | 0.3 pJ  | Route results back to register file           |
  |---------------------|---------|------------------------------------------------|
  | Total overhead      | 2.7 pJ  | Per warp instruction (amortized over 32 ops)  |

Native Execution Units:
  - GPU (CUDA):      4 warp instructions -> 128 MACs (4 partitions x 32 cores)
  - GPU (TensorCore): 4 MMA instructions -> 256 MACs (4 partitions x 64 MACs)

TensorCore has reduced overhead vs CUDA cores due to bulk matrix fragment access:
  | Component           | CUDA Core | TensorCore | Reason                          |
  |---------------------|-----------|------------|----------------------------------|
  | Scoreboard lookup   | 0.3 pJ    | 0.3 pJ     | Still need dependency tracking   |
  | Register addr gen   | 0.6 pJ    | 0.45 pJ    | Fewer addresses (matrix frags)   |
  | Operand collector   | 0.8 pJ    | 0.5 pJ     | Bulk load of 4x4 tiles           |
  | Bank arbitration    | 0.3 pJ    | 0.2 pJ     | Structured access pattern        |
  | Operand routing     | 0.4 pJ    | 0.3 pJ     | Route tiles vs scalars           |
  | Result routing      | 0.3 pJ    | 0.25 pJ    | Route 4x4 result tile            |
  |---------------------|-----------|------------|----------------------------------|
  | Total overhead      | 2.7 pJ    | 2.0 pJ     | Per instruction                  |

### 2.2 Systolic Arrays

#### 2.2.1 Google Tensor Processing Unit
  TPU Systolic (tpu.py lines 8-30):
  - NO instruction fetch per operation (fixed function)
  - Weight-stationary dataflow (weights stay in place)
  - Data flows through systolic array
  - Just configuration + control signals per tile

#### 2.2.2 Stillwater Knowledge Processing Unit
  KPU Domain Flow (kpu.py lines 8-66):
  - NO instruction fetch per operation
  - NO register address generation (operands arrive via SURE network)
  - NO operand collector
  - NO bank arbitration
  - PE-to-PE transfer: ~0.05 pJ (just wire + local latch)
  - vs GPU's ~2.7 pJ for operand collection per instruction

The fundamental insight from kpu.py (lines 213-221):
  GPU request/reply cycle per instruction:
    scoreboard (0.3pJ) + addr_gen (0.6pJ) + operand_collector (0.8pJ)
    + bank_arb (0.3pJ) + routing (0.7pJ) = ~2.7 pJ

  KPU SURE network per operation:
    PE-to-PE wire delay + local register write only
    = ~0.05 pJ (just the wire and latch energy)

This 50x reduction in internal data movement is the main source
of KPU's energy efficiency over stored program machines.

