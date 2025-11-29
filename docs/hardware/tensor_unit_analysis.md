# Streaming Multiprocessor architecture

## SM Architecture Overview (Ampere/Ada Lovelace baseline)

An SM is fundamentally a **wide SIMD processor with hardware multithreading** designed to hide memory latency through massive parallelism rather than caching.

### Top-level organization

```
SM
├── Instruction Cache (L0 I-Cache, ~16-32KB)
├── Constant Cache / Uniform Register File
├── Shared Memory / L1 Data Cache (configurable, 128-192KB combined)
├── Texture Units (4 per SM typically)
└── 4 Processing Blocks (Sub-partitions)
    └── [replicated structure below]
```

### Per processing block

Each of the 4 partitions contains:

```
Processing Block
├── Warp Scheduler (1)
│   ├── Scoreboard
│   ├── I-Buffer (holds decoded instructions per warp)
│   └── Arbitration logic
├── Dispatch Units (1-2, architecture dependent)
├── Register File Segment (~16K x 32-bit registers)
├── Execution Units
│   ├── FP32 Cores (16-32 depending on arch)
│   ├── INT32 Cores (16, sometimes fused with FP32)
│   ├── FP64 Cores (varies: 1:2 to 1:64 ratio vs FP32)
│   ├── Tensor Cores (1-2 per block)
│   ├── Load/Store Units (8)
│   └── Special Function Units (4) - transcendentals
└── Operand Collector
```

### Key shared resources (SM-wide)

**Shared Memory / L1 Cache**
- Banked (32 banks, 4-byte words)
- Configurable split between shared memory and L1
- Bank conflicts serialize accesses within a warp

**Texture Units**
- Dedicated filtering hardware
- Separate cache hierarchy
- 4 tex ops per cycle per SM typically

**Register File**
- Physically distributed across partitions but logically unified per thread block
- 256KB total per SM (Ampere)
- No register renaming—compiler allocates statically

### Execution model

A warp (32 threads) is the atomic scheduling unit. Each cycle, per partition:

1. Scheduler selects one eligible warp
2. Dispatch unit sends one (or two) instructions to execution units
3. All 32 threads execute in lockstep (predication for divergence)
4. Results write back to register file

The SM can have up to **64 warps resident** (2048 threads), distributed across the 4 partitions (~16 warps each). Occupancy depends on register pressure, shared memory usage, and thread block dimensions.

### Memory hierarchy visible to the SM

```
Registers       → 0 cycles (after operand collection)
Shared Memory   → ~20-30 cycles (bank conflict dependent)
L1 Cache        → ~30-40 cycles
L2 Cache        → ~200+ cycles
Global Memory   → ~400-800 cycles
```

The entire design philosophy is: keep enough warps in flight that while some are stalled on memory, others are ready to execute—hiding latency through occupancy rather than deep caches.

# SM Partitioning

SM partitioning is the division into **processing blocks** (sometimes called "SM sub-partitions" or "SM partitions"). The exact number varies by architecture—it's 4 in many recent architectures (Volta, Turing, Ampere, Ada Lovelace), but Hopper uses 4 as well while organizing things somewhat differently.

## Why partition at all?

The fundamental reason is **resource management and scheduling efficiency**, not some magic property of the number 4.

Each partition contains:
- Its own **warp scheduler**
- A portion of the register file
- A subset of the execution units (INT32, FP32, tensor cores, etc.)
- Its own piece of the shared memory/L1 cache banking

## The practical motivations

**1. Warp scheduling scalability**

A single warp scheduler trying to manage all warps and all execution units in a large SM would be a timing nightmare. By partitioning, each scheduler only needs to track ~16 warps (depending on occupancy) and dispatch to a smaller set of functional units. This keeps cycle times reasonable.

**2. Register file banking**

The register file is enormous (256KB in Ampere SMs). Accessing it as a monolithic structure with enough ports for all simultaneous operand reads would be prohibitively expensive in area and power. Partitioning allows banked access with local routing.

**3. Wire lengths and physical layout**

Modern SMs are physically large. Partitioning creates locality—warps scheduled on one partition primarily access nearby resources, reducing cross-chip communication.

**Why 4 specifically?**

There's no deep theoretical reason it must be 4. It's an engineering tradeoff:
- 2 partitions wouldn't provide enough parallelism for the execution width NVIDIA wants
- 8 partitions would add coordination overhead and area cost for diminishing returns

4 has proven to be a sweet spot across multiple generations, but it's empirical, not fundamental. AMD's RDNA uses different partitioning (dual compute units with different internal organization).

# Thread Scheduling

The warp scheduler is one of the most complex pieces of the SM, and its cost is a legitimate concern that shapes GPU architecture decisions.

## What the warp scheduler actually does

**1. Warp eligibility tracking**

Every cycle, the scheduler must determine which warps are *ready* to issue. A warp is stalled if:
- It's waiting on a memory operation (load latency, cache miss)
- It has a data dependency on a previous instruction still in flight
- It's blocked at a barrier (`__syncthreads()`)
- Required execution units are busy
- It's waiting on a scoreboard entry to clear

This requires maintaining a **scoreboard** per warp tracking in-flight instructions and their destination registers.

**2. Instruction fetch and decode/buffering**

The scheduler fetches instructions from the I-cache, decodes them (or retrieves pre-decoded micro-ops depending on architecture), and buffers them. NVIDIA uses an **instruction buffer** (I-Buffer) per warp holding several decoded instructions ready for issue.

**3. Operand collection**

Once an instruction is selected, the scheduler must:
- Read source operands from the register file (coordinating with bank arbitration)
- Handle operand forwarding/bypassing where possible
- Manage uniform register access vs. vector register access

**4. Arbitration and dispatch**

With multiple eligible warps competing for shared execution units, the scheduler arbitrates. NVIDIA typically uses a **greedy-then-oldest** or similar policy—prioritizing warps that can fill execution slots, with age as a tiebreaker to prevent starvation.

**5. Dual-issue logic (architecture dependent)**

In some architectures, a single warp scheduler can issue two independent instructions from the *same warp* in one cycle if there's no conflict (different functional units, no dependencies). This requires dependency checking across the instruction pair.

## The hardware cost

The scheduler needs:
- Per-warp scoreboards (tracking ~24-32 destination registers in flight per warp)
- Warp state machines (active, blocked, barrier, exited)
- Priority logic across all managed warps
- Multi-ported access to the I-Buffer
- Register bank conflict detection

This is why NVIDIA partitions the SM—replicating 4 simpler schedulers is more tractable than one monolithic scheduler managing 64 warps and dozens of functional units.

## How other GPUs compare

**AMD RDNA (consumer) / CDNA (compute)**

AMD uses a different model. RDNA organizes into **Wave32** (or Wave64 in CDNA) as the basic SIMD unit. Their "Compute Unit" has:
- A scheduler, but it operates on wavefronts with a somewhat simpler issue model
- RDNA 3 specifically introduced dual-issue for certain instruction pairs
- Less aggressive latency hiding via scheduling; more reliance on explicit software prefetching and occupancy

The scheduler is arguably simpler because AMD's model doesn't lean as hard into fine-grained latency hiding.

**Intel Xe (Arc / Data Center Max)**

Intel's Xe-cores use **Execution Units (EUs)** grouped into **Xe-cores** (or subslices). Each EU has:
- 2 ALU pipes (XVE in Xe2)
- A thread arbiter managing 8 threads per EU
- Simpler than NVIDIA's scheduler but compensated by compiler-driven scheduling

Intel's approach offloads more scheduling responsibility to the compiler.

**Apple GPUs**

Apple's architecture is less documented, but from reverse engineering it appears to use a relatively wide SIMD with simpler scheduling, relying more heavily on unified memory to reduce memory latency variance.

## The tradeoff NVIDIA chose

NVIDIA's warp scheduler is expensive because they've optimized for **latency hiding through hardware**. The philosophy:

> "Make the hardware smart enough that average code runs fast without heroic compiler optimization."

AMD and Intel lean more toward:

> "Keep hardware simpler; let the compiler and programmer manage scheduling."

Neither is objectively superior—NVIDIA's approach favors CUDA's programming model where you write relatively naive parallel code and the hardware hides latency. AMD/Intel can achieve similar throughput but often requires more tuning.

Is there a particular piece—
scoreboarding
I-Buffer
arbitration policy

# Register File Organization

One of the most challenging pieces of GPU microarchitecture is the register file. The register file is arguably *the* critical design constraint in an SM, and it doesn't work the way a naive reading of the specs suggests.

## The illusion vs. the reality

**What the specs say:** 65,536 32-bit registers per SM, "0 cycle" access

**What actually happens:** Multi-cycle operand collection with latency hidden by the scheduler

### Register file physical organization

The register file is **banked and distributed**. For a partition with ~16K registers:

```
Partition Register File (~16K x 32-bit = 64KB)
├── Bank 0  (512-1K registers)
├── Bank 1
├── ...
└── Bank N  (typically 16-32 banks per partition)
```

Each bank is a relatively small SRAM array with limited ports (typically 1-2 read ports, 1 write port per bank). The key insight:

**A warp's registers are striped across banks by thread ID and register number.**

For a 32-thread warp accessing register `R5`, the physical mapping might be:

```
Thread 0  → R5 → Bank (0 + 5) mod N
Thread 1  → R5 → Bank (1 + 5) mod N
Thread 2  → R5 → Bank (2 + 5) mod N
...
```

This distributes accesses across banks, allowing parallel reads for one operand across all 32 threads—*if* there are no conflicts.

## The operand collector

This is the piece that makes it work. The scheduler doesn't directly feed operands to execution units. Instead:

```
┌─────────────┐    ┌───────────────────┐     ┌─────────────┐
│  I-Buffer   │───>│ Operand Collector │────>│  Exec Unit  │
│ (decoded    │    │   (staging area)  │     │             │
│  instrs)    │    └───────────────────┘     └─────────────┘
└─────────────┘              ▲
                             │
                    ┌────────┴────────┐
                    │  Register File  │
                    │    (banked)     │
                    └─────────────────┘
```

The operand collector:

1. **Receives dispatched instructions** with register operand specifiers
2. **Requests operands from register banks** over multiple cycles
3. **Buffers partial results** until all operands for an instruction are gathered
4. **Releases the instruction** to execution units only when complete

This means an instruction with 3 source operands might take 2-4 cycles in the operand collector before execution, depending on bank conflicts. The "0 cycle" latency is only true *after* operand collection—execution itself has no additional register access stall.

### Collector entries

The operand collector has a fixed number of **collector units** (typically 4-8 per partition). Each can hold one in-flight instruction gathering operands. If all collector units are full, dispatch stalls.

```
Collector Unit State:
├── Instruction opcode + destination
├── Operand 0: [needed] [ready] [value]
├── Operand 1: [needed] [ready] [value]
├── Operand 2: [needed] [ready] [value]
└── Ready-to-issue bit
```

## Bank conflict mechanics

For a warp reading one register across 32 threads with 16 banks:

- Best case: 2 threads per bank, 2 cycles to read all 32 values
- Typical case: 2-4 cycles per operand

For an instruction with 3 source operands:
- Best case: 6 cycles in operand collector
- Worst case: Much worse if the same bank is hammered

**The register-to-bank mapping is specifically designed to minimize conflicts for common access patterns** (all threads reading the same register number, sequential register numbers, etc.).

## Why this doesn't kill performance

**1. Warp-level parallelism hides collector latency**

While warp A is spending 4 cycles collecting operands, warps B, C, D can be issuing. The scheduler keeps the execution units fed.

**2. Execution unit latency matches**

FP32/INT32 operations have ~4-6 cycle latency anyway. The operand collection overlaps with previous instruction execution.

**3. Compiler register allocation matters**

The CUDA compiler (ptxas) is aware of bank mapping and tries to allocate registers to minimize conflicts. This is partly why `-maxrregcount` and occupancy tuning matter so much.

## The routing problem

Routing 64K registers to 128 CUDA cores in one cycle is impossible. The solution is careful staging of requests:

The actual data flow is:

```
Register Banks (distributed SRAM)
        │
        ▼ (banked crossbar, multi-cycle)
Operand Collector Buffers
        │
        ▼ (local routing, single cycle)
Execution Units
```

The crossbar between register banks and collector units is **not** a full 64K-to-128 crossbar. It's a banked crossbar where:

- Each bank connects to all collector units (narrow mux)
- Each collector unit can receive from any bank (but only one per cycle per bank)
- The crossbar width is `num_banks × 32 bits`, not `num_registers × 32 bits`

This makes the routing tractable—16-32 banks × 32 bits × number of collector units is manageable in silicon.

## Rough cycle-level timeline

```
Cycle 0:    Scheduler selects warp W, instruction I
Cycle 1:    Dispatch to collector unit C
Cycle 2-4:  Operand collection (bank reads)
Cycle 5:    All operands ready, issue to FP32 unit
Cycle 6-10: Execution
Cycle 11:   Writeback to register bank
```

Meanwhile, cycles 1-10 the scheduler is issuing other warps' instructions—the pipeline stays full despite the multi-cycle operand gather.

## What this means for optimization

- **Register pressure** isn't just about total count—bank conflicts from access patterns matter
- **Instruction-level parallelism within a warp** helps fill collector units efficiently
- **The 255 register limit per thread** is partly about bank mapping, not just total capacity

# The Register Bank/Collectors crossbar

This is where we get into the real hardware engineering. I'll walk through how the crossbar actually works, because understanding it explains a lot of GPU performance quirks.

## The problem statement

For one partition with ~16K registers across 16 banks, we need to:

1. Allow 32 threads to read up to 3 operands each per instruction
2. Feed 4-8 collector units simultaneously
3. Do this without an impossibly large crossbar

A naive full crossbar (16K sources × 8 destinations) would be roughly 128K switching points—completely impractical at the required speeds.

## The hierarchical banked crossbar

The actual design is a **two-level switching network**:

```
Level 1: Bank Arbitration (per bank)
──────────────────────────────────────────────────────

Bank 0 ──┬── Arbiter ──▶ Single 32-bit output
         │     ▲
         │     │ (select which register in this bank)
         │   Address from collector request
         
Bank 1 ──┬── Arbiter ──▶ Single 32-bit output
         │
        ...

Bank 15 ─┬── Arbiter ──▶ Single 32-bit output


Level 2: Collector Crossbar
──────────────────────────────────────────────────────

         Bank outputs (16 × 32-bit)
              │
              ▼
    ┌─────────────────────────┐
    │   16×8 Crossbar Switch  │
    │   (or 16×N for N        │
    │    collector units)     │
    └─────────────────────────┘
              │
              ▼
    Collector Unit inputs (8 × 32-bit per cycle)
```

### Level 1: Bank arbitration

Each bank is a small SRAM (512-1K registers). Each cycle, a bank can service **one read** and **one write**. The arbiter selects which pending request wins.

```
Bank Internal Structure:
┌────────────────────────────────────────┐
│  SRAM Array (512-1K × 32-bit)          │
│                                        │
│  Read Port ◀── Address Mux ◀── Arbiter │
│                     ▲                  │
│                     │                  │
│  Pending requests from collectors      │
│  [C0: R17] [C1: R45] [C2: R17] [C3: -] │
│                                        │
│  Arbiter grants one, others wait       │
└────────────────────────────────────────┘
```

The arbiter uses **round-robin** or **age-based priority** to prevent starvation. If two collector units want the same bank, one waits.

### Level 2: Collector crossbar

This is the actual routing network. With 16 banks and 8 collector units:

```
        B0  B1  B2  B3  B4  B5  B6  B7  B8  B9  B10 B11 B12 B13 B14 B15
        │   │   │   │   │   │   │   │   │   │   │   │   │   │   │   │
        ▼   ▼   ▼   ▼   ▼   ▼   ▼   ▼   ▼   ▼   ▼   ▼   ▼   ▼   ▼   ▼
C0 ◀── [═══════════════════════════════════════════════════════════] ── 16:1 mux
C1 ◀── [═══════════════════════════════════════════════════════════] ── 16:1 mux
C2 ◀── [═══════════════════════════════════════════════════════════] ── 16:1 mux
C3 ◀── [═══════════════════════════════════════════════════════════] ── 16:1 mux
C4 ◀── [═══════════════════════════════════════════════════════════] ── 16:1 mux
C5 ◀── [═══════════════════════════════════════════════════════════] ── 16:1 mux
C6 ◀── [═══════════════════════════════════════════════════════════] ── 16:1 mux
C7 ◀── [═══════════════════════════════════════════════════════════] ── 16:1 mux
```

Each collector unit has a 16:1 multiplexer selecting from bank outputs. The crossbar is:

- **16 × 8 = 128 switching points** (manageable)
- **Each mux is 16:1 × 32 bits = 512-bit mux** (standard cell library component)

### The timing reality

This crossbar operates in a **pipelined** fashion:

```
Cycle N:    Collector C0 requests Bank 3, Register 47
            Collector C1 requests Bank 3, Register 12  (CONFLICT)
            Collector C2 requests Bank 7, Register 91
            
Cycle N+1:  Bank 3 arbiter grants C0 (C1 queued)
            Bank 7 arbiter grants C2
            SRAM read initiated
            
Cycle N+2:  Bank 3 data arrives at crossbar
            Bank 7 data arrives at crossbar
            Muxes route to C0, C2
            
Cycle N+3:  C0, C2 latch data
            Bank 3 now services C1's request
```

So minimum latency from request to data is **2-3 cycles**, with conflicts adding more.

## Handling 32 threads × 3 operands

Here's where it gets clever. A warp has 32 threads, each needing the same register number (say R5). With the bank mapping:

```
Thread T, Register R → Bank = (T + R) mod NumBanks
```

For R5 across 32 threads with 16 banks:

```
Thread 0  → Bank 5
Thread 1  → Bank 6
Thread 2  → Bank 7
...
Thread 11 → Bank 0  (wraps)
Thread 12 → Bank 1
...
Thread 15 → Bank 4
Thread 16 → Bank 5  (collision with Thread 0!)
Thread 17 → Bank 6  (collision with Thread 1!)
...
```

With 32 threads and 16 banks, we get exactly **2 threads per bank**. The bank can only read one per cycle, so:

**Minimum 2 cycles per operand, 6 cycles for 3 operands** (assuming no inter-operand conflicts).

## The collector unit internal structure

Each collector unit manages this multi-cycle gather:

```
Collector Unit C0:
┌──────────────────────────────────────────────────────────────┐
│ Instruction: FFMA R8, R5, R6, R7                             │
│                                                              │
│ Operand 0 (R5):  [████████████████────────────────]  16/32   │
│                   Threads 0-15 collected                     │
│                                                              │
│ Operand 1 (R6):  [████████████████████████████████]  32/32 ✓ │
│                   Complete                                   │
│                                                              │
│ Operand 2 (R7):  [────────────────────────────────]   0/32   │
│                   Waiting                                    │
│                                                              │
│ Destination: R8 (scoreboard reservation made)                │
│                                                              │
│ Status: GATHERING                                            │
└──────────────────────────────────────────────────────────────┘
```

The collector maintains a **32-wide × 32-bit buffer per operand**—essentially 3 × 128 bytes = 384 bytes of staging per collector unit. With 8 collectors, that's ~3KB just for operand staging per partition.

## Conflict patterns that kill performance

**Intra-operand conflict (unavoidable with 32 threads, 16 banks):**
- Always 2 cycles minimum per operand
- This is baked into the design

**Inter-operand conflict (avoidable with good register allocation):**

```
Bad: FFMA R8, R0, R16, R32
     
     Thread 0: R0 → Bank 0, R16 → Bank 0, R32 → Bank 0
               All three operands hit the same bank!
```

The compiler tries to avoid this by spacing register allocations.

**Inter-instruction conflict (multiple warps hitting same banks):**

If two collector units simultaneously want:
- C0: Warp A, R5, Thread 0 → Bank 5
- C1: Warp B, R21, Thread 0 → Bank 5

One must wait. The scheduler tries to pick warps with non-conflicting access patterns, but this is imperfect.

## Physical layout implications

The crossbar dominates the partition's wire routing:

```
┌─────────────────────────────────────────────────────────┐
│                     Partition Layout                     │
│                                                          │
│  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐     ┌───────────┐  │
│  │Bank 0│ │Bank 1│ │Bank 2│ │Bank 3│ ... │  Bank 15  │  │
│  └──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘     └─────┬─────┘  │
│     │        │        │        │               │        │
│     └────────┴────────┴────────┴───────────────┘        │
│                        │                                 │
│              ┌─────────▼─────────┐                       │
│              │     Crossbar      │  ◀── Wire-dominated   │
│              │   (routing area)  │      region           │
│              └─────────┬─────────┘                       │
│                        │                                 │
│     ┌──────┬──────┬────┴───┬──────┬──────┬──────┐       │
│     ▼      ▼      ▼        ▼      ▼      ▼      ▼       │
│   ┌───┐  ┌───┐  ┌───┐    ┌───┐  ┌───┐  ┌───┐  ┌───┐    │
│   │C0 │  │C1 │  │C2 │    │C3 │  │C4 │  │C5 │  │C6 │    │
│   └─┬─┘  └─┬─┘  └─┬─┘    └─┬─┘  └─┬─┘  └─┬─┘  └─┬─┘    │
│     │      │      │        │      │      │      │       │
│     └──────┴──────┴────────┴──────┴──────┴──────┘       │
│                        │                                 │
│              ┌─────────▼─────────┐                       │
│              │  Execution Units  │                       │
│              └───────────────────┘                       │
└─────────────────────────────────────────────────────────┘
```

The crossbar area is significant—it's why you can't just "add more banks" without cost. More banks = wider crossbar = more area and wire delay.

## Comparison with CPU register files

CPUs take a completely different approach:

| Aspect | GPU (NVIDIA) | CPU (x86-64) |
|--------|--------------|--------------|
| Register count | 64K × 32-bit per SM | 16-32 architectural (hundreds physical) |
| Ports | 1R/1W per bank | 8-12 read, 4-6 write (fully ported) |
| Access latency | 2-4 cycles (hidden) | 0 cycles (in critical path) |
| Routing | Banked crossbar | Full crossbar (small) |
| Renaming | None | Extensive (RAT, PRF) |

CPUs can afford full crossbars because they have ~200 physical registers. GPUs can't, hence the banked design with latency hiding.

## What this means for your code

1. **Register allocation patterns matter**—the compiler does its best, but unusual access patterns can cause conflicts

2. **High occupancy helps**—more warps = more chances to find a non-conflicting warp to schedule while others gather operands

3. **The "256 registers per thread" soft limit** is partly because more registers per thread = fewer warps = less latency hiding for bank conflicts

4. **Inline PTX with explicit register assignment** can hurt performance if you create conflict patterns the compiler would have avoided


writeback path, 
scoreboard interaction
tensor cores complications

# Scoreboard interactions

The scoreboard is the piece that prevents data hazards, and its interaction with the collector units is more nuanced than most descriptions suggest. Let me walk through where it fits.

## The fundamental problem

Consider this instruction sequence:

```
FFMA R8, R5, R6, R7      ; R8 = R5 * R6 + R7
FADD R9, R8, R10         ; R9 = R8 + R10  (RAW hazard on R8)
```

The second instruction cannot read R8 until the first instruction writes it. The scoreboard tracks this.

## Scoreboard structure

Each partition maintains a scoreboard per warp. For ~16 warps per partition:

```
Partition Scoreboard:
┌────────────────────────────────────────────────────────────────┐
│ Warp 0:  [R0:0][R1:0][R2:1][R3:0][R4:0][R5:0][R6:0][R7:0]...  │
│ Warp 1:  [R0:0][R1:1][R2:0][R3:0][R4:1][R5:0][R6:0][R7:0]...  │
│ Warp 2:  [R0:0][R1:0][R2:0][R3:0][R4:0][R5:1][R6:1][R7:0]...  │
│ ...                                                            │
│ Warp 15: [R0:0][R1:0][R2:0][R3:1][R4:0][R5:0][R6:0][R7:1]...  │
└────────────────────────────────────────────────────────────────┘

0 = register available
1 = register has pending write (in-flight)
```

But it's not just a single bit. The scoreboard typically tracks:

```
Scoreboard Entry (per register, per warp):
┌─────────────────────────────────────────┐
│ Pending bit     : 1 = write in flight   │
│ Producer ID     : which collector/unit  │
│ Countdown/cycle : estimated completion  │
└─────────────────────────────────────────┘
```

The extra fields enable **speculative operand fetch**—the collector can start gathering an operand slightly before it's actually written, timing the arrival.

## Where scoreboard checks happen

This is the critical integration point. The scoreboard is consulted at **three stages**:

```
┌─────────────┐     ┌─────────────┐     ┌───────────────┐     ┌──────────┐
│  I-Buffer   │────▶│  Scheduler  │────▶│   Collector   │────▶│   Exec   │
│             │     │  (issue)    │     │   (gather)    │     │          │
└─────────────┘     └──────┬──────┘     └───────┬───────┘     └────┬─────┘
                           │                    │                   │
                    ┌──────▼──────┐      ┌──────▼──────┐     ┌──────▼──────┐
                    │ Scoreboard  │      │ Scoreboard  │     │ Scoreboard  │
                    │ Check #1    │      │ Check #2    │     │   Update    │
                    │ (can issue?)│      │ (can read?) │     │  (clear)    │
                    └─────────────┘      └─────────────┘     └─────────────┘
```

### Check #1: Issue eligibility (Scheduler)

Before the scheduler even considers a warp for issue, it checks:

```
For instruction I with source operands [Rs1, Rs2, Rs3] and destination Rd:

issue_eligible = true

; Check all sources
for each Rs in sources:
    if scoreboard[warp][Rs].pending:
        if scoreboard[warp][Rs].countdown > GATHER_LATENCY:
            issue_eligible = false  ; Data won't be ready in time
            
; Check destination (WAW hazard)
if scoreboard[warp][Rd].pending:
    issue_eligible = false  ; Previous write to Rd not complete
```

This is a **conservative check**. The scheduler won't dispatch an instruction if there's any risk the operand won't be ready by the time collection completes.

### Check #2: Operand fetch (Collector)

Once an instruction is dispatched to a collector unit, it begins gathering operands. But even here, the collector coordinates with the scoreboard:

```
Collector Unit State Machine:

state WAITING_FOR_DISPATCH:
    on dispatch(instruction):
        reserve scoreboard[warp][Rd] = PENDING
        set producer_id = my_collector_id
        transition to GATHERING

state GATHERING:
    for each operand Op:
        if scoreboard[warp][Op.reg].pending:
            if producer_id == my_local_bypass_candidate:
                ; Can capture from writeback bus (forwarding)
                wait_for_bypass(Op)
            else:
                ; Must wait for register file write
                stall_operand(Op)
        else:
            issue_bank_request(Op)
            
    when all operands ready:
        transition to READY_TO_EXECUTE

state READY_TO_EXECUTE:
    wait for execution unit availability
    dispatch to ALU/FMA/etc
```

The key insight: **the collector can begin fetching operands even if some are pending**, as long as it can time the arrival correctly or use bypass paths.

### Scoreboard update: Writeback

When an instruction completes execution and writes back:

```
Writeback Stage:

1. Data arrives on writeback bus
2. Scoreboard lookup: which warp, which register
3. Clear pending bit: scoreboard[warp][Rd].pending = 0
4. Signal any collectors waiting on this register
5. Write to register file bank
```

## The collector-scoreboard integration detail

Here's where collectors and scoreboard tightly couple:

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Collector Unit                               │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ Instruction: FFMA R8, R5, R6, R7    Warp: 3                  │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  Operand Table:                                                     │
│  ┌────────┬───────────┬───────────┬──────────┬──────────────────┐  │
│  │ Operand│ Register  │ Scoreboard│  Status  │     Data         │  │
│  │        │           │  Pending? │          │                  │  │
│  ├────────┼───────────┼───────────┼──────────┼──────────────────┤  │
│  │  Src0  │    R5     │    No     │ FETCHING │ [Bank reqs out]  │  │
│  │  Src1  │    R6     │    YES    │ BLOCKED  │ [Waiting CU #2]  │  │
│  │  Src2  │    R7     │    No     │ COMPLETE │ [32 values]      │  │
│  │  Dst   │    R8     │  CLAIMED  │    --    │       --         │  │
│  └────────┴───────────┴───────────┴──────────┴──────────────────┘  │
│                                                                     │
│  Bypass Monitor:                                                    │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ Watching writeback bus for: Warp 3, R6, from Collector #2    │  │
│  │ When seen: capture data directly, skip register file read    │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

The collector for Src1 (R6) knows:
- R6 is pending (scoreboard says so)
- The producer is Collector #2
- It can either wait for the register file write **or** snoop the writeback bus

## Bypass/forwarding paths

This is a critical optimization. Without bypass:

```
Instruction A writes R8:  Cycle 10 (execution complete)
                          Cycle 11 (writeback to register file)
                          
Instruction B reads R8:   Cycle 12 (earliest bank read)
                          Cycle 13-14 (operand collection)
                          Cycle 15 (execution)
```

With bypass:

```
Instruction A writes R8:  Cycle 10 (execution complete)
                          
Instruction B reads R8:   Cycle 10 (snoop writeback bus)
                          Cycle 11 (execution, other operands permitting)
```

The collector snoops the writeback bus:

```
Writeback Bus:
┌─────────────────────────────────────────────────────────────────┐
│  [Valid] [Warp ID] [Register] [Thread Mask] [32 × 32-bit Data]  │
└──────────────────────────────────────┬──────────────────────────┘
                                       │
           ┌───────────────────────────┼───────────────────────────┐
           ▼                           ▼                           ▼
    ┌─────────────┐             ┌─────────────┐             ┌─────────────┐
    │ Collector 0 │             │ Collector 1 │             │ Collector 2 │
    │             │             │             │             │             │
    │ Watching:   │             │ Watching:   │             │ Watching:   │
    │ W3:R6 ✓     │             │ W7:R12      │             │   (none)    │
    │             │             │             │             │             │
    │ MATCH! ─────┼─capture─────┼─────────────┼─────────────┼───────────▶ │
    └─────────────┘             └─────────────┘             └─────────────┘
```

Each collector has comparators checking every writeback against its pending operand list.

## Scoreboard sizing and complexity

For a partition managing 16 warps, each with 256 possible registers:

```
Naive scoreboard: 16 warps × 256 registers × 1 bit = 4K bits = 512 bytes

Actual scoreboard (with metadata):
  16 warps × 256 registers × ~4 bits = 16K bits = 2KB
  
Plus:
  - Multi-ported access (scheduler reads, collector reads, writeback writes)
  - Associative lookup for bypass matching
  - Priority logic for WAW ordering
```

The scoreboard is a CAM-like structure (Content Addressable Memory) for fast lookups.

## Handling long-latency operations

Memory operations complicate this significantly:

```
LD R8, [R5]          ; Load from global memory (400+ cycles)
FADD R9, R8, R10     ; Depends on load result
```

The scoreboard cannot just decrement a counter—memory latency is unpredictable. Instead:

```
For memory operations:

1. Issue LD: scoreboard[W][R8].pending = 1
             scoreboard[W][R8].type = MEMORY
             scoreboard[W][R8].token = unique_id
             
2. LD goes to Load/Store Unit, eventually to memory

3. (hundreds of cycles later)

4. Memory response arrives with token
   
5. Scoreboard lookup by token:
   scoreboard[W][R8].pending = 0
   signal waiting collectors
```

The token-based completion allows out-of-order memory returns—a load issued later might complete before an earlier one.

## Interaction with warp scheduling

The scheduler's view of the scoreboard determines warp eligibility:

```
Scheduler Eligibility Check (simplified):

for each warp W in partition:
    next_instr = I_buffer[W].head
    
    ; Check source dependencies
    src_ready = true
    for each source S in next_instr.sources:
        if scoreboard[W][S].pending:
            if scoreboard[W][S].type == ALU:
                ; Check if bypass will be available
                if scoreboard[W][S].countdown > bypass_window:
                    src_ready = false
            else:  ; MEMORY type
                src_ready = false  ; Cannot predict memory latency
                
    ; Check destination (WAW)
    dst_ready = not scoreboard[W][next_instr.dst].pending
    
    ; Check structural hazards
    unit_available = execution_unit[next_instr.type].available
    collector_available = any_collector_free()
    
    eligible[W] = src_ready and dst_ready and unit_available and collector_available
```

The scheduler then picks among eligible warps using its priority policy (typically oldest-first or round-robin among same-age warps).

## Summary of the flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Complete Issue-to-Writeback Flow                    │
│                                                                         │
│  I-Buffer ──> Scheduler ──> Collector ──> Exec Unit ──> Writeback       │
│      │            │             │            │              │           │
│      │       ┌────▼────┐   ┌────▼────┐       │         ┌────▼────┐      │
│      │       │ Check   │   │ Gather  │       │         │ Clear   │      │
│      │       │ pending │   │ or wait │       │         │ pending │      │
│      │       │ sources │   │ bypass  │       │         │ signal  │      │
│      │       └────┬────┘   └────┬────┘       │         │ waiters │      │
│      │            │             │            │         └────┬────┘      │
│      │            │             │            │              │           │
│      │            └─────────────┴────────────┴──────────────┘           │
│      │                          │                                       │
│      │                    ┌─────▼─────┐                                 │
│      │                    │SCOREBOARD │                                 │
│      │                    └───────────┘                                 │
│      │                          ▲                                       │
│      │                          │                                       │
│      └─ Reserve dst on dispatch ┘                                       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

The scoreboard is fundamentally the *serialization point* that maintains correctness while the collector units and execution units operate in parallel across multiple warps.

# Writeback path

The writeback path is very energy-intensive, and often glossed over in GPU architecture discussions. The writeback path is energy-intensive, but for reasons more nuanced than simple "coherence"—and the coherence story on GPUs is quite different from CPUs.

## What "coherence" means in the GPU context

First, let's clarify: GPUs do **not** maintain coherence in the CPU sense across the register file. There's no MESI protocol, no snooping for register writes between warps. Each warp's registers are architecturally private.

The energy cost comes from different sources:

1. **Wide data movement** (32 threads × 32 bits = 1024 bits per writeback)
2. **Bank arbitration and routing**
3. **Scoreboard updates and wakeup logic**
4. **Bypass network distribution**

## The writeback datapath

When an execution unit completes:

```
Execution Unit Output:
┌────────────────────────────────────────────────────────────────────────┐
│                                                                        │
│  Result Vector: 32 threads × 32 bits = 1024 bits                       │
│                                                                        │
│  ┌───────┬───────┬───────┬───────┬───────┬─────────────────┬───────┐   │
│  │ T0    │ T1    │ T2    │ T3    │ T4    │      ...        │ T31   │   │
│  │32-bit │32-bit │32-bit │32-bit │32-bit │                 │32-bit │   │
│  └───────┴───────┴───────┴───────┴───────┴─────────────────┴───────┘   │
│                                                                        │
│  Metadata:                                                             │
│  ┌─────────────┬──────────────┬──────────────┬────────────────────┐    │
│  │ Warp ID: 7  │ Dest Reg: R8 │ Pred Mask:   │ Collector ID: 3    │    │
│  │             │              │ 0xFFFFFFFF   │ (for scoreboard)   │    │
│  └─────────────┴──────────────┴──────────────┴────────────────────┘    │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

This 1024-bit result plus metadata must now:

1. Travel to the register file banks
2. Be captured by any snooping collectors (bypass)
3. Update the scoreboard
4. Actually write to SRAM

## The writeback bus architecture

```
                        Execution Units
            ┌───────┬───────┬───────┬───────┬───────┐
            │ FP32  │ FP32  │ INT32 │  SFU  │Tensor │
            │  #0   │  #1   │       │       │ Core  │
            └───┬───┴───┬───┴───┬───┴───┬───┴───┬───┘
                │       │       │       │       │
                ▼       ▼       ▼       ▼       ▼
        ┌─────────────────────────────────────────────┐
        │           Writeback Arbitration             │
        │                                             │
        │  Multiple units may complete same cycle     │
        │  Arbiter serializes onto writeback bus(es)  │
        │                                             │
        │  Typical: 2-4 writeback buses per partition │
        └─────────────────────┬───────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
        ┌──────────┐    ┌──────────┐    ┌──────────┐
        │ WB Bus 0 │    │ WB Bus 1 │    │ WB Bus 2 │
        │ 1024-bit │    │ 1024-bit │    │ 1024-bit │
        │ + meta   │    │ + meta   │    │ + meta   │
        └────┬─────┘    └────┬─────┘    └────┬─────┘
             │               │               │
             └───────────────┴───────────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Distribution    │
                    │ Network         │
                    └────────┬────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ▼                    ▼                    ▼
   ┌─────────┐         ┌───────────┐        ┌─────────┐
   │Collector│         │ Register  │        │Scoreboard│
   │ Bypass  │         │ File      │        │ Update   │
   │ Snoop   │         │ Banks     │        │          │
   └─────────┘         └───────────┘        └──────────┘
```

### Why multiple writeback buses?

A single bus creates a bottleneck. With 4 FP32 units, 4 INT32 units, SFUs, and tensor cores all potentially completing in the same cycle, serialization would stall execution. Multiple buses allow parallel writebacks at the cost of more routing.

## Energy breakdown by component

### 1. Data movement (dominant cost)

```
Writeback data movement per operation:

Source: Execution unit output register
        1024 bits (32 × 32)
        
Path 1: To bypass network
        1024 bits × wire capacitance × (distance to collectors)
        Broadcast to ALL collector units (even those not waiting)
        
Path 2: To register file write ports  
        1024 bits must be demuxed to appropriate banks
        32 threads → 16 banks → 2 writes per bank
        Each bank write: 32 bits × SRAM write energy
        
Total wire length: ~0.5-1mm per writeback (significant at 5nm)
```

The wire capacitance dominates. Moving 1024 bits across even 0.5mm of chip at GHz frequencies costs significant energy. This happens **every instruction that produces a result**.

### 2. Bypass network (second largest cost)

```
Bypass Distribution:
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  Writeback Bus ────┬────────────┬────────────┬────────────┬──────────   │
│                    │            │            │            │             │
│                    ▼            ▼            ▼            ▼             │
│               ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐           │
│               │   C0    │ │   C1    │ │   C2    │ │   C3    │           │
│               │ Compare │ │ Compare │ │ Compare │ │ Compare │           │
│               │ Warp+Reg│ │ Warp+Reg│ │ Warp+Reg│ │ Warp+Reg│           │
│               └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘           │
│                    │            │            │            │             │
│               match?        match?       match?       match?            │
│                    │            │            │            │             │
│                    ▼            ▼            ▼            ▼             │
│               ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐           │
│               │ Capture │ │  Ignore │ │ Capture │ │  Ignore │           │
│               │ 1024b   │ │         │ │ 1024b   │ │         │           │
│               └─────────┘ └─────────┘ └─────────┘ └─────────┘           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

Energy cost:
- Comparators: 8-10 bits compared per collector (warp ID + register)
- Data latches: 1024 bits captured when match (conditional)
- Broadcast wire: driven regardless of whether any collector wants it
```

The broadcast nature is key: the writeback bus is driven to all collectors even if none are waiting. Clock gating and data gating help but don't eliminate this.

### 3. Bank write arbitration

When the writeback reaches the register file:

```
Thread-to-Bank Mapping for Writeback:

Result for Warp 7, R8:
Thread 0  → Bank (0 + 8) mod 16 = Bank 8
Thread 1  → Bank (1 + 8) mod 16 = Bank 9
Thread 2  → Bank (2 + 8) mod 16 = Bank 10
...
Thread 8  → Bank (8 + 8) mod 16 = Bank 0
...
Thread 16 → Bank (16 + 8) mod 16 = Bank 8  (collision!)
Thread 17 → Bank (17 + 8) mod 16 = Bank 9  (collision!)
...

With 32 threads and 16 banks: exactly 2 writes per bank
```

The writeback must serialize:

```
Cycle N:   Threads 0-15 write to banks 0-15 (one write each)
Cycle N+1: Threads 16-31 write to banks 0-15 (one write each)
```

Each bank's write port is active for 2 cycles per writeback. This is unavoidable given the banking.

### 4. Scoreboard update

```
Scoreboard Update Logic:

Input:  Warp ID (4 bits), Register (8 bits), Completion signal

Action:
1. Index into scoreboard RAM: warp × 256 + register
2. Clear pending bit
3. Broadcast wakeup to scheduler

Wakeup fanout:
┌──────────────┐
│  Scoreboard  │
│  Entry Clear │
└──────┬───────┘
       │
       ├──────────> Scheduler warp eligibility logic
       │            (must re-evaluate affected warp)
       │
       ├──────────> Collector units waiting on this reg
       │            (transition from BLOCKED to FETCHING)
       │
       └──────────> Dependency chain tracking
                    (for prefetch/speculation)
```

The wakeup logic has high fanout—one scoreboard clear can affect multiple waiters. This requires careful timing closure.

## The actual energy numbers (order of magnitude)

Based on published research and architectural studies:

```
Energy per writeback operation (32-thread warp, 7nm process):

┌─────────────────────────────────────┬────────────────┐
│ Component                           │ Energy (pJ)    │
├─────────────────────────────────────┼────────────────┤
│ Execution unit output latch         │     ~5-10      │
│ Writeback bus drivers               │    ~20-40      │
│ Bypass network (broadcast + snoop)  │    ~15-30      │
│ Bank write demux + arbitration      │    ~10-20      │
│ SRAM writes (16 banks × 2 writes)   │    ~30-50      │
│ Scoreboard update + wakeup          │     ~5-10      │
├─────────────────────────────────────┼────────────────┤
│ Total                               │   ~85-160 pJ   │
└─────────────────────────────────────┴────────────────┘

For comparison:
- FP32 FMA execution: ~1-2 pJ
- Register file READ: ~20-40 pJ (also significant)
```

The writeback path costs **50-100× more energy than the actual computation** for simple operations. This is why register file access dominates GPU power consumption.

## What about memory coherence?

Now to address the "coherence" claim you mentioned. There are limited coherence concerns:

### Intra-warp coherence (trivial)

All threads in a warp execute in lockstep. No coherence issue—they all see the same instruction's result at the same time.

### Inter-warp register coherence (non-existent)

Warps have private register spaces. Warp 0's R8 and Warp 1's R8 are physically different locations. No coherence protocol needed.

### Shared memory coherence

This is where coherence **does** matter:

```
Warp 0, Thread 0: ST.SHARED [addr], R5
Warp 1, Thread 0: LD.SHARED R6, [addr]
```

Shared memory is banked but shared across warps in a thread block. The GPU handles this through:

1. **Barrier synchronization** (`__syncthreads()`)—software coherence
2. **Memory ordering within a warp**—hardware guarantees
3. **No automatic coherence across warps**—programmer's responsibility

The "writeback coherence" energy cost is really about **shared memory write visibility**, not register coherence.

### L1/L2 cache coherence

Modern GPUs (Volta+) have some cache coherence for global memory:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        L1/L2 Coherence Domain                           │
│                                                                         │
│  SM 0 L1 <──┐                                                           │
│  SM 1 L1 <──┼───── L2 Cache (coherence point) <───── Global Memory      │
│  SM 2 L1 <──┤                                                           │
│  ...        │                                                           │
│  SM N L1 <──┘                                                           │
│                                                                         │
│  Coherence protocol: write-through or write-evict to L2                 │
│  L2 is inclusive and acts as coherence point                            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

This is **not** the writeback path for registers—it's a separate concern for memory operations.

## Why the confusion about "coherence"?

I suspect the analysis you read was conflating:

1. **Register writeback energy** (what we've discussed—dominated by data movement)
2. **Shared memory write energy** (includes bank arbitration and visibility to other warps)
3. **Cache coherence energy** (L1/L2 traffic for global memory)

All three are "writeback" in some sense, but they're architecturally distinct paths.

## The actual coherence-related energy cost

If we include shared memory and cache writes:

```
Shared Memory Write (per 32-thread store):

┌─────────────────────────────────────┬────────────────┐
│ Component                           │ Energy (pJ)    │
├─────────────────────────────────────┼────────────────┤
│ Address calculation                 │     ~5-10      │
│ Bank conflict detection             │     ~5-10      │
│ Bank arbitration (may serialize)    │    ~10-30      │
│ SRAM write (conflict dependent)     │    ~30-100     │
│ Write visibility to other warps    │     ~10-20     │
├─────────────────────────────────────┼────────────────┤
│ Total                               │   ~60-170 pJ   │
└─────────────────────────────────────┴────────────────┘


Global Memory Write (per 128-byte cache line):

┌─────────────────────────────────────┬────────────────┐
│ Component                           │ Energy (pJ)    │
├─────────────────────────────────────┼────────────────┤
│ L1 write + tag update               │    ~50-100     │
│ L1→L2 traffic (if write-through)    │   ~200-500     │
│ L2 write + coherence check          │   ~100-200     │
│ DRAM write (if eviction)            │  ~1000-5000    │
├─────────────────────────────────────┼────────────────┤
│ Total (L1 hit)                      │  ~350-800 pJ   │
│ Total (DRAM)                        │ ~1500-6000 pJ  │
└─────────────────────────────────────┴────────────────┘
```

Memory writes are indeed expensive, but it's the **data movement through the memory hierarchy** that costs, not coherence protocols per se.

## Summary

The register writeback path is energy-intensive primarily due to:

1. **Wide data movement** (1024 bits per writeback)
2. **Bypass broadcast** (driven to all collectors regardless of demand)
3. **Multi-cycle bank writes** (serialization due to banking)

"Coherence" in the CPU sense doesn't apply to the register file. The coherence costs that do exist relate to shared memory and cache hierarchy, which are separate from the register writeback path.

# TensorCore Complications

This is where the SM architecture gets genuinely interesting, because tensor cores break several assumptions that hold for the scalar ALU pipeline.

## The fundamental problem

Tensor cores perform matrix operations. The basic operation (varying by generation) is:

```
D = A × B + C

Where (Ampere generation):
A: 16×16 matrix of FP16/BF16
B: 16×16 matrix of FP16/BF16
C: 16×16 matrix of FP32
D: 16×16 matrix of FP32

Data volume per operation:
A: 16 × 16 × 16 bits = 4096 bits = 512 bytes
B: 16 × 16 × 16 bits = 4096 bits = 512 bytes
C: 16 × 16 × 32 bits = 8192 bits = 1024 bytes
D: 16 × 16 × 32 bits = 8192 bits = 1024 bytes

Total input operands: 2560 bytes
Total output: 1024 bytes
```

Compare this to a scalar FMA:

```
Scalar FMA per warp:
3 inputs × 32 threads × 32 bits = 384 bytes
1 output × 32 threads × 32 bits = 128 bytes
```

Tensor core operands are **~7× larger** than scalar operands. The register file crossbar we discussed earlier cannot handle this directly.

## How tensor core operands are mapped to registers

NVIDIA uses a **distributed register mapping** across threads in a warp. For a 16×16 matrix:

```
Matrix A (16×16 FP16) distribution across 32 threads:

Thread 0:  A[0,0:7]    → 8 × FP16 = 128 bits = 4 registers (R0-R3)
Thread 1:  A[0,8:15]   → 8 × FP16 = 128 bits = 4 registers
Thread 2:  A[1,0:7]    → 8 × FP16 = 128 bits = 4 registers
Thread 3:  A[1,8:15]   → 8 × FP16 = 128 bits = 4 registers
...
Thread 30: A[15,0:7]   → 8 × FP16 = 128 bits = 4 registers
Thread 31: A[15,8:15]  → 8 × FP16 = 128 bits = 4 registers

Total: 32 threads × 4 registers = 128 registers for matrix A
```

The actual mapping is more complex (and architecture-specific), but the principle holds: **matrix fragments are striped across warp lanes and held in regular registers**.

The PTX/SASS instructions reflect this:

```
// PTX wmma (Warp Matrix Multiply Accumulate)
wmma.load.a.sync.aligned.m16n16k16.shared.f16 {%r0,%r1,%r2,%r3}, [%rs1];
wmma.load.b.sync.aligned.m16n16k16.shared.f16 {%r4,%r5,%r6,%r7}, [%rs2];
wmma.load.c.sync.aligned.m16n16k16.shared.f32 {%r8,%r9,%r10,%r11,%r12,%r13,%r14,%r15}, [%rs3];

wmma.mma.sync.aligned.m16n16k16.f32.f32 
    {%r16,%r17,%r18,%r19,%r20,%r21,%r22,%r23},  // D
    {%r0,%r1,%r2,%r3},                           // A
    {%r4,%r5,%r6,%r7},                           // B
    {%r8,%r9,%r10,%r11,%r12,%r13,%r14,%r15};     // C
```

## Operand delivery architecture

Here's where the tensor core integration gets interesting. The standard collector unit cannot gather tensor operands efficiently—the operand count and cross-thread data movement are too high.

### The tensor core operand collector

NVIDIA adds a **separate operand staging mechanism** for tensor cores:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Partition with Tensor Core                           │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                     Register File Banks                          │   │
│  └──────────────────────────────┬──────────────────────────────────┘   │
│                                 │                                       │
│           ┌─────────────────────┴─────────────────────┐                │
│           │                                           │                 │
│           ▼                                           ▼                 │
│  ┌─────────────────────┐                 ┌─────────────────────────┐   │
│  │  Standard Operand   │                 │   Tensor Operand        │   │
│  │  Collectors (4-8)   │                 │   Staging Buffer        │   │
│  │                     │                 │                         │   │
│  │  For: FP32, INT32,  │                 │   Capacity: ~8-16KB     │   │
│  │       SFU, LD/ST    │                 │   (holds A, B, C        │   │
│  │                     │                 │    fragments)           │   │
│  └──────────┬──────────┘                 └────────────┬────────────┘   │
│             │                                         │                 │
│             ▼                                         ▼                 │
│  ┌─────────────────────┐                 ┌─────────────────────────┐   │
│  │  Scalar Execution   │                 │     Tensor Core         │   │
│  │  Units              │                 │                         │   │
│  │  FP32, INT32, SFU   │                 │  Matrix Multiply Unit   │   │
│  └─────────────────────┘                 └─────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Tensor operand staging buffer

This buffer serves as an accumulation point:

```
Tensor Operand Staging Buffer:
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  Matrix A Fragment Storage:                                             │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ Slot 0: [T0:R0-R3][T1:R0-R3][T2:R0-R3]...[T31:R0-R3]           │   │
│  │         4096 bits gathered over multiple cycles                 │   │
│  │         Status: COMPLETE / GATHERING / EMPTY                    │   │
│  ├─────────────────────────────────────────────────────────────────┤   │
│  │ Slot 1: (next A fragment, double buffered)                      │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Matrix B Fragment Storage:                                             │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ Slot 0: [T0:R4-R7][T1:R4-R7]...[T31:R4-R7]                     │   │
│  │         4096 bits                                               │   │
│  ├─────────────────────────────────────────────────────────────────┤   │
│  │ Slot 1: (double buffered)                                       │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Matrix C/D Fragment Storage (FP32, larger):                            │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ Slot 0: [T0:R8-R15][T1:R8-R15]...[T31:R8-R15]                  │   │
│  │         8192 bits                                               │   │
│  ├─────────────────────────────────────────────────────────────────┤   │
│  │ Slot 1: (double buffered)                                       │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### The gathering process

Tensor operand gathering takes **many more cycles** than scalar operands:

```
Tensor Operand Gather Timeline (estimated):

Cycle 0:     Scheduler dispatches WMMA instruction
Cycle 1:     Tensor staging buffer allocated
             Begin gathering Matrix A fragment

Cycles 2-9:  Matrix A gather (32 threads × 4 regs = 128 reg reads)
             With 16 banks, ~2 threads/bank/cycle for one register
             4 registers × 2 cycles minimum = 8 cycles best case
             Bank conflicts extend this

Cycles 10-17: Matrix B gather (same pattern)

Cycles 18-33: Matrix C gather (32 threads × 8 regs = 256 reg reads)
              8 registers × 2 cycles minimum = 16 cycles best case

Cycle 34:    All operands ready, dispatch to tensor core

Cycles 35-N: Tensor core execution (see pipelining below)

Cycles N+1-?: Result writeback (256 register writes)
```

The total operand gather time can be **30-50 cycles** for a single WMMA instruction. This is why tensor core utilization requires careful attention to occupancy and data layout.

## Tensor core internal architecture

Now for the pipelining question. The tensor core is **deeply pipelined**, but differently than scalar units.

### Matrix multiply structure

A 16×16×16 matrix multiply decomposes into:

```
D[i,j] = Σ(k=0 to 15) A[i,k] × B[k,j] + C[i,j]

For each output element: 16 multiply-adds
Total output elements: 256
Total operations: 256 × 16 = 4096 FMAs
```

The tensor core doesn't do 4096 sequential FMAs. It uses a **systolic array** or **spatial dataflow** architecture:

```
Tensor Core Systolic Array (simplified 4×4 view of larger structure):
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│         B[0,0] B[0,1] B[0,2] B[0,3]  (B elements flow down)            │
│            ↓      ↓      ↓      ↓                                       │
│          ┌────┐ ┌────┐ ┌────┐ ┌────┐                                   │
│  A[0,0]→ │ PE │→│ PE │→│ PE │→│ PE │→ D[0,*]                           │
│          └────┘ └────┘ └────┘ └────┘                                   │
│            ↓      ↓      ↓      ↓                                       │
│          ┌────┐ ┌────┐ ┌────┐ ┌────┐                                   │
│  A[1,0]→ │ PE │→│ PE │→│ PE │→│ PE │→ D[1,*]                           │
│          └────┘ └────┘ └────┘ └────┘                                   │
│            ↓      ↓      ↓      ↓                                       │
│          ┌────┐ ┌────┐ ┌────┐ ┌────┐                                   │
│  A[2,0]→ │ PE │→│ PE │→│ PE │→│ PE │→ D[2,*]                           │
│          └────┘ └────┘ └────┘ └────┘                                   │
│            ↓      ↓      ↓      ↓                                       │
│          ┌────┐ ┌────┐ ┌────┐ ┌────┐                                   │
│  A[3,0]→ │ PE │→│ PE │→│ PE │→│ PE │→ D[3,*]                           │
│          └────┘ └────┘ └────┘ └────┘                                   │
│                                                                         │
│  Each PE: FMA unit (multiply-accumulate)                               │
│  Data flows spatially; accumulation happens in registers at each PE    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Processing element detail

Each PE in the array:

```
Processing Element:
┌────────────────────────────────────────┐
│                                        │
│  A_in ──┬──▶ A_out (pass to right)    │
│         │                              │
│         ▼                              │
│      ┌──────┐                          │
│      │  ×   │◀── B_in                  │
│      └──┬───┘      │                   │
│         │          ▼                   │
│         │       B_out (pass down)      │
│         ▼                              │
│      ┌──────┐                          │
│      │  +   │◀── Accumulator           │
│      └──┬───┘                          │
│         │                              │
│         ▼                              │
│    Accumulator (updated)               │
│                                        │
└────────────────────────────────────────┘
```

### Pipelining within the tensor core

The tensor core is pipelined at **multiple levels**:

**Level 1: Inter-PE pipelining (spatial)**

```
Cycle 0:  PE[0,0] receives A[0,0], B[0,0], computes A[0,0]×B[0,0]
Cycle 1:  PE[0,0] receives A[0,1], B[1,0], computes A[0,1]×B[1,0], accumulates
          PE[0,1] receives A[0,0], B[0,1], computes A[0,0]×B[0,1]
          PE[1,0] receives A[1,0], B[0,0], computes A[1,0]×B[0,0]
Cycle 2:  Wave propagates diagonally through array
...
```

The systolic nature means **data reuse is implicit**—each A element is used by a row of PEs, each B element by a column.

**Level 2: Intra-PE pipelining**

Each PE's FMA is itself pipelined:

```
PE FMA Pipeline (estimated 4 stages):

Stage 1: Mantissa multiply (partial products)
Stage 2: Partial product reduction
Stage 3: Alignment and addition with accumulator
Stage 4: Normalization and writeback to accumulator

Throughput: 1 FMA per cycle per PE (after filling)
```

**Level 3: Instruction-level pipelining**

Multiple WMMA instructions can be in flight:

```
WMMA Instruction Pipeline:
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│ Cycle:  0    5   10   15   20   25   30   35   40   45   50   55   60  │
│         │    │    │    │    │    │    │    │    │    │    │    │    │  │
│ WMMA 0: [──Gather A,B,C──][────Execute────][─Writeback─]               │
│                                                                         │
│ WMMA 1:      [──Gather A,B,C──][────Execute────][─Writeback─]          │
│                                                                         │
│ WMMA 2:           [──Gather A,B,C──][────Execute────][─Writeback─]     │
│                                                                         │
│ With double-buffered operand staging, gather can overlap execution     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Tensor core execution latency

Based on microbenchmarking and architectural analysis:

```
WMMA Latency Breakdown (Ampere, m16n16k16):

┌─────────────────────────────────────┬────────────────┐
│ Phase                               │ Cycles         │
├─────────────────────────────────────┼────────────────┤
│ Operand gather (A, B, C)            │   30-50        │
│ Tensor core fill (systolic ramp)    │   ~16          │
│ Tensor core compute                 │   ~16          │
│ Tensor core drain                   │   ~16          │
│ Result writeback                    │   20-40        │
├─────────────────────────────────────┼────────────────┤
│ Total latency (single instruction)  │   ~100-140     │
│ Throughput (pipelined, sustained)   │   ~32 cycles   │
└─────────────────────────────────────┴────────────────┘
```

The key insight: **latency is high, but throughput is excellent** when pipelined.

## Scoreboard integration for tensor operations

Tensor operations complicate the scoreboard significantly:

```
Scoreboard Entries for WMMA:

WMMA.MMA D[R16-R23], A[R0-R3], B[R4-R7], C[R8-R15]

Source registers (must be clear before gather):
  R0, R1, R2, R3       (A fragment, 4 regs × 32 threads)
  R4, R5, R6, R7       (B fragment, 4 regs × 32 threads)  
  R8-R15               (C fragment, 8 regs × 32 threads)
  
Destination registers (marked pending on dispatch):
  R16-R23              (D fragment, 8 regs × 32 threads)

Scoreboard must track:
  - 16 source registers as dependencies
  - 8 destination registers as pending
  - Much longer in-flight duration than scalar ops
```

The scoreboard needs **more entries per WMMA** and must handle the **longer pending duration**:

```
Standard Scoreboard vs Tensor-Extended:
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  Standard Entry:                                                        │
│  [Pending:1][Countdown:4 bits][Producer:3 bits] = ~8 bits/register     │
│                                                                         │
│  Tensor-Extended:                                                       │
│  [Pending:1][Type:1][Countdown:6 bits][Producer:4 bits] = ~12 bits/reg │
│             └─ 0=scalar, 1=tensor (longer countdown)                    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Cross-lane data movement

One aspect we haven't addressed: matrix multiply requires **cross-thread communication** within the warp. Thread 0's A elements must interact with Thread 1's B elements, etc.

The tensor core handles this internally through its dataflow, but there's an implicit **shuffle network**:

```
Implicit Shuffle in Tensor Core:
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  Thread Registers:                                                      │
│  T0: [A0][A1][A2][A3]                                                  │
│  T1: [A4][A5][A6][A7]                                                  │
│  T2: [A8][A9][A10][A11]                                                │
│  ...                                                                    │
│  T16: [B0][B1][B2][B3]                                                 │
│  T17: [B4][B5][B6][B7]                                                 │
│  ...                                                                    │
│                                                                         │
│  The tensor core's input stage must route:                              │
│  - A elements from threads 0-15 to appropriate systolic rows           │
│  - B elements from threads 16-31 to appropriate systolic columns       │
│                                                                         │
│  This is a fixed permutation network, not a general shuffle            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

This permutation network is hardwired for the specific matrix layout NVIDIA chose. It's why you can't use arbitrary matrix layouts with tensor cores—the hardware only supports specific fragment arrangements.

## Putting it together: full WMMA dataflow

```
┌───────────────────────────────────────────────────────────────────────┐
│                   Complete WMMA Execution Flow                        │
│                                                                       │
│  ┌───────────────┐                                                    │
│  │   I-Buffer    │                                                    │
│  │  WMMA instr   │                                                    │
│  └───────┬───────┘                                                    │
│          │                                                            │
│          ▼                                                            │
│  ┌───────────────┐     ┌──────────────────┐                           │
│  │   Scheduler   │────>│   Scoreboard     │                           │
│  │               │     │   Check 16 src   │                           │
│  │ Eligible if   │<────│   regs clear     │                           │
│  │ all srcs ready│     │   Mark 8 dst     │                           │
│  └───────┬───────┘     │   pending        │                           │
│          │             └──────────────────┘                           │
│          ▼                                                            │
│  ┌───────────────────────────────────────┐                            │
│  │     Tensor Operand Staging Buffer     │                            │
│  │                                       │                            │
│  │  ┌──────────┐ ┌──────────┐ ┌────────┐ │                            │
│  │  │ Gather A │ │ Gather B │ │Gather C│ │ <── From register file     │
│  │  │ 30+ cyc  │ │ 30+ cyc  │ │40+ cyc │ │     via banked crossbar    │
│  │  └────┬─────┘ └────┬─────┘ └───┬────┘ │                            │
│  │       │            │           │      │                            │
│  └───────┼────────────┼───────────┼──────┘                            │
│          │            │           │                                   │
│          ▼            ▼           ▼                                   │
│  ┌────────────────────────────────────────┐                           │
│  │         Permutation Network            │                           │
│  │   (Thread→Systolic position mapping)   │                           │
│  └───────────────────┬────────────────────┘                           │
│                      │                                                │
│                      ▼                                                │
│  ┌────────────────────────────────────────┐                           │
│  │          Systolic Array                │                           │
│  │                                        │                           │
│  │    16×16 PE grid (or similar)          │                           │
│  │    Pipelined FMAs at each PE           │                           │
│  │    ~16 cycles to fill                  │                           │
│  │    ~16 cycles compute                  │                           │
│  │    ~16 cycles drain                    │                           │
│  │                                        │                           │
│  └───────────────────┬────────────────────┘                           │
│                      │                                                │
│                      ▼                                                │
│  ┌────────────────────────────────────────┐                           │
│  │      Result Permutation Network        │                           │
│  │   (Systolic position→Thread mapping)   │                           │
│  └───────────────────┬────────────────────┘                           │
│                      │                                                │
│                      ▼                                                │
│  ┌────────────────────────────────────────┐                           │
│  │         Writeback Buffer               │                           │
│  │   8 regs × 32 threads = 8KB            │                           │
│  └───────────────────┬────────────────────┘                           │
│                      │                                                │
│                      ▼                                                │
│  ┌────────────────────────────────────────┐     ┌──────────────────┐  │
│  │      Writeback to Register File        │────>│   Scoreboard     │  │
│  │   256 register writes                  │     │   Clear 8 dst    │  │
│  │   ~20-40 cycles (banked)               │     │   regs pending   │  │
│  └────────────────────────────────────────┘     └──────────────────┘  │
│                                                                       │
└───────────────────────────────────────────────────────────────────────┘
```

## Implications for optimization

This architecture explains several CUDA optimization guidelines:

1. **High occupancy is critical**—tensor ops have long latency; you need many warps to hide it

2. **Register pressure matters more**—each WMMA ties up 24 registers for 100+ cycles

3. **Double buffering operands**—overlap next WMMA's gather with current WMMA's execution

4. **Memory layout matters**—the fragment layout must match hardware expectations; bad layouts cause extra shuffles

5. **Sustained throughput requires pipelining**—a single WMMA is wasteful; you need a stream of them

## Comparison with AMD Matrix Cores

AMD's approach in CDNA (MI100/MI200/MI300):

```
AMD Matrix Core (CDNA2/3):
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  - Wave64 based (64 threads vs NVIDIA's 32)                            │
│  - MFMA instructions (Matrix Fused Multiply-Add)                       │
│  - Different fragment sizes (32×32×8 and variants)                     │
│  - AccVGPR: dedicated accumulator registers (separate from VGPR)       │
│                                                                         │
│  Key difference: AMD separates accumulator storage                      │
│  - Reduces register file pressure                                       │
│  - Accumulator has different access patterns                           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

AMD's dedicated accumulator registers (AccVGPR) avoid the writeback congestion for the C/D matrices, at the cost of less flexibility.

# Competitive Matrix Core Implementations (NVIDIA/AMD/Intel)

You're right to push back on this. I was illustrating the concept with a 16×16 array, but that's not what NVIDIA actually implements. Let me correct this and give you accurate comparisons.

## NVIDIA Tensor Core: Actual Structure

The tensor core is **not** a 16×16 systolic array. Based on die analysis, microbenchmarking, and patent documentation, the actual structure is much smaller:

```
NVIDIA Tensor Core (Volta/Turing/Ampere):

Physical Structure: 4×4 systolic array
Per-cycle throughput: 4×4×4 = 64 FMAs (FP16 multiply, FP32 accumulate)

How 16×16×16 WMMA maps to 4×4 hardware:

  Logical operation: D[16×16] = A[16×16] × B[16×16] + C[16×16]
  Physical execution: Decomposed into smaller tiles processed over multiple cycles

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  16×16×16 decomposition onto 4×4×4 hardware:                           │
│                                                                         │
│  K-dimension (reduction): 16 ÷ 4 = 4 steps                             │
│  M-dimension (rows):      16 ÷ 4 = 4 tiles                             │
│  N-dimension (cols):      16 ÷ 4 = 4 tiles                             │
│                                                                         │
│  Total tiles: 4 × 4 = 16 output tiles                                  │
│  Each tile: 4 K-steps                                                  │
│  Total 4×4×4 operations: 16 × 4 = 64 cycles (minimum, idealized)       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Why 4×4?

The choice is driven by practical constraints:

```
Area and Power Analysis:

4×4 array:
  - 16 PEs
  - 16 multipliers + 16 accumulators
  - Manageable routing for operand distribution
  - Fits within SM partition area budget

16×16 array (hypothetical):
  - 256 PEs
  - Would dominate the entire SM area
  - Operand distribution network would be enormous
  - Most workloads couldn't keep it fed
```

### Tensor core throughput calculation

Per SM (Ampere A100):

```
Ampere A100 Tensor Core Configuration:

Per SM:
  - 4 tensor cores (one per partition)
  - Each tensor core: 4×4 array
  - Per-cycle per tensor core: 64 FP16 FMAs
  - Per-cycle per SM: 4 × 64 = 256 FP16 FMAs

At 1.41 GHz boost:
  - Per SM: 256 × 2 × 1.41 GHz = 722 TFLOPS FP16 (×2 for FMA = 2 ops)
  
Wait, that's per SM. A100 has 108 SMs:
  - Total: 722 × 108 ≈ 78 TFLOPS... 

But A100 claims 312 TFLOPS FP16 Tensor. The discrepancy?
  - Ampere tensor cores can do 2 FP16 FMAs per PE per cycle (packed)
  - Actual: 256 × 2 (FMA) × 2 (packed) × 1.41 GHz × 108 = ~156 TFLOPS
  
Still not 312? NVIDIA counts FMA as 2 ops AND has sparse mode (2:4 sparsity):
  - Dense: 156 TFLOPS
  - Sparse (2× throughput): 312 TFLOPS
```

The 4×4 array does more work per cycle through data packing, not physical array size.

## AMD CDNA Architecture

AMD takes a different approach with their Matrix Cores.

### CDNA 1 (MI100)

```
AMD MI100 Matrix Core:

Compute Unit Structure:
  - 4 SIMD units per CU
  - Each SIMD: 16 lanes (wave64 = 4 cycles to complete)
  - Matrix Core attached to each SIMD pair

Matrix Core Physical Array: 4×4×4
  - 4×4 spatial array
  - 4-element dot product per PE
  - 64 FMAs per cycle per matrix core

Per CU: 2 matrix cores × 64 = 128 FMAs/cycle

MFMA Instruction (Matrix Fused Multiply-Add):
  mfma_f32_32x32x8_f16: 
    - 32×32 output tile
    - K=8 reduction
    - Processed over multiple cycles on 4×4 hardware
```

### CDNA 2 (MI200 series)

```
AMD MI200 Matrix Core (improved):

Physical Array: 4×4×4 (same size, higher frequency + efficiency)

Key architectural change: Unified memory architecture
  - HBM directly addressable by matrix core
  - Infinity Fabric for multi-die coherence

Per-GCD (one die of MI250X):
  - 110 CUs
  - 220 matrix cores
  - 220 × 64 × 2 (FMA) × 1.7 GHz ≈ 47.9 TFLOPS FP16 per die
  - MI250X (2 dies): ~95.7 TFLOPS FP16 dense
```

### CDNA 3 (MI300)

```
AMD MI300X Matrix Core:

Physical Array: Believed to be 4×4×8 (doubled K-dimension)
  - 128 FMAs per cycle per matrix core
  - Or potentially 8×8×4 (different aspect ratio)

AMD hasn't disclosed exact dimensions, but throughput suggests doubling:

Per XCD (compute die):
  - 38 CUs 
  - 76 matrix cores
  
MI300X (8 XCDs):
  - 304 CUs
  - 608 matrix cores
  - Claimed: 1307 TFLOPS FP16
  
Back-calculating array size:
  1307 TFLOPS = 608 × array_ops × 2 (FMA) × 2.1 GHz
  array_ops = 1307 / (608 × 2 × 2.1) ≈ 512 ops/cycle
  
  512 = 8×8×8 or 4×4×32 or similar factorization
  Most likely: 8×8×8 or 4×4 array with 8-way K-unrolling
```

### AMD AccVGPR: The Key Differentiator

AMD's architecture separates accumulator storage:

```
AMD Register Architecture for Matrix Ops:
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  Standard VGPR (Vector General Purpose Registers):                      │
│  - 256 registers per SIMD (wave)                                       │
│  - Used for A and B matrix fragments                                   │
│  - Standard banked access                                              │
│                                                                         │
│  AccVGPR (Accumulator VGPR):                                           │
│  - Separate register file                                              │
│  - Used for C and D matrix fragments                                   │
│  - Dedicated paths to/from matrix core                                 │
│  - Reduces writeback congestion on main VGPR                          │
│                                                                         │
│  Data flow:                                                             │
│                                                                         │
│  VGPR ──────▶ Matrix Core ──────▶ AccVGPR                              │
│  (A, B)           │              (D output)                            │
│                   │                                                     │
│           AccVGPR ┘                                                     │
│           (C input)                                                     │
│                                                                         │
│  Accumulation stays in AccVGPR:                                        │
│    mfma D, A, B, C    ; C read from AccVGPR, D written to AccVGPR     │
│    mfma D, A2, B2, D  ; D reused as accumulator (no VGPR traffic)     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

This eliminates the writeback bottleneck we discussed for NVIDIA. The accumulator never needs to traverse the main register file crossbar during a chain of matrix operations.

## Intel Xe Matrix Extensions (XMX)

Intel's approach in their Xe-HPC (Ponte Vecchio) and Xe-HPG (Arc) architectures:

### Xe-HPC (Data Center Max / Ponte Vecchio)

```
Intel Xe-HPC Matrix Engine:

Structure: Systolic array per Xe-core

Physical Array: 8×8
  - 8 rows × 8 columns of PEs
  - Each PE: 8-element dot product (for FP16/BF16)
  - Per cycle: 8×8×8 = 512 FP16 FMAs per matrix engine

Xe-core organization:
  - 8 Vector Engines (like CUDA cores)
  - 8 Matrix Engines (XMX units)
  
Wait, this differs from consumer Xe. Let me separate:

Xe-HPG (Arc consumer):
  - XMX array: 8×2 systolic (16 MACs per XMX)
  - Smaller, power-constrained design
  
Xe-HPC (Ponte Vecchio):
  - Larger XMX arrays: 8×8 believed
  - Throughput: ~420 TFLOPS FP16 per stack
```

### Intel's Systolic Approach

Intel explicitly uses the term "systolic" and their implementation is closer to the textbook definition:

```
Intel XMX Systolic Data Flow:
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│     A elements (broadcast down columns)                                 │
│          │   │   │   │   │   │   │   │                                 │
│          ▼   ▼   ▼   ▼   ▼   ▼   ▼   ▼                                 │
│        ┌───┬───┬───┬───┬───┬───┬───┬───┐                               │
│   B ──▶│   │   │   │   │   │   │   │   │                               │
│        ├───┼───┼───┼───┼───┼───┼───┼───┤                               │
│   B ──▶│   │   │   │   │   │   │   │   │                               │
│        ├───┼───┼───┼───┼───┼───┼───┼───┤                               │
│   B ──▶│   │   │   │   │   │   │   │   │                               │
│        ├───┼───┼───┼───┼───┼───┼───┼───┤                               │
│   ...  │   │   │   │   │   │   │   │   │                               │
│        ├───┼───┼───┼───┼───┼───┼───┼───┤                               │
│   B ──▶│   │   │   │   │   │   │   │   │──▶ Output row                 │
│        └───┴───┴───┴───┴───┴───┴───┴───┘                               │
│                                                                         │
│   Each PE accumulates partial results                                   │
│   Output streams out after K cycles                                    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Direct Comparison Table

```
┌────────────────────┬─────────────────┬─────────────────┬─────────────────┐
│                    │ NVIDIA          │ AMD             │ Intel           │
│                    │ (Ampere/Ada)    │ (CDNA2/3)       │ (Xe-HPC)        │
├────────────────────┼─────────────────┼─────────────────┼─────────────────┤
│ Array Dimensions   │ 4×4             │ 4×4 (CDNA2)     │ 8×8             │
│                    │                 │ ~8×8 (CDNA3)    │                 │
├────────────────────┼─────────────────┼─────────────────┼─────────────────┤
│ Dot Product Depth  │ 4 (FP16)        │ 4-8             │ 8               │
│ (K per cycle)      │                 │                 │                 │
├────────────────────┼─────────────────┼─────────────────┼─────────────────┤
│ FMAs/cycle/unit    │ 64              │ 64-128          │ 512             │
├────────────────────┼─────────────────┼─────────────────┼─────────────────┤
│ Units per SM/CU    │ 4 tensor cores  │ 2 matrix cores  │ 8 XMX engines   │
│                    │ per SM          │ per CU          │ per Xe-core     │
├────────────────────┼─────────────────┼─────────────────┼─────────────────┤
│ Accumulator        │ Shared with     │ Separate        │ Separate        │
│ Storage            │ main register   │ AccVGPR         │ accumulator     │
│                    │ file            │                 │ registers       │
├────────────────────┼─────────────────┼─────────────────┼─────────────────┤
│ Data Flow          │ Quasi-systolic  │ Systolic        │ True systolic   │
│                    │ (output         │                 │                 │
│                    │ stationary)     │                 │                 │
├────────────────────┼─────────────────┼─────────────────┼─────────────────┤
│ Operand Source     │ Register file   │ VGPR + AccVGPR  │ Register file   │
│                    │ (unified)       │ (split)         │ + accumulators  │
├────────────────────┼─────────────────┼─────────────────┼─────────────────┤
│ Native Tile Size   │ 16×16×16        │ 32×32×8 (vary)  │ 8×8×K           │
│ (instruction)      │ (or 8×8×16)     │ 16×16×16        │ (K varies)      │
├────────────────────┼─────────────────┼─────────────────┼─────────────────┤
│ Top Product        │ H100: 990       │ MI300X: 1307    │ PVC: ~420       │
│ FP16 TFLOPS        │ (dense)         │ (dense)         │ per stack       │
│                    │                 │                 │ (~840 2-stack)  │
└────────────────────┴─────────────────┴─────────────────┴─────────────────┘
```

## Dataflow Strategies

The three vendors use different dataflow optimizations:

### NVIDIA: Output Stationary

```
Output Stationary (NVIDIA):
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  Accumulator stays fixed in PE                                         │
│  A and B elements stream through                                       │
│                                                                         │
│  Advantage: Accumulator doesn't move, reducing write traffic           │
│  Disadvantage: A and B must be delivered repeatedly                    │
│                                                                         │
│  Timeline for one output tile:                                         │
│                                                                         │
│  Cycle 0: A[*,0:3], B[0:3,*] → partial sum in accumulator             │
│  Cycle 1: A[*,4:7], B[4:7,*] → accumulate                             │
│  Cycle 2: A[*,8:11], B[8:11,*] → accumulate                           │
│  Cycle 3: A[*,12:15], B[12:15,*] → accumulate, output ready           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### AMD: Weight Stationary (variant)

```
Weight Stationary (AMD variant):
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  For inference workloads, B (weights) can be preloaded                 │
│  A (activations) streams through                                       │
│                                                                         │
│  MFMA instructions support flexible reuse patterns                     │
│  AccVGPR allows accumulator to persist without main RF pressure        │
│                                                                         │
│  Key: software (rocBLAS) chooses tiling strategy                       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Intel: Row Stationary

```
Row Stationary (Intel XMX):
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  Each row of the systolic array handles one output row                 │
│  B elements flow left-to-right through the row                        │
│  A elements broadcast down                                             │
│                                                                         │
│  Maximizes data reuse for both A and B                                 │
│  Matches well with transformer attention patterns                      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Why the Small Arrays?

You might wonder why everyone uses relatively small arrays (4×4 to 8×8) rather than massive systolic structures. The reasons:

```
Scaling Challenges:
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│ 1. Operand Bandwidth                                                    │
│    - 4×4 array needs 4+4 operands/cycle per PE row/column               │
│    - 16×16 array needs 16+16 operands/cycle                             │
│    - Register file can't scale to feed larger arrays                    │
│                                                                         │
│ 2. Utilization                                                          │
│    - Small matrices (batch size 1, small hidden dims) waste capacity    │
│    - 4×4 has better utilization across workload sizes                   │
│                                                                         │
│ 3. Area/Power                                                           │
│    - Array area scales as N²                                            │
│    - 4× larger array = 16× area, but not 16× throughput (starved)       │
│                                                                         │
│ 4. Latency                                                              │
│    - Larger array = more cycles to fill and drain                       │
│    - Small batches see high overhead                                    │
│                                                                         │
│ 5. Wire Delay                                                           │
│    - Signals must traverse array in one cycle                           │
│    - Larger array = longer wires = lower frequency or more pipelining   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Hopper (H100) Update

NVIDIA's Hopper made changes worth noting:

```
H100 Tensor Core Changes:

Physical array: Still 4×4 base, but with modifications

New capability: TMA (Tensor Memory Accelerator)
  - Dedicated unit for async tensor loads
  - Bypasses register file for some operands
  - Feeds tensor core staging buffer directly from shared memory
  
New instruction: WGMMA (Warpgroup Matrix Multiply-Accumulate)
  - Operates on 4 warps cooperatively (128 threads)
  - Larger logical tiles (64×N×16)
  - Better amortizes operand delivery overhead

Effect: Higher sustained throughput by reducing RF bottleneck
```

The trend across vendors is clear: the physical array size is limited by operand delivery, so innovation focuses on **reducing operand delivery overhead** rather than building bigger arrays.

# Operand Delivery

This is where the architectural philosophies really diverge. Each vendor has made fundamentally different tradeoffs in how they feed their matrix units.

## The Core Problem

Before diving into solutions, let's quantify the challenge:

```
Operand Delivery Requirements (4×4×4 array, FP16):

Per cycle, the array needs:
  - 4 A elements × 4 rows = 16 FP16 values = 256 bits
  - 4 B elements × 4 cols = 16 FP16 values = 256 bits
  - Total input: 512 bits/cycle

For sustained operation at 1.5 GHz:
  - 512 bits × 1.5 GHz = 768 Gbits/sec = 96 GB/s per tensor core
  - 4 tensor cores per SM = 384 GB/s per SM just for tensor operands

For comparison:
  - A100 HBM bandwidth: 2 TB/s total
  - 108 SMs × 384 GB/s = 41 TB/s demand (20× available!)

The only solution: operand reuse from register files and caches
```

## NVIDIA's Approach: Unified Register File with Staging

NVIDIA uses the standard vector register file with a dedicated staging buffer:

### Architecture

```
NVIDIA Operand Delivery (Ampere/Ada):
┌────────────────────────────────────────────────────────────────────────┐
│                                                                        │
│                      Vector Register File                              │
│                    (256KB per SM, banked)                              │
│                             │                                          │
│         ┌───────────────────┼───────────────────┐                      │
│         │                   │                   │                      │
│         ▼                   ▼                   ▼                      │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                 │
│  │  Standard   │    │  Standard   │    │   Tensor    │                 │
│  │  Collector  │    │  Collector  │    │  Staging    │                 │
│  │   (scalar)  │    │   (scalar)  │    │   Buffer    │                 │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘                 │
│         │                  │                  │                        │
│         ▼                  ▼                  │                        │
│  ┌─────────────┐    ┌─────────────┐          │                         │
│  │  FP32/INT   │    │    SFU     │           │                         │
│  │    Units    │    │            │           │                         │
│  └─────────────┘    └─────────────┘          │                         │
│                                              │                         │
│                     ┌────────────────────────┘                         │
│                     │                                                  │
│                     ▼                                                  │
│         ┌───────────────────────┐                                      │
│         │  Operand Reformatter  │                                      │
│         │                       │                                      │
│         │  Thread-to-PE mapping │                                      │
│         │  Lane swizzling       │                                      │
│         └───────────┬───────────┘                                      │
│                     │                                                  │
│                     ▼                                                  │
│         ┌───────────────────────┐                                      │
│         │     Tensor Core       │                                      │
│         │     4×4 Array         │                                      │
│         └───────────┬───────────┘                                      │
│                     │                                                  │
│                     ▼                                                  │
│         ┌───────────────────────┐                                      │
│         │  Result Reformatter   │                                      │
│         │  PE-to-thread mapping │                                      │
│         └───────────┬───────────┘                                      │
│                     │                                                  │
│                     ▼                                                  │
│              Register File                                             │
│              (writeback)                                               │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

### The Staging Buffer Detail

```
Tensor Staging Buffer Organization:
┌────────────────────────────────────────────────────────────────────────┐
│                                                                        │
│  Double-Buffered Design (ping-pong):                                   │
│                                                                        │
│  Buffer A (Active - feeding tensor core):                              │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  A Matrix Fragment:  [16×16 FP16] = 512 bytes                   │   │
│  │  B Matrix Fragment:  [16×16 FP16] = 512 bytes                   │   │
│  │  C Matrix Fragment:  [16×16 FP32] = 1024 bytes                  │   │
│  │                                                                 │   │
│  │  Status: EXECUTING                                              │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                        │
│  Buffer B (Filling - gathering next operands):                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  A Matrix Fragment:  [████████████░░░░░░░░] 60% gathered        │   │
│  │  B Matrix Fragment:  [████████████████████] 100% ready          │   │
│  │  C Matrix Fragment:  [████░░░░░░░░░░░░░░░░] 20% gathered        │   │
│  │                                                                 │   │
│  │  Status: GATHERING                                              │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                        │
│  Total staging buffer: ~4KB per tensor core                            │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

### The Gathering Process

```
NVIDIA Operand Gather Timeline:
┌────────────────────────────────────────────────────────────────────────┐
│                                                                        │
│  WMMA instruction: D = A × B + C                                       │
│  Fragment layout: each thread holds multiple registers                 │
│                                                                        │
│  Thread 0:  A[R0,R1,R2,R3]  B[R4,R5,R6,R7]  C[R8-R15]                  │
│  Thread 1:  A[R0,R1,R2,R3]  B[R4,R5,R6,R7]  C[R8-R15]                  │
│  ...                                                                   │
│  Thread 31: A[R0,R1,R2,R3]  B[R4,R5,R6,R7]  C[R8-R15]                  │
│                                                                        │
│  Gather sequence (simplified, 16 banks):                               │
│                                                                        │
│  Cycles 1-2:   Read R0 across all 32 threads (2 threads/bank)          │
│  Cycles 3-4:   Read R1 across all 32 threads                           │
│  Cycles 5-6:   Read R2 across all 32 threads                           │
│  Cycles 7-8:   Read R3 across all 32 threads                           │
│  ... (A fragment complete after ~8 cycles)                             │
│                                                                        │
│  Cycles 9-16:  Read R4-R7 (B fragment)                                 │
│  Cycles 17-32: Read R8-R15 (C fragment, FP32 = 2× registers)           │
│                                                                        │
│  Total gather: ~32 cycles minimum (more with bank conflicts)           │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

### Operand Reformatter

The critical piece that maps thread-private registers to systolic positions:

```
NVIDIA Operand Reformatter:
┌────────────────────────────────────────────────────────────────────────┐
│                                                                        │
│  Input: 32 threads × 4 registers (A fragment) = 128 values             │
│         Organized as: T0[R0,R1,R2,R3], T1[R0,R1,R2,R3], ...            │
│                                                                        │
│  Required output: 16×16 matrix in systolic-friendly order              │
│                                                                        │
│  The reformatter is a fixed permutation network:                       │
│                                                                        │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                                                                  │  │
│  │   Thread registers         Permutation          Systolic input   │  │
│  │                             Network                              │  │
│  │   T0:R0 ─────────────────────────────────────────> A[0,0]        │  │
│  │   T0:R1 ─────────────────────────────────────────> A[0,1]        │  │
│  │   T0:R2 ─────────────────────────────────────────> A[0,2]        │  │
│  │   T0:R3 ─────────────────────────────────────────> A[0,3]        │  │
│  │   T1:R0 ─────────────────────────────────────────> A[0,4]        │  │
│  │   T1:R1 ─────────────────────────────────────────> A[0,5]        │  │
│  │   ...                                                            │  │
│  │   T16:R0 ────────────────────────────────────────> A[8,0]        │  │
│  │   ...                                                            │  │
│  │                                                                  │  │
│  │   Hardwired permutation (no configurability)                     │  │
│  │   This is why fragment layout is fixed by NVIDIA                 │  │
│  │                                                                  │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

### NVIDIA Bottlenecks

```
NVIDIA Operand Delivery Bottlenecks:

1. Register File Bandwidth
   - 16 banks, each can service 1 read/cycle
   - 32 threads × N registers = many cycles to gather
   - Bank conflicts extend this further

2. Unified Register Pressure
   - A, B, C fragments compete with scalar code for registers
   - High register usage = low occupancy = less latency hiding

3. Writeback Contention  
   - D fragment (8 regs × 32 threads) must write back
   - Shares writeback path with scalar units
   - Can stall subsequent WMMA instructions

4. No Accumulator Persistence
   - Each WMMA must read C and write D through register file
   - Even when D becomes C for next WMMA in chain
```

## AMD's Approach: Split Register Files with AccVGPR

AMD fundamentally restructured their register hierarchy:

### Architecture

```
AMD CDNA Operand Delivery:
┌────────────────────────────────────────────────────────────────────────┐
│                                                                        │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    VGPR (Vector GPR)                            │   │
│  │                   256 registers × 64 lanes                      │   │
│  │                   Used for: A and B fragments                   │   │
│  └──────────────────────────────┬──────────────────────────────────┘   │
│                                 │                                      │
│                                 │  (A, B operands)                     │
│                                 ▼                                      │
│                    ┌─────────────────────────┐                         │
│                    │   Matrix Core Input     │                         │
│                    │   Crossbar              │                         │
│                    └────────────┬────────────┘                         │
│                                 │                                      │
│  ┌──────────────────────────────┼──────────────────────────────────┐   │
│  │                              ▼                                  │   │
│  │                    ┌─────────────────┐                          │   │
│  │                    │  Matrix Core    │                          │   │
│  │                    │  4×4 Systolic   │                          │   │
│  │                    └────────┬────────┘                          │   │
│  │                             │                                   │   │
│  │                             ▼                                   │   │
│  │  ┌──────────────────────────────────────────────────────────┐   │   │
│  │  │                    AccVGPR                               │   │   │
│  │  │              Accumulator Register File                   │   │   │
│  │  │                                                          │   │   │
│  │  │   - Dedicated to matrix accumulation                     │   │   │
│  │  │   - Direct path to/from matrix core                      │   │   │
│  │  │   - No arbitration with scalar ops                       │   │   │
│  │  │   - Persistent across MFMA chains                        │   │   │
│  │  │                                                          │   │   │
│  │  │   C input ──> Matrix Core ──> D output                   │   │   │
│  │  │      ▲                            │                      │   │   │
│  │  │      └────────────────────────────┘                      │   │   │
│  │  │         (D becomes C for next MFMA)                      │   │   │
│  │  │                                                          │   │   │
│  │  └──────────────────────────────────────────────────────────┘   │   │
│  │                                                                 │   │
│  │              Matrix Core + AccVGPR Domain                       │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                        │
│  Data movement to extract results:                                     │
│                                                                        │
│  AccVGPR ──────> VGPR (explicit v_accvgpr_read instruction)            │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

### AccVGPR Details

```
AccVGPR Architecture:
┌────────────────────────────────────────────────────────────────────────┐
│                                                                        │
│  Physical Organization:                                                │
│                                                                        │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  AccVGPR Bank 0   │  AccVGPR Bank 1   │  AccVGPR Bank 2   │ ... │   │
│  │                   │                   │                   │     │   │
│  │  [A0] [A4] [A8]   │  [A1] [A5] [A9]   │  [A2] [A6] [A10]  │     │   │
│  │  ...              │  ...              │  ...              │     │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                        │
│  Capacity: 256 AccVGPRs per wavefront (CDNA2)                          │
│            Same as VGPR count                                          │
│                                                                        │
│  Access Characteristics:                                               │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                                                                 │   │
│  │  Path                          │  Latency    │  Bandwidth       │   │
│  │  ─────────────────────────────────────────────────────────────  │   │
│  │  AccVGPR → Matrix Core (C in)  │  ~4 cycles  │  Very high       │   │
│  │  Matrix Core → AccVGPR (D out) │  ~4 cycles  │  Very high       │   │
│  │  AccVGPR → VGPR (read out)     │  ~8 cycles  │  Medium          │   │
│  │  VGPR → AccVGPR (initialize)   │  ~8 cycles  │  Medium          │   │
│  │                                                                 │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                        │
│  Key insight: AccVGPR↔Matrix Core path is SHORT and WIDE               │
│               AccVGPR↔VGPR path is LONGER but used infrequently        │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

### AMD Operand Gather

```
AMD MFMA Operand Gather:
┌────────────────────────────────────────────────────────────────────────┐
│                                                                        │
│  Instruction: v_mfma_f32_32x32x8_f16 D, A, B, C                        │
│                                                                        │
│  A operands (from VGPR):                                               │
│  - 4 VGPRs across 64 lanes = 256 FP16 values                           │
│  - Gather from VGPR banks                                              │
│  - Similar banking to NVIDIA, but wave64 helps                         │
│                                                                        │
│  B operands (from VGPR):                                               │
│  - 4 VGPRs across 64 lanes = 256 FP16 values                           │
│  - Same gather mechanism                                               │
│                                                                        │
│  C operands (from AccVGPR):                                            │
│  - Already in place from previous MFMA (typically)                     │
│  - OR initialized via v_accvgpr_write                                  │
│  - NO VGPR bank contention for C!                                      │
│                                                                        │
│  Timeline comparison:                                                  │
│                                                                        │
│  NVIDIA (all from shared RF):                                          │
│  [────── A gather ──────][────── B gather ──────][────── C gather ────]│
│  │                                                                     │
│  AMD (C from AccVGPR):                                                 │
│  [────── A gather ──────]                                              │
│  [────── B gather ──────] (can overlap with A if no bank conflict)     │
│  [── C ready instantly ─] (already in AccVGPR)                         │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

### AMD's MFMA Instruction Variants

AMD offers multiple MFMA shapes, each with different operand patterns:

```
AMD MFMA Instruction Family (CDNA2):
┌────────────────────────────────────────────────────────────────────────┐
│                                                                        │
│  Instruction              │ Output  │ A regs │ B regs │ C/D regs       │
│  ───────────────────────────────────────────────────────────────────── │
│  v_mfma_f32_32x32x8_f16   │ 32×32   │ 4      │ 4      │ 16 (AccVGPR)   │
│  v_mfma_f32_16x16x16_f16  │ 16×16   │ 4      │ 4      │ 4 (AccVGPR)    │
│  v_mfma_f32_4x4x4_f16     │ 4×4     │ 2      │ 2      │ 4 (AccVGPR)    │
│  v_mfma_f32_32x32x4_f16   │ 32×32   │ 2      │ 2      │ 16 (AccVGPR)   │
│                                                                        │
│  Flexibility allows software to choose:                                │
│  - Large tiles: better compute density, higher register pressure       │
│  - Small tiles: lower latency, better for small matrices               │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

### AMD Advantages

```
AMD Operand Delivery Advantages:
┌────────────────────────────────────────────────────────────────────────┐
│                                                                        │
│ 1. Accumulator Persistence                                             │
│    - Chained MFMAs: D[i] = A×B + D[i-1]                                │
│    - D stays in AccVGPR, becomes C for next MFMA                       │
│    - Zero VGPR traffic for accumulator in steady state                 │
│                                                                        │
│    MFMA chain:                                                         │
│    mfma D, A0, B0, C      ; C from AccVGPR (initialized)               │
│    mfma D, A1, B1, D      ; D read as C, updated D written             │
│    mfma D, A2, B2, D      ; same AccVGPRs, no external traffic         │
│    mfma D, A3, B3, D      ;                                            │
│    v_accvgpr_read V, D    ; only now move to VGPR (once at end)        │
│                                                                        │
│ 2. Reduced Bank Conflicts                                              │
│    - Only A and B compete for VGPR banks                               │
│    - C/D on separate physical structure                                │
│    - Wave64: more threads = better bank coverage                       │
│                                                                        │
│ 3. Independent Scheduling                                              │
│    - AccVGPR writes don't block VGPR reads                             │
│    - Matrix core can operate more independently                        │
│                                                                        │
│ 4. Lower Register Pressure Impact                                      │
│    - AccVGPRs are "free" (separate allocation)                         │
│    - More VGPRs available for A, B staging and scalar code             │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

## Intel's Approach: Systolic with Explicit Staging

Intel takes yet another approach with dedicated operand staging in their XMX:

### Architecture

```
Intel Xe-HPC XMX Operand Delivery:
┌────────────────────────────────────────────────────────────────────────┐
│                                                                        │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                   General Register File (GRF)                   │   │
│  │                  128 registers × 512 bits (64 bytes)            │   │
│  └────────────────────────────┬────────────────────────────────────┘   │
│                               │                                        │
│           ┌───────────────────┼───────────────────┐                    │
│           │                   │                   │                    │
│           ▼                   ▼                   ▼                    │
│  ┌─────────────────┐ ┌─────────────────┐ ┌───────────────────┐         │
│  │   EU Pipe 0     │ │   EU Pipe 1     │ │     XMX Pipe      │         │
│  │   (FP/INT)      │ │   (FP/INT)      │ │                   │         │
│  └─────────────────┘ └─────────────────┘ │  ┌─────────────┐  │         │
│                                          │  │   Operand   │  │         │
│                                          │  │   Staging   │  │         │
│                                          │  │   Regs      │  │         │
│                                          │  └──────┬──────┘  │         │
│                                          │         │         │         │
│                                          │         ▼         │         │
│                                          │  ┌─────────────┐  │         │
│                                          │  │    8×8      │  │         │
│                                          │  │   Systolic  │  │         │
│                                          │  │    Array    │  │         │
│                                          │  └──────┬──────┘  │         │
│                                          │         │         │         │
│                                          │         ▼         │         │
│                                          │  ┌─────────────┐  │         │
│                                          │  │ Accumulator │  │         │
│                                          │  │    Regs     │  │         │
│                                          │  └─────────────┘  │         │
│                                          └───────────────────┘         │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

### Intel DPAS Instruction

Intel uses DPAS (Dot Product Accumulate Systolic) instructions:

```
Intel DPAS Instruction:
┌────────────────────────────────────────────────────────────────────────┐
│                                                                        │
│  dpas.8x8 (systolic_depth) dst, src0, src1, src2                       │
│                                                                        │
│  dst:  Accumulator destination (8×8 output tile)                       │
│  src0: Accumulator input (same registers as dst typically)             │
│  src1: A matrix fragment                                               │
│  src2: B matrix fragment                                               │
│                                                                        │
│  Systolic depth options: 1, 2, 4, 8                                    │
│  - Depth 8: 8 elements in K dimension per instruction                  │
│  - Higher depth = more work per instruction, but more registers        │
│                                                                        │
│  Example: dpas.8x8 (8) r32:8, r32:8, r16:8, r24:8                      │
│                                                                        │
│  Operand layout:                                                       │
│  - src1 (A): 8 rows × 8 elements = 64 values = 8 registers             │
│  - src2 (B): 8 cols × 8 elements = 64 values = 8 registers             │
│  - dst/src0: 8×8 = 64 values = 8 registers                             │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

### Intel Operand Delivery Mechanism

```
Intel XMX Operand Flow:
┌────────────────────────────────────────────────────────────────────────┐
│                                                                        │
│  DPAS Execution Timeline:                                              │
│                                                                        │
│  Cycle 0:   DPAS instruction issued                                    │
│  Cycle 1:   src1 (A) registers read from GRF                           │
│  Cycle 2:   src2 (B) registers read from GRF                           │
│  Cycle 3:   src0 (C) accumulator read (or forwarded)                   │
│  Cycle 4-11: Systolic computation (8 cycles for depth-8)               │
│  Cycle 12:  Result available in accumulator                            │
│                                                                        │
│  Key features:                                                         │
│                                                                        │
│  1. Wide GRF: 512-bit registers (vs 32-bit in NVIDIA/AMD)              │
│     - Fewer register reads needed                                      │
│     - 8 FP16 values per register read                                  │
│                                                                        │
│  2. Explicit systolic depth control                                    │
│     - Compiler chooses depth based on register pressure                │
│     - Lower depth = lower latency, less register usage                 │
│                                                                        │
│  3. Hardware accumulator forwarding                                    │
│     - Chained DPAS instructions bypass GRF for accumulator             │
│     - Similar benefit to AMD's AccVGPR                                 │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

### Intel's Wide Register Advantage

```
Intel GRF Organization:
┌────────────────────────────────────────────────────────────────────────┐
│                                                                        │
│  Single GRF Register: 512 bits = 64 bytes                              │
│                                                                        │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ FP16: 32 values per register                                    │   │
│  │ ┌────┬────┬────┬────┬────┬────┬────┬────┬─────────────┬────┐    │   │
│  │ │ 0  │ 1  │ 2  │ 3  │ 4  │ 5  │ 6  │ 7  │    ...      │ 31 │    │   │
│  │ └────┴────┴────┴────┴────┴────┴────┴────┴─────────────┴────┘    │   │
│  │                                                                 │   │
│  │ FP32: 16 values per register                                    │   │
│  │ ┌────────┬────────┬────────┬────────┬───────────┬────────┐      │   │
│  │ │   0    │   1    │   2    │   3    │    ...    │   15   │      │   │
│  │ └────────┴────────┴────────┴────────┴───────────┴────────┘      │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                        │
│  Impact on operand delivery:                                           │
│                                                                        │
│  8×8 A matrix (FP16) = 64 values = 2 register reads                    │
│  8×8 B matrix (FP16) = 64 values = 2 register reads                    │
│  8×8 C matrix (FP32) = 64 values = 4 register reads                    │
│                                                                        │
│  vs NVIDIA (32-bit registers):                                         │
│  8×8 A matrix = 64 values = 32 register reads (across threads)         │
│                                                                        │
│  Wide registers dramatically reduce read port pressure                 │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

## Comparative Analysis

### Operand Bandwidth Comparison

```
Operand Delivery Bandwidth:
┌────────────────────────────────────────────────────────────────────────┐
│                                                                        │
│  Metric: Time to deliver operands for one matrix op                    │
│                                                                        │
│                    │ NVIDIA        │ AMD           │ Intel             │
│  ──────────────────────────────────────────────────────────────────────│
│  A matrix reads    │ ~8 cycles     │ ~8 cycles     │ ~2 cycles         │
│  B matrix reads    │ ~8 cycles     │ ~8 cycles     │ ~2 cycles         │
│  C matrix reads    │ ~16 cycles    │ ~0 (AccVGPR)  │ ~4 cycles         │
│  ──────────────────────────────────────────────────────────────────────│
│  Total (no overlap)│ ~32 cycles    │ ~16 cycles    │ ~8 cycles         │
│  With pipelining   │ ~16 cycles    │ ~8 cycles     │ ~4 cycles         │
│                                                                        │
│  Note: Actual numbers vary with bank conflicts and microarchitecture   │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

### Energy per Operand Delivery

```
Energy Analysis (Estimated):
┌───────────────────────────────────────────────────────────────────────┐
│                                                                       │
│  NVIDIA (Unified RF):                                                 │
│  ┌────────────────────────────────────────────────────────────────┐   │
│  │  A operands: 128 reg reads × ~0.5 pJ/read = ~64 pJ             │   │
│  │  B operands: 128 reg reads × ~0.5 pJ/read = ~64 pJ             │   │
│  │  C operands: 256 reg reads × ~0.5 pJ/read = ~128 pJ            │   │
│  │  D writeback: 256 reg writes × ~1 pJ/write = ~256 pJ           │   │
│  │  Crossbar/routing: ~100 pJ                                     │   │
│  │  ──────────────────────────────────────────────────────────    │   │
│  │  Total: ~612 pJ per WMMA operand delivery                      │   │
│  └────────────────────────────────────────────────────────────────┘   │
│                                                                       │
│  AMD (Split RF with AccVGPR):                                         │
│  ┌────────────────────────────────────────────────────────────────┐   │
│  │  A operands: 128 reg reads × ~0.5 pJ/read = ~64 pJ             │   │
│  │  B operands: 128 reg reads × ~0.5 pJ/read = ~64 pJ             │   │
│  │  C operands: ~0 pJ (already in AccVGPR, local read)            │   │
│  │  D writeback: ~10 pJ (AccVGPR local write)                     │   │
│  │  Crossbar/routing: ~50 pJ (A,B only)                           │   │
│  │  ──────────────────────────────────────────────────────────    │   │
│  │  Total: ~188 pJ per MFMA operand delivery                      │   │
│  │                                                                │   │
│  │  Savings: ~70% vs NVIDIA for chained operations                │   │
│  └────────────────────────────────────────────────────────────────┘   │
│                                                                       │
│  Intel (Wide RF + Accumulator):                                       │
│  ┌────────────────────────────────────────────────────────────────┐   │
│  │  A operands: 2 reg reads × ~2 pJ/read = ~4 pJ                  │   │
│  │  B operands: 2 reg reads × ~2 pJ/read = ~4 pJ                  │   │
│  │  C operands: ~5 pJ (accumulator forwarding)                    │   │
│  │  D writeback: ~5 pJ (accumulator local)                        │   │
│  │  Routing: ~20 pJ (simpler due to wide regs)                    │   │
│  │  ──────────────────────────────────────────────────────────    │   │
│  │  Total: ~38 pJ per DPAS operand delivery                       │   │
│  │                                                                │   │
│  │  Note: Intel's larger array (8×8) amortizes this better        │   │
│  └────────────────────────────────────────────────────────────────┘   │
│                                                                       │
│  Caveat: These are rough estimates. Actual values depend on process,  │
│  voltage, temperature, and specific microarchitectural details.       │
│                                                                       │
└───────────────────────────────────────────────────────────────────────┘
```

### Flexibility vs Efficiency Tradeoffs

```
Design Tradeoff Summary:
┌────────────────────────────────────────────────────────────────────────┐
│                                                                        │
│  NVIDIA (Unified RF):                                                  │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │  Pros:                                                         │    │
│  │  + Maximum flexibility: any register usable for any purpose    │    │
│  │  + Simpler programming model                                   │    │
│  │  + Scalar and tensor code share resources naturally            │    │
│  │                                                                │    │
│  │  Cons:                                                         │    │
│  │  - Highest operand delivery overhead                           │    │
│  │  - Writeback contention between tensor and scalar              │    │
│  │  - Register pressure directly impacts occupancy                │    │
│  └────────────────────────────────────────────────────────────────┘    │
│                                                                        │
│  AMD (AccVGPR):                                                        │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │  Pros:                                                         │    │
│  │  + Excellent efficiency for chained matrix ops                 │    │
│  │  + Reduced VGPR pressure for matrix-heavy code                 │    │
│  │  + No writeback contention for accumulator                     │    │
│  │                                                                │    │
│  │  Cons:                                                         │    │
│  │  - Two register namespaces to manage                           │    │
│  │  - AccVGPR↔VGPR transfers add latency                          │    │
│  │  - Less flexible for non-matrix workloads                      │    │
│  └────────────────────────────────────────────────────────────────┘    │
│                                                                        │
│  Intel (Wide RF):                                                      │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │  Pros:                                                         │    │
│  │  + Lowest operand read overhead                                │    │
│  │  + Explicit systolic depth control                             │    │
│  │  + Good balance of flexibility and efficiency                  │    │
│  │                                                                │    │
│  │  Cons:                                                         │    │
│  │  - Wider registers complicate partial access                   │    │
│  │  - Register spill cost is higher (64 bytes per spill)          │    │
│  │  - Less mature software ecosystem                              │    │
│  └────────────────────────────────────────────────────────────────┘    │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

## NVIDIA Hopper's TMA: A Hybrid Approach

NVIDIA recognized the operand delivery bottleneck and introduced TMA in Hopper:

```
Hopper TMA (Tensor Memory Accelerator):
┌────────────────────────────────────────────────────────────────────────┐
│                                                                        │
│  Traditional path (pre-Hopper):                                        │
│                                                                        │
│  Global Mem → L2 → L1/Shared → Registers → Tensor Core                 │
│              ▲                    ▲           ▲                        │
│              └── software managed ┴───────────┘                        │
│                                                                        │
│  TMA path (Hopper):                                                    │
│                                                                        │
│  Global Mem → L2 → TMA Engine → Shared Memory → Tensor Core            │
│                       │              │                                 │
│                       │              └── can bypass registers!         │
│                       │                                                │
│                       └── Async, address generation offloaded          │
│                                                                        │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  TMA Features:                                                  │   │
│  │                                                                 │   │
│  │  1. Hardware address generation                                 │   │
│  │     - Tensor descriptors define layout                          │   │
│  │     - No register-consuming address math                        │   │
│  │                                                                 │   │
│  │  2. Async copy to shared memory                                 │   │
│  │     - Fire-and-forget, hardware tracks completion               │   │
│  │     - Overlaps with computation                                 │   │
│  │                                                                 │   │
│  │  3. Direct shared memory → tensor core path                     │   │
│  │     - WGMMA instructions can read from shared memory            │   │
│  │     - Bypasses register file entirely for A/B matrices          │   │
│  │                                                                 │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                        │
│  Impact on operand delivery:                                           │
│                                                                        │
│  Before TMA:  HBM → RF → Tensor (512 pJ+)                              │
│  With TMA:    HBM → SMEM → Tensor (~200 pJ)                            │
│                                                                        │
│  NVIDIA effectively created a bypass for the RF bottleneck             │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

### WGMMA: Warpgroup Matrix Multiply

```
Hopper WGMMA Instruction:
┌───────────────────────────────────────────────────────────────────────┐
│                                                                       │
│  wgmma.mma_async.sync.aligned.m64n256k16.f16.f16.f32                  │
│       D, [A_smem_ptr], [B_smem_ptr], C                                │
│                                                                       │
│  Key differences from WMMA:                                           │
│                                                                       │
│  1. Operates on 4 warps (warpgroup = 128 threads)                     │
│  2. A and B read directly from shared memory                          │
│  3. Only C/D use registers                                            │
│  4. Async: doesn't stall issuing warp                                 │
│                                                                       │
│  Operand delivery:                                                    │
│  ┌────────────────────────────────────────────────────────────────┐   │
│  │                                                                │   │
│  │  A matrix: 64×16 FP16 = 2KB from shared memory                 │   │
│  │  B matrix: 16×256 FP16 = 8KB from shared memory                │   │
│  │  C matrix: from registers (or previous D)                      │   │
│  │  D matrix: to registers                                        │   │
│  │                                                                │   │
│  │  Shared memory bandwidth: 128 bytes/cycle per SM               │   │
│  │  Time to deliver A: 2KB / 128 B/cyc = 16 cycles                │   │
│  │  Time to deliver B: 8KB / 128 B/cyc = 64 cycles                │   │
│  │                                                                │   │
│  │  But: TMA prefetches overlap with compute                      │   │
│  │       Multiple WGMMA in flight hide latency                    │   │
│  │                                                                │   │
│  └────────────────────────────────────────────────────────────────┘   │
│                                                                       │
└───────────────────────────────────────────────────────────────────────┘
```

## Summary: Evolution of Operand Delivery

```
Historical Progression:
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  Gen 1 (Volta): Everything through unified register file               │
│  - Simple model                                                         │
│  - RF bandwidth is the bottleneck                                      │
│                                                                         │
│  Gen 2 (AMD CDNA): Split accumulator into AccVGPR                      │
│  - Recognizes accumulator reuse pattern                                │
│  - Dramatic efficiency gain for GEMM chains                            │
│                                                                         │
│  Gen 3 (Intel Xe): Wide registers + accumulator forwarding             │
│  - Fewer reads needed due to wide RF                                   │
│  - Compiler-controlled systolic depth                                  │
│                                                                         │
│  Gen 4 (Hopper): TMA + WGMMA bypass register file                      │
│  - A/B operands skip RF entirely                                       │
│  - Async delivery overlaps with compute                                │
│  - Only accumulator uses registers                                     │
│                                                                         │
│  Future: More direct paths, less general-purpose routing               │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

The trend is clear: register file bandwidth is the critical bottleneck, and each generation finds new ways to bypass it for matrix operands.

# NVIDIA Tensor Memory Accelerator descriptors

TMA is one of the most significant architectural additions in Hopper, and the tensor descriptor mechanism is the key to understanding how it works. Let me walk through the full design.

## The Problem TMA Solves

Before TMA, tensor operand delivery required extensive software involvement:

```
Pre-TMA Tensor Load (Ampere and earlier):
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  To load a 64×16 tile of A matrix from global memory:                  │
│                                                                         │
│  1. Calculate base address (registers needed)                          │
│  2. Calculate per-thread offsets (more registers)                      │
│  3. Handle bounds checking (conditionals, predicates)                  │
│  4. Issue 128+ individual load instructions                           │
│  5. Wait for loads to complete                                         │
│  6. Reorganize data in registers for tensor core layout               │
│                                                                         │
│  Software overhead per tile:                                            │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Address calculation:     ~20 instructions                      │   │
│  │  Bounds checking:         ~16 instructions                      │   │
│  │  Load instructions:       ~128 instructions                     │   │
│  │  Layout swizzling:        ~32 instructions (shuffles)           │   │
│  │  ─────────────────────────────────────────────────────────────  │   │
│  │  Total:                   ~196 instructions per tile            │   │
│  │                                                                  │   │
│  │  Registers for addresses: ~16-32                                │   │
│  │  Registers for data:      ~64                                   │   │
│  │  Registers for temps:     ~8-16                                 │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  This overhead is repeated for every tile load!                        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

TMA offloads all of this to dedicated hardware.

## Tensor Descriptor Concept

The tensor descriptor is a compact data structure that describes a multi-dimensional tensor's layout in memory:

```
Tensor Descriptor Structure (64 bytes):
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  Offset  │ Field                │ Size    │ Description                │
│  ────────────────────────────────────────────────────────────────────  │
│  0x00    │ Base Address         │ 8 bytes │ Global memory pointer      │
│  0x08    │ Dimension Count      │ 4 bytes │ 1-5 dimensions supported   │
│  0x0C    │ Data Type            │ 4 bytes │ FP16, BF16, FP32, etc.     │
│  0x10    │ Dim[0] Size          │ 4 bytes │ Size in dimension 0        │
│  0x14    │ Dim[0] Stride        │ 4 bytes │ Stride in bytes            │
│  0x18    │ Dim[1] Size          │ 4 bytes │ Size in dimension 1        │
│  0x1C    │ Dim[1] Stride        │ 4 bytes │ Stride in bytes            │
│  0x20    │ Dim[2] Size          │ 4 bytes │ (optional)                 │
│  0x24    │ Dim[2] Stride        │ 4 bytes │                            │
│  0x28    │ Dim[3] Size          │ 4 bytes │ (optional)                 │
│  0x2C    │ Dim[3] Stride        │ 4 bytes │                            │
│  0x30    │ Dim[4] Size          │ 4 bytes │ (optional)                 │
│  0x34    │ Dim[4] Stride        │ 4 bytes │                            │
│  0x38    │ Swizzle Mode         │ 4 bytes │ Memory access pattern      │
│  0x3C    │ Fill Mode            │ 4 bytes │ Out-of-bounds handling     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Creating a Tensor Descriptor

```cpp
// CUDA API for creating tensor descriptor
cudaTensorMapCreate(
    &tensorMap,                    // Output: tensor descriptor
    CUDA_TENSOR_MAP_DATA_TYPE_FLOAT16,
    tensorRank,                    // Number of dimensions (2 for matrix)
    globalAddress,                 // Base pointer in global memory
    globalDim,                     // Array of dimension sizes
    globalStride,                  // Array of strides (bytes)
    boxDim,                        // Tile size to load
    elementStride,                 // Usually 1
    interleave,                    // Memory interleaving mode
    swizzle,                       // Bank conflict avoidance mode
    l2Promotion,                   // L2 caching hint
    oobFill                        // Out-of-bounds fill value
);
```

### Example: Matrix A Descriptor

```
Matrix A Layout in Memory:
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  Matrix A: M×K, stored row-major                                       │
│  M = 4096, K = 4096, FP16                                              │
│                                                                         │
│  Memory layout:                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ A[0,0] A[0,1] A[0,2] ... A[0,4095]                              │   │
│  │ A[1,0] A[1,1] A[1,2] ... A[1,4095]                              │   │
│  │ ...                                                              │   │
│  │ A[4095,0] A[4095,1] ... A[4095,4095]                            │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Tensor Descriptor for A:                                               │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Base Address:    0x7f0000000000 (example)                      │   │
│  │  Dimensions:      2                                              │   │
│  │  Data Type:       FP16 (2 bytes)                                │   │
│  │  Dim[0] (cols):   Size = 4096, Stride = 2 bytes                 │   │
│  │  Dim[1] (rows):   Size = 4096, Stride = 8192 bytes (4096 × 2)   │   │
│  │  Box Dim:         [16, 64] (tile size: 64 rows × 16 cols)       │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  To load tile at position (row=128, col=256):                          │
│  - Coordinate: [256, 128] in descriptor terms (col, row)              │
│  - Hardware calculates: base + 256×2 + 128×8192 = address            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## TMA Hardware Architecture

```
TMA Engine Block Diagram:
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│                         SM (Streaming Multiprocessor)                   │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │                                                                    │ │
│  │  Warp Scheduler                                                   │ │
│  │       │                                                            │ │
│  │       │ TMA instruction                                           │ │
│  │       ▼                                                            │ │
│  │  ┌─────────────────────────────────────────────────────────────┐  │ │
│  │  │                    TMA Unit                                  │  │ │
│  │  │                                                              │  │ │
│  │  │  ┌────────────────┐  ┌────────────────┐  ┌──────────────┐  │  │ │
│  │  │  │  Descriptor    │  │    Address     │  │   Request    │  │  │ │
│  │  │  │    Cache       │  │   Generator    │  │    Queue     │  │  │ │
│  │  │  │                │  │                │  │              │  │  │ │
│  │  │  │  Holds recent  │  │  Computes all  │  │  Buffers     │  │  │ │
│  │  │  │  descriptors   │  │  addresses     │  │  pending     │  │  │ │
│  │  │  │  for reuse     │  │  for tile      │  │  requests    │  │  │ │
│  │  │  └───────┬────────┘  └───────┬────────┘  └──────┬───────┘  │  │ │
│  │  │          │                   │                  │          │  │ │
│  │  │          └───────────────────┼──────────────────┘          │  │ │
│  │  │                              │                              │  │ │
│  │  │                              ▼                              │  │ │
│  │  │                    ┌─────────────────┐                     │  │ │
│  │  │                    │  Memory Request │                     │  │ │
│  │  │                    │    Coalescer    │                     │  │ │
│  │  │                    └────────┬────────┘                     │  │ │
│  │  │                             │                              │  │ │
│  │  └─────────────────────────────┼──────────────────────────────┘  │ │
│  │                                │                                  │ │
│  │                                ▼                                  │ │
│  │                    ┌───────────────────────┐                     │ │
│  │                    │    L2 Cache / HBM     │                     │ │
│  │                    └───────────┬───────────┘                     │ │
│  │                                │                                  │ │
│  │                                ▼                                  │ │
│  │                    ┌───────────────────────┐                     │ │
│  │                    │   Response Handler    │                     │ │
│  │                    │                       │                     │ │
│  │                    │  - Reorders data      │                     │ │
│  │                    │  - Applies swizzle    │                     │ │
│  │                    │  - Writes to SMEM     │                     │ │
│  │                    └───────────┬───────────┘                     │ │
│  │                                │                                  │ │
│  │                                ▼                                  │ │
│  │                    ┌───────────────────────┐                     │ │
│  │                    │    Shared Memory      │                     │ │
│  │                    │    (destination)      │                     │ │
│  │                    └───────────────────────┘                     │ │
│  │                                                                   │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Address Generation Engine

The address generator is the core of TMA's efficiency:

```
TMA Address Generation:
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  Input: Descriptor + Tile Coordinates                                  │
│  Output: All memory addresses needed for the tile                      │
│                                                                         │
│  For a 64×16 tile (1024 FP16 elements = 2KB):                         │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Address Generator Pipeline                                      │   │
│  │                                                                  │   │
│  │  Stage 1: Coordinate Validation                                 │   │
│  │  ┌────────────────────────────────────────────────────────────┐ │   │
│  │  │  tile_row < tensor_rows ? valid : clamp/fill               │ │   │
│  │  │  tile_col < tensor_cols ? valid : clamp/fill               │ │   │
│  │  └────────────────────────────────────────────────────────────┘ │   │
│  │           │                                                      │   │
│  │           ▼                                                      │   │
│  │  Stage 2: Base Offset Calculation                               │   │
│  │  ┌────────────────────────────────────────────────────────────┐ │   │
│  │  │  base_offset = coord[0] × stride[0] +                      │ │   │
│  │  │                coord[1] × stride[1] +                      │ │   │
│  │  │                ... (for each dimension)                    │ │   │
│  │  └────────────────────────────────────────────────────────────┘ │   │
│  │           │                                                      │   │
│  │           ▼                                                      │   │
│  │  Stage 3: Element Address Enumeration                           │   │
│  │  ┌────────────────────────────────────────────────────────────┐ │   │
│  │  │  for row in 0..63:                                         │ │   │
│  │  │    for col in 0..15:                                       │ │   │
│  │  │      addr[row,col] = base + row×row_stride + col×elem_size│ │   │
│  │  │                                                            │ │   │
│  │  │  Hardware generates 1024 addresses in ~16 cycles          │ │   │
│  │  │  (64 addresses per cycle, pipelined)                      │ │   │
│  │  └────────────────────────────────────────────────────────────┘ │   │
│  │           │                                                      │   │
│  │           ▼                                                      │   │
│  │  Stage 4: Cache Line Grouping                                   │   │
│  │  ┌────────────────────────────────────────────────────────────┐ │   │
│  │  │  Group addresses into 128-byte cache line requests         │ │   │
│  │  │  2KB tile ÷ 128B lines = 16 cache line requests           │ │   │
│  │  │                                                            │ │   │
│  │  │  Coalescing: detect which addresses hit same line         │ │   │
│  │  └────────────────────────────────────────────────────────────┘ │   │
│  │                                                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Swizzle Modes

Swizzling is critical for avoiding shared memory bank conflicts:

```
Shared Memory Bank Conflict Problem:
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  Shared Memory: 32 banks, 4 bytes per bank                             │
│                                                                         │
│  Naive tile layout (64×16 FP16, row-major):                            │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │  Row 0:  [B0][B1][B2][B3][B4][B5][B6][B7] ... [B15]            │    │
│  │  Row 1:  [B0][B1][B2][B3][B4][B5][B6][B7] ... [B15]            │    │
│  │  Row 2:  [B0][B1][B2][B3][B4][B5][B6][B7] ... [B15]            │    │
│  │  ...                                                           │    │
│  │                                                                 │    │
│  │  Problem: Column access hits same banks!                       │    │
│  │  Reading column 0: all rows hit Bank 0 → 64-way conflict!     │    │
│  └────────────────────────────────────────────────────────────────┘    │
│                                                                         │
│  TMA Swizzle solution:                                                 │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │  Row 0:  [B0 ][B1 ][B2 ][B3 ][B4 ][B5 ][B6 ][B7 ] ...         │    │
│  │  Row 1:  [B4 ][B5 ][B6 ][B7 ][B0 ][B1 ][B2 ][B3 ] ... (XOR 4) │    │
│  │  Row 2:  [B8 ][B9 ][B10][B11][B12][B13][B14][B15] ... (XOR 8) │    │
│  │  Row 3:  [B12][B13][B14][B15][B8 ][B9 ][B10][B11] ... (XOR 12)│    │
│  │  ...                                                           │    │
│  │                                                                 │    │
│  │  Column access now distributed across banks!                   │    │
│  └────────────────────────────────────────────────────────────────┘    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Swizzle Mode Details

```
TMA Swizzle Modes:
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  Mode                │ XOR Pattern     │ Use Case                      │
│  ─────────────────────────────────────────────────────────────────────  │
│  SWIZZLE_NONE        │ No swizzling    │ Sequential access patterns    │
│  SWIZZLE_32B         │ XOR bits [0:4]  │ 32-byte aligned tiles         │
│  SWIZZLE_64B         │ XOR bits [0:5]  │ 64-byte aligned tiles         │
│  SWIZZLE_128B        │ XOR bits [0:6]  │ 128-byte aligned tiles        │
│                                                                         │
│  Swizzle formula:                                                       │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                                                                  │   │
│  │  smem_addr = base_addr + (offset ^ swizzle_pattern)             │   │
│  │                                                                  │   │
│  │  For SWIZZLE_128B:                                              │   │
│  │  swizzle_pattern = (row_in_tile × 16) & 0x7F                    │   │
│  │                     ────────────────                             │   │
│  │                     bytes per row                                │   │
│  │                                                                  │   │
│  │  Example (row 5, 128B swizzle):                                 │   │
│  │  swizzle_pattern = (5 × 16) & 0x7F = 80 & 0x7F = 0x50          │   │
│  │  byte 0 of row 5 → byte 0 XOR 0x50 = byte 0x50                 │   │
│  │                                                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Hardware applies swizzle automatically during:                        │
│  - TMA store to shared memory                                         │
│  - WGMMA read from shared memory                                      │
│                                                                         │
│  Software only needs to specify swizzle mode in descriptor            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## TMA Instructions

### cp.async.bulk: The Core TMA Instruction

```
TMA Instruction Anatomy:
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete│
│       [smem_ptr], [tensormap_ptr, {coord0, coord1}], [mbar_ptr]        │
│                                                                         │
│  Breakdown:                                                             │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  cp.async.bulk         │ Asynchronous bulk copy instruction    │   │
│  │  .tensor               │ Using tensor descriptor               │   │
│  │  .2d                   │ 2-dimensional tensor                  │   │
│  │  .shared::cluster      │ Destination is cluster shared memory  │   │
│  │  .global               │ Source is global memory               │   │
│  │  .tile                 │ Copy a tile (vs single element)       │   │
│  │  .mbarrier::complete   │ Signal mbarrier when done             │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Operands:                                                              │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  smem_ptr      │ Destination address in shared memory          │   │
│  │  tensormap_ptr │ Pointer to tensor descriptor                  │   │
│  │  coord0, coord1│ Tile coordinates (which tile to load)         │   │
│  │  mbar_ptr      │ Memory barrier for completion signaling       │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Memory Barrier Integration

TMA uses hardware memory barriers for synchronization:

```
TMA + mbarrier Synchronization:
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  typedef struct {                                                       │
│      uint64_t barrier;    // Hardware barrier state                    │
│      uint32_t phase;      // Alternating phase for double buffering   │
│  } mbarrier_t;                                                         │
│                                                                         │
│  Barrier Lifecycle:                                                     │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                                                                  │   │
│  │  1. Initialize barrier with expected arrival count              │   │
│  │     mbarrier.init(&mbar, expected_bytes);                       │   │
│  │                                                                  │   │
│  │  2. Issue TMA (increments pending count)                        │   │
│  │     cp.async.bulk.tensor ... , [mbar];                          │   │
│  │     // TMA hardware will signal mbar when copy completes        │   │
│  │                                                                  │   │
│  │  3. Wait for completion                                         │   │
│  │     mbarrier.wait(&mbar, phase);                                │   │
│  │     // Blocks until all expected bytes have arrived             │   │
│  │                                                                  │   │
│  │  4. Flip phase for next iteration                               │   │
│  │     phase ^= 1;                                                 │   │
│  │                                                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Hardware mbarrier state machine:                                       │
│                                                                         │
│  ┌──────────────┐    init(N)    ┌──────────────┐                       │
│  │              │──────────────▶│              │                       │
│  │   EMPTY      │               │  WAITING     │                       │
│  │              │◀──────────────│  (count=N)   │                       │
│  └──────────────┘    reset      └──────┬───────┘                       │
│                                        │                                │
│                           TMA complete │ (decrements count)            │
│                                        ▼                                │
│                                 ┌──────────────┐                       │
│                      count==0? │              │                        │
│                        yes ───▶│  COMPLETE    │───▶ wake waiting warps │
│                                │              │                        │
│                                └──────────────┘                        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Multi-Dimensional Tensor Support

TMA supports up to 5 dimensions:

```
5D Tensor Descriptor Example (Batched Attention):
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  Attention tensor: [Batch, Heads, SeqLen, SeqLen, HeadDim]             │
│                                                                         │
│  Descriptor:                                                            │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Dim[0]: HeadDim    size=64    stride=2 (FP16)                  │   │
│  │  Dim[1]: SeqLen_K   size=2048  stride=128                       │   │
│  │  Dim[2]: SeqLen_Q   size=2048  stride=262144                    │   │
│  │  Dim[3]: Heads      size=32    stride=536870912                 │   │
│  │  Dim[4]: Batch      size=8     stride=17179869184               │   │
│  │                                                                  │   │
│  │  Box (tile) size: [64, 64, 1, 1, 1]                             │   │
│  │  = Load 64×64 attention tile for one head, one batch           │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  TMA instruction:                                                       │
│  cp.async.bulk.tensor.5d.shared.global.tile                            │
│      [smem], [desc, {hdim_off, k_off, q_off, head, batch}], [mbar]     │
│                                                                         │
│  Address calculation (hardware):                                        │
│  addr = base + hdim_off×2 + k_off×128 + q_off×262144                  │
│              + head×536870912 + batch×17179869184                      │
│                                                                         │
│  This would require ~20 instructions in software!                      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Out-of-Bounds Handling

TMA handles boundary conditions automatically:

```
TMA Boundary Handling Modes:
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  Scenario: Loading 64×16 tile at edge of 4000×4000 matrix             │
│  Tile position: row=3980, col=0                                        │
│  Valid rows: 3980-3999 (20 rows), need 64 rows                        │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                                                                  │   │
│  │  Mode: OOB_FILL_ZERO                                            │   │
│  │  ┌────────────────────────────────────────────────────────────┐ │   │
│  │  │  Row 0-19:   Valid data from global memory                 │ │   │
│  │  │  Row 20-63:  Filled with zeros                             │ │   │
│  │  │                                                            │ │   │
│  │  │  Hardware detects OOB addresses, skips memory request,    │ │   │
│  │  │  writes zero to shared memory instead                     │ │   │
│  │  └────────────────────────────────────────────────────────────┘ │   │
│  │                                                                  │   │
│  │  Mode: OOB_FILL_NAN                                             │   │
│  │  ┌────────────────────────────────────────────────────────────┐ │   │
│  │  │  Row 0-19:   Valid data                                    │ │   │
│  │  │  Row 20-63:  Filled with NaN (useful for debugging)       │ │   │
│  │  └────────────────────────────────────────────────────────────┘ │   │
│  │                                                                  │   │
│  │  Mode: OOB_CLAMP                                                │   │
│  │  ┌────────────────────────────────────────────────────────────┐ │   │
│  │  │  Row 0-19:   Valid data                                    │ │   │
│  │  │  Row 20-63:  Repeat row 19 (edge clamping)                │ │   │
│  │  │                                                            │ │   │
│  │  │  Useful for convolution-like operations                   │ │   │
│  │  └────────────────────────────────────────────────────────────┘ │   │
│  │                                                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Software no longer needs:                                              │
│  - Bounds checking code                                                │
│  - Predicated loads                                                    │
│  - Manual padding                                                      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## TMA in Practice: GEMM Example

```
TMA-Accelerated GEMM Kernel Structure:
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  // Descriptor setup (done once on host)                               │
│  cudaTensorMapCreate(&descA, ...);  // M×K matrix                      │
│  cudaTensorMapCreate(&descB, ...);  // K×N matrix                      │
│                                                                         │
│  // Kernel code (simplified)                                            │
│  __global__ void gemm_tma(descA, descB, C, M, N, K) {                  │
│                                                                         │
│      __shared__ __align__(128) half smemA[2][TILE_M][TILE_K];         │
│      __shared__ __align__(128) half smemB[2][TILE_K][TILE_N];         │
│      __shared__ mbarrier_t mbar[2];                                    │
│                                                                         │
│      // Double buffering setup                                         │
│      int phase = 0;                                                    │
│      int buffer = 0;                                                   │
│                                                                         │
│      // Prologue: start first loads                                    │
│      if (threadIdx.x == 0) {                                          │
│          cp.async.bulk.tensor.2d [smemA[0]], [descA, {0, tile_m}];    │
│          cp.async.bulk.tensor.2d [smemB[0]], [descB, {tile_n, 0}];    │
│      }                                                                 │
│                                                                         │
│      // Main loop                                                       │
│      for (int k = 0; k < K; k += TILE_K) {                            │
│                                                                         │
│          // Wait for current tile                                      │
│          mbarrier.wait(&mbar[buffer], phase);                         │
│                                                                         │
│          // Start loading next tile (async, non-blocking)             │
│          if (threadIdx.x == 0 && k + TILE_K < K) {                    │
│              int next = buffer ^ 1;                                   │
│              cp.async.bulk.tensor.2d                                  │
│                  [smemA[next]], [descA, {k+TILE_K, tile_m}], [mbar];  │
│              cp.async.bulk.tensor.2d                                  │
│                  [smemB[next]], [descB, {tile_n, k+TILE_K}], [mbar];  │
│          }                                                             │
│                                                                         │
│          // Compute using WGMMA (reads directly from smem!)           │
│          wgmma.mma_async [accum], [smemA[buffer]], [smemB[buffer]];   │
│                                                                         │
│          // Flip buffers                                               │
│          buffer ^= 1;                                                  │
│          phase ^= 1;                                                   │
│      }                                                                 │
│                                                                         │
│      // Store results to C                                             │
│      ...                                                               │
│  }                                                                     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Execution Timeline

```
TMA Double-Buffered Pipeline:
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  Time →                                                                 │
│                                                                         │
│  TMA Engine:                                                            │
│  ├─ Load A[0],B[0] ─┼─ Load A[1],B[1] ─┼─ Load A[2],B[2] ─┼─ ...       │
│                                                                         │
│  Tensor Core:                                                           │
│  ├──── idle ────────┼─ Compute [0] ────┼─ Compute [1] ────┼─ ...       │
│                                                                         │
│  Memory Barrier:                                                        │
│  ├─ mbar[0] wait ───┼─ mbar[1] wait ───┼─ mbar[0] wait ───┼─ ...       │
│                                                                         │
│  Legend:                                                                │
│  ═══ TMA load active                                                   │
│  ███ Tensor core compute                                               │
│  ─── Wait/synchronization                                              │
│                                                                         │
│  Key insight: TMA loads overlap with tensor core compute               │
│  - No warp stalls for memory                                           │
│  - No register file traffic for A/B operands                          │
│  - Hardware handles all address math                                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Performance Impact

```
TMA Performance Benefits:
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  Metric                     │ Pre-TMA (Ampere) │ With TMA (Hopper)     │
│  ─────────────────────────────────────────────────────────────────────  │
│  Instructions per tile load │ ~200             │ 1-2                   │
│  Registers for addressing   │ 16-32            │ 0                     │
│  Software bounds checking   │ Required         │ Hardware              │
│  Bank conflict handling     │ Manual swizzle   │ Automatic             │
│  Async overlap              │ Limited          │ Full                  │
│  Address coalescing         │ Compiler         │ Hardware optimal      │
│                                                                         │
│  Effective memory bandwidth utilization:                               │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Ampere (manual loads):     ~70-80% of peak                     │   │
│  │  Hopper (TMA):              ~90-95% of peak                     │   │
│  │                                                                  │   │
│  │  The 15-20% improvement comes from:                             │   │
│  │  - Better coalescing (hardware optimal)                         │   │
│  │  - No instruction cache pressure from load code                 │   │
│  │  - More registers available for accumulation                    │   │
│  │  - Perfect overlap of load and compute                          │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  GEMM efficiency (TFLOPS achieved / peak TFLOPS):                     │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  A100 (Ampere): ~80% at large sizes, drops at small sizes      │   │
│  │  H100 (Hopper): ~90% sustained, better small-size efficiency   │   │
│  │                                                                  │   │
│  │  TMA is especially impactful for:                               │   │
│  │  - Attention (many small tiles, complex indexing)              │   │
│  │  - Batched GEMM (high descriptor reuse)                        │   │
│  │  - Strided/transposed access (hardware handles complexity)     │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Comparison: TMA vs AMD/Intel Approaches

```
Tensor Load Mechanism Comparison:
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  NVIDIA TMA (Hopper):                                                  │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  + Dedicated hardware for address generation                    │   │
│  │  + Up to 5D tensor support                                      │   │
│  │  + Automatic swizzling and bounds handling                     │   │
│  │  + Direct to shared memory (bypasses RF)                       │   │
│  │  - Requires descriptor setup (host code complexity)            │   │
│  │  - Descriptor cache pressure for many tensors                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  AMD (CDNA2/3):                                                        │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  + Global load → LDS with buffer descriptors                   │   │
│  │  + Simpler model (no separate TMA unit)                        │   │
│  │  + AccVGPR reduces RF pressure for accumulators                │   │
│  │  - Address calculation still in shader                         │   │
│  │  - Manual swizzling often needed                               │   │
│  │  - More instruction overhead for loads                         │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Intel Xe-HPC:                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  + Block loads (2D load instructions)                          │   │
│  │  + Wide registers reduce load count                            │   │
│  │  + Prefetch hints for memory hierarchy                         │   │
│  │  - Less flexible than TMA descriptors                          │   │
│  │  - Still requires address setup in kernel                      │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Summary: NVIDIA invested heavily in TMA to address RF bottleneck     │
│           AMD relies on AccVGPR to reduce RF pressure differently     │
│           Intel uses wide registers to minimize load operations       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

TMA represents NVIDIA's recognition that operand delivery is the critical bottleneck for matrix operations. By creating dedicated hardware for tensor addressing and moving data directly to shared memory, they've effectively created a bypass for the register file that was limiting performance in earlier architectures.

