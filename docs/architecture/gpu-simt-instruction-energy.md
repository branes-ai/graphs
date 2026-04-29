# GPU SIMT instruction energy -- Edge-8nm-LPDDR5 (Jetson Orin-class)

Per-stage energy accounting for one Ampere SM-cycle of 4 subpartitions x 32 lanes = 128 lane-ops, with all data resident in the per-subpartition register file (no L1, no shared memory, no off-chip traffic).

## 1. Technology-profile reference (source of truth)

All energy primitives in this report are derived from the selected `TechnologyProfile` (`edge-8nm-lpddr5`). The full set of values flowing into the model:

| Field | Value |
|---|---|
| Process node | 8 nm |
| Memory technology | LPDDR5 |
| Target market | edge |
| Typical frequency | 1.30 GHz |
| Typical TDP | 60 W |
| &nbsp; |  |
| base_alu_energy_pj (FP32 FMA) | 1.900 pJ |
| simd_mac_energy_pj | 1.710 pJ |
| tensor_core_mac_energy_pj | 1.615 pJ |
| &nbsp; |  |
| instruction_fetch_energy_pj | 0.570 pJ |
| instruction_decode_energy_pj | 0.304 pJ |
| instruction_dispatch_energy_pj | 0.190 pJ |
| &nbsp; |  |
| register_read_energy_pj | 1.069 pJ |
| register_write_energy_pj | 1.282 pJ |

Derived-from-base ratios used in this report:

| Derived field | Formula |
|---|---|
| `ff_read_each_pj` | `register_read * 5%` (a directly-clocked flip-flop is far smaller than a banked RF cell) |
| `oc_per_op_pj`    | `register_read * 5% * 2` (1 flop write + 1 flop read per operand) |
| `alu_disp_pj`     | `register_read * 25%` (operand-collector to lane wire drive) |
| `sched_pj`        | `instruction_decode * 50%` (scoreboard read + arbitration) |
| Op-kind ALU ratio | FADD : FMUL : FMA = 0.22 : 0.88 : 1.00 (Horowitz 45nm tutorial) |
| Precision scaling | FP32 : FP16 : INT8 = 1.00 : 0.50 : 0.20 (multiplier area scales with bits-squared) |
| Packing factor    | FP32 : FP16 : INT8 = 1 : 2 : 4 (HFMA2 / DP4A semantics) |

## 2. SIMT pipeline stages

Stages 1-4 fire ONCE PER SUBPARTITION (each Ampere subpartition has its own L0 I-cache, decoder, warp scheduler, and dispatch -- so they are counted x4). Stages 5-9 fire at the full SM lane count.

| # | Label | Description |
|---|---|---|
| 1 | Fch | Fetch -- L0 instruction cache read; one per subpartition |
| 2 | Dec | Decode -- instruction decoder; one per subpartition |
| 3 | Sch | Warp schedule -- 16-warp scoreboard + priority arbitration; one per subpartition |
| 4 | Dsp | Dispatch -- subpartition control wires energized |
| 5 | Rd | Register-file read -- banked 32-bit reads, one per source operand per lane |
| 6 | OC | Operand collector -- flop write + read to align operands across cycles |
| 7 | Disp | ALU dispatch -- operand wires from OC to lane ALUs |
| 8 | Exe | Compute -- FADD / FMUL / FMA in the per-lane ALU |
| 9 | WB | Writeback -- banked RF write, one per lane (dest only) |

## 3. Baseline ALU energy (irreducible compute floor)

Single ALU stripped of all SIMT overhead: input flip-flops wired directly to the ALU inputs, output flip-flop on the result. No register file, no operand collector, no instruction fetch / decode / scheduler / dispatch.

### 3.FADD

**FADD fp32**  (2 source operands, 1 FLOP/op)

| Stage | FF_read | MUL | ADD | FF_write | Total |
|---|---|---|---|---|---|
| pJ | 0.107 | -- | 0.418 | 0.053 | 0.578 |

Per-op total: **0.578 pJ**, per-FLOP: **0.578 pJ/FLOP**.

**FADD fp16**  (2 source operands, 1 FLOP/op)

| Stage | FF_read | MUL | ADD | FF_write | Total |
|---|---|---|---|---|---|
| pJ | 0.107 | -- | 0.209 | 0.053 | 0.369 |

Per-op total: **0.369 pJ**, per-FLOP: **0.369 pJ/FLOP**.

**FADD int8**  (2 source operands, 1 FLOP/op)

| Stage | FF_read | MUL | ADD | FF_write | Total |
|---|---|---|---|---|---|
| pJ | 0.107 | -- | 0.084 | 0.053 | 0.244 |

Per-op total: **0.244 pJ**, per-FLOP: **0.244 pJ/FLOP**.

### 3.FMUL

**FMUL fp32**  (2 source operands, 1 FLOP/op)

| Stage | FF_read | MUL | ADD | FF_write | Total |
|---|---|---|---|---|---|
| pJ | 0.107 | 1.672 | -- | 0.053 | 1.832 |

Per-op total: **1.832 pJ**, per-FLOP: **1.832 pJ/FLOP**.

**FMUL fp16**  (2 source operands, 1 FLOP/op)

| Stage | FF_read | MUL | ADD | FF_write | Total |
|---|---|---|---|---|---|
| pJ | 0.107 | 0.836 | -- | 0.053 | 0.996 |

Per-op total: **0.996 pJ**, per-FLOP: **0.996 pJ/FLOP**.

**FMUL int8**  (2 source operands, 1 FLOP/op)

| Stage | FF_read | MUL | ADD | FF_write | Total |
|---|---|---|---|---|---|
| pJ | 0.107 | 0.334 | -- | 0.053 | 0.495 |

Per-op total: **0.495 pJ**, per-FLOP: **0.495 pJ/FLOP**.

### 3.FMA

**FMA fp32**  (3 source operands, 2 FLOP/op)

| Stage | FF_read | MUL | ADD | FF_write | Total |
|---|---|---|---|---|---|
| pJ | 0.160 | 1.672 | 0.228 | 0.053 | 2.114 |

Per-op total: **2.114 pJ**, per-FLOP: **1.057 pJ/FLOP**.

**FMA fp16**  (3 source operands, 2 FLOP/op)

| Stage | FF_read | MUL | ADD | FF_write | Total |
|---|---|---|---|---|---|
| pJ | 0.160 | 0.836 | 0.114 | 0.053 | 1.164 |

Per-op total: **1.164 pJ**, per-FLOP: **0.582 pJ/FLOP**.

**FMA int8**  (3 source operands, 2 FLOP/op)

| Stage | FF_read | MUL | ADD | FF_write | Total |
|---|---|---|---|---|---|
| pJ | 0.160 | 0.334 | 0.046 | 0.053 | 0.594 |

Per-op total: **0.594 pJ**, per-FLOP: **0.297 pJ/FLOP**.

## 4. SIMT pipeline energy (one SM-cycle, 128 lanes)

Rows trace each operand / control flow through the 9 pipeline stages; each cell is the energy attributable to that operation at that stage. The Stage total row at the bottom sums vertically; the Row total column on the right sums horizontally; the bold corner cell is the per-instruction total.

### 4.FADD

**FADD fp32** -- 128 lanes x 1 packed = 128 ops (128 FLOPS)

| Operation | Fch | Dec | Sch | Dsp | Rd | OC | Disp | Exe | WB | Row total |
|---|---|---|---|---|---|---|---|---|---|---|
| Instruction control | 2.280 | 1.216 | 0.608 | 0.760 | -- | -- | -- | -- | -- | 4.864 |
| Src operand A | -- | -- | -- | -- | 137 | 13.7 | 34.2 | -- | -- | 185 |
| Src operand B | -- | -- | -- | -- | 137 | 13.7 | 34.2 | -- | -- | 185 |
| ALU compute | -- | -- | -- | -- | -- | -- | -- | 53.5 | -- | 53.5 |
| Dest writeback | -- | -- | -- | -- | -- | -- | -- | -- | 164 | 164 |
| **Stage total** | 2.280 | 1.216 | 0.608 | 0.760 | 274 | 27.4 | 68.4 | 53.5 | 164 | **592** |

Per-instruction: **592 pJ**, per-op: **4.624 pJ/op**, per-FLOP: **4.624 pJ**.

**FADD fp16** -- 128 lanes x 2 packed = 256 ops (256 FLOPS)

| Operation | Fch | Dec | Sch | Dsp | Rd | OC | Disp | Exe | WB | Row total |
|---|---|---|---|---|---|---|---|---|---|---|
| Instruction control | 2.280 | 1.216 | 0.608 | 0.760 | -- | -- | -- | -- | -- | 4.864 |
| Src operand A | -- | -- | -- | -- | 137 | 13.7 | 34.2 | -- | -- | 185 |
| Src operand B | -- | -- | -- | -- | 137 | 13.7 | 34.2 | -- | -- | 185 |
| ALU compute | -- | -- | -- | -- | -- | -- | -- | 53.5 | -- | 53.5 |
| Dest writeback | -- | -- | -- | -- | -- | -- | -- | -- | 164 | 164 |
| **Stage total** | 2.280 | 1.216 | 0.608 | 0.760 | 274 | 27.4 | 68.4 | 53.5 | 164 | **592** |

Per-instruction: **592 pJ**, per-op: **2.312 pJ/op**, per-FLOP: **2.312 pJ**.

**FADD int8** -- 128 lanes x 4 packed = 512 ops (512 IntOPS)

| Operation | Fch | Dec | Sch | Dsp | Rd | OC | Disp | Exe | WB | Row total |
|---|---|---|---|---|---|---|---|---|---|---|
| Instruction control | 2.280 | 1.216 | 0.608 | 0.760 | -- | -- | -- | -- | -- | 4.864 |
| Src operand A | -- | -- | -- | -- | 137 | 13.7 | 34.2 | -- | -- | 185 |
| Src operand B | -- | -- | -- | -- | 137 | 13.7 | 34.2 | -- | -- | 185 |
| ALU compute | -- | -- | -- | -- | -- | -- | -- | 42.8 | -- | 42.8 |
| Dest writeback | -- | -- | -- | -- | -- | -- | -- | -- | 164 | 164 |
| **Stage total** | 2.280 | 1.216 | 0.608 | 0.760 | 274 | 27.4 | 68.4 | 42.8 | 164 | **581** |

Per-instruction: **581 pJ**, per-op: **1.135 pJ/op**, per-IntOP: **1.135 pJ**.

### 4.FMUL

**FMUL fp32** -- 128 lanes x 1 packed = 128 ops (128 FLOPS)

| Operation | Fch | Dec | Sch | Dsp | Rd | OC | Disp | Exe | WB | Row total |
|---|---|---|---|---|---|---|---|---|---|---|
| Instruction control | 2.280 | 1.216 | 0.608 | 0.760 | -- | -- | -- | -- | -- | 4.864 |
| Src operand A | -- | -- | -- | -- | 137 | 13.7 | 34.2 | -- | -- | 185 |
| Src operand B | -- | -- | -- | -- | 137 | 13.7 | 34.2 | -- | -- | 185 |
| ALU compute | -- | -- | -- | -- | -- | -- | -- | 214 | -- | 214 |
| Dest writeback | -- | -- | -- | -- | -- | -- | -- | -- | 164 | 164 |
| **Stage total** | 2.280 | 1.216 | 0.608 | 0.760 | 274 | 27.4 | 68.4 | 214 | 164 | **752** |

Per-instruction: **752 pJ**, per-op: **5.878 pJ/op**, per-FLOP: **5.878 pJ**.

**FMUL fp16** -- 128 lanes x 2 packed = 256 ops (256 FLOPS)

| Operation | Fch | Dec | Sch | Dsp | Rd | OC | Disp | Exe | WB | Row total |
|---|---|---|---|---|---|---|---|---|---|---|
| Instruction control | 2.280 | 1.216 | 0.608 | 0.760 | -- | -- | -- | -- | -- | 4.864 |
| Src operand A | -- | -- | -- | -- | 137 | 13.7 | 34.2 | -- | -- | 185 |
| Src operand B | -- | -- | -- | -- | 137 | 13.7 | 34.2 | -- | -- | 185 |
| ALU compute | -- | -- | -- | -- | -- | -- | -- | 214 | -- | 214 |
| Dest writeback | -- | -- | -- | -- | -- | -- | -- | -- | 164 | 164 |
| **Stage total** | 2.280 | 1.216 | 0.608 | 0.760 | 274 | 27.4 | 68.4 | 214 | 164 | **752** |

Per-instruction: **752 pJ**, per-op: **2.939 pJ/op**, per-FLOP: **2.939 pJ**.

**FMUL int8** -- 128 lanes x 4 packed = 512 ops (512 IntOPS)

| Operation | Fch | Dec | Sch | Dsp | Rd | OC | Disp | Exe | WB | Row total |
|---|---|---|---|---|---|---|---|---|---|---|
| Instruction control | 2.280 | 1.216 | 0.608 | 0.760 | -- | -- | -- | -- | -- | 4.864 |
| Src operand A | -- | -- | -- | -- | 137 | 13.7 | 34.2 | -- | -- | 185 |
| Src operand B | -- | -- | -- | -- | 137 | 13.7 | 34.2 | -- | -- | 185 |
| ALU compute | -- | -- | -- | -- | -- | -- | -- | 171 | -- | 171 |
| Dest writeback | -- | -- | -- | -- | -- | -- | -- | -- | 164 | 164 |
| **Stage total** | 2.280 | 1.216 | 0.608 | 0.760 | 274 | 27.4 | 68.4 | 171 | 164 | **710** |

Per-instruction: **710 pJ**, per-op: **1.386 pJ/op**, per-IntOP: **1.386 pJ**.

### 4.FMA

**FMA fp32** -- 128 lanes x 1 packed = 128 ops (256 FLOPS)

| Operation | Fch | Dec | Sch | Dsp | Rd | OC | Disp | Exe | WB | Row total |
|---|---|---|---|---|---|---|---|---|---|---|
| Instruction control | 2.280 | 1.216 | 0.608 | 0.760 | -- | -- | -- | -- | -- | 4.864 |
| Src operand A | -- | -- | -- | -- | 137 | 13.7 | 34.2 | -- | -- | 185 |
| Src operand B | -- | -- | -- | -- | 137 | 13.7 | 34.2 | -- | -- | 185 |
| Src operand C | -- | -- | -- | -- | 137 | 13.7 | 34.2 | -- | -- | 185 |
| ALU compute | -- | -- | -- | -- | -- | -- | -- | 243 | -- | 243 |
| Dest writeback | -- | -- | -- | -- | -- | -- | -- | -- | 164 | 164 |
| **Stage total** | 2.280 | 1.216 | 0.608 | 0.760 | 410 | 41.0 | 103 | 243 | 164 | **966** |

Per-instruction: **966 pJ**, per-op: **7.549 pJ/op**, per-FLOP: **3.774 pJ**.

**FMA fp16** -- 128 lanes x 2 packed = 256 ops (512 FLOPS)

| Operation | Fch | Dec | Sch | Dsp | Rd | OC | Disp | Exe | WB | Row total |
|---|---|---|---|---|---|---|---|---|---|---|
| Instruction control | 2.280 | 1.216 | 0.608 | 0.760 | -- | -- | -- | -- | -- | 4.864 |
| Src operand A | -- | -- | -- | -- | 137 | 13.7 | 34.2 | -- | -- | 185 |
| Src operand B | -- | -- | -- | -- | 137 | 13.7 | 34.2 | -- | -- | 185 |
| Src operand C | -- | -- | -- | -- | 137 | 13.7 | 34.2 | -- | -- | 185 |
| ALU compute | -- | -- | -- | -- | -- | -- | -- | 243 | -- | 243 |
| Dest writeback | -- | -- | -- | -- | -- | -- | -- | -- | 164 | 164 |
| **Stage total** | 2.280 | 1.216 | 0.608 | 0.760 | 410 | 41.0 | 103 | 243 | 164 | **966** |

Per-instruction: **966 pJ**, per-op: **3.774 pJ/op**, per-FLOP: **1.887 pJ**.

**FMA int8** -- 128 lanes x 4 packed = 512 ops (1024 IntOPS)

| Operation | Fch | Dec | Sch | Dsp | Rd | OC | Disp | Exe | WB | Row total |
|---|---|---|---|---|---|---|---|---|---|---|
| Instruction control | 2.280 | 1.216 | 0.608 | 0.760 | -- | -- | -- | -- | -- | 4.864 |
| Src operand A | -- | -- | -- | -- | 137 | 13.7 | 34.2 | -- | -- | 185 |
| Src operand B | -- | -- | -- | -- | 137 | 13.7 | 34.2 | -- | -- | 185 |
| Src operand C | -- | -- | -- | -- | 137 | 13.7 | 34.2 | -- | -- | 185 |
| ALU compute | -- | -- | -- | -- | -- | -- | -- | 195 | -- | 195 |
| Dest writeback | -- | -- | -- | -- | -- | -- | -- | -- | 164 | 164 |
| **Stage total** | 2.280 | 1.216 | 0.608 | 0.760 | 410 | 41.0 | 103 | 195 | 164 | **918** |

Per-instruction: **918 pJ**, per-op: **1.792 pJ/op**, per-IntOP: **0.896 pJ**.

## 5. Cross-comparison summary

Headline: per-op energy in the full SIMT pipeline vs the irreducible baseline ALU. Ratio = SIMT-overhead tax.

| Op | Precision | Baseline pJ/op | SIMT pJ/op | SIMT pJ/FLOP | SIMT/Baseline | Compute % | RF+OC+wire % |
|---|---|---|---|---|---|---|---|
| FADD | fp32 | 0.578 | 4.624 | 4.624 | 8.0x | 9.0% | 90.1% |
| FADD | fp16 | 0.369 | 2.312 | 2.312 | 6.3x | 9.0% | 90.1% |
| FADD | int8 | 0.244 | 1.135 | 1.135 | 4.7x | 7.4% | 91.8% |
| FMUL | fp32 | 1.832 | 5.878 | 5.878 | 3.2x | 28.4% | 70.9% |
| FMUL | fp16 | 0.996 | 2.939 | 2.939 | 2.9x | 28.4% | 70.9% |
| FMUL | int8 | 0.495 | 1.386 | 1.386 | 2.8x | 24.1% | 75.2% |
| FMA | fp32 | 2.114 | 7.549 | 3.774 | 3.6x | 25.2% | 74.3% |
| FMA | fp16 | 1.164 | 3.774 | 1.887 | 3.2x | 25.2% | 74.3% |
| FMA | int8 | 0.594 | 1.792 | 0.896 | 3.0x | 21.2% | 78.3% |

## 6. Architectural reading

1. **Cheap compute = high overhead share.** FADD has the smallest ALU energy per op, so a larger fraction of the instruction energy goes into RF reads, the operand collector, and the writeback. The SIMT/baseline overhead ratio is highest for FADD and lowest for FMA.

2. **Packing recovers narrow-precision energy efficiency.** Per-instruction energy is approximately constant across fp32 / fp16-packed / int8-packed, because the same 32-bit datapath does 1, 2, or 4 useful ops respectively. Per-op energy halves (fp16) or quarters (int8) -- not from shrinking the RF, but from amortizing it over more useful work.

3. **Per-subpartition control is x4 at SM level.** Each Ampere subpartition runs its own fetch / decode / scheduler / dispatch. Counting any of these once at SM level (the naive accounting) under-reports the control-overhead share by 4x.

4. **The TechnologyProfile RF energies are derived from a CPU-style scaling.** Published GPU RF reads at 8nm are typically 3-5x higher than the values used here, because GPU register files are larger, more banked, and have more ports. The model's relative energy decomposition is correct; absolute SIMT/baseline ratios scale with the assumed RF energy.

5. **Operand collector is poorly characterized in public literature.** This report models it as a per-operand flop write + read. Real GPU operand collectors are more complex (multi-entry, cross-bar to RF banks). Treat the OC column as INTERPOLATED.

## 7. Self-validation

These are the sanity checks the test suite runs against the model output. They are stable across TechnologyProfile changes within +/-30% RF energy.

- fp16-packed / fp32 per-op ratio in [0.45, 0.55] for every op.
- int8-packed / fp32 per-op ratio in [0.20, 0.30] for every op.
- FMA ALU energy / FMUL ALU energy in [1.10, 1.18] (FMA = MUL + small ADD).
- SIMT FADD overhead ratio > FMUL > FMA at every precision.
- Stages 1-4 each fire `sm_subpartitions` times, not once.
