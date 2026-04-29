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

## 1b. Banked SRAM register file (the SIMT energy story)

A SIMT pipeline only works when many warps are in flight concurrently; that requires hundreds-to-thousands of registers per subpartition, which can only come from a banked SRAM. The banked-SRAM cost IS the GPU's architectural overhead vs accelerators (TPU / KPU / CGRA) that either eliminate the general-purpose register file (systolic data flows bank-to-bank) or replace it with FIFOs (dataflow streams). Quantifying this cost is the purpose of this report.

| Bank-model field | Value |
|---|---|
| Bytes per subpartition | 64 KiB |
| Number of banks | 4 |
| Bank size | 16 KiB |
| Bank access width | 1024 bits (128 bytes) |
| Per-byte SRAM dynamic energy | 0.343 pJ (`get_sram_energy_per_byte_pj(8, 'register_file')`) |
| Wide-bank read energy | **43.89 pJ** (per access) |
| Wide-bank write energy | **54.86 pJ** (per access) |
| Reads per warp source operand | 1 (1024-bit bank matches 32 threads x 32 bits exactly) |

At the SM level, one SIMT instruction issues across 4 subpartitions in parallel. RF activity per cycle:

| Op kind | RF reads (SM-cycle) | RF writes (SM-cycle) |
|---|---|---|
| FADD / FMUL (2 sources) | 8 | 4 |
| FMA (3 sources) | 12 | 4 |

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

Single ALU stripped of all SIMT overhead: input flip-flops wired directly to the ALU inputs, output flip-flop on the result. No register file, no operand collector, no instruction fetch / decode / scheduler / dispatch. **All values in pJ**, for one ALU performing one op.

### 3.FADD

**FADD fp32**  (2 source operands, 1 FLOP/op)

| Stage (pJ) | FF_read | MUL | ADD | FF_write | Total |
|---|---|---|---|---|---|
| Energy | 0.107 | -- | 0.418 | 0.053 | 0.578 |

Per-op total: **0.578 pJ**, per-FLOP: **0.578 pJ/FLOP**.

**FADD fp16**  (2 source operands, 1 FLOP/op)

| Stage (pJ) | FF_read | MUL | ADD | FF_write | Total |
|---|---|---|---|---|---|
| Energy | 0.107 | -- | 0.209 | 0.053 | 0.369 |

Per-op total: **0.369 pJ**, per-FLOP: **0.369 pJ/FLOP**.

**FADD int8**  (2 source operands, 1 FLOP/op)

| Stage (pJ) | FF_read | MUL | ADD | FF_write | Total |
|---|---|---|---|---|---|
| Energy | 0.107 | -- | 0.084 | 0.053 | 0.244 |

Per-op total: **0.244 pJ**, per-FLOP: **0.244 pJ/FLOP**.

### 3.FMUL

**FMUL fp32**  (2 source operands, 1 FLOP/op)

| Stage (pJ) | FF_read | MUL | ADD | FF_write | Total |
|---|---|---|---|---|---|
| Energy | 0.107 | 1.672 | -- | 0.053 | 1.832 |

Per-op total: **1.832 pJ**, per-FLOP: **1.832 pJ/FLOP**.

**FMUL fp16**  (2 source operands, 1 FLOP/op)

| Stage (pJ) | FF_read | MUL | ADD | FF_write | Total |
|---|---|---|---|---|---|
| Energy | 0.107 | 0.836 | -- | 0.053 | 0.996 |

Per-op total: **0.996 pJ**, per-FLOP: **0.996 pJ/FLOP**.

**FMUL int8**  (2 source operands, 1 FLOP/op)

| Stage (pJ) | FF_read | MUL | ADD | FF_write | Total |
|---|---|---|---|---|---|
| Energy | 0.107 | 0.334 | -- | 0.053 | 0.495 |

Per-op total: **0.495 pJ**, per-FLOP: **0.495 pJ/FLOP**.

### 3.FMA

**FMA fp32**  (3 source operands, 2 FLOP/op)

| Stage (pJ) | FF_read | MUL | ADD | FF_write | Total |
|---|---|---|---|---|---|
| Energy | 0.160 | 1.672 | 0.228 | 0.053 | 2.114 |

Per-op total: **2.114 pJ**, per-FLOP: **1.057 pJ/FLOP**.

**FMA fp16**  (3 source operands, 2 FLOP/op)

| Stage (pJ) | FF_read | MUL | ADD | FF_write | Total |
|---|---|---|---|---|---|
| Energy | 0.160 | 0.836 | 0.114 | 0.053 | 1.164 |

Per-op total: **1.164 pJ**, per-FLOP: **0.582 pJ/FLOP**.

**FMA int8**  (3 source operands, 2 FLOP/op)

| Stage (pJ) | FF_read | MUL | ADD | FF_write | Total |
|---|---|---|---|---|---|
| Energy | 0.160 | 0.334 | 0.046 | 0.053 | 0.594 |

Per-op total: **0.594 pJ**, per-FLOP: **0.297 pJ/FLOP**.

## 4. SIMT pipeline energy (one SM-cycle, 128 lanes)

Rows trace each operand / control flow through the 9 pipeline stages; each cell is the **energy in pJ** attributable to that operation at that stage. All values are SM-level totals (the x4 subpartition fanout is already applied). The Stage total row at the bottom sums vertically; the Row total column on the right sums horizontally; the bold corner cell is the per-instruction total.

Each precision section also has a collapsible "Stage activity" table showing how each Stage total decomposes into `activity_count x pJ_each` (e.g. an FMA fp32 Rd column of 527 pJ resolves to `12 wide-bank reads x 43.89 pJ each`). Treat the cells in the main energy table as **pJ totals**, not access counts.

### 4.FADD

**FADD fp32** -- 128 lanes x 1 packed = 128 ops (128 FLOPS)

*All values in pJ; SM-level totals.*

| Operation | Fch | Dec | Sch | Dsp | Rd | OC | Disp | Exe | WB | Row total |
|---|---|---|---|---|---|---|---|---|---|---|
| Instruction control | 2.280 | 1.216 | 0.608 | 0.760 | -- | -- | -- | -- | -- | 4.864 |
| Src operand A | -- | -- | -- | -- | 176 | 17.6 | 43.9 | -- | -- | 237 |
| Src operand B | -- | -- | -- | -- | 176 | 17.6 | 43.9 | -- | -- | 237 |
| ALU compute | -- | -- | -- | -- | -- | -- | -- | 53.5 | -- | 53.5 |
| Dest writeback | -- | -- | -- | -- | -- | -- | -- | -- | 219 | 219 |
| **Stage total** | 2.280 | 1.216 | 0.608 | 0.760 | 351 | 35.1 | 87.8 | 53.5 | 219 | **752** |

<details><summary>Stage activity (count x pJ/event)</summary>

| Stage | Activity count (per SM-cycle) | pJ / event | Total pJ |
|---|---|---|---|
| Fch | 4 | 0.570 | 2.280 |
| Dec | 4 | 0.304 | 1.216 |
| Sch | 4 | 0.152 | 0.608 |
| Dsp | 4 | 0.190 | 0.760 |
| Rd | 8 | 43.9 | 351 |
| OC | 8 | 4.389 | 35.1 |
| Disp | 256 | 0.343 | 87.8 |
| Exe | 128 | 0.418 | 53.5 |
| WB | 4 | 54.9 | 219 |

</details>

Per-instruction: **752 pJ**, per-op: **5.873 pJ/op**, per-FLOP: **5.873 pJ**.

**FADD fp16** -- 128 lanes x 2 packed = 256 ops (256 FLOPS)

*All values in pJ; SM-level totals.*

| Operation | Fch | Dec | Sch | Dsp | Rd | OC | Disp | Exe | WB | Row total |
|---|---|---|---|---|---|---|---|---|---|---|
| Instruction control | 2.280 | 1.216 | 0.608 | 0.760 | -- | -- | -- | -- | -- | 4.864 |
| Src operand A | -- | -- | -- | -- | 176 | 17.6 | 43.9 | -- | -- | 237 |
| Src operand B | -- | -- | -- | -- | 176 | 17.6 | 43.9 | -- | -- | 237 |
| ALU compute | -- | -- | -- | -- | -- | -- | -- | 53.5 | -- | 53.5 |
| Dest writeback | -- | -- | -- | -- | -- | -- | -- | -- | 219 | 219 |
| **Stage total** | 2.280 | 1.216 | 0.608 | 0.760 | 351 | 35.1 | 87.8 | 53.5 | 219 | **752** |

<details><summary>Stage activity (count x pJ/event)</summary>

| Stage | Activity count (per SM-cycle) | pJ / event | Total pJ |
|---|---|---|---|
| Fch | 4 | 0.570 | 2.280 |
| Dec | 4 | 0.304 | 1.216 |
| Sch | 4 | 0.152 | 0.608 |
| Dsp | 4 | 0.190 | 0.760 |
| Rd | 8 | 43.9 | 351 |
| OC | 8 | 4.389 | 35.1 |
| Disp | 256 | 0.343 | 87.8 |
| Exe | 128 | 0.418 | 53.5 |
| WB | 4 | 54.9 | 219 |

</details>

Per-instruction: **752 pJ**, per-op: **2.937 pJ/op**, per-FLOP: **2.937 pJ**.

**FADD int8** -- 128 lanes x 4 packed = 512 ops (512 IntOPS)

*All values in pJ; SM-level totals.*

| Operation | Fch | Dec | Sch | Dsp | Rd | OC | Disp | Exe | WB | Row total |
|---|---|---|---|---|---|---|---|---|---|---|
| Instruction control | 2.280 | 1.216 | 0.608 | 0.760 | -- | -- | -- | -- | -- | 4.864 |
| Src operand A | -- | -- | -- | -- | 176 | 17.6 | 43.9 | -- | -- | 237 |
| Src operand B | -- | -- | -- | -- | 176 | 17.6 | 43.9 | -- | -- | 237 |
| ALU compute | -- | -- | -- | -- | -- | -- | -- | 42.8 | -- | 42.8 |
| Dest writeback | -- | -- | -- | -- | -- | -- | -- | -- | 219 | 219 |
| **Stage total** | 2.280 | 1.216 | 0.608 | 0.760 | 351 | 35.1 | 87.8 | 42.8 | 219 | **741** |

<details><summary>Stage activity (count x pJ/event)</summary>

| Stage | Activity count (per SM-cycle) | pJ / event | Total pJ |
|---|---|---|---|
| Fch | 4 | 0.570 | 2.280 |
| Dec | 4 | 0.304 | 1.216 |
| Sch | 4 | 0.152 | 0.608 |
| Dsp | 4 | 0.190 | 0.760 |
| Rd | 8 | 43.9 | 351 |
| OC | 8 | 4.389 | 35.1 |
| Disp | 256 | 0.343 | 87.8 |
| Exe | 128 | 0.334 | 42.8 |
| WB | 4 | 54.9 | 219 |

</details>

Per-instruction: **741 pJ**, per-op: **1.447 pJ/op**, per-IntOP: **1.447 pJ**.

### 4.FMUL

**FMUL fp32** -- 128 lanes x 1 packed = 128 ops (128 FLOPS)

*All values in pJ; SM-level totals.*

| Operation | Fch | Dec | Sch | Dsp | Rd | OC | Disp | Exe | WB | Row total |
|---|---|---|---|---|---|---|---|---|---|---|
| Instruction control | 2.280 | 1.216 | 0.608 | 0.760 | -- | -- | -- | -- | -- | 4.864 |
| Src operand A | -- | -- | -- | -- | 176 | 17.6 | 43.9 | -- | -- | 237 |
| Src operand B | -- | -- | -- | -- | 176 | 17.6 | 43.9 | -- | -- | 237 |
| ALU compute | -- | -- | -- | -- | -- | -- | -- | 214 | -- | 214 |
| Dest writeback | -- | -- | -- | -- | -- | -- | -- | -- | 219 | 219 |
| **Stage total** | 2.280 | 1.216 | 0.608 | 0.760 | 351 | 35.1 | 87.8 | 214 | 219 | **912** |

<details><summary>Stage activity (count x pJ/event)</summary>

| Stage | Activity count (per SM-cycle) | pJ / event | Total pJ |
|---|---|---|---|
| Fch | 4 | 0.570 | 2.280 |
| Dec | 4 | 0.304 | 1.216 |
| Sch | 4 | 0.152 | 0.608 |
| Dsp | 4 | 0.190 | 0.760 |
| Rd | 8 | 43.9 | 351 |
| OC | 8 | 4.389 | 35.1 |
| Disp | 256 | 0.343 | 87.8 |
| Exe | 128 | 1.672 | 214 |
| WB | 4 | 54.9 | 219 |

</details>

Per-instruction: **912 pJ**, per-op: **7.127 pJ/op**, per-FLOP: **7.127 pJ**.

**FMUL fp16** -- 128 lanes x 2 packed = 256 ops (256 FLOPS)

*All values in pJ; SM-level totals.*

| Operation | Fch | Dec | Sch | Dsp | Rd | OC | Disp | Exe | WB | Row total |
|---|---|---|---|---|---|---|---|---|---|---|
| Instruction control | 2.280 | 1.216 | 0.608 | 0.760 | -- | -- | -- | -- | -- | 4.864 |
| Src operand A | -- | -- | -- | -- | 176 | 17.6 | 43.9 | -- | -- | 237 |
| Src operand B | -- | -- | -- | -- | 176 | 17.6 | 43.9 | -- | -- | 237 |
| ALU compute | -- | -- | -- | -- | -- | -- | -- | 214 | -- | 214 |
| Dest writeback | -- | -- | -- | -- | -- | -- | -- | -- | 219 | 219 |
| **Stage total** | 2.280 | 1.216 | 0.608 | 0.760 | 351 | 35.1 | 87.8 | 214 | 219 | **912** |

<details><summary>Stage activity (count x pJ/event)</summary>

| Stage | Activity count (per SM-cycle) | pJ / event | Total pJ |
|---|---|---|---|
| Fch | 4 | 0.570 | 2.280 |
| Dec | 4 | 0.304 | 1.216 |
| Sch | 4 | 0.152 | 0.608 |
| Dsp | 4 | 0.190 | 0.760 |
| Rd | 8 | 43.9 | 351 |
| OC | 8 | 4.389 | 35.1 |
| Disp | 256 | 0.343 | 87.8 |
| Exe | 128 | 1.672 | 214 |
| WB | 4 | 54.9 | 219 |

</details>

Per-instruction: **912 pJ**, per-op: **3.564 pJ/op**, per-FLOP: **3.564 pJ**.

**FMUL int8** -- 128 lanes x 4 packed = 512 ops (512 IntOPS)

*All values in pJ; SM-level totals.*

| Operation | Fch | Dec | Sch | Dsp | Rd | OC | Disp | Exe | WB | Row total |
|---|---|---|---|---|---|---|---|---|---|---|
| Instruction control | 2.280 | 1.216 | 0.608 | 0.760 | -- | -- | -- | -- | -- | 4.864 |
| Src operand A | -- | -- | -- | -- | 176 | 17.6 | 43.9 | -- | -- | 237 |
| Src operand B | -- | -- | -- | -- | 176 | 17.6 | 43.9 | -- | -- | 237 |
| ALU compute | -- | -- | -- | -- | -- | -- | -- | 171 | -- | 171 |
| Dest writeback | -- | -- | -- | -- | -- | -- | -- | -- | 219 | 219 |
| **Stage total** | 2.280 | 1.216 | 0.608 | 0.760 | 351 | 35.1 | 87.8 | 171 | 219 | **869** |

<details><summary>Stage activity (count x pJ/event)</summary>

| Stage | Activity count (per SM-cycle) | pJ / event | Total pJ |
|---|---|---|---|
| Fch | 4 | 0.570 | 2.280 |
| Dec | 4 | 0.304 | 1.216 |
| Sch | 4 | 0.152 | 0.608 |
| Dsp | 4 | 0.190 | 0.760 |
| Rd | 8 | 43.9 | 351 |
| OC | 8 | 4.389 | 35.1 |
| Disp | 256 | 0.343 | 87.8 |
| Exe | 128 | 1.338 | 171 |
| WB | 4 | 54.9 | 219 |

</details>

Per-instruction: **869 pJ**, per-op: **1.698 pJ/op**, per-IntOP: **1.698 pJ**.

### 4.FMA

**FMA fp32** -- 128 lanes x 1 packed = 128 ops (256 FLOPS)

*All values in pJ; SM-level totals.*

| Operation | Fch | Dec | Sch | Dsp | Rd | OC | Disp | Exe | WB | Row total |
|---|---|---|---|---|---|---|---|---|---|---|
| Instruction control | 2.280 | 1.216 | 0.608 | 0.760 | -- | -- | -- | -- | -- | 4.864 |
| Src operand A | -- | -- | -- | -- | 176 | 17.6 | 43.9 | -- | -- | 237 |
| Src operand B | -- | -- | -- | -- | 176 | 17.6 | 43.9 | -- | -- | 237 |
| Src operand C | -- | -- | -- | -- | 176 | 17.6 | 43.9 | -- | -- | 237 |
| ALU compute | -- | -- | -- | -- | -- | -- | -- | 243 | -- | 243 |
| Dest writeback | -- | -- | -- | -- | -- | -- | -- | -- | 219 | 219 |
| **Stage total** | 2.280 | 1.216 | 0.608 | 0.760 | 527 | 52.7 | 132 | 243 | 219 | **1178** |

<details><summary>Stage activity (count x pJ/event)</summary>

| Stage | Activity count (per SM-cycle) | pJ / event | Total pJ |
|---|---|---|---|
| Fch | 4 | 0.570 | 2.280 |
| Dec | 4 | 0.304 | 1.216 |
| Sch | 4 | 0.152 | 0.608 |
| Dsp | 4 | 0.190 | 0.760 |
| Rd | 12 | 43.9 | 527 |
| OC | 12 | 4.389 | 52.7 |
| Disp | 384 | 0.343 | 132 |
| Exe | 128 | 1.900 | 243 |
| WB | 4 | 54.9 | 219 |

</details>

Per-instruction: **1178 pJ**, per-op: **9.207 pJ/op**, per-FLOP: **4.603 pJ**.

**FMA fp16** -- 128 lanes x 2 packed = 256 ops (512 FLOPS)

*All values in pJ; SM-level totals.*

| Operation | Fch | Dec | Sch | Dsp | Rd | OC | Disp | Exe | WB | Row total |
|---|---|---|---|---|---|---|---|---|---|---|
| Instruction control | 2.280 | 1.216 | 0.608 | 0.760 | -- | -- | -- | -- | -- | 4.864 |
| Src operand A | -- | -- | -- | -- | 176 | 17.6 | 43.9 | -- | -- | 237 |
| Src operand B | -- | -- | -- | -- | 176 | 17.6 | 43.9 | -- | -- | 237 |
| Src operand C | -- | -- | -- | -- | 176 | 17.6 | 43.9 | -- | -- | 237 |
| ALU compute | -- | -- | -- | -- | -- | -- | -- | 243 | -- | 243 |
| Dest writeback | -- | -- | -- | -- | -- | -- | -- | -- | 219 | 219 |
| **Stage total** | 2.280 | 1.216 | 0.608 | 0.760 | 527 | 52.7 | 132 | 243 | 219 | **1178** |

<details><summary>Stage activity (count x pJ/event)</summary>

| Stage | Activity count (per SM-cycle) | pJ / event | Total pJ |
|---|---|---|---|
| Fch | 4 | 0.570 | 2.280 |
| Dec | 4 | 0.304 | 1.216 |
| Sch | 4 | 0.152 | 0.608 |
| Dsp | 4 | 0.190 | 0.760 |
| Rd | 12 | 43.9 | 527 |
| OC | 12 | 4.389 | 52.7 |
| Disp | 384 | 0.343 | 132 |
| Exe | 128 | 1.900 | 243 |
| WB | 4 | 54.9 | 219 |

</details>

Per-instruction: **1178 pJ**, per-op: **4.603 pJ/op**, per-FLOP: **2.302 pJ**.

**FMA int8** -- 128 lanes x 4 packed = 512 ops (1024 IntOPS)

*All values in pJ; SM-level totals.*

| Operation | Fch | Dec | Sch | Dsp | Rd | OC | Disp | Exe | WB | Row total |
|---|---|---|---|---|---|---|---|---|---|---|
| Instruction control | 2.280 | 1.216 | 0.608 | 0.760 | -- | -- | -- | -- | -- | 4.864 |
| Src operand A | -- | -- | -- | -- | 176 | 17.6 | 43.9 | -- | -- | 237 |
| Src operand B | -- | -- | -- | -- | 176 | 17.6 | 43.9 | -- | -- | 237 |
| Src operand C | -- | -- | -- | -- | 176 | 17.6 | 43.9 | -- | -- | 237 |
| ALU compute | -- | -- | -- | -- | -- | -- | -- | 195 | -- | 195 |
| Dest writeback | -- | -- | -- | -- | -- | -- | -- | -- | 219 | 219 |
| **Stage total** | 2.280 | 1.216 | 0.608 | 0.760 | 527 | 52.7 | 132 | 195 | 219 | **1130** |

<details><summary>Stage activity (count x pJ/event)</summary>

| Stage | Activity count (per SM-cycle) | pJ / event | Total pJ |
|---|---|---|---|
| Fch | 4 | 0.570 | 2.280 |
| Dec | 4 | 0.304 | 1.216 |
| Sch | 4 | 0.152 | 0.608 |
| Dsp | 4 | 0.190 | 0.760 |
| Rd | 12 | 43.9 | 527 |
| OC | 12 | 4.389 | 52.7 |
| Disp | 384 | 0.343 | 132 |
| Exe | 128 | 1.520 | 195 |
| WB | 4 | 54.9 | 219 |

</details>

Per-instruction: **1130 pJ**, per-op: **2.207 pJ/op**, per-IntOP: **1.103 pJ**.

## 5. Cross-comparison summary

Headline: per-op energy in the full SIMT pipeline vs the irreducible baseline ALU. Ratio = SIMT-overhead tax.

| Op | Precision | Baseline pJ/op | SIMT pJ/op | SIMT pJ/FLOP | SIMT/Baseline | Compute % | RF+OC+wire % |
|---|---|---|---|---|---|---|---|
| FADD | fp32 | 0.578 | 5.873 | 5.873 | 10.2x | 7.1% | 92.2% |
| FADD | fp16 | 0.369 | 2.937 | 2.937 | 8.0x | 7.1% | 92.2% |
| FADD | int8 | 0.244 | 1.447 | 1.447 | 5.9x | 5.8% | 93.6% |
| FMUL | fp32 | 1.832 | 7.127 | 7.127 | 3.9x | 23.5% | 76.0% |
| FMUL | fp16 | 0.996 | 3.564 | 3.564 | 3.6x | 23.5% | 76.0% |
| FMUL | int8 | 0.495 | 1.698 | 1.698 | 3.4x | 19.7% | 79.7% |
| FMA | fp32 | 2.114 | 9.207 | 4.603 | 4.4x | 20.6% | 78.9% |
| FMA | fp16 | 1.164 | 4.603 | 2.302 | 4.0x | 20.6% | 78.9% |
| FMA | int8 | 0.594 | 2.207 | 1.103 | 3.7x | 17.2% | 82.3% |

## 6. Architectural reading

1. **Banked SRAM RF traffic dominates SIMT energy.** Stage 5 (Rd) + stage 9 (WB) together exceed the ALU compute (stage 8) for every (op, precision) combo. This is the GPU's architectural "tax" for keeping hundreds of warps in flight: a 64 KiB banked RF per subpartition that costs tens of pJ per wide-bank access. Accelerators that eliminate the RF (TPU systolic, KPU dataflow, CGRA spatial) skip this tax entirely -- which is exactly the energy gap this report quantifies.

2. **Cheap compute = high overhead share.** FADD has the smallest ALU energy per op, so a larger fraction of the instruction energy is fixed RF + control overhead. The SIMT/baseline ratio is highest for FADD and lowest for FMA.

3. **Packing recovers narrow-precision energy efficiency.** Per-instruction energy is approximately constant across fp32 / fp16-packed / int8-packed, because the same 32-bit datapath does 1, 2, or 4 useful ops respectively, AND the same wide-bank RF reads supply data regardless of packing. Per-op energy halves (fp16) or quarters (int8) -- not from shrinking the RF, but from amortizing it over more useful work.

4. **Per-subpartition control is x4 at SM level.** Each Ampere subpartition runs its own fetch / decode / scheduler / dispatch. Counting any of these once at SM level (the naive accounting) under-reports the control-overhead share by 4x. Same applies to the RF: each subpartition's banks fire independently in parallel.

5. **The wide-bank read assumption.** This model assumes the 1024-bit bank width matches a warp's source operand exactly (32 threads x 32 bits) so 1 wide-bank read suffices per source. Real GPUs have register-bank-conflict cycles when two source operands map to the same bank; the operand collector hides these by buffering operands across cycles. Modeling bank conflicts is future work; the current model captures the no-conflict case (the optimistic floor for RF energy).

6. **Operand collector is poorly characterized in public literature.** This report models it as a per-operand wide-buffer flop write + read at 5% of bank-read energy. Real GPU operand collectors are multi-entry with crossbars to RF banks. Treat the OC column as INTERPOLATED.

## 7. Self-validation

These are the sanity checks the test suite runs against the model output.

- fp16-packed / fp32 per-op ratio in [0.45, 0.55] for every op.
- int8-packed / fp32 per-op ratio in [0.20, 0.30] for every op.
- FMA ALU energy / FMUL ALU energy in [1.10, 1.18] (FMA = MUL + small ADD).
- SIMT FADD overhead ratio > FMUL > FMA at every precision.
- Stages 1-4 each fire `sm_subpartitions` times, not once.
- **RF traffic (Rd + WB) > ALU compute (Exe).** This is the   architectural punchline; if it ever flips, the bank model   is mis-parameterised.
- Default Ampere bank model: 4 banks of 16 KiB / 1024-bit wide   -> 1 wide-bank read per warp source operand.
- Per-bank read energy at 8 nm in [30, 80] pJ.
