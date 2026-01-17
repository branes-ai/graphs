# Chip Area estimates

**In 12nm-class processes, a 1MB on-chip SRAM block typically occupies ~2.1–2.5 mm², scaling roughly linearly up to ~67–80 mm² for 32MB.**

Here's a breakdown based on available data and extrapolated estimates:

---

### Estimated SRAM Area by Capacity (12nm-class nodes)

| Memory Size | Estimated Area (mm²) | Notes |
|-------------|----------------------|-------|
| **1 MB**    | 2.1–2.5 mm²          | Based on ~0.021 µm² bitcell size |
| **2 MB**    | 4.2–5.0 mm²          | Linear scaling assumption |
| **4 MB**    | 8.4–10.0 mm²         | — |
| **8 MB**    | 16.8–20.0 mm²        | — |
| **16 MB**   | 33.6–40.0 mm²        | — |
| **32 MB**   | 67.2–80.0 mm²        | — |

**Sources:**
- TSMC’s 12nm FinFET Compact Plus (12FFC+) SRAM bitcell size is ~0.021 µm².
- SRAM density at this node is ~38 Mb/mm² for newer nodes like N2, but closer to ~32–38 Mb/mm² for 12nm-class nodes.

---

### Notes on Process Variants

- **TSMC 12FFC+**: Optimized for low leakage and embedded memory; SRAM scaling is mature but not as dense as newer nodes.
- **GlobalFoundries 12LP+**: Similar SRAM density; area estimates are comparable.
- **Intel 12nm-class (e.g., Intel 4)**: Not directly marketed as 12nm, but SRAM density is in the same ballpark. Intel often uses high-density SRAM libraries for cache blocks.

---

### Area Estimation Method

- 1 Byte = 8 bits → 1MB = 8 million bits
- At 0.021 µm² per bitcell → 1MB = 168 mm² raw bitcell area
- With peripheral overhead and layout inefficiencies, usable density is ~32–38 Mb/mm² → 1MB ≈ 2.1–2.5 mm²

---

**Here's a CSV-style comparison of SRAM area and energy estimates across nodes from 16nm to 2nm for 1MB to 32MB blocks.** 

Area scales with bitcell density, while energy varies by access type and voltage scaling. These are high-level estimates based on published data and compiler outputs.

---

### SRAM Area and Energy Estimates Across Nodes (1MB–32MB)


| Node   | SRAM Density (Mb/mm²) | Bitcell Size (µm²) | 1MB Area (mm²) | 32MB Area (mm²) | Read Energy (pJ/bit) | Write Energy (pJ/bit) |
|--------|------------------------|---------------------|----------------|------------------|-----------------------|------------------------|
| 16nm   | 18                     | ~0.044              | 4.44           | 142.2            | 0.45–0.60             | 0.55–0.75              |
| 14nm   | 20                     | ~0.040              | 4.00           | 128.0            | 0.40–0.55             | 0.50–0.70              |
| 12nm   | 32                     | ~0.028–0.021        | 2.50–2.10      | 80.0–67.2        | 0.35–0.50             | 0.45–0.65              |
| 7nm    | 45                     | ~0.018–0.020        | 1.78           | 57.0             | 0.30–0.45             | 0.40–0.60              |
| 5nm    | 48                     | ~0.0175–0.018       | 1.67           | 53.4             | 0.28–0.42             | 0.38–0.55              |
| 3nm    | 38                     | ~0.021              | 2.10           | 67.2             | 0.25–0.40             | 0.35–0.50              |
| 2nm    | 38.1                   | ~0.0175             | 2.09           | 66.9             | 0.22–0.35             | 0.30–0.45              |

---


```
Node,SRAM Density (Mb/mm²),Bitcell Size (µm²),1MB Area (mm²),32MB Area (mm²),Read Energy (pJ/bit),Write Energy (pJ/bit)
16nm,18,~0.044,4.44,142.2,0.45–0.60,0.55–0.75
14nm,20,~0.040,4.00,128.0,0.40–0.55,0.50–0.70
12nm,32,~0.028–0.021,2.50–2.10,80.0–67.2,0.35–0.50,0.45–0.65
7nm,45,~0.018–0.020,1.78,57.0,0.30–0.45,0.40–0.60
5nm,48,~0.0175–0.018,1.67,53.4,0.28–0.42,0.38–0.55
3nm,38,~0.021,2.10,67.2,0.25–0.40,0.35–0.50
2nm,38.1,~0.0175,2.09,66.9,0.22–0.35,0.30–0.45
```

---

### Notes on Interpretation

- **Area**: Based on usable SRAM density (Mb/mm²), not raw bitcell area. Includes peripheral overhead.
- **Energy**: Read/write energy per bit varies by compiler settings, voltage, and access patterns. Lower nodes benefit from voltage scaling and improved transistor control (e.g., GAA at 2nm).
- **2nm**: TSMC’s N2 node achieves ~38.1 Mb/mm² with GAA nanosheets, enabling ~2.09 mm² for 1MB SRAM.
- **Compiler Data**: ARM Artisan and Synopsys DesignWare SRAM compilers provide Liberty models with dynamic/static power estimates.

---

**SRAMs in 7/6/5nm nodes are denser than in 4/3/2nm because SRAM scaling stalled at 3nm due to electrostatic and leakage challenges, while logic transistor density continued improving. Only at 2nm did SRAM scaling resume, thanks to gate-all-around (GAA) nanosheet transistors.**

---

### Why SRAM Density Peaked at 7/6/5nm

- **FinFET maturity**: Nodes like 7nm and 5nm used highly optimized FinFETs, which had reached peak SRAM scaling efficiency. Bitcell sizes dropped to ~0.018–0.020 µm², yielding densities of ~45–48 Mb/mm².
- **SRAM-friendly design rules**: These nodes had relaxed metal pitches and well-characterized SRAM libraries, allowing dense layout and efficient compiler integration.
- **Voltage and leakage balance**: FinFETs offered good electrostatic control and leakage suppression, enabling aggressive SRAM scaling without excessive power draw.

---

### Why 4nm and 3nm Didn't Improve SRAM Density

- **SRAM scaling stall**: TSMC’s N3B and N3E nodes retained SRAM bitcell sizes of ~0.021 µm²—identical to N5. No density improvement occurred.
- **Focus on logic scaling**: These nodes prioritized logic transistor density and performance, not SRAM. SRAM cells are more sensitive to process variation and require different optimization.
- **Electrostatic limits**: FinFETs began to struggle with short-channel effects and leakage at these dimensions, making further SRAM scaling impractical.

---

### Why 2nm Resumes SRAM Scaling

- **GAA nanosheets**: TSMC’s N2 node introduces gate-all-around transistors, which wrap the gate fully around the channel. This improves electrostatic control and reduces leakage.
- **Smaller bitcells**: N2 achieves SRAM bitcell sizes of ~0.0175 µm², enabling densities of ~38.1 Mb/mm²—finally surpassing N5 and N3.
- **Better scaling potential**: GAA allows tighter layout and more aggressive voltage scaling, making SRAM scaling viable again.

---

### 📉 Summary of SRAM Bitcell Sizes

| Node   | Bitcell Size (µm²) | SRAM Density (Mb/mm²) |
|--------|---------------------|------------------------|
| 7nm    | ~0.018–0.020        | ~45                    |
| 5nm    | ~0.0175–0.018       | ~48                    |
| 3nm    | ~0.021              | ~38                    |
| 2nm    | ~0.0175             | ~38.1                  |

---

Here’s the annotated diagram showing SRAM density trends across process nodes from 12nm to 2nm. It highlights the scaling stall at 3nm and the modest recovery at 2nm due to GAA nanosheets.


![SRAM Density Trend](./img/SRAM-density-scaling.png)

---

Here's a structured synthesis of what an **Energy-Delay Product (EDP) model** across nodes might look like, tailored for SRAM blocks ranging from 1MB to 32MB. This model helps quantify architectural tradeoffs between energy efficiency and latency, especially relevant for operand streaming and buffer design in your KPU pipeline work.

---

### EDP Model Structure Across Nodes

We define EDP as:

[
\text{EDP} = E \cdot D = (\text{Energy per access}) \cdot (\text{Access latency})
]

Where:
- **Energy per access** includes read/write energy per bit × total bits accessed.
- **Access latency** is typically in nanoseconds, influenced by process node, voltage, and compiler optimizations.

---

### 📊 Sample EDP Estimates (Per 1MB SRAM Block)

| Node   | Read Energy (pJ/bit) | Write Energy (pJ/bit) | Latency (ns) | EDP Read (pJ·ns) | EDP Write (pJ·ns) |
|--------|-----------------------|------------------------|---------------|------------------|-------------------|
| 16nm   | 0.55                  | 0.70                   | ~1.2          | 0.66             | 0.84              |
| 12nm   | 0.45                  | 0.60                   | ~1.0          | 0.45             | 0.60              |
| 7nm    | 0.35                  | 0.50                   | ~0.9          | 0.32             | 0.45              |
| 5nm    | 0.30                  | 0.45                   | ~0.8          | 0.24             | 0.36              |
| 3nm    | 0.28                  | 0.40                   | ~0.8          | 0.22             | 0.32              |
| 2nm    | 0.25                  | 0.35                   | ~0.7          | 0.18             | 0.25              |

---

### Architectural Implications

- **Lower nodes reduce both energy and latency**, improving EDP significantly.
- **SRAM scaling stall at 3nm** means EDP gains are mostly from voltage and transistor improvements, not density.
- **Operand streaming** benefits from lower EDP at 2nm, especially for high-throughput matrix ops.
- **Buffer sizing** can be tuned using EDP curves to balance occupancy vs. energy cost.

---

### Integration with KPU Loop Nest Modeling

You could annotate each loop level with:
- **Access count per tile**
- **Energy per access (read/write)**
- **Latency per access**
- **Total EDP per loop level**

This enables:
- **Gantt chart overlays** showing EDP per stage
- **Credit-based flow control** tuned to energy-delay thresholds
- **Dynamic operand injection** strategies that minimize EDP under concurrency

---

