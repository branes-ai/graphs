# Realistic TDP Model with SoC Infrastructure Overhead

## Problem Statement

The current `cli/estimate_tdp.py` shows a **linear relationship** between number of ALUs and TDP power. However, real SoCs require infrastructure to support those ALUs:

1. **On-Chip Memory (SRAM)** - To avoid memory controller bottleneck for compute-bound operations
2. **Interconnect Network** - To deliver operands to ALUs (H-tree, crossbar, mesh, CLOS)
3. **Power Management** - Idle/leakage power that doesn't scale linearly
4. **Control Overhead** - Schedulers, DMAs, coherence units

These components introduce **non-linear scaling** - doubling ALUs doesn't double performance if interconnect becomes a bottleneck.

---

## Physics Basis

### Current Model (Linear)

```
TDP = num_ALUs x frequency x energy_per_op
```

### Proposed Model (Non-Linear)

```
TDP = P_compute + P_sram + P_interconnect + P_control + P_idle

Where:
  P_compute      = num_ALUs x frequency x energy_per_op
  P_sram         = sram_bytes x sram_energy_per_byte x access_rate
  P_interconnect = f(num_ALUs, topology, wire_length)
  P_control      = f(num_ALUs, control_ratio)
  P_idle         = TDP_envelope x idle_fraction
```

---

## Component 1: On-Chip SRAM for Compute-Bound GEMM

### Why SRAM is Required

The **only BLAS operator that is compute-bound** is matrix multiplication (GEMM). All other operators (elementwise, reductions, convolutions via im2col) are memory-bound.

For GEMM to be compute-bound, we need sufficient on-chip memory to:
- Hold weight tiles between loads from DRAM
- Buffer activations for spatial reuse
- Store partial accumulations

### SRAM Sizing Formula

For an MxK x KxN matmul using tile sizes (Tm, Tk, Tn):

```
Operand FLOPs = 2 x M x K x N
Operand Bytes = (M x K + K x N + M x N) x bytes_per_element

Arithmetic Intensity (AI) = FLOPs / Bytes

For AI > roofline_knee:
  Minimum tile size: AI_required = peak_flops / peak_bandwidth

  Tile SRAM = (Tm x Tk + Tk x Tn + Tm x Tn) x bytes_per_element
```

**Empirical Relationship** (from existing hardware):

| Hardware | ALUs | On-Chip SRAM | SRAM/ALU Ratio |
|----------|------|--------------|----------------|
| TPU v4   | 32K  | 32 MiB       | 1 KB/ALU       |
| H100 SM  | 512  | 256 KB L1/SM | 512 B/ALU      |
| KPU T256 | 64K  | 64 MiB       | 1 KB/ALU       |

**Rule of Thumb**: ~512B to 1KB of SRAM per ALU to keep GEMM compute-bound.

### SRAM Power Model

```python
SRAM_BYTES_PER_ALU = 768  # 768 bytes per ALU (conservative)

def sram_power_watts(num_alus: int, process_nm: int, utilization: float = 1.0) -> float:
    """
    SRAM power from access energy.

    At full utilization, each ALU accesses SRAM every cycle for operand streaming.
    """
    sram_bytes = num_alus * SRAM_BYTES_PER_ALU

    # SRAM energy per byte (from docs/chip_area_estimates.md)
    sram_energy_pj = {
        3: 0.25, 4: 0.28, 5: 0.30, 7: 0.35, 8: 0.38,
        12: 0.45, 14: 0.50, 16: 0.55, 28: 0.65
    }.get(process_nm, 0.40)

    # Access rate: assume 4 bytes per ALU per cycle (FP32 operand)
    bytes_per_cycle = num_alus * 4 * utilization
    frequency_hz = 1.5e9  # Typical accelerator frequency
    bytes_per_second = bytes_per_cycle * frequency_hz

    # Static leakage adds ~30% to dynamic power for SRAM
    dynamic_power = bytes_per_second * sram_energy_pj * 1e-12
    static_power = dynamic_power * 0.30

    return dynamic_power + static_power
```

---

## Component 2: Interconnect Network

### Topology Options

| Topology | Bisection BW | Wire Cost | Latency | Best For |
|----------|-------------|-----------|---------|----------|
| **H-Tree** | O(N) | O(N log N) | O(log N) | Broadcast, reduction |
| **Crossbar** | O(N^2) | O(N^2) | O(1) | Small N (<64) |
| **2D Mesh** | O(sqrt(N)) | O(N) | O(sqrt(N)) | Spatial locality |
| **CLOS** | O(N) | O(N log N) | O(log N) | Rearrangeably non-blocking |

### Interconnect Power Model

```python
from enum import Enum
from math import sqrt, log2

class InterconnectTopology(Enum):
    HTREE = "h_tree"
    CROSSBAR = "crossbar"
    MESH_2D = "mesh_2d"
    CLOS = "clos"

# Wire energy: ~0.5 pJ/mm at 5nm (scales with process)
WIRE_ENERGY_PJ_PER_MM_5NM = 0.5

def interconnect_power_watts(
    num_alus: int,
    topology: InterconnectTopology,
    process_nm: int,
    frequency_ghz: float,
    data_width_bits: int = 32,
) -> float:
    """
    Interconnect power based on topology and wire length.

    Key insight: Power scales super-linearly because:
    1. More ALUs = longer wires to reach distant units
    2. More bandwidth required to feed all ALUs
    """
    # Wire energy scales with process node
    wire_energy_pj = WIRE_ENERGY_PJ_PER_MM_5NM * (process_nm / 5.0)

    # Chip area grows with ALU count (assume 0.1 mm^2 per ALU at 5nm)
    alu_area_mm2 = 0.1 * (process_nm / 5.0) ** 2
    total_area_mm2 = num_alus * alu_area_mm2 * 2  # 2x for SRAM
    chip_side_mm = sqrt(total_area_mm2)

    # Average wire length depends on topology
    if topology == InterconnectTopology.HTREE:
        # H-tree: average wire length is O(chip_side / log(N))
        avg_wire_mm = chip_side_mm / max(1, log2(num_alus) / 2)
        num_wires = num_alus * log2(num_alus)

    elif topology == InterconnectTopology.CROSSBAR:
        # Crossbar: all-to-all, but wire length bounded by chip size
        avg_wire_mm = chip_side_mm / 2
        num_wires = num_alus * sqrt(num_alus)  # Practical crossbar is O(N^1.5)

    elif topology == InterconnectTopology.MESH_2D:
        # Mesh: local connections, avg hop = sqrt(N)/2
        avg_wire_mm = chip_side_mm / sqrt(num_alus) * 2
        num_wires = num_alus * 4  # 4 neighbors per node

    elif topology == InterconnectTopology.CLOS:
        # CLOS: log stages, each wire spans fraction of chip
        stages = int(log2(num_alus) / 2) + 1
        avg_wire_mm = chip_side_mm / stages
        num_wires = num_alus * log2(num_alus) * 2

    # Energy per bit transmission
    energy_per_bit_pj = wire_energy_pj * avg_wire_mm

    # Bandwidth: each ALU needs data_width bits per cycle
    bits_per_second = num_alus * data_width_bits * frequency_ghz * 1e9

    # Power = energy per bit x bits per second x effective wire multiplier
    wire_multiplier = num_wires / num_alus  # Amortized wires per ALU
    power_watts = energy_per_bit_pj * 1e-12 * bits_per_second * wire_multiplier / num_alus

    # Account for router/switch power (adds ~50%)
    router_overhead = 1.5

    return power_watts * router_overhead * num_alus
```

---

## Component 3: Control Overhead

### Control Structures Required

1. **DMA Controllers**: 1 per memory interface (~0.1W each)
2. **Schedulers**: 1 per compute cluster (~0.05W each)
3. **Coherence Units**: For GPU-style architectures (~0.2W per SM)
4. **Clock Distribution**: Scales with chip area

```python
def control_power_watts(
    num_alus: int,
    circuit_type: str,
    frequency_ghz: float,
    process_nm: int,
) -> float:
    """
    Control overhead power based on architecture type.

    Different architectures have different control structures:
    - CPU/GPU: Heavy control (coherence, scheduling, OoO)
    - TPU: Light control (fixed systolic schedule)
    - KPU: Moderate control (domain tracking)
    """
    # Base control energy per ALU (pJ per cycle)
    control_energy_per_alu_pj = {
        'x86_performance': 2.0,    # Heavy OoO, branch prediction
        'x86_efficiency': 1.0,     # Simpler in-order
        'arm_performance': 1.5,
        'arm_efficiency': 0.5,
        'cuda_core': 1.5,          # Warp scheduling, coherence
        'tensor_core': 1.0,        # Shared warp scheduler
        'systolic_mac': 0.2,       # Fixed schedule, minimal control
        'domain_flow': 0.3,        # Domain tracking
    }.get(circuit_type, 0.5)

    # Scale with process node
    control_energy_pj = control_energy_per_alu_pj * (process_nm / 7.0)

    # ALUs per control unit (clustering factor)
    alus_per_cluster = {
        'x86_performance': 4,      # 4 ALUs per core
        'x86_efficiency': 2,
        'arm_performance': 4,
        'arm_efficiency': 2,
        'cuda_core': 32,           # 32 per warp
        'tensor_core': 64,         # 64 MACs per tensor core
        'systolic_mac': 256,       # 16x16 array
        'domain_flow': 256,        # 16x16 tile
    }.get(circuit_type, 64)

    num_clusters = max(1, num_alus // alus_per_cluster)

    # Power = clusters x frequency x energy per cluster
    energy_per_cluster_pj = control_energy_pj * alus_per_cluster
    power_watts = num_clusters * frequency_ghz * 1e9 * energy_per_cluster_pj * 1e-12

    # DMA controllers scale with memory interfaces (sqrt of ALU count)
    num_dma = int(sqrt(num_alus / 1024)) + 1
    dma_power = num_dma * 0.1 * (7.0 / process_nm)  # 0.1W at 7nm

    return power_watts + dma_power
```

---

## Component 4: Idle/Leakage Power

### The 50% Rule

From existing analysis in `test_power_modeling.py`:

```
P_idle = TDP_envelope x 0.5
```

This is a **constant floor** regardless of utilization due to nanoscale leakage.

```python
def idle_power_watts(tdp_envelope_watts: float) -> float:
    """
    Idle power is ~50% of TDP envelope at modern process nodes.

    This represents:
    - Subthreshold leakage (always present)
    - Gate leakage
    - Junction leakage
    - Clock tree power (even when gated)
    """
    IDLE_POWER_FRACTION = 0.50
    return tdp_envelope_watts * IDLE_POWER_FRACTION
```

---

## Integrated TDP Model

### Complete Formula

```python
@dataclass
class RealisticTDPEstimate:
    """Complete TDP breakdown with infrastructure overhead."""
    num_alus: int
    process_node_nm: int
    circuit_type: str
    topology: InterconnectTopology

    # Component powers
    compute_power_w: float
    sram_power_w: float
    interconnect_power_w: float
    control_power_w: float
    idle_power_w: float

    # Totals
    total_dynamic_power_w: float
    total_tdp_w: float

    # Efficiency metrics
    compute_fraction: float  # What % of TDP is "useful" compute
    infrastructure_overhead: float  # Overhead multiplier


def estimate_realistic_tdp(
    num_alus: int,
    process_node_nm: int,
    circuit_type: str,
    precision: str = 'FP32',
    frequency_ghz: float = None,
    topology: InterconnectTopology = InterconnectTopology.MESH_2D,
) -> RealisticTDPEstimate:
    """
    Estimate TDP with realistic infrastructure overhead.
    """
    # Get frequency default
    if frequency_ghz is None:
        frequency_ghz = DEFAULT_FREQUENCY_GHZ.get(circuit_type, 1.5)

    # 1. Compute power (existing model)
    base_energy_pj = get_process_base_energy_pj(process_node_nm)
    circuit_mult = CIRCUIT_TYPE_MULTIPLIER[circuit_type]
    precision_scale = PRECISION_ENERGY_SCALE[precision]
    ops_per_cycle = PRECISION_OPS_PER_CYCLE[precision]

    energy_per_op_pj = base_energy_pj * circuit_mult * precision_scale
    ops_per_second = num_alus * ops_per_cycle * frequency_ghz * 1e9
    compute_power = energy_per_op_pj * 1e-12 * ops_per_second

    # 2. SRAM power
    sram_power = sram_power_watts(num_alus, process_node_nm)

    # 3. Interconnect power
    interconnect_power = interconnect_power_watts(
        num_alus, topology, process_node_nm, frequency_ghz
    )

    # 4. Control power
    control_power = control_power_watts(
        num_alus, circuit_type, frequency_ghz, process_node_nm
    )

    # 5. Total dynamic power
    total_dynamic = compute_power + sram_power + interconnect_power + control_power

    # 6. Idle power (50% of total as floor)
    # Note: idle power sets the TDP envelope, then we add dynamic on top
    # But we need to estimate TDP envelope first...
    # Iterative approach: estimate envelope, then validate
    estimated_envelope = total_dynamic * 1.3  # Initial guess (30% headroom)
    idle_power = idle_power_watts(estimated_envelope)

    # Total TDP
    total_tdp = total_dynamic + idle_power

    # Compute fraction
    compute_fraction = compute_power / total_tdp
    infrastructure_overhead = total_tdp / compute_power

    return RealisticTDPEstimate(
        num_alus=num_alus,
        process_node_nm=process_node_nm,
        circuit_type=circuit_type,
        topology=topology,
        compute_power_w=compute_power,
        sram_power_w=sram_power,
        interconnect_power_w=interconnect_power,
        control_power_w=control_power,
        idle_power_w=idle_power,
        total_dynamic_power_w=total_dynamic,
        total_tdp_w=total_tdp,
        compute_fraction=compute_fraction,
        infrastructure_overhead=infrastructure_overhead,
    )
```

---

## Expected Non-Linear Behavior

### Scaling Curves

```
TDP vs ALU Count (5nm Tensor Core with 2D Mesh):

ALUs      Linear TDP    Realistic TDP    Overhead
------    ----------    -------------    --------
1,024         8 W           15 W          1.9x
4,096        32 W           72 W          2.3x
16,384      128 W          340 W          2.7x
65,536      512 W        1,600 W          3.1x
```

The overhead multiplier **increases with ALU count** because:

1. **Interconnect scales super-linearly**: Longer wires, more hops
2. **SRAM scales linearly**: But with ~30% leakage overhead
3. **Control scales sub-linearly**: Amortized over clusters
4. **Idle power scales with envelope**: Larger chip = more leakage

---

## Validation Targets

| Hardware | ALUs | Published TDP | Model Estimate |
|----------|------|---------------|----------------|
| H100 SXM | 16K TC | 700 W | Should be ~650-750 W |
| TPU v4 | 32K MAC | 350 W | Should be ~320-380 W |
| A100 SXM | 6.9K TC | 400 W | Should be ~380-420 W |

---

## Implementation Plan

### Phase 1: Core Infrastructure Model
1. Add `SoCInfrastructureModel` class to `technology_profile.py`
2. Implement SRAM power model with process scaling
3. Add interconnect topology enum and power models

### Phase 2: Integration with estimate_tdp.py
1. Add `--topology` flag (mesh/htree/crossbar/clos)
2. Add `--include-infrastructure` flag (default: True)
3. Update sweep functions to use realistic model

### Phase 3: Validation
1. Compare against published TDP for known hardware
2. Validate SRAM sizing against TPU/H100 specs
3. Tune constants based on real-world data

### Phase 4: Visualization
1. Stacked area chart: compute + sram + interconnect + control + idle
2. Non-linear sweep showing overhead growth
3. Topology comparison plot

---

## References

1. Horowitz, M. "1.1 Computing's energy problem (and what we can do about it)" ISSCC 2014
2. TSMC N5 SRAM compiler documentation (density, energy)
3. NVIDIA H100 whitepaper (SM count, TDP, memory)
4. Google TPU v4 architecture paper (systolic array, buffer sizing)
5. MIT Eyeriss project (interconnect energy modeling)
