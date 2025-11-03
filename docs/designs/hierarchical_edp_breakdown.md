# Hierarchical EDP Breakdown Design

**Date:** 2025-11-03
**Status:** Design Proposal
**Goal:** Enable per-operator EDP analysis to reveal architectural energy event differences

---

## Problem Statement

Current EDP implementation (Phase 1) calculates Energy-Delay Product at the **model level**:
- Total model energy × total model latency = single EDP value
- Shows which architecture "wins" overall
- **DOES NOT** reveal WHERE energy is spent or WHY architectures differ

### The Hidden Details

Consider an MLP layer: `Linear → Bias → ReLU`

**On TPU/KPU (Fused dataflow):**
- Linear: matmul in systolic array/dataflow fabric → **output stays in scratchpad**
- Bias: add in scratchpad → **output stays in scratchpad**
- ReLU: activation in scratchpad → **output written to DRAM**
- **Memory cycles:** 2 (input read, final output write)
- **Energy events:** 1× instruction setup, 1× DRAM write

**On CPU/GPU (Separate kernels):**
- Linear: matmul → **output written to DRAM**
- Bias: read from DRAM, add → **output written to DRAM**
- ReLU: read from DRAM, activate → **output written to DRAM**
- **Memory cycles:** 6 (3× read, 3× write)
- **Energy events:** 3× instruction fetch, 6× DRAM access

**Impact:**
- TPU/KPU: 2 DRAM accesses, 1 instruction setup
- CPU/GPU: 6 DRAM accesses, 3 instruction fetches
- **3× memory energy difference**
- **3× architectural overhead difference**
- **3× EDP difference for this operator sequence**

This difference is **completely hidden** in the current model-level EDP!

---

## Analysis of Existing Infrastructure

### 1. Data Structures (Already Available!)

#### SubgraphDescriptor (`ir/structures.py:115`)
```python
@dataclass
class SubgraphDescriptor:
    node_id: str
    node_name: str
    operation_type: OperationType
    fusion_pattern: str  # e.g., 'linear_bias_relu'

    flops: int
    total_input_bytes: int
    total_output_bytes: int
    total_weight_bytes: int

    # ... parallelism, dependencies, etc.
```

#### FusedSubgraph (`transform/partitioning/fusion_partitioner.py:37`)
```python
@dataclass
class FusedSubgraph:
    subgraph_id: int
    node_ids: List[str]           # ← Individual operators!
    node_names: List[str]          # ← Human-readable names!
    operation_types: List[OperationType]  # ← Operation types!

    total_flops: int
    total_input_bytes: int
    total_output_bytes: int
    internal_bytes: int  # ← KEY: data saved by fusion!

    fusion_pattern: str  # e.g., "Conv_BN_ReLU"
    num_operators: int   # ← How many ops fused
```

#### EnergyDescriptor (`analysis/energy.py:26`)
```python
@dataclass
class EnergyDescriptor:
    subgraph_id: str
    subgraph_name: str

    compute_energy_j: float
    memory_energy_j: float
    static_energy_j: float
    total_energy_j: float

    latency_s: float
    # ... utilization, efficiency, etc.
```

#### LatencyDescriptor (`analysis/roofline.py:29`)
```python
@dataclass
class LatencyDescriptor:
    subgraph_id: str
    subgraph_name: str

    compute_time: float
    memory_time: float
    actual_latency: float

    bottleneck: BottleneckType
    arithmetic_intensity: float
    # ... utilization, etc.
```

### 2. Key Observations

✅ **We already have per-subgraph energy and latency!**
- `EnergyDescriptor` has per-subgraph energy breakdown
- `LatencyDescriptor` has per-subgraph latency breakdown
- **We just need to combine them into per-subgraph EDP!**

✅ **We track fusion at subgraph level!**
- `FusedSubgraph.node_names` lists individual operators
- `internal_bytes` quantifies fusion savings
- `fusion_pattern` describes what's fused

❌ **We DON'T have per-operator energy/latency** (within a fused subgraph)
- Current energy/latency models operate at subgraph granularity
- Fused ops are treated as atomic unit
- **This is the gap we need to fill**

### 3. Architectural Energy Events (Per Architecture)

From `hardware/architectural_energy.py`:

**StoredProgramEnergyModel (CPU):**
```python
# Per instruction
instruction_fetch: 2.0 pJ
operand_fetch_overhead: 10.0 pJ
pipeline_control: 0.5 pJ/cycle

# If fused: 1× instruction fetch
# If separate: 3× instruction fetch (for linear, bias, relu)
```

**DataParallelEnergyModel (GPU):**
```python
# Inherits from CPU, plus:
coherence_per_request: 5.0 pJ
thread_scheduling: 1.0 pJ
warp_divergence: 3.0 pJ (for ReLU)

# If separate: 3× kernel launch overhead + coherence
```

**SystolicArrayEnergyModel (TPU):**
```python
# One-time setup
schedule_setup: 100.0 pJ (once per fused subgraph)

# Per element
data_injection: 0.5 pJ
data_extraction: 0.5 pJ

# Efficiency gains from spatial dataflow
compute_efficiency: 0.15  # 85% reduction!
memory_efficiency: 0.20   # 80% reduction!

# If fused: 1× setup, N× injection/extraction
# If separate: 3× setup, 3N× injection/extraction
```

**DomainFlowEnergyModel (KPU):**
```python
# Per operation
domain_tracking: 1.0 pJ
wavefront_control: 0.8 pJ

# Per schedule change
schedule_adaptation: 50.0 pJ

# Efficiency gains from programmable spatial dataflow
compute_efficiency: 0.30  # 70% reduction
memory_efficiency: 0.35   # 65% reduction

# If fused: 1× schedule, N× domain tracking
# If separate: 3× schedule, 3N× domain tracking
```

---

## Proposed Hierarchical EDP Design

### Level 0: Model-Level EDP (Already Implemented ✅)

```
Model Total EDP = Σ(subgraph EDPs)
```

**Example output:**
```
ResNet-18 Total EDP:
  KPU: 2.66 nJ·s  ← winner
  CPU: 21.47 nJ·s
  GPU: 21.09 nJ·s
  TPU: 35.16 nJ·s
```

### Level 1: Subgraph-Level EDP (NEW)

```
Subgraph EDP = (subgraph_energy) × (subgraph_latency)
```

**Data source:**
- Energy: `EnergyDescriptor` (already exists)
- Latency: `LatencyDescriptor` (already exists)
- **Just need to multiply and aggregate!**

**Example output:**
```
Subgraph Breakdown (Top 5 by EDP):

Subgraph                    Energy      Latency     EDP         % of Total  Fusion
------------------------------------------------------------------------------------
conv2_x.0.conv1            890.2 µJ    120.5 µs    107.25 nJ·s    40.3%    Conv_BN_ReLU (3 ops)
conv2_x.1.conv1            720.1 µJ    98.2 µs     70.71 nJ·s     26.6%    Conv_BN_ReLU (3 ops)
fc                         450.3 µJ    45.8 µs     20.62 nJ·s      7.7%    Linear_ReLU (2 ops)
conv1                      380.5 µJ    52.1 µs     19.82 nJ·s      7.5%    Conv_BN_ReLU (3 ops)
layer1.0.conv1             290.2 µJ    38.4 µs     11.14 nJ·s      4.2%    Conv_BN_ReLU (3 ops)
...
```

**Key insight:** Shows which subgraphs dominate EDP, revealing optimization targets.

### Level 2: Operator-Level EDP (NEW - Core Contribution)

```
Operator EDP (within subgraph) =
    estimated_operator_energy × estimated_operator_latency
```

#### Challenge: Estimating Per-Operator Contributions

Fused subgraphs are atomic execution units. We need to **decompose** energy/latency back to individual operators.

#### Approach 1: FLOP-Proportional Energy Decomposition

**Assumption:** Within a fused subgraph, energy is proportional to FLOPs.

```python
# For subgraph with operators: [Linear, Bias, ReLU]
total_subgraph_energy = 890.2 µJ
total_subgraph_flops = 1.8 GFLOPs

# Operator FLOPs (estimated from operator type)
linear_flops = 1.79 GFLOPs  # matmul dominates
bias_flops = 0.001 GFLOPs    # negligible
relu_flops = 0.009 GFLOPs    # comparisons

# Energy decomposition (proportional to FLOPs)
linear_energy = 890.2 µJ × (1.79 / 1.8) = 884.6 µJ
bias_energy = 890.2 µJ × (0.001 / 1.8) = 0.5 µJ
relu_energy = 890.2 µJ × (0.009 / 1.8) = 4.5 µJ
```

**Pros:**
- Simple, fast, no architectural modeling needed
- Reasonable for compute-bound operators

**Cons:**
- Ignores memory energy (which may dominate for bandwidth-bound ops)
- Doesn't account for architectural differences

#### Approach 2: Architectural Event Decomposition (RECOMMENDED)

**Assumption:** Different architectures have different event costs for each operator type.

**For CPU/GPU (separate kernels):**
```python
# Each operator pays full overhead
linear_events = {
    'instruction_fetch': 1,
    'operand_fetch': 2,  # weights + inputs
    'dram_write': 1,      # output
}

bias_events = {
    'instruction_fetch': 1,
    'operand_fetch': 1,  # bias vector
    'dram_read': 1,       # linear output
    'dram_write': 1,      # bias output
}

relu_events = {
    'instruction_fetch': 1,
    'operand_fetch': 0,
    'dram_read': 1,       # bias output
    'dram_write': 1,      # relu output
}

# Energy = Σ(events × energy_per_event)
linear_energy = (
    instruction_fetch_energy × 1 +
    operand_fetch_energy × 2 +
    dram_write_energy × 1 +
    compute_energy_from_flops
)
```

**For TPU/KPU (fused execution):**
```python
# Fused subgraph pays setup once, operators share dataflow
fused_events = {
    'schedule_setup': 1,           # one-time
    'data_injection': 1,           # input
    'data_extraction': 1,          # output
    'intermediate_scratchpad': 2,  # bias, relu stay in scratchpad
}

# Energy savings from fusion
internal_bytes_saved = bias_output_bytes + relu_input_bytes

# Per-operator allocation (heuristic)
# Linear: dominant operator, gets bulk of energy
linear_energy = total_fused_energy × 0.95
bias_energy = total_fused_energy × 0.03
relu_energy = total_fused_energy × 0.02
```

**Pros:**
- Captures architectural differences
- Reveals fusion benefits quantitatively
- Aligns with hardware reality

**Cons:**
- Requires architecture-specific event models
- More complex to implement
- Heuristics needed for energy apportionment

#### Approach 3: Hybrid Model (RECOMMENDED for Phase 1)

**Combine FLOP-proportional with architectural modifiers:**

```python
def estimate_operator_edp(
    operator_type: OperationType,
    operator_flops: int,
    subgraph_total_energy: float,
    subgraph_total_flops: int,
    architecture: ArchitectureClass,
    is_fused: bool
) -> float:
    """
    Estimate operator EDP within a fused subgraph.

    Returns (operator_energy, operator_latency, operator_edp)
    """

    # Base allocation (FLOP-proportional)
    flop_fraction = operator_flops / subgraph_total_flops
    base_energy = subgraph_total_energy * flop_fraction

    # Architectural modifier
    if architecture in [SYSTOLIC_ARRAY, DOMAIN_FLOW]:
        # Spatial architectures hide lightweight ops in dataflow
        if operator_type in [BIAS, RELU, ACTIVATION]:
            if is_fused:
                modifier = 0.1  # Almost free when fused
            else:
                modifier = 1.5  # Expensive if separate (setup overhead)
        else:
            modifier = 1.0
    else:  # CPU, GPU
        # Sequential architectures pay per-op overhead
        if operator_type in [BIAS, RELU]:
            if is_fused:
                modifier = 0.5  # Still some overhead, but shared
            else:
                modifier = 2.0  # Full overhead per kernel
        else:
            modifier = 1.0

    operator_energy = base_energy * modifier

    # Latency follows similar pattern
    flop_latency = operator_flops / peak_flops
    memory_latency = operator_bytes / peak_bandwidth
    base_latency = max(flop_latency, memory_latency)

    operator_latency = base_latency * modifier

    operator_edp = operator_energy * operator_latency

    return operator_energy, operator_latency, operator_edp
```

**Example output:**

```
Subgraph: fc (Linear_ReLU, 2 ops fused)
  Total EDP: 20.62 nJ·s

  Operator Breakdown:
  Operator    Energy      Latency     EDP         % of Subgraph  Architectural Impact
  ---------------------------------------------------------------------------------------
  Linear      450.0 µJ    45.5 µs     20.48 nJ·s     99.3%         Dominant (matmul)
  ReLU          0.3 µJ     0.2 µs      0.06 nJ·s      0.3%         Hidden in dataflow ⭐

  Fusion Benefit:
    If separate kernels:
      Linear:  450.0 µJ × 45.5 µs = 20.48 nJ·s  (same)
      ReLU:      4.5 µJ ×  3.2 µs =  0.14 nJ·s  (8× worse!) ❌
      Total:                       = 20.62 nJ·s  (current)
                             vs.     20.76 nJ·s  (if separated)

    Fusion saves: 0.14 nJ·s (0.7% of total) for this subgraph
```

---

## Level 3: Architecture-to-Architecture Comparison (NEW)

Show the **same operator** on **different architectures**:

```
Operator: ReLU (in fc layer)

Architecture  Energy      Latency     EDP         Overhead      Fused?
---------------------------------------------------------------------------
KPU           0.2 µJ      0.1 µs      0.02 nJ·s   Hidden ⭐     Yes (dataflow)
TPU           0.3 µJ      0.2 µs      0.06 nJ·s   Hidden ⭐     Yes (systolic)
CPU           2.1 µJ      1.8 µs      3.78 nJ·s   Moderate      Yes (SIMD)
GPU           4.5 µJ      3.2 µs     14.40 nJ·s   High ❌       No (kernel launch)

Key Insight:
  GPU pays 720× more EDP for ReLU than KPU!
  Reason: GPU kernel launch overhead (coherence machinery, thread scheduling)
          dominates lightweight activation function.

  KPU hides ReLU in wavefront dataflow - essentially free.
```

---

## Level 4: Fusion Impact Analysis (NEW)

Quantify **exactly how much EDP is saved** by fusion:

```
Fusion Analysis: Conv_BN_ReLU

SCENARIO 1: Fused (Current)
  Conv:  880.0 µJ × 118.0 µs = 103.84 nJ·s
  BN:      8.0 µJ ×   1.8 µs =   0.01 nJ·s  (hidden)
  ReLU:    2.2 µJ ×   0.7 µs =   0.00 nJ·s  (hidden)
  Total: 890.2 µJ × 120.5 µs = 107.25 nJ·s  ✅

SCENARIO 2: Separate Kernels
  Conv:  880.0 µJ × 118.0 µs = 103.84 nJ·s  (same)
  BN:     48.0 µJ ×  12.5 µs =   0.60 nJ·s  (6× worse, kernel overhead)
  ReLU:   32.0 µJ ×   8.8 µs =   0.28 nJ·s  (40× worse, kernel overhead)
  Total: 960.0 µJ × 139.3 µs = 133.73 nJ·s  ❌

Fusion Savings:
  Energy: 69.8 µJ (7.3% saved)
  Latency: 18.8 µs (13.5% saved)
  EDP: 26.48 nJ·s (19.8% saved)  ⭐

  Breakdown:
    - Eliminated 2 DRAM writes (BN output, ReLU input): 45 µJ saved
    - Eliminated 2 kernel launches: 12 µs saved
    - Reduced coherence overhead: 24.8 µJ saved
```

---

## Implementation Plan

### Phase 1: Subgraph-Level EDP (Low-Hanging Fruit)

**Effort:** Small (~100 lines)
**Impact:** High (reveals hotspot subgraphs)

**Tasks:**
1. Create `SubgraphEDPDescriptor`:
   ```python
   @dataclass
   class SubgraphEDPDescriptor:
       subgraph_id: str
       subgraph_name: str
       fusion_pattern: str
       num_operators: int

       energy_j: float
       latency_s: float
       edp: float  # energy × latency

       # Component EDPs
       compute_edp: float
       memory_edp: float
       architectural_edp: float

       # Percentage of total
       edp_fraction: float
   ```

2. Add method to `ArchitectureComparator`:
   ```python
   def get_subgraph_edp_breakdown(self, arch_name: str) -> List[SubgraphEDPDescriptor]:
       """Get per-subgraph EDP breakdown for one architecture"""
       result = self.results[arch_name]

       energy_descriptors = result.energy_report.energy_descriptors
       latency_descriptors = result.roofline_report.latencies

       subgraph_edps = []
       for i, (e_desc, l_desc) in enumerate(zip(energy_descriptors, latency_descriptors)):
           edp = e_desc.total_energy_j * l_desc.actual_latency

           subgraph_edps.append(SubgraphEDPDescriptor(
               subgraph_id=e_desc.subgraph_id,
               subgraph_name=e_desc.subgraph_name,
               fusion_pattern=get_fusion_pattern(result, e_desc.subgraph_id),
               num_operators=get_num_operators(result, e_desc.subgraph_id),
               energy_j=e_desc.total_energy_j,
               latency_s=l_desc.actual_latency,
               edp=edp,
               compute_edp=e_desc.compute_energy_j * l_desc.actual_latency,
               memory_edp=e_desc.memory_energy_j * l_desc.actual_latency,
               architectural_edp=0.0,  # Calculate from breakdown
               edp_fraction=0.0,  # Will be normalized
           ))

       # Normalize fractions
       total_edp = sum(d.edp for d in subgraph_edps)
       for d in subgraph_edps:
           d.edp_fraction = d.edp / total_edp if total_edp > 0 else 0.0

       return sorted(subgraph_edps, key=lambda x: x.edp, reverse=True)
   ```

3. Add reporting method:
   ```python
   def generate_subgraph_edp_report(self, arch_name: str, top_n: int = 10) -> str:
       """Generate subgraph EDP breakdown report"""
       subgraph_edps = self.get_subgraph_edp_breakdown(arch_name)

       lines = []
       lines.append(f"Subgraph EDP Breakdown ({arch_name})")
       lines.append("=" * 100)
       lines.append(f"{'Subgraph':<30} {'Energy':<12} {'Latency':<12} {'EDP':<15} {'% Total':<10} {'Pattern'}")
       lines.append("-" * 100)

       for sg in subgraph_edps[:top_n]:
           lines.append(
               f"{sg.subgraph_name:<30} "
               f"{sg.energy_j*1e6:<12.2f} µJ "
               f"{sg.latency_s*1e6:<12.2f} µs "
               f"{sg.edp*1e9:<15.2f} nJ·s "
               f"{sg.edp_fraction*100:<10.1f}% "
               f"{sg.fusion_pattern}"
           )

       return "\n".join(lines)
   ```

**Deliverable:** Shows which subgraphs contribute most to total EDP.

### Phase 2: Operator-Level EDP (Medium Effort)

**Effort:** Medium (~500 lines)
**Impact:** Very High (reveals fusion benefits and architectural differences)

**Tasks:**
1. Create `OperatorEDPDescriptor`:
   ```python
   @dataclass
   class OperatorEDPDescriptor:
       operator_id: str
       operator_name: str
       operator_type: OperationType

       # Within parent subgraph
       parent_subgraph_id: str
       is_fused: bool
       fusion_pattern: str

       # Estimated values
       energy_j: float
       latency_s: float
       edp: float

       # Architectural impact
       architectural_modifier: float  # 1.0 = neutral, <1 = efficient, >1 = overhead
       fusion_benefit_edp: float  # EDP saved by being fused

       # Allocation method
       allocation_method: str  # "flop_proportional", "architectural_model", "hybrid"
   ```

2. Implement energy allocation strategies:
   ```python
   class OperatorEDPAllocator:
       """Allocate subgraph EDP to individual operators"""

       def allocate_flop_proportional(
           self,
           operators: List[OperatorInfo],
           subgraph_energy: float,
           subgraph_latency: float
       ) -> List[OperatorEDPDescriptor]:
           """Simple FLOP-proportional allocation"""
           # Implementation from Approach 1

       def allocate_architectural(
           self,
           operators: List[OperatorInfo],
           subgraph_energy: float,
           subgraph_latency: float,
           architecture: ArchitectureClass,
           is_fused: bool
       ) -> List[OperatorEDPDescriptor]:
           """Architecture-aware allocation"""
           # Implementation from Approach 2

       def allocate_hybrid(
           self,
           operators: List[OperatorInfo],
           subgraph_energy: float,
           subgraph_latency: float,
           architecture: ArchitectureClass,
           is_fused: bool
       ) -> List[OperatorEDPDescriptor]:
           """Hybrid FLOP + architectural modifiers"""
           # Implementation from Approach 3 (RECOMMENDED)
   ```

3. Extract operator information from FusedSubgraph:
   ```python
   def extract_operator_info(subgraph: SubgraphDescriptor, partition_report: PartitionReport) -> List[OperatorInfo]:
       """Extract individual operator details from fused subgraph"""

       # Look up FusedSubgraph in partition_report
       if hasattr(partition_report, 'fusion_report'):
           fusion_report = partition_report.fusion_report
           fused_sg = next(
               (fs for fs in fusion_report.fused_subgraphs if fs.subgraph_id == subgraph.node_id),
               None
           )

           if fused_sg:
               operators = []
               for i, (node_id, node_name, op_type) in enumerate(zip(
                   fused_sg.node_ids,
                   fused_sg.node_names,
                   fused_sg.operation_types
               )):
                   # Estimate per-operator FLOPs
                   op_flops = estimate_operator_flops(op_type, subgraph.input_tensors, subgraph.output_tensors)

                   operators.append(OperatorInfo(
                       operator_id=node_id,
                       operator_name=node_name,
                       operator_type=op_type,
                       flops=op_flops,
                   ))

               return operators

       # Fallback: treat as single operator
       return [OperatorInfo(
           operator_id=subgraph.node_id,
           operator_name=subgraph.node_name,
           operator_type=subgraph.operation_type,
           flops=subgraph.flops,
       )]
   ```

4. Add architectural modifiers per operator type:
   ```python
   ARCHITECTURAL_MODIFIERS = {
       ArchitectureClass.SYSTOLIC_ARRAY: {
           OperationType.MATMUL: {'fused': 1.0, 'separate': 1.2},
           OperationType.CONV: {'fused': 1.0, 'separate': 1.2},
           OperationType.BIAS: {'fused': 0.05, 'separate': 1.5},   # Almost free if fused!
           OperationType.RELU: {'fused': 0.02, 'separate': 2.0},   # Almost free if fused!
           OperationType.BATCHNORM: {'fused': 0.1, 'separate': 1.8},
       },
       ArchitectureClass.DOMAIN_FLOW: {
           OperationType.MATMUL: {'fused': 1.0, 'separate': 1.1},
           OperationType.CONV: {'fused': 1.0, 'separate': 1.1},
           OperationType.BIAS: {'fused': 0.08, 'separate': 1.4},
           OperationType.RELU: {'fused': 0.05, 'separate': 1.6},
           OperationType.BATCHNORM: {'fused': 0.15, 'separate': 1.5},
       },
       ArchitectureClass.DATA_PARALLEL: {  # GPU
           OperationType.MATMUL: {'fused': 1.0, 'separate': 1.1},
           OperationType.CONV: {'fused': 1.0, 'separate': 1.1},
           OperationType.BIAS: {'fused': 0.3, 'separate': 2.5},   # Kernel launch overhead!
           OperationType.RELU: {'fused': 0.2, 'separate': 3.0},   # Kernel launch overhead!
           OperationType.BATCHNORM: {'fused': 0.5, 'separate': 2.0},
       },
       ArchitectureClass.STORED_PROGRAM: {  # CPU
           OperationType.MATMUL: {'fused': 1.0, 'separate': 1.05},
           OperationType.CONV: {'fused': 1.0, 'separate': 1.05},
           OperationType.BIAS: {'fused': 0.4, 'separate': 1.5},
           OperationType.RELU: {'fused': 0.3, 'separate': 1.8},
           OperationType.BATCHNORM: {'fused': 0.6, 'separate': 1.4},
       },
   }
   ```

**Deliverable:** Shows per-operator EDP with architectural context.

### Phase 3: Cross-Architecture Operator Comparison

**Effort:** Small (~200 lines)
**Impact:** Very High (reveals architectural strengths/weaknesses per operator)

**Tasks:**
1. Create comparison table generator:
   ```python
   def compare_operator_across_architectures(
       self,
       operator_name: str,  # e.g., "fc.relu"
       architectures: List[str]
   ) -> str:
       """Compare same operator across different architectures"""

       operator_edps = {}
       for arch in architectures:
           subgraph_edps = self.get_subgraph_edp_breakdown(arch)
           operator_edps_for_arch = self.get_operator_edp_breakdown(arch)

           # Find matching operator
           op = next((o for o in operator_edps_for_arch if o.operator_name == operator_name), None)
           if op:
               operator_edps[arch] = op

       # Generate comparison
       lines = []
       lines.append(f"Operator Comparison: {operator_name}")
       lines.append("=" * 100)
       lines.append(f"{'Architecture':<12} {'Energy':<12} {'Latency':<12} {'EDP':<15} {'Modifier':<10} {'Fused?':<8} {'Impact'}")
       lines.append("-" * 100)

       baseline_edp = operator_edps[self.summary.baseline].edp if self.summary.baseline in operator_edps else 1.0

       for arch in sorted(operator_edps.keys()):
           op = operator_edps[arch]
           ratio = op.edp / baseline_edp

           if ratio < 0.5:
               impact = "Excellent ⭐"
           elif ratio < 1.0:
               impact = "Good ✓"
           elif ratio < 2.0:
               impact = "Moderate"
           else:
               impact = "Poor ❌"

           lines.append(
               f"{arch:<12} "
               f"{op.energy_j*1e6:<12.2f} µJ "
               f"{op.latency_s*1e6:<12.2f} µs "
               f"{op.edp*1e9:<15.2f} nJ·s "
               f"{op.architectural_modifier:<10.2f}× "
               f"{'Yes' if op.is_fused else 'No':<8} "
               f"{impact}"
           )

       return "\n".join(lines)
   ```

**Deliverable:** Cross-architecture operator comparison tables.

### Phase 4: Fusion Impact Quantification

**Effort:** Medium (~300 lines)
**Impact:** High (quantifies fusion benefits)

**Tasks:**
1. Add fusion scenario simulator:
   ```python
   def simulate_fusion_scenarios(
       self,
       arch_name: str,
       subgraph_id: str
   ) -> FusionComparisonReport:
       """Simulate EDP with/without fusion for a subgraph"""

       # Get current (fused) EDP
       current_edp = get_subgraph_edp(arch_name, subgraph_id)

       # Simulate unfused scenario
       operators = extract_operator_info(subgraph_id)
       unfused_edps = []

       for op in operators:
           # Recalculate with 'separate' modifiers
           unfused_energy = estimate_unfused_energy(op, arch_name)
           unfused_latency = estimate_unfused_latency(op, arch_name)
           unfused_edp = unfused_energy * unfused_latency
           unfused_edps.append(unfused_edp)

       total_unfused_edp = sum(unfused_edps)

       fusion_benefit = total_unfused_edp - current_edp
       fusion_benefit_pct = fusion_benefit / total_unfused_edp * 100

       return FusionComparisonReport(
           subgraph_id=subgraph_id,
           fused_edp=current_edp,
           unfused_edp=total_unfused_edp,
           fusion_benefit_edp=fusion_benefit,
           fusion_benefit_pct=fusion_benefit_pct,
           operator_breakdowns=unfused_edps,
       )
   ```

**Deliverable:** Quantified fusion benefits per subgraph.

---

## Example Use Cases

### Use Case 1: Find EDP Hotspots

```bash
$ python cli/analyze_edp.py --model mlp --level subgraph

Subgraph EDP Breakdown (KPU):
================================================================================
Subgraph                        Energy      Latency     EDP         % Total
--------------------------------------------------------------------------------
fc1 (Linear_Bias_ReLU)          450.2 µJ    45.8 µs     20.62 nJ·s     77.6%  ⭐
fc2 (Linear_Bias_ReLU)          320.1 µJ    32.1 µs     10.28 nJ·s     38.7%
fc3 (Linear)                    180.3 µJ    18.5 µs      3.34 nJ·s     12.6%

→ fc1 dominates EDP (77.6%). Optimize this layer first!
```

### Use Case 2: Compare Operator Across Architectures

```bash
$ python cli/analyze_edp.py --model mlp --operator "fc1.relu" --compare-architectures

Operator Comparison: fc1.relu
================================================================================
Architecture  Energy      Latency     EDP         Modifier   Fused?  Impact
--------------------------------------------------------------------------------
KPU           0.2 µJ      0.1 µs      0.02 nJ·s   0.05×      Yes     Excellent ⭐
TPU           0.3 µJ      0.2 µs      0.06 nJ·s   0.02×      Yes     Excellent ⭐
CPU           2.1 µJ      1.8 µs      3.78 nJ·s   0.30×      Yes     Good ✓
GPU           4.5 µJ      3.2 µs     14.40 nJ·s   2.50×      No      Poor ❌

Key Insight:
  GPU pays 720× more EDP for ReLU than KPU due to kernel launch overhead.
  Fusion hides ReLU in KPU wavefront dataflow (almost free).
```

### Use Case 3: Quantify Fusion Benefits

```bash
$ python cli/analyze_edp.py --model resnet18 --subgraph "conv2_x.0" --fusion-impact

Fusion Impact Analysis: conv2_x.0 (Conv_BN_ReLU)
================================================================================

SCENARIO 1: Fused (Current)
  Conv:  880.0 µJ × 118.0 µs = 103.84 nJ·s
  BN:      8.0 µJ ×   1.8 µs =   0.01 nJ·s  (hidden)
  ReLU:    2.2 µJ ×   0.7 µs =   0.00 nJ·s  (hidden)
  ─────────────────────────────────────────────────
  Total: 890.2 µJ × 120.5 µs = 107.25 nJ·s  ✅

SCENARIO 2: Separate Kernels (Simulated)
  Conv:  880.0 µJ × 118.0 µs = 103.84 nJ·s  (same)
  BN:     48.0 µJ ×  12.5 µs =   0.60 nJ·s  (6× worse)
  ReLU:   32.0 µJ ×   8.8 µs =   0.28 nJ·s  (40× worse)
  ─────────────────────────────────────────────────
  Total: 960.0 µJ × 139.3 µs = 133.73 nJ·s  ❌

Fusion Savings:
  Energy: 69.8 µJ (7.3% saved)
  Latency: 18.8 µs (13.5% saved)
  EDP: 26.48 nJ·s (19.8% saved)  ⭐

Why Fusion Helps:
  • Eliminated 2 DRAM writes (BN→ReLU intermediate): 45 µJ
  • Eliminated 2 kernel launches: 12 µs
  • Reduced coherence overhead: 24.8 µJ
```

---

## Validation Strategy

### 1. Subgraph-Level Validation

**Test:** Sum of subgraph EDPs should equal model-level EDP.

```python
assert abs(sum(subgraph_edps) - model_edp) < 1e-6
```

### 2. Operator-Level Validation

**Test:** Sum of operator EDPs within subgraph should equal subgraph EDP.

```python
assert abs(sum(operator_edps) - subgraph_edp) < tolerance
```

Note: Tolerance needed due to allocation heuristics.

### 3. Fusion Impact Validation

**Test:** Fusion benefit should be positive (fusion should help, not hurt).

```python
assert fusion_benefit_edp >= 0  # Fusion should never increase EDP
```

### 4. Architectural Consistency

**Test:** Lightweight ops (bias, relu) should have lower modifiers on spatial architectures.

```python
assert modifier[KPU][RELU]['fused'] < modifier[GPU][RELU]['fused']
assert modifier[TPU][BIAS]['fused'] < modifier[CPU][BIAS]['fused']
```

---

## Success Metrics

1. **Correctness:**
   - ✅ Subgraph EDPs sum to model EDP (within 0.1%)
   - ✅ Operator EDPs sum to subgraph EDP (within 5% tolerance for heuristics)

2. **Insight Quality:**
   - ✅ Clearly identifies EDP hotspot subgraphs
   - ✅ Reveals architectural differences for same operator
   - ✅ Quantifies fusion benefits with concrete numbers

3. **Performance:**
   - ✅ Hierarchical breakdown adds <10% overhead to analysis time
   - ✅ Operator-level allocation runs in <100ms per subgraph

4. **Usability:**
   - ✅ CLI tools support drill-down (model → subgraph → operator)
   - ✅ Reports are human-readable and actionable
   - ✅ HTML visualizations show hierarchy interactively

---

## Future Enhancements

### 1. Per-Precision EDP Analysis
- Show how FP16/INT8 changes operator EDPs
- Quantify precision-accuracy trade-offs

### 2. Memory Hierarchy Breakdown
- Separate L1/L2/DRAM energy per operator
- Show cache hit/miss impact on EDP

### 3. Dynamic Batch Size Analysis
- Show how operator EDP scales with batch size
- Identify batch size sweet spot per architecture

### 4. Interactive Visualization
- HTML drill-down: click subgraph → see operator breakdown
- Sunburst chart: model → subgraph → operator hierarchy
- Comparison mode: side-by-side architecture views

---

## Conclusion

This hierarchical EDP breakdown design provides:

1. **Immediate value** (Phase 1): Subgraph-level EDP reveals hotspots
2. **Deep insight** (Phase 2): Operator-level EDP explains architectural differences
3. **Quantified benefits** (Phase 3-4): Fusion impact and cross-architecture comparison

The design leverages **existing infrastructure** (EnergyDescriptor, LatencyDescriptor, FusedSubgraph) and adds **targeted enhancements** to expose hierarchical detail.

**Key innovation:** Architectural modifiers capture the reality that lightweight ops (bias, relu) are almost free on spatial architectures (TPU/KPU) when fused, but expensive on sequential architectures (CPU/GPU) when separate.

This enables users to:
- Identify which operators dominate EDP
- Understand WHY architectures differ (instruction fetch, coherence, dataflow)
- Quantify fusion benefits concretely
- Make informed architecture selection decisions

**Ready for review and feedback!**
