# Functional Energy Composition Design

**Date:** 2025-11-03
**Status:** 🔍 Analysis Complete → Design Needed
**Context:** Progression towards functional divide-and-conquer with proper idle energy accounting

---

## Executive Summary

**Problem:** Current energy modeling calculates static energy as `idle_power_watts × latency` for the **entire chip**, regardless of how many compute units are actually allocated and powered on. This leads to incorrect energy estimates that don't account for:

1. Per-unit power states (allocated vs unallocated)
2. Power gating (turning off unused units to save idle power)
3. SoC power management features
4. Proper functional composition of energy from unit → subgraph → model

**Impact:**
- Energy estimates can be 2-10× off for low-utilization workloads
- Cannot model the benefit of power gating unused units
- Cannot compare architectures with different power management capabilities

**Solution Direction:**
- Integrate hardware mapper allocation decisions into energy calculation
- Model per-unit power states (active, idle-allocated, idle-unallocated, powered-off)
- Support SoC power management modeling (power gating, DVFS regions)
- Functional energy composition: `unit_energy → subgraph_energy → model_energy`

---

## Current Pipeline Flow

### Phase 1: Graph Analysis → Partitioning

```
User Model (nn.Module)
    ↓
FX Tracing (symbolic_trace)
    ↓
Shape Propagation (ShapeProp)
    ↓
Graph Partitioning (GraphPartitioner)
    ↓
PartitionReport
    - List[SubgraphDescriptor]
    - Concurrency info (execution stages)
    - Subgraph characteristics (FLOPs, bytes, fusion patterns)
```

**Code:** `unified_analyzer.py:459-483 (_trace_and_partition)`

**Key Data Structure:** `SubgraphDescriptor` contains:
- `flops: int` - Total FLOPs for this subgraph
- `total_input_bytes`, `total_output_bytes`, `total_weight_bytes` - Memory footprint
- `parallelism: ParallelismDescriptor` - Thread/warp count estimates
- `fusion_pattern: str` - e.g., "conv_bn_relu"

### Phase 2: Hardware Mapping (MISSING IN CURRENT UNIFIED PIPELINE!)

**Current State:** The `UnifiedAnalyzer` does NOT use hardware mappers!

The hardware mapping capability exists in `hardware/mappers/` but is not integrated into the analysis pipeline:

```python
# What SHOULD happen (but doesn't):
HardwareMapper.map_subgraph(subgraph, precision)
    ↓
HardwareAllocation (per subgraph)
    - compute_units_allocated: int  # 2/132 SMs, 1/2 TPU tiles, etc.
    - occupancy: float
    - utilization: float
    - estimated_latency: float
    - compute_energy, memory_energy, total_energy: float
    ↓
GraphHardwareAllocation (full model)
    - average_compute_units_used: float
    - peak_compute_units_used: int
    - total_latency, total_energy: float
```

**Code:** `hardware/resource_model.py:458-565` (data structures exist but unused)

**Evidence of Gap:**
```python
# unified_analyzer.py:487-495 (_run_roofline_analysis)
def _run_roofline_analysis(...):
    analyzer = RooflineAnalyzer(hardware, precision=precision)
    return analyzer.analyze(partition_report.subgraphs, partition_report)
    # ↑ Creates NEW RooflineAnalyzer, doesn't use HardwareMapper!
```

### Phase 3: Roofline Analysis (Independent, Doesn't Use Mapper)

```
SubgraphDescriptor
    ↓
RooflineAnalyzer.analyze()
    - For each subgraph:
        compute_time = flops / peak_flops
        memory_time = bytes / peak_bandwidth
        actual_latency = max(compute_time, memory_time) + overhead
    ↓
LatencyDescriptor (per subgraph)
    - actual_latency: float
    - flops_utilization: float (0-1)
    - bandwidth_utilization: float (0-1)
    - bottleneck: BottleneckType
    ↓
RooflineReport
    - List[LatencyDescriptor]
    - total_latency: float
```

**Code:** `analysis/roofline.py:200-350 (RooflineAnalyzer.analyze)`

**Key Gap:** `LatencyDescriptor` does NOT contain `compute_units_allocated` or any allocation info!

### Phase 4: Energy Analysis (Uses Latencies, Ignores Allocation)

```
SubgraphDescriptor + LatencyDescriptor
    ↓
EnergyAnalyzer._analyze_subgraph()
    ↓
For each subgraph:
    compute_energy = flops × energy_per_flop
    memory_energy = bytes × energy_per_byte
    static_energy = idle_power_watts × latency  ← PROBLEM: entire chip!

    # Estimate utilization from thread count
    max_threads = compute_units × threads_per_unit
    utilization = min(1.0, sg.parallelism.total_threads / max_threads)

    # Wasted energy
    wasted_energy = static_energy × (1.0 - utilization)
    ↓
EnergyDescriptor (per subgraph)
    - compute_energy_j, memory_energy_j, static_energy_j
    - utilization: float (estimated from threads, not from mapper)
    - wasted_energy_j: float
    ↓
EnergyReport
    - total_energy_j
    - Per-component breakdown
```

**Code:** `analysis/energy.py:362-415 (_analyze_subgraph)`

**Critical Lines:**
```python
# Line 373: Static energy for ENTIRE chip
static_energy = self.idle_power_watts * latency

# Lines 379-384: Utilization estimated from thread count (not mapper allocation)
utilization = 1.0
if sg.parallelism and sg.parallelism.total_threads > 0:
    max_threads = self.resource_model.compute_units * self.resource_model.threads_per_unit
    utilization = min(1.0, sg.parallelism.total_threads / max_threads)

# Line 388: Wasted energy calculation
wasted_energy = static_energy * (1.0 - utilization)
```

### Phase 5: Memory Analysis (Independent)

```
SubgraphDescriptor
    ↓
MemoryEstimator.estimate_memory()
    - Peak memory usage (activations + weights)
    - Memory timeline (live tensors over time)
    ↓
MemoryReport
```

**Code:** `analysis/memory_estimator.py`

---

## Identified Gaps

### Gap 1: Hardware Mapper Not Integrated into Pipeline

**Current:** `UnifiedAnalyzer` creates separate `RooflineAnalyzer` and `EnergyAnalyzer` that estimate latency and energy independently.

**Missing:** Hardware mapper allocation decisions (`compute_units_allocated`) never feed into energy calculation.

**Location:** `unified_analyzer.py:487-516`

**Evidence:**
```python
# Roofline doesn't use mapper
def _run_roofline_analysis(...):
    analyzer = RooflineAnalyzer(hardware, precision=precision)
    return analyzer.analyze(partition_report.subgraphs, partition_report)

# Energy uses roofline latencies, but no allocation info
def _run_energy_analysis(...):
    analyzer = EnergyAnalyzer(hardware, precision=precision)
    latencies = [lat.actual_latency for lat in roofline_report.latencies]
    return analyzer.analyze(partition_report.subgraphs, partition_report, latencies=latencies)
```

### Gap 2: Static Energy Calculated for Entire Chip

**Current:** `idle_power_watts × latency` charges for ALL compute units on the chip.

**Problem Examples:**
- TPU with 2 tiles at 4.7% utilization: Charges for 2 tiles' idle power, but only 0.094 tiles used
- H100 with 132 SMs at 18% utilization: Charges for 132 SMs' idle power, but only 24 SMs used

**Location:** `analysis/energy.py:373`

**Code:**
```python
# This is WRONG - charges for entire chip
static_energy = self.idle_power_watts * latency
```

**What it SHOULD be:**
```python
# Charge only for allocated units (with power gating support)
allocated_units = allocation.compute_units_allocated
if power_gating_enabled:
    # Only allocated units consume idle power
    static_energy = (idle_power_per_unit * allocated_units) * latency
else:
    # All units consume idle power (no power gating)
    static_energy = idle_power_watts * latency
```

### Gap 3: No Per-Unit Power State Modeling

**Current:** Only two states modeled:
- Active (doing work): compute energy + memory energy
- Idle (entire chip): `idle_power_watts × latency`

**Missing States:**
- **Idle-Allocated:** Unit allocated to workload but waiting (occupancy < 100%)
- **Idle-Unallocated:** Unit not allocated, could be power-gated
- **Powered-Off:** Unit power-gated (0 idle power)
- **Transitioning:** Power gating transition overhead

**Example (TPU with 2 tiles, 4.7% util):**

| State | Current Model | Should Be |
|-------|---------------|-----------|
| Allocated | 1 tile (from mapper) | 1 tile |
| Unallocated | Not tracked | 1 tile |
| Idle power | 2 tiles × idle_power_per_tile | 1 tile × idle_power_per_tile (if power-gated) |

### Gap 4: No SoC Power Management Modeling

**Current:** Single `idle_power_watts` for entire chip, no configuration.

**Missing Features:**
- **Power Gating:** Can turn off unused units (e.g., H100 can gate SMs)
- **DVFS Regions:** Different clock/voltage domains (already modeled in `ClockDomain`, not used for energy)
- **Power States:** C-states, P-states, retention modes
- **Transition Overhead:** Time and energy to power gate units

**Example (Jetson Orin):**
- 15W mode: Can power gate 50% of SMs to stay under thermal budget
- 30W mode: All SMs available but may throttle

### Gap 5: Utilization Estimated from Threads, Not Allocation

**Current:** `energy.py:379-384` estimates utilization from thread count:
```python
max_threads = self.resource_model.compute_units * self.resource_model.threads_per_unit
utilization = min(1.0, sg.parallelism.total_threads / max_threads)
```

**Problem:**
- Thread count is a **coarse estimate** from graph partitioning
- Hardware mapper calculates **actual allocation** considering:
  - Wave quantization (must allocate whole SMs/tiles)
  - Occupancy limits (warps per SM, etc.)
  - Resource constraints (registers, shared memory)

**Example (GPU):**
- Thread count: 2048 threads
- Max threads: 132 SMs × 2048 threads/SM = 270,336 threads
- **Estimated utilization:** 2048 / 270,336 = 0.76% (wrong!)
- **Actual allocation (from mapper):** 24 SMs allocated (wave quantization)
- **Actual utilization:** 24 / 132 = 18.2% (correct)

### Gap 6: Energy Composition Not Functional

**Current:** Energy calculated bottom-up:
```
subgraph_energy = compute + memory + static(entire_chip)
model_energy = sum(subgraph_energy)
```

**Problem:** Static energy double-counts if subgraphs run sequentially but we charge entire chip per subgraph.

**Desired (Functional Composition):**
```
# Per-unit energy
unit_energy = active_energy_when_used + idle_energy_when_allocated

# Per-subgraph energy (compose from units)
subgraph_energy = sum(unit_energy for unit in allocated_units)

# Per-model energy (compose from subgraphs, account for concurrency)
model_energy = sum_by_stage(subgraph_energy, accounting_for_parallelism)
```

---

## Root Cause Analysis

### Why Does This Gap Exist?

1. **Historical Evolution:** Original pipeline was roofline-based (no hardware mapping)
2. **Phase 2 Added Later:** Hardware mappers were added but never integrated into `UnifiedAnalyzer`
3. **API Mismatch:** Mappers return `HardwareAllocation`, but analyzers expect `SubgraphDescriptor`
4. **Separation of Concerns:** Roofline/Energy analyzers designed to be standalone, not mapper-aware

### Why Wasn't It Caught?

1. **Small Models Dominate:** Most validation uses small models (ResNet-18, MobileNet) where utilization is low
2. **Static Energy Dominates:** When static energy is 75%, the allocation error is hidden
3. **Relative Comparisons:** Comparing architectures shows trends even if absolute values are wrong
4. **No Ground Truth:** No reference measurements to validate against

---

## Design Requirements

### Functional Requirements

**FR1: Per-Unit Power State Tracking**
- Track each compute unit's state: active, idle-allocated, idle-unallocated, powered-off
- Calculate energy based on actual state

**FR2: Hardware Mapper Integration**
- Use mapper allocation decisions (`compute_units_allocated`) in energy calculation
- Support all mapper types (CPU, GPU, DSP, TPU, KPU, DPU, CGRA)

**FR3: Power Management Modeling**
- Support power gating (turn off unused units)
- Model transition overhead (time and energy to power gate)
- Support DVFS regions (already have `ClockDomain`, extend to energy)

**FR4: Functional Energy Composition**
- Energy composes cleanly: unit → subgraph → model
- No double-counting of idle energy
- Proper accounting for concurrent vs sequential execution

**FR5: Backward Compatibility**
- Existing API unchanged (`UnifiedAnalyzer.analyze_model()`)
- Default behavior: power gating OFF (matches current results)
- Opt-in for power management features

### Non-Functional Requirements

**NFR1: Performance**
- No significant slowdown (< 10% overhead)
- Hardware mapper already exists, just needs integration

**NFR2: Maintainability**
- Clean separation: mapper → latency → energy
- Reusable across all hardware types

**NFR3: Testability**
- Validate against known power gating benefits
- Unit tests for per-unit energy calculation

---

## Proposed Design

### Option 1: Mapper-Integrated Pipeline (Recommended)

**Approach:** Integrate hardware mapper into `UnifiedAnalyzer`, use allocation results in energy calculation.

**Pipeline Flow:**
```
Graph Partitioning
    ↓
    PartitionReport (SubgraphDescriptor list)
    ↓
Hardware Mapping (NEW INTEGRATION)
    ↓
    HardwareAllocation per subgraph
        - compute_units_allocated
        - estimated_latency
        - compute_energy, memory_energy (from mapper)
    ↓
Energy Composition (NEW)
    - Per-unit energy based on state
    - Aggregate to subgraph, then model
    - Account for power gating
    ↓
EnergyReport (enhanced)
    - total_energy_j
    - allocated_units_energy_j
    - unallocated_units_energy_j (if no power gating)
    - power_gating_savings_j (if enabled)
```

**Code Changes:**

1. **Add mapper integration to `UnifiedAnalyzer`:**

```python
# unified_analyzer.py

def _run_hardware_mapping(
    self,
    partition_report: PartitionReport,
    hardware: HardwareResourceModel,
    precision: Precision
) -> GraphHardwareAllocation:
    """Map subgraphs to hardware (NEW)"""
    mapper = hardware.create_mapper()
    allocations = []

    for sg in partition_report.subgraphs:
        allocation = mapper.map_subgraph(sg, precision)
        allocations.append(allocation)

    # Aggregate
    return GraphHardwareAllocation(
        model_name=self.model_name,
        hardware_name=hardware.name,
        batch_size=self.batch_size,
        model_precision=precision,
        subgraph_allocations=allocations,
        # ... compute aggregate stats
    )

def analyze_model_with_custom_hardware(self, ...):
    # ... existing code ...

    # NEW: Run hardware mapping
    if config.run_hardware_mapping:  # Default: True
        result.hardware_allocation = self._run_hardware_mapping(
            partition_report, hardware, precision
        )

    # Roofline uses mapper results
    result.roofline_report = self._run_roofline_analysis(
        partition_report, hardware, precision, result.hardware_allocation
    )

    # Energy uses mapper allocation
    result.energy_report = self._run_energy_analysis(
        partition_report, hardware, result.roofline_report,
        result.hardware_allocation, precision
    )
```

2. **Enhance `EnergyAnalyzer` to use allocation:**

```python
# energy.py

class EnergyAnalyzer:
    def __init__(
        self,
        resource_model: HardwareResourceModel,
        precision: Precision = Precision.FP32,
        power_gating_enabled: bool = False  # NEW
    ):
        self.resource_model = resource_model
        self.precision = precision
        self.power_gating_enabled = power_gating_enabled

        # Energy rates
        self.energy_per_flop = resource_model.precision_profiles[precision].energy_per_op
        self.energy_per_byte = resource_model.precision_profiles[precision].energy_per_byte

        # Power modeling (NEW)
        self.idle_power_watts = resource_model.idle_power_watts
        self.total_compute_units = resource_model.compute_units

        # Per-unit idle power (NEW)
        self.idle_power_per_unit = self.idle_power_watts / self.total_compute_units

    def analyze(
        self,
        subgraphs: List[SubgraphDescriptor],
        partition_report: PartitionReport,
        latencies: Optional[List[float]] = None,
        hardware_allocation: Optional[GraphHardwareAllocation] = None  # NEW
    ) -> EnergyReport:
        """Analyze energy with optional hardware allocation"""

        # Build allocation map (NEW)
        allocation_map = {}
        if hardware_allocation:
            for alloc in hardware_allocation.subgraph_allocations:
                allocation_map[alloc.subgraph_id] = alloc

        # Analyze each subgraph
        energy_descriptors = []
        for i, sg in enumerate(subgraphs):
            latency = latencies[i] if latencies else self._estimate_latency(sg)
            allocation = allocation_map.get(sg.node_id)  # NEW

            descriptor = self._analyze_subgraph(sg, latency, allocation)
            energy_descriptors.append(descriptor)

        # ... rest of aggregation ...

    def _analyze_subgraph(
        self,
        sg: SubgraphDescriptor,
        latency: float,
        allocation: Optional[HardwareAllocation] = None  # NEW
    ) -> EnergyDescriptor:
        """Analyze energy for a single subgraph with allocation info"""

        # Compute and memory energy (unchanged)
        compute_energy = sg.flops * self.energy_per_flop
        total_bytes = sg.total_input_bytes + sg.total_output_bytes + sg.total_weight_bytes
        memory_energy = total_bytes * self.energy_per_byte

        # Static energy (NEW: per-unit power state modeling)
        if allocation and allocation.compute_units_allocated > 0:
            # Use actual allocation from mapper
            allocated_units = allocation.compute_units_allocated
            unallocated_units = self.total_compute_units - allocated_units

            if self.power_gating_enabled:
                # Only allocated units consume idle power
                static_energy_allocated = self.idle_power_per_unit * allocated_units * latency
                static_energy_unallocated = 0.0  # Power gated
                power_gating_savings = self.idle_power_per_unit * unallocated_units * latency
            else:
                # All units consume idle power (no power gating)
                static_energy_allocated = self.idle_power_per_unit * allocated_units * latency
                static_energy_unallocated = self.idle_power_per_unit * unallocated_units * latency
                power_gating_savings = 0.0

            static_energy = static_energy_allocated + static_energy_unallocated
            utilization = allocation.utilization  # From mapper

        else:
            # Fallback to old method (no allocation info)
            static_energy = self.idle_power_watts * latency
            power_gating_savings = 0.0

            # Estimate utilization from thread count
            utilization = 1.0
            if sg.parallelism and sg.parallelism.total_threads > 0:
                max_threads = self.total_compute_units * self.resource_model.threads_per_unit
                utilization = min(1.0, sg.parallelism.total_threads / max_threads)

        # ... rest of method unchanged ...

        return EnergyDescriptor(
            # ... existing fields ...
            power_gating_savings_j=power_gating_savings,  # NEW
        )
```

3. **Enhance `EnergyDescriptor` and `EnergyReport`:**

```python
# energy.py

@dataclass
class EnergyDescriptor:
    # ... existing fields ...

    # NEW: Power management
    allocated_units: int = 0  # Compute units allocated
    static_energy_allocated_j: float = 0.0  # Idle energy for allocated units
    static_energy_unallocated_j: float = 0.0  # Idle energy for unallocated units
    power_gating_savings_j: float = 0.0  # Energy saved by power gating

@dataclass
class EnergyReport:
    # ... existing fields ...

    # NEW: Power management summary
    total_allocated_units_energy_j: float = 0.0
    total_unallocated_units_energy_j: float = 0.0
    total_power_gating_savings_j: float = 0.0
    power_gating_enabled: bool = False
```

**Pros:**
✅ Minimal API changes (backward compatible)
✅ Reuses existing hardware mapper infrastructure
✅ Clean separation of concerns (mapper → energy)
✅ Easy to test (compare power gating on/off)
✅ Solves all 6 identified gaps

**Cons:**
⚠️ Requires hardware mapper to be run (small overhead ~5%)
⚠️ More complex pipeline flow

### Option 2: Energy-Aware Hardware Mapper (Alternative)

**Approach:** Move energy calculation into hardware mapper, return complete `HardwareAllocation` with energy already computed.

**Pipeline Flow:**
```
Graph Partitioning
    ↓
Hardware Mapping (energy-aware)
    ↓
    HardwareAllocation (enhanced)
        - compute_units_allocated
        - per_unit_energy breakdown
        - latency
    ↓
Energy Aggregation (simplified)
    - Just sum HardwareAllocation energies
    ↓
EnergyReport
```

**Pros:**
✅ Single pass (mapper computes latency AND energy)
✅ Tightest coupling (energy uses exact allocation)

**Cons:**
❌ Breaks existing API (mappers now need energy rates)
❌ Mixing concerns (mapper does latency AND energy)
❌ Harder to test (can't test energy without mapper)
❌ Harder to extend (every mapper needs energy logic)

---

## Recommendation

**Use Option 1: Mapper-Integrated Pipeline**

**Rationale:**
1. Minimal changes to existing API (backward compatible)
2. Clean separation: mapper (allocation) → energy analyzer (energy calculation)
3. Reuses all existing infrastructure
4. Easy to add power gating as opt-in feature
5. Testable (can compare with/without mapper, with/without power gating)

**Implementation Plan:**

**Phase 1: Basic Integration (1-2 days)**
- Add `_run_hardware_mapping()` to `UnifiedAnalyzer`
- Pass `hardware_allocation` to `_run_energy_analysis()`
- Use `allocation.compute_units_allocated` in static energy calculation
- Validation: Compare with current results (should match when power_gating=False)

**Phase 2: Per-Unit Power States (1 day)**
- Add `idle_power_per_unit` calculation
- Implement `power_gating_enabled` flag
- Calculate `static_energy_allocated` and `static_energy_unallocated` separately
- Validation: Test on TPU (expect 50% energy reduction with power gating)

**Phase 3: Extended Power Management (2-3 days)**
- Add transition overhead modeling (time and energy to power gate)
- Support DVFS regions (use `ClockDomain` for power calculation)
- Add power state APIs for SoC modeling
- Validation: Compare with vendor power specs

**Phase 4: Validation & Testing (1-2 days)**
- Unit tests for per-unit energy calculation
- Integration tests with all hardware types
- Validation against known power gating benefits
- Update documentation and examples

**Total Effort:** ~1 week for complete implementation and validation

---

## Example: Before vs After

### Before (Current Implementation)

**Scenario:** ResNet-18 on H100 (132 SMs), batch=1

```
Graph Partitioning:
  - 62 subgraphs identified
  - Parallelism analysis: 2048 threads typical

Energy Calculation (per subgraph):
  static_energy = 300W × latency  # ENTIRE chip!

  # Thread-based utilization estimate
  max_threads = 132 SMs × 2048 threads/SM = 270,336
  utilization = 2048 / 270,336 = 0.76%

  wasted_energy = static_energy × (1 - 0.0076) = 99.24% wasted!

Total Energy:
  - Static energy: 75% of total
  - Wasted energy: 74.4% of total (99.24% of static)
  - Charges for 132 SMs idle power even though only ~24 allocated
```

**Problem:** Grossly overestimates idle energy because charges for all 132 SMs!

### After (Mapper-Integrated Pipeline)

**Scenario:** Same ResNet-18 on H100

```
Graph Partitioning:
  - 62 subgraphs identified

Hardware Mapping (NEW):
  - Subgraph 1: 24 SMs allocated (wave quantization)
  - Subgraph 2: 16 SMs allocated
  - ... etc
  - Average: 24 SMs allocated

Energy Calculation (per subgraph, power_gating=OFF):
  idle_power_per_SM = 300W / 132 SMs = 2.27W/SM

  allocated_units = 24 SMs (from mapper)
  unallocated_units = 108 SMs

  # No power gating: all units consume idle power
  static_energy_allocated = 2.27W/SM × 24 SMs × latency
  static_energy_unallocated = 2.27W/SM × 108 SMs × latency
  static_energy = static_energy_allocated + static_energy_unallocated

  utilization = 24 / 132 = 18.2% (from mapper allocation)

Energy Calculation (per subgraph, power_gating=ON):
  # Power gating: only allocated units consume idle power
  static_energy_allocated = 2.27W/SM × 24 SMs × latency
  static_energy_unallocated = 0.0  # Power gated!
  static_energy = static_energy_allocated

  power_gating_savings = 2.27W/SM × 108 SMs × latency

Total Energy (power_gating=ON):
  - Static energy: 18% of previous (only 24/132 SMs)
  - Power gating savings: 82% of static energy
  - More accurate utilization (18.2% vs 0.76%)
```

**Improvement:**
- ✅ Correct idle energy accounting (only charged for allocated SMs)
- ✅ Accurate utilization (18.2% from mapper vs 0.76% from thread estimate)
- ✅ Can model power gating benefits (82% idle energy savings!)
- ✅ Functional composition (per-SM → per-subgraph → per-model)

---

## Open Questions

**Q1:** Should power gating be enabled by default?

**A1:** No. Default to `power_gating_enabled=False` for backward compatibility and conservative estimates. Users opt-in when they know hardware supports it.

**Q2:** How to handle architectures without hardware mapper implementations?

**A2:** Fallback to current thread-based estimation (graceful degradation).

**Q3:** What about power gating transition overhead?

**A3:** Model in Phase 3. Add `power_gating_transition_time` and `power_gating_transition_energy` to `HardwareResourceModel`. Account for transitions when units are powered on/off.

**Q4:** How to validate the energy estimates?

**A4:**
1. Compare relative trends (higher batch → higher util → lower idle energy per inference)
2. Validate power gating savings against vendor specs (e.g., H100: 80% idle power reduction with SM gating)
3. Cross-check with `nvidia-smi` / `rocm-smi` power measurements where possible

**Q5:** Should we integrate DVFS (dynamic voltage/frequency scaling) modeling?

**A5:** Yes, in Phase 3. We already have `ClockDomain` with `sustained_clock_hz` vs `max_boost_clock_hz`. Extend to energy:
- `energy_per_flop` should scale with voltage² (DVFS power law)
- `idle_power_watts` should scale with voltage and frequency

---

## Next Steps

1. **Review this design** with team/maintainers
2. **Prototype Phase 1** (basic mapper integration)
3. **Validate** on ResNet-18 across all architectures
4. **Iterate** based on findings
5. **Document** new APIs and power management options
6. **Update examples** to show power gating benefits

---

## References

- **Hardware Mapper Implementation:** `src/graphs/hardware/mappers/`
- **Current Energy Analysis:** `src/graphs/analysis/energy.py`
- **UnifiedAnalyzer Pipeline:** `src/graphs/analysis/unified_analyzer.py`
- **Data Structures:** `src/graphs/hardware/resource_model.py:458-565`
- **Session Context:** `docs/sessions/2025-11-03_test_organization.md`
