# Subgraph Representation Unification Analysis

**Date:** 2025-11-09
**Issue:** FusionReport incompatibility with downstream analyzers expecting PartitionReport
**Goal:** Unify SubgraphDescriptor and FusedSubgraph into a single representation

---

## Executive Summary

The codebase currently has **two parallel subgraph representations** that emerged from different partitioning strategies:

1. **SubgraphDescriptor** (from GraphPartitioner) - Used with symbolic_trace
2. **FusedSubgraph** (from FusionBasedPartitioner) - Used with Dynamo export

With the migration to Dynamo-first architecture, we need to **unify these representations** to enable FusionBasedPartitioner output to work with all downstream analyzers.

**Key Finding:** SubgraphDescriptor already has compatibility properties designed to match FusedSubgraph! This suggests prior partial unification attempts.

---

## Structure Comparison

### FusedSubgraph (fusion_partitioner.py:37-84)

```python
@dataclass
class FusedSubgraph:
    # Identity
    subgraph_id: int                          # Numeric ID
    node_ids: List[str]                       # List of FX node IDs
    node_names: List[str]                     # Human-readable names
    operation_types: List[OperationType]      # Multiple operations

    # Computation (aggregate)
    total_flops: int
    total_macs: int

    # Memory (external vs internal distinction)
    total_input_bytes: int                    # External inputs only
    total_output_bytes: int                   # External outputs only
    internal_bytes: int                       # Intermediate (saved by fusion!)
    total_weight_bytes: int

    # Parallelism (merged)
    parallelism: Optional[ParallelismDescriptor]

    # Fusion metadata
    fusion_pattern: str                       # e.g., "Conv_BN_ReLU"
    num_operators: int                        # How many ops fused

    # Dependencies
    depends_on: List[int]                     # Other subgraph IDs (numeric)

    # Characterization
    arithmetic_intensity: float
    recommended_bottleneck: BottleneckType

    # Method
    def data_movement_reduction(self) -> float
```

### SubgraphDescriptor (ir/structures.py:115-221)

```python
@dataclass
class SubgraphDescriptor:
    # Identity
    node_id: str                              # Single node ID
    node_name: str                            # Single name
    operation_type: OperationType             # Single operation
    fusion_pattern: str

    # Computation
    flops: int                                # NOT total_flops
    macs: int                                 # NOT total_macs

    # Memory (detailed breakdown)
    input_tensors: List[TensorDescriptor]
    output_tensors: List[TensorDescriptor]
    weight_tensors: List[TensorDescriptor]
    total_input_bytes: int
    total_output_bytes: int
    total_weight_bytes: int
    arithmetic_intensity: float

    # Parallelism
    parallelism: Optional[ParallelismDescriptor]

    # Dependencies
    depends_on: List[str]                     # Other node IDs (strings)
    dependency_type: str

    # Hardware hints
    recommended_bottleneck: BottleneckType

    # Partition metadata
    partition_reason: PartitionReason
    partition_criteria: Dict[str, any]
    fusion_candidates: List[str]

    # COMPATIBILITY PROPERTIES (for FusedSubgraph interface!)
    @property
    def total_flops(self) -> int:
        return self.flops

    @property
    def total_macs(self) -> int:
        return self.macs

    @property
    def node_names(self) -> List[str]:
        return [self.node_name]
```

---

## Container Report Comparison

### FusionReport (fusion_partitioner.py:86-134)

```python
@dataclass
class FusionReport:
    fused_subgraphs: List[FusedSubgraph]      # ⚠️ Different attribute name!
    total_subgraphs: int
    original_operators: int

    # Aggregated stats
    total_flops: int
    total_memory_traffic_fused: int
    total_memory_traffic_unfused: int
    data_movement_reduction: float            # NEW: fusion benefit metric

    # Fusion stats
    fusion_patterns: Dict[str, int]
    avg_fusion_size: float                    # NEW: fusion quality metric
    max_fusion_size: int                      # NEW: fusion quality metric

    def summary_stats(self) -> str
```

### PartitionReport (ir/structures.py:284-339)

```python
@dataclass
class PartitionReport:
    subgraphs: List[SubgraphDescriptor]       # ⚠️ Different attribute name!
    total_subgraphs: int

    # Computation totals
    total_flops: int
    total_macs: int
    total_memory_traffic: int

    # Intensity distribution
    average_arithmetic_intensity: float
    min_arithmetic_intensity: float
    max_arithmetic_intensity: float

    # Distribution counts
    operation_type_counts: Dict[str, int]
    fusion_pattern_counts: Dict[str, int]
    parallelism_distribution: Dict[str, int]
    bottleneck_distribution: Dict[str, int]
    partition_reason_distribution: Dict[str, int]

    # Concurrency
    concurrency: Optional[ConcurrencyDescriptor]

    # Critical path
    critical_path_subgraphs: List[str]

    def summary_stats(self) -> str
```

---

## Attribute Mapping Table

| Concept | SubgraphDescriptor | FusedSubgraph | Compatible? |
|---------|-------------------|---------------|-------------|
| **Identity** |
| Subgraph ID | `node_id: str` | `subgraph_id: int` | ❌ Different types |
| Node names | `node_name: str` + property `node_names: List[str]` | `node_names: List[str]` | ✅ Via property |
| Operations | `operation_type: OperationType` | `operation_types: List[OperationType]` | ⚠️ Single vs list |
| **Computation** |
| FLOPs | `flops: int` + property `total_flops` | `total_flops: int` | ✅ Via property |
| MACs | `macs: int` + property `total_macs` | `total_macs: int` | ✅ Via property |
| **Memory** |
| Input bytes | `total_input_bytes: int` | `total_input_bytes: int` | ✅ Same |
| Output bytes | `total_output_bytes: int` | `total_output_bytes: int` | ✅ Same |
| Weight bytes | `total_weight_bytes: int` | `total_weight_bytes: int` | ✅ Same |
| Internal bytes | ❌ Missing | `internal_bytes: int` | ❌ Not in SubgraphDescriptor |
| Tensor details | `input/output/weight_tensors: List[TensorDescriptor]` | ❌ Missing | ❌ Not in FusedSubgraph |
| **Fusion** |
| Pattern | `fusion_pattern: str` | `fusion_pattern: str` | ✅ Same |
| Operator count | ❌ Missing (always 1) | `num_operators: int` | ❌ Not in SubgraphDescriptor |
| **Dependencies** |
| Depends on | `depends_on: List[str]` | `depends_on: List[int]` | ❌ Different types |
| **Characteristics** |
| AI | `arithmetic_intensity: float` | `arithmetic_intensity: float` | ✅ Same |
| Bottleneck | `recommended_bottleneck: BottleneckType` | `recommended_bottleneck: BottleneckType` | ✅ Same |
| Parallelism | `parallelism: Optional[ParallelismDescriptor]` | `parallelism: Optional[ParallelismDescriptor]` | ✅ Same |
| **Partition Metadata** |
| Reason | `partition_reason: PartitionReason` | ❌ Missing | ❌ Not in FusedSubgraph |
| Candidates | `fusion_candidates: List[str]` | ❌ Missing | ❌ Not in FusedSubgraph |

---

## Report Container Mapping

| Concept | PartitionReport | FusionReport | Compatible? |
|---------|----------------|--------------|-------------|
| **Subgraph list** | `subgraphs` | `fused_subgraphs` | ❌ **CRITICAL** - Different names! |
| **Counts** |
| Total subgraphs | `total_subgraphs: int` | `total_subgraphs: int` | ✅ Same |
| Total FLOPs | `total_flops: int` | `total_flops: int` | ✅ Same |
| Total MACs | `total_macs: int` | ❌ Missing | ❌ Not in FusionReport |
| **Memory** |
| Memory traffic | `total_memory_traffic: int` | `total_memory_traffic_fused: int` | ⚠️ Different semantics |
| Unfused traffic | ❌ Missing | `total_memory_traffic_unfused: int` | ❌ Fusion-specific |
| **Fusion metrics** |
| Original ops | ❌ Missing | `original_operators: int` | ❌ Fusion-specific |
| Avg fusion size | ❌ Missing | `avg_fusion_size: float` | ❌ Fusion-specific |
| Max fusion size | ❌ Missing | `max_fusion_size: int` | ❌ Fusion-specific |
| Data movement saved | ❌ Missing | `data_movement_reduction: float` | ❌ Fusion-specific |
| **Statistics** |
| AI stats | `average/min/max_arithmetic_intensity` | ❌ Missing | ❌ Not in FusionReport |
| Distributions | `operation_type_counts`, `bottleneck_distribution`, etc. | `fusion_patterns: Dict[str, int]` | ⚠️ Different structure |
| **Advanced** |
| Concurrency | `concurrency: Optional[ConcurrencyDescriptor]` | ❌ Missing | ❌ Not in FusionReport |
| Critical path | `critical_path_subgraphs: List[str]` | ❌ Missing | ❌ Not in FusionReport |

---

## Downstream Consumer Analysis

### Files expecting `.subgraphs` attribute (12 files):

1. **src/graphs/analysis/unified_analyzer.py** - Core orchestrator
2. **src/graphs/hardware/table_formatter.py** - Hardware comparison tables
3. **src/graphs/analysis/architecture_comparator.py** - Cross-architecture comparison
4. **src/graphs/reporting/report_generator.py** - Report generation
5. **src/graphs/visualization/mermaid_generator.py** - Graph visualization
6. **src/graphs/hardware/mappers/research/dfm.py** - Research mapper
7. **src/graphs/analysis/memory.py** - Memory analysis
8. **src/graphs/transform/partitioning/graph_partitioner.py** - GraphPartitioner itself
9. **src/graphs/analysis/concurrency.py** - Concurrency analysis
10. **src/graphs/experiment/complexity.py** - Complexity experiments
11. **src/graphs/experiment/estimateEDP.py** - EDP experiments
12. **src/graphs/experiment/sweep.py** - Sweep experiments

### Files using `.fused_subgraphs` attribute (15 files):

All hardware mappers + partitioner internals:
- **Mappers**: cpu.py, gpu.py, dsp.py, hailo.py, tpu.py, kpu.py, dpu.py, cgra.py, dfm.py
- **Partitioners**: fusion_partitioner.py, attention_fusion_partitioner.py
- **Analysis**: unified_analyzer.py, architecture_comparator.py
- **Visualization**: mermaid_generator.py, visualization.py

**CRITICAL**: Hardware mappers are ALREADY using `.fused_subgraphs`! This means they've been updated for FusionBasedPartitioner but other analyzers haven't.

---

## Root Cause Analysis

### Why the incompatibility exists:

1. **GraphPartitioner** (legacy) creates PartitionReport with `.subgraphs`
   - One SubgraphDescriptor per operator (no fusion)
   - Works with symbolic_trace output
   - Single-op descriptors with string node IDs

2. **FusionBasedPartitioner** (current) creates FusionReport with `.fused_subgraphs`
   - Multiple operators fused into FusedSubgraph
   - Works with Dynamo export output
   - Multi-op descriptors with integer subgraph IDs

3. **Hardware mappers** were updated to use `.fused_subgraphs` (smart!)
   - They already work with FusionBasedPartitioner

4. **Analyzers** (roofline, energy, memory, concurrency) still expect `.subgraphs`
   - They fail when given FusionReport from FusionBasedPartitioner

### Evidence of partial migration:

- SubgraphDescriptor has compatibility properties (`total_flops`, `node_names`)
- Hardware mappers already use `.fused_subgraphs`
- UnifiedAnalyzer tries to handle both but gets confused

---

## Proposed Unification Strategy

### Option 1: Alias FusionReport.subgraphs → FusionReport.fused_subgraphs

**Pros:**
- Minimal code change
- Backward compatibility
- Quick fix

**Cons:**
- Maintains dual naming
- Doesn't solve semantic differences
- Confusing for future developers

### Option 2: Rename FusionReport.fused_subgraphs → FusionReport.subgraphs

**Pros:**
- Standard naming across codebase
- Hardware mappers already handle it

**Cons:**
- Need to update all 15 files using `.fused_subgraphs`
- Loses semantic clarity (fused vs unfused)

### Option 3: Make PartitionReport = FusionReport (deprecate PartitionReport)

**Pros:**
- Single report type
- FusionReport has richer metrics (data movement reduction, etc.)
- Natural fit for Dynamo-first architecture

**Cons:**
- Larger refactor
- Lose some PartitionReport-specific stats (concurrency, critical path)

### Option 4: Unified SubgraphDescriptor (RECOMMENDED)

**Make FusedSubgraph inherit from or BE SubgraphDescriptor**

```python
@dataclass
class SubgraphDescriptor:
    """Unified subgraph descriptor (single or fused operators)"""

    # Identity
    subgraph_id: int                          # Always numeric (FusedSubgraph style)
    node_ids: List[str]                       # List (single-element for unfused)
    node_names: List[str]                     # List (single-element for unfused)
    operation_types: List[OperationType]      # List (single-element for unfused)

    # Computation
    total_flops: int                          # Unified naming
    total_macs: int                           # Unified naming

    # Memory
    total_input_bytes: int
    total_output_bytes: int
    total_weight_bytes: int
    internal_bytes: int = 0                   # Zero for unfused
    arithmetic_intensity: float

    # Fusion metadata
    fusion_pattern: str
    num_operators: int = 1                    # 1 for unfused

    # Dependencies
    depends_on: List[int]                     # Unified to int

    # Characterization
    parallelism: Optional[ParallelismDescriptor]
    recommended_bottleneck: BottleneckType

    # Optional detailed tensor info (only for unfused/single-op)
    input_tensors: List[TensorDescriptor] = field(default_factory=list)
    output_tensors: List[TensorDescriptor] = field(default_factory=list)
    weight_tensors: List[TensorDescriptor] = field(default_factory=list)

    # Optional partition metadata
    partition_reason: Optional[PartitionReason] = None
    fusion_candidates: List[str] = field(default_factory=list)

    # Methods
    def data_movement_reduction(self) -> float:
        """Calculate fusion benefit (0.0 for unfused)"""
        if self.num_operators <= 1:
            return 0.0
        external = self.total_input_bytes + self.total_output_bytes + self.total_weight_bytes
        total = external + self.internal_bytes
        return self.internal_bytes / total if total > 0 else 0.0

    # Backward compatibility aliases
    @property
    def node_id(self) -> str:
        """Legacy: first node ID"""
        return self.node_ids[0] if self.node_ids else ""

    @property
    def node_name(self) -> str:
        """Legacy: first node name"""
        return self.node_names[0] if self.node_names else ""

    @property
    def operation_type(self) -> OperationType:
        """Legacy: first operation type"""
        return self.operation_types[0] if self.operation_types else OperationType.UNKNOWN
```

**Unified PartitionReport:**

```python
@dataclass
class PartitionReport:
    """Unified partition/fusion report"""

    subgraphs: List[SubgraphDescriptor]       # Standard name
    total_subgraphs: int
    original_operators: int                   # NEW: track fusion compression

    # Computation
    total_flops: int
    total_macs: int

    # Memory
    total_memory_traffic: int                 # External (fused)
    total_memory_traffic_unfused: int         # What it would be without fusion
    data_movement_reduction: float            # Fusion benefit

    # Statistics
    average_arithmetic_intensity: float
    min_arithmetic_intensity: float
    max_arithmetic_intensity: float

    # Distributions
    operation_type_counts: Dict[str, int]
    fusion_pattern_counts: Dict[str, int]
    bottleneck_distribution: Dict[str, int]
    parallelism_distribution: Dict[str, int]

    # Fusion quality
    avg_fusion_size: float
    max_fusion_size: int

    # Advanced (optional)
    concurrency: Optional[ConcurrencyDescriptor] = None
    critical_path_subgraphs: List[str] = field(default_factory=list)
    partition_reason_distribution: Dict[str, int] = field(default_factory=dict)

    # Backward compatibility
    @property
    def fused_subgraphs(self) -> List[SubgraphDescriptor]:
        """Alias for backward compat with hardware mappers"""
        return self.subgraphs

    def summary_stats(self) -> str
```

---

## Migration Plan

### Phase 1: Update SubgraphDescriptor (Core Unification)

1. Merge FusedSubgraph attributes into SubgraphDescriptor
2. Add defaults for fusion-specific fields (num_operators=1, internal_bytes=0)
3. Keep backward compatibility properties

### Phase 2: Update FusionBasedPartitioner

1. Return PartitionReport instead of FusionReport
2. Create SubgraphDescriptor instances instead of FusedSubgraph
3. Compute all PartitionReport statistics

### Phase 3: Update Consumers (Minimal Changes)

Most consumers already use `.subgraphs` - they'll just work!

Files needing updates:
- Hardware mappers using `.fused_subgraphs` → use `.subgraphs` or alias
- Analysis using report-level fusion stats → use new PartitionReport fields

### Phase 4: Deprecate FusedSubgraph and FusionReport

1. Add deprecation warnings
2. Remove in next major version

---

## Benefits of Unification

1. **Single code path**: No more parallel structures
2. **Dynamo-first**: Naturally supports FusionBasedPartitioner
3. **Backward compatible**: GraphPartitioner can also create unified SubgraphDescriptor
4. **Richer analytics**: All reports get fusion metrics (even if zero for unfused)
5. **Hardware mapper compatibility**: Already using fused_subgraphs, trivial to update
6. **Future-proof**: New partitioners just use unified SubgraphDescriptor

---

## Risk Assessment

**Low Risk:**
- SubgraphDescriptor already has compatibility properties
- Hardware mappers already use fused_subgraphs (just need alias)
- Most analyzers already use .subgraphs (no change needed)

**Medium Risk:**
- Need to update FusionBasedPartitioner to generate PartitionReport
- Some code may assume single-op subgraphs (need to handle lists)

**Mitigation:**
- Backward compatibility properties
- Phased rollout with testing at each phase
- Keep both names initially (.subgraphs and .fused_subgraphs as aliases)

---

## Recommendation

**Implement Option 4: Unified SubgraphDescriptor**

This provides the cleanest long-term solution while maintaining backward compatibility. The migration path is clear and low-risk.

Next steps:
1. Update SubgraphDescriptor definition in ir/structures.py
2. Update FusionBasedPartitioner to use SubgraphDescriptor and return PartitionReport
3. Add alias property fused_subgraphs to PartitionReport
4. Test with YOLO model end-to-end
5. Run full Embodied AI comparison

---

## Open Questions

1. Should we keep GraphPartitioner or fully migrate to FusionBasedPartitioner?
   - **Recommendation**: Deprecate GraphPartitioner, it doesn't work with Dynamo

2. Should dependencies be `List[int]` or `List[str]`?
   - **Recommendation**: Use `List[int]` (subgraph IDs) for inter-subgraph dependencies

3. Should we keep detailed tensor info (input_tensors, etc.) in SubgraphDescriptor?
   - **Recommendation**: Yes, as optional fields - useful for single-op analysis

4. How to handle concurrency and critical path in fused graphs?
   - **Recommendation**: Keep as optional fields, compute only when requested
