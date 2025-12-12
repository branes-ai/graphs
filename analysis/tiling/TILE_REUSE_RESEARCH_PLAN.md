# Tile Reuse Characterization and Optimization Research Plan

## Problem Statement

Modern DNN accelerators must efficiently manage data movement across a memory hierarchy while maximizing compute utilization. The key challenges are:

1. **Block Algebra Blocking**: When tile sizes exceed L2 capacity (even with double-buffering), we need to introduce a second level of blocking - "blocking the block algebra"
2. **Distributed L3 Orchestration**: For SoCs with distributed L3 caches (like Branes.AI), we can use checkerboard tilings where tiles rotate to adjacent L3s to generate new intermediary outputs without returning to DRAM
3. **Tile Reuse Optimization**: Maximizing data reuse at each level of the memory hierarchy to minimize energy and latency

## Goals

### Goal 1: Block Algebra Framework
Create a composable representation for tile schedules that supports:
- Hierarchical blocking (tiles within tiles)
- Memory capacity constraints at each level
- Double-buffering state tracking
- Block matrix algebra operations (A_block @ B_block = C_block)

### Goal 2: Memory-Constrained Tile Scheduling
Implement schedulers that:
- Respect L1/L2/L3 capacity constraints
- Model double-buffering overhead (2x memory for prefetch)
- Automatically introduce blocking when tiles exceed capacity
- Track working set size at each memory level

### Goal 3: Distributed Memory Tile Orchestration
Support distributed L3 configurations:
- Checkerboard tiling patterns for compute/memory
- Tile rotation algorithms (shift tiles to adjacent L3s)
- Local tile reuse vs remote tile fetch trade-offs
- NUMA-aware tile placement

### Goal 4: Tile Reuse Analysis
Comprehensive reuse characterization:
- Temporal reuse (same data accessed multiple times)
- Spatial reuse (adjacent data in same cache line)
- Reuse distance in cycles/operations
- Producer-consumer reuse (output of one op is input to next)

### Goal 5: Optimization Search Space
Tools to explore:
- Tile size sweep for each dimension (Tm, Tk, Tn)
- Dataflow strategy comparison (WS, OS, RS)
- Memory allocation strategies
- Tile rotation schedules for distributed L3

---

## Architecture Overview

```
src/graphs/research/tiling/
    __init__.py

    # Core representations
    block_algebra.py        # Block matrix algebra, hierarchical tiles
    tile_schedule.py        # Enhanced TileSchedule with memory constraints
    memory_model.py         # Memory hierarchy with capacity constraints

    # Schedulers
    capacity_scheduler.py   # Memory-constrained tile scheduling
    double_buffer.py        # Double-buffering state machine
    blocking_scheduler.py   # Automatic blocking when tiles exceed capacity

    # Distributed memory
    distributed_l3.py       # Distributed L3 topology model
    checkerboard.py         # Checkerboard tiling patterns
    tile_rotation.py        # Tile rotation algorithms

    # Analysis
    reuse_analyzer.py       # Comprehensive reuse analysis
    working_set.py          # Working set size tracking
    energy_optimizer.py     # Energy-aware tile optimization

    # Optimization
    tile_search.py          # Tile size search algorithms
    pareto_frontier.py      # Multi-objective optimization

cli/research/
    tile_reuse_analysis.py      # Analyze tile reuse for operations
    distributed_l3_study.py     # Study distributed L3 orchestration
    blocking_optimizer.py       # Find optimal blocking strategies
    tile_rotation_simulator.py  # Simulate tile rotation patterns
```

---

## Phase 1: Block Algebra Framework

### 1.1 Block Matrix Representation

```python
@dataclass
class BlockMatrix:
    """Block matrix representation for tile algebra."""
    name: str                    # "A", "B", "C"
    shape: Tuple[int, int]       # Full matrix shape (M, K) or (K, N) or (M, N)
    block_shape: Tuple[int, int] # Tile shape (Tm, Tk) etc.
    dtype: str                   # "bf16", "fp32", etc.

    @property
    def num_blocks(self) -> Tuple[int, int]:
        """Number of blocks in each dimension."""
        return (ceil(shape[0] / block_shape[0]),
                ceil(shape[1] / block_shape[1]))

    @property
    def block_bytes(self) -> int:
        """Bytes per block."""
        return block_shape[0] * block_shape[1] * dtype_size(dtype)
```

### 1.2 Hierarchical Blocking

```python
@dataclass
class HierarchicalTileSchedule:
    """Multi-level blocking for when tiles exceed memory capacity."""

    # Problem dimensions
    M: int
    K: int
    N: int

    # Level 1: L2-resident tiles (inner)
    l2_tile: Tuple[int, int, int]  # (Tm_l2, Tk_l2, Tn_l2)

    # Level 2: L3-resident blocks (outer)
    l3_block: Tuple[int, int, int]  # (Tm_l3, Tk_l3, Tn_l3)

    # Optional Level 3: DRAM-resident super-blocks
    dram_superblock: Optional[Tuple[int, int, int]] = None

    def fits_in_l2(self, l2_bytes: int, double_buffer: bool = True) -> bool:
        """Check if L2 tiles fit in L2 capacity."""
        multiplier = 2 if double_buffer else 1
        working_set = self.l2_working_set_bytes()
        return working_set * multiplier <= l2_bytes

    def l2_working_set_bytes(self) -> int:
        """Working set size for L2-resident computation."""
        Tm, Tk, Tn = self.l2_tile
        elem_size = 2  # BF16
        input_tile = Tm * Tk * elem_size
        weight_tile = Tk * Tn * elem_size
        output_tile = Tm * Tn * elem_size * 2  # FP32 accumulator
        return input_tile + weight_tile + output_tile
```

### 1.3 Block Algebra Operations

```python
class BlockAlgebra:
    """Block matrix algebra with memory-aware scheduling."""

    def matmul(self, A: BlockMatrix, B: BlockMatrix, C: BlockMatrix,
               memory_budget: MemoryBudget) -> BlockSchedule:
        """
        Schedule C = A @ B with memory constraints.

        If tiles exceed memory budget, automatically introduce blocking.
        """
        # Try single-level tiling
        schedule = self._try_single_level(A, B, C, memory_budget)
        if schedule.fits():
            return schedule

        # Need hierarchical blocking
        return self._hierarchical_blocking(A, B, C, memory_budget)

    def _hierarchical_blocking(self, A, B, C, budget) -> BlockSchedule:
        """Introduce blocking when tiles exceed L2 capacity."""
        # Find largest L2 tile that fits with double-buffering
        l2_tile = self._find_max_l2_tile(A, B, C, budget.l2_bytes)

        # Compute L3 block size based on L3 capacity
        l3_block = self._find_max_l3_block(A, B, C, budget.l3_bytes, l2_tile)

        return HierarchicalBlockSchedule(
            l2_tile=l2_tile,
            l3_block=l3_block,
            dataflow=budget.dataflow
        )
```

---

## Phase 2: Memory-Constrained Scheduling

### 2.1 Memory Budget Model

```python
@dataclass
class MemoryBudget:
    """Memory capacity constraints at each level."""

    # Capacity (bytes)
    l1_bytes: int = 256 * 1024      # 256 KB register file / scratchpad
    l2_bytes: int = 4 * 1024 * 1024  # 4 MB L2 per cluster
    l3_bytes: int = 32 * 1024 * 1024 # 32 MB L3 slice

    # Double-buffering
    double_buffer_l1: bool = False   # Usually not enough space
    double_buffer_l2: bool = True    # Standard practice
    double_buffer_l3: bool = False   # Depends on architecture

    # Allocation strategy
    l2_partition: str = "static"     # "static" or "dynamic"
    l3_partition: str = "shared"     # "shared", "sliced", "distributed"

    @property
    def effective_l2_bytes(self) -> int:
        """Effective L2 capacity accounting for double-buffering."""
        return self.l2_bytes // 2 if self.double_buffer_l2 else self.l2_bytes
```

### 2.2 Double-Buffer State Machine

```python
class DoubleBufferScheduler:
    """
    Schedule tile execution with double-buffering.

    While computing on buffer A, prefetch next tile into buffer B.
    Swap buffers between iterations.
    """

    def __init__(self, l2_bytes: int, prefetch_latency_cycles: int):
        self.buffer_bytes = l2_bytes // 2  # Each buffer gets half
        self.prefetch_latency = prefetch_latency_cycles

    def schedule(self, tiles: List[Tile]) -> DoubleBufferSchedule:
        """
        Generate execution schedule with prefetch/compute overlap.

        Returns schedule with:
        - Compute timeline (which tile executing when)
        - Prefetch timeline (which tile loading when)
        - Stall cycles (when prefetch not ready)
        """
        schedule = DoubleBufferSchedule()

        # Initial load (no overlap)
        schedule.add_prefetch(tiles[0], cycle=0)
        compute_start = self.prefetch_latency

        for i, tile in enumerate(tiles):
            # Start compute
            compute_cycles = tile.compute_cycles
            schedule.add_compute(tile, cycle=compute_start)

            # Prefetch next tile (overlapped with current compute)
            if i + 1 < len(tiles):
                prefetch_start = compute_start
                schedule.add_prefetch(tiles[i+1], cycle=prefetch_start)

                # Check if prefetch completes before compute
                prefetch_done = prefetch_start + self.prefetch_latency
                compute_done = compute_start + compute_cycles

                if prefetch_done > compute_done:
                    # Stall waiting for prefetch
                    stall_cycles = prefetch_done - compute_done
                    schedule.add_stall(stall_cycles, cycle=compute_done)
                    compute_start = prefetch_done
                else:
                    compute_start = compute_done
            else:
                compute_start += compute_cycles

        return schedule
```

### 2.3 Automatic Blocking

```python
class BlockingScheduler:
    """
    Automatically introduce blocking when tiles exceed memory capacity.

    Given: matmul(M, K, N) with target tile sizes (Tm, Tk, Tn)
    If working set > L2 capacity:
        1. Reduce tile sizes to fit L2
        2. Introduce L3-level blocking over reduced tiles
        3. If still too large, introduce DRAM-level super-blocking
    """

    def schedule(self, M: int, K: int, N: int,
                 target_tile: Tuple[int, int, int],
                 budget: MemoryBudget) -> HierarchicalSchedule:

        # Check if target tile fits
        working_set = self._working_set(target_tile)

        if working_set <= budget.effective_l2_bytes:
            # Single-level tiling sufficient
            return SingleLevelSchedule(tile=target_tile)

        # Need to reduce tile size for L2
        l2_tile = self._find_l2_tile(M, K, N, budget.effective_l2_bytes)

        # Compute how many L2 tiles fit in L3
        l3_block = self._find_l3_block(M, K, N, l2_tile, budget.l3_bytes)

        return HierarchicalSchedule(
            l2_tile=l2_tile,
            l3_block=l3_block,
            loop_order=self._compute_loop_order(budget.dataflow)
        )
```

---

## Phase 3: Distributed L3 Orchestration

### 3.1 Distributed L3 Topology

```python
@dataclass
class L3Slice:
    """Single L3 cache slice in a distributed configuration."""
    slice_id: int
    capacity_bytes: int
    position: Tuple[int, int]  # (row, col) in mesh
    bandwidth_gbps: float

@dataclass
class DistributedL3:
    """
    Distributed L3 cache topology.

    Models architectures like:
    - Intel mesh interconnect (L3 slices per core)
    - AMD CCD L3 (per chiplet)
    - Branes.AI distributed L3 (checkerboard)
    """
    slices: List[L3Slice]
    topology: str  # "mesh", "ring", "crossbar"

    # Inter-slice communication
    hop_latency_cycles: int
    hop_energy_pj: float

    def distance(self, src: int, dst: int) -> int:
        """Number of hops between slices."""
        src_pos = self.slices[src].position
        dst_pos = self.slices[dst].position
        return abs(src_pos[0] - dst_pos[0]) + abs(src_pos[1] - dst_pos[1])

    def transfer_cost(self, src: int, dst: int, bytes: int) -> TransferCost:
        """Cost to transfer data between slices."""
        hops = self.distance(src, dst)
        return TransferCost(
            latency_cycles=hops * self.hop_latency_cycles,
            energy_pj=hops * self.hop_energy_pj * bytes
        )
```

### 3.2 Checkerboard Tiling

```python
class CheckerboardTiling:
    """
    Checkerboard pattern for compute/memory tiles.

    Pattern alternates between:
    - Compute tiles: Actively computing matmul
    - Memory tiles: Staging data for adjacent compute tiles

    This enables:
    1. Local data reuse (compute tile uses data from adjacent memory tile)
    2. Tile rotation (memory tiles rotate to become compute tiles)
    3. Reduced DRAM traffic (intermediates stay in L3)
    """

    def __init__(self, l3_topology: DistributedL3, tile_size: Tuple[int, int, int]):
        self.topology = l3_topology
        self.tile_size = tile_size

    def create_pattern(self, M: int, K: int, N: int) -> CheckerboardPattern:
        """
        Create checkerboard assignment of tiles to L3 slices.

        Returns mapping of (tile_m, tile_n) -> L3Slice
        with alternating compute/memory roles.
        """
        num_m_tiles = ceil(M / self.tile_size[0])
        num_n_tiles = ceil(N / self.tile_size[2])

        pattern = CheckerboardPattern()
        for tm in range(num_m_tiles):
            for tn in range(num_n_tiles):
                # Checkerboard: even sum = compute, odd sum = memory
                role = "compute" if (tm + tn) % 2 == 0 else "memory"
                slice_id = self._assign_slice(tm, tn)
                pattern.add(tm, tn, slice_id, role)

        return pattern
```

### 3.3 Tile Rotation Algorithms

```python
class TileRotation:
    """
    Tile rotation for distributed L3 architectures.

    Key insight: After computing C_ij = sum_k(A_ik @ B_kj), the partial
    result C_ij can be rotated to an adjacent L3 slice where it becomes
    an input for the next computation, avoiding DRAM round-trip.

    Algorithms:
    1. Cannon's Algorithm: Systolic rotation for square decompositions
    2. SUMMA: Broadcast-based for rectangular decompositions
    3. 2.5D: Uses extra memory to reduce communication
    """

    def cannon_rotation(self, pattern: CheckerboardPattern,
                        iteration: int) -> RotationSchedule:
        """
        Cannon's algorithm rotation step.

        - A tiles shift left (wrap around)
        - B tiles shift up (wrap around)
        - C tiles accumulate in place
        """
        schedule = RotationSchedule()

        for tile in pattern.a_tiles:
            # Shift A left by one position
            src_slice = tile.current_slice
            dst_slice = self._left_neighbor(src_slice)
            schedule.add_transfer(tile, src_slice, dst_slice)

        for tile in pattern.b_tiles:
            # Shift B up by one position
            src_slice = tile.current_slice
            dst_slice = self._up_neighbor(src_slice)
            schedule.add_transfer(tile, src_slice, dst_slice)

        return schedule

    def summa_broadcast(self, pattern: CheckerboardPattern,
                        k_iteration: int) -> BroadcastSchedule:
        """
        SUMMA algorithm broadcast step.

        - Owner of A[:,k] broadcasts to row
        - Owner of B[k,:] broadcasts to column
        - All nodes compute local C += A @ B
        """
        schedule = BroadcastSchedule()

        # Broadcast A panel to row
        a_owner = k_iteration % pattern.num_cols
        schedule.add_row_broadcast(pattern.a_panel(k_iteration), a_owner)

        # Broadcast B panel to column
        b_owner = k_iteration % pattern.num_rows
        schedule.add_col_broadcast(pattern.b_panel(k_iteration), b_owner)

        return schedule
```

---

## Phase 4: Tile Reuse Analysis

### 4.1 Reuse Classification

```python
class ReuseType(Enum):
    """Types of data reuse in tiled computations."""

    TEMPORAL = "temporal"      # Same element accessed multiple times
    SPATIAL = "spatial"        # Adjacent elements in same cache line
    PRODUCER_CONSUMER = "pc"   # Output of one op is input to next
    MULTICAST = "multicast"    # Same data used by multiple PEs

@dataclass
class ReuseAnalysis:
    """Comprehensive reuse analysis for a tile schedule."""

    # Per-operand reuse factors
    input_temporal_reuse: float   # How many times each input element used
    input_spatial_reuse: float    # Cache line utilization for inputs
    weight_temporal_reuse: float
    weight_spatial_reuse: float
    output_temporal_reuse: float  # Partial sum accumulation

    # Reuse distances (in cycles/operations)
    input_reuse_distance: int     # Cycles between reuses
    weight_reuse_distance: int
    output_reuse_distance: int

    # Producer-consumer reuse (for fused operations)
    pc_reuse_factor: float        # Fraction staying in registers/L1
    pc_reuse_bytes: int           # Bytes avoiding memory round-trip

    # Effective bandwidth
    actual_dram_bytes: int
    minimum_dram_bytes: int       # If perfect reuse
    reuse_efficiency: float       # minimum / actual
```

### 4.2 Reuse Analyzer

```python
class TileReuseAnalyzer:
    """
    Analyze data reuse for tiled matrix operations.

    For C = A @ B with tiles (Tm, Tk, Tn):
    - Input A[m,k]: reused Tn times (across N tiles)
    - Weight B[k,n]: reused Tm times (across M tiles)
    - Output C[m,n]: reused Tk times (K reduction)
    """

    def analyze(self, schedule: TileSchedule,
                cache_config: CacheConfig) -> ReuseAnalysis:

        Tm, Tk, Tn = schedule.tile_sizes
        num_m, num_k, num_n = schedule.num_tiles

        # Temporal reuse
        input_reuse = num_n  # A reused across N tiles
        weight_reuse = num_m  # B reused across M tiles
        output_reuse = num_k  # C accumulated across K tiles

        # Spatial reuse (cache line utilization)
        line_size = cache_config.line_size_bytes
        elem_size = schedule.element_size

        # For row-major A[M,K]: spatial reuse along K dimension
        input_spatial = min(line_size // elem_size, Tk)

        # For row-major B[K,N]: spatial reuse along N dimension
        weight_spatial = min(line_size // elem_size, Tn)

        # Reuse distance (cycles between consecutive accesses)
        # Depends on loop order (dataflow)
        reuse_distances = self._compute_reuse_distances(schedule)

        return ReuseAnalysis(
            input_temporal_reuse=input_reuse,
            input_spatial_reuse=input_spatial,
            weight_temporal_reuse=weight_reuse,
            weight_spatial_reuse=weight_spatial,
            output_temporal_reuse=output_reuse,
            input_reuse_distance=reuse_distances['input'],
            weight_reuse_distance=reuse_distances['weight'],
            output_reuse_distance=reuse_distances['output'],
            ...
        )

    def _compute_reuse_distances(self, schedule: TileSchedule) -> Dict[str, int]:
        """
        Compute reuse distance based on dataflow.

        Weight-stationary: weights have distance 1 (stay in place)
        Output-stationary: outputs have distance 1
        Row-stationary: balanced distances
        """
        if schedule.dataflow == DataflowType.WEIGHT_STATIONARY:
            return {
                'input': schedule.num_n_tiles * schedule.tile_compute_cycles,
                'weight': 1,  # Stays in place
                'output': schedule.num_k_tiles * schedule.tile_compute_cycles,
            }
        # ... other dataflows
```

### 4.3 Working Set Tracking

```python
class WorkingSetTracker:
    """
    Track working set size at each memory level during execution.

    Monitors:
    - Live data at each level (L1, L2, L3, DRAM)
    - Peak working set (max simultaneous live data)
    - Working set timeline (how it changes over execution)
    """

    def __init__(self, memory_config: MemoryConfig):
        self.config = memory_config
        self.timeline = []

    def track_execution(self, schedule: TileSchedule) -> WorkingSetTimeline:
        """Generate working set timeline for schedule execution."""

        timeline = WorkingSetTimeline()

        for cycle, event in schedule.iterate_events():
            if event.type == "load":
                timeline.add_to_level(event.level, event.bytes, cycle)
            elif event.type == "evict":
                timeline.remove_from_level(event.level, event.bytes, cycle)
            elif event.type == "spill":
                # Data moves from higher to lower level
                timeline.remove_from_level(event.src_level, event.bytes, cycle)
                timeline.add_to_level(event.dst_level, event.bytes, cycle)

        return timeline

    def find_capacity_violations(self, timeline: WorkingSetTimeline) -> List[Violation]:
        """Find points where working set exceeds capacity."""
        violations = []
        for level in ['L1', 'L2', 'L3']:
            capacity = self.config.capacity(level)
            for cycle, size in timeline.level_timeline(level):
                if size > capacity:
                    violations.append(Violation(level, cycle, size, capacity))
        return violations
```

---

## Phase 5: Optimization Tools

### 5.1 Tile Size Search

```python
class TileSizeOptimizer:
    """
    Search for optimal tile sizes given constraints.

    Objectives:
    1. Minimize DRAM traffic (maximize reuse)
    2. Minimize latency (maximize parallelism)
    3. Minimize energy (balance compute vs memory)

    Constraints:
    - Tile working set <= memory capacity
    - Tile dimensions divisible by array size (for systolic)
    """

    def search(self, M: int, K: int, N: int,
               hardware: HardwareConfig,
               objective: str = "energy") -> OptimalTileSchedule:

        # Generate candidate tile sizes
        candidates = self._generate_candidates(M, K, N, hardware)

        # Evaluate each candidate
        results = []
        for tile in candidates:
            schedule = TileSchedule(M, K, N, *tile)

            # Check constraints
            if not self._satisfies_constraints(schedule, hardware):
                continue

            # Compute metrics
            reuse = self.reuse_analyzer.analyze(schedule, hardware.cache)
            energy = self.energy_model.estimate(schedule, hardware)
            latency = self.latency_model.estimate(schedule, hardware)

            results.append(TileResult(
                tile=tile,
                dram_bytes=reuse.actual_dram_bytes,
                energy_mj=energy.total_mj,
                latency_ms=latency.total_ms,
                utilization=schedule.compute_utilization(hardware)
            ))

        # Select best based on objective
        if objective == "energy":
            return min(results, key=lambda r: r.energy_mj)
        elif objective == "latency":
            return min(results, key=lambda r: r.latency_ms)
        elif objective == "dram":
            return min(results, key=lambda r: r.dram_bytes)
```

### 5.2 Pareto Frontier

```python
class ParetoOptimizer:
    """
    Multi-objective optimization for tile scheduling.

    Computes Pareto frontier over:
    - Energy
    - Latency
    - DRAM bandwidth
    - Compute utilization
    """

    def compute_frontier(self, candidates: List[TileResult]) -> List[TileResult]:
        """Extract Pareto-optimal tile configurations."""

        pareto = []
        for candidate in candidates:
            dominated = False
            for other in candidates:
                if self._dominates(other, candidate):
                    dominated = True
                    break
            if not dominated:
                pareto.append(candidate)

        return pareto

    def _dominates(self, a: TileResult, b: TileResult) -> bool:
        """Check if a dominates b (better or equal in all objectives)."""
        dominated_all = True
        strictly_better = False

        for obj in ['energy_mj', 'latency_ms', 'dram_bytes']:
            a_val = getattr(a, obj)
            b_val = getattr(b, obj)
            if a_val > b_val:  # a is worse
                dominated_all = False
            if a_val < b_val:  # a is strictly better
                strictly_better = True

        return dominated_all and strictly_better
```

---

## CLI Tools

### analyze_tile_reuse.py
```bash
# Analyze tile reuse for a single operation
python cli/research/analyze_tile_reuse.py \
    --M 1024 --K 512 --N 1024 \
    --tile 128 64 128 \
    --dataflow weight_stationary \
    --l2-size 4MB

# Compare reuse across dataflows
python cli/research/analyze_tile_reuse.py \
    --M 1024 --K 512 --N 1024 \
    --compare-dataflows \
    --output reuse_comparison.json
```

### optimize_blocking.py
```bash
# Find optimal blocking for memory-constrained system
python cli/research/optimize_blocking.py \
    --M 4096 --K 4096 --N 4096 \
    --l2-size 4MB --l3-size 32MB \
    --double-buffer \
    --output blocking_schedule.json
```

### simulate_tile_rotation.py
```bash
# Simulate Cannon's algorithm on distributed L3
python cli/research/simulate_tile_rotation.py \
    --M 4096 --K 4096 --N 4096 \
    --l3-slices 8 \
    --topology mesh \
    --algorithm cannon \
    --output rotation_timeline.json
```

### distributed_l3_study.py
```bash
# Study distributed L3 configurations for Branes.AI
python cli/research/distributed_l3_study.py \
    --model resnet50 \
    --l3-config checkerboard \
    --slices 4 8 16 \
    --compare-dram-traffic \
    --output distributed_l3_study.pdf
```

---

## Implementation Order

### Week 1: Core Framework
1. `memory_model.py` - Memory hierarchy with capacity constraints
2. `block_algebra.py` - Block matrix representation
3. `tile_schedule.py` - Enhanced TileSchedule with hierarchical blocking

### Week 2: Scheduling
4. `double_buffer.py` - Double-buffering state machine
5. `capacity_scheduler.py` - Memory-constrained scheduling
6. `blocking_scheduler.py` - Automatic blocking

### Week 3: Distributed Memory
7. `distributed_l3.py` - Distributed L3 topology
8. `checkerboard.py` - Checkerboard patterns
9. `tile_rotation.py` - Cannon/SUMMA algorithms

### Week 4: Analysis & Optimization
10. `reuse_analyzer.py` - Comprehensive reuse analysis
11. `working_set.py` - Working set tracking
12. `tile_search.py` - Tile size optimization

### Week 5: CLI & Validation
13. CLI tools
14. Test suite
15. Validation against analytical models

---

## References

1. **Cannon's Algorithm**: Cannon, L.E. (1969). A cellular computer to implement the Kalman Filter Algorithm.
2. **SUMMA**: Van De Geijn, R.A. and Watts, J. (1997). SUMMA: Scalable Universal Matrix Multiplication Algorithm.
3. **2.5D Algorithms**: Solomonik, E. and Demmel, J. (2011). Communication-optimal parallel 2.5D matrix multiplication.
4. **Roofline Model**: Williams, S. et al. (2009). Roofline: An insightful visual performance model.
5. **Eyeriss Dataflow**: Chen, Y.H. et al. (2017). Eyeriss: An Energy-Efficient Reconfigurable Accelerator for DNNs.

---

## Success Criteria

1. **Block Algebra**: Can represent hierarchical blocking with arbitrary levels
2. **Memory Constraints**: Automatically introduces blocking when tiles exceed L2
3. **Double-Buffering**: Correctly models prefetch/compute overlap and stalls
4. **Distributed L3**: Can simulate Cannon/SUMMA on checkerboard topology
5. **Reuse Analysis**: Correctly computes temporal, spatial, and PC reuse
6. **Optimization**: Finds Pareto-optimal tile configurations
7. **Validation**: Results match analytical models within 5%
