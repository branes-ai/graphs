# Reuse Analysis

 Tile Reuse Characterization Framework


## Core Modules (src/graphs/research/tiling/):

Files Created

  1. memory_model.py - Memory hierarchy with capacity constraints
    - MemoryLevel, MemoryHierarchy, MemoryBudget, WorkingSetState
    - Predefined hierarchies: TPU v4, H100, Distributed L3, KPU
  2. block_algebra.py - Block matrix algebra
    - BlockMatrix, BlockOperation, BlockSchedule
    - BlockScheduleGenerator for different loop orders
    - HierarchicalBlockSchedule for multi-level blocking
  3. tile_schedule.py - Enhanced tile scheduling
    - TileConfig, TileSchedule with reuse analysis
    - HierarchicalTileSchedule for multi-level blocking
    - TileScheduler respecting memory constraints
    - Dataflow comparison utilities
  4. double_buffer.py - Double-buffering state machine
    - Buffer, DoubleBuffer with ping-pong operation
    - DoubleBufferScheduler, TripleBufferScheduler
    - ExecutionTimeline for tracking prefetch/compute overlap
  5. distributed_l3.py - Distributed L3 topology
    - L3Slice, DistributedL3 with mesh/ring/crossbar topologies
    - CheckerboardPattern for compute/memory role assignment
    - DataDistribution for tile placement
    - Communication cost analysis
  6. tile_rotation.py - Rotation algorithms
    - CannonAlgorithm - Systolic rotation for square grids
    - SUMMAAlgorithm - Broadcast-based for rectangular grids
    - Algorithm25D - Memory-communication trade-off
  7. reuse_analyzer.py - Comprehensive reuse analysis
    - ReuseMetrics, ReuseAnalysis per operand
    - Temporal, spatial, producer-consumer reuse
    - Energy impact estimation
    - Dataflow effectiveness scoring

## CLI Tool (cli/research/):

  8. analyze_tile_reuse.py - Analysis CLI
    - Basic tile analysis
    - Dataflow comparison (--compare-dataflows)
    - Tile size sweep (--sweep-tiles)
    - Distributed L3 analysis (--distributed-l3)
    - Rotation algorithm comparison (--rotation-algorithms)
    - Double-buffering analysis (--double-buffer)
    - JSON/text output

## Usage Examples

```bash
  # Basic analysis
  python cli/research/analyze_tile_reuse.py --M 1024 --K 512 --N 1024 --tile 128 64 128

  # Compare dataflows
  python cli/research/analyze_tile_reuse.py --M 1024 --K 512 --N 1024 --compare-dataflows

  # Sweep tile sizes
  python cli/research/analyze_tile_reuse.py --M 4096 --K 4096 --N 4096 --sweep-tiles

  # Rotation algorithms for distributed L3
  python cli/research/analyze_tile_reuse.py --M 4096 --K 4096 --N 4096 --rotation-algorithms --num-procs 16

  # Double-buffering benefit
  python cli/research/analyze_tile_reuse.py --M 2048 --K 2048 --N 2048 --double-buffer --l2-size 4

  # JSON output
  python cli/research/analyze_tile_reuse.py --M 1024 --K 512 --N 1024 --output results.json
```

## The Problem with (Tm, Tk, Tn) Notation

There is no such thing as a 3D tile. What actually exists:

  - A tile: 2D submatrix of shape (Tm, Tk)
  - B tile: 2D submatrix of shape (Tk, Tn)
  - C tile: 2D submatrix of shape (Tm, Tn)

The (Tm, Tk, Tn) notation comes from the loop nest perspective - it describes the iteration space of a tiled matmul:

```cpp
  for m in range(0, M, Tm):      # Outer M loop
      for n in range(0, N, Tn):  # Outer N loop
          for k in range(0, K, Tk):  # Outer K loop (reduction)
              C[m:m+Tm, n:n+Tn] += A[m:m+Tm, k:k+Tk] @ B[k:k+Tk, n:n+Tn]
```

  This is the compiler/scheduler view, not the memory system view.

### The Correct Approach

For tile reuse analysis, we should think in terms of actual data objects:

  A_tile[i,k]: shape (Tm, Tk), bytes = Tm * Tk * dtype_size
  B_tile[k,j]: shape (Tk, Tn), bytes = Tk * Tn * dtype_size
  C_tile[i,j]: shape (Tm, Tn), bytes = Tm * Tn * acc_dtype_size

The reuse questions become:

  1. A_tile[i,k]: How many times is this tile used before eviction?
    - Used for all j in 0..num_n_tiles (reuse = N/Tn times)
    - Must fit in cache for that duration
  2. B_tile[k,j]: How many times is this tile used before eviction?
    - Used for all i in 0..num_m_tiles (reuse = M/Tm times)
    - Must fit in cache for that duration
  3. C_tile[i,j]: How many times is this tile updated?
    - Accumulated K/Tk times
    - Should stay in registers/L1 during accumulation

Corrected Tile Model:
  - A_tile: (Tm, Tk) - input activation tile
  - B_tile: (Tk, Tn) - weight tile
  - C_tile: (Tm, Tn) - output/accumulator tile

The (Tm, Tk, Tn) notation now correctly describes loop bounds, not a single 3D tile.

Key Files Refactored:

  | File                  | Purpose                                                              |
  |-----------------------|----------------------------------------------------------------------|
  | block_algebra.py      | Core 2D tile types (ATile, BTile, CTile, MatmulTiling, TileSchedule) |
  | reuse_analyzer.py     | Per-tile-type reuse metrics (a_metrics, b_metrics, c_metrics)        |
  | __init__.py           | Updated exports with documentation emphasizing 2D tiles              |
  | analyze_tile_reuse.py | CLI with explicit --Tm, --Tk, --Tn arguments                         |

  Verified Output:
  TILE SHAPES (2D submatrices)
  ----------------------------------------------------------------------
    A_tile: (128, 64) @ BF16  = 16,384 bytes
    B_tile: (64, 128) @ BF16  = 16,384 bytes
    C_tile: (128, 128) @ FP32 = 65,536 bytes

  TILE COUNTS AND REUSE
  ----------------------------------------------------------------------
  Tile     Count      Reuse      Min Bytes       Actual Bytes    Efficiency
  A        64         8.0        1,048,576       1,048,576       100.0%
  B        64         8.0        1,048,576       1,048,576       100.0%
  C        64         8.0        4,194,304       4,194,304       100.0%

The framework now correctly tracks:
  1. Per-tile-type memory footprint and reuse
  2. Tile lifetimes (when tiles are first/last used in schedule)
  3. Peak working set (tiles simultaneously live in cache)
  4. Loop order effects on reuse patterns (MNK=output-stationary, NKM=weight-stationary, MKN=input-stationary)

##  Memory-Constrained Tile Size Optimization

  The new tile_optimizer.py module provides the missing link between memory constraints and blocking decisions:

  Key Components

  | Class                | Purpose                                                |
  |----------------------|--------------------------------------------------------|
  | TileConstraint       | Captures constraint analysis for a memory level        |
  | HierarchicalBlocking | Multi-level blocking scheme (L1, L2, L3 tiles)         |
  | TileSizeOptimizer    | Main optimizer that determines when blocking is needed |

  The Blocking Decision Logic

  # The optimizer checks at each level:
  constraint = optimizer.check_constraint(M, K, N, "L2")

  if constraint.requires_blocking:
      # Working set exceeds capacity - must tile smaller
      # A[M,K] + B[K,N] + C[M,N] > L2_capacity
      Tm, Tk, Tn = optimizer.optimize_for_level(M, K, N, "L2")

  Working Set Calculation (2D Tiles)

  working_set = A_tile + B_tile + C_tile
              = (Tm * Tk * input_bytes) + (Tk * Tn * weight_bytes) + (Tm * Tn * accum_bytes)

  Hierarchical Blocking

  When working set exceeds L2, the optimizer creates:
  - L1 tiles: Innermost compute tiles that fit in L1
  - L2 tiles: Mid-level tiles composed of L1 tiles
  - L3 tiles: Outermost tiles (if L3 is configured)

  This generates the classic 6-loop tiled matmul structure when blocking is required at multiple levels.

