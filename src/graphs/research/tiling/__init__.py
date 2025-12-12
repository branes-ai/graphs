"""
Tile Reuse Characterization and Optimization

Research framework for studying tile scheduling, data reuse, and
distributed memory orchestration for DNN accelerators.

Key Design Principle:
    Tiles are 2D submatrices, not 3D objects. For C = A @ B:
    - A_tile: shape (Tm, Tk) - input activation tile
    - B_tile: shape (Tk, Tn) - weight tile
    - C_tile: shape (Tm, Tn) - output/accumulator tile

    The (Tm, Tk, Tn) notation describes loop bounds, not a single tile.

Modules:
- memory_model: Memory hierarchy with capacity constraints
- tile_optimizer: Memory-constrained tile size selection (triggers blocking)
- block_algebra: Block matrix algebra with explicit 2D tile types
- reuse_analyzer: Per-tile-type reuse analysis
- double_buffer: Double-buffering state machine
- distributed_l3: Distributed L3 topology and checkerboard patterns
- tile_rotation: Cannon/SUMMA rotation algorithms

Example usage:
    from graphs.research.tiling import (
        MatmulTiling, TileSchedule, LoopOrder,
        TileReuseAnalyzer, analyze_tile_reuse
    )

    # Analyze with explicit 2D tile shapes
    analysis = analyze_tile_reuse(
        M=1024, K=512, N=1024,   # Problem dimensions
        Tm=128, Tk=64, Tn=128,   # Tile dimensions
        loop_order="MNK",        # Output-stationary
        verbose=True
    )

    # Access per-tile-type metrics
    print(f"A_tile shape: ({analysis.tiling.Tm}, {analysis.tiling.Tk})")
    print(f"A_tile reuse: {analysis.a_metrics.reuse_factor}x")
    print(f"B_tile shape: ({analysis.tiling.Tk}, {analysis.tiling.Tn})")
    print(f"B_tile reuse: {analysis.b_metrics.reuse_factor}x")
"""

from .memory_model import (
    MemoryLevel,
    MemoryLevelType,
    MemoryHierarchy,
    MemoryBudget,
    WorkingSetState,
    create_tpu_v4_hierarchy,
    create_h100_hierarchy,
    create_distributed_l3_hierarchy,
    create_kpu_hierarchy,
    TPU_V4_BUDGET,
    H100_BUDGET,
    DISTRIBUTED_L3_BUDGET,
    KPU_BUDGET,
)

from .block_algebra import (
    DataType,
    TileShape,
    ATile,
    BTile,
    CTile,
    TileSet,
    MatmulTiling,
    TileOperation,
    LoopOrder,
    TileSchedule,
    analyze_memory_traffic,
)

from .reuse_analyzer import (
    TileReuseMetrics,
    WorkingSetAnalysis,
    TileReuseAnalysis,
    TileReuseAnalyzer,
    print_tile_analysis,
    analyze_tile_reuse,
)

from .double_buffer import (
    BufferState,
    Buffer,
    DoubleBuffer,
    TimelineEvent,
    ExecutionTimeline,
    TileDescriptor,
    DoubleBufferScheduler,
    TripleBufferScheduler,
    analyze_double_buffer_benefit,
    create_tiles_from_schedule,
)

from .distributed_l3 import (
    TopologyType,
    SliceRole,
    Position,
    L3Slice,
    TransferCost,
    DistributedL3,
    CheckerboardPattern,
    TilePlacement,
    DataDistribution,
    DistributedMatmulAnalysis,
    create_mesh_l3,
    create_ring_l3,
)

from .tile_rotation import (
    RotationDirection,
    TileState,
    RotationStep,
    RotationSchedule,
    CannonAlgorithm,
    SUMMAAlgorithm,
    Algorithm25D,
    compare_algorithms,
    optimal_algorithm_for_problem,
)

from .tile_optimizer import (
    TileConstraint,
    HierarchicalBlocking,
    TileSizeOptimizer,
    print_blocking_analysis,
    analyze_with_memory_constraints,
)

__all__ = [
    # memory_model
    'MemoryLevel',
    'MemoryLevelType',
    'MemoryHierarchy',
    'MemoryBudget',
    'WorkingSetState',
    'create_tpu_v4_hierarchy',
    'create_h100_hierarchy',
    'create_distributed_l3_hierarchy',
    'create_kpu_hierarchy',
    'TPU_V4_BUDGET',
    'H100_BUDGET',
    'DISTRIBUTED_L3_BUDGET',
    'KPU_BUDGET',

    # block_algebra - 2D tile types
    'DataType',
    'TileShape',
    'ATile',
    'BTile',
    'CTile',
    'TileSet',
    'MatmulTiling',
    'TileOperation',
    'LoopOrder',
    'TileSchedule',
    'analyze_memory_traffic',

    # reuse_analyzer
    'TileReuseMetrics',
    'WorkingSetAnalysis',
    'TileReuseAnalysis',
    'TileReuseAnalyzer',
    'print_tile_analysis',
    'analyze_tile_reuse',

    # double_buffer
    'BufferState',
    'Buffer',
    'DoubleBuffer',
    'TimelineEvent',
    'ExecutionTimeline',
    'TileDescriptor',
    'DoubleBufferScheduler',
    'TripleBufferScheduler',
    'analyze_double_buffer_benefit',
    'create_tiles_from_schedule',

    # distributed_l3
    'TopologyType',
    'SliceRole',
    'Position',
    'L3Slice',
    'TransferCost',
    'DistributedL3',
    'CheckerboardPattern',
    'TilePlacement',
    'DataDistribution',
    'DistributedMatmulAnalysis',
    'create_mesh_l3',
    'create_ring_l3',

    # tile_rotation
    'RotationDirection',
    'TileState',
    'RotationStep',
    'RotationSchedule',
    'CannonAlgorithm',
    'SUMMAAlgorithm',
    'Algorithm25D',
    'compare_algorithms',
    'optimal_algorithm_for_problem',

    # tile_optimizer
    'TileConstraint',
    'HierarchicalBlocking',
    'TileSizeOptimizer',
    'print_blocking_analysis',
    'analyze_with_memory_constraints',
]
