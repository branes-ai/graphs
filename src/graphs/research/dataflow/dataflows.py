"""
Dataflow Strategy Implementations

Generate loop nests for different dataflow strategies:
- Weight-stationary (TPU-style)
- Output-stationary
- Row-stationary (Eyeriss-style)
"""

from typing import Tuple

from graphs.research.dataflow.tiling import TileSchedule, DataflowType
from graphs.research.dataflow.loop_nests import (
    LoopNest,
    LoopLevel,
    LoopVariable,
    create_loop_nest_from_schedule,
)


def generate_weight_stationary_loop_nest(
    M: int,
    K: int,
    N: int,
    Tm: int,
    Tk: int,
    Tn: int,
) -> LoopNest:
    """
    Generate weight-stationary loop nest (TPU-style).

    Weight-stationary maximizes weight reuse by keeping weights
    in the array registers while streaming activations through.

    Pseudocode:
    ```
    for n_tile in [0, ceil(N/Tn)):
        for k_tile in [0, ceil(K/Tk)):
            load_weights(B[k_tile*Tk:(k_tile+1)*Tk, n_tile*Tn:(n_tile+1)*Tn])
            for m_tile in [0, ceil(M/Tm)):
                load_inputs(A[m_tile*Tm:(m_tile+1)*Tm, k_tile*Tk:(k_tile+1)*Tk])
                for n_i in [0, Tn):
                    for k_i in [0, Tk):
                        for m_i in [0, Tm):
                            C[m_tile*Tm+m_i, n_tile*Tn+n_i] +=
                                A[m_tile*Tm+m_i, k_tile*Tk+k_i] *
                                B[k_tile*Tk+k_i, n_tile*Tn+n_i]
    ```

    Args:
        M: Total output rows
        K: Reduction dimension
        N: Total output columns
        Tm: M tile size
        Tk: K tile size
        Tn: N tile size

    Returns:
        LoopNest with complete structure
    """
    import math

    num_m_tiles = math.ceil(M / Tm)
    num_k_tiles = math.ceil(K / Tk)
    num_n_tiles = math.ceil(N / Tn)

    loops = [
        # Outer tile loops (DRAM level)
        LoopLevel(
            variable=LoopVariable.TN,
            bound=num_n_tiles,
            touches_weight=True,
            touches_output=True,
            memory_level="DRAM",
        ),
        LoopLevel(
            variable=LoopVariable.TK,
            bound=num_k_tiles,
            touches_input=True,
            touches_weight=True,
            memory_level="DRAM",
        ),
        # Middle tile loop (L1 level - weights stay loaded)
        LoopLevel(
            variable=LoopVariable.TM,
            bound=num_m_tiles,
            touches_input=True,
            touches_output=True,
            memory_level="L1",
        ),
        # Inner loops (RF level - systolic computation)
        LoopLevel(
            variable=LoopVariable.N_INTRA,
            bound=min(Tn, N),
            touches_output=True,
            memory_level="RF",
        ),
        LoopLevel(
            variable=LoopVariable.K_INTRA,
            bound=min(Tk, K),
            touches_input=True,
            touches_weight=True,
            memory_level="RF",
        ),
        LoopLevel(
            variable=LoopVariable.M_INTRA,
            bound=min(Tm, M),
            touches_input=True,
            touches_output=True,
            memory_level="RF",
        ),
    ]

    return LoopNest(
        loops=loops,
        dataflow_name="weight_stationary",
        input_access_pattern=f"A[tm*{Tm}:(tm+1)*{Tm}, tk*{Tk}:(tk+1)*{Tk}]",
        weight_access_pattern=f"B[tk*{Tk}:(tk+1)*{Tk}, tn*{Tn}:(tn+1)*{Tn}]",
        output_access_pattern=f"C[tm*{Tm}:(tm+1)*{Tm}, tn*{Tn}:(tn+1)*{Tn}]",
    )


def generate_output_stationary_loop_nest(
    M: int,
    K: int,
    N: int,
    Tm: int,
    Tk: int,
    Tn: int,
) -> LoopNest:
    """
    Generate output-stationary loop nest.

    Output-stationary keeps partial sums in accumulators,
    streaming both inputs and weights through.

    Pseudocode:
    ```
    for m_tile in [0, ceil(M/Tm)):
        for n_tile in [0, ceil(N/Tn)):
            init C[m_tile, n_tile] = 0
            for k_tile in [0, ceil(K/Tk)):
                load_inputs(A[m_tile, k_tile])
                load_weights(B[k_tile, n_tile])
                C[m_tile, n_tile] += A[m_tile, k_tile] @ B[k_tile, n_tile]
            store C[m_tile, n_tile]
    ```

    Args:
        M, K, N: Matrix dimensions
        Tm, Tk, Tn: Tile sizes

    Returns:
        LoopNest
    """
    import math

    num_m_tiles = math.ceil(M / Tm)
    num_k_tiles = math.ceil(K / Tk)
    num_n_tiles = math.ceil(N / Tn)

    loops = [
        # Outer loops - output tiles (outputs stay in accumulators)
        LoopLevel(
            variable=LoopVariable.TM,
            bound=num_m_tiles,
            touches_input=True,
            touches_output=True,
            memory_level="DRAM",
        ),
        LoopLevel(
            variable=LoopVariable.TN,
            bound=num_n_tiles,
            touches_weight=True,
            touches_output=True,
            memory_level="DRAM",
        ),
        # K loop - reduction (outputs stay in place)
        LoopLevel(
            variable=LoopVariable.TK,
            bound=num_k_tiles,
            touches_input=True,
            touches_weight=True,
            memory_level="L1",
        ),
        # Inner computation loops
        LoopLevel(
            variable=LoopVariable.M_INTRA,
            bound=min(Tm, M),
            touches_input=True,
            touches_output=True,
            memory_level="RF",
        ),
        LoopLevel(
            variable=LoopVariable.N_INTRA,
            bound=min(Tn, N),
            touches_weight=True,
            touches_output=True,
            memory_level="RF",
        ),
        LoopLevel(
            variable=LoopVariable.K_INTRA,
            bound=min(Tk, K),
            touches_input=True,
            touches_weight=True,
            memory_level="RF",
        ),
    ]

    return LoopNest(
        loops=loops,
        dataflow_name="output_stationary",
        input_access_pattern=f"A[tm*{Tm}:(tm+1)*{Tm}, tk*{Tk}:(tk+1)*{Tk}]",
        weight_access_pattern=f"B[tk*{Tk}:(tk+1)*{Tk}, tn*{Tn}:(tn+1)*{Tn}]",
        output_access_pattern=f"C[tm*{Tm}:(tm+1)*{Tm}, tn*{Tn}:(tn+1)*{Tn}]",
    )


def generate_row_stationary_loop_nest(
    M: int,
    K: int,
    N: int,
    Tm: int,
    Tk: int,
    Tn: int,
) -> LoopNest:
    """
    Generate row-stationary loop nest (Eyeriss-style).

    Row-stationary balances reuse across all three operands
    by processing rows of the computation together.

    Each PE processes a row of C, accumulating partial sums locally
    while sharing activations horizontally and weights vertically.

    Pseudocode:
    ```
    for m_tile in [0, ceil(M/Tm)):
        for k_tile in [0, ceil(K/Tk)):
            for n_tile in [0, ceil(N/Tn)):
                # Each PE row handles one row of output
                # Activations shared horizontally
                # Weights shared vertically
                for m_i in [0, Tm):
                    for k_i in [0, Tk):
                        for n_i in [0, Tn):
                            C[m_tile*Tm+m_i, n_tile*Tn+n_i] +=
                                A[m_tile*Tm+m_i, k_tile*Tk+k_i] *
                                B[k_tile*Tk+k_i, n_tile*Tn+n_i]
    ```

    Args:
        M, K, N: Matrix dimensions
        Tm, Tk, Tn: Tile sizes

    Returns:
        LoopNest
    """
    import math

    num_m_tiles = math.ceil(M / Tm)
    num_k_tiles = math.ceil(K / Tk)
    num_n_tiles = math.ceil(N / Tn)

    loops = [
        # Outer loops - balanced tiling
        LoopLevel(
            variable=LoopVariable.TM,
            bound=num_m_tiles,
            touches_input=True,
            touches_output=True,
            memory_level="DRAM",
        ),
        LoopLevel(
            variable=LoopVariable.TK,
            bound=num_k_tiles,
            touches_input=True,
            touches_weight=True,
            memory_level="DRAM",
        ),
        LoopLevel(
            variable=LoopVariable.TN,
            bound=num_n_tiles,
            touches_weight=True,
            touches_output=True,
            memory_level="L1",
        ),
        # Inner computation - row-wise processing
        LoopLevel(
            variable=LoopVariable.M_INTRA,
            bound=min(Tm, M),
            touches_input=True,
            touches_output=True,
            memory_level="RF",
        ),
        LoopLevel(
            variable=LoopVariable.K_INTRA,
            bound=min(Tk, K),
            touches_input=True,
            touches_weight=True,
            memory_level="RF",
        ),
        LoopLevel(
            variable=LoopVariable.N_INTRA,
            bound=min(Tn, N),
            touches_weight=True,
            touches_output=True,
            memory_level="RF",
        ),
    ]

    return LoopNest(
        loops=loops,
        dataflow_name="row_stationary",
        input_access_pattern=f"A[tm*{Tm}:(tm+1)*{Tm}, tk*{Tk}:(tk+1)*{Tk}]",
        weight_access_pattern=f"B[tk*{Tk}:(tk+1)*{Tk}, tn*{Tn}:(tn+1)*{Tn}]",
        output_access_pattern=f"C[tm*{Tm}:(tm+1)*{Tm}, tn*{Tn}:(tn+1)*{Tn}]",
    )


def compare_dataflow_loop_nests(
    M: int,
    K: int,
    N: int,
    array_size: int,
) -> Tuple[LoopNest, LoopNest, LoopNest]:
    """
    Generate and compare loop nests for all three dataflows.

    Args:
        M, K, N: Matrix dimensions
        array_size: Systolic array dimension (rows = cols = array_size)

    Returns:
        Tuple of (weight_stationary, output_stationary, row_stationary) loop nests
    """
    Tm = min(M, array_size)
    Tk = K  # Full K for simplicity
    Tn = min(N, array_size)

    ws = generate_weight_stationary_loop_nest(M, K, N, Tm, Tk, Tn)
    os = generate_output_stationary_loop_nest(M, K, N, Tm, Tk, Tn)
    rs = generate_row_stationary_loop_nest(M, K, N, Tm, Tk, Tn)

    return ws, os, rs


def print_loop_nest_comparison(
    M: int,
    K: int,
    N: int,
    array_size: int,
) -> None:
    """
    Print comparison of loop nests for all dataflows.

    Args:
        M, K, N: Matrix dimensions
        array_size: Systolic array dimension
    """
    ws, os, rs = compare_dataflow_loop_nests(M, K, N, array_size)

    print(f"Matrix dimensions: M={M}, K={K}, N={N}")
    print(f"Array size: {array_size}x{array_size}")
    print()

    for nest in [ws, os, rs]:
        print("=" * 60)
        print(nest)
        print()
        print("Pseudocode:")
        print(nest.to_pseudocode())
        print()
