"""
Loop Nest Representation

Explicit loop nest representation for dataflow analysis.
"""

from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum


class LoopVariable(Enum):
    """Loop iteration variable types."""
    M = "m"           # Output row iteration
    K = "k"           # Reduction dimension
    N = "n"           # Output column iteration
    TM = "tm"         # M tile iteration
    TK = "tk"         # K tile iteration
    TN = "tn"         # N tile iteration
    M_INTRA = "m_i"   # Intra-tile M iteration
    K_INTRA = "k_i"   # Intra-tile K iteration
    N_INTRA = "n_i"   # Intra-tile N iteration


@dataclass
class LoopLevel:
    """
    Single loop level in a loop nest.

    Represents one loop in the tiled computation with
    metadata about which operands it touches.
    """
    variable: LoopVariable
    bound: int          # Trip count (number of iterations)
    stride: int = 1     # Step size (usually 1 for tile loops)

    # Which operands are accessed at this loop level
    touches_input: bool = False     # A matrix
    touches_weight: bool = False    # B matrix
    touches_output: bool = False    # C matrix

    # Memory level this loop operates at
    memory_level: str = "DRAM"  # DRAM, L2, L1, RF

    @property
    def is_tile_loop(self) -> bool:
        """Check if this is a tile-level loop."""
        return self.variable in {LoopVariable.TM, LoopVariable.TK, LoopVariable.TN}

    @property
    def is_intra_tile_loop(self) -> bool:
        """Check if this is an intra-tile loop."""
        return self.variable in {LoopVariable.M_INTRA, LoopVariable.K_INTRA, LoopVariable.N_INTRA}

    def __str__(self) -> str:
        touch_str = ""
        if self.touches_input:
            touch_str += "A"
        if self.touches_weight:
            touch_str += "B"
        if self.touches_output:
            touch_str += "C"
        return f"for {self.variable.value} in [0, {self.bound}): [{touch_str}] @{self.memory_level}"


@dataclass
class LoopNest:
    """
    Explicit loop nest with trip counts and orderings.

    Represents the complete loop structure for a tiled matrix multiply
    with annotations for data movement analysis.
    """
    # Loop levels from outer to inner
    loops: List[LoopLevel] = field(default_factory=list)

    # Total iterations (product of all bounds)
    total_iterations: int = 0

    # Memory access patterns
    input_access_pattern: str = ""   # e.g., "A[m_tile, k_tile]"
    weight_access_pattern: str = ""  # e.g., "B[k_tile, n_tile]"
    output_access_pattern: str = ""  # e.g., "C[m_tile, n_tile]"

    # Dataflow name
    dataflow_name: str = ""

    def __post_init__(self):
        if self.loops and self.total_iterations == 0:
            self.total_iterations = 1
            for loop in self.loops:
                self.total_iterations *= loop.bound

    def add_loop(self, loop: LoopLevel) -> None:
        """Add a loop level (appends as innermost)."""
        self.loops.append(loop)
        self.total_iterations = 1
        for l in self.loops:
            self.total_iterations *= l.bound

    def get_reuse_distance(self, operand: str) -> int:
        """
        Calculate reuse distance for an operand.

        Reuse distance = iterations between consecutive accesses
        to the same element.

        Args:
            operand: 'input', 'weight', or 'output'

        Returns:
            Reuse distance in iterations
        """
        if operand == 'input':
            touch_attr = 'touches_input'
        elif operand == 'weight':
            touch_attr = 'touches_weight'
        elif operand == 'output':
            touch_attr = 'touches_output'
        else:
            return 0

        # Find innermost loop that touches this operand
        distance = 1
        found_touch = False

        for loop in reversed(self.loops):
            if getattr(loop, touch_attr):
                found_touch = True
                break
            distance *= loop.bound

        return distance if found_touch else self.total_iterations

    def to_pseudocode(self) -> str:
        """Generate pseudocode representation of loop nest."""
        lines = []
        indent = ""

        for loop in self.loops:
            lines.append(f"{indent}for {loop.variable.value} in range({loop.bound}):")
            indent += "    "

        # Add innermost computation
        lines.append(f"{indent}C[m, n] += A[m, k] * B[k, n]")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            'dataflow_name': self.dataflow_name,
            'total_iterations': self.total_iterations,
            'num_loops': len(self.loops),
            'loops': [
                {
                    'variable': loop.variable.value,
                    'bound': loop.bound,
                    'stride': loop.stride,
                    'touches_input': loop.touches_input,
                    'touches_weight': loop.touches_weight,
                    'touches_output': loop.touches_output,
                    'memory_level': loop.memory_level,
                }
                for loop in self.loops
            ],
            'input_access_pattern': self.input_access_pattern,
            'weight_access_pattern': self.weight_access_pattern,
            'output_access_pattern': self.output_access_pattern,
        }

    def __str__(self) -> str:
        lines = [f"Loop Nest: {self.dataflow_name}"]
        lines.append(f"Total iterations: {self.total_iterations}")
        lines.append("Loops (outer to inner):")
        for i, loop in enumerate(self.loops):
            lines.append(f"  {i}: {loop}")
        return "\n".join(lines)


def create_loop_nest_from_schedule(
    schedule: 'TileSchedule',
) -> LoopNest:
    """
    Create loop nest from a TileSchedule.

    Args:
        schedule: TileSchedule with tiling parameters

    Returns:
        LoopNest with complete loop structure
    """
    from graphs.research.dataflow.tiling import DataflowType

    loops = []

    if schedule.dataflow == DataflowType.WEIGHT_STATIONARY:
        # Weight-stationary: N -> K -> M (outer to inner for tiles)
        loops.append(LoopLevel(
            variable=LoopVariable.TN,
            bound=schedule.num_n_tiles,
            touches_weight=True,
            touches_output=True,
            memory_level="DRAM",
        ))
        loops.append(LoopLevel(
            variable=LoopVariable.TK,
            bound=schedule.num_k_tiles,
            touches_input=True,
            touches_weight=True,
            memory_level="DRAM",
        ))
        loops.append(LoopLevel(
            variable=LoopVariable.TM,
            bound=schedule.num_m_tiles,
            touches_input=True,
            touches_output=True,
            memory_level="L1",
        ))

        # Intra-tile loops
        loops.append(LoopLevel(
            variable=LoopVariable.N_INTRA,
            bound=schedule.Tn,
            touches_output=True,
            memory_level="RF",
        ))
        loops.append(LoopLevel(
            variable=LoopVariable.K_INTRA,
            bound=schedule.Tk,
            touches_input=True,
            touches_weight=True,
            memory_level="RF",
        ))
        loops.append(LoopLevel(
            variable=LoopVariable.M_INTRA,
            bound=schedule.Tm,
            touches_input=True,
            touches_output=True,
            memory_level="RF",
        ))

        dataflow_name = "weight_stationary"

    elif schedule.dataflow == DataflowType.OUTPUT_STATIONARY:
        # Output-stationary: M -> N -> K (output stays in accumulators)
        loops.append(LoopLevel(
            variable=LoopVariable.TM,
            bound=schedule.num_m_tiles,
            touches_input=True,
            touches_output=True,
            memory_level="DRAM",
        ))
        loops.append(LoopLevel(
            variable=LoopVariable.TN,
            bound=schedule.num_n_tiles,
            touches_weight=True,
            touches_output=True,
            memory_level="DRAM",
        ))
        loops.append(LoopLevel(
            variable=LoopVariable.TK,
            bound=schedule.num_k_tiles,
            touches_input=True,
            touches_weight=True,
            memory_level="L1",
        ))

        # Intra-tile
        loops.append(LoopLevel(
            variable=LoopVariable.M_INTRA,
            bound=schedule.Tm,
            touches_input=True,
            touches_output=True,
            memory_level="RF",
        ))
        loops.append(LoopLevel(
            variable=LoopVariable.N_INTRA,
            bound=schedule.Tn,
            touches_weight=True,
            touches_output=True,
            memory_level="RF",
        ))
        loops.append(LoopLevel(
            variable=LoopVariable.K_INTRA,
            bound=schedule.Tk,
            touches_input=True,
            touches_weight=True,
            memory_level="RF",
        ))

        dataflow_name = "output_stationary"

    else:  # ROW_STATIONARY
        # Row-stationary: balanced
        loops.append(LoopLevel(
            variable=LoopVariable.TM,
            bound=schedule.num_m_tiles,
            touches_input=True,
            touches_output=True,
            memory_level="DRAM",
        ))
        loops.append(LoopLevel(
            variable=LoopVariable.TK,
            bound=schedule.num_k_tiles,
            touches_input=True,
            touches_weight=True,
            memory_level="DRAM",
        ))
        loops.append(LoopLevel(
            variable=LoopVariable.TN,
            bound=schedule.num_n_tiles,
            touches_weight=True,
            touches_output=True,
            memory_level="L1",
        ))

        # Intra-tile
        loops.append(LoopLevel(
            variable=LoopVariable.M_INTRA,
            bound=schedule.Tm,
            touches_input=True,
            touches_output=True,
            memory_level="RF",
        ))
        loops.append(LoopLevel(
            variable=LoopVariable.K_INTRA,
            bound=schedule.Tk,
            touches_input=True,
            touches_weight=True,
            memory_level="RF",
        ))
        loops.append(LoopLevel(
            variable=LoopVariable.N_INTRA,
            bound=schedule.Tn,
            touches_weight=True,
            touches_output=True,
            memory_level="RF",
        ))

        dataflow_name = "row_stationary"

    nest = LoopNest(
        loops=loops,
        dataflow_name=dataflow_name,
        input_access_pattern=f"A[tm*{schedule.Tm}:tm*{schedule.Tm}+{schedule.Tm}, tk*{schedule.Tk}:tk*{schedule.Tk}+{schedule.Tk}]",
        weight_access_pattern=f"B[tk*{schedule.Tk}:tk*{schedule.Tk}+{schedule.Tk}, tn*{schedule.Tn}:tn*{schedule.Tn}+{schedule.Tn}]",
        output_access_pattern=f"C[tm*{schedule.Tm}:tm*{schedule.Tm}+{schedule.Tm}, tn*{schedule.Tn}:tn*{schedule.Tn}+{schedule.Tn}]",
    )

    return nest
