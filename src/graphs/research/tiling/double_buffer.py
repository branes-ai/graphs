"""
Double-Buffering State Machine

Implements double-buffering for tile execution with prefetch/compute overlap.
Models the execution timeline including stalls when prefetch is not ready.

Key concepts:
- Buffer: Single buffer holding tile data
- DoubleBuffer: Pair of buffers for ping-pong operation
- BufferState: Current state of buffer (empty, loading, ready, computing)
- DoubleBufferScheduler: Schedule execution with prefetch overlap
- ExecutionTimeline: Complete timeline of operations

Double-buffering principle:
    While computing on buffer A, prefetch next tile into buffer B.
    Swap buffers between iterations.
    Stall if prefetch not ready when compute completes.

    Timeline (ideal):
    |--prefetch[0]--|--compute[0]--|--compute[1]--|--compute[2]--|
                    |--prefetch[1]--|--prefetch[2]--|--prefetch[3]--|

    Timeline (with stall):
    |--prefetch[0]--|--compute[0]--|--stall--|--compute[1]--|
                    |----prefetch[1]--(slow)---|--prefetch[2]--|
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Iterator
from math import ceil


class BufferState(Enum):
    """State of a single buffer."""
    EMPTY = "empty"           # No data
    LOADING = "loading"       # Prefetch in progress
    READY = "ready"           # Data loaded, ready for compute
    COMPUTING = "computing"   # Currently being used for compute
    DIRTY = "dirty"           # Contains result, needs writeback


@dataclass
class Buffer:
    """
    Single buffer in double-buffer pair.

    Tracks state, contents, and timing.
    """
    buffer_id: int           # 0 or 1
    capacity_bytes: int
    state: BufferState = BufferState.EMPTY

    # Current contents
    tile_id: Optional[str] = None
    data_bytes: int = 0

    # Timing
    load_start_cycle: int = 0
    load_end_cycle: int = 0
    compute_start_cycle: int = 0
    compute_end_cycle: int = 0

    def start_load(self, tile_id: str, data_bytes: int, cycle: int, duration: int):
        """Begin loading data into buffer."""
        self.state = BufferState.LOADING
        self.tile_id = tile_id
        self.data_bytes = data_bytes
        self.load_start_cycle = cycle
        self.load_end_cycle = cycle + duration

    def finish_load(self, cycle: int):
        """Complete load operation."""
        if cycle >= self.load_end_cycle:
            self.state = BufferState.READY

    def start_compute(self, cycle: int, duration: int):
        """Begin compute using this buffer."""
        self.state = BufferState.COMPUTING
        self.compute_start_cycle = cycle
        self.compute_end_cycle = cycle + duration

    def finish_compute(self, cycle: int):
        """Complete compute operation."""
        if cycle >= self.compute_end_cycle:
            self.state = BufferState.DIRTY

    def clear(self):
        """Clear buffer contents."""
        self.state = BufferState.EMPTY
        self.tile_id = None
        self.data_bytes = 0

    def is_available(self) -> bool:
        """Check if buffer can accept new data."""
        return self.state in {BufferState.EMPTY, BufferState.DIRTY}

    def is_ready_for_compute(self) -> bool:
        """Check if buffer is ready for computation."""
        return self.state == BufferState.READY


@dataclass
class DoubleBuffer:
    """
    Double-buffer pair for ping-pong operation.

    Manages two buffers, alternating between load and compute.
    """
    capacity_bytes: int        # Total capacity (split between buffers)
    buffer_a: Buffer = None
    buffer_b: Buffer = None

    # Which buffer is active for compute
    compute_buffer: int = 0    # 0 = A, 1 = B
    prefetch_buffer: int = 1   # 0 = A, 1 = B

    def __post_init__(self):
        per_buffer = self.capacity_bytes // 2
        self.buffer_a = Buffer(buffer_id=0, capacity_bytes=per_buffer)
        self.buffer_b = Buffer(buffer_id=1, capacity_bytes=per_buffer)

    def get_compute_buffer(self) -> Buffer:
        """Get buffer currently used for compute."""
        return self.buffer_a if self.compute_buffer == 0 else self.buffer_b

    def get_prefetch_buffer(self) -> Buffer:
        """Get buffer currently used for prefetch."""
        return self.buffer_a if self.prefetch_buffer == 0 else self.buffer_b

    def swap_buffers(self):
        """Swap compute and prefetch buffers."""
        self.compute_buffer, self.prefetch_buffer = (
            self.prefetch_buffer, self.compute_buffer
        )

    def can_prefetch(self) -> bool:
        """Check if prefetch buffer can accept new data."""
        return self.get_prefetch_buffer().is_available()

    def can_compute(self) -> bool:
        """Check if compute buffer has data ready."""
        return self.get_compute_buffer().is_ready_for_compute()


@dataclass
class TimelineEvent:
    """
    Single event in execution timeline.

    Events are: prefetch_start, prefetch_end, compute_start, compute_end, stall, swap
    """
    cycle: int
    event_type: str
    buffer_id: int
    tile_id: Optional[str] = None
    duration_cycles: int = 0
    description: str = ""

    def __repr__(self):
        return f"[{self.cycle:6d}] {self.event_type:15s} buf={self.buffer_id} {self.tile_id or ''}"


@dataclass
class ExecutionTimeline:
    """
    Complete execution timeline for double-buffered tile execution.

    Tracks all events and computes metrics.
    """
    events: List[TimelineEvent] = field(default_factory=list)

    # Summary metrics
    total_cycles: int = 0
    compute_cycles: int = 0
    prefetch_cycles: int = 0
    stall_cycles: int = 0
    overlap_cycles: int = 0

    def add_event(self, event: TimelineEvent):
        """Add event to timeline."""
        self.events.append(event)

    def add_prefetch_start(self, cycle: int, buffer_id: int, tile_id: str, duration: int):
        self.add_event(TimelineEvent(
            cycle=cycle,
            event_type="prefetch_start",
            buffer_id=buffer_id,
            tile_id=tile_id,
            duration_cycles=duration,
        ))

    def add_prefetch_end(self, cycle: int, buffer_id: int, tile_id: str):
        self.add_event(TimelineEvent(
            cycle=cycle,
            event_type="prefetch_end",
            buffer_id=buffer_id,
            tile_id=tile_id,
        ))

    def add_compute_start(self, cycle: int, buffer_id: int, tile_id: str, duration: int):
        self.add_event(TimelineEvent(
            cycle=cycle,
            event_type="compute_start",
            buffer_id=buffer_id,
            tile_id=tile_id,
            duration_cycles=duration,
        ))

    def add_compute_end(self, cycle: int, buffer_id: int, tile_id: str):
        self.add_event(TimelineEvent(
            cycle=cycle,
            event_type="compute_end",
            buffer_id=buffer_id,
            tile_id=tile_id,
        ))

    def add_stall(self, cycle: int, duration: int, reason: str = ""):
        self.add_event(TimelineEvent(
            cycle=cycle,
            event_type="stall",
            buffer_id=-1,
            duration_cycles=duration,
            description=reason,
        ))
        self.stall_cycles += duration

    def add_swap(self, cycle: int):
        self.add_event(TimelineEvent(
            cycle=cycle,
            event_type="swap",
            buffer_id=-1,
        ))

    def compute_metrics(self):
        """Compute summary metrics from events."""
        if not self.events:
            return

        self.total_cycles = max(e.cycle for e in self.events)

        for event in self.events:
            if event.event_type == "compute_start":
                self.compute_cycles += event.duration_cycles
            elif event.event_type == "prefetch_start":
                self.prefetch_cycles += event.duration_cycles

        # Overlap is when prefetch happens during compute
        # overlap = prefetch_cycles - stall_cycles (approximately)
        self.overlap_cycles = max(0, self.prefetch_cycles - self.stall_cycles)

    @property
    def efficiency(self) -> float:
        """Compute efficiency (compute / total)."""
        if self.total_cycles == 0:
            return 0.0
        return self.compute_cycles / self.total_cycles

    @property
    def prefetch_overlap_ratio(self) -> float:
        """Ratio of prefetch that overlaps with compute."""
        if self.prefetch_cycles == 0:
            return 0.0
        return self.overlap_cycles / self.prefetch_cycles

    def summary(self) -> Dict:
        """Generate summary dictionary."""
        self.compute_metrics()
        return {
            'total_cycles': self.total_cycles,
            'compute_cycles': self.compute_cycles,
            'prefetch_cycles': self.prefetch_cycles,
            'stall_cycles': self.stall_cycles,
            'overlap_cycles': self.overlap_cycles,
            'efficiency': self.efficiency,
            'prefetch_overlap_ratio': self.prefetch_overlap_ratio,
            'num_events': len(self.events),
        }

    def print_timeline(self, max_events: int = 50):
        """Print timeline events."""
        print(f"Execution Timeline ({len(self.events)} events)")
        print("-" * 60)
        for event in self.events[:max_events]:
            print(event)
        if len(self.events) > max_events:
            print(f"... and {len(self.events) - max_events} more events")
        print("-" * 60)
        print(f"Total: {self.total_cycles} cycles, "
              f"Compute: {self.compute_cycles}, "
              f"Stall: {self.stall_cycles}")


@dataclass
class TileDescriptor:
    """Description of a tile to be processed."""
    tile_id: str
    input_bytes: int      # Bytes to prefetch
    output_bytes: int     # Bytes to writeback
    compute_cycles: int   # Cycles to compute


class DoubleBufferScheduler:
    """
    Schedule tile execution with double-buffering.

    Generates execution timeline with prefetch/compute overlap.
    """

    def __init__(
        self,
        buffer_capacity_bytes: int,
        prefetch_bandwidth_bytes_per_cycle: float,
        writeback_bandwidth_bytes_per_cycle: float,
    ):
        """
        Initialize scheduler.

        Args:
            buffer_capacity_bytes: Total capacity for double-buffering
            prefetch_bandwidth_bytes_per_cycle: Memory bandwidth for prefetch
            writeback_bandwidth_bytes_per_cycle: Memory bandwidth for writeback
        """
        self.buffer_capacity = buffer_capacity_bytes
        self.prefetch_bw = prefetch_bandwidth_bytes_per_cycle
        self.writeback_bw = writeback_bandwidth_bytes_per_cycle

        self.double_buffer = DoubleBuffer(capacity_bytes=buffer_capacity_bytes)

    def prefetch_latency(self, bytes_to_load: int) -> int:
        """Calculate prefetch latency in cycles."""
        return ceil(bytes_to_load / self.prefetch_bw)

    def writeback_latency(self, bytes_to_store: int) -> int:
        """Calculate writeback latency in cycles."""
        return ceil(bytes_to_store / self.writeback_bw)

    def schedule(self, tiles: List[TileDescriptor]) -> ExecutionTimeline:
        """
        Generate execution timeline for list of tiles.

        Args:
            tiles: List of tiles to process in order

        Returns:
            ExecutionTimeline with all events
        """
        timeline = ExecutionTimeline()
        db = self.double_buffer

        if not tiles:
            return timeline

        current_cycle = 0

        # Initial prefetch (no overlap possible)
        first_tile = tiles[0]
        prefetch_cycles = self.prefetch_latency(first_tile.input_bytes)

        db.get_prefetch_buffer().start_load(
            first_tile.tile_id,
            first_tile.input_bytes,
            current_cycle,
            prefetch_cycles
        )
        timeline.add_prefetch_start(
            current_cycle, db.prefetch_buffer,
            first_tile.tile_id, prefetch_cycles
        )

        current_cycle += prefetch_cycles
        db.get_prefetch_buffer().finish_load(current_cycle)
        timeline.add_prefetch_end(current_cycle, db.prefetch_buffer, first_tile.tile_id)

        # Swap so prefetched data is in compute buffer
        db.swap_buffers()
        timeline.add_swap(current_cycle)

        # Process remaining tiles
        for i, tile in enumerate(tiles):
            compute_buffer = db.get_compute_buffer()
            prefetch_buffer = db.get_prefetch_buffer()

            # Start compute on current tile
            compute_start = current_cycle
            compute_duration = tile.compute_cycles
            compute_buffer.start_compute(compute_start, compute_duration)
            timeline.add_compute_start(
                compute_start, db.compute_buffer,
                tile.tile_id, compute_duration
            )

            # Start prefetch of next tile (if any)
            next_tile_idx = i + 1
            prefetch_end = compute_start  # Default: no prefetch

            if next_tile_idx < len(tiles):
                next_tile = tiles[next_tile_idx]
                prefetch_duration = self.prefetch_latency(next_tile.input_bytes)

                prefetch_buffer.start_load(
                    next_tile.tile_id,
                    next_tile.input_bytes,
                    compute_start,
                    prefetch_duration
                )
                timeline.add_prefetch_start(
                    compute_start, db.prefetch_buffer,
                    next_tile.tile_id, prefetch_duration
                )
                prefetch_end = compute_start + prefetch_duration

            # Wait for compute to complete
            compute_end = compute_start + compute_duration
            compute_buffer.finish_compute(compute_end)
            timeline.add_compute_end(compute_end, db.compute_buffer, tile.tile_id)

            # Check if prefetch is ready
            if next_tile_idx < len(tiles):
                if prefetch_end > compute_end:
                    # Stall waiting for prefetch
                    stall_cycles = prefetch_end - compute_end
                    timeline.add_stall(compute_end, stall_cycles, "prefetch_not_ready")
                    current_cycle = prefetch_end
                else:
                    current_cycle = compute_end

                prefetch_buffer.finish_load(current_cycle)
                timeline.add_prefetch_end(
                    current_cycle, db.prefetch_buffer, tiles[next_tile_idx].tile_id
                )

                # Swap buffers for next iteration
                db.swap_buffers()
                timeline.add_swap(current_cycle)

                # Clear old compute buffer (now prefetch buffer)
                db.get_prefetch_buffer().clear()
            else:
                current_cycle = compute_end

        timeline.compute_metrics()
        return timeline

    def schedule_with_writeback(
        self,
        tiles: List[TileDescriptor],
        writeback_every: int = 1,
    ) -> ExecutionTimeline:
        """
        Schedule with explicit writeback of output tiles.

        Args:
            tiles: Tiles to process
            writeback_every: Writeback after every N tiles

        Returns:
            ExecutionTimeline including writeback events
        """
        # For simplicity, model writeback as overlapped with next compute
        # This is a simplified model - full model would track output buffer
        timeline = self.schedule(tiles)

        # Add writeback overhead (simplified)
        total_writeback_bytes = sum(t.output_bytes for t in tiles)
        writeback_cycles = self.writeback_latency(total_writeback_bytes)

        # Assume writeback overlaps with compute, add non-overlapped portion
        non_overlap_writeback = max(0, writeback_cycles - timeline.compute_cycles)
        timeline.total_cycles += non_overlap_writeback

        return timeline


@dataclass
class TripleBufferScheduler:
    """
    Triple-buffering for higher overlap.

    Uses three buffers:
    - Buffer A: Computing
    - Buffer B: Prefetching next
    - Buffer C: Prefetching next+1

    Reduces stalls when prefetch >> compute.
    """
    buffer_capacity_bytes: int
    prefetch_bandwidth_bytes_per_cycle: float
    writeback_bandwidth_bytes_per_cycle: float

    def __post_init__(self):
        per_buffer = self.buffer_capacity_bytes // 3
        self.buffers = [
            Buffer(buffer_id=i, capacity_bytes=per_buffer)
            for i in range(3)
        ]
        self.compute_idx = 0
        self.prefetch_idx = 1
        self.prefetch2_idx = 2

    def prefetch_latency(self, bytes_to_load: int) -> int:
        return ceil(bytes_to_load / self.prefetch_bandwidth_bytes_per_cycle)

    def schedule(self, tiles: List[TileDescriptor]) -> ExecutionTimeline:
        """
        Schedule with triple buffering.

        Provides better overlap when prefetch latency > compute latency.
        """
        timeline = ExecutionTimeline()

        if not tiles:
            return timeline

        current_cycle = 0

        # Prefetch first two tiles
        for i in range(min(2, len(tiles))):
            tile = tiles[i]
            prefetch_cycles = self.prefetch_latency(tile.input_bytes)

            buffer_idx = (self.prefetch_idx + i) % 3
            self.buffers[buffer_idx].start_load(
                tile.tile_id, tile.input_bytes,
                current_cycle, prefetch_cycles
            )
            timeline.add_prefetch_start(
                current_cycle, buffer_idx, tile.tile_id, prefetch_cycles
            )

        # Wait for first prefetch
        first_prefetch_end = (
            current_cycle +
            self.prefetch_latency(tiles[0].input_bytes)
        )
        current_cycle = first_prefetch_end

        # Rotate buffers: prefetch -> compute
        self.compute_idx = self.prefetch_idx
        self.prefetch_idx = self.prefetch2_idx
        self.prefetch2_idx = (self.prefetch2_idx + 1) % 3

        # Process tiles
        for i, tile in enumerate(tiles):
            compute_start = current_cycle
            compute_duration = tile.compute_cycles

            timeline.add_compute_start(
                compute_start, self.compute_idx,
                tile.tile_id, compute_duration
            )

            # Start prefetch of tile i+2 (if exists)
            if i + 2 < len(tiles):
                next_tile = tiles[i + 2]
                prefetch_cycles = self.prefetch_latency(next_tile.input_bytes)
                self.buffers[self.prefetch2_idx].start_load(
                    next_tile.tile_id, next_tile.input_bytes,
                    compute_start, prefetch_cycles
                )
                timeline.add_prefetch_start(
                    compute_start, self.prefetch2_idx,
                    next_tile.tile_id, prefetch_cycles
                )

            compute_end = compute_start + compute_duration
            timeline.add_compute_end(compute_end, self.compute_idx, tile.tile_id)

            # Check if next prefetch is ready (tile i+1)
            if i + 1 < len(tiles):
                prefetch_ready = (
                    first_prefetch_end +
                    i * self.prefetch_latency(tiles[min(i+1, len(tiles)-1)].input_bytes)
                )
                if prefetch_ready > compute_end:
                    stall = prefetch_ready - compute_end
                    timeline.add_stall(compute_end, stall, "prefetch_not_ready")
                    current_cycle = prefetch_ready
                else:
                    current_cycle = compute_end

                # Rotate buffers
                old_compute = self.compute_idx
                self.compute_idx = self.prefetch_idx
                self.prefetch_idx = self.prefetch2_idx
                self.prefetch2_idx = old_compute
                self.buffers[self.prefetch2_idx].clear()
            else:
                current_cycle = compute_end

        timeline.compute_metrics()
        return timeline


def analyze_double_buffer_benefit(
    tiles: List[TileDescriptor],
    prefetch_bw: float,
    buffer_bytes: int,
) -> Dict:
    """
    Analyze benefit of double-buffering vs no buffering.

    Returns comparison metrics.
    """
    # No buffering: sequential prefetch then compute
    no_buffer_cycles = 0
    for tile in tiles:
        no_buffer_cycles += ceil(tile.input_bytes / prefetch_bw)
        no_buffer_cycles += tile.compute_cycles

    # Double buffering
    scheduler = DoubleBufferScheduler(
        buffer_capacity_bytes=buffer_bytes,
        prefetch_bandwidth_bytes_per_cycle=prefetch_bw,
        writeback_bandwidth_bytes_per_cycle=prefetch_bw,
    )
    timeline = scheduler.schedule(tiles)

    return {
        'no_buffer_cycles': no_buffer_cycles,
        'double_buffer_cycles': timeline.total_cycles,
        'speedup': no_buffer_cycles / timeline.total_cycles if timeline.total_cycles > 0 else 1.0,
        'stall_cycles': timeline.stall_cycles,
        'efficiency': timeline.efficiency,
        'overlap_ratio': timeline.prefetch_overlap_ratio,
    }


def create_tiles_from_schedule(
    M: int, K: int, N: int,
    Tm: int, Tk: int, Tn: int,
    compute_cycles_per_mac: float = 1.0,
    dtype_bytes: int = 2,
) -> List[TileDescriptor]:
    """
    Create tile descriptors from matrix multiply parameters.

    Args:
        M, K, N: Problem dimensions
        Tm, Tk, Tn: Tile dimensions
        compute_cycles_per_mac: Cycles per MAC operation
        dtype_bytes: Element size

    Returns:
        List of TileDescriptor for all tiles in schedule
    """
    tiles = []

    num_m = ceil(M / Tm)
    num_k = ceil(K / Tk)
    num_n = ceil(N / Tn)

    tile_idx = 0
    for im in range(num_m):
        for ik in range(num_k):
            for jn in range(num_n):
                # Actual tile dimensions (handle edges)
                actual_tm = min(Tm, M - im * Tm)
                actual_tk = min(Tk, K - ik * Tk)
                actual_tn = min(Tn, N - jn * Tn)

                # Input: A tile (Tm x Tk) + B tile (Tk x Tn)
                input_bytes = (
                    actual_tm * actual_tk * dtype_bytes +
                    actual_tk * actual_tn * dtype_bytes
                )

                # Output: C tile (Tm x Tn) - FP32 accumulator
                output_bytes = actual_tm * actual_tn * 4

                # Compute: 2 * Tm * Tk * Tn MACs
                macs = actual_tm * actual_tk * actual_tn
                compute_cycles = int(macs * compute_cycles_per_mac)

                tiles.append(TileDescriptor(
                    tile_id=f"tile[{im},{ik},{jn}]",
                    input_bytes=input_bytes,
                    output_bytes=output_bytes,
                    compute_cycles=compute_cycles,
                ))
                tile_idx += 1

    return tiles
