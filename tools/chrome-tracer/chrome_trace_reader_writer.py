#!/usr/bin/env python3
"""
Chrome Trace Event Format Reader and Writer

This module provides classes to read and write JSON trace files compatible with
VizTracer and Perfetto, following the Chrome Trace Event Format specification.

The format supports various event types including:
- Duration events (B/E pairs or X complete events)
- Instant events (I)
- Counter events (C)
- Async events (b/n/e or S/T/F)
- Flow events (s/t/f)
- Memory dump events (v/V)
- Sample events (P)
"""

import json
import gzip
from typing import Dict, List, Any, Optional, Union, TextIO
from dataclasses import dataclass, field, asdict
from pathlib import Path
import time


@dataclass
class TraceEvent:
    """Represents a single trace event in Chrome Trace Event Format."""
    
    # Required fields
    name: str
    ph: str  # Phase: B, E, X, I, C, b, n, e, s, t, f, etc.
    ts: int  # Timestamp in microseconds
    pid: int  # Process ID
    tid: int  # Thread ID
    
    # Optional fields
    cat: Optional[str] = None  # Category
    dur: Optional[int] = None  # Duration in microseconds (for X events)
    args: Optional[Dict[str, Any]] = None  # Event arguments
    id: Optional[Union[str, int]] = None  # Event ID (for async events)
    scope: Optional[str] = None  # Scope for flow events
    bp: Optional[str] = None  # Binding point
    sf: Optional[int] = None  # Stack frame ID
    stack: Optional[List[str]] = None  # Stack trace
    tts: Optional[int] = None  # Thread timestamp
    use_async_tts: Optional[int] = None  # Use async thread timestamp
    
    def __post_init__(self):
        """Validate event data after initialization."""
        if self.args is None:
            self.args = {}
        
        # Validate phase types
        valid_phases = {
            'B', 'E', 'X',  # Duration events
            'I',             # Instant events
            'C',             # Counter events
            'b', 'n', 'e',  # Async events (nestable)
            'S', 'T', 'F',  # Async events (legacy)
            's', 't', 'f',  # Flow events
            'P',             # Sample events
            'M',             # Metadata events
            'N', 'D',        # Object events
            'O',             # Clock sync events
            'v', 'V',        # Memory dump events
        }
        
        if self.ph not in valid_phases:
            raise ValueError(f"Invalid phase '{self.ph}'. Must be one of {valid_phases}")
        
        # Validate duration events
        if self.ph == 'X' and self.dur is None:
            raise ValueError("Complete events (phase 'X') must have a duration")
        
        # Validate async events have IDs
        if self.ph in {'b', 'n', 'e', 'S', 'T', 'F'} and self.id is None:
            raise ValueError(f"Async events (phase '{self.ph}') must have an id")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary format for JSON serialization."""
        result = {
            'name': self.name,
            'ph': self.ph,
            'ts': self.ts,
            'pid': self.pid,
            'tid': self.tid,
        }
        
        # Add optional fields if present
        if self.cat is not None:
            result['cat'] = self.cat
        if self.dur is not None:
            result['dur'] = self.dur
        if self.args:
            result['args'] = self.args
        if self.id is not None:
            result['id'] = self.id
        if self.scope is not None:
            result['scope'] = self.scope
        if self.bp is not None:
            result['bp'] = self.bp
        if self.sf is not None:
            result['sf'] = self.sf
        if self.stack is not None:
            result['stack'] = self.stack
        if self.tts is not None:
            result['tts'] = self.tts
        if self.use_async_tts is not None:
            result['use_async_tts'] = self.use_async_tts
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TraceEvent':
        """Create TraceEvent from dictionary."""
        # Extract required fields
        required_fields = ['name', 'ph', 'ts', 'pid', 'tid']
        kwargs = {}
        
        for field_name in required_fields:
            if field_name not in data:
                raise ValueError(f"Missing required field: {field_name}")
            kwargs[field_name] = data[field_name]
        
        # Extract optional fields
        optional_fields = ['cat', 'dur', 'args', 'id', 'scope', 'bp', 'sf', 'stack', 'tts', 'use_async_tts']
        for field_name in optional_fields:
            if field_name in data:
                kwargs[field_name] = data[field_name]
        
        return cls(**kwargs)


@dataclass
class TraceMetadata:
    """Metadata for the trace file."""
    
    # Common metadata fields
    process_name: Optional[str] = None
    process_sort_index: Optional[int] = None
    thread_name: Optional[str] = None
    thread_sort_index: Optional[int] = None
    
    # Custom metadata
    custom: Dict[str, Any] = field(default_factory=dict)
    
    def to_event(self, pid: int, tid: int = 1) -> TraceEvent:
        """Convert metadata to a metadata event."""
        args = self.custom.copy()
        
        if self.process_name is not None:
            args['name'] = self.process_name
        if self.process_sort_index is not None:
            args['sort_index'] = self.process_sort_index
        if self.thread_name is not None:
            args['name'] = self.thread_name
        if self.thread_sort_index is not None:
            args['sort_index'] = self.thread_sort_index
        
        return TraceEvent(
            name='process_name' if self.process_name else 'thread_name',
            ph='M',  # Metadata event
            ts=0,
            pid=pid,
            tid=tid,
            args=args
        )


class ChromeTraceReader:
    """Reader for Chrome Trace Event Format JSON files."""
    
    def __init__(self):
        self.events: List[TraceEvent] = []
        self.metadata: Dict[str, Any] = {}
        self.stack_frames: Dict[int, Dict[str, Any]] = {}
        self.samples: List[Dict[str, Any]] = []
    
    def read_file(self, filepath: Union[str, Path]) -> None:
        """Read trace data from a file (supports .json and .json.gz)."""
        filepath = Path(filepath)
        
        if filepath.suffix == '.gz':
            with gzip.open(filepath, 'rt', encoding='utf-8') as f:
                self._read_from_file(f)
        else:
            with open(filepath, 'r', encoding='utf-8') as f:
                self._read_from_file(f)
    
    def read_string(self, json_string: str) -> None:
        """Read trace data from a JSON string."""
        data = json.loads(json_string)
        self._parse_trace_data(data)
    
    def _read_from_file(self, file_obj: TextIO) -> None:
        """Read and parse JSON from file object."""
        data = json.load(file_obj)
        self._parse_trace_data(data)
    
    def _parse_trace_data(self, data: Dict[str, Any]) -> None:
        """Parse the trace data structure."""
        self.events.clear()
        self.metadata.clear()
        self.stack_frames.clear()
        self.samples.clear()
        
        # Handle different JSON formats
        if 'traceEvents' in data:
            # Standard format with traceEvents array
            events_data = data['traceEvents']
            
            # Extract other top-level fields
            for key, value in data.items():
                if key != 'traceEvents':
                    self.metadata[key] = value
        elif isinstance(data, list):
            # Simple array format
            events_data = data
        else:
            raise ValueError("Invalid trace format: expected 'traceEvents' field or array")
        
        # Parse events
        for event_data in events_data:
            if not isinstance(event_data, dict):
                continue
            
            try:
                event = TraceEvent.from_dict(event_data)
                self.events.append(event)
            except (ValueError, KeyError) as e:
                # Skip invalid events but continue processing
                print(f"Warning: Skipping invalid event: {e}")
                continue
        
        # Extract stack frames if present
        if 'stackFrames' in self.metadata:
            self.stack_frames = self.metadata['stackFrames']
        
        # Extract samples if present
        if 'samples' in self.metadata:
            self.samples = self.metadata['samples']
    
    def get_events_by_phase(self, phase: str) -> List[TraceEvent]:
        """Get all events with a specific phase."""
        return [event for event in self.events if event.ph == phase]
    
    def get_events_by_name(self, name: str) -> List[TraceEvent]:
        """Get all events with a specific name."""
        return [event for event in self.events if event.name == name]
    
    def get_events_by_process(self, pid: int) -> List[TraceEvent]:
        """Get all events for a specific process."""
        return [event for event in self.events if event.pid == pid]
    
    def get_events_by_thread(self, pid: int, tid: int) -> List[TraceEvent]:
        """Get all events for a specific thread."""
        return [event for event in self.events if event.pid == pid and event.tid == tid]
    
    def get_duration_events(self) -> List[TraceEvent]:
        """Get all duration events (B, E, X phases)."""
        return [event for event in self.events if event.ph in {'B', 'E', 'X'}]
    
    def get_time_range(self) -> tuple[int, int]:
        """Get the time range of all events (min_ts, max_ts)."""
        if not self.events:
            return (0, 0)
        
        min_ts = min(event.ts for event in self.events)
        max_ts = max(event.ts + (event.dur or 0) for event in self.events)
        return (min_ts, max_ts)


class ChromeTraceWriter:
    """Writer for Chrome Trace Event Format JSON files."""
    
    def __init__(self):
        self.events: List[TraceEvent] = []
        self.metadata: Dict[str, Any] = {}
        self.stack_frames: Dict[int, Dict[str, Any]] = {}
        self.samples: List[Dict[str, Any]] = []
    
    def add_event(self, event: TraceEvent) -> None:
        """Add a trace event."""
        self.events.append(event)
    
    def add_duration_event(self, name: str, pid: int, tid: int, start_ts: int, 
                          duration: int, category: str = None, 
                          args: Dict[str, Any] = None) -> None:
        """Add a complete duration event (X phase)."""
        event = TraceEvent(
            name=name,
            ph='X',
            ts=start_ts,
            pid=pid,
            tid=tid,
            dur=duration,
            cat=category,
            args=args or {}
        )
        self.add_event(event)
    
    def add_instant_event(self, name: str, pid: int, tid: int, timestamp: int,
                         category: str = None, args: Dict[str, Any] = None) -> None:
        """Add an instant event (I phase)."""
        event = TraceEvent(
            name=name,
            ph='I',
            ts=timestamp,
            pid=pid,
            tid=tid,
            cat=category,
            args=args or {}
        )
        self.add_event(event)
    
    def add_counter_event(self, name: str, pid: int, tid: int, timestamp: int,
                         value: Union[int, float], category: str = None) -> None:
        """Add a counter event (C phase)."""
        event = TraceEvent(
            name=name,
            ph='C',
            ts=timestamp,
            pid=pid,
            tid=tid,
            cat=category,
            args={name: value}
        )
        self.add_event(event)
    
    def add_async_begin(self, name: str, pid: int, tid: int, timestamp: int,
                       async_id: Union[str, int], category: str = None,
                       args: Dict[str, Any] = None) -> None:
        """Add an async begin event (b phase)."""
        event = TraceEvent(
            name=name,
            ph='b',
            ts=timestamp,
            pid=pid,
            tid=tid,
            id=async_id,
            cat=category,
            args=args or {}
        )
        self.add_event(event)
    
    def add_async_end(self, name: str, pid: int, tid: int, timestamp: int,
                     async_id: Union[str, int], category: str = None,
                     args: Dict[str, Any] = None) -> None:
        """Add an async end event (e phase)."""
        event = TraceEvent(
            name=name,
            ph='e',
            ts=timestamp,
            pid=pid,
            tid=tid,
            id=async_id,
            cat=category,
            args=args or {}
        )
        self.add_event(event)
    
    def add_metadata(self, key: str, value: Any) -> None:
        """Add metadata to the trace."""
        self.metadata[key] = value
    
    def add_process_name(self, pid: int, name: str) -> None:
        """Add process name metadata."""
        event = TraceEvent(
            name='process_name',
            ph='M',
            ts=0,
            pid=pid,
            tid=0,
            args={'name': name}
        )
        self.add_event(event)
    
    def add_thread_name(self, pid: int, tid: int, name: str) -> None:
        """Add thread name metadata."""
        event = TraceEvent(
            name='thread_name',
            ph='M',
            ts=0,
            pid=pid,
            tid=tid,
            args={'name': name}
        )
        self.add_event(event)
    
    def sort_events(self) -> None:
        """Sort events by timestamp."""
        self.events.sort(key=lambda e: e.ts)
    
    def write_file(self, filepath: Union[str, Path], compress: bool = False) -> None:
        """Write trace data to a file."""
        filepath = Path(filepath)
        
        # Sort events before writing
        self.sort_events()
        
        # Prepare data structure
        data = {
            'traceEvents': [event.to_dict() for event in self.events]
        }
        
        # Add metadata
        if self.metadata:
            data.update(self.metadata)
        
        if self.stack_frames:
            data['stackFrames'] = self.stack_frames
        
        if self.samples:
            data['samples'] = self.samples
        
        # Write to file
        if compress or filepath.suffix == '.gz':
            with gzip.open(filepath, 'wt', encoding='utf-8') as f:
                json.dump(data, f, separators=(',', ':'))
        else:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, separators=(',', ':'))
    
    def write_string(self, pretty: bool = False) -> str:
        """Write trace data to a JSON string."""
        self.sort_events()
        
        data = {
            'traceEvents': [event.to_dict() for event in self.events]
        }
        
        if self.metadata:
            data.update(self.metadata)
        
        if self.stack_frames:
            data['stackFrames'] = self.stack_frames
        
        if self.samples:
            data['samples'] = self.samples
        
        if pretty:
            return json.dumps(data, indent=2)
        else:
            return json.dumps(data, separators=(',', ':'))


def microseconds_now() -> int:
    """Get current time in microseconds since epoch."""
    return int(time.time() * 1_000_000)


# Example usage and utility functions
def create_sample_trace() -> ChromeTraceWriter:
    """Create a sample trace for demonstration."""
    writer = ChromeTraceWriter()
    
    # Add process and thread names
    writer.add_process_name(1, "Main Process")
    writer.add_thread_name(1, 1, "Main Thread")
    writer.add_thread_name(1, 2, "Worker Thread")
    
    base_time = microseconds_now()
    
    # Add some sample events
    writer.add_duration_event("main", 1, 1, base_time, 1000, "function")
    writer.add_duration_event("work", 1, 2, base_time + 100, 800, "function")
    writer.add_instant_event("checkpoint", 1, 1, base_time + 500, "debug")
    writer.add_counter_event("memory", 1, 1, base_time + 200, 1024*1024)
    
    # Add async events
    writer.add_async_begin("async_task", 1, 1, base_time + 100, "task1", "async")
    writer.add_async_end("async_task", 1, 1, base_time + 900, "task1", "async")
    
    return writer


if __name__ == "__main__":
    # Example usage
    print("Creating sample trace...")
    writer = create_sample_trace()
    
    # Write to file
    writer.write_file("sample_trace.json")
    print("Sample trace written to sample_trace.json")
    
    # Read it back
    print("\nReading trace back...")
    reader = ChromeTraceReader()
    reader.read_file("sample_trace.json")
    
    print(f"Read {len(reader.events)} events")
    print(f"Time range: {reader.get_time_range()}")
    print(f"Duration events: {len(reader.get_duration_events())}")
    
    # Print first few events
    print("\nFirst 3 events:")
    for i, event in enumerate(reader.events[:3]):
        print(f"  {i+1}. {event.name} ({event.ph}) at {event.ts}μs")
