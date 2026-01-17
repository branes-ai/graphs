# Graph Analysis Tool Extraction Plan

**Date**: 2025-12-23 (Completed: 2025-12-24)
**Status**: ALL PHASES COMPLETE
**Objective**: Extract a DNN graph analysis tool from `graphs` that can be used by `embodied-ai-architect` agentic workflows

## Current State Summary

| Repository | What It Has | What It Needs |
|------------|-------------|---------------|
| **graphs** | UnifiedAnalyzer, 30+ hardware mappers, Phase 3 analyzers (roofline, energy, memory) | Pydantic output models, verdict-first format |
| **embodied-schemas** | Pydantic schemas (HardwareEntry, BenchmarkResult, AnalysisResult), Registry API | Graph analysis result schemas |
| **embodied-ai-architect** | graphs_tools.py (partial integration), tool-use system, orchestrator | Enhanced graph tools, verdict-first wrappers |

## Integration Architecture

```
graphs (analysis engine)
    |
    v  [New: Pydantic adapters]
embodied-schemas (shared schemas)
    |
    v  [Existing: graphs_tools.py]
embodied-ai-architect (agentic tools)
```

## Implementation Phases

### Phase 1: Define Pydantic Schemas in embodied-schemas [COMPLETE]

Created graph analysis result schemas that follow the verdict-first pattern.

**Files created (2025-12-24):**
- `src/embodied_schemas/analysis.py` - Graph analysis Pydantic models (9 schemas)
- `tests/test_analysis_schemas.py` - Schema validation tests (20 tests, all passing)

**Schemas implemented:**
- `Bottleneck` - Enum: compute-bound, memory-bound, balanced
- `RooflineResult` - Latency and bottleneck analysis
- `EnergyResult` - Three-component energy breakdown
- `MemoryResult` - Memory footprint analysis
- `ConcurrencyResult` - Multi-level parallelism analysis
- `SubgraphBreakdown` - Per-subgraph metrics
- `GraphAnalysisResult` - Top-level verdict-first result
- `ComparisonResult` - Multi-hardware comparison
- `BatchSweepResult` - Batch size sweep analysis

### Phase 2: Create Adapters in graphs [COMPLETE]

Added Pydantic output adapters that convert `UnifiedAnalysisResult` to embodied-schemas format.

**Files created (2025-12-24):**
- `src/graphs/adapters/__init__.py` - Package exports
- `src/graphs/adapters/pydantic_output.py` - Adapter functions (~350 lines)
- `tests/test_pydantic_adapter.py` - 19 tests (all passing)

**Key functions:**
- `convert_to_pydantic(result, constraint_metric, constraint_threshold)` - Main entry point
- `convert_roofline_to_pydantic(roofline, latency_ms)` - Roofline conversion
- `convert_energy_to_pydantic(energy, hardware_tdp_w)` - Energy conversion
- `convert_memory_to_pydantic(memory)` - Memory conversion
- `make_verdict(actual, threshold, metric, lower_is_better)` - Verdict helper

**Changes:**
- Added `embodied-schemas` as optional dependency in `pyproject.toml` (`pip install graphs[schemas]`)

### Phase 3: Enhance graphs_tools.py in embodied-ai-architect [COMPLETE]

Updated the existing integration to use the new schemas.

**Files modified (2025-12-24):**
- `src/embodied_ai_architect/llm/graphs_tools.py` (+240 lines)

**New verdict-first tools added:**
| Tool Name | Purpose | Returns |
|-----------|---------|---------|
| `check_latency` | Latency constraint check | GraphAnalysisResult |
| `check_power` | Power budget check | GraphAnalysisResult |
| `check_memory` | Memory constraint check | GraphAnalysisResult |
| `full_analysis` | Comprehensive analysis with optional constraint | GraphAnalysisResult |

**Example output:**
```json
{
  "verdict": "PASS",
  "confidence": "high",
  "summary": "Latency of 0.4 is well under 10.0 target (96% headroom)",
  "constraint": {"metric": "latency", "threshold": 10.0, "actual": 0.43, "margin_pct": 95.7},
  "suggestions": ["Increase batch size to amortize static energy"]
}
```

### Phase 4: Integration Testing [COMPLETE]

End-to-end testing of the full pipeline.

**Files created (2025-12-24):**
- `embodied-ai-architect/tests/test_graphs_integration.py` - 27 integration tests (all passing)
- `embodied-ai-architect/docs/graphs_tools_guide.md` - Usage documentation

**Test coverage:**
- Single model analysis with constraint checking (PASS/FAIL cases)
- Multiple hardware targets (H100, A100, Jetson Orin, TPU v4)
- Multiple models (resnet18, resnet50, mobilenet_v2)
- Verdict-first pattern verification
- Error handling for invalid inputs

**Test results:** 27/27 tests passing

## Detailed Schema Design

### RooflineResult

```python
class RooflineResult(BaseModel):
    """Roofline model analysis results."""
    latency_ms: float
    bottleneck: Literal["compute-bound", "memory-bound", "balanced"]
    utilization_pct: float
    arithmetic_intensity: float
    peak_flops: float
    peak_bandwidth_gbps: float
    achieved_flops: float
    achieved_bandwidth_gbps: float
```

### EnergyResult

```python
class EnergyResult(BaseModel):
    """Three-component energy model results."""
    total_energy_mj: float
    compute_energy_mj: float
    memory_energy_mj: float
    static_energy_mj: float
    average_power_w: float
    peak_power_w: float
    energy_efficiency_pct: float
    power_gating_savings_mj: float | None = None
```

### MemoryResult

```python
class MemoryResult(BaseModel):
    """Memory footprint analysis results."""
    peak_memory_mb: float
    weights_mb: float
    activations_mb: float
    workspace_mb: float
    fits_in_l2: bool
    fits_in_device_memory: bool
    l2_cache_mb: float
    device_memory_mb: float
```

### GraphAnalysisResult (Top-level)

```python
class GraphAnalysisResult(BaseModel):
    """Verdict-first graph analysis output for agentic workflows."""
    # Verdict (required for LLM consumption)
    verdict: Literal["PASS", "FAIL", "PARTIAL", "UNKNOWN"]
    confidence: Literal["high", "medium", "low"]
    summary: str

    # Metadata
    model_id: str
    hardware_id: str
    batch_size: int = 1
    precision: str = "fp32"
    timestamp: datetime

    # Derived metrics (key numbers)
    latency_ms: float
    throughput_fps: float
    energy_per_inference_mj: float
    peak_memory_mb: float

    # Detailed breakdowns
    roofline: RooflineResult
    energy: EnergyResult
    memory: MemoryResult

    # Constraint checking (optional)
    constraint_metric: str | None = None
    constraint_threshold: float | None = None
    constraint_margin_pct: float | None = None

    # Recommendations
    suggestions: list[str] = []
    warnings: list[str] = []
```

## Verdict-First Design Principles

The verdict-first pattern ensures LLMs can trust tool outputs without domain reasoning:

1. **Verdict**: PASS | FAIL | PARTIAL | UNKNOWN
   - PASS: Meets all constraints
   - FAIL: Does not meet constraints
   - PARTIAL: Some constraints met, some failed
   - UNKNOWN: Could not determine (missing data, error)

2. **Confidence**: high | medium | low
   - high: Based on calibrated hardware models
   - medium: Based on roofline estimates
   - low: Based on theoretical peaks only

3. **Summary**: One sentence explaining what was checked and what was found

4. **Suggestion**: If not PASS, actionable next step (or null)

## Data Ownership Boundaries

| Data Type | Owner Repository | Notes |
|-----------|------------------|-------|
| Vendor specs (memory, TDP, cores) | embodied-schemas | Datasheet facts |
| Roofline params (ops_per_clock) | graphs | Calibration data |
| Hardware mappers | graphs | Architecture-specific |
| Tool definitions | embodied-ai-architect | LLM integration |
| Analysis result schemas | embodied-schemas | Shared contract |
| Adapter implementations | graphs | Conversion logic |

## Dependencies

```
embodied-schemas >= 0.2.0
    ^
    |
graphs (adds dependency)
    ^
    |
embodied-ai-architect (existing dependency)
```

## Success Criteria

1. [x] Pydantic schemas defined in embodied-schemas (2025-12-24)
2. [x] Adapter layer in graphs converts UnifiedAnalysisResult to Pydantic (2025-12-24)
3. [x] graphs_tools.py uses new schemas with verdict-first output (2025-12-24)
4. [x] End-to-end test: `check_latency("resnet18", "H100", 10.0)` returns verdict (2025-12-24)
5. [x] Integration tests: 27 tests across 4 hardware targets and 3 models (2025-12-24)
