# Calibration Coverage Analyzer - Requirements Analysis

## Problem Statement

First-principles roofline estimation is difficult because:
1. Real hardware exhibits complex, size-dependent efficiency curves
2. Different operation types have different performance characteristics
3. Calibration data exists at a different semantic level than graph operations

We need a tool that takes a computational graph and reports what calibration data exists (or is missing) for the operations in that graph, enabling us to:
- Understand estimation quality before running inference
- Identify which operations need calibration
- Diagnose why estimation errors occur

## The Semantic Gap

### Graph-Level Operations (what models contain)

From `OperationType` enum in `src/graphs/core/structures.py`:

| Category | Operations |
|----------|------------|
| Convolutions | CONV2D, CONV2D_DEPTHWISE, CONV2D_POINTWISE |
| Linear Algebra | LINEAR, MATMUL |
| Normalization | BATCHNORM, LAYERNORM |
| Activations | RELU, RELU6, SILU, SWISH, GELU, HARDSWISH, SIGMOID |
| Pooling | MAXPOOL, AVGPOOL, ADAPTIVEAVGPOOL |
| Composite | ELEMENTWISE, SQUEEZE_EXCITE, MULTIHEAD_ATTENTION |
| Regularization | DROPOUT, STOCHASTIC_DEPTH |

### Fusion Patterns (what the partitioner produces)

Examples from `validate_estimation_subgraph.py` output:
- `Conv2d_BatchNorm2d_ReLU`
- `LayerNorm_Linear_GELU_+1more`
- `LayerNorm_MultiheadAttention`
- `Linear_Dropout`
- `add_ReLU`
- `Unfused`

### Calibrated Primitives (what we measure)

From `hardware_registry/` JSON files:

| Category | Operations | Metrics |
|----------|------------|---------|
| STREAM | copy, scale, add, triad | bandwidth, latency |
| BLAS L1 | dot, axpy, scal | GFLOPS, latency |
| BLAS L2 | gemv, ger | GFLOPS, latency |
| BLAS L3 | gemm | GFLOPS, efficiency, latency |

### The Mapping Problem

To use calibration data for estimation, we need to map:

```
Graph Operation --> Primitive(s) --> Calibration Entry
     |                  |                  |
   Conv2d          im2col+GEMM       gemm profile
   Linear            GEMM            gemm profile
   BatchNorm         STREAM          stream_copy/scale
   LayerNorm         STREAM+ops      stream + elementwise
   Attention         3xGEMM+softmax  gemm + stream
```

## Requirements

### Functional Requirements

#### FR1: Graph Operation Extraction
- Input: Model name or traced FX graph
- Output: List of operations with:
  - Operation type (OperationType enum)
  - Fusion pattern (from partitioner)
  - FLOPs count
  - Memory footprint (input, output, weights)
  - Arithmetic intensity

#### FR2: Primitive Mapping
- Map each graph operation to one or more calibration primitives
- Handle composite operations (e.g., attention = QKV + softmax + output)
- Track mapping confidence (direct match vs. approximation)

#### FR3: Calibration Lookup
- Query calibration database for each primitive
- Match by:
  - Hardware target
  - Operation type
  - Size range (find closest calibrated size)
  - Precision (FP32, FP16, INT8)
  - Power mode (for edge devices)

#### FR4: Coverage Analysis
- For each operation, report:
  - Calibration status: CALIBRATED, APPROXIMATED, MISSING
  - If calibrated: efficiency, bandwidth, measured latency
  - If approximated: which calibration entry is being used, confidence level
  - If missing: what calibration would be needed

#### FR5: Accuracy Reporting
- If validation data exists (measured vs. estimated):
  - Per-operation error
  - Error attribution (which ops contribute most to total error)
  - Pattern analysis (which fusion patterns are problematic)

### Non-Functional Requirements

#### NFR1: Performance
- Analysis should complete in < 5 seconds for typical models (ResNet, ViT)
- Support caching of calibration database queries

#### NFR2: Extensibility
- Easy to add new operation-to-primitive mappings
- Support for new calibration operation types (Conv2d direct, attention kernels)

#### NFR3: Output Formats
- Terminal output with colored status indicators
- JSON export for programmatic analysis
- CSV export for spreadsheet analysis

## Architecture

### Components

```
+-------------------+     +--------------------+     +-------------------+
|  Graph Analyzer   |---->| Primitive Mapper   |---->| Calibration DB    |
|  (extract ops)    |     | (map to primitives)|     | (lookup profiles) |
+-------------------+     +--------------------+     +-------------------+
         |                         |                         |
         v                         v                         v
+-------------------+     +--------------------+     +-------------------+
| SubgraphDescriptor|     | PrimitiveMappings  |     | CalibrationMatch  |
| - fusion_pattern  |     | - op -> [prims]    |     | - status          |
| - total_flops     |     | - confidence       |     | - efficiency      |
| - memory          |     | - size_params      |     | - bandwidth       |
+-------------------+     +--------------------+     +-------------------+
                                    |
                                    v
                          +--------------------+
                          | Coverage Report    |
                          | - per-op coverage  |
                          | - gaps identified  |
                          | - accuracy data    |
                          +--------------------+
```

### Operation-to-Primitive Mapping Table

This is the core mapping logic needed:

| Graph Operation | Dominant Primitive | Secondary | Size Parameter |
|-----------------|-------------------|-----------|----------------|
| CONV2D | gemm | stream_copy | K*K*C_in*C_out*H*W |
| CONV2D_DEPTHWISE | stream_scale | - | K*K*H*W*C |
| CONV2D_POINTWISE | gemm | - | C_in*C_out*H*W |
| LINEAR | gemm | - | M*K*N |
| MATMUL | gemm | - | M*K*N |
| BATCHNORM | stream_scale | stream_add | 4*C*H*W bytes |
| LAYERNORM | stream_scale | stream_add | 4*N bytes (per token) |
| RELU/activations | stream_copy | - | tensor size |
| MAXPOOL/AVGPOOL | stream_copy | - | tensor size |
| MULTIHEAD_ATTENTION | gemm (x4) | stream_copy | complex formula |
| DROPOUT | stream_copy | - | tensor size |

### Calibration Match Quality Levels

```python
class CalibrationMatchQuality(Enum):
    EXACT = "exact"           # Same op, same size, same HW
    SIZE_INTERPOLATED = "size_interpolated"  # Same op, interpolated between sizes
    SIZE_EXTRAPOLATED = "size_extrapolated"  # Same op, outside calibrated range
    OP_APPROXIMATED = "op_approximated"      # Different op used as proxy
    MISSING = "missing"       # No applicable calibration
```

## Output Format

### Terminal Output (Example)

```
Calibration Coverage Analysis: ViT-B/16 on Jetson-Orin-AGX (30W, FP32)
=======================================================================

Subgraph Coverage:
  Pattern                      Ops   FLOPs      Status        Calibration    Est.Eff
  ---------------------------------------------------------------------------------
  LayerNorm_MHA               x12   12.6G      APPROX        gemm@256M      67%
    - qkv_proj (Linear)              10.1G     CALIBRATED    gemm@256M      72%
    - attention (MatMul)              1.5G     APPROX        gemm@128M      58%
    - softmax                         0.0G     MISSING       -              -
    - output_proj (Linear)            1.0G     CALIBRATED    gemm@64M       71%

  LayerNorm_Linear_GELU       x12   11.2G      CALIBRATED    gemm@256M      68%

  Unfused (reshape, etc.)     x48   0.0G       MISSING       -              -

Summary:
  Operations:     156 total
  Calibrated:      24 (15.4%)   - direct calibration match
  Approximated:    84 (53.8%)   - using similar calibration
  Missing:         48 (30.8%)   - no applicable calibration

  FLOPs Coverage:
  Calibrated:     31.2G (88.7%) of 35.2G total
  Missing:         4.0G (11.3%) - mostly softmax, reshape, layernorm

  Accuracy Risk Assessment:
  HIGH RISK:   Softmax (12 instances) - memory-bound, no calibration
  MEDIUM RISK: LayerNorm (24 instances) - approximated from stream_scale
  LOW RISK:    Linear/MatMul (36 instances) - well-calibrated GEMM
```

### JSON Output Structure

```json
{
  "model": "vit_b_16",
  "hardware": "Jetson-Orin-AGX",
  "power_mode": "30W",
  "precision": "FP32",
  "analysis_timestamp": "2026-02-03T10:30:00Z",

  "summary": {
    "total_ops": 156,
    "calibrated": 24,
    "approximated": 84,
    "missing": 48,
    "total_flops": 35200000000,
    "calibrated_flops": 31200000000,
    "coverage_percent": 88.7
  },

  "subgraphs": [
    {
      "id": 0,
      "pattern": "LayerNorm_MultiheadAttention",
      "flops": 1050000000,
      "primitives": [
        {
          "name": "qkv_projection",
          "type": "gemm",
          "flops": 805000000,
          "calibration_status": "CALIBRATED",
          "calibration_entry": "gemm_m=197_k=768_n=2304",
          "measured_efficiency": 0.72,
          "match_quality": "SIZE_INTERPOLATED"
        },
        {
          "name": "softmax",
          "type": "softmax",
          "flops": 0,
          "bytes": 1500000,
          "calibration_status": "MISSING",
          "calibration_entry": null,
          "suggested_proxy": "stream_copy"
        }
      ]
    }
  ],

  "gaps": [
    {
      "operation_type": "softmax",
      "count": 12,
      "total_bytes": 18000000,
      "suggested_calibration": "Add softmax benchmark to calibration suite"
    }
  ]
}
```

## Implementation Plan

### Phase 1: Core Infrastructure
1. Create `PrimitiveMapper` class with operation-to-primitive mappings
2. Create `CalibrationLookup` class to query calibration database
3. Define match quality scoring logic

### Phase 2: Analysis Tool
1. Create `cli/analyze_calibration_coverage.py`
2. Integrate with existing `UnifiedAnalyzer` for graph extraction
3. Implement terminal and JSON output formats

### Phase 3: Accuracy Integration
1. Connect with `validate_estimation_subgraph.py` results
2. Add historical accuracy tracking
3. Generate accuracy risk assessments

### Phase 4: Enhanced Calibration
1. Identify priority gaps from coverage analysis
2. Add missing operation benchmarks (softmax, layernorm, attention)
3. Validate accuracy improvements

## Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `src/graphs/calibration/primitive_mapper.py` | Create | Map graph ops to calibration primitives |
| `src/graphs/calibration/coverage_analyzer.py` | Create | Core analysis logic |
| `cli/analyze_calibration_coverage.py` | Create | CLI entry point |
| `src/graphs/calibration/calibrator.py` | Modify | Add missing operation benchmarks |

## Open Questions

1. **Precision handling**: How do we handle mixed-precision graphs? Track per-operation?

2. **Fusion efficiency**: Does fusing ops change efficiency vs. sum of parts? (Likely yes - need fusion-aware calibration)

3. **Size interpolation**: What interpolation method for efficiency vs. size? (Linear? Log-linear?)

4. **Confidence scoring**: How to quantify confidence in approximated calibrations?

5. **Historical data**: Should we track coverage over time to measure improvement?

## Success Criteria

1. Tool runs on any model supported by `UnifiedAnalyzer`
2. Clearly identifies calibration gaps
3. Provides actionable recommendations for improving estimation accuracy
4. Integrates with existing validation tools
5. Helps prioritize calibration effort (focus on high-impact gaps)
