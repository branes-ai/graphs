# Hardware Calibration Framework - Documentation Index

This document provides a roadmap to all calibration framework documentation.

## Quick Start

**New to calibration?** Start here:
1. Read `README.md` - Framework overview
2. Run basic calibration: `python calibrator.py --help`
3. Try fusion calibration: See `FUSION_INTEGRATION_EXAMPLE.md`

## Core Documentation

### Framework Overview

- **`README.md`** - Main calibration framework documentation
  - What is hardware calibration?
  - Why measure real performance?
  - Basic usage and examples

- **`INTEGRATION_GUIDE.md`** - Integration with other components
  - How to use calibration data in mappers
  - How to use in cost models
  - How to use in partitioners

- **`MAPPER_CALIBRATION_INTEGRATION.md`** - Mapper integration details
  - How to use calibration in hardware mappers
  - Efficiency-based performance modeling
  - Example implementations

### Implementation Details

- **`schema.py`** - Data structures and API
  - `OperationCalibration` - Individual operation profiles
  - `FusionCalibration` - Fusion pattern profiles
  - `HardwareCalibration` - Complete calibration object
  - Serialization/deserialization

- **`calibrator.py`** - Calibration orchestrator
  - Command-line interface
  - Calibration workflow
  - Result aggregation

## Fusion Calibration (NEW - 2025-11-05)

### Overview Documents

- **`FUSION_WORK_COMPLETE.md`** ⭐ **START HERE**
  - Complete implementation summary
  - What was built (benchmarks, integration, docs)
  - Key findings (CPU: 1.0-1.3×, GPU expected: 2-5×)
  - Testing results
  - All deliverables

### Design and Planning

- **`FUSION_CALIBRATION_DESIGN.md`**
  - Motivation: Why measure fusion?
  - Fusion patterns to benchmark
  - Implementation strategy
  - Integration approach
  - Expected results

- **`FUSION_INTEGRATION_PLAN.md`**
  - Step-by-step integration plan
  - Schema extension design
  - Calibrator updates
  - CLI changes
  - Timeline and milestones

### Results and Analysis

- **`FUSION_BENCHMARKS_SUMMARY.md`**
  - Implementation summary
  - Benchmark results (i7-12700K CPU)
  - Key insights:
    - Fusion benefits are hardware-specific
    - CPU: 1.0-1.3× speedup
    - GPU expected: 2-5× speedup
  - Integration path
  - Recommendations

### Usage and Examples

- **`FUSION_INTEGRATION_EXAMPLE.md`** ⭐ **PRACTICAL GUIDE**
  - Command-line usage
  - Python API usage
  - Graph partitioner integration
  - Cost model examples
  - JSON structure
  - Hardware-aware fusion strategies

## Benchmarks

### Benchmark Documentation

- **`benchmarks/README.md`**
  - Overview of all benchmarks
  - Fusion benchmark design
  - Result dictionary format
  - Implementation details
  - How to add new benchmarks

### Individual Benchmarks

- **`benchmarks/matmul_bench.py`** - Matrix multiplication
- **`benchmarks/memory_bench.py`** - Memory bandwidth
- **`benchmarks/fused_linear_bench.py`** - Linear fusion (450 lines)
- **`benchmarks/fused_conv_bench.py`** - Conv fusion (500 lines)
- **`benchmarks/fused_attention_bench.py`** - Attention fusion (450 lines)

## Bug Fixes and RCAs

- **`CALIBRATION_STATISTICS_FIX.md`**
  - Root cause analysis: "Worst GFLOPS: 0.0 (56.0% efficiency)"
  - Problem: Mixing compute and memory operation statistics
  - Solution: Separate compute from memory operations
  - Implementation and validation

## Integration Guides

- **`MAPPER_CALIBRATION_INTEGRATION.md`**
  - How to integrate calibration with hardware mappers
  - Efficiency-based latency estimation
  - Example mapper implementations
  - Best practices

## Profiles

Calibration profiles are stored in `profiles/`:
- `intel_i7_12700k.json` - Basic calibration (no fusion)
- `i7_12700k_with_fusion.json` - Complete calibration with fusion patterns

## Documentation by Use Case

### I want to...

#### **Run basic calibration**
→ Read `README.md`
→ Run: `python calibrator.py --hardware MyHardware --peak-gflops 1000 --peak-bandwidth 50 --output profile.json`

#### **Run fusion calibration**
→ Read `FUSION_INTEGRATION_EXAMPLE.md` (Usage section)
→ Run: `python calibrator.py --hardware MyHardware --peak-gflops 1000 --peak-bandwidth 50 --fusion-patterns all --output profile.json`

#### **Understand fusion results**
→ Read `FUSION_BENCHMARKS_SUMMARY.md` (Key Insights section)
→ Read `FUSION_WORK_COMPLETE.md` (Key Findings section)

#### **Use calibration in my code**
→ Read `INTEGRATION_GUIDE.md`
→ Read `MAPPER_CALIBRATION_INTEGRATION.md` (for hardware mappers)
→ Read `FUSION_INTEGRATION_EXAMPLE.md` (Python API section)

#### **Add new fusion patterns**
→ Read `benchmarks/README.md` (Extending the Benchmarks section)
→ Look at existing benchmarks as examples

#### **Understand the design**
→ Read `FUSION_CALIBRATION_DESIGN.md`
→ Read `FUSION_INTEGRATION_PLAN.md`

#### **See what was built**
→ Read `FUSION_WORK_COMPLETE.md` ⭐

#### **Debug calibration issues**
→ Read `CALIBRATION_STATISTICS_FIX.md` (example RCA)
→ Check `schema.py` for data structure details

## File Organization

```
calibration/
├── DOCUMENTATION_INDEX.md                # This file - navigation guide
├── README.md                             # Framework overview
├── INTEGRATION_GUIDE.md                  # Integration with other components
├── MAPPER_CALIBRATION_INTEGRATION.md     # Hardware mapper integration
│
├── schema.py                             # Data structures and API
├── calibrator.py                         # Calibration orchestrator
├── __init__.py                           # Package exports
│
├── benchmarks/                           # Microbenchmarks
│   ├── README.md                         # Benchmark documentation
│   ├── matmul_bench.py                  # Matrix multiplication
│   ├── memory_bench.py                  # Memory bandwidth
│   ├── fused_linear_bench.py            # Linear fusion patterns
│   ├── fused_conv_bench.py              # Conv fusion patterns
│   └── fused_attention_bench.py         # Attention fusion patterns
│
├── profiles/                             # Calibration results
│   ├── intel_i7_12700k.json             # Basic calibration
│   └── i7_12700k_with_fusion.json       # With fusion patterns
│
└── docs/                                 # Documentation
    ├── FUSION_WORK_COMPLETE.md           # ⭐ Complete implementation summary
    ├── FUSION_CALIBRATION_DESIGN.md      # Design document
    ├── FUSION_INTEGRATION_PLAN.md        # Integration plan
    ├── FUSION_BENCHMARKS_SUMMARY.md      # Results and analysis
    ├── FUSION_INTEGRATION_EXAMPLE.md     # ⭐ Usage examples
    └── CALIBRATION_STATISTICS_FIX.md     # Bug fix RCA
```

## Reading Order

### For New Users

1. `README.md` - Understand what calibration is
2. `FUSION_WORK_COMPLETE.md` - See what fusion calibration adds
3. `FUSION_INTEGRATION_EXAMPLE.md` - Learn how to use it
4. Run your first calibration!

### For Implementers

1. `FUSION_CALIBRATION_DESIGN.md` - Understand the design
2. `benchmarks/README.md` - Understand benchmark design
3. `schema.py` - Understand data structures
4. `calibrator.py` - Understand orchestration
5. Individual benchmark files - See implementation

### For Researchers

1. `FUSION_BENCHMARKS_SUMMARY.md` - See empirical results
2. `FUSION_WORK_COMPLETE.md` - Understand key findings
3. Run benchmarks yourself to validate
4. Extend to other hardware (GPU, TPU, etc.)

## Key Takeaways

### What is Hardware Calibration?

**Hardware calibration** measures the real-world performance of operations on specific hardware, rather than relying on theoretical peak specifications.

**Why?**
- Theoretical peak: 1000 GFLOPS (datasheet)
- Measured reality: 750 GFLOPS (75% efficiency)
- Calibration captures the **efficiency** of real workloads

### What is Fusion Calibration?

**Fusion calibration** measures the performance benefit of fusing multiple operations together compared to running them separately.

**Example:**
- Unfused: `Y = X @ W.T; Y = Y + bias; Y = relu(Y)` → 3 operations
- Fused: `Y = relu(addmm(bias, X, W.T))` → 1 operation
- Benefit: 1.0-2.2× speedup (CPU), 2-5× expected (GPU)

### Key Findings

1. **Fusion benefits are hardware-specific**
   - CPU: 1.0-1.3× speedup (modest)
   - GPU: 2-5× speedup expected (strong)

2. **Not all fusion helps**
   - Conv+BN: ✓ 1.03-1.07× speedup
   - QK^T: ✓ 1.07-1.28× speedup
   - Full Attention: ✗ 0.60-0.71× slower on CPU!

3. **Memory reduction ≠ speedup**
   - Full Attention: 75% memory reduction but 0.71× speedup
   - CPU caches hide memory latency

4. **Hardware-aware strategies are critical**
   - CPU: Selective fusion (only beneficial patterns)
   - GPU: Aggressive fusion (most patterns beneficial)

## Status

- ✅ **Basic Calibration**: Production-ready (pre-2025-11-05)
- ✅ **Fusion Calibration**: Production-ready (2025-11-05)
- ⏳ **GPU Calibration**: Future work
- ⏳ **TPU/Accelerator Calibration**: Future work

## Contributing

When adding documentation:
1. Update this index
2. Follow existing documentation style
3. Include code examples
4. Cross-reference related docs
5. Add entry to "I want to..." section if applicable

## Questions?

- **General calibration**: See `README.md`
- **Fusion calibration**: See `FUSION_WORK_COMPLETE.md`
- **Usage examples**: See `FUSION_INTEGRATION_EXAMPLE.md`
- **Implementation details**: See individual files in `benchmarks/`
- **Bug reports**: Create issue with RCA like `CALIBRATION_STATISTICS_FIX.md`

## Recent Updates

### 2025-11-05: Fusion Calibration Framework

**Added**:
- 3 fusion benchmark modules (1,400 lines)
- Schema extension with `FusionCalibration`
- Calibrator integration with `--fusion-patterns`
- 5 comprehensive documentation files
- Benchmarks README

**Key Achievement**: Quantified fusion benefits on CPU (1.0-1.3×), identified hardware-specific strategies

See `FUSION_WORK_COMPLETE.md` for complete details.

---

**Last Updated**: 2025-11-05
**Total Documentation**: ~10,000 lines across 10+ files
**Status**: Production-ready ✅
