# Operator-Level EDP Analysis (Phase 2)

**Energy-Delay Product (EDP) breakdown at the operator level for deep neural networks**

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Understanding the Metrics](#understanding-the-metrics)
4. [API Reference](#api-reference)
5. [Architectural Modifiers](#architectural-modifiers)
6. [UnifiedAnalyzer Integration](#unifiedanalyzer-integration)
7. [Interpretation Guide](#interpretation-guide)
8. [Advanced Usage](#advanced-usage)
9. [Validation](#validation)

---

## Overview

### What is Operator-Level EDP?

Operator-level EDP analysis provides a **hierarchical breakdown** of energy-delay product (EDP) from the model level down to individual operators:

```
Model EDP
  └─ Subgraph EDP (per fused operation group)
      └─ Operator EDP (per individual operation)
```

This enables you to:
- **Identify bottlenecks** at the operator level
- **Understand architectural impact** on different operator types
- **Quantify fusion benefits** for specific operations
- **Optimize energy efficiency** by targeting high-EDP operators

### Key Features

- ✅ **Energy-based normalization**: Operator EDPs sum to ~97% of model energy (remaining 3% is static/leakage)
- ✅ **Architectural modifiers**: Architecture-specific cost multipliers (e.g., ReLU is 0.05× on KPU when fused, 3.0× on GPU when separate)
- ✅ **Fusion benefit analysis**: Quantifies EDP savings from operator fusion
- ✅ **Cross-architecture comparison**: Consistent operator extraction across all architectures
- ✅ **UnifiedAnalyzer integration**: Seamless integration with the unified analysis framework

### Why Energy-Based Normalization?

**Key Insight**: At the model level, all operators share the same total latency, but energy is additive.

```
Model EDP = Model Energy × Model Latency
Operator contribution = Operator Energy × Model Latency
Operator fraction = Operator Energy / Model Energy
```

Therefore, **operator EDP fractions represent energy fractions**, not raw EDP fractions. This is the correct way to attribute operator-level contributions to model-level EDP.

---

## Quick Start

### Basic Usage with ArchitectureComparator

```python
from graphs.analysis.architecture_comparator import ArchitectureComparator
from graphs.hardware.mappers.accelerators.kpu import create_kpu_t256_mapper
from graphs.hardware.resource_model import Precision

# Setup
architectures = {'KPU': create_kpu_t256_mapper()}
comparator = ArchitectureComparator(
    model_name='resnet18',
    architectures=architectures,
    batch_size=1,
    precision=Precision.FP32
)

# Run analysis
comparator.analyze_all()

# Get operator-level breakdown
operator_edps = comparator.get_operator_edp_breakdown('KPU')

# Display top operators
print(f"Found {len(operator_edps)} operators")
for i, op in enumerate(operator_edps[:5], 1):
    print(f"{i}. {op.operator_type:<15} "
          f"{op.architectural_edp*1e9:>8.2f} nJ·s "
          f"({op.edp_fraction_of_model*100:>5.1f}% of model)")
```

**Output:**
```
Found 68 operators
1. Conv2d             17.53 nJ·s (  8.1% of model)
2. Conv2d             17.53 nJ·s (  8.1% of model)
3. Conv2d             17.53 nJ·s (  8.1% of model)
4. BatchNorm2d         7.45 nJ·s (  5.2% of model)
5. ReLU                7.44 nJ·s (  5.1% of model)
```

### Generate Report

```python
# Generate formatted report
report = comparator.generate_operator_edp_report('KPU', top_n=10)
print(report)
```

**Output:**
```
========================================================================
Operator-Level EDP Breakdown: KPU
========================================================================

Total Operators: 68
Subgraphs: 68
Operator EDP Coverage: 96.7% of model energy
  (Remaining 3.3% is static/leakage energy)

Top 10 Operators by EDP:

Rank  Operator    Subgraph          EDP (nJ·s)  % Subgraph  % Model  Modifier  Fused
------------------------------------------------------------------------------------
1  ⭐  Conv2d      layer4_0_conv2    17.53       100.0%      8.1%     1.00×     No
2     Conv2d      layer4_1_conv1    17.53       100.0%      8.1%     1.00×     No
...

Operator Type Distribution:
  Conv2d          17     71.03 nJ·s     74.7%
  BatchNorm2d     20     10.08 nJ·s     10.6%
  ReLU            17      9.91 nJ·s     10.4%
```

---

## Understanding the Metrics

### OperatorEDPDescriptor Fields

Each operator has the following metrics:

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `operator_type` | str | Operator type | `"Conv2d"`, `"ReLU"` |
| `subgraph_name` | str | Parent subgraph | `"layer4_0_conv2"` |
| `base_edp` | float (J·s) | Hardware-agnostic EDP | `17.53e-9` |
| `architectural_edp` | float (J·s) | EDP with modifiers applied | `17.53e-9` |
| `architectural_modifier` | float | Architecture-specific multiplier | `1.0` (baseline), `0.05` (efficient), `3.0` (overhead) |
| `edp_fraction_of_subgraph` | float | Fraction within parent subgraph | `1.0` (100% if single-op) |
| `edp_fraction_of_model` | float | **Energy fraction** of total model | `0.081` (8.1%) |
| `is_fused` | bool | Whether fused with other ops | `False` |
| `fusion_benefit` | float | EDP savings from fusion | `20.0` (20× improvement) |

### Key Metrics Explained

#### 1. `architectural_edp`

The operator's EDP after applying architecture-specific modifiers:

```python
architectural_edp = base_edp × architectural_modifier
```

- **Base EDP**: FLOP-proportional allocation from subgraph EDP
- **Architectural modifier**: Architecture-specific cost multiplier (see below)

#### 2. `edp_fraction_of_model`

**Energy fraction** of the total model:

```python
operator_energy = architectural_edp / subgraph_latency
edp_fraction_of_model = operator_energy / model_total_energy
```

This correctly represents the operator's contribution to model-level EDP.

**Expected sum**: ~96.7% (remaining 3.3% is static/leakage energy)

#### 3. `architectural_modifier`

Architecture-specific cost multiplier based on:
- **Architecture class**: STORED_PROGRAM, DATA_PARALLEL, SYSTOLIC_ARRAY, DOMAIN_FLOW
- **Operator type**: Conv2d, ReLU, BatchNorm, etc.
- **Fusion status**: Fused vs. separate

See [Architectural Modifiers](#architectural-modifiers) for details.

---

## API Reference

### ArchitectureComparator

#### `get_operator_edp_breakdown(arch_name: str) -> List[OperatorEDPDescriptor]`

Get operator-level EDP breakdown for one architecture.

**Parameters:**
- `arch_name` (str): Architecture name (e.g., `'KPU'`, `'GPU'`)

**Returns:**
- `List[OperatorEDPDescriptor]`: Operators sorted by EDP (descending)

**Example:**
```python
operator_edps = comparator.get_operator_edp_breakdown('KPU')

# Filter by operator type
conv_ops = [op for op in operator_edps if op.operator_type == 'Conv2d']
print(f"Found {len(conv_ops)} Conv2d operators")

# Find high-modifier operators
high_overhead = [op for op in operator_edps if op.architectural_modifier > 2.0]
```

#### `get_subgraph_edp_breakdown(arch_name: str) -> List[SubgraphEDPDescriptor]`

Get subgraph-level EDP breakdown.

**Parameters:**
- `arch_name` (str): Architecture name

**Returns:**
- `List[SubgraphEDPDescriptor]`: Subgraphs sorted by EDP (descending)

**Example:**
```python
subgraph_edps = comparator.get_subgraph_edp_breakdown('KPU')

# Show top subgraphs
for sg in subgraph_edps[:3]:
    print(f"{sg.subgraph_name}: {sg.edp*1e9:.2f} nJ·s ({sg.edp_fraction*100:.1f}%)")
```

#### `generate_operator_edp_report(arch_name: str, top_n: int = 10) -> str`

Generate formatted operator-level EDP report.

**Parameters:**
- `arch_name` (str): Architecture name
- `top_n` (int): Number of top operators to show (default: 10)

**Returns:**
- `str`: Formatted report

**Example:**
```python
report = comparator.generate_operator_edp_report('KPU', top_n=15)
print(report)

# Save to file
with open('operator_edp_report.txt', 'w') as f:
    f.write(report)
```

---

## Architectural Modifiers

### What are Architectural Modifiers?

Architectural modifiers are **architecture-specific cost multipliers** that reflect the real cost of executing an operator on a particular hardware architecture.

### Modifier Categories

| Modifier Range | Category | Interpretation |
|----------------|----------|----------------|
| **< 0.5×** | Highly Efficient | Hidden in dataflow or pipelined (e.g., ReLU on KPU) |
| **0.5× - 1.5×** | Standard | Baseline execution cost |
| **> 2.0×** | High Overhead | Kernel launch overhead or inefficient mapping |

### Architecture-Specific Modifiers

#### Spatial Dataflow (KPU, TPU)

On spatial architectures, lightweight operations can be **hidden in the dataflow**:

| Operator | Fused | Separate | Fusion Benefit |
|----------|-------|----------|----------------|
| ReLU | 0.05× | 1.0× | 20× |
| Bias | 0.05× | 1.0× | 20× |
| BatchNorm | 1.5× | 1.5× | 1× |
| Conv2d | 1.0× | 1.0× | 1× |

**Example (KPU):**
```python
operator_edps = comparator.get_operator_edp_breakdown('KPU')
relu_ops = [op for op in operator_edps if op.operator_type == 'ReLU']

for relu in relu_ops:
    print(f"{relu.subgraph_name}: modifier={relu.architectural_modifier:.2f}×, "
          f"fused={relu.is_fused}, benefit={relu.fusion_benefit:.1f}×")

# Output:
# relu: modifier=1.00×, fused=False, benefit=None
# (Currently single-op subgraphs, fusion not yet implemented)
```

#### Sequential Execution (CPU, GPU)

On sequential architectures, separate kernel launches have **high overhead**:

| Operator | Fused | Separate | Fusion Benefit |
|----------|-------|----------|----------------|
| ReLU | 0.5× | 3.0× | 6× |
| Bias | 0.5× | 3.0× | 6× |
| BatchNorm | 1.0× | 2.0× | 2× |
| Conv2d | 1.0× | 1.0× | 1× |

**Example (GPU):**
```python
operator_edps = comparator.get_operator_edp_breakdown('GPU')

# Find high-overhead operators
high_overhead = [op for op in operator_edps
                 if op.architectural_modifier > 2.0]

print(f"Found {len(high_overhead)} operators with high overhead (>2.0×)")
for op in high_overhead[:5]:
    print(f"  {op.operator_type}: {op.architectural_modifier:.2f}×")
```

### Fusion Benefit

Fusion benefit quantifies the **EDP savings** from fusing operators:

```python
fusion_benefit = separate_edp / fused_edp
```

- **High benefit (>5×)**: Significant savings from fusion (e.g., ReLU on spatial architectures)
- **Low benefit (<2×)**: Limited savings from fusion
- **None**: Operator not fused or no benefit

---

## UnifiedAnalyzer Integration

### Enable Operator-Level EDP

```python
from graphs.analysis.unified_analyzer import UnifiedAnalyzer, AnalysisConfig
from graphs.hardware.resource_model import Precision

# Create config with operator EDP enabled
config = AnalysisConfig(
    run_operator_edp=True,  # Enable operator-level EDP
    run_concurrency=False,   # Optional: disable for speed
)

# Run analysis
analyzer = UnifiedAnalyzer(verbose=True)
result = analyzer.analyze_model(
    model_name='resnet18',
    hardware_name='kpu-t256',
    batch_size=1,
    precision=Precision.FP32,
    config=config
)

# Access operator breakdown
print(f"Operators: {len(result.operator_edp_breakdown)}")
print(f"Subgraphs: {len(result.subgraph_edp_breakdown)}")

# Show top operators
for i, op in enumerate(result.operator_edp_breakdown[:5], 1):
    print(f"{i}. {op.operator_type}: {op.edp_fraction_of_model*100:.1f}%")
```

### Disable Operator-Level EDP

To skip operator-level analysis (faster execution):

```python
config = AnalysisConfig(run_operator_edp=False)

result = analyzer.analyze_model(
    model_name='resnet18',
    hardware_name='kpu-t256',
    config=config
)

# operator_edp_breakdown will be None
assert result.operator_edp_breakdown is None
```

### Integration with Other Metrics

Operator-level EDP is consistent with other unified metrics:

```python
result = analyzer.analyze_model('resnet18', 'kpu-t256', config=config)

# All metrics available
print(f"Total Energy: {result.total_energy_mj:.2f} mJ")
print(f"Total Latency: {result.total_latency_ms:.2f} ms")
print(f"Peak Memory: {result.peak_memory_mb:.2f} MB")

# Operator breakdown
total_op_energy_frac = sum(op.edp_fraction_of_model
                            for op in result.operator_edp_breakdown)
print(f"Operator coverage: {total_op_energy_frac*100:.1f}% of model energy")
```

---

## Interpretation Guide

### Reading Operator EDP Fractions

**Q: Why do operator EDP fractions sum to 96.7% instead of 100%?**

**A:** This is **correct behavior**! The remaining 3.3% is static/leakage energy that's time-based, not operation-specific.

```
Model Energy = Operator Energy + Static Energy
             = 96.7%           + 3.3%
```

Operator EDP fractions represent **energy fractions**, which are additive.

### Common Patterns

#### Pattern 1: Conv2d Dominance

On most architectures, Conv2d operators dominate EDP:

```
Conv2d:      74.7% of model energy
BatchNorm:   10.6%
ReLU:        10.4%
Other:        1.0%
```

**Interpretation:** Focus optimization on Conv2d operations for maximum impact.

#### Pattern 2: Different Bottlenecks Across Architectures

The same model may have different bottlenecks on different architectures:

```
GPU:  Top operator: layer3_0_conv1 (2.6%) - more balanced distribution
KPU:  Top operator: layer4_0_conv2 (8.1%) - less balanced distribution
```

**Interpretation:** KPU has higher concentration in specific operators, suggesting different parallelism characteristics.

#### Pattern 3: Modifier Impact

Architectural modifiers reveal architecture-specific costs:

```
KPU (spatial):
  Conv2d:      1.0× (baseline)
  BatchNorm:   1.5× (higher overhead on spatial architectures)
  ReLU:        1.0× (would be 0.05× if fused)

GPU (sequential):
  Conv2d:      1.0× (baseline)
  BatchNorm:   1.0× (efficient on GPUs)
  ReLU:        3.0× (if separate - kernel launch overhead)
```

**Interpretation:** Different architectures have different cost structures for the same operators.

---

## Advanced Usage

### Custom Analysis

#### Find Optimization Opportunities

```python
operator_edps = comparator.get_operator_edp_breakdown('KPU')

# Find high-EDP, high-modifier operators (optimization targets)
targets = [op for op in operator_edps
           if op.edp_fraction_of_model > 0.05  # >5% of model
           and op.architectural_modifier > 1.5]  # High modifier

print(f"Found {len(targets)} optimization targets:")
for op in targets:
    print(f"  {op.operator_type} in {op.subgraph_name}: "
          f"{op.edp_fraction_of_model*100:.1f}% "
          f"(modifier: {op.architectural_modifier:.2f}×)")
```

#### Compare Operators Across Architectures

```python
architectures = {
    'GPU': create_h100_mapper(),
    'KPU': create_kpu_t256_mapper(),
}

comparator = ArchitectureComparator(
    model_name='resnet18',
    architectures=architectures,
    batch_size=1,
    precision=Precision.FP32
)
comparator.analyze_all()

# Compare Conv2d operators
for arch_name in ['GPU', 'KPU']:
    ops = comparator.get_operator_edp_breakdown(arch_name)
    conv_ops = [op for op in ops if op.operator_type == 'Conv2d']

    total_conv_frac = sum(op.edp_fraction_of_model for op in conv_ops)
    avg_modifier = sum(op.architectural_modifier for op in conv_ops) / len(conv_ops)

    print(f"{arch_name}:")
    print(f"  Conv2d count: {len(conv_ops)}")
    print(f"  Total Conv2d energy: {total_conv_frac*100:.1f}%")
    print(f"  Average modifier: {avg_modifier:.2f}×")
```

#### Aggregate by Operator Type

```python
from collections import defaultdict

operator_edps = comparator.get_operator_edp_breakdown('KPU')

# Aggregate by type
type_stats = defaultdict(lambda: {
    'count': 0,
    'total_edp': 0.0,
    'total_fraction': 0.0,
    'avg_modifier': 0.0
})

for op in operator_edps:
    stats = type_stats[op.operator_type]
    stats['count'] += 1
    stats['total_edp'] += op.architectural_edp
    stats['total_fraction'] += op.edp_fraction_of_model
    stats['avg_modifier'] += op.architectural_modifier

# Calculate averages
for stats in type_stats.values():
    stats['avg_modifier'] /= stats['count']

# Display
print("Operator Type Summary:")
print(f"{'Type':<20} {'Count':<8} {'Total %':<10} {'Avg Modifier'}")
print("-" * 60)

for op_type, stats in sorted(type_stats.items(),
                              key=lambda x: x[1]['total_fraction'],
                              reverse=True):
    print(f"{op_type:<20} {stats['count']:<8} "
          f"{stats['total_fraction']*100:<10.1f} "
          f"{stats['avg_modifier']:.2f}×")
```

---

## Validation

### Running Validation Tests

Comprehensive validation tests are available:

```bash
python validation/analysis/test_operator_edp_comprehensive.py
```

**Tests:**
1. ✅ Basic operator extraction
2. ✅ EDP fraction normalization
3. ✅ Architectural modifiers
4. ✅ Subgraph-level EDP breakdown
5. ✅ UnifiedAnalyzer integration
6. ✅ Cross-architecture consistency

**Expected output:**
```
Total tests: 6
Passed: 6 ✅
Failed: 0 ❌

🎉 ALL TESTS PASSED! 🎉

Phase 2: Operator-Level EDP is production-ready!
```

### Manual Validation

Quick sanity check:

```python
from graphs.analysis.architecture_comparator import ArchitectureComparator
from graphs.hardware.mappers.accelerators.kpu import create_kpu_t256_mapper
from graphs.hardware.resource_model import Precision

architectures = {'KPU': create_kpu_t256_mapper()}
comparator = ArchitectureComparator(
    model_name='resnet18',
    architectures=architectures,
    batch_size=1,
    precision=Precision.FP32
)

comparator.analyze_all()
operator_edps = comparator.get_operator_edp_breakdown('KPU')

# Validate fractions sum to ~0.97
total_frac = sum(op.edp_fraction_of_model for op in operator_edps)
assert 0.95 <= total_frac <= 1.0, f"Expected 0.95-1.0, got {total_frac}"
print(f"✓ Fractions sum to {total_frac:.4f} (expected ~0.97)")

# Validate all operators have positive EDP
assert all(op.architectural_edp > 0 for op in operator_edps)
print(f"✓ All {len(operator_edps)} operators have positive EDP")

# Validate modifiers are reasonable
assert all(0.01 <= op.architectural_modifier <= 10.0 for op in operator_edps)
print(f"✓ All modifiers in reasonable range [0.01, 10.0]")

print("\n✅ Validation passed!")
```

---

## Summary

Operator-level EDP analysis provides powerful insights into model efficiency:

- **Hierarchical breakdown**: Model → Subgraph → Operator
- **Architecture-aware**: Modifiers reflect real hardware costs
- **Energy-based normalization**: Correct attribution to model-level EDP
- **Fusion analysis**: Quantifies benefits of operator fusion
- **Production-ready**: Comprehensive validation tests passing

**Use cases:**
- Identify energy/latency bottlenecks
- Guide optimization efforts
- Compare architectures
- Validate fusion strategies
- Profile models at fine granularity

**Next steps:**
- Implement true operator fusion (currently single-op subgraphs)
- Add more architectural modifiers for specialized hardware
- Integrate with optimization passes
- Extend to custom operator types

---

**Documentation Version:** Phase 2 Complete (2025-01-15)
