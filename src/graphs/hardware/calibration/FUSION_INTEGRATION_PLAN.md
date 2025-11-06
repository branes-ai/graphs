# Fusion Calibration Integration Plan

## Overview

This document describes the integration of fusion benchmarks into the main hardware calibration framework.

**Status**: Ready to implement
**Date**: 2025-11-05

## What's Already Built

✅ Three fusion benchmark modules:
- `benchmarks/fused_linear_bench.py` (450 lines)
- `benchmarks/fused_conv_bench.py` (500 lines)
- `benchmarks/fused_attention_bench.py` (450 lines)

✅ All benchmarks tested and validated on i7-12700K CPU

✅ Design document and results summary completed

## Integration Goals

1. **Extend Schema** to store fusion calibration results
2. **Update Calibrator** to run fusion benchmarks when requested
3. **Add CLI Support** for fusion patterns
4. **Preserve Backward Compatibility** with existing calibration profiles

## Step 1: Extend Schema

### Add FusionCalibration Data Class

Add to `schema.py`:

```python
@dataclass
class FusionCalibration:
    """
    Calibration for a fused kernel pattern.

    Measures the performance benefit of fusing multiple operations
    compared to running them separately.
    """
    # Fusion pattern identification
    fusion_pattern: str  # "Linear_Bias_ReLU", "Conv_BN_ReLU", etc.
    operators: List[str]  # ["linear", "bias", "relu"]
    num_operators: int    # 3

    # Unfused performance (baseline)
    unfused_latency_ms: float
    unfused_gflops: float
    unfused_memory_bytes: int

    # Fused performance (optimized)
    fused_latency_ms: float
    fused_gflops: float
    fused_memory_bytes: int

    # Fusion benefits
    speedup_factor: float         # unfused_latency / fused_latency
    memory_reduction: float       # 1 - (fused_bytes / unfused_bytes)
    gflops_improvement: float     # (fused_gflops - unfused_gflops) / unfused_gflops

    # Test configuration
    input_shape: Tuple[int, ...]
    extra_params: Dict[str, Any] = field(default_factory=dict)

    # Timing details
    num_trials: int = 100

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'FusionCalibration':
        """Create from dictionary"""
        if isinstance(data.get('input_shape'), list):
            data['input_shape'] = tuple(data['input_shape'])
        if isinstance(data.get('operators'), tuple):
            data['operators'] = list(data['operators'])
        return cls(**data)
```

### Add Fusion Storage to HardwareCalibration

Add field to `HardwareCalibration`:

```python
@dataclass
class HardwareCalibration:
    # ... existing fields ...

    # Fusion pattern calibration profiles
    fusion_profiles: Dict[str, FusionCalibration] = field(default_factory=dict)
```

Add methods to `HardwareCalibration`:

```python
def add_fusion_pattern(self, profile: FusionCalibration):
    """Add a fusion calibration profile"""
    key = profile.fusion_pattern
    self.fusion_profiles[key] = profile

def get_fusion_pattern(self, pattern: str) -> Optional[FusionCalibration]:
    """Get calibration for a fusion pattern"""
    return self.fusion_profiles.get(pattern)

def get_fusion_speedup(self, pattern: str, default: float = 1.0) -> float:
    """Get fusion speedup factor with fallback"""
    profile = self.get_fusion_pattern(pattern)
    return profile.speedup_factor if profile else default
```

### Update Serialization

Update `to_dict()` and `from_dict()` in `HardwareCalibration`:

```python
def to_dict(self) -> Dict:
    return {
        'metadata': self.metadata.to_dict(),
        'theoretical_peak_gflops': self.theoretical_peak_gflops,
        'theoretical_bandwidth_gbps': self.theoretical_bandwidth_gbps,
        # ... existing fields ...
        'operation_profiles': {
            k: v.to_dict() for k, v in self.operation_profiles.items()
        },
        'fusion_profiles': {  # NEW
            k: v.to_dict() for k, v in self.fusion_profiles.items()
        },
    }

@classmethod
def from_dict(cls, data: Dict) -> 'HardwareCalibration':
    # ... existing deserialization ...

    # Deserialize fusion profiles (NEW)
    fusion_profiles = {}
    if 'fusion_profiles' in data:
        for k, v in data['fusion_profiles'].items():
            fusion_profiles[k] = FusionCalibration.from_dict(v)

    cal.fusion_profiles = fusion_profiles
    return cal
```

## Step 2: Update Calibrator

### Add Fusion Patterns Parameter

Update `calibrate_hardware()` signature in `calibrator.py`:

```python
def calibrate_hardware(
    hardware_name: str,
    theoretical_peak_gflops: float,
    theoretical_bandwidth_gbps: float,
    output_path: Optional[Path] = None,
    operations: Optional[List[str]] = None,
    fusion_patterns: Optional[List[str]] = None,  # NEW
    quick: bool = False
) -> HardwareCalibration:
    """
    Run full hardware calibration.

    Args:
        hardware_name: Name of hardware being calibrated
        theoretical_peak_gflops: Theoretical peak GFLOPS
        theoretical_bandwidth_gbps: Theoretical memory bandwidth
        output_path: Optional path to save calibration JSON
        operations: List of operations to calibrate (None = all)
        fusion_patterns: List of fusion patterns to benchmark (NEW)
            e.g., ['linear', 'conv', 'attention'] or ['all']
        quick: If True, run faster but less comprehensive calibration

    Returns:
        Complete HardwareCalibration object
    """
```

### Add Fusion Calibration Logic

Add after matmul calibration in `calibrate_hardware()`:

```python
    # ... existing matmul calibration ...

    if 'matmul' in operations:
        print("2. Matrix Multiplication")
        # ... existing code ...
        print()

    # NEW: Fusion pattern calibration
    if fusion_patterns:
        print("3. Fused Kernel Patterns")
        print("-" * 80)

        # Expand 'all' to all patterns
        if 'all' in fusion_patterns:
            fusion_patterns = ['linear', 'conv', 'attention']

        for pattern in fusion_patterns:
            if pattern == 'linear':
                from .benchmarks.fused_linear_bench import calibrate_linear_fusion_patterns

                print("  Linear fusion patterns...")
                fusion_results = calibrate_linear_fusion_patterns(quick=quick)

                for result in fusion_results:
                    fusion_cal = FusionCalibration(
                        fusion_pattern=result['fusion_pattern'],
                        operators=result['fusion_pattern'].split('_'),
                        num_operators=len(result['fusion_pattern'].split('_')),
                        unfused_latency_ms=result['unfused_latency_ms'],
                        unfused_gflops=result['unfused_gflops'],
                        unfused_memory_bytes=result['unfused_bytes'],
                        fused_latency_ms=result['fused_latency_ms'],
                        fused_gflops=result['fused_gflops'],
                        fused_memory_bytes=result['fused_bytes'],
                        speedup_factor=result['speedup_factor'],
                        memory_reduction=result['memory_reduction'],
                        gflops_improvement=(result['fused_gflops'] - result['unfused_gflops']) / result['unfused_gflops'],
                        input_shape=result['input_shape'],
                        num_trials=50 if quick else 100,
                    )
                    calibration.add_fusion_pattern(fusion_cal)

            elif pattern == 'conv':
                from .benchmarks.fused_conv_bench import calibrate_conv_fusion_patterns

                print("  Conv fusion patterns...")
                fusion_results = calibrate_conv_fusion_patterns(quick=quick)

                for result in fusion_results:
                    fusion_cal = FusionCalibration(
                        fusion_pattern=result['fusion_pattern'],
                        operators=result['fusion_pattern'].split('_'),
                        num_operators=len(result['fusion_pattern'].split('_')),
                        unfused_latency_ms=result['unfused_latency_ms'],
                        unfused_gflops=result['unfused_gflops'],
                        unfused_memory_bytes=result['unfused_bytes'],
                        fused_latency_ms=result['fused_latency_ms'],
                        fused_gflops=result['fused_gflops'],
                        fused_memory_bytes=result['fused_bytes'],
                        speedup_factor=result['speedup_factor'],
                        memory_reduction=result['memory_reduction'],
                        gflops_improvement=(result['fused_gflops'] - result['unfused_gflops']) / result['unfused_gflops'],
                        input_shape=result['input_shape'],
                        extra_params=result.get('extra_params', {}),
                        num_trials=50 if quick else 100,
                    )
                    calibration.add_fusion_pattern(fusion_cal)

            elif pattern == 'attention':
                from .benchmarks.fused_attention_bench import calibrate_attention_fusion_patterns

                print("  Attention fusion patterns...")
                fusion_results = calibrate_attention_fusion_patterns(quick=quick)

                for result in fusion_results:
                    fusion_cal = FusionCalibration(
                        fusion_pattern=result['fusion_pattern'],
                        operators=result['fusion_pattern'].split('_'),
                        num_operators=len(result['fusion_pattern'].split('_')),
                        unfused_latency_ms=result['unfused_latency_ms'],
                        unfused_gflops=result['unfused_gflops'],
                        unfused_memory_bytes=result['unfused_bytes'],
                        fused_latency_ms=result['fused_latency_ms'],
                        fused_gflops=result['fused_gflops'],
                        fused_memory_bytes=result['fused_bytes'],
                        speedup_factor=result['speedup_factor'],
                        memory_reduction=result['memory_reduction'],
                        gflops_improvement=(result['fused_gflops'] - result['unfused_gflops']) / result['unfused_gflops'],
                        input_shape=result['input_shape'],
                        num_trials=50 if quick else 100,
                    )
                    calibration.add_fusion_pattern(fusion_cal)

        print()
```

### Update print_summary()

Add fusion summary to `HardwareCalibration.print_summary()` in schema.py:

```python
def print_summary(self):
    """Print human-readable calibration summary"""
    # ... existing summary ...

    # NEW: Fusion patterns summary
    if self.fusion_profiles:
        print("\nFusion Pattern Performance:")
        print("-" * 80)
        for pattern, profile in sorted(self.fusion_profiles.items()):
            print(f"\n{profile.fusion_pattern}:")
            print(f"  Speedup:  {profile.speedup_factor:.2f}× faster")
            print(f"  Memory:   {profile.memory_reduction*100:.1f}% reduction")
            print(f"  GFLOPS:   {profile.gflops_improvement*100:.1f}% improvement")

            if profile.speedup_factor >= 1.5:
                verdict = "✓ Strong fusion benefit"
            elif profile.speedup_factor >= 1.1:
                verdict = "✓ Moderate fusion benefit"
            elif profile.speedup_factor >= 0.95:
                verdict = "⚠ Minimal benefit"
            else:
                verdict = "✗ Fusion is slower!"

            print(f"  Verdict:  {verdict}")
```

## Step 3: CLI Updates

### Update CLI Arguments

Add to argparse in `calibrator.py` main:

```python
parser.add_argument("--fusion-patterns", type=str, default=None,
                   help="Comma-separated list of fusion patterns (linear,conv,attention,all)")
```

Update parsing:

```python
fusion_patterns = None
if args.fusion_patterns:
    fusion_patterns = [p.strip() for p in args.fusion_patterns.split(',')]

calibration = calibrate_hardware(
    hardware_name=args.hardware,
    theoretical_peak_gflops=args.peak_gflops,
    theoretical_bandwidth_gbps=args.peak_bandwidth,
    output_path=args.output,
    operations=operations,
    fusion_patterns=fusion_patterns,  # NEW
    quick=args.quick
)
```

## Step 4: Usage Examples

### Command Line

```bash
# Calibrate with all fusion patterns
python -m graphs.hardware.calibration.calibrator \
    --hardware i7-12700K \
    --peak-gflops 1000 \
    --peak-bandwidth 50 \
    --output profiles/i7_12700k_fusion.json \
    --fusion-patterns all

# Calibrate specific fusion patterns only
python -m graphs.hardware.calibration.calibrator \
    --hardware i7-12700K \
    --peak-gflops 1000 \
    --peak-bandwidth 50 \
    --fusion-patterns linear,conv

# Quick fusion calibration
python -m graphs.hardware.calibration.calibrator \
    --hardware i7-12700K \
    --peak-gflops 1000 \
    --peak-bandwidth 50 \
    --fusion-patterns all \
    --quick
```

### Python API

```python
from graphs.hardware.calibration import calibrate_hardware
from pathlib import Path

# Calibrate with fusion patterns
calibration = calibrate_hardware(
    hardware_name='i7-12700K',
    theoretical_peak_gflops=1000.0,
    theoretical_bandwidth_gbps=50.0,
    output_path=Path('profiles/i7_12700k_fusion.json'),
    fusion_patterns=['all'],
    quick=False
)

# Query fusion speedup
linear_bias_relu_speedup = calibration.get_fusion_speedup('Linear_Bias_ReLU')
print(f"Linear+Bias+ReLU fusion: {linear_bias_relu_speedup:.2f}× speedup")

# Check if fusion is beneficial
if linear_bias_relu_speedup > 1.2:
    print("Strong fusion benefit - should fuse in graph partitioner")
```

### Graph Partitioner Integration

```python
from graphs.hardware.calibration import load_calibration
from graphs.transform.partitioning import FusionPartitioner

# Load calibration profile
calibration = load_calibration('profiles/i7_12700k_fusion.json')

# Use in partitioner cost model
class CalibrationBasedCostModel:
    def __init__(self, calibration):
        self.calibration = calibration

    def should_fuse(self, op1: str, op2: str) -> bool:
        """Decide whether to fuse based on measured speedup"""
        pattern = f"{op1}_{op2}"
        speedup = self.calibration.get_fusion_speedup(pattern, default=1.0)

        # Only fuse if we get at least 10% speedup
        return speedup > 1.1

    def fusion_benefit(self, pattern: str) -> float:
        """Get measured fusion speedup"""
        return self.calibration.get_fusion_speedup(pattern, default=1.0)
```

## Step 5: Testing

### Unit Tests

Create `tests/calibration/test_fusion_integration.py`:

```python
import pytest
from graphs.hardware.calibration import calibrate_hardware
from graphs.hardware.calibration.schema import FusionCalibration

def test_fusion_calibration_linear():
    """Test linear fusion calibration"""
    cal = calibrate_hardware(
        hardware_name='TestCPU',
        theoretical_peak_gflops=1000.0,
        theoretical_bandwidth_gbps=50.0,
        fusion_patterns=['linear'],
        quick=True
    )

    # Should have Linear fusion patterns
    assert 'Linear_Bias' in cal.fusion_profiles
    assert 'Linear_Bias_ReLU' in cal.fusion_profiles

    # Check structure
    profile = cal.fusion_profiles['Linear_Bias']
    assert isinstance(profile, FusionCalibration)
    assert profile.speedup_factor > 0
    assert 0 <= profile.memory_reduction <= 1

def test_fusion_serialization():
    """Test fusion profile serialization"""
    cal = calibrate_hardware(
        hardware_name='TestCPU',
        theoretical_peak_gflops=1000.0,
        theoretical_bandwidth_gbps=50.0,
        fusion_patterns=['linear'],
        quick=True
    )

    # Save and load
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.json') as f:
        cal.save(f.name)
        loaded = HardwareCalibration.load(f.name)

    # Verify fusion profiles preserved
    assert set(loaded.fusion_profiles.keys()) == set(cal.fusion_profiles.keys())

    for key in cal.fusion_profiles:
        orig = cal.fusion_profiles[key]
        loaded_prof = loaded.fusion_profiles[key]
        assert orig.speedup_factor == loaded_prof.speedup_factor
```

## Step 6: Documentation Updates

### Update README

Add to calibration README:

```markdown
### Fusion Pattern Calibration

Benchmark fused kernel patterns to quantify fusion benefits:

# Calibrate all fusion patterns
python -m graphs.hardware.calibration.calibrator \
    --hardware i7-12700K \
    --peak-gflops 1000 \
    --peak-bandwidth 50 \
    --fusion-patterns all

**Supported Patterns:**
- `linear`: Linear + Bias + ReLU/GELU
- `conv`: Conv2d + BN + ReLU
- `attention`: Q @ K.T + Softmax + @ V

**Benefits:**
- Quantifies real-world fusion speedup
- Informs graph partitioner fusion decisions
- Hardware-specific optimization strategies
```

## Timeline

1. **Schema Extension** (30 min)
   - Add FusionCalibration dataclass
   - Add fusion_profiles field to HardwareCalibration
   - Update serialization

2. **Calibrator Integration** (45 min)
   - Add fusion_patterns parameter
   - Add fusion calibration logic
   - Update print_summary()

3. **CLI Updates** (15 min)
   - Add --fusion-patterns argument
   - Update help text

4. **Testing** (30 min)
   - Unit tests for integration
   - Manual end-to-end test

5. **Documentation** (20 min)
   - Update integration plan (this doc)
   - Update calibration README

**Total Estimate**: 2.5 hours

## Backward Compatibility

✅ **Fully backward compatible**:
- Existing calibration profiles will load correctly (fusion_profiles defaults to empty dict)
- No breaking changes to existing API
- Fusion patterns are opt-in via `--fusion-patterns` flag

## Success Criteria

- [ ] Schema extended with FusionCalibration
- [ ] Calibrator runs fusion benchmarks when requested
- [ ] CLI accepts --fusion-patterns argument
- [ ] Calibration JSON includes fusion profiles
- [ ] print_summary() shows fusion results
- [ ] Unit tests pass
- [ ] End-to-end test: calibrate i7-12700K with fusion patterns
- [ ] Documentation updated

## Next Steps After Integration

1. **GPU Fusion Benchmarks**: Extend to CUDA/cuDNN
2. **More Patterns**: Depthwise Conv, Layer Norm, residual connections
3. **Size Sensitivity**: Measure how fusion benefits vary with tensor size
4. **Partitioner Integration**: Use calibration data in FusionPartitioner cost model
5. **Auto-tuning**: Automatically select fusion threshold based on calibration

## Expected Output

After running calibration with fusion patterns:

```
================================================================================
Hardware Calibration: i7-12700K
================================================================================

1. Memory Bandwidth
--------------------------------------------------------------------------------
  ...

2. Matrix Multiplication
--------------------------------------------------------------------------------
  ...

3. Fused Kernel Patterns
--------------------------------------------------------------------------------
  Linear fusion patterns...
  Linear+Bias (128×512×512)... 2.16× speedup, 25.0% mem reduction
  Linear+Bias+ReLU (128×512×512)... 0.99× speedup, 40.0% mem reduction
  ...

  Conv fusion patterns...
  Conv+BN (B=1, 64→64, 56×56)... 1.23× speedup, 47.8% mem reduction
  ...

  Attention fusion patterns...
  QK^T (B=4, S=256, D=64)... 1.11× speedup, 57.1% mem reduction
  ...

================================================================================
Calibration Summary
================================================================================

Fusion Pattern Performance:
--------------------------------------------------------------------------------

Linear_Bias:
  Speedup:  2.16× faster
  Memory:   25.0% reduction
  GFLOPS:   166.0% improvement
  Verdict:  ✓ Strong fusion benefit

Conv_BN_ReLU:
  Speedup:  1.13× faster
  Memory:   62.8% reduction
  GFLOPS:   13.0% improvement
  Verdict:  ✓ Moderate fusion benefit

FullAttention:
  Speedup:  0.89× slower
  Memory:   75.0% reduction
  GFLOPS:   -11.0% degradation
  Verdict:  ✗ Fusion is slower!
```

This integration makes fusion calibration a first-class citizen in the framework!
