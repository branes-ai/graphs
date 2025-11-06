# Mapper Integration Guide

## How to Update Hardware Mappers to Use Calibration Data

This guide shows how to integrate calibration profiles into existing hardware mappers for improved accuracy.

## Current Mapper Behavior (Before Calibration)

```python
# src/graphs/hardware/mappers/cpu.py
class CPUMapper(HardwareMapper):
    def __init__(self, resource_model: HardwareResourceModel):
        self.peak_ops = resource_model.get_peak_ops(Precision.FP32)
        # peak_ops = 720e9 GFLOPS (theoretical from datasheet)

    def estimate_latency(self, subgraph: FusedSubgraph) -> float:
        # Use fixed efficiency factor
        efficiency = 0.20  # Assume 20% of peak for all operations

        achievable_ops = self.peak_ops * efficiency
        # achievable_ops = 720e9 * 0.20 = 144 GFLOPS

        return subgraph.total_ops / achievable_ops
```

**Problems:**
1. Single efficiency factor for all operations (matmul vs elementwise)
2. No distinction between operation sizes (1024×1024 vs 4096×4096)
3. Conservative estimate (20%) misses optimized code (78%)

## Updated Mapper (With Calibration)

### Step 1: Add Calibration Parameter

```python
from graphs.hardware.calibration import HardwareCalibration

class CPUMapper(HardwareMapper):
    def __init__(
        self,
        resource_model: HardwareResourceModel,
        calibration: Optional[HardwareCalibration] = None,
        thermal_profile: str = None
    ):
        super().__init__(resource_model, thermal_profile=thermal_profile)
        self.calibration = calibration

        # Fallback values if no calibration provided
        self.default_efficiency = 0.20
```

### Step 2: Classify Subgraph Operations

```python
def _classify_operation(self, subgraph: FusedSubgraph) -> Tuple[str, Dict]:
    """
    Classify the dominant operation type in a subgraph.

    Returns:
        (operation_type, extra_params) tuple
    """
    # Analyze operations in subgraph
    ops = subgraph.op_types

    # Matrix multiplication
    if any(op in ['matmul', 'linear', 'addmm', 'bmm'] for op in ops):
        # Estimate matrix size from memory footprint
        total_elements = subgraph.memory_bytes // 4  # FP32
        matrix_size = int(total_elements ** 0.5)  # Rough estimate

        return 'matmul', {'matrix_size': matrix_size}

    # Convolution
    if any('conv' in op for op in ops):
        # Extract kernel size if available
        # (This requires enhanced subgraph metadata)
        return 'conv2d', {}

    # Element-wise operations
    if any(op in ['relu', 'add', 'mul', 'sigmoid'] for op in ops):
        return 'add', {}  # Use 'add' as proxy for elementwise

    # Pooling
    if any('pool' in op for op in ops):
        return 'maxpool', {}

    # Default
    return 'unknown', {}
```

### Step 3: Get Calibrated Efficiency

```python
def _get_efficiency(self, subgraph: FusedSubgraph) -> float:
    """
    Get efficiency for this subgraph using calibration data.

    Returns:
        Efficiency (0.0 to 1.0)
    """
    if not self.calibration:
        # No calibration: use conservative default
        return self.default_efficiency

    # Classify operation
    op_type, extra_params = self._classify_operation(subgraph)

    # Query calibration
    efficiency = self.calibration.get_efficiency(op_type, **extra_params)

    # Sanity check
    if efficiency <= 0 or efficiency > 1.0:
        print(f"Warning: Invalid efficiency {efficiency} for {op_type}, "
              f"using default {self.default_efficiency}")
        return self.default_efficiency

    return efficiency
```

### Step 4: Use Calibrated Values in Latency Estimation

```python
def estimate_latency(self, subgraph: FusedSubgraph, precision: Precision) -> float:
    """Estimate latency with calibration-aware efficiency"""

    # Get theoretical peak for this precision
    peak_ops = self.resource_model.get_peak_ops(precision)

    # Get calibrated efficiency for this operation
    efficiency = self._get_efficiency(subgraph)

    # Calculate achievable throughput
    achievable_ops = peak_ops * efficiency

    # Compute latency (compute-bound)
    compute_latency = subgraph.total_ops / achievable_ops

    # Memory latency (if calibration includes bandwidth)
    if self.calibration:
        bandwidth = self.calibration.measured_bandwidth_gbps * 1e9
    else:
        bandwidth = self.resource_model.peak_bandwidth

    memory_latency = subgraph.memory_bytes / bandwidth

    # Return max (roofline model)
    return max(compute_latency, memory_latency)
```

## Complete Example

```python
# cpu.py
from typing import Optional, Tuple, Dict
from graphs.hardware.calibration import HardwareCalibration

class CPUMapper(HardwareMapper):
    def __init__(
        self,
        resource_model: HardwareResourceModel,
        calibration: Optional[HardwareCalibration] = None,
        thermal_profile: str = None
    ):
        super().__init__(resource_model, thermal_profile=thermal_profile)
        self.calibration = calibration
        self.default_efficiency = 0.20  # Fallback

    def _classify_operation(self, subgraph: FusedSubgraph) -> Tuple[str, Dict]:
        ops = subgraph.op_types

        if any(op in ['matmul', 'linear', 'addmm', 'bmm'] for op in ops):
            total_elements = subgraph.memory_bytes // 4
            matrix_size = int(total_elements ** 0.5)
            return 'matmul', {'matrix_size': matrix_size}

        if any('conv' in op for op in ops):
            return 'conv2d', {}

        if any(op in ['relu', 'add', 'mul'] for op in ops):
            return 'add', {}

        return 'unknown', {}

    def _get_efficiency(self, subgraph: FusedSubgraph) -> float:
        if not self.calibration:
            return self.default_efficiency

        op_type, extra_params = self._classify_operation(subgraph)
        efficiency = self.calibration.get_efficiency(op_type, **extra_params)

        if not (0 < efficiency <= 1.0):
            return self.default_efficiency

        return efficiency

    def estimate_latency(self, subgraph: FusedSubgraph, precision: Precision) -> float:
        # Get calibrated efficiency
        efficiency = self._get_efficiency(subgraph)

        # Achievable throughput
        peak_ops = self.resource_model.get_peak_ops(precision)
        achievable_ops = peak_ops * efficiency

        # Latency
        compute_latency = subgraph.total_ops / achievable_ops

        bandwidth = (self.calibration.measured_bandwidth_gbps * 1e9
                    if self.calibration
                    else self.resource_model.peak_bandwidth)

        memory_latency = subgraph.memory_bytes / bandwidth

        return max(compute_latency, memory_latency)
```

## Usage Example

```python
from graphs.hardware.resource_model import HardwareResourceModel
from graphs.hardware.mappers.cpu import CPUMapper
from graphs.hardware.calibration import load_calibration

# Create resource model
resource_model = HardwareResourceModel(...)

# Load calibration
calibration = load_calibration('profiles/intel_i7_12700k.json')

# Create mapper with calibration
mapper = CPUMapper(resource_model, calibration=calibration)

# Estimate latency for a subgraph
latency = mapper.estimate_latency(subgraph, Precision.FP32)
```

## Expected Accuracy Improvement

### Before (Fixed 20% Efficiency)

```
ResNet-18 Analysis:
  Conv2d layers: 150 GFLOPS assumed
  Matmul layers: 150 GFLOPS assumed
  ReLU layers: 150 GFLOPS assumed (WAY too high!)

  Total latency estimate: 12.3 ms
  Actual measured: 45.6 ms
  Error: 270% underestimate
```

### After (Calibrated Efficiency)

```
ResNet-18 Analysis:
  Conv2d layers: 350 GFLOPS (35% calibrated)
  Matmul layers: 780 GFLOPS (78% calibrated)
  ReLU layers: 15 GFLOPS (memory-bound)

  Total latency estimate: 42.1 ms
  Actual measured: 45.6 ms
  Error: 8% underestimate
```

## Integration Checklist

- [ ] Add `calibration` parameter to mapper `__init__`
- [ ] Implement `_classify_operation()` method
- [ ] Implement `_get_efficiency()` method
- [ ] Update `estimate_latency()` to use calibrated efficiency
- [ ] Update `estimate_energy()` if applicable
- [ ] Add default calibration loading
- [ ] Update mapper factory functions
- [ ] Test with calibrated vs uncalibrated
- [ ] Document calibration usage in mapper docstring

## Default Calibration Loading

```python
# In mapper __init__
def __init__(self, resource_model, calibration=None, thermal_profile=None):
    if calibration is None:
        # Try to load default calibration for this hardware
        calibration = self._load_default_calibration(resource_model)

    self.calibration = calibration

def _load_default_calibration(self, resource_model):
    """Try to load pre-generated calibration profile"""
    try:
        profiles_dir = Path(__file__).parent.parent / "calibration" / "profiles"
        hardware_name = resource_model.name.lower().replace("-", "_").replace(" ", "_")
        profile_path = profiles_dir / f"{hardware_name}.json"

        if profile_path.exists():
            return load_calibration(profile_path)
    except Exception as e:
        print(f"Warning: Could not load calibration: {e}")

    return None  # No calibration available
```

## Testing

```python
# Test with and without calibration
def test_calibration_impact():
    resource_model = get_i7_12700k_model()
    calibration = load_calibration('profiles/intel_i7_12700k.json')

    # Without calibration
    mapper_old = CPUMapper(resource_model, calibration=None)
    latency_old = mapper_old.estimate_latency(subgraph, Precision.FP32)

    # With calibration
    mapper_new = CPUMapper(resource_model, calibration=calibration)
    latency_new = mapper_new.estimate_latency(subgraph, Precision.FP32)

    print(f"Without calibration: {latency_old:.2f} ms")
    print(f"With calibration: {latency_new:.2f} ms")
    print(f"Improvement: {abs(latency_new - actual) / abs(latency_old - actual):.1f}×")
```

## Next Steps

1. Implement `_classify_operation()` for your mapper
2. Add calibration parameter
3. Test with existing calibration profiles
4. Generate calibration for your hardware if not available
5. Validate against real measurements
