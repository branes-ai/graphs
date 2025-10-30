# TPU Performance Fix - Implementation Plan

## Root Cause

The UnifiedAnalyzer's roofline analysis uses naive peak FLOPS division:
```python
compute_time = sg.flops / self.peak_flops  # Assumes full chip utilization!
```

This bypasses the hardware mapper's sophisticated logic:
- GPU mapper: Sequential execution mode, discrete SM allocation (2-48 SMs)
- TPU mapper: Sequential array mode, discrete MXU allocation (1-2 arrays)

## Architecture Problem

```
ArchitectureComparator
  └─> UnifiedAnalyzer.analyze_model_with_custom_hardware(hardware_mapper)
      └─> _run_roofline_analysis(hardware_mapper.resource_model)  # Only passes resource model!
          └─> RooflineAnalyzer(resource_model)
              └─> compute_time = flops / peak_flops  # Naive!
```

**What we need:**
```
ArchitectureComparator
  └─> UnifiedAnalyzer.analyze_model_with_custom_hardware(hardware_mapper)
      └─> _run_roofline_analysis_with_mapper(hardware_mapper, partition_report)  # Pass mapper!
          └─> mapper.map_graph(...) or mapper.compute_sequential_latency(...)
              └─> Uses discrete resource allocation logic
```

## Solution Options

### Option 1: Extend RooflineAnalyzer to use Hardware Mapper (RECOMMENDED)
Modify RooflineAnalyzer to optionally accept a hardware mapper and delegate latency calculation to it.

**Pros:**
- Preserves existing architecture
- Backward compatible (mapper is optional)
- Leverages all mapper sophistication

**Cons:**
- Requires coordination between RooflineAnalyzer and mappers

### Option 2: Bypass RooflineAnalyzer when mapper is available
When hardware_mapper is provided to UnifiedAnalyzer, call mapper.map_graph() directly instead of RooflineAnalyzer.

**Pros:**
- Simple, direct
- Uses mapper's native output format

**Cons:**
- Duplicates latency calculation logic
- May break existing code expecting RooflineReport format

### Option 3: Make HardwareMapper.map_graph() return RooflineReport
Standardize mapper output to RooflineReport format.

**Pros:**
- Clean separation of concerns
- All mappers produce consistent output

**Cons:**
- Large refactoring
- Changes mapper API

## Recommended Implementation: Option 1

### Step 1: Modify UnifiedAnalyzer._run_roofline_analysis

**Before:**
```python
def _run_roofline_analysis(
    self,
    partition_report: PartitionReport,
    hardware: HardwareResourceModel,
    precision: Precision
) -> RooflineReport:
    analyzer = RooflineAnalyzer(hardware, precision=precision)
    return analyzer.analyze(partition_report.subgraphs, partition_report)
```

**After:**
```python
def _run_roofline_analysis(
    self,
    partition_report: PartitionReport,
    hardware: HardwareResourceModel,
    precision: Precision,
    hardware_mapper: Optional[Any] = None,  # NEW: optional mapper
    batch_size: int = 1  # NEW
) -> RooflineReport:
    analyzer = RooflineAnalyzer(hardware, precision=precision, hardware_mapper=hardware_mapper)
    return analyzer.analyze(partition_report.subgraphs, partition_report, batch_size=batch_size)
```

### Step 2: Modify RooflineAnalyzer to use mapper when available

**Add to __init__:**
```python
def __init__(
    self,
    resource_model: HardwareResourceModel,
    precision: Precision = Precision.FP32,
    hardware_mapper: Optional[Any] = None  # NEW
):
    self.hardware_mapper = hardware_mapper
    # ... rest of init
```

**Modify analyze():**
```python
def analyze(
    self,
    subgraphs: List[SubgraphDescriptor],
    partition_report: Optional[PartitionReport] = None,
    batch_size: int = 1  # NEW
) -> RooflineReport:

    # If mapper is available and supports sophisticated latency calculation, use it
    if self.hardware_mapper and hasattr(self.hardware_mapper, 'compute_sequential_latency'):
        return self._analyze_with_mapper(subgraphs, partition_report, batch_size)

    # Otherwise, fall back to naive roofline model
    return self._analyze_with_roofline_model(subgraphs, partition_report)
```

**Add new method:**
```python
def _analyze_with_mapper(
    self,
    subgraphs: List[SubgraphDescriptor],
    partition_report: Optional[PartitionReport],
    batch_size: int
) -> RooflineReport:
    """Use hardware mapper's latency calculation"""

    # Convert PartitionReport to FusionReport (mapper's expected input)
    fusion_report = self._convert_to_fusion_report(partition_report)

    # Call mapper's sequential latency calculation
    total_latency, allocations = self.hardware_mapper.compute_sequential_latency(
        fusion_report, self.precision
    )

    # Convert allocations to LatencyDescriptors
    latencies = [
        LatencyDescriptor(
            subgraph_id=alloc.subgraph_id,
            subgraph_name=alloc.subgraph_name,
            compute_time=alloc.compute_time,
            memory_time=alloc.memory_time,
            actual_latency=alloc.estimated_latency,
            bottleneck=alloc.bottleneck,
            # ... map all fields
        )
        for alloc in allocations
    ]

    # Build RooflineReport from allocations
    return RooflineReport(..., latencies=latencies, total_latency=total_latency)
```

### Step 3: Update UnifiedAnalyzer.analyze_model_with_custom_hardware

```python
def analyze_model_with_custom_hardware(
    self,
    model: nn.Module,
    input_tensor: torch.Tensor,
    model_name: str,
    hardware_mapper: Any,
    precision: Precision = Precision.FP32,
    config: Optional[AnalysisConfig] = None,
) -> UnifiedAnalysisResult:
    hardware = hardware_mapper.resource_model
    batch_size = input_tensor.shape[0]

    # ... trace and partition ...

    # Step 2: Run roofline analysis (NOW WITH MAPPER!)
    if config.run_roofline:
        result.roofline_report = self._run_roofline_analysis(
            partition_report,
            hardware,
            precision,
            hardware_mapper=hardware_mapper,  # PASS MAPPER!
            batch_size=batch_size  # PASS BATCH SIZE!
        )
```

## Expected Results After Fix

**Before (naive):**
- TPU v4 on ResNet18: 0.0999 ms latency, 10,009 FPS
- Assumes 95% utilization of 275 TFLOPS chip

**After (with fix):**
- TPU v4 on ResNet18: ~2-3 ms latency, ~333-500 FPS
- Uses 1 MXU sequentially, 50% array utilization
- Similar to KPU (461 µs) and GPU (431 µs)

## Testing Plan

1. Test TPU with ResNet18 (should show realistic latency)
2. Test GPU (should still work with sequential mode)
3. Test KPU (verify no regression)
4. Test with batch_size > 1 (should use parallel mode)
5. Validate all export formats (JSON, CSV, HTML)
