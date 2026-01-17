# Session: Validation Test Fixes & KPU Energy Model Integration

**Date**: 2025-11-12
**Duration**: ~2 hours
**Focus**: Fix CI validation test failures, add TOPS/W metrics, unify KPU energy models

---

## Summary

Fixed all validation test failures (2 FAILED, 8 ERROR) identified in CI. Enhanced KPU tile energy test with TOPS/W metrics for easier hardware comparison. Unified KPU mappers to consistently use tile energy models from resource definitions.

**Final Result**: ✅ **22 passed, 0 failed, 0 errors** (13.18s runtime)

---

## Session Timeline

### 1. Add TOPS/W Calculations to KPU Tile Energy Test

**User Request**: "can we also add the computation for TOPS/W to the test_kpu_tile_energy script? that makes it easier to compare and contrast"

**Problem**: Test output showed energy breakdowns but lacked efficiency metrics (TOPS/W) for comparing hardware variants.

**Solution**:
- Enhanced `print_energy_breakdown()` function to calculate:
  - Peak throughput (TOPS) from clock × ops/cycle × tiles × utilization
  - Execution time from total_ops / peak_throughput
  - Average power from total_energy / execution_time
  - TOPS/W efficiency metric
- Modified `KPUTileEnergyModel.compute_gemm_energy()` to return hardware config:
  - `clock_frequency_hz`
  - `num_tiles`
  - `ops_per_cycle_per_tile` (via new `_get_ops_per_cycle()` helper)

**Results**:
```
T64 @ 6W:   2.55 TOPS/W (6.9 TOPS peak, 2.7W avg)
T256 @ 30W: 1.68 TOPS/W (50.3 TOPS peak, 29.9W avg)
T768 @ 60W: 2.37 TOPS/W (130.1 TOPS peak, 54.9W avg)
```

**Files Modified**:
- `tests/hardware/test_kpu_tile_energy.py` (enhanced print function)
- `src/graphs/hardware/architectural_energy.py` (added config fields, helper method)

---

### 2. Fix KPU Mapper Inconsistency

**User Observation**: "In kpu.py, I still only see a KPUTileEnergyModel for the T256, I am not seeing it for the T64 nor the T768"

**Problem**:
- T256 mapper was creating duplicate `KPUTileEnergyModel` (35 lines)
- T64 and T768 mappers weren't using tile energy models at all
- Inconsistent architectural energy modeling across KPU variants

**Solution**:
Unified all three KPU mappers to follow same pattern:
```python
def create_kpu_tXXX_mapper(thermal_profile: str = None) -> KPUMapper:
    from ...models.accelerators.kpu_tXXX import kpu_tXXX_resource_model
    from ...architectural_energy import KPUTileEnergyAdapter

    model = kpu_tXXX_resource_model()
    model.architecture_energy_model = KPUTileEnergyAdapter(model.tile_energy_model)
    return KPUMapper(model, thermal_profile=thermal_profile)
```

**Impact**:
- T256: Removed 35 lines of duplicate code
- T64 & T768: Now use architectural energy adapter
- All KPU variants consistently use tile_energy_model from resource definitions

**Files Modified**:
- `src/graphs/hardware/mappers/accelerators/kpu.py` (lines 505-565)

---

### 3. Investigate CI Validation Test Failures

**User Report**: "CI is getting failed validation tests, can you investigate"

**Investigation Results**:

Found 10 test failures in two categories:

**Category 1: FAILURES (2 tests)**
1. `validation/hardware/test_all_hardware.py::test_all_hardware` - FAILED
   - Error: `ValueError: Thermal profile '12W' not found. Available: ['3W', '6W', '10W']`
   - Cause: Test using outdated thermal profile names

2. `validation/analysis/test_operator_edp_comprehensive.py::test_architectural_modifiers` - FAILED
   - Error: `AssertionError: Should have Conv2d operators`
   - Cause: All operators showing as type 'Unknown'

**Category 2: ERRORs (8 tests)**
All had same root cause: `fixture 'name' not found`
- `validation/analysis/test_building_blocks_edp.py::test_building_block` - ERROR
- `validation/analysis/test_building_blocks_simple.py::test_building_block` - ERROR
- `validation/estimators/test_enhanced_attention_fusion_complete.py::test_model` - ERROR
- `validation/estimators/test_vit_automatic_decomposition.py::test_vit_model` - ERROR
- `validation/hardware/test_embodied_ai_comparison.py::test_hardware_on_workload` - ERROR
- `validation/hardware/test_ip_core_mappers.py::test_mapper` - ERROR
- `validation/hardware/test_new_dsp_mappers.py::test_mapper` - ERROR
- `validation/hardware/test_reid_comparison.py::test_hardware` - ERROR

---

### 4. Fix FAILURE Tests

#### Fix 1: test_architectural_modifiers (COMPLETED ✅)

**Root Cause**:
- `SubgraphDescriptor.operation_type` field was `OperationType.UNKNOWN` for all subgraphs
- But `node_name` field had correct values: 'conv2d', 'batch_norm', 'relu_', etc.
- `get_operator_edp_breakdown()` was grouping by operation_type enum → all became 'Unknown'

**Solution**:
Added operator type inference from FX node names as fallback:

```python
def _infer_op_type_from_node_name(self, node_name: str) -> str:
    """Infer operator type from FX node name."""
    name_lower = node_name.lower().rstrip('_').rstrip('0123456789')

    name_mapping = {
        'conv2d': 'Conv2d',
        'batch_norm': 'BatchNorm2d',
        'relu': 'ReLU',
        'max_pool2d': 'MaxPool2d',
        'linear': 'Linear',
        'add': 'Add',
        'matmul': 'MatMul',
        'softmax': 'Softmax',
        # ... 20+ mappings total
    }

    return name_mapping.get(name_lower, node_name)
```

Modified `get_operator_edp_breakdown()` to use fallback:
```python
op_type_str = self._operation_type_to_string(sg_desc.operation_type)

# Fallback: If operation_type is UNKNOWN, try to infer from node_name
if op_type_str == 'Unknown' and sg_desc.node_name:
    op_type_str = self._infer_op_type_from_node_name(sg_desc.node_name)
```

**Test Result**: ✅ PASSED
- Found 1 Conv2d operator correctly
- Found 1 BatchNorm2d operator correctly

**Files Modified**:
- `src/graphs/analysis/architecture_comparator.py` (lines 1209, 1314-1337)

---

#### Fix 2: test_all_hardware (COMPLETED ✅)

**Root Cause**:
Test was using outdated thermal profile names that don't match user's updated KPU resource models:
- Test expected: T64 @ 12W, 24W; T256 @ 12.5W, 25W
- Actual profiles: T64 @ 3W, 6W, 10W; T256 @ 15W, 30W, 50W

**Solution**:
Updated test to use correct thermal profiles and tile counts:

```python
# BEFORE (WRONG):
"Stillwater KPU-T64 @ 12W (70/20/10)": create_kpu_t64_mapper(thermal_profile="12W"),
"Stillwater KPU-T256 @ 12.5W (210/60/30)": create_kpu_t256_mapper(thermal_profile="12.5W"),

# AFTER (CORRECT):
"Stillwater KPU-T64 @ 3W (44/13/7)": create_kpu_t64_mapper(thermal_profile="3W"),
"Stillwater KPU-T64 @ 6W (44/13/7)": create_kpu_t64_mapper(thermal_profile="6W"),
"Stillwater KPU-T64 @ 10W (44/13/7)": create_kpu_t64_mapper(thermal_profile="10W"),
"Stillwater KPU-T256 @ 15W (179/51/26)": create_kpu_t256_mapper(thermal_profile="15W"),
"Stillwater KPU-T256 @ 30W (179/51/26)": create_kpu_t256_mapper(thermal_profile="30W"),
"Stillwater KPU-T256 @ 50W (179/51/26)": create_kpu_t256_mapper(thermal_profile="50W"),
```

**Test Result**: ✅ PASSED

**Files Modified**:
- `validation/hardware/test_all_hardware.py` (lines 105-121)

---

### 5. Fix ERROR Tests (Parametrization Issues)

**Root Cause Analysis**:
All 8 ERROR tests had the same issue:
- Functions named `test_something()` triggered pytest collection
- Pytest treated them as test functions and looked for fixtures
- But they were actually **helper functions** that take custom parameters
- These files are **standalone scripts** meant to be run directly: `python script.py`

**General Solution Pattern**:
1. Rename helper functions with underscore prefix: `test_foo()` → `_test_foo_helper()`
2. Update all function calls to use new name
3. Scripts still work when run directly, but pytest correctly ignores them

---

#### Fix 3: test_building_blocks_edp.py & test_building_blocks_simple.py (COMPLETED ✅)

**Problem**: `test_building_block(model, input_data, model_name)` causing fixture errors

**Solution**:
1. Renamed: `test_building_block()` → `_test_building_block_helper()`
2. Updated 4 calls in each file (lines 248, 264, 280, 297)
3. **Additional fix for test_building_blocks_edp.py**: Added missing parameters to ArchitectureComparator:
   ```python
   comparator = ArchitectureComparator(
       model_name=name,
       architectures=architectures,
       batch_size=input_tensor.shape[0],
       precision=Precision.FP32,
       model=model,                    # Added
       input_tensor=input_tensor       # Added
   )
   ```

**Test Results**: ✅ PASSED (both files)
- All 4 building blocks validated: MLP, Conv2D, ResNet, Attention
- Subgraph EDP breakdown working correctly
- Scripts produce detailed analysis reports

**Files Modified**:
- `validation/analysis/test_building_blocks_edp.py` (lines 132, 158-165, 248, 264, 280, 297)
- `validation/analysis/test_building_blocks_simple.py` (lines 170, 275, 288, 301, 314)

---

#### Fix 4: test_enhanced_attention_fusion_complete.py (COMPLETED ✅)

**Problem**: `test_model(model, input_data, model_name)` causing fixture error

**Solution**:
1. Renamed: `test_model()` → `_test_model_helper()` (line 267)
2. Updated call site (line 358)

**Verification**: Pytest now collects 0 tests from this file (correct behavior)

**Files Modified**:
- `validation/estimators/test_enhanced_attention_fusion_complete.py` (lines 267, 358)

---

#### Fix 5: test_vit_automatic_decomposition.py (COMPLETED ✅)

**Problem**: `test_vit_model(model_factory, model_name, input_shape)` causing fixture error

**Solution**:
1. Renamed: `test_vit_model()` → `_test_vit_model_helper()` (line 141)
2. Updated call site (lines 344-348)

**Verification**: Pytest now collects 0 tests from this file (correct behavior)

**Files Modified**:
- `validation/estimators/test_vit_automatic_decomposition.py` (lines 141, 344-348)

---

#### Fix 6: test_embodied_ai_comparison.py (COMPLETED ✅)

**Problem**: `test_hardware_on_workload(hardware_name, thermal_profile, workload_name, workload_info)` causing fixture error

**Solution**:
1. Renamed: `test_hardware_on_workload()` → `_test_hardware_on_workload_helper()` (lines 216-221)
2. Updated call site (lines 358-360)

**Verification**: Pytest now collects 0 tests from this file (correct behavior)

**Files Modified**:
- `validation/hardware/test_embodied_ai_comparison.py` (lines 216-221, 358-360)

---

#### Fix 7: test_ip_core_mappers.py (COMPLETED ✅)

**Problem**: `test_mapper(mapper_name, mapper_factory, precision='int8')` causing fixture error

**Solution**:
1. Renamed: `test_mapper()` → `_test_mapper_helper()` (line 40)
2. Updated 3 call sites using replace_all (lines 133, 149, 165):
   - `ceva_report, ceva_mapper = _test_mapper_helper(...)`
   - `cadence_report, cadence_mapper = _test_mapper_helper(...)`
   - `synopsys_report, synopsys_mapper = _test_mapper_helper(...)`

**Verification**: Pytest now collects 0 tests from this file (correct behavior)

**Files Modified**:
- `validation/hardware/test_ip_core_mappers.py` (lines 40, 133, 149, 165)

---

#### Fix 8: test_new_dsp_mappers.py (COMPLETED ✅)

**Problem**: `test_mapper(mapper_name, mapper_factory, precision='int8')` causing fixture error

**Solution**:
1. Renamed: `test_mapper()` → `_test_mapper_helper()` (line 43)
2. Updated 7 call sites using replace_all (lines 130-220):
   - QRB5165, SA8775P, RIDE, TDA4VM, TDA4VL, TDA4AL, TDA4VH

**Verification**: Pytest now collects 0 tests from this file (correct behavior)

**Files Modified**:
- `validation/hardware/test_new_dsp_mappers.py` (lines 43, 130-220)

---

#### Fix 9: test_reid_comparison.py (COMPLETED ✅)

**Problem**: `test_hardware(hardware_name, mapper_func)` causing fixture error

**Solution**:
1. Renamed: `test_hardware()` → `_test_hardware_helper()` (line 35)
2. Updated call site (line 127)

**Verification**:
- Pytest now collects 0 tests from this file (correct behavior)
- Script runs correctly: `python validation/hardware/test_reid_comparison.py`
- Output shows Re-ID comparison across 5 hardware platforms

**Files Modified**:
- `validation/hardware/test_reid_comparison.py` (lines 35, 127)

---

## Final Validation

**Test Command**:
```bash
python3 -m pytest validation/ -v
```

**Results**:
```
======================== 22 passed, 26 warnings in 13.18s =======================
```

**Test Breakdown**:
- ✅ `validation/analysis/test_operator_edp_comprehensive.py` (6 tests including test_architectural_modifiers)
- ✅ `validation/analysis/test_phase1_mapper_integration.py` (1 test)
- ✅ `validation/analysis/test_phase2_operator_edp.py` (1 test)
- ✅ `validation/analysis/test_power_management_reporting.py` (1 test)
- ✅ `validation/estimators/test_attention_fusion_patterns.py` (1 test)
- ✅ `validation/hardware/test_all_hardware.py` (1 test)
- ✅ `validation/hardware/test_ampere_ampereone.py` (1 test)
- ✅ `validation/hardware/test_cgra_mapper.py` (1 test)
- ✅ `validation/hardware/test_cpu_vs_gpu_mapping.py` (1 test)
- ✅ `validation/hardware/test_dpu_mapper.py` (1 test)
- ✅ `validation/hardware/test_embodied_quick.py` (1 test)
- ✅ `validation/hardware/test_gpu_cpu_kpu_comparison.py` (1 test)
- ✅ `validation/hardware/test_hardware_mapping.py` (1 test)
- ✅ `validation/hardware/test_performance_characteristics.py` (3 tests)
- ✅ `validation/test_latency_sanity.py` (1 test)

**Standalone Scripts** (correctly excluded from pytest):
- `validation/analysis/test_building_blocks_edp.py` (run directly: ✅ works)
- `validation/analysis/test_building_blocks_simple.py` (run directly: ✅ works)
- `validation/estimators/test_enhanced_attention_fusion_complete.py`
- `validation/estimators/test_vit_automatic_decomposition.py`
- `validation/hardware/test_embodied_ai_comparison.py`
- `validation/hardware/test_ip_core_mappers.py`
- `validation/hardware/test_new_dsp_mappers.py`
- `validation/hardware/test_reid_comparison.py` (verified: ✅ works)

---

## Files Modified Summary

### Source Code (4 files)

1. **`src/graphs/analysis/architecture_comparator.py`**
   - Added `_infer_op_type_from_node_name()` helper method (lines 1314-1337)
   - Modified `get_operator_edp_breakdown()` to use node_name fallback (line 1209)

2. **`src/graphs/hardware/architectural_energy.py`**
   - Enhanced `compute_gemm_energy()` return dict with hardware config fields (line ~2409)
   - Added `_get_ops_per_cycle()` helper method (lines 2439-2450)

3. **`src/graphs/hardware/mappers/accelerators/kpu.py`**
   - Unified T64, T256, T768 mappers to use tile_energy_model from resource models
   - Removed 35 lines of duplicate code in T256 mapper
   - Lines 505-565

4. **`tests/hardware/test_kpu_tile_energy.py`**
   - Enhanced `print_energy_breakdown()` with TOPS/W calculations (lines ~170-195)

### Validation Tests (9 files)

5. **`validation/hardware/test_all_hardware.py`**
   - Updated KPU thermal profile names (lines 105-121)

6. **`validation/analysis/test_building_blocks_edp.py`**
   - Renamed helper function + updated 4 calls (lines 132, 158-165, 248, 264, 280, 297)

7. **`validation/analysis/test_building_blocks_simple.py`**
   - Renamed helper function + updated 4 calls (lines 170, 275, 288, 301, 314)

8. **`validation/estimators/test_enhanced_attention_fusion_complete.py`**
   - Renamed helper function + updated 1 call (lines 267, 358)

9. **`validation/estimators/test_vit_automatic_decomposition.py`**
   - Renamed helper function + updated 1 call (lines 141, 344-348)

10. **`validation/hardware/test_embodied_ai_comparison.py`**
    - Renamed helper function + updated 1 call (lines 216-221, 358-360)

11. **`validation/hardware/test_ip_core_mappers.py`**
    - Renamed helper function + updated 3 calls (lines 40, 133, 149, 165)

12. **`validation/hardware/test_new_dsp_mappers.py`**
    - Renamed helper function + updated 7 calls (lines 43, 130-220)

13. **`validation/hardware/test_reid_comparison.py`**
    - Renamed helper function + updated 1 call (lines 35, 127)

---

## Key Insights

### Pytest Function Naming Convention

**Problem**: Functions starting with `test_` are automatically collected as pytest tests, even if they're helper functions.

**Solution**: Use underscore prefix for helper functions: `_test_helper()` or `_helper()`

**Best Practice**:
- Pytest test functions: `def test_something():` (no parameters or use @pytest.mark.parametrize)
- Helper functions: `def _test_something_helper():` or just `def _something_helper():`
- Standalone scripts: Either use helper naming or don't use pytest at all

### PyTorch FX Operation Type Detection

**Problem**: `operation_type` enum often returns `UNKNOWN` for FX nodes.

**Solution**: Use `node_name` as fallback - it contains the actual operation name as a string.

**Pattern**:
```python
# Primary: Try enum
op_type_str = self._operation_type_to_string(sg_desc.operation_type)

# Fallback: Use node_name if enum is UNKNOWN
if op_type_str == 'Unknown' and sg_desc.node_name:
    op_type_str = self._infer_op_type_from_node_name(sg_desc.node_name)
```

### KPU Architectural Energy Model Integration

**Pattern**: All KPU mappers should follow this structure:
```python
def create_kpu_mapper(thermal_profile: str = None) -> KPUMapper:
    # 1. Import resource model and adapter
    from ...models.accelerators.kpu import kpu_resource_model
    from ...architectural_energy import KPUTileEnergyAdapter

    # 2. Create resource model (includes tile_energy_model)
    model = kpu_resource_model()

    # 3. Wrap tile energy model with adapter
    model.architecture_energy_model = KPUTileEnergyAdapter(model.tile_energy_model)

    # 4. Create mapper
    return KPUMapper(model, thermal_profile=thermal_profile)
```

**Benefit**: Single source of truth for tile energy models in resource model files.

---

## Next Steps

### Immediate
- ✅ All validation tests passing
- ✅ KPU energy models unified
- ✅ TOPS/W metrics available for hardware comparison

### Future Enhancements
- Consider fixing `operation_type` at FX tracing level (populate enum correctly)
- Add TOPS/W metrics to other hardware test scripts
- Create unified test report generator for hardware comparisons

---

## Session Artifacts

**Documentation**:
- `CHANGELOG.md` - Added entry for 2025-11-12
- `docs/sessions/2025-11-12_validation_test_fixes.md` - This session log

**Test Evidence**:
```bash
# Full validation suite
python3 -m pytest validation/ -v
# Result: 22 passed, 0 failed, 0 errors (13.18s)

# Standalone scripts (examples)
python validation/hardware/test_reid_comparison.py  # ✅ works
python validation/analysis/test_building_blocks_edp.py  # ✅ works
```

**Before/After**:
- Before: 2 FAILED, 8 ERROR tests blocking CI
- After: 22 PASSED, 0 FAILED, 0 ERROR tests ✅

---

## Conclusion

Successfully fixed all CI validation test failures through systematic investigation and targeted fixes. Enhanced KPU testing infrastructure with efficiency metrics (TOPS/W) and unified architectural energy modeling across all KPU variants. All 22 validation tests now pass cleanly, with standalone scripts correctly excluded from pytest collection but still functional when run directly.

**Key Achievement**: Zero test failures - validation suite is now fully green ✅
