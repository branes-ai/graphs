---
name: hardware-mapper
description: Guides development of new hardware architecture mappers. Use when adding support for a new chip, accelerator, or hardware platform to the mapper registry.
argument-hint: "[chip-name] [architecture-type]"
---

# New Hardware Mapper Development

Add a hardware mapper for: $ARGUMENTS

## Mandatory Workflow

### Phase 1: Architecture Classification

Classify the target hardware using the taxonomy in `docs/hardware/architecture_taxonomy.md`:

| Property | Value |
|----------|-------|
| Execution model | (MIMD, SIMT, Systolic, Spatial Dataflow, VLIW, FPGA) |
| Closest existing mapper | (CPU, GPU, DSP, TPU, KPU, DPU, CGRA) |
| Key differentiator | (what makes this hardware unique) |

Determine: Can this hardware be modeled as a variant of an existing mapper (preferred) or does it need a new mapper class?

### Phase 2: Resource Model Parameters

Gather from datasheets or vendor documentation:

```python
HardwareResourceModel(
    name="<chip_formfactor_memory>",
    peak_gflops_fp32=...,      # FP32 peak compute
    peak_gflops_fp16=...,      # FP16 peak compute (if supported)
    peak_gflops_int8=...,      # INT8 peak compute (if supported)
    memory_bandwidth_gb_s=..., # Peak memory bandwidth
    total_memory_bytes=...,    # Total on-chip/accessible memory
    l2_cache_bytes=...,        # L2 cache size
    tdp_watts=...,             # Thermal design power
    # Architecture-specific:
    num_compute_units=...,     # SMs, cores, PEs, tiles, etc.
    compute_unit_size=...,     # Threads/warps/lanes per unit
)
```

### Phase 3: Mapper Implementation

If extending an existing mapper class:
1. Create factory function `create_<chip>_<formfactor>_<memory>_mapper()` in the appropriate file
2. Follow naming convention: `{Architecture}-{FormFactor}-{Memory}`

If creating a new mapper class:
1. Create `src/graphs/hardware/mappers/<type>.py` (or under `accelerators/`)
2. Inherit from `HardwareMapper` base class
3. Implement required methods:
   - `map(subgraph, precision) -> HardwareAllocation`
   - Hardware-specific resource allocation logic
4. Create factory functions for specific chips

### Phase 4: Registry Integration

In `hardware/mappers/__init__.py`:
1. Add factory function to `_init_registry()` in the `_MAPPER_REGISTRY`
2. Include metadata: category, vendor, TDP range
3. Verify with `list_all_mappers()` and `get_mapper_by_name()`

### Phase 5: Validation

1. Create `validation/hardware/test_<chip>_mapper.py`
2. Test basic mapping with a simple model (MLP or Conv2D)
3. Verify resource allocation is reasonable (utilization 0-100%)
4. Compare against similar hardware in same category
5. Run `validation/hardware/test_all_hardware.py` to verify integration

## Reference: Naming Convention

```
Datacenter GPU:  create_h100_sxm5_80gb_mapper()
Edge GPU:        create_jetson_orin_agx_64gb_mapper()
CPU:             create_intel_xeon_platinum_8490h_mapper()
DSP:             create_qualcomm_sa8775p_mapper()
Accelerator:     create_hailo8_mapper()
```
