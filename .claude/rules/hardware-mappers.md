---
paths:
  - "src/graphs/hardware/**/*.py"
---

# Hardware Mapper Rules

- Factory functions: `create_<chip>_<formfactor>_<memory>_mapper()`
- All mappers inherit from `HardwareMapper` base class
- Register every new mapper in `hardware/mappers/__init__.py` `_init_registry()`
- Resource model parameters must come from datasheets (cite source in comments)
- Utilization must be in range [0.0, 1.0]
- Do not hardcode precision support -- use the `HardwareResourceModel` precision profiles
- Each mapper must handle at least FP32; FP16/INT8 are optional but preferred
- GPU mappers: account for wave quantization and warp scheduling
- CPU mappers: account for SIMD width and core allocation
- Accelerator mappers: account for architecture-specific constraints (systolic dimensions, tile sizes)
