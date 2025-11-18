# Cache Levels Implementation Status

## Completed ✅

### 1. Schema Updates (schema.py)

**CacheLevel Dataclass** - Lines 304-463
- Complete topology-aware cache level specification
- Required fields: name, level, cache_type, scope
- Scope types: per_core, per_cluster, shared
- Advanced fields added:
  - inclusivity, coherence_protocol, write_policy, replacement_policy
  - prefetcher_enabled, prefetcher_type, prefetch_distance
  - partitioned, slices (for shared L3)
  - banks, read_ports, write_ports
  - hit_latency_cycles, miss_latency_cycles
  - cluster_name (for heterogeneous cores)

**OnChipMemoryHierarchy Dataclass** - Lines 466-622
- cache_levels: List[Dict] = field(default_factory=list) (REQUIRED)
- Helper methods:
  - get_cache_levels() → List[CacheLevel]
  - get_cache_by_level(level, cache_type, cluster)
  - get_per_core_caches(cluster)
  - get_shared_caches()
  - get_per_cluster_caches()
  - compute_total_cache_kb(core_count)
  - get_available_cache_for_core(cluster_name)

**Migration Logic** - Lines 1063-1151
- Backward compatibility: old flat fields → cache_levels
- Assumptions for old data:
  - L1 dcache/icache: per_core
  - L2: per_core
  - L3: shared
- Populates both cache_levels and simple fields

### 2. Testing

**Migration Test**
- ✅ Old flat fields successfully migrate to cache_levels
- ✅ Proper scope assignment (per_core vs shared)
- ✅ Helper methods work correctly
- ✅ Available cache per core: {1: 256, 2: 2048, 3: 16384}

## Remaining Work

### 3. Detector Updates (detector.py)

**DetectedCPU** - Need to add:
```python
@dataclass
class DetectedCacheLevel:
    """Detected cache level info"""
    level: int
    cache_type: str  # 'data', 'instruction', 'unified'
    size_kb: int
    associativity: Optional[int] = None
    line_size_bytes: Optional[int] = None
    # TODO: detect scope (per_core vs shared) from cpuinfo or lscpu
```

**_extract_cache_info()** - Need to:
- Generate cache_levels structure
- Determine scope (may require parsing lscpu or /sys/devices/system/cpu)
- Handle heterogeneous cores (P/E-cores have different L1/L2)

### 4. Auto-Detect Script Updates (auto_detect_and_add.py)

**HardwareSpec Creation** - Need to:
- Generate cache_levels from DetectedCPU
- For heterogeneous CPUs:
  - Create separate L1/L2 entries per cluster (P-cores, E-cores)
  - Link to cluster_name from core_clusters
- Set proper scope based on detection

**Example for i7-12700K:**
```python
cache_levels = [
    {
        "name": "L1 dcache (P-cores)",
        "level": 1,
        "cache_type": "data",
        "scope": "per_core",
        "cluster_name": "Performance Cores",
        "size_per_unit_kb": 48,  # P-cores have 48 KB
        "total_size_kb": 384,     # 48 KB × 8 P-cores
        "associativity": 12,
        "line_size_bytes": 64
    },
    {
        "name": "L1 dcache (E-cores)",
        "level": 1,
        "cache_type": "data",
        "scope": "per_core",
        "cluster_name": "Efficiency Cores",
        "size_per_unit_kb": 32,  # E-cores have 32 KB
        "total_size_kb": 128,    # 32 KB × 4 E-cores
        "associativity": 8,
        "line_size_bytes": 64
    },
    # ... L1 icache for P/E, L2 for P/E, shared L3
]
```

### 5. Hardware Database Examples

Need to create reference examples:
- `docs/hardware_db/cache_levels_example_i7_12700k.json` (heterogeneous)
- `docs/hardware_db/cache_levels_example_epyc_9654.json` (homogeneous)
- `docs/hardware_db/cache_levels_example_h100.json` (GPU)

## Design Decisions

### Scope Types

1. **per_core**: Private to each core
   - L1 dcache, L1 icache (always)
   - L2 (modern CPUs - Intel, AMD, ARM)
   - No snooping needed among cores

2. **per_cluster**: Shared within cluster
   - ARM big.LITTLE L2 (shared within big or little cluster)
   - AMD CCX L3 (shared within CCX, not across CCXs)
   - Multi-socket NUMA L3

3. **shared**: Shared across all cores
   - L3 LLC (most modern CPUs)
   - L4 (if present)
   - May be sliced/partitioned

### Heterogeneous Core Support

- Link via `cluster_name` field
- Same cluster_name as in core_clusters array
- Examples:
  - Intel 12th+ gen: "Performance Cores", "Efficiency Cores"
  - ARM big.LITTLE: "Big", "Little", "Prime"

### Hardware Mapper Usage

```python
# Get L1 dcache for P-cores
l1d_pcore = hierarchy.get_cache_by_level(1, 'data', cluster='Performance Cores')
per_core_kb = l1d_pcore.size_per_unit_kb  # 48 KB

# Get shared L3
l3 = hierarchy.get_shared_caches()[0]
total_shared_kb = l3.total_size_kb  # 25600 KB

# Get all available cache for a core
available = hierarchy.get_available_cache_for_core('Performance Cores')
# {1: 48, 2: 1280, 3: 25600}
```

## Next Steps

1. Update detector.py to generate cache_levels
2. Update auto_detect_and_add.py to create proper cache_levels
3. Test full end-to-end with i7-12700K detection
4. Create example JSON files
5. Update documentation

## Files Modified

- `src/graphs/hardware/database/schema.py`
  - Added CacheLevel dataclass (304-463)
  - Updated OnChipMemoryHierarchy (466-622)
  - Enhanced migration logic (1063-1151)

- `docs/hardware_db/cache_hierarchy_design.md` (new)
  - Complete design documentation
  - Examples for Intel, AMD, NVIDIA

