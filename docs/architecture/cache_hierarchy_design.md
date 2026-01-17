# Enhanced On-Chip Memory Hierarchy Design

## Problem Statement

Current schema doesn't capture:
1. **Scope/Sharing**: L1/L2 are per-core, L3 is shared
2. **Size per unit**: Total L1 = cores × per-core size
3. **Heterogeneous cores**: P-cores and E-cores have different L1/L2 sizes
4. **Cache inclusivity**: Inclusive/exclusive/non-inclusive
5. **Partitioning**: L3 slice-based organization

## Key Requirements

1. Hardware mappers need to know:
   - How much cache per core for allocation
   - Which caches are shared vs private
   - Total available cache at each level
   
2. Support heterogeneous cores:
   - Different L1/L2 for P-cores vs E-cores
   - Different associativity, line sizes

3. Support various topologies:
   - Intel: per-core L1/L2, sliced shared L3
   - AMD: per-core L1/L2, per-CCX L3, shared L3
   - ARM: per-cluster L2, shared L3
   - GPU: per-SM L1, shared L2

## Proposed Design

### Option 1: Structured Cache Levels (Recommended)

```python
@dataclass
class CacheLevel:
    """Single cache level configuration"""
    
    # Identity
    name: str
    """Cache name: 'L1 dcache', 'L1 icache', 'L2', 'L3', 'L4'"""
    
    level: int
    """Cache level: 1, 2, 3, 4"""
    
    cache_type: str
    """Type: 'data', 'instruction', 'unified'"""
    
    # Topology and Sharing
    scope: str
    """
    Scope of sharing:
    - 'per_core': Private to each core (L1, often L2)
    - 'per_cluster': Shared within core cluster (ARM L2, AMD CCX L3)
    - 'shared': Shared across all cores (L3, L4)
    """
    
    # Sizing
    size_per_unit_kb: Optional[int] = None
    """
    Size per unit (core/cluster) in KB.
    For scope='per_core': size per core
    For scope='per_cluster': size per cluster
    For scope='shared': None (use total_size_kb)
    """
    
    total_size_kb: Optional[int] = None
    """
    Total size across all units in KB.
    For per_core: computed as size_per_unit_kb × core_count
    For shared: actual total capacity
    """
    
    # Organization
    associativity: Optional[int] = None
    """N-way set associativity (8, 12, 16, etc.)"""
    
    line_size_bytes: Optional[int] = None
    """Cache line size in bytes (typically 64)"""
    
    sets: Optional[int] = None
    """Number of sets (for detailed modeling)"""
    
    # Advanced Properties
    inclusivity: Optional[str] = None
    """
    Inclusivity policy:
    - 'inclusive': Contains copies of lower-level caches
    - 'exclusive': No overlap with lower levels (victim cache)
    - 'non_inclusive': May or may not contain lower-level data
    """
    
    coherence_protocol: Optional[str] = None
    """Coherence protocol: 'MESI', 'MOESI', 'MESIF'"""
    
    write_policy: Optional[str] = None
    """Write policy: 'write_back', 'write_through'"""
    
    # Partitioning (for shared caches)
    partitioned: Optional[bool] = None
    """Whether cache is physically partitioned (sliced)"""
    
    slices: Optional[int] = None
    """Number of slices/partitions (for shared L3)"""
    
    # Per-cluster variation (for heterogeneous cores)
    cluster_name: Optional[str] = None
    """
    If this cache config applies to specific cluster.
    None = applies to all cores
    'Performance Cores' = only P-cores
    'Efficiency Cores' = only E-cores
    """


@dataclass
class OnChipMemoryHierarchy:
    """Complete on-chip memory hierarchy specification"""
    
    cache_levels: Optional[List[Dict]] = None
    """
    Detailed cache level configurations.
    Use get_cache_levels() to access as CacheLevel objects.
    """
    
    # Simple aggregate fields (backward compatibility)
    l1_dcache_kb: Optional[int] = None
    """L1 dcache per core (deprecated: use cache_levels)"""
    
    l1_icache_kb: Optional[int] = None
    """L1 icache per core (deprecated: use cache_levels)"""
    
    # ... other fields ...
    
    def get_cache_levels(self) -> List[CacheLevel]:
        """Get cache levels as CacheLevel objects"""
        if not self.cache_levels:
            return []
        return [CacheLevel.from_dict(c) for c in self.cache_levels]
    
    def get_cache_by_level(self, level: int, cache_type: str = 'unified') -> Optional[CacheLevel]:
        """Get specific cache level"""
        for cache in self.get_cache_levels():
            if cache.level == level and (cache.cache_type == cache_type or cache.cache_type == 'unified'):
                return cache
        return None
    
    def get_per_core_caches(self) -> List[CacheLevel]:
        """Get all per-core caches"""
        return [c for c in self.get_cache_levels() if c.scope == 'per_core']
    
    def get_shared_caches(self) -> List[CacheLevel]:
        """Get all shared caches"""
        return [c for c in self.get_cache_levels() if c.scope == 'shared']
    
    def compute_total_cache_kb(self, core_count: int) -> int:
        """Compute total on-chip cache capacity"""
        total = 0
        for cache in self.get_cache_levels():
            if cache.scope == 'per_core' and cache.size_per_unit_kb:
                total += cache.size_per_unit_kb * core_count
            elif cache.total_size_kb:
                total += cache.total_size_kb
        return total
```

## Example: Intel i7-12700K (Alder Lake)

```json
{
  "cache_levels": [
    {
      "name": "L1 dcache (P-cores)",
      "level": 1,
      "cache_type": "data",
      "scope": "per_core",
      "cluster_name": "Performance Cores",
      "size_per_unit_kb": 48,
      "total_size_kb": 384,
      "associativity": 12,
      "line_size_bytes": 64,
      "sets": 64,
      "write_policy": "write_back"
    },
    {
      "name": "L1 icache (P-cores)",
      "level": 1,
      "cache_type": "instruction",
      "scope": "per_core",
      "cluster_name": "Performance Cores",
      "size_per_unit_kb": 32,
      "total_size_kb": 256,
      "associativity": 8,
      "line_size_bytes": 64
    },
    {
      "name": "L1 dcache (E-cores)",
      "level": 1,
      "cache_type": "data",
      "scope": "per_core",
      "cluster_name": "Efficiency Cores",
      "size_per_unit_kb": 32,
      "total_size_kb": 128,
      "associativity": 8,
      "line_size_bytes": 64
    },
    {
      "name": "L1 icache (E-cores)",
      "level": 1,
      "cache_type": "instruction",
      "scope": "per_core",
      "cluster_name": "Efficiency Cores",
      "size_per_unit_kb": 64,
      "total_size_kb": 256,
      "associativity": 8,
      "line_size_bytes": 64
    },
    {
      "name": "L2 (P-cores)",
      "level": 2,
      "cache_type": "unified",
      "scope": "per_core",
      "cluster_name": "Performance Cores",
      "size_per_unit_kb": 1280,
      "total_size_kb": 10240,
      "associativity": 10,
      "line_size_bytes": 64,
      "inclusivity": "non_inclusive",
      "write_policy": "write_back"
    },
    {
      "name": "L2 (E-cores)",
      "level": 2,
      "cache_type": "unified",
      "scope": "per_core",
      "cluster_name": "Efficiency Cores",
      "size_per_unit_kb": 2048,
      "total_size_kb": 8192,
      "associativity": 16,
      "line_size_bytes": 64,
      "inclusivity": "non_inclusive"
    },
    {
      "name": "L3 (LLC)",
      "level": 3,
      "cache_type": "unified",
      "scope": "shared",
      "total_size_kb": 25600,
      "associativity": 12,
      "line_size_bytes": 64,
      "inclusivity": "non_inclusive",
      "coherence_protocol": "MESIF",
      "partitioned": true,
      "slices": 12
    }
  ]
}
```

## Example: AMD EPYC 9654 (Zen 4)

```json
{
  "cache_levels": [
    {
      "name": "L1 dcache",
      "level": 1,
      "cache_type": "data",
      "scope": "per_core",
      "size_per_unit_kb": 32,
      "total_size_kb": 3072,
      "associativity": 8,
      "line_size_bytes": 64,
      "write_policy": "write_back"
    },
    {
      "name": "L1 icache",
      "level": 1,
      "cache_type": "instruction",
      "scope": "per_core",
      "size_per_unit_kb": 32,
      "total_size_kb": 3072,
      "associativity": 8,
      "line_size_bytes": 64
    },
    {
      "name": "L2",
      "level": 2,
      "cache_type": "unified",
      "scope": "per_core",
      "size_per_unit_kb": 1024,
      "total_size_kb": 98304,
      "associativity": 8,
      "line_size_bytes": 64,
      "inclusivity": "non_inclusive"
    },
    {
      "name": "L3 (per CCD)",
      "level": 3,
      "cache_type": "unified",
      "scope": "per_cluster",
      "size_per_unit_kb": 32768,
      "total_size_kb": 393216,
      "associativity": 16,
      "line_size_bytes": 64,
      "inclusivity": "non_inclusive",
      "coherence_protocol": "MOESI",
      "slices": 8
    }
  ]
}
```

## Example: NVIDIA H100 (GPU)

```json
{
  "cache_levels": [
    {
      "name": "L1 data cache / Shared Memory",
      "level": 1,
      "cache_type": "unified",
      "scope": "per_core",
      "size_per_unit_kb": 256,
      "total_size_kb": 33792,
      "associativity": 4,
      "line_size_bytes": 128,
      "write_policy": "write_back"
    },
    {
      "name": "L2",
      "level": 2,
      "cache_type": "unified",
      "scope": "shared",
      "total_size_kb": 51200,
      "associativity": 16,
      "line_size_bytes": 128,
      "partitioned": true,
      "slices": 60
    }
  ]
}
```

## Benefits

1. **Hardware Mapper Clarity**: Mappers can query per-core vs shared caches
2. **Heterogeneous Core Support**: Different cache sizes for P/E-cores
3. **Accurate Totals**: Computed correctly from per-core × count
4. **Extensible**: Can add L4, victim caches, etc.
5. **Backward Compatible**: Keep simple fields, add detailed structure optionally

## Migration Path

1. Keep existing simple fields for backward compatibility
2. Add optional `cache_levels` for detailed topology
3. Auto-generate simple fields from `cache_levels` if not provided
4. Hardware mappers use `cache_levels` when available, fall back to simple fields

