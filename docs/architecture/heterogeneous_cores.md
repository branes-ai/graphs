# Heterogeneous Core Architecture Support

The hardware database schema supports detailed specifications for heterogeneous CPU architectures, including Intel P/E-cores and ARM big.LITTLE configurations.

## Overview

Modern CPUs often have multiple types of cores with different characteristics:

- **Intel Hybrid Architecture (12th gen+)**: Performance cores (P-cores) with Hyper-Threading + Efficiency cores (E-cores) without HT
- **ARM big.LITTLE**: Prime/Big cores for performance + Little cores for efficiency
- **ARM DynamIQ**: Up to 3 core types (Prime, Performance, Efficiency)

The `core_clusters` field allows you to specify detailed information for each homogeneous cluster of cores.

## Schema Design

### Simple Fields (Backward Compatible)

For homogeneous CPUs or basic specifications:
```json
{
  "cores": 8,
  "threads": 16,
  "base_frequency_ghz": 3.6,
  "boost_frequency_ghz": 5.0
}
```

### Core Clusters (Heterogeneous CPUs)

For detailed heterogeneous specifications:
```json
{
  "cores": 12,
  "threads": 20,
  "core_clusters": [
    {
      "name": "P-core",
      "type": "performance",
      "count": 8,
      "architecture": "Golden Cove",
      "base_frequency_ghz": 3.6,
      "boost_frequency_ghz": 5.0,
      "has_hyperthreading": true,
      "simd_width_bits": 256
    },
    {
      "name": "E-core",
      "type": "efficiency",
      "count": 4,
      "architecture": "Gracemont",
      "base_frequency_ghz": 2.7,
      "boost_frequency_ghz": 3.8,
      "has_hyperthreading": false,
      "simd_width_bits": 256
    }
  ]
}
```

## CoreCluster Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | Yes | Cluster name: "P-core", "E-core", "Prime", "Big", "Little" |
| `type` | string | Yes | Cluster type: "performance", "efficiency", "balanced" |
| `count` | int | Yes | Number of cores in this cluster |
| `architecture` | string | No | Core microarchitecture: "Golden Cove", "Cortex-X3", etc. |
| `base_frequency_ghz` | float | No | Base/sustained frequency for this cluster |
| `boost_frequency_ghz` | float | No | Maximum boost frequency for this cluster |
| `has_hyperthreading` | bool | No (default: true) | Whether cores support SMT/HT |
| `simd_width_bits` | int | No | SIMD width (256 for AVX2, 512 for AVX512, 128 for NEON) |

## Examples

### Intel Core i7-12700K (Alder Lake)

**Specification**:
- 8 P-cores (Golden Cove) @ 3.6-5.0 GHz with HT = 16 threads
- 4 E-cores (Gracemont) @ 2.7-3.8 GHz without HT = 4 threads
- Total: 12 cores, 20 threads

**JSON** (see `docs/hardware_db/core_clusters_example.json`):
```json
{
  "cores": 12,
  "threads": 20,
  "core_clusters": [
    {
      "name": "P-core",
      "type": "performance",
      "count": 8,
      "architecture": "Golden Cove",
      "base_frequency_ghz": 3.6,
      "boost_frequency_ghz": 5.0,
      "has_hyperthreading": true,
      "simd_width_bits": 256
    },
    {
      "name": "E-core",
      "type": "efficiency",
      "count": 4,
      "architecture": "Gracemont",
      "base_frequency_ghz": 2.7,
      "boost_frequency_ghz": 3.8,
      "has_hyperthreading": false,
      "simd_width_bits": 256
    }
  ]
}
```

### Qualcomm Snapdragon 8 Gen 2 (ARM DynamIQ)

**Specification**:
- 1 Prime core (Cortex-X3) @ 2.0-3.2 GHz
- 4 Performance cores (Cortex-A715) @ 1.8-2.8 GHz
- 3 Efficiency cores (Cortex-A510) @ 1.0-2.0 GHz
- Total: 8 cores, 8 threads (ARM cores don't have SMT)

**JSON** (see `docs/hardware_db/core_clusters_arm_example.json`):
```json
{
  "cores": 8,
  "threads": 8,
  "core_clusters": [
    {
      "name": "Prime",
      "type": "performance",
      "count": 1,
      "architecture": "Cortex-X3",
      "base_frequency_ghz": 2.0,
      "boost_frequency_ghz": 3.2,
      "has_hyperthreading": false,
      "simd_width_bits": 128
    },
    {
      "name": "Performance",
      "type": "performance",
      "count": 4,
      "architecture": "Cortex-A715",
      "base_frequency_ghz": 1.8,
      "boost_frequency_ghz": 2.8,
      "has_hyperthreading": false,
      "simd_width_bits": 128
    },
    {
      "name": "Efficiency",
      "type": "efficiency",
      "count": 3,
      "architecture": "Cortex-A510",
      "base_frequency_ghz": 1.0,
      "boost_frequency_ghz": 2.0,
      "has_hyperthreading": false,
      "simd_width_bits": 128
    }
  ]
}
```

## API Usage

The `HardwareSpec` class provides helper methods for working with core clusters:

```python
from graphs.hardware.database import HardwareSpec

# Load a spec with core clusters
spec = HardwareSpec.from_json("intel_i7_12700k.json")

# Check if heterogeneous
if spec.has_heterogeneous_cores():
    print("This CPU has heterogeneous cores")

# Get cluster objects
clusters = spec.get_core_clusters()
for cluster in clusters:
    print(f"{cluster.name}: {cluster.count} cores")
    print(f"  Frequency: {cluster.base_frequency_ghz}-{cluster.boost_frequency_ghz} GHz")
    print(f"  Threads: {cluster.total_threads()}")

# Compute totals
total_cores = spec.compute_total_cores()      # 12
total_threads = spec.compute_total_threads()  # 20
max_boost = spec.get_max_boost_frequency()    # 5.0 GHz
```

## Validation

The schema validates that:

1. Each cluster has required fields (`name`, `type`, `count`)
2. Core counts are positive
3. If both `cores` and `core_clusters` are specified, they match
4. If both `threads` and `core_clusters` are specified, they match

Example validation error:
```
cores (16) doesn't match core_clusters total (12).
Update cores to match cluster total or remove it.
```

## Migration Guide

### Existing Simple Specs (No Change Needed)

If your CPU has homogeneous cores, no changes are needed:
```json
{
  "cores": 8,
  "threads": 16,
  "base_frequency_ghz": 3.6,
  "boost_frequency_ghz": 4.8
}
```

### Upgrading to Core Clusters

For hybrid CPUs, add the `core_clusters` field:

**Before**:
```json
{
  "cores": 12,
  "threads": 20,
  "e_cores": 4,
  "base_frequency_ghz": 3.6,
  "boost_frequency_ghz": 5.0
}
```

**After**:
```json
{
  "cores": 12,
  "threads": 20,
  "e_cores": 4,  // deprecated but kept for backward compatibility
  "core_clusters": [
    {
      "name": "P-core",
      "type": "performance",
      "count": 8,
      "base_frequency_ghz": 3.6,
      "boost_frequency_ghz": 5.0,
      "has_hyperthreading": true
    },
    {
      "name": "E-core",
      "type": "efficiency",
      "count": 4,
      "base_frequency_ghz": 2.7,
      "boost_frequency_ghz": 3.8,
      "has_hyperthreading": false
    }
  ]
}
```

## Use Cases for Mappers

Hardware mappers can use core cluster information to:

1. **Prefer performance cores** for compute-intensive kernels
2. **Use efficiency cores** for background tasks or low-priority work
3. **Account for frequency differences** in latency estimation
4. **Consider different SIMD widths** per cluster
5. **Handle different thread counts** per cluster for parallelism analysis

Example mapper usage:
```python
def map_to_hardware(self, subgraph, spec: HardwareSpec):
    if spec.has_heterogeneous_cores():
        clusters = spec.get_core_clusters()

        # Prefer P-cores for heavy compute
        if subgraph.is_compute_bound():
            p_cores = [c for c in clusters if c.type == "performance"]
            return self._map_to_cluster(subgraph, p_cores[0])

        # Use E-cores for light tasks
        else:
            e_cores = [c for c in clusters if c.type == "efficiency"]
            if e_cores:
                return self._map_to_cluster(subgraph, e_cores[0])
```

## Future Extensions

Possible future additions:
- Per-cluster cache configurations
- Per-cluster power characteristics
- Cluster interconnect topology
- Core affinity hints
- NUMA node assignments
