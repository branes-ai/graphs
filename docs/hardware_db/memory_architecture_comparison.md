# Memory Architecture Comparison

Visual guide showing how different memory architectures map to the unified `memory_channels` schema.

## CPU: Dual-Channel DDR5

```
┌──────────────────────────────────────┐
│         Intel i7-12700K              │
│  ┌────────────────────────────────┐  │
│  │          CPU Die               │  │
│  │   ┌─────────┐   ┌─────────┐   │  │
│  │   │ MC Ch 0 │   │ MC Ch 1 │   │  │
│  │   └────┬────┘   └────┬────┘   │  │
│  └────────┼─────────────┼─────────┘  │
│           │             │            │
│      64-bit bus    64-bit bus        │
└───────────┼─────────────┼────────────┘
            │             │
    ┌───────▼───────┐  ┌──▼──────────┐
    │   Channel 0   │  │  Channel 1  │
    ├───────────────┤  ├─────────────┤
    │ DIMM 0: 16GB  │  │ DIMM 0: 16GB│
    │ DIMM 1: 16GB  │  │ DIMM 1: 16GB│
    └───────────────┘  └─────────────┘
         32GB               32GB
      38.4 GB/s         38.4 GB/s

Total: 64GB, 76.8 GB/s, 128-bit bus
```

**JSON:**
```json
{
  "memory_channels": [
    {
      "name": "Channel 0",
      "type": "ddr5",
      "size_gb": 32,
      "data_rate_mts": 4800,
      "bus_width_bits": 64,
      "bandwidth_gbps": 38.4,
      "dimm_slots": 2,
      "dimms_populated": 2,
      "dimm_size_gb": 16
    },
    {
      "name": "Channel 1",
      "type": "ddr5",
      "size_gb": 32,
      "data_rate_mts": 4800,
      "bus_width_bits": 64,
      "bandwidth_gbps": 38.4,
      "dimm_slots": 2,
      "dimms_populated": 2,
      "dimm_size_gb": 16
    }
  ]
}
```

---

## Server CPU: 8-Channel DDR5

```
┌──────────────────────────────────────────────────────────────┐
│                    AMD EPYC 9654                             │
│  ┌────────────────────────────────────────────────────────┐  │
│  │                    CPU Die                             │  │
│  │   ┌──┐ ┌──┐ ┌──┐ ┌──┐ ┌──┐ ┌──┐ ┌──┐ ┌──┐            │  │
│  │   │C0│ │C1│ │C2│ │C3│ │C4│ │C5│ │C6│ │C7│            │  │
│  │   └┬─┘ └┬─┘ └┬─┘ └┬─┘ └┬─┘ └┬─┘ └┬─┘ └┬─┘            │  │
│  └────┼────┼────┼────┼────┼────┼────┼────┼──────────────┘  │
└───────┼────┼────┼────┼────┼────┼────┼────┼─────────────────┘
        │    │    │    │    │    │    │    │
        ▼    ▼    ▼    ▼    ▼    ▼    ▼    ▼
      DIMM DIMM DIMM DIMM DIMM DIMM DIMM DIMM
      64GB 64GB 64GB 64GB 64GB 64GB 64GB 64GB

Each channel: 64GB @ 38.4 GB/s
Total: 512GB @ 307.2 GB/s
```

---

## GPU Datacenter: HBM3 (NVIDIA H100)

```
┌────────────────────────────────────────────────────────┐
│                   H100 GPU Die                         │
│                                                        │
│  ┌──────────────────────────────────────────────┐    │
│  │            Compute Cores (SMs)               │    │
│  │                                              │    │
│  └──────┬───────┬───────┬───────┬───────┬──────┘    │
│         │       │       │       │       │            │
│    1024-bit  1024   1024   1024   1024              │
│         │       │       │       │       │            │
└─────────┼───────┼───────┼───────┼───────┼────────────┘
          │       │       │       │       │
      ┌───▼───┐ ┌─▼───┐ ┌─▼───┐ ┌─▼───┐ ┌▼────┐
      │Stack 0│ │Stk 1│ │Stk 2│ │Stk 3│ │Stk 4│
      ├───────┤ ├─────┤ ├─────┤ ├─────┤ ├─────┤
      │ 16GB  │ │16GB │ │16GB │ │16GB │ │16GB │
      │ 8 dies│ │8 die│ │8 die│ │8 die│ │8 die│
      │666GB/s│ │666  │ │666  │ │666  │ │666  │
      └───────┘ └─────┘ └─────┘ └─────┘ └─────┘

Total: 80GB @ 3,350 GB/s, 5,120-bit bus
```

**JSON:**
```json
{
  "memory_channels": [
    {
      "name": "HBM Stack 0",
      "type": "hbm3",
      "size_gb": 16,
      "data_rate_mts": 5200,
      "bus_width_bits": 1024,
      "bandwidth_gbps": 665.6,
      "dies_per_stack": 8
    },
    // ... stacks 1-4
  ]
}
```

---

## GPU Consumer: GDDR6X (NVIDIA RTX 4090)

```
┌──────────────────────────────────────────────────────────────┐
│                     RTX 4090 GPU Die                         │
│  ┌────────────────────────────────────────────────────────┐  │
│  │              Compute Cores (SMs)                       │  │
│  └──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬────────────────────┘  │
└─────┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──────────────────────┘
      │  │  │  │  │  │  │  │  │  │  │  (12 memory controllers)
     32  32 32 32 32 32 32 32 32 32 32 (bits each)
      │  │  │  │  │  │  │  │  │  │  │
      ▼  ▼  ▼  ▼  ▼  ▼  ▼  ▼  ▼  ▼  ▼
    ┌─┐┌─┐┌─┐┌─┐┌─┐┌─┐┌─┐┌─┐┌─┐┌─┐┌─┐┌─┐
    │2││2││2││2││2││2││2││2││2││2││2││2│ GB GDDR6X chips
    └─┘└─┘└─┘└─┘└─┘└─┘└─┘└─┘└─┘└─┘└─┘└─┘

Total: 24GB @ 1,008 GB/s, 384-bit bus
```

**JSON:**
```json
{
  "memory_channels": [
    {
      "name": "GDDR Bank 0",
      "type": "gddr6x",
      "size_gb": 2,
      "data_rate_mts": 21000,
      "bus_width_bits": 32,
      "bandwidth_gbps": 84.0
    },
    // ... banks 1-11 (12 total)
  ]
}
```

---

## Mobile SoC: LPDDR5 (Snapdragon 8 Gen 2)

```
┌────────────────────────────────────────┐
│     Snapdragon 8 Gen 2 SoC (PoP)       │
│  ┌──────────────────────────────────┐  │
│  │         SoC Die                  │  │
│  │  ┌────────┐      ┌────────┐     │  │
│  │  │ MC Ch0 │      │ MC Ch1 │     │  │
│  │  └───┬────┘      └───┬────┘     │  │
│  └──────┼───────────────┼──────────┘  │
│         │               │             │
└─────────┼───────────────┼─────────────┘
    32-bit│         32-bit│
      ┌───▼────┐      ┌───▼────┐
      │LPDDR5X │      │LPDDR5X │
      │  6GB   │      │  6GB   │
      │34.1GB/s│      │34.1GB/s│
      └────────┘      └────────┘
        (PoP)           (PoP)

Total: 12GB @ 68.2 GB/s, 64-bit bus
Package-on-Package (stacked on SoC)
```

**JSON:**
```json
{
  "memory_channels": [
    {
      "name": "LPDDR Channel 0",
      "type": "lpddr5x",
      "size_gb": 6,
      "data_rate_mts": 8533,
      "bus_width_bits": 32,
      "bandwidth_gbps": 34.1,
      "package_on_package": true
    },
    {
      "name": "LPDDR Channel 1",
      "type": "lpddr5x",
      "size_gb": 6,
      "data_rate_mts": 8533,
      "bus_width_bits": 32,
      "bandwidth_gbps": 34.1,
      "package_on_package": true
    }
  ]
}
```

---

## Key Differences Summary

| Architecture | Unit | Bus Width | Count | Capacity Range | Bandwidth Range |
|--------------|------|-----------|-------|----------------|-----------------|
| **CPU DDR** | Channel with DIMMs | 64 bits | 1-8 channels | 8-1024 GB | 20-400 GB/s |
| **GPU HBM** | Memory Stack | 1024 bits | 4-12 stacks | 16-192 GB | 1,600-6,000 GB/s |
| **GPU GDDR** | Memory Controller | 32 bits | 6-16 controllers | 8-24 GB | 400-1,000 GB/s |
| **Mobile LPDDR** | Channel (PoP) | 32 bits | 2-4 channels | 4-24 GB | 30-100 GB/s |

## Bandwidth Calculation Formula

```
Bandwidth (GB/s) = (Data Rate MT/s × Bus Width bits) / 8 / 1000

Examples:
- DDR5-4800, 64-bit: (4800 × 64) / 8 / 1000 = 38.4 GB/s
- HBM3-5200, 1024-bit: (5200 × 1024) / 8 / 1000 = 665.6 GB/s
- GDDR6X-21000, 32-bit: (21000 × 32) / 8 / 1000 = 84.0 GB/s
```

## Usage in Mappers

```python
# Get memory configuration
channels = spec.get_memory_channels()

for ch in channels:
    if ch.type.startswith('ddr'):
        # CPU DDR - NUMA aware
        if ch.ecc_enabled:
            # Account for ECC overhead (~1-2% bandwidth loss)
            effective_bw = ch.bandwidth_gbps * 0.98

    elif ch.type.startswith('hbm'):
        # GPU HBM - very high bandwidth
        # Use for large batch sizes
        if ch.bandwidth_gbps > 500:
            # Favor memory-intensive operations
            pass

    elif ch.type.startswith('gddr'):
        # GPU GDDR - moderate bandwidth
        # Balance compute and memory
        pass

    elif ch.type.startswith('lpddr'):
        # Mobile - power constrained
        # Optimize for power efficiency
        pass
```

## Design Rationale

1. **Unified Structure**: Same schema handles DDR, HBM, GDDR, LPDDR
2. **Per-Channel Details**: Know exact configuration of each channel/stack
3. **Asymmetric Support**: Can represent mixed configurations (rare but possible)
4. **Technology-Specific Fields**: Optional fields for DDR (DIMMs), HBM (dies), mobile (PoP)
5. **Validation**: Totals can be computed and checked
6. **Mapper-Friendly**: Detailed info for optimization decisions
