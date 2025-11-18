# Memory Subsystem Design Proposal

## Problem Statement

Current schema has flat memory fields:
```json
{
  "memory_type": "DDR5",
  "memory_channels": 2,
  "memory_bus_width": 128,
  "peak_bandwidth_gbps": 76.8
}
```

This doesn't capture:
- **CPU**: Multiple channels with DIMMs (size, speed, MT/s per channel)
- **GPU Datacenter**: HBM stacks (dies per stack, bandwidth per stack)
- **GPU Consumer**: GDDR6X configuration (multiple memory controllers)
- **Mobile/Embedded**: LPDDR channels (often asymmetric)

## Proposed Solution: `memory_channels` Array

Similar to `core_clusters`, create structured memory channel/stack descriptions.

## MemoryChannel Dataclass

```python
@dataclass
class MemoryChannel:
    """
    Description of a memory channel, stack, or bank.

    Handles:
    - CPU: DDR4/DDR5 channels with DIMMs
    - GPU Datacenter: HBM2e/HBM3 stacks
    - GPU Consumer: GDDR6/GDDR6X controllers
    - Mobile: LPDDR4/LPDDR5 channels
    """

    name: str
    """
    Channel identifier.
    CPU: "Channel 0", "Channel 1"
    GPU HBM: "HBM Stack 0", "HBM Stack 1"
    GPU GDDR: "GDDR Bank 0", "GDDR Bank 1"
    Mobile: "LPDDR Channel 0"
    """

    type: str
    """
    Memory technology.
    "ddr5", "ddr4", "hbm3", "hbm2e", "gddr6x", "gddr6", "lpddr5", "lpddr4x"
    """

    # Capacity
    size_gb: float
    """Size of this channel/stack in GB"""

    # Frequency/Speed
    frequency_mhz: int
    """Memory clock frequency in MHz"""

    data_rate_mts: int
    """Data rate in MT/s (megatransfers per second)"""

    # Bus Configuration
    bus_width_bits: int
    """
    Bus width in bits.
    DDR: 64 bits per channel (72 with ECC)
    HBM: 1024 bits per stack
    GDDR6X: 32 bits per controller (typical)
    """

    bandwidth_gbps: float
    """
    Theoretical peak bandwidth for this channel/stack.
    Formula: (data_rate_mts × bus_width_bits) / 8 / 1000
    """

    # CPU DDR-specific
    dimm_slots: Optional[int] = None
    """Number of DIMM slots in this channel (CPU only)"""

    dimms_populated: Optional[int] = None
    """Number of DIMMs actually installed (CPU only)"""

    dimm_size_gb: Optional[int] = None
    """Size per DIMM in GB (CPU only)"""

    ecc_enabled: Optional[bool] = None
    """ECC support enabled (CPU/Server only)"""

    rank_count: Optional[int] = None
    """Ranks per DIMM (1=single-rank, 2=dual-rank, 4=quad-rank)"""

    # GPU HBM-specific
    dies_per_stack: Optional[int] = None
    """Number of dies per HBM stack (typically 8 or 16)"""

    stack_height: Optional[int] = None
    """HBM stack height in mm"""

    # Mobile-specific
    package_on_package: Optional[bool] = None
    """PoP configuration (mobile/embedded)"""
```

## Example 1: CPU with DDR5 (Dual Channel)

**Intel i7-12700K with 64GB DDR5-4800**

```json
{
  "memory_type": "DDR5",
  "peak_bandwidth_gbps": 76.8,

  "memory_channels": [
    {
      "name": "Channel 0",
      "type": "ddr5",
      "size_gb": 32,
      "frequency_mhz": 2400,
      "data_rate_mts": 4800,
      "bus_width_bits": 64,
      "bandwidth_gbps": 38.4,
      "dimm_slots": 2,
      "dimms_populated": 2,
      "dimm_size_gb": 16,
      "ecc_enabled": false,
      "rank_count": 1
    },
    {
      "name": "Channel 1",
      "type": "ddr5",
      "size_gb": 32,
      "frequency_mhz": 2400,
      "data_rate_mts": 4800,
      "bus_width_bits": 64,
      "bandwidth_gbps": 38.4,
      "dimm_slots": 2,
      "dimms_populated": 2,
      "dimm_size_gb": 16,
      "ecc_enabled": false,
      "rank_count": 1
    }
  ]
}
```

**Computation:**
- Total capacity: 32 + 32 = **64 GB**
- Total bandwidth: 38.4 + 38.4 = **76.8 GB/s**
- Total bus width: 64 + 64 = **128 bits**

## Example 2: Server CPU with DDR5 (Quad Channel)

**AMD EPYC 9654 with 512GB DDR5-4800, 8 channels**

```json
{
  "memory_type": "DDR5",
  "peak_bandwidth_gbps": 307.2,

  "memory_channels": [
    {
      "name": "Channel 0",
      "type": "ddr5",
      "size_gb": 64,
      "frequency_mhz": 2400,
      "data_rate_mts": 4800,
      "bus_width_bits": 64,
      "bandwidth_gbps": 38.4,
      "dimm_slots": 2,
      "dimms_populated": 1,
      "dimm_size_gb": 64,
      "ecc_enabled": true,
      "rank_count": 2
    },
    // ... repeat for channels 1-7
  ]
}
```

## Example 3: GPU with HBM3 (NVIDIA H100)

**H100 SXM5 with 80GB HBM3**

```json
{
  "memory_type": "HBM3",
  "peak_bandwidth_gbps": 3350.0,

  "memory_channels": [
    {
      "name": "HBM Stack 0",
      "type": "hbm3",
      "size_gb": 16,
      "frequency_mhz": 2600,
      "data_rate_mts": 5200,
      "bus_width_bits": 1024,
      "bandwidth_gbps": 665.6,
      "dies_per_stack": 8,
      "stack_height": 7
    },
    {
      "name": "HBM Stack 1",
      "type": "hbm3",
      "size_gb": 16,
      "frequency_mhz": 2600,
      "data_rate_mts": 5200,
      "bus_width_bits": 1024,
      "bandwidth_gbps": 665.6,
      "dies_per_stack": 8,
      "stack_height": 7
    },
    // ... stacks 2-4 (total 5 stacks for 80GB)
  ]
}
```

**Computation:**
- 5 stacks × 16 GB = **80 GB**
- 5 stacks × 665.6 GB/s = **3,328 GB/s** (≈3,350 GB/s)
- 5 stacks × 1024 bits = **5,120 bits** total bus width

## Example 4: GPU with GDDR6X (NVIDIA RTX 4090)

**RTX 4090 with 24GB GDDR6X**

```json
{
  "memory_type": "GDDR6X",
  "peak_bandwidth_gbps": 1008.0,

  "memory_channels": [
    {
      "name": "GDDR Bank 0",
      "type": "gddr6x",
      "size_gb": 2,
      "frequency_mhz": 1313,
      "data_rate_mts": 21000,
      "bus_width_bits": 32,
      "bandwidth_gbps": 84.0
    },
    // ... 11 more banks (12 total for 384-bit bus)
  ]
}
```

**Computation:**
- 12 banks × 2 GB = **24 GB**
- 12 banks × 84 GB/s = **1,008 GB/s**
- 12 banks × 32 bits = **384 bits** total bus width

## Example 5: Mobile SoC with LPDDR5 (Snapdragon 8 Gen 2)

**Snapdragon 8 Gen 2 with 12GB LPDDR5X**

```json
{
  "memory_type": "LPDDR5X",
  "peak_bandwidth_gbps": 64.0,

  "memory_channels": [
    {
      "name": "LPDDR Channel 0",
      "type": "lpddr5x",
      "size_gb": 6,
      "frequency_mhz": 2000,
      "data_rate_mts": 8533,
      "bus_width_bits": 32,
      "bandwidth_gbps": 34.1,
      "package_on_package": true
    },
    {
      "name": "LPDDR Channel 1",
      "type": "lpddr5x",
      "size_gb": 6,
      "frequency_mhz": 2000,
      "data_rate_mts": 8533,
      "bus_width_bits": 32,
      "bandwidth_gbps": 34.1,
      "package_on_package": true
    }
  ]
}
```

## API Methods

```python
class HardwareSpec:
    def get_memory_channels(self) -> List[MemoryChannel]:
        """Get memory channels as MemoryChannel objects"""
        if not self.memory_channels:
            return []
        return [MemoryChannel.from_dict(ch) for ch in self.memory_channels]

    def compute_total_memory_gb(self) -> float:
        """Compute total memory capacity from channels"""
        if self.memory_channels:
            return sum(ch.size_gb for ch in self.get_memory_channels())
        return 0.0

    def compute_total_bandwidth_gbps(self) -> float:
        """Compute total memory bandwidth from channels"""
        if self.memory_channels:
            return sum(ch.bandwidth_gbps for ch in self.get_memory_channels())
        return self.peak_bandwidth_gbps or 0.0

    def compute_total_bus_width_bits(self) -> int:
        """Compute total memory bus width from channels"""
        if self.memory_channels:
            return sum(ch.bus_width_bits for ch in self.get_memory_channels())
        return self.memory_bus_width or 0

    def has_ecc_memory(self) -> bool:
        """Check if any channel has ECC enabled"""
        channels = self.get_memory_channels()
        return any(ch.ecc_enabled for ch in channels if ch.ecc_enabled is not None)
```

## Backward Compatibility

Old flat fields remain but are computed from channels:

```python
# Old fields (deprecated but supported)
memory_type: str = "DDR5"              # Type of first channel
memory_channels: int = 2               # DEPRECATED: count
memory_bus_width: int = 128            # DEPRECATED: use compute_total_bus_width_bits()
peak_bandwidth_gbps: float = 76.8      # DEPRECATED: use compute_total_bandwidth_gbps()

# New structured field
memory_channels: List[Dict] = [...]    # Array of MemoryChannel dicts
```

## Benefits

1. **Detailed Per-Channel Info**: Know exact DIMM configuration
2. **Flexible**: Handles DDR, HBM, GDDR, LPDDR
3. **Asymmetric Configs**: Can represent non-uniform channels (rare but possible)
4. **Validation**: Check totals match
5. **Mapper-Friendly**: Calculate NUMA effects, channel utilization

## Memory Technology Reference

| Type | Bus Width | Typical Speed | Use Case |
|------|-----------|---------------|----------|
| **DDR5** | 64 bits/ch | 4800-6400 MT/s | Desktop/Server CPU |
| **DDR4** | 64 bits/ch | 2400-3200 MT/s | Older CPU systems |
| **HBM3** | 1024 bits/stack | 5200-6400 MT/s | Datacenter GPU (H100, MI300X) |
| **HBM2e** | 1024 bits/stack | 3200-3600 MT/s | GPU (A100, V100) |
| **GDDR6X** | 32 bits/ctrl | 19000-21000 MT/s | Consumer GPU (RTX 40) |
| **GDDR6** | 32 bits/ctrl | 14000-16000 MT/s | Consumer GPU (RTX 30) |
| **LPDDR5X** | 32 bits/ch | 6400-8533 MT/s | Mobile/Embedded |
| **LPDDR5** | 32 bits/ch | 5500-6400 MT/s | Mobile/Embedded |

## Questions for Review

1. **Naming**: Is `memory_channels` clear, or should it be `memory_subsystem`?
2. **HBM Stack Count**: Should we track stack position (phy 0-4)?
3. **NUMA**: Should we add NUMA node affinity per channel?
4. **ECC Overhead**: Should we track effective vs theoretical bandwidth with ECC?
5. **Asymmetric**: Do we need to support asymmetric channel configs?

## Implementation Checklist

- [ ] Add `MemoryChannel` dataclass to schema.py
- [ ] Add `memory_channels: List[Dict]` to HardwareSpec
- [ ] Add helper methods (get_memory_channels, compute_total_memory_gb, etc.)
- [ ] Update validation to check channel totals
- [ ] Deprecate old flat fields
- [ ] Create example specs for each memory type
- [ ] Update documentation
- [ ] Update auto-detection script (where possible)
