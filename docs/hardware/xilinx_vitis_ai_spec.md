# Xilinx Vitis AI DPU Specifications

*Research conducted: 2025-10-21*
*Sources: AMD/Xilinx documentation, Versal datasheets, Vitis AI 3.5 docs*

## 1. AI Engine (AIE) Clock Frequencies

### AIE-ML v1 (First Generation Versal AI Edge)
- **Maximum frequency**: 1250 MHz (1.25 GHz)
- Example: XCVE2802-2MP device on VEK280 board
- Compute capability: **512 INT8 ops/clock** per tile

### AIE-ML v2 (Gen 2 Versal AI Edge)
- **Typical frequency**: 1000 MHz (1.0 GHz)
- Operating voltage: 0.7V AIE
- Compute capability: **1024 INT8 ops/clock** per tile (2× AIE-ML v1)

**Conclusion**: User's estimate of "1GHz range" was accurate. We'll use **1.25 GHz** for AIE-ML v1 as baseline.

## 2. DPU MAC Array Configurations

### Available Architectures
The DPU IP supports multiple configurable MAC array sizes:
- **B512**: 512 MACs
- **B800**: 800 MACs
- **B1024**: 1024 MACs
- **B1152**: 1152 MACs
- **B1600**: 1600 MACs
- **B2304**: 2304 MACs
- **B3136**: 3136 MACs
- **B4096**: 4096 MACs (largest)

### Multi-Core Scaling
- Up to **4 DPU cores** can be instantiated in one DPU IP
- Multiple DPU IPs can be used for higher performance

### Typical Performance
- **DPUCAHX8L** (HBM variant): 4.0 - 5.3 TOPS
- Higher configurations can exceed 10 TOPS

## 3. Power Consumption (Versal AI Edge Series)

### Edge-Optimized Devices (Embodied AI Target)
| Device   | Power Range | Use Case                    |
|----------|-------------|------------------------------|
| VE2002   | 6-9 W       | Ultra-low power edge        |
| VE2102   | 7-10 W      | Low power edge              |
| VE2022   | 15-20 W     | Balanced edge               |
| VE2302   | 15-20 W     | Balanced edge               |
| VE2602   | 50-60 W     | High-performance edge       |
| VE1752   | 50-60 W     | High-performance edge       |
| VE2802   | 75 W        | Maximum performance edge    |

**Embodied AI sweet spot**: VE2002/VE2102 (6-10W) for drones, VE2022/VE2302 (15-20W) for robots.

### Overall Versal Family Range
- Power: 5-150W
- Voltage options: 0.7V, 0.78V, 0.88V
- Architected for **highest AI performance/watt** in power-constrained systems

## 4. Memory Hierarchy

### On-Chip Memory
- **High RAM Usage mode**: Larger on-chip memory blocks for intermediate data
- Enables higher performance per DPU core
- Specific sizes: Need to confirm (estimate: 32-128KB per tile based on AIE architecture)

### External Memory
- Supports HBM (High Bandwidth Memory) on high-end cards (U50, U280)
- DDR4/DDR5 support on edge devices

## 5. Supported Operations

### Confirmed Operations
- **Convolution** (Conv2D, DepthwiseConv)
- **Pooling** (MaxPool, AvgPool)
- **Element-wise** (Add, Mul, ReLU, etc.)
- **MatMul** (for fully connected layers)

### Attention Support (Critical for ViT)
- **Needs confirmation**: Whether DPU natively supports attention mechanisms
- Likely requires custom AIE kernel implementation for efficient attention
- Alternative: Run attention on ARM cores (performance penalty)

## 6. DPU Mapper Implementation Parameters

### Recommended Configuration for Modeling

**Hardware Profile**:
```python
# B4096 configuration @ AIE-ML v1
mac_units = 4096
clock_freq = 1.25e9  # Hz
ops_per_mac = 2      # Multiply + Accumulate
efficiency = 0.75    # Per user requirement

# Theoretical peak
theoretical_tops = mac_units * ops_per_mac * clock_freq / 1e12
# = 4096 * 2 * 1.25e9 / 1e12 = 10.24 TOPS

# Realistic peak
realistic_tops = theoretical_tops * efficiency
# = 10.24 * 0.75 = 7.68 TOPS INT8
```

**Power Profile** (VE2302 - 15-20W for robots):
```python
power_budget = 17.5  # W (average of 15-20W range)
idle_power = 3.0     # W (estimated ~17% idle)
dynamic_power = 14.5 # W

# Energy per operation
energy_per_top = dynamic_power / realistic_tops
# = 14.5 / 7.68 = 1.89 W/TOPS
```

**Memory Profile**:
```python
# Conservative estimates (need confirmation)
scratchpad_per_tile = 64e3    # 64 KB per tile (to confirm)
ddr_bandwidth = 50e9          # 50 GB/s (typical for edge)
num_tiles = 64                # Typical for B4096 config
total_scratchpad = 64 * 64e3  # 4 MB
```

## 7. Comparison to Other Architectures

### DPU vs KPU (from our existing model)
| Metric              | KPU-T100      | DPU (B4096)  | Notes                          |
|---------------------|---------------|--------------|--------------------------------|
| INT8 TOPS           | 100           | 7.68         | KPU ~13× faster (theoretical) |
| Power (W)           | 25            | 17.5         | DPU more efficient             |
| TOPS/W              | 4.0           | 0.44         | KPU better perf/watt          |
| Flexibility         | Fixed tiles   | Programmable | DPU advantage: FPGA fabric    |
| Reconfigurability   | No            | Yes          | DPU advantage                  |
| Edge deployment     | Yes           | Yes          | Both suitable                  |

### DPU vs GPU (H100)
| Metric              | H100          | DPU (B4096)  | Ratio         |
|---------------------|---------------|--------------|---------------|
| INT8 TOPS           | 3958          | 7.68         | 515× slower   |
| Power (W)           | 700           | 17.5         | 40× less      |
| TOPS/W              | 5.65          | 0.44         | 13× worse     |
| Cost ($)            | ~30,000       | ~1,000       | 30× cheaper   |

**Key insight**: DPU trades absolute performance for power efficiency and cost, ideal for embodied AI.

## 8. Implementation Strategy

### Phase 1: Conservative Model (Week 1)
Use B4096 @ 1.25 GHz with 75% efficiency:
- 7.68 TOPS INT8
- 17.5W power
- Similar tiling strategy to KPU (scratchpad constraints)

### Phase 2: Multi-Configuration Support (Future)
Add support for:
- B1024, B2304, B4096 configurations
- Multi-core scaling (2-4 cores)
- AIE-ML v2 (1024 ops/clock)

### Phase 3: Attention Optimization (For ViT)
- Model attention on ARM cores vs AIE tiles
- Estimate reconfiguration overhead for custom kernels
- Compare to native transformer accelerators (KPU, TPU)

## 9. Open Questions / TODO

1. **Exact scratchpad size per tile**: Need AIE architecture manual
2. **Attention kernel support**: Check Vitis AI 3.5 docs for transformer support
3. **Reconfiguration latency**: Time to load DPU configuration
4. **Multi-core efficiency**: How well does 4-core scale? (Assume 90% for now)
5. **BF16 support**: Does DPU support BF16 or only INT8/FP16?

## 10. References

- AMD Versal AI Edge Series Product Brief
- Xilinx Vitis AI 3.5 Documentation
- DPU IP Product Guide (PG338)
- Versal ACAP AI Engine Architecture Manual (AM009)
- AIE Clock Frequency Scaling Guide (UG1304)

---

**Next Steps**:
1. Implement `xilinx_vitis_ai_dpu_resource_model()` in `hardware_mapper.py`
2. Implement `DPUMapper` class in `dpu_mapper.py`
3. Test on ResNet-18, MobileNet-V2, ViT-Tiny
4. Add to 6-way hardware comparison
