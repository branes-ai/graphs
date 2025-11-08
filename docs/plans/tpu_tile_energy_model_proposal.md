# TPU Tile-Based Energy Model Proposal

## Executive Summary

This proposal focuses exclusively on improving TPU energy estimation by modeling the tile-based data movement that dominates energy consumption in TPU architectures from v1 through v5 and Coral Edge TPU. We propose versioned mappers (TPUv1Mapper, TPUv3Mapper, TPUv4Mapper, TPUv5Mapper, CoralMapper) to enable correlation with published data across different TPU generations.

## Problem Statement

### Current TPU Energy Model Limitations

The `SystolicArrayEnergyModel` (src/graphs/hardware/architectural_energy.py:500-586) only models:
1. **Schedule setup**: 100 pJ (one-time)
2. **Data injection**: 0.5 pJ per element
3. **Data extraction**: 0.5 pJ per element
4. **Efficiency multipliers**: 15% compute overhead, 20% memory overhead (fixed)

**What's Missing:**
- **Weight tile loading**: Loading weight tiles from off-chip memory → Weight FIFO → Matrix Unit
- **Accumulator management**: Double-buffered accumulator read/write cycles
- **Unified Buffer staging**: Intermediate result movement through on-chip SRAM
- **Pipeline fill/drain overhead**: 256 cycles (v1) or 128 cycles (v3+) to fill systolic array pipeline
- **DRAM access energy**: Weight Memory (off-chip) vs HBM (on-chip) differences across generations
- **Architectural differences**: v1 (256×256 array) vs v3/v4/v5 (128×128 arrays) utilization characteristics

### Current Mapper Limitations

The `TPUMapper` (src/graphs/hardware/mappers/accelerators/tpu.py) is generation-agnostic:
- Single mapper for all TPU versions
- Doesn't capture v1 vs v3+ architectural differences (array size, memory hierarchy)
- Can't correlate with published v1 ISCA'17 paper or v4/v5 datacenter data
- Doesn't model Coral Edge TPU's ultra-low-power constraints

## TPU Architecture Evolution

### TPU v1 (2015, ISCA 2017 paper)

**Systolic Array**: 256×256 MACs = 65,536 MACs total
**Clock**: ~700 MHz
**Peak Performance**: 92 TOPS INT8 (256×256 × 2 ops × 700 MHz)

**Memory Hierarchy:**
```
┌─────────────────┐
│  Weight Memory  │ 8 GiB DDR3 (off-chip, read-only)
│   (Off-chip)    │ Bandwidth: ~34 GB/s
└────────┬────────┘
         │ PCIe Gen3 x16
         ↓
┌─────────────────┐
│  Weight FIFO    │ 4 tiles × 64 KiB = 256 KiB
│  (On-chip)      │ Buffers 4 weight tiles
└────────┬────────┘
         │ 256-byte path
         ↓
┌─────────────────┐
│  Matrix Unit    │ 256×256 MACs (65,536 total)
│  Weight Buffer  │ 2 tiles × 64 KiB = 128 KiB (double-buffered)
│                 │ 256 cycles to shift in one 64 KiB tile
└────────┬────────┘
         │ 256 elements/cycle
         ↓
┌─────────────────┐
│  Accumulators   │ 4 MiB (4096 × 256 × 32-bit)
│  (On-chip SRAM) │ Double-buffered, sized for ~1350 ops/byte
└────────┬────────┘
         │ 256-byte path
         ↓
┌─────────────────┐
│ Unified Buffer  │ 24 MiB on-chip SRAM
│  (UB)           │ Staging for activations and intermediate results
└─────────────────┘
```

**Key Energy Events (per 64 KiB weight tile):**
1. DRAM read (Weight Memory): 64 KiB @ 10 pJ/byte = 655,360 pJ = 655 nJ
2. Weight FIFO buffer: 64 KiB @ 0.5 pJ/byte = 32,768 pJ = 33 nJ
3. Matrix Unit shift-in: 256 cycles @ 700 MHz = 366 ns pipeline fill
4. Accumulator writes: 256 elements/cycle × 4 bytes × 0.4 pJ/byte
5. Unified Buffer staging: Variable based on activation size

**Roofline Knee**: ~1350 ops/byte (why accumulators are 4 MiB)

---

### TPU v3 (2018)

**Systolic Array**: 128×128 MACs = 16,384 MACs per MXU, **2 MXUs per chip**
- Total: 32,768 MACs per chip (vs 65,536 in v1)
- **Why smaller**: Better utilization on diverse workloads

**Clock**: ~940 MHz
**Peak Performance**:
- BF16: 123 TFLOPS per chip (2 MXUs)
- INT8: 246 TOPS per chip

**Memory**: 16 GB HBM (on-chip, much faster than DDR3)
- Bandwidth: ~900 GB/s
- Energy: ~5 pJ/byte (vs 10 pJ/byte for DDR3)

**Weight Tile Size**: Likely 32 KiB per tile (proportional to 128×128 array)

**Pipeline Depth**: 128 cycles (shorter than v1's 256 cycles)

**Key Change**: On-chip HBM dramatically reduces weight loading energy

---

### TPU v4 (2020)

**Systolic Array**: 128×128 MACs = 16,384 MACs per MXU, **2 MXUs per chip**

**Clock**: ~1050 MHz (estimated)
**Peak Performance**:
- BF16: 275 TFLOPS per chip (2 MXUs × 137.5 TFLOPS/MXU)
- INT8: 550 TOPS per chip (2× BF16)

**Memory**: 32 GB HBM2e
- Bandwidth: ~1.2 TB/s
- Energy: ~10 pJ/byte (HBM2e)

**Precision**: BF16 native (FP32 emulated at half speed)

**Weight Tile Size**: 32 KiB per tile (estimated)

**TDP**: 350W (datacenter pod configuration)

---

### TPU v5e/v5p (2023+)

**TPU v5e** (cost-optimized):
- Peak: 393 TFLOPS BF16
- Similar 128×128 MXU architecture (likely 2 MXUs)

**TPU v5p** (performance-optimized):
- Peak: 459 TFLOPS BF16 per chip
- Likely 128×128 MXUs, possibly more per chip

**Memory**: HBM2e/HBM3
- Bandwidth: ~1.6-2.0 TB/s (estimated)

**Key Improvements**:
- Sparsity support (dynamic zero-skipping in matrix multiply)
- Better FP8 support for transformer models
- Improved interconnect (ICI) for multi-chip scaling

---

### Coral Edge TPU (2019)

**Systolic Array**: Estimated 64×64 or 128×128 (scaled down)
**Clock**: ~500 MHz (estimated)
**Peak Performance**: 4 TOPS INT8 **only** (no FP16/BF16/FP32)

**Memory**: No HBM, uses **host CPU memory** over USB 3.0/PCIe
- Bandwidth: ~4 GB/s (USB 3.0 limited)
- Energy: ~20 pJ/byte (off-chip over USB)

**Power Budget**: 0.5W idle, 2W peak (ultra-low power)

**Weight Storage**: Stored in host memory, streamed over USB
- No Weight FIFO (too small for edge)
- Likely tiny on-chip buffer (~512 KB)

**Key Constraint**: Must fit inference in 2W power budget

---

## Architectural Differences Summary

| Feature | TPU v1 | TPU v3 | TPU v4 | TPU v5p | Coral Edge |
|---------|--------|--------|--------|---------|------------|
| **Array Size** | 256×256 | 128×128 × 2 | 128×128 × 2 | 128×128 × N | ~64×64 |
| **Total MACs** | 65,536 | 32,768 | 32,768 | TBD | ~4,096 |
| **Clock (MHz)** | 700 | 940 | 1050 | ~1100 | 500 |
| **Weight Tile** | 64 KiB | ~32 KiB | ~32 KiB | ~32 KiB | Minimal |
| **Memory** | 8GB DDR3 | 16GB HBM | 32GB HBM2e | HBM3 | Host (USB) |
| **Bandwidth** | 34 GB/s | 900 GB/s | 1200 GB/s | ~1600 GB/s | 4 GB/s |
| **Energy/byte** | 10 pJ | 5 pJ | 10 pJ | ~8 pJ | 20 pJ |
| **Pipeline** | 256 cycles | 128 cycles | 128 cycles | 128 cycles | ~64 cycles |
| **TDP** | ~75W | ~200W | 350W | ~400W | 2W |
| **Precision** | INT8 | BF16, INT8 | BF16, INT8 | BF16, FP8 | INT8 only |

**Key Insight**: v3/v4/v5 use smaller arrays (128×128) for better utilization, but compensate with:
- Higher clock speeds
- Multiple MXUs per chip
- Faster on-chip HBM memory

---

## Proposed Tile Energy Model

### TPU Tile Energy Components

```python
@dataclass
class TPUTileEnergyModel:
    """
    Energy model for TPU tile-based operations.

    Captures the tile-based data movement through TPU memory hierarchy:
    Weight Memory → Weight FIFO → Matrix Unit → Accumulators → Unified Buffer
    """

    # ============================================================
    # Architectural Parameters (generation-specific)
    # ============================================================

    # Systolic array configuration
    array_width: int  # 256 (v1) or 128 (v3/v4/v5)
    array_height: int  # 256 (v1) or 128 (v3/v4/v5)
    num_arrays: int  # 1 (v1, Coral) or 2 (v3/v4/v5)

    # Weight tile configuration
    weight_tile_size: int  # 64 KiB (v1) or 32 KiB (v3+)
    weight_fifo_depth: int  # 4 tiles (v1) or 2 tiles (v3+, estimated)

    # Pipeline depth
    pipeline_fill_cycles: int  # 256 (v1) or 128 (v3+)
    clock_frequency_hz: float  # 700 MHz (v1), 1050 MHz (v4), etc.

    # Accumulator configuration
    accumulator_size: int  # 4 MiB (v1), likely 2 MiB per MXU (v3+)
    accumulator_width: int  # 256 (v1) or 128 (v3+)

    # Unified Buffer size
    unified_buffer_size: int  # 24 MiB (v1), estimated 32 MiB (v4)

    # ============================================================
    # Energy Coefficients (technology-dependent)
    # ============================================================

    # Memory energy (varies by generation)
    weight_memory_energy_per_byte: float  # 10 pJ (DDR3), 5-10 pJ (HBM), 20 pJ (USB)
    weight_fifo_energy_per_byte: float  # 0.5 pJ (on-chip SRAM buffering)
    unified_buffer_read_energy_per_byte: float  # 0.5 pJ (on-chip SRAM)
    unified_buffer_write_energy_per_byte: float  # 0.5 pJ (on-chip SRAM)

    # Accumulator energy (on-chip SRAM, 32-bit wide)
    accumulator_write_energy_per_element: float  # 0.4 pJ (32-bit write)
    accumulator_read_energy_per_element: float  # 0.3 pJ (32-bit read)

    # Matrix unit data movement
    weight_shift_in_energy_per_element: float  # 0.3 pJ (shift register energy)
    activation_stream_energy_per_element: float  # 0.2 pJ (stream into array)

    # Computation energy
    mac_energy: float  # 0.2 pJ (8-bit MAC), 0.3 pJ (BF16 MAC)

    # Static power during pipeline fill
    static_power_watts: float  # TDP-dependent


    def compute_tile_energy(
        self,
        num_weight_tiles: int,
        ops_per_tile: int,
        input_elements_per_tile: int,
        output_elements_per_tile: int,
        batch_size: int = 1,
        precision: str = "INT8"
    ) -> Dict[str, float]:
        """
        Compute energy for tile-based matrix operation.

        Args:
            num_weight_tiles: Number of weight tiles to load
            ops_per_tile: MACs per weight tile
            input_elements_per_tile: Input activation elements per tile
            output_elements_per_tile: Output elements per tile
            batch_size: Batch size (weight reuse factor)
            precision: Operation precision (INT8, BF16, FP32, FP8)

        Returns:
            Dictionary with detailed energy breakdown
        """

        # ============================================================
        # 1. Weight Tile Loading (amortized by batch size)
        # ============================================================

        # Weight Memory → Weight FIFO (off-chip or HBM)
        weight_dram_energy = (
            num_weight_tiles * self.weight_tile_size *
            self.weight_memory_energy_per_byte
        ) / batch_size  # Amortized: weights loaded once, reused for batch

        # Weight FIFO buffering (on-chip staging)
        weight_fifo_energy = (
            num_weight_tiles * self.weight_tile_size *
            self.weight_fifo_energy_per_byte
        )

        # Weight shift-in to Matrix Unit (shift register energy)
        elements_per_tile = self.weight_tile_size // self._get_bytes_per_element(precision)
        weight_shift_energy = (
            num_weight_tiles * elements_per_tile *
            self.weight_shift_in_energy_per_element
        )

        total_weight_energy = weight_dram_energy + weight_fifo_energy + weight_shift_energy

        # ============================================================
        # 2. Input Activation Loading (Unified Buffer → Matrix Unit)
        # ============================================================

        # Unified Buffer read (activations staged in UB)
        input_bytes = input_elements_per_tile * num_weight_tiles * batch_size * \
                      self._get_bytes_per_element(precision)
        input_read_energy = input_bytes * self.unified_buffer_read_energy_per_byte

        # Stream into Matrix Unit
        total_input_elements = input_elements_per_tile * num_weight_tiles * batch_size
        activation_stream_energy = (
            total_input_elements * self.activation_stream_energy_per_element
        )

        total_input_energy = input_read_energy + activation_stream_energy

        # ============================================================
        # 3. Computation (Systolic Array MACs)
        # ============================================================

        # Adjust MAC energy for precision
        mac_energy = self._get_mac_energy(precision)
        total_ops = ops_per_tile * num_weight_tiles * batch_size
        compute_energy = total_ops * mac_energy

        # ============================================================
        # 4. Accumulator Management
        # ============================================================

        # Partial sums written to accumulators during computation
        # Accumulators produce output_elements per tile
        total_output_elements = output_elements_per_tile * num_weight_tiles * batch_size

        # Write partial sums (during computation)
        accumulator_write_energy = (
            total_output_elements * self.accumulator_write_energy_per_element
        )

        # Read completed results (after tile finishes)
        accumulator_read_energy = (
            total_output_elements * self.accumulator_read_energy_per_element
        )

        total_accumulator_energy = accumulator_write_energy + accumulator_read_energy

        # ============================================================
        # 5. Output Write (Accumulators → Unified Buffer)
        # ============================================================

        output_bytes = total_output_elements * self._get_bytes_per_element(precision)
        output_write_energy = output_bytes * self.unified_buffer_write_energy_per_byte

        # ============================================================
        # 6. Pipeline Fill/Drain Overhead
        # ============================================================

        # Each tile incurs pipeline fill/drain cost
        pipeline_time_per_tile = self.pipeline_fill_cycles / self.clock_frequency_hz
        total_pipeline_time = num_weight_tiles * pipeline_time_per_tile

        # Static power consumed during pipeline fill
        pipeline_energy = total_pipeline_time * self.static_power_watts

        # ============================================================
        # Energy Breakdown
        # ============================================================

        return {
            # Weight loading (off-chip → on-chip)
            'weight_dram_energy_j': weight_dram_energy,
            'weight_fifo_energy_j': weight_fifo_energy,
            'weight_shift_energy_j': weight_shift_energy,
            'total_weight_energy_j': total_weight_energy,

            # Input activation loading
            'input_read_energy_j': input_read_energy,
            'activation_stream_energy_j': activation_stream_energy,
            'total_input_energy_j': total_input_energy,

            # Computation
            'compute_energy_j': compute_energy,

            # Accumulator management
            'accumulator_write_energy_j': accumulator_write_energy,
            'accumulator_read_energy_j': accumulator_read_energy,
            'total_accumulator_energy_j': total_accumulator_energy,

            # Output staging
            'output_write_energy_j': output_write_energy,

            # Pipeline overhead
            'pipeline_energy_j': pipeline_energy,

            # Total
            'total_energy_j': (
                total_weight_energy + total_input_energy + compute_energy +
                total_accumulator_energy + output_write_energy + pipeline_energy
            ),

            # Metrics
            'num_tiles': num_weight_tiles,
            'batch_size': batch_size,
            'weight_reuse_factor': batch_size,
            'arithmetic_intensity': total_ops / (input_bytes + output_bytes) if (input_bytes + output_bytes) > 0 else 0,
        }


    def _get_bytes_per_element(self, precision: str) -> int:
        """Get bytes per element for precision"""
        return {
            'FP32': 4,
            'BF16': 2,
            'FP16': 2,
            'INT8': 1,
            'FP8': 1,
        }.get(precision, 4)


    def _get_mac_energy(self, precision: str) -> float:
        """Get MAC energy for precision"""
        # INT8 is most efficient, FP32 least efficient
        base_int8_energy = self.mac_energy
        return {
            'INT8': base_int8_energy,
            'FP8': base_int8_energy,
            'BF16': base_int8_energy * 1.5,  # BF16 ~1.5× INT8 energy
            'FP16': base_int8_energy * 1.5,
            'FP32': base_int8_energy * 3.0,  # FP32 ~3× INT8 energy
        }.get(precision, base_int8_energy)
```

---

## Versioned TPU Mappers

### Factory Functions for Each TPU Generation

```python
def create_tpu_v1_mapper() -> TPUMapper:
    """
    TPU v1 mapper (ISCA 2017 paper architecture).

    Key characteristics:
    - 256×256 systolic array (65,536 MACs)
    - 64 KiB weight tiles
    - 8 GB DDR3 Weight Memory (off-chip)
    - 24 MiB Unified Buffer
    - 256-cycle pipeline fill
    """
    from ...models.datacenter.tpu_v1 import tpu_v1_resource_model

    model = tpu_v1_resource_model()

    # Configure tile energy model for v1
    model.tile_energy_model = TPUTileEnergyModel(
        # Array configuration
        array_width=256,
        array_height=256,
        num_arrays=1,

        # Tile configuration
        weight_tile_size=64 * 1024,  # 64 KiB
        weight_fifo_depth=4,

        # Pipeline
        pipeline_fill_cycles=256,
        clock_frequency_hz=700e6,  # 700 MHz

        # Accumulator
        accumulator_size=4 * 1024 * 1024,  # 4 MiB
        accumulator_width=256,

        # Unified Buffer
        unified_buffer_size=24 * 1024 * 1024,  # 24 MiB

        # Energy coefficients (DDR3 era)
        weight_memory_energy_per_byte=10.0e-12,  # 10 pJ (DDR3)
        weight_fifo_energy_per_byte=0.5e-12,
        unified_buffer_read_energy_per_byte=0.5e-12,
        unified_buffer_write_energy_per_byte=0.5e-12,
        accumulator_write_energy_per_element=0.4e-12,
        accumulator_read_energy_per_element=0.3e-12,
        weight_shift_in_energy_per_element=0.3e-12,
        activation_stream_energy_per_element=0.2e-12,
        mac_energy=0.2e-12,  # 0.2 pJ per 8-bit MAC
        static_power_watts=75.0,  # 75W TDP
    )

    return TPUMapper(model, version="v1")


def create_tpu_v3_mapper() -> TPUMapper:
    """
    TPU v3 mapper (2018, first with HBM).

    Key changes from v1:
    - 128×128 arrays × 2 MXUs (better utilization)
    - 32 KiB weight tiles (smaller)
    - 16 GB HBM (on-chip, much faster)
    - 128-cycle pipeline (shorter)
    """
    from ...models.datacenter.tpu_v3 import tpu_v3_resource_model

    model = tpu_v3_resource_model()

    model.tile_energy_model = TPUTileEnergyModel(
        array_width=128,
        array_height=128,
        num_arrays=2,  # 2 MXUs
        weight_tile_size=32 * 1024,  # 32 KiB
        weight_fifo_depth=2,
        pipeline_fill_cycles=128,
        clock_frequency_hz=940e6,  # 940 MHz
        accumulator_size=2 * 1024 * 1024,  # 2 MiB per MXU (estimated)
        accumulator_width=128,
        unified_buffer_size=32 * 1024 * 1024,  # 32 MiB (estimated)
        weight_memory_energy_per_byte=5.0e-12,  # 5 pJ (HBM, lower than DDR3)
        weight_fifo_energy_per_byte=0.5e-12,
        unified_buffer_read_energy_per_byte=0.5e-12,
        unified_buffer_write_energy_per_byte=0.5e-12,
        accumulator_write_energy_per_element=0.4e-12,
        accumulator_read_energy_per_element=0.3e-12,
        weight_shift_in_energy_per_element=0.3e-12,
        activation_stream_energy_per_element=0.2e-12,
        mac_energy=0.25e-12,  # BF16 slightly higher than INT8
        static_power_watts=200.0,  # 200W TDP
    )

    return TPUMapper(model, version="v3")


def create_tpu_v4_mapper(thermal_profile: str = None) -> TPUMapper:
    """
    TPU v4 mapper (2020, current datacenter standard).

    Key characteristics:
    - 128×128 arrays × 2 MXUs
    - 32 GB HBM2e (1.2 TB/s)
    - BF16 native, INT8 2× speedup
    - 350W TDP
    """
    from ...models.datacenter.tpu_v4 import tpu_v4_resource_model

    model = tpu_v4_resource_model()

    model.tile_energy_model = TPUTileEnergyModel(
        array_width=128,
        array_height=128,
        num_arrays=2,
        weight_tile_size=32 * 1024,
        weight_fifo_depth=2,
        pipeline_fill_cycles=128,
        clock_frequency_hz=1050e6,  # 1050 MHz
        accumulator_size=2 * 1024 * 1024,
        accumulator_width=128,
        unified_buffer_size=32 * 1024 * 1024,
        weight_memory_energy_per_byte=10.0e-12,  # 10 pJ (HBM2e)
        weight_fifo_energy_per_byte=0.5e-12,
        unified_buffer_read_energy_per_byte=0.5e-12,
        unified_buffer_write_energy_per_byte=0.5e-12,
        accumulator_write_energy_per_element=0.4e-12,
        accumulator_read_energy_per_element=0.3e-12,
        weight_shift_in_energy_per_element=0.3e-12,
        activation_stream_energy_per_element=0.2e-12,
        mac_energy=0.25e-12,  # BF16
        static_power_watts=350.0,  # 350W TDP
    )

    return TPUMapper(model, version="v4", thermal_profile=thermal_profile)


def create_coral_edge_tpu_mapper(thermal_profile: str = None) -> TPUMapper:
    """
    Coral Edge TPU mapper (2019, ultra-low-power edge).

    Key characteristics:
    - Tiny systolic array (~64×64 estimated)
    - No Weight Memory (uses host via USB)
    - 4 TOPS INT8 only
    - 2W power budget
    """
    from ...models.edge.coral_edge_tpu import coral_edge_tpu_resource_model

    model = coral_edge_tpu_resource_model()

    model.tile_energy_model = TPUTileEnergyModel(
        array_width=64,  # Estimated (not published)
        array_height=64,
        num_arrays=1,
        weight_tile_size=4 * 1024,  # 4 KiB (tiny tiles)
        weight_fifo_depth=1,  # Minimal buffering
        pipeline_fill_cycles=64,
        clock_frequency_hz=500e6,  # 500 MHz
        accumulator_size=512 * 1024,  # 512 KB (estimated)
        accumulator_width=64,
        unified_buffer_size=512 * 1024,  # 512 KB on-chip
        weight_memory_energy_per_byte=20.0e-12,  # 20 pJ (USB 3.0, off-chip)
        weight_fifo_energy_per_byte=0.5e-12,
        unified_buffer_read_energy_per_byte=0.5e-12,
        unified_buffer_write_energy_per_byte=0.5e-12,
        accumulator_write_energy_per_element=0.4e-12,
        accumulator_read_energy_per_element=0.3e-12,
        weight_shift_in_energy_per_element=0.3e-12,
        activation_stream_energy_per_element=0.2e-12,
        mac_energy=0.15e-12,  # Very efficient 8-bit MAC
        static_power_watts=2.0,  # 2W power budget
    )

    return TPUMapper(model, version="Coral", thermal_profile=thermal_profile)
```

---

## Integration with TPUMapper

### Modified TPUMapper Methods

```python
class TPUMapper(HardwareMapper):
    """TPU hardware mapper with version-specific tiling"""

    def __init__(
        self,
        resource_model: HardwareResourceModel,
        version: str = "v4",  # NEW: version identifier
        thermal_profile: str = None
    ):
        super().__init__(resource_model, thermal_profile=thermal_profile)
        self.version = version
        self.tile_energy_model = getattr(resource_model, 'tile_energy_model', None)

        if self.tile_energy_model is None:
            raise ValueError(f"TPU {version} requires tile_energy_model in resource_model")


    def _calculate_energy(
        self,
        ops: int,
        bytes_transferred: int,
        precision: Precision
    ) -> Tuple[float, float]:
        """
        Calculate energy using tile-aware model.

        Overrides base implementation to use TPUTileEnergyModel.
        """
        if self.tile_energy_model is None:
            # Fallback to base implementation
            return super()._calculate_energy(ops, bytes_transferred, precision)

        # Extract tiling information from operation
        # Estimate weight tile count
        weight_bytes = bytes_transferred * 0.5  # Rough estimate: 50% weights, 50% activations
        num_weight_tiles = math.ceil(weight_bytes / self.tile_energy_model.weight_tile_size)

        # Estimate elements per tile
        bytes_per_element = self.tile_energy_model._get_bytes_per_element(precision.name)
        ops_per_tile = ops / max(1, num_weight_tiles)
        input_elements_per_tile = (bytes_transferred * 0.3) / bytes_per_element / num_weight_tiles
        output_elements_per_tile = (bytes_transferred * 0.2) / bytes_per_element / num_weight_tiles

        # Compute tile energy (batch_size=1 for single inference)
        energy_breakdown = self.tile_energy_model.compute_tile_energy(
            num_weight_tiles=num_weight_tiles,
            ops_per_tile=int(ops_per_tile),
            input_elements_per_tile=int(input_elements_per_tile),
            output_elements_per_tile=int(output_elements_per_tile),
            batch_size=1,
            precision=precision.name,
        )

        # Split into compute and memory energy
        compute_energy = energy_breakdown['compute_energy_j']
        memory_energy = (
            energy_breakdown['total_weight_energy_j'] +
            energy_breakdown['total_input_energy_j'] +
            energy_breakdown['total_accumulator_energy_j'] +
            energy_breakdown['output_write_energy_j']
        )

        return compute_energy, memory_energy
```

---

## Validation Strategy

### Correlation with Published Data

1. **TPU v1 (ISCA 2017 paper)**:
   - Validate roofline knee (~1350 ops/byte)
   - Validate energy efficiency (5-10× better than CPU)
   - Cross-check with Google's published TCO data

2. **TPU v4 (current)**:
   - Validate against datacenter power measurements (350W TDP)
   - Validate BF16 vs INT8 energy ratio (2×)
   - Cross-check latency estimates with published benchmarks

3. **Coral Edge TPU**:
   - Validate 2W power budget
   - Validate 4 TOPS INT8 performance
   - Cross-check with MobileNet/EfficientNet edge benchmarks

### Test Cases

```python
# Test: ResNet-50 inference on TPU v1 vs v4
def test_resnet50_tpu_v1_vs_v4():
    # v1: 256×256 array, DDR3
    v1_mapper = create_tpu_v1_mapper()
    v1_result = v1_mapper.map_graph(resnet50_fusion_report, ...)

    # v4: 128×128 × 2, HBM2e
    v4_mapper = create_tpu_v4_mapper()
    v4_result = v4_mapper.map_graph(resnet50_fusion_report, ...)

    # Expected: v4 faster (higher clock) but similar energy efficiency
    assert v4_result.total_latency < v1_result.total_latency
    assert abs(v4_result.total_energy - v1_result.total_energy) / v1_result.total_energy < 0.3


# Test: Batch size impact on weight loading energy
def test_batch_size_amortization():
    mapper = create_tpu_v4_mapper()

    batch1 = mapper.map_graph(resnet50_fusion_report, batch_size=1)
    batch64 = mapper.map_graph(resnet50_fusion_report, batch_size=64)

    # Expected: Energy per sample decreases with batch size
    energy_per_sample_b1 = batch1.total_energy
    energy_per_sample_b64 = batch64.total_energy / 64

    # Should see ~50% reduction (weight loading amortized)
    assert energy_per_sample_b64 < energy_per_sample_b1 * 0.6
```

---

## Expected Impact

### Accuracy Improvements

1. **Tile-aware energy**:
   - Current: Fixed 15%/20% overhead multipliers
   - Proposed: Per-tile breakdown showing weight loading, accumulator management, pipeline overhead
   - Expected accuracy: ±20% vs measured (vs ±50% current)

2. **Batch size effects**:
   - Current: No batch-aware energy modeling
   - Proposed: Weight reuse amortization explicit
   - Impact: Explains why TPU is efficient at large batches

3. **Generation-specific**:
   - Current: Single generic TPU model
   - Proposed: v1, v3, v4, v5, Coral with different architectures
   - Impact: Can correlate with published data per generation

### Educational Value

- **Why TPU is efficient**: Explicit weight reuse, systolic array eliminates instruction fetch
- **Why batch size matters**: Weight loading amortized over batch
- **Why v3+ uses 128×128**: Better utilization on diverse workloads
- **Roofline connection**: Energy per op decreases as arithmetic intensity increases (visible in tile model)

---

## Implementation Plan

### Phase 1: Core Tile Energy Model (Week 1)
1. Implement `TPUTileEnergyModel` class
2. Add unit tests for tile energy calculations
3. Validate against hand calculations

### Phase 2: TPU v4 Integration (Week 1)
1. Integrate tile model into existing `TPUMapper`
2. Modify `_calculate_energy()` to use tile model
3. Test with ResNet-18/50 on v4

### Phase 3: Versioned Mappers (Week 2)
1. Create `tpu_v1_resource_model()`
2. Create `tpu_v3_resource_model()`
3. Implement factory functions: `create_tpu_v1_mapper()`, etc.
4. Add version-specific tests

### Phase 4: Validation (Week 2)
1. Correlate v1 with ISCA 2017 paper
2. Correlate v4 with datacenter benchmarks
3. Correlate Coral with edge benchmarks
4. Document discrepancies and limitations

### Phase 5: Documentation (Week 3)
1. Update CLAUDE.md with TPU versioning
2. Add energy model documentation
3. Create tutorial: "Why TPU is energy efficient"
4. Add batch size optimization guide

---

## Next Steps

**Should we:**
1. Start with Phase 1 (implement `TPUTileEnergyModel`)?
2. Create `tpu_v1_resource_model()` first to validate against ISCA paper?
3. Focus on v4 integration to get immediate improvements?

**Recommendation**: Start with Phase 2 (v4 integration) since:
- v4 is the current standard
- Existing `tpu_v4_resource_model()` provides foundation
- Can validate quickly against datacenter data
- Then backport to v1 for historical correlation

---

## References

1. **TPU v1 Paper**: Jouppi et al., "In-Datacenter Performance Analysis of a Tensor Processing Unit", ISCA 2017
2. **TPU v3**: Google Cloud TPU v3 documentation
3. **TPU v4**: Google Cloud TPU v4 Pod documentation
4. **Coral Edge TPU**: Google Coral Edge TPU specifications
5. **Current Implementation**:
   - `src/graphs/hardware/architectural_energy.py` (lines 500-586)
   - `src/graphs/hardware/mappers/accelerators/tpu.py`
   - `src/graphs/hardware/models/datacenter/tpu_v4.py`
   - `src/graphs/hardware/models/edge/coral_edge_tpu.py`
