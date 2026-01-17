# Fingerprint System Design

This document explains the hardware and software fingerprint system used for calibration tracking.

## Problem Statement

When collecting performance calibration data across many systems over time, we face several challenges:

1. **Identity Problem**: How do we know we're measuring the same physical hardware?
2. **Reproducibility Problem**: How do we correlate performance changes with software updates?
3. **Comparison Problem**: How do we compare measurements from different sessions?
4. **User Burden**: Requiring users to provide hardware IDs is error-prone and inconsistent

### The Traditional Approach (And Its Failures)

Traditional benchmarking systems require users to provide hardware identifiers:

```bash
# Traditional approach - error-prone
./benchmark --hardware-id "my-workstation" --gpu "rtx4090"
```

Problems:
- Users invent inconsistent names ("workstation1", "my-pc", "test-box")
- Same hardware gets different IDs across reinstalls
- No way to detect hardware changes (CPU upgrade, RAM addition)
- Software stack changes are invisible

### Our Approach: Self-Organizing Identity

The calibration system **auto-detects everything** and generates deterministic fingerprints:

```bash
# Our approach - zero user input
./cli/benchmark_sweep.py
# Hardware fingerprint: c3f840a080356806
# Software fingerprint: 6286d41799837f08
```

## Fingerprint Generation

### Hardware Fingerprint

The hardware fingerprint identifies the physical system configuration. It is generated from attributes that are:
- **Stable across reboots** - survives power cycles
- **Stable across OS reinstalls** - tied to silicon, not software
- **Sensitive to hardware changes** - new CPU = new fingerprint

#### Algorithm

```python
def _generate_fingerprint(cpu, gpu, memory) -> str:
    components = [
        cpu.model,           # "12th Gen Intel(R) Core(TM) i7-12700K"
        cpu.vendor,          # "GenuineIntel"
        str(cpu.stepping),   # "2" (silicon revision)
        cpu.microcode,       # "0x3a"
        str(cpu.cores_physical),  # "12"
    ]

    if gpu:
        components.extend([
            gpu.pci_id,      # "10de:2684" (vendor:device)
            gpu.vbios_version,  # "95.02.18.40.84"
            str(gpu.memory_mb),  # "24576"
        ])
    else:
        components.append("no_gpu")

    components.append(str(int(memory.total_gb)))  # "32"

    fingerprint_str = "|".join(components)
    full_hash = hashlib.sha256(fingerprint_str.encode()).hexdigest()
    return full_hash[:16]  # 16 hex chars = 64 bits
```

#### Example

For your i7-12700K system:

```
Input string:
"12th Gen Intel(R) Core(TM) i7-12700K|GenuineIntel|2|0x3a|12|no_gpu|31"

SHA256 hash:
c3f840a080356806d8f1a2b3c4d5e6f7...

Fingerprint (first 16 chars):
c3f840a080356806
```

#### What Changes the Hardware Fingerprint

| Change | Fingerprint Changes? | Reason |
|--------|---------------------|--------|
| Reboot | No | Same silicon |
| OS reinstall | No | Same silicon |
| Driver update | No | Driver is software |
| BIOS update | Maybe | May update microcode |
| Microcode update | Yes | Silicon-level change |
| CPU upgrade | Yes | Different model/stepping |
| GPU upgrade | Yes | Different PCI ID/VBIOS |
| RAM upgrade | Yes | Different total GB |
| GPU VBIOS flash | Yes | Different VBIOS version |

#### What Does NOT Affect the Fingerprint

These are intentionally excluded:

| Attribute | Why Excluded |
|-----------|--------------|
| CPU frequency | Changes with DVFS/power modes |
| GPU clock speeds | Changes with power/thermal state |
| Hostname | User-configurable, not hardware |
| Cache sizes | Derived from model (redundant) |
| Driver version | Software, not hardware |

### Software Fingerprint

The software fingerprint identifies the software stack. It captures everything that could affect benchmark results.

#### Algorithm

```python
def _generate_fingerprint(*components) -> str:
    # Key components that affect performance
    components = [
        os_release,      # "6.8.0-90-generic"
        gpu_driver,      # "560.35.03"
        cuda_version,    # "12.4"
        pytorch_version, # "2.7.1+cu126"
        numpy_version,   # "2.2.6"
    ]

    fingerprint_str = "|".join(components)
    full_hash = hashlib.sha256(fingerprint_str.encode()).hexdigest()
    return full_hash[:16]
```

#### Example

```
Input string:
"6.8.0-90-generic|N/A|N/A|2.7.1+cu126|2.2.6"

Fingerprint:
6286d41799837f08
```

#### What Changes the Software Fingerprint

| Change | Fingerprint Changes? |
|--------|---------------------|
| Kernel update | Yes |
| GPU driver update | Yes |
| CUDA version change | Yes |
| PyTorch upgrade | Yes |
| NumPy upgrade | Yes |
| pip install other-package | No (not tracked) |
| Python version change | No (not in hash) |

## Design Decisions

### Why SHA256 Truncated to 16 Characters?

- **64 bits of entropy** is sufficient for uniqueness across expected dataset sizes
- **Human-readable** - easy to compare visually, copy/paste
- **URL-safe** - hexadecimal, no special characters
- **Database-friendly** - fixed width, efficient indexing

Birthday paradox analysis: With 16 hex chars (64 bits), collision probability stays under 1% until ~600 million entries.

### Why Separate Hardware and Software Fingerprints?

This enables powerful queries:

```sql
-- How did driver updates affect this hardware?
SELECT * FROM calibration_runs
WHERE hardware_fingerprint = 'c3f840a080356806'
ORDER BY timestamp;

-- Did the kernel update cause regression across all hardware?
SELECT hardware_fingerprint, peak_measured_gops
FROM calibration_runs
WHERE os_kernel IN ('6.8.0-89-generic', '6.8.0-90-generic')
GROUP BY hardware_fingerprint;
```

If we combined them into one fingerprint, we couldn't answer "same hardware, different software."

### Why Include Microcode and Stepping?

CPU microcode and stepping identify specific silicon revisions:

- **Stepping** = silicon revision (e.g., B0, C0). Different steppings can have different performance characteristics.
- **Microcode** = firmware patches. Can fix bugs, change branch prediction, affect performance.

Example: Intel's Spectre mitigations in microcode reduced performance 5-15% on some workloads. We want to track this.

### Why Include GPU VBIOS?

GPU VBIOS (Video BIOS) defines:
- Default power limits
- Voltage curves
- Fan profiles
- Boost behavior

Two identical GPU models with different VBIOS versions can have measurably different performance.

### Why NOT Include Frequencies?

Frequencies are **environmental**, not **identity**:

```
Same i7-12700K at different times:
- Idle: 800 MHz (governor scaling down)
- Load: 4900 MHz (turbo boost)
- Thermal throttle: 3200 MHz

Same fingerprint, three very different measurements.
```

Frequencies are captured in `EnvironmentalContext`, not the fingerprint.

## How Fingerprints Are Used

### 1. Append-Only Storage

Every calibration run creates a NEW record with:
- `hardware_fingerprint` - which physical system
- `software_fingerprint` - which software stack
- `timestamp` - when it happened

```sql
-- Never UPDATE, always INSERT
INSERT INTO calibration_runs (
    hardware_fingerprint,
    software_fingerprint,
    timestamp,
    peak_measured_gops,
    ...
) VALUES (?, ?, ?, ?, ...);
```

### 2. Trajectory Analysis

Track performance over time for specific hardware:

```python
trajectory = db.get_trajectory("c3f840a080356806")
for run in trajectory:
    print(f"{run.timestamp}: {run.peak_measured_gops:.1f} GOPS "
          f"(driver: {run.gpu_driver_version})")
```

Output:
```
2025-01-15: 450.2 GOPS (driver: 550.54.14)
2025-02-20: 465.8 GOPS (driver: 555.42.02)  # +3.5%
2025-03-10: 452.1 GOPS (driver: 560.35.03)  # Regression!
```

### 3. Regression Detection

Automatically detect when newer software performs worse:

```python
alerts = db.detect_regressions(threshold_pct=5.0)
for alert in alerts:
    print(alert.summary())
```

Output:
```
REGRESSION: peak_measured_gops dropped 5.2%
  Hardware: c3f840a080356806
  Previous: 465.8 (driver: 555.42.02)
  Current:  441.5 (driver: 560.35.03)
```

### 4. Similarity Search

Find calibrated hardware similar to a target specification:

```python
similar = db.find_comparable(target_gops=500, target_bandwidth=200)
for run, similarity in similar:
    print(f"{run.cpu_model}: {similarity:.3f} similarity")
```

## Versioning and Migration

### When Do We Need to Version the Fingerprint?

The fingerprint algorithm should be versioned when:

1. **Adding new attributes** that change existing fingerprints
2. **Removing attributes** that were previously included
3. **Changing the hash algorithm** (SHA256 -> something else)
4. **Changing the truncation length** (16 chars -> different)

### Current Fingerprint Versions

| Version | Hardware Attributes | Software Attributes |
|---------|--------------------|--------------------|
| v1 (current) | model, vendor, stepping, microcode, cores, [gpu_pci_id, vbios, memory], total_gb | os_release, gpu_driver, cuda, pytorch, numpy |

### How to Add New Attributes (Backward Compatible)

**Option A: Append with default** (Recommended for optional attributes)

```python
# Add GPU compute capability without breaking existing fingerprints
components = [
    cpu.model, cpu.vendor, str(cpu.stepping),
    cpu.microcode, str(cpu.cores_physical),
]
if gpu:
    components.extend([
        gpu.pci_id, gpu.vbios_version, str(gpu.memory_mb),
        gpu.compute_capability or "",  # New, defaults to empty
    ])
```

Empty string preserves existing fingerprints for GPUs where we didn't detect compute capability.

**Option B: Version prefix** (For breaking changes)

```python
# New fingerprint includes version prefix
components = ["v2"] + [  # Version marker
    cpu.model, cpu.vendor, str(cpu.stepping),
    # ... including new required attributes
]
```

This creates a new fingerprint namespace. Old `c3f840a...` fingerprints remain valid for historical data.

### Database Migration Strategy

The calibration database is **append-only by design**. Migration is straightforward:

```python
def migrate_v1_to_v2():
    """
    Migrate to v2 fingerprints while preserving v1 data.

    Strategy: Add new columns, don't modify existing data.
    """
    # 1. Add new column for v2 fingerprint
    cursor.execute("""
        ALTER TABLE calibration_runs
        ADD COLUMN hardware_fingerprint_v2 TEXT
    """)

    # 2. New runs use v2 fingerprint in new column
    # 3. Old data keeps v1 fingerprint in original column
    # 4. Queries can join on either version
```

### Re-fingerprinting Historical Data

If you need to regenerate fingerprints for historical data:

```python
def regenerate_fingerprints(db):
    """
    Regenerate fingerprints from stored attributes.

    This works because we denormalize all attributes into each record.
    """
    for run in db.get_all_runs():
        # Reconstruct identity from stored attributes
        new_hw_fp = generate_hw_fingerprint(
            model=run.cpu_model,
            vendor=run.cpu_vendor,
            stepping=run.cpu_stepping,
            microcode=run.cpu_microcode,
            cores=run.cpu_cores_physical,
            gpu_pci_id=run.gpu_pci_id,
            gpu_vbios=run.gpu_vbios,
            gpu_memory=run.gpu_memory_mb,
            memory_gb=run.memory_total_gb,
        )

        # Store in new column
        db.update_fingerprint_v2(run.run_id, new_hw_fp)
```

This is possible because we **denormalize** all attributes into each record. The fingerprint is a convenience for grouping, but the source attributes are preserved.

## Collision Handling

### Theoretical Risk

With 16 hex characters (64 bits), the birthday paradox gives:
- 50% collision probability at ~5 billion entries
- 1% collision probability at ~600 million entries

For a calibration database, this is effectively zero risk.

### Practical Handling

If collisions ever become a concern:

1. **Increase to 24 characters** (96 bits) - trivial code change
2. **Add version prefix** - `v2:` prefix creates new namespace
3. **Use full SHA256** (64 chars) - always available in code

## Summary

| Fingerprint | Identifies | Changes When | Use Case |
|-------------|-----------|--------------|----------|
| Hardware | Physical silicon | CPU/GPU/RAM change, microcode update | Group by machine |
| Software | Software stack | Driver/CUDA/PyTorch update | Regression analysis |

Key design principles:
1. **Auto-detected** - zero user input required
2. **Deterministic** - same system = same fingerprint
3. **Stable** - survives reboots and reinstalls
4. **Sensitive** - detects meaningful changes
5. **Append-only** - never update, always insert
6. **Denormalized** - attributes preserved for re-fingerprinting

## References

- `src/graphs/hardware/calibration/auto_detect.py` - Fingerprint generation
- `src/graphs/hardware/calibration/calibration_db.py` - Database schema and queries
- `cli/benchmark_sweep.py` - CLI tool using fingerprints
