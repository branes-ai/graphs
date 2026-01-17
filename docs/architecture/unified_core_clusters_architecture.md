# Unified Core Clusters: CPU + GPU

The hardware database uses a **unified `core_clusters` structure** to describe the hierarchical compute organization of both CPUs and GPUs.

## Design Philosophy

Modern processors have **hierarchical compute structures**:

```
CPU (Heterogeneous):
├── P-core cluster: 8 cores @ 3.6-5.0 GHz, HT enabled
└── E-core cluster: 4 cores @ 2.7-3.8 GHz, HT disabled

GPU (Data Parallel):
└── SM cluster: 132 SMs × (128 CUDA + 4 Tensor cores)

ARM SoC (Heterogeneous):
├── Prime cluster: 1 core @ 3.2 GHz (Cortex-X3)
├── Performance cluster: 4 cores @ 2.8 GHz (Cortex-A715)
└── Efficiency cluster: 3 cores @ 2.0 GHz (Cortex-A510)
```

Rather than having separate schemas for CPUs and GPUs, we use a **unified `CoreCluster`** structure that handles both.

## CoreCluster Structure

```python
@dataclass
class CoreCluster:
    # Common fields (CPU + GPU)
    name: str                           # "P-core", "E-core", "SM", "CU"
    type: str                           # CPU: "performance", "efficiency"
                                        # GPU: "data_parallel"
    count: int                          # Number of cores/SMs/CUs
    architecture: str                   # "Golden Cove", "Hopper", "CDNA 3"
    base_frequency_ghz: float          # Base frequency
    boost_frequency_ghz: float         # Boost frequency

    # CPU-specific
    has_hyperthreading: bool           # SMT/HT support
    simd_width_bits: int               # 256 for AVX2, 512 for AVX512

    # GPU-specific (per SM/CU)
    cuda_cores_per_cluster: int        # CUDA cores or Stream Processors
    tensor_cores_per_cluster: int      # Tensor/Matrix cores
    rt_cores_per_cluster: int          # RT cores (NVIDIA RTX)
    max_threads_per_cluster: int       # Max resident threads
    max_warps_per_cluster: int         # Max warps/wavefronts
    shared_memory_kb: int              # Shared memory/LDS
    register_file_kb: int              # Register file size
    l1_cache_kb: int                   # L1 cache per cluster
```

## Supported Architectures

### CPUs
- ✅ **Intel Hybrid** (12th gen+): P-cores + E-cores
- ✅ **ARM big.LITTLE**: Prime + Performance + Efficiency cores
- ✅ **AMD Threadripper**: Homogeneous high-core-count
- ✅ **Ampere Altra**: ARM server CPUs

### GPUs
- ✅ **NVIDIA Datacenter**: H100, A100 (SMs with Tensor Cores)
- ✅ **NVIDIA Consumer**: RTX 40/30 series (SMs with Tensor + RT Cores)
- ✅ **AMD CDNA**: MI300X, MI250X (CUs with Matrix Cores)
- ✅ **AMD RDNA**: RX 7900, RX 6900 (CUs for gaming)
- ✅ **Intel Xe**: Arc GPUs (Xe-cores)

## Quick Examples

### CPU: Intel i7-12700K
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
      "has_hyperthreading": true
    },
    {
      "name": "E-core",
      "type": "efficiency",
      "count": 4,
      "architecture": "Gracemont",
      "base_frequency_ghz": 2.7,
      "boost_frequency_ghz": 3.8,
      "has_hyperthreading": false
    }
  ]
}
```

### GPU: NVIDIA H100
```json
{
  "sms": 132,
  "cuda_cores": 16896,
  "tensor_cores": 528,
  "core_clusters": [
    {
      "name": "SM",
      "type": "data_parallel",
      "count": 132,
      "cuda_cores_per_cluster": 128,
      "tensor_cores_per_cluster": 4,
      "max_threads_per_cluster": 2048,
      "shared_memory_kb": 228
    }
  ]
}
```

### ARM: Snapdragon 8 Gen 2
```json
{
  "cores": 8,
  "core_clusters": [
    {
      "name": "Prime",
      "type": "performance",
      "count": 1,
      "architecture": "Cortex-X3",
      "boost_frequency_ghz": 3.2
    },
    {
      "name": "Performance",
      "type": "performance",
      "count": 4,
      "architecture": "Cortex-A715",
      "boost_frequency_ghz": 2.8
    },
    {
      "name": "Efficiency",
      "type": "efficiency",
      "count": 3,
      "architecture": "Cortex-A510",
      "boost_frequency_ghz": 2.0
    }
  ]
}
```

## API Usage

### Query Core Clusters
```python
spec = HardwareSpec.from_json("intel_i7_12700k.json")

# Check if using clusters
if spec.has_heterogeneous_cores():
    clusters = spec.get_core_clusters()

    for cluster in clusters:
        if cluster.is_cpu_cluster():
            print(f"{cluster.name}: {cluster.count} cores @ {cluster.boost_frequency_ghz} GHz")
            print(f"  HT: {cluster.has_hyperthreading}")

        elif cluster.is_gpu_cluster():
            print(f"{cluster.name}: {cluster.count} SMs/CUs")
            print(f"  CUDA cores: {cluster.cuda_cores_per_cluster}/SM")
            print(f"  Tensor cores: {cluster.tensor_cores_per_cluster}/SM")
```

### Compute Totals
```python
# CPU
total_cores = spec.compute_total_cores()
total_threads = spec.compute_total_threads()

# GPU
total_sms = spec.compute_total_sms()
total_cuda_cores = spec.compute_total_cuda_cores()
total_tensor_cores = spec.compute_total_tensor_cores()
total_rt_cores = spec.compute_total_rt_cores()
```

### Get Maximum Frequencies
```python
max_boost = spec.get_max_boost_frequency()  # Max across all clusters
```

## Benefits

### 1. **Structured Information**
Old flat structure:
```json
{"sms": 132, "cuda_cores": 16896, "tensor_cores": 528}
```

New hierarchical structure:
```json
{
  "core_clusters": [{
    "name": "SM",
    "count": 132,
    "cuda_cores_per_cluster": 128,
    "tensor_cores_per_cluster": 4,
    "shared_memory_kb": 228
  }]
}
```

### 2. **Per-Cluster Details**
Enables fine-grained mapping decisions:
- Map compute-intensive kernels to P-cores
- Map background tasks to E-cores
- Calculate SM occupancy from shared memory limits
- Determine Tensor Core availability per SM

### 3. **Extensibility**
Easy to add:
- Multiple GPU chip types in one package
- Heterogeneous GPU configurations
- Custom accelerator clusters
- Future hybrid architectures

### 4. **Validation**
Schema validates:
- Core counts match cluster totals
- Thread counts computed correctly
- Required fields present per cluster type

## Comparison: Old vs New

### Old GPU Schema (Flat)
```json
{
  "sms": 132,
  "cuda_cores": 16896,
  "tensor_cores": 528,
  "rt_cores": 0,
  "shared_memory_per_sm": 228
}
```

**Problems:**
- No per-SM breakdown
- Hard to compute derived values
- Can't distinguish cluster types
- Missing context (threads/SM, warps/SM)

### New GPU Schema (Structured)
```json
{
  "core_clusters": [
    {
      "name": "SM",
      "type": "data_parallel",
      "count": 132,
      "cuda_cores_per_cluster": 128,
      "tensor_cores_per_cluster": 4,
      "rt_cores_per_cluster": 0,
      "max_threads_per_cluster": 2048,
      "max_warps_per_cluster": 64,
      "shared_memory_kb": 228,
      "register_file_kb": 256,
      "l1_cache_kb": 256
    }
  ]
}
```

**Benefits:**
- ✅ Per-SM resource breakdown
- ✅ Can compute totals automatically
- ✅ Extensible to heterogeneous GPUs
- ✅ Complete occupancy calculation info

## Backward Compatibility

Old specs without `core_clusters` continue to work:

```python
# Old style
spec.cuda_cores          # 16896
spec.sms                 # 132
spec.tensor_cores        # 528

# New style (computes from clusters if available)
spec.compute_total_cuda_cores()      # 16896
spec.compute_total_sms()             # 132
spec.compute_total_tensor_cores()    # 528
```

When migrating, you can keep old fields for compatibility:
```json
{
  "sms": 132,
  "cuda_cores": 16896,
  "core_clusters": [...]
}
```

The validator will check they match.

## Documentation

- 📘 **CPU Details**: `docs/hardware_db/heterogeneous_cores.md`
- 🎮 **GPU Details**: `docs/hardware_db/gpu_core_clusters.md`
- 📋 **Examples**:
  - `docs/hardware_db/core_clusters_example.json` (Intel i7-12700K)
  - `docs/hardware_db/core_clusters_arm_example.json` (Snapdragon 8 Gen 2)
  - `docs/hardware_db/gpu_h100_example.json` (NVIDIA H100)
  - `docs/hardware_db/gpu_amd_mi300x_example.json` (AMD MI300X)
  - `docs/hardware_db/gpu_rtx_4090_example.json` (NVIDIA RTX 4090)

## Next Steps

1. **Add your hardware**:
   ```bash
   # Auto-detect base info
   python scripts/hardware_db/auto_detect_and_add.py -o my_cpu.json

   # Edit to add core_clusters
   vim my_cpu.json

   # Add to database
   mv my_cpu.json hardware_database/cpu/intel/
   ```

2. **Update existing specs**: Add `core_clusters` to existing hardware entries

3. **Use in mappers**: Access per-cluster details for better mapping decisions

4. **Validate**: Ensure totals match cluster sums
