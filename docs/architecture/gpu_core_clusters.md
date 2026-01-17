# GPU Core Clusters (Unified Structure)

The hardware database now uses a **unified `core_clusters` structure** for both CPUs and GPUs. This provides detailed, structured information about GPU Streaming Multiprocessors (NVIDIA), Compute Units (AMD), and Xe-cores (Intel).

## Why Unified Core Clusters?

Both CPUs and GPUs have hierarchical compute structures:

| Architecture | Cluster Unit | Contains |
|-------------|--------------|----------|
| CPU Heterogeneous | P-core/E-core | Individual cores with different frequencies/capabilities |
| NVIDIA GPU | SM (Streaming Multiprocessor) | CUDA cores, Tensor Cores, RT cores, shared memory |
| AMD GPU | CU (Compute Unit) | Stream Processors, Matrix Cores, LDS memory |
| Intel GPU | Xe-core | Execution Units (EUs), shared local memory |

The same `CoreCluster` structure handles both cases with optional CPU-specific and GPU-specific fields.

## GPU CoreCluster Fields

### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | "SM" (NVIDIA), "CU" (AMD), "Xe-core" (Intel) |
| `type` | string | "data_parallel" for GPUs |
| `count` | int | Number of SMs/CUs |

### GPU-Specific Fields

| Field | Type | Description |
|-------|------|-------------|
| `cuda_cores_per_cluster` | int | CUDA cores/SM (NVIDIA) or Stream Processors/CU (AMD) |
| `tensor_cores_per_cluster` | int | Tensor Cores/SM (NVIDIA) or Matrix Cores/CU (AMD) |
| `rt_cores_per_cluster` | int | RT cores/SM (NVIDIA RTX only) |
| `max_threads_per_cluster` | int | Maximum resident threads per SM/CU |
| `max_warps_per_cluster` | int | Max warps/SM (NVIDIA) or wavefronts/CU (AMD) |
| `shared_memory_kb` | int | Shared memory per SM/CU (configurable on some architectures) |
| `register_file_kb` | int | Register file size per SM/CU |
| `l1_cache_kb` | int | L1 cache per SM/CU |

### Optional Fields

| Field | Type | Description |
|-------|------|-------------|
| `architecture` | string | "Hopper", "Ada", "CDNA 3", "RDNA 3", "Xe-HPG" |
| `base_frequency_ghz` | float | Base GPU frequency |
| `boost_frequency_ghz` | float | Maximum boost frequency |

## Example: NVIDIA H100 SXM5 80GB

```json
{
  "id": "nvidia_h100_sxm5_80gb",
  "vendor": "NVIDIA",
  "model": "H100 SXM5 80GB",
  "architecture": "Hopper",
  "device_type": "gpu",
  "cuda_capability": "9.0",

  "core_clusters": [
    {
      "name": "SM",
      "type": "data_parallel",
      "count": 132,
      "architecture": "Hopper",
      "base_frequency_ghz": 1.095,
      "boost_frequency_ghz": 1.83,
      "cuda_cores_per_cluster": 128,
      "tensor_cores_per_cluster": 4,
      "rt_cores_per_cluster": 0,
      "max_threads_per_cluster": 2048,
      "max_warps_per_cluster": 64,
      "shared_memory_kb": 228,
      "register_file_kb": 256,
      "l1_cache_kb": 256
    }
  ],

  "memory_type": "HBM3",
  "peak_bandwidth_gbps": 3350.0,

  "theoretical_peaks": {
    "fp64": 33500.0,
    "fp32": 67000.0,
    "fp16": 1979000.0,
    "fp8": 3958000.0
  }
}
```

**Computed Totals:**
- SMs: 132
- CUDA cores: 132 × 128 = **16,896**
- Tensor Cores: 132 × 4 = **528**
- Max threads: 132 × 2048 = **270,336**

## Example: AMD Instinct MI300X

```json
{
  "id": "amd_instinct_mi300x",
  "vendor": "AMD",
  "model": "Instinct MI300X",
  "architecture": "CDNA 3",
  "device_type": "gpu",

  "core_clusters": [
    {
      "name": "CU",
      "type": "data_parallel",
      "count": 304,
      "architecture": "CDNA 3",
      "base_frequency_ghz": 1.7,
      "boost_frequency_ghz": 2.1,
      "cuda_cores_per_cluster": 64,
      "tensor_cores_per_cluster": 4,
      "max_threads_per_cluster": 2560,
      "max_warps_per_cluster": 40,
      "shared_memory_kb": 64,
      "register_file_kb": 512
    }
  ],

  "memory_type": "HBM3",
  "peak_bandwidth_gbps": 5300.0,

  "theoretical_peaks": {
    "fp64": 81700.0,
    "fp32": 163400.0,
    "fp16": 1307000.0,
    "fp8": 2614000.0
  }
}
```

**Computed Totals:**
- CUs: 304
- Stream Processors: 304 × 64 = **19,456**
- Matrix Cores: 304 × 4 = **1,216**
- Max threads: 304 × 2560 = **778,240**

## Example: NVIDIA RTX 4090 (with RT Cores)

```json
{
  "id": "nvidia_rtx_4090",
  "vendor": "NVIDIA",
  "model": "GeForce RTX 4090",
  "architecture": "Ada Lovelace",
  "cuda_capability": "8.9",

  "core_clusters": [
    {
      "name": "SM",
      "type": "data_parallel",
      "count": 128,
      "cuda_cores_per_cluster": 128,
      "tensor_cores_per_cluster": 4,
      "rt_cores_per_cluster": 1,
      "max_threads_per_cluster": 1536,
      "shared_memory_kb": 100,
      "l1_cache_kb": 128
    }
  ]
}
```

**Computed Totals:**
- SMs: 128
- CUDA cores: 128 × 128 = **16,384**
- Tensor Cores: 128 × 4 = **512**
- RT Cores: 128 × 1 = **128** (for ray tracing)

## API Usage

```python
from graphs.hardware.database import HardwareSpec

# Load GPU spec
spec = HardwareSpec.from_json("nvidia_h100_sxm5_80gb.json")

# Check if using core clusters
if spec.has_heterogeneous_cores():
    clusters = spec.get_core_clusters()

    for cluster in clusters:
        if cluster.is_gpu_cluster():
            print(f"{cluster.name}: {cluster.count} units")
            print(f"  CUDA cores/unit: {cluster.cuda_cores_per_cluster}")
            print(f"  Tensor cores/unit: {cluster.tensor_cores_per_cluster}")

# Compute totals
total_sms = spec.compute_total_sms()              # 132
total_cuda = spec.compute_total_cuda_cores()      # 16,896
total_tensor = spec.compute_total_tensor_cores()  # 528
total_rt = spec.compute_total_rt_cores()          # 0
```

## Backward Compatibility

Old specs with simple fields still work:

```json
{
  "cuda_cores": 16896,
  "tensor_cores": 528,
  "sms": 132
}
```

But new specs should use `core_clusters` for detailed structure:

```json
{
  "core_clusters": [
    {
      "name": "SM",
      "count": 132,
      "cuda_cores_per_cluster": 128,
      "tensor_cores_per_cluster": 4
    }
  ]
}
```

When both are present, `core_clusters` is authoritative.

## NVIDIA GPU Architectures

| Architecture | SM Structure | CUDA/SM | Tensor/SM | RT/SM | Notes |
|--------------|--------------|---------|-----------|-------|-------|
| **Hopper** (H100) | 132-142 SMs | 128 | 4 (Gen 4) | 0 | Datacenter, FP8 |
| **Ada Lovelace** (RTX 40) | 128 SMs | 128 | 4 (Gen 4) | 1 (Gen 3) | Consumer, RT |
| **Ampere** (A100) | 108 SMs | 64 | 4 (Gen 3) | 0 | Datacenter |
| **Ampere** (RTX 30) | 68-84 SMs | 128 | 4 (Gen 3) | 1 (Gen 2) | Consumer, RT |
| **Turing** (RTX 20) | 48-72 SMs | 64 | 8 (Gen 2) | 1 (Gen 1) | First RT cores |

## AMD GPU Architectures

| Architecture | CU Structure | SP/CU | Matrix/CU | Notes |
|--------------|--------------|-------|-----------|-------|
| **CDNA 3** (MI300X) | 304 CUs | 64 | 4 | Datacenter, chiplet |
| **CDNA 2** (MI250X) | 220 CUs | 64 | 4 | Datacenter, dual-die |
| **RDNA 3** (RX 7900) | 96 CUs | 64 | 2 | Consumer |
| **RDNA 2** (RX 6900) | 80 CUs | 64 | 0 | Consumer, no matrix |

## Use Cases for GPU Mappers

Hardware mappers can use SM/CU cluster information to:

1. **Occupancy calculation**:
   ```python
   max_blocks_per_sm = cluster.max_warps_per_cluster // warps_per_block
   ```

2. **Shared memory budgeting**:
   ```python
   available_shmem = cluster.shared_memory_kb * 1024
   ```

3. **Register allocation**:
   ```python
   registers_per_thread = cluster.register_file_kb * 1024 // cluster.max_threads_per_cluster
   ```

4. **Tensor Core utilization**:
   ```python
   if cluster.tensor_cores_per_cluster > 0:
       # Use tensor core instructions
   ```

5. **Multi-GPU configurations**:
   ```python
   total_compute = num_gpus * spec.compute_total_cuda_cores()
   ```

## Migration Guide

### Step 1: Convert Old GPU Specs

**Before:**
```json
{
  "sms": 132,
  "cuda_cores": 16896,
  "tensor_cores": 528
}
```

**After:**
```json
{
  "core_clusters": [
    {
      "name": "SM",
      "type": "data_parallel",
      "count": 132,
      "cuda_cores_per_cluster": 128,
      "tensor_cores_per_cluster": 4
    }
  ]
}
```

### Step 2: Add Per-SM Details

Enhance with shared memory, threads, etc:
```json
{
  "core_clusters": [
    {
      "name": "SM",
      "type": "data_parallel",
      "count": 132,
      "cuda_cores_per_cluster": 128,
      "tensor_cores_per_cluster": 4,
      "max_threads_per_cluster": 2048,
      "max_warps_per_cluster": 64,
      "shared_memory_kb": 228,
      "register_file_kb": 256
    }
  ]
}
```

### Step 3: Update Mapper Code

Old code using simple fields:
```python
num_sms = spec.sms
cuda_cores = spec.cuda_cores
```

New code using clusters:
```python
num_sms = spec.compute_total_sms()
cuda_cores = spec.compute_total_cuda_cores()

# Or access per-SM details:
for cluster in spec.get_core_clusters():
    if cluster.is_gpu_cluster():
        threads_per_sm = cluster.max_threads_per_cluster
        shmem_per_sm = cluster.shared_memory_kb
```

## Future Extensions

Possible additions:
- Heterogeneous GPU clusters (e.g., different SM types in same GPU)
- Sub-cluster details (e.g., Tensor Core generations within same GPU)
- Dynamic frequency scaling profiles per cluster
- Power capping effects per SM/CU
