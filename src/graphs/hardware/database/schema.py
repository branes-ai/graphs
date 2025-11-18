"""
Hardware Database Schema

Defines the structure for hardware specifications in the database.
Each hardware entry is a complete specification including identification,
detection hints, performance characteristics, and mapper configuration.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
from pathlib import Path
import json
from datetime import datetime


@dataclass
class CoreCluster:
    """
    Description of a homogeneous compute cluster (CPU cores or GPU SMs/CUs).

    Used for:
    - CPU: Heterogeneous cores (Intel P/E-cores, ARM big.LITTLE)
    - GPU: Streaming Multiprocessors (NVIDIA SMs), Compute Units (AMD CUs), Xe-cores (Intel)

    The same structure handles both CPU and GPU hierarchies.
    """

    name: str
    """
    Cluster name.
    CPU: 'P-core', 'E-core', 'Big', 'Little', 'Medium', 'Prime'
    GPU: 'SM', 'CU', 'Xe-core', 'EU'
    """

    type: str
    """
    Cluster type.
    CPU: 'performance', 'efficiency', 'balanced'
    GPU: 'data_parallel', 'compute', 'graphics', 'ray_tracing'
    """

    count: int
    """
    Number of clusters.
    CPU: Number of cores in this cluster (e.g., 8 P-cores)
    GPU: Number of SMs/CUs (e.g., 132 SMs)
    """

    architecture: Optional[str] = None
    """
    Microarchitecture name.
    CPU: 'Golden Cove', 'Gracemont', 'Cortex-X3', 'Cortex-A510'
    GPU: 'Ada', 'Ampere', 'RDNA 3', 'Xe-HPG'
    """

    base_frequency_ghz: Optional[float] = None
    """Base frequency for this cluster in GHz"""

    boost_frequency_ghz: Optional[float] = None
    """Maximum boost/turbo frequency for this cluster in GHz"""

    # CPU-specific fields
    has_hyperthreading: Optional[bool] = None
    """Whether cores support SMT/Hyper-Threading (CPU only)"""

    simd_width_bits: Optional[int] = None
    """SIMD width in bits (CPU: 256 for AVX2, 512 for AVX512; GPU: N/A)"""

    # GPU-specific fields (per SM/CU)
    cuda_cores_per_cluster: Optional[int] = None
    """CUDA cores per SM (NVIDIA) or Stream Processors per CU (AMD)"""

    tensor_cores_per_cluster: Optional[int] = None
    """Tensor Cores per SM (NVIDIA) or Matrix Cores per CU (AMD)"""

    rt_cores_per_cluster: Optional[int] = None
    """RT cores per SM (NVIDIA RTX only)"""

    max_threads_per_cluster: Optional[int] = None
    """Maximum resident threads per SM/CU"""

    max_warps_per_cluster: Optional[int] = None
    """Maximum warps per SM (NVIDIA, 32 threads/warp) or wavefronts per CU (AMD, 64 threads/wave)"""

    shared_memory_kb: Optional[int] = None
    """Shared memory per SM/CU in KB"""

    register_file_kb: Optional[int] = None
    """Register file size per SM/CU in KB"""

    l1_cache_kb: Optional[int] = None
    """L1 cache per SM/CU in KB (may be shared with shared memory)"""

    def threads_per_core(self) -> int:
        """
        Calculate threads per core (CPU only).

        Returns 2 if hyperthreading, 1 otherwise.
        For GPUs, use max_threads_per_cluster instead.
        """
        if self.has_hyperthreading is None:
            return 1
        return 2 if self.has_hyperthreading else 1

    def total_threads(self) -> int:
        """
        Calculate total hardware threads for this cluster.

        CPU: count × threads_per_core
        GPU: count × max_threads_per_cluster
        """
        if self.max_threads_per_cluster is not None:
            # GPU: total threads = SMs × threads per SM
            return self.count * self.max_threads_per_cluster
        else:
            # CPU: total threads = cores × (2 if HT else 1)
            return self.count * self.threads_per_core()

    def total_cuda_cores(self) -> int:
        """Calculate total CUDA cores across all SMs/CUs in this cluster"""
        if self.cuda_cores_per_cluster is not None:
            return self.count * self.cuda_cores_per_cluster
        return 0

    def total_tensor_cores(self) -> int:
        """Calculate total Tensor Cores across all SMs/CUs in this cluster"""
        if self.tensor_cores_per_cluster is not None:
            return self.count * self.tensor_cores_per_cluster
        return 0

    def total_rt_cores(self) -> int:
        """Calculate total RT Cores across all SMs in this cluster"""
        if self.rt_cores_per_cluster is not None:
            return self.count * self.rt_cores_per_cluster
        return 0

    def is_cpu_cluster(self) -> bool:
        """Check if this is a CPU core cluster"""
        return self.type in ['performance', 'efficiency', 'balanced']

    def is_gpu_cluster(self) -> bool:
        """Check if this is a GPU SM/CU cluster"""
        return self.type in ['data_parallel', 'compute', 'graphics', 'ray_tracing']

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'CoreCluster':
        """Create from dictionary (from JSON)"""
        return cls(**data)


@dataclass
class HardwareSpec:
    """
    Complete hardware specification for database.

    This represents a single hardware configuration (CPU, GPU, accelerator)
    with all information needed for detection, mapping, and calibration.
    """

    # =========================================================================
    # IDENTIFICATION
    # =========================================================================

    id: str
    """Unique identifier: 'intel_i7_12700k', 'nvidia_h100_sxm5_80gb'"""

    vendor: str
    """Vendor name: 'Intel', 'NVIDIA', 'AMD', 'Ampere Computing', 'Google'"""

    model: str
    """Full model name: 'Core i7-12700K', 'H100 SXM5 80GB', 'Jetson Orin AGX 64GB'"""

    architecture: str
    """Architecture name: 'Alder Lake', 'Hopper', 'Zen 4', 'Altra', 'Cortex-A78AE'"""

    device_type: str
    """Device type: 'cpu', 'gpu', 'tpu', 'kpu', 'dpu', 'cgra'"""

    platform: str
    """Platform architecture: 'x86_64', 'aarch64', 'arm64'"""

    # =========================================================================
    # DETECTION HINTS (for auto-matching)
    # =========================================================================

    detection_patterns: List[str] = field(default_factory=list)
    """
    Regex patterns for matching detected hardware.

    Examples:
    - "12th Gen Intel.*Core.*i7-12700K"
    - "NVIDIA H100.*80GB"
    - "Cortex-A78AE.*Orin"
    - "AMD Ryzen 9 7950X"

    Used for fuzzy matching during auto-detection.
    """

    os_compatibility: List[str] = field(default_factory=lambda: ["linux", "windows", "macos"])
    """Supported operating systems: 'linux', 'windows', 'macos'"""

    # =========================================================================
    # CORE SPECIFICATIONS
    # =========================================================================

    # CPU/GPU Core Configuration
    cores: Optional[int] = None
    """
    Physical cores (total across all clusters).
    For homogeneous CPUs, this is the simple core count.
    For heterogeneous CPUs, this is P-cores + E-cores.
    """

    threads: Optional[int] = None
    """Hardware threads (with SMT/Hyper-Threading, total across all clusters)"""

    e_cores: Optional[int] = None
    """
    Efficiency cores (for hybrid CPUs like Alder Lake).
    DEPRECATED: Use core_clusters for detailed heterogeneous specifications.
    """

    base_frequency_ghz: Optional[float] = None
    """
    Base/sustained frequency in GHz.
    For heterogeneous CPUs, this is typically the P-core/Big core frequency.
    For detailed per-cluster frequencies, use core_clusters.
    """

    boost_frequency_ghz: Optional[float] = None
    """
    Maximum boost/turbo frequency in GHz.
    For heterogeneous CPUs, this is typically the maximum across all clusters.
    For detailed per-cluster frequencies, use core_clusters.
    """

    core_clusters: Optional[List[Dict]] = None
    """
    Detailed core cluster specifications for heterogeneous CPUs.

    Each cluster describes a homogeneous set of cores with same architecture,
    frequency, and features. Used for Intel P/E-cores, ARM big.LITTLE, etc.

    Example (Intel i7-12700K):
    [
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

    Example (ARM big.LITTLE - Snapdragon):
    [
        {
            "name": "Prime",
            "type": "performance",
            "count": 1,
            "architecture": "Cortex-X3",
            "base_frequency_ghz": 2.0,
            "boost_frequency_ghz": 3.2
        },
        {
            "name": "Performance",
            "type": "performance",
            "count": 3,
            "architecture": "Cortex-A715",
            "base_frequency_ghz": 1.8,
            "boost_frequency_ghz": 2.8
        },
        {
            "name": "Efficiency",
            "type": "efficiency",
            "count": 4,
            "architecture": "Cortex-A510",
            "base_frequency_ghz": 1.0,
            "boost_frequency_ghz": 2.0
        }
    ]

    When core_clusters is specified, it should be considered the authoritative
    source for core configuration, with cores/threads being computed totals.
    """

    # GPU-specific (DEPRECATED - use core_clusters instead)
    cuda_cores: Optional[int] = None
    """
    Total CUDA cores (NVIDIA GPUs).
    DEPRECATED: Use core_clusters with cuda_cores_per_cluster instead.
    """

    tensor_cores: Optional[int] = None
    """
    Total Tensor cores for matrix ops (NVIDIA).
    DEPRECATED: Use core_clusters with tensor_cores_per_cluster instead.
    """

    sms: Optional[int] = None
    """
    Number of Streaming Multiprocessors (NVIDIA) or Compute Units (AMD).
    DEPRECATED: Use core_clusters with count instead.
    """

    rt_cores: Optional[int] = None
    """
    Total Ray Tracing cores (NVIDIA RTX).
    DEPRECATED: Use core_clusters with rt_cores_per_cluster instead.
    """

    cuda_capability: Optional[str] = None
    """CUDA compute capability: '8.9', '9.0'"""

    # =========================================================================
    # MEMORY CONFIGURATION
    # =========================================================================

    memory_type: str = "DDR4"
    """Memory type: 'DDR5', 'DDR4', 'HBM3', 'HBM2e', 'LPDDR5', 'GDDR6X'"""

    memory_channels: Optional[int] = None
    """Number of memory channels"""

    memory_bus_width: Optional[int] = None
    """Memory bus width in bits"""

    peak_bandwidth_gbps: float = 0.0
    """Theoretical peak memory bandwidth in GB/s"""

    # =========================================================================
    # ISA & FEATURES
    # =========================================================================

    isa_extensions: List[str] = field(default_factory=list)
    """
    Instruction set extensions.

    Examples:
    - x86_64: ["AVX2", "AVX512F", "AVX512BW", "FMA3", "VNNI"]
    - aarch64: ["NEON", "SVE", "SVE2", "BF16", "FP16"]
    """

    special_features: List[str] = field(default_factory=list)
    """
    Special hardware features.

    Examples:
    - ["Tensor Cores", "RT Cores", "DLSS"]
    - ["Matrix Engine", "Neural Engine"]
    - ["Systolic Array", "TPU Cores"]
    """

    # =========================================================================
    # THEORETICAL PERFORMANCE
    # =========================================================================

    theoretical_peaks: Dict[str, float] = field(default_factory=dict)
    """
    Theoretical peak performance for each precision (GFLOPS for float, GIOPS for int).

    Example:
    {
        "fp64": 360.0,
        "fp32": 720.0,
        "fp16": 1440.0,
        "int64": 360.0,
        "int32": 360.0,
        "int16": 720.0,
        "int8": 1440.0
    }

    Note: These are theoretical maximums based on:
    - Cores × Operations/Cycle × Frequency
    - May not account for memory bottlenecks
    - Actual performance will be lower (measured via calibration)
    """

    # =========================================================================
    # CACHE (CPU-specific)
    # =========================================================================

    l1_data_cache_kb: Optional[int] = None
    """L1 data cache size in KB (total across all cores)"""

    l1_instruction_cache_kb: Optional[int] = None
    """L1 instruction cache size in KB (total across all cores)"""

    l1_cache_kb: Optional[int] = None
    """L1 cache size in KB (deprecated: use l1_data_cache_kb + l1_instruction_cache_kb)"""

    l2_cache_kb: Optional[int] = None
    """L2 cache size in KB (total or per-core depending on architecture)"""

    l3_cache_kb: Optional[int] = None
    """L3 cache size in KB (shared across all cores)"""

    l1_cache_line_size_bytes: Optional[int] = None
    """L1 cache line size in bytes (typically 64)"""

    l2_cache_line_size_bytes: Optional[int] = None
    """L2 cache line size in bytes (typically 64 or 128)"""

    l3_cache_line_size_bytes: Optional[int] = None
    """L3 cache line size in bytes (typically 64 or 128)"""

    l1_cache_associativity: Optional[int] = None
    """L1 cache associativity (n-way set associative)"""

    l2_cache_associativity: Optional[int] = None
    """L2 cache associativity (n-way set associative)"""

    l3_cache_associativity: Optional[int] = None
    """L3 cache associativity (n-way set associative)"""

    # =========================================================================
    # POWER
    # =========================================================================

    tdp_watts: Optional[float] = None
    """Thermal Design Power in watts"""

    max_power_watts: Optional[float] = None
    """Maximum power consumption in watts"""

    # =========================================================================
    # METADATA
    # =========================================================================

    release_date: Optional[str] = None
    """Release date: 'Q4 2021', '2022-11', 'November 2021'"""

    end_of_life: Optional[str] = None
    """End of life / discontinuation date"""

    manufacturer_url: Optional[str] = None
    """Official product page URL"""

    notes: Optional[str] = None
    """Additional notes, quirks, known issues"""

    data_source: str = "manufacturer"
    """Source of specifications: 'manufacturer', 'measured', 'estimated', 'community'"""

    last_updated: Optional[str] = None
    """ISO 8601 timestamp of last update: '2025-01-17T12:00:00Z'"""

    # =========================================================================
    # MAPPER CONFIGURATION
    # =========================================================================

    mapper_class: str = "CPUMapper"
    """
    Mapper class name for this hardware.

    Examples:
    - 'CPUMapper' (x86_64, aarch64 CPUs)
    - 'GPUMapper' (NVIDIA, AMD GPUs)
    - 'TPUMapper' (Google TPU)
    - 'KPUMapper' (Kendryte K210)
    - 'DPUMapper' (Xilinx Vitis AI)
    - 'CGRAMapper' (Spatial dataflow accelerators)
    """

    mapper_config: Dict[str, Any] = field(default_factory=dict)
    """
    Mapper-specific configuration parameters.

    Examples:
    - CPUMapper: {"simd_width": 256, "cores_to_use": 12}
    - GPUMapper: {"waves_per_sm": 32, "warps_per_block": 32}
    - TPUMapper: {"systolic_array_dim": 128}
    """

    # =========================================================================
    # SERIALIZATION
    # =========================================================================

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'HardwareSpec':
        """Create from dictionary (from JSON)"""
        return cls(**data)

    def to_json(self, filepath: Path):
        """
        Save to JSON file with pretty formatting.

        Args:
            filepath: Path to output JSON file
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        data = self.to_dict()
        # Update timestamp
        data['last_updated'] = datetime.utcnow().isoformat() + 'Z'

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, sort_keys=False)

    @classmethod
    def from_json(cls, filepath: Path) -> 'HardwareSpec':
        """
        Load from JSON file.

        Args:
            filepath: Path to JSON file

        Returns:
            HardwareSpec instance
        """
        with open(filepath) as f:
            data = json.load(f)
        return cls.from_dict(data)

    def get_core_clusters(self) -> List[CoreCluster]:
        """
        Get core clusters as CoreCluster objects.

        Returns:
            List of CoreCluster objects, or empty list if not specified
        """
        if not self.core_clusters:
            return []

        return [CoreCluster.from_dict(cluster) for cluster in self.core_clusters]

    def has_heterogeneous_cores(self) -> bool:
        """
        Check if this CPU has heterogeneous core architecture.

        Returns:
            True if core_clusters is specified, False otherwise
        """
        return self.core_clusters is not None and len(self.core_clusters) > 0

    def compute_total_cores(self) -> int:
        """
        Compute total core count from clusters or use cores field.

        Returns:
            Total number of physical cores
        """
        if self.has_heterogeneous_cores():
            return sum(cluster.count for cluster in self.get_core_clusters())
        return self.cores or 0

    def compute_total_threads(self) -> int:
        """
        Compute total thread count from clusters or use threads field.

        Returns:
            Total number of hardware threads
        """
        if self.has_heterogeneous_cores():
            return sum(cluster.total_threads() for cluster in self.get_core_clusters())
        return self.threads or 0

    def get_max_boost_frequency(self) -> Optional[float]:
        """
        Get maximum boost frequency across all clusters.

        Returns:
            Maximum boost frequency in GHz, or None if not specified
        """
        if self.has_heterogeneous_cores():
            boost_freqs = [
                cluster.boost_frequency_ghz
                for cluster in self.get_core_clusters()
                if cluster.boost_frequency_ghz is not None
            ]
            return max(boost_freqs) if boost_freqs else None
        return self.boost_frequency_ghz

    def compute_total_sms(self) -> int:
        """
        Compute total SM/CU count from clusters or use sms field.

        Returns:
            Total number of SMs/CUs
        """
        if self.has_heterogeneous_cores():
            clusters = self.get_core_clusters()
            gpu_clusters = [c for c in clusters if c.is_gpu_cluster()]
            return sum(cluster.count for cluster in gpu_clusters)
        return self.sms or 0

    def compute_total_cuda_cores(self) -> int:
        """
        Compute total CUDA cores from clusters or use cuda_cores field.

        Returns:
            Total number of CUDA cores
        """
        if self.has_heterogeneous_cores():
            return sum(cluster.total_cuda_cores() for cluster in self.get_core_clusters())
        return self.cuda_cores or 0

    def compute_total_tensor_cores(self) -> int:
        """
        Compute total Tensor Cores from clusters or use tensor_cores field.

        Returns:
            Total number of Tensor Cores
        """
        if self.has_heterogeneous_cores():
            return sum(cluster.total_tensor_cores() for cluster in self.get_core_clusters())
        return self.tensor_cores or 0

    def compute_total_rt_cores(self) -> int:
        """
        Compute total RT Cores from clusters or use rt_cores field.

        Returns:
            Total number of RT Cores
        """
        if self.has_heterogeneous_cores():
            return sum(cluster.total_rt_cores() for cluster in self.get_core_clusters())
        return self.rt_cores or 0

    def validate(self) -> List[str]:
        """
        Validate hardware spec for completeness and correctness.

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        # Required fields
        if not self.id:
            errors.append("Missing required field: id")
        if not self.vendor:
            errors.append("Missing required field: vendor")
        if not self.model:
            errors.append("Missing required field: model")
        if not self.device_type:
            errors.append("Missing required field: device_type")
        if self.device_type not in ['cpu', 'gpu', 'tpu', 'kpu', 'dpu', 'cgra']:
            errors.append(f"Invalid device_type: {self.device_type}")
        if not self.platform:
            errors.append("Missing required field: platform")
        if self.platform not in ['x86_64', 'aarch64', 'arm64']:
            errors.append(f"Invalid platform: {self.platform}")

        # Validate core_clusters if specified
        if self.core_clusters:
            for i, cluster_dict in enumerate(self.core_clusters):
                try:
                    cluster = CoreCluster.from_dict(cluster_dict)
                    if not cluster.name:
                        errors.append(f"core_clusters[{i}]: Missing name")
                    if not cluster.type:
                        errors.append(f"core_clusters[{i}]: Missing type")
                    if cluster.count <= 0:
                        errors.append(f"core_clusters[{i}]: count must be positive")
                except Exception as e:
                    errors.append(f"core_clusters[{i}]: Invalid cluster definition: {e}")

            # Check that cores/threads match cluster totals (if both specified)
            computed_cores = self.compute_total_cores()
            if self.cores and computed_cores != self.cores:
                errors.append(
                    f"cores ({self.cores}) doesn't match core_clusters total ({computed_cores}). "
                    "Update cores to match cluster total or remove it."
                )

            computed_threads = self.compute_total_threads()
            if self.threads and computed_threads != self.threads:
                errors.append(
                    f"threads ({self.threads}) doesn't match core_clusters total ({computed_threads}). "
                    "Update threads to match cluster total or remove it."
                )

        # Theoretical peaks should have at least fp32
        if not self.theoretical_peaks or 'fp32' not in self.theoretical_peaks:
            errors.append("theoretical_peaks must include 'fp32'")

        # Peak bandwidth should be positive
        if self.peak_bandwidth_gbps <= 0:
            errors.append("peak_bandwidth_gbps must be positive")

        # Mapper class should be specified
        if not self.mapper_class:
            errors.append("Missing required field: mapper_class")

        return errors


@dataclass
class HardwareDetectionResult:
    """
    Result of hardware auto-detection.

    Contains detected properties and confidence score for matching
    against database entries.
    """

    # Detected properties
    cpu_model: Optional[str] = None
    cpu_vendor: Optional[str] = None
    cpu_cores: Optional[int] = None
    cpu_threads: Optional[int] = None
    cpu_frequency_mhz: Optional[float] = None

    gpu_vendor: Optional[str] = None
    gpu_model: Optional[str] = None
    gpu_memory_gb: Optional[float] = None
    gpu_cuda_capability: Optional[str] = None

    platform: Optional[str] = None
    os_type: Optional[str] = None  # 'linux', 'windows', 'macos'

    isa_extensions: List[str] = field(default_factory=list)

    # Matching
    matched_spec: Optional[HardwareSpec] = None
    confidence_score: float = 0.0  # 0.0 - 1.0
    match_reason: Optional[str] = None

    # Raw detection data (for debugging)
    raw_data: Dict[str, Any] = field(default_factory=dict)
