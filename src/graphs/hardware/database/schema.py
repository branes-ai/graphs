"""
Hardware Database Schema

Defines the structure for hardware specifications in the database.
Each hardware entry is a complete specification including identification,
detection hints, performance characteristics, and mapper configuration.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
from pathlib import Path
from datetime import datetime
import json

# Standard set of precisions that MUST be present in theoretical_peaks
# If a precision is not supported, it MUST be set to 0.0
REQUIRED_PRECISIONS = [
    'fp64',   # Double precision floating point
    'fp32',   # Single precision floating point
    'fp16',   # Half precision floating point
    'fp8',    # Quarter precision floating point
    'fp4',    # 4-bit floating point
    'bf16',   # Brain float 16
    'int64',  # 64-bit integer
    'int32',  # 32-bit integer
    'int16',  # 16-bit integer
    'int8',   # 8-bit integer
    'int4',   # 4-bit integer
]


@dataclass
class MemorySubsystem:
    """
    Description of a memory channel, stack, or bank in the memory subsystem.

    Handles diverse memory architectures:
    - CPU: DDR4/DDR5 channels with DIMMs
    - GPU Datacenter: HBM2e/HBM3 stacks
    - GPU Consumer: GDDR6/GDDR6X controllers
    - Mobile/Embedded: LPDDR4/LPDDR5 channels
    - Grace Hopper: LPDDR5X with ECC and NUMA
    """

    name: str
    """
    Memory channel/stack/bank identifier.
    CPU: "Channel 0", "Channel 1", "Channel 2", "Channel 3"
    GPU HBM: "HBM Stack 0", "HBM Stack 1", "HBM Stack 2"
    GPU GDDR: "GDDR Bank 0", "GDDR Bank 1"
    Mobile: "LPDDR Channel 0", "LPDDR Channel 1"
    """

    type: str
    """
    Memory technology type.
    Values: "ddr5", "ddr4", "hbm3", "hbm2e", "gddr6x", "gddr6", "lpddr5", "lpddr5x", "lpddr4x"
    """

    # Capacity
    size_gb: float
    """Memory size for this channel/stack/bank in GB"""

    # Frequency and Data Rate
    frequency_mhz: int
    """Memory clock frequency in MHz (base clock)"""

    data_rate_mts: int
    """
    Data rate in MT/s (megatransfers per second).
    This is the effective transfer rate (double data rate).
    """

    # Bus Configuration
    bus_width_bits: int
    """
    Memory bus width in bits.
    - DDR4/DDR5: 64 bits per channel (72 with ECC)
    - HBM: 1024 bits per stack
    - GDDR6/GDDR6X: 32 bits per controller
    - LPDDR: 32 bits per channel
    """

    # Bandwidth
    bandwidth_gbps: float
    """
    Theoretical peak bandwidth for this channel/stack/bank in GB/s.
    Formula: (data_rate_mts × bus_width_bits) / 8 / 1000
    """

    effective_bandwidth_gbps: Optional[float] = None
    """
    Effective bandwidth accounting for ECC overhead, protocol overhead, etc.
    Typically 1-2% lower than theoretical for ECC, 5-10% for other overheads.
    """

    # NUMA and Topology
    numa_node: Optional[int] = None
    """
    NUMA node this memory is attached to (CPU servers, Grace Hopper).
    0-indexed. None for non-NUMA systems.
    """

    physical_position: Optional[int] = None
    """
    Physical position/index of this channel/stack/bank.
    Useful for: HBM stack positions, memory controller mapping, thermal modeling.
    0-indexed.
    """

    # CPU DDR-specific fields
    dimm_slots: Optional[int] = None
    """Number of DIMM slots available in this channel (CPU only)"""

    dimms_populated: Optional[int] = None
    """Number of DIMMs actually installed in this channel (CPU only)"""

    dimm_size_gb: Optional[int] = None
    """Size per DIMM in GB (CPU only). Total = dimms_populated × dimm_size_gb"""

    ecc_enabled: Optional[bool] = None
    """
    Error Correcting Code (ECC) enabled.
    CPU/Server: Typically optional (consumer: false, server: true)
    Grace Hopper: true
    """

    rank_count: Optional[int] = None
    """
    Number of ranks per DIMM (CPU only).
    1=single-rank, 2=dual-rank, 4=quad-rank
    Higher rank can improve bandwidth but may limit speed/capacity.
    """

    # GPU HBM-specific fields
    dies_per_stack: Optional[int] = None
    """
    Number of dies in an HBM stack (GPU HBM only).
    HBM2e: typically 8 or 12 dies
    HBM3: typically 8 or 16 dies
    """

    stack_height: Optional[int] = None
    """Physical height of HBM stack in mm (GPU HBM only)"""

    # Mobile/Embedded-specific
    package_on_package: Optional[bool] = None
    """
    Package-on-Package (PoP) configuration (mobile/embedded).
    Memory die stacked directly on SoC die.
    """

    def compute_effective_bandwidth(self, ecc_overhead: float = 0.02) -> float:
        """
        Compute effective bandwidth accounting for ECC overhead if enabled.

        Args:
            ecc_overhead: Overhead fraction (default 2% = 0.02)

        Returns:
            Effective bandwidth in GB/s
        """
        if self.effective_bandwidth_gbps is not None:
            return self.effective_bandwidth_gbps

        if self.ecc_enabled:
            return self.bandwidth_gbps * (1.0 - ecc_overhead)

        return self.bandwidth_gbps

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'MemorySubsystem':
        """Create from dictionary (from JSON)"""
        return cls(**data)


@dataclass
class CoreInfo:
    """
    Consolidated core/compute unit information.

    Groups all core-related information in one place:
    - Aggregate counts (cores, threads)
    - Aggregate frequencies (base, boost)
    - Detailed cluster breakdown (for heterogeneous CPUs/GPUs)

    For homogeneous CPUs: Just use aggregate fields.
    For heterogeneous CPUs: Use both aggregate + core_clusters.
    For GPUs: Use core_clusters to describe SM/CU configuration.
    """

    cores: int
    """Total number of physical cores (CPU) or SMs/CUs (GPU)"""

    threads: int
    """Total number of hardware threads"""

    base_frequency_ghz: Optional[float] = None
    """
    Aggregate base frequency in GHz.
    For heterogeneous CPUs: typically the P-core/Big core frequency.
    For detailed per-cluster frequencies, see core_clusters.
    """

    boost_frequency_ghz: Optional[float] = None
    """
    Aggregate maximum boost/turbo frequency in GHz.
    For heterogeneous CPUs: maximum across all clusters.
    For detailed per-cluster frequencies, see core_clusters.
    """

    core_clusters: Optional[List[Dict]] = None
    """
    Detailed core cluster specifications.
    Use get_core_clusters() to access as CoreCluster objects.

    For heterogeneous CPUs (Intel P/E-cores, ARM big.LITTLE):
    - One entry per cluster type
    - Each cluster has count, frequencies, architecture, etc.

    For GPUs (NVIDIA SMs, AMD CUs):
    - One or more entries describing SM/CU configuration
    - Includes CUDA cores, tensor cores, shared memory per SM/CU
    """

    def get_core_clusters(self) -> List['CoreCluster']:
        """Get core clusters as CoreCluster objects"""
        if not self.core_clusters:
            return []
        return [CoreCluster.from_dict(c) for c in self.core_clusters]

    def compute_total_cores(self) -> int:
        """Compute total cores from clusters"""
        if not self.core_clusters:
            return self.cores
        return sum(c.get('count', 0) for c in self.core_clusters)

    def compute_total_threads(self) -> int:
        """Compute total threads from clusters"""
        if not self.core_clusters:
            return self.threads

        total = 0
        for cluster_dict in self.core_clusters:
            cluster = CoreCluster.from_dict(cluster_dict)
            total += cluster.threads_per_core() * cluster.count
        return total

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'CoreInfo':
        """Create from dictionary (from JSON)"""
        return cls(**data)


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
class CacheLevel:
    """
    Single cache level configuration with topology information.

    Captures per-core vs shared caches, heterogeneous core variations,
    and advanced cache properties needed for accurate performance modeling.
    """

    # Identity
    name: str
    """Cache name: 'L1 dcache', 'L1 icache', 'L2', 'L3', 'L4'"""

    level: int
    """Cache level: 1, 2, 3, 4"""

    cache_type: str
    """
    Cache type:
    - 'data': Data cache (L1 dcache)
    - 'instruction': Instruction cache (L1 icache)
    - 'unified': Unified data+instruction (L2, L3, L4)
    """

    # Topology and Sharing
    scope: str
    """
    Scope of sharing (critical for hardware mappers):
    - 'per_core': Private to each core (L1, often L2). No snooping needed among cores.
    - 'per_cluster': Shared within core cluster (ARM L2, AMD CCX L3, multi-socket)
    - 'shared': Shared across all cores (L3, L4). May be sliced/partitioned.
    """

    # Sizing
    size_per_unit_kb: Optional[int] = None
    """
    Size per unit (core/cluster) in KB.
    - For scope='per_core': size per core (e.g., 48 KB L1 dcache per P-core)
    - For scope='per_cluster': size per cluster (e.g., 32 MB L3 per CCX)
    - For scope='shared': None (use total_size_kb instead)

    Hardware mappers use this for per-core allocation calculations.
    """

    total_size_kb: Optional[int] = None
    """
    Total size across all units in KB.
    - For per_core: computed as size_per_unit_kb × core_count (e.g., 384 KB = 48 KB × 8 P-cores)
    - For per_cluster: computed as size_per_unit_kb × cluster_count
    - For shared: actual total capacity (e.g., 25600 KB shared L3)
    """

    # Organization
    associativity: Optional[int] = None
    """N-way set associativity (8-way, 12-way, 16-way, etc.)"""

    line_size_bytes: Optional[int] = None
    """Cache line size in bytes (typically 64 for CPU, 128 for GPU)"""

    sets: Optional[int] = None
    """Number of sets (for detailed modeling: sets × associativity × line_size = total_size)"""

    # Advanced Cache Properties
    inclusivity: Optional[str] = None
    """
    Inclusivity policy (affects eviction behavior):
    - 'inclusive': Contains copies of lower-level caches (Intel older gens)
    - 'exclusive': No overlap with lower levels (acts as victim cache)
    - 'non_inclusive': May or may not contain lower-level data (Intel recent, AMD)
    """

    coherence_protocol: Optional[str] = None
    """
    Cache coherence protocol:
    - 'MESI': Modified, Exclusive, Shared, Invalid
    - 'MOESI': Modified, Owned, Exclusive, Shared, Invalid (AMD)
    - 'MESIF': MESI + Forward state (Intel)
    """

    write_policy: Optional[str] = None
    """
    Write policy:
    - 'write_back': Write to cache, update memory later (typical for L1/L2/L3)
    - 'write_through': Write to cache and memory simultaneously
    - 'write_around': Write directly to memory, bypass cache
    """

    replacement_policy: Optional[str] = None
    """
    Cache replacement policy:
    - 'lru': Least Recently Used
    - 'lfu': Least Frequently Used
    - 'plru': Pseudo-LRU (common in high-associativity caches)
    - 'random': Random replacement
    - 'fifo': First In First Out
    """

    # Prefetcher Configuration
    prefetcher_enabled: Optional[bool] = None
    """Whether hardware prefetcher is enabled for this cache level"""

    prefetcher_type: Optional[str] = None
    """
    Prefetcher type:
    - 'stride': Stride prefetcher (detects regular patterns)
    - 'stream': Stream prefetcher (detects sequential access)
    - 'spatial': Spatial prefetcher (prefetches adjacent lines)
    - 'adaptive': Adaptive/learning prefetcher
    """

    prefetch_distance: Optional[int] = None
    """Prefetch distance in cache lines (how far ahead to prefetch)"""

    # Partitioning (for shared caches)
    partitioned: Optional[bool] = None
    """Whether cache is physically partitioned into slices"""

    slices: Optional[int] = None
    """
    Number of slices/partitions (for shared L3).
    Each slice typically associated with a memory controller or core group.
    """

    # Per-cluster Variation (for heterogeneous cores)
    cluster_name: Optional[str] = None
    """
    Links cache to specific core cluster for heterogeneous configurations.
    - None: Applies to all cores (homogeneous)
    - 'Performance Cores': Only P-cores (Intel 12th+ gen)
    - 'Efficiency Cores': Only E-cores (Intel 12th+ gen)
    - 'Prime': ARM big.LITTLE Prime cores
    - 'Big': ARM big.LITTLE Big cores
    - 'Little': ARM big.LITTLE Little cores
    """

    # Banking and Ports
    banks: Optional[int] = None
    """Number of banks (for parallel access within cache)"""

    read_ports: Optional[int] = None
    """Number of read ports (affects simultaneous read bandwidth)"""

    write_ports: Optional[int] = None
    """Number of write ports (affects simultaneous write bandwidth)"""

    # Latency
    hit_latency_cycles: Optional[int] = None
    """Cache hit latency in CPU cycles"""

    miss_latency_cycles: Optional[int] = None
    """Cache miss latency in CPU cycles (to next level or memory)"""

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'CacheLevel':
        """Create from dictionary (from JSON)"""
        return cls(**data)


@dataclass
class OnChipMemoryHierarchy:
    """
    Complete on-chip memory hierarchy specification.

    Organizes cache topology with proper per-core vs shared distinction,
    heterogeneous core support, and detailed cache properties.
    """

    cache_levels: List[Dict] = field(default_factory=list)
    """
    Detailed cache level configurations (REQUIRED - provide at least one cache level).
    Each entry is a CacheLevel specification.
    Use get_cache_levels() to access as CacheLevel objects.

    Example:
    [
        {
            "name": "L1 dcache",
            "level": 1,
            "cache_type": "data",
            "scope": "per_core",
            "size_per_unit_kb": 48,
            "associativity": 12,
            "line_size_bytes": 64
        },
        ...
    ]
    """

    # DEPRECATED SIMPLE FIELDS (for backward compatibility)
    # Use cache_levels instead - these will be auto-populated from cache_levels if not set
    # ==================================================================================

    l1_dcache_kb: Optional[int] = None
    """DEPRECATED: Use cache_levels. L1 dcache per core."""

    l1_dcache_associativity: Optional[int] = None
    """DEPRECATED: Use cache_levels. L1 dcache associativity."""

    l1_icache_kb: Optional[int] = None
    """DEPRECATED: Use cache_levels. L1 icache per core."""

    l1_icache_associativity: Optional[int] = None
    """DEPRECATED: Use cache_levels. L1 icache associativity."""

    l1_cache_line_size_bytes: Optional[int] = None
    """DEPRECATED: Use cache_levels. L1 cache line size."""

    l2_cache_kb: Optional[int] = None
    """DEPRECATED: Use cache_levels. L2 per core or total."""

    l2_cache_associativity: Optional[int] = None
    """DEPRECATED: Use cache_levels. L2 associativity."""

    l2_cache_line_size_bytes: Optional[int] = None
    """DEPRECATED: Use cache_levels. L2 cache line size."""

    l3_cache_kb: Optional[int] = None
    """DEPRECATED: Use cache_levels. L3 total size."""

    l3_cache_associativity: Optional[int] = None
    """DEPRECATED: Use cache_levels. L3 associativity."""

    l3_cache_line_size_bytes: Optional[int] = None
    """DEPRECATED: Use cache_levels. L3 cache line size."""

    # Helper Methods
    # ==================================================================================

    def get_cache_levels(self) -> List[CacheLevel]:
        """Get cache levels as CacheLevel objects"""
        return [CacheLevel.from_dict(c) for c in self.cache_levels]

    def get_cache_by_level(
        self,
        level: int,
        cache_type: Optional[str] = None,
        cluster: Optional[str] = None
    ) -> Optional[CacheLevel]:
        """
        Get specific cache level by level number, type, and optional cluster.

        Args:
            level: Cache level (1, 2, 3, 4)
            cache_type: 'data', 'instruction', 'unified', or None for any
            cluster: Cluster name filter, or None for any

        Returns:
            CacheLevel object or None if not found
        """
        for cache in self.get_cache_levels():
            if cache.level != level:
                continue
            if cache_type and cache.cache_type != cache_type:
                continue
            if cluster and cache.cluster_name != cluster:
                continue
            return cache
        return None

    def get_per_core_caches(self, cluster: Optional[str] = None) -> List[CacheLevel]:
        """
        Get all per-core caches.

        Args:
            cluster: Filter by cluster name, or None for all

        Returns:
            List of per-core CacheLevel objects
        """
        caches = [c for c in self.get_cache_levels() if c.scope == 'per_core']
        if cluster:
            caches = [c for c in caches if c.cluster_name == cluster]
        return caches

    def get_shared_caches(self) -> List[CacheLevel]:
        """Get all shared caches"""
        return [c for c in self.get_cache_levels() if c.scope == 'shared']

    def get_per_cluster_caches(self) -> List[CacheLevel]:
        """Get all per-cluster caches"""
        return [c for c in self.get_cache_levels() if c.scope == 'per_cluster']

    def compute_total_cache_kb(self, core_count: int) -> int:
        """
        Compute total on-chip cache capacity.

        Args:
            core_count: Total number of cores

        Returns:
            Total cache in KB
        """
        total = 0
        for cache in self.get_cache_levels():
            if cache.total_size_kb:
                total += cache.total_size_kb
        return total

    def get_available_cache_for_core(self, cluster_name: Optional[str] = None) -> Dict[int, int]:
        """
        Get available cache per core (useful for hardware mappers).

        Args:
            cluster_name: Core cluster name, or None for all cores

        Returns:
            Dictionary mapping level -> KB available per core
        """
        result = {}
        for cache in self.get_cache_levels():
            if cluster_name and cache.cluster_name != cluster_name:
                continue

            if cache.scope == 'per_core' and cache.size_per_unit_kb:
                result[cache.level] = cache.size_per_unit_kb
            elif cache.scope == 'shared' and cache.total_size_kb:
                # Shared cache - report total (mapper decides how to use it)
                result[cache.level] = cache.total_size_kb

        return result

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'OnChipMemoryHierarchy':
        """Create from dictionary (from JSON)"""
        return cls(**data)


@dataclass
class SystemInfo:
    """
    Consolidated system/platform information.

    Groups all hardware identity, platform details, power specs,
    lifecycle information, and compatibility/capabilities.

    This consolidates what were previously loose top-level fields:
    - Identity: vendor, model, architecture
    - Platform: device_type, platform
    - Power: tdp_watts, max_power_watts
    - Lifecycle: release_date, end_of_life
    - Reference: manufacturer_url, notes
    - Compatibility: os_compatibility, isa_extensions, special_features
    """

    # Hardware Identity
    vendor: str
    """Hardware vendor (Intel, NVIDIA, AMD, ARM, Google, Qualcomm, etc.)"""

    model: str
    """Full model name (e.g., "12th Gen Intel(R) Core(TM) i7-12700K")"""

    architecture: str
    """
    Microarchitecture name.
    CPU: Alder Lake, Zen 4, Neoverse V1
    GPU: Ada Lovelace, RDNA 3, Ampere
    TPU: v4, v5e
    """

    # Platform Classification
    device_type: str
    """
    Device category: cpu, gpu, tpu, kpu, dsp, dpu, cgra, npu, vpu
    """

    platform: str
    """
    Platform/ISA: x86_64, aarch64, cuda, rocm, opencl, oneapi, etc.
    """

    # Power Specifications
    tdp_watts: Optional[float] = None
    """Thermal Design Power in watts (typical/base power)"""

    max_power_watts: Optional[float] = None
    """Maximum power consumption in watts (peak/turbo)"""

    # Lifecycle Information
    release_date: Optional[str] = None
    """Release/launch date (ISO 8601 format: YYYY-MM-DD)"""

    end_of_life: Optional[str] = None
    """End of life/support date (ISO 8601 format: YYYY-MM-DD)"""

    # Reference Information
    manufacturer_url: Optional[str] = None
    """URL to manufacturer's product page"""

    notes: Optional[str] = None
    """Additional notes about this hardware"""

    # Compatibility and Capabilities
    os_compatibility: Optional[List[str]] = None
    """
    Supported operating systems: linux, windows, macos, android, ios, etc.
    """

    isa_extensions: Optional[List[str]] = None
    """
    ISA extensions and capabilities.
    CPU: AVX2, AVX512, NEON, SVE
    GPU: CUDA, ROCm, OneAPI
    """

    special_features: Optional[List[str]] = None
    """
    Special hardware features.
    Examples: ECC, NVLink, Infinity Fabric, AMX, SME
    """

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'SystemInfo':
        """Create from dictionary (from JSON)"""
        return cls(**data)


@dataclass
class MapperInfo:
    """
    Consolidated mapper configuration.

    Groups all mapper-related information and hints:
    - mapper_class: Which mapper to use (CPUMapper, GPUMapper, etc.)
    - mapper_config: Mapper-specific configuration parameters
    - hints: Additional hints for the mapper (optional)

    This consolidates what were previously loose top-level fields.
    """

    mapper_class: str
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

    mapper_config: Optional[Dict[str, Any]] = None
    """
    Mapper-specific configuration parameters.

    Examples:
    - CPUMapper: {"simd_width": 256, "cores_to_use": 12}
    - GPUMapper: {"waves_per_sm": 32, "warps_per_block": 32}
    - TPUMapper: {"systolic_array_dim": 128}
    """

    hints: Optional[Dict[str, Any]] = None
    """
    Additional hints for the mapper (optional).

    Can include architecture-specific optimization hints,
    preferred tile sizes, scheduling strategies, etc.

    Examples:
    - {"preferred_tile_size": [128, 128]}
    - {"enable_tensor_cores": true}
    - {"memory_coalescing_hint": "row_major"}
    """

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'MapperInfo':
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

    # =========================================================================
    # SYSTEM/PLATFORM INFORMATION
    # =========================================================================

    system: Optional[Dict] = None
    """
    Consolidated system/platform information.
    Use get_system_info() to access as SystemInfo dataclass.

    Groups all hardware identity, platform details, power specs,
    lifecycle information, and compatibility/capabilities:
    - Identity: vendor, model, architecture
    - Platform: device_type, platform
    - Power: tdp_watts, max_power_watts
    - Lifecycle: release_date, end_of_life
    - Reference: manufacturer_url, notes
    - Compatibility: os_compatibility, isa_extensions, special_features

    Example:
    {
        "vendor": "Intel",
        "model": "12th Gen Intel(R) Core(TM) i7-12700K",
        "architecture": "Alder Lake",
        "device_type": "cpu",
        "platform": "x86_64",
        "tdp_watts": 125,
        "max_power_watts": 190,
        "release_date": "2021-11-04",
        "manufacturer_url": "https://ark.intel.com/content/www/us/en/ark/products/134594/...",
        "notes": "12th generation hybrid architecture with P-cores and E-cores",
        "os_compatibility": ["linux", "windows", "macos"],
        "isa_extensions": ["AVX2", "FMA3", "SSE4.2", "VNNI"],
        "special_features": []
    }
    """

    # DEPRECATED FIELDS - Use system instead
    # Kept for backward compatibility when reading old specs

    vendor: Optional[str] = None
    """DEPRECATED: Use system.vendor"""

    model: Optional[str] = None
    """DEPRECATED: Use system.model"""

    architecture: Optional[str] = None
    """DEPRECATED: Use system.architecture"""

    device_type: Optional[str] = None
    """DEPRECATED: Use system.device_type"""

    platform: Optional[str] = None
    """DEPRECATED: Use system.platform"""

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

    os_compatibility: Optional[List[str]] = None
    """DEPRECATED: Use system.os_compatibility"""

    # =========================================================================
    # CORE SPECIFICATIONS
    # =========================================================================

    core_info: Optional[Dict] = None
    """
    Consolidated core/compute unit information.
    Use get_core_info() to access as CoreInfo dataclass.

    Groups all core-related information:
    - cores, threads (aggregate counts)
    - base_frequency_ghz, boost_frequency_ghz (aggregate frequencies)
    - core_clusters (detailed breakdown for heterogeneous CPUs/GPUs)

    Example (Homogeneous CPU):
    {
        "cores": 16,
        "threads": 32,
        "base_frequency_ghz": 3.4,
        "boost_frequency_ghz": 5.0
    }

    Example (Heterogeneous CPU - Intel i7-12700K):
    {
        "cores": 12,
        "threads": 20,
        "base_frequency_ghz": 3.6,
        "boost_frequency_ghz": 5.0,
        "core_clusters": [
            {
                "name": "Performance Cores",
                "type": "performance",
                "count": 8,
                "architecture": "Golden Cove",
                "base_frequency_ghz": 3.6,
                "boost_frequency_ghz": 5.0,
                "has_hyperthreading": true,
                "simd_width_bits": 256
            },
            {
                "name": "Efficiency Cores",
                "type": "efficiency",
                "count": 4,
                "architecture": "Gracemont",
                "base_frequency_ghz": 2.7,
                "boost_frequency_ghz": 3.8,
                "has_hyperthreading": false,
                "simd_width_bits": 128
            }
        ]
    }
    """

    # DEPRECATED FIELDS - Use core_info instead
    # Kept for backward compatibility when reading old specs

    cores: Optional[int] = None
    """DEPRECATED: Use core_info.cores"""

    threads: Optional[int] = None
    """DEPRECATED: Use core_info.threads"""

    e_cores: Optional[int] = None
    """DEPRECATED: Use core_info.core_clusters"""

    base_frequency_ghz: Optional[float] = None
    """DEPRECATED: Use core_info.base_frequency_ghz"""

    boost_frequency_ghz: Optional[float] = None
    """DEPRECATED: Use core_info.boost_frequency_ghz"""

    core_clusters: Optional[List[Dict]] = None
    """DEPRECATED: Use core_info.core_clusters
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
    """
    Memory technology type (primary/first channel).
    DEPRECATED: Use memory_subsystem for detailed configuration.
    """

    memory_channels: Optional[int] = None
    """
    Number of memory channels/stacks/banks.
    DEPRECATED: Use memory_subsystem array length instead.
    """

    memory_bus_width: Optional[int] = None
    """
    Total memory bus width in bits.
    DEPRECATED: Use compute_total_bus_width_bits() with memory_subsystem.
    """

    peak_bandwidth_gbps: float = 0.0
    """
    Theoretical peak memory bandwidth in GB/s.
    DEPRECATED: Use memory_subsystem.peak_bandwidth_gbps instead.
    """

    memory_subsystem: Optional[Dict] = None
    """
    Consolidated memory subsystem configuration.

    Groups all memory-related information:
    - total_size_gb: Total memory capacity
    - peak_bandwidth_gbps: Aggregate memory bandwidth
    - memory_channels: Detailed per-channel/stack/bank configuration

    For homogeneous memory: Just use aggregate fields.
    For multi-channel/NUMA: Use both aggregate + memory_channels.

    Example (CPU Dual-Channel DDR5):
    {
        "total_size_gb": 64,
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
                "numa_node": 0
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
                "numa_node": 0
            }
        ]
    }

    Example (GPU HBM3):
    {
        "total_size_gb": 80,
        "peak_bandwidth_gbps": 3328.0,
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
                "physical_position": 0
            }
            // ... stacks 1-4
        ]
    }

    When memory_subsystem is specified, it is the authoritative source for
    memory configuration. Legacy fields (memory_type, peak_bandwidth_gbps)
    should be computed from memory_subsystem.
    """

    # =========================================================================
    # ISA & FEATURES (DEPRECATED - Use system instead)
    # =========================================================================

    isa_extensions: Optional[List[str]] = None
    """DEPRECATED: Use system.isa_extensions"""

    special_features: Optional[List[str]] = None
    """DEPRECATED: Use system.special_features"""

    # =========================================================================
    # THEORETICAL PERFORMANCE
    # =========================================================================

    theoretical_peaks: Dict[str, float] = field(default_factory=dict)
    """
    Theoretical peak performance for each precision (GFLOPS for float, GIOPS for int).

    REQUIRED: All precisions must be explicitly stated. If a precision is not supported,
    it MUST be set to 0.0 (not null, not missing). This ensures mappers fail loudly when
    trying to use unsupported precisions.

    Required precisions (see REQUIRED_PRECISIONS):
    - fp64, fp32, fp16, bf16
    - int64, int32, int16, int8

    Example (CPU with AVX2, no native fp16/bf16):
    {
        "fp64": 86.4,
        "fp32": 172.8,
        "fp16": 0.0,       # Not natively supported
        "bf16": 0.0,       # Not natively supported
        "int64": 43.2,
        "int32": 86.4,
        "int16": 172.8,
        "int8": 345.6
    }

    Example (GPU with Tensor Cores):
    {
        "fp64": 19.5,
        "fp32": 82.6,
        "fp16": 330.0,
        "bf16": 330.0,
        "int64": 0.0,      # Not supported on GPU
        "int32": 165.0,
        "int16": 0.0,
        "int8": 660.0
    }

    Note: These are theoretical maximums based on:
    - Cores × Operations/Cycle × Frequency
    - May not account for memory bottlenecks
    - Actual performance will be lower (measured via calibration)
    """

    # =========================================================================
    # ON-CHIP MEMORY HIERARCHY (Cache Subsystem)
    # =========================================================================

    onchip_memory_hierarchy: Optional[Dict] = None
    """
    On-chip memory hierarchy (cache subsystem).
    Contains L1 dcache/icache, L2, L3 configuration with sizes, associativity, line sizes.
    Use get_onchip_memory_hierarchy() to access as OnChipMemoryHierarchy dataclass.
    """

    # DEPRECATED CACHE FIELDS (for backward compatibility)
    # Use onchip_memory_hierarchy instead
    # =========================================================================

    l1_data_cache_kb: Optional[int] = None
    """DEPRECATED: Use onchip_memory_hierarchy.l1_dcache_kb"""

    l1_instruction_cache_kb: Optional[int] = None
    """DEPRECATED: Use onchip_memory_hierarchy.l1_icache_kb"""

    l1_cache_kb: Optional[int] = None
    """DEPRECATED: Use onchip_memory_hierarchy"""

    l2_cache_kb: Optional[int] = None
    """DEPRECATED: Use onchip_memory_hierarchy.l2_cache_kb"""

    l3_cache_kb: Optional[int] = None
    """DEPRECATED: Use onchip_memory_hierarchy.l3_cache_kb"""

    l1_cache_line_size_bytes: Optional[int] = None
    """DEPRECATED: Use onchip_memory_hierarchy.l1_cache_line_size_bytes"""

    l2_cache_line_size_bytes: Optional[int] = None
    """DEPRECATED: Use onchip_memory_hierarchy.l2_cache_line_size_bytes"""

    l3_cache_line_size_bytes: Optional[int] = None
    """DEPRECATED: Use onchip_memory_hierarchy.l3_cache_line_size_bytes"""

    l1_cache_associativity: Optional[int] = None
    """DEPRECATED: Use onchip_memory_hierarchy.l1_dcache_associativity or l1_icache_associativity"""

    l2_cache_associativity: Optional[int] = None
    """DEPRECATED: Use onchip_memory_hierarchy.l2_cache_associativity"""

    l3_cache_associativity: Optional[int] = None
    """DEPRECATED: Use onchip_memory_hierarchy.l3_cache_associativity"""

    # =========================================================================
    # POWER (DEPRECATED - Use system instead)
    # =========================================================================

    tdp_watts: Optional[float] = None
    """DEPRECATED: Use system.tdp_watts"""

    max_power_watts: Optional[float] = None
    """DEPRECATED: Use system.max_power_watts"""

    # =========================================================================
    # METADATA (DEPRECATED - Use system instead)
    # =========================================================================

    release_date: Optional[str] = None
    """DEPRECATED: Use system.release_date"""

    end_of_life: Optional[str] = None
    """DEPRECATED: Use system.end_of_life"""

    manufacturer_url: Optional[str] = None
    """DEPRECATED: Use system.manufacturer_url"""

    notes: Optional[str] = None
    """DEPRECATED: Use system.notes"""

    data_source: str = "manufacturer"
    """Source of specifications: 'manufacturer', 'measured', 'estimated', 'community'"""

    last_updated: Optional[str] = None
    """ISO 8601 timestamp of last update: '2025-01-17T12:00:00Z'"""

    # =========================================================================
    # MAPPER CONFIGURATION
    # =========================================================================

    mapper: Optional[Dict] = None
    """
    Consolidated mapper configuration.
    Use get_mapper_info() to access as MapperInfo dataclass.

    Groups all mapper-related information:
    - mapper_class: Which mapper to use
    - mapper_config: Mapper-specific parameters
    - hints: Additional optimization hints (optional)

    Example:
    {
        "mapper_class": "CPUMapper",
        "mapper_config": {
            "simd_width": 256,
            "cores_to_use": 12
        },
        "hints": {
            "preferred_tile_size": [128, 128]
        }
    }
    """

    # DEPRECATED FIELDS - Use mapper instead
    # Kept for backward compatibility when reading old specs

    mapper_class: Optional[str] = None
    """DEPRECATED: Use mapper.mapper_class"""

    mapper_config: Optional[Dict[str, Any]] = None
    """DEPRECATED: Use mapper.mapper_config"""

    # =========================================================================
    # SERIALIZATION
    # =========================================================================

    def to_dict(self) -> Dict:
        """
        Convert to dictionary for JSON serialization.

        Excludes deprecated and null fields to keep JSON clean:
        - GPU-specific fields (cuda_cores, sms, etc.) are excluded for CPU specs
        - Deprecated memory fields are excluded when memory_subsystem is present
        - Deprecated cache fields are excluded when onchip_memory_hierarchy.cache_levels is present
        """
        data = asdict(self)

        # List of fields to potentially exclude
        deprecated_gpu_fields = ['cuda_cores', 'tensor_cores', 'sms', 'rt_cores', 'cuda_capability']
        deprecated_memory_fields = ['memory_type', 'memory_channels', 'memory_bus_width']
        deprecated_cache_fields = [
            'l1_data_cache_kb', 'l1_instruction_cache_kb', 'l1_cache_kb',
            'l2_cache_kb', 'l3_cache_kb',
            'l1_cache_line_size_bytes', 'l2_cache_line_size_bytes', 'l3_cache_line_size_bytes',
            'l1_cache_associativity', 'l2_cache_associativity', 'l3_cache_associativity'
        ]

        # Exclude GPU fields for CPU specs (if they're null)
        device_type = None
        if self.system:
            system_info = self.get_system_info()
            if system_info:
                device_type = system_info.device_type
        else:
            device_type = self.device_type

        if device_type == 'cpu':
            for field in deprecated_gpu_fields:
                if field in data and data[field] is None:
                    del data[field]

        # Exclude deprecated memory fields if memory_subsystem is present
        if self.memory_subsystem:
            for field in deprecated_memory_fields:
                if field in data:
                    del data[field]
            # Also exclude peak_bandwidth_gbps (now inside memory_subsystem)
            if 'peak_bandwidth_gbps' in data:
                del data['peak_bandwidth_gbps']

        # Exclude deprecated cache fields if onchip_memory_hierarchy.cache_levels is present
        if self.onchip_memory_hierarchy:
            hierarchy_dict = self.onchip_memory_hierarchy if isinstance(self.onchip_memory_hierarchy, dict) else asdict(self.onchip_memory_hierarchy)
            if hierarchy_dict.get('cache_levels'):
                # Remove deprecated fields from top level
                for field in deprecated_cache_fields:
                    if field in data:
                        del data[field]

                # Also remove duplicate simple fields from inside onchip_memory_hierarchy
                # Keep only cache_levels when it's present
                if 'onchip_memory_hierarchy' in data and isinstance(data['onchip_memory_hierarchy'], dict):
                    hierarchy_fields_to_remove = [
                        'l1_dcache_kb', 'l1_icache_kb',
                        'l1_dcache_associativity', 'l1_icache_associativity',
                        'l1_cache_line_size_bytes',
                        'l2_cache_kb', 'l2_cache_associativity', 'l2_cache_line_size_bytes',
                        'l3_cache_kb', 'l3_cache_associativity', 'l3_cache_line_size_bytes',
                        'l4_cache_kb', 'l4_cache_associativity', 'l4_cache_line_size_bytes'
                    ]
                    for field in hierarchy_fields_to_remove:
                        if field in data['onchip_memory_hierarchy']:
                            del data['onchip_memory_hierarchy'][field]

        # Exclude deprecated core fields if core_info is present
        if self.core_info:
            deprecated_core_fields = [
                'cores', 'threads', 'e_cores',
                'base_frequency_ghz', 'boost_frequency_ghz',
                'core_clusters'
            ]
            for field in deprecated_core_fields:
                if field in data:
                    del data[field]

        # Exclude deprecated system fields if system is present
        if self.system:
            deprecated_system_fields = [
                'vendor', 'model', 'architecture',
                'device_type', 'platform',
                'os_compatibility',
                'isa_extensions', 'special_features',
                'tdp_watts', 'max_power_watts',
                'release_date', 'end_of_life',
                'manufacturer_url', 'notes'
            ]
            for field in deprecated_system_fields:
                if field in data:
                    del data[field]

        # Exclude deprecated mapper fields if mapper is present
        if self.mapper:
            deprecated_mapper_fields = [
                'mapper_class', 'mapper_config'
            ]
            for field in deprecated_mapper_fields:
                if field in data:
                    del data[field]

        return data

    @classmethod
    def from_dict(cls, data: Dict) -> 'HardwareSpec':
        """
        Create from dictionary (from JSON).

        Handles backward compatibility by migrating old cache fields to onchip_memory_hierarchy.
        """
        # Make a copy to avoid mutating the input
        data = data.copy()

        # Migrate old core fields to core_info if not already present
        if 'core_info' not in data or data['core_info'] is None:
            # Check if any old core fields are present
            old_core_fields = ['cores', 'threads', 'base_frequency_ghz', 'boost_frequency_ghz', 'core_clusters', 'e_cores']
            has_old_core_fields = any(data.get(field) is not None for field in old_core_fields)

            if has_old_core_fields:
                # Migrate to new core_info structure
                core_info = {}

                if data.get('cores') is not None:
                    core_info['cores'] = data['cores']

                if data.get('threads') is not None:
                    core_info['threads'] = data['threads']

                if data.get('base_frequency_ghz') is not None:
                    core_info['base_frequency_ghz'] = data['base_frequency_ghz']

                if data.get('boost_frequency_ghz') is not None:
                    core_info['boost_frequency_ghz'] = data['boost_frequency_ghz']

                if data.get('core_clusters') is not None:
                    core_info['core_clusters'] = data['core_clusters']

                if core_info:
                    data['core_info'] = core_info

        # Migrate old system fields to system if not already present
        if 'system' not in data or data['system'] is None:
            # Check if any old system fields are present
            old_system_fields = [
                'vendor', 'model', 'architecture', 'device_type', 'platform',
                'os_compatibility', 'isa_extensions', 'special_features',
                'tdp_watts', 'max_power_watts',
                'release_date', 'end_of_life', 'manufacturer_url', 'notes'
            ]
            has_old_system_fields = any(data.get(field) is not None for field in old_system_fields)

            if has_old_system_fields:
                # Migrate to new system structure
                system = {}

                # Required fields (must be present)
                if data.get('vendor') is not None:
                    system['vendor'] = data['vendor']
                if data.get('model') is not None:
                    system['model'] = data['model']
                if data.get('architecture') is not None:
                    system['architecture'] = data['architecture']
                if data.get('device_type') is not None:
                    system['device_type'] = data['device_type']
                if data.get('platform') is not None:
                    system['platform'] = data['platform']

                # Optional fields
                if data.get('tdp_watts') is not None:
                    system['tdp_watts'] = data['tdp_watts']
                if data.get('max_power_watts') is not None:
                    system['max_power_watts'] = data['max_power_watts']
                if data.get('release_date') is not None:
                    system['release_date'] = data['release_date']
                if data.get('end_of_life') is not None:
                    system['end_of_life'] = data['end_of_life']
                if data.get('manufacturer_url') is not None:
                    system['manufacturer_url'] = data['manufacturer_url']
                if data.get('notes') is not None:
                    system['notes'] = data['notes']
                if data.get('os_compatibility') is not None:
                    system['os_compatibility'] = data['os_compatibility']
                if data.get('isa_extensions') is not None:
                    system['isa_extensions'] = data['isa_extensions']
                if data.get('special_features') is not None:
                    system['special_features'] = data['special_features']

                if system:
                    data['system'] = system

        # Migrate old mapper fields to mapper if not already present
        if 'mapper' not in data or data['mapper'] is None:
            # Check if any old mapper fields are present
            has_old_mapper_fields = (
                data.get('mapper_class') is not None or
                data.get('mapper_config') is not None
            )

            if has_old_mapper_fields:
                # Migrate to new mapper structure
                mapper = {}

                if data.get('mapper_class') is not None:
                    mapper['mapper_class'] = data['mapper_class']

                if data.get('mapper_config') is not None:
                    mapper['mapper_config'] = data['mapper_config']

                if mapper:
                    data['mapper'] = mapper

        # Migrate old memory_subsystem list format to new dict format
        if 'memory_subsystem' in data and data['memory_subsystem'] is not None:
            if isinstance(data['memory_subsystem'], list):
                # Old format: list of channels
                old_channels = data['memory_subsystem']

                # Calculate total size and bandwidth
                total_size_gb = sum(ch.get('size_gb', 0) for ch in old_channels)
                total_bandwidth_gbps = sum(ch.get('bandwidth_gbps', 0) for ch in old_channels)

                # Create new structure
                new_memory_subsystem = {
                    "total_size_gb": total_size_gb,
                    "peak_bandwidth_gbps": total_bandwidth_gbps,
                    "memory_channels": old_channels
                }

                data['memory_subsystem'] = new_memory_subsystem

                # If there was a loose peak_bandwidth_gbps, remove it (it's now inside memory_subsystem)
                if 'peak_bandwidth_gbps' in data:
                    del data['peak_bandwidth_gbps']

        # Migrate old cache fields to onchip_memory_hierarchy.cache_levels if not already present
        if 'onchip_memory_hierarchy' not in data or data['onchip_memory_hierarchy'] is None:
            # Check if any old cache fields are present
            old_cache_fields = [
                'l1_data_cache_kb', 'l1_instruction_cache_kb', 'l1_cache_kb',
                'l2_cache_kb', 'l3_cache_kb',
                'l1_cache_line_size_bytes', 'l2_cache_line_size_bytes', 'l3_cache_line_size_bytes',
                'l1_cache_associativity', 'l2_cache_associativity', 'l3_cache_associativity'
            ]

            has_old_fields = any(data.get(field) is not None for field in old_cache_fields)

            if has_old_fields:
                # Migrate to new cache_levels structure
                cache_levels = []

                # L1 dcache (data cache) - assume per_core
                if data.get('l1_data_cache_kb') is not None or data.get('l1_dcache_kb') is not None:
                    l1d_size = data.get('l1_data_cache_kb') or data.get('l1_dcache_kb')
                    cache_levels.append({
                        "name": "L1 dcache",
                        "level": 1,
                        "cache_type": "data",
                        "scope": "per_core",
                        "size_per_unit_kb": l1d_size,
                        "associativity": data.get('l1_cache_associativity') or data.get('l1_dcache_associativity'),
                        "line_size_bytes": data.get('l1_cache_line_size_bytes')
                    })

                # L1 icache (instruction cache) - assume per_core
                if data.get('l1_instruction_cache_kb') is not None or data.get('l1_icache_kb') is not None:
                    l1i_size = data.get('l1_instruction_cache_kb') or data.get('l1_icache_kb')
                    cache_levels.append({
                        "name": "L1 icache",
                        "level": 1,
                        "cache_type": "instruction",
                        "scope": "per_core",
                        "size_per_unit_kb": l1i_size,
                        "associativity": data.get('l1_cache_associativity') or data.get('l1_icache_associativity'),
                        "line_size_bytes": data.get('l1_cache_line_size_bytes')
                    })

                # L2 cache - assume per_core (modern CPUs)
                if data.get('l2_cache_kb') is not None:
                    cache_levels.append({
                        "name": "L2",
                        "level": 2,
                        "cache_type": "unified",
                        "scope": "per_core",
                        "size_per_unit_kb": data.get('l2_cache_kb'),
                        "associativity": data.get('l2_cache_associativity'),
                        "line_size_bytes": data.get('l2_cache_line_size_bytes')
                    })

                # L3 cache - assume shared
                if data.get('l3_cache_kb') is not None:
                    cache_levels.append({
                        "name": "L3",
                        "level": 3,
                        "cache_type": "unified",
                        "scope": "shared",
                        "total_size_kb": data.get('l3_cache_kb'),
                        "associativity": data.get('l3_cache_associativity'),
                        "line_size_bytes": data.get('l3_cache_line_size_bytes')
                    })

                if cache_levels:
                    # Create onchip_memory_hierarchy with cache_levels
                    data['onchip_memory_hierarchy'] = {
                        "cache_levels": cache_levels,
                        # Also populate simple fields for backward compatibility
                        "l1_dcache_kb": data.get('l1_data_cache_kb') or data.get('l1_dcache_kb'),
                        "l1_icache_kb": data.get('l1_instruction_cache_kb') or data.get('l1_icache_kb'),
                        "l1_dcache_associativity": data.get('l1_cache_associativity') or data.get('l1_dcache_associativity'),
                        "l1_icache_associativity": data.get('l1_cache_associativity') or data.get('l1_icache_associativity'),
                        "l1_cache_line_size_bytes": data.get('l1_cache_line_size_bytes'),
                        "l2_cache_kb": data.get('l2_cache_kb'),
                        "l2_cache_associativity": data.get('l2_cache_associativity'),
                        "l2_cache_line_size_bytes": data.get('l2_cache_line_size_bytes'),
                        "l3_cache_kb": data.get('l3_cache_kb'),
                        "l3_cache_associativity": data.get('l3_cache_associativity'),
                        "l3_cache_line_size_bytes": data.get('l3_cache_line_size_bytes'),
                    }

        # Ensure onchip_memory_hierarchy exists (even if empty cache_levels)
        if 'onchip_memory_hierarchy' not in data or data['onchip_memory_hierarchy'] is None:
            data['onchip_memory_hierarchy'] = {"cache_levels": []}

        # Clean up deprecated fields that have been migrated
        # These fields should NOT be passed to the constructor
        deprecated_fields = [
            # Old core fields (migrated to core_info)
            'cores', 'threads', 'e_cores', 'base_frequency_ghz', 'boost_frequency_ghz', 'core_clusters',
            # Old memory fields (migrated to memory_subsystem)
            'peak_bandwidth_gbps',
            # Old cache fields (migrated to onchip_memory_hierarchy)
            'l1_data_cache_kb', 'l1_instruction_cache_kb', 'l1_cache_kb',
            'l1_dcache_kb', 'l1_icache_kb',
            'l2_cache_kb', 'l3_cache_kb',
            'l1_cache_line_size_bytes', 'l2_cache_line_size_bytes', 'l3_cache_line_size_bytes',
            'l1_cache_associativity', 'l2_cache_associativity', 'l3_cache_associativity',
            'l1_dcache_associativity', 'l1_icache_associativity',
            # Old system fields (migrated to system)
            'vendor', 'model', 'architecture', 'device_type', 'platform',
            'os_compatibility', 'isa_extensions', 'special_features',
            'tdp_watts', 'max_power_watts',
            'release_date', 'end_of_life', 'manufacturer_url', 'notes',
            # Old mapper fields (migrated to mapper)
            'mapper_class', 'mapper_config',
        ]

        for field in deprecated_fields:
            if field in data:
                del data[field]

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

    def get_core_info(self) -> Optional[CoreInfo]:
        """
        Get core_info as CoreInfo dataclass.

        Returns:
            CoreInfo object if core_info is specified, None otherwise
        """
        if not self.core_info:
            return None
        return CoreInfo.from_dict(self.core_info)

    def get_system_info(self) -> Optional['SystemInfo']:
        """
        Get system as SystemInfo dataclass.

        Returns:
            SystemInfo object if system is specified, None otherwise
        """
        if not self.system:
            return None
        return SystemInfo.from_dict(self.system)

    def get_mapper_info(self) -> Optional['MapperInfo']:
        """
        Get mapper as MapperInfo dataclass.

        Returns:
            MapperInfo object if mapper is specified, None otherwise
        """
        if not self.mapper:
            return None
        return MapperInfo.from_dict(self.mapper)

    def has_heterogeneous_cores(self) -> bool:
        """
        Check if this CPU has heterogeneous core architecture.

        Returns:
            True if core_clusters is specified, False otherwise
        """
        # Check both core_info.core_clusters and legacy core_clusters
        if self.core_info:
            core_info_obj = self.get_core_info()
            return core_info_obj and core_info_obj.core_clusters is not None and len(core_info_obj.core_clusters) > 0
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

    def get_memory_subsystem(self) -> List['MemorySubsystem']:
        """
        Get memory channels from memory_subsystem as MemorySubsystem objects.

        Returns:
            List of MemorySubsystem objects (memory channels), or empty list if not specified
        """
        if not self.memory_subsystem:
            return []

        # New format: dict with memory_channels
        if isinstance(self.memory_subsystem, dict):
            channels = self.memory_subsystem.get('memory_channels', [])
            return [MemorySubsystem.from_dict(mem) for mem in channels]

        # Old format: list (for backward compatibility - should have been migrated)
        return [MemorySubsystem.from_dict(mem) for mem in self.memory_subsystem]

    def has_memory_subsystem(self) -> bool:
        """
        Check if detailed memory subsystem is specified.

        Returns:
            True if memory_subsystem is specified, False otherwise
        """
        return self.memory_subsystem is not None and len(self.memory_subsystem) > 0

    def get_onchip_memory_hierarchy(self) -> Optional['OnChipMemoryHierarchy']:
        """
        Get on-chip memory hierarchy (cache configuration) as OnChipMemoryHierarchy object.

        Returns:
            OnChipMemoryHierarchy object, or None if not specified
        """
        if not self.onchip_memory_hierarchy:
            return None

        return OnChipMemoryHierarchy.from_dict(self.onchip_memory_hierarchy)

    def has_onchip_memory_hierarchy(self) -> bool:
        """
        Check if on-chip memory hierarchy is specified.

        Returns:
            True if onchip_memory_hierarchy is specified, False otherwise
        """
        return self.onchip_memory_hierarchy is not None

    def compute_total_memory_gb(self) -> float:
        """
        Compute total memory capacity from subsystem.

        Returns:
            Total memory in GB
        """
        if self.has_memory_subsystem():
            return sum(mem.size_gb for mem in self.get_memory_subsystem())
        return 0.0

    def compute_total_bandwidth_gbps(self) -> float:
        """
        Compute total memory bandwidth from subsystem.

        Returns:
            Total theoretical bandwidth in GB/s
        """
        if self.has_memory_subsystem():
            return sum(mem.bandwidth_gbps for mem in self.get_memory_subsystem())
        return self.peak_bandwidth_gbps

    def compute_total_effective_bandwidth_gbps(self) -> float:
        """
        Compute total effective memory bandwidth accounting for ECC overhead.

        Returns:
            Total effective bandwidth in GB/s
        """
        if self.has_memory_subsystem():
            return sum(mem.compute_effective_bandwidth() for mem in self.get_memory_subsystem())
        return self.peak_bandwidth_gbps

    def compute_total_bus_width_bits(self) -> int:
        """
        Compute total memory bus width from subsystem.

        Returns:
            Total bus width in bits
        """
        if self.has_memory_subsystem():
            return sum(mem.bus_width_bits for mem in self.get_memory_subsystem())
        return self.memory_bus_width or 0

    def get_numa_nodes(self) -> List[int]:
        """
        Get list of NUMA nodes present in memory subsystem.

        Returns:
            Sorted list of unique NUMA node IDs
        """
        if not self.has_memory_subsystem():
            return []

        numa_nodes = set()
        for mem in self.get_memory_subsystem():
            if mem.numa_node is not None:
                numa_nodes.add(mem.numa_node)

        return sorted(numa_nodes)

    def has_ecc_memory(self) -> bool:
        """
        Check if any memory channel has ECC enabled.

        Returns:
            True if at least one channel has ECC enabled
        """
        if not self.has_memory_subsystem():
            return False

        return any(mem.ecc_enabled for mem in self.get_memory_subsystem() if mem.ecc_enabled is not None)

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

        # Validate system info (consolidated fields)
        if self.system:
            system_info = self.get_system_info()
            if not system_info:
                errors.append("system is specified but cannot be parsed as SystemInfo")
            else:
                if not system_info.vendor:
                    errors.append("Missing required field: system.vendor")
                if not system_info.model:
                    errors.append("Missing required field: system.model")
                if not system_info.device_type:
                    errors.append("Missing required field: system.device_type")
                if system_info.device_type and system_info.device_type not in ['cpu', 'gpu', 'tpu', 'kpu', 'dpu', 'cgra']:
                    errors.append(f"Invalid system.device_type: {system_info.device_type}")
                if not system_info.platform:
                    errors.append("Missing required field: system.platform")
                if system_info.platform and system_info.platform not in ['x86_64', 'aarch64', 'arm64']:
                    errors.append(f"Invalid system.platform: {system_info.platform}")
        else:
            # Fall back to legacy fields (for backward compatibility)
            if not self.vendor:
                errors.append("Missing required field: vendor (or system.vendor)")
            if not self.model:
                errors.append("Missing required field: model (or system.model)")
            if not self.device_type:
                errors.append("Missing required field: device_type (or system.device_type)")
            if self.device_type and self.device_type not in ['cpu', 'gpu', 'tpu', 'kpu', 'dpu', 'cgra']:
                errors.append(f"Invalid device_type: {self.device_type}")
            if not self.platform:
                errors.append("Missing required field: platform (or system.platform)")
            if self.platform and self.platform not in ['x86_64', 'aarch64', 'arm64']:
                errors.append(f"Invalid platform: {self.platform}")

        # Validate theoretical_peaks - ALL precisions must be present
        if not self.theoretical_peaks:
            errors.append("Missing required field: theoretical_peaks")
        else:
            missing_precisions = []
            for precision in REQUIRED_PRECISIONS:
                if precision not in self.theoretical_peaks:
                    missing_precisions.append(precision)

            if missing_precisions:
                errors.append(
                    f"theoretical_peaks missing required precisions: {', '.join(missing_precisions)}. "
                    f"All precisions must be explicitly set. Use 0.0 for unsupported precisions."
                )

            # Validate that values are non-negative
            for precision, value in self.theoretical_peaks.items():
                if value < 0:
                    errors.append(f"theoretical_peaks[{precision}] must be >= 0 (got {value})")

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

        # Validate memory_subsystem if specified
        if self.memory_subsystem:
            if isinstance(self.memory_subsystem, dict):
                # New format: dict with total_size_gb, peak_bandwidth_gbps, memory_channels
                if 'total_size_gb' not in self.memory_subsystem or self.memory_subsystem['total_size_gb'] <= 0:
                    errors.append("memory_subsystem: total_size_gb must be specified and positive")

                if 'peak_bandwidth_gbps' not in self.memory_subsystem or self.memory_subsystem['peak_bandwidth_gbps'] <= 0:
                    errors.append("memory_subsystem: peak_bandwidth_gbps must be specified and positive")

                if 'memory_channels' not in self.memory_subsystem:
                    errors.append("memory_subsystem: memory_channels must be specified")
                else:
                    # Validate each channel
                    for i, mem_dict in enumerate(self.memory_subsystem['memory_channels']):
                        try:
                            mem = MemorySubsystem.from_dict(mem_dict)
                            if not mem.name:
                                errors.append(f"memory_subsystem.memory_channels[{i}]: Missing name")
                            if not mem.type:
                                errors.append(f"memory_subsystem.memory_channels[{i}]: Missing type")
                            if mem.size_gb <= 0:
                                errors.append(f"memory_subsystem.memory_channels[{i}]: size_gb must be positive")
                            if mem.bandwidth_gbps <= 0:
                                errors.append(f"memory_subsystem.memory_channels[{i}]: bandwidth_gbps must be positive")
                        except Exception as e:
                            errors.append(f"memory_subsystem.memory_channels[{i}]: Invalid memory configuration: {e}")
            else:
                # Old format: list (should have been migrated, but validate anyway)
                for i, mem_dict in enumerate(self.memory_subsystem):
                    try:
                        mem = MemorySubsystem.from_dict(mem_dict)
                        if not mem.name:
                            errors.append(f"memory_subsystem[{i}]: Missing name")
                        if not mem.type:
                            errors.append(f"memory_subsystem[{i}]: Missing type")
                        if mem.size_gb <= 0:
                            errors.append(f"memory_subsystem[{i}]: size_gb must be positive")
                        if mem.bandwidth_gbps <= 0:
                            errors.append(f"memory_subsystem[{i}]: bandwidth_gbps must be positive")
                    except Exception as e:
                        errors.append(f"memory_subsystem[{i}]: Invalid memory configuration: {e}")

            # Check for uniform configuration within same NUMA node (no asymmetric)
            # Channels on same NUMA node should have same size and type
            # Channels on different NUMA nodes can differ (e.g., Grace Hopper)
            mem_configs = self.get_memory_subsystem()
            if len(mem_configs) > 1:
                # Group by NUMA node
                numa_groups = {}
                for mem in mem_configs:
                    numa_node = mem.numa_node if mem.numa_node is not None else 0
                    if numa_node not in numa_groups:
                        numa_groups[numa_node] = []
                    numa_groups[numa_node].append(mem)

                # Check uniformity within each NUMA node
                for numa_node, mems in numa_groups.items():
                    if len(mems) > 1:
                        first_mem = mems[0]
                        for mem in mems[1:]:
                            if mem.type != first_mem.type:
                                errors.append(
                                    f"Asymmetric memory configuration on NUMA node {numa_node}: "
                                    f"{mem.name} type ({mem.type}) differs from {first_mem.name} ({first_mem.type}). "
                                    f"Memory controllers down-configure mixed types."
                                )
                            if mem.size_gb != first_mem.size_gb:
                                errors.append(
                                    f"Asymmetric memory configuration on NUMA node {numa_node}: "
                                    f"{mem.name} size ({mem.size_gb}GB) differs from {first_mem.name} ({first_mem.size_gb}GB). "
                                    f"Consider using uniform memory sizes."
                                )
                            if mem.data_rate_mts != first_mem.data_rate_mts:
                                errors.append(
                                    f"Asymmetric memory configuration on NUMA node {numa_node}: "
                                    f"{mem.name} speed ({mem.data_rate_mts}MT/s) differs from {first_mem.name} ({first_mem.data_rate_mts}MT/s). "
                                    f"Memory controller will down-configure to lowest speed."
                                )

        # Theoretical peaks should have at least fp32
        if not self.theoretical_peaks or 'fp32' not in self.theoretical_peaks:
            errors.append("theoretical_peaks must include 'fp32'")

        # Peak bandwidth should be positive (if not using memory_subsystem)
        if not self.has_memory_subsystem() and self.peak_bandwidth_gbps <= 0:
            errors.append("peak_bandwidth_gbps must be positive (or use memory_subsystem)")

        # Mapper configuration should be specified
        if self.mapper:
            mapper_info = self.get_mapper_info()
            if not mapper_info:
                errors.append("mapper is specified but cannot be parsed as MapperInfo")
            elif not mapper_info.mapper_class:
                errors.append("Missing required field: mapper.mapper_class")
        else:
            # Fall back to legacy field (for backward compatibility)
            if not self.mapper_class:
                errors.append("Missing required field: mapper_class (or mapper.mapper_class)")

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
