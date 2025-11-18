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
class OnChipMemoryHierarchy:
    """
    On-chip memory hierarchy (cache subsystem).

    Organizes L1/L2/L3 cache configuration with separate tracking for
    L1 data cache (dcache) and instruction cache (icache).
    """

    # L1 Data Cache
    l1_dcache_kb: Optional[int] = None
    """L1 data cache size in KB (per core or total, see notes)"""

    l1_dcache_associativity: Optional[int] = None
    """L1 data cache associativity (e.g., 8-way, 12-way)"""

    # L1 Instruction Cache
    l1_icache_kb: Optional[int] = None
    """L1 instruction cache size in KB (per core or total, see notes)"""

    l1_icache_associativity: Optional[int] = None
    """L1 instruction cache associativity (e.g., 8-way)"""

    # L1 Common
    l1_cache_line_size_bytes: Optional[int] = None
    """L1 cache line size in bytes (typically 64 bytes for modern CPUs)"""

    # L2 Cache
    l2_cache_kb: Optional[int] = None
    """L2 cache size in KB (per core or total, see notes)"""

    l2_cache_associativity: Optional[int] = None
    """L2 cache associativity (e.g., 16-way, 20-way)"""

    l2_cache_line_size_bytes: Optional[int] = None
    """L2 cache line size in bytes (typically 64 bytes)"""

    # L3 Cache
    l3_cache_kb: Optional[int] = None
    """L3 cache size in KB (typically shared, total capacity)"""

    l3_cache_associativity: Optional[int] = None
    """L3 cache associativity (e.g., 12-way, 16-way)"""

    l3_cache_line_size_bytes: Optional[int] = None
    """L3 cache line size in bytes (typically 64 bytes)"""

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'OnChipMemoryHierarchy':
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
    DEPRECATED: Use compute_total_bandwidth_gbps() with memory_subsystem.
    """

    memory_subsystem: Optional[List[Dict]] = None
    """
    Detailed memory subsystem configuration (channels/stacks/banks).

    Each entry describes a memory channel (CPU DDR), HBM stack (GPU datacenter),
    GDDR bank (GPU consumer), or LPDDR channel (mobile/embedded).

    Example (CPU Dual-Channel DDR5):
    [
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

    Example (GPU HBM3):
    [
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
        },
        // ... stacks 1-4
    ]

    When memory_subsystem is specified, it is the authoritative source for
    memory configuration. Legacy fields (memory_type, peak_bandwidth_gbps)
    should be computed from memory_subsystem.
    """

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
        """
        Create from dictionary (from JSON).

        Handles backward compatibility by migrating old cache fields to onchip_memory_hierarchy.
        """
        # Make a copy to avoid mutating the input
        data = data.copy()

        # Migrate old cache fields to onchip_memory_hierarchy if not already present
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
                # Migrate to new structure
                onchip = {}

                # L1 dcache (data cache)
                if data.get('l1_data_cache_kb') is not None:
                    onchip['l1_dcache_kb'] = data['l1_data_cache_kb']

                # L1 icache (instruction cache)
                if data.get('l1_instruction_cache_kb') is not None:
                    onchip['l1_icache_kb'] = data['l1_instruction_cache_kb']

                # L1 associativity (apply to both dcache and icache for now)
                if data.get('l1_cache_associativity') is not None:
                    onchip['l1_dcache_associativity'] = data['l1_cache_associativity']
                    onchip['l1_icache_associativity'] = data['l1_cache_associativity']

                # L1 cache line size
                if data.get('l1_cache_line_size_bytes') is not None:
                    onchip['l1_cache_line_size_bytes'] = data['l1_cache_line_size_bytes']

                # L2 cache
                if data.get('l2_cache_kb') is not None:
                    onchip['l2_cache_kb'] = data['l2_cache_kb']
                if data.get('l2_cache_associativity') is not None:
                    onchip['l2_cache_associativity'] = data['l2_cache_associativity']
                if data.get('l2_cache_line_size_bytes') is not None:
                    onchip['l2_cache_line_size_bytes'] = data['l2_cache_line_size_bytes']

                # L3 cache
                if data.get('l3_cache_kb') is not None:
                    onchip['l3_cache_kb'] = data['l3_cache_kb']
                if data.get('l3_cache_associativity') is not None:
                    onchip['l3_cache_associativity'] = data['l3_cache_associativity']
                if data.get('l3_cache_line_size_bytes') is not None:
                    onchip['l3_cache_line_size_bytes'] = data['l3_cache_line_size_bytes']

                if onchip:
                    data['onchip_memory_hierarchy'] = onchip

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

    def get_memory_subsystem(self) -> List['MemorySubsystem']:
        """
        Get memory subsystem as MemorySubsystem objects.

        Returns:
            List of MemorySubsystem objects, or empty list if not specified
        """
        if not self.memory_subsystem:
            return []

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

        # Validate memory_subsystem if specified
        if self.memory_subsystem:
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
