# characterize/arch_profiles.py

from graphs.compile.tiling import (
    CPUTilingStrategy,
    GPUTilingStrategy,
    TPUTilingStrategy,
    KPUTilingStrategy
)

class ArchitectureProfile:
    def __init__(self, name, peak_flops, mem_bandwidth, energy_per_flop,
                 energy_per_byte, scheduler_model, tiling_strategy):
        self.name = name
        self.peak_flops = peak_flops
        self.mem_bandwidth = mem_bandwidth
        self.energy_per_flop = energy_per_flop
        self.energy_per_byte = energy_per_byte
        self.scheduler_model = scheduler_model
        self.tiling_strategy = tiling_strategy

# Scheduler models (can be refined later)
def default_scheduler(_): return 1.0
def fused_scheduler(_): return 0.6


# Multi-core CPU profiles
# Precision   Intel Core i7      AMD Ryzen 7
# FP32         1.5 TFLOPS          1.0 TFLOPS
# BF16
# INT8        6 TOPS               4 TOPS
# DDR5         80 GB/s             80 GB/s


# Profiles
intel_i7_profile = ArchitectureProfile(
    name="Intel Core i7",
    peak_flops=1.5e12,
    mem_bandwidth=80e9,
    energy_per_flop=1e-9,
    energy_per_byte=100e-12,
    scheduler_model=default_scheduler,
    tiling_strategy=CPUTilingStrategy()
)

amd_ryzen7_profile = ArchitectureProfile(
    name="AMD Ryzen 7",
    peak_flops=1.0e12,
    mem_bandwidth=80e9,
    energy_per_flop=1e-9,
    energy_per_byte=100e-12,
    scheduler_model=default_scheduler,
    tiling_strategy=CPUTilingStrategy()
)

#gpu_profile = ArchitectureProfile(
#    name="GPU",
#    peak_flops=10e12,
#    mem_bandwidth=900e9,
#    energy_per_flop=0.5e-9,
#    energy_per_byte=30e-12,
#    scheduler_model=fused_scheduler,
#    tiling_strategy=GPUTilingStrategy()
#)

# NVIDIA H100 SXM5 and PCIe profiles
# Precision     SXM5           PCIe
# FP32         67 TFLOPS    51 TFLOPS
# BF16        980 TFLOPS   750 TFLOPS  Raw Peak
# BF16       1979 TFLOPS  1513 TFLOPS  Sparsity
# INT8       1979 TOPS    1513 TOPS    Raw Peak
# INT8       3958 TOPS    3026 TOPS    Sparsity
# HBM3       3350 GB/s
# HBM2e                   2000 GB/s
h100_pcie_profile = ArchitectureProfile(
    name="H100-PCIe",
    peak_flops=750e12,    # use BF16 as baseline
    mem_bandwidth=2000e9,
    energy_per_flop=0.5e-9,
    energy_per_byte=30e-12,
    scheduler_model=fused_scheduler,
    tiling_strategy=GPUTilingStrategy()
)

# Google TPUv4 profile
# Precision    v4
# BF16         275 TFLOPS
# INT8         275 TOPS
# HBM2e        1200 GB/s
tpu_v4_profile = ArchitectureProfile(
    name="TPU",
    peak_flops=275e12,
    mem_bandwidth=1200e9,
    energy_per_flop=0.2e-9,
    energy_per_byte=10e-12,
    scheduler_model=fused_scheduler,
    tiling_strategy=TPUTilingStrategy()
)

# Chip Structure: 
#   A single TPU v4 chip typically contains two TensorCores, which Google refers to 
#   as a "megacore" configuration that share a unified 32 GB HBM memory space.
# 
# TPU v4i: 
#   A specific inference variant, the TPU v4i, is a single-core chip (half the compute 
#   of the full TPU v4) designed for air cooling and optimized for inference workloads. 
#   Its peak performance is roughly half the dual-core TPU v4.
# 
# Scalability: 
#   TPUs are designed to scale massively using a custom, high-speed optical interconnect, 
#   which allows a single TPU v4 Pod of 4,096 chips to achieve an aggregate compute of 1.1 ExaFLOPS.


# KPUs are very scalable, so there are many different profiles possible.
# The nomenclature is T###, representing the available TOPS capability of the SoC

kpu_t2_profile = ArchitectureProfile(
    name="KPU-T2",
    peak_flops=2e12,
    mem_bandwidth=165e9,    # 86bit bus
    energy_per_flop=0.1e-9,
    energy_per_byte=5e-12,
    scheduler_model=fused_scheduler,
    tiling_strategy=KPUTilingStrategy()
)

kpu_t100_profile = ArchitectureProfile(
    name="KPU-T100",
    peak_flops=100e12,
    mem_bandwidth=1000e9,   # low-end HBM
    energy_per_flop=0.1e-9,
    energy_per_byte=2.5e-12,
    scheduler_model=fused_scheduler,
    tiling_strategy=KPUTilingStrategy()
)


# GDDR6 memory subsystem profile
#  Component           Typical Specifications        Peak System Bandwidth
# Data Rate (per pin)  14 to 18 Gbps                  N/A
# Mid-Range Bus Width  192-bit or 256-bit             336 to 576 GB/s
# High-End Bus Width   384-bit                        672 to 768 GB/s
