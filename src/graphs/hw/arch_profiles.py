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

# Profiles
cpu_profile = ArchitectureProfile(
    name="CPU",
    peak_flops=100e9,
    mem_bandwidth=50e9,
    energy_per_flop=1e-9,
    energy_per_byte=100e-12,
    scheduler_model=default_scheduler,
    tiling_strategy=CPUTilingStrategy()
)

gpu_profile = ArchitectureProfile(
    name="GPU",
    peak_flops=10e12,
    mem_bandwidth=900e9,
    energy_per_flop=0.5e-9,
    energy_per_byte=30e-12,
    scheduler_model=fused_scheduler,
    tiling_strategy=GPUTilingStrategy()
)

tpu_profile = ArchitectureProfile(
    name="TPU",
    peak_flops=45e12,
    mem_bandwidth=600e9,
    energy_per_flop=0.2e-9,
    energy_per_byte=10e-12,
    scheduler_model=fused_scheduler,
    tiling_strategy=TPUTilingStrategy()
)

kpu_profile = ArchitectureProfile(
    name="KPU",
    peak_flops=1e12,
    mem_bandwidth=100e9,
    energy_per_flop=0.1e-9,
    energy_per_byte=5e-12,
    scheduler_model=fused_scheduler,
    tiling_strategy=KPUTilingStrategy()
)
