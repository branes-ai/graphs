# Jetson Orin AGX power statistics

Are there CLI commands that can provide the same information as the Jetson Power GUI?

```bash
  # Power mode and clock limits
  sudo nvpmodel -q                    # Current power mode
  sudo nvpmodel -p --verbose          # All modes with frequency limits

  # Current clocks
  sudo jetson_clocks --show           # CPU, GPU, EMC current frequencies

  # Thermal
  cat /sys/class/thermal/thermal_zone*/type /sys/class/thermal/thermal_zone*/temp 2>/dev/null

  # Power consumption (real-time)
  cat /sys/bus/i2c/drivers/ina3221/*/hwmon/hwmon*/in*_input 2>/dev/null   # voltage
  cat /sys/bus/i2c/drivers/ina3221/*/hwmon/hwmon*/curr*_input 2>/dev/null  # current

  # Or use tegrastats for a live dashboard (all-in-one)
  sudo tegrastats --interval 1000
```

**tegrastats** is the most useful -- it streams CPU/GPU load, frequencies, temperatures, power draw, and memory usage in one line per second. 
It's essentially the Power GUI in text form.

```bash
cjuser@sdhome-jetson:/mnt/nvme/dev/branes/clones/graphs$ sudo jetson_clocks --show
  SOC family:tegra234  Machine:Jetson AGX Orin for DSBOARD-AGX
  Online CPUs: 0-11
  cpu0: Online=1 Governor=schedutil MinFreq=729600 MaxFreq=2201600 CurrentFreq=2201600 IdleStates: WFI=1 c7=1
  cpu1: Online=1 Governor=schedutil MinFreq=729600 MaxFreq=2201600 CurrentFreq=2201600 IdleStates: WFI=1 c7=1
  cpu10: Online=1 Governor=schedutil MinFreq=729600 MaxFreq=2201600 CurrentFreq=2201600 IdleStates: WFI=1 c7=1
  cpu11: Online=1 Governor=schedutil MinFreq=729600 MaxFreq=2201600 CurrentFreq=2201600 IdleStates: WFI=1 c7=1
  cpu2: Online=1 Governor=schedutil MinFreq=729600 MaxFreq=2201600 CurrentFreq=2201600 IdleStates: WFI=1 c7=1
  cpu3: Online=1 Governor=schedutil MinFreq=729600 MaxFreq=2201600 CurrentFreq=2201600 IdleStates: WFI=1 c7=1
  cpu4: Online=1 Governor=schedutil MinFreq=729600 MaxFreq=2201600 CurrentFreq=2201600 IdleStates: WFI=1 c7=1
  cpu5: Online=1 Governor=schedutil MinFreq=729600 MaxFreq=2201600 CurrentFreq=2201600 IdleStates: WFI=1 c7=1
  cpu6: Online=1 Governor=schedutil MinFreq=729600 MaxFreq=2201600 CurrentFreq=2201600 IdleStates: WFI=1 c7=1
  cpu7: Online=1 Governor=schedutil MinFreq=729600 MaxFreq=2201600 CurrentFreq=2201600 IdleStates: WFI=1 c7=1
  cpu8: Online=1 Governor=schedutil MinFreq=729600 MaxFreq=2201600 CurrentFreq=2201600 IdleStates: WFI=1 c7=1
  cpu9: Online=1 Governor=schedutil MinFreq=729600 MaxFreq=2201600 CurrentFreq=2201600 IdleStates: WFI=1 c7=1
  GPU MinFreq=306000000 MaxFreq=1300500000 CurrentFreq=510000000
  EMC MinFreq=204000000 MaxFreq=3199000000 CurrentFreq=2133000000 FreqOverride=0
  DLA0_CORE:   Online=1 MinFreq=0 MaxFreq=1600000000 CurrentFreq=1600000000
  DLA0_FALCON: Online=1 MinFreq=0 MaxFreq=844800000 CurrentFreq=844800000
  DLA1_CORE:   Online=1 MinFreq=0 MaxFreq=1600000000 CurrentFreq=1600000000
  DLA1_FALCON: Online=1 MinFreq=0 MaxFreq=844800000 CurrentFreq=844800000
  PVA0_VPS0: Online=1 MinFreq=0 MaxFreq=1152000000 CurrentFreq=1152000000
  PVA0_AXI:  Online=1 MinFreq=0 MaxFreq=832000000 CurrentFreq=832000000
  FAN Dynamic Speed control=active hwmon4_pwm1=3
  NV Power Mode: MAXN
```

## Here's what each resource is:

  EMC (External Memory Controller) -- The DRAM controller. Frequency determines memory bandwidth. At 3199 MHz max, this drives the ~204 GB/s LPDDR5
  bandwidth. FreqOverride=0 means DVFS is managing it dynamically.

  DLA0, DLA1 (Deep Learning Accelerators) -- Two fixed-function neural network inference engines. Each has:
  - CORE: The MAC (multiply-accumulate) array that runs convolutions, matmuls, etc. At 1.6 GHz, each DLA delivers ~5.3 INT8 TOPS.
  - FALCON: A small microcontroller (Falcon Architecture for Low-power CONtrol) that manages DLA scheduling, DMA, and sequencing. It's the control
  processor, not a compute unit.

  PVA0 (Programmable Vision Accelerator) -- A vector processor for classical computer vision (stereo depth, optical flow, feature detection). Not used
  for DNN inference. Has:
  - VPS0: Vector Processing Subsystem -- the VLIW compute cores.
  - AXI: The bus interface connecting PVA to the memory fabric.

  FAN -- Fan speed control. Dynamic Speed control=active means the fan adjusts based on thermal load. pwm1=3 is a very low duty cycle (cool system).

  In summary, the AGX Orin has 5 distinct compute engines:
```text
  ┌──────────────┬─────────────────┬────────────────────────────────┬───────────────────┐
  │    Engine    │      Count      │            Purpose             │     Framework     │
  ├──────────────┼─────────────────┼────────────────────────────────┼───────────────────┤
  │ CPU (A78AE)  │ 12 cores        │ General compute                │ PyTorch, NumPy    │
  ├──────────────┼─────────────────┼────────────────────────────────┼───────────────────┤
  │ GPU (Ampere) │ 2048 CUDA cores │ Parallel compute, DNN training │ PyTorch CUDA      │
  ├──────────────┼─────────────────┼────────────────────────────────┼───────────────────┤
  │ DLA          │ 2 engines       │ DNN inference only             │ TensorRT          │
  ├──────────────┼─────────────────┼────────────────────────────────┼───────────────────┤
  │ PVA          │ 1 engine        │ Classical CV                   │ VisionWorks, VPI  │
  ├──────────────┼─────────────────┼────────────────────────────────┼───────────────────┤
  │ NVENC/NVDEC  │ (not shown)     │ Video encode/decode            │ GStreamer, FFmpeg │
  └──────────────┴─────────────────┴────────────────────────────────┴───────────────────┘
```

Our calibration currently only exercises CPU and GPU. The DLAs are accessible through TensorRT, not PyTorch directly.

