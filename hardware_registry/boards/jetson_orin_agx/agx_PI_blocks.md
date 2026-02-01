# NVIDIA Jetson Orin AGX IP Blocks

The Jetson Orin AGX is one SoC (NVIDIA Tegra234). The SOC1/SOC2/SOC3 labels in tegrastats are thermal sensors placed at different physical
locations on the die, not separate chips. Here's what's on the SoC:

## Orin SoC (Tegra234) IP Blocks

```text
  ┌──────────┬─────────────────────────────────────┬──────────────────────────────┐
  │ Category │               Engines               │           Purpose            │
  ├──────────┼─────────────────────────────────────┼──────────────────────────────┤
  │ CPU      │ 12x Arm Cortex-A78AE                │ General compute              │
  ├──────────┼─────────────────────────────────────┼──────────────────────────────┤
  │ GPU      │ Ampere (2048 CUDA, 64 Tensor Cores) │ Graphics + ML inference      │
  ├──────────┼─────────────────────────────────────┼──────────────────────────────┤
  │ DLA      │ 2x DLA 2.0 cores                    │ Fixed-function CNN inference │
  ├──────────┼─────────────────────────────────────┼──────────────────────────────┤
  │ PVA      │ Programmable Vision Accelerator     │ Computer vision (stereo, OF) │
  ├──────────┼─────────────────────────────────────┼──────────────────────────────┤
  │ NVJPG    │ JPEG encoder/decoder                │ Image codec                  │
  ├──────────┼─────────────────────────────────────┼──────────────────────────────┤
  │ NVENC    │ Video encoder                       │ H.264/H.265 encode           │
  ├──────────┼─────────────────────────────────────┼──────────────────────────────┤
  │ NVDEC    │ Video decoder                       │ H.264/H.265/AV1 decode       │
  ├──────────┼─────────────────────────────────────┼──────────────────────────────┤
  │ VIC      │ Video Image Compositor              │ Scaling, color conversion    │
  ├──────────┼─────────────────────────────────────┼──────────────────────────────┤
  │ ISP      │ Image Signal Processor              │ Camera RAW processing        │
  ├──────────┼─────────────────────────────────────┼──────────────────────────────┤
  │ VI       │ Video Input                         │ Camera capture               │
  ├──────────┼─────────────────────────────────────┼──────────────────────────────┤
  │ EMC      │ External Memory Controller          │ LPDDR5 interface             │
  ├──────────┼─────────────────────────────────────┼──────────────────────────────┤
  │ OFA      │ Optical Flow Accelerator            │ Dense optical flow           │
  ├──────────┼─────────────────────────────────────┼──────────────────────────────┤
  │ APE      │ Audio Processing Engine             │ Audio DSP                    │
  ├──────────┼─────────────────────────────────────┼──────────────────────────────┤
  │ SE       │ Security Engine                     │ Crypto/secure boot           │
  └──────────┴─────────────────────────────────────┴──────────────────────────────┘
```

## Thermal Zones in tegrastats

The SOC0/SOC1/SOC2 readings are on-die thermal sensors monitoring different regions of the SoC fabric (interconnect, memory controllers, peripheral
subsystems) -- not the CPU or GPU clusters which have their own dedicated sensors. NVIDIA doesn't publicly document which specific IP block each SOC
sensor sits nearest. They're used alongside CPU-therm and GPU-therm in the weighted Tmargin calculation that drives fan speed and thermal throttling.

You can see all thermal zones on your board:
```bash
for z in /sys/class/thermal/thermal_zone*/; do echo "$(cat $z/type): $(cat $z/temp)"; done
```

Sources:
  - https://docs.nvidia.com/jetson/archives/r35.6.2/DeveloperGuide/SD/PlatformPowerAndPerformance/JetsonOrinNanoSeriesJetsonOrinNxSeriesAndJetsonAgxOri
nSeries.html
  - https://forums.developer.nvidia.com/t/tegrastats-cv-temperature-on-jetson-orin/238424

