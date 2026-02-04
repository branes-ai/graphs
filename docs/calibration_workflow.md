# Calibration Workflow


## Directory Structure

```text
  calibration_data/
    i7-12700K/
      measurements/
        resnet18.json
        resnet50.json
        ...
      efficiency_curves.json    # Aggregated curves
      calibration_report.md     # Summary

    ryzen-7-7800x3d/
      measurements/
        ...
      efficiency_curves.json

    jetson-orin-agx-50w/
      measurements/
        ...
      efficiency_curves.json
    ...
```


  Pre-configured Hardware IDs
  ┌──────────────────────┬────────┬─────────┬──────────────────────┐
  │     Hardware ID      │ Device │ Profile │     Description      │
  ├──────────────────────┼────────┼─────────┼──────────────────────┤
  │ i7-12700K            │ cpu    │ default │ Intel Core i7-12700K │
  ├──────────────────────┼────────┼─────────┼──────────────────────┤
  │ ryzen-7-7800x3d      │ cpu    │ default │ AMD Ryzen 7 7800X3D  │
  ├──────────────────────┼────────┼─────────┼──────────────────────┤
  │ ryzen-9-7950x        │ cpu    │ default │ AMD Ryzen 9 7950X    │
  ├──────────────────────┼────────┼─────────┼──────────────────────┤
  │ jetson-orin-agx-50w  │ cuda   │ 50W     │ Jetson AGX Orin 64GB │
  ├──────────────────────┼────────┼─────────┼──────────────────────┤
  │ jetson-orin-agx-30w  │ cuda   │ 30W     │ Jetson AGX Orin 64GB │
  ├──────────────────────┼────────┼─────────┼──────────────────────┤
  │ jetson-orin-nano-15w │ cuda   │ 15W     │ Jetson Orin Nano 8GB │
  ├──────────────────────┼────────┼─────────┼──────────────────────┤
  │ jetson-orin-nx-25w   │ cuda   │ 25W     │ Jetson Orin NX 16GB  │
  └──────────────────────┴────────┴─────────┴──────────────────────┘

## Commands for Each System

```bash
  # Intel i7:
  ./cli/calibrate_efficiency.py --hardware-id i7-12700K --device cpu

  # AMD Ryzen 7:
  ./cli/calibrate_efficiency.py --hardware-id ryzen-7-7800x3d --device cpu

  # AMD Ryzen 9:
  ./cli/calibrate_efficiency.py --hardware-id ryzen-9-7950x --device cpu

  # Jetson Orin AGX (run each power mode):
  ./cli/calibrate_efficiency.py --hardware-id jetson-orin-agx-50w --device cuda
  ./cli/calibrate_efficiency.py --hardware-id jetson-orin-agx-30w --device cuda

  # Jetson Orin Nano:
  ./cli/calibrate_efficiency.py --hardware-id jetson-orin-nano-15w --device cuda

  Quick mode (6 models, ~5 min):
  ./cli/calibrate_efficiency.py --hardware-id <id> --device <cpu|cuda> --quick
```

## Models Calibrated (16 total)

  - ResNet: resnet18, resnet34, resnet50, resnet101
  - MobileNet: mobilenet_v2, mobilenet_v3_small, mobilenet_v3_large
  - EfficientNet: efficientnet_b0, efficientnet_b1
  - VGG: vgg11, vgg16
  - ViT: vit_b_16, vit_b_32, vit_l_16
  - Other: maxvit_t

Commit pushed: c670bd5

