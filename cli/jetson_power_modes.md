# NVIDIA Jetson Power Modes

The registry shows 15W/30W/50W/MAXN with GPU clocks up to 1300 MHz -- this matches the 64GB AGX Orin (different from the 32GB). The device detection
showed MODE_30W at 306/612 MHz, which is the 30W mode but the GPU wasn't boosted.

Here are the Jetson AGX Orin 64GB power modes:
```text
  ┌──────┬───────┬──────┬───────────────┬───────────────┬───────────┐
  │ Mode │ Power │ CPUs │ CPU Max (MHz) │ GPU Max (MHz) │ EMC (MHz) │
  ├──────┼───────┼──────┼───────────────┼───────────────┼───────────┤
  │ 0    │ MAXN  │ 12   │ 2201          │ 1300          │ 3200      │
  ├──────┼───────┼──────┼───────────────┼───────────────┼───────────┤
  │ 1    │ 15W   │ 4    │ 1114          │ 510           │ 2133      │
  ├──────┼───────┼──────┼───────────────┼───────────────┼───────────┤
  │ 2    │ 30W   │ 8    │ 1728          │ 612           │ 3200      │
  ├──────┼───────┼──────┼───────────────┼───────────────┼───────────┤
  │ 3    │ 50W   │ 12   │ 2201          │ 1020          │ 3200      │
  └──────┴───────┴──────┴───────────────┴───────────────┴───────────┘
```

AGX Orin 64GB power modes:

```text
  ┌─────────────────────────────────┬─────────┬───────────────┬──────┬──────────┬──────────┬──────────┐
  │             Command             │ Mode ID │     Power     │ CPUs │ CPU Max  │ GPU Max  │   EMC    │
  ├─────────────────────────────────┼─────────┼───────────────┼──────┼──────────┼──────────┼──────────┤
  │ ./cli/jetson_power_mode.sh 15w  │ 1       │ 15W           │ 4    │ 1114 MHz │ 510 MHz  │ 2133 MHz │
  ├─────────────────────────────────┼─────────┼───────────────┼──────┼──────────┼──────────┼──────────┤
  │ ./cli/jetson_power_mode.sh 30w  │ 2       │ 30W           │ 8    │ 1728 MHz │ 612 MHz  │ 3200 MHz │
  ├─────────────────────────────────┼─────────┼───────────────┼──────┼──────────┼──────────┼──────────┤
  │ ./cli/jetson_power_mode.sh 50w  │ 3       │ 50W           │ 12   │ 2201 MHz │ 1020 MHz │ 3200 MHz │
  ├─────────────────────────────────┼─────────┼───────────────┼──────┼──────────┼──────────┼──────────┤
  │ ./cli/jetson_power_mode.sh maxn │ 0       │ Unconstrained │ 12   │ 2201 MHz │ 1300 MHz │ 3200 MHz │
  └─────────────────────────────────┴─────────┴───────────────┴──────┴──────────┴──────────┴──────────┘
```

Usage:

```bash
  ./cli/jetson_power_mode.sh              # Show current status + available modes
  ./cli/jetson_power_mode.sh status       # Detailed status with clocks/temps
  ./cli/jetson_power_mode.sh maxn         # Set MAXN mode
  ./cli/jetson_power_mode.sh maxn --lock  # Set MAXN and lock clocks to max (needs reboot to change after)
```

The --lock flag runs jetson_clocks which forces all clocks to their maximum for the mode, 
preventing the DVFS governor from idling the GPU at 306 MHz during benchmarks.

