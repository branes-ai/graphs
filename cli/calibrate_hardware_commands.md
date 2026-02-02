# Calibrate Hardware commands

## Usage:

```bash
  # Multi-core CPU STREAM only
  ./cli/calibrate_hardware.py --operations multicore_stream

  # Concurrent all-engines (Jetson)
  ./cli/calibrate_hardware.py --operations concurrent_stream

  # Combined with regular benchmarks
  ./cli/calibrate_hardware.py --operations blas,stream,multicore_stream,concurrent_stream
```
