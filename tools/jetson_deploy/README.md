# Jetson Calibration Deployment Package

Deploy and run hardware calibration on remote Jetson devices.

## Quick Start

On your local machine:
```bash
# Create deployment package
./tools/jetson_deploy/create_package.sh

# Copy to Jetson
scp jetson_calibration.tar.gz user@jetson:/tmp/

# SSH to Jetson and run
ssh user@jetson
cd /tmp && tar xzf jetson_calibration.tar.gz
cd jetson_calibration
./install.sh
./calibrate.sh
```

Retrieve results:
```bash
# On local machine
scp user@jetson:/tmp/jetson_calibration/results/*.json ./
```

## Package Contents

- `install.sh` - Install dependencies and setup environment
- `calibrate.sh` - Run calibration suite (supports `--quick` and `--full` modes)
- `collect_results.sh` - Package results for retrieval
- `portable_calibration.py` - Standalone calibration script
- `src/` - Minimal graphs package for calibration
- `hardware_registry/` - Hardware specs (including target device)

## Requirements

The Jetson should have:
- JetPack 5.x or 6.x (provides PyTorch, CUDA)
- Python 3.10+
- Network access (for pip install psutil)

## Calibration Modes

### Quick Mode (~5 minutes)
```bash
./calibrate.sh --quick
```
- Fewer matrix sizes (256, 1024, 2048)
- Fewer precisions (FP32, FP16)
- Good for initial testing

### Full Mode (~30-60 minutes)
```bash
./calibrate.sh
```
- All matrix sizes (128 to 8192)
- All precisions (FP32, FP16, BF16, INT8)
- Production quality results

### Specific Power Mode
```bash
# Set power mode before calibration
sudo nvpmodel -m 0  # MAXN mode
./calibrate.sh
```

## Results

Results are saved to `results/` directory:
- `calibration.json` - Full calibration data
- `calibration_summary.txt` - Human-readable summary
- `system_info.json` - System configuration snapshot

## Troubleshooting

### Permission denied for nvpmodel
```bash
sudo nvpmodel -m 0
```

### PyTorch not found
JetPack should include PyTorch. If not:
```bash
# Check JetPack version
cat /etc/nv_tegra_release
# Reinstall JetPack PyTorch wheel
```

### Out of memory
Use quick mode or reduce matrix sizes:
```bash
./calibrate.sh --quick
```
