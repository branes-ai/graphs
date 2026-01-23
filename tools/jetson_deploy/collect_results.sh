#!/bin/bash
#
# Collect and package calibration results
#
# This script packages calibration results for easy retrieval from the
# remote Jetson device.
#
# Usage:
#   ./collect_results.sh [output_name]
#
# Output:
#   jetson_results_<hostname>_<timestamp>.tar.gz
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results"
HOSTNAME=$(hostname)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Output name
OUTPUT_NAME="${1:-jetson_results_${HOSTNAME}_${TIMESTAMP}}"

echo "=========================================="
echo "Collecting Calibration Results"
echo "=========================================="
echo ""

# Check if results exist
if [ ! -d "$RESULTS_DIR" ]; then
    echo "ERROR: Results directory not found: $RESULTS_DIR"
    echo "Run ./calibrate.sh first."
    exit 1
fi

if [ ! -f "$RESULTS_DIR/calibration.json" ]; then
    echo "WARNING: No calibration.json found."
    echo "Calibration may not have completed successfully."
fi

# List results
echo "Results found:"
ls -la "$RESULTS_DIR/"
echo ""

# Add additional system diagnostics
echo "Collecting additional system diagnostics..."

# Power mode info
if command -v nvpmodel &> /dev/null; then
    nvpmodel -q 2>/dev/null > "$RESULTS_DIR/power_mode.txt" || \
    sudo nvpmodel -q 2>/dev/null > "$RESULTS_DIR/power_mode.txt" || \
    echo "Could not query power mode" > "$RESULTS_DIR/power_mode.txt"
fi

# Tegra release info
if [ -f /etc/nv_tegra_release ]; then
    cp /etc/nv_tegra_release "$RESULTS_DIR/tegra_release.txt"
fi

# JetPack version (if available)
if [ -f /etc/apt/sources.list.d/nvidia-l4t-apt-source.list ]; then
    cat /etc/apt/sources.list.d/nvidia-l4t-apt-source.list > "$RESULTS_DIR/jetpack_source.txt"
fi

# CPU info
cat /proc/cpuinfo > "$RESULTS_DIR/cpuinfo.txt" 2>/dev/null || true

# Memory info
cat /proc/meminfo > "$RESULTS_DIR/meminfo.txt" 2>/dev/null || true

# GPU info via Python/PyTorch
if command -v python3 &> /dev/null; then
    python3 -c "
import json
try:
    import torch
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info = {
            'name': props.name,
            'total_memory_gb': props.total_memory / (1024**3),
            'major': props.major,
            'minor': props.minor,
            'multi_processor_count': props.multi_processor_count,
            'cuda_version': torch.version.cuda,
            'pytorch_version': torch.__version__,
        }
        print(json.dumps(info, indent=2))
except Exception as e:
    print(json.dumps({'error': str(e)}))
" > "$RESULTS_DIR/gpu_properties.json" 2>/dev/null || true
fi

# Create manifest
cat > "$RESULTS_DIR/MANIFEST.txt" << EOF
Jetson Calibration Results
==========================
Hostname: $HOSTNAME
Collected: $(date -Iseconds)

Files:
$(ls -1 "$RESULTS_DIR/")

To import these results:
1. Copy to your local machine
2. Run: python cli/calibrate.py import --file calibration.json --device-id <device_id>
EOF

echo ""
echo "Creating results package..."

# Create tarball
cd "$SCRIPT_DIR"
tar czf "${OUTPUT_NAME}.tar.gz" results/

# Get size
SIZE=$(du -h "${OUTPUT_NAME}.tar.gz" | cut -f1)

echo ""
echo "=========================================="
echo "Results Packaged Successfully!"
echo "=========================================="
echo ""
echo "Output: $SCRIPT_DIR/${OUTPUT_NAME}.tar.gz"
echo "Size: $SIZE"
echo ""
echo "Contents:"
tar tzf "${OUTPUT_NAME}.tar.gz" | head -20
TOTAL=$(tar tzf "${OUTPUT_NAME}.tar.gz" | wc -l)
if [ "$TOTAL" -gt 20 ]; then
    echo "... and $((TOTAL - 20)) more files"
fi
echo ""
echo "To retrieve results:"
echo "  scp user@$HOSTNAME:$SCRIPT_DIR/${OUTPUT_NAME}.tar.gz ./"
echo ""
echo "To import into hardware registry:"
echo "  tar xzf ${OUTPUT_NAME}.tar.gz"
echo "  python cli/calibrate.py import --file results/calibration.json"
echo ""
