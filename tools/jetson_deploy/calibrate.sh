#!/bin/bash
#
# Run calibration on Jetson device
#
# This script runs hardware calibration benchmarks and saves results.
#
# Usage:
#   ./calibrate.sh [options]
#
# Options:
#   --quick     Run quick calibration (fewer trials, ~5 minutes)
#   --full      Run full calibration (all trials, ~30-60 minutes)
#   --help      Show this help message
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results"

# Default to full calibration
MODE="full"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            MODE="quick"
            shift
            ;;
        --full)
            MODE="full"
            shift
            ;;
        --help)
            head -20 "$0" | tail -15
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "Jetson Calibration - $MODE mode"
echo "=========================================="
echo ""

# Set up environment
export PYTHONPATH="$SCRIPT_DIR/src:$PYTHONPATH"

# Check Python
if command -v python3 &> /dev/null; then
    PYTHON=python3
else
    echo "ERROR: Python 3 not found"
    exit 1
fi

# Create results directory
mkdir -p "$RESULTS_DIR"

# Get system info
echo "Collecting system information..."
HOSTNAME=$(hostname)
TIMESTAMP=$(date -Iseconds)

# Detect device
DEVICE_ID="unknown"
if [ -f /etc/nv_tegra_release ]; then
    TEGRA_RELEASE=$(cat /etc/nv_tegra_release | head -1)
    # Try to detect specific Orin variant
    if $PYTHON -c "import torch; name=torch.cuda.get_device_name(0); print(name)" 2>/dev/null | grep -qi "orin"; then
        # Check memory to distinguish variants
        MEM_GB=$($PYTHON -c "import torch; print(torch.cuda.get_device_properties(0).total_memory // (1024**3))" 2>/dev/null || echo "0")
        if [ "$MEM_GB" -ge 14 ]; then
            DEVICE_ID="jetson_orin_nx_16gb_gpu"
        elif [ "$MEM_GB" -ge 6 ]; then
            DEVICE_ID="jetson_orin_nx_8gb_gpu"
        elif [ "$MEM_GB" -ge 4 ]; then
            DEVICE_ID="jetson_orin_nano_8gb_gpu"
        fi
    fi
fi

echo "Detected device: $DEVICE_ID"
echo ""

# Save system info
cat > "$RESULTS_DIR/system_info.json" << EOF
{
    "hostname": "$HOSTNAME",
    "timestamp": "$TIMESTAMP",
    "device_id": "$DEVICE_ID",
    "tegra_release": "$(cat /etc/nv_tegra_release 2>/dev/null | head -1 || echo 'N/A')",
    "kernel": "$(uname -r)",
    "architecture": "$(uname -m)",
    "python_version": "$($PYTHON --version 2>&1)",
    "pytorch_version": "$($PYTHON -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'N/A')",
    "cuda_version": "$($PYTHON -c 'import torch; print(torch.version.cuda)' 2>/dev/null || echo 'N/A')",
    "gpu_name": "$($PYTHON -c 'import torch; print(torch.cuda.get_device_name(0))' 2>/dev/null || echo 'N/A')",
    "gpu_memory_gb": "$($PYTHON -c 'import torch; print(torch.cuda.get_device_properties(0).total_memory / (1024**3))' 2>/dev/null || echo 'N/A')"
}
EOF

echo "System info saved to $RESULTS_DIR/system_info.json"
echo ""

# Set calibration parameters based on mode
if [ "$MODE" = "quick" ]; then
    QUICK_FLAG="--quick"
    echo "Running quick calibration..."
    echo "  Matrix sizes: 256, 1024, 2048"
    echo "  Precisions: fp32, fp16"
else
    QUICK_FLAG=""
    echo "Running full calibration..."
    echo "  Matrix sizes: 128, 256, 512, 1024, 2048, 4096, 8192"
    echo "  Precisions: fp32, fp16, bf16, int8"
fi
echo ""

# Check for power mode
if command -v nvpmodel &> /dev/null; then
    echo "Current power mode:"
    nvpmodel -q 2>/dev/null || sudo nvpmodel -q 2>/dev/null || echo "  Could not query"
    echo ""
fi

# Run the portable calibration script
echo "Starting calibration benchmarks..."
echo "=========================================="

# Run GPU calibration (the Jetson GPU)
$PYTHON "$SCRIPT_DIR/portable_calibration.py" \
    --device cuda:0 \
    --output "$RESULTS_DIR/calibration.json" \
    $QUICK_FLAG

CALIBRATION_EXIT=$?

if [ $CALIBRATION_EXIT -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Calibration Complete!"
    echo "=========================================="
    echo ""
    echo "Results saved to:"
    echo "  $RESULTS_DIR/calibration.json"
    echo "  $RESULTS_DIR/system_info.json"
    echo ""

    # Generate summary
    if [ -f "$RESULTS_DIR/calibration.json" ]; then
        echo "Generating summary..."
        $PYTHON -c "
import json
with open('$RESULTS_DIR/calibration.json') as f:
    data = json.load(f)

print('Calibration Summary')
print('=' * 40)

meta = data.get('metadata', {})
print(f'Hardware ID: {meta.get(\"hardware_id\", \"unknown\")}')
print(f'Hardware: {meta.get(\"hardware_name\", \"unknown\")}')
print(f'Date: {meta.get(\"calibration_date\", \"unknown\")}')
print(f'Framework: {meta.get(\"framework\", \"unknown\")}')

print('')
print(f'Best Measured: {data.get(\"best_measured_gflops\", 0):.1f} GFLOPS')
print(f'Avg Measured: {data.get(\"avg_measured_gflops\", 0):.1f} GFLOPS')
print(f'Bandwidth: {data.get(\"measured_bandwidth_gbps\", 0):.1f} GB/s')

peaks = data.get('per_precision_peaks', {})
if peaks:
    print('')
    print('Peak GFLOPS by precision:')
    for prec in ['fp32', 'fp16', 'bf16', 'int8']:
        gflops = peaks.get(prec, 0)
        if gflops > 0:
            print(f'  {prec}: {gflops:.1f} GFLOPS')

print('')
" > "$RESULTS_DIR/calibration_summary.txt" 2>/dev/null || true

        if [ -f "$RESULTS_DIR/calibration_summary.txt" ]; then
            cat "$RESULTS_DIR/calibration_summary.txt"
        fi
    fi

    echo ""
    echo "Next steps:"
    echo "  1. Review results: cat $RESULTS_DIR/calibration.json"
    echo "  2. Package results: ./collect_results.sh"
    echo "  3. Copy to local machine for analysis"
else
    echo ""
    echo "=========================================="
    echo "Calibration Failed (exit code: $CALIBRATION_EXIT)"
    echo "=========================================="
    echo ""
    echo "Check the output above for errors."
    echo "Common issues:"
    echo "  - Out of memory: try --quick mode"
    echo "  - CUDA not available: check JetPack installation"
    echo "  - Permission denied: some operations may need sudo"
    exit $CALIBRATION_EXIT
fi
