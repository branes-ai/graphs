#!/bin/bash
#
# Install dependencies for Jetson calibration
#
# This script sets up the environment on a Jetson device for running
# hardware calibration benchmarks.
#
# Usage:
#   ./install.sh
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "Jetson Calibration - Installation"
echo "=========================================="
echo ""

# Check if we're on a Jetson
if [ -f /etc/nv_tegra_release ]; then
    echo "Detected Jetson platform:"
    cat /etc/nv_tegra_release
    echo ""
else
    echo "WARNING: This doesn't appear to be a Jetson device."
    echo "Continuing anyway..."
    echo ""
fi

# Check Python
echo "Checking Python..."
if command -v python3 &> /dev/null; then
    PYTHON=python3
    echo "Found: $($PYTHON --version)"
else
    echo "ERROR: Python 3 not found"
    exit 1
fi

# Check PyTorch
echo ""
echo "Checking PyTorch..."
if $PYTHON -c "import torch; print(f'PyTorch {torch.__version__}')" 2>/dev/null; then
    # Check CUDA
    if $PYTHON -c "import torch; assert torch.cuda.is_available(), 'No CUDA'; print(f'CUDA available: {torch.cuda.get_device_name(0)}')" 2>/dev/null; then
        echo "CUDA is available"
    else
        echo "WARNING: CUDA not available. GPU calibration will fail."
    fi
else
    echo "ERROR: PyTorch not found"
    echo "JetPack should include PyTorch. Please check your JetPack installation."
    exit 1
fi

# Check/install psutil
echo ""
echo "Checking psutil..."
if ! $PYTHON -c "import psutil" 2>/dev/null; then
    echo "Installing psutil..."
    pip3 install --user psutil
else
    echo "psutil already installed"
fi

# Check NumPy
echo ""
echo "Checking NumPy..."
if $PYTHON -c "import numpy; print(f'NumPy {numpy.__version__}')" 2>/dev/null; then
    echo "NumPy is available"
else
    echo "Installing NumPy..."
    pip3 install --user numpy
fi

# Set up PYTHONPATH
echo ""
echo "Setting up environment..."
export PYTHONPATH="$SCRIPT_DIR/src:$PYTHONPATH"

# Verify graphs package can be imported
echo ""
echo "Verifying graphs package..."
if $PYTHON -c "import sys; sys.path.insert(0, '$SCRIPT_DIR/src'); from graphs.hardware.calibration import calibrator; print('graphs.hardware.calibration OK')" 2>/dev/null; then
    echo "Package verification successful"
else
    echo "WARNING: Could not import graphs package. Calibration may fail."
fi

# Show system info
echo ""
echo "=========================================="
echo "System Information"
echo "=========================================="
echo "Hostname: $(hostname)"
echo "Kernel: $(uname -r)"
echo "Architecture: $(uname -m)"

if command -v nvpmodel &> /dev/null; then
    echo ""
    echo "Current power mode:"
    sudo nvpmodel -q 2>/dev/null || nvpmodel -q 2>/dev/null || echo "Could not query power mode"
fi

if command -v tegrastats &> /dev/null; then
    echo ""
    echo "Tegrastats available for monitoring"
fi

echo ""
echo "=========================================="
echo "Installation Complete"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. (Optional) Set power mode: sudo nvpmodel -m 0  # MAXN"
echo "  2. Run calibration: ./calibrate.sh"
echo "  3. Collect results: ./collect_results.sh"
echo ""
