#!/bin/bash
#
# Create Jetson Calibration Deployment Package
#
# Creates a self-contained tarball that can be deployed to a Jetson device
# for running hardware calibration benchmarks.
#
# Usage:
#   ./tools/jetson_deploy/create_package.sh [output_dir]
#
# Output:
#   jetson_calibration.tar.gz - Deployment package
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUTPUT_DIR="${1:-$REPO_ROOT}"
PACKAGE_NAME="jetson_calibration"
PACKAGE_DIR="/tmp/$PACKAGE_NAME"

echo "=========================================="
echo "Creating Jetson Calibration Package"
echo "=========================================="
echo "Repo root: $REPO_ROOT"
echo "Output dir: $OUTPUT_DIR"
echo ""

# Clean up previous build
rm -rf "$PACKAGE_DIR"
mkdir -p "$PACKAGE_DIR"

# Create directory structure
mkdir -p "$PACKAGE_DIR/src/graphs/hardware/calibration"
mkdir -p "$PACKAGE_DIR/hardware_registry"
mkdir -p "$PACKAGE_DIR/results"

echo "Copying source files..."

# Create minimal __init__.py files for package structure
# (graphs uses namespace packages but we need explicit inits for deployment)
cat > "$PACKAGE_DIR/src/graphs/__init__.py" << 'EOF'
"""Graphs package - minimal deployment for calibration."""
__version__ = "0.1.0"
EOF

cat > "$PACKAGE_DIR/src/graphs/hardware/__init__.py" << 'EOF'
"""Hardware module."""
EOF

# Copy hardware/calibration module (required for portable_calibration.py)
cp -r "$REPO_ROOT/src/graphs/hardware/calibration/"* "$PACKAGE_DIR/src/graphs/hardware/calibration/"

# Replace the __init__.py with a minimal version that only provides registry_sync
# (the original has deprecation code that tries to import from graphs.calibration)
cat > "$PACKAGE_DIR/src/graphs/hardware/calibration/__init__.py" << 'EOF'
"""Hardware Calibration Framework - Minimal Deployment Package.

This package provides registry_sync for hardware matching in portable_calibration.py.
"""
# Only import what's needed for registry_sync (avoids complex dependency chains)
# Full calibration imports are available via direct module import
__all__ = ['registry_sync', 'calibration_db', 'auto_detect']
EOF

# Also fix benchmarks __init__.py to avoid the deprecation import issue
cat > "$PACKAGE_DIR/src/graphs/hardware/calibration/benchmarks/__init__.py" << 'EOF'
"""Calibration Benchmarks - Minimal Deployment Package."""
# Benchmarks available via direct module import
__all__ = []
EOF

# Remove unnecessary benchmark subdirectories to reduce size
rm -rf "$PACKAGE_DIR/src/graphs/hardware/calibration/benchmarks/numpy" 2>/dev/null || true
rm -rf "$PACKAGE_DIR/src/graphs/hardware/calibration/benchmarks/pytorch" 2>/dev/null || true
rm -rf "$PACKAGE_DIR/src/graphs/hardware/calibration/profiles" 2>/dev/null || true
rm -rf "$PACKAGE_DIR/src/graphs/hardware/calibration/__pycache__" 2>/dev/null || true
rm -rf "$PACKAGE_DIR/src/graphs/hardware/calibration/benchmarks/__pycache__" 2>/dev/null || true

# Copy hardware registry (only GPU entries for Jetson)
cp -r "$REPO_ROOT/hardware_registry/gpu" "$PACKAGE_DIR/hardware_registry/"

echo "Copying deployment scripts..."

# Copy deployment scripts
cp "$SCRIPT_DIR/install.sh" "$PACKAGE_DIR/"
cp "$SCRIPT_DIR/calibrate.sh" "$PACKAGE_DIR/"
cp "$SCRIPT_DIR/collect_results.sh" "$PACKAGE_DIR/"
cp "$SCRIPT_DIR/README.md" "$PACKAGE_DIR/"

# Copy portable calibration tool
cp "$REPO_ROOT/tools/portable_calibration.py" "$PACKAGE_DIR/"

# Make scripts executable
chmod +x "$PACKAGE_DIR"/*.sh

echo "Creating package manifest..."

# Create manifest
cat > "$PACKAGE_DIR/MANIFEST.txt" << EOF
Jetson Calibration Package
==========================
Created: $(date -Iseconds)
Source: $REPO_ROOT
Git commit: $(cd "$REPO_ROOT" && git rev-parse --short HEAD 2>/dev/null || echo "unknown")

Contents:
- src/graphs/hardware/calibration/ - Hardware calibration framework
- hardware_registry/gpu/ - GPU hardware specifications
- portable_calibration.py - Standalone calibration script
- install.sh - Installation script
- calibrate.sh - Calibration runner
- collect_results.sh - Results packaging

Usage:
1. Transfer to Jetson: scp jetson_calibration.tar.gz user@jetson:/tmp/
2. Extract: tar xzf jetson_calibration.tar.gz
3. Install deps: cd jetson_calibration && ./install.sh
4. Run calibration: ./calibrate.sh
5. Collect results: ./collect_results.sh
EOF

echo "Creating tarball..."

# Create tarball
cd /tmp
tar czf "$OUTPUT_DIR/$PACKAGE_NAME.tar.gz" "$PACKAGE_NAME"

# Calculate size
SIZE=$(du -h "$OUTPUT_DIR/$PACKAGE_NAME.tar.gz" | cut -f1)

echo ""
echo "=========================================="
echo "Package created successfully!"
echo "=========================================="
echo "Output: $OUTPUT_DIR/$PACKAGE_NAME.tar.gz"
echo "Size: $SIZE"
echo ""
echo "Deploy to Jetson:"
echo "  scp $OUTPUT_DIR/$PACKAGE_NAME.tar.gz user@jetson:/tmp/"
echo "  ssh user@jetson 'cd /tmp && tar xzf $PACKAGE_NAME.tar.gz && cd $PACKAGE_NAME && ./install.sh && ./calibrate.sh'"
echo ""

# Cleanup
rm -rf "$PACKAGE_DIR"
