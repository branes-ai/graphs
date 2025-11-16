#!/usr/bin/env python3
"""
Test script to demonstrate device detection functionality.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "cli"))

from calibrate_hardware import detect_actual_device, detect_platform

print("=" * 80)
print("DEVICE DETECTION TEST")
print("=" * 80)
print()

# Test platform detection
print("1. Platform Detection:")
print("-" * 80)
platform_info = detect_platform()
for key, value in platform_info.items():
    print(f"  {key}: {value}")
print()

# Test device detection for CPU
print("2. Device Detection (CPU requested):")
print("-" * 80)
cpu_info = detect_actual_device('cpu')
for key, value in cpu_info.items():
    print(f"  {key}: {value}")
print()

# Test device detection for CUDA
print("3. Device Detection (CUDA requested):")
print("-" * 80)
cuda_info = detect_actual_device('cuda')
for key, value in cuda_info.items():
    print(f"  {key}: {value}")
print()

# Summary
print("=" * 80)
print("SUMMARY")
print("=" * 80)
if cuda_info['fallback_occurred']:
    print("⚠ CUDA requested but will fall back to CPU")
    print(f"  Reason: {cuda_info['fallback_reason']}")
else:
    print("✓ CUDA is available and will be used")
print()
