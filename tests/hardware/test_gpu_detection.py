#!/usr/bin/env python3
"""
Quick test of GPU detection
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

print("=" * 80)
print("Testing GPU Detection")
print("=" * 80)
print()

try:
    from graphs.hardware.database.detector import HardwareDetector

    detector = HardwareDetector()

    print("Detecting GPUs...")
    gpus = detector.detect_gpu()

    if not gpus:
        print("✗ No GPUs detected")
        print()
        print("This could mean:")
        print("  1. No NVIDIA GPU is installed")
        print("  2. nvidia-smi is not in PATH")
        print("  3. NVIDIA drivers are not installed")
        print()
        print("To test nvidia-smi directly:")
        print("  nvidia-smi --query-gpu=name,memory.total,compute_cap,driver_version --format=csv,noheader")
    else:
        print(f"✓ Detected {len(gpus)} GPU(s):")
        print()

        for i, gpu in enumerate(gpus):
            print(f"GPU {i}:")
            print(f"  Model:           {gpu.model_name}")
            print(f"  Vendor:          {gpu.vendor}")
            print(f"  Memory:          {gpu.memory_gb} GB")
            print(f"  CUDA Capability: {gpu.cuda_capability}")
            print(f"  Driver Version:  {gpu.driver_version}")
            print()

        print("To create hardware specs for these GPUs:")
        print("  python scripts/hardware_db/auto_detect_and_add.py --gpus-only -o .")

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
