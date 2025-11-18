#!/usr/bin/env python3
"""
Quick check to verify you have the Windows cache detection fix
"""
import sys
from pathlib import Path

# Add src to path  
sys.path.insert(0, str(Path(__file__).parent / "src"))

print("Checking for Windows cache detection fix...")
print()

try:
    detector_path = Path(__file__).parent / "src" / "graphs" / "hardware" / "database" / "detector.py"
    with open(detector_path, 'r') as f:
        content = f.read()
        
    if 'has_cache_sizes = any(' in content:
        print("✓ FIX IS PRESENT")
        print()
        print("  The code has the corrected logic:")
        print("  - Checks for cache SIZE fields specifically")
        print("  - Not just whether cache_info dict is empty")
        print()
        print("  You should now see cache_levels populated!")
    else:
        print("✗ FIX NOT FOUND")
        print()
        print("  Your code still has the old buggy condition:")
        print("    if not cache_info and platform.system() == 'Windows':")
        print()
        print("  You need to PULL the latest changes!")
        print()
        print("  Expected in detector.py around line 656:")
        print("    has_cache_sizes = any(")
        print("        key in cache_info")
        print("        for key in ['l1_dcache_kb', 'l1_icache_kb', 'l2_cache_kb', 'l3_cache_kb']")
        print("    )")
except FileNotFoundError:
    print("✗ Could not find detector.py")
    print(f"  Looking for: {detector_path}")
    print("  Make sure you're running from the repo root")
