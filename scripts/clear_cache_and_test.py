#!/usr/bin/env python3
"""
Clear Python bytecode cache and test cache detection
"""
import sys
import shutil
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

print("=" * 80)
print("Clearing Python bytecode cache...")
print("=" * 80)
print()

# Find and remove all __pycache__ directories
repo_root = Path(__file__).parent
pycache_dirs = list(repo_root.rglob("__pycache__"))

if pycache_dirs:
    print(f"Found {len(pycache_dirs)} __pycache__ directories:")
    for pdir in pycache_dirs:
        print(f"  - {pdir.relative_to(repo_root)}")
        shutil.rmtree(pdir, ignore_errors=True)
    print()
    print("✓ All bytecode cache cleared")
else:
    print("No __pycache__ directories found")

print()
print("=" * 80)
print("Testing cache detection with fresh imports...")
print("=" * 80)
print()

try:
    import cpuinfo
    from graphs.hardware.database.detector import HardwareDetector

    # Get CPU info
    cpu_info = cpuinfo.get_cpu_info()

    print("Step 1: Test _extract_cache_info")
    print("-" * 60)
    detector = HardwareDetector()
    cache_info = detector._extract_cache_info(cpu_info)

    print(f"cache_info keys: {list(cache_info.keys())}")
    print()

    for key in ['l1_dcache_kb', 'l1_icache_kb', 'l2_cache_kb', 'l3_cache_kb']:
        value = cache_info.get(key)
        if value:
            print(f"  ✓ {key}: {value} KB")
        else:
            print(f"  ✗ {key}: NOT FOUND")

    if 'cache_levels' in cache_info:
        print(f"  ✓ cache_levels: {len(cache_info['cache_levels'])} levels")
    else:
        print(f"  ✗ cache_levels: NOT FOUND")
    print()

    print("Step 2: Test full detect_cpu()")
    print("-" * 60)
    cpu = detector.detect_cpu()

    print(f"DetectedCPU attributes:")
    print(f"  model_name: {cpu.model_name}")
    print(f"  cores: {cpu.cores}")
    print(f"  threads: {cpu.threads}")
    print()

    print(f"Cache fields from DetectedCPU:")
    for key in ['l1_dcache_kb', 'l1_icache_kb', 'l2_cache_kb', 'l3_cache_kb']:
        value = getattr(cpu, key)
        if value:
            print(f"  ✓ {key}: {value} KB")
        else:
            print(f"  ✗ {key}: None")

    if cpu.cache_levels:
        print(f"  ✓ cache_levels: {len(cpu.cache_levels)} levels")
        for level in cpu.cache_levels:
            size_kb = level.get('size_per_unit_kb') or level.get('total_size_kb')
            print(f"    - {level['name']}: {size_kb} KB")
    else:
        print(f"  ✗ cache_levels: None")
    print()

    print("=" * 80)
    if cpu.cache_levels and len(cpu.cache_levels) > 0:
        print("✓ SUCCESS: Cache detection is working!")
    else:
        print("✗ FAILED: Cache levels still empty")
        print()
        print("This means bytecode cache was not the issue.")
        print("We need to investigate the DetectedCPU construction.")
    print("=" * 80)

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
