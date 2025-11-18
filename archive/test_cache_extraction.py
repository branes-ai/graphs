#!/usr/bin/env python3
"""
Direct test of _extract_cache_info to see if it's working
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

print("Testing _extract_cache_info directly...")
print()

try:
    import cpuinfo
    from graphs.hardware.database.detector import HardwareDetector
    
    # Get CPU info
    cpu_info = cpuinfo.get_cpu_info()
    
    print("Step 1: Check what cpuinfo returns")
    print("-" * 60)
    if 'l2_cache_size' in cpu_info:
        print(f"✓ l2_cache_size in cpu_info: {cpu_info['l2_cache_size']} bytes")
    else:
        print("✗ l2_cache_size NOT in cpu_info")
    
    if 'l3_cache_size' in cpu_info:
        print(f"✓ l3_cache_size in cpu_info: {cpu_info['l3_cache_size']} bytes")
    else:
        print("✗ l3_cache_size NOT in cpu_info")
    print()
    
    # Call _extract_cache_info directly
    print("Step 2: Call _extract_cache_info")
    print("-" * 60)
    detector = HardwareDetector()
    cache_info = detector._extract_cache_info(cpu_info)
    
    print(f"Returned cache_info dict:")
    for key, value in cache_info.items():
        if not key.startswith('cache_levels'):  # Don't print the whole array
            print(f"  {key}: {value}")
    
    if 'cache_levels' in cache_info:
        print(f"  cache_levels: {len(cache_info['cache_levels'])} levels")
        for level in cache_info['cache_levels']:
            print(f"    - {level}")
    print()
    
    # Check if sizes were extracted
    print("Step 3: Verify extraction")
    print("-" * 60)
    if cache_info.get('l2_cache_kb'):
        print(f"✓ L2 cache extracted: {cache_info['l2_cache_kb']} KB")
    else:
        print("✗ L2 cache NOT extracted")
    
    if cache_info.get('l3_cache_kb'):
        print(f"✓ L3 cache extracted: {cache_info['l3_cache_kb']} KB")
    else:
        print("✗ L3 cache NOT extracted")
    
    # Test has_cache_sizes logic
    print()
    print("Step 4: Test has_cache_sizes logic")
    print("-" * 60)
    has_cache_sizes = any(
        key in cache_info
        for key in ['l1_dcache_kb', 'l1_icache_kb', 'l2_cache_kb', 'l3_cache_kb']
    )
    print(f"has_cache_sizes: {has_cache_sizes}")
    print(f"  Checking: {['l1_dcache_kb', 'l1_icache_kb', 'l2_cache_kb', 'l3_cache_kb']}")
    print(f"  Found: {[k for k in ['l1_dcache_kb', 'l1_icache_kb', 'l2_cache_kb', 'l3_cache_kb'] if k in cache_info]}")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
