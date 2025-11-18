#!/usr/bin/env python3
"""
Debug script to test Windows cache detection
Run this on Windows to see what's happening
"""
import sys
import platform
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

print("=" * 80)
print("Windows Cache Detection Debug")
print("=" * 80)
print()

# Check platform
print(f"Platform: {platform.system()}")
print(f"Platform version: {platform.version()}")
print()

# Test cpuinfo
print("Testing py-cpuinfo...")
try:
    import cpuinfo
    cpu_info = cpuinfo.get_cpu_info()
    
    print(f"✓ py-cpuinfo version: {cpu_info.get('cpuinfo_version_string', 'unknown')}")
    print(f"  CPU: {cpu_info.get('brand_raw', 'unknown')}")
    print()
    
    # Check for cache fields
    cache_fields = [
        'l1_data_cache_size',
        'l1_instruction_cache_size', 
        'l2_cache_size',
        'l3_cache_size',
        'l2_cache_associativity',
        'l2_cache_line_size'
    ]
    
    print("Cache fields in cpuinfo:")
    found_any = False
    for field in cache_fields:
        if field in cpu_info:
            print(f"  ✓ {field}: {cpu_info[field]}")
            found_any = True
        else:
            print(f"  ✗ {field}: NOT FOUND")
    
    if not found_any:
        print()
        print("⚠ No cache fields found in cpuinfo (expected on Windows)")
    print()
    
except ImportError as e:
    print(f"✗ py-cpuinfo not available: {e}")
    sys.exit(1)

# Test wmic
if platform.system() == 'Windows':
    print("Testing wmic cache detection...")
    try:
        import subprocess
        
        result = subprocess.run(
            ['wmic', 'cpu', 'get', 'L2CacheSize,L3CacheSize', '/format:list'],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        print(f"  wmic return code: {result.returncode}")
        
        if result.returncode == 0:
            print(f"  wmic stdout:")
            print("  " + "-" * 60)
            for line in result.stdout.strip().split('\n'):
                print(f"  {repr(line)}")
            print("  " + "-" * 60)
            print()
            
            # Parse output
            cache_info = {}
            for line in result.stdout.strip().split('\n'):
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    print(f"  Parsed: key='{key}', value='{value}'")
                    
                    if key == 'L2CacheSize' and value:
                        cache_info['l2_cache_kb'] = int(value)
                        print(f"    → L2 cache: {value} KB")
                    elif key == 'L3CacheSize' and value:
                        cache_info['l3_cache_kb'] = int(value)
                        print(f"    → L3 cache: {value} KB")
            
            print()
            print(f"✓ Detected cache: {cache_info}")
        else:
            print(f"  ✗ wmic failed")
            print(f"  stderr: {result.stderr}")
    
    except FileNotFoundError:
        print("  ✗ wmic command not found")
    except Exception as e:
        print(f"  ✗ wmic failed with exception: {e}")
        import traceback
        traceback.print_exc()
else:
    print("⚠ Not running on Windows, skipping wmic test")

print()
print("=" * 80)
print("Now testing actual detector...")
print("=" * 80)
print()

try:
    from graphs.hardware.database.detector import HardwareDetector
    
    detector = HardwareDetector()
    cpu = detector.detect_cpu()
    
    print(f"Detected CPU: {cpu.model_name}")
    print(f"  Vendor: {cpu.vendor}")
    print(f"  Cores: {cpu.cores}")
    print(f"  Threads: {cpu.threads}")
    print()
    
    print("Cache information:")
    print(f"  l1_dcache_kb: {cpu.l1_dcache_kb}")
    print(f"  l1_icache_kb: {cpu.l1_icache_kb}")
    print(f"  l2_cache_kb: {cpu.l2_cache_kb}")
    print(f"  l3_cache_kb: {cpu.l3_cache_kb}")
    print()
    
    if cpu.cache_levels:
        print(f"✓ cache_levels detected ({len(cpu.cache_levels)} levels):")
        for level in cpu.cache_levels:
            print(f"  - {level['name']}: {level.get('size_per_unit_kb') or level.get('total_size_kb')} KB")
    else:
        print("✗ No cache_levels detected")
        print()
        print("This means the Windows fallback didn't work.")
        print("Make sure you have the fix - run: python verify_fix.py")
    
except Exception as e:
    print(f"✗ Detector failed: {e}")
    import traceback
    traceback.print_exc()
