#!/usr/bin/env python3
"""
Add Hardware to Database

Interactive wizard to add new hardware specifications to the database.

Usage:
    python scripts/hardware_db/add_hardware.py
    python scripts/hardware_db/add_hardware.py --from-file hardware.json
    python scripts/hardware_db/add_hardware.py --from-detection detection.json
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from graphs.hardware.database import HardwareDatabase, HardwareSpec


def prompt_required(prompt_text: str, default=None) -> str:
    """Prompt for required input with optional default"""
    if default:
        prompt_text = f"{prompt_text} [{default}]"

    while True:
        value = input(f"{prompt_text}: ").strip()
        if value:
            return value
        elif default:
            return default
        else:
            print("  This field is required. Please enter a value.")


def prompt_optional(prompt_text: str, default=None) -> str:
    """Prompt for optional input"""
    if default:
        prompt_text = f"{prompt_text} [{default}]"

    value = input(f"{prompt_text}: ").strip()
    return value if value else default


def prompt_int(prompt_text: str, default=None) -> int:
    """Prompt for integer input"""
    while True:
        value = prompt_optional(prompt_text, str(default) if default else None)
        if not value:
            return default
        try:
            return int(value)
        except ValueError:
            print("  Please enter a valid integer.")


def prompt_float(prompt_text: str, default=None) -> float:
    """Prompt for float input"""
    while True:
        value = prompt_optional(prompt_text, str(default) if default else None)
        if not value:
            return default
        try:
            return float(value)
        except ValueError:
            print("  Please enter a valid number.")


def prompt_list(prompt_text: str, delimiter=',') -> list:
    """Prompt for comma-separated list input"""
    value = input(f"{prompt_text} (comma-separated): ").strip()
    if not value:
        return []
    return [item.strip() for item in value.split(delimiter) if item.strip()]


def prompt_choice(prompt_text: str, choices: list, default=None) -> str:
    """Prompt for choice from list"""
    print(f"\n{prompt_text}")
    for i, choice in enumerate(choices, 1):
        marker = " (default)" if choice == default else ""
        print(f"  {i}. {choice}{marker}")

    while True:
        value = input("Select (number or text): ").strip()

        # Try as number
        try:
            idx = int(value)
            if 1 <= idx <= len(choices):
                return choices[idx - 1]
        except ValueError:
            pass

        # Try as text match
        if value in choices:
            return value

        # Use default if empty
        if not value and default:
            return default

        print(f"  Please enter a number 1-{len(choices)} or a valid choice.")


def interactive_add_hardware(db: HardwareDatabase) -> HardwareSpec:
    """Interactive wizard to create hardware spec"""
    print()
    print("=" * 80)
    print("Add New Hardware - Interactive Wizard")
    print("=" * 80)
    print()
    print("Press Ctrl+C at any time to cancel")
    print()

    # Basic identification
    print("--- Basic Identification ---")
    vendor = prompt_required("Vendor (e.g., Intel, NVIDIA, AMD)")
    model = prompt_required("Model (e.g., Core i7-12700K, H100 SXM5 80GB)")

    # Generate default ID from vendor and model
    default_id = f"{vendor}_{model}".lower()
    default_id = default_id.replace(' ', '_').replace('-', '_')
    default_id = ''.join(c for c in default_id if c.isalnum() or c == '_')

    hw_id = prompt_required("Hardware ID", default=default_id)

    architecture = prompt_optional("Architecture (e.g., Alder Lake, Hopper, Zen 4)")

    device_type = prompt_choice(
        "Device Type",
        ['cpu', 'gpu', 'tpu', 'kpu', 'dpu', 'cgra'],
        default='cpu'
    )

    platform = prompt_choice(
        "Platform",
        ['x86_64', 'aarch64', 'arm64'],
        default='x86_64'
    )
    print()

    # Detection patterns
    print("--- Detection Patterns ---")
    print("Enter regex patterns to match this hardware during auto-detection")
    detection_patterns = prompt_list("Detection patterns")
    if not detection_patterns:
        # Suggest default pattern
        detection_patterns = [model]
        print(f"  Using default: [{model}]")
    print()

    # Core specifications
    print("--- Core Specifications ---")
    cores = None
    threads = None
    e_cores = None
    cuda_cores = None
    sms = None

    if device_type == 'cpu':
        cores = prompt_int("Physical cores")
        threads = prompt_int("Hardware threads", default=cores)

        # Check for hybrid CPU
        if cores and threads and threads > cores:
            e_cores = prompt_int("E-cores (efficiency cores, leave empty if not hybrid)")

    elif device_type == 'gpu':
        cuda_cores = prompt_int("CUDA cores (or compute units)")
        sms = prompt_int("SMs (streaming multiprocessors)")
        tensor_cores = prompt_int("Tensor cores")

    base_frequency_ghz = prompt_float("Base frequency (GHz)")
    boost_frequency_ghz = prompt_float("Boost frequency (GHz)")
    print()

    # Memory
    print("--- Memory Specifications ---")
    memory_type = prompt_optional("Memory type (e.g., DDR5, HBM3, GDDR6X)")
    peak_bandwidth_gbps = prompt_float("Peak memory bandwidth (GB/s)")
    print()

    # ISA extensions
    print("--- ISA Extensions ---")
    isa_extensions = prompt_list("ISA extensions (e.g., AVX2, AVX512, NEON)")
    print()

    # Theoretical peaks
    print("--- Theoretical Performance Peaks ---")
    print("Enter GFLOPS for float precisions, GIOPS for integer precisions")
    theoretical_peaks = {}

    # Always ask for fp32
    fp32 = prompt_float("fp32 (required)")
    if fp32:
        theoretical_peaks['fp32'] = fp32

    # Optional precisions
    for prec in ['fp64', 'fp16', 'bf16', 'fp8', 'int64', 'int32', 'int16', 'int8']:
        value = prompt_float(f"{prec} (optional)")
        if value:
            theoretical_peaks[prec] = value
    print()

    # Cache (optional)
    print("--- Cache (Optional) ---")
    l1_cache_kb = prompt_int("L1 cache (KB)")
    l2_cache_kb = prompt_int("L2 cache (KB)")
    l3_cache_kb = prompt_int("L3 cache (KB)")
    print()

    # Power (optional)
    print("--- Power (Optional) ---")
    tdp_watts = prompt_float("TDP (watts)")
    max_power_watts = prompt_float("Max power (watts)")
    print()

    # Mapper
    print("--- Mapper Configuration ---")
    mapper_class = prompt_choice(
        "Mapper class",
        ['CPUMapper', 'GPUMapper', 'TPUMapper', 'KPUMapper', 'DPUMapper', 'CGRAMapper'],
        default='CPUMapper' if device_type == 'cpu' else 'GPUMapper'
    )
    print()

    # Metadata
    print("--- Metadata (Optional) ---")
    release_date = prompt_optional("Release date (e.g., Q4 2021, 2022-11)")
    manufacturer_url = prompt_optional("Manufacturer URL")
    notes = prompt_optional("Notes")
    print()

    # Create HardwareSpec
    spec = HardwareSpec(
        id=hw_id,
        vendor=vendor,
        model=model,
        architecture=architecture or "Unknown",
        device_type=device_type,
        platform=platform,

        detection_patterns=detection_patterns,
        os_compatibility=["linux", "windows", "macos"],

        cores=cores,
        threads=threads,
        e_cores=e_cores,
        base_frequency_ghz=base_frequency_ghz,
        boost_frequency_ghz=boost_frequency_ghz,

        cuda_cores=cuda_cores,
        sms=sms,
        tensor_cores=tensor_cores if device_type == 'gpu' else None,

        memory_type=memory_type,
        peak_bandwidth_gbps=peak_bandwidth_gbps or 0.0,

        isa_extensions=isa_extensions,
        theoretical_peaks=theoretical_peaks,

        l1_cache_kb=l1_cache_kb,
        l2_cache_kb=l2_cache_kb,
        l3_cache_kb=l3_cache_kb,

        tdp_watts=tdp_watts,
        max_power_watts=max_power_watts,

        release_date=release_date,
        manufacturer_url=manufacturer_url,
        notes=notes,

        data_source="user",
        last_updated=datetime.utcnow().isoformat() + "Z",

        mapper_class=mapper_class,
        mapper_config={}
    )

    return spec


def add_from_file(db: HardwareDatabase, file_path: Path, overwrite: bool = False) -> bool:
    """Add hardware from JSON file"""
    print(f"Loading hardware spec from: {file_path}")

    try:
        spec = HardwareSpec.from_json(file_path)

        # Validate
        errors = spec.validate()
        if errors:
            print("\n⚠ Validation warnings:")
            for error in errors:
                print(f"  - {error}")
            print()

        # Add to database
        success = db.add(spec, overwrite=overwrite)

        if success:
            print(f"✓ Added hardware: {spec.id}")
            return True
        else:
            print(f"✗ Hardware already exists: {spec.id}")
            print("  Use --overwrite to replace existing spec")
            return False

    except Exception as e:
        print(f"✗ Error loading hardware spec: {e}")
        return False


def add_from_detection(db: HardwareDatabase, detection_file: Path) -> bool:
    """Add hardware from detection results JSON"""
    print(f"Loading detection results from: {detection_file}")

    try:
        with open(detection_file) as f:
            detection = json.load(f)

        # Check if CPU detected
        if not detection.get('cpu'):
            print("✗ No CPU detected in file")
            return False

        cpu = detection['cpu']

        print(f"\nDetected CPU: {cpu['model_name']}")
        print(f"Vendor: {cpu['vendor']}")
        print(f"Cores: {cpu['cores']}, Threads: {cpu['threads']}")

        # Prompt for additional info
        proceed = input("\nCreate hardware spec from detection? (y/n): ").strip().lower()
        if proceed != 'y':
            print("Cancelled")
            return False

        # Generate ID
        hw_id = f"{cpu['vendor']}_{cpu['model_name']}".lower()
        hw_id = hw_id.replace(' ', '_').replace('-', '_')
        hw_id = ''.join(c for c in hw_id if c.isalnum() or c == '_')

        hw_id = prompt_required("Hardware ID", default=hw_id)

        # Prompt for missing required fields
        print("\n--- Additional Required Information ---")
        peak_bandwidth_gbps = prompt_float("Peak memory bandwidth (GB/s)")

        print("\n--- Theoretical Performance Peaks ---")
        fp32 = prompt_float("fp32 GFLOPS (required)")

        theoretical_peaks = {'fp32': fp32}
        for prec in ['fp64', 'fp16', 'int64', 'int32', 'int16', 'int8']:
            value = prompt_float(f"{prec} (optional)")
            if value:
                theoretical_peaks[prec] = value

        # Create spec from detection
        spec = HardwareSpec(
            id=hw_id,
            vendor=cpu['vendor'],
            model=cpu['model_name'],
            architecture=cpu['architecture'],
            device_type='cpu',
            platform=detection['platform']['architecture'],

            detection_patterns=[cpu['model_name']],
            os_compatibility=[detection['platform']['os']],

            cores=cpu['cores'],
            threads=cpu['threads'],
            base_frequency_ghz=cpu.get('base_frequency_ghz'),

            peak_bandwidth_gbps=peak_bandwidth_gbps or 0.0,
            isa_extensions=cpu.get('isa_extensions', []),
            theoretical_peaks=theoretical_peaks,

            data_source="detected",
            last_updated=datetime.utcnow().isoformat() + "Z",

            mapper_class="CPUMapper",
            mapper_config={}
        )

        # Validate
        errors = spec.validate()
        if errors:
            print("\n⚠ Validation warnings:")
            for error in errors:
                print(f"  - {error}")

        # Add to database
        success = db.add(spec)

        if success:
            print(f"\n✓ Added hardware: {spec.id}")
            return True
        else:
            print(f"\n✗ Hardware already exists: {spec.id}")
            return False

    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Add new hardware to database"
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=Path(__file__).parent.parent.parent / "hardware_database",
        help="Path to hardware database"
    )
    parser.add_argument(
        "--from-file",
        type=Path,
        help="Add hardware from JSON file"
    )
    parser.add_argument(
        "--from-detection",
        type=Path,
        help="Add hardware from detection results JSON"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing hardware"
    )

    args = parser.parse_args()

    # Load database
    db = HardwareDatabase(args.db)
    db.load_all()

    try:
        # Add from file
        if args.from_file:
            success = add_from_file(db, args.from_file, overwrite=args.overwrite)
            return 0 if success else 1

        # Add from detection
        if args.from_detection:
            success = add_from_detection(db, args.from_detection)
            return 0 if success else 1

        # Interactive mode
        spec = interactive_add_hardware(db)

        # Validate
        print("=" * 80)
        print("Validating Hardware Specification")
        print("=" * 80)
        errors = spec.validate()

        if errors:
            print("⚠ Validation warnings:")
            for error in errors:
                print(f"  - {error}")
            print()
        else:
            print("✓ Specification is valid")
            print()

        # Review
        print("=" * 80)
        print("Review Hardware Specification")
        print("=" * 80)
        print(f"ID:       {spec.id}")
        print(f"Vendor:   {spec.vendor}")
        print(f"Model:    {spec.model}")
        print(f"Type:     {spec.device_type}")
        print(f"Platform: {spec.platform}")
        if spec.cores:
            print(f"Cores:    {spec.cores}")
        if spec.theoretical_peaks:
            peaks_str = ", ".join([f"{k}={v:.0f}" for k, v in list(spec.theoretical_peaks.items())[:3]])
            print(f"Peaks:    {peaks_str}")
        print()

        # Confirm
        confirm = input("Add this hardware to database? (y/n): ").strip().lower()
        if confirm != 'y':
            print("Cancelled")
            return 1

        # Add to database
        success = db.add(spec, overwrite=args.overwrite)

        if success:
            print()
            print(f"✓ Added hardware: {spec.id}")
            spec_file = db._find_spec_file(spec.id)
            if spec_file:
                print(f"  File: {spec_file}")
            print()
            print("Next steps:")
            print("  1. Verify with: python scripts/hardware_db/query_hardware.py --id", spec.id)
            print("  2. Test detection: python scripts/hardware_db/detect_hardware.py")
            return 0
        else:
            print()
            print(f"✗ Hardware already exists: {spec.id}")
            print("  Use --overwrite to replace")
            return 1

    except KeyboardInterrupt:
        print("\n\nCancelled by user")
        return 1
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
