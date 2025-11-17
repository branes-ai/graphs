#!/usr/bin/env python3
"""
Update Hardware in Database

Update existing hardware specifications in the database.

Usage:
    python scripts/hardware_db/update_hardware.py --id intel_i7_12700k
    python scripts/hardware_db/update_hardware.py --id h100_sxm5 --field architecture --value "Hopper"
    python scripts/hardware_db/update_hardware.py --id h100_sxm5 --interactive
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from graphs.hardware.database import HardwareDatabase


def interactive_update(db: HardwareDatabase, hw_id: str) -> bool:
    """Interactive update of hardware spec"""
    spec = db.get(hw_id)
    if not spec:
        print(f"✗ Hardware not found: {hw_id}")
        return False

    print()
    print("=" * 80)
    print(f"Update Hardware: {hw_id}")
    print("=" * 80)
    print()
    print("Current values shown in [brackets]. Press Enter to keep current value.")
    print("Press Ctrl+C to cancel")
    print()

    def prompt_update(field_name: str, current_value, value_type=str):
        """Prompt for field update"""
        current_str = str(current_value) if current_value is not None else "None"
        prompt = f"{field_name} [{current_str}]: "

        new_value = input(prompt).strip()

        if not new_value:
            return current_value

        try:
            if value_type == int:
                return int(new_value)
            elif value_type == float:
                return float(new_value)
            elif value_type == list:
                return [item.strip() for item in new_value.split(',') if item.strip()]
            else:
                return new_value
        except ValueError:
            print(f"  Invalid {value_type.__name__}, keeping current value")
            return current_value

    # Basic identification
    print("--- Basic Identification ---")
    spec.vendor = prompt_update("Vendor", spec.vendor)
    spec.model = prompt_update("Model", spec.model)
    spec.architecture = prompt_update("Architecture", spec.architecture)
    print()

    # Detection patterns
    print("--- Detection Patterns ---")
    patterns_str = ", ".join(spec.detection_patterns) if spec.detection_patterns else ""
    new_patterns = input(f"Detection patterns (comma-separated) [{patterns_str}]: ").strip()
    if new_patterns:
        spec.detection_patterns = [p.strip() for p in new_patterns.split(',') if p.strip()]
    print()

    # Core specifications
    print("--- Core Specifications ---")
    if spec.device_type == 'cpu':
        spec.cores = prompt_update("Physical cores", spec.cores, int)
        spec.threads = prompt_update("Hardware threads", spec.threads, int)
        spec.e_cores = prompt_update("E-cores", spec.e_cores, int)

    elif spec.device_type == 'gpu':
        spec.cuda_cores = prompt_update("CUDA cores", spec.cuda_cores, int)
        spec.sms = prompt_update("SMs", spec.sms, int)
        spec.tensor_cores = prompt_update("Tensor cores", spec.tensor_cores, int)

    spec.base_frequency_ghz = prompt_update("Base frequency (GHz)", spec.base_frequency_ghz, float)
    spec.boost_frequency_ghz = prompt_update("Boost frequency (GHz)", spec.boost_frequency_ghz, float)
    print()

    # Memory
    print("--- Memory ---")
    spec.memory_type = prompt_update("Memory type", spec.memory_type)
    spec.peak_bandwidth_gbps = prompt_update("Peak bandwidth (GB/s)", spec.peak_bandwidth_gbps, float)
    print()

    # ISA extensions
    print("--- ISA Extensions ---")
    isa_str = ", ".join(spec.isa_extensions) if spec.isa_extensions else ""
    new_isa = input(f"ISA extensions (comma-separated) [{isa_str}]: ").strip()
    if new_isa:
        spec.isa_extensions = [e.strip() for e in new_isa.split(',') if e.strip()]
    print()

    # Theoretical peaks
    print("--- Theoretical Peaks ---")
    print("Enter new values to update, or press Enter to keep current")
    for prec in ['fp64', 'fp32', 'fp16', 'bf16', 'fp8', 'int64', 'int32', 'int16', 'int8']:
        current = spec.theoretical_peaks.get(prec)
        new_value = prompt_update(f"  {prec}", current, float)
        if new_value is not None:
            spec.theoretical_peaks[prec] = new_value
        elif prec in spec.theoretical_peaks:
            # Remove if set to None
            remove = input(f"  Remove {prec}? (y/n): ").strip().lower()
            if remove == 'y':
                del spec.theoretical_peaks[prec]
    print()

    # Cache
    print("--- Cache ---")
    spec.l1_cache_kb = prompt_update("L1 cache (KB)", spec.l1_cache_kb, int)
    spec.l2_cache_kb = prompt_update("L2 cache (KB)", spec.l2_cache_kb, int)
    spec.l3_cache_kb = prompt_update("L3 cache (KB)", spec.l3_cache_kb, int)
    print()

    # Power
    print("--- Power ---")
    spec.tdp_watts = prompt_update("TDP (watts)", spec.tdp_watts, float)
    spec.max_power_watts = prompt_update("Max power (watts)", spec.max_power_watts, float)
    print()

    # Metadata
    print("--- Metadata ---")
    spec.release_date = prompt_update("Release date", spec.release_date)
    spec.manufacturer_url = prompt_update("Manufacturer URL", spec.manufacturer_url)
    spec.notes = prompt_update("Notes", spec.notes)
    print()

    # Update timestamp
    spec.last_updated = datetime.utcnow().isoformat() + "Z"

    # Validate
    errors = spec.validate()
    if errors:
        print("⚠ Validation warnings:")
        for error in errors:
            print(f"  - {error}")
        print()

    # Confirm
    print("=" * 80)
    confirm = input("Save changes? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Cancelled")
        return False

    # Update in database (use add with overwrite)
    success = db.add(spec, overwrite=True)

    if success:
        print(f"✓ Updated hardware: {hw_id}")
        return True
    else:
        print(f"✗ Failed to update hardware: {hw_id}")
        return False


def update_field(db: HardwareDatabase, hw_id: str, field: str, value: str) -> bool:
    """Update a single field in hardware spec"""
    spec = db.get(hw_id)
    if not spec:
        print(f"✗ Hardware not found: {hw_id}")
        return False

    # Convert value to appropriate type
    spec_dict = spec.to_json()

    if field not in spec_dict:
        print(f"✗ Unknown field: {field}")
        print("Available fields:")
        for f in sorted(spec_dict.keys()):
            print(f"  - {f}")
        return False

    # Get current value and type
    current_value = getattr(spec, field)
    current_type = type(current_value)

    # Convert value
    try:
        if current_type == int:
            new_value = int(value)
        elif current_type == float:
            new_value = float(value)
        elif current_type == bool:
            new_value = value.lower() in ('true', 'yes', '1', 'y')
        elif current_type == list:
            new_value = [item.strip() for item in value.split(',') if item.strip()]
        elif current_type == dict:
            new_value = json.loads(value)
        else:
            new_value = value

        # Set new value
        setattr(spec, field, new_value)

        # Update timestamp
        spec.last_updated = datetime.utcnow().isoformat() + "Z"

        # Validate
        errors = spec.validate()
        if errors:
            print("⚠ Validation warnings:")
            for error in errors:
                print(f"  - {error}")
            print()

        # Update in database (use add with overwrite)
        success = db.add(spec, overwrite=True)

        if success:
            print(f"✓ Updated {hw_id}.{field}: {current_value} → {new_value}")
            return True
        else:
            print(f"✗ Failed to update hardware")
            return False

    except (ValueError, json.JSONDecodeError) as e:
        print(f"✗ Invalid value for field {field}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Update hardware in database"
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=Path(__file__).parent.parent.parent / "hardware_database",
        help="Path to hardware database"
    )
    parser.add_argument(
        "--id",
        required=True,
        help="Hardware ID to update"
    )
    parser.add_argument(
        "--field",
        help="Field to update (use with --value)"
    )
    parser.add_argument(
        "--value",
        help="New value for field"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Interactive update mode"
    )

    args = parser.parse_args()

    # Load database
    db = HardwareDatabase(args.db)
    print(f"Loading database from: {args.db}")
    db.load_all()
    print()

    try:
        # Field update mode
        if args.field and args.value:
            success = update_field(db, args.id, args.field, args.value)
            return 0 if success else 1

        # Interactive mode
        if args.interactive or (not args.field and not args.value):
            success = interactive_update(db, args.id)
            return 0 if success else 1

        # Missing arguments
        print("Please specify either:")
        print("  --field FIELD --value VALUE")
        print("  --interactive")
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
