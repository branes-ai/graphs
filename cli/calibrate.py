#!/usr/bin/env python3
"""
Hardware Calibration CLI (Simplified)

One-command hardware calibration using the unified registry.

Usage:
    # Auto-detect and calibrate current hardware (most common)
    ./cli/calibrate.py

    # Calibrate specific hardware from registry
    ./cli/calibrate.py --id i7_12700k

    # Quick calibration (fewer trials)
    ./cli/calibrate.py --quick

    # List available hardware in registry
    ./cli/calibrate.py --list

    # Show current hardware profile
    ./cli/calibrate.py --show i7_12700k
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from graphs.hardware.registry import get_registry, HardwareProfile


def list_hardware(registry):
    """List all hardware in the registry."""
    profiles = registry.list_all()

    if not profiles:
        print("No hardware profiles found in registry.")
        print(f"Registry path: {registry.path}")
        return

    print(f"\nHardware Registry ({len(profiles)} profiles)")
    print("=" * 60)

    # Group by device type
    by_type = {}
    for pid in profiles:
        profile = registry.get(pid)
        dtype = profile.device_type
        if dtype not in by_type:
            by_type[dtype] = []
        by_type[dtype].append(profile)

    for dtype in sorted(by_type.keys()):
        print(f"\n{dtype.upper()}:")
        for profile in sorted(by_type[dtype], key=lambda p: p.id):
            status = "✓" if profile.is_calibrated else "○"
            print(f"  {status} {profile.id:30s} {profile.model}")

    print()
    print("Legend: ✓ = calibrated, ○ = not calibrated")
    print()


def show_profile(registry, hardware_id: str):
    """Show details of a hardware profile."""
    profile = registry.get(hardware_id)

    if not profile:
        print(f"Hardware '{hardware_id}' not found in registry.")
        print("\nAvailable:")
        for pid in registry.list_all()[:10]:
            print(f"  {pid}")
        if len(registry.list_all()) > 10:
            print(f"  ... and {len(registry.list_all()) - 10} more")
        return 1

    profile.print_summary()
    return 0


def calibrate_hardware(registry, hardware_id: str = None, quick: bool = False,
                       operations: list = None, framework: str = None, force: bool = False):
    """Calibrate hardware and save to registry."""
    try:
        if hardware_id:
            # Calibrate specific hardware
            profile = registry.calibrate(
                hardware_id,
                quick=quick,
                operations=operations,
                framework=framework,
                force=force,
            )
        else:
            # Auto-detect and calibrate
            profile = registry.detect_and_calibrate(
                quick=quick,
                operations=operations,
                framework=framework,
                create_if_missing=True,
                force=force,
            )

        print()
        print("=" * 60)
        print("Calibration Complete!")
        print("=" * 60)
        print()
        print(f"Profile saved: {profile.id}")
        print(f"Location:      {registry.path / profile.device_type / profile.id}")
        print()
        print("Use in analysis:")
        print(f"  ./cli/analyze.py --model resnet18 --hardware {profile.id}")
        print()

        return 0

    except ValueError as e:
        print(f"Error: {e}")
        return 1
    except RuntimeError as e:
        print(f"Error: {e}")
        return 1


def main():
    parser = argparse.ArgumentParser(
        description="Hardware Calibration Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Actions
    action_group = parser.add_mutually_exclusive_group()
    action_group.add_argument(
        '--list', '-l',
        action='store_true',
        help="List all hardware in registry"
    )
    action_group.add_argument(
        '--show', '-s',
        type=str,
        metavar='ID',
        help="Show details of a hardware profile"
    )

    # Hardware selection
    parser.add_argument(
        '--id',
        type=str,
        help="Hardware ID to calibrate (default: auto-detect)"
    )

    # Calibration options
    parser.add_argument(
        '--quick', '-q',
        action='store_true',
        help="Run quick calibration (fewer sizes/trials)"
    )
    parser.add_argument(
        '--force', '-f',
        action='store_true',
        help="Force calibration even if pre-flight checks fail (results flagged as non-representative)"
    )
    parser.add_argument(
        '--operations',
        type=str,
        default=None,
        help="Comma-separated operations to calibrate (default: blas,stream)"
    )
    parser.add_argument(
        '--framework',
        type=str,
        choices=['numpy', 'pytorch'],
        default=None,
        help="Override framework selection"
    )

    # Registry options
    parser.add_argument(
        '--registry',
        type=Path,
        default=None,
        help="Path to hardware registry (default: auto-detect)"
    )

    args = parser.parse_args()

    # Get registry
    if args.registry:
        from graphs.hardware.registry import RegistryConfig
        config = RegistryConfig(registry_path=args.registry)
        registry = get_registry(config)
    else:
        registry = get_registry()

    # Load profiles
    count = registry.load_all()
    if count == 0 and not args.list:
        print(f"Warning: No hardware profiles found in {registry.path}")
        print("You may need to run the migration script first:")
        print("  python scripts/hardware_db/migrate_to_registry.py")
        print()

    # Handle actions
    if args.list:
        list_hardware(registry)
        return 0

    if args.show:
        return show_profile(registry, args.show)

    # Default: calibrate
    operations = None
    if args.operations:
        operations = [op.strip() for op in args.operations.split(',')]

    return calibrate_hardware(
        registry,
        hardware_id=args.id,
        quick=args.quick,
        operations=operations,
        framework=args.framework,
        force=args.force,
    )


if __name__ == "__main__":
    sys.exit(main())
