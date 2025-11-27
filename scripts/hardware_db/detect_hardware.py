#!/usr/bin/env python3
"""
Detect Hardware

Auto-detect CPU, GPU, and platform information and match against database.
Supports Linux, Windows (x86_64), and macOS.

Usage:
    python scripts/hardware_db/detect_hardware.py
    python scripts/hardware_db/detect_hardware.py --verbose
    python scripts/hardware_db/detect_hardware.py --export detection.json
"""

import sys
import json
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from graphs.hardware.database import (
    HardwareDatabase,
    HardwareDetector,
)


def format_cpu_info(cpu, verbose=False):
    """Format detected CPU information for display"""
    lines = []
    lines.append(f"Model:        {cpu.model_name}")
    lines.append(f"Vendor:       {cpu.vendor}")
    lines.append(f"Architecture: {cpu.architecture}")

    # Format cores with P-core/E-core breakdown if hybrid CPU
    if cpu.e_cores:
        p_cores = cpu.cores - cpu.e_cores
        lines.append(f"Cores:        {cpu.cores} cores ({p_cores}P + {cpu.e_cores}E), {cpu.threads} threads")
    else:
        lines.append(f"Cores:        {cpu.cores} cores, {cpu.threads} threads")

    if cpu.base_frequency_ghz:
        lines.append(f"Frequency:    {cpu.base_frequency_ghz:.2f} GHz")

    if verbose and cpu.isa_extensions:
        ext_str = ", ".join(cpu.isa_extensions[:8])
        if len(cpu.isa_extensions) > 8:
            ext_str += f" ... ({len(cpu.isa_extensions)} total)"
        lines.append(f"ISA:          {ext_str}")

    return "\n".join(lines)


def format_gpu_info(gpu, verbose=False):
    """Format detected GPU information for display"""
    lines = []
    lines.append(f"Model:        {gpu.model_name}")
    lines.append(f"Vendor:       {gpu.vendor}")

    if gpu.memory_gb:
        lines.append(f"Memory:       {gpu.memory_gb} GB")

    if gpu.cuda_capability:
        lines.append(f"CUDA Cap:     {gpu.cuda_capability}")

    if verbose and gpu.driver_version:
        lines.append(f"Driver:       {gpu.driver_version}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Auto-detect hardware and match against database"
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=Path(__file__).parent.parent.parent / "hardware_database",
        help="Path to hardware database"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed information"
    )
    parser.add_argument(
        "--export",
        type=Path,
        help="Export detection results to JSON file"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("Hardware Detection")
    print("=" * 80)
    print()

    # Load database
    db = HardwareDatabase(args.db)
    print(f"Loading database from: {args.db}")
    db.load_all()
    print(f"Loaded {len(db._cache)} hardware specs")
    print()

    # Create detector
    detector = HardwareDetector()

    # Show platform info
    print("Platform Information")
    print("-" * 80)
    print(f"OS:           {detector.os_type}")
    print(f"Architecture: {detector.platform_arch}")
    print()

    # Auto-detect all hardware
    results = detector.auto_detect(db)

    # Display CPU
    if results['cpu']:
        print("Detected CPU")
        print("-" * 80)
        print(format_cpu_info(results['cpu'], verbose=args.verbose))
        print()

        # Show database matches
        if results['cpu_matches']:
            print("Database Matches (CPU)")
            print("-" * 80)
            for i, match in enumerate(results['cpu_matches'][:3]):  # Top 3 matches
                spec = match.matched_spec
                conf_pct = match.confidence * 100
                print(f"{i+1}. {spec.id} (confidence: {conf_pct:.0f}%)")
                print(f"   {spec.vendor} {spec.model}")
                print(f"   Platform: {spec.platform}, Mapper: {spec.mapper_class}")
                if args.verbose and spec.theoretical_peaks:
                    peaks_str = ", ".join([
                        f"{k}={v:.0f}"
                        for k, v in list(spec.theoretical_peaks.items())[:3]
                    ])
                    print(f"   Peaks: {peaks_str} (GFLOPS/GIOPS)")
                print()
        else:
            print("⚠ No matching CPU found in database")
            print()

    else:
        print("⚠ CPU detection failed")
        print()

    # Display GPUs
    if results['gpus']:
        for i, gpu in enumerate(results['gpus']):
            print(f"Detected GPU #{i+1}")
            print("-" * 80)
            print(format_gpu_info(gpu, verbose=args.verbose))
            print()

            # Show database matches
            if i < len(results['gpu_matches']) and results['gpu_matches'][i]:
                print(f"Database Matches (GPU #{i+1})")
                print("-" * 80)
                for j, match in enumerate(results['gpu_matches'][i][:3]):  # Top 3
                    spec = match.matched_spec
                    conf_pct = match.confidence * 100
                    print(f"{j+1}. {spec.id} (confidence: {conf_pct:.0f}%)")
                    print(f"   {spec.vendor} {spec.model}")
                    print(f"   Platform: {spec.platform}, Mapper: {spec.mapper_class}")
                    if args.verbose and spec.theoretical_peaks:
                        peaks_str = ", ".join([
                            f"{k}={v:.0f}"
                            for k, v in list(spec.theoretical_peaks.items())[:3]
                        ])
                        print(f"   Peaks: {peaks_str} (GFLOPS/GIOPS)")
                    print()
            else:
                print(f"⚠ No matching GPU found in database")
                print()

    else:
        print("No GPUs detected")
        print()

    # Display board detection (for embedded/SoC devices)
    if results.get('board'):
        board = results['board']
        print("Detected Board/SoC")
        print("-" * 80)
        print(f"Model:        {board.model}")
        print(f"Vendor:       {board.vendor}")
        if board.family:
            print(f"Family:       {board.family}")
        if board.soc:
            print(f"SoC:          {board.soc}")
        if args.verbose:
            if board.device_tree_model:
                print(f"Device Tree:  {board.device_tree_model}")
            if board.tegra_release:
                print(f"Tegra:        {board.tegra_release}")
            if board.compatible_strings:
                print(f"Compatible:   {board.compatible_strings[:3]}")
        print()

        if results.get('board_match'):
            board_match = results['board_match']
            conf_pct = board_match.confidence * 100
            print("Board Match")
            print("-" * 80)
            print(f"Board ID:     {board_match.board_id}")
            print(f"Confidence:   {conf_pct:.0f}%")
            print(f"Components:   CPU={board_match.components.get('cpu', 'N/A')}, "
                  f"GPU={board_match.components.get('gpu', 'N/A')}")
            if args.verbose and board_match.matched_signals:
                print(f"Signals:      {', '.join(board_match.matched_signals)}")
            print()

    # Export if requested
    if args.export:
        export_data = {
            'platform': {
                'os': results['os'],
                'architecture': results['platform']
            },
            'cpu': None,
            'cpu_best_match': None,
            'gpus': [],
            'gpu_best_matches': []
        }

        if results['cpu']:
            cpu = results['cpu']
            export_data['cpu'] = {
                'model_name': cpu.model_name,
                'vendor': cpu.vendor,
                'architecture': cpu.architecture,
                'cores': cpu.cores,
                'threads': cpu.threads,
                'base_frequency_ghz': cpu.base_frequency_ghz,
                'isa_extensions': cpu.isa_extensions
            }

            if results['cpu_matches']:
                match = results['cpu_matches'][0]
                export_data['cpu_best_match'] = {
                    'id': match.matched_spec.id,
                    'confidence': match.confidence,
                    'model': match.matched_spec.model
                }

        for i, gpu in enumerate(results['gpus']):
            gpu_data = {
                'model_name': gpu.model_name,
                'vendor': gpu.vendor,
                'memory_gb': gpu.memory_gb,
                'cuda_capability': gpu.cuda_capability,
                'driver_version': gpu.driver_version
            }
            export_data['gpus'].append(gpu_data)

            if i < len(results['gpu_matches']) and results['gpu_matches'][i]:
                match = results['gpu_matches'][i][0]
                match_data = {
                    'id': match.matched_spec.id,
                    'confidence': match.confidence,
                    'model': match.matched_spec.model
                }
                export_data['gpu_best_matches'].append(match_data)
            else:
                export_data['gpu_best_matches'].append(None)

        with open(args.export, 'w') as f:
            json.dump(export_data, f, indent=2)

        print(f"✓ Detection results exported to: {args.export}")
        print()

    # Summary
    print("=" * 80)
    print("Summary")
    print("=" * 80)

    # Check if matches came via board detection
    board_match = results.get('board_match')
    via_board = ""
    if board_match and board_match.confidence > 0.5:
        via_board = f" (via board: {board_match.board_id})"

    cpu_match_status = "✓ Matched" + via_board if results['cpu_matches'] else "✗ Not found"
    gpu_count = len(results['gpus'])
    gpu_matched = sum(1 for matches in results['gpu_matches'] if matches)

    print(f"CPU:   {cpu_match_status}")
    print(f"GPUs:  {gpu_count} detected, {gpu_matched} matched in database{via_board if gpu_matched and via_board else ''}")
    if board_match:
        print(f"Board: ✓ {board_match.board_id} ({board_match.confidence*100:.0f}% confidence)")
    print()

    if not results['cpu_matches'] or gpu_count != gpu_matched:
        print("⚠ Some hardware not found in database.")
        print("  Consider adding missing hardware with:")
        print("  python scripts/hardware_db/add_hardware.py")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
