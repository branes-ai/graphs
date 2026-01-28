#!/usr/bin/env python
"""
Comprehensive Device Detection Tool

Detects and displays complete system information for characterization:
- Platform: OS, kernel, Python version, framework versions (PyTorch, NumPy, CUDA, ROCm)
- CPU: Processor details, cores, threads, cache, ISA extensions
- GPU: NVIDIA, AMD, and Intel GPUs with memory and driver info
- Board: Embedded system detection (Jetson, Raspberry Pi, etc.)
- Memory: System RAM and NUMA configuration
- Runtime: Current clock frequencies, power modes, temperatures

Usage:
    ./cli/device_detection.py              # Full detection report
    ./cli/device_detection.py --json       # JSON output
    ./cli/device_detection.py --section platform  # Show only platform info
    ./cli/device_detection.py --section cpu       # Show only CPU info
    ./cli/device_detection.py --section gpu       # Show only GPU info
    ./cli/device_detection.py --section board     # Show only board info
    ./cli/device_detection.py --section memory    # Show only memory info
    ./cli/device_detection.py --section runtime   # Show only runtime state (clocks, power)
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from calibrate_hardware import detect_actual_device, detect_platform
from graphs.hardware.database import HardwareDetector
from graphs.calibration.gpu_clock import get_gpu_clock_info, GPUClockInfo
from graphs.calibration.cpu_clock import get_cpu_clock_info, CPUClockInfo


def format_size(size_kb: int) -> str:
    """Format size in KB to human-readable string."""
    if size_kb >= 1024 * 1024:
        return f"{size_kb / (1024 * 1024):.1f} GB"
    elif size_kb >= 1024:
        return f"{size_kb / 1024:.1f} MB"
    else:
        return f"{size_kb} KB"


def print_section_header(title: str):
    """Print a section header."""
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def print_platform_info(platform_info: dict):
    """Print platform and software stack information."""
    print_section_header("PLATFORM & SOFTWARE STACK")

    # Operating System
    print()
    print("Operating System:")
    print(f"  OS:              {platform_info.get('os', 'Unknown')}")
    if platform_info.get('distro_name'):
        print(f"  Distribution:    {platform_info.get('distro_name')} {platform_info.get('distro_version', '')}")
    print(f"  Kernel/Release:  {platform_info.get('os_release', 'Unknown')}")
    print(f"  Architecture:    {platform_info.get('architecture', 'Unknown')}")
    print(f"  Hostname:        {platform_info.get('hostname', 'Unknown')}")

    # Python
    print()
    print("Python:")
    print(f"  Version:         {platform_info.get('python_version', 'Unknown')}")
    print(f"  Implementation:  {platform_info.get('python_implementation', 'Unknown')}")
    print(f"  Compiler:        {platform_info.get('python_compiler', 'Unknown')}")

    # NumPy
    print()
    print("NumPy:")
    if platform_info.get('numpy_version'):
        print(f"  Version:         {platform_info.get('numpy_version')}")
        if platform_info.get('numpy_blas'):
            print(f"  BLAS Backend:    {platform_info.get('numpy_blas')}")
    else:
        print("  Status:          Not installed")

    # PyTorch
    print()
    print("PyTorch:")
    if platform_info.get('pytorch_version'):
        print(f"  Version:         {platform_info.get('pytorch_version')}")
        if platform_info.get('cuda_available'):
            print(f"  CUDA Version:    {platform_info.get('cuda_version', 'Unknown')}")
            if platform_info.get('cudnn_version'):
                print(f"  cuDNN Version:   {platform_info.get('cudnn_version')}")
            print(f"  CUDA Devices:    {platform_info.get('cuda_device_count', 0)}")
        else:
            print("  CUDA:            Not available")

        if platform_info.get('rocm_available'):
            print(f"  ROCm Version:    {platform_info.get('rocm_version', 'Unknown')}")

        if platform_info.get('mps_available'):
            print("  MPS (Apple):     Available")
    else:
        print("  Status:          Not installed")

    # Intel oneAPI / MKL
    if platform_info.get('mkl_root') or platform_info.get('oneapi_root'):
        print()
        print("Intel Libraries:")
        if platform_info.get('oneapi_root'):
            print(f"  oneAPI Root:     {platform_info.get('oneapi_root')}")
        if platform_info.get('mkl_root'):
            print(f"  MKL Root:        {platform_info.get('mkl_root')}")

    # Thread settings
    thread_vars = []
    if platform_info.get('omp_num_threads'):
        thread_vars.append(f"OMP_NUM_THREADS={platform_info.get('omp_num_threads')}")
    if platform_info.get('mkl_num_threads'):
        thread_vars.append(f"MKL_NUM_THREADS={platform_info.get('mkl_num_threads')}")
    if platform_info.get('openblas_num_threads'):
        thread_vars.append(f"OPENBLAS_NUM_THREADS={platform_info.get('openblas_num_threads')}")

    if thread_vars:
        print()
        print("Thread Configuration:")
        for var in thread_vars:
            print(f"  {var}")


def print_cpu_info(cpu):
    """Print CPU information."""
    print_section_header("CPU")

    if cpu is None:
        print()
        print("  No CPU information detected")
        return

    print()
    print(f"  Model:           {cpu.model_name}")
    print(f"  Vendor:          {cpu.vendor}")
    print(f"  Architecture:    {cpu.architecture}")
    print(f"  Physical Cores:  {cpu.cores}")
    print(f"  Logical Threads: {cpu.threads}")

    if cpu.e_cores:
        p_cores = cpu.cores - cpu.e_cores
        print(f"  Core Types:      {p_cores} P-cores + {cpu.e_cores} E-cores (hybrid)")

    if cpu.base_frequency_ghz:
        print(f"  Base Frequency:  {cpu.base_frequency_ghz:.2f} GHz")
    if cpu.boost_frequency_ghz:
        print(f"  Boost Frequency: {cpu.boost_frequency_ghz:.2f} GHz")

    # Cache information
    if cpu.l1_dcache_kb or cpu.l2_cache_kb or cpu.l3_cache_kb:
        print()
        print("  Cache Hierarchy:")
        if cpu.l1_dcache_kb:
            print(f"    L1 Data:       {format_size(cpu.l1_dcache_kb)}")
        if cpu.l1_icache_kb:
            print(f"    L1 Instruction:{format_size(cpu.l1_icache_kb)}")
        if cpu.l2_cache_kb:
            print(f"    L2:            {format_size(cpu.l2_cache_kb)}")
        if cpu.l3_cache_kb:
            print(f"    L3:            {format_size(cpu.l3_cache_kb)}")

    # ISA extensions
    if cpu.isa_extensions:
        print()
        print("  ISA Extensions:")
        # Group extensions by category
        simd_ext = [e for e in cpu.isa_extensions if any(x in e.lower() for x in ['sse', 'avx', 'neon', 'sve'])]
        other_ext = [e for e in cpu.isa_extensions if e not in simd_ext]

        if simd_ext:
            print(f"    SIMD:          {', '.join(simd_ext[:6])}")
            if len(simd_ext) > 6:
                print(f"                   {', '.join(simd_ext[6:])}")
        if other_ext:
            print(f"    Other:         {', '.join(other_ext[:6])}")
            if len(other_ext) > 6:
                print(f"                   {', '.join(other_ext[6:])}")


def print_gpu_info(gpus: list):
    """Print GPU information."""
    print_section_header("GPU(s)")

    if not gpus:
        print()
        print("  No GPUs detected")
        return

    for i, gpu in enumerate(gpus):
        print()
        print(f"  GPU #{i + 1}:")
        print(f"    Model:         {gpu.model_name}")
        print(f"    Vendor:        {gpu.vendor}")
        if gpu.memory_gb:
            print(f"    Memory:        {gpu.memory_gb} GB")
        if gpu.cuda_capability:
            print(f"    CUDA Compute:  {gpu.cuda_capability}")
        if gpu.driver_version:
            print(f"    Driver:        {gpu.driver_version}")


def print_board_info(board):
    """Print board/embedded system information."""
    print_section_header("BOARD / EMBEDDED SYSTEM")

    if board is None:
        print()
        print("  Not an embedded system or board not detected")
        return

    print()
    print(f"  Model:           {board.model}")
    print(f"  Vendor:          {board.vendor}")
    if board.family:
        print(f"  Family:          {board.family}")
    if board.soc:
        print(f"  SoC:             {board.soc}")
    if board.device_tree_model:
        print(f"  Device Tree:     {board.device_tree_model}")
    if board.tegra_release:
        print(f"  Tegra Release:   {board.tegra_release}")


def print_memory_info(memory):
    """Print memory information."""
    print_section_header("SYSTEM MEMORY")

    if memory is None:
        print()
        print("  Memory detection not available")
        return

    print()
    print(f"  Total:           {memory.total_gb:.1f} GB")
    if memory.numa_nodes:
        print(f"  NUMA Nodes:      {memory.numa_nodes}")

    if memory.channels:
        print()
        print("  Memory Channels:")
        for ch in memory.channels:
            details = []
            if ch.type:
                details.append(ch.type.upper())
            if ch.speed_mts:
                details.append(f"{ch.speed_mts} MT/s")
            if ch.size_gb:
                details.append(f"{ch.size_gb:.0f} GB")

            print(f"    {ch.name}: {' / '.join(details)}")


def print_runtime_info(platform_info: dict):
    """Print runtime state: current clocks, power modes, temperatures."""
    print_section_header("RUNTIME STATE")

    # CPU Runtime
    print()
    print("CPU:")
    try:
        cpu_clock = get_cpu_clock_info()
        if cpu_clock.query_success:
            if cpu_clock.current_freq_mhz:
                print(f"  Current Frequency: {cpu_clock.current_freq_mhz:.0f} MHz")
            if cpu_clock.min_freq_mhz and cpu_clock.max_freq_mhz:
                print(f"  Frequency Range:   {cpu_clock.min_freq_mhz:.0f} - {cpu_clock.max_freq_mhz:.0f} MHz")
            if cpu_clock.governor:
                print(f"  Governor:          {cpu_clock.governor}")
            if cpu_clock.driver:
                print(f"  Scaling Driver:    {cpu_clock.driver}")
            print(f"  Query Method:      {cpu_clock.query_method}")
        else:
            print(f"  Query failed: {cpu_clock.error_message or 'Unknown error'}")
    except Exception as e:
        print(f"  Query failed: {e}")

    # GPU Runtime (only if CUDA available)
    if platform_info.get('cuda_available'):
        print()
        print("GPU:")
        try:
            gpu_clock = get_gpu_clock_info()
            if gpu_clock.query_success:
                if gpu_clock.sm_clock_mhz:
                    max_str = f" (max: {gpu_clock.max_sm_clock_mhz})" if gpu_clock.max_sm_clock_mhz else ""
                    print(f"  SM Clock:          {gpu_clock.sm_clock_mhz} MHz{max_str}")
                if gpu_clock.mem_clock_mhz:
                    max_str = f" (max: {gpu_clock.max_mem_clock_mhz})" if gpu_clock.max_mem_clock_mhz else ""
                    print(f"  Memory Clock:      {gpu_clock.mem_clock_mhz} MHz{max_str}")
                if gpu_clock.power_mode_name:
                    mode_num = f" (mode {gpu_clock.nvpmodel_mode})" if gpu_clock.nvpmodel_mode is not None else ""
                    print(f"  Power Mode:        {gpu_clock.power_mode_name}{mode_num}")
                if gpu_clock.power_draw_watts:
                    limit_str = f" / {gpu_clock.power_limit_watts:.0f}W limit" if gpu_clock.power_limit_watts else ""
                    print(f"  Power Draw:        {gpu_clock.power_draw_watts:.1f}W{limit_str}")
                if gpu_clock.temperature_c:
                    print(f"  Temperature:       {gpu_clock.temperature_c} C")
                print(f"  Query Method:      {gpu_clock.query_method}")
            else:
                print(f"  Query failed: {gpu_clock.error_message or 'Unknown error'}")
        except Exception as e:
            print(f"  Query failed: {e}")


def print_device_availability(platform_info: dict):
    """Print device availability summary for compute."""
    print_section_header("COMPUTE DEVICE AVAILABILITY")

    print()
    print("  CPU:             Always available")

    # CUDA (NVIDIA)
    if platform_info.get('cuda_available'):
        count = platform_info.get('cuda_device_count', 1)
        print(f"  CUDA (NVIDIA):   Available ({count} device{'s' if count > 1 else ''})")
    else:
        print("  CUDA (NVIDIA):   Not available")

    # ROCm (AMD)
    if platform_info.get('rocm_available'):
        print(f"  ROCm (AMD):      Available (v{platform_info.get('rocm_version', 'Unknown')})")
    else:
        print("  ROCm (AMD):      Not available")

    # MPS (Apple)
    if platform_info.get('mps_available'):
        print("  MPS (Apple):     Available")
    else:
        print("  MPS (Apple):     Not available")


def collect_all_info() -> dict:
    """Collect all detection information into a dictionary."""
    detector = HardwareDetector()
    hardware = detector.detect_all()
    platform_info = detect_platform()

    result = {
        'platform': platform_info,
        'cpu': None,
        'gpus': [],
        'board': None,
        'memory': None,
    }

    # Convert dataclasses to dicts for JSON serialization
    if hardware.get('cpu'):
        cpu = hardware['cpu']
        result['cpu'] = {
            'model_name': cpu.model_name,
            'vendor': cpu.vendor,
            'architecture': cpu.architecture,
            'cores': cpu.cores,
            'threads': cpu.threads,
            'base_frequency_ghz': cpu.base_frequency_ghz,
            'boost_frequency_ghz': cpu.boost_frequency_ghz,
            'e_cores': cpu.e_cores,
            'isa_extensions': cpu.isa_extensions,
            'l1_dcache_kb': cpu.l1_dcache_kb,
            'l1_icache_kb': cpu.l1_icache_kb,
            'l2_cache_kb': cpu.l2_cache_kb,
            'l3_cache_kb': cpu.l3_cache_kb,
        }

    if hardware.get('gpus'):
        for gpu in hardware['gpus']:
            result['gpus'].append({
                'model_name': gpu.model_name,
                'vendor': gpu.vendor,
                'memory_gb': gpu.memory_gb,
                'cuda_capability': gpu.cuda_capability,
                'driver_version': gpu.driver_version,
            })

    if hardware.get('board'):
        board = hardware['board']
        result['board'] = {
            'model': board.model,
            'vendor': board.vendor,
            'family': board.family,
            'soc': board.soc,
            'device_tree_model': board.device_tree_model,
            'tegra_release': board.tegra_release,
        }

    if hardware.get('memory'):
        memory = hardware['memory']
        result['memory'] = {
            'total_gb': memory.total_gb,
            'numa_nodes': memory.numa_nodes,
            'channels': [
                {
                    'name': ch.name,
                    'type': ch.type,
                    'size_gb': ch.size_gb,
                    'speed_mts': ch.speed_mts,
                }
                for ch in memory.channels
            ] if memory.channels else [],
        }

    # Runtime state (clocks, power modes)
    runtime = {'cpu': None, 'gpu': None}
    try:
        cpu_clock = get_cpu_clock_info()
        if cpu_clock.query_success:
            runtime['cpu'] = {
                'current_freq_mhz': cpu_clock.current_freq_mhz,
                'min_freq_mhz': cpu_clock.min_freq_mhz,
                'max_freq_mhz': cpu_clock.max_freq_mhz,
                'governor': cpu_clock.governor,
                'driver': cpu_clock.driver,
                'query_method': cpu_clock.query_method,
            }
    except Exception:
        pass

    if platform_info.get('cuda_available'):
        try:
            gpu_clock = get_gpu_clock_info()
            if gpu_clock.query_success:
                runtime['gpu'] = {
                    'sm_clock_mhz': gpu_clock.sm_clock_mhz,
                    'max_sm_clock_mhz': gpu_clock.max_sm_clock_mhz,
                    'mem_clock_mhz': gpu_clock.mem_clock_mhz,
                    'max_mem_clock_mhz': gpu_clock.max_mem_clock_mhz,
                    'power_mode_name': gpu_clock.power_mode_name,
                    'nvpmodel_mode': gpu_clock.nvpmodel_mode,
                    'power_draw_watts': gpu_clock.power_draw_watts,
                    'power_limit_watts': gpu_clock.power_limit_watts,
                    'temperature_c': gpu_clock.temperature_c,
                    'query_method': gpu_clock.query_method,
                }
        except Exception:
            pass

    result['runtime'] = runtime

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive Device Detection Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument("--json", action="store_true",
                        help="Output in JSON format")
    parser.add_argument("--section", type=str,
                        choices=['platform', 'cpu', 'gpu', 'board', 'memory', 'runtime', 'all'],
                        default='all',
                        help="Show only specific section (default: all)")

    args = parser.parse_args()

    # Collect all info
    detector = HardwareDetector()
    hardware = detector.detect_all()
    platform_info = detect_platform()

    # JSON output
    if args.json:
        all_info = collect_all_info()
        if args.section != 'all':
            if args.section == 'gpu':
                all_info = {'gpus': all_info['gpus']}
            else:
                all_info = {args.section: all_info.get(args.section)}
        print(json.dumps(all_info, indent=2, default=str))
        return 0

    # Text output
    print()
    print("=" * 80)
    print("DEVICE DETECTION REPORT")
    print("=" * 80)

    if args.section in ['all', 'platform']:
        print_platform_info(platform_info)

    if args.section in ['all', 'cpu']:
        print_cpu_info(hardware.get('cpu'))

    if args.section in ['all', 'gpu']:
        print_gpu_info(hardware.get('gpus', []))

    if args.section in ['all', 'board']:
        print_board_info(hardware.get('board'))

    if args.section in ['all', 'memory']:
        print_memory_info(hardware.get('memory'))

    if args.section in ['all', 'runtime']:
        print_runtime_info(platform_info)

    if args.section == 'all':
        print_device_availability(platform_info)

    # Summary
    if args.section == 'all':
        print()
        print("=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print()

        cpu = hardware.get('cpu')
        if cpu:
            print(f"  CPU: {cpu.model_name} ({cpu.cores}C/{cpu.threads}T)")
        else:
            print("  CPU: Not detected")

        gpus = hardware.get('gpus', [])
        if gpus:
            for i, gpu in enumerate(gpus):
                mem_str = f" {gpu.memory_gb}GB" if gpu.memory_gb else ""
                print(f"  GPU: {gpu.vendor} {gpu.model_name}{mem_str}")
        else:
            print("  GPU: None detected")

        board = hardware.get('board')
        if board:
            print(f"  Board: {board.model}")

        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
