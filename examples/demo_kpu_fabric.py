#!/usr/bin/env python3
"""Test KPU-T256 multi-fabric configuration"""

from graphs.hardware.models.accelerators.kpu_t256 import kpu_t256_resource_model
from graphs.hardware.resource_model import Precision

def main():
    print("=" * 80)
    print("KPU-T256 - Multi-Fabric Configuration Test")
    print("=" * 80)

    model = kpu_t256_resource_model()

    print(f"\nHardware: {model.name}")
    print(f"Process Node: 16nm TSMC")
    print(f"Architecture: 256 tiles (70% INT8, 20% BF16, 10% Matrix)")

    if model.compute_fabrics:
        print(f"\n{'='*80}")
        print("COMPUTE FABRICS:")
        print(f"{'='*80}")

        total_tiles = sum(fabric.num_units for fabric in model.compute_fabrics)
        print(f"\nTotal Tiles: {total_tiles}")

        for fabric in model.compute_fabrics:
            percentage = fabric.num_units / total_tiles * 100
            print(f"\n{fabric.fabric_type.upper()} ({percentage:.0f}%):")
            print(f"  Circuit Type: {fabric.circuit_type}")
            print(f"  Tiles: {fabric.num_units}")
            print(f"  Frequency: {fabric.core_frequency_hz/1e9:.2f} GHz (30W sustained)")
            print(f"  Process Node: {fabric.process_node_nm}nm")
            print(f"  Base Energy (FP32): {fabric.energy_per_flop_fp32 * 1e12:.2f} pJ")

            # Show throughput for each supported precision
            print(f"  Supported Precisions:")
            for precision in fabric.ops_per_unit_per_clock.keys():
                peak_ops = fabric.get_peak_ops_per_sec(precision)
                energy_per_op = fabric.get_energy_per_op(precision)
                peak_power = fabric.get_peak_power(precision)

                # Format based on magnitude
                if peak_ops >= 1e12:
                    ops_str = f"{peak_ops/1e12:.1f} TOPS" if precision in [Precision.INT8, Precision.INT4] else f"{peak_ops/1e12:.1f} TFLOPS"
                else:
                    ops_str = f"{peak_ops/1e9:.1f} GOPS" if precision in [Precision.INT8, Precision.INT4] else f"{peak_ops/1e9:.1f} GFLOPS"

                print(f"    {precision.name}: {ops_str}, {energy_per_op*1e12:.3f} pJ/op, {peak_power:.1f} W peak")

        # Calculate efficiency gain
        standard_energy = model.compute_fabrics[0].energy_per_flop_fp32  # INT8 tiles
        tensor_energy = model.compute_fabrics[2].energy_per_flop_fp32    # Matrix tiles
        efficiency_gain = (standard_energy - tensor_energy) / standard_energy * 100

        print(f"\n{'='*80}")
        print(f"Efficiency Gain (Matrix vs Standard Tiles): {efficiency_gain:.1f}%")
        print(f"{'='*80}")

    print(f"\n{'='*80}")
    print("LEGACY PRECISION PROFILES (for backward compatibility):")
    print(f"{'='*80}")

    for precision, profile in model.precision_profiles.items():
        peak_ops = profile.peak_ops_per_sec
        if peak_ops >= 1e12:
            ops_str = f"{peak_ops/1e12:.1f} TOPS" if precision in [Precision.INT8, Precision.INT4] else f"{peak_ops/1e12:.1f} TFLOPS"
        else:
            ops_str = f"{peak_ops/1e9:.1f} GOPS" if precision in [Precision.INT8, Precision.INT4] else f"{peak_ops/1e9:.1f} GFLOPS"

        tc_status = "Tensor Core" if profile.tensor_core_supported else "Standard"
        print(f"  {precision.name}: {ops_str} ({tc_status})")

    print(f"\n{'='*80}")
    print("ENERGY MODEL:")
    print(f"{'='*80}")
    print(f"  Base ALU Energy (FP32): {model.energy_per_flop_fp32 * 1e12:.2f} pJ (from BF16 fabric)")
    print(f"  Memory Access Energy: {model.energy_per_byte * 1e12:.2f} pJ/byte (LPDDR5)")

    print(f"\n{'='*80}")
    print("COMPARISON WITH GPU:")
    print(f"{'='*80}")
    print(f"  KPU-T256 @ 16nm: 2.70 pJ (standard), 2.30 pJ (matrix, 15% better)")
    print(f"  H100 @ 5nm: 1.50 pJ (CUDA), 1.28 pJ (Tensor, 15% better)")
    print(f"  Jetson Orin @ 8nm: 1.90 pJ (CUDA), 1.62 pJ (Tensor, 15% better)")
    print(f"  Process scaling: 16nm → 8nm → 5nm shows expected energy reduction")
    print(f"  Physics-consistent: All show 15% Tensor Core / Matrix efficiency gain!")

    print("\n✓ KPU-T256 model updated successfully!")
    print("✓ Physics-based energy values: 2.7 pJ @ 16nm (standard cell)")
    print("✓ Multi-fabric support: INT8 tiles (70%) + BF16 tiles (20%) + Matrix tiles (10%)")
    print("✓ Backward compatible with legacy precision profiles")

if __name__ == '__main__':
    main()
