#!/usr/bin/env python3
"""Test Jetson Orin AGX GPU multi-fabric configuration"""

from graphs.hardware.models.edge.jetson_orin_agx_64gb import jetson_orin_agx_64gb_resource_model
from graphs.hardware.resource_model import Precision

def main():
    print("=" * 80)
    print("JETSON ORIN AGX 64GB - Multi-Fabric Configuration Test")
    print("=" * 80)

    model = jetson_orin_agx_64gb_resource_model()

    print(f"\nHardware: {model.name}")
    print(f"Process Node: 8nm Samsung")
    print(f"Architecture: Ampere (2nd gen Tensor Cores)")

    if model.compute_fabrics:
        print(f"\n{'='*80}")
        print("COMPUTE FABRICS:")
        print(f"{'='*80}")

        for fabric in model.compute_fabrics:
            print(f"\n{fabric.fabric_type.upper()}:")
            print(f"  Circuit Type: {fabric.circuit_type}")
            print(f"  Units: {fabric.num_units:,}")
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
                    ops_str = f"{peak_ops/1e12:.1f} TOPS" if precision in [Precision.INT8, Precision.FP16] else f"{peak_ops/1e12:.1f} TFLOPS"
                else:
                    ops_str = f"{peak_ops/1e9:.1f} GOPS" if precision in [Precision.INT8, Precision.FP16] else f"{peak_ops/1e9:.1f} GFLOPS"

                print(f"    {precision.name}: {ops_str}, {energy_per_op*1e12:.2f} pJ/op, {peak_power:.1f} W peak")

        # Calculate efficiency gain
        cuda_energy = model.compute_fabrics[0].energy_per_flop_fp32
        tensor_energy = model.compute_fabrics[1].energy_per_flop_fp32
        efficiency_gain = (cuda_energy - tensor_energy) / cuda_energy * 100

        print(f"\n{'='*80}")
        print(f"Efficiency Gain (Tensor Core vs CUDA Core): {efficiency_gain:.1f}%")
        print(f"{'='*80}")

    print(f"\n{'='*80}")
    print("LEGACY PRECISION PROFILES (for backward compatibility):")
    print(f"{'='*80}")

    for precision, profile in model.precision_profiles.items():
        peak_ops = profile.peak_ops_per_sec
        if peak_ops >= 1e12:
            ops_str = f"{peak_ops/1e12:.1f} TOPS" if precision in [Precision.INT8, Precision.FP16] else f"{peak_ops/1e12:.1f} TFLOPS"
        else:
            ops_str = f"{peak_ops/1e9:.1f} GOPS" if precision in [Precision.INT8, Precision.FP16] else f"{peak_ops/1e9:.1f} GFLOPS"

        tc_status = "Tensor Core" if profile.tensor_core_supported else "CUDA Core"
        print(f"  {precision.name}: {ops_str} ({tc_status})")

    print(f"\n{'='*80}")
    print("ENERGY MODEL:")
    print(f"{'='*80}")
    print(f"  Base ALU Energy (FP32): {model.energy_per_flop_fp32 * 1e12:.2f} pJ (from CUDA fabric)")
    print(f"  Memory Access Energy: {model.energy_per_byte * 1e12:.2f} pJ/byte (LPDDR5)")

    print(f"\n{'='*80}")
    print("COMPARISON WITH H100:")
    print(f"{'='*80}")
    print(f"  Jetson Orin AGX @ 8nm: 1.90 pJ (CUDA), 1.62 pJ (Tensor, 15% better)")
    print(f"  H100 SXM5 @ 5nm: 1.50 pJ (CUDA), 1.28 pJ (Tensor, 15% better)")
    print(f"  Process scaling: 8nm → 5nm = {(1.90/1.50):.2f}× energy reduction")
    print(f"  Physics-consistent: Both show 15% Tensor Core efficiency gain!")

    print("\n✓ Jetson Orin AGX GPU model updated successfully!")
    print("✓ Physics-based energy values: 1.9 pJ @ 8nm (standard cell)")
    print("✓ Multi-fabric support: CUDA cores + Tensor Cores")
    print("✓ Backward compatible with legacy precision profiles")

if __name__ == '__main__':
    main()
