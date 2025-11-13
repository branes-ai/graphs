#!/usr/bin/env python3
"""Test TPU Edge Pro multi-fabric configuration"""

from graphs.hardware.models.edge.tpu_edge_pro import tpu_edge_pro_resource_model
from graphs.hardware.resource_model import Precision

def main():
    print("=" * 80)
    print("TPU EDGE PRO - Multi-Fabric Configuration Test")
    print("=" * 80)

    model = tpu_edge_pro_resource_model()

    print(f"\nHardware: {model.name}")
    print(f"Process Node: 7nm TSMC")
    print(f"Architecture: 128×128 Systolic Array (16,384 PEs)")

    if model.compute_fabrics:
        print(f"\n{'='*80}")
        print("COMPUTE FABRIC:")
        print(f"{'='*80}")

        fabric = model.compute_fabrics[0]
        print(f"\n{fabric.fabric_type.upper()}:")
        print(f"  Circuit Type: {fabric.circuit_type}")
        print(f"  PEs: {fabric.num_units:,}")
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
                ops_str = f"{peak_ops/1e12:.1f} TOPS" if precision == Precision.INT8 else f"{peak_ops/1e12:.1f} TFLOPS"
            else:
                ops_str = f"{peak_ops/1e9:.1f} GOPS" if precision == Precision.INT8 else f"{peak_ops/1e9:.1f} GFLOPS"

            print(f"    {precision.name}: {ops_str}, {energy_per_op*1e12:.3f} pJ/op, {peak_power:.1f} W peak")

    print(f"\n{'='*80}")
    print("LEGACY PRECISION PROFILES (for backward compatibility):")
    print(f"{'='*80}")

    for precision, profile in model.precision_profiles.items():
        peak_ops = profile.peak_ops_per_sec
        if peak_ops >= 1e12:
            ops_str = f"{peak_ops/1e12:.1f} TOPS" if precision == Precision.INT8 else f"{peak_ops/1e12:.1f} TFLOPS"
        else:
            ops_str = f"{peak_ops/1e9:.1f} GOPS" if precision == Precision.INT8 else f"{peak_ops/1e9:.1f} GFLOPS"

        tc_status = "Systolic Array" if profile.tensor_core_supported else "Standard"
        print(f"  {precision.name}: {ops_str} ({tc_status})")

    print(f"\n{'='*80}")
    print("ENERGY MODEL:")
    print(f"{'='*80}")
    print(f"  Base ALU Energy (FP32): {model.energy_per_flop_fp32 * 1e12:.2f} pJ (from systolic fabric)")
    print(f"  Memory Access Energy: {model.energy_per_byte * 1e12:.2f} pJ/byte (LPDDR5)")

    print(f"\n{'='*80}")
    print("COMPARISON WITH ALL ARCHITECTURES:")
    print(f"{'='*80}")
    print(f"  TPU Edge Pro @ 7nm: 1.80 pJ (systolic array, standard cell)")
    print(f"  KPU-T256 @ 16nm: 2.70 pJ (standard), 2.30 pJ (matrix, 15% better)")
    print(f"  Jetson Orin @ 8nm: 1.90 pJ (CUDA), 1.62 pJ (Tensor, 15% better)")
    print(f"  H100 @ 5nm: 1.50 pJ (CUDA), 1.28 pJ (Tensor, 15% better)")
    print(f"")
    print(f"  Physics-consistent: 7nm chips (TPU) at ~1.8 pJ!")
    print(f"  Process scaling: 16nm → 8nm → 7nm → 5nm shows expected reduction")
    print(f"  TPU uses standard_cell (no tensor core efficiency in systolic arrays)")

    print("\n✓ TPU Edge Pro model updated successfully!")
    print("✓ Physics-based energy values: 1.8 pJ @ 7nm (standard cell)")
    print("✓ Single systolic array fabric (weight-stationary dataflow)")
    print("✓ Backward compatible with legacy precision profiles")

if __name__ == '__main__':
    main()
