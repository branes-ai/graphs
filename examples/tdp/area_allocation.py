from graphs.hardware.soc_infrastructure import (
    SoCInfrastructureModel,
    InterconnectTopology,
)

print('='*90)
print('Area vs Power Fraction Comparison (16K ALUs, 5nm, mesh_2d)')
print('='*90)
print(f"{'Circuit':<15} {'Compute Area':>12} {'SRAM Area':>10} {'Area Frac':>10} {'Power Frac':>11}")
print('-'*90)

circuits = ['x86_performance', 'tensor_core', 'systolic_mac', 'domain_flow']

for circuit in circuits:
    model = SoCInfrastructureModel(
        num_alus=16384,
        process_node_nm=5,
        circuit_type=circuit,
        topology=InterconnectTopology.MESH_2D,
    )
    result = model.compute_power_breakdown(frequency_ghz=1.5, precision='FP32')
    print(f'{circuit:<15} {result.compute_area_mm2:>10.2f}mm2 {result.sram_area_mm2:>8.2f}mm2 '
            f'{result.compute_area_fraction*100:>9.1f}% {result.compute_fraction*100:>10.1f}%')
