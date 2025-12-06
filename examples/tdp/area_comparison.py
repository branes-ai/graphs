from graphs.hardware.soc_infrastructure import (
    SoCInfrastructureModel,
    InterconnectTopology,
    CIRCUIT_AREA_MULTIPLIER,
)

print('='*100)
print('Area vs Power Fraction Comparison (16K ALUs, 5nm, mesh_2d)')
print('='*100)
print(f"{'Circuit':<15} {'Area Mult':>10} {'Compute Area':>12} {'SRAM Area':>10} {'Total Area':>11} {'Area Frac':>10}")
print('-'*100)

circuits = ['x86_performance', 'tensor_core', 'systolic_mac', 'domain_flow']

for circuit in circuits:
    model = SoCInfrastructureModel(
        num_alus=16384,
        process_node_nm=5,
        circuit_type=circuit,
        topology=InterconnectTopology.MESH_2D,
    )
    result = model.compute_power_breakdown(frequency_ghz=1.5, precision='FP32')
    area_mult = CIRCUIT_AREA_MULTIPLIER.get(circuit, 4.0)
    print(f'{circuit:<15} {area_mult:>10.1f}x {result.compute_area_mm2:>10.1f}mm2 {result.sram_area_mm2:>8.1f}mm2 '
            f'{result.total_area_mm2:>9.1f}mm2 {result.compute_area_fraction*100:>9.1f}%')