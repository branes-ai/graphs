from graphs.hardware.soc_infrastructure import (
    SoCInfrastructureModel,
    InterconnectTopology,
)

print('='*100)
print('Architecture Comparison (16K ALUs, 5nm, mesh_2d)')
print('='*100)
print()
print(f"{'Circuit':<18} {'Power%':>8} {'Area%':>8} {'TDP':>8} {'Clusters':>10} {'Description':<30}")
print('-'*100)

circuits = [
    ('x86_performance', 'High-perf OoO CPU core'),
    ('tensor_core', 'NVIDIA Tensor Core (GPU)'),
    ('systolic_mac', 'TPU-style systolic array'),
    ('domain_flow', 'KPU domain flow accelerator'),
]

for circuit, desc in circuits:
    model = SoCInfrastructureModel(
        num_alus=16384,
        process_node_nm=5,
        circuit_type=circuit,
        topology=InterconnectTopology.MESH_2D,
    )
    result = model.compute_power_breakdown(frequency_ghz=1.5, precision='FP32')
    print(f'{circuit:<18} {result.compute_fraction*100:>7.1f}% {result.compute_area_fraction*100:>7.1f}% '
            f'{result.total_tdp_w:>7.0f}W {model.num_clusters:>10} {desc:<30}')

print()
print('Analysis:')
print('  - Power Fraction: Shows ALU switching power as % of TDP')
print('  - Area Fraction:  Shows compute silicon as % of die area')
print('  - Accelerators (systolic, domain_flow) have HIGHER area efficiency')
print('  - Accelerators also have LOWER total TDP (more efficient per op)')