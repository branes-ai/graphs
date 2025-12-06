from graphs.hardware.soc_infrastructure import (
    estimate_infrastructure_power,
    InterconnectTopology,
    COMPUTE_GRANULARITY_PRESETS,
)

print('='*80)
print('Compute Fraction Comparison (16K ALUs, 5nm, mesh_2d)')
print('='*80)
print(f"{'Circuit Type':<20} {'Granularity':<15} {'Clusters':>10} {'Compute%':>10} {'TDP (W)':>10}")
print('-'*80)

circuit_granularity_map = [
    ('x86_performance', 'cpu_core'),
    ('tensor_core', 'nvidia_tc'),
    ('systolic_mac', 'tpu_mxu'),
    ('domain_flow', 'kpu_tile'),
]

for circuit, granularity in circuit_granularity_map:
    result = estimate_infrastructure_power(
        num_alus=16384,
        process_node_nm=5,
        circuit_type=circuit,
        topology=InterconnectTopology.MESH_2D,
        frequency_ghz=1.5,
        precision='FP32',
    )
    g = COMPUTE_GRANULARITY_PRESETS[granularity]
    num_clusters = max(1, 16384 // g.alus_per_cluster)
    print(f'{circuit:<20} {granularity:<15} {num_clusters:>10} {result.compute_fraction*100:>9.1f}% {result.total_tdp_w:>10.1f}')



# ================================================================================
# Compute Fraction Comparison (16K ALUs, 5nm, mesh_2d)
# ================================================================================
# Circuit Type         Granularity       Clusters   Compute%    TDP (W)
# --------------------------------------------------------------------------------
# x86_performance      cpu_core              2048      27.3%      674.9
# tensor_core          nvidia_tc               64      20.6%      303.9
# systolic_mac         tpu_mxu                  1      23.4%      252.3
# domain_flow          kpu_tile                64      20.1%      275.5
