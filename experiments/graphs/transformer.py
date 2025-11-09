import torch
from torch.fx.passes.shape_prop import ShapeProp
from transformers import ViTModel, ViTConfig
from graphs.transform.partitioning.fusion_partitioner import FusionBasedPartitioner
from fvcore.nn import FlopCountAnalysis

print('='*80)
print('TRANSFORMER VALIDATION - Vision Transformer (ViT)')
print('='*80)

# Create a smaller ViT for testing (ViT-Tiny)
config = ViTConfig(
    hidden_size=192,
    num_hidden_layers=12,
    num_attention_heads=3,
    intermediate_size=768,
    image_size=224,
    patch_size=16,
)

model = ViTModel(config)
model.eval()

# Input: batch=1, 3 channels, 224x224
input_tensor = torch.randn(1, 3, 224, 224)

# Warm-up
with torch.no_grad():
    _ = model(input_tensor)

print('\nModel: ViT-Tiny (12 layers, 192 hidden dim)')
print('Input: 1x3x224x224')

# Dynamo export
try:
    exported_program = torch.export.export(model, (input_tensor,))
    fx_graph = exported_program.module()

    # Shape propagation
    shape_prop = ShapeProp(fx_graph)
    shape_prop.propagate(input_tensor)

    # Our partitioner
    partitioner = FusionBasedPartitioner()
    partition_report = partitioner.partition(fx_graph)

    our_macs = partition_report.total_macs
    our_flops = partition_report.total_flops

    print(f'\nOur calculation:')
    print(f'  Total FLOPs: {our_flops / 1e9:.6f} GFLOPs')
    print(f'  Total MACs:  {our_macs / 1e9:.6f} GMACs')
    print(f'  Subgraphs:   {len(partition_report.subgraphs)}')

    # Calculate arithmetic intensity
    ai = partition_report.total_flops / max(1, partition_report.total_memory_traffic)
    print(f'  Arithmetic Intensity: {ai:.2f} FLOPs/Byte')

    # FVCore comparison
    try:
        flop_counter = FlopCountAnalysis(model, input_tensor)
        fvcore_macs = flop_counter.total()

        diff_pct = abs(our_macs - fvcore_macs) / fvcore_macs * 100

        print(f'\nFVCore:')
        print(f'  Total MACs: {fvcore_macs / 1e9:.6f} GMACs')

        print(f'\nComparison:')
        print(f'  Difference: {diff_pct:.2f}%')

        if diff_pct < 5:
            print(f'  ✅ PASS - Within 5%')
        elif diff_pct < 15:
            print(f'  ⚠️  ACCEPTABLE - Within 15%')
        else:
            print(f'  ❌ FAIL - Differs by more than 15%')

    except Exception as e:
        print(f'\n⚠️  FVCore failed: {e}')

    # Analyze operation types
    print('\n' + '='*80)
    print('Operation Analysis')
    print('='*80)

    op_types = {}
    for node in fx_graph.graph.nodes:
        if node.op == 'call_function':
            target_str = str(node.target)
            # Simplify target name
            if 'matmul' in target_str or 'bmm' in target_str:
                key = 'matmul/bmm'
            elif 'linear' in target_str or 'addmm' in target_str:
                key = 'linear'
            elif 'softmax' in target_str:
                key = 'softmax'
            elif 'layer_norm' in target_str:
                key = 'layer_norm'
            elif 'gelu' in target_str:
                key = 'gelu'
            elif any(x in target_str for x in ['add', 'mul', 'div', 'sub']):
                key = 'elementwise'
            else:
                key = 'other'

            op_types[key] = op_types.get(key, 0) + 1

    print('\nOperation type counts:')
    for op, count in sorted(op_types.items(), key=lambda x: x[1], reverse=True):
        print(f'  {op:20s}: {count:4d}')

    # Top FLOP consumers
    print('\n' + '='*80)
    print('Top 10 Subgraphs by FLOPs')
    print('='*80)

    sorted_subgraphs = sorted(partition_report.subgraphs, key=lambda sg: sg.total_flops, reverse=True)
    for i, sg in enumerate(sorted_subgraphs[:10]):
        print(f'  {i+1:2d}. {sg.total_flops / 1e9:8.4f} GFLOPs')

except Exception as e:
    print(f'\n❌ Error: {e}')
    import traceback
    traceback.print_exc()

print('\n' + '='*80)
