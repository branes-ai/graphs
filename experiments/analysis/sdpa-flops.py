import torch
from torch.fx.passes.shape_prop import ShapeProp
from transformers import ViTModel, ViTConfig
from graphs.transform.partitioning.fusion_partitioner import FusionBasedPartitioner

print('='*80)
print('TESTING SDPA FLOP CALCULATION')
print('='*80)

# Create ViT-Tiny
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
input_tensor = torch.randn(1, 3, 224, 224)

# Warm-up
with torch.no_grad():
    _ = model(input_tensor)

# Dynamo export
exported_program = torch.export.export(model, (input_tensor,))
fx_graph = exported_program.module()

# Shape propagation
shape_prop = ShapeProp(fx_graph)
shape_prop.propagate(input_tensor)

# Partition
partitioner = FusionBasedPartitioner()
partition_report = partitioner.partition(fx_graph)

print(f'\nWith SDPA support:')
print(f'  Total FLOPs: {partition_report.total_flops / 1e9:.6f} GFLOPs')
print(f'  Total MACs:  {partition_report.total_macs / 1e9:.6f} GMACs')

# Calculate theoretical attention FLOPs
batch = 1
seq_len = (224 // 16) ** 2 + 1  # 197
num_heads = 3
head_dim = 192 // 3  # 64
num_layers = 12

qkt_macs = batch * num_heads * seq_len * seq_len * head_dim
attn_v_macs = batch * num_heads * seq_len * seq_len * head_dim
total_attention_macs = (qkt_macs + attn_v_macs) * num_layers

print(f'\nTheoretical attention MACs (12 layers): {total_attention_macs / 1e9:.6f} GMACs')
print(f'Linear layer MACs (from before): 1.046 GMACs')
print(f'Expected total MACs: ~{(total_attention_macs + 1.046e9) / 1e9:.3f} GMACs')

# FVCore won't have attention, so our total should be higher now
from fvcore.nn import FlopCountAnalysis
flop_counter = FlopCountAnalysis(model, input_tensor)
fvcore_macs = flop_counter.total()

print(f'\nComparison:')
print(f'  Our MACs (with SDPA):    {partition_report.total_macs / 1e9:.6f} GMACs')
print(f'  FVCore MACs (no SDPA):   {fvcore_macs / 1e9:.6f} GMACs')
print(f'  Difference (our - fvcore): {(partition_report.total_macs - fvcore_macs) / 1e9:.6f} GMACs')
print(f'  Expected difference (attention): {total_attention_macs / 1e9:.6f} GMACs')

diff = abs((partition_report.total_macs - fvcore_macs) - total_attention_macs) / total_attention_macs * 100
print(f'\nSDPA calculation accuracy: {diff:.2f}% difference from theoretical')

if diff < 5:
    print('✅ PASS - SDPA FLOPs calculated correctly')
else:
    print('❌ FAIL - SDPA calculation may be incorrect')
