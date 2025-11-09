import torch
from transformers import ViTModel, ViTConfig
from torch.fx.passes.shape_prop import ShapeProp

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

with torch.no_grad():
    _ = model(input_tensor)

exported_program = torch.export.export(model, (input_tensor,))
fx_graph = exported_program.module()

print('='*80)
print('Investigating scaled_dot_product_attention')
print('='*80)

# Find all SDPA nodes
sdpa_nodes = []
linear_nodes = []
matmul_nodes = []

for node in fx_graph.graph.nodes:
    if node.op == 'call_function':
        target_str = str(node.target)
        if 'scaled_dot_product_attention' in target_str:
            sdpa_nodes.append(node)
        elif 'linear' in target_str:
            linear_nodes.append(node)
        elif 'matmul' in target_str or 'bmm' in target_str:
            matmul_nodes.append(node)

print(f'\nOperation counts:')
print(f'  scaled_dot_product_attention: {len(sdpa_nodes)}')
print(f'  linear operations: {len(linear_nodes)}')
print(f'  matmul/bmm operations: {len(matmul_nodes)}')

print(f'\nLinear operations breakdown:')
# Count linear ops in ViT
# Each transformer block has:
# - 3 linear ops for QKV projection
# - 1 linear op for output projection
# - 2 linear ops for FFN (fc1, fc2)
# Total: 6 linear ops per block

expected_linear = 12 * 6  # 12 layers × 6 linear per layer
print(f'  Expected (12 layers × 6): {expected_linear}')
print(f'  Found: {len(linear_nodes)}')

# Check if SDPA is being decomposed
print(f'\n' + '='*80)
print('HYPOTHESIS: SDPA may be getting decomposed into matmul ops')
print('='*80)

# Theoretical FLOP calculation for attention
# SDPA does: QK^T (matmul), softmax, @V (matmul)
# Q, K, V shapes: [batch, num_heads, seq_len, head_dim]

batch = 1
seq_len = (224 // 16) ** 2 + 1  # patches + cls token = 197
num_heads = 3
head_dim = 192 // 3  # 64
num_layers = 12

# Per attention layer:
# QK^T: [B, H, S, D] @ [B, H, D, S] = [B, H, S, S]
#   FLOPs = B * H * S * S * D * 2
qkt_flops = batch * num_heads * seq_len * seq_len * head_dim * 2

# Softmax: ~5 ops per element (exp, sum, div, ...)
softmax_flops = batch * num_heads * seq_len * seq_len * 5

# Attn @ V: [B, H, S, S] @ [B, H, S, D] = [B, H, S, D]
#   FLOPs = B * H * S * D * S * 2
attn_v_flops = batch * num_heads * seq_len * head_dim * seq_len * 2

total_attention_flops_per_layer = qkt_flops + softmax_flops + attn_v_flops
total_attention_flops = total_attention_flops_per_layer * num_layers

print(f'\nTheoretical Attention FLOPs (all 12 layers):')
print(f'  QK^T:    {qkt_flops * num_layers / 1e9:.6f} GFLOPs')
print(f'  Softmax: {softmax_flops * num_layers / 1e9:.6f} GFLOPs')
print(f'  Attn@V:  {attn_v_flops * num_layers / 1e9:.6f} GFLOPs')
print(f'  TOTAL:   {total_attention_flops / 1e9:.6f} GFLOPs')

# Compare with linear layer FLOPs
# QKV projection: 3 × (seq_len × hidden × hidden) × 2
# Output proj: 1 × (seq_len × hidden × hidden) × 2
# FFN: 2 × (seq_len × hidden × intermediate + seq_len × intermediate × hidden) × 2

hidden = 192
intermediate = 768

qkv_flops = 3 * seq_len * hidden * hidden * 2
out_proj_flops = seq_len * hidden * hidden * 2
ffn_flops = 2 * (seq_len * hidden * intermediate * 2)

total_linear_flops_per_layer = qkv_flops + out_proj_flops + ffn_flops
total_linear_flops = total_linear_flops_per_layer * num_layers

print(f'\nLinear Layer FLOPs (all 12 layers):')
print(f'  QKV proj:    {qkv_flops * num_layers / 1e9:.6f} GFLOPs')
print(f'  Output proj: {out_proj_flops * num_layers / 1e9:.6f} GFLOPs')
print(f'  FFN:         {ffn_flops * num_layers / 1e9:.6f} GFLOPs')
print(f'  TOTAL:       {total_linear_flops / 1e9:.6f} GFLOPs')

print(f'\nRatio:')
print(f'  Attention FLOPs / Linear FLOPs = {total_attention_flops / total_linear_flops:.2f}')

print(f'\n' + '='*80)
print('CONCLUSION')
print('='*80)
print(f'''
If SDPA is being decomposed by Dynamo into matmul operations,
then our linear/matmul handlers ARE counting attention FLOPs.

If SDPA remains fused (not decomposed), we're missing ~{total_attention_flops / 1e9:.2f} GFLOPs.

Our FVCore accuracy of 0.44% suggests SDPA is likely decomposed.
Let's verify by checking if we see matmul nodes in the graph.
''')

if matmul_nodes:
    print(f'✅ Found {len(matmul_nodes)} matmul/bmm nodes - SDPA likely decomposed')
else:
    print(f'❌ No matmul nodes found - SDPA may be fused (missing FLOPs!)')
