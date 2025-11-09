from transformers import ViTModel, ViTConfig
import torch
from fvcore.nn import FlopCountAnalysis

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

flop_counter = FlopCountAnalysis(model, input_tensor)

print('='*80)
print('FVCore Operation Breakdown')
print('='*80)

# Get per-operator breakdown
by_operator = flop_counter.by_operator()
total = flop_counter.total()

print(f'\nTotal FLOPs (FVCore reports as MACs): {total / 1e9:.6f} GMACs')
print(f'\nPer-operator breakdown:')

for op, count in sorted(by_operator.items(), key=lambda x: x[1], reverse=True):
    pct = (count / total * 100) if total > 0 else 0
    print(f'  {op:40s}: {count / 1e9:8.4f} GMACs ({pct:5.1f}%)')

print(f'\n' + '='*80)
print('Analysis')
print('='*80)

if 'scaled_dot_product_attention' in by_operator or 'aten::scaled_dot_product_attention' in by_operator:
    print('✅ FVCore DOES count scaled_dot_product_attention')
else:
    print('❌ FVCore does NOT count scaled_dot_product_attention')
    print('   → Both our implementation and FVCore miss this operator')
    print('   → This explains why our accuracy is excellent despite missing it')
