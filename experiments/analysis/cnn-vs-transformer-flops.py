import torch
from torch.fx.passes.shape_prop import ShapeProp
from torchvision import models
from transformers import ViTModel, ViTConfig
from graphs.transform.partitioning.fusion_partitioner import FusionBasedPartitioner
from fvcore.nn import FlopCountAnalysis

print('='*80)
print('COMPREHENSIVE FLOP VALIDATION')
print('Architecture Coverage: CNNs + Transformers')
print('='*80)

test_cases = []

# CNNs
test_cases.append({
    'name': 'ResNet18 (CNN)',
    'model': lambda: models.resnet18(weights=None),
    'input': torch.randn(1, 3, 224, 224),
    'category': 'CNN'
})

test_cases.append({
    'name': 'ResNet50 (CNN)',
    'model': lambda: models.resnet50(weights=None),
    'input': torch.randn(1, 3, 224, 224),
    'category': 'CNN'
})

test_cases.append({
    'name': 'MobileNetV2 (CNN)',
    'model': lambda: models.mobilenet_v2(weights=None),
    'input': torch.randn(1, 3, 224, 224),
    'category': 'CNN'
})

test_cases.append({
    'name': 'EfficientNet-B0 (CNN)',
    'model': lambda: models.efficientnet_b0(weights=None),
    'input': torch.randn(1, 3, 224, 224),
    'category': 'CNN'
})

# Transformers
test_cases.append({
    'name': 'ViT-Tiny (Transformer)',
    'model': lambda: ViTModel(ViTConfig(
        hidden_size=192,
        num_hidden_layers=12,
        num_attention_heads=3,
        intermediate_size=768,
        image_size=224,
        patch_size=16,
    )),
    'input': torch.randn(1, 3, 224, 224),
    'category': 'Transformer'
})

test_cases.append({
    'name': 'ViT-Small (Transformer)',
    'model': lambda: ViTModel(ViTConfig(
        hidden_size=384,
        num_hidden_layers=12,
        num_attention_heads=6,
        intermediate_size=1536,
        image_size=224,
        patch_size=16,
    )),
    'input': torch.randn(1, 3, 224, 224),
    'category': 'Transformer'
})

results = []

for test_case in test_cases:
    print(f'\n{test_case["name"]}:')
    print('-' * 60)

    try:
        model = test_case['model']()
        model.eval()
        input_tensor = test_case['input']

        # Warm-up
        with torch.no_grad():
            _ = model(input_tensor)

        # Dynamo export
        exported_program = torch.export.export(model, (input_tensor,))
        fx_graph = exported_program.module()

        # Shape propagation
        shape_prop = ShapeProp(fx_graph)
        shape_prop.propagate(input_tensor)

        # Our partitioner
        partitioner = FusionBasedPartitioner()
        partition_report = partitioner.partition(fx_graph)

        our_macs = partition_report.total_macs

        # FVCore
        flop_counter = FlopCountAnalysis(model, input_tensor)
        fvcore_macs = flop_counter.total()

        diff_pct = abs(our_macs - fvcore_macs) / fvcore_macs * 100

        status = '✅ PASS' if diff_pct < 5 else ('⚠️  ACCEPT' if diff_pct < 15 else '❌ FAIL')

        print(f'  Our MACs:    {our_macs / 1e9:.6f} GMACs')
        print(f'  FVCore MACs: {fvcore_macs / 1e9:.6f} GMACs')
        print(f'  Difference:  {diff_pct:.2f}%')
        print(f'  Status:      {status}')

        results.append({
            'name': test_case['name'],
            'category': test_case['category'],
            'diff_pct': diff_pct,
            'status': status
        })

    except Exception as e:
        print(f'  ❌ ERROR: {e}')
        results.append({
            'name': test_case['name'],
            'category': test_case['category'],
            'diff_pct': float('inf'),
            'status': '❌ ERROR'
        })

# Summary
print('\n' + '='*80)
print('SUMMARY')
print('='*80)

cnn_results = [r for r in results if r['category'] == 'CNN']
transformer_results = [r for r in results if r['category'] == 'Transformer']

print(f'\nCNNs ({len(cnn_results)} models):')
for r in cnn_results:
    print(f'  {r["name"]:30s} {r["diff_pct"]:6.2f}% {r["status"]}')

print(f'\nTransformers ({len(transformer_results)} models):')
for r in transformer_results:
    print(f'  {r["name"]:30s} {r["diff_pct"]:6.2f}% {r["status"]}')

# Overall stats
all_diff = [r['diff_pct'] for r in results if r['diff_pct'] != float('inf')]
if all_diff:
    avg_diff = sum(all_diff) / len(all_diff)
    max_diff = max(all_diff)

    print(f'\nOverall Statistics:')
    print(f'  Average difference: {avg_diff:.2f}%')
    print(f'  Maximum difference: {max_diff:.2f}%')
    print(f'  Models tested: {len(results)}')
    print(f'  All passed: {"✅ YES" if all(r["diff_pct"] < 15 for r in results) else "❌ NO"}')

print('\n' + '='*80)
