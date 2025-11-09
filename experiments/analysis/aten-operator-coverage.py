import torch
from torch.fx.passes.shape_prop import ShapeProp
from torchvision import models
from transformers import ViTModel, ViTConfig
from graphs.transform.partitioning.fusion_partitioner import FusionBasedPartitioner

print('='*80)
print('COMPREHENSIVE ATEN OPERATOR COVERAGE VALIDATION')
print('CNNs + Transformers + Segmentation Models')
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
    'name': 'MobileNetV2 (CNN)',
    'model': lambda: models.mobilenet_v2(weights=None),
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

# Segmentation models
test_cases.append({
    'name': 'DeepLabV3-ResNet50 (Segmentation)',
    'model': lambda: models.segmentation.deeplabv3_resnet50(weights=None),
    'input': torch.randn(1, 3, 224, 224),
    'category': 'Segmentation'
})

test_cases.append({
    'name': 'FCN-ResNet50 (Segmentation)',
    'model': lambda: models.segmentation.fcn_resnet50(weights=None),
    'input': torch.randn(1, 3, 224, 224),
    'category': 'Segmentation'
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

        our_flops = partition_report.total_flops
        our_macs = partition_report.total_macs

        print(f'  Total FLOPs: {our_flops / 1e9:.6f} GFLOPs')
        print(f'  Total MACs:  {our_macs / 1e9:.6f} GMACs')
        print(f'  Subgraphs:   {len(partition_report.subgraphs)}')

        # Arithmetic intensity
        ai = our_flops / max(1, partition_report.total_memory_traffic)
        print(f'  Arithmetic Intensity: {ai:.2f} FLOPs/Byte')

        status = '✅ SUCCESS'

        results.append({
            'name': test_case['name'],
            'category': test_case['category'],
            'flops_gf': our_flops / 1e9,
            'status': status
        })

    except Exception as e:
        print(f'  ❌ ERROR: {e}')
        results.append({
            'name': test_case['name'],
            'category': test_case['category'],
            'flops_gf': 0,
            'status': f'❌ ERROR'
        })

# Summary
print('\n' + '='*80)
print('SUMMARY')
print('='*80)

categories = {}
for r in results:
    cat = r['category']
    if cat not in categories:
        categories[cat] = []
    categories[cat].append(r)

for cat, cat_results in categories.items():
    print(f'\n{cat} ({len(cat_results)} models):')
    for r in cat_results:
        print(f'  {r["name"]:40s} {r["flops_gf"]:8.3f} GFLOPs  {r["status"]}')

all_success = all(r['status'] == '✅ SUCCESS' for r in results)
print(f'\n{"="*80}')
print(f'All models processed: {"✅ YES" if all_success else "❌ NO"}')
print(f'Total models: {len(results)}')
print('='*80)
