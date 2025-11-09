from graphs.analysis.unified_analyzer import UnifiedAnalyzer
from graphs.hardware.resource_model import Precision
import time

print('='*80)
print('AUTOMOTIVE HARDWARE COMPARISON')
print('Segmentation + Vision Workloads')
print('='*80)

# Define hardware platforms
automotive_hardware = [
    # NVIDIA Automotive
    ('Jetson-Orin-AGX', 'NVIDIA Automotive'),
    ('Jetson-Thor', 'NVIDIA Next-Gen'),

    # Qualcomm Automotive
    ('Snapdragon-Ride', 'Qualcomm Premium'),
    ('SA8775P', 'Qualcomm SoC'),

    # TI ADAS
    ('TDA4VH', 'TI High-End ADAS'),
    ('TDA4VM', 'TI Mid-Range ADAS'),

    # Stillwater KPU
    ('KPU-T256', 'Stillwater Edge AI'),
    ('KPU-T768', 'Stillwater High-Perf'),

    # Hailo Edge AI
    ('Hailo-10H', 'Hailo High-Perf'),
    ('Hailo-8', 'Hailo Edge'),
]

# Define workloads
workloads = [
    ('resnet50', 'ResNet50 (Classification)', 'FP32'),
    ('deeplabv3_resnet50', 'DeepLabV3 (Segmentation)', 'FP32'),
]

analyzer = UnifiedAnalyzer(verbose=False)

results = []

for model_name, model_desc, precision_str in workloads:
    precision = Precision.FP32 if precision_str == 'FP32' else Precision.FP16

    print(f'\n{"="*80}')
    print(f'Model: {model_desc}')
    print(f'Precision: {precision_str}')
    print(f'{"="*80}')

    for hw_name, hw_category in automotive_hardware:
        try:
            print(f'\n  Analyzing {hw_name}...', end=' ', flush=True)

            start = time.time()
            result = analyzer.analyze_model(
                model_name=model_name,
                hardware_name=hw_name,
                batch_size=1,
                precision=precision
            )
            elapsed = time.time() - start

            results.append({
                'model': model_desc,
                'hardware': hw_name,
                'category': hw_category,
                'latency_ms': result.total_latency_ms,
                'energy_mj': result.total_energy_mj,
                'throughput_fps': result.throughput_fps,
                'memory_mb': result.peak_memory_mb,
                'flops_gf': result.partition_report.total_flops / 1e9,
            })

            print(f'✓ ({elapsed:.1f}s)')

        except Exception as e:
            print(f'✗ Error: {str(e)[:50]}')
            results.append({
                'model': model_desc,
                'hardware': hw_name,
                'category': hw_category,
                'latency_ms': -1,
                'energy_mj': -1,
                'throughput_fps': -1,
                'memory_mb': -1,
                'flops_gf': -1,
            })

print(f'\n{"="*80}')
print('RESULTS SUMMARY')
print(f'{"="*80}')

# Group by model
for model_desc in ['ResNet50 (Classification)', 'DeepLabV3 (Segmentation)']:
    model_results = [r for r in results if r['model'] == model_desc]

    print(f'\n{model_desc}:')
    print('-' * 80)
    print(f'{"Hardware":<25} {"Category":<25} {"Latency":>10} {"Energy":>10} {"FPS":>8}')
    print('-' * 80)

    for r in model_results:
        if r['latency_ms'] > 0:
            print(f'{r["hardware"]:<25} {r["category"]:<25} '
                    f'{r["latency_ms"]:>9.2f}ms {r["energy_mj"]:>9.1f}mJ {r["throughput_fps"]:>7.1f}')
        else:
            print(f'{r["hardware"]:<25} {r["category"]:<25} {"ERROR":>10}')

# Find best performers
print(f'\n{"="*80}')
print('BEST PERFORMERS')
print(f'{"="*80}')

for model_desc in ['ResNet50 (Classification)', 'DeepLabV3 (Segmentation)']:
    model_results = [r for r in results if r['model'] == model_desc and r['latency_ms'] > 0]

    if not model_results:
        continue

    print(f'\n{model_desc}:')

    # Best latency
    best_latency = min(model_results, key=lambda x: x['latency_ms'])
    print(f'  Lowest Latency:  {best_latency["hardware"]:<20} {best_latency["latency_ms"]:>8.2f}ms')

    # Best energy
    best_energy = min(model_results, key=lambda x: x['energy_mj'])
    print(f'  Lowest Energy:   {best_energy["hardware"]:<20} {best_energy["energy_mj"]:>8.1f}mJ')

    # Best throughput
    best_fps = max(model_results, key=lambda x: x['throughput_fps'])
    print(f'  Highest FPS:     {best_fps["hardware"]:<20} {best_fps["throughput_fps"]:>8.1f} FPS')

print(f'\n{"="*80}')