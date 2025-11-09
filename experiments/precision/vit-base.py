from graphs.analysis.unified_analyzer import UnifiedAnalyzer
from graphs.hardware.resource_model import Precision

analyzer = UnifiedAnalyzer(verbose=False)

print('='*80)
print('TRANSFORMER PRECISION TEST - ViT-Base')
print('='*80)

# Test with a standard ViT-Base model
print('\nTesting FP32...')
fp32_result = analyzer.analyze_model('vit_b_16', 'H100', batch_size=1, precision=Precision.FP32)
print(f'FP32:')
print(f'  Latency: {fp32_result.total_latency_ms:.6f} ms')
print(f'  Energy:  {fp32_result.total_energy_mj:.6f} mJ')
print(f'  FLOPs:   {fp32_result.partition_report.total_flops / 1e9:.3f} GFLOPs')

print('\nTesting FP16...')
fp16_result = analyzer.analyze_model('vit_b_16', 'H100', batch_size=1, precision=Precision.FP16)
print(f'FP16:')
print(f'  Latency: {fp16_result.total_latency_ms:.6f} ms')
print(f'  Energy:  {fp16_result.total_energy_mj:.6f} mJ')
print(f'  FLOPs:   {fp16_result.partition_report.total_flops / 1e9:.3f} GFLOPs')

print('\n' + '='*80)
print('RESULTS')
print('='*80)

latency_speedup = fp32_result.total_latency_ms / fp16_result.total_latency_ms
energy_savings = fp32_result.total_energy_mj / fp16_result.total_energy_mj

print(f'\nLatency speedup (FP32/FP16): {latency_speedup:.2f}×')
print(f'Energy reduction (FP32/FP16): {energy_savings:.2f}×')

# Calculate arithmetic intensity
ai = fp32_result.partition_report.total_flops / max(1, fp32_result.partition_report.total_memory_traffic)
print(f'\nArithmetic Intensity: {ai:.2f} FLOPs/Byte')

# H100 ridge point
fp32_ridge = 60e12 / 2e12  # 30 FLOPs/Byte
fp16_ridge = 750e12 / 2e12  # 375 FLOPs/Byte

if ai > fp32_ridge:
    print(f'  Compute-bound (AI={ai:.1f} > {fp32_ridge:.1f})')
    print(f'  → Expect significant FP16 speedup')
else:
    print(f'  Memory-bound (AI={ai:.1f} < {fp32_ridge:.1f})')
    print(f'  → FP16 speedup limited by bandwidth')

if latency_speedup > 1.05:
    print(f'\n✅ PASS - FP16 shows {latency_speedup:.2f}× speedup')
else:
    print(f'\n⚠️  Modest speedup - model may be memory-bound at batch=1')
