import torch
from torchvision import models
from torch.fx.passes.shape_prop import ShapeProp
from graphs.transform.partitioning.fusion_partitioner import FusionBasedPartitioner
from graphs.hardware.models.datacenter.h100_pcie import h100_pcie_resource_model
from graphs.hardware.resource_model import Precision
from graphs.analysis.roofline import RooflineAnalyzer

# Create model and partition
model = models.resnet18(weights=None)
input_tensor = torch.randn(1, 3, 224, 224)

model.eval()
with torch.no_grad():
    _ = model(input_tensor)

exported_program = torch.export.export(model, (input_tensor,))
fx_graph = exported_program.module()

shape_prop = ShapeProp(fx_graph)
shape_prop.propagate(input_tensor)

partitioner = FusionBasedPartitioner()
partition_report = partitioner.partition(fx_graph)

# Get hardware
hw = h100_pcie_resource_model()

# Test FP32 roofline
print('='*80)
print('FP32 Roofline Analysis')
print('='*80)

roofline_fp32 = RooflineAnalyzer(hw, precision=Precision.FP32)
report_fp32 = roofline_fp32.analyze(partition_report.subgraphs, partition_report)

total_latency_fp32 = sum(lat.actual_latency for lat in report_fp32.latencies)
print(f'Peak FLOPS: {roofline_fp32.peak_flops / 1e12:.1f} TFLOPS')
print(f'Total latency: {total_latency_fp32 * 1000:.6f} ms')
print(f'Total FLOPs: {partition_report.total_flops / 1e9:.6f} GFLOPs')

# Check first few subgraphs
print(f'\\nFirst 5 subgraph latencies:')
for i, (sg, lat) in enumerate(zip(partition_report.subgraphs[:5], report_fp32.latencies[:5])):
    print(f'  {i}: {sg.total_flops / 1e6:.2f} MFLOPs -> {lat.actual_latency * 1e6:.2f} µs ({lat.bottleneck})')

# Test FP16 roofline
print('\\n' + '='*80)
print('FP16 Roofline Analysis')
print('='*80)

roofline_fp16 = RooflineAnalyzer(hw, precision=Precision.FP16)
report_fp16 = roofline_fp16.analyze(partition_report.subgraphs, partition_report)

total_latency_fp16 = sum(lat.actual_latency for lat in report_fp16.latencies)
print(f'Peak FLOPS: {roofline_fp16.peak_flops / 1e12:.1f} TFLOPS')
print(f'Total latency: {total_latency_fp16 * 1000:.6f} ms')
print(f'Total FLOPs: {partition_report.total_flops / 1e9:.6f} GFLOPs')

# Check first few subgraphs
print(f'\\nFirst 5 subgraph latencies:')
for i, (sg, lat) in enumerate(zip(partition_report.subgraphs[:5], report_fp16.latencies[:5])):
    print(f'  {i}: {sg.total_flops / 1e6:.2f} MFLOPs -> {lat.actual_latency * 1e6:.2f} µs ({lat.bottleneck})')

print('\\n' + '='*80)
print(f'Speedup: {total_latency_fp32 / total_latency_fp16:.2f}×')
print('='*80)
