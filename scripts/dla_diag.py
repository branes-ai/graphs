#!/usr/bin/env python3
"""Diagnostic: check if DLA is actually being used by TRT engine builder."""
import sys, os, json, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'src'))

import tensorrt as trt
import torch
import torch.nn as nn

print(f"TensorRT: {trt.__version__}")
print(f"DLA cores: ", end="")
logger = trt.Logger(trt.Logger.INFO)  # Use INFO to see DLA assignment messages
runtime = trt.Runtime(logger)
print(runtime.num_DLA_cores)

# Simple Conv2D model
class SimpleConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

model = SimpleConv().eval()
dummy = torch.randn(1, 3, 224, 224)

# Export ONNX
onnx_path = tempfile.mktemp(suffix='.onnx')
torch.onnx.export(model, dummy, onnx_path, opset_version=13,
                  input_names=['input'], output_names=['output'])

# Build with DLA
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, logger)
with open(onnx_path, 'rb') as f:
    parser.parse(f.read())

config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 256 << 20)
config.set_flag(trt.BuilderFlag.FP16)

# DLA config
config.default_device_type = trt.DeviceType.DLA
config.DLA_core = 0
config.set_flag(trt.BuilderFlag.GPU_FALLBACK)

print("\nBuilding engine with DLA core 0 + GPU fallback...")
print("(TRT INFO messages below show DLA assignment)\n")

serialized = builder.build_serialized_network(network, config)
if serialized is None:
    print("ENGINE BUILD FAILED")
    sys.exit(1)

engine = runtime.deserialize_cuda_engine(serialized)
print(f"\nEngine built: {engine.num_layers} layers")

# Dump inspector JSON for each layer
try:
    inspector = engine.create_engine_inspector()
    for i in range(engine.num_layers):
        info_json = inspector.get_layer_information(i, trt.LayerInformationFormat.JSON)
        info = json.loads(info_json)
        print(f"\n--- Layer {i} ---")
        print(json.dumps(info, indent=2))
except Exception as e:
    print(f"Inspector failed: {e}")
    print("Trying string format...")
    try:
        inspector = engine.create_engine_inspector()
        for i in range(engine.num_layers):
            info_str = inspector.get_layer_information(i, trt.LayerInformationFormat.ONELINE)
            print(f"  Layer {i}: {info_str}")
    except Exception as e2:
        print(f"String format also failed: {e2}")

os.unlink(onnx_path)
