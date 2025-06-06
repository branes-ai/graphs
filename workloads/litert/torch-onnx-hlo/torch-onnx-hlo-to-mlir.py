import torch
from transformers import EfficientNetForImageClassification
import iree.turbine.aot as aot

# Load model
model_name = "google/efficientnet-b0"
model = EfficientNetForImageClassification.from_pretrained(model_name)
model.eval()
dummy_input = torch.randn(1, 3, 224, 224)

# 1. PyTorch MLIR
export_output = aot.export(model, args=(dummy_input,))
export_output.save_mlir("mlir/efficientnet_b0_pytorch.mlir")
print("PyTorch MLIR saved to efficientnet_b0_pytorch.mlir")

# 2. ONNX Export
torch.onnx.export(
    model,
    dummy_input,
    "efficientnet_b0.onnx",
    opset_version=13,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    export_params=True
)
print("ONNX model saved to efficientnet_b0.onnx")
print("Run manually:")

#iree-import-onnx did not seem reliable.so built onnx-mlir and onnx-mlir-opt for onnx/stablehlo
#onnx-mlir creates its own extension onnx.mlir so leaving out the extension here.

print("  onnx-mlir efficientnet_b0.onnx --EmitONNXIR -o efficientnet_b0_opset13_onnx")
print("  onnx-mlir-opt efficientnet_b0_opset13_onnx.onnx.mlir --convert-onnx-to-stablehlo -o efficientnet_b0_stablehlo.mlir")
