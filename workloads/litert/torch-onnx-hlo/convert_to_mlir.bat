@echo off
REM #iree-import-onnx efficientnet_b0.onnx -o mlir/efficientnet_b0_onnx.mlir
REM #iree-import-onnx efficientnet_b0.onnx -o mlir/efficientnet_b0_stablehlo.mlir
onnx-mlir efficientnet_b0_opset13.onnx --EmitONNXIR -o efficientnet_b0_opset13_onnx
REM #onnx-mlir efficientnet_b0_opset13_onnx.mlir --EmitMLIR -o efficientnet_b0_stablehlo.mlir
onnx-mlir-opt efficientnet_b0_opset13_onnx.onnx.mlir --convert-onnx-to-stablehlo -o efficientnet_b0_stablehlo.mlir
echo Done converting to ONNX and StableHLO MLIR
pause
