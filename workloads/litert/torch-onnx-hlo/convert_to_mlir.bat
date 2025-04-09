@echo off
iree-import-onnx efficientnet_b0.onnx -o mlir/efficientnet_b0_onnx.mlir
iree-import-onnx efficientnet_b0.onnx -o mlir/efficientnet_b0_stablehlo.mlir
echo Done converting to ONNX and StableHLO MLIR
pause
