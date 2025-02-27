
  ##########          oneLayerMLP           #################

iree-import-tflite tflite/oneLayerMLP_3x3.tflite -o mlir/oneLayerMLP_3x3_tosa.mlir
iree-import-tflite tflite/oneLayerMLP_10x10.tflite -o mlir/oneLayerMLP_10x10_tosa.mlir
iree-import-tflite tflite/oneLayerMLP_32x32.tflite -o mlir/oneLayerMLP_32x32_tosa.mlir
iree-import-tflite tflite/oneLayerMLP_100x100.tflite -o mlir/oneLayerMLP_100x100_tosa.mlir

## **Output**: `oneLayerMLP_tosa.mlir` (TOSA dialect MLIR file).
## 3.2 Compile the Model for x86 CPU
## Compile the imported MLIR file to an IREE VM FlatBuffer (`.vmfb`) targeting the x86_64 CPU backend:

iree-compile --iree-input-type=tosa --iree-hal-target-backends=llvm-cpu mlir/oneLayerMLP_3x3_tosa.mlir -o vmfb/oneLayerMLP_3x3.vmfb
iree-compile --iree-input-type=tosa --iree-hal-target-backends=llvm-cpu mlir/oneLayerMLP_10x10_tosa.mlir -o vmfb/oneLayerMLP_10x10.vmfb
iree-compile --iree-input-type=tosa --iree-hal-target-backends=llvm-cpu mlir/oneLayerMLP_32x32_tosa.mlir -o vmfb/oneLayerMLP_32x322.vmfb
iree-compile --iree-input-type=tosa --iree-hal-target-backends=llvm-cpu mlir/oneLayerMLP_100x100_tosa.mlir -o vmfb/oneLayerMLP_100x100.vmfb

## - `--iree-input-type=tosa`: Specifies the input is in TOSA dialect.
##  - `--iree-hal-target-backends=cpu`: Targets the x86 CPU backend.
##- **Output**: `oneLayerMLP.vmfb` (compiled IREE bytecode module).
##C:\IREE\iree\oneLayerMLP\grok> iree_env\Scripts\activate
# Now this is dynamic with latest code -- run through API
#iree-run-module --module=oneLayerMLP.vmfb --input="1x4xf32=[1.0 2.0 3.0 4.0]"
