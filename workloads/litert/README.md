# LiteRT Workloads

test programs representing computational graphs expressed in LiteRT
Compilation of the iree-runtime-api module to generate tflite, mlir, vmfb files


## Step 1: Preprocess mobilenet_v1_tosa.mlir. Follow same for V2 by using V2 files

python walk_mobilenetv1_graph.py mobilenet_v1_tosa.mlir > tmp.txt
	Outputs:	preprocessedv1.mlir: The adjusted MLIR file.
			tmp.txt: Preprocessing logs (optional, for debugging).

#Step 2: Generate the Graph Walk from preprocessedv1.mlir
python walk_mlirMNV1_text_complete.py preprocessedv1.mlir > tmp_text_complete_v1.txt
	Outputs:	graph_walk_complete_v1.txt: The complete graph walk with all 88 nodes and 59 edges.
			tmp_text_complete_v1.txt: Parsing logs (optional, for debugging).


These two scripts and steps take you from the original mobilenet_v1_tosa.mlir to the complete graph walk. No further adjustments are needed unless you want to include constant-to-op edges (e.g., %55 -> %57), which would require a minor tweak. Run these commands anytime to regenerate the graph walk



  ##########          oneLayerMLP           #################

iree-import-tflite tflite/oneLayerMLP.tflite -o mlir/oneLayerMLP_tosa.mlir

## **Output**: `oneLayerMLP_tosa.mlir` (TOSA dialect MLIR file).
## 3.2 Compile the Model for x86 CPU
## Compile the imported MLIR file to an IREE VM FlatBuffer (`.vmfb`) targeting the x86_64 CPU backend:

iree-compile --iree-input-type=tosa --iree-hal-target-backends=llvm-cpu mlir/oneLayerMLP_tosa.mlir -o vmfb/oneLayerMLP.vmfb

## - `--iree-input-type=tosa`: Specifies the input is in TOSA dialect.
##  - `--iree-hal-target-backends=cpu`: Targets the x86 CPU backend.
##- **Output**: `oneLayerMLP.vmfb` (compiled IREE bytecode module).

##C:\IREE\iree\oneLayerMLP\grok> iree_env\Scripts\activate
# Now this is dynamic with latest code -- run through API
#iree-run-module --module=oneLayerMLP.vmfb --input="1x4xf32=[1.0 2.0 3.0 4.0]"


##########          twoLayerMLP            #################

iree-import-tflite twoLayerMLP.tflite -o twoLayerMLP_tosa.mlir

## **Output**: `twoLayerMLP_tosa.mlir` (TOSA dialect MLIR file).
## 3.2 Compile the Model for x86 CPU
## Compile the imported MLIR file to an IREE VM FlatBuffer (`.vmfb`) targeting the x86_64 CPU backend:

iree-compile --iree-input-type=tosa --iree-hal-target-backends=llvm-cpu twoLayerMLP_tosa.mlir -o twoLayerMLP.vmfb

## - `--iree-input-type=tosa`: Specifies the input is in TOSA dialect.
##  - `--iree-hal-target-backends=cpu`: Targets the x86 CPU backend.
##- **Output**: `twoLayerMLP.vmfb` (compiled IREE bytecode module).

##C:\IREE\iree\oneLayerMLP\grok> iree_env\Scripts\activate
iree-run-module --module=twoLayerMLP.vmfb --function=main --input="1x4xf32=[1.0 2.0 3.0 4.0]"

##########          threeLayerMLP            #################

iree-import-tflite threeLayerMLP.tflite -o threeLayerMLP_tosa.mlir

## **Output**: `threeLayerMLP_tosa.mlir` (TOSA dialect MLIR file).
## 3.2 Compile the Model for x86 CPU
## Compile the imported MLIR file to an IREE VM FlatBuffer (`.vmfb`) targeting the x86_64 CPU backend:

iree-compile --iree-input-type=tosa --iree-hal-target-backends=llvm-cpu threeLayerMLP_tosa.mlir -o threeLayerMLP.vmfb

## - `--iree-input-type=tosa`: Specifies the input is in TOSA dialect.
##  - `--iree-hal-target-backends=cpu`: Targets the x86 CPU backend.
##- **Output**: `threeLayerMLP.vmfb` (compiled IREE bytecode module).

##C:\IREE\iree\oneLayerMLP\grok> iree_env\Scripts\activate

iree-run-module --module=threeLayerMLP.vmfb --function=main --input="1x4xf32=[1.0 2.0 3.0 4.0]"
