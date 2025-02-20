# Import the TensorFlow SavedModel
iree-import-tf --tf-import-type=savedmodel_v1 --tf-savedmodel-exported-names=predict ./savedmodel/oneLayerMLP -o ./mlir/iree_onelayer.mlir

iree-import-tf --tf-import-type=savedmodel_v1 --tf-savedmodel-exported-names=predict ./savedmodel/twoLayerMLP -o ./mlir/iree_twolayer.mlir

iree-import-tf --tf-import-type=savedmodel_v1 --tf-savedmodel-exported-names=predict ./savedmodel/threeLayerMLP -o ./mlir/iree_threelayer.mlir
