# TFLite Converter Command-Line Examples

## Prerequisites
Ensure TensorFlow is installed and the command-line tools are available:
```bash
pip install tensorflow
```

## Basic Command-Line Conversion Syntax
```bash
# General syntax
tflite_convert \
  --output_file=output.tflite \
  --graph_def_file=input_graph.pb \
  [additional options]
```

## TensorFlow Model to TFLite Conversion
```bash
# From SavedModel
tflite_convert \
  --saved_model_dir=/path/to/saved_model \
  --output_file=converted_model.tflite

# From Frozen GraphDef
tflite_convert \
  --graph_def_file=frozen_graph.pb \
  --output_file=converted_model.tflite \
  --input_arrays=input_tensor_name \
  --output_arrays=output_tensor_name
```

## MLIR and TFLite Conversion Workflows

### TensorFlow Model to MLIR
```bash
# Convert TensorFlow model to MLIR
tensorflow_model_to_mlir \
  --saved_model_path=/path/to/saved_model \
  --output_file=model.mlir
```

### MLIR to TFLite
```bash
# Convert MLIR to TFLite
mlir_to_tflite \
  --input_file=model.mlir \
  --output_file=converted.tflite
```

### Quantization with MLIR
```bash
# Quantize model using MLIR
mlir_quantize \
  --input_file=model.mlir \
  --output_file=quantized_model.mlir \
  --quantization_method=post_training
```

## Advanced MLIR Conversion Options
```bash
# Detailed MLIR conversion with specific configurations
tf-mlir-translate \
  --input-dialect=tf \
  --output-dialect=tflite \
  --mlir-print-op-generic \
  input_model.mlir > converted_model.mlir
```

## Troubleshooting Conversion
```bash
# Verbose conversion with debugging
tflite_convert \
  --saved_model_dir=/path/to/saved_model \
  --output_file=converted_model.tflite \
  --debug_info_file=conversion_log.txt
```

## Notes
- Exact command names may vary slightly depending on TensorFlow version
- Some advanced MLIR conversion tools might require building TensorFlow from source
- Always verify model compatibility and performance after conversion
