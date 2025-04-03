import tensorflow as tf
from transformers import MobileNetV1ForImageClassification, MobileNetV2ForImageClassification
import torch
import onnx
try:
    from onnx_tf.backend import prepare
except ImportError as e:
    print(f"onnx-tf import failed: {e}. Ensure onnx-tf is installed.")
    prepare = None
import subprocess
import numpy as np

# Function to convert TFLite to TOSA MLIR using IREE
def convert_to_tosa_mlir(tflite_file, mlir_file):
    try:
        result = subprocess.run(
            ["iree-import-tflite", tflite_file, "-o", mlir_file],
            check=True,
            capture_output=True,
            text=True
        )
        print(f"Converted {tflite_file} to {mlir_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error converting {tflite_file} to MLIR: {e.stderr}")
    except FileNotFoundError:
        print("iree-import-tflite not found. Ensure IREE is installed.")

# Define models from Hugging Face
models = {
    "mobilenet_v1": "google/mobilenet_v1_1.0_224",
    "mobilenet_v2": "google/mobilenet_v2_1.0_224"
}

# Process each model
for model_name, repo_id in models.items():
    print(f"Processing {model_name} from {repo_id}...")

    # Load PyTorch model from Hugging Face
    try:
        if model_name == "mobilenet_v1":
            pytorch_model = MobileNetV1ForImageClassification.from_pretrained(repo_id)
        else:
            pytorch_model = MobileNetV2ForImageClassification.from_pretrained(repo_id)
        print(f"Loaded {model_name} PyTorch model from Hugging Face.")
    except Exception as e:
        print(f"Error loading {model_name} from Hugging Face: {e}")
        continue

    # Export PyTorch model to ONNX
    try:
        dummy_input = torch.randn(1, 3, 224, 224)
        onnx_file = f"{model_name}_1.0_224.onnx"
        torch.onnx.export(
            pytorch_model,
            dummy_input,
            onnx_file,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            opset_version=13
        )
        print(f"Exported {model_name} to ONNX: {onnx_file}")
    except Exception as e:
        print(f"Error exporting {model_name} to ONNX: {e}")
        continue

    # Convert ONNX to TensorFlow
    if prepare:
        try:
            onnx_model = onnx.load(onnx_file)
            tf_rep = prepare(onnx_model)
            tf_model = tf.keras.models.Model(inputs=tf_rep.inputs, outputs=tf_rep.outputs)
            print(f"Converted {model_name} from ONNX to TensorFlow.")
        except Exception as e:
            print(f"Error converting {model_name} from ONNX to TensorFlow: {e}")
            continue

        # Convert to TFLite
        tflite_file = f"{model_name}_1.0_224.tflite"
        try:
            converter = tf.lite.TFLiteConverter.from_keras_model(tf_model)
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
            tflite_model = converter.convert()
            with open(tflite_file, "wb") as f:
                f.write(tflite_model)
            print(f"Saved {tflite_file}")
        except Exception as e:
            print(f"Error converting {model_name} to TFLite: {e}")
            continue

        # Convert TFLite to TOSA MLIR
        mlir_file = f"{model_name}_tosa.mlir"
        convert_to_tosa_mlir(tflite_file, mlir_file)
    else:
        print(f"Skipping TensorFlow/TFLite conversion for {model_name} due to missing onnx-tf.")

print("Processing complete!")