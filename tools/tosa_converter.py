import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import argparse
import sys
import tensorflow as tf

def convert_tflite_to_tosa(tflite_file_path, tosa_bytecode_file_path):
    """
    Converts a TensorFlow Lite flatbuffer to TOSA MLIR bytecode.

    Args:
        tflite_file_path: Path to the input TFLite flatbuffer file.
        tosa_bytecode_file_path: Path to save the output TOSA MLIR bytecode file.
    """
    try:
        tf.mlir.experimental.tflite_to_tosa_bytecode(
            flatbuffer=tflite_file_path,
            bytecode=tosa_bytecode_file_path,
        )

        print(f"TFLite model converted to TOSA bytecode and saved to: {tosa_bytecode_file_path}")

    except Exception as e:
        print(f"Error during conversion: {e}")
        sys.exit(1)  # exit with error code



# Converts a TensorFlow Lite flatbuffer to TOSA MLIR bytecode.

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert TFLite to TOSA bytecode.")
    parser.add_argument("tflite_file", help="Input TFLite file, for example: model.tflite.")
    parser.add_argument("tosa_file", help="Output TOSA bytecode file, for example: tosa_mlir.bc.")

    args = parser.parse_args()

    print("input flatbuffer: {}", args.tflite_file)
    print("output bytecode : {}", args.tosa_file)
    convert_tflite_to_tosa(args.tflite_file, args.tosa_file)

