import tensorflow as tf
from tensorflow.compiler.mlir.stablehlo import stablehlo 
from tensorflow.compiler.mlir.tensorflow_to_stablehlo.python import pywrap_tensorflow_to_stablehlo as tensorflow_to_stablehlo
import numpy as np

# Define the OneLayerMLP model
class OneLayerMLP(tf.keras.Model):
    def __init__(self, input_dim=4, output_dim=2):
        super(OneLayerMLP, self).__init__()
        self.dense = tf.keras.layers.Dense(output_dim, input_shape=(input_dim,), activation='linear')

    def call(self, inputs):
        return self.dense(inputs)

# Create and initialize model instance
model = OneLayerMLP(input_dim=4, output_dim=2)

# Sample input for tracing
sample_input = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)

# Build the model by running a forward pass
model(sample_input)  # Initializes weights and biases

# Define the serving function
@tf.function(input_signature=[tf.TensorSpec(shape=[1, 4], dtype=tf.float32)])
def predict(inputs):
    return model(inputs)

# Get the concrete function
concrete_func = predict.get_concrete_function()

# Freeze variables to constants
from tensorflow.python.framework import convert_to_constants
frozen_func = convert_to_constants.convert_variables_to_constants_v2(concrete_func)

# Save the frozen graph as a SavedModel
saved_model_dir = "savedmodel/oneLayerMLP"
tf.saved_model.save(
    model,
    saved_model_dir,
    signatures={"serving_default": frozen_func}
)
print(f"Model saved to {saved_model_dir}")

# Test the SavedModel
loaded_model = tf.saved_model.load(saved_model_dir)
infer = loaded_model.signatures["serving_default"]
output = infer(tf.constant(sample_input))["output_0"]
print("SavedModel output:", output)

# Convert SavedModel to StableHLO and save to .mlirbc file
module_bytecode = tensorflow_to_stablehlo.savedmodel_to_stablehlo(input_path=saved_model_dir)
target = stablehlo.get_current_version()
portable_byte_str = stablehlo.serialize_portable_artifact_str(module_bytecode, target)
output_file = "oneLayerMLP.mlirbc"
with open(output_file, "wb") as f:  # Use "wb" for binary write
    f.write(portable_byte_str.encode('utf-8') if isinstance(portable_byte_str, str) else portable_byte_str)
print(f"StableHLO bytecode saved to {output_file}")