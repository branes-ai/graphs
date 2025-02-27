import tensorflow as tf
import numpy as np
import os

# Ensure output directory exists
os.makedirs("tflite", exist_ok=True)

# Define the OneLayerMLP model
class OneLayerMLP(tf.keras.Model):
    def __init__(self, input_dim, output_dim=2):
        super(OneLayerMLP, self).__init__()
        self.dense = tf.keras.layers.Dense(output_dim, input_shape=(input_dim,), activation='linear')

    def call(self, inputs):
        return self.dense(inputs)

# Generate 10 square dimensions growing exponentially to ~1 billion elements
# Target: 31623x31623 ≈ 1 billion elements
base_sizes = [3, 10, 32, 100, 316, 1000, 3162, 10000, 31623, 99999]
max_steps = 4
output_dim = 2

# Process each dimension
for i in range(max_steps):
    size = base_sizes[i]
    input_dim = size * size  # Flatten the 2D input to 1D
    
    # Create model instance
    model = OneLayerMLP(input_dim=input_dim, output_dim=output_dim)
    
    # Create sample input (all ones for simplicity)
    sample_input = np.ones((1, input_dim), dtype=np.float32)
    
    # Define the prediction function with dynamic input signature
    @tf.function(input_signature=[tf.TensorSpec(shape=[1, input_dim], dtype=tf.float32)])
    def predict(inputs):
        return model(inputs)
    
    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_concrete_functions([predict.get_concrete_function()])
    tflite_model = converter.convert()
    
    # Generate filename based on dimension
    filename = f"tflite/oneLayerMLP_{size}x{size}.tflite"
    
    # Save the TFLite model
    with open(filename, "wb") as f:
        f.write(tflite_model)
    
    print(f"OneLayerMLP TFLite model ({size}x{size}, {input_dim:,} elements) saved as '{filename}'")

print("All TFLite models have been generated successfully!")

