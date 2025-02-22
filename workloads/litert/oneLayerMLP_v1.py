import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import numpy as np
import tensorflow as tf

# Define input parameters
input_dim = 5
output_dim = 3
batch_size = 10

# Create a model builder for TFLite
def build_tflite_mlp():
    # Create a concrete function that represents our MLP
    @tf.function(input_signature=[tf.TensorSpec(shape=[batch_size, input_dim], dtype=tf.float32)])
    def one_layer_mlp(x):
        # Create variables for weights and biases
        w = tf.constant(np.random.normal(size=[input_dim, output_dim]).astype(np.float32))
        b = tf.constant(np.zeros(output_dim).astype(np.float32))
        
        # Linear transformation
        logits = tf.matmul(x, w) + b
        
        # Apply softmax activation
        output = tf.nn.softmax(logits)
        
        return output
    
    # Get concrete function
    concrete_func = one_layer_mlp.get_concrete_function()
    
    # Convert to TFLite model
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    tflite_model = converter.convert()
    
    return tflite_model

# Build and save TFLite model
tflite_model = build_tflite_mlp()
with open('native_one_layer_mlp.tflite', 'wb') as f:
    f.write(tflite_model)
print("Native TFLite MLP model saved successfully")

# Test the TFLite model
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Generate test data
test_input = np.random.normal(size=(batch_size, input_dim)).astype(np.float32)

# Run inference
interpreter.set_tensor(input_details[0]['index'], test_input)
interpreter.invoke()
output = interpreter.get_tensor(output_details[0]['index'])

print(f"Input shape: {test_input.shape}")
print(f"Output shape: {output.shape}")
print("Sample outputs:")
print(output[:3])  # Show first 3 predictions

# Verify outputs are proper probabilities (sum to 1)
output_sums = np.sum(output, axis=1)
print(f"Output sums (should be close to 1.0): {output_sums[:5]}")

# Using the TFLite model in a simple prediction function
def predict(input_data):
    """Make prediction using the TFLite model"""
    if input_data.shape[1] != input_dim:
        raise ValueError(f"Expected input dimension {input_dim}, got {input_data.shape[1]}")
    
    # Convert input to float32 if needed
    if input_data.dtype != np.float32:
        input_data = input_data.astype(np.float32)
        
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])

# Test the prediction function with new data
new_data = np.random.normal(size=(batch_size, input_dim)).astype(np.float32)
predictions = predict(new_data)
print("\nPredictions for new data:")
print(predictions)
