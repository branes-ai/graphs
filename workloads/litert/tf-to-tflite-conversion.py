import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf
import numpy as np

# Define a 1-layer MLP
class OneLayerMLP(tf.keras.Model):
    def __init__(self, input_dim, output_dim):
        super(OneLayerMLP, self).__init__()
        self.fc = tf.keras.layers.Dense(output_dim, activation='softmax')   # relu, sigmoid, atanh, softmax
    def call(self, x):
        return self.fc(x)

# Set dimensions
input_dim = 5
output_dim = 3
batch_size = 10

# Create and build the model
model = OneLayerMLP(input_dim, output_dim)
# Build the model with a sample input
sample_input = tf.random.normal([batch_size, input_dim])
model(sample_input)
print(model.summary())

# Test the original model
x = tf.random.normal([batch_size, input_dim])
y = model(x)
print("Output of original 1-layer MLP:", y.numpy())

# Convert to TFLite
# 1. Save the model
model.save('one_layer_mlp')

# 2. Convert the saved model to TFLite format
converter = tf.lite.TFLiteConverter.from_saved_model('one_layer_mlp')
tflite_model = converter.convert()

# 3. Save the TFLite model to a file
with open('one_layer_mlp.tflite', 'wb') as f:
    f.write(tflite_model)
print("TFLite model saved successfully")

# 4. Test the TFLite model
# Load the TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path='one_layer_mlp.tflite')
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test the model with the same input
input_data = x.numpy()
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()
tflite_output = interpreter.get_tensor(output_details[0]['index'])

print("Output of TFLite model:", tflite_output)

# Verify the outputs match
print("\nVerifying outputs:")
print("Max difference:", np.max(np.abs(y.numpy() - tflite_output)))
