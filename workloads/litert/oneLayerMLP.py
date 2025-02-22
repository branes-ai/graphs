import tensorflow as tf
import numpy as np

# Define the OneLayerMLP model
class OneLayerMLP(tf.keras.Model):
    def __init__(self, input_dim=4, output_dim=2):
        super(OneLayerMLP, self).__init__()
        self.dense = tf.keras.layers.Dense(output_dim, input_shape=(input_dim,), activation='linear')

    def call(self, inputs):
        return self.dense(inputs)

# Create model instance
model = OneLayerMLP(input_dim=4, output_dim=2)

# Sample input for tracing (needed for TFLite conversion)
sample_input = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)

# Convert to TFLite
@tf.function(input_signature=[tf.TensorSpec(shape=[1, 4], dtype=tf.float32)])
def predict(inputs):
    return model(inputs)

# Convert the model to TFLite format
converter = tf.lite.TFLiteConverter.from_concrete_functions([predict.get_concrete_function()])
tflite_model = converter.convert()

# Save the TFLite model to a file
with open("tflite/oneLayerMLP.tflite", "wb") as f:
    f.write(tflite_model)

print("OneLayerMLP TFLite model saved as 'tflite/oneLayerMLP.tflite'")
