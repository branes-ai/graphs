import tensorflow as tf
import numpy as np

# Define the threeLayerMLP model
class threeLayerMLP(tf.keras.Model):
    def __init__(self, input_dim=4, hidden_dim1=16, hidden_dim2=8, output_dim=2):
        super(threeLayerMLP, self).__init__()
        self.hidden1 = tf.keras.layers.Dense(hidden_dim1, input_shape=(input_dim,), activation='relu')
        self.hidden2 = tf.keras.layers.Dense(hidden_dim2, activation='relu')
        self.output_layer = tf.keras.layers.Dense(output_dim, activation='linear')

    def call(self, inputs):
        x = self.hidden1(inputs)
        x = self.hidden2(x)
        return self.output_layer(x)

# Create model instance
model = threeLayerMLP(input_dim=4, hidden_dim1=16, hidden_dim2=8, output_dim=2)

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
with open("threeLayerMLP.tflite", "wb") as f:
    f.write(tflite_model)

print("threeLayerMLP TFLite model saved as 'threeLayerMLP.tflite'")