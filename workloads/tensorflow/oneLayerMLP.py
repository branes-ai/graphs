import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf

# Define a 1-layer MLP
class OneLayerMLP(tf.keras.Model):
    def __init__(self, input_dim, output_dim):
        super(OneLayerMLP, self).__init__()
        self.fc = tf.keras.layers.Dense(output_dim)

    def call(self, x):
        return self.fc(x)

# Example usage
input_dim = 5
output_dim = 3
model = OneLayerMLP(input_dim, output_dim)

# Example input
x = tf.random.normal([1, input_dim])
y = model(x)
print("Output of 1-layer MLP:", y)
