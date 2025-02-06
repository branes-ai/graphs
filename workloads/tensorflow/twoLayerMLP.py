import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf

# Define a 2-layer MLP
class TwoLayerMLP(tf.keras.Model):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TwoLayerMLP, self).__init__()
        self.fc1 = tf.keras.layers.Dense(hidden_dim, activation='tanh')
        self.fc2 = tf.keras.layers.Dense(output_dim)

    def call(self, x):
        h = self.fc1(x)
        return self.fc2(h)

# Example usage
input_dim = 5
hidden_dim = 4
output_dim = 3
model = TwoLayerMLP(input_dim, hidden_dim, output_dim)

# Example input
x = tf.random.normal([1, input_dim])
y = model(x)
print("Output of 2-layer MLP:", y)
