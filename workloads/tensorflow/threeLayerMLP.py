import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf

# Define a 3-layer MLP
class ThreeLayerMLP(tf.keras.Model):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        super(ThreeLayerMLP, self).__init__()
        self.fc1 = tf.keras.layers.Dense(hidden_dim1, activation='tanh')
        self.fc2 = tf.keras.layers.Dense(hidden_dim2, activation='tanh')
        self.fc3 = tf.keras.layers.Dense(output_dim)

    def call(self, x):
        h1 = self.fc1(x)
        h2 = self.fc2(h1)
        return self.fc3(h2)

# Example usage
input_dim = 5
hidden_dim1 = 4
hidden_dim2 = 4
output_dim = 3
model = ThreeLayerMLP(input_dim, hidden_dim1, hidden_dim2, output_dim)

# Example input
x = tf.random.normal([1, input_dim])
y = model(x)
print("Output of 3-layer MLP:", y)
