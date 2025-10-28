import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf

# Define a 3-layer MLP
class ThreeLayerMLP(tf.keras.Model):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        super(ThreeLayerMLP, self).__init__()
        self.fc1 = tf.keras.layers.Dense(hidden_dim1, activation='tanh')
        self.fc2 = tf.keras.layers.Dense(hidden_dim2, activation='tanh')
        self.fc3 = tf.keras.layers.Dense(output_dim, activation='softmax')

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
print(model)

# Example input
batch_size = 10
x = tf.random.normal([batch_size, input_dim])
y = model(x)
print("Output of 3-layer MLP:", y)

# save it in a format that IREE command-line tools can run
# export WRAPT_DISABLE_EXTENSIONS=true
# to remove: TypeError: this __dict__ descriptor does not support '_DictWrapper' objects
tf.saved_model.save(model, "./saved_model/threeLayerMLP")
