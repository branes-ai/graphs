import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf

# Define a 2-layer MLP
class TwoLayerMLP(tf.keras.Model):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TwoLayerMLP, self).__init__()
        self.fc1 = tf.keras.layers.Dense(hidden_dim, activation='tanh')
        self.fc2 = tf.keras.layers.Dense(output_dim, activation='softmax')

    def call(self, x):
        h = self.fc1(x)
        return self.fc2(h)

# Example usage
input_dim = 5
hidden_dim = 4
output_dim = 3
model = TwoLayerMLP(input_dim, hidden_dim, output_dim)

# Example input
batch_size = 10
x = tf.random.normal([batch_size, input_dim])
y = model(x)
print("Output of 2-layer MLP:", y)

# save it in a format that IREE command-line tools can run
# export WRAPT_DISABLE_EXTENSIONS=true
# to remove: TypeError: this __dict__ descriptor does not support '_DictWrapper' objects
tf.saved_model.save(model, "./saved_model/twoLayerMLP")
