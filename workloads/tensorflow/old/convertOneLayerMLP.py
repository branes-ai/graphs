import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf

# Define a 1-layer MLP
class OneLayerMLP(tf.keras.Model):
    def __init__(self, input_dim, output_dim):
        super(OneLayerMLP, self).__init__()
        self.fc = tf.keras.layers.Dense(output_dim, activation='softmax')   # relu, sigmoid, atanh, softmax

    def call(self, x):
        return self.fc(x)

# Create an entry point function that IREE can call
@tf.function(input_signature=[tf.TensorSpec([None, input_dim], tf.float32)])
def forward(x):
    """Entry point function for IREE to call.
    This function takes a batch of inputs and returns the model's predictions.
    """
    return model(x)

# Example usage
if __name__ == "__main__":
    # Define model parameters
    input_dim = 5
    output_dim = 3
    model = OneLayerMLP(input_dim, output_dim)
    print(model)
    
    # Example input
    batch_size = 10
    x = tf.random.normal([batch_size, input_dim])
    y = model(x)
    print("Output of 1-layer MLP:", y)
    
    # Show the traced function signature for IREE
    print("\nFunction signature that IREE will use:")
    print(forward.get_concrete_function().pretty_printed_signature())

    # save it in a format that IREE command-line tools can run
    # export WRAPT_DISABLE_EXTENSIONS=true
    # to remove: TypeError: this __dict__ descriptor does not support '_DictWrapper' objects
    tf.saved_model.save(model, "./savedmodel/oneLayerMLP")
