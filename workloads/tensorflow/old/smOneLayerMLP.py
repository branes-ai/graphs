import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf
import tempfile

# Define a 1-layer MLP
class OneLayerMLP(tf.keras.Model):
    def __init__(self, input_dim, output_dim):
        super(OneLayerMLP, self).__init__()
        self.fc = tf.keras.layers.Dense(output_dim, activation='softmax')
        
    def call(self, x):
        return self.fc(x)

def export_model_for_iree(input_dim=5, output_dim=3, export_dir=None):
    """Creates and exports a model with proper signature for IREE.
    
    Returns:
        export_dir: Directory where model was saved
    """
    # Create model
    model = OneLayerMLP(input_dim, output_dim)
    
    # Define the serving signature
    @tf.function(input_signature=[tf.TensorSpec([None, input_dim], tf.float32, name="input")])
    def serving_default(x):
        return {"output": model(x)}
    
    # Create a module with the function
    module = tf.Module()
    module.serving_default = serving_default
    module.f = serving_default  # Also export as a simple name
    
    # Save the module
    if export_dir is None:
        export_dir = tempfile.mkdtemp()
    
    tf.saved_model.save(
        module, 
        export_dir,
        signatures={
            "serving_default": module.serving_default,
            "f": module.f
        }
    )
    
    print(f"Model exported to: {export_dir}")
    return export_dir

# Example usage
if __name__ == "__main__":
    # Define model parameters
    input_dim = 5
    output_dim = 3
    
    # Export the model
    export_dir = export_model_for_iree(input_dim, output_dim, "./savedmodel/oneLayerMLP")
    
    # Print compilation command
    print("\nIREE Compilation Command:")
    print(f"iree-compile \\")
    print(f"  --tf-saved-model-input-dir={export_dir} \\")
    print(f"  --output=model.vmfb \\")
    print(f"  --iree-hal-target-backends=llvm-cpu")
    
    # Print execution command
    print("\nIREE Execution Command:")
    print(f"iree-run-module \\")
    print(f"  --module=model.vmfb \\")
    print(f"  --function=serving_default \\")
    print(f"  --input=1x5xf32=1,2,3,4,5")
