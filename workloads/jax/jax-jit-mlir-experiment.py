import jax
print(jax.__version__)
import jax.numpy as jnp
from jax import random, grad

# JAX has a native mlir serialization
#from jax.experimental import compilation

# Initialize parameters for a 1-layer MLP
def init_params_1_layer(key, input_dim, output_dim):
    key, subkey = random.split(key)
    w = random.normal(subkey, (input_dim, output_dim))
    b = jnp.zeros(output_dim)
    return w, b

# Define forward pass for a 1-layer MLP
def forward_1_layer_v1(params, x):
    w, b = params
    return jnp.dot(x, w) + b

# Define forward pass for a 1-layer MLP with Softmax activation
def forward_1_layer(params, x):
    w, b = params
    logits = jnp.dot(x, w) + b
    return jnp.exp(logits) / jnp.sum(jnp.exp(logits), axis=-1, keepdims=True)

# Example usage
key = random.PRNGKey(0)
input_dim = 5
output_dim = 3
params = init_params_1_layer(key, input_dim, output_dim)
batch_size = 10
x = random.normal(key, (batch_size, input_dim))  # Example input
y = forward_1_layer(params, x)
print("Output of 1-layer MLP:", y)

# create a MLIR serialization of the graph
#jitted_func = jax.jit(forward_1_layer)
# Use compilation.mlir_computation to get the mlir string
#mlir_str = compilation.mlir_computation(jitted_func.lower(x, y).compiler_data(dialect='stablehlo'))()
#
#with open("oneLayerMLP_jax_module.mlir", "w") as f:
#    f.write(str(mlir_module))
