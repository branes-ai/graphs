import jax.numpy as jnp
from jax import random, grad

# Initialize parameters for a 1-layer MLP
def init_params_1_layer(key, input_dim, output_dim):
    key, subkey = random.split(key)
    w = random.normal(subkey, (input_dim, output_dim))
    b = jnp.zeros(output_dim)
    return w, b

# Define forward pass for a 1-layer MLP
def forward_1_layer(params, x):
    w, b = params
    return jnp.dot(x, w) + b

# Example usage
key = random.PRNGKey(0)
input_dim = 5
output_dim = 3
params = init_params_1_layer(key, input_dim, output_dim)
x = random.normal(key, (1, input_dim))  # Example input
y = forward_1_layer(params, x)
print("Output of 1-layer MLP:", y)
