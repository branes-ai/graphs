import jax.numpy as jnp
from jax import random, grad

# Initialize parameters for a 2-layer MLP
def init_params_2_layer(key, input_dim, hidden_dim, output_dim):
    key, subkey = random.split(key)
    w1 = random.normal(subkey, (input_dim, hidden_dim))
    b1 = jnp.zeros(hidden_dim)
    key, subkey = random.split(key)
    w2 = random.normal(subkey, (hidden_dim, output_dim))
    b2 = jnp.zeros(output_dim)
    return (w1, b1), (w2, b2)

# Define forward pass for a 2-layer MLP
def forward_2_layer_v1(params, x):
    (w1, b1), (w2, b2) = params
    h = jnp.tanh(jnp.dot(x, w1) + b1)
    return jnp.dot(h, w2) + b2

# Define forward pass for a 2-layer MLP with Softmax activation
def forward_2_layer(params, x):
    (w1, b1), (w2, b2) = params
    h = jnp.tanh(jnp.dot(x, w1) + b1)
    logits = jnp.dot(h, w2) + b2
    return jnp.exp(logits) / jnp.sum(jnp.exp(logits), axis=-1, keepdims=True)

# Example usage
key = random.PRNGKey(0)
input_dim = 5
hidden_dim = 4
output_dim = 3
params = init_params_2_layer(key, input_dim, hidden_dim, output_dim)
batch_size = 10
x = random.normal(key, (batch_size, input_dim))  # Example input
y = forward_2_layer(params, x)
print("Output of 2-layer MLP:", y)
