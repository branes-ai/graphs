import jax.numpy as jnp
from jax import random, grad

# Initialize parameters for a 3-layer MLP
def init_params_3_layer(key, input_dim, hidden_dim1, hidden_dim2, output_dim):
    key, subkey = random.split(key)
    w1 = random.normal(subkey, (input_dim, hidden_dim1))
    b1 = jnp.zeros(hidden_dim1)
    key, subkey = random.split(key)
    w2 = random.normal(subkey, (hidden_dim1, hidden_dim2))
    b2 = jnp.zeros(hidden_dim2)
    key, subkey = random.split(key)
    w3 = random.normal(subkey, (hidden_dim2, output_dim))
    b3 = jnp.zeros(output_dim)
    return (w1, b1), (w2, b2), (w3, b3)

# Define forward pass for a 3-layer MLP
def forward_3_layer_v1(params, x):
    (w1, b1), (w2, b2), (w3, b3) = params
    h1 = jnp.tanh(jnp.dot(x, w1) + b1)
    h2 = jnp.tanh(jnp.dot(h1, w2) + b2)
    return jnp.dot(h2, w3) + b3

# Define forward pass for a 3-layer MLP with Softmax activation
def forward_3_layer(params, x):
    (w1, b1), (w2, b2), (w3, b3) = params
    h1 = jnp.tanh(jnp.dot(x, w1) + b1)
    h2 = jnp.tanh(jnp.dot(h1, w2) + b2)
    logits = jnp.dot(h2, w3) + b3
    return jnp.exp(logits) / jnp.sum(jnp.exp(logits), axis=-1, keepdims=True)


# Example usage
key = random.PRNGKey(0)
input_dim = 5
hidden_dim1 = 4
hidden_dim2 = 4
output_dim = 3
params = init_params_3_layer(key, input_dim, hidden_dim1, hidden_dim2, output_dim)
batch_size = 10
x = random.normal(key, (batch_size, input_dim))  # Example input
y = forward_3_layer(params, x)
print("Output of 3-layer MLP:", y)
