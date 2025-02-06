import jax.numpy as jnp
from jax import random
from jax.scipy.signal import convolve2d

# Generate a sample 2D signal (image)
signal = jnp.array([
    [1, 2, 3, 4, 5],
    [6, 7, 8, 9, 10],
    [11, 12, 13, 14, 15],
    [16, 17, 18, 19, 20],
    [21, 22, 23, 24, 25]
], dtype=jnp.float32)

# Define a simple 2D filter (kernel) for convolution
filter = jnp.array([
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1]
], dtype=jnp.float32)

# Perform the 2D convolution operation using jax.scipy.signal.convolve2d
convolved_signal = convolve2d(signal, filter, mode='same')

# Print the results
print("Original Signal:\n", signal)
print("Convolved Signal:\n", convolved_signal)
