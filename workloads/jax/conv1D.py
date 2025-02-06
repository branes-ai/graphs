import jax.numpy as jnp
from jax import random
from jax.scipy.signal import convolve

# Generate a sample signal (1D array)
signal = jnp.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=jnp.float32)

# Define a simple filter (kernel) for convolution
filter = jnp.array([1, 0, -1], dtype=jnp.float32)

# Perform the convolution operation using jax.scipy.signal.convolve
convolved_signal = convolve(signal, filter, mode='same')

# Print the results
print("Original Signal: ", signal)
print("Convolved Signal: ", convolved_signal)

# In JAX, we use jax.numpy (imported as jnp) to work with arrays 
# and jax.scipy.signal.convolve to perform the convolution operation. 
# The convolve function in JAX's scipy.signal module is used 
# similarly to the convolve function in SciPy.