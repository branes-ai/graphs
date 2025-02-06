import jax.numpy as jnp
from scipy.ndimage import gaussian_filter

def create_gaussian_filter(size, sigma=1.0):
    # Create an identity matrix
    identity = jnp.eye(size)
    
    # Apply Gaussian filter
    gauss_filter = gaussian_filter(identity, sigma=sigma)
    
    return gauss_filter

# Generate 3x3, 5x5, and 7x7 Gaussian filters
filter_3x3 = create_gaussian_filter(3)
filter_5x5 = create_gaussian_filter(5)
filter_7x7 = create_gaussian_filter(7)

print("3x3 Gaussian Filter:\n", filter_3x3)
print("5x5 Gaussian Filter:\n", filter_5x5)
print("7x7 Gaussian Filter:\n", filter_7x7)
