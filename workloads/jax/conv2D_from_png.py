import jax.numpy as jnp
from jax.scipy.signal import convolve2d
from PIL import Image
import numpy as np
import cv2

# Load the PNG file and convert it to a numpy array
image = Image.open('white_circle.png').convert('L')  # Convert to grayscale
signal = np.array(image, dtype=np.float32)

# Convert the numpy array to a jax numpy array
signal = jnp.array(signal)

# Define a simple 5x5 Guassian 2D filter (kernel) for convolution
filter = jnp.array([
    [0.50179990, 0.32346246, 0.13351893, 0.03493138, 0.00628734],
    [0.32346246, 0.31185636, 0.22487491, 0.10487489, 0.03493138],
    [0.13351893, 0.22487491, 0.28321230, 0.22487491, 0.13351893],
    [0.03493138, 0.10487489, 0.22487491, 0.31185636, 0.32346246],
    [0.00628734, 0.03493138, 0.13351893, 0.32346246, 0.5017999 ]
], dtype=jnp.float32)

# Perform the 2D convolution operation using jax.scipy.signal.convolve2d
convolved_signal = convolve2d(signal, filter, mode='same')

# Print the results
print("Original Signal:\n", signal)
print("Convolved Signal:\n", convolved_signal)

# Convert the convolved signal to a numpy array
convolved_signal_np = np.array(convolved_signal)

# Normalize the values to the range [0, 255]
convolved_signal_np = (convolved_signal_np - convolved_signal_np.min()) / (convolved_signal_np.max() - convolved_signal_np.min()) * 255
convolved_signal_np = convolved_signal_np.astype(np.uint8)

# Convert the numpy array to a PIL image and save it as a PNG file
convolved_image = Image.fromarray(convolved_signal_np)
convolved_image.save('white_circle_5x5_Gaussian.png')

