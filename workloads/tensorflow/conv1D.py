import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf
import numpy as np

# Generate a sample signal (1D array)
signal = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float32)

# Reshape the signal to match TensorFlow's expected input shape for conv1d
signal = signal.reshape(1, -1, 1)  # shape: (batch_size, length, channels)

# Define a simple filter (kernel) for convolution
filter = np.array([1, 0, -1], dtype=np.float32).reshape(-1, 1, 1)  # shape: (filter_length, in_channels, out_channels)

# Convert the filter to a TensorFlow constant
filter_tf = tf.constant(filter)

# Perform the convolution operation using tf.nn.conv1d
convolved_signal = tf.nn.conv1d(signal, filter_tf, stride=1, padding='SAME')

# Initialize a TensorFlow session and run the convolution in TF2.x style
convolved_signal = tf.function(lambda: convolved_signal)()

# Print the results
print("Original Signal: ", signal.flatten())
print("Convolved Signal: ", convolved_signal.numpy().flatten())
