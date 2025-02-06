import torch
import numpy as np

# Generate a sample signal (1D array)
signal = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float32)

# Reshape the signal to match PyTorch's expected input shape for conv1d
signal = torch.tensor(signal).reshape(1, 1, -1)  # shape: (batch_size, channels, length)

# Define a simple filter (kernel) for convolution
filter = np.array([1, 0, -1], dtype=np.float32).reshape(1, 1, -1)  # shape: (out_channels, in_channels, filter_length)
filter = torch.tensor(filter)

# Perform the convolution operation using torch.nn.functional.conv1d
convolved_signal = torch.nn.functional.conv1d(signal, filter, stride=1, padding='same')

# Print the results
print("Original Signal: ", signal.numpy().flatten())
print("Convolved Signal: ", convolved_signal.numpy().flatten())

# In this PyTorch version, we use torch.tensor to convert the NumPy arrays to PyTorch tensors
# and torch.nn.functional.conv1d to perform the convolution operation. PyTorch expects the input 
# shape for conv1d to be (batch_size, channels, length), so we reshape the signal accordingly. 
# We also reshape the filter to match PyTorch's expected filter shape (out_channels, in_channels, filter_length).
    