import torch

from multi_layer_perceptrons import ThreeLayerMLP 

# Define a large 3-layer MLP model
input_dim = 1024
hidden_dim1 = 8*1024
hidden_dim2 = 8*1024
output_dim = 128
model = ThreeLayerMLP(input_dim, hidden_dim1, hidden_dim2, output_dim)

# Example input
batch_size = 10
x = torch.randn(batch_size, input_dim)
y = model(x)
print("Output of 3-layer MLP:", y)

# Break down the **memory requirements** for each layer in your `ThreeLayerMLP` model. We'll consider both:

# 1. **Parameter memory** (weights and biases)
# 2. **Activation memory** (intermediate outputs during forward pass)

# ---

# ## Layer-by-Layer Memory Breakdown

# ### Assumptions
# - Each `float32` value takes **4 bytes**
# - Batch size = **10**
# - Input dimension = **1024**
# - Hidden layers = **32,768** neurons each
# - Output dimension = **128**

# ---

# ### Layer 1: `fc1 = nn.Linear(input_dim, hidden_dim1)`

# #### Parameters:
# - **Weights**: `1024 × 32768` = 33,554,432
# - **Biases**: `32768`
# - **Total params**: `33,554,432 + 32,768 = 33,587,200`
# - **Memory**: `33,587,200 × 4 bytes ≈ 128 MB`

# #### Activations:
# - Output shape: `[10, 32768]`
# - Memory: `10 × 32768 × 4 bytes ≈ 1.25 MB`

# ---

# ### Layer 2: `fc2 = nn.Linear(hidden_dim1, hidden_dim2)`

# #### Parameters:
# - **Weights**: `32768 × 32768` = 1,073,741,824
# - **Biases**: `32768`
# - **Total params**: `1,073,741,824 + 32768 = 1,073,774,592`
# - **Memory**: `1,073,774,592 × 4 bytes ≈ 4.0 GB`

# #### Activations:
# - Output shape: `[10, 32768]`
# - Memory: `1.25 MB` (same as above)

# ---

# ### Layer 3: `fc3 = nn.Linear(hidden_dim2, output_dim)`

# #### Parameters:
# - **Weights**: `32768 × 128` = 4,194,304
# - **Biases**: `128`
# - **Total params**: `4,194,304 + 128 = 4,194,432`
# - **Memory**: `4,194,432 × 4 bytes ≈ 16 MB`

# #### Activations:
# - Output shape: `[10, 128]`
# - Memory: `10 × 128 × 4 bytes ≈ 5 KB`

# ---

# ## Summary Table

# | Layer   | Parameters | Param Memory | Activation Memory |
# |---------|------------|--------------|-------------------|
# | fc1     | 33.6M      | ~128 MB      | ~1.25 MB          |
# | fc2     | 1.07B      | ~4.0 GB      | ~1.25 MB          |
# | fc3     | 4.2M       | ~16 MB       | ~5 KB             |
# | Total   | 1.11B      | ~4.14 GB     | ~2.5 MB           |

# ---

# ## Notes
# - The **dominant memory cost** is clearly in `fc2`, due to the massive weight matrix.
# - If you're running this on GPU, you’ll need to ensure your device has enough memory (~4.2 GB just for parameters).
# - During backpropagation, **gradient buffers** will roughly double the memory footprint.

