import torch
import torch.nn as nn
import iree.runtime as ireert
import iree.turbine.aot as aot

from multi_layer_perceptrons import OneLayerMLP 
        
if __name__ == "__main__":
    # Define a small 1-layer MLP model.
    in_features  = 1024
    out_features = 128

    model = OneLayerMLP(in_features, out_features)

    # Example input
    batch_size = 10
    x = torch.randn(batch_size, in_features)
    y = model(x)
    print("Output of 1-layer MLP:", y)