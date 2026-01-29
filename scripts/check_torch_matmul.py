import torch

A = torch.randn(32, 32)
B = torch.randn(32, 32)

C = A @ B

print('GEMM Works:', C.shape)
