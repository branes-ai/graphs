import torch
from validation.model_v4.workloads.matmul import build_matmul

w = build_matmul(1024, 1024, 1024, 'fp16')
print('matmul input[0] device:', w.inputs[0].device)
print('matmul input[1] device:', w.inputs[1].device)
print()

from validation.model_v4.workloads.linear import build_linear

w = build_linear(1024, 1024, 1024, 'fp16')
print('linear input[0] device:', w.inputs[0].device)
print('linear weight   device:', next(w.model.parameters().device))

