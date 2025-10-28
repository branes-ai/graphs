import torch

#
# Simple NN model
# linear(X) = X^T * W
# ReLU(X) = max(0, X)
#
# input tensor X(2, 16)
# weight tensor W(16, 10)
#
# X^T = (16, 2)
# (16, 2) * (

class SimpleNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 10, bias=False)
        self.linear.weight = torch.nn.Parameter(torch.ones(10, 16))
        print(self.linear.weight)
        self.relu = torch.nn.ReLU()
        self.train(False)

    def forward(self, input):
        return self.relu(self.linear(input))

input = torch.randn(2, 16)
print(input)
output = SimpleNN()(input)
print(output)


torch.jit.script(SimpleNN())

for function in torch.jit._state._python_cu.get_functions():
    print(function.graph)


# module = torch_mlir.compiler(SimpleNN(), input, output_type="raw")

print(SimpleNN())
