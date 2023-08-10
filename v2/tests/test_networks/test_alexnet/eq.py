import torch
import torch.nn as nn
input_size = 16 * 7 * 7
output_size = 1
fc = nn.Linear(input_size, output_size)
input_tensor = torch.randn(1, input_size)
print("input_tensor\n ", input_tensor)
print("fc.weight\n ", fc.weight)
print("fc.bias\n ", fc.bias)
output_tensor = fc(input_tensor)
print("output_tensor\n ", output_tensor)

input_tensor_conv = input_tensor.reshape([1, 16, 7, 7])
print("input_tensor_conv\n ", input_tensor_conv)
conv = nn.Conv2d(16, 1, 7, 1, 0)
conv.weight = nn.Parameter(fc.weight.reshape([1, 16, 7, 7]))
conv.bias = nn.Parameter(fc.bias.reshape([1]))
print("conv.weight\n ", conv.weight)
output_tensor_conv = conv(input_tensor_conv)
print("output_tensor_conv: \n", output_tensor_conv)
print(torch.isclose(output_tensor_conv, output_tensor))



