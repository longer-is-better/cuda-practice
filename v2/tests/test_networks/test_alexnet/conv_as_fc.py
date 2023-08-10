import os
import torch

class CAF(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(9216, 1)

        # weight_0 = torch.zeros([1, 9216])
        # for i in range(16, 17):
        #     weight_0[0][i] = 1.
        # self.fc.weight = torch.nn.Parameter(weight_0)
        # self.fc.bias = torch.nn.Parameter(torch.zeros([1, 1]))

    def forward(self, x):
        return self.fc(x)
    



caf = CAF()
print(caf)
input = torch.zeros([1, 9216])
for i in range(9216):
    input[0][i] = i
output = caf(input)
print("output: \n", output)

# Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing
traced_script_module = torch.jit.trace(caf, input)

# Output and Input
# output = traced_script_module(torch.ones(1,3,224,224))
# print(type(output), output[0,:10],output.shape)

# This will produce a traced_alexnet_model.pt file
# in working dir
traced_script_module.save(
    os.path.join(
        os.path.dirname(
            os.path.abspath(__file__)
        ),
        "traced_alexnet_model_CAF.pt"
    )
)