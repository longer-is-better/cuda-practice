import os
import torch
import torchvision

# An instance of your model
model = torchvision.models.alexnet(pretrained=True)
model.avgpool = torch.nn.Sequential(model.avgpool, torch.nn.Flatten(1))
model.classifier[0] = torch.nn.Dropout(p = 0)
model.classifier[1] = torch.nn.Linear(9216, 1)
model.classifier[3] = torch.nn.Dropout(p = 0)


# 0 20 0.006918 24.882370  38.8842
# 0 10 -0.001676 9.706610 -0.0016762
# 10 20 -0.000461 15.166705  13.8590
# 10 15 0.005601 6.935339 0.0056015
# 15 20 0.000000 8.237428  15.3095
# 15 18 0.000000 8.237428  16.4749
# 15 16 0.000000 7.654719  0
# 16 17 0.000000 0.582709  1.1654


# weight_0 = torch.zeros([1, 9216])
# for i in range(16, 17):
#     weight_0[0][i] = 1.
# model.classifier[1].weight = torch.nn.Parameter(weight_0)
# model.classifier[1].bias = torch.nn.Parameter(torch.zeros([1, 1]))

def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.features(x)
    x = self.avgpool(x)
    x = self.classifier[0:2](x)
    return x

model.forward = forward.__get__(model, torchvision.models.alexnet)

print(model)
# AlexNet(
#   (features): Sequential(
#     (0): Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
#     (1): ReLU(inplace=True)
#     (2): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
#     (3): Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
#     (4): ReLU(inplace=True)
#     (5): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
#     (6): Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (7): ReLU(inplace=True)
#     (8): Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (9): ReLU(inplace=True)
#     (10): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (11): ReLU(inplace=True)
#     (12): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
#   )
#   (avgpool): AdaptiveAvgPool2d(output_size=(6, 6))
#   (classifier): Sequential(
#     (0): Dropout(p=0.5, inplace=False)
#     (1): Linear(in_features=9216, out_features=4096, bias=True)
#     (2): ReLU(inplace=True)
#     (3): Dropout(p=0.5, inplace=False)
#     (4): Linear(in_features=4096, out_features=4096, bias=True)
#     (5): ReLU(inplace=True)
#     (6): Linear(in_features=4096, out_features=1000, bias=True)
#   )
# )

# classifier.1.bias   Tensor Shape: 4096 
# classifier.1.weight   Tensor Shape: 4096 9216 
# classifier.4.bias   Tensor Shape: 4096 
# classifier.4.weight   Tensor Shape: 4096 4096 
# classifier.6.bias   Tensor Shape: 1000 
# classifier.6.weight   Tensor Shape: 1000 4096 
# features.0.bias   Tensor Shape: 64 
# features.0.weight   Tensor Shape: 64 3 11 11 
# features.10.bias   Tensor Shape: 256 
# features.10.weight   Tensor Shape: 256 256 3 3 
# features.3.bias   Tensor Shape: 192 
# features.3.weight   Tensor Shape: 192 64 5 5 
# features.6.bias   Tensor Shape: 384 
# features.6.weight   Tensor Shape: 384 192 3 3 
# features.8.bias   Tensor Shape: 256 
# features.8.weight   Tensor Shape: 256 384 3 3 

# Conv2d_0_output_desc: n(1), c(64), h(55), w(55), 
# MaxPool2d_2_output_desc: n(1), c(64), h(27), w(27), 
# Conv2d_3_output_desc: n(1), c(192), h(27), w(27), 
# MaxPool2d_5_output_desc: n(1), c(192), h(13), w(13), 
# Conv2d_6_output_desc: n(1), c(384), h(13), w(13), 
# Conv2d_8_output_desc: n(1), c(256), h(13), w(13), 
# Conv2d_10_output_desc: n(1), c(256), h(13), w(13), 
# MaxPool2d_12_output_desc: n(1), c(256), h(6), w(6), 
# AvgPool2d_output_desc: n(1), c(256), h(6), w(6), 
# Linear_1_output_desc: n(1), c(4096), h(1), w(1), 
# Linear_4_output_desc: n(1), c(4096), h(1), w(1), 
# Linear_6_output_desc: n(1), c(1000), h(1), w(1), 

# An example input you would normally provide to
# your model's forward() method
example = torch.rand(1,3,224,224)

# Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing
traced_script_module = torch.jit.trace(model, example)

# Output and Input
output = traced_script_module(torch.ones(1,3,224,224))
print(type(output), output[0,:10],output.shape)

# This will produce a traced_alexnet_model.pt file
# in working dir
traced_script_module.save(
    os.path.join(
        os.path.dirname(
            os.path.abspath(__file__)
        ),
        "traced_alexnet_model_FAFC02_mod.pt"
    )
)