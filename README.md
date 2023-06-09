# cuda-practice

Implement alexnet from 0 to 1 as an practice.

## ENV

| device           | cuda |
| ---------------- | ---- |
| GeForce 3080 12G | 11.7 |

cuda 11.7

## pytorch Implement

```
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 96, 11, 4), # in_channels, out_channels, kernel_size, stride, padding
            nn.ReLU(),
            nn.MaxPool2d(3, 2), # kernel_size, stride
            # 减小卷积窗口，使用填充为2来使得输入与输出的高和宽一致，且增大输出通道数
            nn.Conv2d(96, 256, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            # 连续3个卷积层，且使用更小的卷积窗口。除了最后的卷积层外，进一步增大了输出通道数。
            nn.Conv2d(256, 384, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(384, 384, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(384, 256, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(3, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(256*5*5, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 10),
        )

    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        return output
```

## 问题和解决方案

## diary

### v1

将所有算子实现为 __global\_\_ 从主机顺序调用

### Implement Completely alone, check Accuracy and performance

1. Implement relu, test and optimize it

# 新知识

![DynamicParallelism](cuda-playground/multifile/DynamicParallelism.png "DynamicParallelism")

```
# --relocatable-device-code {true|false}          (-rdc)                      
#         Enable (disable) the generation of relocatable device code.  If disabled,
#         executable device code is generated.  Relocatable device code must be linked
#         before it can be executed.
#         Default value:  false.
```
