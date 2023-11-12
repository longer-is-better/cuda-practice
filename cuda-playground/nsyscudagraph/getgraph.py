import onnx
import torch
import numpy as np
import onnx_graphsurgeon as ogs

# X1 = ogs.Variable("X1", np.float32, shape=(1, 3, 5, 5))
# X2 = ogs.Variable("X2", np.float32, shape=(1, 3, 5, 5))
# Y1 = ogs.Variable("Y1", np.float32, shape=(1, 3, 1, 1))
# Y2 = ogs.Variable("Y2", np.float32, shape=(1, 3, 1, 1))
# Z = ogs.Variable("Z", np.float32, shape=(1, 3, 1, 1))

# graph = ogs.Graph(
#     [
#         ogs.Node(op="GlobalLpPool", attrs={"p": 2}, inputs=[X], outputs=[Y1]),
#         ogs.Node(op="GlobalLpPool", attrs={"p": 2}, inputs=[X], outputs=[Y2]),
#         ogs.Node(op="GlobalLpPool", attrs={"p": 2}, inputs=[Y1, Y2], outputs=[Z])
#     ],
#     [X],
#     [Z]
# )

device = torch.device("cuda")

class snet(torch.nn.Module):
    def __init__(self):
        super(snet, self).__init__()
        self.conv00 = torch.nn.Conv2d(3, 3, 2, 1)
        self.conv01 = torch.nn.Conv2d(3, 3, 2, 1)
        self.conv02 = torch.nn.Conv2d(3, 3, 2, 1)
        self.conv03 = torch.nn.Conv2d(3, 3, 2, 1)


        self.conv10 = torch.nn.Conv2d(3, 3, 2, 1)
        self.conv11 = torch.nn.Conv2d(3, 3, 2, 1)
        self.conv12 = torch.nn.Conv2d(3, 3, 2, 1)
        self.conv13 = torch.nn.Conv2d(3, 3, 2, 1)

    def forward(self, x0, x1):
        y0 = self.conv03(self.conv02(self.conv01(self.conv00(x0))))
        y1 = self.conv13(self.conv12(self.conv11(self.conv10(x1))))
        return y0 + y1

m = snet().eval().to(device)
x0 = torch.randn(1, 3, 5, 5)   # 生成张量
x0 = x0.to(device)
x1 = torch.randn(1, 3, 5, 5)   # 生成张量
x1 = x1.to(device)
export_onnx_file = "test_globallppool_torch.onnx"		# 目的ONNX文件名
torch.onnx.export(
    model=m,
    args=(x0, x1),
    f=export_onnx_file,
    opset_version=10,
    do_constant_folding=True,	# 是否执行常量折叠优化
    input_names=["input0", "input1"],	# 输入名
    output_names=["output"] #,	# 输出名
    # dynamic_axes={"input":{0:"batch_size"},  # 批处理变量
    #                 "output":{0:"batch_size"}}
)


# onnx.save(ogs.export_onnx(graph), export_onnx_file)