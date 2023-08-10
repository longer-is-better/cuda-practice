import torch
from torchvision import models, transforms
from PIL import Image

torch.set_printoptions(edgeitems=8, linewidth=300, precision=6, sci_mode=False)


# 加载预训练的 AlexNet 模型
model = models.alexnet(pretrained=True)
# del model.classifier[3]
# del model.classifier[0]
model.eval()

# 定义图像预处理的转换
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图像大小为 256x256
    # transforms.CenterCrop(224),  # 中心裁剪图像为 224x224
    transforms.ToTensor(),  # 将图像转换为张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化图像
])

# 打开图像并进行预处理
image_path = "/home/dongwei/Workspace/cuda-practice/v1/tests/test_networks/test_alexnet/dog.png"
image = Image.open(image_path)
image = image.convert("RGB")
# image = transforms.ToTensor()(image)
# print(image.shape)

image = preprocess(image)

# 增加一个维度，将图像转换为批次大小为 1 的张量
image = image.unsqueeze(0)

# print("input shape: \n", image.shape)
# print("input: \n", image)

# print("features.0.weight: \n")
# print(model.state_dict()['features.0.weight'])

with torch.no_grad():
    # 获取中间层的输出
    intermediate_output = model.features(image)

# print("intermediate_output shape : \n", intermediate_output.shape)
# print("intermediate_output : \n", intermediate_output)

# exit()

# 运行推理
with torch.no_grad():
    output = model(image)

# 获取预测结果
_, predicted_idx = torch.max(output, 1)
predicted_label = predicted_idx.item()

# 加载预训练模型的标签文件
with open("/home/dongwei/Workspace/cuda-practice/v1/tests/test_networks/test_alexnet/synset_words.txt") as f:
    labels = f.readlines()

# 打印预测结果
print("Predicted label: ", labels[predicted_label])
