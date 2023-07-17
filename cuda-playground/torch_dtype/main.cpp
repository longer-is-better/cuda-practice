#include <iostream>
#include <torch/torch.h>

int main() {
    // 创建一个包含随机值的 2x3 的浮点数张量
    at::Tensor tensor = torch::rand({2, 3});

    // 获取张量的数据类型
    if (tensor.dtype().Match<float>()) {
        // 访问元素并转换为 float 类型
        float value = tensor[0][1].item<float>();
        std::cout << "Float value: " << value << std::endl;
    } else if (tensor.dtype().Match<int64_t>()) {
        // 访问元素并转换为 int64_t 类型
        int64_t value = tensor[1][2].item<int64_t>();
        std::cout << "Int64 value: " << value << std::endl;
    } else {
        // 处理其他数据类型
        std::cout << "Unsupported data type." << std::endl;
    }

    return 0;
}
