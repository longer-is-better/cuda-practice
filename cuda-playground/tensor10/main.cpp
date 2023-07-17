#include <iostream>
#include <torch/torch.h>

void printFirst10Values(const at::Tensor& tensor) {
    int numel = tensor.numel();
    int maxPrint = std::min(numel, 10);  // 输出最多前 10 个值

    std::cout << "Tensor values:" << std::endl;
    for (int i = 0; i < maxPrint; ++i) {
        std::cout << tensor[i].item<float>() << " ";
    }
    std::cout << std::endl;

    if (numel > maxPrint) {
        std::cout << "..." << std::endl;
    }
}

int main() {
    // 创建一个示例的 at::Tensor 对象
    at::Tensor tensor = torch::ones({3, 4, 2});

    // 输出前 10 个值
    printFirst10Values(tensor);

    return 0;
}
