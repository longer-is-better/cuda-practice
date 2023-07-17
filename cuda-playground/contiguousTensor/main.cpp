#include <iostream>
#include <torch/torch.h>

int main() {
    // 创建一个非连续存储的 2x3 张量
    float data[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    at::Tensor tensor = torch::from_blob(data, {2, 3});
    at::Tensor nonContiguousTensor = tensor.transpose(0, 1);  // 创建非连续张量

    // 检查张量是否连续
    if (!nonContiguousTensor.is_contiguous()) {
        // 进行连续化操作
        // at::Tensor contiguousTensor = nonContiguousTensor.contiguous();

        // 获取底层数据的指针
        // float* dataPtr = contiguousTensor.data_ptr<float>();
        float* dataPtr = nonContiguousTensor.data_ptr<float>();

        // 获取张量的维度信息
        c10::IntArrayRef sizes = nonContiguousTensor.sizes();

        std::cout << nonContiguousTensor << std::endl;
        // std::cout << ContiguousTensor << std::endl;

        // 遍历张量的值
        for (int i = 0; i < sizes[0]; ++i) {
            for (int j = 0; j < sizes[1]; ++j) {
                // 计算元素在内存中的索引
                int index = i * sizes[1] + j;

                // 访问张量的值
                float value = dataPtr[index];

                // 打印值
                std::cout << "Value at (" << i << ", " << j << "): " << value << "vs" << nonContiguousTensor[i][j] << std::endl;
            }
        }
    }

    return 0;
}
