#include "helper_cuda.h"
#include "alexnet.cuh"

Alexnet::Alexnet(std::string model_apath) {
    torch::jit::script::Module module;
    try {
        module = torch::jit::load(model_apath);
    } catch (const c10::Error& e) {
        std::cerr << "Error loading the model\n";
        return;
    }

    for (const auto& pair : module.named_parameters()) {
        float *d = nullptr;
        checkCudaErrors(
            cudaMalloc(
                (void**)&d,
                pair.value.numel() * pair.value.itemsize()
            )
        );
        weights.insert(make_pair(pair.name, d));
        // weights[pair.name] = pair.value.clone();
    }

    // 打印权重
    // for (const auto& pair : state_dict) {
    //     std::cout << pair.first << ":\n" << pair.second << "\n";
    // }

    return 0;

}