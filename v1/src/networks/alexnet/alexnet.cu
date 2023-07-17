// #include <utility>
#include <iostream>

#include <torch/script.h> //one-stop header
#include <torch/torch.h>

#include "helper_cuda.h"
#include "conv2d.cuh"
#include "pool2d.cuh"
#include "alexnet.cuh"
#include "descriptor.h"
#include "print_tensor.h"
#include"log.h"

Alexnet::Alexnet(torch::jit::script::Module module, TensorDesc *input_desc) {
    for (const auto& pair : module.named_parameters()) {
        torch::IntArrayRef shape = pair.value.sizes();
        TensorDesc* data_desc = nullptr;
        if (pair.name.find("bias") != std::string::npos) {
            data_desc = new TensorDesc("c", {static_cast<int>(shape.begin()[0])});
        } else if (pair.name.find("weight") != std::string::npos) {
            if (pair.name.find("classifier") != std::string::npos) {
                // if (pair.name.find("classifier.1.weight") != std::string::npos) {
                //     data_desc = new TensorDesc("oihw", {static_cast<int>(shape.begin()[0]), 256, 6, 6});
                // } else {
                    data_desc = new TensorDesc("oihw", {static_cast<int>(shape.begin()[0]), static_cast<int>(shape.begin()[1]), 1, 1});
                // }
            } else if (pair.name.find("features") != std::string::npos) {
                data_desc = new TensorDesc("oihw", shape.begin(), shape.size());
            } else LOGERR("unrecognized layer" + pair.name);
        } else LOGERR("unrecognized layer" + pair.name);
        

        float *data = nullptr;
        checkCudaErrors(
            cudaMallocManaged(
                (void**)&data,
                pair.value.numel() * pair.value.itemsize()
            )
        );

        if (pair.value.dtype().Match<float>()) {
            checkCudaErrors(
                cudaMemcpy(
                    data,
                    pair.value.contiguous().data_ptr<float>(),
                    pair.value.numel() * pair.value.itemsize(),
                    cudaMemcpyHostToHost
                )
            );
        } else LOGERR("not float type");
        weights.insert(make_pair(pair.name, std::make_pair(data_desc, data)));
    }


    checkCudaErrors(cudaMallocManaged((void**)&Conv2d_0, sizeof(Conv2dDesc)));
    Conv2d_0->padding = 2;
    Conv2d_0->stride = 4;


    checkCudaErrors(cudaMallocManaged((void**)&Conv2d_3, sizeof(Conv2dDesc)));
    Conv2d_3->padding = 2;
    Conv2d_3->stride = 1;


    checkCudaErrors(cudaMallocManaged((void**)&Conv2d_6, sizeof(Conv2dDesc)));
    Conv2d_6->padding = 1;
    Conv2d_6->stride = 1;


    checkCudaErrors(cudaMallocManaged((void**)&Conv2d_8, sizeof(Conv2dDesc)));
    Conv2d_8->padding = 1;
    Conv2d_8->stride = 1;


    checkCudaErrors(cudaMallocManaged((void**)&Conv2d_10, sizeof(Conv2dDesc)));
    Conv2d_10->padding = 1;
    Conv2d_10->stride = 1;


    checkCudaErrors(cudaMallocManaged((void**)&Linear_1, sizeof(Conv2dDesc)));
    Linear_1->padding = 0;
    Linear_1->stride = 1;


    checkCudaErrors(cudaMallocManaged((void**)&Linear_4, sizeof(Conv2dDesc)));
    Linear_4->padding = 0;
    Linear_4->stride = 1;


    checkCudaErrors(cudaMallocManaged((void**)&Linear_6, sizeof(Conv2dDesc)));
    Linear_6->padding = 0;
    Linear_6->stride = 1;


    MaxPool2d_2 = new Pool2dDesc({3, 3}, {0, 0}, {2, 2});
    MaxPool2d_5 = new Pool2dDesc({3, 3}, {0, 0}, {2, 2});
    MaxPool2d_12 = new Pool2dDesc({3, 3}, {0, 0}, {2, 2});
    AvgPool2d = new Pool2dDesc({1, 1}, {0, 0}, {1, 1});

    this->input_desc = new TensorDesc(std::move(*input_desc));

    Conv2d_0_output_desc = conv2d_forward_shape_infer(this->input_desc, weights["features.0.weight"].first, Conv2d_0);
    checkCudaErrors(cudaMallocManaged((void**)&Conv2d_0_output, Conv2d_0_output_desc->tensor_size()));
    std::cout << "Conv2d_0_output_desc: " << *Conv2d_0_output_desc << std::endl;

    MaxPool2d_2_output_desc = pool2d_forward_shape_infer(Conv2d_0_output_desc, MaxPool2d_2);
    checkCudaErrors(cudaMallocManaged((void**)&MaxPool2d_2_output, MaxPool2d_2_output_desc->tensor_size()));
    std::cout << "MaxPool2d_2_output_desc: " << *MaxPool2d_2_output_desc << std::endl;

    Conv2d_3_output_desc = conv2d_forward_shape_infer(MaxPool2d_2_output_desc, weights["features.3.weight"].first, Conv2d_3);
    checkCudaErrors(cudaMallocManaged((void**)&Conv2d_3_output, Conv2d_3_output_desc->tensor_size()));
    std::cout << "Conv2d_3_output_desc: " << *Conv2d_3_output_desc << std::endl;

    MaxPool2d_5_output_desc = pool2d_forward_shape_infer(Conv2d_3_output_desc, MaxPool2d_5);
    checkCudaErrors(cudaMallocManaged((void**)&MaxPool2d_5_output, MaxPool2d_5_output_desc->tensor_size()));
    std::cout << "MaxPool2d_5_output_desc: " << *MaxPool2d_5_output_desc << std::endl;

    Conv2d_6_output_desc = conv2d_forward_shape_infer(MaxPool2d_5_output_desc, weights["features.6.weight"].first, Conv2d_6);
    checkCudaErrors(cudaMallocManaged((void**)&Conv2d_6_output, Conv2d_6_output_desc->tensor_size()));
    std::cout << "Conv2d_6_output_desc: " << *Conv2d_6_output_desc << std::endl;

    Conv2d_8_output_desc = conv2d_forward_shape_infer(Conv2d_6_output_desc, weights["features.8.weight"].first, Conv2d_8);
    checkCudaErrors(cudaMallocManaged((void**)&Conv2d_8_output, Conv2d_8_output_desc->tensor_size()));
    std::cout << "Conv2d_8_output_desc: " << *Conv2d_8_output_desc << std::endl;

    Conv2d_10_output_desc = conv2d_forward_shape_infer(Conv2d_8_output_desc, weights["features.10.weight"].first, Conv2d_10);
    checkCudaErrors(cudaMallocManaged((void**)&Conv2d_10_output, Conv2d_10_output_desc->tensor_size()));
    std::cout << "Conv2d_10_output_desc: " << *Conv2d_10_output_desc << std::endl;

    MaxPool2d_12_output_desc = pool2d_forward_shape_infer(Conv2d_10_output_desc, MaxPool2d_12);
    checkCudaErrors(cudaMallocManaged((void**)&MaxPool2d_12_output, MaxPool2d_12_output_desc->tensor_size()));
    std::cout << "MaxPool2d_12_output_desc: " << *MaxPool2d_12_output_desc << std::endl;


    AvgPool2d_output_desc = pool2d_forward_shape_infer(MaxPool2d_12_output_desc, AvgPool2d);
    checkCudaErrors(cudaMallocManaged((void**)&AvgPool2d_output, AvgPool2d_output_desc->tensor_size()));
    std::cout << "AvgPool2d_output_desc: " << *AvgPool2d_output_desc << std::endl;


    flatten_output_desc = new TensorDesc("nchw", {1, 9216, 1, 1});


    // Linear_1_output_desc = conv2d_forward_shape_infer(AvgPool2d_output_desc, weights["classifier.1.weight"].first, Linear_1);
    Linear_1_output_desc = conv2d_forward_shape_infer(flatten_output_desc, weights["classifier.1.weight"].first, Linear_1);
    checkCudaErrors(cudaMallocManaged((void**)&Linear_1_output, Linear_1_output_desc->tensor_size()));
    std::cout << "Linear_1_output_desc: " << *Linear_1_output_desc << std::endl;

    Linear_4_output_desc = conv2d_forward_shape_infer(Linear_1_output_desc, weights["classifier.4.weight"].first, Linear_4);
    checkCudaErrors(cudaMallocManaged((void**)&Linear_4_output, Linear_4_output_desc->tensor_size()));
    std::cout << "Linear_4_output_desc: " << *Linear_4_output_desc << std::endl;

    Linear_6_output_desc = conv2d_forward_shape_infer(Linear_4_output_desc, weights["classifier.6.weight"].first, Linear_6);
    checkCudaErrors(cudaMallocManaged((void**)&Linear_6_output, Linear_6_output_desc->tensor_size()));
    std::cout << "Linear_6_output_desc: " << *Linear_6_output_desc << std::endl;
}

Alexnet::~Alexnet() {
    for (auto& pair : weights) {
        delete pair.second.first;
        checkCudaErrors(
            cudaFree(
                pair.second.second
            )
        );
    }
    checkCudaErrors(cudaFree(Conv2d_0));
    checkCudaErrors(cudaFree(Conv2d_3));
    checkCudaErrors(cudaFree(Conv2d_6));
    checkCudaErrors(cudaFree(Conv2d_8));
    checkCudaErrors(cudaFree(Conv2d_10));
    checkCudaErrors(cudaFree(Linear_1));
    checkCudaErrors(cudaFree(Linear_4));
    checkCudaErrors(cudaFree(Linear_6));
    delete MaxPool2d_2;
    delete MaxPool2d_5;
    delete MaxPool2d_12;
    delete AvgPool2d;

    delete Conv2d_0_output_desc;
    delete MaxPool2d_2_output_desc;
    delete Conv2d_3_output_desc;
    delete MaxPool2d_5_output_desc;
    delete Conv2d_6_output_desc;
    delete Conv2d_8_output_desc;
    delete Conv2d_10_output_desc;
    delete MaxPool2d_12_output_desc;
    delete AvgPool2d_output_desc;
    delete Linear_1_output_desc;
    delete Linear_4_output_desc;
    delete Linear_6_output_desc;

    checkCudaErrors(cudaFree(Conv2d_0_output));
    checkCudaErrors(cudaFree(MaxPool2d_2_output));
    checkCudaErrors(cudaFree(Conv2d_3_output));
    checkCudaErrors(cudaFree(MaxPool2d_5_output));
    checkCudaErrors(cudaFree(Conv2d_6_output));
    checkCudaErrors(cudaFree(Conv2d_8_output));
    checkCudaErrors(cudaFree(Conv2d_10_output));
    checkCudaErrors(cudaFree(MaxPool2d_12_output));
    checkCudaErrors(cudaFree(AvgPool2d_output));
    checkCudaErrors(cudaFree(Linear_1_output));
    checkCudaErrors(cudaFree(Linear_4_output));
    checkCudaErrors(cudaFree(Linear_6_output));

    if (input) checkCudaErrors(cudaFree(input));
    if (input_desc) delete input_desc;
}

std::pair<TensorDesc*, float*> Alexnet::forward(float* input) {
    checkCudaErrors(cudaMallocManaged((void**)&this->input, this->input_desc->tensor_size()));
    checkCudaErrors(cudaMemcpy(this->input, input, this->input_desc->tensor_size(), cudaMemcpyHostToHost));

    // std::cout << "input_desc: " << *this->input_desc << std::endl;
    // PrintTensor(
    //     this->input,
    //     this->input_desc->shape,
    //     this->input_desc->dim_n[0], 
    //     "input"
    // );


    // std::cout << "features.0.weight: " << *this->weights["features.0.weight"].first << std::endl;
    // PrintTensor(
    //     this->weights["features.0.weight"].second,
    //     this->weights["features.0.weight"].first->shape,
    //     this->weights["features.0.weight"].first->dim_n[0], 
    //     "features.0.weight"
    // );


    // std::cout << "features.0.bias_desc: " << *this->weights["features.0.bias"].first << std::endl;
    // PrintTensor(
    //     this->weights["features.0.bias"].second,
    //     this->weights["features.0.bias"].first->shape,
    //     this->weights["features.0.bias"].first->dim_n[0], 
    //     "features.0.bias"
    // );

    conv2d_bias_active_forward_naive<<<dim3(8, 8, 8), dim3(8, 8, 8)>>>(
        this->input,
        this->input_desc,
        this->weights["features.0.weight"].second,
        this->weights["features.0.weight"].first,
        this->Conv2d_0,
        this->weights["features.0.bias"].second,
        this->weights["features.0.bias"].first,
        RELU,
        this->Conv2d_0_output,
        this->Conv2d_0_output_desc,
        false
    );
    checkCudaErrors(cudaDeviceSynchronize());
    

    // std::cout << "Conv2d_0_output_desc: " << *Conv2d_0_output_desc << std::endl;
    // PrintTensor(
    //     this->Conv2d_0_output,
    //     this->Conv2d_0_output_desc->shape,
    //     this->Conv2d_0_output_desc->dim_n[0], 
    //     "Conv2d_0_output shape"
    // );


    max_pool2d<<<dim3(8, 4, 4), dim3(8, 8, 8)>>>(
        this->Conv2d_0_output,
        this->Conv2d_0_output_desc,
        MaxPool2d_2,
        MaxPool2d_2_output,
        MaxPool2d_2_output_desc
    );
    checkCudaErrors(cudaDeviceSynchronize());


    // std::cout << "MaxPool2d_2_output_desc: " << *MaxPool2d_2_output_desc << std::endl;
    // PrintTensor(
    //     this->MaxPool2d_2_output,
    //     this->MaxPool2d_2_output_desc->shape,
    //     this->MaxPool2d_2_output_desc->dim_n[0], 
    //     "MaxPool2d_2_output"
    // );









    conv2d_bias_active_forward_naive<<<dim3(12, 4, 4), dim3(16, 8, 8)>>>(
        MaxPool2d_2_output,
        MaxPool2d_2_output_desc,
        this->weights["features.3.weight"].second,
        this->weights["features.3.weight"].first,
        this->Conv2d_3,
        this->weights["features.3.bias"].second,
        this->weights["features.3.bias"].first,
        RELU,
        this->Conv2d_3_output,
        this->Conv2d_3_output_desc,
        false
    );
    checkCudaErrors(cudaDeviceSynchronize());
    max_pool2d<<<dim3(12, 2, 2), dim3(16, 8, 8)>>>(
        this->Conv2d_3_output,
        this->Conv2d_3_output_desc,
        MaxPool2d_5,
        MaxPool2d_5_output,
        MaxPool2d_5_output_desc
    );
    checkCudaErrors(cudaDeviceSynchronize());



    // std::cout << "MaxPool2d_5_output_desc: " << *MaxPool2d_5_output_desc << std::endl;
    // PrintTensor(
    //     this->MaxPool2d_5_output,
    //     this->MaxPool2d_5_output_desc->shape,
    //     this->MaxPool2d_5_output_desc->dim_n[0], 
    //     "MaxPool2d_5_output"
    // );


    conv2d_bias_active_forward_naive<<<dim3(48, 2, 2), dim3(8, 10, 10)>>>(
        MaxPool2d_5_output,
        MaxPool2d_5_output_desc,
        this->weights["features.6.weight"].second,
        this->weights["features.6.weight"].first,
        this->Conv2d_6,
        this->weights["features.6.bias"].second,
        this->weights["features.6.bias"].first,
        RELU,
        this->Conv2d_6_output,
        this->Conv2d_6_output_desc,
        false
    );
    checkCudaErrors(cudaDeviceSynchronize());
    conv2d_bias_active_forward_naive<<<dim3(32, 2, 2), dim3(8, 10, 10)>>>(
        this->Conv2d_6_output,
        this->Conv2d_6_output_desc,
        this->weights["features.8.weight"].second,
        this->weights["features.8.weight"].first,
        this->Conv2d_8,
        this->weights["features.8.bias"].second,
        this->weights["features.8.bias"].first,
        RELU,
        this->Conv2d_8_output,
        this->Conv2d_8_output_desc,
        false
    );
    checkCudaErrors(cudaDeviceSynchronize());
    conv2d_bias_active_forward_naive<<<dim3(32, 3, 3), dim3(8, 8, 8)>>>(
        this->Conv2d_8_output,
        this->Conv2d_8_output_desc,
        this->weights["features.10.weight"].second,
        this->weights["features.10.weight"].first,
        this->Conv2d_10,
        this->weights["features.10.bias"].second,
        this->weights["features.10.bias"].first,
        RELU,
        this->Conv2d_10_output,
        this->Conv2d_10_output_desc,
        false
    );
    checkCudaErrors(cudaDeviceSynchronize());
    max_pool2d<<<dim3(32, 1, 1), dim3(8, 10, 10)>>>(
        this->Conv2d_10_output,
        this->Conv2d_10_output_desc,
        MaxPool2d_12,
        MaxPool2d_12_output,
        MaxPool2d_12_output_desc
    );
    checkCudaErrors(cudaDeviceSynchronize());



    // std::cout << "MaxPool2d_12_output_desc: " << *MaxPool2d_12_output_desc << std::endl;
    // PrintTensor(
    //     this->MaxPool2d_12_output,
    //     this->MaxPool2d_12_output_desc->shape,
    //     this->MaxPool2d_12_output_desc->dim_n[0], 
    //     "MaxPool2d_12_output"
    // );






    avg_pool2d<<<dim3(16, 1, 1), dim3(16, 6, 6)>>>(
        MaxPool2d_12_output,
        MaxPool2d_12_output_desc,
        AvgPool2d,
        AvgPool2d_output,
        AvgPool2d_output_desc
    );
    checkCudaErrors(cudaDeviceSynchronize());



    // std::cout << "flatten_output_desc: " << *flatten_output_desc << std::endl;
    // PrintTensor(
    //     this->AvgPool2d_output,
    //     this->flatten_output_desc->shape,
    //     this->flatten_output_desc->dim_n[0], 
    //     "flatten_output"
    // );

    // std::cout << "classifier.1.weight shape: " << *this->weights["classifier.1.weight"].first << std::endl;
    // PrintTensor(
    //     this->weights["classifier.1.weight"].second,
    //     this->weights["classifier.1.weight"].first->shape,
    //     this->weights["classifier.1.weight"].first->dim_n[0], 
    //     "classifier.1.weight"
    // );
    
    
    
    // std::cout << "classifier.1.bias shape: " << *this->weights["classifier.1.bias"].first << std::endl;
    // PrintTensor(
    //     this->weights["classifier.1.bias"].second,
    //     this->weights["classifier.1.bias"].first->shape,
    //     this->weights["classifier.1.bias"].first->dim_n[0], 
    //     "classifier.1.bias"
    // );



    conv2d_bias_active_forward_naive<<<dim3(4, 1, 1), dim3(1024, 1, 1)>>>(
        AvgPool2d_output,
        flatten_output_desc,
        this->weights["classifier.1.weight"].second,
        this->weights["classifier.1.weight"].first,
        this->Linear_1,
        this->weights["classifier.1.bias"].second,
        this->weights["classifier.1.bias"].first,
        RELU,
        this->Linear_1_output,
        this->Linear_1_output_desc,
        false
    );
    checkCudaErrors(cudaDeviceSynchronize());



    // std::cout << "Linear_1_output_desc: " << *Linear_1_output_desc << std::endl;
    // PrintTensor(
    //     this->Linear_1_output,
    //     this->Linear_1_output_desc->shape,
    //     this->Linear_1_output_desc->dim_n[0], 
    //     "Linear_1_output"
    // );






    conv2d_bias_active_forward_naive<<<dim3(4, 1, 1), dim3(1024, 1, 1)>>>(
        this->Linear_1_output,
        this->Linear_1_output_desc,
        this->weights["classifier.4.weight"].second,
        this->weights["classifier.4.weight"].first,
        this->Linear_4,
        this->weights["classifier.4.bias"].second,
        this->weights["classifier.4.bias"].first,
        RELU,
        this->Linear_4_output,
        this->Linear_4_output_desc,
        false
    );
    checkCudaErrors(cudaDeviceSynchronize());



    // std::cout << "Linear_4_output_desc: " << *Linear_4_output_desc << std::endl;
    // PrintTensor(
    //     this->Linear_4_output,
    //     this->Linear_4_output_desc->shape,
    //     this->Linear_4_output_desc->dim_n[0], 
    //     "Linear_4_output"
    // );









    conv2d_forward_naive<<<dim3(4, 1, 1), dim3(1024, 1, 1)>>>(
        this->Linear_4_output,
        this->Linear_4_output_desc,
        this->weights["classifier.6.weight"].second,
        this->weights["classifier.6.weight"].first,
        this->Linear_6,
        this->Linear_6_output,
        this->Linear_6_output_desc,
        false
    );
    checkCudaErrors(cudaDeviceSynchronize());

    std::cout << "Linear_6_output_desc: " << *Linear_6_output_desc << std::endl;
    PrintTensor(
        this->Linear_6_output,
        this->Linear_6_output_desc->shape,
        this->Linear_6_output_desc->dim_n[0], 
        "Linear_6_output"
    );

    return std::pair<TensorDesc*, float*>(this->Linear_6_output_desc, this->Linear_6_output);
    // return std::pair<TensorDesc*, float*>(this->flatten_output_desc, this->AvgPool2d_output);
}
