#include<string>
#include<string.h>
#include<vector>
#include<iostream>
#include<cuda_runtime.h>
#include"helper_cuda.h"
#include"descriptor.h"
#include"log.h"


TensorDesc::TensorDesc(const std::string& layout, const std::vector<int>& shape) {
    if (layout.size() != shape.size()) LOGERR("layout.size() != shape.size()");

    checkCudaErrors(cudaMallocManaged((void**)&this->dim_n, sizeof(int)));
    *dim_n = layout.size();

    checkCudaErrors(cudaMallocManaged((void**)&this->layout, layout.size() + 1));
    memcpy(this->layout, layout.c_str(), layout.size() + 1);

    checkCudaErrors(cudaMallocManaged((void**)&this->shape, shape.size() * sizeof(int)));
    memcpy(this->shape, shape.data(), shape.size() * sizeof(int));

    checkCudaErrors(cudaMallocManaged((void**)&this->stride, shape.size() * sizeof(int)));
    this->stride[shape.size() - 1] = 1;
    for (int i = shape.size() - 1; i > 0; i--) this->stride[i - 1] = this->stride[i] * this->shape[i];
}

TensorDesc::~TensorDesc(){
    cudaFree(dim_n);
    cudaFree(layout);
    cudaFree(shape);
    cudaFree(stride);
}

void TensorDesc::init(const std::string& layout, const std::vector<int>& shape) {
    if (layout.size() != shape.size()) LOGERR("layout.size() != shape.size()");

    checkCudaErrors(cudaMallocManaged((void**)&this->dim_n, sizeof(int)));
    *dim_n = layout.size();

    checkCudaErrors(cudaMallocManaged((void**)&this->layout, layout.size() + 1));
    memcpy(this->layout, layout.c_str(), layout.size() + 1);

    checkCudaErrors(cudaMallocManaged((void**)&this->shape, shape.size() * sizeof(int)));
    memcpy(this->shape, shape.data(), shape.size() * sizeof(int));

    checkCudaErrors(cudaMallocManaged((void**)&this->stride, shape.size() * sizeof(int)));
    this->stride[shape.size() - 1] = 1;
    for (int i = shape.size() - 1; i > 0; i--) this->stride[i - 1] = this->stride[i] * this->shape[i];
}

void* TensorDesc::operator new(std::size_t size) {
    void* ptr = nullptr;
    checkCudaErrors(cudaMallocManaged(&ptr, size));
    return ptr;
}

void TensorDesc::operator delete(void* ptr) {
    checkCudaErrors(cudaFree(ptr));
}

TensorDesc::TensorDesc(TensorDesc&& rvalue){
    std::cout << "move TensorDesc, yes!" << std::endl;
    this->dim_n = rvalue.dim_n;
    rvalue.dim_n = 0;
    this->layout = rvalue.layout; rvalue.layout = nullptr;
    this->shape = rvalue.shape; rvalue.shape = nullptr;
    this->stride = rvalue.stride; rvalue.stride = nullptr;
}

TensorDesc& TensorDesc::operator=(TensorDesc&& rvalue){
    std::cout << "move assign TensorDesc, yes!" << std::endl;
    this->dim_n = rvalue.dim_n; rvalue.dim_n = 0;
    this->layout = rvalue.layout; rvalue.layout = nullptr;
    this->shape = rvalue.shape; rvalue.shape = nullptr;
    this->stride = rvalue.stride; rvalue.stride = nullptr;
    return *this;
};