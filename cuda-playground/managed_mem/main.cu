#include<stdio.h>
#include<cuda_runtime.h>
#include"helper_cuda.h"
// #include<cudnn.h>
#include"log.h"
#include<iostream>



// __global__ void test(float* I){
//     printf("%f", I[0]);
// }

int main(){
    // float* I = nullptr;
    // checkCudaErrors(cudaMallocManaged((void**)&I, 4));
    // cudnnHandle_t _h;
    // cudnnCreate(&_h);
    // cudnnDestroy(_h);

    // float num = 234.f;

    // I[0] = 123.0f;
    // // memcpy(I, &num, 4);

    // test<<<1, 1>>>(I);
    // checkCudaErrors(cudaDeviceSynchronize());


#ifdef __CUDACC_DEBUG__
    std::cout << "__CUDACC_DEBUG__" << std::endl;
#else
    std::cout << "__CUDACC_DEBUG__ not defined" << std::endl;
#endif

    std::cout << "out" << std::endl;
    D(std::cout << "in D" << std::endl;)
    R(std::cout << "in R" << std::endl;)
}