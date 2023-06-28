#include<stdio.h>
#include<cuda_runtime.h>
#include"helper_cuda.h"



__global__ void test(float* I){
    printf("%f", I[0]);
}

int main(){
    float* I = nullptr;
    checkCudaErrors(cudaMallocManaged((void**)&I, 4));

    float num = 234.f;

    I[0] = 123.0f;
    // memcpy(I, &num, 4);

    test<<<1, 1>>>(I);
    checkCudaErrors(cudaDeviceSynchronize());
}